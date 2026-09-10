// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt;
use std::ops::Range;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, future, stream::BoxStream};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OSObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    RenameOptions,
};
use object_store_opendal::OpendalStore as InnerOpendalStore;
use opendal::Operator;

/// Adapts OpenDAL listing paths to the spelling used by the request.
///
/// The upstream bridge builds listed locations with [`Path::from`], which
/// percent-encodes reserved characters. Lance builds dataset base paths with
/// [`Path::from_url_path`], so mismatched listed locations must be decoded.
/// Locations that already match the requested prefix retain their spelling to
/// preserve paths containing literal percent escapes.
#[derive(Debug, Clone)]
pub(super) struct OpendalStore {
    inner: InnerOpendalStore,
}

impl OpendalStore {
    pub(super) fn new(operator: Operator) -> Self {
        Self {
            // The scheduler has already coalesced ranges and reserved their
            // bytes. Do not fetch additional gaps outside that reservation.
            inner: InnerOpendalStore::new(operator).with_get_ranges_gap(0),
        }
    }
}

impl fmt::Display for OpendalStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(formatter)
    }
}

fn normalize_location(location: &Path, prefix: Option<&Path>) -> object_store::Result<Path> {
    if prefix.is_none_or(|prefix| location.prefix_matches(prefix)) {
        return Ok(location.clone());
    }

    Path::from_url_path(location.as_ref()).map_err(|source| object_store::Error::Generic {
        store: "OpendalStore",
        source: Box::new(source),
    })
}

fn normalize_object_meta(
    mut meta: ObjectMeta,
    prefix: Option<&Path>,
) -> object_store::Result<ObjectMeta> {
    meta.location = normalize_location(&meta.location, prefix)?;
    Ok(meta)
}

#[async_trait]
impl OSObjectStore for OpendalStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        options: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, options).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<Bytes>> {
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        let listed = self.inner.list(prefix);
        let prefix = prefix.cloned();
        listed
            .map(move |result| result.and_then(|meta| normalize_object_meta(meta, prefix.as_ref())))
            .boxed()
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        if self.inner.info().capability().list_with_start_after {
            let listed = self.inner.list_with_offset(prefix, offset);
            let prefix = prefix.cloned();
            listed
                .map(move |result| {
                    result.and_then(|meta| normalize_object_meta(meta, prefix.as_ref()))
                })
                .boxed()
        } else {
            // The bridge's fallback compares its encoded output with the raw
            // offset. Filter normalized locations so both sides use one form.
            let offset = offset.clone();
            self.list(prefix)
                .try_filter(move |meta| future::ready(meta.location > offset))
                .boxed()
        }
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        let mut result = self.inner.list_with_delimiter(prefix).await?;
        for object in &mut result.objects {
            object.location = normalize_location(&object.location, prefix)?;
        }
        for common_prefix in &mut result.common_prefixes {
            *common_prefix = normalize_location(common_prefix, prefix)?;
        }
        Ok(result)
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }

    async fn rename_opts(
        &self,
        from: &Path,
        to: &Path,
        options: RenameOptions,
    ) -> object_store::Result<()> {
        self.inner.rename_opts(from, to, options).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use futures::TryStreamExt;
    use object_store::ObjectStoreExt;
    use opendal::raw::{self, oio};
    use opendal::services::Memory;
    use rstest::rstest;

    use crate::object_reader::CloudObjectReader;
    use crate::object_store::{DEFAULT_DOWNLOAD_RETRY_COUNT, ObjectStore};
    use crate::scheduler::{ScanScheduler, SchedulerConfig};
    use crate::traits::Reader;
    use crate::utils::CachedFileSize;

    use super::*;

    #[derive(Clone, Debug, Default)]
    struct ReadCounter {
        readers: Arc<AtomicUsize>,
        ranges: Arc<Mutex<Vec<opendal::BytesRange>>>,
    }

    impl raw::Layer for ReadCounter {
        fn apply_service(&self, inner: raw::Servicer) -> raw::Servicer {
            Arc::new(CountingService {
                inner,
                counter: self.clone(),
            })
        }
    }

    #[derive(Debug)]
    struct CountingService {
        inner: raw::Servicer,
        counter: ReadCounter,
    }

    impl raw::Service for CountingService {
        type Reader = oio::Reader;
        type Writer = oio::Writer;
        type Lister = oio::Lister;
        type Deleter = oio::Deleter;
        type Copier = oio::Copier;
        type Composer = oio::Composer;

        fn info(&self) -> raw::ServiceInfo {
            self.inner.info()
        }
        fn capability(&self) -> opendal::Capability {
            self.inner.capability()
        }
        async fn create_dir(
            &self,
            ctx: &opendal::OperationContext,
            path: &str,
            args: opendal::raw::OpCreateDir,
        ) -> opendal::Result<opendal::raw::RpCreateDir> {
            self.inner.create_dir(ctx, path, args).await
        }

        async fn stat(
            &self,
            ctx: &opendal::OperationContext,
            path: &str,
            args: opendal::raw::OpStat,
        ) -> opendal::Result<opendal::raw::RpStat> {
            self.inner.stat(ctx, path, args).await
        }

        fn write(
            &self,
            ctx: &opendal::OperationContext,
            path: &str,
            args: opendal::raw::OpWrite,
        ) -> opendal::Result<Self::Writer> {
            self.inner.write(ctx, path, args)
        }

        fn delete(&self, ctx: &opendal::OperationContext) -> opendal::Result<Self::Deleter> {
            self.inner.delete(ctx)
        }

        fn list(
            &self,
            ctx: &opendal::OperationContext,
            path: &str,
            args: opendal::raw::OpList,
        ) -> opendal::Result<Self::Lister> {
            self.inner.list(ctx, path, args)
        }

        fn copy(
            &self,
            ctx: &opendal::OperationContext,
            from: &str,
            to: &str,
            args: opendal::raw::OpCopy,
        ) -> opendal::Result<Self::Copier> {
            self.inner.copy(ctx, from, to, args)
        }

        async fn rename(
            &self,
            ctx: &opendal::OperationContext,
            from: &str,
            to: &str,
            args: opendal::raw::OpRename,
        ) -> opendal::Result<opendal::raw::RpRename> {
            self.inner.rename(ctx, from, to, args).await
        }

        async fn presign(
            &self,
            ctx: &opendal::OperationContext,
            path: &str,
            args: opendal::raw::OpPresign,
        ) -> opendal::Result<opendal::raw::RpPresign> {
            self.inner.presign(ctx, path, args).await
        }
        fn read(
            &self,
            ctx: &opendal::OperationContext,
            path: &str,
            args: raw::OpRead,
        ) -> opendal::Result<Self::Reader> {
            self.counter.readers.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(CountingReader {
                inner: self.inner.read(ctx, path, args)?,
                counter: self.counter.clone(),
            }))
        }
    }

    struct CountingReader {
        inner: oio::Reader,
        counter: ReadCounter,
    }

    impl oio::Read for CountingReader {
        async fn open(
            &self,
            range: opendal::BytesRange,
        ) -> opendal::Result<(raw::RpRead, Box<dyn oio::ReadStreamDyn>)> {
            self.counter.ranges.lock().unwrap().push(range);
            self.inner.open(range).await
        }

        async fn read(
            &self,
            range: opendal::BytesRange,
        ) -> opendal::Result<(raw::RpRead, opendal::Buffer)> {
            self.counter.ranges.lock().unwrap().push(range);
            self.inner.read(range).await
        }
    }

    #[rstest]
    #[case::standard(false)]
    #[case::lite(true)]
    #[tokio::test]
    async fn test_scheduler_batch_shares_opendal_reader(#[case] use_lite: bool) {
        let operator = Operator::new(Memory::default()).unwrap();
        operator.write("data", "0123456789abcdef").await.unwrap();
        let counter = ReadCounter::default();
        let store = Arc::new(OpendalStore::new(operator.layer(counter.clone())));
        let object_store = Arc::new(ObjectStore::new(
            store.clone(),
            url::Url::parse("memory://").unwrap(),
            Some(1),
            None,
            false,
            false,
            4,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
            None,
        ));
        let scheduler = ScanScheduler::new(
            object_store,
            SchedulerConfig {
                io_buffer_size_bytes: 64,
                use_lite_scheduler: Some(use_lite),
            },
        );
        let reader = Arc::new(
            CloudObjectReader::new(
                store,
                Path::from("data"),
                1,
                Some(16),
                DEFAULT_DOWNLOAD_RETRY_COUNT,
            )
            .unwrap(),
        );
        let file = scheduler
            .open_file(&Path::from("data"), &CachedFileSize::new(16))
            .await
            .unwrap();
        // Nearby ranges coalesce once in Lance; distant ranges stay separate.
        let ranges = vec![0..2, 1..3, 8..10, 14..16];
        let actual = file.submit_request(ranges, 0).await.unwrap();
        assert_eq!(
            actual,
            vec![
                Bytes::from_static(b"01"),
                Bytes::from_static(b"12"),
                Bytes::from_static(b"89"),
                Bytes::from_static(b"ef")
            ]
        );
        assert_eq!(counter.readers.load(Ordering::Relaxed), 1);
        let mut reads = counter.ranges.lock().unwrap().clone();
        reads.sort_by_key(opendal::BytesRange::offset);
        assert_eq!(
            reads,
            vec![
                opendal::BytesRange::new(0, Some(3)),
                opendal::BytesRange::new(8, Some(2)),
                opendal::BytesRange::new(14, Some(2))
            ]
        );
        let stats = scheduler.stats();
        assert_eq!(stats.iops, 3);
        assert_eq!(stats.bytes_read, 7);

        // The Reader API also preserves unsorted, duplicate and empty ranges.
        let actual = reader
            .get_ranges(vec![14..16, 4..4, 0..2, 14..16])
            .await
            .unwrap();
        assert_eq!(
            actual,
            vec![
                Bytes::from_static(b"ef"),
                Bytes::new(),
                Bytes::from_static(b"01"),
                Bytes::from_static(b"ef")
            ]
        );
        assert_eq!(counter.readers.load(Ordering::Relaxed), 2);
        assert!(reader.get_ranges(vec![]).await.unwrap().is_empty());
        assert_eq!(
            reader.get_ranges(vec![4..4]).await.unwrap(),
            vec![Bytes::new()]
        );
        assert_eq!(counter.readers.load(Ordering::Relaxed), 2);
        let error = reader
            .get_ranges(vec![Range { start: 4, end: 2 }])
            .await
            .unwrap_err();
        assert!(matches!(error, object_store::Error::Generic { .. }));
        assert!(
            error
                .to_string()
                .contains("Invalid read range 4..2 for data")
        );
        assert_eq!(counter.readers.load(Ordering::Relaxed), 2);
    }

    #[rstest]
    #[case::raw_reserved_character("tables/run~1/t.lance")]
    #[case::literal_percent_escape("tables/run%25231/t.lance")]
    #[tokio::test]
    async fn test_list_preserves_request_path_spelling(#[case] base_url: &str) {
        let operator = Operator::new(Memory::default()).unwrap();
        let store = OpendalStore::new(operator);
        let base = Path::from_url_path(base_url).unwrap();
        let direct_location = base.clone().join("manifest.lance");
        let nested_location = Path::from_url_path(format!("{base_url}/data/part.lance")).unwrap();
        for location in [&direct_location, &nested_location] {
            store
                .put(location, Bytes::from_static(b"data").into())
                .await
                .unwrap();
        }

        let listed = store
            .list(Some(&base))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut listed_locations = listed
            .into_iter()
            .map(|meta| meta.location)
            .collect::<Vec<_>>();
        listed_locations.sort();
        let mut expected_locations = vec![direct_location.clone(), nested_location.clone()];
        expected_locations.sort();
        assert_eq!(listed_locations, expected_locations);
        assert!(
            listed_locations
                .iter()
                .all(|location| location.prefix_matches(&base))
        );

        let listed_after_nested = store
            .list_with_offset(Some(&base), &nested_location)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(listed_after_nested.len(), 1);
        assert_eq!(listed_after_nested[0].location, direct_location);

        let delimited = store.list_with_delimiter(Some(&base)).await.unwrap();
        assert_eq!(delimited.objects.len(), 1);
        assert_eq!(delimited.objects[0].location, direct_location);
        assert_eq!(delimited.common_prefixes, vec![base.clone().join("data")]);
    }
}
