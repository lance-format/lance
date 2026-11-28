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
use opendal::{ErrorKind, Operator};

use crate::object_reader::ConditionalGetSupport;
use crate::object_store::hf_throttle::mark_huggingface_rate_limited;

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
    /// Mark rate-limit errors so Hugging Face's outer retry loops stop repeating
    /// requests that OpenDAL has already retried.
    marks_huggingface_rate_limit: bool,
}

impl OpendalStore {
    pub(super) fn new(operator: Operator) -> Self {
        Self {
            inner: InnerOpendalStore::new(operator),
            marks_huggingface_rate_limit: false,
        }
    }

    #[cfg(feature = "huggingface")]
    pub(super) fn new_huggingface(operator: Operator) -> Self {
        Self {
            inner: InnerOpendalStore::new(operator),
            marks_huggingface_rate_limit: true,
        }
    }
}

fn maybe_mark_huggingface_rate_limit(
    mark: bool,
    error: object_store::Error,
) -> object_store::Error {
    if mark && is_opendal_rate_limited(&error) {
        mark_huggingface_rate_limited(error)
    } else {
        error
    }
}

fn is_opendal_rate_limited(error: &object_store::Error) -> bool {
    let object_store::Error::Generic { source, .. } = error else {
        return false;
    };
    source
        .downcast_ref::<opendal::Error>()
        .is_some_and(|error| error.kind() == ErrorKind::RateLimited)
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
        let mut result = self
            .inner
            .get_opts(location, options)
            .await
            .map_err(|error| {
                maybe_mark_huggingface_rate_limit(self.marks_huggingface_rate_limit, error)
            })?;
        // The bridge can fall back to stat, passing the same conditions to both
        // operations. Require support on both before using a guard to resume.
        let capability = self.inner.info().capability();
        result.extensions.insert(ConditionalGetSupport {
            has_if_match: capability.read_with_if_match && capability.stat_with_if_match,
            has_version: capability.read_with_version && capability.stat_with_version,
        });
        Ok(result)
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> object_store::Result<Vec<Bytes>> {
        self.inner
            .get_ranges(location, ranges)
            .await
            .map_err(|error| {
                maybe_mark_huggingface_rate_limit(self.marks_huggingface_rate_limit, error)
            })
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
        let mark = self.marks_huggingface_rate_limit;
        listed
            .map(move |result| {
                result
                    .map_err(|error| maybe_mark_huggingface_rate_limit(mark, error))
                    .and_then(|meta| normalize_object_meta(meta, prefix.as_ref()))
            })
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
            let mark = self.marks_huggingface_rate_limit;
            listed
                .map(move |result| {
                    result
                        .map_err(|error| maybe_mark_huggingface_rate_limit(mark, error))
                        .and_then(|meta| normalize_object_meta(meta, prefix.as_ref()))
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
        let mut result = self
            .inner
            .list_with_delimiter(prefix)
            .await
            .map_err(|error| {
                maybe_mark_huggingface_rate_limit(self.marks_huggingface_rate_limit, error)
            })?;
        if let Some(prefix) = prefix
            && let Some(first) = result.objects.first()
            && !first.location.prefix_matches(prefix)
        {
            // OpenDAL falls back to the literal request path on invalid UTF-8.
            let encode_for_read = !matches!(
                Path::from_url_path(prefix),
                Err(object_store::path::Error::NonUnicode { .. })
            );
            result
                .extensions
                .insert(super::DecodedFileNames { encode_for_read });
        }
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
    use std::sync::Arc;

    use bytes::Bytes;
    use futures::TryStreamExt;
    use object_store::ObjectStoreExt;
    use object_store::prefix::PrefixStore;
    use opendal::services::Memory;
    use rstest::rstest;

    use super::*;
    use crate::object_store::ObjectStore;

    #[rstest]
    #[case::plain("tables/plain")]
    #[case::reserved("tables/run~1")]
    #[case::mixed("tables/run~1/hash%2523/t.lance")]
    #[case::encoded_separator("tables/run%252F%252Fone/t.lance")]
    #[case::invalid_utf8_escape("tables/%25FF")]
    #[case::root("")]
    #[tokio::test]
    async fn test_delimited_location_reopens_listed_key(
        #[case] base_url: &str,
        #[values(false, true)] native: bool,
        #[values(0, 1, 2)] mounts: u8,
    ) {
        let mut store: Arc<dyn OSObjectStore> = if native {
            Arc::new(object_store::memory::InMemory::new())
        } else {
            Arc::new(OpendalStore::new(Operator::new(Memory::default()).unwrap()))
        };
        for mount in ["mount~1", "part%23one"].into_iter().take(mounts.into()) {
            store = Arc::new(PrefixStore::new(store, Path::parse(mount).unwrap()));
        }
        let store = ObjectStore {
            inner: store,
            ..ObjectStore::memory()
        };
        let base = Path::from_url_path(base_url).unwrap();
        let report = base.clone().join("July%20report.json");
        store.put(&report, b"monthly totals").await.unwrap();

        let prefix = (!base.as_ref().is_empty()).then_some(&base);
        let listed = store.list_with_delimiter(prefix).await.unwrap();
        assert_eq!(listed.objects.len(), 1);
        let selected = &listed.objects[0].location;

        assert_eq!(selected, &report, "listed location must reopen the report");
        assert_eq!(
            store
                .inner
                .get(selected)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
            &b"monthly totals"[..]
        );
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
