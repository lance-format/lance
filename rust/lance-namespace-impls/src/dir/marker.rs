// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use object_store::{
    Error as ObjectStoreError, ObjectStore as OSObjectStore, ObjectStoreExt, PutMode, PutOptions,
};

/// Error from [`put_marker_file_atomic`].
#[derive(Debug)]
pub enum MarkerFileError {
    /// The final marker path is already present (Create / rename race).
    AlreadyExists { description: String },
    /// Staging or publish failed for a non-conflict reason.
    Other { message: String },
}

impl std::fmt::Display for MarkerFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyExists { description } => {
                write!(f, "{} already exists", description)
            }
            Self::Other { message } => write!(f, "{}", message),
        }
    }
}

/// Atomically create a marker file (e.g. `.lance-reserved`) with Create semantics.
///
/// S3 uses `PutMode::Create` directly: some S3-compatible stores enforce
/// create-only PUTs but ignore destination conditions on multipart copy
/// completion, making conditional rename/copy unsafe for claiming markers.
/// A failed S3 PUT is returned without falling back to rename/copy.
///
/// Other stores may implement `PutMode::Create` via temp+rename that reuses the
/// final basename. Dotfile targets such as `.lance-reserved` therefore produce
/// temp names containing `..`, which these stores reject. Stage under a non-dot
/// sibling, then claim the final path with `rename_if_not_exists`.
///
/// When `rename_if_not_exists` is unavailable, fall back to
/// `copy_if_not_exists(staging → target)`, then `PutMode::Create` on the target.
///
/// Some object stores also fail to flush empty objects, so the conditional rename
/// can fail with NotFound. Use a tiny non-empty payload.
///
/// Staging cleanup is best-effort with a few short retries. A delete that still
/// fails after retries leaves a tiny `lance-marker.staging.*` orphan. Async Drop
/// cannot await object-store I/O, so RAII is not used here. Each staged call uses a
/// unique staging UUID, so concurrent callers never contend on the same cleanup.
pub async fn put_marker_file_atomic(
    object_store: &ObjectStore,
    path: &Path,
    file_description: &str,
) -> std::result::Result<(), MarkerFileError> {
    let publish_result = if matches!(object_store.scheme(), "s3" | "s3+ddb") {
        object_store
            .inner
            .put_opts(
                path,
                bytes::Bytes::from_static(b"reserved").into(),
                PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                },
            )
            .await
            .map(|_| ())
    } else {
        let staging_name = format!("lance-marker.staging.{}", uuid::Uuid::new_v4().simple());
        // Build staging path from the parent Path object (preserving its encoding)
        // rather than string-rebuilding, which would double-encode non-ASCII parents.
        let staging_path = match path.parent() {
            Some(parent) => parent.join(staging_name.as_str()),
            None => Path::from(staging_name.as_str()),
        };

        object_store
            .inner
            .put(&staging_path, bytes::Bytes::from_static(b"reserved").into())
            .await
            .map_err(|e| MarkerFileError::Other {
                message: format!("Failed to stage {}: {:?}", file_description, e),
            })?;

        // Successful rename consumes the staging object; every other path must
        // delete it (best-effort) so conflict/fallback races do not accumulate.
        let mut staging_consumed = false;
        let publish_result = match object_store
            .inner
            .rename_if_not_exists(&staging_path, path)
            .await
        {
            Ok(()) => {
                staging_consumed = true;
                Ok(())
            }
            Err(ObjectStoreError::NotImplemented { .. })
            | Err(ObjectStoreError::NotSupported { .. }) => {
                match object_store
                    .inner
                    .copy_if_not_exists(&staging_path, path)
                    .await
                {
                    Ok(()) => Ok(()),
                    Err(ObjectStoreError::NotImplemented { .. })
                    | Err(ObjectStoreError::NotSupported { .. }) => object_store
                        .inner
                        .put_opts(
                            path,
                            bytes::Bytes::from_static(b"reserved").into(),
                            PutOptions {
                                mode: PutMode::Create,
                                ..Default::default()
                            },
                        )
                        .await
                        .map(|_| ()),
                    Err(e) => Err(e),
                }
            }
            Err(e) => Err(e),
        };

        if !staging_consumed {
            delete_staging_marker_best_effort(object_store, &staging_path).await;
        }
        publish_result
    };

    match publish_result {
        Ok(()) => Ok(()),
        Err(ObjectStoreError::AlreadyExists { .. })
        | Err(ObjectStoreError::Precondition { .. }) => Err(MarkerFileError::AlreadyExists {
            description: file_description.to_string(),
        }),
        Err(e) => Err(MarkerFileError::Other {
            message: format!("Failed to create {}: {:?}", file_description, e),
        }),
    }
}

/// Best-effort delete of a per-call staging marker, with short retries for
/// transient store errors. `NotFound` is treated as success (delete may have
/// succeeded despite an earlier ambiguous failure).
async fn delete_staging_marker_best_effort(object_store: &ObjectStore, staging_path: &Path) {
    const MAX_ATTEMPTS: u32 = 3;
    const BACKOFF_MS: [u64; 2] = [20, 50];

    let mut last_err: Option<ObjectStoreError> = None;
    for attempt in 0..MAX_ATTEMPTS {
        match object_store.inner.delete(staging_path).await {
            Ok(()) => return,
            Err(ObjectStoreError::NotFound { .. }) => return,
            Err(e) => {
                last_err = Some(e);
                if let Some(&delay_ms) = BACKOFF_MS.get(attempt as usize) {
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                }
            }
        }
    }
    if let Some(del_err) = last_err {
        log::warn!(
            "Failed to delete staging marker at '{}' after {} attempts: {:?}",
            staging_path,
            MAX_ATTEMPTS,
            del_err
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use futures::{TryStreamExt, stream::BoxStream};
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
        PutMultipartOptions, PutPayload, PutResult, Result as OSResult, memory::InMemory,
    };
    use rstest::rstest;
    use url::Url;

    #[derive(Debug, Clone, Copy)]
    enum StoreBehavior {
        IgnoresCopyCondition,
        RequiresStaging,
    }

    #[derive(Debug)]
    struct MarkerStore {
        inner: InMemory,
        behavior: StoreBehavior,
        copy_calls: AtomicUsize,
    }

    impl std::fmt::Display for MarkerStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MarkerStore({:?})", self.behavior)
        }
    }

    #[async_trait]
    impl OSObjectStore for MarkerStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> OSResult<PutResult> {
            if matches!(self.behavior, StoreBehavior::RequiresStaging)
                && opts.mode == PutMode::Create
                && location
                    .filename()
                    .is_some_and(|name| name.starts_with('.'))
            {
                return Err(ObjectStoreError::NotSupported {
                    source: "basename-derived temporary path contains '..'".into(),
                });
            }
            // Let concurrent publishers reach the store before either mutation.
            tokio::task::yield_now().await;
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(&self, location: &Path, opts: GetOptions) -> OSResult<GetResult> {
            self.inner.get_opts(location, opts).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, OSResult<Path>>,
        ) -> BoxStream<'static, OSResult<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
            self.copy_calls.fetch_add(1, Ordering::SeqCst);
            // Model an S3-compatible endpoint that silently ignores conditional copy.
            let opts = match self.behavior {
                StoreBehavior::IgnoresCopyCondition => CopyOptions::default(),
                StoreBehavior::RequiresStaging => opts,
            };
            self.inner.copy_opts(from, to, opts).await
        }
    }

    #[rstest]
    #[case::s3("s3://bucket/", StoreBehavior::IgnoresCopyCondition)]
    #[case::s3_ddb("s3+ddb://bucket/", StoreBehavior::IgnoresCopyCondition)]
    #[case::staged("custom://bucket/", StoreBehavior::RequiresStaging)]
    #[tokio::test]
    async fn test_put_marker_file_atomic(
        #[case] uri: &str,
        #[case] behavior: StoreBehavior,
        #[values(".lance-reserved", ".lance-deregistered")] marker_name: &str,
    ) {
        let inner = Arc::new(MarkerStore {
            inner: InMemory::new(),
            behavior,
            copy_calls: AtomicUsize::new(0),
        });
        let store = ObjectStore::new(
            inner.clone(),
            Url::parse(uri).unwrap(),
            None,
            None,
            false,
            true,
            1,
            0,
            None,
        );
        let path = Path::from("namespace/表.lance").join(marker_name);
        inner.put(&path, "existing marker".into()).await.unwrap();

        let err = put_marker_file_atomic(&store, &path, "table marker")
            .await
            .unwrap_err();
        assert!(matches!(err, MarkerFileError::AlreadyExists { .. }));
        assert_eq!(err.to_string(), "table marker already exists");
        assert_eq!(
            inner.get(&path).await.unwrap().bytes().await.unwrap(),
            "existing marker"
        );

        inner.delete(&path).await.unwrap();
        let (first, second) = tokio::join!(
            put_marker_file_atomic(&store, &path, "table marker"),
            put_marker_file_atomic(&store, &path, "table marker"),
        );
        let err = match (first, second) {
            (Ok(()), Err(err)) | (Err(err), Ok(())) => err,
            results => panic!("expected exactly one marker owner, got {results:?}"),
        };
        assert!(matches!(err, MarkerFileError::AlreadyExists { .. }));
        assert_eq!(err.to_string(), "table marker already exists");
        assert_eq!(
            inner.get(&path).await.unwrap().bytes().await.unwrap(),
            "reserved"
        );
        let objects: Vec<_> = inner.list(None).try_collect().await.unwrap();
        assert_eq!(objects.len(), 1, "staging objects must be cleaned up");
        assert_eq!(objects[0].location, path);
        match behavior {
            StoreBehavior::IgnoresCopyCondition => {
                assert_eq!(inner.copy_calls.load(Ordering::SeqCst), 0);
            }
            StoreBehavior::RequiresStaging => {
                assert!(inner.copy_calls.load(Ordering::SeqCst) > 0);
            }
        }
    }

    #[tokio::test]
    async fn test_s3_marker_create_error_does_not_fall_back_to_copy() {
        let inner = Arc::new(MarkerStore {
            inner: InMemory::new(),
            behavior: StoreBehavior::RequiresStaging,
            copy_calls: AtomicUsize::new(0),
        });
        let store = ObjectStore::new(
            inner.clone(),
            Url::parse("s3://bucket/").unwrap(),
            None,
            None,
            false,
            true,
            1,
            0,
            None,
        );
        let err = put_marker_file_atomic(&store, &Path::from(".lance-reserved"), "table marker")
            .await
            .unwrap_err();
        assert!(matches!(err, MarkerFileError::Other { .. }));
        assert!(err.to_string().contains("Failed to create table marker"));
        assert!(err.to_string().contains("NotSupported"));
        assert_eq!(inner.copy_calls.load(Ordering::SeqCst), 0);
        assert!(
            inner
                .list(None)
                .try_collect::<Vec<_>>()
                .await
                .unwrap()
                .is_empty()
        );
    }
}
