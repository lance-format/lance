// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_io::object_store::ObjectStore;
use lance_io::object_store::path::Path;
use lance_io::object_store::{Error as StorageError, PutMode, PutOptions};

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
/// Some object stores implement `PutMode::Create` via temp+rename that reuses the
/// final basename. Dotfile targets such as `.lance-reserved` therefore produce
/// temp names containing `..`, which these stores reject. Stage under a non-dot
/// sibling, then claim the final path with `rename_if_not_exists`.
///
/// When `rename_if_not_exists` is unavailable, fall back to
/// `copy_if_not_exists(staging → target)`, then `PutMode::Create` on the target.
/// That Create path is only for stores whose Create is a true conditional PUT
/// (not basename-derived temp+rename); such stores are exactly the ones that
/// typically omit rename/copy conditionals.
///
/// Some object stores also fail to flush empty objects, so the conditional rename
/// can fail with NotFound. Use a tiny non-empty payload.
///
/// Staging cleanup is best-effort with a few short retries. A delete that still
/// fails after retries leaves a tiny `lance-marker.staging.*` orphan. Async Drop
/// cannot await object-store I/O, so RAII is not used here. Each call uses a
/// unique staging UUID, so concurrent callers never contend on the same cleanup.
pub async fn put_marker_file_atomic(
    storage: &ObjectStore,
    path: &Path,
    file_description: &str,
) -> std::result::Result<(), MarkerFileError> {
    let staging_name = format!("lance-marker.staging.{}", uuid::Uuid::new_v4().simple());
    // Build staging path from the parent Path object (preserving its encoding)
    // rather than string-rebuilding, which would double-encode non-ASCII parents.
    let staging_path = match path.parent() {
        Some(parent) => parent.join(staging_name.as_str()),
        None => Path::from(staging_name.as_str()),
    };

    storage
        .put_payload(&staging_path, bytes::Bytes::from_static(b"reserved").into())
        .await
        .map_err(|e| MarkerFileError::Other {
            message: format!("Failed to stage {}: {:?}", file_description, e),
        })?;

    // Successful rename consumes the staging object; every other path must
    // delete it (best-effort) so conflict/fallback races do not accumulate.
    let mut staging_consumed = false;
    let publish_result = match storage.rename_if_not_exists(&staging_path, path).await {
        Ok(()) => {
            staging_consumed = true;
            Ok(())
        }
        Err(StorageError::NotSupported { .. }) => {
            match storage.copy_if_not_exists(&staging_path, path).await {
                Ok(()) => Ok(()),
                Err(StorageError::NotSupported { .. }) => storage
                    .put_payload_opts(
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
        delete_staging_marker_best_effort(storage, &staging_path).await;
    }

    match publish_result {
        Ok(()) => Ok(()),
        Err(StorageError::AlreadyExists { .. }) | Err(StorageError::Precondition { .. }) => {
            Err(MarkerFileError::AlreadyExists {
                description: file_description.to_string(),
            })
        }
        Err(e) => Err(MarkerFileError::Other {
            message: format!("Failed to create {}: {:?}", file_description, e),
        }),
    }
}

/// Best-effort delete of a per-call staging marker, with short retries for
/// transient store errors. `NotFound` is treated as success (delete may have
/// succeeded despite an earlier ambiguous failure).
async fn delete_staging_marker_best_effort(storage: &ObjectStore, staging_path: &Path) {
    const MAX_ATTEMPTS: u32 = 3;
    const BACKOFF_MS: [u64; 2] = [20, 50];

    let mut last_err: Option<lance_core::Error> = None;
    for attempt in 0..MAX_ATTEMPTS {
        match storage.delete(staging_path).await {
            Ok(()) => return,
            Err(lance_core::Error::NotFound { .. }) => return,
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
    use futures::{StreamExt, TryStreamExt};
    use lance_io::testing::MockStorageBackend;
    use lance_io::utils::testing::TestWrappingObjectStore;
    use std::sync::{Arc, Mutex};

    #[rstest::rstest]
    #[case::rename(true, false, false, 0)]
    #[case::copy(false, true, false, 0)]
    #[case::put(false, false, true, 0)]
    #[case::cleanup_retry(false, true, false, 2)]
    #[case::unsupported(false, false, false, 0)]
    #[tokio::test]
    async fn marker_publication_preserves_backend_fallbacks(
        #[case] rename: bool,
        #[case] copy: bool,
        #[case] put: bool,
        #[case] cleanup_failures: usize,
    ) {
        let target = Path::from("encoded parent/目录/.lance-reserved");
        let observed = Arc::new(Mutex::new(Vec::new()));
        let calls = observed.clone();
        let destination = target.clone();
        let pending_failures = std::sync::atomic::AtomicUsize::new(cleanup_failures);
        let before = Arc::new(
            move |op: &str, path: &Path| -> lance_io::object_store::Result<()> {
                calls.lock().unwrap().push(op.to_string());
                if op == "delete"
                    && pending_failures
                        .fetch_update(
                            std::sync::atomic::Ordering::SeqCst,
                            std::sync::atomic::Ordering::SeqCst,
                            |left| left.checked_sub(1),
                        )
                        .is_ok()
                {
                    return Err(lance_io::object_store::Error::Generic {
                        store: "test",
                        source: "transient cleanup failure".into(),
                    });
                }
                let unavailable = match op {
                    "rename" => !rename,
                    "copy" => !copy,
                    "put" if path == &destination => !put,
                    _ => false,
                };
                if unavailable {
                    Err(lance_io::object_store::Error::NotSupported {
                        source: format!("{op} unavailable").into(),
                    })
                } else {
                    Ok(())
                }
            },
        );
        let mut store = ObjectStore::memory();
        store.apply_wrapper(&TestWrappingObjectStore::new(|backend| {
            let mut mock = MockStorageBackend::new();
            // Mockall's async expectations return values synchronously; these
            // delegated in-memory operations complete without suspending.
            let target = backend.clone();
            let check = before.clone();
            mock.expect_put_opts()
                .returning(move |path, bytes, options| {
                    check("put", path)?;
                    futures::executor::block_on(target.put_opts(path, bytes, options))
                });
            let target = backend.clone();
            let check = before.clone();
            mock.expect_rename_opts()
                .returning(move |from, to, options| {
                    check("rename", from)?;
                    futures::executor::block_on(target.rename_opts(from, to, options))
                });
            let target = backend.clone();
            let check = before.clone();
            mock.expect_copy_opts().returning(move |from, to, options| {
                check("copy", from)?;
                futures::executor::block_on(target.copy_opts(from, to, options))
            });
            let target = backend.clone();
            let check = before.clone();
            mock.expect_delete_stream().returning(move |paths| {
                let check = check.clone();
                target.delete_stream(
                    paths
                        .and_then(move |path| {
                            futures::future::ready(check("delete", &path).map(|()| path))
                        })
                        .boxed(),
                )
            });
            let target = backend.clone();
            mock.expect_list()
                .returning(move |prefix| target.list(prefix));
            mock.expect_get_opts().returning(move |path, options| {
                let target = backend.clone();
                let path = path.clone();
                Box::pin(async move { target.get_opts(&path, options).await })
            });
            Arc::new(mock)
        }));
        let result = put_marker_file_atomic(&store, &target, "reservation").await;
        if !(rename || copy || put) {
            let error = result.unwrap_err();
            assert!(matches!(error, MarkerFileError::Other { .. }));
            assert!(error.to_string().contains("unavailable"));
            assert!(
                store
                    .list_objects(None)
                    .try_collect::<Vec<_>>()
                    .await
                    .unwrap()
                    .is_empty()
            );
            return;
        }
        result.unwrap();
        let publication_calls = observed.lock().unwrap().clone();
        let mut expected_calls = if rename {
            vec!["put", "rename"]
        } else if copy {
            vec!["put", "rename", "copy", "delete"]
        } else {
            vec!["put", "rename", "copy", "put", "delete"]
        };
        expected_calls.extend(std::iter::repeat_n("delete", cleanup_failures));
        assert_eq!(publication_calls, expected_calls);
        let error = put_marker_file_atomic(&store, &target, "reservation")
            .await
            .unwrap_err();
        assert!(matches!(error, MarkerFileError::AlreadyExists { .. }));
        assert_eq!(
            store
                .read_object(&target)
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap(),
            "reserved"
        );
        let objects = store
            .list_objects(None)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].location, target);
    }
}
