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
    object_store: &ObjectStore,
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
