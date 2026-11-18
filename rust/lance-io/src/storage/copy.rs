// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::storage::{Error as ObjectStoreError, path::Path};
use bytes::Bytes;
use futures::StreamExt;

use super::Storage;

/// The most conservative server-side-copy size limit across the object
/// stores we support. This is not S3-specific: S3's `CopyObject` and GCS's
/// single-shot `Objects: copy` both reject sources above ~5 GiB, so we use
/// 5 GiB as a backend-agnostic threshold. Above it we stream the source
/// through the client and re-upload via multipart instead of relying on a
/// server-side copy. Stores that have no such cap (e.g. local filesystem)
/// also take the fallback above this size — correctness is preserved; only
/// the rare >5 GiB copy is slower than a native copy would be.
const MAX_SERVER_SIDE_COPY_BYTES: u64 = 5 * 1024 * 1024 * 1024;

/// Part size for the read+rewrite fallback. Multipart-capable stores
/// (S3, GCS) require every part except the last to be ≥5 MB and allow up to
/// 10,000 parts. 100 MB sits comfortably inside both bounds and keeps the
/// part count low (~140 parts for a 14 GB manifest) without large per-part
/// RAM.
const COPY_REWRITE_PART_SIZE: usize = 100 * 1024 * 1024;

impl Storage {
    /// Copy an immutable object whose size is already known.
    ///
    /// Use native copy below 5 GiB and sequential multipart upload above it.
    /// The streaming fallback uses 100 MiB parts and aborts on read, upload, or
    /// completion errors. It does not clean up uploads on task cancellation.
    /// Source-not-found errors retain their classification for commit recovery.
    ///
    /// ```
    /// # use lance_io::storage::Storage;
    /// # use lance_core::storage::path::Path;
    /// # async fn example(store: &Storage) -> lance_core::Result<()> {
    /// let source = Path::from("staging/manifest");
    /// let size = store.stat(&source).await?.size;
    /// store.copy_with_size(&source, &"manifest".into(), size).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn copy_with_size(
        &self,
        from: &Path,
        to: &Path,
        size: u64,
    ) -> std::result::Result<(), ObjectStoreError> {
        if size < MAX_SERVER_SIDE_COPY_BYTES {
            self.inner.copy(from, to).await
        } else {
            copy_via_read_rewrite(self.inner.as_ref(), from, to).await
        }
    }
}

// NOTE: parts are uploaded sequentially. This could be parallelized (a
// bounded JoinSet, like lance-io/src/object_writer.rs's
// LANCE_UPLOAD_CONCURRENCY) or sidestepped entirely by switching to
// `object_store::WriteMultipart` (which also handles abort-on-drop). Left
// sequential here: this is a cold path (only >5 GiB manifests) and the
// helper is itself a stopgap until `object_store` exposes UploadPartCopy.
async fn copy_via_read_rewrite(
    store: &dyn super::StorageBackend,
    from: &Path,
    to: &Path,
) -> std::result::Result<(), ObjectStoreError> {
    // NotFound here propagates upward unchanged.
    let mut stream = store.get(from).await?.into_stream();

    // From here on, errors must `abort()` the upload to avoid leaving an
    // orphan multipart upload on stores that support them (e.g. S3, GCS),
    // which would otherwise incur storage charges until the bucket's
    // lifecycle policy cleans it up.
    //
    // Note: this does NOT cover task cancellation — `MultipartUpload`'s
    // upstream Drop is documented as a no-op for S3/GCS. Callers that
    // need cancellation cleanliness should run this with a guard or
    // switch to `object_store::WriteMultipart` (planned follow-up).
    let mut upload = store.put_multipart(to).await?;
    let mut part_buf: Vec<u8> = Vec::with_capacity(COPY_REWRITE_PART_SIZE);

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(b) => b,
            Err(e) => {
                let _ = upload.abort().await;
                return Err(e);
            }
        };
        // Append the chunk in COPY_REWRITE_PART_SIZE-bounded slices so a
        // single oversized chunk (e.g., LocalFileSystem returning a whole
        // file) cannot push part_buf past the backend's per-part size limit
        // (5 GiB on S3/GCS). COPY_REWRITE_PART_SIZE is well under every
        // backend's cap, so each flushed part is always valid.
        let mut offset = 0;
        while offset < chunk.len() {
            let want = COPY_REWRITE_PART_SIZE - part_buf.len();
            let take = want.min(chunk.len() - offset);
            part_buf.extend_from_slice(&chunk[offset..offset + take]);
            offset += take;

            if part_buf.len() >= COPY_REWRITE_PART_SIZE {
                let payload =
                    std::mem::replace(&mut part_buf, Vec::with_capacity(COPY_REWRITE_PART_SIZE));
                if let Err(e) = upload.put_part(Bytes::from(payload).into()).await {
                    let _ = upload.abort().await;
                    return Err(e);
                }
            }
        }
    }

    // Flush the final (possibly-short) part. The last part of a multipart
    // upload is exempt from the per-part minimum on S3/GCS.
    if !part_buf.is_empty()
        && let Err(e) = upload.put_part(Bytes::from(part_buf).into()).await
    {
        let _ = upload.abort().await;
        return Err(e);
    }

    if let Err(e) = upload.complete().await {
        let _ = upload.abort().await;
        return Err(e);
    }
    Ok(())
}
