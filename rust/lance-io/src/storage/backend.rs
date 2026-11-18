// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The backend operations consumed by Lance readers, writers, and middleware.
//!
//! This trait uses Lance-owned paths, requests, results, and errors. Providers and
//! wrappers retain their credential, connection-pool, and cache identity when
//! passing this handle.
//! Native object_store implementations enter through [`object_store`].

use crate::storage::{
    CopyMode, CopyOptions, GetOptions, GetResult, ListResult, ObjectMeta, PutMultipartOptions,
    PutOptions, PutPayload, PutResult, RenameOptions, RenameTargetMode, Result, path::Path,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, stream::BoxStream};
use std::{
    fmt::{Debug, Display},
    ops::Range,
};

pub mod object_store;

/// One streaming upload owned by a Lance backend.
///
/// The caller completes it to publish or aborts it after an error. Dropping the
/// upload does not promise remote cleanup.
#[async_trait]
pub trait MultipartUpload: Debug + Send {
    /// Upload a part; implementations preserve call order when completing.
    fn put_part(&mut self, data: PutPayload) -> crate::storage::UploadPart;
    /// Publish the uploaded parts and return generation metadata.
    async fn complete(&mut self) -> Result<PutResult>;
    /// Discard the unfinished upload.
    async fn abort(&mut self) -> Result<()>;
}

/// Runtime storage operations used by Lance.
///
/// Conditional publication must be atomic, or return NotSupported. Implementors
/// must retain native bulk-delete and start-after listing where available.
/// No authorization or cache identity may be reconstructed from a URI here.
#[async_trait]
pub trait StorageBackend: Debug + Display + Send + Sync + 'static {
    /// Atomically publish a payload using the requested write condition.
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult>;

    /// Start a streaming upload; completion is the publication point.
    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>>;

    /// Read bytes or metadata, preserving range and conditional-read semantics.
    async fn get_opts(&self, location: &Path, options: GetOptions) -> Result<GetResult>;

    /// Read multiple ranges without changing their order.
    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> Result<Vec<Bytes>>;

    /// Delete objects through the backend bulk path, retaining per-object results.
    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>>;

    /// Recursively list a path prefix. Ordering is provider-defined.
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, Result<ObjectMeta>>;

    /// List keys strictly after an offset; preserve native request pushdown.
    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>>;

    /// List immediate objects and child prefixes.
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> Result<ListResult>;

    /// Return one native page, preserving opaque tokens, or NotSupported.
    /// Wrappers must forward, filter, or reject pagination consistently with listing.
    /// Storage falls back to wrapped full listing only for NotSupported.
    async fn list_paginated(
        &self,
        prefix: Option<&str>,
        options: crate::storage::PaginatedListOptions,
    ) -> Result<crate::storage::PaginatedListResult>;

    /// Copy with an atomic destination condition when requested.
    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> Result<()>;

    /// Move an object, preserving the destination publication condition.
    async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> Result<()>;
}

impl dyn StorageBackend {
    /// Publish an unconditional payload.
    pub async fn put(&self, path: &Path, payload: PutPayload) -> Result<PutResult> {
        self.put_opts(path, payload, PutOptions::default()).await
    }
    /// Begin an upload with default options.
    pub async fn put_multipart(&self, path: &Path) -> Result<Box<dyn MultipartUpload>> {
        self.put_multipart_opts(path, PutMultipartOptions::default())
            .await
    }
    /// Read an entire object.
    pub async fn get(&self, path: &Path) -> Result<GetResult> {
        self.get_opts(path, GetOptions::default()).await
    }
    /// Read a range, retaining a backend's short response.
    pub async fn get_range(&self, path: &Path, range: Range<u64>) -> Result<Bytes> {
        self.get_opts(
            path,
            GetOptions {
                range: Some(range),
                ..Default::default()
            },
        )
        .await?
        .bytes()
        .await
    }
    /// Read metadata without the object body.
    pub async fn head(&self, path: &Path) -> Result<ObjectMeta> {
        Ok(self
            .get_opts(
                path,
                GetOptions {
                    head: true,
                    ..Default::default()
                },
            )
            .await?
            .meta)
    }
    /// Delete one object and validate the bulk response cardinality.
    pub async fn delete(&self, path: &Path) -> Result<()> {
        let path = path.clone();
        let mut results =
            self.delete_stream(futures::stream::once(async move { Ok(path) }).boxed());
        results
            .try_next()
            .await?
            .ok_or_else(|| crate::storage::Error::Generic {
                store: "lance-io",
                source: "delete returned no result".into(),
            })?;
        if results.next().await.is_some() {
            return Err(crate::storage::Error::Generic {
                store: "lance-io",
                source: "delete returned more than one result".into(),
            });
        }
        Ok(())
    }
    /// Replace the destination with a copy of the source.
    pub async fn copy(&self, from: &Path, to: &Path) -> Result<()> {
        self.copy_opts(
            from,
            to,
            CopyOptions {
                mode: CopyMode::Overwrite,
                ..Default::default()
            },
        )
        .await
    }
    /// Copy only if the destination does not exist.
    pub async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> Result<()> {
        self.copy_opts(
            from,
            to,
            CopyOptions {
                mode: CopyMode::Create,
                ..Default::default()
            },
        )
        .await
    }
    /// Move, replacing the destination.
    pub async fn rename(&self, from: &Path, to: &Path) -> Result<()> {
        self.rename_opts(
            from,
            to,
            RenameOptions {
                target_mode: RenameTargetMode::Overwrite,
                ..Default::default()
            },
        )
        .await
    }
    /// Move only if the destination does not exist.
    pub async fn rename_if_not_exists(&self, from: &Path, to: &Path) -> Result<()> {
        self.rename_opts(
            from,
            to,
            RenameOptions {
                target_mode: RenameTargetMode::Create,
                ..Default::default()
            },
        )
        .await
    }
}
