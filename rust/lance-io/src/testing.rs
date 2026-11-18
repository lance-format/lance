// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::fmt::{self, Display, Formatter};

use crate::object_store::RenameOptions;
use crate::object_store::backend::MultipartUpload;
use crate::object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, ObjectMeta, PutMultipartOptions, PutOptions,
    PutPayload, PutResult, Result as OSResult, path::Path,
};
use async_trait::async_trait;
use futures::stream::BoxStream;
use mockall::mock;
use std::future::Future;

mock! {
    pub StorageBackend {}

    #[async_trait]
    impl crate::object_store::StorageBackend for StorageBackend {
    async fn list_paginated<'a, 'b>(&'a self, prefix: Option<&'b str>, options: crate::object_store::PaginatedListOptions) -> crate::object_store::Result<crate::object_store::PaginatedListResult>;

        async fn put_opts(&self, location: &Path, bytes: PutPayload, opts: PutOptions) -> OSResult<PutResult>;
        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> OSResult<Box<dyn MultipartUpload>>;
        fn get_opts<'life0, 'life1, 'async_trait>(
            &'life0 self,
            location: &'life1 Path,
            options: GetOptions
        ) -> std::pin::Pin<Box<dyn Future<Output=OSResult<GetResult> > +Send+'async_trait> > where
        Self: 'async_trait,
        'life0: 'async_trait,
        'life1: 'async_trait;
        fn delete_stream(&self, locations: BoxStream<'static, OSResult<Path>>) -> BoxStream<'static, OSResult<Path>>;
        fn list<'a>(&self, prefix: Option<&'a Path>) -> BoxStream<'static, OSResult<ObjectMeta>>;
        fn list_with_offset<'a, 'b>(&self, prefix: Option<&'a Path>, offset: &'b Path) -> BoxStream<'static, OSResult<ObjectMeta>>;
        async fn get_ranges(&self, location: &Path, ranges: &[std::ops::Range<u64>]) -> OSResult<Vec<bytes::Bytes>>;
        async fn rename_opts(&self, from: &Path, to: &Path, opts: RenameOptions) -> OSResult<()>;
        async fn list_with_delimiter<'a, 'b>(&'a self, prefix: Option<&'b Path>) -> OSResult<ListResult>;
        async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()>;
    }
}

impl std::fmt::Debug for MockStorageBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "MockStorageBackend")
    }
}

impl Display for MockStorageBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "MockStorageBackend")
    }
}
