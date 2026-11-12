// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Test-only object store wrapper that fails the Nth matching request and can
//! delay every GET to stand in for remote round trips. A put fails either
//! before it reaches storage, so the write never happened, or after it
//! succeeded, so the write happened but the writer never learned. Used to
//! simulate crashes at exact points of a fragment metadata tree commit.
//!
//! Only `put_opts` and `get_opts` are intercepted. Multipart uploads, copies,
//! deletes and lists pass through.
//!
//! `ProxyObjectStore` in `lance_core::utils::testing` has no hook after a put,
//! so it cannot model that lost response.

use std::fmt;
use std::sync::{Arc, Mutex};

use crate::object_store::WrappingObjectStore;
use async_trait::async_trait;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    CopyOptions, Error as OSError, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OSObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    Result as OSResult,
};

/// Which request class a failpoint targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailOn {
    /// `put_opts` requests.
    Put,
    /// `get_opts` requests, which include HEAD and range reads. These always
    /// fail before the request, whatever [`Failpoint::when`] says, because a
    /// read leaves no stored state to lose.
    Get,
}

/// When a put failpoint reports its failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailWhen {
    /// The put never reaches storage.
    Before,
    /// The put succeeds and the writer still sees an error.
    After,
}

#[derive(Debug, Clone)]
pub struct Failpoint {
    pub on: FailOn,
    pub when: FailWhen,
    /// Path substring filter, empty matches every path of that class.
    pub path_contains: String,
    /// Fail the Nth matching request (1-based).
    pub nth: usize,
}

#[derive(Debug, Default)]
struct State {
    failpoint: Option<Failpoint>,
    matched: usize,
    tripped: bool,
    /// Simulated GET latency, a stand-in for remote round trips.
    latency: Option<std::time::Duration>,
}

/// Shared failpoint controller, cloned into every wrapped store.
#[derive(Debug, Clone, Default)]
pub struct FailpointController {
    state: Arc<Mutex<State>>,
}

impl FailpointController {
    pub fn arm(&self, failpoint: Failpoint) {
        let mut state = self.state.lock().unwrap();
        state.failpoint = Some(failpoint);
        state.matched = 0;
        state.tripped = false;
    }

    /// Delay every GET by `latency`, simulating a remote round trip.
    pub fn set_get_latency(&self, latency: std::time::Duration) {
        self.state.lock().unwrap().latency = Some(latency);
    }

    fn get_latency(&self) -> Option<std::time::Duration> {
        self.state.lock().unwrap().latency
    }

    pub fn disarm(&self) {
        let mut state = self.state.lock().unwrap();
        state.failpoint = None;
    }

    /// Whether the armed failpoint fired.
    pub fn tripped(&self) -> bool {
        self.state.lock().unwrap().tripped
    }

    /// Decide whether this request must fail, and when.
    fn check(&self, on: FailOn, path: &Path) -> Option<FailWhen> {
        let mut state = self.state.lock().unwrap();
        let failpoint = state.failpoint.clone()?;
        if state.tripped {
            return None;
        }
        if failpoint.on != on || !path.as_ref().contains(&failpoint.path_contains) {
            return None;
        }
        state.matched += 1;
        if state.matched == failpoint.nth {
            state.tripped = true;
            Some(failpoint.when)
        } else {
            None
        }
    }
}

impl WrappingObjectStore for FailpointController {
    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        _original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        None
    }
    fn wrap(
        &self,
        _store_prefix: &str,
        original: Arc<dyn OSObjectStore>,
    ) -> Arc<dyn OSObjectStore> {
        Arc::new(FailpointStore {
            target: original,
            controller: self.clone(),
        })
    }
}

fn injected(path: &Path) -> OSError {
    OSError::Generic {
        store: "failpoint",
        source: format!("injected failure at {path}").into(),
    }
}

#[derive(Debug)]
struct FailpointStore {
    target: Arc<dyn OSObjectStore>,
    controller: FailpointController,
}

impl fmt::Display for FailpointStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FailpointStore({})", self.target)
    }
}

#[async_trait]
impl OSObjectStore for FailpointStore {
    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        match self.controller.check(FailOn::Put, location) {
            Some(FailWhen::Before) => Err(injected(location)),
            Some(FailWhen::After) => {
                self.target.put_opts(location, bytes, opts).await?;
                Err(injected(location))
            }
            None => self.target.put_opts(location, bytes, opts).await,
        }
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        self.target.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        if let Some(latency) = self.controller.get_latency() {
            tokio::time::sleep(latency).await;
        }
        match self.controller.check(FailOn::Get, location) {
            Some(_) => Err(injected(location)),
            None => self.target.get_opts(location, options).await,
        }
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OSResult<Path>>,
    ) -> BoxStream<'static, OSResult<Path>> {
        self.target.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.target.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.target.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, opts: CopyOptions) -> OSResult<()> {
        self.target.copy_opts(from, to, opts).await
    }
}
