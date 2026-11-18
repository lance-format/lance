// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Lance-owned storage requests and results.
//!
//! The adapter converts these values at the provider boundary. No backend
//! implementation is required to implement the object_store traits.

use super::{Path, Result};
use chrono::{DateTime, Utc};
use http::Extensions;
use std::borrow::Cow;
use std::ops::Range;

mod attributes;
mod payload;
mod read;
pub use attributes::{Attribute, Attributes};
pub use payload::PutPayload;
pub use read::{GetResult, GetResultPayload};

/// Completion future for one multipart upload part.
pub type UploadPart = futures::future::BoxFuture<'static, Result<()>>;

/// Result of a list call that includes objects, prefixes (directories) and a
/// token for the next set of results. Individual result sets may be limited to
/// 1,000 objects based on the underlying object storage's limitations.
#[derive(Debug)]
pub struct ListResult {
    /// Prefixes that are common (like directories)
    pub common_prefixes: Vec<Path>,
    /// Object metadata for the listing
    pub objects: Vec<ObjectMeta>,
    /// Implementation-specific extensions. Intended for use by storage backend implementations
    /// that need to return context-specific information (like cache status) from trait methods.
    ///
    /// HTTP-backed stores in this crate populate this with the extensions of the HTTP
    /// response, allowing custom HTTP middleware to propagate information to callers.
    /// Where a result is assembled from multiple paginated requests, the extensions of
    /// each response are merged, with those of later responses taking precedence.
    pub extensions: Extensions,
}

/// The metadata that describes an object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectMeta {
    /// The full path to the object
    pub location: Path,
    /// The last modified time
    pub last_modified: DateTime<Utc>,
    /// The size in bytes of the object.
    ///
    /// Note this is not `usize` to support 32-bit architectures such as WASM
    pub size: u64,
    /// The unique identifier for the object
    ///
    /// <https://datatracker.ietf.org/doc/html/rfc9110#name-etag>
    pub e_tag: Option<String>,
    /// A version indicator for this object
    pub version: Option<String>,
}

/// Options for a get request, such as range
#[derive(Debug, Default, Clone)]
pub struct GetOptions {
    /// Read only if the ETag matches (or `*` for any existing object).
    /// A mismatch returns [`Error::Precondition`](super::Error::Precondition).
    pub if_match: Option<String>,
    /// Read only if the ETag does not match (or `*` for any existing object).
    /// A match returns [`Error::NotModified`](super::Error::NotModified).
    pub if_none_match: Option<String>,
    /// An exclusive-end byte range. An end past EOF returns the remaining bytes.
    /// Empty or reversed ranges, and starts at or past EOF, return an error.
    pub range: Option<Range<u64>>,
    /// Fetch metadata without an object body.
    pub head: bool,
    /// Backend-specific context, forwarded by wrappers and provider adapters.
    pub extensions: Extensions,
}

/// Configure preconditions for the put operation
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PutMode {
    /// Perform an atomic write operation, overwriting any object present at the provided path
    #[default]
    Overwrite,
    /// Perform an atomic write operation, returning [`Error::AlreadyExists`](super::Error::AlreadyExists) if an
    /// object already exists at the provided path
    Create,
    /// Perform an atomic write operation if the current version of the object matches the
    /// provided [`UpdateVersion`], returning [`Error::Precondition`](super::Error::Precondition) otherwise
    Update(UpdateVersion),
}

/// Uniquely identifies a version of an object to update
///
/// Stores will use differing combinations of `e_tag` and `version` to provide conditional
/// updates, and it is therefore recommended applications preserve both
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateVersion {
    /// The unique identifier for the newly created object
    ///
    /// <https://datatracker.ietf.org/doc/html/rfc9110#name-etag>
    pub e_tag: Option<String>,
    /// A version indicator for the newly created object
    pub version: Option<String>,
}

/// Options for a put request
#[derive(Debug, Clone, Default)]
pub struct PutOptions {
    /// Configure the [`PutMode`] for this operation
    pub mode: PutMode,
    /// Provide a set of [`Attributes`]
    ///
    /// Implementations that don't support an attribute should return an error
    pub attributes: Attributes,
    /// Implementation-specific extensions. Intended for use by storage backend implementations
    /// that need to pass context-specific information (like tracing spans) via trait methods.
    ///
    /// Backends and wrappers may interpret these extensions. Unsupported extensions
    /// do not change the operation's storage semantics.
    pub extensions: Extensions,
}

impl From<PutMode> for PutOptions {
    fn from(mode: PutMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }
}

/// Options for storage backend
#[derive(Debug, Clone, Default)]
pub struct PutMultipartOptions {
    /// Provide a set of [`Attributes`]
    ///
    /// Implementations that don't support an attribute should return an error
    pub attributes: Attributes,
    /// Implementation-specific extensions. Intended for use by storage backend implementations
    /// that need to pass context-specific information (like tracing spans) via trait methods.
    ///
    /// Backends may interpret these extensions without changing storage semantics.
    pub extensions: Extensions,
}

/// Result for a put request
#[derive(Debug, Clone)]
pub struct PutResult {
    /// The unique identifier for the newly created object
    ///
    /// <https://datatracker.ietf.org/doc/html/rfc9110#name-etag>
    pub e_tag: Option<String>,
    /// A version indicator for the newly created object
    pub version: Option<String>,
    /// Implementation-specific extensions. Intended for use by storage backend implementations
    /// that need to return context-specific information (like cache status) from trait methods.
    ///
    /// HTTP-backed stores in this crate populate this with the extensions of the HTTP
    /// response, allowing custom HTTP middleware to propagate information to callers.
    pub extensions: Extensions,
}

/// Configure preconditions for the copy operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CopyMode {
    /// Perform an atomic write operation, overwriting any object present at the provided path
    #[default]
    Overwrite,
    /// Perform an atomic write operation, returning [`Error::AlreadyExists`](super::Error::AlreadyExists) if an
    /// object already exists at the provided path
    Create,
}

/// Options for a copy request
#[derive(Debug, Clone, Default)]
pub struct CopyOptions {
    /// Configure the [`CopyMode`] for this operation
    pub mode: CopyMode,
    /// Implementation-specific extensions. Intended for use by storage backend implementations
    /// that need to pass context-specific information (like tracing spans) via trait methods.
    ///
    /// Backends and wrappers may interpret these extensions. Unsupported extensions
    /// do not change the operation's storage semantics.
    pub extensions: Extensions,
}

/// Configure preconditions for the target of rename operation.
///
/// Note though that the source location may or not be deleted at the same time in an atomic operation. There is
/// currently NO flag to control the atomicity of "delete source at the same time as creating the target".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenameTargetMode {
    /// Perform a write operation on the target, overwriting any object present at the provided path.
    #[default]
    Overwrite,
    /// Perform an atomic write operation of the target, returning [`Error::AlreadyExists`](super::Error::AlreadyExists) if an
    /// object already exists at the provided path.
    Create,
}

/// Options for a rename request
#[derive(Debug, Clone, Default)]
pub struct RenameOptions {
    /// Configure the [`RenameTargetMode`] for this operation
    pub target_mode: RenameTargetMode,
    /// Implementation-specific extensions. Intended for use by storage backend implementations
    /// that need to pass context-specific information (like tracing spans) via trait methods.
    ///
    /// Backends and wrappers may interpret these extensions. Unsupported extensions
    /// do not change the operation's storage semantics.
    pub extensions: Extensions,
}

/// Options for a paginated list request
#[derive(Debug, Default, Clone)]
pub struct PaginatedListOptions {
    /// Path to start listing from
    ///
    /// Note: Not all stores support this
    ///
    /// For stores that do support this, the returned
    /// result should not include the object with this key.
    pub offset: Option<String>,

    /// A delimiter use to group keys with a common prefix
    ///
    /// Note: Some stores only support `/`
    pub delimiter: Option<Cow<'static, str>>,

    /// The maximum number of paths to return
    pub max_keys: Option<usize>,

    /// A page token from a previous request
    ///
    /// Note: Behaviour is implementation defined if the previous request
    /// used a different prefix or options
    pub page_token: Option<String>,

    /// Implementation-specific extensions. Intended for use by implementations
    /// that need to pass context-specific information (like tracing spans) via trait methods.
    ///
    /// Backends and wrappers may interpret these extensions. Unsupported extensions
    /// do not change the operation's storage semantics.
    pub extensions: http::Extensions,
}

/// A [`ListResult`] with optional pagination token
#[derive(Debug)]
pub struct PaginatedListResult {
    /// The list result
    pub result: ListResult,
    /// If result set truncated, the pagination token to fetch next results
    pub page_token: Option<String>,
}
