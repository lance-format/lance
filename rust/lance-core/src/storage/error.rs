// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

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

//! Backend-independent storage error categories.

use super::path;

/// A specialized `Result` for storage errors
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A specialized `Error` for storage errors
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A fallback error type when no variant matches
    #[error("Generic {} error: {}", store, source)]
    Generic {
        /// The store this error originated from
        store: &'static str,
        /// The wrapped error
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Error when the object is not found at given location
    #[error("Object at location {} not found: {}", path, source)]
    NotFound {
        /// The path to file
        path: String,
        /// The wrapped error
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Error for invalid path
    #[error("Encountered object with invalid path: {}", source)]
    InvalidPath {
        /// The wrapped error
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Invalid request or provider configuration; retrying unchanged cannot succeed.
    #[error("Invalid storage input: {source}")]
    InvalidInput {
        /// The rejected value and validation context.
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Error when the attempted operation is not supported
    #[error("Operation not supported: {}", source)]
    NotSupported {
        /// The wrapped error
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Error when the object already exists
    #[error("Object at location {} already exists: {}", path, source)]
    AlreadyExists {
        /// The path to the
        path: String,
        /// The wrapped error
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Error when the required conditions failed for the operation
    #[error("Request precondition failure for path {}: {}", path, source)]
    Precondition {
        /// The path to the file
        path: String,
        /// The wrapped error
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Error when the object at the location isn't modified
    #[error("Object at location {} not modified: {}", path, source)]
    NotModified {
        /// The path to the file
        path: String,
        /// The wrapped error
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Error when the used credentials don't have enough permission
    /// to perform the requested operation
    #[error(
        "The operation lacked the necessary privileges to complete for path {}: {}",
        path,
        source
    )]
    PermissionDenied {
        /// The path to the file
        path: String,
        /// The wrapped error
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// Error when the used credentials lack valid authentication
    #[error(
        "The operation lacked valid authentication credentials for path {}: {}",
        path,
        source
    )]
    Unauthenticated {
        /// The path to the file
        path: String,
        /// The wrapped error
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
}

impl From<Error> for std::io::Error {
    fn from(e: Error) -> Self {
        let kind = match &e {
            Error::NotFound { .. } => std::io::ErrorKind::NotFound,
            _ => std::io::ErrorKind::Other,
        };
        Self::new(kind, e)
    }
}

impl From<path::Error> for Error {
    fn from(source: path::Error) -> Self {
        Self::InvalidPath {
            source: Box::new(source),
        }
    }
}

impl From<crate::Error> for Error {
    fn from(source: crate::Error) -> Self {
        Self::Generic {
            store: "lance",
            source: Box::new(source),
        }
    }
}
