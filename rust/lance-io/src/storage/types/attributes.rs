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

use std::borrow::Cow;
use std::collections::HashMap;

/// Additional object attribute types
#[non_exhaustive]
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub enum Attribute {
    /// Specifies how the object should be handled by a browser
    ///
    /// See [Content-Disposition](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Disposition)
    ContentDisposition,
    /// Specifies the encodings applied to the object
    ///
    /// See [Content-Encoding](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Encoding)
    ContentEncoding,
    /// Specifies the language of the object
    ///
    /// See [Content-Language](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Language)
    ContentLanguage,
    /// Specifies the MIME type of the object
    ///
    /// This takes precedence over the provider's default content type.
    ///
    /// See [Content-Type](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Type)
    ContentType,
    /// Overrides cache control policy of the object
    ///
    /// See [Cache-Control](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control)
    CacheControl,
    /// Specifies the storage class of the object.
    ///
    /// See [AWS](https://aws.amazon.com/s3/storage-classes/),
    /// [GCP](https://cloud.google.com/storage/docs/storage-classes), and
    /// [Azure](https://learn.microsoft.com/en-us/rest/api/storageservices/set-blob-tier).
    /// `StorageClass` is used as the name for this attribute because 2 of the 3 storage providers
    /// use that name
    StorageClass,
    /// Specifies a user-defined metadata field for the object
    ///
    /// The String is a user-defined key
    Metadata(Cow<'static, str>),
}

/// Object attributes keyed by their storage meaning. Values may borrow static strings.
pub type Attributes = HashMap<Attribute, Cow<'static, str>>;
