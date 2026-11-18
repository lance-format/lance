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

use bytes::Bytes;
use std::sync::Arc;

/// A cheaply cloneable, ordered collection of [`Bytes`]
///
/// A [`PutPayload`] is fully materialized in memory. To upload large objects
/// without buffering them entirely in memory, use the multipart API instead:
/// [`StorageBackend::put_multipart_opts`], the Lance object writer
///
/// [`StorageBackend::put_multipart_opts`]: crate::object_store::backend::StorageBackend::put_multipart_opts
#[derive(Debug, Clone, Default)]
pub struct PutPayload(Arc<[Bytes]>);

impl PutPayload {
    /// Creates a [`PutPayload`] from a static slice
    pub fn from_static(s: &'static [u8]) -> Self {
        s.into()
    }

    /// Returns the total length of the [`Bytes`] in this payload
    pub fn content_length(&self) -> usize {
        self.0.iter().map(|b| b.len()).sum()
    }

    /// Returns an iterator over the [`Bytes`] in this payload
    pub fn iter(&self) -> std::slice::Iter<'_, Bytes> {
        self.0.iter()
    }
}

impl From<Bytes> for PutPayload {
    fn from(value: Bytes) -> Self {
        Self(Arc::new([value]))
    }
}

impl From<Vec<u8>> for PutPayload {
    fn from(value: Vec<u8>) -> Self {
        Self(Arc::new([value.into()]))
    }
}

impl From<&'static str> for PutPayload {
    fn from(value: &'static str) -> Self {
        Bytes::from(value).into()
    }
}

impl From<&'static [u8]> for PutPayload {
    fn from(value: &'static [u8]) -> Self {
        Bytes::from(value).into()
    }
}

impl From<String> for PutPayload {
    fn from(value: String) -> Self {
        Bytes::from(value).into()
    }
}

impl FromIterator<Bytes> for PutPayload {
    fn from_iter<T: IntoIterator<Item = Bytes>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl From<PutPayload> for Bytes {
    fn from(value: PutPayload) -> Self {
        match value.0.len() {
            0 => Self::new(),
            1 => value.0[0].clone(),
            _ => {
                let mut buf = Vec::with_capacity(value.content_length());
                value.iter().for_each(|x| buf.extend_from_slice(x));
                buf.into()
            }
        }
    }
}
