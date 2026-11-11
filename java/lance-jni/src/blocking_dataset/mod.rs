// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod construction;
pub(crate) mod conversions;
mod index;
mod maintenance;
mod mutation;
mod read;
mod refs;
mod schema;

pub(crate) use index::optimize_options_from_java;

use jni::JNIEnv;
use jni::objects::JObject;
use lance::dataset::Dataset;

use crate::error::Result;

pub const NATIVE_DATASET: &str = "nativeDatasetHandle";

#[derive(Clone)]
pub struct BlockingDataset {
    pub(crate) inner: Dataset,
}

/// Clones the native dataset out of a Java `Dataset` and releases the field's mutex.
///
/// Java runs reads under a shared read lock so they can proceed concurrently, but
/// `get_rust_field` hands out a guard on a per-object mutex. Holding that guard across a read's
/// I/O serializes every concurrent read on the same Java handle. Cloning is cheap: `Dataset`
/// shares its manifest, session, and object store through `Arc`s.
pub(crate) fn clone_dataset(env: &mut JNIEnv, java_dataset: &JObject) -> Result<BlockingDataset> {
    let dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    Ok(dataset_guard.clone())
}

impl BlockingDataset {
    pub fn new(dataset: Dataset) -> Self {
        Self { inner: dataset }
    }

    pub fn close(&self) {}
}
