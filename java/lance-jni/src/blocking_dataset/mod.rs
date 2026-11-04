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

use lance::dataset::Dataset;

pub const NATIVE_DATASET: &str = "nativeDatasetHandle";

#[derive(Clone)]
pub struct BlockingDataset {
    pub(crate) inner: Dataset,
}

impl BlockingDataset {
    pub fn new(dataset: Dataset) -> Self {
        Self { inner: dataset }
    }

    pub fn close(&self) {}
}
