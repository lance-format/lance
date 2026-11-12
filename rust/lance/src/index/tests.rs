// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::{WriteMode, WriteParams};
use crate::index::lifecycle::build_index_metadata_from_segments;
use crate::index::loading::load_all_indices_by_name;
use crate::index::segment::unsupported_index_version;
use crate::index::statistics::IndexDescriptionImpl;
use crate::index::vector::VectorIndexParams;
use crate::index::vector::details::{
    target_partition_size_from_details, vector_index_details_default,
};
use crate::session::Session;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount, copy_test_data_to_tmp};
use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, Int32Type, Int64Type};
use arrow_array::Int32Array;
use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::stream::TryStreamExt;
use lance_arrow::*;
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::gen_batch;
use lance_datagen::{BatchCount, ByteCount, Dimension, RowCount, array};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::optimize::OptimizeOptions;
use lance_index::pbold::{BTreeIndexDetails, InvertedIndexDetails};
use lance_index::scalar::bitmap::BITMAP_LOOKUP_NAME;
use lance_index::scalar::inverted::query::{FtsQuery, PhraseQuery};
use lance_index::scalar::inverted::{
    INVERTED_INDEX_VERSION_V1, INVERTED_INDEX_VERSION_V2, INVERTED_INDEX_VERSION_V3,
    InvertedListFormatVersion,
};
use lance_index::scalar::{
    BuiltinIndexType, FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams,
};
use lance_index::vector::{
    hnsw::builder::HnswBuildParams,
    ivf::IvfBuildParams,
    kmeans::{KMeansParams, train_kmeans},
    pq::builder::PQBuildParams,
    sq::builder::SQBuildParams,
};
use lance_index::{
    INDEX_FILE_NAME, IndexDescription, IndexFragmentCoverage, IndexParams, IndexType,
    frag_reuse::FRAG_REUSE_INDEX_NAME,
};
use lance_io::{
    assert_io_eq, assert_io_lt, object_store::WrappingObjectStore, utils::tracking_store::IoStats,
};
use lance_linalg::distance::{DistanceType, MetricType};
use lance_testing::datagen::generate_random_array;
use object_store::ObjectStoreExt;
use roaring::RoaringBitmap;
use rstest::rstest;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, atomic::Ordering},
};
use uuid::Uuid;

use crate::{Dataset, Error};
use lance_table::format::IndexMetadata;
use lance_table::io::manifest::read_manifest_indexes;

mod create;
mod lifecycle;
mod loading_behaviors;
mod merge;
mod nested_fields;
mod optimize;
mod statistics;
pub(super) mod support;
