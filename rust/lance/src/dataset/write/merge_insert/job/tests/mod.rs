// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod indexed;
mod plans;
mod spilled_lineage;
mod units;

pub(super) use super::super::tests::{create_new_batch, create_test_dataset, create_test_schema};
pub(super) use super::super::{MergeInsertBuilder, WhenNotMatched};
pub(super) use super::*;
pub(super) use crate::dataset::optimize::{CompactionOptions, compact_files};
pub(super) use crate::dataset::rowids::{
    INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, SPILL_ROW_LINEAGE_CONFIG_KEY,
};
pub(super) use crate::dataset::scanner::ColumnOrdering;
pub(super) use crate::dataset::transaction::Operation;
pub(super) use crate::dataset::write::CommitBuilder;
pub(super) use crate::index::DatasetIndexExt;
pub(super) use crate::{
    dataset::{InsertBuilder, ProjectionRequest, UpdateBuilder, WriteDestination, WriteParams},
    utils::test::assert_plan_node_equals,
};
pub(super) use arrow_array::ArrayRef;
pub(super) use arrow_array::RecordBatch;
pub(super) use arrow_array::cast::AsArray;
pub(super) use arrow_array::types::UInt64Type;
pub(super) use arrow_array::{
    Array, Int64Array, RecordBatchIterator, RecordBatchReader, StringArray, UInt32Array,
    types::UInt32Type,
};
pub(super) use arrow_schema::{DataType, Field, Schema};
pub(super) use arrow_select::concat::concat_batches;
pub(super) use datafusion::datasource::MemTable;
pub(super) use datafusion::logical_expr::JoinType;
pub(super) use datafusion::physical_plan::joins::{HashJoinExec, PartitionMode};
pub(super) use datafusion::physical_plan::{StatisticsArgs, StatisticsContext};
pub(super) use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
pub(super) use futures::TryStreamExt;
pub(super) use lance_core::utils::tempfile::TempStrDir;
pub(super) use lance_core::{
    ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION,
};
pub(super) use lance_datafusion::{exec::new_session_context, utils::reader_to_stream};
pub(super) use lance_datagen::{BatchCount, RowCount, Seed, array};
pub(super) use lance_file::version::LanceFileVersion;
pub(super) use lance_index::IndexType;
pub(super) use lance_index::scalar::ScalarIndexParams;
pub(super) use lance_table::format::RowIdMeta;
pub(super) use lance_table::rowids::version::RowDatasetVersionMeta;
pub(super) use std::collections::HashMap;
pub(super) use std::io::Write;
