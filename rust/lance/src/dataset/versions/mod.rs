// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset policies that differ across exact Lance file versions.
//!
//! File grammar belongs to `lance_file::versions`. This module contains only
//! operation-level dataset choices whose behavior actually differs by version.

use std::{collections::HashMap, ops::Range, sync::Arc};

use arrow_schema::{DataType, Field as ArrowField};
use datafusion::catalog::Session;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use lance_arrow::DataTypeExt;
use lance_core::{
    Error, Result,
    datatypes::{Field, Projection, Schema},
};
use lance_file::version::ConcreteFileVersion;
use lance_table::format::Fragment;

use super::Dataset;
use super::fragment::FileFragment;
use super::scanner::{PlannedFilteredScan, Scanner};
use super::schema_evolution::optimize::{
    ChainedNewColumnTransformOptimizer, SqlToAllNullsOptimizer,
};
use super::statistics::FieldStatistics;
use super::write::GenericWriter;
use crate::io::exec::filtered_read::{FilteredReadExec, FilteredReadOptions};
use crate::io::exec::{
    AddRowAddrExec, FilterPlan as ExprFilterPlan, LanceScanConfig, LanceStream, TakeExec,
};
pub use lance_table_runtime::write::{schema_compare_options, writes_blob_v2};

pub use lance_table_runtime::fragment::validate_write_version;

#[allow(clippy::too_many_arguments)]
pub fn create_scan_stream(
    version: ConcreteFileVersion,
    dataset: Arc<Dataset>,
    fragments: Arc<Vec<Fragment>>,
    offsets: Option<Range<u64>>,
    projection: Arc<Schema>,
    config: LanceScanConfig,
    metrics: &ExecutionPlanMetricsSet,
    partition: usize,
) -> datafusion::error::Result<LanceStream> {
    match version {
        ConcreteFileVersion::V1 => LanceStream::try_new_v1(
            dataset, fragments, offsets, projection, config, metrics, partition,
        ),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => LanceStream::try_new_v2(
            dataset, fragments, offsets, projection, config, metrics, partition,
        ),
    }
}

pub async fn open_update_writer(
    version: ConcreteFileVersion,
    dataset: &Dataset,
    schema: &Schema,
    allow_external_blob_outside_bases: bool,
) -> Result<Box<dyn GenericWriter>> {
    let table = dataset.table_view().to_table();
    Box::pin(lance_table_runtime::write::open_update_writer(
        version,
        &table,
        schema,
        allow_external_blob_outside_bases,
    ))
    .await
}

pub async fn create_fragment_from_file(
    file_version: ConcreteFileVersion,
    dataset_version: ConcreteFileVersion,
    filename: &str,
    dataset: &Dataset,
    fragment_id: usize,
    physical_rows: Option<usize>,
) -> Result<Fragment> {
    let same_family = matches!(
        (file_version, dataset_version),
        (ConcreteFileVersion::V1, ConcreteFileVersion::V1)
            | (
                ConcreteFileVersion::V2_0
                    | ConcreteFileVersion::V2_1
                    | ConcreteFileVersion::V2_2
                    | ConcreteFileVersion::V2_3,
                ConcreteFileVersion::V2_0
                    | ConcreteFileVersion::V2_1
                    | ConcreteFileVersion::V2_2
                    | ConcreteFileVersion::V2_3
            )
    );
    if !same_family {
        return Err(Error::invalid_input(format!(
            "File version family mismatch. Dataset default: {:?} Fragment version: {:?}",
            dataset_version, file_version
        )));
    }
    match file_version {
        ConcreteFileVersion::V1 => {
            FileFragment::create_from_v1_file(filename, dataset, fragment_id, physical_rows).await
        }
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => {
            FileFragment::create_from_current_file(filename, dataset, fragment_id).await
        }
    }
}

pub fn index_file_version(version: ConcreteFileVersion) -> ConcreteFileVersion {
    match version {
        ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0 => ConcreteFileVersion::V2_0,
        ConcreteFileVersion::V2_1 => ConcreteFileVersion::V2_1,
        ConcreteFileVersion::V2_2 => ConcreteFileVersion::V2_2,
        ConcreteFileVersion::V2_3 => ConcreteFileVersion::V2_3,
    }
}

pub async fn row_group_size_for_rewrite(
    version: ConcreteFileVersion,
    fragment: &FileFragment,
) -> Result<Option<u32>> {
    match version {
        ConcreteFileVersion::V1 => {
            let reader = fragment
                .open_v1_fragment_reader(fragment.schema(), &Default::default())
                .await?;
            Ok(reader.num_rows_in_batch(0))
        }
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => Ok(None),
    }
}

pub fn is_upcast_downcast(
    version: ConcreteFileVersion,
    from_type: &DataType,
    to_type: &DataType,
) -> bool {
    is_upcast_downcast_impl(
        from_type,
        to_type,
        !matches!(version, ConcreteFileVersion::V1),
    )
}

fn is_upcast_downcast_impl(
    from_type: &DataType,
    to_type: &DataType,
    dictionary_materialization: bool,
) -> bool {
    use DataType::*;
    match (from_type, to_type) {
        (_, Dictionary(_, _)) if !dictionary_materialization => false,
        (Dictionary(_, from_value_type), _) => {
            is_upcast_downcast_impl(from_value_type, to_type, dictionary_materialization)
        }
        (_, Dictionary(_, to_value_type)) => {
            is_upcast_downcast_impl(from_type, to_value_type, dictionary_materialization)
        }
        (from, to) if from.is_integer() => to.is_integer(),
        (from, to) if from.is_floating() => to.is_floating(),
        (from, to) if from.is_temporal() => to.is_temporal(),
        (Boolean, to) => matches!(to, Boolean),
        (Utf8 | LargeUtf8, to) => matches!(to, Utf8 | LargeUtf8),
        (Binary | LargeBinary, to) => matches!(to, Binary | LargeBinary),
        (Decimal128(_, _) | Decimal256(_, _), to) => {
            matches!(to, Decimal128(_, _) | Decimal256(_, _))
        }
        (List(from_field) | LargeList(from_field) | FixedSizeList(from_field, _), to_type) => {
            match to_type {
                List(to_field) | LargeList(to_field) | FixedSizeList(to_field, _) => {
                    is_upcast_downcast_impl(
                        from_field.data_type(),
                        to_field.data_type(),
                        dictionary_materialization,
                    )
                }
                _ => false,
            }
        }
        _ => false,
    }
}

pub fn validate_nulls(
    version: ConcreteFileVersion,
    column_name: &str,
    datatype: &DataType,
    has_nulls: bool,
) -> Result<()> {
    let supported = match version {
        ConcreteFileVersion::V1 => matches!(
            datatype,
            DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Binary
                | DataType::List(_)
                | DataType::FixedSizeBinary(_)
                | DataType::FixedSizeList(_, _)
        ),
        ConcreteFileVersion::V2_0 => !matches!(datatype, DataType::Struct(..)),
        ConcreteFileVersion::V2_1 | ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => true,
    };
    if has_nulls && !supported {
        return Err(Error::invalid_input(format!(
            "Column '{}' has null values of type: {:?}, but storing nulls for this data type is not supported by the dataset's current Lance file format version: {:?}. This can be caused by an explicit null in the new data.",
            column_name, datatype, version
        )));
    }
    Ok(())
}

fn reject_nested_column_add(field: &ArrowField, version: ConcreteFileVersion) -> Result<()> {
    Err(Error::invalid_input(format!(
        "Column {} is a struct col, add sub column is not supported in Lance file version {}",
        field.name(),
        version
    )))
}

fn reject_nested_v1(field: &ArrowField) -> Result<()> {
    reject_nested_column_add(field, ConcreteFileVersion::V1)
}

fn reject_nested_v2_0(field: &ArrowField) -> Result<()> {
    reject_nested_column_add(field, ConcreteFileVersion::V2_0)
}

fn reject_nested_v2_1(field: &ArrowField) -> Result<()> {
    reject_nested_column_add(field, ConcreteFileVersion::V2_1)
}

fn allow_nested(_field: &ArrowField) -> Result<()> {
    Ok(())
}

pub fn check_field_conflict(
    version: ConcreteFileVersion,
    left: &ArrowField,
    right: &ArrowField,
) -> Result<()> {
    let validate = match version {
        ConcreteFileVersion::V1 => reject_nested_v1,
        ConcreteFileVersion::V2_0 => reject_nested_v2_0,
        ConcreteFileVersion::V2_1 => reject_nested_v2_1,
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => allow_nested,
    };
    super::schema_evolution::check_field_conflict_with(left, right, validate)
}

fn exclude_struct_field(field: &Field, other: &Field) -> Option<Field> {
    field
        .data_type()
        .is_struct()
        .then(|| field.exclude(other))
        .flatten()
}

fn exclude_nested_field(field: &Field, other: &Field) -> Option<Field> {
    field
        .data_type()
        .is_nested()
        .then(|| field.exclude(other))
        .flatten()
}

pub fn exclude_schema(
    version: ConcreteFileVersion,
    source: &Schema,
    other: &Schema,
) -> Result<Schema> {
    let exclude = match version {
        ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0 | ConcreteFileVersion::V2_1 => {
            exclude_struct_field
        }
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => exclude_nested_field,
    };
    super::schema_evolution::exclude_with(source, other, exclude)
}

pub fn configure_new_column_optimizers(
    version: ConcreteFileVersion,
    optimizer: &mut ChainedNewColumnTransformOptimizer,
) {
    match version {
        ConcreteFileVersion::V1 => {}
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => {
            optimizer.add_optimizer(Box::new(SqlToAllNullsOptimizer::new()));
        }
    }
}

pub fn validate_metadata_only_null_columns(version: ConcreteFileVersion) -> Result<()> {
    match version {
        ConcreteFileVersion::V1 => Err(Error::not_supported_source(
            "Cannot add all-null columns to legacy dataset version.".into(),
        )),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => Ok(()),
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::dataset) async fn filtered_read(
    version: ConcreteFileVersion,
    scanner: &Scanner,
    filter_plan: &ExprFilterPlan,
    projection: Projection,
    make_deletions_null: bool,
    fragments: Option<Arc<Vec<Fragment>>>,
    scan_range: Option<Range<u64>>,
    is_prefilter: bool,
    session: Option<&dyn Session>,
) -> Result<PlannedFilteredScan> {
    match version {
        ConcreteFileVersion::V1 => {
            scanner
                .legacy_filtered_read(
                    filter_plan,
                    projection,
                    make_deletions_null,
                    fragments,
                    scan_range,
                    is_prefilter,
                )
                .await
        }
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => {
            let limit_pushed_down = scan_range.is_some();
            let plan = scanner
                .new_filtered_read(
                    filter_plan,
                    projection,
                    make_deletions_null,
                    fragments,
                    scan_range,
                    session,
                )
                .await?;
            Ok(PlannedFilteredScan {
                filter_pushed_down: true,
                limit_pushed_down,
                plan,
            })
        }
    }
}

pub fn take(
    version: ConcreteFileVersion,
    scanner: &Scanner,
    input: Arc<dyn ExecutionPlan>,
    output_projection: Projection,
) -> Result<Arc<dyn ExecutionPlan>> {
    match version {
        ConcreteFileVersion::V1 => scanner.take_legacy(input, output_projection),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => scanner.take_current(input, output_projection),
    }
}

pub async fn collect_data_stats(
    version: ConcreteFileVersion,
    dataset: &Arc<Dataset>,
    field_stats: &mut HashMap<u32, FieldStatistics>,
) -> Result<()> {
    match version {
        ConcreteFileVersion::V1 => Ok(()),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => {
            super::statistics::collect_current_data_stats(dataset, field_stats).await
        }
    }
}

pub fn merge_insert_indexed_take(
    version: ConcreteFileVersion,
    dataset: Arc<Dataset>,
    mut index_mapper: Arc<dyn ExecutionPlan>,
    projection: Projection,
    add_row_addr: bool,
) -> Result<Arc<dyn ExecutionPlan>> {
    match version {
        ConcreteFileVersion::V1 => {
            if add_row_addr {
                let position = index_mapper.schema().fields().len();
                index_mapper = Arc::new(AddRowAddrExec::try_new(
                    index_mapper,
                    dataset.clone(),
                    position,
                )?);
            }
            Ok(Arc::new(
                TakeExec::try_new(dataset, index_mapper, projection)?.ok_or_else(|| {
                    Error::internal("merge-insert legacy take unexpectedly needed no columns")
                })?,
            ))
        }
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => {
            let mut projection = projection.with_row_id();
            if add_row_addr {
                projection = projection.with_row_addr();
            }
            Ok(Arc::new(FilteredReadExec::try_new(
                dataset,
                FilteredReadOptions::new(projection),
                Some(index_mapper),
            )?))
        }
    }
}

pub fn validate_row_stream_read(version: ConcreteFileVersion) -> Result<()> {
    match version {
        ConcreteFileVersion::V1 => Err(Error::not_supported_source(
            "taking rows through FilteredReadExec requires the v2 storage format"
                .to_string()
                .into(),
        )),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => Ok(()),
    }
}
