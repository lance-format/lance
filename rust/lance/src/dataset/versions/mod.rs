// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset policies that differ across exact Lance file versions.
//!
//! File grammar belongs to `lance_file::versions`. This module contains only
//! operation-level dataset choices whose behavior actually differs by version.

use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    sync::Arc,
};

use arrow_schema::{DataType, Field as ArrowField};
use datafusion::catalog::Session;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::{StreamExt, TryStreamExt};
use lance_arrow::DataTypeExt;
use lance_core::{
    Error, Result,
    datatypes::{Field, Projection, Schema, SchemaCompareOptions},
};
use lance_datafusion::chunker::{
    break_stream, break_stream_with_sizes, chunk_stream, chunk_stream_with_sizes,
};
use lance_file::{
    version::ConcreteFileVersion,
    versions as file_versions,
    writer::{FileWriter, FileWriterOptions},
};
use lance_index::scalar::seed::IndexSeedWriter;
use lance_io::object_store::ObjectStore;
use lance_io::traits::Writer as ObjectWriter;
use lance_table::format::{DataFile, DataStorageFormat, Fragment, Manifest};
use object_store::path::Path;

use super::Dataset;
use super::fragment::{
    FileFragment, FragReadConfig, GenericFileReader, MetadataMode, V1FragmentReader,
    write::FragmentCreateBuilder,
};
use super::optimize::CompactionOptions;
use super::scanner::{PlannedFilteredScan, Scanner};
use super::schema_evolution::optimize::{
    ChainedNewColumnTransformOptimizer, SqlToAllNullsOptimizer,
};
use super::statistics::FieldStatistics;
use super::utils::SchemaAdapter;
use super::write::{self, GenericWriter, TargetBaseInfo, WriteParams, WriterOptions};
use crate::io::exec::filtered_read::{FilteredReadExec, FilteredReadOptions};
use crate::io::exec::{
    AddRowAddrExec, FilterPlan as ExprFilterPlan, LanceScanConfig, LanceStream, TakeExec,
};

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

pub fn schema_compare_options(version: ConcreteFileVersion) -> SchemaCompareOptions {
    match version {
        ConcreteFileVersion::V1 => SchemaCompareOptions {
            compare_dictionary: true,
            ..Default::default()
        },
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => SchemaCompareOptions::default(),
    }
}

async fn create_seed_writers(
    version: ConcreteFileVersion,
    dataset: Option<&Dataset>,
    params: &WriteParams,
) -> Result<Vec<Box<dyn IndexSeedWriter>>> {
    match version {
        ConcreteFileVersion::V1 => Ok(Vec::new()),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => write::create_seed_writers_current(dataset, params).await,
    }
}

fn create_current_file_writer(
    version: ConcreteFileVersion,
    object_writer: Box<dyn ObjectWriter>,
    schema: Schema,
    filename: String,
    base_id: Option<u32>,
) -> Result<(FileWriter, DataFile)> {
    let writer =
        file_versions::create_writer(version, object_writer, schema, FileWriterOptions::default())?;
    let mut data_file = DataFile::new_unstarted(filename, version);
    data_file.base_id = base_id;
    Ok((writer, data_file))
}

#[allow(clippy::too_many_arguments)]
pub async fn write_fragments(
    version: ConcreteFileVersion,
    dataset: Option<&Dataset>,
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    normalized_schema: Schema,
    data: SendableRecordBatchStream,
    params: WriteParams,
    target_bases_info: Option<Vec<TargetBaseInfo>>,
    file_row_counts: Option<Vec<usize>>,
) -> Result<(Vec<Fragment>, Schema)> {
    let version_name = format!("{version:?}");
    let schema = write::prepare_write_schema(
        dataset,
        normalized_schema,
        &params,
        schema_compare_options(version),
    )?;
    match version {
        ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0 | ConcreteFileVersion::V2_1 => {
            write::validate_legacy_blob_write_schema(&schema, &version_name)?;
        }
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => {
            write::validate_blob_v2_write_schema(&schema)?;
        }
    }
    let seed_writers = create_seed_writers(version, dataset, &params).await?;
    let fragments = write_fragments_direct(
        version,
        dataset,
        object_store,
        base_dir,
        &schema,
        data,
        params,
        target_bases_info,
        seed_writers,
        file_row_counts,
    )
    .await?;
    Ok((fragments, schema))
}

#[allow(clippy::too_many_arguments)]
pub async fn write_fragments_direct(
    version: ConcreteFileVersion,
    dataset: Option<&Dataset>,
    object_store: Arc<ObjectStore>,
    base_dir: &Path,
    schema: &Schema,
    data: SendableRecordBatchStream,
    params: WriteParams,
    target_bases_info: Option<Vec<TargetBaseInfo>>,
    seed_writers: Vec<Box<dyn IndexSeedWriter>>,
    file_row_counts: Option<Vec<usize>>,
) -> Result<Vec<Fragment>> {
    let adapter = SchemaAdapter::new(data.schema());
    let data = adapter.to_physical_stream(data);
    let buffered_reader = if let Some(file_row_counts) = file_row_counts.as_ref() {
        if file_row_counts.contains(&0) {
            return Err(Error::invalid_input(
                "File row counts must be greater than zero",
            ));
        }
        match version {
            ConcreteFileVersion::V1 => {
                if params.max_rows_per_group == 0 {
                    return Err(Error::invalid_input(
                        "max_rows_per_group must be greater than zero when file row counts are specified",
                    ));
                }
                let max_rows_per_group = params.max_rows_per_group;
                let batch_row_counts =
                    file_row_counts
                        .clone()
                        .into_iter()
                        .flat_map(move |file_rows| {
                            (0..file_rows)
                                .step_by(max_rows_per_group)
                                .map(move |offset| (file_rows - offset).min(max_rows_per_group))
                        });
                chunk_stream_with_sizes(data, batch_row_counts)
            }
            ConcreteFileVersion::V2_0
            | ConcreteFileVersion::V2_1
            | ConcreteFileVersion::V2_2
            | ConcreteFileVersion::V2_3 => break_stream_with_sizes(data, file_row_counts.clone()),
        }
    } else {
        match version {
            ConcreteFileVersion::V1 => chunk_stream(data, params.max_rows_per_group),
            ConcreteFileVersion::V2_0
            | ConcreteFileVersion::V2_1
            | ConcreteFileVersion::V2_2
            | ConcreteFileVersion::V2_3 => break_stream(data, params.max_rows_per_file)
                .map_ok(|batch| vec![batch])
                .boxed(),
        }
    };
    let external_base_resolver = match version {
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => {
            write::blob_v2_external_base_resolver(dataset, &params, schema).await?
        }
        ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0 | ConcreteFileVersion::V2_1 => None,
    };
    write::do_write_fragments_impl(
        dataset,
        object_store,
        base_dir,
        schema,
        buffered_reader,
        params,
        move |object_store, schema, base_dir, options| async move {
            open_writer(version, &object_store, &schema, &base_dir, options).await
        },
        external_base_resolver,
        target_bases_info,
        seed_writers,
        file_row_counts,
    )
    .await
}

fn binary_copy_files_match(fragments: &[Fragment], expected: ConcreteFileVersion) -> Result<bool> {
    for fragment in fragments {
        for data_file in &fragment.files {
            if data_file.file_version()? != expected {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

pub async fn can_use_binary_copy(
    version: ConcreteFileVersion,
    dataset: &Dataset,
    options: &CompactionOptions,
    fragments: &[Fragment],
) -> Result<bool> {
    match version {
        ConcreteFileVersion::V1 => Ok(false),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => {
            if !binary_copy_files_match(fragments, version)? {
                return Ok(false);
            }
            super::optimize::can_use_binary_copy_current(dataset, options, fragments).await
        }
    }
}

pub async fn rewrite_files_binary_copy(
    version: ConcreteFileVersion,
    dataset: &Dataset,
    fragments: &[Fragment],
    params: &WriteParams,
    read_batch_bytes: Option<usize>,
) -> Result<Vec<Fragment>> {
    match version {
        ConcreteFileVersion::V1 => Err(Error::not_supported(
            "binary-copy compaction is not supported for Lance file version 1".to_string(),
        )),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => {
            super::optimize::binary_copy::rewrite_files_binary_copy(
                version,
                dataset,
                fragments,
                params,
                read_batch_bytes,
            )
            .await
        }
    }
}

pub fn check_manifest_storage_version(manifest: &mut Manifest) -> Result<()> {
    let version = manifest.data_storage_format.lance_file_format();
    match version {
        ConcreteFileVersion::V1 => repair_legacy_manifest_storage(manifest),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => validate_exact_manifest_storage(manifest, version),
    }
}

pub fn validate_column_indices(manifest: &Manifest) -> Result<()> {
    match manifest.data_storage_format.lance_file_format() {
        ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0 => Ok(()),
        ConcreteFileVersion::V2_1 | ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => {
            validate_leaf_column_indices(manifest)
        }
    }
}

fn validate_leaf_column_indices(manifest: &Manifest) -> Result<()> {
    let mut fields_by_id: HashMap<i32, (&Field, bool)> = HashMap::new();
    for field in manifest.schema.fields_pre_order() {
        let needs_column = field.is_leaf() || field.is_packed_struct() || field.is_blob();
        fields_by_id
            .entry(field.id)
            .or_insert((field, needs_column));
    }

    let mut validated_lists: HashSet<(usize, usize)> = HashSet::new();

    for fragment in manifest.fragments.iter() {
        for data_file in &fragment.files {
            let file_version = data_file.file_version()?;
            if file_version == ConcreteFileVersion::V1 || data_file.column_indices.is_empty() {
                continue;
            }
            if data_file.fields.len() != data_file.column_indices.len() {
                return Err(Error::invalid_input(format!(
                    "Data file '{}' (fragment {}) has {} field ids but {} column indices. These must be the same length.",
                    data_file.path,
                    fragment.id,
                    data_file.fields.len(),
                    data_file.column_indices.len()
                )));
            }
            if file_version == ConcreteFileVersion::V2_0 {
                continue;
            }
            let list_key = (
                data_file.fields.as_ptr() as usize,
                data_file.column_indices.as_ptr() as usize,
            );
            if !validated_lists.insert(list_key) {
                continue;
            }
            for (field_id, column_index) in
                data_file.fields.iter().zip(data_file.column_indices.iter())
            {
                let Some((field, needs_column)) = fields_by_id.get(field_id).copied() else {
                    continue;
                };
                if needs_column && *column_index == -1 {
                    return Err(Error::invalid_input(format!(
                        "Field '{}' (id={}) in data file '{}' (fragment {}) has column_index=-1, but leaf fields, packed structs, and blob fields must have a valid column index in file format 2.1+.",
                        field.name, field_id, data_file.path, fragment.id
                    )));
                }
                if !needs_column && *column_index != -1 {
                    return Err(Error::invalid_input(format!(
                        "Non-leaf field '{}' (id={}) in data file '{}' (fragment {}) has column_index={}, but non-leaf fields should have column_index=-1 in file format 2.1+.",
                        field.name, field_id, data_file.path, fragment.id, column_index
                    )));
                }
            }
        }
    }
    Ok(())
}

pub async fn write_fragment(
    version: ConcreteFileVersion,
    builder: &FragmentCreateBuilder<'_>,
    stream: SendableRecordBatchStream,
    schema: Schema,
    id: u64,
) -> Result<Fragment> {
    match version {
        ConcreteFileVersion::V1 => builder.write_v1_impl(stream, schema, id).await,
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => {
            builder
                .write_current_impl(
                    move |object_writer, schema, filename| {
                        create_current_file_writer(version, object_writer, schema, filename, None)
                    },
                    stream,
                    schema,
                    id,
                )
                .await
        }
    }
}

pub async fn open_writer(
    version: ConcreteFileVersion,
    object_store: &ObjectStore,
    schema: &Schema,
    base_dir: &Path,
    options: WriterOptions,
) -> Result<Box<dyn GenericWriter>> {
    match version {
        ConcreteFileVersion::V1 => {
            write::open_v1_writer(object_store, schema, base_dir, options).await
        }
        ConcreteFileVersion::V2_0 | ConcreteFileVersion::V2_1 => {
            write::open_current_writer(
                move |object_writer, schema, filename, base_id| {
                    create_current_file_writer(version, object_writer, schema, filename, base_id)
                },
                object_store,
                schema,
                base_dir,
                options,
            )
            .await
        }
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => {
            write::open_current_blob_v2_writer(
                move |object_writer, schema, filename, base_id| {
                    create_current_file_writer(version, object_writer, schema, filename, base_id)
                },
                object_store,
                schema,
                base_dir,
                options,
            )
            .await
        }
    }
}

pub async fn open_update_writer(
    version: ConcreteFileVersion,
    dataset: &Dataset,
    schema: &Schema,
    allow_external_blob_outside_bases: bool,
) -> Result<Box<dyn GenericWriter>> {
    let external_base_resolver = match version {
        ConcreteFileVersion::V2_2 | ConcreteFileVersion::V2_3 => {
            write::blob_v2_external_base_resolver(Some(dataset), &WriteParams::default(), schema)
                .await?
        }
        ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0 | ConcreteFileVersion::V2_1 => None,
    };
    open_writer(
        version,
        &dataset.object_store,
        schema,
        &dataset.base,
        WriterOptions::update(
            dataset.session.store_registry(),
            external_base_resolver,
            allow_external_blob_outside_bases,
        ),
    )
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
    if file_version != dataset_version {
        return Err(Error::invalid_input(format!(
            "File version mismatch. Dataset version: {:?} Fragment version: {:?}",
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

pub async fn open_file_reader(
    version: ConcreteFileVersion,
    fragment: &FileFragment,
    data_file: &DataFile,
    projection: Option<&Schema>,
    read_config: &FragReadConfig,
    metadata_mode: MetadataMode,
) -> Result<Option<Box<dyn GenericFileReader>>> {
    match version {
        ConcreteFileVersion::V1 => fragment.open_v1_file_reader(data_file, projection).await,
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => {
            fragment
                .open_current_file_reader(data_file, projection, read_config, metadata_mode)
                .await
        }
    }
}

pub async fn open_v1_fragment_reader(
    fragment: &FileFragment,
    projection: &Schema,
    read_config: &FragReadConfig,
) -> Result<V1FragmentReader> {
    for data_file in &fragment.metadata().files {
        let actual = data_file.file_version()?;
        if actual != ConcreteFileVersion::V1 {
            return Err(Error::invalid_input(format!(
                "Cannot open file {} with the v1 reader because it has version {}",
                data_file.path, actual
            )));
        }
    }
    fragment
        .open_v1_fragment_reader(projection, read_config)
        .await
}

pub async fn row_group_size_for_rewrite(
    version: ConcreteFileVersion,
    fragment: &FileFragment,
) -> Result<Option<u32>> {
    match version {
        ConcreteFileVersion::V1 => {
            let reader = open_v1_fragment_reader(
                fragment,
                fragment.dataset().schema(),
                &FragReadConfig::default(),
            )
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

/// Return whether `version` can store physical nulls for `datatype`.
///
/// The V1 arm is a whitelist rather than a general capability model: it enumerates the
/// types the legacy writer was known to handle when [`validate_nulls`] was added for
/// the hash-join path. Treat a `false` there as "not known to be supported", and
/// prefer to widen it only alongside a legacy round-trip test.
pub fn supports_nulls(version: ConcreteFileVersion, datatype: &DataType) -> bool {
    match version {
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
    }
}

pub fn validate_nulls(
    version: ConcreteFileVersion,
    datatype: &DataType,
    has_nulls: bool,
) -> Result<()> {
    if has_nulls && !supports_nulls(version, datatype) {
        return Err(Error::invalid_input(format!(
            "Join produced null values for type: {:?}, but storing nulls for this data type is not supported by the dataset's current Lance file format version: {:?}. This can be caused by an explicit null in the new data.",
            datatype, version
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

fn repair_legacy_manifest_storage(manifest: &mut Manifest) -> Result<()> {
    let declared = manifest.data_storage_format.lance_file_format();
    if let Some(actual) = Fragment::try_infer_version(&manifest.fragments)
        .map_err(|error| {
            Error::internal(format!(
                "The dataset contains a mixture of file versions. You will need to rollback to an earlier version: {error}"
            ))
        })?
        && actual != ConcreteFileVersion::V1
    {
        log::warn!(
            "Data storage version {} is less than the actual file version {}. This has been automatically updated.",
            declared,
            actual
        );
        manifest.data_storage_format = DataStorageFormat::new(actual);
    }
    Ok(())
}

fn validate_exact_manifest_storage(
    manifest: &Manifest,
    expected: ConcreteFileVersion,
) -> Result<()> {
    if let Some(actual) = Fragment::try_infer_version(&manifest.fragments)?
        && actual != expected
    {
        return Err(Error::internal(format!(
            "The operation added files with version {}. However, the data storage version is {}.",
            actual, expected
        )));
    }
    Ok(())
}
