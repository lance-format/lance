// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Zone map write seeds follow a seeded column into every new data file.
//!
//! When a column has a ZoneMap index with write seeds enabled, each writer
//! that produces a new data file for that column must embed a seed built
//! from the values it wrote. These tests drive every such writer and check
//! that the data file now serving the column carries a seed identical to one
//! built from the column's current values.

use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::types::Float32Type;
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, FixedSizeListArray, Int32Array, Int64Array,
    RecordBatch, RecordBatchIterator, StringArray, StringViewArray, StructArray,
    TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, TimeUnit};
use bytes::Bytes;
use futures::{TryStreamExt, stream};
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::utils::tempfile::TempStrDir;
use lance_datafusion::utils::reader_to_stream;
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::seed::{IndexSeedWriter, SEED_META_KEY_PREFIX};
use lance_index::scalar::zonemap::ZoneMapSeedWriter;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_io::ReadBatchParams;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_table::format::{DataFile, Fragment};

use crate::Dataset;
use crate::blob::{BlobArrayBuilder, blob_field};
use crate::dataset::optimize::CompactionScope;
use crate::dataset::schema_evolution::{ColumnAlteration, NewColumnTransform};
use crate::dataset::transaction::{DataReplacementGroup, Operation, Transaction};
use crate::dataset::write::merge_insert::{WhenMatched, WhenNotMatched};
use crate::dataset::write::seeds::{
    ZONE_WRITE_SEEDS_ENABLED_KEY, ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY,
    ZONE_WRITE_SEEDS_STORE_NULL_BITMAP_KEY,
};
use crate::dataset::{
    CommitBuilder, DATA_DIR, InsertBuilder, MergeInsertBuilder, MergeInsertWriteMode,
    UpdateBuilder, WriteDestination, WriteMode, WriteParams,
};
use crate::index::DatasetIndexExt;

const ROWS_PER_ZONE: u64 = 4;
const ROWS_PER_FILE: usize = 10;

#[rstest::rstest]
#[case::current_seed(true, true, Some(100), false)]
#[case::current_all_null_seed(true, true, None, false)]
#[case::missing_current_seed(true, false, Some(100), false)]
#[case::seeds_disabled(false, false, Some(100), false)]
#[case::shallow_clone(true, true, Some(100), true)]
#[tokio::test]
async fn test_zone_map_seed_harvest_after_repeated_column_replacement(
    #[case] use_seeds: bool,
    #[case] seed_current_file: bool,
    #[case] replacement_value: Option<i32>,
    #[case] shallow_clone: bool,
) {
    let dir = TempStrDir::default();
    let dataset = dataset_with_val_index(dir.as_str(), use_seeds).await;
    let full_schema = dataset.schema().clone();
    let replacement = dataset
        .get_fragment(0)
        .unwrap()
        .write_columns(stream::iter([Ok(rows(0..10, |_| Some(1)))]), &full_schema)
        .await
        .unwrap();
    let read_version = dataset.version_id();
    let mut dataset = Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataReplacement {
            replacements: vec![replacement],
            data_change: true,
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments()[0].files.len(), 1);
    let original_file = dataset.fragments()[0].files[0].path.clone();
    assert_eq!(
        seed_in_file(&dataset, &original_file, "val")
            .await
            .is_some(),
        use_seeds
    );
    let index_name = dataset.load_indices().await.unwrap()[0].name.clone();
    if !seed_current_file {
        dataset.drop_index(&index_name).await.unwrap();
    }
    dataset = replace_column(dataset, "val", |id| id == 0, |_, _| replacement_value).await;
    assert_eq!(dataset.fragments()[0].files.len(), 2);
    let current_file = serving_file(&dataset.fragments()[0], field_id(&dataset, "val"));
    assert_ne!(current_file.path, original_file);
    assert_eq!(
        seed_in_file(&dataset, &current_file.path, "val")
            .await
            .is_some(),
        seed_current_file
    );
    if !seed_current_file {
        // Recreate the index on the other fragments. Maintenance must scan
        // fragment 0's current file, which was written while no index existed.
        let segment = dataset
            .create_index_builder(&["val"], IndexType::ZoneMap, &zone_map_params(use_seeds))
            .name(index_name.clone())
            .fragments(vec![1, 2])
            .execute_uncommitted()
            .await
            .unwrap();
        dataset
            .commit_existing_index_segments(&index_name, "val", vec![segment])
            .await
            .unwrap();
    }
    let clone_dir = TempStrDir::default();
    let uri = if shallow_clone {
        dataset = dataset
            .shallow_clone(clone_dir.as_str(), dataset.version_id(), None)
            .await
            .unwrap();
        assert!(
            dataset.fragments()[0]
                .files
                .iter()
                .all(|file| file.base_id.is_some())
        );
        clone_dir.as_str()
    } else {
        dir.as_str()
    };
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let dataset = Dataset::open(uri).await.unwrap();
    for filter in ["val = 100", "val IS NULL", "val < 10"] {
        let mut flat = dataset.scan();
        flat.use_scalar_index(false).filter(filter).unwrap();
        let expected = flat.try_into_batch().await.unwrap().num_rows();
        if filter == "val = 100" {
            assert_eq!(expected, if replacement_value.is_some() { 10 } else { 0 });
        }
        let mut indexed = dataset.scan();
        indexed.filter(filter).unwrap();
        assert_eq!(
            indexed.try_into_batch().await.unwrap().num_rows(),
            expected,
            "{filter}"
        );
    }
}

#[rstest::rstest]
#[case::same_name("val")]
#[case::different_name("other")]
#[tokio::test]
async fn test_zone_map_seed_harvest_respects_field_identity(#[case] top_level_name: &str) {
    let dir = TempStrDir::default();
    let child_field = Arc::new(ArrowField::new("val", DataType::Int32, false));
    let nested = StructArray::from(vec![(
        child_field,
        Arc::new(Int32Array::from(vec![100, 100])) as ArrayRef,
    )]);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(top_level_name, DataType::Int32, false),
        ArrowField::new("st", nested.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 1])), Arc::new(nested)],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], schema.clone()),
        dir.as_str(),
        None,
    )
    .await
    .unwrap();
    for column in [top_level_name, "st.val"] {
        dataset
            .create_index(
                &[column],
                IndexType::ZoneMap,
                None,
                &zone_map_params(true),
                false,
            )
            .await
            .unwrap();
    }
    dataset
        .append(RecordBatchIterator::new([Ok(batch)], schema), None)
        .await
        .unwrap();
    assert_eq!(dataset.fragments().len(), 2);
    let appended = &dataset.fragments()[1].files[0];
    assert!(
        seed_in_file(&dataset, &appended.path, top_level_name)
            .await
            .is_some()
    );
    assert!(
        seed_in_file(&dataset, &appended.path, "st.val")
            .await
            .is_none()
    );

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    for indexed in [false, true] {
        let mut scan = dataset.scan();
        scan.use_scalar_index(indexed)
            .filter("st.val = 100")
            .unwrap();
        assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), 4);
    }
}

#[tokio::test]
async fn test_zone_map_seed_harvest_after_column_name_swap() {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_val_index(dir.as_str(), true).await;
    dataset
        .create_index(
            &["id"],
            IndexType::ZoneMap,
            None,
            &zone_map_params(true),
            false,
        )
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let replacement_schema = dataset.schema().project(&["id", "val"]).unwrap();
    let arrow_schema = Arc::new(ArrowSchema::from(&replacement_schema));
    let batch = RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int32Array::from(vec![1; 10])),
            Arc::new(Int32Array::from(vec![100; 10])),
        ],
    )
    .unwrap();
    let replacement = dataset
        .get_fragment(0)
        .unwrap()
        .write_columns(stream::iter([Ok(batch)]), &replacement_schema)
        .await
        .unwrap();
    let read_version = dataset.version_id();
    dataset = Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataReplacement {
            replacements: vec![replacement],
            data_change: true,
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments()[0].files.len(), 2);
    let current_file = serving_file(&dataset.fragments()[0], field_id(&dataset, "val"));
    assert!(
        seed_in_file(&dataset, &current_file.path, "id")
            .await
            .is_some()
    );
    assert!(
        seed_in_file(&dataset, &current_file.path, "val")
            .await
            .is_some()
    );
    assert!(
        seed_in_file(&dataset, &dataset.fragments()[0].files[0].path, "val")
            .await
            .is_none()
    );
    dataset
        .alter_columns(&[
            ColumnAlteration::new("id".into()).rename("val".into()),
            ColumnAlteration::new("val".into()).rename("id".into()),
        ])
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    let mut flat = dataset.scan();
    flat.use_scalar_index(false).filter("id = 100").unwrap();
    let expected = flat.try_into_batch().await.unwrap().num_rows();
    assert_eq!(expected, 10);
    let mut indexed = dataset.scan();
    indexed.filter("id = 100").unwrap();
    assert_eq!(indexed.try_into_batch().await.unwrap().num_rows(), expected);
}

/// A zone map on a blob column itself is built from what the scan yields
/// (the blob descriptor), on creation and on a later update.
#[tokio::test]
async fn test_zone_map_on_blob_column() {
    let dir = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        blob_field("blob", true),
    ]));
    let make_batch = |start: i32| {
        let mut blobs = BlobArrayBuilder::new(10);
        for i in 0..10 {
            if i % 3 == 0 {
                blobs.push_null().unwrap();
            } else {
                blobs.push_bytes(b"blob").unwrap();
            }
        }
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(start..start + 10)),
                blobs.finish().unwrap(),
            ],
        )
        .unwrap()
    };
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(make_batch(0))], schema.clone()),
        dir.as_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["blob"],
            IndexType::ZoneMap,
            None,
            &zone_map_params(true),
            false,
        )
        .await
        .unwrap();
    dataset
        .append(RecordBatchIterator::new([Ok(make_batch(10))], schema), None)
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    for filter in ["blob IS NULL", "blob IS NOT NULL"] {
        let mut flat = dataset.scan();
        flat.use_scalar_index(false)
            .project(&["id"])
            .unwrap()
            .filter(filter)
            .unwrap();
        let expected = flat.try_into_batch().await.unwrap().num_rows();
        let mut indexed = dataset.scan();
        indexed.project(&["id"]).unwrap().filter(filter).unwrap();
        assert_eq!(
            indexed.try_into_batch().await.unwrap().num_rows(),
            expected,
            "{filter}"
        );
    }
}

#[rstest::rstest]
#[case::blob_first(true, false)]
#[case::blob_last(false, false)]
#[case::blob_first_renamed(true, true)]
#[tokio::test]
async fn test_zone_map_seed_harvest_with_blob_columns(
    #[case] blob_first: bool,
    #[case] rename: bool,
) {
    let dir = TempStrDir::default();
    let mut fields = ["a", "b", "c", "d", "e", "f"]
        .map(|name| ArrowField::new(name, DataType::Int32, false))
        .to_vec();
    let blob_position = if blob_first { 0 } else { fields.len() };
    fields.insert(blob_position, blob_field("blob", false));
    let schema = Arc::new(ArrowSchema::new(fields));
    let make_batch = || {
        let mut blobs = BlobArrayBuilder::new(10);
        for _ in 0..10 {
            blobs.push_bytes(b"blob").unwrap();
        }
        let mut arrays = [1, 2, 3, 4, 5, 100]
            .map(|v| Arc::new(Int32Array::from(vec![v; 10])) as ArrayRef)
            .to_vec();
        arrays.insert(blob_position, blobs.finish().unwrap());
        RecordBatch::try_new(schema.clone(), arrays).unwrap()
    };
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(make_batch())], schema.clone()),
        dir.as_str(),
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    for name in ["a", "f"] {
        dataset
            .create_index(
                &[name],
                IndexType::ZoneMap,
                None,
                &zone_map_params(true),
                false,
            )
            .await
            .unwrap();
    }
    dataset
        .append(RecordBatchIterator::new([Ok(make_batch())], schema), None)
        .await
        .unwrap();
    let file = &dataset.fragments()[1].files[0];
    assert!(seed_in_file(&dataset, &file.path, "a").await.is_some());
    assert!(seed_in_file(&dataset, &file.path, "f").await.is_some());
    let column = if rename {
        dataset
            .alter_columns(&[
                ColumnAlteration::new("a".into()).rename("f".into()),
                ColumnAlteration::new("f".into()).rename("a".into()),
            ])
            .await
            .unwrap();
        "a"
    } else {
        "f"
    };
    let filter = format!("{column} = 100");
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let dataset = Dataset::open(dir.as_str()).await.unwrap();
    let mut flat = dataset.scan();
    flat.use_scalar_index(false)
        .project(&[column])
        .unwrap()
        .filter(&filter)
        .unwrap();
    let expected = flat.try_into_batch().await.unwrap().num_rows();
    assert_eq!(expected, 20);
    let mut indexed = dataset.scan();
    indexed.project(&[column]).unwrap().filter(&filter).unwrap();
    assert_eq!(indexed.try_into_batch().await.unwrap().num_rows(), expected);
}

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("val", DataType::Int32, true),
        ArrowField::new("untouched", DataType::Utf8, true),
    ]))
}

fn rows(ids: std::ops::Range<i32>, value: impl Fn(i32) -> Option<i32>) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from_iter_values(ids.clone())),
            Arc::new(Int32Array::from_iter(ids.clone().map(value))),
            Arc::new(StringArray::from_iter_values(ids.map(|i| format!("u{i}")))),
        ],
    )
    .unwrap()
}

fn base_value(i: i32) -> Option<i32> {
    (i % 7 != 3).then_some(i)
}

fn zone_map_params(use_seeds: bool) -> ScalarIndexParams {
    ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
        .with_params(&serde_json::json!({"rows_per_zone": ROWS_PER_ZONE, "use_seeds": use_seeds}))
}

/// Twenty rows in two fragments, a zone map on `val`, then a third fragment
/// appended after the index exists.
async fn dataset_with_val_index(uri: &str, use_seeds: bool) -> Dataset {
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(rows(0..20, base_value))], schema()),
        uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FILE,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["val"],
            IndexType::ZoneMap,
            None,
            &zone_map_params(use_seeds),
            false,
        )
        .await
        .unwrap();
    dataset
        .append(
            RecordBatchIterator::new([Ok(rows(20..30, base_value))], schema()),
            None,
        )
        .await
        .unwrap();
    dataset
}

fn field_id(dataset: &Dataset, column: &str) -> i32 {
    dataset.schema().field(column).unwrap().id
}

/// The newest data file that stores `field_id`.
fn serving_file(fragment: &Fragment, field_id: i32) -> &DataFile {
    fragment
        .files
        .iter()
        .rev()
        .find(|file| {
            file.fields
                .iter()
                .zip(file.column_indices.iter())
                .any(|(id, column_index)| *id == field_id && *column_index >= 0)
        })
        .expect("a data file serves the field")
}

async fn open_data_file(dataset: &Dataset, path: &str) -> FileReader {
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );
    let path = dataset.base.clone().join(DATA_DIR).join(path);
    let file_scheduler = scheduler
        .open_file(&path, &CachedFileSize::unknown())
        .await
        .unwrap();
    FileReader::try_open(
        file_scheduler,
        None,
        Default::default(),
        &dataset.metadata_cache.file_metadata_cache(&path),
        FileReaderOptions::default(),
    )
    .await
    .unwrap()
}

/// The seed stored for `column` in the data file at `path`, if any.
async fn seed_in_file(dataset: &Dataset, path: &str, column: &str) -> Option<Bytes> {
    let reader = open_data_file(dataset, path).await;
    let value = reader
        .metadata()
        .file_schema
        .metadata
        .get(&format!("{SEED_META_KEY_PREFIX}{column}"))?
        .clone();
    let buf_index: u32 = value.split(':').next().unwrap().parse().unwrap();
    Some(reader.read_global_buffer(buf_index).await.unwrap())
}

/// The physical values of `column` stored in the data file at `path`,
/// deleted rows included.
async fn physical_column(dataset: &Dataset, path: &str, column: &str) -> ArrayRef {
    let reader = open_data_file(dataset, path).await;
    let batches: Vec<RecordBatch> = reader
        .read_stream(
            ReadBatchParams::RangeFull,
            1024,
            4,
            FilterExpression::no_filter(),
        )
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let batch = concat_batches(&batches[0].schema(), &batches).unwrap();
    batch.column_by_name(column).unwrap().clone()
}

/// The seed a writer produces from `values`, observed in `batches` chunks.
fn seed_from_values(column: &str, values: &ArrayRef, batches: usize) -> Bytes {
    let mut writer =
        ZoneMapSeedWriter::new(column, ROWS_PER_ZONE, values.data_type().clone()).unwrap();
    let chunk = values.len().div_ceil(batches).max(1);
    let mut offset = 0;
    while offset < values.len() {
        let len = chunk.min(values.len() - offset);
        writer.observe_batch(&values.slice(offset, len)).unwrap();
        offset += len;
    }
    writer.finish().unwrap().unwrap()
}

/// The seed a writer produces from the fragment's current values of `column`.
async fn seed_from_current_values(dataset: &Dataset, fragment: &Fragment, column: &str) -> Bytes {
    let mut scanner = dataset.scan();
    scanner
        .with_fragments(vec![fragment.clone()])
        .project(&[column])
        .unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    let values = batch.column_by_name(column).unwrap();
    let mut writer =
        ZoneMapSeedWriter::new(column, ROWS_PER_ZONE, values.data_type().clone()).unwrap();
    writer.observe_batch(values).unwrap();
    writer.finish().unwrap().unwrap()
}

/// Every fragment in `fragment_ids` stores a seed for `column` in the data
/// file that serves it, and the seed matches the column's current values.
async fn assert_seed_matches_data(dataset: &Dataset, fragment_ids: &[u64], column: &str) {
    let field = field_id(dataset, column);
    for fragment in dataset.fragments().iter() {
        if !fragment_ids.contains(&fragment.id) {
            continue;
        }
        let file = serving_file(fragment, field);
        let stored = seed_in_file(dataset, &file.path, column)
            .await
            .unwrap_or_else(|| panic!("fragment {} has no seed for {column}", fragment.id));
        let expected = seed_from_current_values(dataset, fragment, column).await;
        assert_eq!(stored, expected, "fragment {} seed", fragment.id);
    }
}

/// Stage a replacement for `column` in the fragments selected by `rewrite`
/// and commit it as a data replacement, the way computed and function column
/// refreshes do.
async fn replace_column(
    dataset: Dataset,
    column: &str,
    rewrite: impl Fn(u64) -> bool,
    value: impl Fn(i32, i32) -> Option<i32>,
) -> Dataset {
    replace_column_in_batches(dataset, column, rewrite, value, 1).await
}

/// Like [`replace_column`], streaming each fragment's replacement as
/// `batches` batches so the writer sees zone boundaries inside a batch and
/// across batches.
async fn replace_column_in_batches(
    dataset: Dataset,
    column: &str,
    rewrite: impl Fn(u64) -> bool,
    value: impl Fn(i32, i32) -> Option<i32>,
    batches: usize,
) -> Dataset {
    let column_schema = LanceSchema {
        fields: vec![dataset.schema().field(column).unwrap().clone()],
        metadata: Default::default(),
    };
    let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        column,
        DataType::Int32,
        true,
    )]));
    let mut replacements: Vec<DataReplacementGroup> = Vec::new();
    for fragment in dataset.get_fragments() {
        if !rewrite(fragment.id() as u64) {
            continue;
        }
        let rows = fragment.physical_rows().await.unwrap() as i32;
        let fragment_id = fragment.id() as i32;
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from_iter(
                (0..rows).map(|i| value(fragment_id, i)),
            ))],
        )
        .unwrap();
        let chunk = batch.num_rows().div_ceil(batches).max(1);
        let chunks: Vec<_> = (0..batch.num_rows())
            .step_by(chunk)
            .map(|offset| Ok(batch.slice(offset, chunk.min(batch.num_rows() - offset))))
            .collect();
        replacements.push(
            fragment
                .write_columns(stream::iter(chunks), &column_schema)
                .await
                .unwrap(),
        );
    }
    let read_version = dataset.manifest.version;
    Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataReplacement {
            replacements,
            data_change: true,
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap()
}

fn replacement_value(fragment_id: i32, i: i32) -> Option<i32> {
    (i % 4 != 1).then_some(fragment_id * 1000 + i)
}

/// merge_insert updating `val` for `ids` in column-rewrite mode.
async fn merge_insert_val(dataset: Dataset, ids: std::ops::Range<i32>) -> Dataset {
    let source_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("val", DataType::Int32, true),
    ]));
    let source = RecordBatch::try_new(
        source_schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(ids.clone())),
            Arc::new(Int32Array::from_iter(
                ids.map(|i| (i % 5 != 0).then_some(i + 1000)),
            )),
        ],
    )
    .unwrap();
    let job = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap();
    let reader = Box::new(RecordBatchIterator::new([Ok(source)], source_schema));
    let (dataset, _) = job.execute(reader_to_stream(reader)).await.unwrap();
    dataset.as_ref().clone()
}

/// Appending after the index exists seeds the new fragment; the fragments
/// written before the index have no seed.
#[tokio::test]
async fn test_append_seeds_new_fragment() {
    let dir = TempStrDir::default();
    let dataset = dataset_with_val_index(dir.as_str(), true).await;
    let val = field_id(&dataset, "val");
    for fragment in &dataset.fragments()[..2] {
        let file = serving_file(fragment, val);
        assert!(seed_in_file(&dataset, &file.path, "val").await.is_none());
    }
    assert_seed_matches_data(&dataset, &[2], "val").await;
}

/// merge_insert rewriting `val` for whole fragments writes a replacement
/// column file per fragment, each with a seed of the new values.
#[tokio::test]
async fn test_merge_insert_full_fragment_rewrite_seeds_new_file() {
    let dir = TempStrDir::default();
    let dataset = dataset_with_val_index(dir.as_str(), true).await;
    let dataset = merge_insert_val(dataset, 10..30).await;

    let val = field_id(&dataset, "val");
    let fragments = dataset.fragments();
    assert_eq!(fragments[0].files.len(), 1, "fragment 0 was not touched");
    for fragment in &fragments[1..] {
        assert_eq!(fragment.files.len(), 2);
        assert_eq!(serving_file(fragment, val).path, fragment.files[1].path);
    }
    assert_seed_matches_data(&dataset, &[1, 2], "val").await;
}

/// merge_insert that matches only some rows of a fragment still rewrites the
/// whole column file for that fragment, with a seed of the merged values.
#[tokio::test]
async fn test_merge_insert_partial_fragment_rewrite_seeds_new_file() {
    let dir = TempStrDir::default();
    let dataset = dataset_with_val_index(dir.as_str(), true).await;
    let dataset = merge_insert_val(dataset, 13..27).await;

    let val = field_id(&dataset, "val");
    let fragments = dataset.fragments();
    assert_eq!(fragments[0].files.len(), 1, "fragment 0 was not touched");
    for fragment in &fragments[1..] {
        assert_eq!(serving_file(fragment, val).path, fragment.files[1].path);
    }
    assert_seed_matches_data(&dataset, &[1, 2], "val").await;
}

/// `FileFragment::write_columns`, used by computed and function column
/// refreshes, stages a replacement column file that carries a seed, even
/// when the column already moved out of the fragment's first file.
#[tokio::test]
async fn test_write_columns_replacement_seeds_new_file() {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_val_index(dir.as_str(), true).await;
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("twice".to_string(), "val * 2".to_string())]),
            None,
            None,
        )
        .await
        .unwrap();
    let dataset = replace_column(dataset, "val", |id| id >= 1, replacement_value).await;

    let val = field_id(&dataset, "val");
    let fragments = dataset.fragments();
    assert_eq!(fragments[0].files.len(), 2, "fragment 0 was not rewritten");
    for fragment in &fragments[1..] {
        assert_eq!(fragment.files.len(), 3);
        assert_eq!(serving_file(fragment, val).path, fragment.files[2].path);
        // The derived column has no index, so its file carries no seed.
        assert!(
            seed_in_file(&dataset, &fragment.files[1].path, "twice")
                .await
                .is_none()
        );
    }
    assert_seed_matches_data(&dataset, &[1, 2], "val").await;
}

/// An index with write seeds disabled gets no seed from any writer.
#[tokio::test]
async fn test_seeds_disabled_index_gets_no_seed_on_rewrite() {
    let dir = TempStrDir::default();
    let dataset = dataset_with_val_index(dir.as_str(), false).await;
    let val = field_id(&dataset, "val");
    let appended = serving_file(&dataset.fragments()[2], val);
    assert!(
        seed_in_file(&dataset, &appended.path, "val")
            .await
            .is_none()
    );

    let dataset = merge_insert_val(dataset, 10..30).await;
    let dataset = replace_column(dataset, "val", |id| id == 0, replacement_value).await;
    for fragment in dataset.fragments().iter() {
        let file = serving_file(fragment, val);
        assert!(seed_in_file(&dataset, &file.path, "val").await.is_none());
    }
}

/// A fragment with deleted rows gets a partial update. The replacement
/// column file still covers every physical row, with the deleted rows
/// restored as nulls, and its seed describes exactly that file: zone
/// boundaries, null counts and null positions match the physical content,
/// not the deletion-filtered view.
#[tokio::test]
async fn test_partial_update_of_fragment_with_deletions_seeds_physical_rows() {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_val_index(dir.as_str(), true).await;
    dataset.delete("id % 3 = 0").await.unwrap();
    let dataset = merge_insert_val(dataset, 13..27).await;

    let val = field_id(&dataset, "val");
    for fragment in &dataset.fragments()[1..] {
        assert!(fragment.deletion_file.is_some());
        let file = serving_file(fragment, val);
        let physical = physical_column(&dataset, &file.path, "val").await;
        assert_eq!(physical.len(), fragment.physical_rows.unwrap());
        let stored = seed_in_file(&dataset, &file.path, "val").await.unwrap();
        assert_eq!(stored, seed_from_values("val", &physical, 1));
        // The deletion-filtered view would describe fewer rows.
        assert_ne!(
            stored,
            seed_from_current_values(&dataset, fragment, "val").await
        );
    }
}

/// `write_columns` fed several batches on a fragment with deleted rows. Ten
/// rows in four batches put batch edges at rows 3, 6 and 9, so with four-row
/// zones every zone but the last spans two batches. The seed is the same as
/// from one batch, and matches the physical file.
#[tokio::test]
async fn test_write_columns_in_batches_seeds_physical_rows() {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_val_index(dir.as_str(), true).await;
    dataset.delete("id % 3 = 0").await.unwrap();
    let dataset =
        replace_column_in_batches(dataset, "val", |id| id >= 1, replacement_value, 4).await;

    let val = field_id(&dataset, "val");
    for fragment in &dataset.fragments()[1..] {
        let file = serving_file(fragment, val);
        let physical = physical_column(&dataset, &file.path, "val").await;
        assert_eq!(physical.len(), fragment.physical_rows.unwrap());
        let stored = seed_in_file(&dataset, &file.path, "val").await.unwrap();
        assert_eq!(stored, seed_from_values("val", &physical, 1));
        assert_eq!(stored, seed_from_values("val", &physical, 4));
    }
}

/// Every fragment that gained a data file carries, in the file serving
/// `column`, a seed matching that file's physical content.
async fn assert_new_files_seed_physical_rows(
    dataset: &Dataset,
    files_before: &std::collections::HashSet<String>,
    column: &str,
) {
    let field = field_id(dataset, column);
    let mut checked = 0;
    for fragment in dataset.fragments().iter() {
        if fragment
            .files
            .iter()
            .all(|file| files_before.contains(&file.path))
        {
            continue;
        }
        let file = serving_file(fragment, field);
        assert!(
            !files_before.contains(&file.path),
            "fragment {} keeps its old file",
            fragment.id
        );
        let physical = physical_column(dataset, &file.path, column).await;
        assert_eq!(physical.len(), fragment.physical_rows.unwrap());
        let stored = seed_in_file(dataset, &file.path, column)
            .await
            .unwrap_or_else(|| panic!("fragment {} has no seed for {column}", fragment.id));
        assert_eq!(stored, seed_from_values(column, &physical, 1));
        checked += 1;
    }
    assert!(checked > 0, "no fragment gained a data file");
}

fn data_file_paths(dataset: &Dataset) -> std::collections::HashSet<String> {
    dataset
        .fragments()
        .iter()
        .flat_map(|fragment| fragment.files.iter().map(|file| file.path.clone()))
        .collect()
}

/// `Dataset::update` rewrites matching rows into new data files written in
/// create mode against the existing dataset; they still get seeds.
#[tokio::test]
async fn test_update_rewritten_fragments_get_seeds() {
    let dir = TempStrDir::default();
    let dataset = dataset_with_val_index(dir.as_str(), true).await;
    let before = data_file_paths(&dataset);
    let result = UpdateBuilder::new(Arc::new(dataset))
        .update_where("id >= 10")
        .unwrap()
        .set("val", "val + 1000")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();
    let dataset = result.new_dataset.as_ref().clone();
    assert_new_files_seed_physical_rows(&dataset, &before, "val").await;
}

/// merge_insert inserting unmatched rows writes new fragments; they get seeds.
#[tokio::test]
async fn test_merge_insert_inserted_fragments_get_seeds() {
    let dir = TempStrDir::default();
    let dataset = dataset_with_val_index(dir.as_str(), true).await;
    let before = data_file_paths(&dataset);
    let source = rows(25..45, |i| (i % 6 != 1).then_some(i + 5000));
    let job = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();
    let reader = Box::new(RecordBatchIterator::new([Ok(source)], schema()));
    let (dataset, _) = job.execute(reader_to_stream(reader)).await.unwrap();
    let dataset = dataset.as_ref().clone();
    assert_new_files_seed_physical_rows(&dataset, &before, "val").await;
}

/// A column repack moving `val` into a new file writes that file's seed from
/// the moved values, deleted rows included.
#[tokio::test]
async fn test_column_repack_seeds_moved_column() {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_val_index(dir.as_str(), true).await;
    dataset.delete("id % 3 = 0").await.unwrap();
    // A backfill gives each fragment a second file, so a limit of 1 repacks.
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("ext".into(), "id + 1".into())]),
            None,
            None,
        )
        .await
        .unwrap();
    let before = data_file_paths(&dataset);
    let fragment_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
    let options = CompactionOptions {
        max_data_files_per_fragment: Some(1),
        scope: CompactionScope::RepackColumns,
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();
    assert_eq!(
        dataset.fragments().iter().map(|f| f.id).collect::<Vec<_>>(),
        fragment_ids
    );
    assert_new_files_seed_physical_rows(&dataset, &before, "val").await;
}

fn view_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("name", DataType::Utf8, true),
        ArrowField::new("blob", DataType::Binary, true),
    ]))
}

fn name_value(i: i32) -> Option<String> {
    (i % 5 != 0).then(|| format!("n{i:03}"))
}

fn blob_value(i: i32) -> Option<Vec<u8>> {
    (i % 6 != 2).then(|| format!("b{i:03}").into_bytes())
}

/// Twenty rows in two fragments with seeded zone maps on a Utf8 and a
/// Binary column, written from classic (non-view) arrays.
async fn dataset_with_seeded_view_targets(uri: &str) -> Dataset {
    let batch = RecordBatch::try_new(
        view_schema(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..20)),
            Arc::new(StringArray::from_iter(
                (0..20).map(|i| name_value(i).map(|s| s.to_lowercase())),
            )),
            Arc::new(BinaryArray::from_iter((0..20).map(blob_value))),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], view_schema()),
        uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FILE,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    for column in ["name", "blob"] {
        dataset
            .create_index(
                &[column],
                IndexType::ZoneMap,
                None,
                &zone_map_params(true),
                false,
            )
            .await
            .unwrap();
    }
    dataset
}

/// The data file serving `column` in every fragment carries a seed matching
/// the file's physical content, which for a view input is the classic
/// offset-based column the file stores.
async fn assert_all_fragments_seed_physical_rows(dataset: &Dataset, column: &str) {
    let field = field_id(dataset, column);
    for fragment in dataset.fragments().iter() {
        let file = serving_file(fragment, field);
        let physical = physical_column(dataset, &file.path, column).await;
        assert_eq!(physical.len(), fragment.physical_rows.unwrap());
        assert!(
            physical.null_count() > 0,
            "the test data must contain nulls"
        );
        let stored = seed_in_file(dataset, &file.path, column)
            .await
            .unwrap_or_else(|| panic!("fragment {} has no seed for {column}", fragment.id));
        assert_eq!(
            stored,
            seed_from_values(column, &physical, 1),
            "fragment {}",
            fragment.id
        );
    }
}

/// A full-fragment merge_insert fed Utf8View values for a seeded Utf8
/// column: the data file stores classic Utf8, and the seed must describe
/// that stored column, built from the same normalized batches the writer
/// received.
#[tokio::test]
async fn test_merge_insert_rewrite_with_view_input_seeds_stored_column() {
    let dir = TempStrDir::default();
    let dataset = dataset_with_seeded_view_targets(dir.as_str()).await;

    let source_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("name", DataType::Utf8View, true),
    ]));
    let source = RecordBatch::try_new(
        source_schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..20)),
            Arc::new(StringViewArray::from_iter(
                (0..20).map(|i| name_value(i).map(|s| s.to_uppercase())),
            )),
        ],
    )
    .unwrap();
    let job = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap();
    let reader = Box::new(RecordBatchIterator::new([Ok(source)], source_schema));
    let (dataset, _) = job.execute(reader_to_stream(reader)).await.unwrap();
    let dataset = dataset.as_ref().clone();

    // The written values are the uppercase names, stored as classic Utf8.
    let name = field_id(&dataset, "name");
    for fragment in dataset.fragments().iter() {
        let file = serving_file(fragment, name);
        let physical = physical_column(&dataset, &file.path, "name").await;
        assert_eq!(physical.data_type(), &DataType::Utf8);
        let start = fragment.id as i32 * ROWS_PER_FILE as i32;
        let expected: Vec<Option<String>> = (start..start + physical.len() as i32)
            .map(|i| name_value(i).map(|s| s.to_uppercase()))
            .collect();
        let actual: Vec<Option<String>> = physical
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|v| v.map(str::to_string))
            .collect();
        assert_eq!(actual, expected);
    }
    assert_all_fragments_seed_physical_rows(&dataset, "name").await;
}

/// A full-fragment merge_insert fed BinaryView values for a seeded Binary
/// column, in three-row source batches whose edges do not line up with the
/// four-row zones: the stored seed describes the classic Binary column the
/// file holds.
#[tokio::test]
async fn test_merge_insert_rewrite_with_binary_view_input_seeds_stored_column() {
    let dir = TempStrDir::default();
    let dataset = dataset_with_seeded_view_targets(dir.as_str()).await;

    let source_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("blob", DataType::BinaryView, true),
    ]));
    let source = RecordBatch::try_new(
        source_schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..20)),
            Arc::new(BinaryViewArray::from_iter(
                (0..20).map(|i| blob_value(i + 1000)),
            )),
        ],
    )
    .unwrap();
    let chunks: Vec<_> = (0..source.num_rows())
        .step_by(3)
        .map(|offset| Ok(source.slice(offset, 3.min(source.num_rows() - offset))))
        .collect();
    let job = MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap();
    let reader = Box::new(RecordBatchIterator::new(chunks, source_schema));
    let (dataset, _) = job.execute(reader_to_stream(reader)).await.unwrap();
    let dataset = dataset.as_ref().clone();

    let blob = field_id(&dataset, "blob");
    for fragment in dataset.fragments().iter() {
        let file = serving_file(fragment, blob);
        let physical = physical_column(&dataset, &file.path, "blob").await;
        assert_eq!(physical.data_type(), &DataType::Binary);
        let start = fragment.id as i32 * ROWS_PER_FILE as i32;
        let expected: Vec<Option<Vec<u8>>> = (start..start + physical.len() as i32)
            .map(|i| blob_value(i + 1000))
            .collect();
        let actual: Vec<Option<Vec<u8>>> = physical
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap()
            .iter()
            .map(|v| v.map(<[u8]>::to_vec))
            .collect();
        assert_eq!(actual, expected);
    }
    assert_all_fragments_seed_physical_rows(&dataset, "blob").await;
}

// --- Binary-copy compaction seed carrying -----------------------------------
//
// Binary copy moves data pages verbatim; the output file's seed is the
// concatenation of its inputs' zone records, per output and per column, and a
// column whose inputs cannot be represented under the fixed-stride layout
// simply loses its seed while the data still copies.

use crate::dataset::optimize::{
    CompactionMetrics, CompactionMode, CompactionOptions, compact_files,
};
use lance_table::feature_flags::FLAG_ZONE_WRITE_SEEDS;

fn assert_seed_flags_set(dataset: &Dataset) {
    assert_ne!(
        dataset.manifest.reader_feature_flags & FLAG_ZONE_WRITE_SEEDS,
        0,
        "reader flag"
    );
    assert_ne!(
        dataset.manifest.writer_feature_flags & FLAG_ZONE_WRITE_SEEDS,
        0,
        "writer flag"
    );
}

async fn force_binary_copy(dataset: &mut Dataset, target_rows: u64) -> CompactionMetrics {
    compact_files(
        dataset,
        CompactionOptions {
            target_rows_per_fragment: target_rows as usize,
            compaction_mode: Some(CompactionMode::ForceBinaryCopy),
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap()
}

/// Like [`all_seeded_dataset`] but keeps the unseeded pre-index fragment.
/// The planner never mixes fragments with different index coverage in one
/// bin, so the covered pre-index fragment stays put and compaction takes the
/// seeded appends alone, while the index keeps live training data (an index
/// whose covered fragments are all gone is rebuilt by scan, not updated).
async fn seeded_appends_dataset(uri: &str, appends: usize, rows_per_append: i32) -> Dataset {
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(rows(0..8, base_value))], schema()),
        uri,
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["val"],
            IndexType::ZoneMap,
            None,
            &zone_map_params(true),
            false,
        )
        .await
        .unwrap();
    for append in 0..appends as i32 {
        let start = 8 + append * rows_per_append;
        dataset
            .append(
                RecordBatchIterator::new(
                    [Ok(rows(start..start + rows_per_append, base_value))],
                    schema(),
                ),
                None,
            )
            .await
            .unwrap();
    }
    dataset
}

/// The fragments compaction added: every fragment with an id above the
/// pre-compaction maximum.
fn fragments_above(dataset: &Dataset, max_before: u64) -> Vec<Fragment> {
    dataset
        .fragments()
        .iter()
        .filter(|fragment| fragment.id > max_before)
        .cloned()
        .collect()
}

/// A dataset whose every fragment carries a `val` seed: the pre-index
/// fragment is fully deleted after seeded appends, leaving only aligned
/// (or ragged, per `rows_per_append`) seeded fragments.
async fn all_seeded_dataset(uri: &str, appends: usize, rows_per_append: i32) -> Dataset {
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(rows(0..8, base_value))], schema()),
        uri,
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["val"],
            IndexType::ZoneMap,
            None,
            &zone_map_params(true),
            false,
        )
        .await
        .unwrap();
    for append in 0..appends as i32 {
        let start = 8 + append * rows_per_append;
        dataset
            .append(
                RecordBatchIterator::new(
                    [Ok(rows(start..start + rows_per_append, base_value))],
                    schema(),
                ),
                None,
            )
            .await
            .unwrap();
    }
    // Drop the unseeded pre-index fragment so every remaining input has a
    // `val` seed.
    dataset.delete("id < 8").await.unwrap();
    assert_eq!(dataset.fragments().len(), appends);
    for fragment in dataset.fragments().iter() {
        assert!(fragment.deletion_file.is_none());
        let file = serving_file(fragment, field_id(&dataset, "val"));
        assert!(seed_in_file(&dataset, &file.path, "val").await.is_some());
    }
    dataset
}

/// Aligned seeded inputs merge into one output whose seed matches the
/// output's physical values exactly, and the sticky flags stay set.
#[tokio::test]
async fn test_binary_copy_merges_aligned_seeds() {
    let dir = TempStrDir::default();
    // 8-row fragments with rows_per_zone = 4: every tail zone is full.
    let mut dataset = all_seeded_dataset(dir.as_str(), 3, 8).await;
    assert_seed_flags_set(&dataset);
    let before = dataset.scan().try_into_batch().await.unwrap();

    let metrics = force_binary_copy(&mut dataset, 1_000_000).await;
    assert_eq!(metrics.fragments_added, 1);
    assert_eq!(metrics.fragments_removed, 3);

    let output_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
    assert_eq!(output_ids.len(), 1);
    assert_seed_matches_data(&dataset, &output_ids, "val").await;
    assert_seed_flags_set(&dataset);

    let after = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(before, after);
}

/// Several output files: each output's seed covers exactly its own inputs,
/// with the null bitmap offsets reset per output.
#[tokio::test]
async fn test_binary_copy_multi_output_seeds_are_independent() {
    let dir = TempStrDir::default();
    let mut dataset = all_seeded_dataset(dir.as_str(), 4, 8).await;

    let metrics = force_binary_copy(&mut dataset, 16).await;
    assert_eq!(metrics.fragments_added, 2);

    let output_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
    assert_eq!(output_ids.len(), 2);
    for fragment in dataset.fragments().iter() {
        assert_eq!(fragment.physical_rows, Some(16));
    }
    assert_seed_matches_data(&dataset, &output_ids, "val").await;
}

/// An output fed by exactly one input carries that input's seed verbatim,
/// ragged tail and all; an output merging ragged inputs carries their
/// concatenated records.
#[tokio::test]
async fn test_binary_copy_single_source_output_carries_seed_verbatim() {
    let dir = TempStrDir::default();
    // 10-row seeded appends with rows_per_zone = 4: zones [4, 4, 2] each.
    let mut dataset = seeded_appends_dataset(dir.as_str(), 3, 10).await;
    let val = field_id(&dataset, "val");
    let last_input = dataset.fragments().last().unwrap().clone();
    let last_input_seed = seed_in_file(&dataset, &serving_file(&last_input, val).path, "val")
        .await
        .unwrap();
    let max_before = dataset.fragments().iter().map(|f| f.id).max().unwrap();

    // The bin flushes an output at 15 rows: the first two sources merge into
    // a 20-row output, the third lands alone as a whole source file.
    let metrics = force_binary_copy(&mut dataset, 15).await;
    assert_eq!(metrics.fragments_added, 2);

    let outputs = fragments_above(&dataset, max_before);
    assert_eq!(outputs.len(), 2);
    let merged = outputs
        .iter()
        .find(|f| f.physical_rows == Some(20))
        .unwrap();
    let single = outputs
        .iter()
        .find(|f| f.physical_rows == Some(10))
        .unwrap();
    assert!(
        seed_in_file(&dataset, &serving_file(merged, val).path, "val")
            .await
            .is_some(),
        "ragged sources concatenate under the self-describing layout"
    );
    let carried = seed_in_file(&dataset, &serving_file(single, val).path, "val")
        .await
        .unwrap();
    assert_eq!(
        carried, last_input_seed,
        "single-source outputs carry bytes verbatim"
    );
}

/// Ragged inputs (short tail zones) concatenate into a seed with short zones
/// mid file: version 1 records are self-describing, the next index update
/// reuses every record that lies inside one configured zone and scans only
/// the intervals of the records that cross a zone boundary, and indexed
/// queries match unindexed scans.
#[tokio::test]
async fn test_binary_copy_merges_ragged_seeds_and_feeds_index_update() {
    use arrow::array::AsArray;
    use arrow::datatypes::Int32Type;

    let dir = TempStrDir::default();
    // 10-row seeded appends with rows_per_zone = 4: zones [4, 4, 2] per
    // input, so the merged seed holds short zones mid file. The covered
    // pre-index fragment stays out of the bin and keeps the index alive.
    let mut dataset = seeded_appends_dataset(dir.as_str(), 2, 10).await;
    let before = dataset.scan().try_into_batch().await.unwrap();
    let max_before = dataset.fragments().iter().map(|f| f.id).max().unwrap();

    let metrics = force_binary_copy(&mut dataset, 1_000_000).await;
    assert_eq!(metrics.fragments_added, 1);

    let fragment = fragments_above(&dataset, max_before)[0].clone();
    let file = serving_file(&fragment, field_id(&dataset, "val"));
    assert!(
        seed_in_file(&dataset, &file.path, "val").await.is_some(),
        "self-describing version 1 records carry mid-file short zones"
    );
    let after = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(before, after);
    assert_seed_flags_set(&dataset);

    // The compacted fragment is unindexed. Its merged seed holds records
    // [0,4) [4,8) [8,10) [10,14) [14,18) [18,20) for 4-row zones: the four
    // records inside one zone are reused, the two that cross a zone
    // boundary ([10,14) over 12, [14,18) over 16) are scanned by interval.
    use crate::index::scalar::zonemap::SEED_REUSE_REPORTS;
    let reports = SEED_REUSE_REPORTS
        .scope(std::sync::Mutex::new(Vec::new()), async {
            dataset
                .optimize_indices(&OptimizeOptions::default())
                .await
                .unwrap();
            SEED_REUSE_REPORTS.with(|reports| reports.lock().unwrap().clone())
        })
        .await;
    assert_eq!(reports.len(), 1, "{reports:?}");
    assert_eq!(reports[0].partial_fragments, vec![fragment.id]);
    assert_eq!(reports[0].reused_records, 4);
    assert_eq!(reports[0].scanned_rows, 8);
    assert!(
        reports[0].masked,
        "the crossing intervals are scanned through a mask"
    );

    for filter in ["val >= 10 AND val < 15", "val IS NULL", "val = 23"] {
        let mut indexed = dataset.scan();
        indexed.filter(filter).unwrap().project(&["id"]).unwrap();
        let plan = indexed.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery") && plan.contains("ZoneMap"),
            "filter {filter} must use the zone map index: {plan}"
        );
        let indexed_ids: Vec<i32> = indexed.try_into_batch().await.unwrap()["id"]
            .as_primitive::<Int32Type>()
            .values()
            .to_vec();

        let mut unindexed = dataset.scan();
        unindexed
            .use_scalar_index(false)
            .filter(filter)
            .unwrap()
            .project(&["id"])
            .unwrap();
        let expected_ids: Vec<i32> = unindexed.try_into_batch().await.unwrap()["id"]
            .as_primitive::<Int32Type>()
            .values()
            .to_vec();
        assert_eq!(indexed_ids, expected_ids, "filter {filter}");
    }
}

/// A binary copy carries seed records unchanged under a locator that
/// declares the column's current zone size, which can be smaller than those
/// records; a later index whose zones fit the records still reuses them
/// instead of scanning.
#[tokio::test]
async fn test_binary_copy_keeps_large_carried_records_reusable_after_config_change() {
    use crate::index::scalar::zonemap::SEED_REUSE_REPORTS;

    fn params(rows_per_zone: u64) -> ScalarIndexParams {
        ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
            .with_params(&serde_json::json!({"rows_per_zone": rows_per_zone, "use_seeds": true}))
    }
    async fn locator_rows_per_zone(dataset: &Dataset, fragment: &Fragment) -> u64 {
        let file = serving_file(fragment, field_id(dataset, "val"));
        let reader = open_data_file(dataset, &file.path).await;
        let value = reader
            .metadata()
            .file_schema
            .metadata
            .get(&format!("{SEED_META_KEY_PREFIX}val"))
            .unwrap()
            .clone();
        value.split_once(':').unwrap().1.parse().unwrap()
    }

    let dir = TempStrDir::default();
    // Three 8-row fragments with 4-row seed records.
    let mut dataset = all_seeded_dataset(dir.as_str(), 3, 8).await;
    // The column's index now wants 2-row records: the copy's output
    // declares that locator while carrying the 4-row records verbatim.
    dataset
        .create_index(&["val"], IndexType::ZoneMap, None, &params(2), true)
        .await
        .unwrap();
    let metrics = force_binary_copy(&mut dataset, 1_000_000).await;
    assert_eq!(metrics.fragments_added, 1);
    let output = dataset.fragments()[0].clone();
    assert_eq!(output.physical_rows, Some(24));
    assert_eq!(locator_rows_per_zone(&dataset, &output).await, 2);

    // Eight-row zones fit every carried 4-row record: reuse, no scan.
    let reports = SEED_REUSE_REPORTS
        .scope(std::sync::Mutex::new(Vec::new()), async {
            dataset
                .create_index(&["val"], IndexType::ZoneMap, None, &params(8), true)
                .await
                .unwrap();
            SEED_REUSE_REPORTS.with(|reports| reports.lock().unwrap().clone())
        })
        .await;
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].reused_fragments, vec![output.id]);
    assert_eq!(reports[0].reused_records, 6);
    assert_eq!(reports[0].scanned_rows, 0);

    for filter in ["val >= 10 AND val < 15", "val IS NULL", "val = 23"] {
        let mut indexed = dataset.scan();
        indexed.filter(filter).unwrap().project(&["id"]).unwrap();
        let indexed = indexed.try_into_batch().await.unwrap();
        let mut unindexed = dataset.scan();
        unindexed
            .use_scalar_index(false)
            .filter(filter)
            .unwrap()
            .project(&["id"])
            .unwrap();
        let expected = unindexed.try_into_batch().await.unwrap();
        assert_eq!(indexed, expected, "filter {filter}");
    }
}

/// Seed locators keep their `<buffer_index>:<rows_per_zone>` encoding: a
/// normal write declares the index's granularity, and a binary-copy output
/// declares the column's current index granularity.
#[tokio::test]
async fn test_seed_locator_keeps_rows_per_zone() {
    async fn locator(dataset: &Dataset, fragment: &Fragment) -> (u32, u64) {
        let file = serving_file(fragment, field_id(dataset, "val"));
        let reader = open_data_file(dataset, &file.path).await;
        let value = reader
            .metadata()
            .file_schema
            .metadata
            .get(&format!("{SEED_META_KEY_PREFIX}val"))
            .unwrap()
            .clone();
        let (index, rows_per_zone) = value
            .split_once(':')
            .unwrap_or_else(|| panic!("locator '{value}' lacks the rows_per_zone suffix"));
        (index.parse().unwrap(), rows_per_zone.parse().unwrap())
    }

    let dir = TempStrDir::default();
    let mut dataset = all_seeded_dataset(dir.as_str(), 2, 8).await;
    let fragment = dataset.fragments()[0].clone();
    assert_eq!(locator(&dataset, &fragment).await.1, ROWS_PER_ZONE);

    let metrics = force_binary_copy(&mut dataset, 1_000_000).await;
    assert_eq!(metrics.fragments_added, 1);
    let output = dataset.fragments()[0].clone();
    assert_eq!(locator(&dataset, &output).await.1, ROWS_PER_ZONE);
}

/// Boundary queries around every internal short zone of a merged ragged
/// seed, after a fresh reopen (so the persisted index, not session memory,
/// answers); then a later seeded append over the ragged index; then a full
/// re-encoding maintenance pass followed by the ordinary index refresh.
#[tokio::test]
async fn test_binary_copy_ragged_zone_boundaries_reopen_and_maintenance() {
    use crate::dataset::optimize::compact_files;
    use arrow::array::AsArray;
    use arrow::datatypes::Int32Type;

    async fn assert_indexed_matches_scan(
        phase: &str,
        dataset: &Dataset,
        filters: &[String],
        require_index: bool,
    ) {
        for filter in filters {
            let mut indexed = dataset.scan();
            indexed.filter(filter).unwrap().project(&["id"]).unwrap();
            let plan = indexed.explain_plan(false).await.unwrap();
            if require_index {
                assert!(
                    plan.contains("ScalarIndexQuery") && plan.contains("ZoneMap"),
                    "[{phase}] filter {filter} must use the zone map index: {plan}"
                );
            }
            let indexed_ids: Vec<i32> = indexed.try_into_batch().await.unwrap()["id"]
                .as_primitive::<Int32Type>()
                .values()
                .to_vec();
            let mut unindexed = dataset.scan();
            unindexed
                .use_scalar_index(false)
                .filter(filter)
                .unwrap()
                .project(&["id"])
                .unwrap();
            let expected_ids: Vec<i32> = unindexed.try_into_batch().await.unwrap()["id"]
                .as_primitive::<Int32Type>()
                .values()
                .to_vec();
            assert_eq!(indexed_ids, expected_ids, "[{phase}] filter {filter}");
        }
    }

    fn boundary_filters(boundaries: &[i32]) -> Vec<String> {
        let mut filters = vec!["val IS NULL".to_string(), "val IS NOT NULL".to_string()];
        for boundary in boundaries {
            for value in [boundary - 1, *boundary, boundary + 1] {
                filters.push(format!("val = {value}"));
            }
            filters.push(format!(
                "val >= {} AND val < {}",
                boundary - 1,
                boundary + 1
            ));
        }
        filters
    }

    let dir = TempStrDir::default();
    // Appends of 10 rows (ids 8..18 and 18..28) under rows_per_zone = 4: the
    // merged output's zones are [4, 4, 2, 4, 4, 2] at output offsets
    // 0, 4, 8, 10, 14, 18, that is ids 8, 12, 16, 18, 22, 26.
    let mut dataset = seeded_appends_dataset(dir.as_str(), 2, 10).await;
    let metrics = force_binary_copy(&mut dataset, 1_000_000).await;
    assert_eq!(metrics.fragments_added, 1);
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();

    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(reopened.fragments().len(), 2);
    let internal_boundary_ids = [12, 16, 18, 22, 26];
    assert_indexed_matches_scan(
        "after reopen",
        &reopened,
        &boundary_filters(&internal_boundary_ids),
        true,
    )
    .await;

    let mut dataset = reopened;
    dataset
        .append(
            RecordBatchIterator::new([Ok(rows(28..38, base_value))], schema()),
            None,
        )
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    let mut filters = boundary_filters(&internal_boundary_ids);
    filters.extend(boundary_filters(&[32, 36]));
    assert_indexed_matches_scan("after later append + update", &reopened, &filters, true).await;

    // Full re-encoding maintenance: results must match the scan whether or
    // not the planner routes through the remapped index right away.
    let mut dataset = reopened;
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1_000_000,
            compaction_mode: Some(CompactionMode::Reencode),
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_indexed_matches_scan("after re-encode", &reopened, &filters, false).await;

    // The ordinary refresh then integrates the re-encoded output's fresh seed
    // and queries route through the index again.
    let mut dataset = reopened;
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_indexed_matches_scan("after re-encode + refresh", &reopened, &filters, true).await;
}

/// A column seeded in only some inputs loses its seed (partial statistics
/// must not impersonate the whole output), while a column seeded in every
/// input keeps its merged seed.
#[tokio::test]
async fn test_binary_copy_partial_column_coverage() {
    let dir = TempStrDir::default();
    let mut dataset = seeded_appends_dataset(dir.as_str(), 1, 8).await;
    // The `id` index arrives after the first append, so that fragment has a
    // `val` seed but no `id` seed; the next append carries both. Dropping the
    // index afterwards leaves the written seeds in place and gives both
    // appends the same (empty) index coverage, so they share one bin.
    dataset
        .create_index(
            &["id"],
            IndexType::ZoneMap,
            None,
            &zone_map_params(true),
            false,
        )
        .await
        .unwrap();
    dataset
        .append(
            RecordBatchIterator::new([Ok(rows(16..24, base_value))], schema()),
            None,
        )
        .await
        .unwrap();
    let id = field_id(&dataset, "id");
    let newest = dataset.fragments().last().unwrap().clone();
    assert!(
        seed_in_file(&dataset, &serving_file(&newest, id).path, "id")
            .await
            .is_some()
    );
    let id_index = dataset
        .load_indices()
        .await
        .unwrap()
        .iter()
        .find(|index| index.fields == vec![id])
        .unwrap()
        .name
        .clone();
    dataset.drop_index(&id_index).await.unwrap();
    let max_before = dataset.fragments().iter().map(|f| f.id).max().unwrap();

    let metrics = force_binary_copy(&mut dataset, 1_000_000).await;
    assert_eq!(metrics.fragments_added, 1);

    let outputs = fragments_above(&dataset, max_before);
    assert_eq!(outputs.len(), 1);
    let file = serving_file(&outputs[0], id);
    assert!(
        seed_in_file(&dataset, &file.path, "id").await.is_none(),
        "one input has no 'id' seed, so the output must not claim one"
    );
    let output_ids: Vec<u64> = outputs.iter().map(|f| f.id).collect();
    assert_seed_matches_data(&dataset, &output_ids, "val").await;
}

/// A rename between the write and the compaction: seeds match by field id and
/// the output key uses the current column name.
#[tokio::test]
async fn test_binary_copy_carries_seed_across_rename() {
    let dir = TempStrDir::default();
    let mut dataset = all_seeded_dataset(dir.as_str(), 2, 8).await;
    dataset
        .alter_columns(&[ColumnAlteration::new("val".into()).rename("renamed".into())])
        .await
        .unwrap();

    let metrics = force_binary_copy(&mut dataset, 1_000_000).await;
    assert_eq!(metrics.fragments_added, 1);

    let fragment = &dataset.fragments()[0];
    let file = serving_file(fragment, field_id(&dataset, "renamed"));
    assert!(
        seed_in_file(&dataset, &file.path, "val").await.is_none(),
        "the stale write-time name must not key the output seed"
    );
    let output_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
    assert_seed_matches_data(&dataset, &output_ids, "renamed").await;
}

// Version 1 seeds are announced by the paired `FLAG_ZONE_WRITE_SEEDS` bits.
// The write path embeds such a seed only when the writer's snapshot already
// carries the bits, and every commit sets them for the seed-requesting
// indices in the manifest it builds, so no manifest references a seed its
// bits do not announce. These tests pin that rule across the commit paths.

fn seed_flags_set(dataset: &Dataset) -> bool {
    dataset.manifest.reader_feature_flags & FLAG_ZONE_WRITE_SEEDS != 0
        && dataset.manifest.writer_feature_flags & FLAG_ZONE_WRITE_SEEDS != 0
}

async fn last_fragment_seeded(dataset: &Dataset) -> bool {
    let fragment = dataset.fragments().last().unwrap();
    let file = serving_file(fragment, field_id(dataset, "val"));
    seed_in_file(dataset, &file.path, "val").await.is_some()
}

async fn append_rows(dataset: &mut Dataset, ids: std::ops::Range<i32>) {
    dataset
        .append(
            RecordBatchIterator::new([Ok(rows(ids, base_value))], schema()),
            None,
        )
        .await
        .unwrap();
}

async fn stage_append(dataset: &Arc<Dataset>, ids: std::ops::Range<i32>) -> Transaction {
    InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(vec![rows(ids, base_value)])
        .await
        .unwrap()
}

/// The transaction as a reader of its serialized form sees it.
fn serialized_round_trip(transaction: &Transaction) -> Transaction {
    let encoded = lance_table::format::pb::Transaction::try_from(transaction).unwrap();
    Transaction::try_from(encoded).unwrap()
}

/// Commit a copy of the latest manifest with the zone seed bits cleared, the
/// way a build that predates the flag left a seeded index behind, and reopen.
async fn reopen_with_seed_flags_cleared(dataset: &Dataset) -> Dataset {
    use crate::dataset::{ManifestWriteConfig, write_manifest_file};
    use crate::index::loading::load_all_indices;

    let mut manifest = dataset.manifest.as_ref().clone();
    manifest.reader_feature_flags &= !FLAG_ZONE_WRITE_SEEDS;
    manifest.writer_feature_flags &= !FLAG_ZONE_WRITE_SEEDS;
    manifest.version += 1;
    let indices = load_all_indices(dataset).await.unwrap().as_ref().clone();
    let index_count = indices.len();
    write_manifest_file(
        dataset.object_store.as_ref(),
        dataset.commit_handler.as_ref(),
        &dataset.base,
        &mut manifest,
        Some(indices),
        &ManifestWriteConfig {
            auto_set_feature_flags: false,
            ..Default::default()
        },
        dataset.manifest_location.naming_scheme,
        None,
        false,
    )
    .await
    .unwrap();
    let mut reopened = dataset.clone();
    reopened.checkout_latest().await.unwrap();
    assert!(!seed_flags_set(&reopened));
    assert_eq!(reopened.load_indices().await.unwrap().len(), index_count);
    reopened
}

async fn dataset_with_seeded_index(uri: &str) -> Dataset {
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(rows(0..8, base_value))], schema()),
        uri,
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["val"],
            IndexType::ZoneMap,
            None,
            &zone_map_params(true),
            false,
        )
        .await
        .unwrap();
    dataset
}

/// Creating a seed-requesting index activates the bits in that commit, before
/// any seeded file exists; an index without seeds and plain writes do not; the
/// bits stay set after the index is dropped, across a restore and in a clone.
#[tokio::test]
async fn test_zone_seed_flags_follow_index_configuration() {
    let unseeded_dir = TempStrDir::default();
    let mut unseeded = Dataset::write(
        RecordBatchIterator::new([Ok(rows(0..8, base_value))], schema()),
        unseeded_dir.as_str(),
        None,
    )
    .await
    .unwrap();
    assert!(!seed_flags_set(&unseeded));
    unseeded
        .create_index(
            &["val"],
            IndexType::ZoneMap,
            None,
            &zone_map_params(false),
            false,
        )
        .await
        .unwrap();
    assert!(!seed_flags_set(&unseeded));
    append_rows(&mut unseeded, 8..16).await;
    assert!(
        !seed_flags_set(&unseeded),
        "an index that does not ask for seeds must not activate the bits"
    );
    assert!(!last_fragment_seeded(&unseeded).await);

    let dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(rows(0..8, base_value))], schema()),
        dir.as_str(),
        None,
    )
    .await
    .unwrap();
    let pre_index_version = dataset.manifest.version;
    dataset
        .create_index(
            &["val"],
            IndexType::ZoneMap,
            None,
            &zone_map_params(true),
            false,
        )
        .await
        .unwrap();
    assert!(
        seed_flags_set(&dataset),
        "the commit that configures seeds sets the bits before any seeded file exists"
    );
    append_rows(&mut dataset, 8..16).await;
    assert!(seed_flags_set(&dataset));
    assert!(last_fragment_seeded(&dataset).await);

    let clone_dir = TempStrDir::default();
    let cloned = dataset
        .shallow_clone(clone_dir.as_str(), dataset.version_id(), None)
        .await
        .unwrap();
    assert!(seed_flags_set(&cloned), "sticky into a shallow clone");

    let index_name = dataset.load_indices().await.unwrap()[0].name.clone();
    dataset.drop_index(&index_name).await.unwrap();
    assert!(
        seed_flags_set(&dataset),
        "sticky after the index is dropped"
    );
    append_rows(&mut dataset, 16..24).await;
    assert!(seed_flags_set(&dataset));

    let mut restored = dataset.checkout_version(pre_index_version).await.unwrap();
    restored.restore().await.unwrap();
    assert!(seed_flags_set(&restored), "sticky across restore");
}

/// A seeded index left behind by a build that did not set the bits: writes
/// against that snapshot embed no seed, the next commit activates the bits,
/// and writes after that are seeded.
#[tokio::test]
async fn test_zone_seed_flags_legacy_index_activates_before_seeding() {
    let dir = TempStrDir::default();
    let dataset = dataset_with_seeded_index(dir.as_str()).await;
    let mut legacy = reopen_with_seed_flags_cleared(&dataset).await;

    append_rows(&mut legacy, 8..16).await;
    assert!(
        seed_flags_set(&legacy),
        "the first commit activates the bits"
    );
    assert!(
        !last_fragment_seeded(&legacy).await,
        "a write against an unflagged snapshot must not embed a seed"
    );

    append_rows(&mut legacy, 16..24).await;
    assert!(last_fragment_seeded(&legacy).await);
}

/// A staged append serialized, committed on a fresh handle at a later version
/// where the index has been dropped: the legacy case publishes no seed, the
/// flagged case keeps its seed and the sticky bits.
#[tokio::test]
async fn test_zone_seed_flags_staged_append_committed_after_index_drop() {
    let legacy_dir = TempStrDir::default();
    let dataset = dataset_with_seeded_index(legacy_dir.as_str()).await;
    let legacy = Arc::new(reopen_with_seed_flags_cleared(&dataset).await);
    let index_name = legacy.load_indices().await.unwrap()[0].name.clone();
    let staged = serialized_round_trip(&stage_append(&legacy, 8..16).await);
    let mut dropped = (*legacy).clone();
    dropped.drop_index(&index_name).await.unwrap();
    assert!(!seed_flags_set(&dropped), "nothing left to announce");
    let fresh = Dataset::open(legacy_dir.as_str()).await.unwrap();
    assert_eq!(fresh.manifest.version, dropped.manifest.version);
    let committed = CommitBuilder::new(Arc::new(fresh))
        .execute(staged)
        .await
        .unwrap();
    assert!(!last_fragment_seeded(&committed).await);
    assert!(!seed_flags_set(&committed));

    let flagged_dir = TempStrDir::default();
    let flagged = Arc::new(dataset_with_seeded_index(flagged_dir.as_str()).await);
    assert!(seed_flags_set(&flagged));
    let index_name = flagged.load_indices().await.unwrap()[0].name.clone();
    let staged = serialized_round_trip(&stage_append(&flagged, 8..16).await);
    let mut dropped = (*flagged).clone();
    dropped.drop_index(&index_name).await.unwrap();
    assert!(seed_flags_set(&dropped));
    let fresh = Dataset::open(flagged_dir.as_str()).await.unwrap();
    let committed = CommitBuilder::new(Arc::new(fresh))
        .execute(staged)
        .await
        .unwrap();
    assert!(last_fragment_seeded(&committed).await);
    assert!(seed_flags_set(&committed));
}

/// Rebase over an interleaved commit and detached commits, from a flagged
/// base and from a legacy base.
#[tokio::test]
async fn test_zone_seed_flags_rebase_and_detached_commits() {
    let dir = TempStrDir::default();
    let flagged = Arc::new(dataset_with_seeded_index(dir.as_str()).await);

    let staged = stage_append(&flagged, 8..16).await;
    let mut interleaved = (*flagged).clone();
    interleaved.delete("id = 0").await.unwrap();
    let rebased = CommitBuilder::new(flagged.clone())
        .execute(staged)
        .await
        .unwrap();
    assert!(seed_flags_set(&rebased));
    assert!(last_fragment_seeded(&rebased).await);

    let staged = stage_append(&flagged, 16..24).await;
    let detached = CommitBuilder::new(flagged.clone())
        .with_detached(true)
        .execute(staged)
        .await
        .unwrap();
    assert!(seed_flags_set(&detached));
    assert!(last_fragment_seeded(&detached).await);

    let legacy_dir = TempStrDir::default();
    let dataset = dataset_with_seeded_index(legacy_dir.as_str()).await;
    let legacy = Arc::new(reopen_with_seed_flags_cleared(&dataset).await);
    let staged = stage_append(&legacy, 8..16).await;
    let detached = CommitBuilder::new(legacy.clone())
        .with_detached(true)
        .execute(staged)
        .await
        .unwrap();
    assert!(
        seed_flags_set(&detached),
        "a detached commit activates the bits for the index it publishes"
    );
    assert!(!last_fragment_seeded(&detached).await);
}

/// Re-encoding compaction: outputs written against a legacy snapshot carry no
/// seed while the rewrite commit activates the bits; outputs written against a
/// flagged snapshot carry seeds.
#[tokio::test]
async fn test_zone_seed_flags_reencode_compaction() {
    let dir = TempStrDir::default();
    let dataset = seeded_appends_dataset(dir.as_str(), 2, 8).await;
    let mut legacy = reopen_with_seed_flags_cleared(&dataset).await;
    let max_before = legacy.fragments().iter().map(|f| f.id).max().unwrap();
    let metrics = compact_files(
        &mut legacy,
        CompactionOptions {
            compaction_mode: Some(CompactionMode::Reencode),
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert!(metrics.fragments_added > 0);
    assert!(
        seed_flags_set(&legacy),
        "the rewrite commit activates the bits"
    );
    for fragment in fragments_above(&legacy, max_before) {
        let file = serving_file(&fragment, field_id(&legacy, "val"));
        assert!(
            seed_in_file(&legacy, &file.path, "val").await.is_none(),
            "a rewrite against an unflagged snapshot must not embed seeds"
        );
    }

    append_rows(&mut legacy, 100..108).await;
    append_rows(&mut legacy, 108..116).await;
    let max_before = legacy.fragments().iter().map(|f| f.id).max().unwrap();
    let metrics = compact_files(
        &mut legacy,
        CompactionOptions {
            compaction_mode: Some(CompactionMode::Reencode),
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert!(metrics.fragments_added > 0);
    for fragment in fragments_above(&legacy, max_before) {
        let file = serving_file(&fragment, field_id(&legacy, "val"));
        assert!(seed_in_file(&legacy, &file.path, "val").await.is_some());
    }
}

/// The fragments a commit added, ordered by id, with the ids of the rows
/// each one holds, so a test can say which staged write produced it.
async fn new_fragments_with_ids(dataset: &Dataset, max_before: u64) -> Vec<(Fragment, Vec<i32>)> {
    let mut out = Vec::new();
    for fragment in fragments_above(dataset, max_before) {
        let batch = dataset
            .scan()
            .with_fragments(vec![fragment.clone()])
            .project(&["id"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids: Vec<i32> = batch["id"]
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        out.push((fragment, ids));
    }
    out
}

async fn fragment_seeded(dataset: &Dataset, fragment: &Fragment) -> bool {
    let file = serving_file(fragment, field_id(dataset, "val"));
    seed_in_file(dataset, &file.path, "val").await.is_some()
}

/// A worker stages fragments against a snapshot (`execute_uncommitted`), the
/// transaction crosses a process boundary (proto round trip) and a
/// coordinator commits it on a freshly opened handle at a later version.
#[tokio::test]
async fn test_zone_seed_flags_distributed_publication() {
    // Flagged snapshot: the staged file carries its seed and the bits travel
    // through sticky inheritance from the later version.
    let dir = TempStrDir::default();
    let flagged = Arc::new(dataset_with_seeded_index(dir.as_str()).await);
    let staged = serialized_round_trip(&stage_append(&flagged, 8..16).await);
    let mut elsewhere = (*flagged).clone();
    elsewhere.delete("id = 0").await.unwrap();
    let fresh = Dataset::open(dir.as_str()).await.unwrap();
    assert!(fresh.manifest.version > staged.read_version);
    let committed = CommitBuilder::new(Arc::new(fresh))
        .execute(staged)
        .await
        .unwrap();
    assert!(seed_flags_set(&committed));
    assert!(last_fragment_seeded(&committed).await);

    // Legacy snapshot, index still present at commit time: the staged file
    // carries no seed; the commit activates the bits for the index it
    // publishes.
    let legacy_dir = TempStrDir::default();
    let dataset = dataset_with_seeded_index(legacy_dir.as_str()).await;
    let legacy = Arc::new(reopen_with_seed_flags_cleared(&dataset).await);
    let staged = serialized_round_trip(&stage_append(&legacy, 8..16).await);
    let fresh = Dataset::open(legacy_dir.as_str()).await.unwrap();
    let committed = CommitBuilder::new(Arc::new(fresh))
        .execute(staged)
        .await
        .unwrap();
    assert!(seed_flags_set(&committed));
    assert!(!last_fragment_seeded(&committed).await);

    // Legacy snapshot, index dropped before the commit: nothing announces
    // seeds and no file carries one.
    let legacy_dir = TempStrDir::default();
    let dataset = dataset_with_seeded_index(legacy_dir.as_str()).await;
    let legacy = Arc::new(reopen_with_seed_flags_cleared(&dataset).await);
    let index_name = legacy.load_indices().await.unwrap()[0].name.clone();
    let staged = serialized_round_trip(&stage_append(&legacy, 8..16).await);
    let mut dropped = (*legacy).clone();
    dropped.drop_index(&index_name).await.unwrap();
    assert!(
        !seed_flags_set(&dropped),
        "empty prepared index list after the drop"
    );
    let fresh = Dataset::open(legacy_dir.as_str()).await.unwrap();
    let committed = CommitBuilder::new(Arc::new(fresh))
        .execute(staged)
        .await
        .unwrap();
    assert!(!seed_flags_set(&committed));
    assert!(!last_fragment_seeded(&committed).await);
}

/// `execute_batch` merges transactions staged against different snapshots
/// and takes the minimum read version. Files staged before activation carry
/// no seed, files staged after it do, and the published manifest carries the
/// bits whenever any seeded file is published.
#[tokio::test]
async fn test_zone_seed_flags_execute_batch_mixed_snapshots() {
    let dir = TempStrDir::default();
    let dataset = dataset_with_seeded_index(dir.as_str()).await;
    let legacy = Arc::new(reopen_with_seed_flags_cleared(&dataset).await);
    let staged_legacy = stage_append(&legacy, 100..108).await;

    // An activating commit from the legacy snapshot: its own file is unseeded.
    let mut activated = (*legacy).clone();
    append_rows(&mut activated, 8..16).await;
    assert!(seed_flags_set(&activated));
    assert!(!last_fragment_seeded(&activated).await);
    let activated = Arc::new(activated);
    let staged_flagged = stage_append(&activated, 200..208).await;

    let max_before = activated.fragments().iter().map(|f| f.id).max().unwrap();
    let result = CommitBuilder::new(activated.clone())
        .execute_batch(vec![staged_legacy, staged_flagged])
        .await
        .unwrap();
    let batched = result.dataset;
    assert!(seed_flags_set(&batched));
    let added = new_fragments_with_ids(&batched, max_before).await;
    assert_eq!(added.len(), 2);
    for (fragment, ids) in &added {
        let seeded = fragment_seeded(&batched, fragment).await;
        if ids.contains(&100) {
            assert!(
                !seeded,
                "the file staged against the legacy snapshot has no seed"
            );
        } else {
            assert!(ids.contains(&200));
            assert!(
                seeded,
                "the file staged against the flagged snapshot is seeded"
            );
        }
    }

    // The index disappears before publication: the file staged after the
    // activation is still seeded and the bits are sticky.
    let legacy = Arc::new(reopen_with_seed_flags_cleared(&batched).await);
    let staged_legacy = stage_append(&legacy, 300..308).await;
    let mut activated = (*legacy).clone();
    append_rows(&mut activated, 16..24).await;
    let activated = Arc::new(activated);
    let staged_flagged = stage_append(&activated, 400..408).await;
    let index_name = activated.load_indices().await.unwrap()[0].name.clone();
    let mut dropped = (*activated).clone();
    dropped.drop_index(&index_name).await.unwrap();
    assert!(seed_flags_set(&dropped));
    let max_before = dropped.fragments().iter().map(|f| f.id).max().unwrap();
    let result = CommitBuilder::new(Arc::new(dropped))
        .execute_batch(vec![staged_legacy, staged_flagged])
        .await
        .unwrap();
    let batched = result.dataset;
    assert!(seed_flags_set(&batched));
    let added = new_fragments_with_ids(&batched, max_before).await;
    assert_eq!(added.len(), 2);
    for (fragment, ids) in &added {
        let seeded = fragment_seeded(&batched, fragment).await;
        assert_eq!(seeded, ids.contains(&400), "ids {ids:?}");
    }

    // The index disappears before any activation: no file was ever seeded
    // and nothing announces seeds.
    let dir = TempStrDir::default();
    let dataset = dataset_with_seeded_index(dir.as_str()).await;
    let legacy = Arc::new(reopen_with_seed_flags_cleared(&dataset).await);
    let staged_before = stage_append(&legacy, 100..108).await;
    let index_name = legacy.load_indices().await.unwrap()[0].name.clone();
    let mut dropped = (*legacy).clone();
    dropped.drop_index(&index_name).await.unwrap();
    assert!(!seed_flags_set(&dropped));
    let dropped = Arc::new(dropped);
    let staged_after = stage_append(&dropped, 200..208).await;
    let max_before = dropped.fragments().iter().map(|f| f.id).max().unwrap();
    let result = CommitBuilder::new(dropped.clone())
        .execute_batch(vec![staged_before, staged_after])
        .await
        .unwrap();
    let batched = result.dataset;
    assert!(!seed_flags_set(&batched));
    let added = new_fragments_with_ids(&batched, max_before).await;
    assert_eq!(added.len(), 2);
    for (fragment, _) in &added {
        assert!(!fragment_seeded(&batched, fragment).await);
    }
}

/// A detached commit whose supplied handle is newer than the transaction's
/// read version: the detached manifest descends from the handle, so the bits
/// follow that ancestry, and the file's seed presence follows the snapshot it
/// was written against.
#[tokio::test]
async fn test_zone_seed_flags_detached_commit_with_newer_handle() {
    // Flagged: staged at F, handle at F + 1.
    let dir = TempStrDir::default();
    let flagged = Arc::new(dataset_with_seeded_index(dir.as_str()).await);
    let staged = stage_append(&flagged, 8..16).await;
    let mut newer = (*flagged).clone();
    newer.delete("id = 0").await.unwrap();
    assert!(newer.manifest.version > staged.read_version);
    let detached = CommitBuilder::new(Arc::new(newer))
        .with_detached(true)
        .execute(staged)
        .await
        .unwrap();
    assert!(seed_flags_set(&detached));
    assert!(last_fragment_seeded(&detached).await);

    // Legacy: staged at L (no seed), handle at L + 1 whose commit activated
    // the bits.
    let legacy_dir = TempStrDir::default();
    let dataset = dataset_with_seeded_index(legacy_dir.as_str()).await;
    let legacy = Arc::new(reopen_with_seed_flags_cleared(&dataset).await);
    let staged = stage_append(&legacy, 8..16).await;
    let mut newer = (*legacy).clone();
    newer.delete("id = 0").await.unwrap();
    assert!(seed_flags_set(&newer));
    let detached = CommitBuilder::new(Arc::new(newer))
        .with_detached(true)
        .execute(staged)
        .await
        .unwrap();
    assert!(seed_flags_set(&detached));
    assert!(!last_fragment_seeded(&detached).await);

    // Legacy: staged at L (no seed), handle at L + 1 where the index was
    // dropped without activation: nothing to announce.
    let legacy_dir = TempStrDir::default();
    let dataset = dataset_with_seeded_index(legacy_dir.as_str()).await;
    let legacy = Arc::new(reopen_with_seed_flags_cleared(&dataset).await);
    let staged = stage_append(&legacy, 8..16).await;
    let index_name = legacy.load_indices().await.unwrap()[0].name.clone();
    let mut dropped = (*legacy).clone();
    dropped.drop_index(&index_name).await.unwrap();
    assert!(!seed_flags_set(&dropped));
    let detached = CommitBuilder::new(Arc::new(dropped))
        .with_detached(true)
        .execute(staged)
        .await
        .unwrap();
    assert!(!seed_flags_set(&detached));
    assert!(!last_fragment_seeded(&detached).await);
}

/// Detached snapshots remain valid writing bases when the commit uses that
/// exact snapshot, including the batch publication path.
#[rstest::rstest]
#[tokio::test]
async fn test_zone_seed_flags_same_detached_snapshot(#[values(false, true)] is_batch: bool) {
    let dir = TempStrDir::default();
    let initial = Arc::new(dataset_with_seeded_index(dir.as_str()).await);
    let staged = stage_append(&initial, 100..108).await;
    let detached = Arc::new(
        CommitBuilder::new(initial)
            .with_detached(true)
            .execute(staged)
            .await
            .unwrap(),
    );
    let first = stage_append(&detached, 200..208).await;
    let builder = CommitBuilder::new(detached.clone()).with_detached(true);
    let committed = if is_batch {
        let second = stage_append(&detached, 300..308).await;
        builder
            .execute_batch(vec![first, second])
            .await
            .unwrap()
            .dataset
    } else {
        builder.execute(first).await.unwrap()
    };
    assert!(seed_flags_set(&committed));
    assert!(last_fragment_seeded(&committed).await);
}

/// Imported files may remap field ids in `DataFile.fields` while their
/// footers keep the original ids; carried seeds must follow the physical
/// column, or another column's bounds would prune matching rows.
#[tokio::test]
async fn test_binary_copy_translates_remapped_seed_fields() {
    let source_dir = TempStrDir::default();
    let initial = arrow_array::record_batch!(
        ("discarded", Int32, [0, 0]),
        ("id", Int32, [0, 1]),
        ("val", Int32, [0, 10]),
        ("untouched", Utf8, ["u0", "u1"])
    )
    .unwrap();
    let mut source = Dataset::write(
        RecordBatchIterator::new([Ok(initial.clone())], initial.schema()),
        source_dir.as_str(),
        None,
    )
    .await
    .unwrap();
    source.drop_columns(&["discarded"]).await.unwrap();
    assert_eq!(field_id(&source, "id"), 1);
    assert_eq!(field_id(&source, "val"), 2);
    for column in ["id", "val"] {
        source
            .create_index(
                &[column],
                IndexType::ZoneMap,
                None,
                &zone_map_params(true),
                false,
            )
            .await
            .unwrap();
    }
    for start in [100, 108] {
        source
            .append(
                RecordBatchIterator::new([Ok(rows(start..start + 8, |_| Some(999)))], schema()),
                None,
            )
            .await
            .unwrap();
    }

    let destination_dir = TempStrDir::default();
    let mut destination = dataset_with_seeded_index(destination_dir.as_str()).await;
    assert_seed_flags_set(&destination);
    let mut fragments = source.fragments()[1..].to_vec();
    for fragment in &mut fragments {
        for file in &mut fragment.files {
            assert!(seed_in_file(&source, &file.path, "id").await.is_some());
            assert!(seed_in_file(&source, &file.path, "val").await.is_some());
            destination
                .object_store
                .copy(
                    &source.base.clone().join(DATA_DIR).join(file.path.as_str()),
                    &destination
                        .base
                        .clone()
                        .join(DATA_DIR)
                        .join(file.path.as_str()),
                )
                .await
                .unwrap();
            file.fields = file
                .fields
                .iter()
                .map(|id| id - 1)
                .collect::<Vec<_>>()
                .into();
        }
    }
    let transaction = Transaction::new(
        destination.version_id(),
        Operation::Append { fragments },
        None,
    );
    destination = CommitBuilder::new(Arc::new(destination))
        .execute(transaction)
        .await
        .unwrap();
    let mut flat = destination.scan();
    flat.use_scalar_index(false).filter("val = 999").unwrap();
    assert_eq!(flat.try_into_batch().await.unwrap().num_rows(), 16);

    let metrics = force_binary_copy(&mut destination, 1_000_000).await;
    assert_eq!(metrics.fragments_added, 1);
    assert_eq!(metrics.fragments_removed, 2);
    destination
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let destination = Dataset::open(destination_dir.as_str()).await.unwrap();
    let mut flat = destination.scan();
    flat.use_scalar_index(false).filter("val = 999").unwrap();
    let expected = flat.try_into_batch().await.unwrap().num_rows();
    assert_eq!(expected, 16);
    let mut indexed = destination.scan();
    indexed.filter("val = 999").unwrap();
    let plan = indexed.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ScalarIndexQuery") && plan.contains("ZoneMap"),
        "{plan}"
    );
    assert_eq!(indexed.try_into_batch().await.unwrap().num_rows(), expected);
}

// --- Zone write seeds from the dataset config --------------------------------
//
// `lance.zone_write_seeds.*` seeds every column that no seeded ZoneMap index
// covers, so an index created later can be built from the seeds.

const CONFIG_COLUMNS: [&str; 4] = ["id", "val", "name", "ts"];
/// Columns of [`config_schema`] whose types the config never seeds.
const UNSEEDED_CONFIG_COLUMNS: [&str; 2] = ["vec", "meta"];

fn config_meta_field() -> Arc<ArrowField> {
    Arc::new(ArrowField::new("a", DataType::Int32, true))
}

fn config_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int64, false),
        ArrowField::new("val", DataType::Int64, true),
        ArrowField::new("name", DataType::Utf8, true),
        ArrowField::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        ArrowField::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                4,
            ),
            true,
        ),
        ArrowField::new(
            "meta",
            DataType::Struct(vec![config_meta_field()].into()),
            true,
        ),
    ]))
}

fn config_rows(ids: std::ops::Range<i64>) -> RecordBatch {
    RecordBatch::try_new(
        config_schema(),
        vec![
            Arc::new(Int64Array::from_iter_values(ids.clone())),
            Arc::new(Int64Array::from_iter(
                ids.clone().map(|i| (i % 7 != 3).then_some(i)),
            )),
            Arc::new(StringArray::from_iter(
                ids.clone()
                    .map(|i| (i % 5 != 2).then(|| format!("n{i:03}"))),
            )),
            Arc::new(TimestampMicrosecondArray::from_iter(
                ids.clone()
                    .map(|i| Some(1_700_000_000_000_000 + i * 1_000_000)),
            )),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    ids.clone().map(|i| Some(vec![Some(i as f32); 4])),
                    4,
                ),
            ),
            Arc::new(StructArray::from(vec![(
                config_meta_field(),
                Arc::new(Int32Array::from_iter(ids.map(|i| Some(i as i32)))) as ArrayRef,
            )])),
        ],
    )
    .unwrap()
}

/// Ten rows in one fragment, written before any config is set.
async fn dataset_with_config_rows(uri: &str) -> Dataset {
    Dataset::write(
        RecordBatchIterator::new([Ok(config_rows(0..10))], config_schema()),
        uri,
        None,
    )
    .await
    .unwrap()
}

async fn append_config_rows(dataset: &mut Dataset, ids: std::ops::Range<i64>) {
    dataset
        .append(
            RecordBatchIterator::new([Ok(config_rows(ids))], config_schema()),
            None,
        )
        .await
        .unwrap();
}

async fn enable_zone_write_seeds(dataset: &mut Dataset, rows_per_zone: u64) {
    let rows_per_zone = rows_per_zone.to_string();
    dataset
        .update_config([
            (ZONE_WRITE_SEEDS_ENABLED_KEY, "true"),
            (ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY, rows_per_zone.as_str()),
        ])
        .await
        .unwrap();
}

/// The `rows_per_zone` the seed locator of `column` declares in the data file
/// serving it, or `None` when that file carries no seed for the column.
async fn seed_locator_rows_per_zone(
    dataset: &Dataset,
    fragment: &Fragment,
    column: &str,
) -> Option<u64> {
    let file = serving_file(fragment, field_id(dataset, column));
    let reader = open_data_file(dataset, &file.path).await;
    let value = reader
        .metadata()
        .file_schema
        .metadata
        .get(&format!("{SEED_META_KEY_PREFIX}{column}"))?
        .clone();
    Some(value.split_once(':').unwrap().1.parse().unwrap())
}

/// The commit that enables the setting publishes the seed flags, so the next
/// append seeds every column at the configured size; earlier files stay
/// unseeded.
#[tokio::test]
async fn test_config_seeds_columns_without_index() {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_config_rows(dir.as_str()).await;
    assert!(!seed_flags_set(&dataset));
    enable_zone_write_seeds(&mut dataset, ROWS_PER_ZONE).await;
    assert_seed_flags_set(&dataset);

    append_config_rows(&mut dataset, 10..20).await;
    let fragments = dataset.fragments().clone();
    assert_eq!(fragments.len(), 2);
    for column in CONFIG_COLUMNS {
        assert_eq!(
            seed_locator_rows_per_zone(&dataset, &fragments[0], column).await,
            None,
            "first file, column {column}"
        );
        assert_eq!(
            seed_locator_rows_per_zone(&dataset, &fragments[1], column).await,
            Some(ROWS_PER_ZONE),
            "appended file, column {column}"
        );
    }
    for column in UNSEEDED_CONFIG_COLUMNS {
        assert!(
            !fragment_has_seed_locator(&dataset, &fragments[1], column).await,
            "appended file, unsupported column {column}"
        );
    }
}

/// Whether any data file of `fragment` carries a seed locator for `column`.
/// Nested columns may have no file that stores the parent field itself.
async fn fragment_has_seed_locator(dataset: &Dataset, fragment: &Fragment, column: &str) -> bool {
    let key = format!("{SEED_META_KEY_PREFIX}{column}");
    for file in &fragment.files {
        let reader = open_data_file(dataset, &file.path).await;
        if reader.metadata().file_schema.metadata.contains_key(&key) {
            return true;
        }
    }
    false
}

/// A snapshot that enables the setting but does not carry the seed flags
/// writes no seed; its commit sets the flags and the next write is seeded.
#[tokio::test]
async fn test_config_seeds_wait_for_seed_flags() {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_config_rows(dir.as_str()).await;
    enable_zone_write_seeds(&mut dataset, ROWS_PER_ZONE).await;
    let mut legacy = reopen_with_seed_flags_cleared(&dataset).await;

    append_config_rows(&mut legacy, 10..20).await;
    assert_seed_flags_set(&legacy);
    append_config_rows(&mut legacy, 20..30).await;
    let fragments = legacy.fragments().clone();
    assert_eq!(fragments.len(), 3);
    for column in CONFIG_COLUMNS {
        assert_eq!(
            seed_locator_rows_per_zone(&legacy, &fragments[1], column).await,
            None,
            "append without flags, column {column}"
        );
        assert_eq!(
            seed_locator_rows_per_zone(&legacy, &fragments[2], column).await,
            Some(ROWS_PER_ZONE),
            "append with flags, column {column}"
        );
    }
}

/// Without the setting, or with it set to false, appends carry no seeds and
/// the seed flags stay unset.
#[rstest::rstest]
#[case::absent(None)]
#[case::disabled(Some("false"))]
#[tokio::test]
async fn test_config_seeds_off(#[case] enabled: Option<&'static str>) {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_config_rows(dir.as_str()).await;
    if let Some(enabled) = enabled {
        dataset
            .update_config([
                (ZONE_WRITE_SEEDS_ENABLED_KEY, enabled),
                (ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY, "4"),
            ])
            .await
            .unwrap();
    }
    append_config_rows(&mut dataset, 10..20).await;
    assert!(!seed_flags_set(&dataset));
    let fragments = dataset.fragments().clone();
    for column in CONFIG_COLUMNS {
        assert_eq!(
            seed_locator_rows_per_zone(&dataset, &fragments[1], column).await,
            None,
            "column {column}"
        );
    }
}

/// A ZoneMap index created after a seeded append reuses the configured seeds
/// for that fragment and answers queries like a scan.
#[tokio::test]
async fn test_index_built_from_config_seeds() {
    use crate::index::scalar::zonemap::SEED_REUSE_REPORTS;

    let dir = TempStrDir::default();
    let mut dataset = dataset_with_config_rows(dir.as_str()).await;
    enable_zone_write_seeds(&mut dataset, ROWS_PER_ZONE).await;
    append_config_rows(&mut dataset, 10..20).await;
    let appended = dataset.fragments()[1].id;

    // An index without a null bitmap ignores the positions the seeds carry.
    let params =
        ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap).with_params(&serde_json::json!({
            "rows_per_zone": ROWS_PER_ZONE,
            "use_seeds": true,
            "store_null_bitmap": false,
        }));
    let reports = SEED_REUSE_REPORTS
        .scope(std::sync::Mutex::new(Vec::new()), async {
            dataset
                .create_index(&["val"], IndexType::ZoneMap, None, &params, false)
                .await
                .unwrap();
            SEED_REUSE_REPORTS.with(|reports| reports.lock().unwrap().clone())
        })
        .await;
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].reused_fragments, vec![appended]);

    for filter in ["val >= 5 AND val < 15", "val = 12", "val IS NULL"] {
        let mut indexed = dataset.scan();
        indexed.filter(filter).unwrap().project(&["id"]).unwrap();
        let indexed = indexed.try_into_batch().await.unwrap();
        let mut unindexed = dataset.scan();
        unindexed
            .use_scalar_index(false)
            .filter(filter)
            .unwrap()
            .project(&["id"])
            .unwrap();
        let expected = unindexed.try_into_batch().await.unwrap();
        assert!(expected.num_rows() > 0, "filter {filter}");
        assert_eq!(indexed, expected, "filter {filter}");
    }
}

/// Whether a ZoneMap index on `val` built from configured seeds reuses the
/// appended fragment, depending on the null-position setting of the seeds
/// (`None` leaves the key absent) and the index's `store_null_bitmap`.
/// Results of null filters always match a scan.
#[rstest::rstest]
#[case::default_seeds_bitmap_index(None, true, true)]
#[case::default_seeds_no_bitmap_index(None, false, true)]
#[case::explicit_seeds_bitmap_index(Some("true"), true, true)]
#[case::no_position_seeds_bitmap_index(Some("false"), true, false)]
#[tokio::test]
async fn test_config_seed_null_positions_serve_index(
    #[case] seeds_store_null_bitmap: Option<&'static str>,
    #[case] index_store_null_bitmap: bool,
    #[case] expect_reuse: bool,
) {
    use crate::index::scalar::zonemap::SEED_REUSE_REPORTS;

    let dir = TempStrDir::default();
    let mut dataset = dataset_with_config_rows(dir.as_str()).await;
    enable_zone_write_seeds(&mut dataset, ROWS_PER_ZONE).await;
    if let Some(value) = seeds_store_null_bitmap {
        dataset
            .update_config([(ZONE_WRITE_SEEDS_STORE_NULL_BITMAP_KEY, value)])
            .await
            .unwrap();
    }
    append_config_rows(&mut dataset, 10..20).await;
    let appended = dataset.fragments()[1].id;

    let params =
        ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap).with_params(&serde_json::json!({
            "rows_per_zone": ROWS_PER_ZONE,
            "use_seeds": true,
            "store_null_bitmap": index_store_null_bitmap,
        }));
    let reports = SEED_REUSE_REPORTS
        .scope(std::sync::Mutex::new(Vec::new()), async {
            dataset
                .create_index(&["val"], IndexType::ZoneMap, None, &params, false)
                .await
                .unwrap();
            SEED_REUSE_REPORTS.with(|reports| reports.lock().unwrap().clone())
        })
        .await;
    assert_eq!(reports.len(), 1);
    if expect_reuse {
        assert_eq!(reports[0].reused_fragments, vec![appended]);
        assert!(!reports[0].scanned_fragments.contains(&appended));
    } else {
        // A seed without null positions is rejected and its fragment scanned.
        assert!(reports[0].reused_fragments.is_empty());
        assert!(reports[0].partial_fragments.is_empty());
        assert!(reports[0].scanned_fragments.contains(&appended));
    }

    assert_indexed_ids_match_scan(
        &dataset,
        &["val IS NULL", "val IS NOT NULL", "val >= 5 AND val < 15"],
        "null positions",
    )
    .await;
}

/// A column with a seeded ZoneMap index follows that index's zone size even
/// when the configured size is smaller; a ZoneMap index without seeds does
/// not take part, so its column gets the configured size like any other.
#[rstest::rstest]
#[case::seeded_index(true, 8)]
#[case::unseeded_index(false, ROWS_PER_ZONE)]
#[tokio::test]
async fn test_config_seed_size_with_zone_map_index(
    #[case] use_seeds: bool,
    #[case] expected_val_rows_per_zone: u64,
) {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_config_rows(dir.as_str()).await;
    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
        .with_params(&serde_json::json!({"rows_per_zone": 8, "use_seeds": use_seeds}));
    dataset
        .create_index(&["val"], IndexType::ZoneMap, None, &params, false)
        .await
        .unwrap();
    enable_zone_write_seeds(&mut dataset, ROWS_PER_ZONE).await;
    append_config_rows(&mut dataset, 10..20).await;

    let appended = dataset.fragments()[1].clone();
    for column in CONFIG_COLUMNS {
        let expected = if column == "val" {
            expected_val_rows_per_zone
        } else {
            ROWS_PER_ZONE
        };
        assert_eq!(
            seed_locator_rows_per_zone(&dataset, &appended, column).await,
            Some(expected),
            "column {column}"
        );
    }
}

/// Binary-copy compaction declares the configured zone size for carried
/// seeds of columns without a seeded index, as the write path does.
#[tokio::test]
async fn test_binary_copy_declares_configured_rows_per_zone() {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_config_rows(dir.as_str()).await;
    enable_zone_write_seeds(&mut dataset, ROWS_PER_ZONE).await;
    append_config_rows(&mut dataset, 10..20).await;
    append_config_rows(&mut dataset, 20..30).await;
    // Drop the unseeded first fragment so every compaction input is seeded.
    dataset.delete("id < 10").await.unwrap();
    assert_eq!(dataset.fragments().len(), 2);

    let metrics = force_binary_copy(&mut dataset, 1_000_000).await;
    assert_eq!(metrics.fragments_added, 1);
    let output = dataset.fragments()[0].clone();
    assert_eq!(output.physical_rows, Some(20));
    for column in CONFIG_COLUMNS {
        assert_eq!(
            seed_locator_rows_per_zone(&dataset, &output, column).await,
            Some(ROWS_PER_ZONE),
            "column {column}"
        );
    }
}

/// Setting an invalid zone write seed value is rejected by its commit and
/// leaves the config as it was; unrelated config changes still commit.
#[rstest::rstest]
#[case::enabled(ZONE_WRITE_SEEDS_ENABLED_KEY, "yes")]
#[case::rows_per_zone(ZONE_WRITE_SEEDS_ROWS_PER_ZONE_KEY, "0")]
#[case::store_null_bitmap(ZONE_WRITE_SEEDS_STORE_NULL_BITMAP_KEY, "yes")]
#[tokio::test]
async fn test_config_rejects_invalid_zone_write_seeds_value(
    #[case] key: &'static str,
    #[case] value: &'static str,
) {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_config_rows(dir.as_str()).await;
    enable_zone_write_seeds(&mut dataset, ROWS_PER_ZONE).await;
    let before = dataset.config().clone();
    let version = dataset.manifest.version;

    let err = dataset.update_config([(key, value)]).await.unwrap_err();
    assert!(err.to_string().contains(key), "{err}");
    assert_eq!(dataset.config(), &before);
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(reopened.manifest.version, version);
    assert_eq!(reopened.config(), &before);

    dataset
        .update_config([("app.owner", "analytics")])
        .await
        .unwrap();
    assert_eq!(
        dataset.config().get("app.owner").map(String::as_str),
        Some("analytics")
    );
    assert_eq!(
        dataset
            .config()
            .get(ZONE_WRITE_SEEDS_ENABLED_KEY)
            .map(String::as_str),
        Some("true")
    );
}

/// Disabling the setting stops seeding new files while the flags and the
/// seeds already written stay; re-enabling with another size seeds new files
/// at that size, also after a reopen.
#[tokio::test]
async fn test_config_seeds_lifecycle() {
    const RESIZED_ROWS_PER_ZONE: u64 = 8;

    async fn assert_locators(dataset: &Dataset, expected: &[Option<u64>]) {
        let fragments = dataset.fragments().clone();
        assert_eq!(fragments.len(), expected.len());
        for (fragment, expected) in fragments.iter().zip(expected) {
            for column in CONFIG_COLUMNS {
                assert_eq!(
                    seed_locator_rows_per_zone(dataset, fragment, column).await,
                    *expected,
                    "fragment {}, column {column}",
                    fragment.id
                );
            }
        }
    }

    let dir = TempStrDir::default();
    let mut dataset = dataset_with_config_rows(dir.as_str()).await;
    enable_zone_write_seeds(&mut dataset, ROWS_PER_ZONE).await;
    append_config_rows(&mut dataset, 10..20).await;
    assert_locators(&dataset, &[None, Some(ROWS_PER_ZONE)]).await;

    dataset
        .update_config([(ZONE_WRITE_SEEDS_ENABLED_KEY, "false")])
        .await
        .unwrap();
    append_config_rows(&mut dataset, 20..30).await;
    assert_seed_flags_set(&dataset);
    assert_locators(&dataset, &[None, Some(ROWS_PER_ZONE), None]).await;

    enable_zone_write_seeds(&mut dataset, RESIZED_ROWS_PER_ZONE).await;
    append_config_rows(&mut dataset, 30..40).await;
    assert_locators(
        &dataset,
        &[None, Some(ROWS_PER_ZONE), None, Some(RESIZED_ROWS_PER_ZONE)],
    )
    .await;

    let mut dataset = Dataset::open(dir.as_str()).await.unwrap();
    append_config_rows(&mut dataset, 40..50).await;
    assert_locators(
        &dataset,
        &[
            None,
            Some(ROWS_PER_ZONE),
            None,
            Some(RESIZED_ROWS_PER_ZONE),
            Some(RESIZED_ROWS_PER_ZONE),
        ],
    )
    .await;
}

/// Indexed results of `filters` on `dataset` equal those of a scan without
/// scalar indices, compared as sorted `id` lists.
async fn assert_indexed_ids_match_scan(dataset: &Dataset, filters: &[&str], context: &str) {
    use arrow::array::AsArray;
    use arrow::datatypes::Int64Type;

    fn sorted_ids(batch: &RecordBatch) -> Vec<i64> {
        let mut ids = batch["id"].as_primitive::<Int64Type>().values().to_vec();
        ids.sort_unstable();
        ids
    }

    for filter in filters {
        let mut indexed = dataset.scan();
        indexed.filter(filter).unwrap().project(&["id"]).unwrap();
        let indexed = sorted_ids(&indexed.try_into_batch().await.unwrap());
        let mut unindexed = dataset.scan();
        unindexed
            .use_scalar_index(false)
            .filter(filter)
            .unwrap()
            .project(&["id"])
            .unwrap();
        let expected = sorted_ids(&unindexed.try_into_batch().await.unwrap());
        assert!(!expected.is_empty(), "{context}: filter {filter}");
        assert_eq!(indexed, expected, "{context}: filter {filter}");
    }
}

/// ZoneMap params that build from seeds without a null bitmap.
fn config_seed_index_params(rows_per_zone: u64) -> ScalarIndexParams {
    ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap).with_params(&serde_json::json!({
        "rows_per_zone": rows_per_zone,
        "use_seeds": true,
        "store_null_bitmap": false,
    }))
}

fn typed_schema(data_type: &DataType) -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int64, false),
        ArrowField::new("col", data_type.clone(), true),
    ]))
}

/// Deterministic values in `0..61` with a null every ninth row.
fn typed_value(i: i64) -> Option<i64> {
    (i % 9 != 4).then_some((i * 7) % 61)
}

/// [`typed_value`] as a float, with NaN, infinities and negative zero mixed in.
fn typed_float(i: i64) -> Option<f64> {
    typed_value(i).map(|v| match i % 13 {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => f64::NEG_INFINITY,
        3 => -0.0,
        _ => v as f64 / 2.0,
    })
}

fn typed_column(data_type: &DataType, ids: std::ops::Range<i64>) -> ArrayRef {
    use arrow_array::{
        BooleanArray, Date32Array, Decimal128Array, DurationMicrosecondArray, Float32Array,
        Float64Array, Int8Array, LargeBinaryArray, LargeStringArray, Time32SecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray, UInt16Array, UInt64Array,
    };

    let values = ids.clone().map(typed_value);
    match data_type {
        DataType::Boolean => Arc::new(BooleanArray::from_iter(
            values.map(|v| v.map(|v| v % 3 == 0)),
        )),
        DataType::Int8 => Arc::new(Int8Array::from_iter(values.map(|v| v.map(|v| v as i8)))),
        DataType::Int32 => Arc::new(Int32Array::from_iter(values.map(|v| v.map(|v| v as i32)))),
        DataType::Int64 => Arc::new(Int64Array::from_iter(values)),
        DataType::UInt16 => Arc::new(UInt16Array::from_iter(values.map(|v| v.map(|v| v as u16)))),
        DataType::UInt64 => Arc::new(UInt64Array::from_iter(values.map(|v| v.map(|v| v as u64)))),
        DataType::Float32 => Arc::new(Float32Array::from_iter(
            ids.map(|i| typed_float(i).map(|v| v as f32)),
        )),
        DataType::Float64 => Arc::new(Float64Array::from_iter(ids.map(typed_float))),
        DataType::Date32 => Arc::new(Date32Array::from_iter(
            values.map(|v| v.map(|v| 19_000 + v as i32)),
        )),
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            Arc::new(TimestampMillisecondArray::from_iter(
                values.map(|v| v.map(|v| 1_700_000_000_000 + v * 1_000)),
            ))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, Some(tz)) => Arc::new(
            TimestampNanosecondArray::from_iter(
                values.map(|v| v.map(|v| 1_700_000_000_000_000_000 + v * 1_000)),
            )
            .with_timezone(tz.clone()),
        ),
        DataType::Duration(TimeUnit::Microsecond) => Arc::new(DurationMicrosecondArray::from_iter(
            values.map(|v| v.map(|v| v * 1_000)),
        )),
        DataType::Time32(TimeUnit::Second) => Arc::new(Time32SecondArray::from_iter(
            values.map(|v| v.map(|v| (v * 60) as i32)),
        )),
        DataType::Utf8 => Arc::new(StringArray::from_iter(
            values.map(|v| v.map(|v| format!("v{v:03}"))),
        )),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from_iter(
            values.map(|v| v.map(|v| format!("v{v:03}"))),
        )),
        DataType::Binary => Arc::new(BinaryArray::from_iter(
            values.map(|v| v.map(|v| v.to_be_bytes().to_vec())),
        )),
        DataType::LargeBinary => Arc::new(LargeBinaryArray::from_iter(
            values.map(|v| v.map(|v| v.to_be_bytes().to_vec())),
        )),
        DataType::Decimal128(precision, scale) => Arc::new(
            Decimal128Array::from_iter(values.map(|v| v.map(|v| v as i128 * 10_000)))
                .with_precision_and_scale(*precision, *scale)
                .unwrap(),
        ),
        other => panic!("no generator for {other}"),
    }
}

fn typed_rows(data_type: &DataType, ids: std::ops::Range<i64>) -> RecordBatch {
    RecordBatch::try_new(
        typed_schema(data_type),
        vec![
            Arc::new(Int64Array::from_iter_values(ids.clone())),
            typed_column(data_type, ids),
        ],
    )
    .unwrap()
}

const INT_FILTERS: &[&str] = &["col >= 10 AND col < 30", "col = 21"];
const FLOAT_FILTERS: &[&str] = &["col >= 2.0 AND col < 20.0", "col > 1000.0"];
const STRING_FILTERS: &[&str] = &["col >= 'v010' AND col < 'v030'", "col = 'v021'"];

/// For every candidate type family, an append after enabling the setting
/// seeds the column, a ZoneMap index created afterwards reuses that seed for
/// the appended fragment, and indexed filters match a scan. `filters` adds
/// value predicates to the null checks every type runs.
#[rstest::rstest]
#[case::boolean(DataType::Boolean, &["col = true"])]
#[case::int8(DataType::Int8, INT_FILTERS)]
#[case::int32(DataType::Int32, INT_FILTERS)]
#[case::int64(DataType::Int64, INT_FILTERS)]
#[case::uint16(DataType::UInt16, INT_FILTERS)]
#[case::uint64(DataType::UInt64, INT_FILTERS)]
#[case::float32(DataType::Float32, FLOAT_FILTERS)]
#[case::float64(DataType::Float64, FLOAT_FILTERS)]
#[case::date32(DataType::Date32, &["col >= DATE '2022-02-07'"])]
#[case::timestamp_ms(
    DataType::Timestamp(TimeUnit::Millisecond, None),
    &["col >= TIMESTAMP '2023-11-14 22:13:50'"]
)]
#[case::timestamp_ns_utc(DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())), &[])]
#[case::duration_us(DataType::Duration(TimeUnit::Microsecond), &[])]
#[case::time32_s(
    DataType::Time32(TimeUnit::Second),
    &["col >= arrow_cast(1800, 'Time32(Second)')"]
)]
#[case::utf8(DataType::Utf8, STRING_FILTERS)]
#[case::large_utf8(DataType::LargeUtf8, STRING_FILTERS)]
#[case::binary(DataType::Binary, &["col >= X'000000000000001E'"])]
#[case::large_binary(DataType::LargeBinary, &["col >= X'000000000000001E'"])]
#[case::decimal128(DataType::Decimal128(38, 4), &["col >= 30"])]
#[tokio::test]
async fn test_config_seed_round_trip_per_type(
    #[case] data_type: DataType,
    #[case] filters: &'static [&'static str],
) {
    use crate::index::scalar::zonemap::SEED_REUSE_REPORTS;
    const TYPED_ROWS_PER_ZONE: u64 = 8;

    let dir = TempStrDir::default();
    let schema = typed_schema(&data_type);
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(typed_rows(&data_type, 0..40))], schema.clone()),
        dir.as_str(),
        None,
    )
    .await
    .unwrap();
    enable_zone_write_seeds(&mut dataset, TYPED_ROWS_PER_ZONE).await;
    dataset
        .append(
            RecordBatchIterator::new([Ok(typed_rows(&data_type, 40..140))], schema),
            None,
        )
        .await
        .unwrap();
    let fragments = dataset.fragments().clone();
    assert_eq!(fragments.len(), 2, "{data_type}");
    let appended = fragments[1].id;
    assert_eq!(
        seed_locator_rows_per_zone(&dataset, &fragments[1], "col").await,
        Some(TYPED_ROWS_PER_ZONE),
        "{data_type}"
    );

    let params = config_seed_index_params(TYPED_ROWS_PER_ZONE);
    let reports = SEED_REUSE_REPORTS
        .scope(std::sync::Mutex::new(Vec::new()), async {
            dataset
                .create_index(&["col"], IndexType::ZoneMap, None, &params, false)
                .await
                .unwrap();
            SEED_REUSE_REPORTS.with(|reports| reports.lock().unwrap().clone())
        })
        .await;
    assert_eq!(reports.len(), 1, "{data_type}");
    assert_eq!(reports[0].reused_fragments, vec![appended], "{data_type}");

    let mut all_filters: Vec<&str> = filters.to_vec();
    all_filters.extend(["col IS NULL", "col IS NOT NULL"]);
    assert_indexed_ids_match_scan(&dataset, &all_filters, &data_type.to_string()).await;
}

/// Binary-copy compaction of files whose configured seeds were written at
/// different sizes still carries a seed, an index whose zones end at the
/// input file boundaries is built from it without a scan, and it answers
/// filters like a scan.
#[tokio::test]
async fn test_binary_copy_carries_config_seeds_of_different_sizes() {
    use crate::index::scalar::zonemap::SEED_REUSE_REPORTS;
    const RESIZED_ROWS_PER_ZONE: u64 = 8;
    // Each input holds 10 rows, so no carried record crosses a target zone.
    const INDEX_ROWS_PER_ZONE: u64 = 10;

    let dir = TempStrDir::default();
    let mut dataset = dataset_with_config_rows(dir.as_str()).await;
    enable_zone_write_seeds(&mut dataset, ROWS_PER_ZONE).await;
    append_config_rows(&mut dataset, 10..20).await;
    enable_zone_write_seeds(&mut dataset, RESIZED_ROWS_PER_ZONE).await;
    append_config_rows(&mut dataset, 20..30).await;
    // Drop the unseeded first fragment so every compaction input is seeded.
    dataset.delete("id < 10").await.unwrap();
    let inputs = dataset.fragments().clone();
    assert_eq!(inputs.len(), 2);
    assert_eq!(
        seed_locator_rows_per_zone(&dataset, &inputs[0], "val").await,
        Some(ROWS_PER_ZONE)
    );
    assert_eq!(
        seed_locator_rows_per_zone(&dataset, &inputs[1], "val").await,
        Some(RESIZED_ROWS_PER_ZONE)
    );

    let metrics = force_binary_copy(&mut dataset, 1_000_000).await;
    assert_eq!(metrics.fragments_added, 1);
    let output = dataset.fragments()[0].clone();
    assert_eq!(output.physical_rows, Some(20));
    for column in CONFIG_COLUMNS {
        assert!(
            seed_locator_rows_per_zone(&dataset, &output, column)
                .await
                .is_some(),
            "column {column}"
        );
    }

    let params = config_seed_index_params(INDEX_ROWS_PER_ZONE);
    let reports = SEED_REUSE_REPORTS
        .scope(std::sync::Mutex::new(Vec::new()), async {
            dataset
                .create_index(&["val"], IndexType::ZoneMap, None, &params, false)
                .await
                .unwrap();
            SEED_REUSE_REPORTS.with(|reports| reports.lock().unwrap().clone())
        })
        .await;
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].reused_fragments, vec![output.id]);
    assert!(reports[0].partial_fragments.is_empty());
    assert!(reports[0].scanned_fragments.is_empty());
    assert_eq!(reports[0].scanned_rows, 0);

    assert_indexed_ids_match_scan(
        &dataset,
        &["val >= 12 AND val < 22", "val = 25", "val IS NULL"],
        "binary copy",
    )
    .await;
}

/// Rows of the creating write in [`test_config_seed_boundary_matrix`]; that
/// write happens before the config is enabled and carries no seed.
const BOUNDARY_CREATE_ROWS: i64 = 11;
/// Fragments appended after the config is enabled.
const BOUNDARY_APPENDED_FRAGMENTS: i64 = 3;

fn boundary_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int64, false),
        ArrowField::new("val", DataType::Int64, true),
    ]))
}

/// The generator every expectation is computed from: three times the row
/// position, null on every seventh position.
fn boundary_value(position: i64) -> Option<i64> {
    (position % 7 != 0).then_some(position * 3)
}

fn boundary_rows(positions: std::ops::Range<i64>) -> RecordBatch {
    RecordBatch::try_new(
        boundary_schema(),
        vec![
            Arc::new(Int64Array::from_iter_values(positions.clone())),
            Arc::new(Int64Array::from_iter(positions.map(boundary_value))),
        ],
    )
    .unwrap()
}

/// Appends `positions` as one fragment, streamed in batches of `batch_size`.
async fn append_boundary_fragment(
    dataset: &mut Dataset,
    positions: std::ops::Range<i64>,
    batch_size: i64,
) {
    let mut batches = Vec::new();
    let mut start = positions.start;
    while start < positions.end {
        let end = (start + batch_size).min(positions.end);
        batches.push(boundary_rows(start..end));
        start = end;
    }
    dataset
        .append(
            RecordBatchIterator::new(
                batches
                    .into_iter()
                    .map(Ok::<RecordBatch, arrow_schema::ArrowError>),
                boundary_schema(),
            ),
            Some(WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: (positions.end - positions.start) as usize,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
}

/// `(id, val)` pairs, sorted by `id`.
type BoundaryRows = Vec<(i64, Option<i64>)>;

/// Sorted `(id, val)` pairs a filter returns.
async fn boundary_query(dataset: &Dataset, filter: &str, use_index: bool) -> BoundaryRows {
    use arrow::array::AsArray;
    use arrow::datatypes::Int64Type;

    let mut scan = dataset.scan();
    scan.use_scalar_index(use_index)
        .filter(filter)
        .unwrap()
        .project(&["id", "val"])
        .unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let mut rows: BoundaryRows = batch["id"]
        .as_primitive::<Int64Type>()
        .values()
        .iter()
        .copied()
        .zip(batch["val"].as_primitive::<Int64Type>().iter())
        .collect();
    rows.sort_unstable();
    rows
}

/// Configured seeds of one zone size serve a ZoneMap index of another size
/// across fragment and batch boundaries. A seed record is reused when it lies
/// inside one index zone (index zones restart at every fragment); records that
/// cross an index zone boundary leave their fragment partly scanned. Indexed
/// results always match both the generator and an unindexed scan.
#[rstest::rstest]
#[case::equal_aligned(8, 8, 64, 64)]
#[case::equal_partial_final_zone(8, 8, 61, 61)]
#[case::batches_cross_zones(8, 8, 61, 5)]
#[case::coarsen_multiple(8, 32, 64, 13)]
#[case::coarsen_multiple_partial_fragment(8, 32, 61, 13)]
#[case::coarsen_not_multiple(8, 20, 64, 13)]
#[case::seed_coarser_than_index(32, 8, 64, 13)]
#[case::fragment_shorter_than_seed_zone(8, 8, 5, 5)]
#[tokio::test]
async fn test_config_seed_boundary_matrix(
    #[case] seed_rows_per_zone: u64,
    #[case] index_rows_per_zone: u64,
    #[case] rows_per_fragment: i64,
    #[case] batch_size: i64,
) {
    use crate::index::scalar::zonemap::SEED_REUSE_REPORTS;

    let context = format!(
        "seed {seed_rows_per_zone}, index {index_rows_per_zone}, \
         fragment {rows_per_fragment}, batch {batch_size}"
    );
    let dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(
            [Ok(boundary_rows(0..BOUNDARY_CREATE_ROWS))],
            boundary_schema(),
        ),
        dir.as_str(),
        None,
    )
    .await
    .unwrap();
    let creating = dataset.fragments()[0].id;
    enable_zone_write_seeds(&mut dataset, seed_rows_per_zone).await;
    for fragment in 0..BOUNDARY_APPENDED_FRAGMENTS {
        let start = BOUNDARY_CREATE_ROWS + fragment * rows_per_fragment;
        append_boundary_fragment(&mut dataset, start..start + rows_per_fragment, batch_size).await;
    }
    let fragments = dataset.fragments().clone();
    assert_eq!(
        fragments.len() as i64,
        BOUNDARY_APPENDED_FRAGMENTS + 1,
        "{context}"
    );
    let mut appended: Vec<u64> = fragments
        .iter()
        .map(|f| f.id)
        .filter(|id| *id != creating)
        .collect();
    appended.sort_unstable();
    for fragment in &fragments[1..] {
        assert_eq!(
            seed_locator_rows_per_zone(&dataset, fragment, "val").await,
            Some(seed_rows_per_zone),
            "{context}: fragment {}",
            fragment.id
        );
    }

    let params = config_seed_index_params(index_rows_per_zone);
    let reports = SEED_REUSE_REPORTS
        .scope(std::sync::Mutex::new(Vec::new()), async {
            dataset
                .create_index(&["val"], IndexType::ZoneMap, None, &params, false)
                .await
                .unwrap();
            SEED_REUSE_REPORTS.with(|reports| reports.lock().unwrap().clone())
        })
        .await;
    assert_eq!(reports.len(), 1, "{context}: {reports:?}");
    let report = &reports[0];
    let sorted = |ids: &[u64]| {
        let mut ids = ids.to_vec();
        ids.sort_unstable();
        ids
    };
    let reused = sorted(&report.reused_fragments);
    let partial = sorted(&report.partial_fragments);
    let scanned = sorted(&report.scanned_fragments);
    // A partly reused fragment is listed as both partial and scanned.
    assert_eq!(
        reused.len() + scanned.len(),
        fragments.len(),
        "{context}: {report:?}"
    );
    assert!(
        reused.iter().all(|id| !scanned.contains(id)),
        "{context}: {report:?}"
    );
    assert!(
        partial.iter().all(|id| scanned.contains(id)),
        "{context}: {report:?}"
    );

    // `scanned_rows` includes the creating fragment's BOUNDARY_CREATE_ROWS (11)
    // rows.
    let index_multiple_of_seed = index_rows_per_zone.is_multiple_of(seed_rows_per_zone);
    if index_multiple_of_seed {
        // Every seed record lies inside one index zone: no appended fragment
        // is scanned.
        assert_eq!(reused, appended, "{context}: {report:?}");
        assert!(partial.is_empty(), "{context}: {report:?}");
        assert_eq!(scanned, vec![creating], "{context}: {report:?}");
        assert_eq!(report.scanned_rows, 11, "{context}: {report:?}");
    } else if seed_rows_per_zone < index_rows_per_zone {
        // 8-row records over index zones [0,20), [20,40), [40,60), [60,64)
        // in a 64-row fragment: the records [16,24) and [56,64) cross an
        // index zone boundary and are scanned, the other six are reused.
        assert_eq!(
            (seed_rows_per_zone, index_rows_per_zone, rows_per_fragment),
            (8, 20, 64),
            "{context}: expectation only derived for this shape"
        );
        assert!(reused.is_empty(), "{context}: {report:?}");
        assert_eq!(partial, appended, "{context}: {report:?}");
        let mut all_fragments = appended.clone();
        all_fragments.push(creating);
        all_fragments.sort_unstable();
        assert_eq!(scanned, all_fragments, "{context}: {report:?}");
        // Six reused records and 16 scanned rows per appended fragment, plus
        // the creating fragment's 11 rows.
        assert_eq!(report.reused_records, 18, "{context}: {report:?}");
        assert_eq!(report.scanned_rows, 59, "{context}: {report:?}");
    } else {
        // In this fixture every 32-row record crosses 8-row index zones and
        // none fits, so each appended seed is rejected before any reuse is
        // recorded and its fragment is scanned in full.
        assert_eq!(
            (seed_rows_per_zone, index_rows_per_zone, rows_per_fragment),
            (32, 8, 64),
            "{context}: expectation only derived for this shape"
        );
        assert!(reused.is_empty(), "{context}: {report:?}");
        assert!(partial.is_empty(), "{context}: {report:?}");
        let mut all_fragments = appended.clone();
        all_fragments.push(creating);
        all_fragments.sort_unstable();
        assert_eq!(scanned, all_fragments, "{context}: {report:?}");
        assert_eq!(report.reused_records, 0, "{context}: {report:?}");
        // The creating fragment's 11 rows plus three fully scanned 64-row
        // fragments.
        assert_eq!(report.scanned_rows, 203, "{context}: {report:?}");
    }

    // Positions at an index zone boundary inside the first appended fragment,
    // and just before and after the first boundary between appended fragments.
    let first_appended = BOUNDARY_CREATE_ROWS;
    assert_boundary_predicates(
        &dataset,
        BOUNDARY_CREATE_ROWS + BOUNDARY_APPENDED_FRAGMENTS * rows_per_fragment,
        first_appended + (index_rows_per_zone as i64).min(rows_per_fragment),
        first_appended + rows_per_fragment,
        &context,
    )
    .await;

    let mut fragment_ranges = vec![(creating, 0..BOUNDARY_CREATE_ROWS)];
    for (k, fragment) in fragments[1..].iter().enumerate() {
        let start = BOUNDARY_CREATE_ROWS + k as i64 * rows_per_fragment;
        fragment_ranges.push((fragment.id, start..start + rows_per_fragment));
    }
    assert_boundary_zone_statistics(&dataset, index_rows_per_zone, &fragment_ranges, &context)
        .await;
}

/// One stored or expected zone: fragment id, zone start, zone length, null
/// count, min, max, NaN count and negative NaN count.
type BoundaryZone = (
    u64,
    u64,
    u64,
    u32,
    Option<i64>,
    Option<i64>,
    u32,
    Option<u32>,
);

/// Every zone stored by the ZoneMap index on `val`, across all its segments,
/// equals the zone computed from the generator: zones of `index_rows_per_zone`
/// rows restart at each fragment's first row, the last one short.
/// `fragment_ranges` maps each fragment id to its generator positions.
async fn assert_boundary_zone_statistics(
    dataset: &Dataset,
    index_rows_per_zone: u64,
    fragment_ranges: &[(u64, std::ops::Range<i64>)],
    context: &str,
) {
    use arrow::array::AsArray;
    use arrow::datatypes::{Int64Type, UInt32Type, UInt64Type};
    use lance_index::scalar::lance_format::LanceIndexStore;
    use lance_index::scalar::zonemap::{ZoneMapIndex, ZoneMapIndexVersion};

    use crate::dataset::index::LanceIndexStoreExt;

    let val = field_id(dataset, "val");
    let indices = dataset.load_indices().await.unwrap();
    let segments: Vec<_> = indices
        .iter()
        .filter(|index| index.fields == vec![val])
        .collect();
    assert!(!segments.is_empty(), "{context}: no index on val");

    let mut batches = Vec::new();
    let mut stored: Vec<BoundaryZone> = Vec::new();
    for segment in segments {
        let store = LanceIndexStore::from_dataset_for_existing(dataset, segment)
            .await
            .unwrap();
        let version = ZoneMapIndexVersion::from_metadata(segment.index_version as u32).unwrap();
        let batch = ZoneMapIndex::load_zone_statistics(&store, version)
            .await
            .unwrap();
        let min = batch["min"].as_primitive::<Int64Type>();
        let max = batch["max"].as_primitive::<Int64Type>();
        let null_count = batch["null_count"].as_primitive::<UInt32Type>();
        let nan_count = batch["nan_count"].as_primitive::<UInt32Type>();
        let negative_nan_count = batch["negative_nan_count"].as_primitive::<UInt32Type>();
        let fragment_id = batch["fragment_id"].as_primitive::<UInt64Type>();
        let zone_start = batch["zone_start"].as_primitive::<UInt64Type>();
        let zone_length = batch["zone_length"].as_primitive::<UInt64Type>();
        for row in 0..batch.num_rows() {
            stored.push((
                fragment_id.value(row),
                zone_start.value(row),
                zone_length.value(row),
                null_count.value(row),
                min.is_valid(row).then(|| min.value(row)),
                max.is_valid(row).then(|| max.value(row)),
                nan_count.value(row),
                negative_nan_count
                    .is_valid(row)
                    .then(|| negative_nan_count.value(row)),
            ));
        }
        batches.push(batch);
    }
    stored.sort_unstable_by_key(|zone| (zone.0, zone.1));

    let zone_rows = index_rows_per_zone as i64;
    let mut expected: Vec<BoundaryZone> = Vec::new();
    for (fragment_id, positions) in fragment_ranges {
        let mut start = positions.start;
        while start < positions.end {
            let end = (start + zone_rows).min(positions.end);
            let values: Vec<Option<i64>> = (start..end).map(boundary_value).collect();
            let non_null = values.iter().flatten();
            expected.push((
                *fragment_id,
                (start - positions.start) as u64,
                (end - start) as u64,
                values.iter().filter(|v| v.is_none()).count() as u32,
                non_null.clone().min().copied(),
                non_null.max().copied(),
                0,
                Some(0),
            ));
            start = end;
        }
    }
    expected.sort_unstable_by_key(|zone| (zone.0, zone.1));

    assert_eq!(
        stored.len(),
        expected.len(),
        "{context}: zone count, stored {batches:?}"
    );
    for (stored, expected) in stored.iter().zip(&expected) {
        // `zone_start` is accepted as a fragment-local offset or as the row
        // address of the zone's first row.
        let row_address = (expected.0 << 32) | expected.1;
        assert!(
            stored.1 == expected.1 || stored.1 == row_address,
            "{context}: zone start of {expected:?}, stored {stored:?} in {batches:?}"
        );
        let mut normalized = *stored;
        normalized.1 = expected.1;
        assert_eq!(
            normalized, *expected,
            "{context}: zone {expected:?}, stored {stored:?} in {batches:?}"
        );
    }
}

/// Range, null and equality predicates around `zone_boundary` and
/// `fragment_boundary` return the rows the generator yields for positions
/// `0..total_rows`, with and without the scalar index.
async fn assert_boundary_predicates(
    dataset: &Dataset,
    total_rows: i64,
    zone_boundary: i64,
    fragment_boundary: i64,
    context: &str,
) {
    let generated: BoundaryRows = (0..total_rows).map(|p| (p, boundary_value(p))).collect();
    let oracle = |keep: &dyn Fn(Option<i64>) -> bool| -> BoundaryRows {
        generated
            .iter()
            .filter(|(_, v)| keep(*v))
            .copied()
            .collect()
    };

    let zone_value = zone_boundary * 3;
    let mut cases: Vec<(String, BoundaryRows)> = vec![
        (
            format!("val < {zone_value}"),
            oracle(&|v: Option<i64>| v.is_some_and(|v| v < zone_value)),
        ),
        (
            format!("val >= {zone_value}"),
            oracle(&|v: Option<i64>| v.is_some_and(|v| v >= zone_value)),
        ),
        (
            "val IS NULL".to_string(),
            oracle(&|v: Option<i64>| v.is_none()),
        ),
        (
            "val IS NOT NULL".to_string(),
            oracle(&|v: Option<i64>| v.is_some()),
        ),
    ];
    for position in [zone_boundary, fragment_boundary - 1, fragment_boundary] {
        let x = position * 3;
        cases.push((format!("val = {x}"), oracle(&|v: Option<i64>| v == Some(x))));
    }

    for (filter, expected) in cases {
        let indexed = boundary_query(dataset, &filter, true).await;
        let unindexed = boundary_query(dataset, &filter, false).await;
        assert_eq!(unindexed, expected, "{context}: unindexed {filter}");
        assert_eq!(indexed, expected, "{context}: indexed {filter}");
    }
}

/// Configured seeds at production zone sizes serve a ZoneMap index over one
/// appended fragment whose length no zone size divides. Seed records are
/// `seed_rows_per_zone` rows from the fragment start with a short final
/// record; with seed records no larger than index zones, a record is reused
/// when it lies inside one index zone and scanned when it straddles an index
/// zone boundary.
#[rstest::rstest]
#[case::small_seed_crosses_index_boundary(8192, 1_000_000, 1_050_000, 100_003)]
#[case::equal_sizes_short_final_zone(1_000_000, 1_000_000, 1_050_000, 100_003)]
#[case::equal_small_sizes(8192, 8192, 20_000, 4_099)]
#[tokio::test]
async fn test_config_seed_boundary_real_sizes(
    #[case] seed_rows_per_zone: u64,
    #[case] index_rows_per_zone: u64,
    #[case] rows_per_fragment: i64,
    #[case] batch_size: i64,
) {
    use crate::index::scalar::zonemap::SEED_REUSE_REPORTS;

    let context = format!(
        "seed {seed_rows_per_zone}, index {index_rows_per_zone}, \
         fragment {rows_per_fragment}, batch {batch_size}"
    );
    let dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(
            [Ok(boundary_rows(0..BOUNDARY_CREATE_ROWS))],
            boundary_schema(),
        ),
        dir.as_str(),
        None,
    )
    .await
    .unwrap();
    let creating = dataset.fragments()[0].id;
    enable_zone_write_seeds(&mut dataset, seed_rows_per_zone).await;
    let first_appended = BOUNDARY_CREATE_ROWS;
    append_boundary_fragment(
        &mut dataset,
        first_appended..first_appended + rows_per_fragment,
        batch_size,
    )
    .await;
    let fragments = dataset.fragments().clone();
    assert_eq!(fragments.len(), 2, "{context}");
    let appended = fragments[1].id;
    assert_eq!(
        seed_locator_rows_per_zone(&dataset, &fragments[1], "val").await,
        Some(seed_rows_per_zone),
        "{context}"
    );

    // Expected reuse, computed from the sizes alone.
    let seed = seed_rows_per_zone as i64;
    let index = index_rows_per_zone as i64;
    assert!(
        seed <= index,
        "{context}: expectation assumes seed <= index"
    );
    let records: Vec<(i64, i64)> = (0..rows_per_fragment)
        .step_by(seed as usize)
        .map(|start| (start, (start + seed).min(rows_per_fragment)))
        .collect();
    let crossing: Vec<&(i64, i64)> = records
        .iter()
        .filter(|(start, end)| start / index != (end - 1) / index)
        .collect();
    let expected_reused_records = records.len() - crossing.len();
    let expected_scanned_rows =
        BOUNDARY_CREATE_ROWS + crossing.iter().map(|(start, end)| end - start).sum::<i64>();

    let params = config_seed_index_params(index_rows_per_zone);
    let reports = SEED_REUSE_REPORTS
        .scope(std::sync::Mutex::new(Vec::new()), async {
            dataset
                .create_index(&["val"], IndexType::ZoneMap, None, &params, false)
                .await
                .unwrap();
            SEED_REUSE_REPORTS.with(|reports| reports.lock().unwrap().clone())
        })
        .await;
    assert_eq!(reports.len(), 1, "{context}: {reports:?}");
    let report = &reports[0];
    let mut scanned = report.scanned_fragments.clone();
    scanned.sort_unstable();
    if crossing.is_empty() {
        assert_eq!(
            report.reused_fragments,
            vec![appended],
            "{context}: {report:?}"
        );
        assert!(report.partial_fragments.is_empty(), "{context}: {report:?}");
        assert_eq!(scanned, vec![creating], "{context}: {report:?}");
    } else {
        // A partly reused fragment is listed as both partial and scanned.
        assert!(report.reused_fragments.is_empty(), "{context}: {report:?}");
        assert_eq!(
            report.partial_fragments,
            vec![appended],
            "{context}: {report:?}"
        );
        let mut all_fragments = vec![creating, appended];
        all_fragments.sort_unstable();
        assert_eq!(scanned, all_fragments, "{context}: {report:?}");
    }
    assert_eq!(
        report.reused_records, expected_reused_records,
        "{context}: {report:?}"
    );
    assert_eq!(
        report.scanned_rows, expected_scanned_rows as u64,
        "{context}: {report:?}"
    );

    // With a single appended fragment the fragment boundary is the one
    // between the creating fragment and the appended fragment.
    assert_boundary_predicates(
        &dataset,
        BOUNDARY_CREATE_ROWS + rows_per_fragment,
        first_appended + index.min(rows_per_fragment),
        first_appended,
        &context,
    )
    .await;

    assert_boundary_zone_statistics(
        &dataset,
        index_rows_per_zone,
        &[
            (creating, 0..BOUNDARY_CREATE_ROWS),
            (appended, first_appended..first_appended + rows_per_fragment),
        ],
        &context,
    )
    .await;
}
