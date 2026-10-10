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
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Int32Array, RecordBatch, RecordBatchIterator,
    StringArray, StringViewArray,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use bytes::Bytes;
use futures::{TryStreamExt, stream};
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::utils::tempfile::TempStrDir;
use lance_datafusion::utils::reader_to_stream;
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_index::IndexType;
use lance_index::scalar::seed::{IndexSeedWriter, SEED_META_KEY_PREFIX};
use lance_index::scalar::zonemap::ZoneMapSeedWriter;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_io::ReadBatchParams;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_table::format::{DataFile, Fragment};

use crate::Dataset;
use crate::dataset::schema_evolution::NewColumnTransform;
use crate::dataset::transaction::{DataReplacementGroup, Operation};
use crate::dataset::write::merge_insert::{WhenMatched, WhenNotMatched};
use crate::dataset::{
    DATA_DIR, MergeInsertBuilder, MergeInsertWriteMode, UpdateBuilder, WriteDestination,
    WriteParams,
};
use crate::index::DatasetIndexExt;

const ROWS_PER_ZONE: u64 = 4;
const ROWS_PER_FILE: usize = 10;

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
        Operation::DataReplacement { replacements },
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
