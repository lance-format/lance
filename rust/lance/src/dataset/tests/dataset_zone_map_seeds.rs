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

use arrow_array::{Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use bytes::Bytes;
use futures::stream;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::utils::tempfile::TempStrDir;
use lance_datafusion::utils::reader_to_stream;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_index::IndexType;
use lance_index::scalar::seed::{IndexSeedWriter, SEED_META_KEY_PREFIX};
use lance_index::scalar::zonemap::ZoneMapSeedWriter;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_table::format::{DataFile, Fragment};

use crate::Dataset;
use crate::dataset::schema_evolution::NewColumnTransform;
use crate::dataset::transaction::{DataReplacementGroup, Operation};
use crate::dataset::write::merge_insert::{WhenMatched, WhenNotMatched};
use crate::dataset::{
    DATA_DIR, MergeInsertBuilder, MergeInsertWriteMode, WriteDestination, WriteParams,
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

/// The seed stored for `column` in the data file at `path`, if any.
async fn seed_in_file(dataset: &Dataset, path: &str, column: &str) -> Option<Bytes> {
    let scheduler = ScanScheduler::new(
        dataset.object_store.clone(),
        SchedulerConfig::max_bandwidth(&dataset.object_store),
    );
    let path = dataset.base.clone().join(DATA_DIR).join(path);
    let file_scheduler = scheduler
        .open_file(&path, &CachedFileSize::unknown())
        .await
        .unwrap();
    let reader = FileReader::try_open(
        file_scheduler,
        None,
        Default::default(),
        &dataset.metadata_cache.file_metadata_cache(&path),
        FileReaderOptions::default(),
    )
    .await
    .unwrap();
    let value = reader
        .metadata()
        .file_schema
        .metadata
        .get(&format!("{SEED_META_KEY_PREFIX}{column}"))?
        .clone();
    let buf_index: u32 = value.split(':').next().unwrap().parse().unwrap();
    Some(reader.read_global_buffer(buf_index).await.unwrap())
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
        replacements.push(
            fragment
                .write_columns(stream::iter([Ok(batch)]), &column_schema)
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
