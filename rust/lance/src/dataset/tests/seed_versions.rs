// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Seeds of a format version this build does not support are recognized but
//! never interpreted: binary copy still copies the data and omits only that
//! column's seed, an index update refuses them instead of publishing partial
//! coverage, and writes beside them neither read them nor slow down.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, RecordBatchIterator};
use arrow_ipc::reader::FileReader as IpcReader;
use arrow_ipc::writer::FileWriter as IpcWriter;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use bytes::Bytes;
use lance_core::Error;
use lance_core::utils::tempfile::TempStrDir;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::versions as file_versions;
use lance_file::writer::FileWriterOptions;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::seed::{IndexSeedWriter, SEED_META_KEY_PREFIX};
use lance_index::scalar::zonemap::{
    CarriedSeed, MergeableSeed, ZoneMapSeedWriter, parse_seed_for_binary_copy,
};
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_table::format::{DataFile, Fragment};

use crate::Dataset;
use crate::dataset::DATA_DIR;
use crate::dataset::optimize::{CompactionMode, CompactionOptions, compact_files};
use crate::dataset::transaction::{Operation, TransactionBuilder};
use crate::dataset::write::CommitBuilder;
use crate::index::DatasetIndexExt;

const ROWS_PER_ZONE: u64 = 8;
const ROWS_PER_FILE: i64 = 64;
/// Rows written before any index exists.
const TRAINING_ROWS: i64 = 16;

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("val", DataType::Int64, true),
        ArrowField::new("other", DataType::Int64, true),
    ]))
}

/// `val` is null every seventh row; `other` is never null.
fn rows(start: i64, count: i64) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from_iter(
                (start..start + count).map(|i| (i % 7 != 3).then_some(i)),
            )),
            Arc::new(Int64Array::from_iter_values(
                (start..start + count).map(|i| i * 100),
            )),
        ],
    )
    .unwrap()
}

/// A table whose ZoneMap indices on `columns` request seeds; the index
/// commits activate the seed feature flags, so later writes carry version 1
/// seeds.
async fn seeded_dataset(uri: &str, columns: &[&str]) -> Dataset {
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(rows(0, TRAINING_ROWS))], schema()),
        uri,
        None,
    )
    .await
    .unwrap();
    for column in columns {
        dataset
            .create_index(
                &[column],
                IndexType::ZoneMap,
                None,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap).with_params(
                    &serde_json::json!({"rows_per_zone": ROWS_PER_ZONE, "use_seeds": true}),
                ),
                false,
            )
            .await
            .unwrap();
    }
    dataset
}

async fn append(dataset: &mut Dataset, start: i64) {
    dataset
        .append(
            RecordBatchIterator::new([Ok(rows(start, ROWS_PER_FILE))], schema()),
            None,
        )
        .await
        .unwrap();
}

/// The version 1 seed this build writes for `values`.
fn seed_bytes(column: &str, values: &ArrayRef) -> Bytes {
    let mut writer =
        ZoneMapSeedWriter::new(column, ROWS_PER_ZONE, values.data_type().clone()).unwrap();
    writer.observe_batch(values).unwrap();
    writer.finish().unwrap().unwrap()
}

/// `bytes` with its seed `version` metadata replaced: a synthetic future
/// format whose columns happen to match version 1.
fn relabel_version(bytes: &Bytes, version: &str) -> Bytes {
    let mut reader = IpcReader::try_new(Cursor::new(bytes.as_ref()), None).unwrap();
    let batch = reader.next().unwrap().unwrap();
    let mut schema = batch.schema().as_ref().clone();
    schema
        .metadata
        .insert("version".to_string(), version.to_string());
    let batch = RecordBatch::try_new(Arc::new(schema), batch.columns().to_vec()).unwrap();
    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = IpcWriter::try_new(&mut buf, batch.schema_ref()).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }
    Bytes::from(buf.into_inner())
}

/// Commit one fragment whose data file carries exactly `seeds`, the way an
/// external writer would publish it.
async fn commit_crafted_fragment(
    dataset: Dataset,
    batch: RecordBatch,
    seeds: &[(&str, Bytes)],
) -> Dataset {
    let version = dataset.manifest.data_storage_format.lance_file_format();
    let schema = dataset.schema().clone();
    let filename = format!("crafted-{}.lance", dataset.manifest.version);
    let path = dataset.base.clone().join(DATA_DIR).join(filename.as_str());
    let object_writer = dataset.object_store.create(&path).await.unwrap();
    let mut writer = file_versions::create_writer(
        version,
        object_writer,
        schema.clone(),
        FileWriterOptions::default(),
    )
    .unwrap();
    writer.write_batch(&batch).await.unwrap();
    for (column, bytes) in seeds {
        let buf_index = writer.add_global_buffer(bytes.clone()).await.unwrap();
        writer.add_schema_metadata(
            format!("{SEED_META_KEY_PREFIX}{column}"),
            format!("{buf_index}:{ROWS_PER_ZONE}"),
        );
    }
    writer.finish().await.unwrap();

    let (field_ids, column_indices) = file_versions::data_file_columns(version, &schema);
    let mut data_file = DataFile::new_unstarted(filename, version);
    data_file.fields = field_ids.into();
    data_file.column_indices = column_indices.into();
    let mut fragment = Fragment::new(0);
    fragment.files.push(data_file);
    fragment.physical_rows = Some(batch.num_rows());
    let transaction = TransactionBuilder::new(
        dataset.manifest.version,
        Operation::Append {
            fragments: vec![fragment],
        },
    )
    .build();
    CommitBuilder::new(Arc::new(dataset))
        .execute(transaction)
        .await
        .unwrap()
}

/// The seeds stored in the data file at `path`, by column, and the number of
/// global buffers beyond the schema descriptor.
async fn file_seeds(dataset: &Dataset, path: &str) -> (usize, HashMap<String, Bytes>) {
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
    let metadata = reader.metadata().clone();
    let extra_buffers = metadata.file_buffers.len().saturating_sub(1);
    let mut seeds = HashMap::new();
    for (key, value) in metadata.file_schema.metadata.iter() {
        if let Some(column) = key.strip_prefix(SEED_META_KEY_PREFIX) {
            let buf_index: u32 = value.split(':').next().unwrap().parse().unwrap();
            seeds.insert(
                column.to_string(),
                reader.read_global_buffer(buf_index).await.unwrap(),
            );
        }
    }
    (extra_buffers, seeds)
}

fn supported(bytes: &Bytes) -> MergeableSeed {
    match parse_seed_for_binary_copy(bytes, ROWS_PER_ZONE).unwrap() {
        CarriedSeed::Supported(seed) => seed,
        CarriedSeed::UnsupportedVersion(version) => {
            panic!("expected a version 1 seed, found version {version}")
        }
    }
}

async fn count(dataset: &Dataset, filter: &str, use_scalar_index: bool) -> usize {
    let mut scanner = dataset.scan();
    scanner
        .filter(filter)
        .unwrap()
        .use_scalar_index(use_scalar_index);
    scanner.try_into_batch().await.unwrap().num_rows()
}

fn binary_copy_options() -> CompactionOptions {
    CompactionOptions {
        target_rows_per_fragment: 1_000_000,
        compaction_mode: Some(CompactionMode::ForceBinaryCopy),
        ..Default::default()
    }
}

#[derive(Clone, Copy, Debug)]
enum ValSeed {
    FutureVersion,
    Missing,
}

/// Binary copy decides each output column's seed on its own: `val`, which
/// one input seeds with a future version or not at all, is omitted (no
/// buffer, no locator), while `other`, version 1 in every input, is carried.
/// `ForceBinaryCopy` fails the run if the data were re-encoded instead.
#[rstest::rstest]
#[case::future_version(ValSeed::FutureVersion)]
#[case::missing(ValSeed::Missing)]
#[tokio::test]
async fn test_binary_copy_omits_unsupported_seed_per_column(#[case] val_seed: ValSeed) {
    let dir = TempStrDir::default();
    let mut dataset = seeded_dataset(dir.as_str(), &["val", "other"]).await;
    append(&mut dataset, TRAINING_ROWS).await;
    append(&mut dataset, TRAINING_ROWS + ROWS_PER_FILE).await;
    let crafted = rows(TRAINING_ROWS + 2 * ROWS_PER_FILE, ROWS_PER_FILE);
    let mut seeds = vec![("other", seed_bytes("other", crafted.column(1)))];
    if let ValSeed::FutureVersion = val_seed {
        seeds.push((
            "val",
            relabel_version(&seed_bytes("val", crafted.column(0)), "2"),
        ));
    }
    let mut dataset = commit_crafted_fragment(dataset, crafted, &seeds).await;
    let rows_before = dataset.count_rows(None).await.unwrap();
    let max_before = dataset.fragments().iter().map(|f| f.id).max().unwrap();

    let metrics = compact_files(&mut dataset, binary_copy_options(), None)
        .await
        .unwrap();
    assert_eq!(metrics.fragments_added, 1, "{val_seed:?}");
    assert_eq!(dataset.count_rows(None).await.unwrap(), rows_before);

    let output: Vec<Fragment> = dataset
        .fragments()
        .iter()
        .filter(|fragment| fragment.id > max_before)
        .cloned()
        .collect();
    assert_eq!(output.len(), 1);
    let (extra_buffers, seeds) = file_seeds(&dataset, &output[0].files[0].path).await;
    assert_eq!(
        extra_buffers, 1,
        "only `other` carries a seed ({val_seed:?})"
    );
    assert!(!seeds.contains_key("val"), "{val_seed:?}");
    let other = supported(&seeds["other"]);
    assert!(other.is_version_1());
    assert_eq!(
        other.total_rows(),
        output[0].physical_rows.unwrap() as u64,
        "the carried seed covers the whole output"
    );

    // `other` updates from the carried seed and `val` by scan; both agree
    // with the unindexed scan.
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    for filter in ["val > 100", "val IS NULL", "other > 5000", "other < 1000"] {
        assert_eq!(
            count(&dataset, filter, true).await,
            count(&dataset, filter, false).await,
            "{filter} ({val_seed:?})"
        );
    }
}

/// Version 1 inputs with nulls merge into one output seed whose null offsets
/// are shifted into the output's row order.
#[tokio::test]
async fn test_binary_copy_carries_shifted_null_offsets() {
    let dir = TempStrDir::default();
    let mut dataset = seeded_dataset(dir.as_str(), &["val"]).await;
    append(&mut dataset, TRAINING_ROWS).await;
    append(&mut dataset, TRAINING_ROWS + ROWS_PER_FILE).await;
    let max_before = dataset.fragments().iter().map(|f| f.id).max().unwrap();

    compact_files(&mut dataset, binary_copy_options(), None)
        .await
        .unwrap();
    let output: Vec<Fragment> = dataset
        .fragments()
        .iter()
        .filter(|fragment| fragment.id > max_before)
        .cloned()
        .collect();
    assert_eq!(output.len(), 1);
    let (_, seeds) = file_seeds(&dataset, &output[0].files[0].path).await;
    let seed = supported(&seeds["val"]);

    let mut scanner = dataset.scan();
    scanner
        .with_fragments(output.clone())
        .project(&["val"])
        .unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    let values = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let expected: Vec<u32> = (0..values.len())
        .filter(|&i| values.is_null(i))
        .map(|i| i as u32)
        .collect();
    assert!(
        expected.iter().any(|&i| i >= ROWS_PER_FILE as u32),
        "nulls past the first input exercise the offset shift"
    );
    let bitmap = seed
        .null_bitmap()
        .expect("version 1 inputs record null positions");
    assert_eq!(bitmap.iter().collect::<Vec<_>>(), expected);
    assert_eq!(seed.total_rows(), values.len() as u64);
}

/// A version 0 seed whose statistics cannot satisfy the version 1 contract
/// (here a null count above the zone length) is left unused: the update
/// scans the fragment, publishes a version 1 segment covering it, and the
/// indexed answers equal a scan.
#[tokio::test]
async fn test_index_update_scans_invalid_version_0_seed() {
    use arrow_array::{UInt32Array, UInt64Array};

    let dir = TempStrDir::default();
    let dataset = seeded_dataset(dir.as_str(), &["val"]).await;
    let crafted = rows(TRAINING_ROWS, ROWS_PER_FILE);
    // One version 0 record over the whole file with more nulls than rows.
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("min", DataType::Int64, true),
        ArrowField::new("max", DataType::Int64, true),
        ArrowField::new("null_count", DataType::UInt32, false),
        ArrowField::new("nan_count", DataType::UInt32, false),
        ArrowField::new("zone_length", DataType::UInt64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![Some(TRAINING_ROWS)])),
            Arc::new(Int64Array::from(vec![Some(
                TRAINING_ROWS + ROWS_PER_FILE - 1,
            )])),
            Arc::new(UInt32Array::from(vec![ROWS_PER_FILE as u32 + 1])),
            Arc::new(UInt32Array::from(vec![0u32])),
            Arc::new(UInt64Array::from(vec![ROWS_PER_FILE as u64])),
        ],
    )
    .unwrap();
    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = IpcWriter::try_new(&mut buf, batch.schema_ref()).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }
    let seeds = vec![("val", Bytes::from(buf.into_inner()))];
    let mut dataset = commit_crafted_fragment(dataset, crafted, &seeds).await;

    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    let indices = reopened.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1, "{indices:?}");
    assert_eq!(indices[0].index_version, 1);
    assert_eq!(
        indices[0].fragment_bitmap.as_ref().unwrap(),
        reopened.fragment_bitmap.as_ref(),
        "the scanned fragment is covered"
    );
    for filter in ["val > 20", "val = 70", "val < 100 AND val > 30"] {
        assert_eq!(
            count(&reopened, filter, true).await,
            count(&reopened, filter, false).await,
            "{filter}"
        );
    }
}

/// An index update that harvests a future-version seed fails before
/// publishing anything; it never covers the fragment with statistics it
/// could not read.
#[tokio::test]
async fn test_index_update_refuses_unsupported_seed_version() {
    let dir = TempStrDir::default();
    let dataset = seeded_dataset(dir.as_str(), &["val"]).await;
    let crafted = rows(TRAINING_ROWS, ROWS_PER_FILE);
    let seeds = vec![(
        "val",
        relabel_version(&seed_bytes("val", crafted.column(0)), "2"),
    )];
    let mut dataset = commit_crafted_fragment(dataset, crafted, &seeds).await;
    let version_before = dataset.manifest.version;
    let before = dataset.load_indices().await.unwrap();

    let error = dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");

    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(reopened.manifest.version, version_before);
    let after = reopened.load_indices().await.unwrap();
    assert_eq!(before.len(), after.len());
    for (before, after) in before.iter().zip(after.iter()) {
        assert_eq!(before.uuid, after.uuid);
        assert_eq!(before.fragment_bitmap, after.fragment_bitmap);
    }
    // The uncovered fragment is still scanned by indexed queries.
    assert_eq!(
        count(&reopened, "val > 20", true).await,
        count(&reopened, "val > 20", false).await
    );
}

/// A file carrying a future-version seed is never read by a later write:
/// the append issues the same requests as on a table without it and writes
/// a version 1 seed.
#[tokio::test]
async fn test_append_beside_unsupported_seed_version_writes_version_1() {
    let plain_dir = TempStrDir::default();
    let mut plain = seeded_dataset(plain_dir.as_str(), &["val"]).await;
    append(&mut plain, TRAINING_ROWS).await;

    let dir = TempStrDir::default();
    let dataset = seeded_dataset(dir.as_str(), &["val"]).await;
    let crafted = rows(TRAINING_ROWS, ROWS_PER_FILE);
    let seeds = vec![(
        "val",
        relabel_version(&seed_bytes("val", crafted.column(0)), "2"),
    )];
    let mut beside = commit_crafted_fragment(dataset, crafted, &seeds).await;

    plain.object_store.io_stats_incremental();
    beside.object_store.io_stats_incremental();
    append(&mut plain, TRAINING_ROWS + ROWS_PER_FILE).await;
    append(&mut beside, TRAINING_ROWS + ROWS_PER_FILE).await;
    let plain_io = plain.object_store.io_stats_incremental();
    let beside_io = beside.object_store.io_stats_incremental();
    assert_eq!(
        (beside_io.read_iops, beside_io.write_iops),
        (plain_io.read_iops, plain_io.write_iops),
        "(read_iops, write_iops) of an append must not depend on another file's seed version"
    );

    let appended = beside.fragments().last().unwrap();
    let (extra_buffers, seeds) = file_seeds(&beside, &appended.files[0].path).await;
    assert_eq!(extra_buffers, 1);
    assert!(supported(&seeds["val"]).is_version_1());
    assert_eq!(
        count(&beside, "val > 20", true).await,
        count(&beside, "val > 20", false).await
    );
}
