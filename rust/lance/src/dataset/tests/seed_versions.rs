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

/// Harvesting a seed opens the serving data file with the size the manifest
/// records, so an index build over files with known sizes issues no size
/// probe (`head`) on them, only the footer/seed reads; a fragment whose
/// manifest entry lacks the size (older manifests) is still harvested
/// through the scheduler's probe, exactly once.
#[tokio::test]
async fn test_seed_harvest_uses_the_manifest_file_size() {
    use lance_io::utils::CachedFileSize;

    use crate::index::append::SeedHarvester;
    use crate::index::scalar::zonemap::SEED_REUSE_REPORTS;

    let dir = TempStrDir::default();
    let mut dataset = seeded_dataset(dir.as_str(), &["val"]).await;
    append(&mut dataset, TRAINING_ROWS).await;
    let crafted = rows(TRAINING_ROWS + ROWS_PER_FILE, ROWS_PER_FILE);
    let seeds = vec![("val", seed_bytes("val", crafted.column(0)))];
    let mut dataset = commit_crafted_fragment(dataset, crafted, &seeds).await;
    let fragments: Vec<Fragment> = dataset.fragments().iter().skip(1).cloned().collect();
    assert_eq!(fragments.len(), 2);
    for fragment in &fragments {
        assert!(
            fragment.files[0].file_size_bytes.get().is_some(),
            "committed files carry their size in the manifest: {fragment:?}"
        );
    }
    let requests_on =
        |io: &lance_io::utils::tracking_store::IoStats, fragment: &Fragment, method: &str| {
            let path = fragment.files[0].path.clone();
            io.requests
                .iter()
                .filter(|r| r.method == method && r.path.to_string().ends_with(&path))
                .count()
        };

    // The build: both seeds reused, no size probe on either data file.
    dataset.object_store.io_stats_incremental();
    let reports = SEED_REUSE_REPORTS
        .scope(std::sync::Mutex::new(Vec::new()), async {
            dataset
                .optimize_indices(&OptimizeOptions::default())
                .await
                .unwrap();
            SEED_REUSE_REPORTS.with(|reports| reports.lock().unwrap().clone())
        })
        .await;
    let io = dataset.object_store.io_stats_incremental();
    assert_eq!(reports.len(), 1, "{reports:?}");
    assert_eq!(
        reports[0].reused_fragments,
        fragments.iter().map(|f| f.id).collect::<Vec<_>>()
    );
    assert_eq!(reports[0].scanned_rows, 0);
    for fragment in &fragments {
        assert_eq!(
            requests_on(&io, fragment, "head"),
            0,
            "a known size needs no probe: {:?}",
            io.requests
        );
        assert!(
            requests_on(&io, fragment, "get_range") >= 1,
            "the seed is read from the file: {:?}",
            io.requests
        );
    }

    // A manifest entry without a size: the scheduler probes it once and the
    // seed is harvested all the same.
    let mut without_size = fragments[1].clone();
    without_size.files[0].file_size_bytes = CachedFileSize::unknown();
    let harvester = SeedHarvester::for_column(&dataset, "val").unwrap();
    dataset.object_store.io_stats_incremental();
    let seed = harvester
        .harvest(&without_size)
        .await
        .unwrap()
        .expect("the seed is harvested through the size probe");
    let io = dataset.object_store.io_stats_incremental();
    assert_eq!(seed.physical_rows, ROWS_PER_FILE as u64);
    assert_eq!(
        requests_on(&io, &without_size, "head"),
        1,
        "an unknown size is probed once: {:?}",
        io.requests
    );
    assert!(requests_on(&io, &without_size, "get_range") >= 1);
    dataset.object_store.io_stats_incremental();
    assert!(harvester.harvest(&fragments[1]).await.unwrap().is_some());
    let io = dataset.object_store.io_stats_incremental();
    assert_eq!(requests_on(&io, &fragments[1], "head"), 0);

    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    for filter in ["val > 20", "val = 70", "val < 100 AND val > 30"] {
        assert_eq!(
            count(&reopened, filter, true).await,
            count(&reopened, filter, false).await,
            "{filter}"
        );
    }
}

/// One version 1 segment built from fragments whose seeds have different
/// versions and contents. Every record of the output meets the version 1
/// contract (ordinary extrema, exact NaN sign split), and a legacy record
/// contributes only when its stored `nan_count` is zero:
/// - a NaN-free version 0 seed is reused with a zero sign split;
/// - the NaN-bearing record of a version 0 seed (its `max` is the NaN
///   marker, its split unknown) is scanned, which recovers the ordinary
///   maximum and the exact split, while the seed's other record is reused;
/// - a version 1 seed with NaNs of both signs is reused as is;
/// - a version 0 seed without a `nan_count` column is undecodable: the
///   fragment is scanned, and no zero is inferred.
/// The target zone equals the record size, so version eligibility alone
/// decides what is scanned. The index keeps no null bitmap (version 0 seeds
/// carry no positions; with a bitmap they would be ineligible for that
/// reason instead).
#[tokio::test]
async fn test_mixed_version_seeds_build_one_complete_version_1_segment() {
    use arrow_array::{Float64Array, Int32Array, UInt32Array, UInt64Array};
    use lance_index::scalar::IndexStore;
    use lance_index::scalar::lance_format::LanceIndexStore;
    use lance_index::scalar::zonemap::ZONEMAP_INDEX_FILENAME;

    use crate::dataset::index::LanceIndexStoreExt;
    use crate::index::scalar::zonemap::SEED_REUSE_REPORTS;

    const RECORD: usize = ROWS_PER_ZONE as usize;
    let float_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("x", DataType::Float64, true),
    ]));
    let batch = |start: i32, values: &[Option<f64>]| {
        RecordBatch::try_new(
            float_schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(
                    start..start + values.len() as i32,
                )),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .unwrap()
    };
    /// A version 0 writer's seed: the original five columns, no version
    /// key, the `+NaN` max marker on records with NaNs, no positions.
    /// `with_nan_count: false` leaves the `nan_count` column out entirely.
    fn v0_seed(records: &[(Option<f64>, Option<f64>, u32, u32)], with_nan_count: bool) -> Bytes {
        let mut fields = vec![
            ArrowField::new("min", DataType::Float64, true),
            ArrowField::new("max", DataType::Float64, true),
            ArrowField::new("null_count", DataType::UInt32, false),
        ];
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(
                records.iter().map(|r| r.0).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                records.iter().map(|r| r.1).collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from_iter_values(records.iter().map(|r| r.2))),
        ];
        if with_nan_count {
            fields.push(ArrowField::new("nan_count", DataType::UInt32, false));
            columns.push(Arc::new(UInt32Array::from_iter_values(
                records.iter().map(|r| r.3),
            )));
        }
        fields.push(ArrowField::new("zone_length", DataType::UInt64, false));
        columns.push(Arc::new(UInt64Array::from_iter_values(
            records.iter().map(|_| RECORD as u64),
        )));
        let batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).unwrap();
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = IpcWriter::try_new(&mut buf, batch.schema_ref()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        Bytes::from(buf.into_inner())
    }
    let plain =
        |start: f64| -> Vec<Option<f64>> { (0..RECORD).map(|i| Some(start + i as f64)).collect() };

    // Fragment 0 (training): plain values, indexed without seeds in play.
    let dir = TempStrDir::default();
    let mut training = plain(50.0);
    training.extend(plain(60.0));
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch(0, &training))], float_schema.clone()),
        dir.as_str(),
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["x"],
            IndexType::ZoneMap,
            None,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap).with_params(
                &serde_json::json!({"rows_per_zone": ROWS_PER_ZONE, "use_seeds": true, "store_null_bitmap": false}),
            ),
            false,
        )
        .await
        .unwrap();

    // Fragment 1: NaN-free version 0 seed (one null in its second record).
    let mut f1 = plain(100.0);
    f1.extend(plain(110.0));
    f1[RECORD + 3] = None;
    let f1_seed = v0_seed(
        &[
            (Some(100.0), Some(107.0), 0, 0),
            (Some(110.0), Some(117.0), 1, 0),
        ],
        true,
    );
    let dataset = commit_crafted_fragment(dataset, batch(100, &f1), &[("x", f1_seed)]).await;
    let f1_id = dataset.fragments().last().unwrap().id;

    // Fragment 2: version 0 seed whose second record holds NaNs of both
    // signs behind the legacy marker; its ordinary maximum (5.0) is hidden.
    let mut f2 = plain(1.0);
    f2.extend([
        Some(1.0),
        Some(5.0),
        Some(f64::NAN),
        Some(-f64::NAN),
        Some(2.0),
        Some(3.0),
        Some(4.0),
        Some(1.5),
    ]);
    let f2_seed = v0_seed(
        &[
            (Some(1.0), Some(8.0), 0, 0),
            (Some(1.0), Some(f64::NAN), 0, 2),
        ],
        true,
    );
    let dataset = commit_crafted_fragment(dataset, batch(200, &f2), &[("x", f2_seed)]).await;
    let f2_id = dataset.fragments().last().unwrap().id;

    // Fragment 3: version 1 seed written by this build, NaNs of both signs.
    let mut f3: Vec<Option<f64>> = vec![
        Some(10.0),
        Some(20.0),
        Some(f64::NAN),
        Some(-f64::NAN),
        Some(-f64::NAN),
        Some(15.0),
        Some(12.0),
        Some(11.0),
    ];
    f3.extend(plain(30.0));
    let f3_values: ArrayRef = Arc::new(Float64Array::from(f3.clone()));
    let f3_seed = seed_bytes("x", &f3_values);
    let dataset = commit_crafted_fragment(dataset, batch(300, &f3), &[("x", f3_seed)]).await;
    let f3_id = dataset.fragments().last().unwrap().id;

    // Fragment 4: version 0 seed without a `nan_count` column.
    let mut f4 = plain(200.0);
    f4.extend(plain(210.0));
    let f4_seed = v0_seed(
        &[
            (Some(200.0), Some(207.0), 0, 0),
            (Some(210.0), Some(217.0), 0, 0),
        ],
        false,
    );
    let mut dataset = commit_crafted_fragment(dataset, batch(400, &f4), &[("x", f4_seed)]).await;
    let f4_id = dataset.fragments().last().unwrap().id;

    // One build over all four new fragments.
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
    let report = &reports[0];
    assert_eq!(report.reused_fragments, vec![f1_id, f3_id], "{report:?}");
    assert_eq!(report.partial_fragments, vec![f2_id], "{report:?}");
    assert_eq!(report.scanned_fragments, vec![f2_id, f4_id], "{report:?}");
    assert_eq!(
        report.scanned_rows,
        (RECORD + 2 * RECORD) as u64,
        "the NaN-bearing record of fragment 2 and all of fragment 4"
    );
    assert_eq!(report.reused_records, 2 + 1 + 2, "{report:?}");
    assert!(report.masked);

    // The produced segment: version 1, covering everything, every record
    // complete.
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    let indices = reopened.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1, "{indices:?}");
    let segment = &indices[0];
    assert_eq!(segment.index_version, 1);
    assert_eq!(
        segment.fragment_bitmap.as_ref().unwrap(),
        reopened.fragment_bitmap.as_ref()
    );
    let store = LanceIndexStore::from_dataset_for_existing(&reopened, segment)
        .await
        .unwrap();
    let file = store.open_index_file(ZONEMAP_INDEX_FILENAME).await.unwrap();
    let schema = file.schema().clone();
    let negative_field = schema
        .field("negative_nan_count")
        .expect("version 1 files carry the sign split column");
    assert!(!negative_field.nullable);
    assert_eq!(negative_field.data_type(), DataType::UInt32);
    let zones = file.read_range(0..file.num_rows(), None).await.unwrap();
    let col = |name: &str| zones.column_by_name(name).unwrap().clone();
    let mins = col("min");
    let maxs = col("max");
    let mins = mins.as_any().downcast_ref::<Float64Array>().unwrap();
    let maxs = maxs.as_any().downcast_ref::<Float64Array>().unwrap();
    let u32s = |name: &str| {
        col(name)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .values()
            .to_vec()
    };
    let u64s = |name: &str| {
        col(name)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec()
    };
    let (nulls, nans, negatives) = (
        u32s("null_count"),
        u32s("nan_count"),
        u32s("negative_nan_count"),
    );
    let (fragments, starts, lengths) =
        (u64s("fragment_id"), u64s("zone_start"), u64s("zone_length"));
    // Five fragments of two zones each; the layout is the target's.
    assert_eq!(zones.num_rows(), 10);
    for i in 0..zones.num_rows() {
        assert_eq!(lengths[i], RECORD as u64);
        assert_eq!(starts[i] % ROWS_PER_ZONE, 0);
        assert!(
            negatives[i] <= nans[i],
            "zone {i}: split above its NaN count"
        );
        assert!(u64::from(nulls[i]) + u64::from(nans[i]) <= lengths[i]);
        assert!(
            !mins.is_valid(i) || !mins.value(i).is_nan(),
            "zone {i}: NaN min in a version 1 file"
        );
        assert!(
            !maxs.is_valid(i) || !maxs.value(i).is_nan(),
            "zone {i}: NaN max (the legacy marker) in a version 1 file"
        );
    }
    let zone = |fragment: u64, start: u64| {
        (0..zones.num_rows())
            .find(|&i| fragments[i] == fragment && starts[i] == start)
            .unwrap_or_else(|| panic!("no zone for fragment {fragment} at {start}"))
    };
    // Fragment 1 (reused version 0): bounds and null count from the seed, zero split.
    let z = zone(f1_id, 0);
    assert_eq!(
        (
            mins.value(z),
            maxs.value(z),
            nulls[z],
            nans[z],
            negatives[z]
        ),
        (100.0, 107.0, 0, 0, 0)
    );
    let z = zone(f1_id, 8);
    assert_eq!(
        (
            mins.value(z),
            maxs.value(z),
            nulls[z],
            nans[z],
            negatives[z]
        ),
        (110.0, 117.0, 1, 0, 0)
    );
    // Fragment 2: first record reused; second scanned, ordinary maximum
    // recovered and the split exact.
    let z = zone(f2_id, 0);
    assert_eq!(
        (mins.value(z), maxs.value(z), nans[z], negatives[z]),
        (1.0, 8.0, 0, 0)
    );
    let z = zone(f2_id, 8);
    assert_eq!(
        (mins.value(z), maxs.value(z), nans[z], negatives[z]),
        (1.0, 5.0, 2, 1)
    );
    // Fragment 3 (reused version 1): exact split and ordinary extrema kept.
    let z = zone(f3_id, 0);
    assert_eq!(
        (mins.value(z), maxs.value(z), nans[z], negatives[z]),
        (10.0, 20.0, 3, 2)
    );
    let z = zone(f3_id, 8);
    assert_eq!(
        (mins.value(z), maxs.value(z), nans[z], negatives[z]),
        (30.0, 37.0, 0, 0)
    );
    // Fragment 4 (scanned, undecodable seed): statistics from the data.
    let z = zone(f4_id, 0);
    assert_eq!(
        (mins.value(z), maxs.value(z), nans[z], negatives[z]),
        (200.0, 207.0, 0, 0)
    );
    let z = zone(f4_id, 8);
    assert_eq!(
        (mins.value(z), maxs.value(z), nans[z], negatives[z]),
        (210.0, 217.0, 0, 0)
    );

    for filter in [
        "x > 4.5",
        "x < 1.5",
        "x = 5.0",
        "x >= 20.0",
        "x < 11.0 AND x > 9.0",
        "x IS NULL",
        "x IS NOT NULL",
    ] {
        assert_eq!(
            count(&reopened, filter, true).await,
            count(&reopened, filter, false).await,
            "{filter}"
        );
    }
}

/// An index update that harvests a future-version seed treats it as
/// unavailable and scans that fragment's column, while a valid version 1
/// seed beside it is reused; the published segment is version 1 and its
/// answers equal a scan. The decoder itself still refuses the unknown
/// version (see `test_parse_seed_rejects_corrupt_and_unknown`).
#[tokio::test]
async fn test_index_update_scans_unsupported_seed_version() {
    use crate::index::scalar::zonemap::SEED_REUSE_REPORTS;

    let dir = TempStrDir::default();
    let dataset = seeded_dataset(dir.as_str(), &["val"]).await;
    let crafted = rows(TRAINING_ROWS, ROWS_PER_FILE);
    let seeds = vec![(
        "val",
        relabel_version(&seed_bytes("val", crafted.column(0)), "2"),
    )];
    let mut dataset = commit_crafted_fragment(dataset, crafted, &seeds).await;
    let unreadable = dataset.fragments().last().unwrap().id;
    append(&mut dataset, TRAINING_ROWS + ROWS_PER_FILE).await;
    let seeded = dataset.fragments().last().unwrap().id;
    assert_ne!(unreadable, seeded);

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
    assert_eq!(reports[0].scanned_fragments, vec![unreadable]);
    assert_eq!(reports[0].reused_fragments, vec![seeded]);

    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    let indices = reopened.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1, "{indices:?}");
    assert_eq!(indices[0].index_version, 1);
    assert_eq!(
        indices[0].fragment_bitmap.as_ref().unwrap(),
        reopened.fragment_bitmap.as_ref(),
        "every fragment is covered after the update"
    );
    for filter in ["val > 20", "val = 70", "val < 100 AND val > 30"] {
        assert_eq!(
            count(&reopened, filter, true).await,
            count(&reopened, filter, false).await,
            "{filter}"
        );
    }
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
    // Both tables have written once (the process-wide null-position policy
    // read is behind them) before the compared appends.
    append(&mut beside, TRAINING_ROWS + ROWS_PER_FILE).await;

    plain.object_store.io_stats_incremental();
    beside.object_store.io_stats_incremental();
    append(&mut plain, TRAINING_ROWS + ROWS_PER_FILE).await;
    append(&mut beside, TRAINING_ROWS + 2 * ROWS_PER_FILE).await;
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
