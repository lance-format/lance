// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! ZoneMap index file versions at the dataset level.
//!
//! A version 0 segment is read under its own contract and, once maintenance
//! selects it, rebuilt from source data into a version 1 segment rather than
//! updated in place. A version this build does not know is hidden from the
//! usable-index view but preserved through unrelated commits.

use std::sync::Arc;

use arrow_array::{
    Array, Decimal128Array, Float64Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator,
    UInt64Array,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_core::ROW_ADDR;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use lance_index::scalar::zonemap::{
    ZoneMapIndexBuilder, ZoneMapIndexBuilderParams, ZoneMapIndexVersion,
};
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams, index_files_to_table};
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::Dataset;
use crate::dataset::WriteParams;
use crate::dataset::index::LanceIndexStoreExt;
use crate::dataset::transaction::{Operation, Transaction};
use crate::index::DatasetIndexExt;
use crate::utils::test::copy_test_data_to_tmp;

const ROWS_PER_ZONE: u64 = 2;
const INDEX_NAME: &str = "val_zonemap";

/// Indices the manifest itself carries, bypassing the usable-index filter.
async fn raw_manifest_indices(dataset: &Dataset) -> Vec<IndexMetadata> {
    lance_table::io::manifest::read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await
    .unwrap()
}

/// Query `predicate`, with and without the scalar index, and check that the
/// indexed plan (when `expect_index`) consults the index and that both agree.
async fn assert_query_agrees(dataset: &Dataset, predicate: &str, expect_index: bool) -> Vec<i64> {
    async fn ids(dataset: &Dataset, predicate: &str, use_scalar_index: bool) -> (String, Vec<i64>) {
        let mut scan = dataset.scan();
        scan.project(&["id"])
            .unwrap()
            .use_scalar_index(use_scalar_index)
            .filter(predicate)
            .unwrap();
        let plan = scan.explain_plan(false).await.unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let column = batch["id"].clone();
        let mut ids: Vec<i64> = match column.data_type() {
            DataType::Int64 => column
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .to_vec(),
            DataType::Int32 => column
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .iter()
                .map(|id| i64::from(*id))
                .collect(),
            other => panic!("unexpected id type {other:?}"),
        };
        ids.sort_unstable();
        (plan, ids)
    }
    let (indexed_plan, indexed_ids) = ids(dataset, predicate, true).await;
    let (flat_plan, flat_ids) = ids(dataset, predicate, false).await;
    assert_eq!(
        indexed_plan.contains("ScalarIndexQuery"),
        expect_index,
        "{indexed_plan}"
    );
    assert!(!flat_plan.contains("ScalarIndexQuery"), "{flat_plan}");
    assert_eq!(
        indexed_ids, flat_ids,
        "indexed query diverged for {predicate}"
    );
    flat_ids
}

fn zone_map_params() -> ScalarIndexParams {
    ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
        .with_params(&serde_json::json!({"rows_per_zone": ROWS_PER_ZONE, "use_seeds": false}))
}

fn float_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("val", DataType::Float64, true),
    ]))
}

fn float_rows(ids: std::ops::Range<i32>, values: &[Option<f64>]) -> RecordBatch {
    assert_eq!(ids.len(), values.len());
    RecordBatch::try_new(
        float_schema(),
        vec![
            Arc::new(Int32Array::from_iter_values(ids)),
            Arc::new(Float64Array::from(values.to_vec())),
        ],
    )
    .unwrap()
}

/// Writes a version 0 ZoneMap segment file for `values`, laid out as
/// `fragments` of four rows each, into a fresh index directory of `dataset`.
/// Returns the metadata to commit for it.
async fn write_v0_segment(
    dataset: &Dataset,
    template: &IndexMetadata,
    fragments: &[u32],
    values: &[Option<f64>],
) -> IndexMetadata {
    assert_eq!(values.len(), fragments.len() * 4);
    let uuid = Uuid::new_v4();
    let store = LanceIndexStore::from_dataset_for_new(dataset, &uuid).unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(VALUE_COLUMN_NAME, DataType::Float64, true),
        ArrowField::new(ROW_ADDR, DataType::UInt64, false),
    ]));
    let row_addrs: Vec<u64> = fragments
        .iter()
        .flat_map(|fragment| (0..4u64).map(move |row| (u64::from(*fragment) << 32) | row))
        .collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(values.to_vec())),
            Arc::new(UInt64Array::from(row_addrs)),
        ],
    )
    .unwrap();
    let mut builder = ZoneMapIndexBuilder::try_new(
        ZoneMapIndexBuilderParams::new(ROWS_PER_ZONE),
        DataType::Float64,
    )
    .unwrap()
    .with_version(ZoneMapIndexVersion::V0);
    builder
        .train(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(batch))),
        )))
        .await
        .unwrap();
    let files = builder.write_index(&store).await.unwrap();
    IndexMetadata {
        uuid,
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(RoaringBitmap::from_iter(fragments.iter().copied())),
        index_version: ZoneMapIndexVersion::V0 as i32,
        files: Some(index_files_to_table(files)),
        created_at: None,
        base_id: None,
        ..template.clone()
    }
}

/// Rows 0..8: `0:1.0 1:NaN 2:3.0 3:4.0 4:10.0 5:-NaN 6:null 7:13.0`.
const V0_VALUES: [Option<f64>; 8] = [
    Some(1.0),
    Some(f64::NAN),
    Some(3.0),
    Some(4.0),
    Some(10.0),
    Some(-f64::NAN),
    None,
    Some(13.0),
];

/// A dataset with eight Float64 rows in two fragments whose only ZoneMap
/// segment is a hand-written version 0 file (legacy NaN marker, unknown sign
/// split) covering both fragments.
async fn dataset_with_v0_segment(uri: &str) -> Dataset {
    dataset_with_v0_coverage(uri, &[]).await
}

/// `V0_VALUES` in fragments 0 and 1, covered by a hand-written version 0
/// segment, followed by `extra_fragments` of four rows each that no segment
/// covers.
async fn dataset_with_v0_coverage(uri: &str, extra_fragments: &[[Option<f64>; 4]]) -> Dataset {
    let values = V0_VALUES;
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(float_rows(0..8, &values))], float_schema()),
        uri,
        Some(WriteParams {
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);

    // A real version 1 segment supplies the details to copy, then is replaced
    // by the hand-written version 0 segment in one commit.
    dataset
        .create_index_builder(&["val"], IndexType::ZoneMap, &zone_map_params())
        .name(INDEX_NAME.to_string())
        .await
        .unwrap();
    let current = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].index_version, 1);
    let v0 = write_v0_segment(&dataset, &current[0], &[0, 1], &values).await;
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![v0],
            removed_indices: current,
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();
    for (i, rows) in extra_fragments.iter().enumerate() {
        let start = 8 + 4 * i as i32;
        dataset
            .append(
                RecordBatchIterator::new([Ok(float_rows(start..start + 4, rows))], float_schema()),
                None,
            )
            .await
            .unwrap();
    }
    dataset
}

fn segments_named(indices: &[IndexMetadata]) -> Vec<IndexMetadata> {
    let mut segments: Vec<IndexMetadata> = indices
        .iter()
        .filter(|index| index.name == INDEX_NAME)
        .cloned()
        .collect();
    segments.sort_by_key(|segment| segment.fragment_bitmap.as_ref().and_then(|b| b.min()));
    segments
}

/// Checks a result against the rows whose membership does not depend on how
/// the engine orders NaNs: `must` are finite rows that satisfy the predicate,
/// `must_not` are finite or null rows that do not. NaN rows follow the
/// engine's comparison semantics and are the same with or without the index,
/// which `assert_query_agrees` already pins.
fn assert_finite_rows(ids: &[i64], must: &[i64], must_not: &[i64]) {
    for id in must {
        assert!(ids.contains(id), "row {id} missing from {ids:?}");
    }
    for id in must_not {
        assert!(!ids.contains(id), "row {id} wrongly in {ids:?}");
    }
}

fn only_segment(indices: &[IndexMetadata]) -> &IndexMetadata {
    let segments: Vec<_> = indices
        .iter()
        .filter(|index| index.name == INDEX_NAME)
        .collect();
    assert_eq!(segments.len(), 1, "{segments:?}");
    segments[0]
}

/// The released version 0 fixture answers queries under the version 0
/// contract; appending and optimizing rebuilds its single segment as a
/// version 1 segment covering every fragment, rather than updating the
/// version 0 file in place.
#[tokio::test]
async fn test_v8_zonemap_segment_is_rebuilt_as_version_1() {
    let test_dir = copy_test_data_to_tmp("v8.0.0/decimal_zonemap").unwrap();
    let mut dataset = Dataset::open(&test_dir.path_str()).await.unwrap();

    let before = raw_manifest_indices(&dataset).await;
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].index_version, 0);
    assert_eq!(dataset.load_indices().await.unwrap().len(), 1);
    let equals_two = "value = arrow_cast(2.00, 'Decimal128(10, 2)')";
    assert_eq!(assert_query_agrees(&dataset, equals_two, true).await, [2]);

    let arrow_schema = Arc::new(ArrowSchema::from(dataset.schema()));
    let ids = Int64Array::from(vec![100, 101, 102]);
    let values = Decimal128Array::from(vec![200i128, 900, 250])
        .with_precision_and_scale(10, 2)
        .unwrap();
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(ids) as _, Arc::new(values) as _],
    )
    .unwrap_or_else(|err| panic!("fixture schema changed: {err}; {arrow_schema:?}"));
    dataset
        .append(RecordBatchIterator::new([Ok(batch)], arrow_schema), None)
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();

    let after = raw_manifest_indices(&dataset).await;
    assert_eq!(after.len(), 1, "{after:?}");
    assert_eq!(after[0].index_version, 1);
    assert_ne!(after[0].uuid, before[0].uuid);
    assert_eq!(
        after[0].fragment_bitmap.as_ref().unwrap(),
        dataset.fragment_bitmap.as_ref(),
        "the rebuilt segment covers every fragment"
    );
    assert_eq!(
        assert_query_agrees(&dataset, equals_two, true).await,
        [2, 100]
    );
    assert_eq!(
        assert_query_agrees(
            &dataset,
            "value > arrow_cast(2.40, 'Decimal128(10, 2)')",
            true
        )
        .await,
        [3, 101, 102]
    );
}

/// A version 0 float segment stays conservative while it is read, and the
/// first maintenance after an append rebuilds it into a version 1 segment
/// whose NaN sign split is exact.
#[tokio::test]
async fn test_version_0_float_segment_is_read_then_rebuilt() {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_v0_segment(dir.as_str()).await;
    let v0 = only_segment(&raw_manifest_indices(&dataset).await).clone();
    assert_eq!(v0.index_version, 0);
    assert_eq!(
        only_segment(&dataset.load_indices().await.unwrap()).uuid,
        v0.uuid,
        "a version 0 segment is usable"
    );

    // Version 0 contract: zones with NaNs carry the marker and an unknown
    // split, which keeps them as candidates; results are still exact after
    // the recheck. Rows: 0:1.0 1:NaN 2:3.0 3:4.0 4:10.0 5:-NaN 6:null 7:13.0.
    assert_finite_rows(
        &assert_query_agrees(&dataset, "val > 2.0", true).await,
        &[2, 3, 4, 7],
        &[0, 6],
    );
    assert_finite_rows(
        &assert_query_agrees(&dataset, "val < 2.0", true).await,
        &[0],
        &[2, 3, 4, 6, 7],
    );
    assert_eq!(assert_query_agrees(&dataset, "val = 10.0", true).await, [4]);
    assert_eq!(
        assert_query_agrees(&dataset, "val IS NULL", true).await,
        [6]
    );

    dataset
        .append(
            RecordBatchIterator::new(
                [Ok(float_rows(
                    8..12,
                    &[Some(20.0), Some(f64::NAN), Some(-f64::NAN), Some(2.5)],
                ))],
                float_schema(),
            ),
            None,
        )
        .await
        .unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::default())
        .await
        .unwrap();

    let v1 = only_segment(&raw_manifest_indices(&dataset).await).clone();
    assert_eq!(v1.index_version, 1);
    assert_ne!(
        v1.uuid, v0.uuid,
        "the version 0 file is not updated in place"
    );
    assert_eq!(
        v1.fragment_bitmap.as_ref().unwrap(),
        dataset.fragment_bitmap.as_ref()
    );
    // Appended rows: 8:20.0 9:NaN 10:-NaN 11:2.5.
    assert_finite_rows(
        &assert_query_agrees(&dataset, "val > 2.0", true).await,
        &[2, 3, 4, 7, 8, 11],
        &[0, 6],
    );
    assert_finite_rows(
        &assert_query_agrees(&dataset, "val < 2.0", true).await,
        &[0],
        &[2, 3, 4, 6, 7, 8, 11],
    );
    assert_eq!(assert_query_agrees(&dataset, "val = 10.0", true).await, [4]);
    assert_eq!(
        assert_query_agrees(&dataset, "val IS NULL", true).await,
        [6]
    );
}

/// A ZoneMap segment whose version this build does not know is hidden from
/// the usable-index view, so queries plan without it, while unrelated
/// commits carry its manifest entry forward unchanged. Seeds are disabled so
/// nothing about seed flags can be what hides it.
#[tokio::test]
async fn test_unsupported_zonemap_version_is_hidden_but_preserved() {
    let dir = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(
            [Ok(float_rows(
                0..4,
                &[Some(1.0), Some(2.0), Some(3.0), Some(4.0)],
            ))],
            float_schema(),
        ),
        dir.as_str(),
        None,
    )
    .await
    .unwrap();
    dataset
        .create_index_builder(&["val"], IndexType::ZoneMap, &zone_map_params())
        .name(INDEX_NAME.to_string())
        .await
        .unwrap();
    assert_eq!(
        assert_query_agrees(&dataset, "val > 2.0", true).await,
        [2, 3]
    );

    let current = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
    assert_eq!(current.len(), 1);
    let mut future = current.clone();
    future[0].index_version = ZoneMapIndexVersion::CURRENT as i32 + 1;
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: future.clone(),
            removed_indices: current,
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();

    assert!(
        dataset.load_indices().await.unwrap().is_empty(),
        "an unsupported version must not be offered to queries"
    );
    assert_eq!(
        assert_query_agrees(&dataset, "val > 2.0", false).await,
        [2, 3]
    );

    // An unrelated commit keeps the entry exactly as it was.
    dataset
        .append(
            RecordBatchIterator::new(
                [Ok(float_rows(4..6, &[Some(5.0), Some(6.0)]))],
                float_schema(),
            ),
            None,
        )
        .await
        .unwrap();
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert!(reopened.load_indices().await.unwrap().is_empty());
    // The manifest stores `created_at` at a coarser precision than the
    // in-memory entry; everything else must be carried forward verbatim.
    let without_created_at = |mut indices: Vec<IndexMetadata>| {
        for index in &mut indices {
            index.created_at = None;
        }
        indices
    };
    assert_eq!(
        without_created_at(raw_manifest_indices(&reopened).await),
        without_created_at(future)
    );
    assert_eq!(
        assert_query_agrees(&reopened, "val > 2.0", false).await,
        [2, 3, 4, 5]
    );
}

/// The staged maintenance flow: workers build uncommitted version 1
/// segments for fragments no committed segment covers, merge those
/// intermediates, and commit the result beside the existing version 0
/// coverage. The old segment is neither absorbed nor rewritten, both
/// segments serve queries, and a direct merge that mixes the two versions
/// is refused without publishing anything.
#[tokio::test]
async fn test_staged_version_1_intermediates_merge_beside_version_0_coverage() {
    let dir = TempStrDir::default();
    let mut dataset = dataset_with_v0_coverage(
        dir.as_str(),
        &[
            [Some(20.0), Some(f64::NAN), Some(-f64::NAN), Some(2.5)],
            [Some(30.0), None, Some(31.0), Some(0.5)],
        ],
    )
    .await;
    assert_eq!(dataset.get_fragments().len(), 4);
    let v0 = only_segment(&raw_manifest_indices(&dataset).await).clone();
    assert_eq!(v0.index_version, 0);

    // Fresh intermediates over the uncovered fragments, one per worker. The
    // name is taken, so the builder needs `replace`; nothing is removed here
    // because the segments stay uncommitted, and the later commit replaces
    // by coverage overlap only.
    let mut intermediates = Vec::new();
    for fragment in [2u32, 3] {
        let segment = dataset
            .create_index_builder(&["val"], IndexType::ZoneMap, &zone_map_params())
            .name(INDEX_NAME.to_string())
            .replace(true)
            .fragments(vec![fragment])
            .execute_uncommitted()
            .await
            .unwrap();
        assert_eq!(segment.index_version, 1);
        intermediates.push(segment);
    }
    let merged = dataset
        .merge_existing_index_segments(intermediates.clone())
        .await
        .unwrap();
    assert_eq!(merged.index_version, 1);
    assert_eq!(
        merged.fragment_bitmap.as_ref().unwrap(),
        &RoaringBitmap::from_iter([2u32, 3])
    );
    dataset
        .commit_existing_index_segments(INDEX_NAME, "val", vec![merged.clone()])
        .await
        .unwrap();

    let segments = segments_named(&raw_manifest_indices(&dataset).await);
    assert_eq!(segments.len(), 2, "{segments:?}");
    assert_eq!(
        segments[0].uuid, v0.uuid,
        "the version 0 segment is untouched"
    );
    assert_eq!(segments[0].index_version, 0);
    assert_eq!(segments[1].uuid, merged.uuid);
    assert_eq!(segments[1].index_version, 1);
    let usable = segments_named(&dataset.load_indices().await.unwrap());
    assert_eq!(usable.len(), 2, "both versions serve queries");

    // Rows 8..16: 8:20.0 9:NaN 10:-NaN 11:2.5 12:30.0 13:null 14:31.0 15:0.5.
    assert_finite_rows(
        &assert_query_agrees(&dataset, "val > 2.0", true).await,
        &[2, 3, 4, 7, 8, 11, 12, 14],
        &[0, 6, 13, 15],
    );
    assert_finite_rows(
        &assert_query_agrees(&dataset, "val < 2.0", true).await,
        &[0, 15],
        &[2, 3, 4, 6, 7, 8, 11, 12, 13, 14],
    );
    assert_eq!(
        assert_query_agrees(&dataset, "val = 30.0", true).await,
        [12]
    );
    assert_eq!(
        assert_query_agrees(&dataset, "val IS NULL", true).await,
        [6, 13]
    );

    // A direct merge across the two versions is refused before any write;
    // the committed coverage and the answers do not change.
    let before = raw_manifest_indices(&dataset).await;
    let err = dataset
        .merge_existing_index_segments(vec![v0.clone(), merged.clone()])
        .await
        .expect_err("merging version 0 and version 1 segments must be refused");
    assert!(
        matches!(err, crate::Error::NotSupported { .. }),
        "unexpected error: {err}"
    );
    assert_eq!(raw_manifest_indices(&dataset).await, before);
    let reopened = Dataset::open(dir.as_str()).await.unwrap();
    assert_eq!(
        segments_named(&reopened.load_indices().await.unwrap()).len(),
        2
    );
    assert_eq!(
        assert_query_agrees(&reopened, "val = 30.0", true).await,
        [12]
    );
    assert_eq!(
        assert_query_agrees(&reopened, "val IS NULL", true).await,
        [6, 13]
    );
}
