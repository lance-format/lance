// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::array::as_primitive_array;
use arrow::datatypes::{Float64Type, Int32Type};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch,
    RecordBatchIterator, StringArray,
};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use arrow_select::take;
use datafusion::logical_expr::col;
use futures::TryStreamExt;
use lance_arrow::{FixedSizeListArrayExt, SchemaExt};
use lance_core::utils::address::RowAddress;
use lance_core::utils::tempfile::TempStrDir;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{
    ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION, ROW_OFFSET,
};
use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::query::{BooleanQuery, BoostQuery, FtsQuery, MatchQuery, Occur};
use lance_index::vector::DIST_COL;
use lance_index::vector::ivf::IvfBuildParams;
use lance_io::stream::RecordBatchStream;
use lance_linalg::distance::MetricType;
use rstest::rstest;

use crate::Error;
use crate::dataset::scanner::test_dataset::{TestVectorDataset, batch_row_ids};
use crate::dataset::scanner::{
    ColumnOrdering, FragmentSlice, RowAddrMask, RowAddrTreeMap, Scanner,
};
use crate::dataset::{Dataset, WriteMode, WriteParams};
use crate::index::DatasetIndexExt;
use crate::index::vector::VectorIndexParams;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};

#[rstest]
#[case::without_stable_row_ids(false)]
#[case::with_stable_row_ids(true)]
#[tokio::test]
async fn physical_row_selection_intersects_deletions_and_scalar_index(
    #[case] stable_row_ids: bool,
) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, stable_row_ids)
        .await
        .unwrap();
    test_ds.dataset.delete("i = 2 OR i = 203").await.unwrap();
    test_ds.make_scalar_index().await.unwrap();
    let ds = &test_ds.dataset;
    let fragments = ds.get_fragments();
    assert_eq!(fragments.len(), 2);

    let mut physical_rows = RowAddrTreeMap::new();
    for (fragment_id, start, end) in [
        (fragments[0].id() as u32, 1, 4),
        (fragments[1].id() as u32, 2, 5),
    ] {
        physical_rows.insert_range(
            u64::from(RowAddress::new_from_parts(fragment_id, start))
                ..u64::from(RowAddress::new_from_parts(fragment_id, end)),
        );
    }

    let mut scan = ds.scan();
    scan.with_physical_row_addr_prefilter(physical_rows.clone());
    scan.filter("i >= 3").unwrap();
    scan.project(&["i"]).unwrap();
    assert!(
        scan.explain_plan(false)
            .await
            .unwrap()
            .contains("ScalarIndexQuery")
    );
    let batch = scan.try_into_batch().await.unwrap();
    let values = batch["i"].as_primitive::<Int32Type>().values().to_vec();
    assert_eq!(values, vec![3, 202, 204]);

    // Limit and offset apply after physical selection and deletion filtering,
    // including when there is no SQL filter to prevent scan-range pushdown.
    let mut limited_scan = ds.scan();
    limited_scan.with_physical_row_addr_prefilter(physical_rows.clone());
    limited_scan.project(&["i"]).unwrap();
    limited_scan.limit(Some(2), Some(1)).unwrap();
    let batch = limited_scan.try_into_batch().await.unwrap();
    assert_eq!(batch["i"].as_primitive::<Int32Type>().values(), &[3, 202]);

    let mut count_scan = ds.scan();
    count_scan.with_physical_row_addr_prefilter(physical_rows);
    count_scan.filter("i >= 3").unwrap();
    assert_eq!(count_scan.count_rows().await.unwrap(), 3);
}

#[rstest]
#[case::physical_ids(false)]
#[case::stable_ids(true)]
#[tokio::test]
async fn fragment_slices_core_semantics(#[case] stable_ids: bool) {
    let mut fixture = TestVectorDataset::new(LanceFileVersion::Stable, stable_ids)
        .await
        .unwrap();
    fixture.dataset.delete("i = 2").await.unwrap();
    let ds = &fixture.dataset;
    let fragments = ds.get_fragments();
    let first = fragments[0].id() as u32;
    let second = fragments[1].id() as u32;
    let slices = [
        FragmentSlice {
            fragment_id: second,
            row_offset: 2,
            row_count: 2,
        },
        FragmentSlice {
            fragment_id: first,
            row_offset: 1,
            row_count: 3,
        },
        FragmentSlice {
            fragment_id: first,
            row_offset: 2,
            row_count: 2,
        },
    ];
    let mut scan = ds.scan();
    scan.with_fragment_slices(&slices).await.unwrap();
    scan.project(&["i"]).unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    assert_eq!(
        batch["i"].as_primitive::<Int32Type>().values(),
        &[1, 3, 202, 203]
    );
    // Fragment scope intersects slices regardless of setter order.
    for scope_first in [true, false] {
        let mut scan = ds.scan();
        let scope = vec![fragments[1].metadata().clone()];
        if scope_first {
            scan.with_fragments(scope.clone());
        }
        scan.with_fragment_slices(&slices).await.unwrap();
        if !scope_first {
            scan.with_fragments(scope);
        }
        scan.project(&["i"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(batch["i"].as_primitive::<Int32Type>().values(), &[202, 203]);
    }
    scan.with_fragment_slices(&[]).await.unwrap();
    assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), 0);
    scan.with_fragment_slices(&[FragmentSlice {
        fragment_id: first,
        row_offset: 200,
        row_count: 0,
    }])
    .await
    .unwrap();
    assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), 0);
}

#[rstest]
#[case::ann(true)]
#[case::fts(false)]
#[tokio::test]
async fn physical_row_selection_rejects_search(#[case] is_ann: bool) {
    let fixture = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    let mut scan = fixture.dataset.scan();
    scan.with_fragment_slices(&[FragmentSlice {
        fragment_id: 0,
        row_offset: 0,
        row_count: 1,
    }])
    .await
    .unwrap();
    if is_ann {
        scan.nearest("vec", &Float32Array::from(vec![0.0; 32]), 1)
            .unwrap();
    } else {
        scan.full_text_search(FullTextSearchQuery::new("query".into()))
            .unwrap();
    }
    let error = scan.create_plan().await.unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("not supported for vector or full-text search")
    );
}

#[tokio::test]
async fn physical_row_addr_prefilter_rejects_invalid_addresses() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    let ds = &test_ds.dataset;

    let mut missing_fragment = RowAddrTreeMap::new();
    missing_fragment.insert_fragment(42);
    let mut scan = ds.scan();
    scan.with_physical_row_addr_prefilter(missing_fragment);
    let error = scan.create_plan().await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("references fragment_id=42, which is not present")
    );

    let fragment = &ds.get_fragments()[0];
    let physical_row_count = fragment.physical_rows().await.unwrap();
    let invalid_offset = u32::try_from(physical_row_count).unwrap();
    let mut past_fragment_end = RowAddrTreeMap::new();
    past_fragment_end.insert(u64::from(RowAddress::new_from_parts(
        fragment.id() as u32,
        invalid_offset,
    )));
    let mut scan = ds.scan();
    scan.with_physical_row_addr_prefilter(past_fragment_end);
    let error = scan.create_plan().await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains(&format!(
        "contains row_offset={invalid_offset}, but the fragment has physical_row_count={physical_row_count}"
    )));

    let mut physical_rows = RowAddrTreeMap::new();
    physical_rows.insert(u64::from(RowAddress::new_from_parts(
        fragment.id() as u32,
        0,
    )));
    let mut scan = ds.scan();
    scan.with_physical_row_addr_prefilter(physical_rows);
    scan.full_text_search(FullTextSearchQuery::new("query".into()))
        .unwrap();
    let error = scan.create_plan().await.unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(
        error
            .to_string()
            .contains("not supported for vector or full-text search")
    );
}

#[tokio::test]
async fn physical_and_logical_row_prefilters_are_intersected() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    let ds = &test_ds.dataset;
    let fragment_id = ds.get_fragments()[0].id() as u32;

    let mut physical_rows = RowAddrTreeMap::new();
    physical_rows.insert_range(
        u64::from(RowAddress::new_from_parts(fragment_id, 1))
            ..u64::from(RowAddress::new_from_parts(fragment_id, 4)),
    );
    let logical_rows = RowAddrTreeMap::from_iter([
        u64::from(RowAddress::new_from_parts(fragment_id, 0)),
        u64::from(RowAddress::new_from_parts(fragment_id, 2)),
    ]);

    let mut scan = ds.scan();
    scan.with_physical_row_addr_prefilter(physical_rows);
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(logical_rows));
    scan.project(&["i"]).unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let values = batch["i"].as_primitive::<Int32Type>().values().to_vec();
    assert_eq!(values, vec![2]);
}

#[tokio::test]
async fn take_shortcut_respects_physical_row_selection() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    let ds = &test_ds.dataset;
    let fragment_id = ds.get_fragments()[0].id() as u32;
    let excluded = u64::from(RowAddress::new_from_parts(fragment_id, 0));

    let mut physical_rows = RowAddrTreeMap::new();
    physical_rows.insert(u64::from(RowAddress::new_from_parts(fragment_id, 1)));

    let mut scan = ds.scan();
    scan.with_physical_row_addr_prefilter(physical_rows);
    scan.filter(&format!("_rowaddr = {excluded}")).unwrap();
    scan.project(&["i"]).unwrap();
    assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), 0);
}

#[tokio::test]
#[allow(deprecated)]
async fn row_addr_prefilter_preserves_stable_row_id_semantics() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    let ds = &test_ds.dataset;

    let mut all = ds.scan();
    all.with_row_id();
    let target = batch_row_ids(&all.try_into_batch().await.unwrap())[250];

    let mut scan = ds.scan();
    scan.with_row_id();
    scan.with_row_addr_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([
        target,
    ])));
    assert_eq!(
        batch_row_ids(&scan.try_into_batch().await.unwrap()),
        vec![target]
    );
}

#[rstest]
#[case::without_stable_row_ids(false)]
#[case::with_stable_row_ids(true)]
#[tokio::test]
async fn row_addr_mask_plain_scan_allow_block_refine(#[case] stable_row_ids: bool) {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, stable_row_ids)
        .await
        .unwrap();
    let ds = &test_ds.dataset;

    let mut scan = ds.scan();
    scan.with_row_id();
    let all_ids = batch_row_ids(&scan.try_into_batch().await.unwrap());
    let all_set: BTreeSet<u64> = all_ids.iter().copied().collect();
    let allow: Vec<u64> = all_ids.iter().copied().step_by(2).collect();
    let allow_set: BTreeSet<u64> = allow.iter().copied().collect();

    // Allow-mask plain scan returns exactly the allowed rows.
    let mut scan = ds.scan();
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
        allow.iter().copied(),
    )));
    scan.with_row_id();
    let got: BTreeSet<u64> = batch_row_ids(&scan.try_into_batch().await.unwrap())
        .into_iter()
        .collect();
    assert_eq!(got, allow_set);

    // Block-mask plain scan returns every row except the blocked ones, which
    // also exercises FilteredReadExec index-input serialization of a BlockList.
    let block: Vec<u64> = all_ids.iter().copied().step_by(3).collect();
    let block_set: BTreeSet<u64> = block.iter().copied().collect();
    let mut scan = ds.scan();
    scan.with_row_id_prefilter(RowAddrMask::from_block(RowAddrTreeMap::from_iter(
        block.iter().copied(),
    )));
    scan.with_row_id();
    let got: BTreeSet<u64> = batch_row_ids(&scan.try_into_batch().await.unwrap())
        .into_iter()
        .collect();
    let expected: BTreeSet<u64> = all_set.difference(&block_set).copied().collect();
    assert_eq!(got, expected);

    // With a SQL refine, the result is the allowed rows that also match the filter.
    let mut scan = ds.scan();
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
        allow.iter().copied(),
    )));
    scan.filter("i >= 200").unwrap();
    scan.project(&["i"]).unwrap();
    scan.with_row_id();
    let refined = scan.try_into_batch().await.unwrap();
    let refined_ids: BTreeSet<u64> = batch_row_ids(&refined).into_iter().collect();
    assert!(refined_ids.is_subset(&allow_set) && !refined_ids.is_empty());
    let is = refined
        .column_by_name("i")
        .unwrap()
        .as_primitive::<Int32Type>();
    assert!(is.values().iter().all(|v| *v >= 200));
}

#[tokio::test]
async fn row_addr_mask_plain_scan_rejected_on_legacy() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Legacy, false)
        .await
        .unwrap();
    let ds = &test_ds.dataset;
    let mut scan = ds.scan();
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([0u64])));
    let Err(err) = scan.try_into_stream().await else {
        panic!("expected legacy-storage masked plain scan to be rejected");
    };
    assert!(
        err.to_string().contains("legacy-storage"),
        "unexpected: {err}"
    );
}

#[rstest]
#[case::without_stable_row_ids(false)]
#[case::with_stable_row_ids(true)]
#[tokio::test]
async fn row_addr_mask_ann_search_only_allowed(#[case] stable_row_ids: bool) {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, stable_row_ids)
        .await
        .unwrap();
    test_ds.make_vector_index().await.unwrap();
    // Append after indexing so the appended fragment is unindexed (flat branch).
    test_ds.append_new_data().await.unwrap();
    let ds = &test_ds.dataset;

    let mut scan = ds.scan();
    scan.with_row_id();
    let all_ids = batch_row_ids(&scan.try_into_batch().await.unwrap());
    let allow: Vec<u64> = all_ids.iter().copied().step_by(3).collect();
    let allow_set: BTreeSet<u64> = allow.iter().copied().collect();

    let key: Float32Array = (0..32).map(|v| v as f32).collect();
    let mut scan = ds.scan();
    scan.nearest("vec", &key, 15).unwrap();
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
        allow.iter().copied(),
    )));
    scan.with_row_id();
    let got = batch_row_ids(&scan.try_into_batch().await.unwrap());
    assert!(!got.is_empty());
    for id in got {
        assert!(
            allow_set.contains(&id),
            "returned _rowid {id} not in allowlist"
        );
    }
}

#[tokio::test]
async fn row_addr_mask_plain_scan_with_limit() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    let ds = &test_ds.dataset;

    let mut scan = ds.scan();
    scan.with_row_id();
    let all_ids = batch_row_ids(&scan.try_into_batch().await.unwrap());
    let allow: Vec<u64> = all_ids.iter().copied().step_by(2).collect();
    let allow_set: BTreeSet<u64> = allow.iter().copied().collect();

    // limit must apply AFTER masking: 5 rows, all from the allowlist.
    let mut scan = ds.scan();
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
        allow.iter().copied(),
    )));
    scan.limit(Some(5), None).unwrap();
    scan.with_row_id();
    let got = batch_row_ids(&scan.try_into_batch().await.unwrap());
    assert_eq!(got.len(), 5, "masked limit should yield 5 masked rows");
    for id in &got {
        assert!(allow_set.contains(id), "returned {id} not allowed");
    }
}

#[tokio::test]
async fn row_addr_mask_plain_scan_filter_unprojected_column() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    let ds = &test_ds.dataset;

    let mut scan = ds.scan();
    scan.with_row_id();
    let all_ids = batch_row_ids(&scan.try_into_batch().await.unwrap());

    // Allow everything; filter on `i` but project only `s` (unrelated column).
    let mut scan = ds.scan();
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
        all_ids.iter().copied(),
    )));
    scan.filter("i >= 200").unwrap();
    scan.project(&["s"]).unwrap();
    let out = scan.try_into_batch().await.unwrap();
    assert_eq!(out.num_rows(), 200, "expected 200 rows with i>=200");
}

#[tokio::test]
async fn row_addr_mask_plain_scan_exact_index_filter_unprojected_column() {
    // A scalar index on `i` turns `i >= 200` into an exact index query with no
    // refine. Under an external mask that predicate is demoted to a refine over
    // the masked rows, so `i` must still be projected for the read even though
    // the user only asked for `s`.
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    test_ds.make_scalar_index().await.unwrap();
    let ds = &test_ds.dataset;

    let mut scan = ds.scan();
    scan.with_row_id();
    let all_ids = batch_row_ids(&scan.try_into_batch().await.unwrap());

    let mut scan = ds.scan();
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
        all_ids.iter().copied(),
    )));
    scan.filter("i >= 200").unwrap();
    scan.project(&["s"]).unwrap();
    let out = scan.try_into_batch().await.unwrap();
    assert_eq!(out.num_rows(), 200, "expected 200 rows with i>=200");
}

/// A `_rowid` predicate is recognized as a TakeOperation and short-circuits
/// straight to `take_source`, which used to skip the mask entirely.
#[tokio::test]
async fn row_addr_mask_take_shortcut_respects_mask() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    let ds = &test_ds.dataset;

    let mut scan = ds.scan();
    scan.with_row_id();
    let all_ids = batch_row_ids(&scan.try_into_batch().await.unwrap());
    let target = all_ids[0];

    // Sanity: unmasked, the shortcut returns the row.
    let mut scan = ds.scan();
    scan.with_row_id();
    scan.filter(&format!("_rowid = {target}")).unwrap();
    assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), 1);

    // Masked to nothing, it must return nothing.
    let mut scan = ds.scan();
    scan.with_row_id();
    scan.filter(&format!("_rowid = {target}")).unwrap();
    scan.with_row_id_prefilter(RowAddrMask::allow_nothing());
    assert_eq!(
        scan.try_into_batch().await.unwrap().num_rows(),
        0,
        "the take shortcut must not return rows the mask excludes"
    );

    // And an allow-list restricts it rather than being ignored.
    let mut scan = ds.scan();
    scan.with_row_id();
    scan.filter(&format!("_rowid = {target}")).unwrap();
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([
        target,
    ])));
    assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), 1);
}

/// A same-column compound query (Boost here) is optimized into
/// CompoundFtsScorer, a scorer that built its prefilter without the mask.
#[tokio::test]
async fn row_addr_mask_compound_fts_respects_mask() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    test_ds.make_fts_index().await.unwrap();
    let ds = &test_ds.dataset;

    let compound = || {
        let positive = MatchQuery::new("4".to_owned()).with_column(Some("s".to_owned()));
        let negative = MatchQuery::new("9".to_owned()).with_column(Some("s".to_owned()));
        FullTextSearchQuery::new_query(
            BoostQuery::new(positive.into(), negative.into(), Some(1.0)).into(),
        )
    };

    let mut scan = ds.scan();
    scan.full_text_search(compound()).unwrap();
    scan.with_row_id();
    let plan = scan.explain_plan(true).await.unwrap();
    assert!(
        plan.contains("CompoundFtsScorer"),
        "expected the compound scorer path, got:\n{plan}"
    );
    let base = batch_row_ids(&scan.try_into_batch().await.unwrap());
    assert!(!base.is_empty(), "compound query matched nothing");

    let mut scan = ds.scan();
    scan.full_text_search(compound()).unwrap();
    scan.with_row_id();
    scan.with_row_id_prefilter(RowAddrMask::allow_nothing());
    assert_eq!(
        scan.try_into_batch().await.unwrap().num_rows(),
        0,
        "the compound scorer must not return rows the mask excludes"
    );

    // Allow exactly one baseline hit; only that one may come back.
    let keep = base[0];
    let mut scan = ds.scan();
    scan.full_text_search(compound()).unwrap();
    scan.with_row_id();
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([keep])));
    assert_eq!(
        batch_row_ids(&scan.try_into_batch().await.unwrap()),
        vec![keep]
    );
}

/// A cross-column boolean query plans into CrossColumnCompoundFtsScorer,
/// which is a different exec from the same-column CompoundFtsScorer and
/// builds its own prefilter, so it needs the mask threaded separately.
#[tokio::test]
async fn row_addr_mask_cross_column_fts_respects_mask() {
    use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("title", DataType::Utf8, true),
        ArrowField::new("body", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from_iter_values(
                (0..64).map(|v| format!("alpha title {v}")),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..64).map(|v| format!("alpha body {v}")),
            )),
        ],
    )
    .unwrap();

    let path = TempStrDir::default();
    let reader = RecordBatchIterator::new([Ok(batch)], schema.clone());
    let mut dataset = Dataset::write(reader, &path, None).await.unwrap();
    let params = InvertedIndexParams::default()
        .with_position(true)
        .remove_stop_words(false);
    for column in ["title", "body"] {
        dataset
            .create_index(&[column], IndexType::Inverted, None, &params, true)
            .await
            .unwrap();
    }

    // Two leaves on different columns is what selects the cross-column
    // scorer; a bounded limit is required by that exec.
    let cross_column = || {
        FullTextSearchQuery::new_query(FtsQuery::Boolean(BooleanQuery::new([
            (
                Occur::Should,
                MatchQuery::new("title".to_string())
                    .with_column(Some("title".to_string()))
                    .into(),
            ),
            (
                Occur::Should,
                MatchQuery::new("body".to_string())
                    .with_column(Some("body".to_string()))
                    .into(),
            ),
        ])))
        .limit(Some(10))
    };

    let mut scan = dataset.scan();
    scan.full_text_search(cross_column()).unwrap();
    scan.with_row_id();
    let plan = scan.explain_plan(true).await.unwrap();
    assert!(
        plan.contains("CrossColumnCompoundFtsScorer"),
        "expected the cross-column compound scorer path, got:\n{plan}"
    );
    let base = batch_row_ids(&scan.try_into_batch().await.unwrap());
    assert!(!base.is_empty(), "cross-column query matched nothing");

    let mut scan = dataset.scan();
    scan.full_text_search(cross_column()).unwrap();
    scan.with_row_id();
    scan.with_row_id_prefilter(RowAddrMask::allow_nothing());
    assert_eq!(
        scan.try_into_batch().await.unwrap().num_rows(),
        0,
        "the cross-column scorer must not return rows the mask excludes"
    );

    let keep = base[0];
    let mut scan = dataset.scan();
    scan.full_text_search(cross_column()).unwrap();
    scan.with_row_id();
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter([keep])));
    assert_eq!(
        batch_row_ids(&scan.try_into_batch().await.unwrap()),
        vec![keep]
    );
}

#[tokio::test]
async fn row_addr_mask_fts_search_only_allowed() {
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    test_ds.make_fts_index().await.unwrap();
    // Re-append the low-i rows AFTER indexing so token "4" matches both an
    // indexed row (index prefilter path) and an unindexed one (flat FTS branch).
    test_ds.append_data_with_range(0, 10).await.unwrap();
    let ds = &test_ds.dataset;

    // Baseline: the row ids an unmasked FTS query matches.
    let mut scan = ds.scan();
    scan.full_text_search(FullTextSearchQuery::new("4".into()))
        .unwrap();
    scan.with_row_id();
    let base_ids = batch_row_ids(&scan.try_into_batch().await.unwrap());
    let base_set: BTreeSet<u64> = base_ids.iter().copied().collect();
    assert!(
        base_ids.len() >= 2,
        "expected indexed + unindexed matches for token 4, got {base_ids:?}"
    );

    // Allow only every other matching row; the mask must prefilter BM25 so the
    // result is exactly the allowed subset of the baseline matches.
    let allow: Vec<u64> = base_ids.iter().copied().step_by(2).collect();
    let allow_set: BTreeSet<u64> = allow.iter().copied().collect();

    let mut scan = ds.scan();
    scan.full_text_search(FullTextSearchQuery::new("4".into()))
        .unwrap();
    scan.with_row_id_prefilter(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
        allow.iter().copied(),
    )));
    scan.with_row_id();
    let got: BTreeSet<u64> = batch_row_ids(&scan.try_into_batch().await.unwrap())
        .into_iter()
        .collect();
    let expected: BTreeSet<u64> = base_set.intersection(&allow_set).copied().collect();
    assert_eq!(got, expected, "masked FTS must return allowed matches only");
    assert!(!got.is_empty());

    // Block every match -> empty, proving the mask actually filters FTS results
    // on both the indexed and flat branches.
    let mut scan = ds.scan();
    scan.full_text_search(FullTextSearchQuery::new("4".into()))
        .unwrap();
    scan.with_row_id_prefilter(RowAddrMask::from_block(RowAddrTreeMap::from_iter(
        base_ids.iter().copied(),
    )));
    scan.with_row_id();
    let blocked = batch_row_ids(&scan.try_into_batch().await.unwrap());
    assert!(blocked.is_empty(), "block-mask must drop all FTS matches");
}

#[rstest]
#[tokio::test]
async fn test_only_row_id(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_ds = TestVectorDataset::new(data_storage_version, false)
        .await
        .unwrap();
    let dataset = &test_ds.dataset;

    let mut scan = dataset.scan();
    scan.project::<&str>(&[]).unwrap().with_row_id();

    let batch = scan.try_into_batch().await.unwrap();

    assert_eq!(batch.num_columns(), 1);
    assert_eq!(batch.num_rows(), 400);
    let expected_schema = ArrowSchema::new(vec![ArrowField::new(ROW_ID, DataType::UInt64, true)])
        .with_metadata(dataset.schema().metadata.clone());
    assert_eq!(batch.schema().as_ref(), &expected_schema,);

    let expected_row_ids: Vec<u64> = (0..200_u64).chain((1 << 32)..((1 << 32) + 200)).collect();
    let actual_row_ids: Vec<u64> = as_primitive_array::<UInt64Type>(batch.column(0).as_ref())
        .values()
        .iter()
        .copied()
        .collect();
    assert_eq!(expected_row_ids, actual_row_ids);
}

#[tokio::test]
async fn test_filter_legacy_dataset_with_stable_row_ids() {
    let test_ds = TestVectorDataset::new(LanceFileVersion::Legacy, true)
        .await
        .unwrap();

    let batch = test_ds
        .dataset
        .scan()
        .batch_readahead(get_num_compute_intensive_cpus())
        .project(&["vec"])
        .unwrap()
        .with_row_id()
        .filter_expr(col("vec").is_not_null())
        .try_into_batch()
        .await
        .unwrap();

    let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
    let unique_row_ids = row_ids.values().iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(unique_row_ids.len(), 400);
    assert_eq!(row_ids.len(), unique_row_ids.len());
}

#[tokio::test]
async fn test_scan_unordered_with_row_id() {
    // This test doesn't make sense for v2 files, there is no way to get an out-of-order scan
    let test_ds = TestVectorDataset::new(LanceFileVersion::Legacy, false)
        .await
        .unwrap();
    let dataset = &test_ds.dataset;

    let mut scan = dataset.scan();
    scan.with_row_id();

    let ordered_batches = scan
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<RecordBatch>>()
        .await
        .unwrap();
    assert!(ordered_batches.len() > 2);
    let ordered_batch =
        concat_batches(&ordered_batches[0].schema(), ordered_batches.iter()).unwrap();

    // Attempt to get out-of-order scan, but that might take multiple attempts.
    scan.scan_in_order(false);
    for _ in 0..10 {
        let unordered_batches = scan
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();
        let unordered_batch =
            concat_batches(&unordered_batches[0].schema(), unordered_batches.iter()).unwrap();

        assert_eq!(ordered_batch.num_rows(), unordered_batch.num_rows());

        // If they aren't equal, they should be equal if we sort by row id
        if ordered_batch != unordered_batch {
            let sort_indices = sort_to_indices(&unordered_batch[ROW_ID], None, None).unwrap();

            let ordered_i = ordered_batch["i"].clone();
            let sorted_i = take::take(&unordered_batch["i"], &sort_indices, None).unwrap();

            assert_eq!(&ordered_i, &sorted_i);

            break;
        }
    }
}

#[tokio::test]
async fn test_scan_with_wildcard() {
    let data = gen_batch()
        .col("x", array::step::<Float64Type>())
        .col("y", array::step::<Float64Type>())
        .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(100))
        .await
        .unwrap();

    let check_cols = async |projection: &[&str], expected_cols: &[&str]| {
        let mut scan = data.scan();
        scan.project(projection).unwrap();
        let stream = scan.try_into_stream().await.unwrap();
        let schema = stream.schema();
        let field_names = schema.field_names();
        assert_eq!(field_names, expected_cols);
    };

    check_cols(&["*"], &["x", "y"]).await;
    check_cols(&["x", "y"], &["x", "y"]).await;
    check_cols(&["x"], &["x"]).await;
    check_cols(&["_rowid", "*"], &["_rowid", "x", "y"]).await;
    check_cols(&["*", "_rowid"], &["x", "y", "_rowid"]).await;
    check_cols(
        &["_rowid", "*", "_rowoffset"],
        &["_rowid", "x", "y", "_rowoffset"],
    )
    .await;

    let check_exprs = async |exprs: &[&str], expected_cols: &[&str]| {
        let mut scan = data.scan();
        let projection = exprs
            .iter()
            .map(|e| (e.to_string(), e.to_string()))
            .collect::<Vec<_>>();
        scan.project_with_transform(&projection).unwrap();
        let stream = scan.try_into_stream().await.unwrap();
        let schema = stream.schema();
        let field_names = schema.field_names();
        assert_eq!(field_names, expected_cols);
    };

    // Make sure we can reference * fields in exprs and add new columns
    check_exprs(&["_rowid", "*", "x * 2"], &["_rowid", "x", "y", "x * 2"]).await;

    let check_fails = |projection: &[&str]| {
        let mut scan = data.scan();
        assert!(scan.project(projection).is_err());
    };

    // Would duplicate x
    check_fails(&["x", "*"]);
    check_fails(&["_rowid", "_rowid"]);
}

#[rstest]
#[tokio::test]
async fn test_scan_order(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        true,
    )]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))],
    )
    .unwrap();

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![6, 7, 8]))],
    )
    .unwrap();

    let params = WriteParams {
        mode: WriteMode::Append,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };

    let write_batch = |batch: RecordBatch| async {
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
        Dataset::write(reader, test_uri, Some(params)).await
    };

    write_batch.clone()(batch1.clone()).await.unwrap();
    write_batch(batch2.clone()).await.unwrap();

    let dataset = Arc::new(Dataset::open(test_uri).await.unwrap());
    let fragment1 = dataset.get_fragment(0).unwrap().metadata().clone();
    let fragment2 = dataset.get_fragment(1).unwrap().metadata().clone();

    // 1 then 2
    let mut scanner = dataset.scan();
    scanner.with_fragments(vec![fragment1.clone(), fragment2.clone()]);
    let output = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(output.len(), 2);
    assert_eq!(output[0], batch1);
    assert_eq!(output[1], batch2);

    // 2 then 1
    let mut scanner = dataset.scan();
    scanner.with_fragments(vec![fragment2, fragment1]);
    let output = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(output.len(), 2);
    assert_eq!(output[0], batch2);
    assert_eq!(output[1], batch1);
}

#[rstest]
#[tokio::test]
async fn test_scan_sort(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = gen_batch()
        .col("int", array::cycle::<Int32Type>(vec![5, 4, 1, 2, 3]))
        .col(
            "str",
            array::cycle_utf8_literals(&["a", "b", "c", "e", "d"]),
        );

    let sorted_by_int = gen_batch()
        .col("int", array::cycle::<Int32Type>(vec![1, 2, 3, 4, 5]))
        .col(
            "str",
            array::cycle_utf8_literals(&["c", "e", "d", "b", "a"]),
        )
        .into_batch_rows(RowCount::from(5))
        .unwrap();

    let sorted_by_str = gen_batch()
        .col("int", array::cycle::<Int32Type>(vec![5, 4, 1, 3, 2]))
        .col(
            "str",
            array::cycle_utf8_literals(&["a", "b", "c", "d", "e"]),
        )
        .into_batch_rows(RowCount::from(5))
        .unwrap();

    Dataset::write(
        data.into_reader_rows(RowCount::from(5), BatchCount::from(1)),
        test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let dataset = Arc::new(Dataset::open(test_uri).await.unwrap());

    let batches_by_int = dataset
        .scan()
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
            "int".to_string(),
        )]))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(batches_by_int[0], sorted_by_int);

    let batches_by_str = dataset
        .scan()
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
            "str".to_string(),
        )]))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(batches_by_str[0], sorted_by_str);

    // Ensure an empty sort vec does not break anything (sorting is disabled)
    dataset
        .scan()
        .order_by(Some(vec![]))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
}

#[rstest]
#[tokio::test]
async fn test_sort_multi_columns(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;

    let data = gen_batch()
        .col("int", array::cycle::<Int32Type>(vec![5, 5, 1, 1, 3]))
        .col(
            "float",
            array::cycle::<Float32Type>(vec![7.3, -f32::NAN, f32::NAN, 4.3, f32::INFINITY]),
        );

    let sorted_by_int_then_float = gen_batch()
        .col("int", array::cycle::<Int32Type>(vec![1, 1, 3, 5, 5]))
        .col(
            "float",
            // floats should be sorted using total order so -NAN is before all and NAN is after all
            array::cycle::<Float32Type>(vec![4.3, f32::NAN, f32::INFINITY, -f32::NAN, 7.3]),
        )
        .into_batch_rows(RowCount::from(5))
        .unwrap();

    Dataset::write(
        data.into_reader_rows(RowCount::from(5), BatchCount::from(1)),
        test_uri,
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let dataset = Arc::new(Dataset::open(test_uri).await.unwrap());

    let batches_by_int_then_float = dataset
        .scan()
        .order_by(Some(vec![
            ColumnOrdering::asc_nulls_first("int".to_string()),
            ColumnOrdering::asc_nulls_first("float".to_string()),
        ]))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(batches_by_int_then_float[0], sorted_by_int_then_float);
}

async fn test_row_offset_read_helper(
    ds: &Dataset,
    scan_builder: impl FnOnce(&mut Scanner) -> &mut Scanner,
    expected_cols: &[&str],
    expected_row_offsets: &[u64],
) {
    let mut scanner = ds.scan();
    let scanner = scan_builder(&mut scanner);
    let stream = scanner.try_into_stream().await.unwrap();

    let schema = stream.schema();
    let actual_cols = schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect::<Vec<_>>();
    assert_eq!(&actual_cols, expected_cols);

    let batches = stream.try_collect::<Vec<_>>().await.unwrap();
    let batch = arrow_select::concat::concat_batches(&schema, &batches).unwrap();

    let row_offsets = batch
        .column_by_name(ROW_OFFSET)
        .unwrap()
        .as_primitive::<UInt64Type>()
        .values();
    assert_eq!(row_offsets.as_ref(), expected_row_offsets);
}

#[tokio::test]
async fn test_row_offset_read() {
    let mut ds = lance_datagen::gen_batch()
        .col("idx", array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(3))
        .await
        .unwrap();
    // [0, 1, 2], [3, 4, 5], [6, 7, 8]

    // Delete [2, 3, 4, 5, 6]
    ds.delete("idx >= 2 AND idx <= 6").await.unwrap();

    // Normal read, all columns plus row offset
    test_row_offset_read_helper(
        &ds,
        |scanner| scanner.project(&["idx", ROW_OFFSET]).unwrap(),
        &["idx", ROW_OFFSET],
        &[0, 1, 2, 3],
    )
    .await;

    // Read with row offset only
    test_row_offset_read_helper(
        &ds,
        |scanner| scanner.project(&[ROW_OFFSET]).unwrap(),
        &[ROW_OFFSET],
        &[0, 1, 2, 3],
    )
    .await;

    // Filtered read of row offset
    test_row_offset_read_helper(
        &ds,
        |scanner| {
            scanner
                .filter("idx > 1")
                .unwrap()
                .project(&[ROW_OFFSET])
                .unwrap()
        },
        &[ROW_OFFSET],
        &[2, 3],
    )
    .await;
}

#[rstest]
#[case::address_offset_only(false, false)]
#[case::address_with_id(false, true)]
#[case::stable_offset_only(true, false)]
#[case::stable_with_id(true, true)]
#[tokio::test]
async fn test_ann_row_offset_projection(
    #[case] has_stable_row_ids: bool,
    #[case] has_id_projection: bool,
) {
    let mut dataset = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col(
            "vec",
            array::cycle_vec(array::step::<Float32Type>(), Dimension::from(2)),
        )
        .into_ram_dataset_with_params(
            FragmentCount::from(3),
            FragmentRowCount::from(4),
            Some(WriteParams {
                max_rows_per_file: 4,
                enable_stable_row_ids: has_stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    let centroids = Arc::new(
        FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0, 1.0]), 2).unwrap(),
    );
    let params = VectorIndexParams::with_ivf_flat_params(
        MetricType::L2,
        IvfBuildParams::try_with_centroids(1, centroids).unwrap(),
    );
    dataset
        .create_index(&["vec"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    dataset.delete("id IN (1, 5)").await.unwrap();
    assert_eq!(dataset.get_fragments().len(), 3);

    let mut scanner = dataset.scan();
    scanner
        .nearest("vec", &Float32Array::from(vec![14.5, 15.5]), 4)
        .unwrap()
        .minimum_nprobes(1)
        .maximum_nprobes(1);
    let projection = if has_id_projection {
        vec!["id", ROW_OFFSET]
    } else {
        vec![ROW_OFFSET]
    };
    scanner.project(&projection).unwrap();
    // Keep the default distance projection and ordering: they cause the
    // physical optimizer to request statistics above AddRowOffsetExec.
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(plan.contains("ANNSubIndex"), "{plan}");
    assert!(plan.contains("AddRowOffsetExec"), "{plan}");
    let batch = scanner.try_into_batch().await.unwrap();
    let mut expected_columns = projection;
    expected_columns.push(DIST_COL);
    assert_eq!(
        batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        expected_columns
    );

    // Vectors are [2 * id, 2 * id + 1]. The exact neighbors are 7, 8, 6, 9;
    // deleting 1 and 5 shifts their logical offsets by two.
    let expected_offsets = [5, 6, 4, 7];
    let offsets = batch[ROW_OFFSET].as_primitive::<UInt64Type>().values();
    let recall = offsets
        .iter()
        .filter(|offset| expected_offsets.contains(offset))
        .count() as f64
        / expected_offsets.len() as f64;
    assert_eq!(recall, 1.0);
    assert_eq!(offsets.as_ref(), expected_offsets);
    assert_eq!(
        batch[DIST_COL].as_primitive::<Float32Type>().values(),
        &[0.5, 4.5, 12.5, 24.5]
    );
    if has_id_projection {
        assert_eq!(
            batch["id"].as_primitive::<Int32Type>().values(),
            &[7, 8, 6, 9]
        );
    }
}

#[tokio::test]
async fn test_filter_to_take_with_stable_row_ids() {
    let ds = lance_datagen::gen_batch()
        .col("idx", array::step::<Int32Type>())
        .into_ram_dataset_with_params(
            FragmentCount::from(3),
            FragmentRowCount::from(4),
            Some(WriteParams {
                max_rows_per_file: 4,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let row_addrs = [
        (u64::from(RowAddress::new_from_parts(0, 1)), 1),
        (u64::from(RowAddress::new_from_parts(1, 1)), 5),
        (u64::from(RowAddress::new_from_parts(2, 1)), 9),
    ];
    for (row_addr, expected_idx) in row_addrs {
        let batch = ds
            .scan()
            .filter(&format!("{ROW_ADDR} = {row_addr}"))
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            batch["idx"].as_primitive::<Int32Type>().values(),
            &[expected_idx]
        );
    }

    let batch = ds
        .scan()
        .filter(&format!(
            "{ROW_ADDR} IN ({}, {})",
            row_addrs[1].0, row_addrs[2].0
        ))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch["idx"].as_primitive::<Int32Type>().values(), &[5, 9]);

    let batch = ds
        .scan()
        .filter(&format!("{ROW_ADDR} = 5"))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch.num_rows(), 0);

    let batch = ds
        .scan()
        .filter(&format!("{ROW_OFFSET} IN (5, 9)"))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch["idx"].as_primitive::<Int32Type>().values(), &[5, 9]);

    // NOT IN is the complement of IN and must not be lowered into a take
    // of the listed ids. The list must be large enough that DataFusion's
    // expression simplifier does not expand it into a conjunction of
    // `!=` comparisons first (it only expands small lists) — with the
    // pre-fix code the negated InList reached the lowering and returned
    // exactly the listed rows.
    let not_in_list = (0..10)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let batch = ds
        .scan()
        .filter(&format!("{ROW_ID} NOT IN ({not_in_list})"))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch["idx"].as_primitive::<Int32Type>().values(), &[10, 11]);

    // `_rowoffset` shares the arm, and the contract that matters is that a
    // negated list never turns into a take of the listed values. It cannot
    // be answered as a scan filter either: `_rowoffset` is only reachable
    // through this lowering, so it is absent from the filterable read
    // schema and any predicate that survives to the planner is rejected.
    // Pin that it fails rather than returning the complement. Plain
    // comparisons such as `_rowoffset > 2` fail the same way, so this is
    // the column's existing limit, not something the guard introduced.
    let err = ds
        .scan()
        .filter(&format!("{ROW_OFFSET} NOT IN (0, 1, 2, 4)"))
        .unwrap()
        .try_into_batch()
        .await
        .expect_err("a negated _rowoffset list must not be answered from a take");
    assert!(
        err.to_string().contains(ROW_OFFSET),
        "the error should name the column, got: {err}"
    );
}

#[tokio::test]
async fn test_stale_row_address_does_not_follow_stable_id_after_update() {
    let ds = lance_datagen::gen_batch()
        .col("idx", array::step::<Int32Type>())
        .into_ram_dataset_with_params(
            FragmentCount::from(2),
            FragmentRowCount::from(3),
            Some(WriteParams {
                max_rows_per_file: 3,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let old_row_addr = u64::from(RowAddress::new_from_parts(0, 1));
    let ds = crate::dataset::UpdateBuilder::new(Arc::new(ds))
        .update_where("idx = 1")
        .unwrap()
        .set("idx", "101")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap()
        .new_dataset;

    let batch = ds
        .scan()
        .filter(&format!("{ROW_ADDR} = {old_row_addr}"))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch.num_rows(), 0);
}

#[tokio::test]
async fn test_filter_to_take() {
    let mut ds = lance_datagen::gen_batch()
        .col("idx", array::step::<Int32Type>())
        .col("other", array::step_custom::<Int32Type>(1000, 1))
        .into_ram_dataset(FragmentCount::from(3), FragmentRowCount::from(100))
        .await
        .unwrap();

    let row_ids = ds
        .scan()
        .project(&Vec::<&str>::default())
        .unwrap()
        .with_row_id()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let schema = row_ids[0].schema();
    let row_ids = concat_batches(&schema, row_ids.iter()).unwrap();
    let row_ids = row_ids.column(0).as_primitive::<UInt64Type>().clone();

    let row_addrs = ds
        .scan()
        .project(&Vec::<&str>::default())
        .unwrap()
        .with_row_address()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let schema = row_addrs[0].schema();
    let row_addrs = concat_batches(&schema, row_addrs.iter()).unwrap();
    let row_addrs = row_addrs.column(0).as_primitive::<UInt64Type>().clone();

    ds.delete("idx >= 190 AND idx < 210").await.unwrap();

    let ds_copy = ds.clone();
    let do_check = async move |filt: &str, expected_idx: &[i32], applies_optimization: bool| {
        let mut scanner = ds_copy.scan();
        scanner.filter(filt).unwrap();
        // Verify the optimization is applied
        let plan = scanner.explain_plan(true).await.unwrap();
        if applies_optimization {
            assert!(
                plan.contains("OneShotStream"),
                "expected take optimization to be applied. Filter: '{}'.  Plan:\n{}",
                filt,
                plan
            );
        } else {
            assert!(
                !plan.contains("OneShotStream"),
                "expected take optimization to not be applied. Filter: '{}'.  Plan:\n{}",
                filt,
                plan
            );
        }

        // Verify the results
        let stream = scanner.try_into_stream().await.unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let idx = batches
            .iter()
            .map(|b| b.column_by_name("idx").unwrap().as_ref())
            .collect::<Vec<_>>();

        if idx.is_empty() {
            assert!(expected_idx.is_empty());
            return;
        }

        let idx = arrow::compute::concat(&idx).unwrap();
        assert_eq!(idx.as_primitive::<Int32Type>().values(), expected_idx);
    };
    let check = async |filt: &str, expected_idx: &[i32]| do_check(filt, expected_idx, true).await;
    let check_no_opt = async |filt: &str, expected_idx: &[i32]| {
        do_check(filt, expected_idx, false).await;
    };

    // Simple case, no deletions yet
    check("_rowid = 50", &[50]).await;
    check("_rowaddr = 50", &[50]).await;
    check("_rowoffset = 50", &[50]).await;

    check(
        "_rowid = 50 OR _rowid = 51 OR _rowid = 52 OR _rowid = 49",
        &[49, 50, 51, 52],
    )
    .await;
    check(
        "_rowaddr = 50 OR _rowaddr = 51 OR _rowaddr = 52 OR _rowaddr = 49",
        &[49, 50, 51, 52],
    )
    .await;
    check(
        "_rowoffset = 50 OR _rowoffset = 51 OR _rowoffset = 52 OR _rowoffset = 49",
        &[49, 50, 51, 52],
    )
    .await;

    check("_rowid IN (52, 51, 50, 17)", &[17, 50, 51, 52]).await;
    check("_rowaddr IN (52, 51, 50, 17)", &[17, 50, 51, 52]).await;
    check("_rowoffset IN (52, 51, 50, 17)", &[17, 50, 51, 52]).await;

    // Taking _rowid / _rowaddr of deleted row

    // When using rowid / rowaddr we get an empty
    check(&format!("_rowid = {}", row_ids.value(190)), &[]).await;
    check(&format!("_rowaddr = {}", row_addrs.value(190)), &[]).await;
    // When using rowoffset it just skips the deleted rows (impossible to create an offset
    // into a deleted row)
    check("_rowoffset = 190", &[210]).await;

    // Grabbing after the deleted rows
    check(&format!("_rowid = {}", row_ids.value(250)), &[250]).await;
    check(&format!("_rowaddr = {}", row_addrs.value(250)), &[250]).await;
    check("_rowoffset = 250", &[270]).await;

    // Grabbing past the end
    check("_rowoffset = 1000", &[]).await;

    // Combine take and filter
    check("_rowid IN (5, 10, 15) AND idx > 10", &[15]).await;
    check("_rowaddr IN (5, 10, 15) AND idx > 10", &[15]).await;
    check("_rowoffset IN (5, 10, 15) AND idx > 10", &[15]).await;
    check("idx > 10 AND _rowid IN (5, 10, 15)", &[15]).await;
    check("idx > 10 AND _rowaddr IN (5, 10, 15)", &[15]).await;
    check("idx > 10 AND _rowoffset IN (5, 10, 15)", &[15]).await;
    // Get's simplified into _rowid = 50 and so we catch it
    check("_rowid = 50 AND _rowid = 50", &[50]).await;

    // Filters that cannot be converted into a take
    check_no_opt("_rowid = 50 AND _rowid = 51", &[]).await;
    check_no_opt("(_rowid = 50 AND idx < 100) OR _rowid = 51", &[50, 51]).await;

    // The take's row count is inexact, so the plan must keep the remainder
    // filter from splitting the take's batches across partitions
    let mut scanner = ds.scan();
    scanner
        .target_parallelism(4)
        .filter(&format!(
            "_rowid IN ({}, {}, {}) AND idx >= 0",
            row_ids.value(250),
            row_ids.value(101),
            row_ids.value(1)
        ))
        .unwrap();
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(!plan.contains("RepartitionExec"), "{plan}");
    let batch = scanner.try_into_batch().await.unwrap();
    assert_eq!(
        batch["idx"].as_primitive::<Int32Type>().values(),
        &[1, 101, 250]
    );

    // Dynamic projection
    let mut scanner = ds.scan();
    scanner.filter("_rowoffset = 77").unwrap();
    scanner
        .project_with_transform(&[("foo", "idx * 2")])
        .unwrap();
    let stream = scanner.try_into_stream().await.unwrap();
    let batches = stream.try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(batches[0].schema().field(0).name(), "foo");
    let val = batches[0].column(0).as_primitive::<Int32Type>().values()[0];
    assert_eq!(val, 154);

    // The remainder filter reads `idx`, which the projection leaves out
    let mut scanner = ds.scan();
    scanner
        .project(&["other"])
        .unwrap()
        .filter("_rowid IN (5, 10, 15) AND idx > 10")
        .unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    assert_eq!(batch.schema().fields().len(), 1, "{:?}", batch.schema());
    assert_eq!(batch["other"].as_primitive::<Int32Type>().values(), &[1015]);

    // A count reads no data columns, and the take keeps the scanner's batch size
    let mut scanner = ds.scan();
    scanner
        .batch_size(1)
        .filter("_rowid IN (5, 10, 15)")
        .unwrap();
    let plan = scanner.analyze_count_plan().await.unwrap();
    assert!(plan.contains("projection=[], "), "{plan}");
    assert_eq!(scanner.count_rows().await.unwrap(), 3);
    let batches = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(
        batches
            .iter()
            .map(|batch| batch.num_rows())
            .collect::<Vec<_>>(),
        vec![1, 1, 1]
    );
}

#[tokio::test]
async fn test_filter_not_in_is_not_inverted() {
    let ds = lance_datagen::gen_batch()
        .col("idx", array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(1), FragmentRowCount::from(10))
        .await
        .unwrap();

    let ds_copy = ds.clone();
    let check = async move |filt: &str, expected_idx: &[i32]| {
        let mut scanner = ds_copy.scan();
        scanner.filter(filt).unwrap();
        let batch = scanner.try_into_batch().await.unwrap();
        let idx = batch.column_by_name("idx").unwrap();
        assert_eq!(
            idx.as_primitive::<Int32Type>().values(),
            expected_idx,
            "wrong rows for filter '{filt}'"
        );
    };

    // Single fragment, so _rowid == _rowaddr == idx.
    let complement: &[i32] = &[0, 1, 3, 5, 7, 9];
    check("_rowid NOT IN (2, 4, 6, 8)", complement).await;
    check("_rowaddr NOT IN (2, 4, 6, 8)", complement).await;
}

#[tokio::test]
async fn test_nested_field_ordering() {
    use arrow_array::StructArray;

    // Create test data with nested structs
    let id_array = Int32Array::from(vec![3, 1, 2]);
    let nested_values = Int32Array::from(vec![30, 10, 20]);
    let nested_struct = StructArray::from(vec![(
        Arc::new(ArrowField::new("value", DataType::Int32, false)),
        Arc::new(nested_values) as ArrayRef,
    )]);

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new(
            "nested",
            DataType::Struct(vec![ArrowField::new("value", DataType::Int32, false)].into()),
            false,
        ),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(id_array), Arc::new(nested_struct)],
    )
    .unwrap();

    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone());

    let dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    // Test ordering by nested field
    let mut scanner = dataset.scan();
    scanner
        .order_by(Some(vec![ColumnOrdering {
            column_name: "nested.value".to_string(),
            ascending: true,
            nulls_first: true,
        }]))
        .unwrap(); // ascending order

    let stream = scanner.try_into_stream().await.unwrap();
    let batches = stream.try_collect::<Vec<_>>().await.unwrap();

    // Check that results are sorted by nested.value
    let sorted_ids = batches[0].column(0).as_primitive::<Int32Type>().values();
    assert_eq!(sorted_ids[0], 1); // id=1 has nested.value=10
    assert_eq!(sorted_ids[1], 2); // id=2 has nested.value=20
    assert_eq!(sorted_ids[2], 3); // id=3 has nested.value=30
}

#[tokio::test]
async fn test_limit_with_ordering_not_pushed_down() {
    // This test verifies the fix for a bug where limit/offset could be pushed down
    // even when ordering was specified. When ordering is present, we need to load
    // all data first to sort it before applying limits.

    // Create test data with specific ordering
    let id_array = Int32Array::from(vec![5, 2, 8, 1, 3, 7, 4, 6]);
    let value_array = Int32Array::from(vec![50, 20, 80, 10, 30, 70, 40, 60]);

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("value", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(id_array), Arc::new(value_array)],
    )
    .unwrap();

    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone());

    let dataset = Dataset::write(reader, test_uri, None).await.unwrap();

    // Test 1: limit with ordering should return top N after sorting
    let mut scanner = dataset.scan();
    scanner
        .order_by(Some(vec![ColumnOrdering {
            column_name: "value".to_string(),
            ascending: true,
            nulls_first: true,
        }]))
        .unwrap();
    scanner.limit(Some(3), None).unwrap();

    let stream = scanner.try_into_stream().await.unwrap();
    let batches = stream.try_collect::<Vec<_>>().await.unwrap();

    // Results should be sorted by value and limited to 3
    let sorted_ids = batches[0].column(0).as_primitive::<Int32Type>().values();
    let sorted_values = batches[0].column(1).as_primitive::<Int32Type>().values();
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(sorted_ids[0], 1); // value=10
    assert_eq!(sorted_ids[1], 2); // value=20
    assert_eq!(sorted_ids[2], 3); // value=30
    assert_eq!(sorted_values[0], 10);
    assert_eq!(sorted_values[1], 20);
    assert_eq!(sorted_values[2], 30);

    // Test 2: offset with ordering should skip first N after sorting
    let mut scanner = dataset.scan();
    scanner
        .order_by(Some(vec![ColumnOrdering {
            column_name: "value".to_string(),
            ascending: true,
            nulls_first: true,
        }]))
        .unwrap();
    scanner.limit(Some(3), Some(2)).unwrap(); // Skip first 2, take next 3

    let stream = scanner.try_into_stream().await.unwrap();
    let batches = stream.try_collect::<Vec<_>>().await.unwrap();

    let sorted_ids = batches[0].column(0).as_primitive::<Int32Type>().values();
    let sorted_values = batches[0].column(1).as_primitive::<Int32Type>().values();
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(sorted_ids[0], 3); // value=30 (skipped 10, 20)
    assert_eq!(sorted_ids[1], 4); // value=40
    assert_eq!(sorted_ids[2], 5); // value=50
    assert_eq!(sorted_values[0], 30);
    assert_eq!(sorted_values[1], 40);
    assert_eq!(sorted_values[2], 50);

    // Test 3: without ordering, limit can be pushed down (different behavior)
    let mut scanner = dataset.scan();
    scanner.limit(Some(3), None).unwrap();

    let stream = scanner.try_into_stream().await.unwrap();
    let batches = stream.try_collect::<Vec<_>>().await.unwrap();

    // Should get first 3 rows in storage order (not sorted)
    assert_eq!(batches[0].num_rows(), 3);
    let unsorted_values = batches[0].column(1).as_primitive::<Int32Type>().values();
    // These will be in original insertion order, not sorted
    assert_eq!(unsorted_values[0], 50);
    assert_eq!(unsorted_values[1], 20);
    assert_eq!(unsorted_values[2], 80);
}

#[tokio::test]
async fn test_scan_with_version_columns() {
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};

    // Create a simple dataset
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        DataType::Int32,
        false,
    )]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();

    let test_dir = lance_core::utils::tempfile::TempStrDir::default();
    let test_uri = test_dir.as_str();

    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let write_params = WriteParams {
        enable_stable_row_ids: true,
        ..Default::default()
    };
    Dataset::write(reader, test_uri, Some(write_params))
        .await
        .unwrap();

    let dataset = Dataset::open(test_uri).await.unwrap();
    let mut scanner = dataset.scan();

    scanner
        .project(&[ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION])
        .unwrap();

    // Check that the schema includes version columns
    let output_schema = scanner.schema().await.unwrap();
    assert!(
        output_schema
            .column_with_name("_row_last_updated_at_version")
            .is_some(),
        "Schema should include _row_last_updated_at_version"
    );
    assert!(
        output_schema
            .column_with_name("_row_created_at_version")
            .is_some(),
        "Schema should include _row_created_at_version"
    );

    // Actually read the data to ensure version columns are materialized
    let batches = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    assert_eq!(batches.len(), 1);
    let batch = &batches[0];

    // Verify version columns exist in the output
    let last_updated = batch
        .column_by_name("_row_last_updated_at_version")
        .expect("Should have _row_last_updated_at_version column");
    let created_at = batch
        .column_by_name("_row_created_at_version")
        .expect("Should have _row_created_at_version column");

    // Verify they have the correct values (all rows created in version 1)
    let last_updated_array = last_updated
        .as_any()
        .downcast_ref::<arrow_array::UInt64Array>()
        .unwrap();
    let created_at_array = created_at
        .as_any()
        .downcast_ref::<arrow_array::UInt64Array>()
        .unwrap();

    for i in 0..batch.num_rows() {
        assert_eq!(
            last_updated_array.value(i),
            1,
            "All rows last updated at version 1"
        );
        assert_eq!(
            created_at_array.value(i),
            1,
            "All rows created at version 1"
        );
    }
}
