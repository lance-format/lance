// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use std::collections::HashMap;

use arrow::datatypes::{Float32Type, Int32Type};
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
use lance_index::scalar::BuiltinIndexType;
use lance_linalg::distance::DistanceType;
use rstest::rstest;

use crate::dataset::transaction::{Operation, Transaction};
use crate::index::DatasetIndexExt;
use crate::index::segment::index_type_is_known;
use crate::index::vector::VectorIndexParams;
use crate::utils::test::copy_test_data_to_tmp;

#[rstest]
#[case::missing_details("v0.18.2/index_no_details.lance", None)]
#[case::v036(
    "0.36.0/btree_in_index_pkg.lance",
    Some("/lance.index.pb.BTreeIndexDetails")
)]
#[tokio::test]
async fn test_historical_scalar_indices_support_remapping(
    #[case] fixture: &str,
    #[case] type_url: Option<&str>,
) {
    let test_dir = copy_test_data_to_tmp(fixture).unwrap();
    let dataset = Dataset::open(&test_dir.path_str()).await.unwrap();
    let indices = dataset.load_indices().await.unwrap();
    assert!(!indices.is_empty());

    for index in indices.iter() {
        assert_eq!(
            index
                .index_details
                .as_ref()
                .map(|details| details.type_url.as_str()),
            type_url
        );
        if let Some(details) = &index.index_details {
            assert_eq!(
                IndexDetails(details.clone()).get_plugin().unwrap().name(),
                "BTree"
            );
        }
        assert!(index_type_is_known(index));
        let result = remap_index(
            &dataset,
            &index.uuid,
            &RowAddrTranslator::sync(RowAddrRemap::empty()),
            None,
        )
        .await
        .unwrap();
        assert!(matches!(result, RemapResult::Remapped(_)), "{result:?}");
    }
}

#[tokio::test]
async fn test_remap_empty_chain() {
    let data = gen_batch()
        .col("int", array::step::<Int32Type>())
        .col(
            "vector",
            array::rand_vec::<Float32Type>(Dimension::from(16)),
        )
        .into_reader_rows(RowCount::from(256), BatchCount::from(1));
    let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();

    let params = VectorIndexParams::ivf_pq(1, 8, 1, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    let index_uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let row_count = dataset.count_all_rows().await.unwrap();
    let first_half = (0..row_count / 2)
        .map(|i| (i as u64, None))
        .collect::<HashMap<_, _>>();
    let second_half = (row_count / 2..row_count)
        .map(|i| (i as u64, None))
        .collect::<HashMap<_, _>>();
    let remap_to_empty = RowAddrRemap::chained([
        RowAddrRemap::direct(first_half),
        RowAddrRemap::direct(second_half),
    ]);
    let new_uuid = remap_index(
        &dataset,
        &index_uuid,
        &RowAddrTranslator::sync(remap_to_empty.clone()),
        None,
    )
    .await
    .unwrap();
    assert_eq!(new_uuid, RemapResult::Keep(index_uuid));
}

#[tokio::test]
async fn test_remap_partial_map_does_not_keep_index() {
    let data = gen_batch()
        .col("int", array::step::<Int32Type>())
        .col(
            "vector",
            array::rand_vec::<Float32Type>(Dimension::from(16)),
        )
        .into_reader_rows(RowCount::from(256), BatchCount::from(1));
    let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();

    let params = VectorIndexParams::ivf_pq(1, 8, 1, DistanceType::L2, 1);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await
        .unwrap();

    let index_meta = dataset.load_indices().await.unwrap()[0].clone();
    assert_eq!(dataset.count_all_rows().await.unwrap(), 256);
    assert_eq!(
        index_meta.fragment_bitmap,
        Some(RoaringBitmap::from_iter([0u32]))
    );

    let remap = RowAddrRemap::direct(HashMap::from([(0u64, None)]));
    assert_eq!(remap.get(1), None);

    let result = remap_index(
        &dataset,
        &index_meta.uuid,
        &RowAddrTranslator::sync(remap.clone()),
        None,
    )
    .await
    .unwrap();
    assert_ne!(result, RemapResult::Keep(index_meta.uuid));

    let complete_remap = RowAddrRemap::direct((0u64..256).map(|offset| (offset, None)).collect());
    let mut legacy_dataset = dataset.clone();
    let manifest = Arc::make_mut(&mut legacy_dataset.manifest);
    Arc::make_mut(&mut manifest.fragments)[0].physical_rows = None;
    assert!(
        !remap_deletes_all_indexed_rows(
            &legacy_dataset,
            index_meta.fragment_bitmap.as_ref().unwrap(),
            &RowAddrTranslator::sync(complete_remap.clone()),
        )
        .await
        .unwrap()
    );
}

/// The `fields.len() > 1` rejection in `remap_index`, which had no dedicated
/// coverage. A covered index never reaches it: the withdrawal above returns
/// `RemapResult::Drop` first, which is what
/// `test_compaction_withdraws_a_covered_index_without_failing` in
/// `dataset/tests/dataset_index.rs` exercises, and
/// `test_remap_column_index_refuses_a_covered_index` in
/// `dataset/optimize/remapping.rs` never reaches `remap_index` at all since
/// `remap_column_index` refuses first. So only a genuinely composite index --
/// two keyed fields, no covering declaration -- arrives here, and it must
/// still be rejected.
#[tokio::test]
async fn test_remap_index_rejects_composite_index() {
    let data = gen_batch()
        .col("a", array::step::<Int32Type>())
        .col("b", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(20), BatchCount::from(1));
    let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();
    dataset
        .create_index(
            &["a"],
            IndexType::BTree,
            None,
            &lance_index::scalar::ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();

    let a_id = dataset.schema().field("a").unwrap().id;
    let b_id = dataset.schema().field("b").unwrap().id;
    let current = dataset.load_indices().await.unwrap();
    let mut composite = current[0].clone();
    // No covering declaration at all -- genuinely composite, not covered.
    composite.fields = vec![a_id, b_id];
    composite.covering_fields = vec![];

    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![composite],
            removed_indices: current.to_vec(),
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();

    let index_uuid = dataset.load_indices().await.unwrap()[0].uuid;
    let error = remap_index(
        &dataset,
        &index_uuid,
        &RowAddrTranslator::sync(RowAddrRemap::empty()),
        None,
    )
    .await
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("only one keyed field is supported"),
        "{error}"
    );
}

/// Decoding a manifest validates the declaration, so `matched` cannot be
/// malformed by that route -- but it is not the only route: metadata
/// constructed in-process (segment conversion, a distributed build's output)
/// reaches this guard without ever passing through
/// `TryFrom<pb::IndexMetadata>`. This guard is therefore load-bearing on its
/// own, and must produce a named error rather than either a panic (plain
/// subtraction on `usize` underflows) or a silent
/// saturation that lets corrupt metadata through as an ordinary covered index
/// and gets it quietly withdrawn. The test reaches it by seeding the index
/// cache, the same way the migration-path test in `dataset/index.rs` does.
#[tokio::test]
async fn test_remap_index_rejects_malformed_covering_fields() {
    let data = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(20), BatchCount::from(1));
    let mut dataset = Dataset::write(data, "memory://", None).await.unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            None,
            &lance_index::scalar::ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();

    let index_uuid = dataset.load_indices().await.unwrap()[0].uuid;

    // Malform the cached metadata directly: more carried fields than fields
    // at all. No commit and no manifest decode can produce this -- both run
    // `validate_covering_fields` -- which is exactly why the guard has to
    // stand on its own for metadata that reached it by neither route.
    let mut indices = dataset.load_indices().await.unwrap().as_ref().clone();
    for idx in &mut indices {
        if idx.uuid == index_uuid {
            idx.covering_fields = idx
                .fields
                .iter()
                .copied()
                .chain(std::iter::once(999))
                .collect();
        }
    }
    assert!(
        indices
            .iter()
            .any(|idx| idx.uuid == index_uuid && idx.covering_fields.len() > idx.fields.len()),
        "test setup should have produced a malformed entry"
    );
    let metadata_key = crate::session::index_caches::IndexMetadataKey {
        version: dataset.version().version,
        store_identity: &dataset.object_store.store_prefix,
        e_tag: dataset.manifest_location.e_tag.as_deref(),
    };
    dataset
        .index_cache
        .insert_with_key(&metadata_key, Arc::new(indices))
        .await;

    // The validation runs first in `remap_index`, ahead of the withdrawal and
    // of every row-map-dependent branch, so an empty remap reaches it.
    let error = remap_index(
        &dataset,
        &index_uuid,
        &RowAddrTranslator::sync(RowAddrRemap::empty()),
        None,
    )
    .await
    .expect_err("malformed covering metadata must not be silently accepted");
    assert!(
        error.to_string().contains("are not among its fields"),
        "expected the validator's message, got: {error}"
    );
}
