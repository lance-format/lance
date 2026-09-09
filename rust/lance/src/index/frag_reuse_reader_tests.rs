// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::tests::{fixture, fixture_with_index, install, persist_fixture, prepare};
use super::*;
use crate::dataset::WriteParams;
use crate::index::create::CreateIndexBuilder;
use crate::index::frag_reuse_remapping::vector_supports_batch_remapping;
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use crate::session::index_caches::IndexMetadataKey;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
use arrow_array::types::Int32Type;
use arrow_array::{RecordBatch, RecordBatchIterator, cast::AsArray};
#[cfg(feature = "geo")]
use geo_types::line_string;
#[cfg(feature = "geo")]
use geoarrow_array::{GeoArrowArray, builder::LineStringBuilder};
#[cfg(feature = "geo")]
use geoarrow_schema::{Dimension, LineStringType};
use lance_index::IndexType;
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::ScalarIndexParams;
use lance_table::format::pb;
use lance_table::format::pb::fragment_reuse_index_details::{
    FragmentDigest, InlineContent, StablePartition, Transition, transition,
};
use lance_table::system_index::frag_reuse::ledger::Mapping;
use prost::Message;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

#[rstest::rstest]
#[case::complete("complete")]
#[case::missing_segment("missing")]
#[case::unsupported_version("version")]
#[case::unsupported_async_plugin("plugin")]
#[tokio::test]
async fn destination_coverage_requires_every_contributing_segment(#[case] scenario: &str) {
    let mut dataset = fixture().await;
    let params = ScalarIndexParams::default();
    let fragments: Vec<_> = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    let mut segments = Vec::new();
    for fragment in &fragments {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["i"], IndexType::BTree, &params)
                .name("i_idx".into())
                .replace(true)
                .fragments(vec![*fragment])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    dataset
        .commit_existing_index_segments("i_idx", "i", segments)
        .await
        .unwrap();
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    let mut indices = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    let second = indices
        .iter()
        .position(|index| {
            index.name == "i_idx"
                && index
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|bitmap| bitmap.contains(fragments[1]))
        })
        .unwrap();
    match scenario {
        "missing" => {
            indices.remove(second);
        }
        "version" => {
            indices[second].index_version = i32::MAX;
        }
        "plugin" => {
            // A segment requiring an unsupported consumer must not be opened or
            // counted toward the logical index's destination coverage.
            indices[second].index_details = Some(Arc::new(
                prost_types::Any::from_msg(&lance_index::pb::FmIndexDetails::default()).unwrap(),
            ));
        }
        "complete" => {}
        _ => unreachable!(),
    }
    persist_fixture(&mut dataset, indices).await;
    let complete = scenario == "complete";
    let usable = crate::index::scalar_logical::load_named_scalar_segments(&dataset, "i", "i_idx")
        .await
        .unwrap();
    assert_eq!(usable.len(), if complete { 2 } else { 0 });
    for segment in &usable {
        assert_eq!(
            segment.fragment_bitmap.as_ref().unwrap(),
            dataset.fragment_bitmap.as_ref()
        );
    }
    for value in 0..8 {
        let mut scan = dataset.scan();
        scan.filter(&format!("i = {value}")).unwrap();
        let plan = scan.explain_plan(false).await.unwrap();
        assert_eq!(plan.contains("ScalarIndexQuery"), complete, "{plan}");
        let batch = scan.try_into_batch().await.unwrap();
        assert_eq!(
            batch.num_rows(),
            1,
            "missing or duplicated row {value}: {plan}"
        );
        assert_eq!(batch["i"].as_primitive::<Int32Type>().value(0), value);
    }
}

#[tokio::test]
async fn projected_coverage_does_not_skip_address_translation() {
    let mut dataset = fixture().await;
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    let mut indices = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    let index = indices
        .iter_mut()
        .find(|index| index.name == "i_idx")
        .unwrap();
    index.fragment_bitmap = Some(dataset.fragment_bitmap.as_ref().clone());
    let key = IndexMetadataKey {
        version: dataset.manifest.version,
        store_identity: &dataset.object_store.store_prefix,
        e_tag: dataset.manifest_location.e_tag.as_deref(),
    };
    dataset
        .index_cache
        .insert_with_key(&key, Arc::new(indices))
        .await;
    for value in 0..8 {
        assert_eq!(
            dataset
                .count_rows(Some(format!("i = {value}")))
                .await
                .unwrap(),
            1
        );
    }
}

#[rstest::rstest]
#[case::inline(false)]
#[case::external(true)]
#[tokio::test]
async fn scalar_queries_translate_deleted_sources_and_lazy_blocks(
    #[case] external: bool,
    #[values(
        IndexType::BTree,
        IndexType::Bitmap,
        IndexType::ZoneMap,
        IndexType::BloomFilter
    )]
    index_type: IndexType,
) {
    let mut dataset = fixture_with_index(index_type).await;
    dataset.delete("i = 3").await.unwrap();
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    let fri = install(&mut dataset, content, destinations, external).await;
    let mapping = FragmentReuseIndex::open(&dataset, &fri).await.unwrap();

    let index = dataset.load_index_by_name("i_idx").await.unwrap().unwrap();
    assert_eq!(
        index.fragment_bitmap.as_ref().unwrap(),
        dataset.fragment_bitmap.as_ref()
    );

    let plan = dataset
        .scan()
        .filter("i = 2")
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
    for value in 0..8 {
        assert_eq!(
            dataset
                .count_rows(Some(format!("i = {value}")))
                .await
                .unwrap(),
            usize::from(value != 3)
        );
    }

    let metrics = lance_index::metrics::LocalMetricsCollector::default();
    crate::index::scalar::open_scalar_index(&dataset, "i", &index, &metrics)
        .await
        .unwrap();
    assert_eq!(
        metrics
            .index_loads
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    let mut other_snapshot = dataset.clone();
    other_snapshot.manifest_location.path =
        dataset.base.clone().join("_versions").join("999.manifest");
    crate::index::scalar::open_scalar_index(&other_snapshot, "i", &index, &metrics)
        .await
        .unwrap();
    assert_eq!(
        metrics
            .index_loads
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    let inputs = [
        RowAddress::new_from_parts(1, 3),
        RowAddress::new_from_parts(0, 2),
        RowAddress::new_from_parts(0, 2),
        RowAddress::new_from_parts(0, 3),
    ];
    assert_eq!(
        mapping.translate(&inputs).await.unwrap(),
        vec![
            Some(RowAddress::new_from_parts(11, 2)),
            Some(RowAddress::new_from_parts(10, 1)),
            Some(RowAddress::new_from_parts(10, 1)),
            None
        ]
    );
}

#[rstest::rstest]
#[case::ngram(IndexType::NGram)]
#[case::inverted(IndexType::Inverted)]
#[tokio::test]
async fn text_indices_translate_through_the_shared_remapper(#[case] index_type: IndexType) {
    let batch = arrow_array::record_batch!(
        ("i", Int32, [0, 1, 2, 3, 4, 5, 6, 7]),
        (
            "text",
            Utf8,
            [
                Some("even"),
                Some("odd"),
                Some("even"),
                Some("odd"),
                None,
                Some("odd"),
                Some("even"),
                Some("odd")
            ]
        )
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    if index_type == IndexType::Inverted {
        dataset
            .create_index(
                &["text"],
                index_type,
                Some("text_idx".into()),
                &lance_index::scalar::InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();
    } else {
        dataset
            .create_index(
                &["text"],
                index_type,
                Some("text_idx".into()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
    }
    dataset.delete("i = 3").await.unwrap();
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    let index = dataset
        .load_index_by_name("text_idx")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        index.fragment_bitmap.as_ref().unwrap(),
        dataset.fragment_bitmap.as_ref()
    );
    for _ in 0..2 {
        let mut scan = dataset.scan();
        if index_type == IndexType::Inverted {
            scan.full_text_search(lance_index::scalar::FullTextSearchQuery::new("even".into()))
                .unwrap();
        } else {
            scan.filter("contains(text, 'even')").unwrap();
            let plan = scan.explain_plan(false).await.unwrap();
            assert!(plan.contains("ScalarIndexQuery"), "{plan}");
        }
        let result = scan.try_into_batch().await.unwrap();
        let actual = result
            .column_by_name("i")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(actual, std::collections::BTreeSet::from([0, 2, 6]));
    }
}

#[tokio::test]
async fn label_list_translates_values_and_null_rows() {
    let labels = arrow_array::ListArray::from_iter_primitive::<arrow_array::types::Int64Type, _, _>(
        (0..8).map(|i| {
            if i == 4 {
                None
            } else {
                Some(vec![Some(i % 2)])
            }
        }),
    );
    let batch = RecordBatch::try_from_iter([
        (
            "i",
            Arc::new(arrow_array::Int32Array::from_iter_values(0..8)) as arrow_array::ArrayRef,
        ),
        ("labels", Arc::new(labels) as arrow_array::ArrayRef),
    ])
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["labels"],
            IndexType::LabelList,
            Some("labels_idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    dataset.delete("i = 3").await.unwrap();
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    for (predicate, expected) in [
        ("array_has_any(labels, [0])", vec![0, 2, 6]),
        ("NOT array_has_any(labels, [0])", vec![1, 5, 7]),
    ] {
        let mut scan = dataset.scan();
        scan.filter(predicate).unwrap();
        let plan = scan.explain_plan(false).await.unwrap();
        assert!(plan.contains("ScalarIndexQuery"), "{plan}");
        let result = scan.try_into_batch().await.unwrap();
        let actual = result
            .column_by_name("i")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(actual, expected.into_iter().collect());
    }
}

#[cfg(feature = "geo")]
#[tokio::test]
async fn rtree_queries_translate_partition_addresses() {
    let geometry_type = LineStringType::new(Dimension::XY, Default::default());
    let mut geometry = LineStringBuilder::new(geometry_type.clone());
    for i in 0..8 {
        let line = line_string![(x: i as f64, y: 0.0), (x: i as f64, y: 1.0)];
        geometry
            .push_line_string((i != 7).then_some(&line))
            .unwrap();
    }
    let batch = RecordBatch::try_new(
        Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("i", arrow_schema::DataType::Int32, false),
            geometry_type.to_field("geometry", true),
        ])),
        vec![
            Arc::new(arrow_array::Int32Array::from_iter_values(0..8)),
            geometry.finish().to_array_ref(),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["geometry"],
            IndexType::RTree,
            Some("geometry_idx".into()),
            &ScalarIndexParams::new("RTree".into()),
            true,
        )
        .await
        .unwrap();
    dataset.delete("i = 1").await.unwrap();
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    let sql = "SELECT i FROM dataset WHERE ST_Intersects(geometry, ST_GeomFromText('LINESTRING (0 0.5, 6 0.5)')) ORDER BY i";
    let batches = dataset
        .sql(sql)
        .build()
        .await
        .unwrap()
        .into_batch_records()
        .await
        .unwrap();
    let ids: Vec<_> = batches
        .iter()
        .flat_map(|batch| {
            batch["i"]
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(ids, vec![0, 2, 3, 4, 5, 6]);
    let plan = dataset
        .sql(&format!("EXPLAIN {sql}"))
        .build()
        .await
        .unwrap()
        .into_batch_records()
        .await
        .unwrap();
    let plan = arrow::util::pretty::pretty_format_batches(&plan)
        .unwrap()
        .to_string();
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
    assert_eq!(
        dataset
            .count_rows(Some("geometry IS NULL".into()))
            .await
            .unwrap(),
        1
    );
}

async fn assert_legacy_metadata(dataset: &Dataset) -> Option<IndexMetadata> {
    let flag = lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
    assert_eq!(dataset.manifest.reader_feature_flags & flag, 0);
    assert_eq!(dataset.manifest.writer_feature_flags & flag, 0);
    let indices = crate::index::load_all_indices(dataset).await.unwrap();
    let fri = indices
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME);
    for index in indices
        .iter()
        .filter(|index| index.name != FRAG_REUSE_INDEX_NAME)
    {
        let resolved =
            super::super::frag_reuse::open_row_id_remapping(dataset, index, &NoOpMetricsCollector)
                .await
                .unwrap();
        if let Some(fri) = fri {
            assert_eq!(fri.index_version, 0);
            let (uuid, remapping) = resolved.unwrap();
            assert_eq!(uuid, fri.uuid);
            let lance_index::scalar::RowIdRemapping::InMemory(remapper) = remapping else {
                panic!("V1 must use the synchronous remapper");
            };
            let legacy = dataset
                .open_frag_reuse_index(&NoOpMetricsCollector)
                .await
                .unwrap()
                .unwrap();
            for version in &legacy.details.versions {
                for group in &version.groups {
                    for source in &group.old_frags {
                        for offset in 0..source.physical_rows {
                            let address =
                                RowAddress::new_from_parts(source.id as u32, offset as u32).into();
                            assert_eq!(
                                remapper.remap_row_id(address),
                                legacy.remap_row_id(address)
                            );
                        }
                    }
                }
            }
        } else {
            assert!(resolved.is_none());
        }
    }
    fri.cloned()
}

// Exercise external history with the exact InlineContent bytes emitted by
// the legacy writer, without manufacturing a 200-KB history in every test.
async fn externalize_legacy_history(dataset: &mut Dataset) {
    let mut indices = crate::index::load_all_indices(dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    let fri = indices
        .iter_mut()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
        .unwrap();
    let mut details: pb::FragmentReuseIndexDetails =
        fri.index_details.as_ref().unwrap().to_msg().unwrap();
    let Some(pb::fragment_reuse_index_details::Content::Inline(content)) = details.content.take()
    else {
        panic!("expected inline legacy writer output");
    };
    assert!(content.transitions.is_empty());
    let bytes = content.encode_to_vec();
    // A fresh identity prevents a previously opened inline history from
    // satisfying this test through the shared session cache.
    fri.uuid = Uuid::new_v4();
    let name = "legacy-external.binpb";
    let path = dataset.indices_dir().join(fri.uuid.to_string()).join(name);
    let mut writer = dataset.object_store.create(&path).await.unwrap();
    writer.write_all(&bytes).await.unwrap();
    writer.shutdown().await.unwrap();
    details.content = Some(pb::fragment_reuse_index_details::Content::External(
        pb::ExternalFile {
            path: name.into(),
            offset: 0,
            size: bytes.len() as u64,
        },
    ));
    fri.index_details = Some(Arc::new(prost_types::Any::from_msg(&details).unwrap()));
    persist_fixture(dataset, indices).await;
}

async fn assert_legacy_scalar_queries(dataset: &Dataset) {
    assert_legacy_metadata(dataset).await;
    let index = dataset
        .load_index_by_name("value_idx")
        .await
        .unwrap()
        .unwrap();
    // Force loading even when the planner chooses a scan for a small fixture.
    dataset
        .open_scalar_index("value", &index.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    for filter in ["value = 2", "value IS NULL", "value >= 2"] {
        let mut scan = dataset.scan();
        scan.use_scalar_index(false)
            .filter(filter)
            .unwrap()
            .project(&["id"])
            .unwrap();
        let expected = scan.try_into_batch().await.unwrap();
        let ids = |batch: &RecordBatch| {
            batch["id"]
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
        };
        for _ in 0..2 {
            let actual = dataset
                .scan()
                .filter(filter)
                .unwrap()
                .project(&["id"])
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            assert_eq!(
                ids(&actual),
                ids(&expected),
                "filter {filter}, version {}",
                dataset.version_id()
            );
        }
    }
}

#[rstest::rstest]
#[case::btree_inline(IndexType::BTree, false)]
#[case::btree_external(IndexType::BTree, true)]
#[case::bitmap(IndexType::Bitmap, false)]
#[case::zonemap(IndexType::ZoneMap, false)]
#[case::bloom(IndexType::BloomFilter, false)]
#[tokio::test]
async fn legacy_scalar_lifecycle_never_enters_new_reader(
    #[case] index_type: IndexType,
    #[case] external: bool,
    #[values(false, true)] defer_index_remap: bool,
) {
    LEGACY_READER_ONLY
        .scope((), async {
            let batch = arrow_array::record_batch!(
                ("id", Int32, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                (
                    "value",
                    Int32,
                    [
                        Some(0),
                        None,
                        Some(2),
                        Some(3),
                        None,
                        Some(2),
                        Some(6),
                        Some(7),
                        None,
                        Some(2),
                        Some(10),
                        Some(11)
                    ]
                )
            )
            .unwrap();
            let mut dataset = Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                "memory://",
                Some(WriteParams {
                    max_rows_per_file: 3,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
            dataset
                .create_index(
                    &["value"],
                    index_type,
                    Some("value_idx".into()),
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();
            assert!(assert_legacy_metadata(&dataset).await.is_none());
            assert_legacy_scalar_queries(&dataset).await;
            crate::dataset::optimize::compact_files(
                &mut dataset,
                crate::dataset::optimize::CompactionOptions {
                    target_rows_per_fragment: 6,
                    defer_index_remap,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            assert_eq!(
                assert_legacy_metadata(&dataset).await.is_some(),
                defer_index_remap
            );
            if external && defer_index_remap {
                externalize_legacy_history(&mut dataset).await;
            }
            assert_legacy_scalar_queries(&dataset).await;
            let first = dataset.clone();
            dataset.delete("id = 2").await.unwrap();
            assert_legacy_scalar_queries(&dataset).await;
            let appended =
                arrow_array::record_batch!(("id", Int32, [12]), ("value", Int32, [Some(2)]))
                    .unwrap();
            dataset
                .append(
                    RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
                    None,
                )
                .await
                .unwrap();
            assert_legacy_scalar_queries(&dataset).await;
            crate::dataset::optimize::compact_files(
                &mut dataset,
                crate::dataset::optimize::CompactionOptions {
                    target_rows_per_fragment: 100,
                    defer_index_remap,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            assert_legacy_scalar_queries(&dataset).await;
            // Checkout shares the session cache, but must use its own FRI history.
            let historical = dataset.checkout_version(first.version_id()).await.unwrap();
            assert!(Arc::ptr_eq(&historical.session(), &dataset.session()));
            assert_legacy_scalar_queries(&historical).await;
            assert_legacy_scalar_queries(&dataset).await;
            dataset.index_statistics("value_idx").await.unwrap();
            if defer_index_remap {
                dataset
                    .index_statistics(FRAG_REUSE_INDEX_NAME)
                    .await
                    .unwrap();
            }
            let before_remap = dataset.version_id();
            let result = crate::dataset::optimize::remapping::remap_column_index(
                &mut dataset,
                &["value"],
                Some("value_idx".into()),
            )
            .await;
            if defer_index_remap {
                result.unwrap();
            } else {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::NotSupported { .. }));
                assert!(error.to_string().contains("Fragment reuse index not found"));
                assert_eq!(dataset.version_id(), before_remap);
            }
            assert_legacy_scalar_queries(&dataset).await;
            crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut dataset)
                .await
                .unwrap();
            assert_legacy_scalar_queries(&dataset).await;
        })
        .await;
}

#[tokio::test]
async fn legacy_reader_guard_rejects_new_entry_points() {
    let dataset = fixture().await;
    let index = dataset.load_index_by_name("i_idx").await.unwrap().unwrap();
    LEGACY_READER_ONLY
        .scope((), async {
            let Err(error) = FragmentReuseIndex::open(&dataset, &index).await else {
                panic!("guard must reject new reader construction");
            };
            assert!(matches!(error, Error::Internal { .. }));
            assert!(
                error
                    .to_string()
                    .contains("V1 operation entered the new FRI reader")
            );
            let error = load_ledger(&dataset, &index).await.unwrap_err();
            assert!(matches!(error, Error::Internal { .. }));
            assert!(
                error
                    .to_string()
                    .contains("V1 operation entered the new FRI reader")
            );
        })
        .await;
}

#[rstest::rstest]
#[case::inline(false)]
#[case::external(true)]
#[tokio::test]
async fn released_v1_dataset_uses_legacy_reader(#[case] external: bool) {
    LEGACY_READER_ONLY
        .scope((), async {
            let dir = crate::utils::test::copy_test_data_to_tmp(
                "fri_straddle_pre_6610/fri_straddle_dataset",
            )
            .unwrap();
            let mut dataset = Dataset::open(dir.std_path().to_str().unwrap())
                .await
                .unwrap();
            let initial_rows = dataset.count_rows(None).await.unwrap();
            assert!(assert_legacy_metadata(&dataset).await.is_some());
            if external {
                externalize_legacy_history(&mut dataset).await;
            }
            for _ in 0..2 {
                let index = crate::index::load_all_indices(&dataset)
                    .await
                    .unwrap()
                    .iter()
                    .find(|index| index.name != FRAG_REUSE_INDEX_NAME)
                    .unwrap()
                    .clone();
                dataset
                    .open_vector_index("vec", &index.uuid, &NoOpMetricsCollector)
                    .await
                    .unwrap();
                assert_eq!(
                    dataset
                        .count_rows(Some("vec IS NOT NULL".into()))
                        .await
                        .unwrap(),
                    initial_rows
                );
                assert_legacy_metadata(&dataset).await;
            }
            let before_append = dataset.clone();
            let batch = dataset
                .scan()
                .limit(Some(1), None)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap();
            dataset
                .append(
                    RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                    None,
                )
                .await
                .unwrap();
            assert_legacy_metadata(&dataset).await;
            assert_eq!(dataset.count_rows(None).await.unwrap(), initial_rows + 1);
            let historical = dataset
                .checkout_version(before_append.version_id())
                .await
                .unwrap();
            assert!(Arc::ptr_eq(&dataset.session(), &historical.session()));
            assert_legacy_metadata(&historical).await;
            assert_eq!(historical.count_rows(None).await.unwrap(), initial_rows);
        })
        .await;
}

async fn assert_legacy_vector_queries(dataset: &Dataset) {
    assert_legacy_metadata(dataset).await;
    let index = dataset
        .load_index_by_name("vector_idx")
        .await
        .unwrap()
        .unwrap();
    dataset
        .open_vector_index("vector", &index.uuid, &NoOpMetricsCollector)
        .await
        .unwrap();
    let query = arrow_array::Float32Array::from(vec![5.25, 5.25, 5.25, 5.25]);
    let mut scan = dataset.scan();
    scan.nearest("vector", &query, 3)
        .unwrap()
        .use_index(false)
        .project(&["id"])
        .unwrap();
    let expected = scan.try_into_batch().await.unwrap();
    let ids = |batch: &RecordBatch| {
        batch["id"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
    };
    for _ in 0..2 {
        let mut scan = dataset.scan();
        scan.nearest("vector", &query, 3)
            .unwrap()
            .project(&["id"])
            .unwrap();
        assert!(scan.explain_plan(false).await.unwrap().contains("ANN"));
        let result = scan.try_into_batch().await.unwrap();
        // IVF-Flat with one partition is exact: recall must be 1.0.
        assert_eq!(ids(&result), ids(&expected));
    }
}

#[rstest::rstest]
#[case::eager(false, false)]
#[case::deferred_inline(true, false)]
#[case::deferred_external(true, true)]
#[tokio::test]
async fn legacy_vector_lifecycle_never_enters_new_reader(
    #[case] defer_index_remap: bool,
    #[case] external: bool,
) {
    LEGACY_READER_ONLY
        .scope((), async {
            let vectors = arrow_array::FixedSizeListArray::from_iter_primitive::<
                arrow_array::types::Float32Type,
                _,
                _,
            >((0..12).map(|id| Some(vec![Some(id as f32); 4])), 4);
            let batch = RecordBatch::try_from_iter([
                (
                    "id",
                    Arc::new(arrow_array::Int32Array::from_iter_values(0..12))
                        as arrow_array::ArrayRef,
                ),
                ("vector", Arc::new(vectors) as arrow_array::ArrayRef),
            ])
            .unwrap();
            let mut dataset = Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                "memory://",
                Some(WriteParams {
                    max_rows_per_file: 3,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
            let params = crate::index::vector::VectorIndexParams::ivf_flat(
                1,
                lance_linalg::distance::DistanceType::L2,
            );
            dataset
                .create_index(
                    &["vector"],
                    IndexType::Vector,
                    Some("vector_idx".into()),
                    &params,
                    true,
                )
                .await
                .unwrap();
            assert!(assert_legacy_metadata(&dataset).await.is_none());
            assert_legacy_vector_queries(&dataset).await;
            crate::dataset::optimize::compact_files(
                &mut dataset,
                crate::dataset::optimize::CompactionOptions {
                    target_rows_per_fragment: 6,
                    defer_index_remap,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            assert_eq!(
                assert_legacy_metadata(&dataset).await.is_some(),
                defer_index_remap
            );
            if external && defer_index_remap {
                externalize_legacy_history(&mut dataset).await;
            }
            assert_legacy_vector_queries(&dataset).await;
            let first = dataset.clone();
            dataset.delete("id = 5").await.unwrap();
            assert_legacy_vector_queries(&dataset).await;
            let appended = batch.slice(11, 1);
            dataset
                .append(
                    RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
                    None,
                )
                .await
                .unwrap();
            assert_legacy_vector_queries(&dataset).await;
            crate::dataset::optimize::compact_files(
                &mut dataset,
                crate::dataset::optimize::CompactionOptions {
                    target_rows_per_fragment: 100,
                    defer_index_remap,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            dataset.prewarm_index("vector_idx").await.unwrap();
            assert_legacy_vector_queries(&dataset).await;
            let historical = dataset.checkout_version(first.version_id()).await.unwrap();
            assert!(Arc::ptr_eq(&dataset.session(), &historical.session()));
            assert_legacy_vector_queries(&historical).await;
            assert_legacy_vector_queries(&dataset).await;
            dataset.index_statistics("vector_idx").await.unwrap();
            let before_remap = dataset.version_id();
            let result = crate::dataset::optimize::remapping::remap_column_index(
                &mut dataset,
                &["vector"],
                Some("vector_idx".into()),
            )
            .await;
            if defer_index_remap {
                result.unwrap();
            } else {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::NotSupported { .. }));
                assert!(error.to_string().contains("Fragment reuse index not found"));
                assert_eq!(dataset.version_id(), before_remap);
            }
            crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut dataset)
                .await
                .unwrap();
            assert_legacy_vector_queries(&dataset).await;
        })
        .await;
}

#[tokio::test]
async fn released_legacy_fri_keeps_version_zero_on_append() {
    let dir =
        crate::utils::test::copy_test_data_to_tmp("fri_straddle_pre_6610/fri_straddle_dataset")
            .unwrap();
    let uri = dir.std_path().to_str().unwrap();
    let mut dataset = Dataset::open(uri).await.unwrap();
    let indices = crate::index::load_all_indices(&dataset).await.unwrap();
    let fri = indices
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
        .unwrap();
    assert_eq!(fri.index_version, 0);
    // Decode the released writer's actual Any, not a reserialized modern protobuf.
    let ledger = load_ledger(&dataset, fri).await.unwrap();
    assert!(!ledger.transitions().is_empty());
    let legacy = dataset
        .open_frag_reuse_index(&NoOpMetricsCollector)
        .await
        .unwrap()
        .unwrap();
    for transition in ledger.transitions() {
        for source in transition.sources() {
            for offset in 0..source.physical_rows {
                let original = RowAddress::new_from_parts(source.id as u32, offset as u32);
                let mut translated = Some(u64::from(original));
                while let Some(current) = translated {
                    let Some(index) = ledger.consumer(RowAddress::from(current).fragment_id())
                    else {
                        break;
                    };
                    let Mapping::OrderedCompaction(remap) = ledger.transitions()[index].mapping()
                    else {
                        unreachable!()
                    };
                    translated = remap.get(current).unwrap();
                }
                assert_eq!(translated, legacy.remap_row_id(original.into()));
            }
        }
    }
    let original_rows = dataset.count_rows(None).await.unwrap();
    let batch = dataset
        .scan()
        .limit(Some(1), None)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            None,
        )
        .await
        .unwrap();
    let reopened = Dataset::open(uri).await.unwrap();
    assert_eq!(reopened.count_rows(None).await.unwrap(), original_rows + 1);
    let indices = crate::index::load_all_indices(&reopened).await.unwrap();
    let fri = indices
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
        .unwrap();
    assert_eq!(fri.index_version, 0);
    let flag = lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
    assert_eq!(reopened.manifest.reader_feature_flags & flag, 0);
    assert_eq!(reopened.manifest.writer_feature_flags & flag, 0);
}

#[tokio::test]
async fn legacy_vector_format_is_excluded_from_rewritten_coverage() {
    let dir = crate::utils::test::copy_test_data_to_tmp("v0.10.15/non_divisible_pq").unwrap();
    let mut dataset = Dataset::open(dir.std_path().to_str().unwrap())
        .await
        .unwrap();
    let indices = crate::index::load_all_indices(&dataset).await.unwrap();
    let vector = indices
        .iter()
        .find(|index| !index.fields.is_empty())
        .unwrap()
        .clone();
    assert!(
        !vector_supports_batch_remapping(&dataset, &vector)
            .await
            .unwrap()
    );
    let mut destinations = dataset.fragments().to_vec();
    let sources = destinations
        .iter()
        .map(|fragment| FragmentDigest {
            id: fragment.id,
            physical_rows: 1,
            num_deleted_rows: 0,
        })
        .collect();
    for fragment in &mut destinations {
        fragment.id += 10;
    }
    let transition = Transition {
        sources,
        destinations: destinations
            .iter()
            .map(|fragment| FragmentDigest {
                id: fragment.id,
                physical_rows: 1,
                num_deleted_rows: 0,
            })
            .collect(),
        mapping: Some(transition::Mapping::StablePartition(StablePartition {
            map_id: Uuid::new_v4().to_string(),
            map_size_bytes: 1,
            base_id: None,
        })),
    };
    // No row-map file is written: the unsupported vector segment must fall back.
    install(
        &mut dataset,
        InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec(),
        destinations,
        false,
    )
    .await;
    assert!(
        dataset
            .load_index_by_name(&vector.name)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(dataset.count_rows(Some("id = 0".into())).await.unwrap(), 1);
}

#[rstest::rstest]
#[case::lazy(false)]
#[case::prewarmed(true)]
#[tokio::test]
async fn vector_partition_uses_shared_remapping_and_cached_reconstruction(#[case] prewarm: bool) {
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "vector",
            lance_datagen::array::rand_vec::<arrow_array::types::Float32Type>(4.into()),
        )
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(4))
        .await
        .unwrap();
    let params = crate::index::vector::VectorIndexParams::ivf_flat(
        1,
        lance_linalg::distance::DistanceType::L2,
    );
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".into()),
            &params,
            true,
        )
        .await
        .unwrap();
    let original = dataset
        .scan()
        .filter("i = 2")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let query = original
        .column_by_name("vector")
        .unwrap()
        .as_fixed_size_list()
        .value(0);
    let query = query.as_primitive::<arrow_array::types::Float32Type>();
    dataset.delete("i = 3").await.unwrap();
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    if prewarm {
        dataset.prewarm_index("vector_idx").await.unwrap();
    }
    for _ in 0..2 {
        let mut scan = dataset.scan();
        scan.nearest("vector", query, 1).unwrap();
        let plan = scan.explain_plan(false).await.unwrap();
        assert!(plan.contains("ANN"), "{plan}");
        let result = scan.try_into_batch().await.unwrap();
        assert_eq!(result.num_rows(), 1);
        assert_eq!(
            result
                .column_by_name("i")
                .unwrap()
                .as_primitive::<Int32Type>()
                .value(0),
            2
        );
    }
}

#[tokio::test]
async fn unsupported_scalar_type_is_not_advertised_for_rewritten_fragments() {
    let mut dataset = fixture().await;
    // Only metadata is needed: the unsupported reader must never open it.
    let mut indices = dataset.load_indices().await.unwrap().as_ref().clone();
    indices[0].index_version = 0;
    indices[0].index_details = Some(Arc::new(
        prost_types::Any::from_msg(&lance_index::pb::FmIndexDetails::default()).unwrap(),
    ));
    let key = IndexMetadataKey {
        version: dataset.manifest.version,
        store_identity: &dataset.object_store.store_prefix,
        e_tag: dataset.manifest_location.e_tag.as_deref(),
    };
    dataset
        .index_cache
        .insert_with_key(&key, Arc::new(indices))
        .await;
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    assert!(dataset.load_index_by_name("i_idx").await.unwrap().is_none());
    assert_eq!(dataset.count_rows(Some("i = 2".into())).await.unwrap(), 1);
}
