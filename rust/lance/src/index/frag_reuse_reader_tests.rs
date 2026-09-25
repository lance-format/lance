// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::tests::{field, fixture, fixture_with_index, install, persist_fixture, prepare};
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
#[case::default_search("default")]
#[case::all_segments("all")]
#[case::partial_selection("partial")]
#[case::reconverged("reconverged")]
#[tokio::test]
async fn vector_fragment_search_requires_every_contributor(
    #[case] selection: &str,
    #[values(false, true)] filtered_destination: bool,
) {
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
    let fragments: Vec<_> = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    let mut segments = Vec::new();
    for fragment in fragments {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
                .name("vector_idx".into())
                .fragments(vec![fragment])
                .execute_uncommitted()
                .await
                .unwrap(),
        );
    }
    let ids: Vec<_> = segments.iter().map(|segment| segment.uuid).collect();
    dataset
        .commit_existing_index_segments("vector_idx", "vector", segments)
        .await
        .unwrap();
    let original = dataset
        .scan()
        .filter("i = 6")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let query = original["vector"].as_fixed_size_list().value(0);
    let query = query.as_primitive::<arrow_array::types::Float32Type>();
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    if selection == "reconverged" {
        // Prefer the new C segment while old A/B segments still provide D.
        // After C,D -> E their row sets must remain disjoint within E.
        let c = dataset.fragments()[0].id as u32;
        let direct = CreateIndexBuilder::new(&mut dataset, &["vector"], IndexType::Vector, &params)
            .name("vector_idx".into())
            .replace(true)
            .fragments(vec![c])
            .execute_uncommitted()
            .await
            .unwrap();
        let mut indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        indices.push(direct);
        let sources: Vec<_> = dataset
            .fragments()
            .iter()
            .map(|fragment| FragmentDigest {
                id: fragment.id,
                physical_rows: fragment.physical_rows.unwrap() as u64,
                num_deleted_rows: 0,
            })
            .collect();
        let mut changed_row_addrs = Vec::new();
        roaring::RoaringTreemap::from_iter(sources.iter().flat_map(|source| {
            (0..source.physical_rows).map(move |offset| {
                u64::from(RowAddress::new_from_parts(source.id as u32, offset as u32))
            })
        }))
        .serialize_into(&mut changed_row_addrs)
        .unwrap();
        let batch = dataset.scan().try_into_batch().await.unwrap();
        let rewrite = crate::dataset::InsertBuilder::new(Arc::new(dataset.clone()))
            .with_params(&WriteParams {
                mode: crate::dataset::WriteMode::Append,
                ..Default::default()
            })
            .execute_uncommitted(vec![batch])
            .await
            .unwrap();
        let lance_table::transaction::Operation::Append { mut fragments } = rewrite.operation
        else {
            panic!("expected append fixture");
        };
        assert_eq!(fragments.len(), 1);
        fragments[0].id = 20;
        let fri = indices
            .iter_mut()
            .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
            .unwrap();
        let mut details = pb::FragmentReuseIndexDetails::decode(
            fri.index_details.as_ref().unwrap().value.as_slice(),
        )
        .unwrap();
        let Some(pb::fragment_reuse_index_details::Content::Inline(content)) = &mut details.content
        else {
            panic!("expected inline fixture");
        };
        content.transitions.push(Transition {
            sources,
            destinations: vec![FragmentDigest {
                id: 20,
                physical_rows: 8,
                num_deleted_rows: 0,
            }],
            mapping: Some(transition::Mapping::OrderedCompaction(
                pb::fragment_reuse_index_details::OrderedCompaction { changed_row_addrs },
            )),
        });
        fri.uuid = Uuid::new_v4();
        fri.fragment_bitmap = Some(RoaringBitmap::from_iter([20]));
        fri.index_details = Some(Arc::new(prost_types::Any::from_msg(&details).unwrap()));
        Arc::make_mut(&mut dataset.manifest).fragments = fragments.into();
        persist_fixture(&mut dataset, indices).await;
    }

    let mut scan = dataset.scan();
    let expected_rows = if selection == "reconverged" && !filtered_destination {
        8
    } else {
        1
    };
    scan.nearest("vector", query, expected_rows).unwrap();
    if filtered_destination {
        scan.with_fragments(vec![dataset.fragments()[0].clone()]);
        scan.filter("i = 6").unwrap();
        scan.prefilter(true);
    } else {
        scan.with_fragments(dataset.fragments().iter().cloned().collect());
    }
    match selection {
        "all" => {
            scan.with_index_segments(ids).unwrap();
        }
        "partial" => {
            scan.with_index_segments(vec![ids[0]]).unwrap();
        }
        "default" | "reconverged" => {}
        _ => unreachable!(),
    }
    let plan = scan.explain_plan(false).await.unwrap();
    assert_eq!(plan.contains("ANN"), selection != "partial", "{plan}");
    let result = scan.try_into_batch().await.unwrap();
    assert_eq!(result.num_rows(), expected_rows);
    if expected_rows == 8 {
        let mut values = result["i"].as_primitive::<Int32Type>().values().to_vec();
        values.sort_unstable();
        assert_eq!(values, (0..8).collect::<Vec<_>>(), "{plan}");
    }
    // IVF_FLAT with one partition is exact: the nearest row must come from Y,
    // including when explicit selection of X requires a flat fallback.
    assert_eq!(
        result["i"].as_primitive::<Int32Type>().value(0),
        6,
        "{plan}"
    );
}

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
            assert!(crate::index::index_is_usable(&indices[second]));
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

#[rstest::rstest]
#[case::only_c(false)]
#[case::both_destinations(true)]
#[tokio::test]
async fn direct_destination_coverage_preserves_partial_index_queries(#[case] both: bool) {
    let mut dataset = fixture().await;
    let params = ScalarIndexParams::default();
    let sources: Vec<_> = dataset
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    let mut segments = Vec::new();
    for source in sources {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["i"], IndexType::BTree, &params)
                .name("i_idx".into())
                .replace(true)
                .fragments(vec![source])
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
    let covered: Vec<_> = dataset
        .fragments()
        .iter()
        .take(if both { 2 } else { 1 })
        .map(|fragment| fragment.id as u32)
        .collect();
    let direct = CreateIndexBuilder::new(&mut dataset, &["i"], IndexType::BTree, &params)
        .name("i_idx".into())
        .replace(true)
        .fragments(covered.clone())
        .execute_uncommitted()
        .await
        .unwrap();
    let mut indices = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    if !both {
        // No index path reaches D, but direct coverage of C must remain usable.
        indices.retain(|index| index.name == FRAG_REUSE_INDEX_NAME);
    }
    indices.push(direct.clone());
    persist_fixture(&mut dataset, indices).await;
    let usable = crate::index::scalar_logical::load_named_scalar_segments(&dataset, "i", "i_idx")
        .await
        .unwrap();
    assert_eq!(usable.len(), 1);
    assert_eq!(usable[0].uuid, direct.uuid);
    assert_eq!(
        usable[0].fragment_bitmap,
        Some(RoaringBitmap::from_iter(covered))
    );
    let mut scan = dataset.scan();
    scan.filter("i >= 0").unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
    let batch = scan.try_into_batch().await.unwrap();
    let mut values = batch["i"].as_primitive::<Int32Type>().values().to_vec();
    values.sort_unstable();
    assert_eq!(values, (0..8).collect::<Vec<_>>(), "{plan}");
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
    // Translated entries are keyed by the segment's translation identity, not
    // by the manifest snapshot: a snapshot whose path differs but whose
    // coverage, exclusions and mapping path are unchanged is a cache hit.
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
        0,
        "a path-only snapshot change must not evict translated entries"
    );
    let inputs = [
        u64::from(RowAddress::new_from_parts(1, 3)),
        u64::from(RowAddress::new_from_parts(0, 2)),
        u64::from(RowAddress::new_from_parts(0, 2)),
        u64::from(RowAddress::new_from_parts(0, 3)),
    ];
    assert_eq!(
        mapping.remap_row_ids(&inputs).await.unwrap(),
        vec![
            Some(u64::from(RowAddress::new_from_parts(11, 2))),
            Some(u64::from(RowAddress::new_from_parts(10, 1))),
            Some(u64::from(RowAddress::new_from_parts(10, 1))),
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
            let super::super::frag_reuse::ResolvedRemapping::V0(remapper) = remapping else {
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
    lance_index::scalar::LEGACY_TRAFFIC_ONLY
        .scope(
            (),
            LEGACY_READER_ONLY.scope((), async {
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
                dataset.optimize_indices(&Default::default()).await.unwrap();
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
            }),
        )
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
    lance_index::scalar::LEGACY_TRAFFIC_ONLY
        .scope(
            (),
            LEGACY_READER_ONLY.scope((), async {
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
                // Merge-all keeps a single segment, which the query helper
                // expects when reopening the index by name.
                dataset
                    .optimize_indices(&lance_index::optimize::OptimizeOptions::merge(usize::MAX))
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
            }),
        )
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
    let indices = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    persist_fixture(&mut dataset, indices).await;
    if prewarm {
        dataset.prewarm_index("vector_idx").await.unwrap();
    }
    let search = |dataset: Dataset| async move {
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
    };
    for _ in 0..2 {
        search(dataset.clone()).await;
    }

    // The IVF state embeds no rows and the translated partition is keyed by
    // the segment's translation identity, which an append leaves unchanged:
    // both survive the commit. Before this keying every commit cold-started
    // the whole index.
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(original.clone())], original.schema()),
            None,
        )
        .await
        .unwrap();
    let uuid = dataset
        .load_index_by_name("vector_idx")
        .await
        .unwrap()
        .unwrap()
        .uuid;
    let metrics = lance_index::metrics::LocalMetricsCollector::default();
    let index = dataset
        .open_vector_index("vector", &uuid, &metrics)
        .await
        .unwrap();
    assert_eq!(
        metrics
            .index_loads
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "the IVF state entry survives an append"
    );
    let ivf = index
        .as_any()
        .downcast_ref::<crate::index::vector::ivf::v2::IvfFlatIndex>()
        .unwrap();
    ivf.load_partition(0, true, &metrics).await.unwrap();
    assert_eq!(
        metrics
            .parts_loaded
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "the translated partition entry survives an append"
    );
    search(dataset.clone()).await;
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

#[tokio::test]
async fn tagged_remapping_plans_coverage_once_per_snapshot() {
    use crate::index::frag_reuse::{FriQueryPlanKey, ResolvedRemapping, open_row_id_remapping};

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

    let indices = dataset.load_indices().await.unwrap();
    let fri = indices
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
        .unwrap();
    assert_ne!(fri.index_version, 0);
    let siblings: Vec<_> = indices
        .iter()
        .filter(|index| index.name == "i_idx")
        .cloned()
        .collect();
    assert_eq!(siblings.len(), 2);

    let plan_cache = dataset
        .index_cache
        .with_key_prefix(dataset.manifest_location.path.as_ref());
    let key = FriQueryPlanKey {
        fri_uuid: &fri.uuid,
    };
    assert!(
        plan_cache.get_with_key(&key).await.is_none(),
        "no plan may exist before the first tagged open"
    );

    let first = open_row_id_remapping(&dataset, &siblings[0], &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(matches!(
        first,
        Some((_, ResolvedRemapping::V1Translate { .. }))
    ));
    let plan = plan_cache
        .get_with_key(&key)
        .await
        .expect("the first tagged open must publish the snapshot plan");
    assert!(plan.segments.contains_key(&siblings[1].uuid));

    // Later opens must resolve from the published plan instead of re-running
    // the whole-dataset sibling scan: with the sibling dropped from the cached
    // plan, its open must fail with the plan-miss error.
    let mut pruned = plan.as_ref().clone();
    pruned.segments.remove(&siblings[1].uuid);
    plan_cache.insert_with_key(&key, Arc::new(pruned)).await;
    let error = open_row_id_remapping(&dataset, &siblings[1], &NoOpMetricsCollector)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires committed segment metadata"),
        "{error}"
    );

    // With the real plan restored, the sibling opens as a cache hit on the
    // same plan entry; nothing is recomputed or re-published.
    plan_cache.insert_with_key(&key, plan.clone()).await;
    let second = open_row_id_remapping(&dataset, &siblings[1], &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(matches!(
        second,
        Some((_, ResolvedRemapping::V1Translate { .. }))
    ));
    let republished = plan_cache.get_with_key(&key).await.unwrap();
    assert!(
        Arc::ptr_eq(&plan, &republished),
        "the second open must reuse the published plan instead of rebuilding it"
    );
    for value in 0..8 {
        assert_eq!(
            dataset
                .count_rows(Some(format!("i = {value}")))
                .await
                .unwrap(),
            1
        );
    }

    // Warm opens must not invoke load_indices (whose tagged post-processing
    // recomputes coverage): replace the session metadata listing with one that
    // keeps only the FRI entry, so any listing-derived resolution of the
    // sibling would fail. The open must still succeed from the cached plan.
    let metadata_key = IndexMetadataKey {
        version: dataset.manifest.version,
        store_identity: &dataset.object_store.store_prefix,
        e_tag: dataset.manifest_location.e_tag.as_deref(),
    };
    dataset
        .index_cache
        .insert_with_key(&metadata_key, Arc::new(vec![fri.clone()]))
        .await;
    let warm = open_row_id_remapping(&dataset, &siblings[1], &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        matches!(warm, Some((_, ResolvedRemapping::V1Translate { .. }))),
        "a warm open must resolve purely from the cached plan"
    );
}

#[tokio::test]
async fn resolver_dispatch_follows_fri_version_and_segment_need() {
    use crate::index::frag_reuse::{ResolvedRemapping, open_row_id_remapping};

    // No FRI: nothing to resolve.
    let mut dataset = fixture().await;
    let segment = dataset.load_index_by_name("i_idx").await.unwrap().unwrap();
    assert!(
        open_row_id_remapping(&dataset, &segment, &NoOpMetricsCollector)
            .await
            .unwrap()
            .is_none()
    );

    // The transition rewrites both original fragments; a fragment appended
    // afterwards stays outside the mapped graph entirely.
    let (transition, mut destinations) = prepare(&dataset).await;
    let appended = arrow_array::record_batch!(("i", Int32, [100])).unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
            None,
        )
        .await
        .unwrap();
    let untouched_fragment = dataset
        .fragments()
        .iter()
        .max_by_key(|fragment| fragment.id)
        .unwrap()
        .clone();
    let untouched = CreateIndexBuilder::new(
        &mut dataset,
        &["i"],
        IndexType::BTree,
        &ScalarIndexParams::default(),
    )
    .name("i_untouched".into())
    .replace(true)
    .fragments(vec![untouched_fragment.id as u32])
    .execute_uncommitted()
    .await
    .unwrap();
    destinations.push(untouched_fragment);
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    let mut all = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    all.push(untouched.clone());
    persist_fixture(&mut dataset, all).await;

    // The committed segment covers rewritten fragments: it needs translation.
    let indices = dataset.load_indices().await.unwrap();
    let segment = indices.iter().find(|index| index.name == "i_idx").unwrap();
    let resolved = open_row_id_remapping(&dataset, segment, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        matches!(resolved, Some((_, ResolvedRemapping::V1Translate { .. }))),
        "{resolved:?}"
    );

    // The untouched segment resolves to identity and carries no remapper.
    let resolved = open_row_id_remapping(&dataset, &untouched, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        matches!(resolved, Some((_, ResolvedRemapping::V1Identity))),
        "{resolved:?}"
    );

    // A segment absent from the committed listing cannot be resolved.
    let mut unknown = untouched.clone();
    unknown.uuid = Uuid::new_v4();
    let error = open_row_id_remapping(&dataset, &unknown, &NoOpMetricsCollector)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("requires committed segment metadata"),
        "{error}"
    );
}

#[tokio::test]
async fn resolver_dispatch_v0_uses_the_compact_handle() {
    let mut dataset = fixture().await;
    crate::dataset::optimize::compact_files(
        &mut dataset,
        crate::dataset::optimize::CompactionOptions {
            target_rows_per_fragment: 100,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    // assert_legacy_metadata asserts every segment resolves to V0 with a
    // handle equivalent to the compact legacy index.
    assert!(assert_legacy_metadata(&dataset).await.is_some());
}

#[tokio::test]
async fn legacy_api_plugin_keeps_untouched_segments_available() {
    use crate::index::frag_reuse::{ResolvedRemapping, open_row_id_remapping};
    use lance_index::scalar::BuiltinIndexType;

    // The FM plugin only implements the legacy synchronous API
    // (supports_batch_row_id_remapping is false).
    let batch = arrow_array::record_batch!(
        ("i", Int32, [0, 1, 2, 3, 4, 5, 6, 7]),
        (
            "text",
            Utf8,
            [
                "row0", "row1", "row2", "row3", "row4", "row5", "row6", "row7"
            ]
        )
    )
    .unwrap();
    let mut dataset = crate::Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::Fm);
    dataset
        .create_index(
            &["text"],
            IndexType::Fm,
            Some("text_idx".into()),
            &params,
            true,
        )
        .await
        .unwrap();
    let (transition, mut destinations) = prepare(&dataset).await;
    let appended =
        arrow_array::record_batch!(("i", Int32, [8]), ("text", Utf8, ["extra8"])).unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
            None,
        )
        .await
        .unwrap();
    let untouched_fragment = dataset
        .fragments()
        .iter()
        .max_by_key(|fragment| fragment.id)
        .unwrap()
        .clone();
    let untouched = CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::Fm, &params)
        .name("text_idx".into())
        .replace(true)
        .fragments(vec![untouched_fragment.id as u32])
        .execute_uncommitted()
        .await
        .unwrap();
    destinations.push(untouched_fragment);
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;

    // The committed segment needs translation, which the FM plugin cannot do:
    // it is excluded from coverage and substring queries fall back to a scan,
    // still returning correct rows.
    let usable =
        crate::index::scalar_logical::load_named_scalar_segments(&dataset, "text", "text_idx")
            .await
            .unwrap();
    assert!(usable.is_empty());
    let mut scan = dataset.scan();
    scan.filter("contains(text, 'row2')").unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(!plan.contains("ScalarIndexQuery"), "{plan}");
    let rows = scan.try_into_batch().await.unwrap();
    assert_eq!(rows.num_rows(), 1, "{plan}");
    assert_eq!(rows["i"].as_primitive::<Int32Type>().value(0), 2);

    // The untouched FM segment resolves to identity, loads through the
    // plugin's original entry point, and serves queries: the availability win
    // for legacy-API-only plugins.
    let mut all = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    all.push(untouched.clone());
    persist_fixture(&mut dataset, all).await;
    let resolved = open_row_id_remapping(&dataset, &untouched, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(
        matches!(resolved, Some((_, ResolvedRemapping::V1Identity))),
        "{resolved:?}"
    );
    let usable =
        crate::index::scalar_logical::load_named_scalar_segments(&dataset, "text", "text_idx")
            .await
            .unwrap();
    assert_eq!(usable.len(), 1);
    assert_eq!(usable[0].uuid, untouched.uuid);
    let mut scan = dataset.scan();
    scan.filter("contains(text, 'extra8')").unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
    let rows = scan.try_into_batch().await.unwrap();
    assert_eq!(rows.num_rows(), 1, "{plan}");
    assert_eq!(rows["i"].as_primitive::<Int32Type>().value(0), 8);
    // Rows in fragments the identity segment does not cover still come back
    // correctly through the scan side of the plan.
    assert_eq!(
        dataset
            .count_rows(Some("contains(text, 'row2')".into()))
            .await
            .unwrap(),
        1
    );
}

#[tokio::test]
async fn legacy_traffic_guard_rejects_batch_entry_points() {
    let mut dataset = fixture().await;
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    let indices = dataset.load_indices().await.unwrap();
    let segment = indices
        .iter()
        .find(|index| index.name == "i_idx")
        .unwrap()
        .clone();
    lance_index::scalar::LEGACY_TRAFFIC_ONLY
        .scope((), async {
            let error = crate::index::frag_reuse::open_row_id_remapping(
                &dataset,
                &segment,
                &NoOpMetricsCollector,
            )
            .await
            .unwrap_err();
            assert!(
                error.to_string().contains("entered batch row-ID remapping"),
                "{error}"
            );
            let error = lance_index::scalar::check_batch_remapping_entry().unwrap_err();
            assert!(
                error.to_string().contains("entered batch row-ID remapping"),
                "{error}"
            );
        })
        .await;
}

#[tokio::test]
async fn tagged_append_preserves_stored_segment_provenance() {
    let mut dataset = fixture().await;
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    let indices = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    persist_fixture(&mut dataset, indices).await;

    let serialize = |bitmap: &Option<roaring::RoaringBitmap>| {
        bitmap.as_ref().map(|bitmap| {
            let mut bytes = Vec::new();
            bitmap.serialize_into(&mut bytes).unwrap();
            bytes
        })
    };
    let before: Vec<_> = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .iter()
        .map(|index| (index.uuid, serialize(&index.fragment_bitmap)))
        .collect();
    assert!(!before.is_empty());

    // Query first so the session holds rewritten (derived) coverage; the
    // commit below must not leak it into stored metadata.
    assert_eq!(dataset.count_rows(Some("i = 2".into())).await.unwrap(), 1);
    let appended = arrow_array::record_batch!(("i", Int32, [100])).unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
            None,
        )
        .await
        .unwrap();

    // Snapshot-derived coverage must never be persisted back to a manifest:
    // every stored segment keeps byte-identical provenance.
    let after = crate::index::load_all_indices(&dataset).await.unwrap();
    for (uuid, bitmap) in &before {
        let stored = after.iter().find(|index| index.uuid == *uuid).unwrap();
        assert_eq!(&serialize(&stored.fragment_bitmap), bitmap, "{uuid}");
    }
    assert_eq!(dataset.count_rows(Some("i = 100".into())).await.unwrap(), 1);
}

// End-to-end composition: a v1 FRI whose only transition is DROPPED (unknown
// field) over an indexed dataset. The scalar consumer must fall back to scanning
// (the segment cannot claim coverage through a dropped mapping) and every value
// count must match the pre-FRI direct-scan truth, so no rows are lost and no
// stale pre-transition rows are returned. This exercises the full query path
// (planner + scalar consumer + FRI reader) end to end; the guard's own
// fail-without-fix witness is the unit test
// `dropped_unknown_transition_denies_identity_to_live_segment`.
#[rstest::rstest]
#[case::inline(false)]
#[case::external(true)]
#[tokio::test]
async fn dropped_unknown_transition_forces_scalar_query_to_scan(#[case] external: bool) {
    let mut dataset = fixture_with_index(IndexType::BTree).await;

    let (mut transition, destinations) = prepare(&dataset).await;
    transition.mapping = None;
    let mut raw = transition.encode_to_vec();
    raw.extend(field(17, b"future mapping"));
    let fri = install(&mut dataset, field(2, &raw), destinations, external).await;

    // Ground truth from an explicitly index-DISABLED scan of the reclustered
    // snapshot, so the comparison is against a real scan and not the index path.
    let truth: Vec<usize> = {
        let mut counts = Vec::new();
        for value in 0..8 {
            let mut scan = dataset.scan();
            scan.filter(&format!("i = {value}")).unwrap();
            scan.use_scalar_index(false);
            counts.push(scan.try_into_batch().await.unwrap().num_rows());
        }
        counts
    };
    let mapping = FragmentReuseIndex::open(&dataset, &fri).await.unwrap();
    assert!(mapping.ledger.has_unsupported_transitions());
    assert!(mapping.ledger.transitions().is_empty());

    // The scalar index must not be used: the dropped unknown transition forbids
    // identity, so the segment cannot claim coverage and the query scans.
    let plan = dataset
        .scan()
        .filter("i = 2")
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();
    assert!(!plan.contains("ScalarIndexQuery"), "{plan}");

    // Every value count still matches the direct-scan truth: no rows lost and no
    // stale pre-transition rows returned.
    for (value, &expected) in truth.iter().enumerate() {
        assert_eq!(
            dataset
                .count_rows(Some(format!("i = {value}")))
                .await
                .unwrap(),
            expected,
            "value {value}"
        );
    }
}

// Combination: FRI effective coverage x #9449 fragment-scope pruning.
//
// Two BTree segments each cover one source fragment (S1 -> F0, S2 -> F1). An FRI
// reclusters F0,F1 into destinations {10,11} with rows interleaved, so after the
// rewrite each segment's EFFECTIVE coverage spans both destinations. A scope over
// one destination must therefore keep BOTH segments (querying one reclustered
// fragment can still need multiple old segments). An unrelated index that never
// covers the destinations must be pruned, proving the pruning actually fires
// rather than opening everything.
#[tokio::test]
async fn fragment_scope_prunes_on_fri_effective_coverage() {
    use crate::index::scalar_logical::{open_scalar_index_segments, scalar_index_fragment_bitmap};

    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(4))
        .await
        .unwrap();
    let params = ScalarIndexParams::default();
    let source_fragments: Vec<u32> = dataset.fragments().iter().map(|f| f.id as u32).collect();

    // One BTree segment per source fragment, committed as one logical index.
    let mut segments = Vec::new();
    for fragment in &source_fragments {
        segments.push(
            CreateIndexBuilder::new(&mut dataset, &["i"], IndexType::BTree, &params)
                .name("i_idx".into())
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

    // Recluster F0,F1 -> destinations {10,11}, rows interleaved by parity
    // (F10 = even i {0,2,4,6}, F11 = odd i {1,3,5,7}).
    let (transition, destinations) = prepare(&dataset).await;
    let destination_ids: Vec<u32> = destinations.iter().map(|f| f.id as u32).collect();
    assert_eq!(destination_ids, vec![10, 11]);
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;

    // Add an UNRELATED third segment S3 over a fresh fragment the recluster never
    // touched (values 8..12), extending the same logical index. Its coverage does
    // not intersect the {10,11} destinations, so a destination-scoped query must
    // prune it.
    let appended = arrow_array::record_batch!(("i", Int32, [8, 9, 10, 11])).unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(appended.clone())], appended.schema()),
            None,
        )
        .await
        .unwrap();
    // The appended fragment is the one that is NOT a recluster destination.
    // (`max()` would pick destination 11 and silently build S3 directly over
    // F11, where "direct coverage wins" would hand F11 to S3 and narrow S1/S2 to
    // {10}: a different scenario.)
    let s3_fragment = dataset
        .fragments()
        .iter()
        .map(|f| f.id as u32)
        .find(|id| !destination_ids.contains(id))
        .unwrap();
    let s3 = CreateIndexBuilder::new(&mut dataset, &["i"], IndexType::BTree, &params)
        .name("i_idx".into())
        .replace(true)
        .fragments(vec![s3_fragment])
        .execute_uncommitted()
        .await
        .unwrap();
    let mut indices = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    indices.push(s3);
    persist_fixture(&mut dataset, indices).await;

    // Effective coverage of the reclustered pair is the destinations; S3 stays on
    // its own fragment. Pruning consumes the EFFECTIVE bitmaps.
    let effective = scalar_index_fragment_bitmap(&dataset, "i", "i_idx")
        .await
        .unwrap()
        .unwrap();
    assert!(effective.contains(10) && effective.contains(11));

    // Open the index scoped to destination F10 and COUNT segment loads: both
    // reclustered contributors (S1, S2) must load, S3 must not (its coverage does
    // not intersect {10}).
    let scope_f10 = RoaringBitmap::from_iter([10]);
    let metrics = lance_index::metrics::LocalMetricsCollector::default();
    let scoped = open_scalar_index_segments(&dataset, "i", "i_idx", Some(&scope_f10), &metrics)
        .await
        .unwrap();
    assert_eq!(
        metrics
            .index_loads
            .load(std::sync::atomic::Ordering::Relaxed),
        2,
        "exactly the two reclustered contributors load under an F10 scope; S3 is pruned"
    );

    // Opener layer: the scoped open selects WHICH segments to load; it does NOT
    // row-filter. S1 and S2 both cover F10 and F11, so a full-range search over
    // the opened index legitimately returns the COMPLETE contributor address set
    // (every reclustered row, in both destinations). This proves both
    // contributors were opened and searched, not that rows are scoped here.
    use lance_index::scalar::{SargableQuery, SearchResult};
    use std::ops::Bound;
    let result = scoped
        .search(
            &SargableQuery::Range(
                Bound::Included(datafusion::scalar::ScalarValue::Int32(Some(0))),
                Bound::Included(datafusion::scalar::ScalarValue::Int32(Some(7))),
            ),
            &metrics,
        )
        .await
        .unwrap();
    let SearchResult::Exact(row_addrs) = result else {
        panic!("expected exact scalar search result");
    };
    let found: std::collections::BTreeSet<u32> = row_addrs
        .true_rows()
        .row_addrs()
        .unwrap()
        .map(|row_addr| RowAddress::from(u64::from(row_addr)).fragment_id())
        .collect();
    assert_eq!(
        found,
        std::collections::BTreeSet::from([10, 11]),
        "both contributors are opened, so the opener returns rows in BOTH destinations \
         (row-level fragment scope is the scanner's job, not the opener's)"
    );
    let mut opener_values: Vec<(u32, u32)> = row_addrs
        .true_rows()
        .row_addrs()
        .unwrap()
        .map(|row_addr| {
            let addr = RowAddress::from(u64::from(row_addr));
            (addr.fragment_id(), addr.row_offset())
        })
        .collect();
    opener_values.sort_unstable();
    assert_eq!(
        opener_values,
        vec![
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3)
        ],
        "the opener search returns every reclustered row from both contributors"
    );

    // Scanner layer: a full query that scopes to F10 must return EXACTLY the F10
    // rows (even values 0,2,4,6), excluding F11, proving the downstream consumer
    // applies the row-level fragment scope on top of the opened segments.
    let f10 = dataset
        .fragments()
        .iter()
        .find(|f| f.id == 10)
        .unwrap()
        .clone();
    let mut scan = dataset.scan();
    scan.with_fragments(vec![f10]);
    scan.filter("i >= 0").unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let mut values = batch["i"].as_primitive::<Int32Type>().values().to_vec();
    values.sort_unstable();
    assert_eq!(
        values,
        vec![0, 2, 4, 6],
        "an F10-scoped scanner query returns exactly F10 rows, from both contributors"
    );

    // A scope disjoint from the effective coverage prunes every segment.
    assert!(
        open_scalar_index_segments(
            &dataset,
            "i",
            "i_idx",
            Some(&RoaringBitmap::from_iter([999])),
            &lance_index::metrics::NoOpMetricsCollector,
        )
        .await
        .is_err(),
        "a scope disjoint from effective coverage must prune every segment"
    );
}

// Combination: the derived-listing cache holds the FULL effective listing per
// snapshot; each scoped query filters an independent copy. Querying one
// destination then the other on the same snapshot must both be correct: the
// first (cold) query populates the cache, the second (warm) query must not be
// served a listing trimmed to the first scope.
#[tokio::test]
async fn fragment_scope_cache_hit_is_scope_independent() {
    let mut dataset = fixture_with_index(IndexType::BTree).await;
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;

    // After the parity recluster the destinations both hold predicate-satisfying
    // rows: F10 = even values {0,2,4,6}, F11 = odd values {1,3,5,7}.
    let fragment = |id: u32| {
        dataset
            .fragments()
            .iter()
            .find(|f| f.id as u32 == id)
            .unwrap()
            .clone()
    };

    // Collect the sorted `i` values a scan returns for one destination fragment,
    // with the scalar index either used or disabled. Every query runs on the SAME
    // dataset, so the first (F10) query populates the derived-listing cache and
    // the second (F11) query hits it.
    let values_for = |frag_id: u32, use_scalar: bool| {
        let dataset = dataset.clone();
        async move {
            let mut scan = dataset.scan();
            scan.with_fragments(vec![fragment(frag_id)]);
            scan.filter("i >= 0").unwrap();
            scan.use_scalar_index(use_scalar);
            let batch = scan.try_into_batch().await.unwrap();
            let mut values = batch["i"].as_primitive::<Int32Type>().values().to_vec();
            values.sort_unstable();
            values
        }
    };

    // F10 first (cold cache), then F11 (warm cache). Each scoped, index-using
    // result must equal the same fragment scanned with the index disabled, by
    // value (row identity + multiplicity). A cache poisoned by the F10 scope would
    // make the F11 result wrong.
    let f10_index = values_for(10, true).await;
    let f10_scan = values_for(10, false).await;
    let f11_index = values_for(11, true).await;
    let f11_scan = values_for(11, false).await;

    assert_eq!(f10_index, f10_scan, "F10 scoped result must match a scan");
    assert_eq!(
        f11_index, f11_scan,
        "F11 scoped result must match a scan after the F10 query populated the cache"
    );
    assert_eq!(f10_index, vec![0, 2, 4, 6], "{f10_index:?}");
    assert_eq!(f11_index, vec![1, 3, 5, 7], "{f11_index:?}");
    // No cross-scope leak: F10's rows never appear in F11's result and vice versa.
    assert!(f10_index.iter().all(|v| v % 2 == 0));
    assert!(f11_index.iter().all(|v| v % 2 == 1));

    // The derived listing is still served from the cache (same Arc) across calls.
    let first = dataset.load_indices().await.unwrap();
    let second = dataset.load_indices().await.unwrap();
    assert!(Arc::ptr_eq(&first, &second));
}

// P1 guard witness: index coverage PROJECTED onto live destinations {10,11}
// while the segment's STORED addresses are the retired sources F0/F1, and the
// transition that would map F0/F1 -> F10/F11 is DROPPED (unknown field). Direct
// coverage of the live destinations would be granted, but the translator cannot
// map the pre-transition addresses, so those rows would be dropped and the scan
// fallback suppressed. The third-state guard must instead grant NO coverage, so
// the query scans and matches an index-disabled scan. Verified cold and warm,
// and this fails without the guard.
#[tokio::test]
async fn projected_bitmap_with_dropped_transition_scans_not_drops() {
    let mut dataset = fixture_with_index(IndexType::BTree).await;

    let (mut transition, destinations) = prepare(&dataset).await;
    transition.mapping = None;
    let mut raw = transition.encode_to_vec();
    raw.extend(field(17, b"future mapping"));
    install(&mut dataset, field(2, &raw), destinations, false).await;

    // Project the index coverage onto the live destination fragments while its
    // stored addresses remain the retired sources. This is the identity-eligible
    // "direct coverage of live fragments" shape the guard must reject.
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

    // Truth: an index-disabled scan of the reclustered snapshot.
    let mut truth = Vec::new();
    for value in 0..8 {
        let mut scan = dataset.scan();
        scan.filter(&format!("i = {value}")).unwrap();
        scan.use_scalar_index(false);
        truth.push(scan.try_into_batch().await.unwrap().num_rows());
    }

    // Index-using queries must match the truth, both cold (first derive) and warm
    // (cache hit). Without the guard, coverage is granted, the translator drops
    // the F0/F1 addresses, and these counts come up short.
    for _pass in 0..2 {
        for (value, &expected) in truth.iter().enumerate() {
            let mut scan = dataset.scan();
            scan.filter(&format!("i = {value}")).unwrap();
            assert_eq!(
                scan.try_into_batch().await.unwrap().num_rows(),
                expected,
                "value {value}"
            );
        }
    }
}

// Shared fixture for the vector and FTS prefilter tests: eight rows in two
// fragments, `vector = [i, 0, 0, 0]` so distances are deterministic, and every
// row's `text` is "hit" so a text query matches all rows.
fn scoped_prefilter_batch() -> RecordBatch {
    use arrow_array::types::Float32Type;
    use arrow_array::{ArrayRef, FixedSizeListArray, Int32Array, StringArray};
    let i: Vec<i32> = (0..8).collect();
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        i.iter()
            .map(|v| Some(vec![Some(*v as f32), Some(0.0), Some(0.0), Some(0.0)])),
        4,
    );
    RecordBatch::try_from_iter(vec![
        ("i", Arc::new(Int32Array::from(i)) as ArrayRef),
        ("vector", Arc::new(vectors) as ArrayRef),
        (
            "text",
            Arc::new(StringArray::from(vec!["hit"; 8])) as ArrayRef,
        ),
    ])
    .unwrap()
}

// Build the non-trivial subset-scope scenario: a scalar index on `i` created
// BEFORE the recluster (so it is FRI-translated and its effective coverage is
// both destinations {10,11}), then a search index (`build` closure) created
// AFTER the recluster over destination F10 ONLY. With `fast_search`, the search
// range is exactly F10 while the scalar prefilter covers F10 and F11, and F11
// holds rows that satisfy the scalar predicate.
async fn scoped_prefilter_dataset<F>(build: F) -> Dataset
where
    F: for<'a> FnOnce(
        &'a mut Dataset,
    )
        -> std::pin::Pin<Box<dyn std::future::Future<Output = IndexMetadata> + 'a>>,
{
    let batch = scoped_prefilter_batch();
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
            &["i"],
            IndexType::BTree,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let (transition, destinations) = prepare(&dataset).await;
    assert_eq!(
        destinations.iter().map(|f| f.id as u32).collect::<Vec<_>>(),
        vec![10, 11]
    );
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    let direct = build(&mut dataset).await;
    let mut indices = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    indices.push(direct);
    persist_fixture(&mut dataset, indices).await;
    dataset
}

// Combination: vector search with a scalar prefilter over an FRI-reclustered
// dataset, with a NON-TRIVIAL subset scope. The vector index covers only F10
// and `fast_search` restricts the search to that range, so the scalar prefilter
// (scanner `partition_frags_by_coverage` -> `with_fragment_scope({10})`) runs
// under a scope strictly smaller than its {10,11} effective coverage. F11 holds
// rows that satisfy the predicate AND whose vectors are closer to the query, so
// any leak of F11 rows, or filtering applied only after top-k, shows up in the
// result. The control keeps the vector index and only switches the scalar
// prefilter to its scan-based form (it is not an index-free search).
#[tokio::test]
async fn vector_prefilter_scopes_to_subset_with_fri_translation() {
    let dataset = scoped_prefilter_dataset(|dataset| {
        Box::pin(async move {
            let vparams = crate::index::vector::VectorIndexParams::ivf_flat(
                1,
                lance_linalg::distance::DistanceType::L2,
            );
            CreateIndexBuilder::new(dataset, &["vector"], IndexType::Vector, &vparams)
                .name("vector_idx".into())
                .fragments(vec![10])
                .execute_uncommitted()
                .await
                .unwrap()
        })
    })
    .await;

    // Query sits on i = 7 (in F11): distances are 7 -> 0, 6 -> 1, 5 -> 2, 4 -> 3.
    // Predicate `i >= 4` matches {4,6} in F10 and {5,7} in F11.
    let query = arrow_array::Float32Array::from(vec![7.0f32, 0.0, 0.0, 0.0]);
    let run = |k: usize, use_scalar: bool| {
        let query = query.clone();
        let dataset = dataset.clone();
        async move {
            let mut scan = dataset.scan();
            scan.nearest("vector", &query, k).unwrap();
            scan.fast_search();
            scan.filter("i >= 4").unwrap();
            scan.prefilter(true);
            scan.use_scalar_index(use_scalar);
            let batch = scan.try_into_batch().await.unwrap();
            let mut values = batch["i"].as_primitive::<Int32Type>().values().to_vec();
            values.sort_unstable();
            values
        }
    };

    // The query reaches the scoped scalar prefilter feeding ANN.
    let plan = {
        let mut scan = dataset.scan();
        scan.nearest("vector", &query, 8).unwrap();
        scan.fast_search();
        scan.filter("i >= 4").unwrap();
        scan.prefilter(true);
        scan.explain_plan(false).await.unwrap()
    };
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
    assert!(plan.contains("ANN"), "{plan}");

    // All F10 matches are kept and no F11 match leaks, even though F11's
    // vectors are the closest.
    assert_eq!(run(8, true).await, vec![4, 6]);
    // Filtering happens before top-k: the single nearest row is 6 (F10), not
    // the globally closest 7 (F11).
    assert_eq!(run(1, true).await, vec![6]);
    // Scan-based scalar prefilter control (vector index still used).
    assert_eq!(run(8, false).await, vec![4, 6]);
}

// Combination: full-text search with a scalar prefilter under the same
// non-trivial subset scope. The inverted index covers only F10 and
// `fast_search` restricts the search range to it, so the FRI-translated scalar
// prefilter (effective coverage {10,11}) runs scoped to {10}. Every row's text
// matches, and F11 holds predicate-satisfying rows, so any F11 leak shows up.
#[tokio::test]
async fn fts_prefilter_scopes_to_subset_with_fri_translation() {
    let dataset = scoped_prefilter_dataset(|dataset| {
        Box::pin(async move {
            let direct = CreateIndexBuilder::new(
                dataset,
                &["text"],
                IndexType::Inverted,
                &lance_index::scalar::InvertedIndexParams::default(),
            )
            .name("text_idx".into())
            .fragments(vec![10])
            .execute_uncommitted()
            .await
            .unwrap();
            // An uncommitted inverted index only has staged partition files; the
            // CreateIndex commit normally merges them into the root layout the
            // reader probes (`metadata.lance`). A CreateIndex commit is refused
            // on a tagged history at this layer and the fixture persists the
            // manifest directly, so run that finalize step here.
            use crate::dataset::index::LanceIndexStoreExt;
            let store = lance_index::scalar::lance_format::LanceIndexStore::from_dataset_for_new(
                dataset,
                &direct.uuid,
            )
            .unwrap();
            let index_dir = dataset.indices_dir().join(direct.uuid.to_string());
            lance_index::scalar::inverted::builder::merge_index_files(
                dataset.object_store.as_ref(),
                &index_dir,
                Arc::new(store),
                Arc::new(crate::index::NoopIndexBuildProgress),
            )
            .await
            .unwrap();
            direct
        })
    })
    .await;

    let run = |use_scalar: bool| {
        let dataset = dataset.clone();
        async move {
            let mut scan = dataset.scan();
            scan.full_text_search(lance_index::scalar::FullTextSearchQuery::new("hit".into()))
                .unwrap();
            scan.fast_search();
            scan.filter("i >= 4").unwrap();
            scan.prefilter(true);
            scan.use_scalar_index(use_scalar);
            let batch = scan.try_into_batch().await.unwrap();
            let mut values = batch["i"].as_primitive::<Int32Type>().values().to_vec();
            values.sort_unstable();
            values
        }
    };

    // The query reaches the scoped scalar prefilter feeding the text search.
    let plan = {
        let mut scan = dataset.scan();
        scan.full_text_search(lance_index::scalar::FullTextSearchQuery::new("hit".into()))
            .unwrap();
        scan.fast_search();
        scan.filter("i >= 4").unwrap();
        scan.prefilter(true);
        scan.explain_plan(false).await.unwrap()
    };
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
    assert!(plan.contains("MatchQuery"), "{plan}");

    // All F10 matches, no F11 leak; scan-based scalar prefilter control agrees.
    assert_eq!(run(true).await, vec![4, 6]);
    assert_eq!(run(false).await, vec![4, 6]);
}

// ---- cache identity: translated entries follow the translation state ----

/// Open `i_idx` through the real scalar path, returning how many index loads
/// it took (0 = served from the cache) and the cached container.
async fn open_i_idx(dataset: &Dataset) -> (usize, Arc<dyn lance_index::scalar::ScalarIndex>) {
    let index = dataset.load_index_by_name("i_idx").await.unwrap().unwrap();
    let (loads, container) = open_segment(dataset, &index).await;
    (
        loads,
        container.expect("a translating open leaves its container in the cache"),
    )
}

/// `(index loads, cached container)`: the container is absent on a plugin's
/// non-batch path when it caches state instead of a whole object.
async fn open_segment(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> (usize, Option<Arc<dyn lance_index::scalar::ScalarIndex>>) {
    let metrics = lance_index::metrics::LocalMetricsCollector::default();
    crate::index::scalar::open_scalar_index(dataset, "i", index, &metrics)
        .await
        .unwrap();
    let container = crate::index::scalar::cached_scalar_index_container(dataset, &index.uuid).await;
    (
        metrics
            .index_loads
            .load(std::sync::atomic::Ordering::Relaxed),
        container,
    )
}

fn same_container(
    a: &Arc<dyn lance_index::scalar::ScalarIndex>,
    b: &Arc<dyn lance_index::scalar::ScalarIndex>,
) -> bool {
    std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
}

async fn count(dataset: &Dataset, value: i32) -> usize {
    dataset
        .count_rows(Some(format!("i = {value}")))
        .await
        .unwrap()
}

/// A translating `i_idx` over the reclustered fixture, persisted so later
/// commits (appends, deletes, new histories) build on a real snapshot.
async fn persisted_translating_fixture() -> Dataset {
    let mut dataset = fixture().await;
    let (transition, destinations) = prepare(&dataset).await;
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    let indices = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    persist_fixture(&mut dataset, indices).await;
    let plan = dataset
        .scan()
        .filter("i = 2")
        .unwrap()
        .explain_plan(false)
        .await
        .unwrap();
    assert!(plan.contains("ScalarIndexQuery"), "{plan}");
    dataset
}

fn fri_entry(indices: &[IndexMetadata]) -> &IndexMetadata {
    indices
        .iter()
        .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
        .unwrap()
}

async fn append_rows(dataset: &mut Dataset, values: &[i32]) -> lance_table::format::Fragment {
    let batch = RecordBatch::try_from_iter([(
        "i",
        Arc::new(arrow_array::Int32Array::from(values.to_vec())) as arrow_array::ArrayRef,
    )])
    .unwrap();
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            None,
        )
        .await
        .unwrap();
    dataset
        .fragments()
        .iter()
        .max_by_key(|fragment| fragment.id)
        .unwrap()
        .clone()
}

/// Rewrite fragment `source_id` into a new fragment `dest_id` with an
/// ordered-compaction transition (rows keep their order): the transition and
/// the uncommitted destination fragment.
async fn compact_fragment(
    dataset: &Dataset,
    source_id: u64,
    dest_id: u64,
) -> (Transition, lance_table::format::Fragment) {
    let source = dataset
        .fragments()
        .iter()
        .find(|fragment| fragment.id == source_id)
        .unwrap()
        .clone();
    let batch = {
        let mut scan = dataset.scan();
        scan.with_fragments(vec![source.clone()]);
        scan.try_into_batch().await.unwrap()
    };
    let transaction = crate::dataset::InsertBuilder::new(Arc::new(dataset.clone()))
        .with_params(&WriteParams {
            mode: crate::dataset::WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(vec![batch])
        .await
        .unwrap();
    let lance_table::transaction::Operation::Append { fragments } = transaction.operation else {
        unreachable!()
    };
    let mut destination = fragments.into_iter().next().unwrap();
    destination.id = dest_id;
    let physical_rows = source.physical_rows.unwrap() as u64;
    let deleted: Vec<u32> = dataset
        .get_fragment(source_id as usize)
        .unwrap()
        .get_deletion_vector()
        .await
        .unwrap()
        .map(|vector| vector.iter().collect())
        .unwrap_or_default();
    let mut changed_row_addrs = Vec::new();
    roaring::RoaringTreemap::from_iter(
        (0..physical_rows as u32)
            .filter(|row| !deleted.contains(row))
            .map(|row| u64::from(RowAddress::new_from_parts(source_id as u32, row))),
    )
    .serialize_into(&mut changed_row_addrs)
    .unwrap();
    let transition = Transition {
        sources: vec![FragmentDigest {
            id: source_id,
            physical_rows,
            num_deleted_rows: deleted.len() as u64,
        }],
        destinations: vec![FragmentDigest {
            id: dest_id,
            physical_rows: destination.physical_rows.unwrap() as u64,
            num_deleted_rows: 0,
        }],
        mapping: Some(transition::Mapping::OrderedCompaction(
            pb::fragment_reuse_index_details::OrderedCompaction { changed_row_addrs },
        )),
    };
    (transition, destination)
}

/// Persist a snapshot over exactly the `live` fragments with every index
/// entry, the FRI included, left as it is.
async fn persist_live(dataset: &mut Dataset, live: Vec<lance_table::format::Fragment>) {
    let indices = crate::index::load_all_indices(dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    Arc::make_mut(&mut dataset.manifest).fragments = live.into();
    persist_fixture(dataset, indices).await;
}

/// Replace the FRI entry with a fresh one (new UUID) holding `transitions`,
/// over exactly the `live` fragments, and persist the snapshot. Stored
/// segment provenance is taken from the raw listing, as a real rewrite
/// commit would keep it.
async fn install_history(
    dataset: &mut Dataset,
    transitions: Vec<Transition>,
    live: Vec<lance_table::format::Fragment>,
) -> IndexMetadata {
    let mut indices = crate::index::load_all_indices(dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    indices.retain(|index| index.name != FRAG_REUSE_INDEX_NAME);
    let content = InlineContent {
        legacy_versions: vec![],
        transitions,
    }
    .encode_to_vec();
    let fri = IndexMetadata {
        uuid: Uuid::new_v4(),
        fields: vec![],
        covering_fields: vec![],
        name: FRAG_REUSE_INDEX_NAME.into(),
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(live.iter().map(|fragment| fragment.id as u32).collect()),
        index_details: Some(Arc::new(prost_types::Any {
            type_url: "/lance.table.FragmentReuseIndexDetails".into(),
            value: field(1, &content),
        })),
        index_version: 1,
        created_at: None,
        base_id: None,
        files: None,
    };
    indices.push(fri.clone());
    Arc::make_mut(&mut dataset.manifest).fragments = live.into();
    persist_fixture(dataset, indices).await;
    fri
}

// An append adds fragments no stored address refers to: the segment's
// coverage, exclusions and mapping path are unchanged, so its translated
// entries stay warm. Before this keying, every commit cold-started them.
#[tokio::test]
async fn tagged_append_keeps_translated_entries_warm() {
    let mut dataset = persisted_translating_fixture().await;
    let (loads, before) = open_i_idx(&dataset).await;
    assert_eq!(loads, 1, "cold open loads the index");
    assert_eq!(count(&dataset, 2).await, 1);

    append_rows(&mut dataset, &[100]).await;
    let (loads, after) = open_i_idx(&dataset).await;
    assert_eq!(loads, 0, "an append must not evict translated entries");
    assert!(same_container(&before, &after));
    assert_eq!(count(&dataset, 2).await, 1);
    assert_eq!(count(&dataset, 100).await, 1, "the appended row is scanned");
}

// Dropping a live destination changes the segment's coverage: the entry must
// go cold even though the FRI entry (and its UUID) is untouched.
#[tokio::test]
async fn coverage_change_misses_without_new_fri() {
    let mut dataset = persisted_translating_fixture().await;
    let (loads, _) = open_i_idx(&dataset).await;
    assert_eq!(loads, 1);
    let fri_before = fri_entry(&crate::index::load_all_indices(&dataset).await.unwrap()).uuid;

    // F10 holds the even values. Drop it from the manifest (what deleting all
    // of its rows does once tagged maintenance lands) with the FRI entry left
    // exactly as it is.
    let live: Vec<_> = dataset
        .fragments()
        .iter()
        .filter(|fragment| fragment.id != 10)
        .cloned()
        .collect();
    persist_live(&mut dataset, live).await;
    assert!(
        !dataset.fragment_bitmap.contains(10),
        "{:?}",
        dataset.fragment_bitmap
    );
    let fri_after = fri_entry(&crate::index::load_all_indices(&dataset).await.unwrap()).uuid;
    assert_eq!(fri_before, fri_after);

    let (loads, _) = open_i_idx(&dataset).await;
    assert_eq!(loads, 1, "a coverage change must miss");
    for value in 0..8 {
        assert_eq!(count(&dataset, value).await, usize::from(value % 2 == 1));
    }
}

// A new transition over a fragment on the segment's path changes the path:
// the entry must go cold, and the new hop must be applied.
#[tokio::test]
async fn relevant_transition_misses_translated_entries() {
    let mut dataset = persisted_translating_fixture().await;
    let (loads, _) = open_i_idx(&dataset).await;
    assert_eq!(loads, 1);
    let first = decode_transitions(&dataset).await;
    assert_eq!(first.len(), 1);

    let (second, destination) = compact_fragment(&dataset, 10, 20).await;
    let live = vec![
        dataset
            .fragments()
            .iter()
            .find(|f| f.id == 11)
            .unwrap()
            .clone(),
        destination,
    ];
    let mut transitions = first;
    transitions.push(second);
    install_history(&mut dataset, transitions, live).await;

    let (loads, _) = open_i_idx(&dataset).await;
    assert_eq!(loads, 1, "a transition on the path must miss");
    for value in 0..8 {
        assert_eq!(count(&dataset, value).await, 1, "value {value}");
    }
}

// A new history (fresh FRI UUID) whose extra transition rewrites fragments
// the segment never covered leaves the segment's translation identity
// unchanged: its entries stay warm. This pins the "relevant state only"
// rule; keying on the FRI UUID would cold-start every segment here.
#[tokio::test]
async fn unrelated_transition_keeps_translated_entries_warm() {
    let mut dataset = persisted_translating_fixture().await;
    let (loads, before) = open_i_idx(&dataset).await;
    assert_eq!(loads, 1);
    let fri_before = fri_entry(&crate::index::load_all_indices(&dataset).await.unwrap()).uuid;
    let first = decode_transitions(&dataset).await;

    let appended = append_rows(&mut dataset, &[100, 101]).await;
    let (second, destination) = compact_fragment(&dataset, appended.id, 20).await;
    let mut live: Vec<_> = dataset
        .fragments()
        .iter()
        .filter(|fragment| fragment.id != appended.id)
        .cloned()
        .collect();
    live.push(destination);
    let mut transitions = first;
    transitions.push(second);
    let fri_after = install_history(&mut dataset, transitions, live).await.uuid;
    assert_ne!(fri_before, fri_after);

    let (loads, after) = open_i_idx(&dataset).await;
    assert_eq!(
        loads, 0,
        "a transition off the segment's path must not evict its entries"
    );
    assert!(same_container(&before, &after));
    for value in 0..8 {
        assert_eq!(count(&dataset, value).await, 1, "value {value}");
    }
    assert_eq!(count(&dataset, 100).await, 1);
    assert_eq!(count(&dataset, 101).await, 1);
}

/// The transitions of the dataset's current (inline) FRI history.
async fn decode_transitions(dataset: &Dataset) -> Vec<Transition> {
    let indices = crate::index::load_all_indices(dataset).await.unwrap();
    let fri = fri_entry(&indices);
    let details = fri.index_details.as_ref().unwrap();
    let details = pb::FragmentReuseIndexDetails::decode(details.value.as_slice()).unwrap();
    let Some(pb::fragment_reuse_index_details::Content::Inline(inline)) = details.content else {
        panic!("fixture histories are inline");
    };
    inline.transitions
}

// An untouched segment loads without a remapper, so its entries depend only
// on the index file: two appends later they are still warm.
#[tokio::test]
async fn identity_segment_survives_two_appends() {
    use crate::index::frag_reuse::{ResolvedRemapping, open_row_id_remapping};

    let mut dataset = fixture().await;
    let (transition, mut destinations) = prepare(&dataset).await;
    let untouched_fragment = append_rows(&mut dataset, &[100]).await;
    let untouched = CreateIndexBuilder::new(
        &mut dataset,
        &["i"],
        IndexType::BTree,
        &ScalarIndexParams::default(),
    )
    .name("i_untouched".into())
    .replace(true)
    .fragments(vec![untouched_fragment.id as u32])
    .execute_uncommitted()
    .await
    .unwrap();
    destinations.push(untouched_fragment);
    let content = InlineContent {
        legacy_versions: vec![],
        transitions: vec![transition],
    }
    .encode_to_vec();
    install(&mut dataset, content, destinations, false).await;
    let mut all = crate::index::load_all_indices(&dataset)
        .await
        .unwrap()
        .as_ref()
        .clone();
    all.push(untouched.clone());
    persist_fixture(&mut dataset, all).await;
    let resolved = open_row_id_remapping(&dataset, &untouched, &NoOpMetricsCollector)
        .await
        .unwrap();
    assert!(matches!(resolved, Some((_, ResolvedRemapping::V1Identity))));

    let (loads, before) = open_segment(&dataset, &untouched).await;
    assert_eq!(loads, 1);
    assert_eq!(count(&dataset, 100).await, 1);
    append_rows(&mut dataset, &[200]).await;
    append_rows(&mut dataset, &[300]).await;
    let (loads, after) = open_segment(&dataset, &untouched).await;
    assert_eq!(loads, 0, "identity entries depend only on the index file");
    if let (Some(before), Some(after)) = (&before, &after) {
        assert!(same_container(before, after));
    }
    assert_eq!(count(&dataset, 100).await, 1);
    assert_eq!(count(&dataset, 300).await, 1);
}

// A legacy-layout full-text index (tokens, postings and documents at the
// index root, no metadata file) keeps its per-document lengths aligned with
// the row ids it was built with, so it cannot be translated under a tagged
// history. The tagged reader excludes the segment from coverage, and a real
// full-text query on the tagged table returns exactly what an index-free scan
// returns: neither misaligned scores nor an error.
#[tokio::test]
async fn legacy_layout_fts_on_tagged_table_is_excluded_and_scans() {
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
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            Some("text_idx".into()),
            &lance_index::scalar::InvertedIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let index = dataset
        .load_index_by_name("text_idx")
        .await
        .unwrap()
        .unwrap();

    // Rewrite the single-partition index into the legacy layout: the
    // partition files move to the root names and the metadata file goes.
    let index_dir = dataset.indices_dir().join(index.uuid.to_string());
    for name in ["tokens.lance", "invert.lance", "docs.lance"] {
        let from = index_dir.clone().join(format!("part_0_{name}"));
        let to = index_dir.clone().join(name);
        dataset.object_store.copy(&from, &to).await.unwrap();
        dataset.object_store.delete(&from).await.unwrap();
    }
    dataset
        .object_store
        .delete(&index_dir.clone().join("metadata.lance"))
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

    // Excluded from coverage: no usable segment is advertised for the column.
    let usable =
        crate::index::scalar_logical::load_named_scalar_segments(&dataset, "text", "text_idx")
            .await
            .unwrap();
    assert!(usable.is_empty(), "{usable:?}");

    // The truth: an index-free scan of the tagged snapshot.
    let mut scan = dataset.scan();
    scan.filter("text = 'even'").unwrap();
    scan.use_scalar_index(false);
    let expected = scan
        .try_into_batch()
        .await
        .unwrap()
        .column_by_name("i")
        .unwrap()
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(expected, std::collections::BTreeSet::from([0, 2, 6]));

    // A real full-text query on the tagged table matches it, cold and warm.
    for _ in 0..2 {
        let mut scan = dataset.scan();
        // The column is named: a column-less query discovers its columns from
        // the usable segments, and this snapshot advertises none.
        scan.full_text_search(
            lance_index::scalar::FullTextSearchQuery::new("even".into())
                .with_column("text".into())
                .unwrap(),
        )
        .unwrap();
        let actual = scan
            .try_into_batch()
            .await
            .unwrap()
            .column_by_name("i")
            .unwrap()
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(actual, expected);
    }
}
