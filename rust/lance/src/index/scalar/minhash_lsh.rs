// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use futures::future::try_join_all;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::index_files_to_table;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::minhash_lsh::{MinHashLshIndex, merge_minhash_indices};
use lance_table::format::IndexMetadata;
use uuid::Uuid;

use crate::index::append::build_per_segment_filters;
use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

/// Merge one caller-defined group of MinHash LSH segments into a single segment.
///
/// Rows of fragments that no longer exist are dropped through per-segment
/// filters; the merged segment is rebuilt from the stored signatures without
/// tokenizing the column again.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    if segments.is_empty() {
        return Err(Error::index("No segment metadata was provided".to_string()));
    }

    let field_id = *segments[0].fields.first().ok_or_else(|| {
        Error::invalid_input(format!(
            "CreateIndex: segment {} is missing field ids",
            segments[0].uuid
        ))
    })?;
    let field_path = dataset.schema().field_path(field_id)?;
    let dataset_version = segments
        .iter()
        .map(|segment| segment.dataset_version)
        .min()
        .unwrap_or(dataset.manifest.version);
    // The merged segment is always written in the current (row-address)
    // format with each source's identifiers copied as-is, so a pre-migration
    // segment's row ids would be misread as addresses wherever the two differ.
    if dataset.manifest.uses_stable_row_ids()
        && let Some(legacy) = segments
            .iter()
            .find(|segment| !segment.results_are_row_addrs())
    {
        return Err(Error::invalid_input(format!(
            "MinHash LSH segment {} predates row-address-domain support and cannot be merged \
             on a dataset with stable row IDs; rebuild the index first (e.g. with \
             create_index(..., replace: true))",
            legacy.uuid,
        )));
    }
    let segment_refs: Vec<&IndexMetadata> = segments.iter().collect();
    // Every segment here stores row addresses so the filters are built
    // in the address domain.
    let (fragment_bitmap, filters) =
        build_per_segment_filters(dataset, &segment_refs, staged, true).await?;

    let scalar_indices = try_join_all(segments.iter().map(|segment| {
        let field_path = &field_path;
        async move {
            let scalar_index = super::open_scalar_index_with_plan(
                dataset,
                field_path,
                segment,
                staged.and_then(|plans| plans.get(&segment.uuid)),
                crate::index::frag_reuse::OpenPurpose::Maintenance,
                &NoOpMetricsCollector,
            )
            .await?;
            Ok::<_, Error>((segment.uuid, scalar_index))
        }
    }))
    .await?;
    let mut sources = Vec::with_capacity(scalar_indices.len());
    for ((segment_uuid, scalar_index), filter) in scalar_indices.iter().zip(&filters) {
        let minhash_index = scalar_index
            .as_any()
            .downcast_ref::<MinHashLshIndex>()
            .ok_or_else(|| {
                Error::index(format!(
                    "merge_existing_index_segments: expected MinHash LSH segment {}, got {:?}",
                    segment_uuid,
                    scalar_index.index_type()
                ))
            })?;
        sources.push((minhash_index, filter.as_ref()));
    }

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
    let created_index = merge_minhash_indices(&sources, &new_store).await?;

    Ok(IndexMetadata {
        uuid: new_uuid,
        dataset_version,
        fragment_bitmap: Some(fragment_bitmap),
        index_details: Some(Arc::new(created_index.index_details)),
        index_version: created_index.index_version as i32,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: Some(index_files_to_table(created_index.files)),
        clustering_id: None,
        metadata: Default::default(),
        ..segments[0].clone()
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, UInt64Type};
    use arrow_array::{
        ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StringArray, UInt64Array,
    };
    use arrow_schema::{DataType, Field, Schema};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::IndexType;
    use lance_index::optimize::OptimizeOptions;
    use lance_index::scalar::index_files_to_table;
    use lance_index::scalar::lance_format::LanceIndexStore;
    use lance_index::scalar::minhash_lsh::{
        BANDS_FILENAME, MINHASH_LSH_INDEX_VERSION, MinHashQuery, SIGNATURES_FILENAME,
    };
    use lance_index::scalar::{BuiltinIndexType, IndexStore, ScalarIndexParams};
    use uuid::Uuid;

    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::index::LanceIndexStoreExt;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::{CreateIndexBuilder, DatasetIndexExt};
    use crate::{Dataset, Error};

    const INDEX_NAME: &str = "text_minhash";

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
        ]))
    }

    /// Rows form one near-duplicate cluster that differs only in the row number,
    /// so every row is its own exact match and its neighbors are close.
    fn row_text(id: i32) -> String {
        format!(
            "row number {id} shares one long tail of words alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        )
    }

    fn batch(ids: std::ops::Range<i32>) -> RecordBatch {
        let texts: Vec<String> = ids.clone().map(row_text).collect();
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from_iter_values(ids)) as ArrayRef,
                Arc::new(StringArray::from(texts)) as ArrayRef,
            ],
        )
        .unwrap()
    }

    fn params() -> ScalarIndexParams {
        ScalarIndexParams::for_builtin(BuiltinIndexType::MinHashLsh)
            .with_params(&serde_json::json!({"num_hashes": 32, "num_bands": 8}))
    }

    async fn write(uri: &str, ids: std::ops::Range<i32>, rows_per_file: usize) -> Dataset {
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch(ids))], schema()),
            uri,
            Some(WriteParams {
                max_rows_per_file: rows_per_file,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    async fn append(uri: &str, ids: std::ops::Range<i32>) -> Dataset {
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch(ids))], schema()),
            uri,
            Some(WriteParams {
                max_rows_per_file: 4,
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    async fn search_ids(dataset: &Dataset, id: i32, limit: usize) -> Vec<i32> {
        let mut scan = dataset.scan();
        scan.minhash_search(MinHashQuery::new(row_text(id), "text"))
            .unwrap()
            .limit(Some(limit as i64), None)
            .unwrap()
            .project(&["id"])
            .unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        batch["id"].as_primitive::<Int32Type>().values().to_vec()
    }

    async fn explain(dataset: &Dataset) -> String {
        let mut scan = dataset.scan();
        scan.minhash_search(MinHashQuery::new(row_text(0), "text"))
            .unwrap()
            .limit(Some(1), None)
            .unwrap();
        scan.explain_plan(false).await.unwrap()
    }

    #[tokio::test]
    async fn test_distributed_segments_commit_merge_and_query() {
        let mut dataset = write("memory://", 0..12, 4).await;
        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 3);

        let mut staged = Vec::with_capacity(fragments.len());
        for fragment in &fragments {
            staged.push(
                CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::MinHashLsh, &params())
                    .name(INDEX_NAME.to_string())
                    .fragments(vec![fragment.id() as u32])
                    .execute_uncommitted()
                    .await
                    .unwrap(),
            );
        }
        dataset
            .commit_existing_index_segments(INDEX_NAME, "text", staged.clone())
            .await
            .unwrap();
        assert_eq!(
            dataset
                .load_indices_by_name(INDEX_NAME)
                .await
                .unwrap()
                .len(),
            3
        );
        assert!(explain(&dataset).await.contains("segments=3"));
        for id in [0, 5, 11] {
            assert_eq!(search_ids(&dataset, id, 1).await, vec![id]);
        }
        let neighbors = search_ids(&dataset, 5, 12).await;
        assert_eq!(neighbors[0], 5);
        assert!(
            neighbors.len() > 1,
            "cluster rows must be candidates: {neighbors:?}"
        );

        // A segment built with different parameters cannot join the index
        let drifted_params = ScalarIndexParams::for_builtin(BuiltinIndexType::MinHashLsh)
            .with_params(&serde_json::json!({"num_hashes": 32, "num_bands": 8, "shingle_size": 4}));
        let drifted = CreateIndexBuilder::new(
            &mut dataset,
            &["text"],
            IndexType::MinHashLsh,
            &drifted_params,
        )
        .name(INDEX_NAME.to_string())
        .replace(true)
        .fragments(vec![fragments[1].id() as u32])
        .execute_uncommitted()
        .await
        .unwrap();
        let err = dataset
            .commit_existing_index_segments(INDEX_NAME, "text", vec![drifted])
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
        assert!(err.to_string().contains("identical parameters"), "{err}");
        // Merging the committed segments yields one segment with the same answers
        let merged = dataset
            .merge_existing_index_segments(staged.clone())
            .await
            .unwrap();
        assert_eq!(
            merged.fragment_bitmap.as_ref().unwrap(),
            dataset.fragment_bitmap.as_ref()
        );
        assert!(
            merged
                .index_details
                .as_ref()
                .unwrap()
                .type_url
                .ends_with("MinHashLshIndexDetails")
        );
        dataset
            .commit_existing_index_segments(INDEX_NAME, "text", vec![merged])
            .await
            .unwrap();
        assert_eq!(
            dataset
                .load_indices_by_name(INDEX_NAME)
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(explain(&dataset).await.contains("segments=1"));
        assert_eq!(search_ids(&dataset, 5, 12).await, neighbors);
        dataset.prewarm_index(INDEX_NAME).await.unwrap();
        assert_eq!(search_ids(&dataset, 11, 1).await, vec![11]);
    }

    #[tokio::test]
    async fn test_optimize_appends_delta_then_merges() {
        let test_dir = TempStrDir::default();
        let uri = test_dir.as_str();
        let mut dataset = write(uri, 0..4, 4).await;
        dataset
            .create_index(
                &["text"],
                IndexType::MinHashLsh,
                Some(INDEX_NAME.into()),
                &params(),
                true,
            )
            .await
            .unwrap();

        // append(): the new fragment gets its own delta segment
        let mut dataset = append(uri, 4..8).await;
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
        assert_eq!(
            dataset
                .load_indices_by_name(INDEX_NAME)
                .await
                .unwrap()
                .len(),
            2
        );
        assert_eq!(search_ids(&dataset, 6, 1).await, vec![6]);
        assert_eq!(search_ids(&dataset, 1, 1).await, vec![1]);

        // default optimize: the newest segment absorbs the next fragment
        let _ = dataset;
        let mut dataset = append(uri, 8..12).await;
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
        let segments = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(segments.len(), 2);
        assert!(
            segments
                .iter()
                .any(|segment| { segment.fragment_bitmap.as_ref().unwrap().len() == 2 })
        );
        for id in [1, 6, 10] {
            assert_eq!(search_ids(&dataset, id, 1).await, vec![id]);
        }

        // merging every segment rebuilds one segment from the derived params
        let mut dataset = dataset;
        dataset
            .optimize_indices(&OptimizeOptions::merge(2))
            .await
            .unwrap();
        let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
        assert_eq!(
            dataset
                .load_indices_by_name(INDEX_NAME)
                .await
                .unwrap()
                .len(),
            1
        );
        for id in [1, 6, 10] {
            assert_eq!(search_ids(&dataset, id, 1).await, vec![id]);
        }
        assert_eq!(dataset.count_rows(None).await.unwrap(), 12);
    }

    #[tokio::test]
    async fn test_deferred_compaction_survives_merge_and_optimize() {
        // A deferred compaction leaves the stored row ids in the old fragments;
        // merges and updates must remap them the way queries do, otherwise the
        // rebuilt segment claims the compacted fragment without its rows.
        let test_dir = TempStrDir::default();
        let uri = test_dir.as_str();
        let mut dataset = write(uri, 0..6, 2).await;
        dataset
            .create_index(
                &["text"],
                IndexType::MinHashLsh,
                Some(INDEX_NAME.into()),
                &params(),
                true,
            )
            .await
            .unwrap();
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 10,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // An explicit merge of the deferred segment
        let mut dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
        let segments = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        let merged = dataset
            .merge_existing_index_segments(segments)
            .await
            .unwrap();
        dataset
            .commit_existing_index_segments(INDEX_NAME, "text", vec![merged])
            .await
            .unwrap();
        for id in 0..6 {
            assert_eq!(search_ids(&dataset, id, 1).await, vec![id]);
        }

        // An append followed by the default optimize updates the segment
        let mut dataset = append(uri, 6..7).await;
        dataset
            .optimize_indices(&OptimizeOptions::default())
            .await
            .unwrap();
        let segments = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(
            segments[0].fragment_bitmap.as_ref().unwrap(),
            dataset.fragment_bitmap.as_ref()
        );
        for id in 0..7 {
            assert_eq!(search_ids(&dataset, id, 1).await, vec![id]);
        }
    }

    #[tokio::test]
    async fn test_compaction_remaps_index() {
        let test_dir = TempStrDir::default();
        let uri = test_dir.as_str();
        let mut dataset = write(uri, 0..6, 2).await;
        assert_eq!(dataset.get_fragments().len(), 3);
        dataset
            .create_index(
                &["text"],
                IndexType::MinHashLsh,
                Some(INDEX_NAME.into()),
                &params(),
                true,
            )
            .await
            .unwrap();
        dataset.delete("id = 2").await.unwrap();

        let metrics = compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 10,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(metrics.fragments_removed > 0 && metrics.fragments_added > 0);
        let dataset = DatasetBuilder::from_uri(uri).load().await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 1);
        let segments = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(
            segments[0].fragment_bitmap.as_ref().unwrap(),
            dataset.fragment_bitmap.as_ref()
        );
        for id in [0, 3, 5] {
            assert_eq!(search_ids(&dataset, id, 1).await, vec![id]);
        }
        let neighbors = search_ids(&dataset, 2, 6).await;
        assert!(
            !neighbors.contains(&2),
            "deleted row must stay gone: {neighbors:?}"
        );
    }

    /// A merge always writes the current (row-address-domain) layout and
    /// copies each source's stored identifiers as-is. A segment persisted
    /// before that move stores row ids: on a stable-row-id dataset they differ
    /// from addresses, so merging it must be refused rather than corrupt every
    /// id; without stable row ids they coincide, so the merge succeeds and
    /// upgrades the segment to the current version. Build such a segment by
    /// hand -- the only way a real segment this old would still exist.
    #[rstest::rstest]
    #[tokio::test]
    async fn test_merge_legacy_row_id_domain_segment(
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch(0..6))], schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                enable_stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let mut segment =
            CreateIndexBuilder::new(&mut dataset, &["text"], IndexType::MinHashLsh, &params())
                .name(INDEX_NAME.into())
                .execute_uncommitted()
                .await
                .unwrap();

        // Rewrite the freshly trained segment's files as a pre-migration
        // (row-id domain) layout: rename `_rowaddr` to `_rowid`, convert each
        // stored address into the dataset's row id (with stable row ids, its
        // logical id; otherwise the address itself), and stamp file version 0.
        let source = LanceIndexStore::from_dataset_for_existing(&dataset, &segment)
            .await
            .unwrap();
        let legacy_uuid = Uuid::new_v4();
        let legacy = LanceIndexStore::from_dataset_for_new(&dataset, &legacy_uuid).unwrap();
        let mut files = Vec::new();
        for name in [BANDS_FILENAME, SIGNATURES_FILENAME] {
            let reader = source.open_index_file(name).await.unwrap();
            let batch = reader.read_range(0..reader.num_rows(), None).await.unwrap();
            let mut metadata = reader.schema().metadata.clone();
            metadata.insert("minhash_lsh_index_version".into(), "0".into());
            let old_schema = batch.schema();
            let fields = old_schema
                .fields()
                .iter()
                .map(|field| {
                    if field.name() == lance_core::ROW_ADDR {
                        Arc::new(field.as_ref().clone().with_name(lance_core::ROW_ID))
                    } else {
                        field.clone()
                    }
                })
                .collect::<Vec<_>>();
            let schema = Arc::new(Schema::new(fields));
            let mut columns = batch.columns().to_vec();
            if name == SIGNATURES_FILENAME && enable_stable_row_ids {
                let ids = batch[lance_core::ROW_ADDR].as_primitive::<UInt64Type>();
                columns[0] = Arc::new(UInt64Array::from_iter_values(
                    ids.values()
                        .iter()
                        .map(|address| (address >> 32) * 2 + (address & 0xffff_ffff)),
                ));
            }
            let mut writer = legacy.new_index_file(name, schema.clone()).await.unwrap();
            writer
                .write_record_batch(RecordBatch::try_new(schema, columns).unwrap())
                .await
                .unwrap();
            if name == BANDS_FILENAME {
                let key = "minhash_lsh_page_table_buffer";
                let buffer = reader
                    .read_global_buffer(metadata[key].parse().unwrap())
                    .await
                    .unwrap();
                let id = writer.add_global_buffer(buffer).await.unwrap();
                metadata.insert(key.into(), id.to_string());
            }
            files.push(writer.finish_with_metadata(metadata).await.unwrap());
        }
        segment.uuid = legacy_uuid;
        segment.index_version = 0;
        segment.files = Some(index_files_to_table(files));
        dataset
            .commit_existing_index_segments(INDEX_NAME, "text", vec![segment])
            .await
            .unwrap();

        // The legacy segment is still readable on its own.
        let before = search_ids(&dataset, 2, 6).await;
        assert!(before.contains(&2), "before merge: {before:?}");

        let segments = dataset.load_indices_by_name(INDEX_NAME).await.unwrap();
        let merged = dataset.merge_existing_index_segments(segments).await;
        if enable_stable_row_ids {
            // Merging must be refused outright, not silently corrupt the ids.
            let err = merged.unwrap_err();
            assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
            assert!(err.to_string().contains("cannot be merged"), "{err}");
        } else {
            let merged = merged.unwrap();
            assert_eq!(merged.index_version, MINHASH_LSH_INDEX_VERSION as i32);
            dataset
                .commit_existing_index_segments(INDEX_NAME, "text", vec![merged])
                .await
                .unwrap();
        }

        // Either way the index still answers correctly.
        assert_eq!(search_ids(&dataset, 2, 6).await, before);
    }
}
