// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::record_batch;

#[tokio::test]
async fn test_plan_upsert() {
    let data = lance_datagen::gen_batch()
        .with_seed(Seed::from(1))
        .col("value", array::step::<UInt32Type>())
        .col("key", array::rand_pseudo_uuid_hex());
    let data = data.into_reader_rows(RowCount::from(1024), BatchCount::from(32));
    let _schema = data.schema();

    // Create dataset with initial data
    let ds = Dataset::write(data, "memory://", None).await.unwrap();

    // Create upsert job
    let merge_insert_job =
        crate::dataset::MergeInsertBuilder::try_new(Arc::new(ds), vec!["key".to_string()])
            .unwrap()
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .try_build()
            .unwrap();

    // Create new data for upsert
    let new_data = lance_datagen::gen_batch()
        .with_seed(Seed::from(2))
        .col("value", array::step::<UInt32Type>())
        .col("key", array::rand_pseudo_uuid_hex());
    let new_data = new_data.into_reader_rows(RowCount::from(512), BatchCount::from(16));
    let new_data_stream = reader_to_stream(Box::new(new_data));

    let plan = merge_insert_job
        .create_plan(one_shot_provider(new_data_stream).unwrap())
        .await
        .unwrap();

    // Assert the plan structure using portable plan matching
    // The optimized plan should have:
    // 1. FullSchemaMergeInsertExec at the top
    // 2. ProjectionExec that creates action based on _rowaddr nullness (sentinel is constant
    //    true so DataFusion folds `sentinel IS NOT NULL` away from the CASE expression)
    // 3. HashJoin with projection that includes the sentinel column
    // 4. LanceScan that only reads the key column (projection pushdown working!)
    // 5. ProjectionExec on the source side that materializes the sentinel literal
    assert_plan_node_equals(
        plan,
        "MergeInsert: on=[key], when_matched=UpdateAll, when_not_matched=InsertAll, when_not_matched_by_source=Keep
  CoalescePartitionsExec
    ProjectionExec: expr=[_rowid@0 as _rowid, _rowaddr@1 as _rowaddr, value@2 as value, key@3 as key, __merge_source_sentinel@4 as __merge_source_sentinel, CASE WHEN _rowaddr@1 IS NULL THEN 2 WHEN _rowaddr@1 IS NOT NULL THEN 1 ELSE 0 END as __action]
      HashJoinExec: mode=Partitioned, join_type=Right, on=[(key@0, key@1)], projection=[_rowid@1, _rowaddr@2, value@3, key@4, __merge_source_sentinel@5]
        RepartitionExec: partitioning=Hash([key@0], ...), input_partitions=1
          LanceRead: uri=..., projection=[key], num_fragments=1, range_before=None, range_after=None, \
          row_id=true, row_addr=true, full_filter=--, refine_filter=--
        RepartitionExec: partitioning=Hash([key@1], ...), input_partitions=1
          ProjectionExec: expr=[value@0 as value, key@1 as key, true as __merge_source_sentinel]
            StreamingTableExec: partition_sizes=1, projection=[value, key]"
    ).await.unwrap();
}

/// #4583 use case 3: which side of the merge_insert hash join gets buffered
/// is decided by the source's statistics, not by the order `create_plan`
/// writes the join in. `create_plan` always puts the target on the left, so
/// without a swap the target is always the build side.
///
/// The target here is one row past DataFusion's
/// `hash_join_single_partition_threshold_rows`, and its estimated width puts it
/// past the 1 MiB byte threshold too, so the target cannot be collected on
/// either count. That leaves the source: a materialized one reports exact
/// statistics and fits under the threshold, so `JoinSelection` swaps it onto
/// the build side and rewrites `Right` into `Left`. A one-shot stream reports
/// `Absent` for everything, neither side qualifies for `CollectLeft`, and the
/// plan falls back to a partitioned join whose build side is still the target.
///
/// The one-shot provider used below stands in for every non-materialized
/// source: `stream_source_to_provider` sends the default path through
/// `spilling_table_provider`, which also hands back a `StreamingTable` and so
/// reports the same absent statistics.
///
/// This is about which side is buffered, not about how much the target reads.
/// The target scan projects `other` either way, because the row-rewrite fill
/// reads it from the target side of the join.
///
/// Both expectations characterise DataFusion's choice rather than any Lance
/// logic, and Lance sets no `hash_join_single_partition_threshold*` of its own,
/// so this rides on DataFusion's defaults (1 MiB / 128 Ki rows). A DataFusion
/// upgrade that changes them fails this test without anything in Lance
/// regressing, which is the point: the plan shape is what merge_insert's memory
/// use depends on, so a silent change to it should not go unnoticed.
#[tokio::test]
async fn test_plan_join_build_side_follows_source_statistics() {
    fn find_hash_join(plan: &dyn ExecutionPlan) -> Option<&HashJoinExec> {
        if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
            return Some(join);
        }
        for child in plan.children() {
            if let Some(join) = find_hash_join(child.as_ref()) {
                return Some(join);
            }
        }
        None
    }

    fn sides(join: &HashJoinExec) -> (String, String) {
        let render = |plan: &Arc<dyn ExecutionPlan>| {
            format!(
                "{}",
                datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
            )
        };
        (render(join.left()), render(join.right()))
    }

    // One row past datafusion.optimizer.hash_join_single_partition_threshold_rows.
    const TARGET_ROWS: u64 = 128 * 1024 + 1;

    let target = lance_datagen::gen_batch()
        .with_seed(Seed::from(1))
        .col("key", array::step::<UInt32Type>())
        .col("value", array::step::<UInt32Type>())
        .col("other", array::step::<UInt32Type>())
        .into_reader_rows(RowCount::from(TARGET_ROWS), BatchCount::from(1));
    let ds = Arc::new(Dataset::write(target, "memory://", None).await.unwrap());

    // Partial schema: the source omits `other`, so the row-rewrite fill makes
    // the target scan read it. In the streaming half below, where the target is
    // the build side, that means it is held for every buffered row. This test
    // asserts which side is the build side, not the projection.
    let source = record_batch!(
        ("key", UInt32, [0, 1, 2, 3]),
        ("value", UInt32, [10, 11, 12, 13])
    )
    .unwrap();

    let new_job = || {
        crate::dataset::MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
            .unwrap()
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
    };

    let materialized: Arc<dyn TableProvider> = Arc::new(
        datafusion::datasource::MemTable::try_new(source.schema(), vec![vec![source.clone()]])
            .unwrap(),
    );
    let plan = new_job().create_plan(materialized).await.unwrap();
    let join = find_hash_join(plan.as_ref()).expect("materialized source must plan a hash join");
    let (build, probe) = sides(join);
    assert_eq!(
        (*join.partition_mode(), *join.join_type()),
        (PartitionMode::CollectLeft, JoinType::Left),
        "the target is past the collect threshold and the source is not, so the inputs \
             are swapped and Right is rewritten to Left. build side was:\n{build}"
    );
    assert!(
        build.contains("DataSourceExec") && !build.contains("LanceRead"),
        "the source must be the collected side:\n{build}"
    );
    assert!(
        probe.contains("LanceRead"),
        "the target must be the probe side, which is the side a hash join offers its \
             dynamic filter to:\n{probe}"
    );

    let reader = RecordBatchIterator::new([Ok(source.clone())], source.schema());
    let stream_plan = new_job()
        .create_plan(one_shot_provider(reader_to_stream(Box::new(reader))).unwrap())
        .await
        .unwrap();
    let join = find_hash_join(stream_plan.as_ref()).expect("stream source must plan a hash join");
    let (build, probe) = sides(join);
    assert_eq!(
        (*join.partition_mode(), *join.join_type()),
        (PartitionMode::Partitioned, JoinType::Right),
        "the target is past the collect threshold and the source reports no statistics, \
             so neither side qualifies and both are hash-repartitioned. build side was:\n{build}"
    );
    assert!(
        build.contains("LanceRead"),
        "the target stays the build side, so every one of its rows is \
             buffered:\n{build}"
    );
    assert!(
        probe.contains("StreamingTableExec"),
        "the source stays the probe side:\n{probe}"
    );
}

/// The same build-side choice, now driven by a width measured from the target's
/// string column instead of the decoder's cold seed. The target is small enough
/// to be collected under the seed, while its decoded values are larger than the
/// materialized source and the collection threshold.
#[tokio::test]
async fn test_plan_join_build_side_uses_measured_widths() {
    use crate::dataset::builder::DatasetBuilder;

    fn find_hash_join(plan: &dyn ExecutionPlan) -> Option<&HashJoinExec> {
        if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
            return Some(join);
        }
        for child in plan.children() {
            if let Some(join) = find_hash_join(child.as_ref()) {
                return Some(join);
            }
        }
        None
    }

    fn sides(join: &HashJoinExec) -> (String, String) {
        let render = |plan: &Arc<dyn ExecutionPlan>| {
            format!(
                "{}",
                datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
            )
        };
        (render(join.left()), render(join.right()))
    }

    let wide: &'static str = Box::leak("x".repeat(4_096).into_boxed_str());
    let tmp = TempStrDir::default();
    let target = lance_datagen::gen_batch()
        .with_seed(Seed::from(1))
        .col("key", array::step::<UInt32Type>())
        .col("value", array::step::<UInt32Type>())
        .col("payload", array::cycle_utf8_literals(&[wide]))
        .into_reader_rows(RowCount::from(1_024), BatchCount::from(1));
    let written = Dataset::write(target, tmp.as_str(), None).await.unwrap();
    let ds = Arc::new(
        DatasetBuilder::from_uri(written.uri())
            .with_session(written.session())
            .load()
            .await
            .unwrap(),
    );

    // Larger than the target's cold seed estimate (1,024 rows x ~72 B) but
    // smaller than its measured payload (1,024 x 4 KB).
    let source = lance_datagen::gen_batch()
        .with_seed(Seed::from(2))
        .col("key", array::step::<UInt32Type>())
        .col("value", array::step::<UInt32Type>())
        .into_batch_rows(RowCount::from(20_000))
        .unwrap();
    let new_job = || {
        crate::dataset::MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
            .unwrap()
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
    };
    let materialized = || {
        Arc::new(MemTable::try_new(source.schema(), vec![vec![source.clone()]]).unwrap())
            as Arc<dyn TableProvider>
    };

    let cold_plan = new_job().create_plan(materialized()).await.unwrap();
    let cold_join = find_hash_join(cold_plan.as_ref()).expect("cold plan must have a hash join");
    let (cold_build, _) = sides(cold_join);
    assert_eq!(
        (*cold_join.partition_mode(), *cold_join.join_type()),
        (PartitionMode::CollectLeft, JoinType::Right),
        "the seed makes the target look smaller than the source:\n{cold_build}"
    );
    assert!(cold_build.contains("LanceRead"), "{cold_build}");

    ds.scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let warm_plan = new_job().create_plan(materialized()).await.unwrap();
    let warm_join = find_hash_join(warm_plan.as_ref()).expect("warm plan must have a hash join");
    let (warm_build, warm_probe) = sides(warm_join);
    assert_eq!(
        (*warm_join.partition_mode(), *warm_join.join_type()),
        (PartitionMode::CollectLeft, JoinType::Left),
        "the measured width must make the source the smaller, collected side:\n{warm_build}"
    );
    assert!(
        warm_build.contains("DataSourceExec") && !warm_build.contains("LanceRead"),
        "the materialized source must be the build side:\n{warm_build}"
    );
    assert!(
        warm_probe.contains("LanceRead"),
        "the wide target must be the probe side:\n{warm_probe}"
    );
}

#[tokio::test]
async fn test_first_seen_native_source_matches_ordered_scanner() {
    let source_batch = record_batch!(
        ("key", UInt32, [2, 2, 3, 3, 5]),
        ("value", UInt32, [100, 200, 400, 500, 600])
    )
    .unwrap();
    let source = Arc::new(
        Dataset::write(
            RecordBatchIterator::new([Ok(source_batch.clone())], source_batch.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    let initial =
        record_batch!(("key", UInt32, [1, 2, 3]), ("value", UInt32, [10, 20, 30])).unwrap();
    let mut results = Vec::new();
    for native in [true, false] {
        let target = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(initial.clone())], initial.schema()),
                "memory://",
                None,
            )
            .await
            .unwrap(),
        );
        let job = crate::dataset::MergeInsertBuilder::try_new(target, vec!["key".into()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .source_dedupe_behavior(SourceDedupeBehavior::FirstSeen)
            .try_build()
            .unwrap();
        let (result, stats) = if native {
            job.execute_dataset(source.clone()).await.unwrap()
        } else {
            job.execute(source.scan().try_into_stream().await.unwrap().into())
                .await
                .unwrap()
        };
        assert_eq!(stats.num_skipped_duplicates, 2);
        let batch = result.scan().try_into_batch().await.unwrap();
        let keys = batch.column(0).as_primitive::<UInt32Type>();
        let values = batch.column(1).as_primitive::<UInt32Type>();
        let mut rows = keys
            .values()
            .iter()
            .zip(values.values())
            .map(|(&k, &v)| (k, v))
            .collect::<Vec<_>>();
        rows.sort_unstable();
        results.push(rows);
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[0], vec![(1, 10), (2, 100), (3, 400), (5, 600)]);
}

/// `analyze_plan` is a diagnostic, so it has to report the plan the source it
/// was handed would actually run. The batches entry point must therefore not
/// fall back to the streaming plan: with the row counts used below the join
/// collects the materialized source and rewrites the join type, and a stream
/// gets neither. Which side wins is a size comparison, not a property of the
/// entry point; see the fixture comment.
#[tokio::test]
async fn test_analyze_plan_reports_the_given_source_shape() {
    let data = lance_datagen::gen_batch()
        .with_seed(Seed::from(1))
        .col("key", array::step::<UInt32Type>())
        .col("value", array::step::<UInt32Type>())
        .into_reader_rows(RowCount::from(64), BatchCount::from(1));
    let ds = Arc::new(Dataset::write(data, "memory://", None).await.unwrap());

    // The source covers the dataset's schema, so nothing is filled from the
    // target side. Two rows against the target's 64 keep the source smaller
    // in estimated bytes as well as row count, so the join collects it.
    let source = record_batch!(("key", UInt32, [1, 100]), ("value", UInt32, [999, 999])).unwrap();

    let new_job = || {
        crate::dataset::MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
            .unwrap()
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .when_not_matched(crate::dataset::WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
    };

    let materialized = new_job()
        .analyze_plan_batches(vec![source.clone()])
        .await
        .unwrap();
    assert!(
        materialized.contains("DataSourceExec") && !materialized.contains("StreamingTableExec"),
        "materialized batches must be reported as an in-memory source:\n{materialized}"
    );
    assert!(
        materialized.contains("join_type=Left"),
        "collecting the source, which is the smaller side here, rewrites the join type:\n{materialized}"
    );

    // The provider entry is public too, and the batches entry is a thin wrapper
    // over it, so pin it directly rather than only through that wrapper.
    let provider: Arc<dyn TableProvider> = Arc::new(
        datafusion::datasource::MemTable::try_new(source.schema(), vec![vec![source.clone()]])
            .unwrap(),
    );
    let from_provider = new_job().analyze_plan_provider(provider).await.unwrap();
    assert!(
        from_provider.contains("DataSourceExec") && from_provider.contains("join_type=Left"),
        "a provider with exact statistics reports the same shape as its batches:\n{from_provider}"
    );

    let reader = RecordBatchIterator::new([Ok(source.clone())], source.schema());
    let streaming = new_job()
        .analyze_plan(reader_to_stream(Box::new(reader)))
        .await
        .unwrap();
    assert!(
        streaming.contains("StreamingTableExec") && !streaming.contains("DataSourceExec"),
        "a stream must still be reported as a stream:\n{streaming}"
    );
    assert!(
        streaming.contains("join_type=Right"),
        "nothing is swapped without source statistics:\n{streaming}"
    );
}

#[tokio::test]
async fn test_fast_path_update_only() {
    let data = lance_datagen::gen_batch()
        .with_seed(Seed::from(1))
        .col("value", array::step::<UInt32Type>())
        .col("key", array::rand_pseudo_uuid_hex());
    let data = data.into_reader_rows(RowCount::from(1024), BatchCount::from(32));

    // Create dataset with initial data
    let ds = Dataset::write(data, "memory://", None).await.unwrap();

    // Create update-only job (insert_not_matched = false)
    let merge_insert_job =
        crate::dataset::MergeInsertBuilder::try_new(Arc::new(ds), vec!["key".to_string()])
            .unwrap()
            .when_matched(crate::dataset::WhenMatched::UpdateAll)
            .when_not_matched(crate::dataset::WhenNotMatched::DoNothing)
            .try_build()
            .unwrap();

    // Create new data for update
    let new_data = lance_datagen::gen_batch()
        .with_seed(Seed::from(2))
        .col("value", array::step::<UInt32Type>())
        .col("key", array::rand_pseudo_uuid_hex());
    let new_data = new_data.into_reader_rows(RowCount::from(512), BatchCount::from(16));
    let new_data_stream = reader_to_stream(Box::new(new_data));

    // This should use the fast path (execute_uncommitted_v2)
    let plan = merge_insert_job
        .create_plan(one_shot_provider(new_data_stream).unwrap())
        .await
        .unwrap();

    // The optimized plan should use Inner join instead of Right join since we're not
    // inserting unmatched rows.  The sentinel IS NOT NULL condition is folded away by
    // DataFusion because the sentinel is lit(true), so the CASE only checks _rowaddr.
    assert_plan_node_equals(
        plan,
        "MergeInsert: on=[key], when_matched=UpdateAll, when_not_matched=DoNothing, when_not_matched_by_source=Keep
  CoalescePartitionsExec
    ProjectionExec: expr=[_rowid@0 as _rowid, _rowaddr@1 as _rowaddr, value@2 as value, key@3 as key, __merge_source_sentinel@4 as __merge_source_sentinel, CASE WHEN _rowaddr@1 IS NOT NULL THEN 1 ELSE 0 END as __action]
      HashJoinExec: mode=Partitioned, join_type=Inner, on=[(key@0, key@1)], projection=[_rowid@1, _rowaddr@2, value@3, key@4, __merge_source_sentinel@5]
        RepartitionExec: partitioning=Hash([key@0], ...), input_partitions=1
          LanceRead: uri=..., projection=[key], num_fragments=1, range_before=None, range_after=None, row_id=true, row_addr=true, full_filter=--, refine_filter=--
        RepartitionExec...
          ProjectionExec: expr=[value@0 as value, key@1 as key, true as __merge_source_sentinel]
            StreamingTableExec: partition_sizes=1, projection=[value, key]"
    ).await.unwrap();
}

#[tokio::test]
async fn test_fast_path_conditional_update() {
    let data = lance_datagen::gen_batch()
        .with_seed(Seed::from(1))
        .col("value", array::step::<UInt32Type>())
        .col("key", array::rand_pseudo_uuid_hex());
    let data = data.into_reader_rows(RowCount::from(1024), BatchCount::from(32));

    // Create dataset with initial data
    let ds = Dataset::write(data, "memory://", None).await.unwrap();

    // Create conditional update job (WhenMatched::UpdateIf)
    let merge_insert_job =
        crate::dataset::MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["key".to_string()])
            .unwrap()
            .when_matched(crate::dataset::WhenMatched::update_if(&ds, "source.value > 20").unwrap())
            .when_not_matched(crate::dataset::WhenNotMatched::DoNothing)
            .try_build()
            .unwrap();

    // Create new data for conditional update
    let new_data = lance_datagen::gen_batch()
        .with_seed(Seed::from(2))
        .col("value", array::step::<UInt32Type>())
        .col("key", array::rand_pseudo_uuid_hex());
    let new_data_reader = new_data.into_reader_rows(RowCount::from(512), BatchCount::from(16));
    let new_data_stream = reader_to_stream(Box::new(new_data_reader));

    let plan = merge_insert_job
        .create_plan(one_shot_provider(new_data_stream).unwrap())
        .await
        .unwrap();

    // The optimized plan should use Inner join and include the UpdateIf condition.
    // The sentinel IS NOT NULL condition is folded away (sentinel is lit(true)).
    assert_plan_node_equals(
        plan,
        "MergeInsert: on=[key], when_matched=UpdateIf(source.value > 20), when_not_matched=DoNothing, when_not_matched_by_source=Keep
  CoalescePartitionsExec
    ProjectionExec: expr=[_rowid@0 as _rowid, _rowaddr@1 as _rowaddr, value@2 as value, key@3 as key, __merge_source_sentinel@4 as __merge_source_sentinel, CASE WHEN _rowaddr@1 IS NOT NULL AND value@2 > 20 THEN 1 ELSE 0 END as __action]
      HashJoinExec: mode=Partitioned, join_type=Inner, on=[(key@0, key@1)], projection=[_rowid@1, _rowaddr@2, value@3, key@4, __merge_source_sentinel@5]
        RepartitionExec: partitioning=Hash([key@0], ...), input_partitions=1
          LanceRead: uri=..., projection=[key], num_fragments=1, range_before=None, range_after=None, row_id=true, row_addr=true, full_filter=--, refine_filter=--
        RepartitionExec...
          ProjectionExec: expr=[value@0 as value, key@1 as key, true as __merge_source_sentinel]
            StreamingTableExec: partition_sizes=1, projection=[value, key]"
    ).await.unwrap();
}

/// Verifies that a default find-or-create merge insert
/// (`WhenMatched::DoNothing` + `WhenNotMatched::InsertAll`) is routed
/// through the v2 `FullSchemaMergeInsertExec` path. Prior to this
/// change, `can_use_create_plan` rejected `DoNothing` outright and the
/// operation fell back to the legacy v1 `Merger`; the assertion below
/// would fail on `main`. See lance-format/lance#6441.
#[tokio::test]
async fn test_fast_path_find_or_create() {
    let data = lance_datagen::gen_batch()
        .with_seed(Seed::from(1))
        .col("value", array::step::<UInt32Type>())
        .col("key", array::rand_pseudo_uuid_hex());
    let data = data.into_reader_rows(RowCount::from(1024), BatchCount::from(32));

    // Create dataset with initial data
    let ds = Dataset::write(data, "memory://", None).await.unwrap();

    // Default MergeInsertBuilder config is find-or-create:
    //   when_matched = DoNothing, when_not_matched = InsertAll.
    let merge_insert_job =
        crate::dataset::MergeInsertBuilder::try_new(Arc::new(ds), vec!["key".to_string()])
            .unwrap()
            .try_build()
            .unwrap();

    // Source data with a mix of already-present and new keys.
    let new_data = lance_datagen::gen_batch()
        .with_seed(Seed::from(2))
        .col("value", array::step::<UInt32Type>())
        .col("key", array::rand_pseudo_uuid_hex());
    let new_data = new_data.into_reader_rows(RowCount::from(512), BatchCount::from(16));
    let new_data_stream = reader_to_stream(Box::new(new_data));

    // Should reach the v2 fast path (`create_plan` + FullSchemaMergeInsertExec).
    // Dropping to v1 here would return an error from create_plan instead.
    let plan = merge_insert_job
        .create_plan(one_shot_provider(new_data_stream).unwrap())
        .await
        .unwrap();

    // The join is Right because we keep unmatched source rows (InsertAll)
    // but discard unmatched target rows (DoNothing on when_matched,
    // Keep on when_not_matched_by_source). The CASE expression simplifies
    // to `_rowaddr IS NULL → Insert, else Nothing`.
    assert_plan_node_equals(
        plan,
        "MergeInsert: on=[key], when_matched=DoNothing, when_not_matched=InsertAll, when_not_matched_by_source=Keep
  CoalescePartitionsExec
    ProjectionExec: expr=[_rowid@0 as _rowid, _rowaddr@1 as _rowaddr, value@2 as value, key@3 as key, __merge_source_sentinel@4 as __merge_source_sentinel, CASE WHEN _rowaddr@1 IS NULL THEN 2 ELSE 0 END as __action]
      HashJoinExec: mode=Partitioned, join_type=Right, on=[(key@0, key@1)], projection=[_rowid@1, _rowaddr@2, value@3, key@4, __merge_source_sentinel@5]
        RepartitionExec: partitioning=Hash([key@0], ...), input_partitions=1
          LanceRead: uri=..., projection=[key], num_fragments=1, range_before=None, range_after=None, row_id=true, row_addr=true, full_filter=--, refine_filter=--
        RepartitionExec...
          ProjectionExec: expr=[value@0 as value, key@1 as key, true as __merge_source_sentinel]
            StreamingTableExec: partition_sizes=1, projection=[value, key]"
    )
    .await
    .unwrap();
}

/// Test case for Issue #5323: merge_insert should use the full schema path
/// when columns are provided in a different order than the dataset schema.
#[tokio::test]
async fn test_merge_insert_reordered_columns() {
    use arrow_array::record_batch;

    let initial_data = record_batch!(
        ("id", Int32, [1, 2, 3]),
        ("value", Float64, [1.1, 2.2, 3.3]),
        ("extra", Utf8, ["a", "b", "c"])
    )
    .unwrap();

    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(initial_data.clone())], initial_data.schema()),
        "memory://test_issue_5323",
        None,
    )
    .await
    .unwrap();

    // Source data with reordered columns: [extra, id, value] instead of [id, value, extra]
    let new_data = record_batch!(
        ("extra", Utf8, ["x", "y"]),
        ("id", Int32, [2, 4]), // id 2 exists, 4 is new
        ("value", Float64, [22.2, 44.4])
    )
    .unwrap();

    // Verify reordered columns can use the fast path
    let job = MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();
    assert!(
        job.can_use_create_plan(&new_data.schema()).await.unwrap(),
        "Reordered schema should be able to use fast path"
    );

    // Execute and verify data correctness
    let (merged_dataset, _) =
        MergeInsertBuilder::try_new(Arc::new(dataset), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new(
                vec![Ok(new_data.clone())],
                new_data.schema(),
            )))
            .await
            .unwrap();

    let result = merged_dataset
        .scan()
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first(
            "id".to_string(),
        )]))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();

    let expected = record_batch!(
        ("id", Int32, [1, 2, 3, 4]),
        ("value", Float64, [1.1, 22.2, 3.3, 44.4]),
        ("extra", Utf8, ["a", "x", "c", "y"])
    )
    .unwrap();

    assert_eq!(result, expected);
}

/// Test WhenMatched::Delete with full schema source data.
/// Source contains all columns (key, value, filterme) but we only use it to identify
/// rows to delete - no data is written back.
#[rstest::rstest]
#[tokio::test]
async fn test_when_matched_delete_full_schema(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
    #[values(true, false)] enable_stable_row_ids: bool,
) {
    let schema = create_test_schema();
    let test_uri = "memory://test_delete_full.lance";

    // Create dataset with keys 1-6 (value=1)
    let ds = create_test_dataset(test_uri, version, enable_stable_row_ids).await;

    // Source data has keys 4, 5, 6, 7, 8, 9 with full schema
    // Keys 4, 5, 6 match existing rows and should be deleted
    // Keys 7, 8, 9 don't match (and we're not inserting)
    let new_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![4, 5, 6, 7, 8, 9])),
            Arc::new(UInt32Array::from(vec![2, 2, 2, 2, 2, 2])),
            Arc::new(StringArray::from(vec!["A", "B", "C", "A", "B", "C"])),
        ],
    )
    .unwrap();

    let keys = vec!["key".to_string()];

    // First, verify the execution plan structure
    // Delete-only should use Inner join and only include key columns (optimization)
    // Action 3 = Delete
    let plan_job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();
    let plan_stream = reader_to_stream(Box::new(RecordBatchIterator::new(
        [Ok(new_batch.clone())],
        schema.clone(),
    )));
    let plan = plan_job
        .create_plan(one_shot_provider(plan_stream).unwrap())
        .await
        .unwrap();
    assert_plan_node_equals(
        plan,
        "DeleteOnlyMergeInsert: on=[key], when_matched=Delete, when_not_matched=DoNothing
  ...
    HashJoinExec: ...join_type=Inner...
      ...
      ...
        StreamingTableExec: partition_sizes=1, projection=[key]",
    )
    .await
    .unwrap();
    let job = MergeInsertBuilder::try_new(ds.clone(), keys)
        .unwrap()
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();

    let new_reader = Box::new(RecordBatchIterator::new([Ok(new_batch)], schema.clone()));
    let new_stream = reader_to_stream(new_reader);

    let (merged_dataset, merge_stats) = job.execute(new_stream).await.unwrap();

    // Should have deleted 3 rows (keys 4, 5, 6)
    assert_eq!(merge_stats.num_deleted_rows, 3);
    assert_eq!(merge_stats.num_inserted_rows, 0);
    assert_eq!(merge_stats.num_updated_rows, 0);

    // Verify remaining data - only keys 1, 2, 3 should remain
    let batches = merged_dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let merged = concat_batches(&schema, &batches).unwrap();
    let mut remaining_keys: Vec<u32> = merged
        .column(0)
        .as_primitive::<UInt32Type>()
        .values()
        .to_vec();
    remaining_keys.sort();
    assert_eq!(remaining_keys, vec![1, 2, 3]);
}

/// Test WhenMatched::Delete with ID-only source data (just key column).
/// This is the optimized bulk delete case where we only need key columns for matching.
#[rstest::rstest]
#[tokio::test]
async fn test_when_matched_delete_id_only(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
    #[values(true, false)] enable_stable_row_ids: bool,
) {
    let test_uri = "memory://test_delete_id_only.lance";

    // Create dataset with keys 1-6 (full schema: key, value, filterme)
    let ds = create_test_dataset(test_uri, version, enable_stable_row_ids).await;
    let id_only_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::UInt32, true)]));
    let new_batch = RecordBatch::try_new(
        id_only_schema.clone(),
        vec![Arc::new(UInt32Array::from(vec![2, 4, 6]))], // Delete keys 2, 4, 6
    )
    .unwrap();

    let keys = vec!["key".to_string()];

    // ID-only delete should use Inner join with key-only projection
    // on=[(key@0, key@0)] because key is at position 0 in both target and source
    let plan_job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();
    let plan_stream = reader_to_stream(Box::new(RecordBatchIterator::new(
        [Ok(new_batch.clone())],
        id_only_schema.clone(),
    )));
    let plan = plan_job
        .create_plan(one_shot_provider(plan_stream).unwrap())
        .await
        .unwrap();
    assert_plan_node_equals(
        plan,
        "DeleteOnlyMergeInsert: on=[key], when_matched=Delete, when_not_matched=DoNothing
  ...
    HashJoinExec: ...join_type=Inner...
      ...
      ...
        StreamingTableExec: partition_sizes=1, projection=[key]",
    )
    .await
    .unwrap();
    let job = MergeInsertBuilder::try_new(ds.clone(), keys)
        .unwrap()
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();

    let new_reader = Box::new(RecordBatchIterator::new(
        [Ok(new_batch)],
        id_only_schema.clone(),
    ));
    let new_stream = reader_to_stream(new_reader);

    let (merged_dataset, merge_stats) = job.execute(new_stream).await.unwrap();

    // Should have deleted 3 rows (keys 2, 4, 6)
    assert_eq!(merge_stats.num_deleted_rows, 3);
    assert_eq!(merge_stats.num_inserted_rows, 0);
    assert_eq!(merge_stats.num_updated_rows, 0);

    // Verify remaining data - only keys 1, 3, 5 should remain
    let full_schema = create_test_schema();
    let batches = merged_dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let merged = concat_batches(&full_schema, &batches).unwrap();
    let mut remaining_keys: Vec<u32> = merged
        .column(0)
        .as_primitive::<UInt32Type>()
        .values()
        .to_vec();
    remaining_keys.sort();
    assert_eq!(remaining_keys, vec![1, 3, 5]);
}

/// Test WhenMatched::Delete combined with WhenNotMatched::InsertAll.
/// This replaces existing matching rows with nothing (delete) while inserting new rows.
#[rstest::rstest]
#[tokio::test]
async fn test_when_matched_delete_with_insert(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
) {
    let schema = create_test_schema();
    let test_uri = "memory://test_delete_with_insert.lance";

    // Create dataset with keys 1-6
    let ds = create_test_dataset(test_uri, version, false).await;

    // Source has keys 4, 5, 6 (match - will be deleted) and 7, 8, 9 (new - will be inserted)
    let new_batch = create_new_batch(schema.clone());

    let keys = vec!["key".to_string()];

    // Delete + Insert should use Right join to see unmatched rows for insertion
    let plan_job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();
    let plan_stream = reader_to_stream(Box::new(RecordBatchIterator::new(
        [Ok(new_batch.clone())],
        schema.clone(),
    )));
    let plan = plan_job
        .create_plan(one_shot_provider(plan_stream).unwrap())
        .await
        .unwrap();
    assert_plan_node_equals(
        plan,
        "MergeInsert: on=[key], when_matched=Delete, when_not_matched=InsertAll, when_not_matched_by_source=Keep...THEN 2 WHEN...THEN 3 ELSE 0 END as __action]...projection=[key, value, filterme]"
    ).await.unwrap();

    // Delete matched rows, insert unmatched rows
    let job = MergeInsertBuilder::try_new(ds.clone(), keys)
        .unwrap()
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();

    let new_reader = Box::new(RecordBatchIterator::new([Ok(new_batch)], schema.clone()));
    let new_stream = reader_to_stream(new_reader);

    let (merged_dataset, merge_stats) = job.execute(new_stream).await.unwrap();

    // Deleted 3 (keys 4, 5, 6), inserted 3 (keys 7, 8, 9)
    assert_eq!(merge_stats.num_deleted_rows, 3);
    assert_eq!(merge_stats.num_inserted_rows, 3);
    assert_eq!(merge_stats.num_updated_rows, 0);

    // Verify: keys 1, 2, 3 (original, not matched), 7, 8, 9 (new inserts)
    let batches = merged_dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let merged = concat_batches(&schema, &batches).unwrap();
    let mut remaining_keys: Vec<u32> = merged
        .column(0)
        .as_primitive::<UInt32Type>()
        .values()
        .to_vec();
    remaining_keys.sort();
    assert_eq!(remaining_keys, vec![1, 2, 3, 7, 8, 9]);

    // Verify values: keys 1, 2, 3 have value=1 (original), keys 7, 8, 9 have value=2 (new)
    let keyvals: Vec<(u32, u32)> = merged
        .column(0)
        .as_primitive::<UInt32Type>()
        .values()
        .iter()
        .zip(
            merged
                .column(1)
                .as_primitive::<UInt32Type>()
                .values()
                .iter(),
        )
        .map(|(&k, &v)| (k, v))
        .collect();

    for (key, value) in keyvals {
        if key <= 3 {
            assert_eq!(value, 1, "Original keys should have value=1");
        } else {
            assert_eq!(value, 2, "New keys should have value=2");
        }
    }
}

/// Test WhenMatched::Delete when source data has no matching keys.
/// This should result in zero deletes and the dataset remains unchanged.
#[rstest::rstest]
#[tokio::test]
async fn test_when_matched_delete_no_matches(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
) {
    let schema = create_test_schema();
    let test_uri = "memory://test_delete_no_matches.lance";

    // Create dataset with keys 1-6
    let ds = create_test_dataset(test_uri, version, false).await;

    // Source data has keys 100, 200, 300 - none match existing keys 1-6
    let non_matching_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![100, 200, 300])),
            Arc::new(UInt32Array::from(vec![10, 20, 30])),
            Arc::new(StringArray::from(vec!["X", "Y", "Z"])),
        ],
    )
    .unwrap();

    let keys = vec!["key".to_string()];

    // Even with no matches, the plan structure should be the same
    let plan_job = MergeInsertBuilder::try_new(ds.clone(), keys.clone())
        .unwrap()
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();
    let plan_stream = reader_to_stream(Box::new(RecordBatchIterator::new(
        [Ok(non_matching_batch.clone())],
        schema.clone(),
    )));
    let plan = plan_job
        .create_plan(one_shot_provider(plan_stream).unwrap())
        .await
        .unwrap();
    assert_plan_node_equals(
        plan,
        "DeleteOnlyMergeInsert: on=[key], when_matched=Delete, when_not_matched=DoNothing
  ...
    HashJoinExec: ...join_type=Inner...
      ...
      ...
        StreamingTableExec: partition_sizes=1, projection=[key]",
    )
    .await
    .unwrap();
    let job = MergeInsertBuilder::try_new(ds.clone(), keys)
        .unwrap()
        .when_matched(WhenMatched::Delete)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();

    let new_reader = Box::new(RecordBatchIterator::new(
        [Ok(non_matching_batch)],
        schema.clone(),
    ));
    let new_stream = reader_to_stream(new_reader);

    let (merged_dataset, merge_stats) = job.execute(new_stream).await.unwrap();

    // Should have deleted 0 rows since no keys matched
    assert_eq!(merge_stats.num_deleted_rows, 0);
    assert_eq!(merge_stats.num_inserted_rows, 0);
    assert_eq!(merge_stats.num_updated_rows, 0);

    // Verify all original data remains unchanged - keys 1-6 should all still be present
    let batches = merged_dataset
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let merged = concat_batches(&schema, &batches).unwrap();
    let mut remaining_keys: Vec<u32> = merged
        .column(0)
        .as_primitive::<UInt32Type>()
        .values()
        .to_vec();
    remaining_keys.sort();
    assert_eq!(remaining_keys, vec![1, 2, 3, 4, 5, 6]);
}

/// Test that MergeInsertPlanner::is_delete_only correctly identifies delete-only operations.
///
/// Delete-only is true only when:
/// - when_matched = Delete
/// - insert_not_matched = false (WhenNotMatched::DoNothing)
/// - delete_not_matched_by_source = Keep
///
/// This test iterates through all valid combinations of WhenMatched, WhenNotMatched,
/// and WhenNotMatchedBySource to verify the is_delete_only logic.
#[tokio::test]
async fn test_is_delete_only() {
    use itertools::iproduct;

    // All variants to test (excluding UpdateIf and DeleteIf because they require expressions)
    let when_matched_variants = [
        WhenMatched::UpdateAll,
        WhenMatched::DoNothing,
        WhenMatched::Fail,
        WhenMatched::Delete,
    ];
    let when_not_matched_variants = [WhenNotMatched::InsertAll, WhenNotMatched::DoNothing];
    let when_not_matched_by_source_variants =
        [WhenNotMatchedBySource::Keep, WhenNotMatchedBySource::Delete];

    let schema = create_test_schema();

    for (idx, (when_matched, when_not_matched, when_not_matched_by_source)) in iproduct!(
        when_matched_variants.iter().cloned(),
        when_not_matched_variants.iter().cloned(),
        when_not_matched_by_source_variants.iter().cloned()
    )
    .enumerate()
    {
        // Check if this is a valid (non-no-op) combination, since this would fail try_build()
        let is_no_op = matches!(when_matched, WhenMatched::DoNothing | WhenMatched::Fail)
            && matches!(when_not_matched, WhenNotMatched::DoNothing)
            && matches!(when_not_matched_by_source, WhenNotMatchedBySource::Keep);
        if is_no_op {
            continue;
        }

        let test_uri = format!("memory://test_is_delete_only_{}.lance", idx);
        let ds = create_test_dataset(&test_uri, LanceFileVersion::V2_0, false).await;

        let new_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![4, 5, 6])),
                Arc::new(UInt32Array::from(vec![2, 2, 2])),
                Arc::new(StringArray::from(vec!["A", "B", "C"])),
            ],
        )
        .unwrap();

        let keys = vec!["key".to_string()];

        let mut builder = MergeInsertBuilder::try_new(ds.clone(), keys).unwrap();
        builder
            .when_matched(when_matched.clone())
            .when_not_matched(when_not_matched.clone())
            .when_not_matched_by_source(when_not_matched_by_source.clone());

        let job = builder.try_build().unwrap();

        let plan_stream = reader_to_stream(Box::new(RecordBatchIterator::new(
            [Ok(new_batch)],
            schema.clone(),
        )));
        let plan = job
            .create_plan(one_shot_provider(plan_stream).unwrap())
            .await
            .unwrap();

        let plan_str = datafusion::physical_plan::displayable(plan.as_ref())
            .indent(true)
            .to_string();

        let expected_delete_only = matches!(when_matched, WhenMatched::Delete)
            && matches!(when_not_matched, WhenNotMatched::DoNothing)
            && matches!(when_not_matched_by_source, WhenNotMatchedBySource::Keep);

        if expected_delete_only {
            assert!(
                plan_str.contains("DeleteOnlyMergeInsert"),
                "Expected DeleteOnlyMergeInsert for ({:?}, {:?}, {:?}), but got:\n{}",
                when_matched,
                when_not_matched,
                when_not_matched_by_source,
                plan_str
            );
        } else {
            assert!(
                plan_str.contains("MergeInsert:") && !plan_str.contains("DeleteOnlyMergeInsert"),
                "Expected MergeInsert (not DeleteOnlyMergeInsert) for ({:?}, {:?}, {:?}), but got:\n{}",
                when_matched,
                when_not_matched,
                when_not_matched_by_source,
                plan_str
            );
        }
    }
}

/// Tests that apply_deletions correctly handles an error when applying the row deletions.
#[tokio::test]
async fn test_apply_deletions_invalid_row_address() {
    use crate::dataset::write::merge_insert::exec::apply_deletions;
    use roaring::RoaringTreemap;

    let test_uri = "memory://test_apply_deletions_error.lance";

    // Create a dataset with 2 fragments, each with 3 rows
    let ds = create_test_dataset(test_uri, LanceFileVersion::V2_0, false).await;
    let fragment_id = ds.get_fragments()[0].id() as u32;

    // Create row addresses with invalid row offsets for this fragment
    // Row address format: high 32 bits = fragment_id, low 32 bits = row_offset
    // Each fragment has only 3 rows (offsets 0, 1, 2).
    //
    // The error in extend_deletions is triggered when deletion_vector.len() >= physical_rows
    // AND at least one row ID is >= physical_rows.
    // So we need to add enough deletions (at least 3) with some being invalid (>= 3).
    let mut invalid_row_addrs = RoaringTreemap::new();
    let base = (fragment_id as u64) << 32;
    // Add 4 deletions: rows 10, 11, 12, 13 (all invalid since only rows 0-2 exist)
    for row_offset in 10..14u64 {
        invalid_row_addrs.insert(base | row_offset);
    }

    let result = apply_deletions(&ds, &invalid_row_addrs).await;

    assert!(result.is_err(), "Expected error for invalid row addresses");
    let err = result.unwrap_err();
    assert!(
        err.to_string()
            .contains("Deletion vector includes rows that aren't in the fragment"),
        "Expected 'rows that aren't in the fragment' error, got: {}",
        err
    );
}

fn id_value_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        Field::new("value", DataType::UInt32, false),
    ]))
}

fn collect_exact_row_counts(plan: &Arc<dyn ExecutionPlan>, out: &mut Vec<usize>) {
    if let Ok(stats) = StatisticsContext::new().compute(plan.as_ref(), &StatisticsArgs::new())
        && let datafusion::common::stats::Precision::Exact(n) = stats.num_rows
    {
        out.push(n);
    }
    for child in plan.children() {
        collect_exact_row_counts(child, out);
    }
}

/// Use case 3: planning against the provider exposes its exact source
/// statistics to the optimizer.
#[tokio::test]
async fn test_merge_insert_source_statistics_in_plan() {
    let schema = id_value_schema();
    let target_rows = 1000u32;
    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..target_rows)),
            Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(
                0,
                target_rows as usize,
            ))),
        ],
    )
    .unwrap();
    let dataset = Arc::new(
        InsertBuilder::new("memory://")
            .execute(vec![initial])
            .await
            .unwrap(),
    );

    // A small source whose exact row count is distinct from the target's.
    let source_rows = 10usize;
    let new_data = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..source_rows as u32)),
            Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(
                1,
                source_rows,
            ))),
        ],
    )
    .unwrap();
    let provider: Arc<dyn TableProvider> =
        Arc::new(MemTable::try_new(schema.clone(), vec![vec![new_data]]).unwrap());

    let job = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .try_build()
        .unwrap();

    // The provider's exact row count reaches the plan's statistics.
    let plan = job.create_plan(provider).await.unwrap();
    let mut row_counts = Vec::new();
    collect_exact_row_counts(&plan, &mut row_counts);
    assert!(
        row_counts.contains(&source_rows),
        "source provider's exact row count ({source_rows}) should reach the plan; got {row_counts:?}"
    );
}

fn list_dataset_files(base_dir: &str) -> Vec<std::path::PathBuf> {
    fn walk(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, files);
            } else {
                files.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(std::path::Path::new(base_dir), &mut files);
    files.sort();
    files
}

/// `analyze_plan` executes the merge without committing it, so every file
/// the execution wrote is unreferenced and must be removed afterwards.
/// Each case drives a different write node: the full-schema upsert writes a
/// new fragment plus deletion files, the in-place update writes a patch
/// file onto existing fragments, and the delete-only path writes only
/// deletion files.
#[derive(Debug, Clone, Copy)]
enum AnalyzedWriteNode {
    FullSchema,
    InPlace,
    DeleteOnly,
}

#[rstest::rstest]
#[case::full_schema_upsert(AnalyzedWriteNode::FullSchema)]
#[case::in_place_update(AnalyzedWriteNode::InPlace)]
#[case::delete_only(AnalyzedWriteNode::DeleteOnly)]
#[tokio::test]
async fn test_analyze_plan_removes_written_files(#[case] write_node: AnalyzedWriteNode) {
    let tempdir = TempStrDir::default();
    let data = lance_datagen::gen_batch()
        .col("key", array::step::<UInt32Type>())
        .col("value", array::step::<UInt32Type>())
        .col("other", array::cycle_utf8_literals(&["a", "b"]))
        .into_reader_rows(RowCount::from(4), BatchCount::from(2));
    let write_params = WriteParams {
        max_rows_per_file: 4,
        ..Default::default()
    };
    let dataset = Arc::new(
        Dataset::write(data, &tempdir, Some(write_params))
            .await
            .unwrap(),
    );
    assert_eq!(dataset.get_fragments().len(), 2);
    let files_before = list_dataset_files(&tempdir);

    // Keys 1 and 5 hit both fragments; key 100 is new.
    let builder = || MergeInsertBuilder::try_new(dataset.clone(), vec!["key".to_string()]).unwrap();
    let (job, source, expected_node) = match write_node {
        AnalyzedWriteNode::FullSchema => (
            builder()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::InsertAll)
                .try_build()
                .unwrap(),
            record_batch!(
                ("key", UInt32, [1, 5, 100]),
                ("value", UInt32, [10, 50, 1000]),
                ("other", Utf8, ["x", "y", "z"])
            )
            .unwrap(),
            "MergeInsert:",
        ),
        AnalyzedWriteNode::InPlace => (
            builder()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing)
                .write_mode(MergeInsertWriteMode::RewriteColumns)
                .try_build()
                .unwrap(),
            record_batch!(("key", UInt32, [1, 5]), ("value", UInt32, [10, 50])).unwrap(),
            "InPlaceMergeInsert:",
        ),
        AnalyzedWriteNode::DeleteOnly => (
            builder()
                .when_matched(WhenMatched::Delete)
                .when_not_matched(WhenNotMatched::DoNothing)
                .try_build()
                .unwrap(),
            record_batch!(("key", UInt32, [1, 5])).unwrap(),
            "DeleteOnlyMergeInsert:",
        ),
    };

    let analysis = job.analyze_plan_batches(vec![source]).await.unwrap();
    assert!(
        analysis.trim_start().starts_with(expected_node),
        "expected the {expected_node} write node:\n{analysis}"
    );

    assert_eq!(list_dataset_files(&tempdir), files_before);
    let mut dataset = (*dataset).clone();
    dataset.checkout_latest().await.unwrap();
    assert_eq!(dataset.version().version, 1);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 8);
}

/// With managed blobs enabled, Blob v2 payloads above the inline threshold
/// live in their own `_blobs/<uuid>.blob` objects rather than beside the
/// data file, so removing the analyzed data files alone leaves them behind. Repeated
/// analyses must not accumulate those objects, and the committed blobs the
/// target rows still reference must stay readable.
#[rstest::rstest]
#[case::packed(65 * 1024)]
#[case::dedicated(4 * 1024 * 1024 + 1)]
#[tokio::test]
async fn test_analyze_plan_removes_written_blob_payloads(#[case] payload_size: usize) {
    let tempdir = TempStrDir::default();
    let schema = Arc::new(crate::dataset::blob::test_util::managed_blob_schema(vec![
        Field::new("id", DataType::Int64, true),
        crate::blob_field("payload", true),
    ]));
    let make_batch = |ids: Vec<i64>, value: u8| {
        let payload = vec![value; payload_size];
        let mut blobs = crate::BlobArrayBuilder::new(ids.len());
        for _ in &ids {
            blobs.push_bytes(&payload).unwrap();
        }
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(ids)), blobs.finish().unwrap()],
        )
        .unwrap()
    };
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(make_batch(vec![0, 1], 1))], schema.clone()),
            &tempdir,
            Some(WriteParams {
                max_rows_per_file: 1,
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    assert_eq!(dataset.get_fragments().len(), 2);
    let files_before = list_dataset_files(&tempdir);
    let blobs_dir = std::path::Path::new(tempdir.as_ref() as &str).join("_blobs");
    assert!(
        files_before.iter().any(|path| path.starts_with(&blobs_dir)),
        "the fixture must store its payloads as managed objects: {files_before:?}"
    );

    let job = MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();
    for _ in 0..2 {
        let analysis = job
            .analyze_plan_batches(vec![make_batch(vec![0, 2], 2)])
            .await
            .unwrap();
        assert!(
            analysis.trim_start().starts_with("MergeInsert:"),
            "expected the full-schema write node:\n{analysis}"
        );
    }

    assert_eq!(list_dataset_files(&tempdir), files_before);
    assert_eq!(dataset.version().version, 1);
    for blob in dataset
        .take_blobs_by_indices(&[0, 1], "payload")
        .await
        .unwrap()
    {
        assert_eq!(
            blob.unwrap().read().await.unwrap().as_ref(),
            vec![1; payload_size]
        );
    }
}
