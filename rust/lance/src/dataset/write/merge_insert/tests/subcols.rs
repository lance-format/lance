// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::rowids::{
    INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, SPILL_ROW_LINEAGE_CONFIG_KEY,
};
use lance_core::{ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION};
use lance_table::format::RowDatasetVersionMeta;
use rstest::rstest;

struct Fixtures {
    ds: Arc<Dataset>,
    new_data: RecordBatch,
}

async fn setup(scalar_index: bool) -> Fixtures {
    let data = lance_datagen::gen_batch()
        .with_seed(Seed::from(1))
        .col("other", array::rand_utf8(4.into(), false))
        .col("value", array::step::<UInt32Type>())
        .col("key", array::rand_pseudo_uuid_hex());
    let batch = data.into_batch_rows(RowCount::from(1024 + 2)).unwrap();
    let batch1 = batch.slice(0, 512);
    let batch2 = batch.slice(512, 512);
    let batch3 = batch.slice(1024, 2);
    let schema = batch.schema();

    let reader = Box::new(RecordBatchIterator::new(
        [Ok(batch1.clone())],
        schema.clone(),
    ));
    let write_params = WriteParams {
        max_rows_per_file: 256,
        max_rows_per_group: 32, // Non-standard group size to hit edge cases
        ..Default::default()
    };
    let mut ds = Dataset::write(reader, "memory://", Some(write_params.clone()))
        .await
        .unwrap();

    if scalar_index {
        let index_params = ScalarIndexParams::default();
        ds.create_index(&["key"], IndexType::Scalar, None, &index_params, false)
            .await
            .unwrap();
    }

    // Another two files, not in the scalar index (if there is one)
    let reader = Box::new(RecordBatchIterator::new(
        [Ok(batch2.clone())],
        batch2.schema(),
    ));
    ds.append(reader, Some(write_params)).await.unwrap();

    let ds = Arc::new(ds);

    // New data with only a subset of columns
    let update_schema = Arc::new(schema.project(&[2, 1]).unwrap());
    // Full second file and part of third file. Also two more new rows.
    let indices: Int64Array = (256..512).chain(600..612).chain([712, 715]).collect();
    let keys = arrow::compute::take(batch["key"].as_ref(), &indices, None).unwrap();
    let keys = arrow::compute::concat(&[&keys, &batch3["key"]]).unwrap();
    let num_rows = keys.len();
    let new_data = RecordBatch::try_new(
        update_schema,
        vec![
            keys,
            Arc::new((1024..(1024 + num_rows as u32)).collect::<UInt32Array>()),
        ],
    )
    .unwrap();

    Fixtures { ds, new_data }
}

#[tokio::test]
async fn test_delete_not_matched_by_source_on_v2_subcols() {
    // Historical note: this combination used to be rejected outright
    // on the v1 subcols path because v1 could not delete by source
    // while rewriting only a subset of columns. The v2 path handles
    // it uniformly through the action column, so the previously-
    // rejected configuration now succeeds. This test asserts the
    // successful path to keep the negative-test history explicit.
    let Fixtures { ds, new_data } = Box::pin(setup(false)).await;

    let rows_before = ds.count_rows(None).await.unwrap() as u64;

    let reader = Box::new(RecordBatchIterator::new(
        [Ok(new_data.clone())],
        new_data.schema(),
    ));

    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();
    // assert_send also pins us to the "returned future is Send"
    // contract that the previous (negative) version of this test
    // used to guard.
    let (updated_ds, stats) = assert_send(job.execute_reader(reader))
        .await
        .expect("partial-schema + delete-by-source should succeed on v2");

    // 272 rows in source — 2 of them are inserts (ignored because
    // when_not_matched is DoNothing), the other 270 match existing
    // rows and update them. The remaining 754 target rows that are
    // not matched by the source get deleted.
    assert_eq!(stats.num_updated_rows, 270);
    assert_eq!(stats.num_inserted_rows, 0);
    assert_eq!(stats.num_deleted_rows, rows_before - 270);
    assert_eq!(
        updated_ds.count_rows(None).await.unwrap() as u64,
        270,
        "only the 270 updated rows should remain after the delete-by-source"
    );
}

#[tokio::test]
async fn test_errors_on_bad_schema() {
    let Fixtures { ds, new_data } = Box::pin(setup(false)).await;

    // Schema with different names, which should be rejected.
    let bad_schema = Arc::new(Schema::new(vec![
        Field::new("wrong_key", DataType::Utf8, false),
        Field::new("wrong_value", DataType::UInt32, false),
    ]));

    // Should reject when data is not a subschema.
    let bad_batch = RecordBatch::try_new(bad_schema.clone(), new_data.columns().to_vec()).unwrap();
    let reader = Box::new(RecordBatchIterator::new([Ok(bad_batch)], bad_schema));

    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();
    let res = job.execute_reader(reader).await;
    assert!(
        matches!(
            &res,
            &Err(Error::SchemaMismatch { ref difference, .. })
                if difference.clone().contains("fields did not match")
        ),
        "Expected SchemaMismatch error, got: {:?}",
        res
    );
}

#[rstest]
#[tokio::test]
async fn test_merge_insert_subcols(
    #[values(false, true)] scalar_index: bool,
    #[values(false, true)] insert: bool,
) {
    let Fixtures { ds, new_data } = Box::pin(setup(scalar_index)).await;
    let reader = Box::new(RecordBatchIterator::new(
        [Ok(new_data.clone())],
        new_data.schema(),
    ));
    let fragments_before = ds
        .get_fragments()
        .iter()
        .map(|f| f.metadata().clone())
        .collect::<Vec<_>>();
    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(if insert {
            WhenNotMatched::InsertAll
        } else {
            WhenNotMatched::DoNothing
        })
        .try_build()
        .unwrap();

    let (ds, stats) = job.execute_reader(reader).await.unwrap();

    let fragments_after = ds
        .get_fragments()
        .iter()
        .map(|f| f.metadata().clone())
        .collect::<Vec<_>>();

    // Stats are path-independent: 272 source rows = 2 inserts
    // (if insert) + 270 updates, nothing deleted.
    assert_eq!(stats.num_updated_rows, (new_data.num_rows() - 2) as u64);
    assert_eq!(stats.num_deleted_rows, 0);
    if insert {
        assert_eq!(stats.num_inserted_rows, 2);
    } else {
        assert_eq!(stats.num_inserted_rows, 0);
    }

    if scalar_index {
        // v1 path: partial-schema upserts with a scalar index on the
        // join key fall back to the in-place Merger that rewrites
        // only the changed columns. Verify the legacy file-layout
        // optimization (tombstoned field ids + new partial data
        // files) is still produced on this path, including
        // unchanged fragment ids.
        assert_eq!(
            fragments_before.iter().map(|f| f.id).collect::<Vec<_>>(),
            fragments_after
                .iter()
                .take(fragments_before.len())
                .map(|f| f.id)
                .collect::<Vec<_>>()
        );
        assert_eq!(fragments_before[0], fragments_after[0]);
        assert_ne!(fragments_before[1], fragments_after[1]);
        assert_ne!(fragments_before[2], fragments_after[2]);
        assert_eq!(fragments_before[3], fragments_after[3]);

        let has_added_files = |frag: &Fragment| {
            assert_eq!(frag.files.len(), 2);
            let data_files = &frag.files;
            // Updated columns should be only columns in new data files
            // -2 field ids are tombstoned.
            assert_eq!(data_files[0].fields.as_ref(), &[0, -2, 2]);
            assert_eq!(data_files[1].fields.as_ref(), &[1]);
        };
        has_added_files(&fragments_after[1]);
        has_added_files(&fragments_after[2]);

        if insert {
            assert_eq!(fragments_after.len(), 5);
        } else {
            assert_eq!(fragments_after.len(), 4);
        }
    } else {
        // v2 path: partial-schema upserts run through the same
        // FullSchemaMergeInsertExec as full-schema upserts and
        // write brand-new fragments. In-place column patching is
        // opt-in (see the `test_merge_insert_subcols_in_place_*`
        // tests), so the default stays here. Fragment 1 is entirely
        // matched (all 256 rows) so it is removed; fragment 2 is
        // partially matched so it keeps its id with a deletion
        // vector; a new fragment holds the 270 updated rows (and the
        // 2 inserted rows when `insert` is set).
        let ids_after: Vec<u64> = fragments_after.iter().map(|f| f.id).collect();
        assert_eq!(
            fragments_after.len(),
            4,
            "expected [frag 0, frag 2, frag 3, new frag], got {:?}",
            ids_after
        );
        assert_eq!(
            fragments_before[0], fragments_after[0],
            "frag 0 (untouched) should be identical"
        );
        assert!(
            !ids_after.contains(&1),
            "frag 1 was fully matched by source and should have been removed"
        );
        assert!(
            ids_after.contains(&2),
            "frag 2 was only partially matched and should still be present"
        );
        assert!(
            ids_after.contains(&3),
            "frag 3 (untouched) should still be present"
        );
    }

    // Semantic data check (shared across both code paths): look
    // rows up by key so we don't depend on the scan-order mechanics
    // of v1 vs v2. For updated rows, `value` must be the new
    // source value and `other` must be the original (preserved
    // from the target). For untouched rows, both columns must be
    // unchanged.
    let data = ds
        .scan()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(data.num_rows(), if insert { 1024 + 2 } else { 1024 });
    assert_eq!(data.num_columns(), 3);

    use std::collections::HashMap;
    let other_col = data
        .column_by_name("other")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();
    let value_col = data
        .column_by_name("value")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let key_col = data
        .column_by_name("key")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();
    let mut row_by_key: HashMap<String, (u32, String)> = HashMap::new();
    for i in 0..data.num_rows() {
        row_by_key.insert(
            key_col.value(i).to_string(),
            (value_col.value(i), other_col.value(i).to_string()),
        );
    }

    // Pull original column data for reference lookups.
    let orig_batch_schema = new_data.schema();
    assert_eq!(orig_batch_schema.field(0).name(), "key");
    assert_eq!(orig_batch_schema.field(1).name(), "value");
    let new_keys = new_data
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();
    let new_values = new_data
        .column(1)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    // Every updated source row (270 of them) should be present
    // with its new value and a preserved `other` string.
    for i in 0..(new_data.num_rows() - 2) {
        let key = new_keys.value(i).to_string();
        let (value, other) = row_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("updated key {} missing from result", key));
        assert_eq!(*value, new_values.value(i));
        assert!(
            !other.is_empty(),
            "updated row for key {} should retain its original `other` value",
            key
        );
    }
    // The 2 batch3 rows at the end of new_data are inserts.
    for i in (new_data.num_rows() - 2)..new_data.num_rows() {
        let key = new_keys.value(i).to_string();
        let found = row_by_key.get(&key);
        if insert {
            let (value, _) =
                found.unwrap_or_else(|| panic!("inserted key {} missing from result", key));
            assert_eq!(*value, new_values.value(i));
        } else {
            assert!(
                found.is_none(),
                "unmatched source row for key {} must not be present when insert=false",
                key
            );
        }
    }
}

/// Verifies that `explain_plan` succeeds for a partial-schema upsert
/// and emits a plan that uses the v2 `FullSchemaMergeInsertExec`
/// path. This is the explicit acceptance criterion for #6442: the
/// partial-schema path must go through the same physical plan as
/// full-schema upserts instead of falling back to v1.
#[tokio::test]
async fn test_merge_insert_subcols_v2_explain_plan() {
    let Fixtures { ds, new_data } = Box::pin(setup(false)).await;

    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap();

    let source_schema: Schema = new_data.schema().as_ref().clone();
    let plan = job
        .explain_plan(Some(&source_schema), false)
        .await
        .expect("explain_plan must succeed for partial-schema upsert on v2");

    // The `MergeInsert: on=[...]` header is rendered by the v2
    // extension node's `fmt_for_explain` — it only appears when
    // `create_plan` was used (legacy v1 does not go through this
    // path at all). Combined with the presence of `HashJoinExec`
    // this uniquely identifies the v2 physical plan.
    assert!(
        plan.contains("MergeInsert: on=[key]"),
        "expected MergeInsert extension node in plan (v2 marker), got: {}",
        plan
    );
    assert!(
        plan.contains("HashJoinExec"),
        "expected HashJoinExec in plan, got: {}",
        plan
    );
    // Evidence that the partial-schema fix is active: the target
    // side of the join reads the `other` column (which is missing
    // from the source) and an explicit projection carries it
    // through to the write exec alongside source columns.
    assert!(
        plan.contains("LanceRead") && plan.contains("projection=[other"),
        "target-side scan should include the filled `other` column: {}",
        plan
    );
    assert!(
        plan.contains("other@0 as other"),
        "expected post-join projection to carry `other` from the target side: {}",
        plan
    );
}

/// The opt-in counterpart of `test_merge_insert_subcols_v2_explain_plan`:
/// with in-place column writes allowed, the columns absent from the
/// source must not appear anywhere in the plan. That absence is the
/// read-side half of the win — the target scan projects only the join
/// key instead of carrying the filled column through the join.
#[tokio::test]
async fn test_merge_insert_subcols_in_place_explain_plan() {
    let Fixtures { ds, new_data } = Box::pin(setup(false)).await;

    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap();

    let source_schema: Schema = new_data.schema().as_ref().clone();
    let plan = job
        .explain_plan(Some(&source_schema), false)
        .await
        .expect("explain_plan must succeed for partial-schema upsert on v2");

    assert!(
        plan.contains("InPlaceMergeInsert: on=[key]") && plan.contains("mode=RewriteColumns"),
        "expected InPlaceMergeInsert node in plan, got: {}",
        plan
    );
    assert!(
        plan.contains("HashJoinExec"),
        "expected HashJoinExec in plan, got: {}",
        plan
    );
    // `UpdateAll` references no target column, so `other` is absent
    // from the plan entirely. An `UpdateIf` whose condition reads an
    // omitted column takes it for the matched rows — see
    // `test_merge_insert_subcols_in_place_update_if_reads_omitted_column`.
    assert!(
        !plan.contains("other"),
        "column absent from the source must not be read or projected: {}",
        plan
    );
}

/// An `UpdateIf` condition that reads a column the source omits must
/// not pull that column into the target scan; it is taken for the
/// matched rows only. The write-side win is unaffected — the column
/// is read but never rewritten.
#[tokio::test]
async fn test_merge_insert_subcols_in_place_update_if_reads_omitted_column() {
    let Fixtures { ds, new_data } = Box::pin(setup(false)).await;

    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::update_if(&ds, "target.other != 'zzzz'").unwrap())
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap();

    let source_schema: Schema = new_data.schema().as_ref().clone();
    let plan = job
        .explain_plan(Some(&source_schema), false)
        .await
        .expect("explain_plan must succeed for partial-schema upsert on v2");

    assert!(
        plan.contains("InPlaceMergeInsert: on=[key]"),
        "the condition must not change the write sink: {plan}"
    );
    let scan = plan
        .lines()
        .find(|line| line.contains("LanceRead"))
        .unwrap_or_else(|| panic!("no LanceRead in plan: {plan}"));
    assert!(
        !scan.contains("other"),
        "the condition's column must not be in the target scan: {plan}"
    );
    assert!(
        plan.contains("MergeCondition:") && plan.contains("take=[other]"),
        "the condition must be evaluated on the matched rows: {plan}"
    );

    // And the condition is genuinely evaluated against the stored
    // values rather than silently seeing nulls: `other` is 4 random
    // characters, so it never equals 'zzzz' and every match updates.
    let reader = Box::new(RecordBatchIterator::new(
        [Ok(new_data.clone())],
        new_data.schema(),
    ));
    let (_, stats) = job.execute_reader(reader).await.unwrap();
    assert_eq!(stats.num_updated_rows, (new_data.num_rows() - 2) as u64);
}

/// Partial-schema upserts with `insert_not_matched=InsertAll` must
/// reject non-nullable missing columns at the API boundary instead
/// of producing a confusing downstream writer error. The user-
/// facing error message must name the offending column(s).
#[tokio::test]
async fn test_merge_insert_subcols_v2_rejects_non_nullable_insert() {
    // Build a dataset whose `other` column is explicitly non-nullable
    // so that a partial source missing it cannot safely insert new rows.
    let full_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::UInt32, true),
        Field::new("other", DataType::Utf8, false),
    ]));
    let full_batch = RecordBatch::try_new(
        full_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["k0", "k1", "k2"])),
            Arc::new(UInt32Array::from(vec![0, 1, 2])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();
    let ds = Dataset::write(
        Box::new(RecordBatchIterator::new([Ok(full_batch)], full_schema)),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let ds = Arc::new(ds);

    // Source source lacks `other` and tries to insert a new key.
    let partial_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::UInt32, true),
    ]));
    let partial_batch = RecordBatch::try_new(
        partial_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["k1", "k_new"])),
            Arc::new(UInt32Array::from(vec![11, 99])),
        ],
    )
    .unwrap();
    let reader = Box::new(RecordBatchIterator::new(
        [Ok(partial_batch)],
        partial_schema,
    ));

    let res = MergeInsertBuilder::try_new(ds, vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
        .execute_reader(reader)
        .await;

    match res {
        Err(Error::InvalidInput { source, .. }) => {
            let msg = source.to_string();
            assert!(
                msg.contains("partial-schema")
                    && msg.contains("non-nullable")
                    && msg.contains("\"other\""),
                "expected descriptive partial-schema / non-nullable error naming \
                         the `other` column, got: {}",
                msg
            );
        }
        other => panic!(
            "expected InvalidInput error for non-nullable missing column on insert path, got: {:?}",
            other
        ),
    }
}

/// Partial-schema v2 upsert must correctly handle `camelCase` column
/// names both in the join key and in a column that is *omitted* from
/// the source. DataFusion's `col()` lowercases unquoted identifiers,
/// so the partial-schema fill-in wraps the target reference in double
/// quotes (`target."<name>"`) and `on_cols` are likewise quoted. This
/// test pins that behavior down — prior to this, the v2 partial-schema
/// path had no coverage for case-sensitive column names.
#[tokio::test]
async fn test_merge_insert_subcols_v2_camel_case_column() {
    // Target dataset: camelCase join key AND a camelCase nullable
    // column that will be omitted from the source schema.
    let full_schema = Arc::new(Schema::new(vec![
        Field::new("userId", DataType::Utf8, false),
        Field::new("score", DataType::UInt32, true),
        Field::new("extraData", DataType::Utf8, true),
    ]));
    let full_batch = RecordBatch::try_new(
        full_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["u1", "u2", "u3"])),
            Arc::new(UInt32Array::from(vec![10, 20, 30])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();
    let ds = Dataset::write(
        Box::new(RecordBatchIterator::new([Ok(full_batch)], full_schema)),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let ds = Arc::new(ds);

    // Partial-schema source: no `extraData`. Updates `u2` and inserts `u_new`.
    let partial_schema = Arc::new(Schema::new(vec![
        Field::new("userId", DataType::Utf8, false),
        Field::new("score", DataType::UInt32, true),
    ]));
    let partial_batch = RecordBatch::try_new(
        partial_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["u2", "u_new"])),
            Arc::new(UInt32Array::from(vec![22, 99])),
        ],
    )
    .unwrap();
    let reader = Box::new(RecordBatchIterator::new(
        [Ok(partial_batch)],
        partial_schema,
    ));

    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["userId".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap();
    let (updated_ds, stats) = job
        .execute_reader(reader)
        .await
        .expect("camelCase partial-schema upsert must succeed on v2");

    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(stats.num_inserted_rows, 1);
    assert_eq!(stats.num_deleted_rows, 0);

    // Read the whole dataset back and index by userId so assertions
    // are independent of physical row order.
    let data = updated_ds
        .scan()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(data.num_rows(), 4);
    assert_eq!(data.num_columns(), 3);

    let user_ids = data
        .column_by_name("userId")
        .expect("camelCase join key column must be present in result")
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let scores = data
        .column_by_name("score")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let extra = data
        .column_by_name("extraData")
        .expect("camelCase omitted column must be present in result")
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    let mut by_user: std::collections::HashMap<String, (u32, Option<String>)> =
        std::collections::HashMap::new();
    for i in 0..data.num_rows() {
        let extra_val = if extra.is_null(i) {
            None
        } else {
            Some(extra.value(i).to_string())
        };
        by_user.insert(user_ids.value(i).to_string(), (scores.value(i), extra_val));
    }

    // Untouched rows: unchanged.
    assert_eq!(by_user["u1"], (10, Some("a".to_string())));
    assert_eq!(by_user["u3"], (30, Some("c".to_string())));
    // Updated row: score bumped, camelCase `extraData` preserved from target.
    assert_eq!(
        by_user["u2"],
        (22, Some("b".to_string())),
        "partial-schema update must preserve camelCase `extraData` from the target side of the join"
    );
    // Inserted row: camelCase column must be NULL (outer-join target side is NULL).
    assert_eq!(
        by_user["u_new"],
        (99, None),
        "partial-schema insert must produce NULL for omitted camelCase column"
    );
}

/// End-to-end bloom-filter conflict-detection check for a
/// partial-schema upsert. With the v2 path enabled for partial
/// schema, the returned transaction must carry a populated
/// `inserted_rows_filter` whenever the join key is an unenforced
/// primary key. Previously (v1 path) this filter was always
/// `None` for partial schema.
#[tokio::test]
async fn test_merge_insert_subcols_v2_bloom_filter() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false).with_metadata(
            vec![(
                "lance-schema:unenforced-primary-key".to_string(),
                "true".to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new("value", DataType::UInt32, true),
        Field::new("tag", DataType::Utf8, true),
    ]));
    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1, 2])),
            Arc::new(UInt32Array::from(vec![0, 0, 0])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();
    let dataset = InsertBuilder::new("memory://")
        .execute(vec![initial])
        .await
        .unwrap();
    let dataset = Arc::new(dataset);

    // Partial source — only `id` and `value`, missing nullable `tag`.
    let partial_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false).with_metadata(
            vec![(
                "lance-schema:unenforced-primary-key".to_string(),
                "true".to_string(),
            )]
            .into_iter()
            .collect(),
        ),
        Field::new("value", DataType::UInt32, true),
    ]));
    let partial = RecordBatch::try_new(
        partial_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![1, 5])), // one update (1), one insert (5)
            Arc::new(UInt32Array::from(vec![42, 99])),
        ],
    )
    .unwrap();
    let stream =
        RecordBatchStreamAdapter::new(partial_schema, futures::stream::iter(vec![Ok(partial)]));

    let UncommittedMergeInsert { transaction, .. } =
        MergeInsertBuilder::try_new(dataset.clone(), vec!["id".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap()
            .execute_uncommitted(Box::pin(stream) as SendableRecordBatchStream)
            .await
            .unwrap();

    // The committed transaction must carry the populated bloom
    // filter — this is the core conflict-detection acceptance
    // criterion for #6442.
    let committed = CommitBuilder::new(dataset.clone())
        .execute(transaction)
        .await
        .unwrap();
    let tx_path = committed
        .manifest()
        .transaction_file
        .clone()
        .expect("transaction file must be written");
    let tx_read = read_transaction_file(dataset.object_store.as_ref(), &dataset.base, &tx_path)
        .await
        .unwrap();
    match &tx_read.operation {
        Operation::Update {
            inserted_rows_filter,
            ..
        } => {
            let filter = inserted_rows_filter
                .as_ref()
                .expect("partial-schema upsert on a PK must emit a bloom filter");
            // Exactly one key field (id).
            assert_eq!(filter.field_ids.len(), 1);
        }
        other => panic!("expected Operation::Update, got: {:?}", other),
    }
}

/// An `UpdateIf` condition still routes in place, and rows the
/// condition rejects must keep their stored values rather than being
/// written back or dropped.
#[tokio::test]
async fn test_merge_insert_subcols_in_place_update_if() {
    let Fixtures { ds, new_data } = Box::pin(setup(false)).await;

    // Only rewrite rows whose existing value is below the midpoint of
    // the target's `value` range, so the condition rejects part of the
    // matched set.
    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::update_if(&ds, "target.value < 400").unwrap())
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap();

    let before = ds
        .scan()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    let fragments_before = ds
        .get_fragments()
        .iter()
        .map(|f| f.metadata().clone())
        .collect::<Vec<_>>();

    let reader = Box::new(RecordBatchIterator::new(
        [Ok(new_data.clone())],
        new_data.schema(),
    ));
    let (updated_ds, stats) = job.execute_reader(reader).await.unwrap();

    // 270 source rows match; the 256 rows of fragment 1 hold values
    // 256..512, so only those below 400 are rewritten.
    assert!(
        stats.num_updated_rows > 0 && stats.num_updated_rows < 270,
        "condition should accept some but not all matches, got {}",
        stats.num_updated_rows
    );
    assert_eq!(stats.num_deleted_rows, 0);

    let fragments_after = updated_ds
        .get_fragments()
        .iter()
        .map(|f| f.metadata().clone())
        .collect::<Vec<_>>();
    assert_eq!(
        fragments_before.iter().map(|f| f.id).collect::<Vec<_>>(),
        fragments_after.iter().map(|f| f.id).collect::<Vec<_>>()
    );

    let after = updated_ds
        .scan()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(after.num_rows(), before.num_rows());

    let index_by_key = |batch: &RecordBatch| {
        let keys = batch
            .column_by_name("key")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        let values = batch
            .column_by_name("value")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let others = batch
            .column_by_name("other")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        (0..batch.num_rows())
            .map(|i| {
                (
                    keys.value(i).to_string(),
                    (values.value(i), others.value(i).to_string()),
                )
            })
            .collect::<std::collections::HashMap<_, _>>()
    };
    let before_by_key = index_by_key(&before);
    let after_by_key = index_by_key(&after);

    let new_keys = new_data
        .column_by_name("key")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .unwrap();
    let new_values = new_data
        .column_by_name("value")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();

    let mut rewritten = 0u64;
    for i in 0..new_data.num_rows() {
        let key = new_keys.value(i).to_string();
        let Some((old_value, old_other)) = before_by_key.get(&key) else {
            continue; // one of the two insert-only rows
        };
        let (new_value, new_other) = after_by_key
            .get(&key)
            .unwrap_or_else(|| panic!("key {} disappeared", key));
        assert_eq!(
            old_other, new_other,
            "`other` is absent from the source and must never change"
        );
        if *old_value < 400 {
            assert_eq!(
                *new_value,
                new_values.value(i),
                "key {} should be updated",
                key
            );
            rewritten += 1;
        } else {
            assert_eq!(
                new_value, old_value,
                "key {} was rejected by the condition and must keep its value",
                key
            );
        }
    }
    assert_eq!(stats.num_updated_rows, rewritten);
}

/// A source that duplicates a key matches the same target row twice.
/// The in-place path must apply the same dedupe policy as the
/// row-rewrite one: fail by default, skip when told to keep the first.
#[rstest]
#[tokio::test]
async fn test_merge_insert_subcols_in_place_duplicate_keys(
    #[values(false, true)] first_seen: bool,
) {
    let Fixtures { ds, new_data } = Box::pin(setup(false)).await;

    // Repeat the first source row so two source rows carry the same key.
    let dup = arrow_select::concat::concat_batches(
        &new_data.schema(),
        &[new_data.slice(0, 1), new_data.slice(0, 1)],
    )
    .unwrap();
    let reader = Box::new(RecordBatchIterator::new([Ok(dup.clone())], dup.schema()));

    let mut builder = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()]).unwrap();
    builder
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns);
    if first_seen {
        builder.source_dedupe_behavior(SourceDedupeBehavior::FirstSeen);
    }
    let result = builder.try_build().unwrap().execute_reader(reader).await;

    if first_seen {
        let (_, stats) = result.expect("FirstSeen must skip the duplicate");
        assert_eq!(stats.num_updated_rows, 1);
        assert_eq!(stats.num_skipped_duplicates, 1);
    } else {
        let err = result.expect_err("duplicate keys must fail by default");
        assert!(
            err.to_string().contains("Ambiguous merge inserts"),
            "unexpected error: {err}"
        );
    }
}

/// With stable row ids, patching columns in place must stamp
/// `_row_last_updated_at_version` on exactly the rows it rewrote and
/// leave every other row's stamp alone. Covers both branches inside
/// `update_fragments`: a fragment the source covers entirely (the
/// direct-write path) and one it covers partially (the updater path).
#[rstest]
#[tokio::test]
async fn test_merge_insert_subcols_in_place_stable_row_id_versions(
    #[values(false, true)] full_fragment: bool,
) {
    use lance_core::ROW_LAST_UPDATED_AT_VERSION;

    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("tag", DataType::Utf8, true),
        Field::new("other", DataType::Utf8, true),
    ]));
    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..8)),
            Arc::new(StringArray::from(vec!["t"; 8])),
            Arc::new(StringArray::from(vec!["o"; 8])),
        ],
    )
    .unwrap();
    // Two fragments of 4 rows each.
    let ds = Dataset::write(
        Box::new(RecordBatchIterator::new([Ok(initial)], schema.clone())),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 4,
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(ds.version().version, 1);
    let ds = Arc::new(ds);

    // Either all of fragment 0, or two of its four rows.
    let keys: Vec<u32> = if full_fragment {
        vec![0, 1, 2, 3]
    } else {
        vec![1, 2]
    };
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("tag", DataType::Utf8, true),
    ]));
    let source = RecordBatch::try_new(
        source_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(keys.clone())),
            Arc::new(StringArray::from(vec!["patched"; keys.len()])),
        ],
    )
    .unwrap();

    let (updated_ds, stats) = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            [Ok(source)],
            source_schema,
        )))
        .await
        .unwrap();
    assert_eq!(stats.num_updated_rows, keys.len() as u64);
    let new_version = updated_ds.version().version;
    assert_eq!(new_version, 2);
    assert_eq!(
        updated_ds.get_fragments().len(),
        2,
        "in-place patches must not add fragments"
    );

    let mut scanner = updated_ds.scan();
    scanner
        .project(&["key", "tag", ROW_LAST_UPDATED_AT_VERSION])
        .unwrap();
    let result = scanner.try_into_batch().await.unwrap();
    assert_eq!(result.num_rows(), 8);

    let result_keys = result
        .column_by_name("key")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let tags = result
        .column_by_name("tag")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let versions = result
        .column_by_name(ROW_LAST_UPDATED_AT_VERSION)
        .expect("stable row ids must expose the last-updated version")
        .as_any()
        .downcast_ref::<arrow_array::UInt64Array>()
        .unwrap();

    for i in 0..result.num_rows() {
        let key = result_keys.value(i);
        let was_patched = keys.contains(&key);
        assert_eq!(
            tags.value(i),
            if was_patched { "patched" } else { "t" },
            "key {key} has the wrong tag"
        );
        assert_eq!(
            versions.value(i),
            if was_patched { new_version } else { 1 },
            "key {key} has the wrong last-updated version"
        );
    }
}

/// The version `update_fragments` stamps into the row-version metadata is
/// only the version this transaction expected at prepare time. A
/// compatible commit can land first, making the real commit version
/// later, so the operation carries the patched offsets and
/// `build_manifest` re-stamps exactly those rows with the version the
/// commit actually got.
#[rstest]
#[tokio::test]
async fn test_merge_insert_subcols_in_place_stable_row_id_rebase(
    #[values(false, true)] full_fragment: bool,
) {
    use lance_core::ROW_LAST_UPDATED_AT_VERSION;

    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("tag", DataType::Utf8, true),
        Field::new("other", DataType::Utf8, true),
    ]));
    let rows = |keys: Vec<u32>, tag: &str| {
        let n = keys.len();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(keys)),
                Arc::new(StringArray::from(vec![tag; n])),
                Arc::new(StringArray::from(vec!["o"; n])),
            ],
        )
        .unwrap()
    };
    let ds = Dataset::write(
        Box::new(RecordBatchIterator::new(
            [Ok(rows(vec![0, 1], "t"))],
            schema.clone(),
        )),
        "memory://",
        Some(WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(ds.version().version, 1);
    let ds = Arc::new(ds);

    // Either both rows of the only fragment (the no-read-back write
    // path) or one of them (the updater path). Both stamp the version.
    let keys: Vec<u32> = if full_fragment { vec![0, 1] } else { vec![0] };
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("tag", DataType::Utf8, true),
    ]));
    let source = RecordBatch::try_new(
        source_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(keys.clone())),
            Arc::new(StringArray::from(vec!["patched"; keys.len()])),
        ],
    )
    .unwrap();
    let stream =
        RecordBatchStreamAdapter::new(source_schema.clone(), futures::stream::iter([Ok(source)]));

    // Prepare against v1 but do not commit yet.
    let UncommittedMergeInsert { transaction, .. } =
        MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .write_mode(MergeInsertWriteMode::RewriteColumns)
            .try_build()
            .unwrap()
            .execute_uncommitted(Box::pin(stream) as SendableRecordBatchStream)
            .await
            .unwrap();
    match &transaction.operation {
        Operation::Update {
            updated_fragment_offsets,
            ..
        } => {
            let UpdatedFragmentOffsets(off_map) = updated_fragment_offsets
                .as_ref()
                .expect("a RewriteColumns update must carry its patched offsets");
            let frag_id = ds.get_fragments()[0].id() as u64;
            let offsets = off_map
                .get(&frag_id)
                .expect("the patched fragment must be present");
            assert_eq!(
                offsets.iter().collect::<Vec<u32>>(),
                keys.clone(),
                "the offsets must name exactly the patched rows"
            );
        }
        other => panic!("expected Operation::Update, got: {other:?}"),
    }

    // A concurrent append commits as v2, so the prepared transaction
    // lands on v3 rather than the v2 it stamped.
    let mut appended = ds.as_ref().clone();
    appended
        .append(
            RecordBatchIterator::new([Ok(rows(vec![2], "t"))], schema.clone()),
            None,
        )
        .await
        .unwrap();
    assert_eq!(appended.version().version, 2);

    let committed = CommitBuilder::new(Arc::new(appended))
        .execute(transaction)
        .await
        .unwrap();
    let commit_version = committed.version().version;
    assert_eq!(commit_version, 3);

    let mut scanner = committed.scan();
    scanner
        .project(&["key", "tag", ROW_LAST_UPDATED_AT_VERSION])
        .unwrap();
    let result = scanner.try_into_batch().await.unwrap();
    assert_eq!(result.num_rows(), 3);

    let result_keys = result
        .column_by_name("key")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let tags = result
        .column_by_name("tag")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let versions = result
        .column_by_name(ROW_LAST_UPDATED_AT_VERSION)
        .expect("stable row ids must expose the last-updated version")
        .as_any()
        .downcast_ref::<arrow_array::UInt64Array>()
        .unwrap();

    for i in 0..result.num_rows() {
        let key = result_keys.value(i);
        let was_patched = keys.contains(&key);
        assert_eq!(
            tags.value(i),
            if was_patched { "patched" } else { "t" },
            "key {key} has the wrong tag"
        );
        let expected = if was_patched {
            // Patched by the rebased transaction: the version it landed
            // on, not the version it guessed.
            commit_version
        } else if key == 2 {
            // Appended by the concurrent commit.
            2
        } else {
            // Untouched by either.
            1
        };
        assert_eq!(
            versions.value(i),
            expected,
            "key {key} has the wrong last-updated version"
        );
    }
}

/// An in-place column rewrite fills a replacement file covering every row
/// of each fragment it touches, from the snapshot it read, and
/// `build_manifest` then tombstones every overlay for the fields it
/// rewrote. So committing one over a concurrent overlay would drop that
/// overlay's value on a row the merge never matched. The conflict
/// resolver must retry instead, and the retry re-reads the overlaid
/// value.
#[tokio::test]
async fn test_merge_insert_subcols_in_place_rebase_over_overlay() {
    use crate::dataset::WriteDestination;
    use crate::dataset::transaction::DataOverlayGroup;
    use arrow_array::ArrayRef;
    use lance_file::writer::FileWriterOptions;
    use lance_io::utils::CachedFileSize;
    use lance_table::format::DataFile;
    use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};

    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("tag", DataType::Utf8, true),
        // Omitted by the source, so `RewriteColumns` has something to skip.
        Field::new("other", DataType::Utf8, true),
    ]));
    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0, 1])),
            Arc::new(StringArray::from(vec!["base0", "base1"])),
            Arc::new(StringArray::from(vec!["o", "o"])),
        ],
    )
    .unwrap();
    let ds = Arc::new(
        Dataset::write(
            Box::new(RecordBatchIterator::new([Ok(initial)], schema.clone())),
            "memory://",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    assert_eq!(ds.version().version, 1);
    let tag_field_id = ds.schema().field("tag").unwrap().id;
    let fragment_id = ds.get_fragments()[0].id() as u64;

    // Prepare an in-place patch of key 0 against v1, without committing.
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("tag", DataType::Utf8, true),
    ]));
    let source = RecordBatch::try_new(
        source_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["merge0"])),
        ],
    )
    .unwrap();
    let stream =
        RecordBatchStreamAdapter::new(source_schema.clone(), futures::stream::iter([Ok(source)]));
    let UncommittedMergeInsert { transaction, .. } =
        MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .write_mode(MergeInsertWriteMode::RewriteColumns)
            .try_build()
            .unwrap()
            .execute_uncommitted(Box::pin(stream) as SendableRecordBatchStream)
            .await
            .unwrap();

    // A concurrent overlay commits as v2, changing `tag` on key 1, which
    // the prepared merge did not match.
    let overlay_schema = ds.schema().project_by_ids(&[tag_field_id], true);
    let path = ds.base.clone().join("data").join("overlay.lance");
    let obj_writer = ds.object_store.create(&path).await.unwrap();
    let mut writer = lance_file::versions::v2_1::create_writer(
        obj_writer,
        overlay_schema,
        FileWriterOptions::default(),
    )
    .unwrap();
    writer
        .write_column(0, Arc::new(StringArray::from(vec!["overlay1"])) as ArrayRef)
        .await
        .unwrap();
    let summary = writer.finish().await.unwrap();
    let mut data_file = DataFile::new_unstarted(
        "overlay.lance".to_string(),
        lance_file::version::ConcreteFileVersion::V2_1,
    );
    data_file.fields = vec![tag_field_id].into();
    data_file.column_indices = vec![0].into();
    data_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);
    let overlaid = Dataset::commit(
        WriteDestination::Dataset(ds.clone()),
        Operation::DataOverlay {
            groups: vec![DataOverlayGroup {
                fragment_id,
                overlays: vec![DataOverlayFile {
                    data_file,
                    // Physical offset 1 is key 1.
                    coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([1u32])),
                    committed_version: 0,
                }],
            }],
        },
        Some(1),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();
    assert_eq!(overlaid.version().version, 2);
    let tag_of = |ds: &Dataset, key: u32| {
        let ds = ds.clone();
        async move {
            let mut scanner = ds.scan();
            scanner
                .filter(&format!("key = {key}"))
                .unwrap()
                .project(&["tag"])
                .unwrap();
            let batch = scanner.try_into_batch().await.unwrap();
            batch
                .column_by_name("tag")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(0)
                .to_string()
        }
    };
    assert_eq!(tag_of(&overlaid, 1).await, "overlay1");

    // The prepared transaction must not be able to commit as-is: doing so
    // would tombstone the overlay and resurrect "base1".
    let error = CommitBuilder::new(Arc::new(overlaid.clone()))
        .with_max_retries(0)
        .execute(transaction)
        .await
        .expect_err("committing over an overlapping overlay must conflict");
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "expected a retryable conflict, got: {error:?}"
    );

    // Re-running the merge against the overlaid dataset patches key 0 and
    // leaves the overlay on key 1 intact.
    let source = RecordBatch::try_new(
        source_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![0])),
            Arc::new(StringArray::from(vec!["merge0"])),
        ],
    )
    .unwrap();
    let (retried, _) = MergeInsertBuilder::try_new(Arc::new(overlaid), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            [Ok(source)],
            source_schema,
        )))
        .await
        .unwrap();
    assert_eq!(tag_of(&retried, 0).await, "merge0");
    assert_eq!(
        tag_of(&retried, 1).await,
        "overlay1",
        "the overlay on an unmatched row must survive the retried merge"
    );
}

/// The overlay resolution rules require an in-place column rewrite to
/// tombstone any overlay covering the fields it replaced, otherwise the
/// stale overlay value keeps shadowing the freshly written one.
/// `build_manifest` does that off `fields_modified`, so this asserts the
/// in-place path reports them.
#[tokio::test]
async fn test_merge_insert_subcols_in_place_reports_fields_modified() {
    let Fixtures { ds, new_data } = Box::pin(setup(false)).await;

    // Join keys remain unchanged even though the source carries them.
    let value_field_id = ds.schema().field("value").unwrap().id as u32;
    let key_field_id = ds.schema().field("key").unwrap().id as u32;
    let other_field_id = ds.schema().field("other").unwrap().id as u32;

    let stream = RecordBatchStreamAdapter::new(
        new_data.schema(),
        futures::stream::iter(vec![Ok(new_data.clone())]),
    );
    let UncommittedMergeInsert { transaction, .. } =
        MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .write_mode(MergeInsertWriteMode::RewriteColumns)
            .try_build()
            .unwrap()
            .execute_uncommitted(Box::pin(stream) as SendableRecordBatchStream)
            .await
            .unwrap();

    match &transaction.operation {
        Operation::Update {
            fields_modified,
            update_mode,
            ..
        } => {
            assert!(matches!(update_mode, Some(RewriteColumns)));
            assert!(
                fields_modified.contains(&value_field_id),
                "patched fields must be reported, got {fields_modified:?}"
            );
            assert!(
                !fields_modified.contains(&key_field_id),
                "join keys must not be reported as modified, got {fields_modified:?}"
            );
            assert!(
                !fields_modified.contains(&other_field_id),
                "a field the source never provided must not be reported as modified, \
                         got {fields_modified:?}"
            );
        }
        other => panic!("expected Operation::Update, got: {other:?}"),
    }
}

#[rstest]
#[case::indexed_update(true, false, false)]
#[case::indexed_upsert(true, true, false)]
#[case::planned_update(false, false, false)]
#[case::indexed_key_only(true, true, true)]
#[tokio::test]
async fn test_merge_insert_subcols_preserves_join_columns(
    #[case] scalar_index: bool,
    #[case] insert: bool,
    #[case] key_only: bool,
    #[values(false, true)] composite_key: bool,
    #[values(false, true)] stable_row_ids: bool,
) {
    let initial = record_batch!(
        ("id", UInt32, [0, 1, 2, 3, 4, 5]),
        ("id2", UInt32, [0, 1, 0, 1, 0, 1]),
        ("value", UInt32, [10, 11, 12, 13, 14, 15]),
        ("other", UInt32, [20, 21, 22, 23, 24, 25])
    )
    .unwrap();
    let mut ds = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            max_rows_per_file: 3,
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        })
        .execute(vec![initial])
        .await
        .unwrap();
    let on = if composite_key {
        vec!["id", "id2"]
    } else {
        vec!["id"]
    };
    if scalar_index {
        for key in &on {
            ds.create_index(
                &[*key],
                IndexType::BTree,
                None,
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        }
    }
    ds.create_index(
        &["value"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    let mut ds = Arc::new(ds);

    // Cover fragment 0 entirely and fragment 1 partially, plus an
    // unmatched key. Reorder source columns to exercise name matching.
    for (iteration, value) in [100, 200].into_iter().enumerate() {
        let source = record_batch!(
            (
                "value",
                UInt32,
                [Some(value), None, Some(value), Some(value), Some(value)]
            ),
            ("id2", UInt32, [0, 1, 0, 0, 0]),
            ("id", UInt32, [0, 1, 2, 4, 6])
        )
        .unwrap();
        let projection = match (composite_key, key_only) {
            (false, false) => vec![0, 2],
            (true, false) => vec![0, 1, 2],
            (false, true) => vec![2],
            (true, true) => vec![1, 2],
        };
        let source = source.project(&projection).unwrap();
        let (updated_ds, stats) =
            MergeInsertBuilder::try_new(ds.clone(), on.iter().map(|key| key.to_string()).collect())
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(if insert {
                    WhenNotMatched::InsertAll
                } else {
                    WhenNotMatched::DoNothing
                })
                .write_mode(if scalar_index {
                    MergeInsertWriteMode::Auto
                } else {
                    MergeInsertWriteMode::RewriteColumns
                })
                .try_build()
                .unwrap()
                .execute_reader(RecordBatchIterator::new(
                    [Ok(source.clone())],
                    source.schema(),
                ))
                .await
                .unwrap();
        ds = updated_ds;
        assert_eq!(
            stats.num_updated_rows,
            if insert && iteration == 1 { 5 } else { 4 }
        );
        assert_eq!(stats.num_inserted_rows, u64::from(insert && iteration == 0));
        assert_eq!(stats.num_deleted_rows, 0);

        for fragment in ds.get_fragments().iter().take(2) {
            let files = &fragment.metadata().files;
            assert_eq!(
                files[0].fields.as_ref(),
                &[0, 1, if key_only { 2 } else { -2 }, 3]
            );
            assert_eq!(files.len(), if key_only { 1 } else { 2 });
            if !key_only {
                assert_eq!(files.last().unwrap().fields.as_ref(), &[2]);
            }
        }
        if scalar_index {
            for key in &on {
                let index = ds
                    .load_scalar_index(IndexCriteria::default().with_name(&format!("{key}_idx")))
                    .await
                    .unwrap()
                    .expect("join key index must remain usable after each merge");
                assert_eq!(
                    index.fragment_bitmap,
                    Some(RoaringBitmap::from_iter([0, 1]))
                );
            }
        }
        let indices = ds.load_indices().await.unwrap();
        let value_index = indices
            .iter()
            .find(|index| index.name == "value_idx")
            .unwrap();
        assert_eq!(
            value_index.fragment_bitmap,
            Some(if key_only {
                RoaringBitmap::from_iter([0, 1])
            } else {
                RoaringBitmap::new()
            })
        );

        let mut scanner = ds.scan();
        scanner.scan_in_order(true);
        if stable_row_ids {
            scanner
                .project(&[
                    "id",
                    "id2",
                    "value",
                    "other",
                    lance_core::ROW_LAST_UPDATED_AT_VERSION,
                ])
                .unwrap();
        }
        let result = scanner.try_into_batch().await.unwrap();
        assert_eq!(result.num_rows(), if insert { 7 } else { 6 });
        for row in 0..6 {
            assert_eq!(
                result["id"].as_primitive::<UInt32Type>().value(row),
                row as u32
            );
            assert_eq!(
                result["id2"].as_primitive::<UInt32Type>().value(row),
                (row % 2) as u32
            );
            assert_eq!(
                result["other"].as_primitive::<UInt32Type>().value(row),
                row as u32 + 20
            );
            let values = result["value"].as_primitive::<UInt32Type>();
            if !key_only && [0, 1, 2, 4].contains(&row) {
                assert_eq!(values.is_null(row), row == 1);
                if row != 1 {
                    assert_eq!(values.value(row), value);
                }
            } else {
                assert_eq!(values.value(row), row as u32 + 10);
            }
            if stable_row_ids {
                let versions =
                    result[lance_core::ROW_LAST_UPDATED_AT_VERSION].as_primitive::<UInt64Type>();
                assert_eq!(
                    versions.value(row),
                    if [0, 1, 2, 4].contains(&row) {
                        ds.version().version
                    } else {
                        1
                    }
                );
            }
        }
        if insert {
            assert_eq!(result["id"].as_primitive::<UInt32Type>().value(6), 6);
            assert_eq!(result["id2"].is_null(6), !composite_key);
            if composite_key {
                assert_eq!(result["id2"].as_primitive::<UInt32Type>().value(6), 0);
            }
            assert!(result["other"].is_null(6));
            assert_eq!(result["value"].is_null(6), key_only);
            if !key_only {
                assert_eq!(result["value"].as_primitive::<UInt32Type>().value(6), value);
            }
            if stable_row_ids {
                assert_eq!(
                    result[lance_core::ROW_LAST_UPDATED_AT_VERSION]
                        .as_primitive::<UInt64Type>()
                        .value(6),
                    ds.version().version
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ConcurrentWrite {
    Compaction,
    ColumnRewrite,
    IndexBuild,
}

#[rstest]
#[case::compaction(ConcurrentWrite::Compaction)]
#[case::column_rewrite(ConcurrentWrite::ColumnRewrite)]
#[case::index_build(ConcurrentWrite::IndexBuild)]
#[tokio::test]
async fn test_merge_insert_key_only_concurrent_write(
    #[case] concurrent_write: ConcurrentWrite,
    #[values(false, true)] stable_row_ids: bool,
) {
    let initial = record_batch!(
        ("id", UInt32, [0, 1, 2, 3, 4, 5]),
        ("value", UInt32, [10, 11, 12, 13, 14, 15]),
        ("other", UInt32, [20, 21, 22, 23, 24, 25])
    )
    .unwrap();
    let mut ds = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            max_rows_per_file: 3,
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        })
        .execute(vec![initial])
        .await
        .unwrap();
    ds.create_index(
        &["id"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    let ds = Arc::new(ds);
    let source = record_batch!(("id", UInt32, [0, 4])).unwrap();
    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["id".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .commit_retries(0)
        .try_build()
        .unwrap();
    let prepared = job
        .clone()
        .execute_uncommitted_batches(vec![source.clone()])
        .await
        .unwrap();
    assert_eq!(prepared.stats.num_updated_rows, 2);
    assert_eq!(prepared.stats.num_files_written, 0);
    let Operation::Update {
        updated_fragments,
        fields_modified,
        ..
    } = &prepared.transaction.operation
    else {
        panic!("expected an in-place update");
    };
    assert!(fields_modified.is_empty());
    assert_eq!(updated_fragments.len(), 2);
    if !stable_row_ids {
        assert_eq!(updated_fragments.as_slice(), ds.fragments().as_slice());
    }

    let mut other = ds.as_ref().clone();
    match concurrent_write {
        ConcurrentWrite::Compaction => {
            compact_files(
                &mut other,
                CompactionOptions {
                    target_rows_per_fragment: 6,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            assert_eq!(other.fragments().len(), 1);
        }
        ConcurrentWrite::ColumnRewrite => {
            let patch = record_batch!(("id", UInt32, [0]), ("value", UInt32, [99])).unwrap();
            let (patched, _) = MergeInsertBuilder::try_new(Arc::new(other), vec!["id".into()])
                .unwrap()
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing)
                .write_mode(MergeInsertWriteMode::RewriteColumns)
                .try_build()
                .unwrap()
                .execute_batches(vec![patch])
                .await
                .unwrap();
            other = patched.as_ref().clone();
        }
        ConcurrentWrite::IndexBuild => {
            other
                .create_index(
                    &["value"],
                    IndexType::BTree,
                    None,
                    &ScalarIndexParams::default(),
                    false,
                )
                .await
                .unwrap();
        }
    }

    let mut columns = vec!["id", "value", "other"];
    if stable_row_ids {
        columns.push(ROW_LAST_UPDATED_AT_VERSION);
    }
    let before = other
        .scan()
        .project(&columns)
        .unwrap()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    // Compaction physically rewrites fragments, and on a
    // stable-row-id dataset the "id" BTree index stores row
    // addresses. Unlike ZoneMap/BloomFilter/FM, BTree supports remap,
    // so an eager compaction remaps it in place instead of dropping
    // its coverage of the rewritten fragment -- the in-place patch
    // below still finds the fragment covered and writes zero files.
    let commit = CommitBuilder::new(Arc::new(other.clone()))
        .with_max_retries(0)
        .execute(prepared.transaction)
        .await;
    let merged = if matches!(concurrent_write, ConcurrentWrite::IndexBuild) {
        commit.unwrap()
    } else {
        let error = commit.unwrap_err();
        assert!(
            matches!(error, Error::RetryableCommitConflict { .. }),
            "{error}"
        );
        assert!(error.to_string().contains("preempted"), "{error}");
        // Start from the stale handle again: the merge must replay on
        // the latest fragments instead of resurrecting the old files.
        let (merged, stats) = job.execute_batches(vec![source]).await.unwrap();
        assert_eq!(stats.num_attempts, 2);
        assert_eq!(stats.num_updated_rows, 2);
        assert_eq!(stats.num_files_written, 0);
        merged.as_ref().clone()
    };
    assert_eq!(merged.fragments().len(), other.fragments().len());
    for (merged_fragment, current) in merged.fragments().iter().zip(other.fragments().iter()) {
        assert_eq!(merged_fragment.id, current.id);
        assert_eq!(merged_fragment.files, current.files);
        if !stable_row_ids {
            assert_eq!(merged_fragment, current);
        }
    }
    let after = merged
        .scan()
        .project(&columns)
        .unwrap()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    // `before`'s and `after`'s row order are both original-id order:
    // the patch is always in-place, so no row ever relocates. Looking
    // rows up by id (rather than assuming position) degrades to
    // position here, but stays correct if a case is added later where
    // the patch does relocate rows.
    let after_row_for_id = |id: u32| -> usize {
        let ids = after["id"].as_primitive::<UInt32Type>();
        (0..ids.len()).find(|&i| ids.value(i) == id).unwrap()
    };
    assert_eq!(
        before.project(&[0, 1, 2]).unwrap(),
        after.project(&[0, 1, 2]).unwrap()
    );
    if stable_row_ids {
        for id in 0u32..6 {
            assert_eq!(
                after[ROW_LAST_UPDATED_AT_VERSION]
                    .as_primitive::<UInt64Type>()
                    .value(after_row_for_id(id)),
                if [0, 4].contains(&id) {
                    merged.version().version
                } else {
                    before[ROW_LAST_UPDATED_AT_VERSION]
                        .as_primitive::<UInt64Type>()
                        .value(id as usize)
                }
            );
        }
    }
    let live_fragments = merged
        .fragments()
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect::<RoaringBitmap>();
    let indices = merged.load_indices().await.unwrap();
    let expected_indices = if matches!(concurrent_write, ConcurrentWrite::IndexBuild) {
        vec!["id_idx", "value_idx"]
    } else {
        vec!["id_idx"]
    };
    // Address-domain indices usually cannot cover a fragment a rewrite
    // replaces -- their stored addresses describe the fragment they
    // were built against, not whatever replaced it (see
    // `test_addr_domain_index_does_not_cover_rewritten_update_fragment`).
    // "id" is a BTree, though, which supports remap: any rewrite that
    // touched it (the concurrent compaction) remapped it in place, so
    // it still covers every live fragment here.
    for name in expected_indices {
        let index = indices.iter().find(|index| index.name == name).unwrap();
        assert_eq!(index.fragment_bitmap.as_ref(), Some(&live_fragments));
    }
    merged.validate().await.unwrap();
}

#[rstest]
#[tokio::test]
async fn test_merge_insert_subcols_deleted_and_spilled_lineage(
    #[values(false, true)] stable_row_ids: bool,
    #[values(false, true)] key_only: bool,
) {
    let initial = record_batch!(
        ("id", UInt32, [0, 1, 2, 3, 4, 5]),
        ("value", UInt32, [10, 11, 12, 13, 14, 15]),
        ("other", UInt32, [20, 21, 22, 23, 24, 25])
    )
    .unwrap();
    let mut ds = InsertBuilder::new("memory://")
        .with_params(&WriteParams {
            max_rows_per_file: if stable_row_ids { 1 } else { 3 },
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        })
        .execute(vec![initial])
        .await
        .unwrap();
    if stable_row_ids {
        ds.update_config([
            (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
            (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
        ])
        .await
        .unwrap();
        compact_files(
            &mut ds,
            CompactionOptions {
                target_rows_per_fragment: 3,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(ds.fragments().len(), 2);
        for fragment in ds.fragments().iter() {
            assert!(matches!(
                fragment.last_updated_at_version_meta,
                Some(RowDatasetVersionMeta::Column)
            ));
        }
    }
    ds.create_index(
        &["id"],
        IndexType::BTree,
        None,
        &ScalarIndexParams::default(),
        false,
    )
    .await
    .unwrap();
    ds.delete("id = 1").await.unwrap();
    assert!(
        ds.fragments()
            .iter()
            .any(|fragment| fragment.deletion_file.is_some())
    );
    let mut columns = vec!["id", "value", "other"];
    if stable_row_ids {
        columns.extend([ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION]);
    }
    let before = ds
        .scan()
        .project(&columns)
        .unwrap()
        .with_row_id()
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first("id".into())]))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let source = record_batch!(
        ("id", UInt32, [0, 1, 2, 4]),
        ("value", UInt32, [Some(100), Some(100), None, Some(100)])
    )
    .unwrap();
    let source = if key_only {
        source.project(&[0]).unwrap()
    } else {
        source
    };
    let (merged, stats) = MergeInsertBuilder::try_new(Arc::new(ds.clone()), vec!["id".into()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .execute_batches(vec![source])
        .await
        .unwrap();
    assert_eq!(stats.num_updated_rows, 3);
    assert_eq!(stats.num_inserted_rows, 0);
    assert_eq!(stats.num_deleted_rows, 0);
    for (merged_fragment, original) in merged.fragments().iter().zip(ds.fragments().iter()) {
        assert_eq!(merged_fragment.id, original.id);
        assert_eq!(merged_fragment.deletion_file, original.deletion_file);
        if key_only {
            assert_eq!(merged_fragment.files, original.files);
        } else {
            assert_eq!(merged_fragment.files.len(), original.files.len() + 1);
            assert_eq!(merged_fragment.files.last().unwrap().fields.as_ref(), &[1]);
            assert_eq!(merged_fragment.files[0].fields[0], 0);
            assert_eq!(merged_fragment.files[0].fields[1], -2);
        }
    }
    let after = merged
        .scan()
        .project(&columns)
        .unwrap()
        .with_row_id()
        .order_by(Some(vec![ColumnOrdering::asc_nulls_first("id".into())]))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(
        after["id"].as_primitive::<UInt32Type>().values().as_ref(),
        &[0, 2, 3, 4, 5]
    );
    assert_eq!(&after[ROW_ID], &before[ROW_ID]);
    assert_eq!(&after["other"], &before["other"]);
    if stable_row_ids {
        assert_eq!(
            &after[ROW_CREATED_AT_VERSION],
            &before[ROW_CREATED_AT_VERSION]
        );
    }
    for (row, key) in [0, 2, 3, 4, 5].into_iter().enumerate() {
        let matched = [0, 2, 4].contains(&key);
        let values = after["value"].as_primitive::<UInt32Type>();
        assert_eq!(values.is_null(row), !key_only && key == 2);
        if !values.is_null(row) {
            assert_eq!(
                values.value(row),
                if !key_only && matched { 100 } else { key + 10 }
            );
        }
        if stable_row_ids {
            assert_eq!(
                after[ROW_LAST_UPDATED_AT_VERSION]
                    .as_primitive::<UInt64Type>()
                    .value(row),
                if matched {
                    merged.version().version
                } else {
                    before[ROW_LAST_UPDATED_AT_VERSION]
                        .as_primitive::<UInt64Type>()
                        .value(row)
                }
            );
        }
    }
    assert!(
        merged
            .scan()
            .filter("id = 4")
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap()
            .contains("ScalarIndexQuery")
    );
    merged.validate().await.unwrap();
}

/// Fragments can carry the same column in different data-file layouts:
/// one whose `tag` still lives in the file it was written with, another
/// whose `tag` has already been moved to a patch file by an earlier
/// merge. `Operation::DataReplacement` refuses that case outright, so
/// this asserts `RewriteColumns` does not inherit the restriction — it
/// appends a new data file per fragment rather than swapping one in, so
/// each fragment's existing layout is irrelevant.
#[tokio::test]
async fn test_merge_insert_subcols_in_place_heterogeneous_layouts() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("tag", DataType::Utf8, true),
        Field::new("other", DataType::Utf8, true),
    ]));
    let rows = |keys: Vec<u32>| {
        let n = keys.len();
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(keys)),
                Arc::new(StringArray::from(vec!["t"; n])),
                Arc::new(StringArray::from(vec!["o"; n])),
            ],
        )
        .unwrap()
    };
    let write_params = WriteParams {
        max_rows_per_file: 2,
        ..Default::default()
    };
    let ds = Dataset::write(
        Box::new(RecordBatchIterator::new(
            [Ok(rows(vec![0, 1]))],
            schema.clone(),
        )),
        "memory://",
        Some(write_params.clone()),
    )
    .await
    .unwrap();

    let patch_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("tag", DataType::Utf8, true),
    ]));
    let patch = |keys: Vec<u32>, tag: &'static str| {
        let n = keys.len();
        RecordBatch::try_new(
            patch_schema.clone(),
            vec![
                Arc::new(UInt32Array::from(keys)),
                Arc::new(StringArray::from(vec![tag; n])),
            ],
        )
        .unwrap()
    };
    async fn run_patch(ds: Arc<Dataset>, batch: RecordBatch) -> (Arc<Dataset>, MergeStats) {
        let schema = batch.schema();
        MergeInsertBuilder::try_new(ds, vec!["key".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::DoNothing)
            .write_mode(MergeInsertWriteMode::RewriteColumns)
            .try_build()
            .unwrap()
            .execute_reader(Box::new(RecordBatchIterator::new([Ok(batch)], schema)))
            .await
            .unwrap()
    }

    // Patch fragment 0 so its `tag` moves into a second data file and
    // the original file's `tag` field id is tombstoned.
    let (ds, _) = run_patch(Arc::new(ds), patch(vec![0, 1], "first")).await;
    let layout_of = |ds: &Dataset, frag: usize| {
        ds.get_fragments()[frag]
            .metadata()
            .files
            .iter()
            .map(|f| f.fields.to_vec())
            .collect::<Vec<_>>()
    };
    let frag0_layout = layout_of(&ds, 0);
    assert_eq!(
        frag0_layout.len(),
        2,
        "frag 0 should now carry a patch file"
    );

    // Append a fresh fragment, whose `tag` is still in its original file.
    let mut ds = Arc::unwrap_or_clone(ds);
    ds.append(
        Box::new(RecordBatchIterator::new(
            [Ok(rows(vec![2, 3]))],
            schema.clone(),
        )),
        Some(write_params),
    )
    .await
    .unwrap();
    let ds = Arc::new(ds);
    assert_eq!(
        layout_of(&ds, 1).len(),
        1,
        "frag 1 should be a single unpatched file"
    );
    assert_ne!(
        frag0_layout,
        layout_of(&ds, 1),
        "the two fragments must disagree on where `tag` lives for this test to mean anything"
    );

    // Patch across both layouts in one merge.
    let (updated_ds, stats) = run_patch(ds, patch(vec![0, 2], "patched")).await;
    assert_eq!(stats.num_updated_rows, 2);
    assert_eq!(
        updated_ds.get_fragments().len(),
        2,
        "in-place patches must not add fragments"
    );

    let result = updated_ds
        .scan()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(result.num_rows(), 4);
    let keys = result
        .column_by_name("key")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let tags = result
        .column_by_name("tag")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let others = result
        .column_by_name("other")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..result.num_rows() {
        let key = keys.value(i);
        let expected_tag = match key {
            0 | 2 => "patched",
            // Patched in the first merge, untouched by the second.
            1 => "first",
            _ => "t",
        };
        assert_eq!(tags.value(i), expected_tag, "key {key} has the wrong tag");
        assert_eq!(
            others.value(i),
            "o",
            "key {key}: a column the source never provided must not change"
        );
    }
}

/// An `UpdateIf` condition that rejects every matched row leaves the
/// patch stream empty. The in-place path must still commit cleanly —
/// no new fragments (which its `debug_assert` requires), no fragment
/// churn, and no value changed.
#[tokio::test]
async fn test_merge_insert_subcols_in_place_all_rows_rejected() {
    let Fixtures { ds, new_data } = Box::pin(setup(false)).await;

    let before = ds
        .scan()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    let fragments_before = ds
        .get_fragments()
        .iter()
        .map(|f| f.metadata().clone())
        .collect::<Vec<_>>();

    // `value` is a monotonic step starting at 0, so nothing exceeds
    // this bound and every matched row becomes `Action::Nothing`.
    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::update_if(&ds, "target.value > 999999").unwrap())
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap();
    let reader = Box::new(RecordBatchIterator::new(
        [Ok(new_data.clone())],
        new_data.schema(),
    ));
    let (updated_ds, stats) = job
        .execute_reader(reader)
        .await
        .expect("an all-rejected in-place merge must not fail");

    assert_eq!(stats.num_updated_rows, 0);
    assert_eq!(stats.num_deleted_rows, 0);
    assert_eq!(stats.num_inserted_rows, 0);
    assert_eq!(
        updated_ds
            .get_fragments()
            .iter()
            .map(|f| f.metadata().clone())
            .collect::<Vec<_>>(),
        fragments_before,
        "rejecting every row must leave the fragments untouched"
    );
    let after = updated_ds
        .scan()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(before, after, "no value should have changed");
}

/// `Auto` and an explicit `RewriteRows` must plan identically, and an
/// operation blocked several ways must have every blocker named rather
/// than only the first one found.
#[tokio::test]
async fn test_merge_insert_write_mode_auto_and_explicit_rows_agree() {
    let Fixtures { ds, new_data } = Box::pin(setup(false)).await;
    let source_schema: Schema = new_data.schema().as_ref().clone();

    let plan_for = |mode: Option<MergeInsertWriteMode>| {
        let ds = ds.clone();
        let source_schema = source_schema.clone();
        async move {
            let mut builder = MergeInsertBuilder::try_new(ds, vec!["key".to_string()]).unwrap();
            builder
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing);
            if let Some(mode) = mode {
                builder.write_mode(mode);
            }
            builder
                .try_build()
                .unwrap()
                .explain_plan(Some(&source_schema), false)
                .await
                .unwrap()
        }
    };
    assert_eq!(
        plan_for(None).await,
        plan_for(Some(MergeInsertWriteMode::RewriteRows)).await,
        "Auto must plan the same as an explicit RewriteRows"
    );

    // Inserts and delete-by-source both block column patching, and the
    // source here covers every dataset column too.
    let full_schema: Schema = ds.schema().into();
    let error = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .when_not_matched_by_source(WhenNotMatchedBySource::Delete)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .explain_plan(Some(&full_schema), false)
        .await
        .expect_err("a merge blocked several ways must be rejected");
    let message = error.to_string();
    for blocker in ["covers every dataset column", "adds rows", "removes rows"] {
        assert!(
            message.contains(blocker),
            "the error should name {blocker:?}: {message}"
        );
    }
}

/// A scalar index on the join key routes a partial-schema update to the
/// legacy path, which patches columns unconditionally. So an explicit
/// `RewriteRows` has to leave that path to be honored, while `Auto`
/// keeps the index probe and stays on it.
#[rstest]
#[tokio::test]
async fn test_merge_insert_rewrite_rows_leaves_indexed_path(
    #[values(None, Some(MergeInsertWriteMode::Auto))] auto: Option<MergeInsertWriteMode>,
) {
    let Fixtures { ds, new_data } = Box::pin(setup(true)).await;
    let source_schema: Schema = new_data.schema().as_ref().clone();

    let job = |mode: Option<MergeInsertWriteMode>| {
        let ds = ds.clone();
        async move {
            let mut builder = MergeInsertBuilder::try_new(ds, vec!["key".to_string()]).unwrap();
            builder
                .when_matched(WhenMatched::UpdateAll)
                .when_not_matched(WhenNotMatched::DoNothing);
            if let Some(mode) = mode {
                builder.write_mode(mode);
            }
            builder.try_build().unwrap()
        }
    };

    // The indexed path has no physical plan, so `explain_plan` rejecting
    // the job is how "stayed on the legacy path" is observable.
    let error = job(auto)
        .await
        .explain_plan(Some(&source_schema), false)
        .await
        .expect_err("Auto must keep the scalar-index route, which has no plan");
    assert!(
        error.to_string().contains("does not support explain_plan"),
        "unexpected error: {error}"
    );

    let plan = job(Some(MergeInsertWriteMode::RewriteRows))
        .await
        .explain_plan(Some(&source_schema), false)
        .await
        .expect("RewriteRows must fall through to the v2 plan");
    assert!(
        plan.contains("MergeInsert: on=[key]") && !plan.contains("mode=RewriteColumns"),
        "expected the row-rewrite v2 node, got: {plan}"
    );

    // Patching columns is what the indexed path already does, so asking
    // for it explicitly keeps the index probe.
    let error = job(Some(MergeInsertWriteMode::RewriteColumns))
        .await
        .explain_plan(Some(&source_schema), false)
        .await
        .expect_err("RewriteColumns must keep the scalar-index route");
    assert!(
        error.to_string().contains("does not support explain_plan"),
        "unexpected error: {error}"
    );
}

/// A source carrying nothing but the join key has no values to write, so
/// `RewriteColumns` requires at least one non-key column to patch.
/// Asking for it with only the key must error rather than quietly
/// writing whole rows instead.
#[tokio::test]
async fn test_merge_insert_rewrite_columns_rejects_key_only_source() {
    let Fixtures { ds, new_data } = Box::pin(setup(false)).await;

    let key_only = new_data.project(&[0]).unwrap();
    assert_eq!(key_only.schema().field(0).name(), "key");
    let source_schema: Schema = key_only.schema().as_ref().clone();

    let error = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .explain_plan(Some(&source_schema), false)
        .await
        .expect_err("a key-only source must be rejected, not silently rewritten");
    assert!(
        matches!(error, Error::InvalidInput { .. }),
        "got: {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains("no column besides the join key"),
        "the error must name the blocker: {message}"
    );

    // `Auto` accepts the same merge and writes whole rows.
    let plan = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .explain_plan(Some(&source_schema), false)
        .await
        .unwrap();
    assert!(
        plan.contains("MergeInsert: on=[key]") && !plan.contains("InPlaceMergeInsert"),
        "Auto should take the row-rewrite sink: {plan}"
    );
}

/// A blob nested inside a struct the source provides cannot be patched:
/// the fragment updater reads blobs in their stored descriptor form, and
/// only the row-rewrite path converts between the two representations.
/// Rooting the check on the top-level fields the source carries, rather
/// than matching nested field names against the source's top-level
/// columns, is what makes the nested case visible.
#[tokio::test]
async fn test_merge_insert_rewrite_columns_rejects_nested_blob() {
    use crate::{BlobArrayBuilder, blob_field};
    use arrow_array::{ArrayRef, StructArray};
    use arrow_schema::Fields;

    let info_fields: Fields = vec![
        Field::new("name", DataType::Utf8, false),
        blob_field("blob", true),
    ]
    .into();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("info", DataType::Struct(info_fields.clone()), true),
        Field::new("payload", DataType::Utf8, true),
    ]));

    let mk_info = |n: usize, tag: &str| -> ArrayRef {
        let mut builder = BlobArrayBuilder::new(n);
        for i in 0..n {
            builder
                .push_bytes(format!("{tag}-blob-{i}").as_bytes())
                .unwrap();
        }
        Arc::new(
            StructArray::try_new(
                info_fields.clone(),
                vec![
                    Arc::new(StringArray::from_iter_values(
                        (0..n).map(|i| format!("{tag}-name-{i}")),
                    )) as ArrayRef,
                    Arc::new(builder.finish().unwrap()) as ArrayRef,
                ],
                None,
            )
            .unwrap(),
        )
    };

    let initial = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..4)),
            mk_info(4, "old"),
            Arc::new(StringArray::from(vec!["p"; 4])),
        ],
    )
    .unwrap();
    let ds = Arc::new(
        Dataset::write(
            Box::new(RecordBatchIterator::new([Ok(initial)], schema.clone())),
            "memory://",
            Some(WriteParams {
                // Blob v2 requires >= 2.2.
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );

    // Omits `payload` (so the merge qualifies) but carries `info`,
    // whose subtree holds a blob.
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("info", DataType::Struct(info_fields.clone()), true),
    ]));
    let source = RecordBatch::try_new(
        source_schema.clone(),
        vec![
            Arc::new(UInt32Array::from(vec![1u32, 2])),
            mk_info(2, "new"),
        ],
    )
    .unwrap();

    let error = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap()
        .explain_plan(Some(source_schema.as_ref()), false)
        .await
        .expect_err("a nested blob must be rejected, not patched");
    assert!(
        matches!(error, Error::InvalidInput { .. }),
        "got: {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains("blob column"),
        "the error must name the blocker: {message}"
    );

    // `Auto` accepts the same merge, takes the row-rewrite sink, and
    // leaves the blobs readable.
    let plan = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .explain_plan(Some(source_schema.as_ref()), false)
        .await
        .unwrap();
    assert!(
        plan.contains("MergeInsert: on=[key]") && !plan.contains("InPlaceMergeInsert"),
        "Auto should take the row-rewrite sink: {plan}"
    );

    let (updated_ds, stats) = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::DoNothing)
        .try_build()
        .unwrap()
        .execute_reader(Box::new(RecordBatchIterator::new(
            [Ok(source)],
            source_schema,
        )))
        .await
        .expect("nested-blob merge must succeed on the row-rewrite path");
    assert_eq!(stats.num_updated_rows, 2);

    let keys = updated_ds
        .scan()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    let key_col = keys
        .column_by_name("key")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    let blobs = updated_ds
        .take_blobs_by_indices(&[0, 1, 2, 3], "info.blob")
        .await
        .unwrap();
    for (row, blob) in blobs.iter().enumerate() {
        let bytes = blob
            .as_ref()
            .expect("no blob cell should be null")
            .read()
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let key = key_col.value(row);
        let expected_tag = if key == 1 || key == 2 { "new" } else { "old" };
        assert!(
            text.starts_with(expected_tag),
            "key {key} should hold a {expected_tag}-* blob, got {text}"
        );
    }
}
