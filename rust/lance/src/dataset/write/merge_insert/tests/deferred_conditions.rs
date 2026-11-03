// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::{ArrayRef, LargeBinaryArray};
use lance_datagen::{ArrayGeneratorExt, ByteCount};
use lance_io::{object_store::WrappingObjectStore, utils::tracking_store::IOTracker};
use rstest::rstest;

const PAYLOAD_BYTES: u64 = 16 * 1024;
const ROWS_PER_FRAGMENT: usize = 50;
const NUM_FRAGMENTS: usize = 8;
const NUM_ROWS: usize = ROWS_PER_FRAGMENT * NUM_FRAGMENTS;
const COLUMN_BYTES: u64 = PAYLOAD_BYTES * NUM_ROWS as u64;

struct Fixture {
    ds: Arc<Dataset>,
    batch: RecordBatch,
    io: IOTracker,
}

/// `key` and `other` step 0..400 over eight 50-row fragments; `payload`
/// is 16 KiB per row and null at local row 25 of every fragment.
async fn fixture() -> Fixture {
    let mut nulls = vec![false; ROWS_PER_FRAGMENT];
    nulls[25] = true;
    let batch = lance_datagen::gen_batch()
        .with_seed(Seed::from(7))
        .col("key", array::step::<Int32Type>())
        .col("other", array::step::<UInt32Type>())
        .col(
            "payload",
            array::rand_fixedbin(ByteCount::from(PAYLOAD_BYTES), true).with_nulls(&nulls),
        )
        .into_batch_rows(RowCount::from(NUM_ROWS as u64))
        .unwrap();
    let ds = Dataset::write(
        Box::new(RecordBatchIterator::new(
            [Ok(batch.clone())],
            batch.schema(),
        )),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FRAGMENT,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let io = IOTracker::default();
    let ds = Arc::new(
        ds.with_object_store_wrappers([Arc::new(io.clone()) as Arc<dyn WrappingObjectStore>]),
    );
    io.incremental_stats();
    Fixture { ds, batch, io }
}

/// A partial-schema source of `key` + `payload`.
fn source(keys: &[i32]) -> RecordBatch {
    record_batch!(
        ("key", Int32, keys.to_vec()),
        (
            "payload",
            LargeBinary,
            keys.iter().map(|_| b"new".as_slice()).collect::<Vec<_>>()
        )
    )
    .unwrap()
}

fn builder(ds: &Arc<Dataset>) -> MergeInsertBuilder {
    let mut builder = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()]).unwrap();
    builder.when_not_matched(WhenNotMatched::DoNothing);
    builder
}

async fn explain(job: &MergeInsertJob, source: &RecordBatch) -> String {
    job.explain_plan(Some(source.schema().as_ref()), false)
        .await
        .unwrap()
}

fn scan_line(plan: &str) -> &str {
    plan.lines()
        .find(|line| line.contains("LanceRead"))
        .unwrap_or_else(|| panic!("no LanceRead in plan: {plan}"))
}

async fn run(job: MergeInsertJob, source: &RecordBatch) -> (Arc<Dataset>, MergeStats) {
    let reader = Box::new(RecordBatchIterator::new(
        [Ok(source.clone())],
        source.schema(),
    ));
    job.execute_reader(reader).await.unwrap()
}

/// `(key, payload)` for every row matching `filter`, in key order.
async fn payloads(ds: &Dataset, filter: &str) -> Vec<(i32, Option<Vec<u8>>)> {
    let batch = ds
        .scan()
        .project(&["key", "payload"])
        .unwrap()
        .filter(filter)
        .unwrap()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    let keys = batch["key"].as_primitive::<Int32Type>();
    let payloads = batch["payload"].as_binary::<i64>();
    let mut rows = (0..batch.num_rows())
        .map(|i| {
            (
                keys.value(i),
                payloads.is_valid(i).then(|| payloads.value(i).to_vec()),
            )
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|(key, _)| *key);
    rows
}

fn stored(batch: &RecordBatch, key: i32) -> Option<Vec<u8>> {
    let payloads = batch["payload"].as_binary::<i64>();
    payloads
        .is_valid(key as usize)
        .then(|| payloads.value(key as usize).to_vec())
}

/// The condition's column must not enter the target scan, which reads
/// every row however small the source is; it is taken for the matched
/// rows alone. The source is out of key order and spans fragments, so
/// the take has to restore the join's row order.
#[rstest]
#[case::rewrite_columns(MergeInsertWriteMode::RewriteColumns)]
#[case::rewrite_rows(MergeInsertWriteMode::RewriteRows)]
#[tokio::test]
async fn update_if_reads_target_column_for_matched_rows_only(
    #[case] write_mode: MergeInsertWriteMode,
) {
    let Fixture { ds, batch, io } = fixture().await;
    // Fragments 6, 0, 2, 1; keys 325 and 75 sit at local row 25.
    let source = source(&[325, 30, 130, 75]);
    let job = builder(&ds)
        .when_matched(WhenMatched::update_if(&ds, "target.payload IS NULL").unwrap())
        .write_mode(write_mode)
        .try_build()
        .unwrap();

    let plan = explain(&job, &source).await;
    assert!(
        !scan_line(&plan).contains("payload"),
        "the condition's column must not be in the target scan: {plan}"
    );
    assert!(
        plan.contains("MergeCondition: update_if=") && plan.contains("take=[payload]"),
        "the condition must be evaluated on the matched rows: {plan}"
    );
    assert!(
        plan.contains("when_matched=UpdateIf(MergeCondition)"),
        "the write node must not display a predicate it does not evaluate: {plan}"
    );

    io.incremental_stats();
    let (updated, stats) = run(job, &source).await;
    assert_eq!(stats.num_updated_rows, 2);
    let read_bytes = io.incremental_stats().read_bytes;
    assert!(
        read_bytes < COLUMN_BYTES / 2,
        "read {read_bytes} bytes against a {COLUMN_BYTES}-byte column"
    );
    assert_eq!(
        payloads(&updated, "key IN (325, 30, 130, 75)").await,
        vec![
            (30, stored(&batch, 30)),
            (75, Some(b"new".to_vec())),
            (130, stored(&batch, 130)),
            (325, Some(b"new".to_vec())),
        ]
    );
    assert_eq!(updated.count_rows(None).await.unwrap(), NUM_ROWS);
}

/// A full-schema source: the write path rewrites whole rows and the
/// target scan carries only the join key.
#[tokio::test]
async fn update_if_full_schema_source() {
    let Fixture { ds, batch, io } = fixture().await;
    let source = record_batch!(
        ("key", Int32, [25, 30]),
        ("other", UInt32, [1000u32, 1001]),
        ("payload", LargeBinary, [b"new".as_slice(), b"new"])
    )
    .unwrap();
    let job = builder(&ds)
        .when_matched(WhenMatched::update_if(&ds, "target.payload IS NULL").unwrap())
        .try_build()
        .unwrap();

    let plan = explain(&job, &source).await;
    assert!(
        !scan_line(&plan).contains("payload") && plan.contains("take=[payload]"),
        "{plan}"
    );
    io.incremental_stats();
    let (updated, stats) = run(job, &source).await;
    assert_eq!(stats.num_updated_rows, 1);
    assert!(io.incremental_stats().read_bytes < COLUMN_BYTES / 2);
    assert_eq!(
        payloads(&updated, "key IN (25, 30)").await,
        vec![(25, Some(b"new".to_vec())), (30, stored(&batch, 30))]
    );
}

/// With whole-row rewrites the source's omitted columns are filled
/// from the target side of the join, so the condition's column is in
/// the plan twice: unqualified for the write and `target.`-qualified
/// for the condition. Both spellings of the condition must resolve.
#[rstest]
#[case::qualified("target.payload IS NULL")]
#[case::unqualified("payload IS NULL")]
#[tokio::test]
async fn update_if_rewrite_rows_omitted_column(#[case] condition: &str) {
    let Fixture { ds, batch, .. } = fixture().await;
    let source =
        record_batch!(("key", Int32, [25, 30]), ("other", UInt32, [1000u32, 1001])).unwrap();
    let job = builder(&ds)
        .when_matched(WhenMatched::update_if(&ds, condition).unwrap())
        .try_build()
        .unwrap();
    // The filled column is bound in place, not taken again.
    let plan = explain(&job, &source).await;
    assert!(plan.contains("take=[]"), "{plan}");
    let (updated, stats) = run(job, &source).await;
    assert_eq!(stats.num_updated_rows, 1);
    let rows = updated
        .scan()
        .filter("key IN (25, 30)")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let keys = rows["key"].as_primitive::<Int32Type>();
    let others = rows["other"].as_primitive::<UInt32Type>();
    let mut rows = keys
        .values()
        .iter()
        .zip(others.values())
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(rows, vec![(&25, &1000), (&30, &30)]);
    assert_eq!(
        payloads(&updated, "key IN (25, 30)").await,
        vec![(25, None), (30, stored(&batch, 30))]
    );
}

/// The join key is a target column like any other: the condition's
/// copy is taken, since the scanned copy does not survive projection
/// pruning.
#[tokio::test]
async fn update_if_over_join_key() {
    let Fixture { ds, .. } = fixture().await;
    let source = source(&[25, 30, 35]);
    let job = builder(&ds)
        .when_matched(WhenMatched::update_if(&ds, "target.key >= 30").unwrap())
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap();
    let plan = explain(&job, &source).await;
    assert!(plan.contains("take=[key]"), "{plan}");
    let (updated, stats) = run(job, &source).await;
    assert_eq!(stats.num_updated_rows, 2);
    assert_eq!(
        payloads(&updated, "key IN (30, 35)").await,
        vec![(30, Some(b"new".to_vec())), (35, Some(b"new".to_vec()))]
    );
}

/// A condition over both sides reads the source column from the join
/// output and takes only the target column.
#[tokio::test]
async fn update_if_mixed_source_and_target_columns() {
    let Fixture { ds, .. } = fixture().await;
    let source = record_batch!(
        ("key", Int32, [25, 30, 35]),
        ("other", UInt32, [0u32, 31, 40])
    )
    .unwrap();
    let job = builder(&ds)
        .when_matched(WhenMatched::update_if(&ds, "source.other > target.other").unwrap())
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap();
    let plan = explain(&job, &source).await;
    assert!(plan.contains("take=[other]"), "{plan}");
    let (updated, stats) = run(job, &source).await;
    assert_eq!(stats.num_updated_rows, 2);
    let rows = updated
        .scan()
        .filter("key IN (25, 30, 35)")
        .unwrap()
        .scan_in_order(true)
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(
        rows["other"].as_primitive::<UInt32Type>().values(),
        &[25, 31, 40]
    );
}

/// `_rowid` is not a dataset column, so it is read from the join output
/// while the dataset column next to it is taken.
#[tokio::test]
async fn update_if_row_id_with_target_column() {
    let Fixture { ds, batch, .. } = fixture().await;
    let source = source(&[25, 30, 75]);
    let condition = logical_expr::col("target._rowid")
        .lt(logical_expr::lit(ROWS_PER_FRAGMENT as u64))
        .and(logical_expr::col("target.payload").is_null());
    let job = builder(&ds)
        .when_matched(WhenMatched::UpdateIfExpr(condition))
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap();
    let plan = explain(&job, &source).await;
    assert!(plan.contains("take=[payload]"), "{plan}");
    let (updated, stats) = run(job, &source).await;
    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(
        payloads(&updated, "key IN (25, 30, 75)").await,
        vec![
            (25, Some(b"new".to_vec())),
            (30, stored(&batch, 30)),
            (75, None),
        ]
    );
}

/// A null verdict rejects the row, as a `WHERE` clause would: the
/// null-payload row's verdict is `false OR NULL`.
#[tokio::test]
async fn update_if_null_verdict_rejects() {
    let Fixture { ds, .. } = fixture().await;
    let source = source(&[25, 30]);
    let job = builder(&ds)
        .when_matched(WhenMatched::update_if(&ds, "target.payload IS NOT NULL OR NULL").unwrap())
        .write_mode(MergeInsertWriteMode::RewriteColumns)
        .try_build()
        .unwrap();
    let (updated, stats) = run(job, &source).await;
    assert_eq!(stats.num_updated_rows, 1);
    assert_eq!(
        payloads(&updated, "key IN (25, 30)").await,
        vec![(25, None), (30, Some(b"new".to_vec()))]
    );
}

/// Duplicate source keys reach the condition before the write node
/// rejects them; the take must tolerate the repeated address so the
/// write node's error is the one the caller sees.
#[tokio::test]
async fn update_if_duplicate_source_keys_still_rejected() {
    let Fixture { ds, .. } = fixture().await;
    let source = source(&[25, 30, 25]);
    let job = builder(&ds)
        .when_matched(WhenMatched::update_if(&ds, "target.payload IS NULL").unwrap())
        .try_build()
        .unwrap();
    let reader = Box::new(RecordBatchIterator::new(
        [Ok(source.clone())],
        source.schema(),
    ));
    let err = job.execute_reader(reader).await.unwrap_err().to_string();
    assert!(err.contains("Ambiguous merge inserts"), "{err}");
}

/// The node sits between the write sink and the join.
#[tokio::test]
async fn condition_node_sits_above_the_join() {
    let Fixture { ds, .. } = fixture().await;
    let source = source(&[25]);
    let job = builder(&ds)
        .when_matched(WhenMatched::update_if(&ds, "target.payload IS NULL").unwrap())
        .try_build()
        .unwrap();
    let plan = explain(&job, &source).await;
    let position = |needle: &str| {
        plan.lines()
            .position(|line| line.contains(needle))
            .unwrap_or_else(|| panic!("no {needle} in plan: {plan}"))
    };
    assert!(
        position("MergeInsert: on=") < position("MergeCondition:")
            && position("MergeCondition:") < position("HashJoinExec"),
        "{plan}"
    );
}

/// A `DeleteIf` over a target column is gated the same way, on the
/// target-only rows. That set is the table minus the matches, so the
/// take reads about as many bytes as the scan did; what it saves is
/// holding the column on the join's build side.
#[tokio::test]
async fn delete_if_reads_target_column_for_unmatched_rows_only() {
    let Fixture { ds, io, .. } = fixture().await;
    let source = source(&[25, 30, 35]);
    let job = builder(&ds)
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched_by_source(
            WhenNotMatchedBySource::delete_if(&ds, "payload IS NULL").unwrap(),
        )
        .try_build()
        .unwrap();

    let plan = explain(&job, &source).await;
    assert!(
        !scan_line(&plan).contains("payload"),
        "the condition's column must not be in the target scan: {plan}"
    );
    assert!(
        plan.contains("delete_if=") && plan.contains("take=[payload]"),
        "the condition must be evaluated on the unmatched rows: {plan}"
    );

    io.incremental_stats();
    let (updated, stats) = run(job, &source).await;
    assert_eq!(stats.num_updated_rows, 3);
    // One null payload per fragment; fragment 0's (key 25) is matched.
    assert_eq!(stats.num_deleted_rows, (NUM_FRAGMENTS - 1) as u64);
    let read_bytes = io.incremental_stats().read_bytes;
    assert!(
        read_bytes < COLUMN_BYTES + COLUMN_BYTES / 10,
        "read {read_bytes} bytes against a {COLUMN_BYTES}-byte column"
    );
    assert_eq!(payloads(&updated, "payload IS NULL").await, vec![]);
    assert_eq!(
        updated.count_rows(None).await.unwrap(),
        NUM_ROWS - (NUM_FRAGMENTS - 1)
    );
    assert_eq!(
        payloads(&updated, "key IN (25, 75)").await,
        vec![(25, Some(b"new".to_vec()))]
    );
}

/// Both conditions deferred at once, over different columns. The
/// source carries every column so that neither is filled from the
/// target side.
#[tokio::test]
async fn update_if_and_delete_if_together() {
    let Fixture { ds, batch, .. } = fixture().await;
    let source = record_batch!(
        ("key", Int32, [25, 30, 35]),
        ("other", UInt32, [25u32, 30, 35]),
        ("payload", LargeBinary, [b"new".as_slice(), b"new", b"new"])
    )
    .unwrap();
    let job = builder(&ds)
        .when_matched(WhenMatched::update_if(&ds, "target.payload IS NULL").unwrap())
        .when_not_matched_by_source(WhenNotMatchedBySource::delete_if(&ds, "other < 3").unwrap())
        .try_build()
        .unwrap();
    let plan = explain(&job, &source).await;
    assert!(
        !scan_line(&plan).contains("payload")
            && !scan_line(&plan).contains("other")
            && plan.contains("take=[other, payload]"),
        "{plan}"
    );
    let (updated, stats) = run(job, &source).await;
    assert_eq!((stats.num_updated_rows, stats.num_deleted_rows), (1, 3));
    assert_eq!(
        payloads(&updated, "key < 4 OR key IN (25, 30)").await,
        vec![
            (3, stored(&batch, 3)),
            (25, Some(b"new".to_vec())),
            (30, stored(&batch, 30)),
        ]
    );
}

/// Gate reproducer: a legacy blob column must reach the condition as
/// `LargeBinary`, as the target scan presents it, not as a descriptor.
#[tokio::test]
async fn gate_legacy_blob_condition() {
    let mut metadata = HashMap::new();
    metadata.insert(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string());
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::LargeBinary, true).with_metadata(metadata),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(LargeBinaryArray::from(vec![Some(b"old".as_slice()), None])),
        ],
    )
    .unwrap();
    let ds = Arc::new(
        Dataset::write(
            reader(batch),
            "memory://",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    let input = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1])) as ArrayRef,
            Arc::new(LargeBinaryArray::from(vec![Some(b"new".as_slice())])),
        ],
    )
    .unwrap();
    let condition = WhenMatched::update_if(&ds, "target.payload != source.payload").unwrap();
    let mut builder = MergeInsertBuilder::try_new(ds, vec!["key".to_string()]).unwrap();
    builder
        .when_matched(condition)
        .when_not_matched(WhenNotMatched::DoNothing)
        .use_index(false);
    let (_, stats) = builder
        .try_build()
        .unwrap()
        .execute_reader(reader(input))
        .await
        .unwrap();
    assert_eq!(stats.num_updated_rows, 1);
}

fn reader(
    batch: RecordBatch,
) -> Box<RecordBatchIterator<std::vec::IntoIter<std::result::Result<RecordBatch, ArrowError>>>> {
    let schema = batch.schema();
    Box::new(RecordBatchIterator::new(
        vec![Ok(batch)].into_iter(),
        schema,
    ))
}

/// Control: the same legacy blob predicate where the source omits
/// `payload`, so a whole-row rewrite fills it from the scan and the
/// condition reads that copy rather than taking it.
#[tokio::test]
async fn legacy_blob_condition_filled_control() {
    let mut metadata = HashMap::new();
    metadata.insert(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string());
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("other", DataType::Int32, false),
        Field::new("payload", DataType::LargeBinary, true).with_metadata(metadata),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(Int32Array::from(vec![0, 0])) as ArrayRef,
            Arc::new(LargeBinaryArray::from(vec![Some(b"old".as_slice()), None])),
        ],
    )
    .unwrap();
    let ds = Arc::new(
        Dataset::write(
            reader(batch),
            "memory://",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    let input = record_batch!(("key", Int32, [1, 2]), ("other", Int32, [7, 7])).unwrap();
    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::update_if(&ds, "target.payload = X'6F6C64'").unwrap())
        .when_not_matched(WhenNotMatched::DoNothing)
        .use_index(false)
        .try_build()
        .unwrap();
    let plan = job
        .explain_plan(Some(input.schema().as_ref()), false)
        .await
        .unwrap();
    assert!(plan.contains("take=[]"), "{plan}");
    let (_, stats) = job.execute_reader(reader(input)).await.unwrap();
    assert_eq!(stats.num_updated_rows, 1);
}

/// A Blob v2 column reaches the condition as the descriptor the target
/// scan presents (`position`, `size`), so a predicate over the
/// descriptor works through the deferred take.
#[tokio::test]
async fn blob_v2_descriptor_condition() {
    use crate::{BlobArrayBuilder, blob_field};
    use datafusion::functions::core::expr_fn::get_field;

    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        blob_field("blobs", true),
    ]));
    let make_batch = |keys: Vec<i32>, values: &[&[u8]]| {
        let mut blobs = BlobArrayBuilder::new(values.len());
        for value in values {
            blobs.push_bytes(value).unwrap();
        }
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(keys)) as ArrayRef,
                blobs.finish().unwrap(),
            ],
        )
        .unwrap()
    };
    let ds = Arc::new(
        Dataset::write(
            reader(make_batch(vec![1, 2, 3], &[b"foo", b"quux", b"ab"])),
            "memory://",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    let input = make_batch(vec![1, 2, 3], &[b"new", b"new", b"new"]);
    let condition =
        get_field(logical_expr::col("target.blobs"), "size").gt(logical_expr::lit(3u64));
    let job = MergeInsertBuilder::try_new(ds.clone(), vec!["key".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateIfExpr(condition))
        .when_not_matched(WhenNotMatched::DoNothing)
        .use_index(false)
        .try_build()
        .unwrap();
    let plan = job
        .explain_plan(Some(input.schema().as_ref()), false)
        .await
        .unwrap();
    assert!(
        !scan_line(&plan).contains("blobs") && plan.contains("take=[blobs]"),
        "{plan}"
    );
    let (updated, stats) = job.execute_reader(reader(input)).await.unwrap();
    assert_eq!(stats.num_updated_rows, 1);
    let blobs = updated
        .take_blobs_by_indices(&[0, 1, 2], "blobs")
        .await
        .unwrap();
    let mut values = Vec::new();
    for blob in blobs.iter() {
        values.push(
            blob.as_ref()
                .unwrap()
                .read()
                .await
                .unwrap()
                .as_ref()
                .to_vec(),
        );
    }
    values.sort();
    assert_eq!(
        values,
        vec![b"ab".to_vec(), b"foo".to_vec(), b"new".to_vec()]
    );
}

/// A condition over source columns alone stays in the join's action
/// expression; nothing is taken.
#[tokio::test]
async fn source_only_condition_is_not_deferred() {
    let Fixture { ds, .. } = fixture().await;
    let source = source(&[25, 30]);
    let job = builder(&ds)
        .when_matched(WhenMatched::update_if(&ds, "source.payload IS NOT NULL").unwrap())
        .try_build()
        .unwrap();
    let plan = explain(&job, &source).await;
    assert!(!plan.contains("MergeCondition:"), "{plan}");
    assert!(
        plan.contains("when_matched=UpdateIf(source.payload IS NOT NULL)"),
        "{plan}"
    );
}

/// Without an update no row written needs a stored value, so the
/// columns the source omits are filled with NULL rather than scanned:
/// a delete by key reads only the key.
#[tokio::test]
async fn delete_by_key_scans_only_the_key() {
    let Fixture { ds, io, .. } = fixture().await;
    let source = record_batch!(("key", Int32, [325, 30, 130])).unwrap();
    let job = builder(&ds)
        .when_matched(WhenMatched::Delete)
        .try_build()
        .unwrap();

    let plan = explain(&job, &source).await;
    assert!(scan_line(&plan).contains("projection=[key]"), "{plan}");

    io.incremental_stats();
    let (updated, stats) = run(job, &source).await;
    assert_eq!(stats.num_deleted_rows, 3);
    let read_bytes = io.incremental_stats().read_bytes;
    assert!(
        read_bytes < COLUMN_BYTES / 2,
        "read {read_bytes} bytes against a {COLUMN_BYTES}-byte column"
    );
    assert_eq!(payloads(&updated, "key IN (325, 30, 130)").await, vec![]);
    assert_eq!(updated.count_rows(None).await.unwrap(), NUM_ROWS - 3);
}

/// A `DeleteIf` over a column the source omits reads the filled copy,
/// so that column is still filled from the target when nothing is
/// updated; filled with NULL, no row would pass the condition.
#[tokio::test]
async fn delete_if_over_column_the_source_omits() {
    let Fixture { ds, batch, .. } = fixture().await;
    let source = record_batch!(("key", Int32, [25, 30, 1000])).unwrap();
    let job = builder(&ds)
        .when_not_matched(WhenNotMatched::InsertAll)
        .when_not_matched_by_source(WhenNotMatchedBySource::delete_if(&ds, "other < 3").unwrap())
        .try_build()
        .unwrap();

    let plan = explain(&job, &source).await;
    assert!(
        scan_line(&plan).contains("other")
            && !scan_line(&plan).contains("payload")
            && plan.contains("take=[]"),
        "{plan}"
    );

    let (updated, stats) = run(job, &source).await;
    assert_eq!((stats.num_inserted_rows, stats.num_deleted_rows), (1, 3));
    assert_eq!(payloads(&updated, "key < 3").await, vec![]);
    assert_eq!(
        payloads(&updated, "key IN (25, 30, 1000)").await,
        vec![(25, None), (30, stored(&batch, 30)), (1000, None)]
    );
    assert_eq!(updated.count_rows(None).await.unwrap(), NUM_ROWS - 3 + 1);
}
