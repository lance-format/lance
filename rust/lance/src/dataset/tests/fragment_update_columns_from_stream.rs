// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `FileFragment::update_columns_from_stream`: overwriting existing columns
//! from a stream that is aligned with the fragment's live rows, so no join and
//! no read-back of the old values is needed. Every row carries its `_rowaddr`,
//! and a stream that drifts from the fragment's scan order must be rejected
//! before any of its values can land on the wrong row.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::types::{Int32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StringArray, StringViewArray,
    StructArray, UInt64Array,
};
use arrow_schema::{ArrowError, DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tempfile::TempStrDir;
use lance_core::{Error, ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};
use lance_file::version::LanceFileVersion;
use rstest::rstest;

use crate::Dataset;
use crate::dataset::WriteDestination;
use crate::dataset::fragment::{FileFragment, FragmentUpdateColumnsResult};
use crate::dataset::transaction::{Operation, UpdateMode, UpdatedFragmentOffsets};
use crate::dataset::write::WriteParams;

const ROWS: i32 = 37;
/// Rows 0..9 cover the first two 8-row read batches' worth of a leading
/// deleted run, so the updater hands out a batch with no live row in it; the
/// rest are scattered so later batches are short by different amounts.
const DELETE_PREDICATE: &str = "id < 9 OR id IN (15, 16, 30)";
/// Small enough that the fragment spans several updater batches.
const BATCH_SIZE: u32 = 8;
/// Stream batch sizes, cycled: none of them line up with `BATCH_SIZE`, and the
/// zero exercises empty batches in the middle of the stream.
const CHUNKS: &[usize] = &[3, 1, 11, 0, 2, 5];

fn base_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("v", DataType::Int32, true),
        ArrowField::new("s", DataType::Utf8, true),
    ]))
}

/// One fragment of `ROWS` rows: `id` = 0..ROWS, `v` = -1, `s` = "old{id}".
async fn base_dataset(
    uri: &str,
    version: LanceFileVersion,
    stable_row_ids: bool,
    deletions: bool,
) -> Dataset {
    let schema = base_schema();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..ROWS)),
            Arc::new(Int32Array::from(vec![-1; ROWS as usize])),
            Arc::new(StringArray::from_iter_values(
                (0..ROWS).map(|i| format!("old{i}")),
            )),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        uri,
        Some(WriteParams {
            data_storage_version: Some(version),
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    if deletions {
        dataset.delete(DELETE_PREDICATE).await.unwrap();
    }
    dataset
}

fn only_fragment(dataset: &Dataset) -> FileFragment {
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 1);
    fragments.into_iter().next().unwrap()
}

/// The fragment's live rows in scan order: `(_rowaddr, id)`.
async fn live_rows(fragment: &FileFragment) -> (Vec<u64>, Vec<i32>) {
    let mut scanner = fragment.scan();
    scanner.project(&["id"]).unwrap().with_row_address();
    let batch = scanner.try_into_batch().await.unwrap();
    let addrs = batch[ROW_ADDR]
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec();
    let ids = batch["id"].as_primitive::<Int32Type>().values().to_vec();
    (addrs, ids)
}

fn values_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        ArrowField::new("v", DataType::Int32, true),
        ArrowField::new("s", DataType::Utf8, true),
    ]))
}

/// New values for every live row: `v` = id * 10, `s` = "new{id}".
fn new_values(addrs: &[Option<u64>], ids: &[i32]) -> RecordBatch {
    RecordBatch::try_new(
        values_schema(),
        vec![
            Arc::new(UInt64Array::from(addrs.to_vec())),
            Arc::new(Int32Array::from_iter_values(ids.iter().map(|id| id * 10))),
            Arc::new(StringArray::from_iter_values(
                ids.iter().map(|id| format!("new{id}")),
            )),
        ],
    )
    .unwrap()
}

/// Split `batch` by `CHUNKS`, cycled, with a trailing empty batch.
fn chunked(batch: &RecordBatch) -> Vec<RecordBatch> {
    let mut out = Vec::new();
    let mut offset = 0;
    for &size in CHUNKS.iter().cycle() {
        if offset >= batch.num_rows() {
            break;
        }
        let size = size.min(batch.num_rows() - offset);
        out.push(batch.slice(offset, size));
        offset += size;
    }
    out.push(batch.slice(batch.num_rows(), 0));
    out
}

fn reader_of(
    schema: Arc<ArrowSchema>,
    batches: Vec<RecordBatch>,
) -> RecordBatchIterator<Vec<Result<RecordBatch, ArrowError>>> {
    RecordBatchIterator::new(batches.into_iter().map(Ok).collect::<Vec<_>>(), schema)
}

async fn commit_rewrite(dataset: &Dataset, result: FragmentUpdateColumnsResult) -> Dataset {
    let read_version = dataset.manifest.version;
    let offsets = HashMap::from([(result.fragment.id, result.matched_offsets)]);
    Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset.clone())),
        Operation::Update {
            removed_fragment_ids: vec![],
            updated_fragments: vec![result.fragment],
            new_fragments: vec![],
            fields_modified: result.fields_modified,
            compacted_sstables: vec![],
            fields_for_preserving_frag_bitmap: vec![],
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: Some(UpdatedFragmentOffsets(offsets)),
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap()
}

async fn count_files(dataset: &Dataset) -> usize {
    dataset
        .object_store
        .read_dir_all(&dataset.data_dir(), None)
        .try_fold(0usize, |count, _| async move { Ok(count + 1) })
        .await
        .unwrap()
}

fn live_ids(deletions: bool) -> Vec<i32> {
    (0..ROWS)
        .filter(|id| !deletions || !(*id < 9 || [15, 16, 30].contains(id)))
        .collect()
}

#[rstest]
#[tokio::test]
async fn test_overwrites_live_rows_in_place(
    #[values(LanceFileVersion::V2_0, LanceFileVersion::Stable)] version: LanceFileVersion,
    #[values(false, true)] deletions: bool,
) {
    let test_uri = TempStrDir::default();
    let dataset = base_dataset(&test_uri, version, false, deletions).await;
    let fragment = only_fragment(&dataset);
    let (addrs, ids) = live_rows(&fragment).await;
    assert_eq!(ids, live_ids(deletions));

    let addrs = addrs.into_iter().map(Some).collect::<Vec<_>>();
    let stream = reader_of(values_schema(), chunked(&new_values(&addrs, &ids)));
    let result = fragment
        .update_columns_from_stream(stream, Some(BATCH_SIZE))
        .await
        .unwrap();

    let v_id = dataset.schema().field("v").unwrap().id as u32;
    let s_id = dataset.schema().field("s").unwrap().id as u32;
    let mut fields_modified = result.fields_modified.clone();
    fields_modified.sort_unstable();
    assert_eq!(fields_modified, vec![v_id, s_id]);
    let expected_offsets = ids.iter().map(|id| *id as u32).collect::<Vec<_>>();
    assert_eq!(
        result.matched_offsets.iter().collect::<Vec<_>>(),
        expected_offsets
    );
    // The replaced fields are tombstoned in the original file, which keeps `id`.
    let layout: Vec<Vec<i32>> = result
        .fragment
        .files
        .iter()
        .map(|file| file.fields.as_ref().to_vec())
        .collect();
    assert_eq!(layout, vec![vec![0, -2, -2], vec![1, 2]]);

    let dataset = commit_rewrite(&dataset, result).await;
    dataset.validate().await.unwrap();
    only_fragment(&dataset).validate().await.unwrap();

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(batch["id"].as_ref(), &Int32Array::from(ids.clone()));
    assert_eq!(
        batch["v"].as_ref(),
        &Int32Array::from_iter_values(ids.iter().map(|id| id * 10))
    );
    assert_eq!(
        batch["s"].as_ref(),
        &StringArray::from_iter_values(ids.iter().map(|id| format!("new{id}")))
    );
    assert_eq!(
        only_fragment(&dataset).count_deletions().await.unwrap(),
        (ROWS as usize) - ids.len()
    );
}

/// How a stream can drift from the fragment's live rows. Each must fail before
/// it can write a value onto a row other than the one it names.
#[derive(Debug, Clone, Copy)]
enum Drift {
    /// Right values, wrong order.
    SwappedRows,
    /// One live row skipped, so every later value shifts up a row.
    MissingRow,
    /// A deleted row's address is supplied as if it were live.
    DeletedRow,
    /// An address from another fragment.
    ForeignFragment,
    NullRowAddr,
    TooFewRows,
    TooManyRows,
    StreamError,
}

/// Rows before this live index are well formed, so the updater has already
/// written a few batches when the drift is found and must discard them.
const DRIFT_AT: usize = 15;

fn drifted_batches(
    drift: Drift,
    addrs: &[u64],
    ids: &[i32],
) -> Vec<Result<RecordBatch, ArrowError>> {
    let mut addrs = addrs.iter().copied().map(Some).collect::<Vec<_>>();
    let mut ids = ids.to_vec();
    match drift {
        Drift::SwappedRows => {
            addrs.swap(DRIFT_AT, DRIFT_AT + 1);
            ids.swap(DRIFT_AT, DRIFT_AT + 1);
        }
        Drift::MissingRow => {
            addrs.remove(DRIFT_AT);
            ids.remove(DRIFT_AT);
        }
        Drift::DeletedRow => {
            // Physical row 15 is deleted; it sorts just before live row 17.
            let at = ids.iter().position(|id| *id == 17).unwrap();
            let frag_id = RowAddress::from(addrs[0].unwrap()).fragment_id();
            addrs.insert(at, Some(RowAddress::new_from_parts(frag_id, 15).into()));
            ids.insert(at, 15);
        }
        Drift::ForeignFragment => {
            let addr = RowAddress::from(addrs[DRIFT_AT].unwrap());
            addrs[DRIFT_AT] =
                Some(RowAddress::new_from_parts(addr.fragment_id() + 1, addr.row_offset()).into());
        }
        Drift::NullRowAddr => addrs[DRIFT_AT] = None,
        Drift::TooFewRows => {
            addrs.truncate(addrs.len() - 3);
            ids.truncate(ids.len() - 3);
        }
        Drift::TooManyRows => {
            let last = RowAddress::from(addrs.last().unwrap().unwrap());
            addrs.push(Some(
                RowAddress::new_from_parts(last.fragment_id(), last.row_offset() + 1).into(),
            ));
            ids.push(ROWS);
        }
        Drift::StreamError => {}
    }
    let mut batches = chunked(&new_values(&addrs, &ids))
        .into_iter()
        .map(Ok)
        .collect::<Vec<_>>();
    if matches!(drift, Drift::StreamError) {
        // Fail after the batches covering the first DRIFT_AT rows.
        let mut rows = 0;
        let at = batches
            .iter()
            .position(|batch| {
                rows += batch.as_ref().unwrap().num_rows();
                rows > DRIFT_AT
            })
            .unwrap();
        batches.insert(at, Err(ArrowError::ComputeError("source failed".into())));
    }
    batches
}

#[rstest]
#[case::swapped_rows(Drift::SwappedRows, "_rowaddr")]
#[case::missing_row(Drift::MissingRow, "_rowaddr")]
#[case::deleted_row(Drift::DeletedRow, "_rowaddr")]
#[case::foreign_fragment(Drift::ForeignFragment, "_rowaddr")]
#[case::null_rowaddr(Drift::NullRowAddr, "null")]
#[case::too_few_rows(Drift::TooFewRows, "ended")]
#[case::too_many_rows(Drift::TooManyRows, "more rows")]
#[case::stream_error(Drift::StreamError, "source failed")]
#[tokio::test]
async fn test_rejects_drifting_stream(#[case] drift: Drift, #[case] expected: &str) {
    let test_uri = TempStrDir::default();
    let dataset = base_dataset(&test_uri, LanceFileVersion::Stable, false, true).await;
    let fragment = only_fragment(&dataset);
    let (addrs, ids) = live_rows(&fragment).await;
    let files_before = count_files(&dataset).await;

    let stream = RecordBatchIterator::new(drifted_batches(drift, &addrs, &ids), values_schema());
    let err = fragment
        .update_columns_from_stream(stream, Some(BATCH_SIZE))
        .await
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(expected),
        "expected {expected:?} in {message:?}"
    );
    if !matches!(drift, Drift::StreamError) {
        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
    }
    assert_eq!(
        count_files(&dataset).await,
        files_before,
        "a rejected stream must not leave data files behind"
    );
}

#[rstest]
#[case::missing_rowaddr(vec![ArrowField::new("v", DataType::Int32, true)], "_rowaddr")]
#[case::rowaddr_type(
    vec![
        ArrowField::new(ROW_ADDR, DataType::Int64, true),
        ArrowField::new("v", DataType::Int32, true),
    ],
    "UInt64"
)]
#[case::no_value_columns(
    vec![ArrowField::new(ROW_ADDR, DataType::UInt64, true)],
    "no columns to update"
)]
#[case::unknown_column(
    vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        ArrowField::new("w", DataType::Int32, true),
    ],
    "does not exist"
)]
#[case::system_column(
    vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        ArrowField::new(ROW_ID, DataType::UInt64, true),
        ArrowField::new("v", DataType::Int32, true),
    ],
    "reserved"
)]
#[case::type_mismatch(
    vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        ArrowField::new("v", DataType::Int64, true),
    ],
    "different types"
)]
#[case::duplicate_column(
    vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        ArrowField::new("v", DataType::Int32, true),
        ArrowField::new("v", DataType::Int32, true),
    ],
    "appears twice"
)]
#[tokio::test]
async fn test_rejects_bad_schema(#[case] fields: Vec<ArrowField>, #[case] expected: &str) {
    let test_uri = TempStrDir::default();
    let dataset = base_dataset(&test_uri, LanceFileVersion::Stable, false, false).await;
    let files_before = count_files(&dataset).await;

    // The schema is checked before any batch is pulled, so none is needed.
    let stream = reader_of(Arc::new(ArrowSchema::new(fields)), vec![]);
    let err = only_fragment(&dataset)
        .update_columns_from_stream(stream, None)
        .await
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(expected),
        "expected {expected:?} in {message:?}"
    );
    assert_eq!(count_files(&dataset).await, files_before);
}

/// A batch whose columns differ from the declared stream schema would be
/// written positionally, so swapping two same-typed columns would silently
/// store each one's values under the other.
#[tokio::test]
async fn test_rejects_batch_not_matching_stream_schema() {
    let test_uri = TempStrDir::default();
    let dataset = base_dataset(&test_uri, LanceFileVersion::Stable, false, false).await;
    let fragment = only_fragment(&dataset);
    let (addrs, ids) = live_rows(&fragment).await;

    let declared = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("v", DataType::Int32, true),
    ]));
    let swapped = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        ArrowField::new("v", DataType::Int32, true),
        ArrowField::new("id", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        swapped,
        vec![
            Arc::new(UInt64Array::from(addrs)),
            Arc::new(Int32Array::from(vec![7; ids.len()])),
            Arc::new(Int32Array::from(ids)),
        ],
    )
    .unwrap();

    let err = fragment
        .update_columns_from_stream(reader_of(declared, vec![batch]), None)
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("does not match the stream schema"),
        "{err}"
    );
}

/// Structs are written whole: supplying one child would leave its siblings in
/// the old file under a parent whose validity now comes from the new one.
#[tokio::test]
async fn test_rejects_partial_struct() {
    let children = Fields::from(vec![
        ArrowField::new("a", DataType::Int32, true),
        ArrowField::new("b", DataType::Int32, true),
    ]);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new("st", DataType::Struct(children.clone()), true),
    ]));
    let st = StructArray::new(
        children,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(Int32Array::from(vec![3, 4])) as ArrayRef,
        ],
        None,
    );
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![0, 1])), Arc::new(st)],
    )
    .unwrap();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let fragment = only_fragment(&dataset);
    let (addrs, _) = live_rows(&fragment).await;

    let only_a = Fields::from(vec![ArrowField::new("a", DataType::Int32, true)]);
    let stream_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        ArrowField::new("st", DataType::Struct(only_a.clone()), true),
    ]));
    let partial = RecordBatch::try_new(
        stream_schema.clone(),
        vec![
            Arc::new(UInt64Array::from(addrs)),
            Arc::new(StructArray::new(
                only_a,
                vec![Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef],
                None,
            )),
        ],
    )
    .unwrap();
    let err = fragment
        .update_columns_from_stream(reader_of(stream_schema, vec![partial]), None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("whole"), "{err}");
}

/// The returned offsets are what a stable-row-id commit stamps with the new
/// version, so every live row must advance and no deleted one may.
#[tokio::test]
async fn test_stable_row_ids_advance_last_updated() {
    let test_uri = TempStrDir::default();
    let dataset = base_dataset(&test_uri, LanceFileVersion::Stable, true, true).await;
    let fragment = only_fragment(&dataset);
    let (addrs, ids) = live_rows(&fragment).await;
    let created_before = row_versions(&dataset, ROW_CREATED_AT_VERSION).await;

    let addrs = addrs.into_iter().map(Some).collect::<Vec<_>>();
    let result = fragment
        .update_columns_from_stream(
            reader_of(values_schema(), chunked(&new_values(&addrs, &ids))),
            Some(BATCH_SIZE),
        )
        .await
        .unwrap();
    let dataset = commit_rewrite(&dataset, result).await;
    let new_version = dataset.manifest.version;

    let last_updated = row_versions(&dataset, ROW_LAST_UPDATED_AT_VERSION).await;
    assert_eq!(last_updated, vec![new_version; ids.len()]);
    assert_eq!(
        row_versions(&dataset, ROW_CREATED_AT_VERSION).await,
        created_before
    );
    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(
        batch["v"].as_ref(),
        &Int32Array::from_iter_values(ids.iter().map(|id| id * 10))
    );
}

async fn row_versions(dataset: &Dataset, column: &str) -> Vec<u64> {
    let mut scanner = dataset.scan();
    scanner.project(&[column]).unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    batch[column].as_primitive::<UInt64Type>().values().to_vec()
}

/// Struct children may arrive in any order; each must keep its own values,
/// including in a packed struct, which stores all children in one column.
#[rstest]
#[tokio::test]
async fn test_struct_children_matched_by_name(#[values(false, true)] packed: bool) {
    use lance_encoding::constants::PACKED_STRUCT_META_KEY;

    let children = Fields::from(vec![
        ArrowField::new("a", DataType::Int32, true),
        ArrowField::new("b", DataType::Int32, true),
    ]);
    let mut st_field = ArrowField::new("st", DataType::Struct(children.clone()), true);
    if packed {
        st_field = st_field.with_metadata(HashMap::from([(
            PACKED_STRUCT_META_KEY.to_string(),
            "true".to_string(),
        )]));
    }
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        st_field,
    ]));
    let st = StructArray::new(
        children,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(Int32Array::from(vec![10, 20, 30])) as ArrayRef,
        ],
        None,
    );
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2])), Arc::new(st)],
    )
    .unwrap();
    let test_uri = TempStrDir::default();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let fragment = only_fragment(&dataset);
    let (addrs, _) = live_rows(&fragment).await;

    // b before a.
    let reordered = Fields::from(vec![
        ArrowField::new("b", DataType::Int32, true),
        ArrowField::new("a", DataType::Int32, true),
    ]);
    let stream_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        ArrowField::new("st", DataType::Struct(reordered.clone()), true),
    ]));
    let update = RecordBatch::try_new(
        stream_schema.clone(),
        vec![
            Arc::new(UInt64Array::from(addrs)),
            Arc::new(StructArray::new(
                reordered,
                vec![
                    Arc::new(Int32Array::from(vec![400, 500, 600])) as ArrayRef,
                    Arc::new(Int32Array::from(vec![4, 5, 6])) as ArrayRef,
                ],
                None,
            )),
        ],
    )
    .unwrap();
    let result = fragment
        .update_columns_from_stream(reader_of(stream_schema, vec![update]), None)
        .await
        .unwrap();
    let dataset = commit_rewrite(&dataset, result).await;

    let batch = dataset.scan().try_into_batch().await.unwrap();
    let st = batch["st"].as_struct();
    assert_eq!(
        st.column_by_name("a").unwrap().as_ref(),
        &Int32Array::from(vec![4, 5, 6])
    );
    assert_eq!(
        st.column_by_name("b").unwrap().as_ref(),
        &Int32Array::from(vec![400, 500, 600])
    );
}

/// A rewrite replaces every live row, so a data overlay over the same field
/// must stop answering for it once the update commits.
#[tokio::test]
async fn test_rewrite_supersedes_overlay() {
    let test_uri = TempStrDir::default();
    let dataset = base_dataset(&test_uri, LanceFileVersion::Stable, false, false).await;
    let fragment = only_fragment(&dataset);
    let (addrs, ids) = live_rows(&fragment).await;

    let v_schema = dataset.schema().project(&["v"]).unwrap();
    let mut overlay = fragment.write_overlay(&v_schema).await.unwrap();
    let overlay_batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new(ROW_ADDR, DataType::UInt64, false),
            ArrowField::new("v", DataType::Int32, true),
        ])),
        vec![
            Arc::new(UInt64Array::from(vec![addrs[3], addrs[20]])),
            Arc::new(Int32Array::from(vec![777, 777])),
        ],
    )
    .unwrap();
    overlay.write_batch(&overlay_batch).await.unwrap();
    let group = overlay.finish().await.unwrap().unwrap();
    let dataset = Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset.clone())),
        Operation::DataOverlay {
            groups: vec![group],
        },
        Some(dataset.manifest.version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap();
    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(batch["v"].as_primitive::<Int32Type>().value(3), 777);

    let addrs = addrs.into_iter().map(Some).collect::<Vec<_>>();
    let result = only_fragment(&dataset)
        .update_columns_from_stream(
            reader_of(values_schema(), chunked(&new_values(&addrs, &ids))),
            Some(BATCH_SIZE),
        )
        .await
        .unwrap();
    let dataset = commit_rewrite(&dataset, result).await;

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(
        batch["v"].as_ref(),
        &Int32Array::from_iter_values(ids.iter().map(|id| id * 10))
    );
}

/// Legacy files would read some supplied values back differently (null list
/// rows as values, empty strings as null), so the format is refused before
/// anything is written.
#[tokio::test]
async fn test_rejects_legacy_format() {
    let test_uri = TempStrDir::default();
    let dataset = base_dataset(&test_uri, LanceFileVersion::Legacy, false, false).await;
    let fragment = only_fragment(&dataset);
    let (addrs, ids) = live_rows(&fragment).await;
    let files_before = count_files(&dataset).await;

    let addrs = addrs.into_iter().map(Some).collect::<Vec<_>>();
    let err = fragment
        .update_columns_from_stream(
            reader_of(values_schema(), vec![new_values(&addrs, &ids)]),
            None,
        )
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");
    assert!(err.to_string().contains("legacy"), "{err}");
    assert_eq!(count_files(&dataset).await, files_before);
}

/// A non-nullable column must not take a null from the stream.
#[tokio::test]
async fn test_rejects_null_in_non_nullable_column() {
    let test_uri = TempStrDir::default();
    let dataset = base_dataset(&test_uri, LanceFileVersion::Stable, false, false).await;
    let fragment = only_fragment(&dataset);
    let (addrs, ids) = live_rows(&fragment).await;
    let files_before = count_files(&dataset).await;

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        ArrowField::new("id", DataType::Int32, true),
    ]));
    let mut values = ids.into_iter().map(Some).collect::<Vec<_>>();
    values[5] = None;
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(addrs)),
            Arc::new(Int32Array::from(values)),
        ],
    )
    .unwrap();
    let err = fragment
        .update_columns_from_stream(reader_of(schema, vec![batch]), None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("non-null"), "{err}");
    assert_eq!(count_files(&dataset).await, files_before);
}

/// Arrow JSON and view types arrive in their logical form and are stored in
/// the physical one; the values must read back unchanged.
#[tokio::test]
async fn test_overwrites_json_and_view_columns() {
    use lance_arrow::ARROW_EXT_NAME_KEY;
    use lance_arrow::json::ARROW_JSON_EXT_NAME;

    let json_metadata = HashMap::from([(
        ARROW_EXT_NAME_KEY.to_string(),
        ARROW_JSON_EXT_NAME.to_string(),
    )]);
    let json_field =
        || ArrowField::new("meta", DataType::Utf8, true).with_metadata(json_metadata.clone());
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        json_field(),
        ArrowField::new("desc", DataType::Utf8View, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![
                r#"{"x":1}"#,
                r#"{"x":2}"#,
                r#"{"x":3}"#,
            ])),
            Arc::new(StringViewArray::from(vec!["d1", "d2", "d3"])),
        ],
    )
    .unwrap();
    let test_uri = TempStrDir::default();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        None,
    )
    .await
    .unwrap();
    let fragment = only_fragment(&dataset);
    let (addrs, _) = live_rows(&fragment).await;

    let stream_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        json_field(),
        ArrowField::new("desc", DataType::Utf8View, true),
    ]));
    let update = RecordBatch::try_new(
        stream_schema.clone(),
        vec![
            Arc::new(UInt64Array::from(addrs)),
            Arc::new(StringArray::from(vec![
                r#"{"y":1}"#,
                r#"{"y":2}"#,
                r#"{"y":3}"#,
            ])),
            Arc::new(StringViewArray::from(vec!["n1", "n2", "n3"])),
        ],
    )
    .unwrap();
    let result = fragment
        .update_columns_from_stream(reader_of(stream_schema, vec![update]), None)
        .await
        .unwrap();
    let dataset = commit_rewrite(&dataset, result).await;

    let batch = dataset.scan().try_into_batch().await.unwrap();
    let meta = batch["meta"].as_string::<i32>();
    assert_eq!(
        meta.iter().collect::<Vec<_>>(),
        vec![Some(r#"{"y":1}"#), Some(r#"{"y":2}"#), Some(r#"{"y":3}"#)]
    );
    let desc = batch["desc"]
        .as_any()
        .downcast_ref::<StringArray>()
        .map(|array| array.iter().collect::<Vec<_>>())
        .unwrap_or_else(|| batch["desc"].as_string_view().iter().collect());
    assert_eq!(desc, vec![Some("n1"), Some("n2"), Some("n3")]);
}

/// Field metadata such as the JSON extension is taken from the declared stream
/// schema, so a batch that omits it is still converted, and the result does not
/// depend on how the stream happens to be chunked.
#[rstest]
#[tokio::test]
async fn test_conversion_follows_declared_schema(#[values(1, 3)] chunk_rows: usize) {
    use lance_arrow::ARROW_EXT_NAME_KEY;
    use lance_arrow::json::ARROW_JSON_EXT_NAME;

    let json_field =
        ArrowField::new("meta", DataType::Utf8, true).with_metadata(HashMap::from([(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        )]));
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        json_field.clone(),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![
                r#"{"x":1}"#,
                r#"{"x":2}"#,
                r#"{"x":3}"#,
            ])),
        ],
    )
    .unwrap();
    let test_uri = TempStrDir::default();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        None,
    )
    .await
    .unwrap();
    let fragment = only_fragment(&dataset);
    let (addrs, _) = live_rows(&fragment).await;

    let row_addr = ArrowField::new(ROW_ADDR, DataType::UInt64, true);
    let declared = Arc::new(ArrowSchema::new(vec![row_addr.clone(), json_field]));
    let bare = Arc::new(ArrowSchema::new(vec![
        row_addr,
        ArrowField::new("meta", DataType::Utf8, true),
    ]));
    let update = RecordBatch::try_new(
        bare,
        vec![
            Arc::new(UInt64Array::from(addrs)),
            Arc::new(StringArray::from(vec![
                r#"{"y":1}"#,
                r#"{"y":2}"#,
                r#"{"y":3}"#,
            ])),
        ],
    )
    .unwrap();
    let batches = (0..update.num_rows())
        .step_by(chunk_rows)
        .map(|offset| update.slice(offset, chunk_rows.min(update.num_rows() - offset)))
        .collect();
    let result = fragment
        .update_columns_from_stream(reader_of(declared, batches), None)
        .await
        .unwrap();
    let dataset = commit_rewrite(&dataset, result).await;

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(
        batch["meta"].as_string::<i32>().iter().collect::<Vec<_>>(),
        vec![Some(r#"{"y":1}"#), Some(r#"{"y":2}"#), Some(r#"{"y":3}"#)]
    );
}

/// The backfill shape: a column declared all-null has no data file yet, so
/// the written file is the first to answer for it and nothing is tombstoned.
#[tokio::test]
async fn test_backfills_column_declared_all_null() {
    use crate::dataset::schema_evolution::NewColumnTransform;

    let test_uri = TempStrDir::default();
    let mut dataset = base_dataset(&test_uri, LanceFileVersion::Stable, false, true).await;
    dataset
        .add_columns(
            NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "tag",
                DataType::Int32,
                true,
            )]))),
            None,
            None,
        )
        .await
        .unwrap();
    let fragment = only_fragment(&dataset);
    let files_before = fragment.metadata().files.clone();
    let (addrs, ids) = live_rows(&fragment).await;

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        ArrowField::new("tag", DataType::Int32, true),
    ]));
    let update = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(addrs)),
            Arc::new(Int32Array::from_iter_values(ids.iter().map(|id| id + 1000))),
        ],
    )
    .unwrap();
    let result = fragment
        .update_columns_from_stream(reader_of(schema, chunked(&update)), Some(BATCH_SIZE))
        .await
        .unwrap();
    let tag_id = dataset.schema().field("tag").unwrap().id;
    assert_eq!(result.fields_modified, vec![tag_id as u32]);
    assert_eq!(
        &result.fragment.files[..files_before.len()],
        &files_before[..]
    );
    let dataset = commit_rewrite(&dataset, result).await;
    dataset.validate().await.unwrap();

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(
        batch["tag"].as_ref(),
        &Int32Array::from_iter_values(ids.iter().map(|id| id + 1000))
    );
    assert_eq!(batch["v"].as_ref(), &Int32Array::from(vec![-1; ids.len()]));
}

/// An index over a rewritten column still holds the old values, so the commit
/// has to stop it answering for this fragment; a filter must see the new ones.
#[tokio::test]
async fn test_index_on_rewritten_column_is_not_stale() {
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;

    use crate::index::DatasetIndexExt;

    let test_uri = TempStrDir::default();
    let mut dataset = base_dataset(&test_uri, LanceFileVersion::Stable, false, false).await;
    dataset
        .create_index(
            &["v"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let fragment = only_fragment(&dataset);
    let (addrs, ids) = live_rows(&fragment).await;

    let addrs = addrs.into_iter().map(Some).collect::<Vec<_>>();
    let result = fragment
        .update_columns_from_stream(
            reader_of(values_schema(), chunked(&new_values(&addrs, &ids))),
            Some(BATCH_SIZE),
        )
        .await
        .unwrap();
    let dataset = commit_rewrite(&dataset, result).await;

    let count = |filter: &'static str| {
        let dataset = dataset.clone();
        async move {
            let mut scanner = dataset.scan();
            scanner.filter(filter).unwrap();
            scanner.try_into_batch().await.unwrap().num_rows()
        }
    };
    assert_eq!(count("v = -1").await, 0);
    assert_eq!(count("v = 100").await, 1);
}

/// The complete logical blob layout names a byte range of an external object.
/// The manifest stores the minimal `data, uri` layout, so projecting onto it
/// would drop the range and store the whole object instead. One of the three
/// rows is deleted, and with one row per batch its placeholder is written
/// before, between or after the live rows, in the same layout.
#[rstest]
#[tokio::test]
async fn test_keeps_external_blob_range(
    #[values(false, true)] nested: bool,
    #[values(false, true)] reversed: bool,
    #[values(0, 1, 2)] deleted: i32,
) {
    use crate::blob::{BlobArrayBuilder, blob_field};
    use arrow_array::LargeBinaryArray;
    use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME};
    use lance_core::datatypes::BLOB_V2_LOGICAL_FIELDS;
    use lance_table::format::BasePath;

    let table = TempStrDir::default();
    let objects = TempStrDir::default();
    let payload = std::path::Path::new(objects.as_ref()).join("payload.bin");
    std::fs::write(&payload, b"0123456789").unwrap();
    let payload_uri = format!("file://{}", payload.display());

    // Wraps a blob column in `info: struct<blob>` when `nested`.
    let wrap = |field: ArrowField, array: ArrayRef| -> (ArrowField, ArrayRef) {
        if !nested {
            return (field, array);
        }
        let children = Fields::from(vec![field]);
        (
            ArrowField::new("info", DataType::Struct(children.clone()), true),
            Arc::new(StructArray::new(children, vec![array], None)),
        )
    };
    let column = if nested { "info.blob" } else { "blob" };

    let mut builder = BlobArrayBuilder::new(3);
    builder.push_bytes(b"one").unwrap();
    builder.push_bytes(b"deleted").unwrap();
    builder.push_bytes(b"two").unwrap();
    let (base_field, base_array) = wrap(blob_field("blob", true), builder.finish().unwrap());
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        base_field,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2])), base_array],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &table,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            initial_bases: Some(vec![BasePath::new(
                7,
                format!("file://{}", objects.as_ref()),
                None,
                false,
            )]),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.delete(&format!("id = {deleted}")).await.unwrap();
    let fragment = only_fragment(&dataset);
    let (addrs, _) = live_rows(&fragment).await;

    // In `reversed`, the children arrive as size, position, uri, data.
    let mut children: Vec<(Arc<ArrowField>, ArrayRef)> = BLOB_V2_LOGICAL_FIELDS
        .iter()
        .cloned()
        .zip([
            Arc::new(LargeBinaryArray::from(vec![Some(b"zero".as_slice()), None])) as ArrayRef,
            Arc::new(StringArray::from(vec![None, Some(payload_uri.as_str())])),
            Arc::new(UInt64Array::from(vec![None, Some(2)])),
            Arc::new(UInt64Array::from(vec![None, Some(3)])),
        ])
        .collect();
    if reversed {
        children.reverse();
    }
    let (child_fields, child_arrays): (Vec<_>, Vec<_>) = children.into_iter().unzip();
    let complete =
        ArrowField::new("blob", DataType::Struct(child_fields.clone().into()), true).with_metadata(
            HashMap::from([(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string())]),
        );
    let blobs = StructArray::try_new(child_fields.into(), child_arrays, None).unwrap();
    let (field, array) = wrap(complete, Arc::new(blobs));
    let stream_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        field,
    ]));
    let update = RecordBatch::try_new(
        stream_schema.clone(),
        vec![Arc::new(UInt64Array::from(addrs.clone())), array],
    )
    .unwrap();
    let result = fragment
        .update_columns_from_stream(reader_of(stream_schema, vec![update]), Some(1))
        .await
        .unwrap();
    let dataset = Arc::new(commit_rewrite(&dataset, result).await);

    let mut contents = Vec::new();
    for blob in dataset
        .take_blobs_by_addresses(&addrs, column)
        .await
        .unwrap()
    {
        contents.push(blob.unwrap().read().await.unwrap().to_vec());
    }
    assert_eq!(contents, vec![b"zero".to_vec(), b"234".to_vec()]);
}

/// Blob input other than logical blobs is refused before any batch is pulled:
/// prepared and descriptor blobs reference sidecars relative to a data file the
/// caller cannot name, and a binary column tagged as a blob has no children.
#[rstest]
#[case::prepared("prepared")]
#[case::descriptor("descriptor")]
#[case::tagged_binary("tagged_binary")]
#[case::wrong_child_type("wrong_child_type")]
#[tokio::test]
async fn test_rejects_non_logical_blob_input(#[case] shape: &str) {
    use crate::blob::{BlobArrayBuilder, blob_field};
    use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME};
    use lance_core::datatypes::{BLOB_V2_DESC_FIELDS, BLOB_V2_PREPARED_FIELDS};

    let mut builder = BlobArrayBuilder::new(1);
    builder.push_bytes(b"one").unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![blob_field("blob", true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![builder.finish().unwrap()]).unwrap();
    let test_uri = TempStrDir::default();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let data_type = match shape {
        "prepared" => DataType::Struct(BLOB_V2_PREPARED_FIELDS.clone()),
        "descriptor" => DataType::Struct(BLOB_V2_DESC_FIELDS.clone()),
        // The minimal layout's names with the wrong `uri` type.
        "wrong_child_type" => DataType::Struct(Fields::from(vec![
            ArrowField::new("data", DataType::LargeBinary, true),
            ArrowField::new("uri", DataType::LargeUtf8, true),
        ])),
        _ => DataType::LargeBinary,
    };
    let blob = ArrowField::new("blob", data_type, true).with_metadata(HashMap::from([(
        ARROW_EXT_NAME_KEY.to_string(),
        BLOB_V2_EXT_NAME.to_string(),
    )]));
    let stream_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        blob,
    ]));
    let err = only_fragment(&dataset)
        .update_columns_from_stream(reader_of(stream_schema, vec![]), None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("logical blobs"), "{err}");
}

/// File version 2.0 cannot store struct validity at any depth, and its writer
/// refuses a null struct a reader can see. The refusal must leave no file
/// behind, while a null under a null list slot, or in a row a slice drops,
/// must pass, as must the nulls 2.0 can store, which read back unchanged.
#[rstest]
#[case::nested_struct("nested_struct", Some("inner"))]
#[case::list_struct_item("list_struct", Some("item"))]
#[case::masked_under_null_list("masked_list_struct", None)]
#[case::sliced_visible("sliced_list_struct_visible", Some("item"))]
#[case::sliced_away("sliced_list_struct_dropped", None)]
#[case::struct_child("struct_child", None)]
#[case::list_item("list_item", None)]
#[case::null_list_row("null_list_row", None)]
#[case::null_vector_row("null_vector_row", None)]
#[tokio::test]
async fn test_rejects_struct_nulls_on_v2_0(
    #[case] shape: &str,
    #[case] rejected_path: Option<&str>,
) {
    use arrow_array::{FixedSizeListArray, ListArray};
    use arrow_buffer::{NullBuffer, OffsetBuffer};

    let int = |values: Vec<Option<i32>>| Arc::new(Int32Array::from(values)) as ArrayRef;
    let item = Arc::new(ArrowField::new("item", DataType::Int32, true));
    let x = Fields::from(vec![ArrowField::new("x", DataType::Int32, true)]);
    let struct_item = Arc::new(ArrowField::new("item", DataType::Struct(x.clone()), true));
    // A list of single-field structs, with `item_nulls` on the structs and
    // `list_nulls` on the list slots.
    let struct_list =
        |lengths: &[usize], item_nulls: Option<Vec<bool>>, list_nulls: Option<Vec<bool>>| {
            let len = lengths.iter().sum::<usize>() as i32;
            let items = StructArray::new(
                x.clone(),
                vec![int((0..len).map(Some).collect())],
                item_nulls.map(NullBuffer::from),
            );
            ListArray::new(
                struct_item.clone(),
                OffsetBuffer::from_lengths(lengths.iter().copied()),
                Arc::new(items),
                list_nulls.map(NullBuffer::from),
            )
        };
    // Each shape gives the column, the base value and the update, three rows.
    let (field, base, update): (ArrowField, ArrayRef, ArrayRef) = match shape {
        "struct_child" => {
            let make =
                |values| Arc::new(StructArray::new(x.clone(), vec![int(values)], None)) as ArrayRef;
            (
                ArrowField::new("c", DataType::Struct(x.clone()), true),
                make(vec![Some(1), Some(2), Some(3)]),
                make(vec![Some(10), None, Some(30)]),
            )
        }
        "nested_struct" => {
            let outer = Fields::from(vec![ArrowField::new(
                "inner",
                DataType::Struct(x.clone()),
                true,
            )]);
            let nest = |nulls: Option<NullBuffer>| {
                let inner = StructArray::new(x.clone(), vec![int(vec![Some(1); 3])], nulls);
                Arc::new(StructArray::new(outer.clone(), vec![Arc::new(inner)], None)) as ArrayRef
            };
            (
                ArrowField::new("c", DataType::Struct(outer.clone()), true),
                nest(None),
                nest(Some(NullBuffer::from(vec![true, false, true]))),
            )
        }
        "list_struct" | "masked_list_struct" => {
            // The second struct item is null; in the masked shape it sits
            // under a null list slot, so no reader can see it.
            let list_nulls = (shape == "masked_list_struct").then(|| vec![true, false, true]);
            (
                ArrowField::new("c", DataType::List(struct_item.clone()), true),
                Arc::new(struct_list(&[1, 2, 1], None, None)),
                Arc::new(struct_list(
                    &[1, 2, 1],
                    Some(vec![true, false, true, true]),
                    list_nulls,
                )),
            )
        }
        "sliced_list_struct_visible" | "sliced_list_struct_dropped" => {
            // Sliced to rows 1..4, so the offsets do not start at 0. The null
            // struct item is either in a kept row or in the dropped row 0.
            let item_nulls = if shape == "sliced_list_struct_visible" {
                vec![true, true, true, false, true, true]
            } else {
                vec![false, true, true, true, true, true]
            };
            let full = struct_list(&[2, 1, 2, 1], Some(item_nulls), None);
            (
                ArrowField::new("c", DataType::List(struct_item.clone()), true),
                Arc::new(struct_list(&[1, 2, 1], None, None)),
                Arc::new(full.slice(1, 3)),
            )
        }
        "list_item" | "null_list_row" => {
            let list = |values: Vec<Option<i32>>, nulls: Option<NullBuffer>| {
                Arc::new(ListArray::new(
                    item.clone(),
                    OffsetBuffer::from_lengths([1, 2, 1]),
                    int(values),
                    nulls,
                )) as ArrayRef
            };
            let update = if shape == "null_list_row" {
                list(
                    vec![Some(10), Some(20), Some(21), Some(30)],
                    Some(NullBuffer::from(vec![true, false, true])),
                )
            } else {
                list(vec![Some(10), None, Some(21), Some(30)], None)
            };
            (
                ArrowField::new("c", DataType::List(item.clone()), true),
                list(vec![Some(1), Some(2), Some(2), Some(3)], None),
                update,
            )
        }
        "null_vector_row" => {
            let list = |values: Vec<Option<i32>>, nulls: Option<NullBuffer>| {
                Arc::new(FixedSizeListArray::new(item.clone(), 2, int(values), nulls)) as ArrayRef
            };
            (
                ArrowField::new("c", DataType::FixedSizeList(item.clone(), 2), true),
                list(vec![Some(1); 6], None),
                list(
                    vec![Some(10), Some(10), None, None, Some(30), Some(30)],
                    Some(NullBuffer::from(vec![true, false, true])),
                ),
            )
        }
        other => panic!("unknown shape {other}"),
    };
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        field.clone(),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![0, 1, 2])), base],
    )
    .unwrap();
    let test_uri = TempStrDir::default();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_0),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let fragment = only_fragment(&dataset);
    let (addrs, _) = live_rows(&fragment).await;
    let files_before = count_files(&dataset).await;

    let stream_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        field,
    ]));
    let update_batch = RecordBatch::try_new(
        stream_schema.clone(),
        vec![Arc::new(UInt64Array::from(addrs)), update.clone()],
    )
    .unwrap();
    let result = fragment
        .update_columns_from_stream(reader_of(stream_schema, vec![update_batch]), None)
        .await;
    if let Some(path) = rejected_path {
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains(&format!("struct field `{path}`")),
            "{err}"
        );
        assert_eq!(count_files(&dataset).await, files_before);
        return;
    }
    let dataset = commit_rewrite(&dataset, result.unwrap()).await;
    let batch = dataset.scan().try_into_batch().await.unwrap();
    let read = batch["c"].as_ref();
    for row in 0..3 {
        assert_eq!(read.is_valid(row), update.is_valid(row), "row {row}");
        if update.is_valid(row) {
            assert_eq!(
                read.slice(row, 1).to_data(),
                update.slice(row, 1).to_data(),
                "row {row}"
            );
        }
    }
}

/// Blob v2 values arrive as logical blobs and are written through the update
/// writer's blob preprocessing.
#[tokio::test]
async fn test_overwrites_blob_v2_column() {
    use crate::blob::{BlobArrayBuilder, blob_field};

    let blobs = |values: &[&[u8]]| {
        let mut builder = BlobArrayBuilder::new(values.len());
        for value in values {
            builder.push_bytes(value).unwrap();
        }
        builder.finish().unwrap()
    };
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        blob_field("blob", true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2])),
            blobs(&[b"zero", b"one", b"two"]),
        ],
    )
    .unwrap();
    let test_uri = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.delete("id = 1").await.unwrap();
    let fragment = only_fragment(&dataset);
    let (addrs, _) = live_rows(&fragment).await;

    let stream_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(ROW_ADDR, DataType::UInt64, true),
        blob_field("blob", true),
    ]));
    let update = RecordBatch::try_new(
        stream_schema.clone(),
        vec![
            Arc::new(UInt64Array::from(addrs.clone())),
            blobs(&[b"new zero", b"new two"]),
        ],
    )
    .unwrap();
    let result = fragment
        .update_columns_from_stream(reader_of(stream_schema, vec![update]), None)
        .await
        .unwrap();
    let dataset = Arc::new(commit_rewrite(&dataset, result).await);

    let blobs = dataset
        .take_blobs_by_addresses(&addrs, "blob")
        .await
        .unwrap();
    let mut contents = Vec::new();
    for blob in blobs {
        contents.push(blob.unwrap().read().await.unwrap().to_vec());
    }
    assert_eq!(contents, vec![b"new zero".to_vec(), b"new two".to_vec()]);
}

/// The new data file holds a placeholder for every deleted row. Wherever the
/// deleted rows fall, the placeholders must be storable on the file version (no
/// struct null on 2.0, no null in a non-nullable column), and every live value
/// must land on its own row.
#[rstest]
#[tokio::test]
async fn test_fills_deleted_rows(
    #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1, LanceFileVersion::V2_2)]
    version: LanceFileVersion,
    #[values("id < 17", "id >= 20", "id >= 8 AND id < 24", DELETE_PREDICATE)] predicate: &str,
) {
    use arrow_array::builder::{ListBuilder, StringBuilder};

    let st_fields = Fields::from(vec![
        ArrowField::new("x", DataType::Int32, true),
        ArrowField::new("s", DataType::Utf8, true),
        ArrowField::new("k", DataType::Utf8, false),
    ]);
    let value_fields = vec![
        ArrowField::new("st", DataType::Struct(st_fields.clone()), true),
        ArrowField::new("name", DataType::Utf8, false),
        ArrowField::new(
            "tags",
            DataType::List(Arc::new(ArrowField::new("item", DataType::Utf8, true))),
            true,
        ),
    ];
    let tags = |rows: Vec<Option<Vec<String>>>| {
        let mut builder = ListBuilder::new(StringBuilder::new());
        for row in rows {
            if let Some(values) = &row {
                values.iter().for_each(|v| builder.values().append_value(v));
            }
            builder.append(row.is_some());
        }
        Arc::new(builder.finish()) as ArrayRef
    };
    let st = |x: Vec<Option<i32>>, s: Vec<Option<String>>, k: Vec<String>| {
        Arc::new(StructArray::new(
            st_fields.clone(),
            vec![
                Arc::new(Int32Array::from(x)),
                Arc::new(StringArray::from(s)),
                Arc::new(StringArray::from(k)),
            ],
            None,
        )) as ArrayRef
    };

    let schema = Arc::new(ArrowSchema::new(
        [
            vec![ArrowField::new("id", DataType::Int32, false)],
            value_fields.clone(),
        ]
        .concat(),
    ));
    let all = (0..ROWS).collect::<Vec<_>>();
    let base = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(all.clone())),
            st(
                vec![Some(-1); all.len()],
                vec![Some("old".into()); all.len()],
                vec!["old".into(); all.len()],
            ),
            Arc::new(StringArray::from_iter_values(
                all.iter().map(|id| format!("old{id}")),
            )),
            tags(vec![Some(vec!["old".into()]); all.len()]),
        ],
    )
    .unwrap();
    let test_uri = TempStrDir::default();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(base)], schema),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.delete(predicate).await.unwrap();
    let fragment = only_fragment(&dataset);
    let (addrs, ids) = live_rows(&fragment).await;

    // The stream's first row holds nulls wherever the column allows them, since
    // placeholders may be derived from it.
    let first = ids[0];
    let stream_schema = Arc::new(ArrowSchema::new(
        [
            vec![ArrowField::new(ROW_ADDR, DataType::UInt64, true)],
            value_fields,
        ]
        .concat(),
    ));
    let update = RecordBatch::try_new(
        stream_schema.clone(),
        vec![
            Arc::new(UInt64Array::from(addrs)),
            st(
                ids.iter()
                    .map(|id| (id % 5 != 0).then_some(id * 10))
                    .collect(),
                ids.iter()
                    .map(|id| (*id != first).then(|| format!("s{id}")))
                    .collect(),
                ids.iter().map(|id| format!("k{id}")).collect(),
            ),
            Arc::new(StringArray::from_iter_values(
                ids.iter().map(|id| format!("n{id}")),
            )),
            tags(
                ids.iter()
                    .map(|id| (*id != first).then(|| vec![format!("t{id}"); (id % 3) as usize]))
                    .collect(),
            ),
        ],
    )
    .unwrap();
    let result = fragment
        .update_columns_from_stream(reader_of(stream_schema, chunked(&update)), Some(BATCH_SIZE))
        .await
        .unwrap();
    let dataset = commit_rewrite(&dataset, result).await;
    dataset.validate().await.unwrap();
    let fragment = only_fragment(&dataset);
    fragment.validate().await.unwrap();
    assert_eq!(
        fragment.count_deletions().await.unwrap(),
        ROWS as usize - ids.len()
    );

    let read = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(read["id"].as_ref(), &Int32Array::from(ids));
    for column in ["st", "name", "tags"] {
        assert_eq!(read[column].to_data(), update[column].to_data(), "{column}");
    }
}

/// An empty slice of a large batch has no rows but still holds the large
/// batch's buffers. Each empty batch must be released before the next one is
/// pulled, rather than kept until enough live rows have arrived.
#[tokio::test]
async fn test_releases_empty_batches() {
    use datafusion::error::DataFusionError;
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use std::sync::Mutex;

    const EMPTY_BATCHES: usize = 4;
    let test_uri = TempStrDir::default();
    let dataset = base_dataset(&test_uri, LanceFileVersion::Stable, false, false).await;
    let fragment = only_fragment(&dataset);
    let (addrs, ids) = live_rows(&fragment).await;
    let addrs = addrs.into_iter().map(Some).collect::<Vec<_>>();
    let update = new_values(&addrs, &ids);

    // The empty batches are slices of `parent`; `tracked` is its `v` buffer.
    let parent = new_values(&addrs, &ids);
    let tracked = parent["v"].to_data().buffers()[0].clone();
    // How many references the buffer has as each batch is about to be produced.
    let counts = Arc::new(Mutex::new(Vec::new()));
    let observed = counts.clone();
    let batches = (0..=EMPTY_BATCHES).map(move |i| {
        observed.lock().unwrap().push(tracked.strong_count());
        let batch = if i < EMPTY_BATCHES {
            parent.slice(0, 0)
        } else {
            update.clone()
        };
        Ok::<_, DataFusionError>(batch)
    });
    let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
        values_schema(),
        futures::stream::iter(batches),
    ));
    let result = fragment
        .update_columns_from_stream(stream, None)
        .await
        .unwrap();
    commit_rewrite(&dataset, result).await;

    let counts = counts.lock().unwrap().clone();
    assert_eq!(counts, vec![counts[0]; EMPTY_BATCHES + 1]);
}
