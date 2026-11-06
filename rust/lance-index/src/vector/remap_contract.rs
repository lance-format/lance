// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The contract every vector storage upholds when a partition is read
//! through a [`Remapper`], run against each one: the partition, as an index
//! file stores it, is translated without changing its layout and decoded to
//! exactly the storage an index rewrite through the same translation
//! produces. Moved rows keep their codes at their current addresses, and
//! deleted rows are gone, whether or not the storage's code columns are
//! row-aligned.
//!
//! The script stands in for a fragment reuse history, so no history is
//! built here; the histories themselves are tested against their own
//! contract in lance-table.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow_array::Array;
use arrow_array::types::Float32Type;
use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt8Array, UInt64Array,
};
use arrow_schema::{Field, Schema};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::utils::address::RowAddress;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::{ROW_ID, ROW_ID_FIELD};
use lance_index_core::remapping::translate_preserving_layout;
use lance_linalg::distance::DistanceType;
use lance_select::remapper::ScriptedRemapper;

use super::bq::RQRotationType;
use super::bq::builder::RabitQuantizer;
use super::bq::storage::{RABIT_CODE_COLUMN, RabitQuantizationStorage};
use super::bq::transform::{ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, SCALE_FACTORS_COLUMN};
use super::flat::index::{FlatBinQuantizer, FlatQuantizer};
use super::pq::{PQBuildParams, ProductQuantizer};
use super::quantizer::{Quantization, QuantizerStorage};
use super::sq::ScalarQuantizer;
use super::sq::builder::SQBuildParams;
use super::storage::{StorageBuilder, VectorStore, drop_tombstones};

const DIM: usize = 16;
/// Rows of fragment 0, which the script rewrites, and of fragment 2, which
/// it leaves alone. Enough rows to train an 8-bit product quantizer.
const REWRITTEN: u32 = 256;
const UNTOUCHED: u32 = 32;

fn address(fragment: u32, offset: u32) -> u64 {
    RowAddress::new_from_parts(fragment, offset).into()
}

/// The addresses the partition was built with.
fn stored_row_ids() -> UInt64Array {
    let rewritten = (0..REWRITTEN).map(|offset| address(0, offset));
    let untouched = (0..UNTOUCHED).map(|offset| address(2, offset));
    UInt64Array::from_iter_values(rewritten.chain(untouched))
}

/// Fragment 0 was rewritten into fragment 1 in reverse order, and every
/// seventh row deleted.
fn moves() -> HashMap<u64, Option<u64>> {
    (0..REWRITTEN)
        .map(|offset| {
            let to = (offset % 7 != 3).then(|| address(1, REWRITTEN - 1 - offset));
            (address(0, offset), to)
        })
        .collect()
}

fn script() -> ScriptedRemapper {
    moves()
        .into_iter()
        .fold(ScriptedRemapper::new(), |script, (from, to)| match to {
            Some(to) => script.moved(from, to),
            None => script.deleted(from),
        })
}

fn vectors() -> FixedSizeListArray {
    let rows = (REWRITTEN + UNTOUCHED) as usize;
    let values = Float32Array::from_iter_values(
        (0..rows * DIM).map(|i| ((i * 7919) % 1000) as f32 / 1000.0),
    );
    FixedSizeListArray::try_new_from_values(values, DIM as i32).unwrap()
}

fn built<Q: Quantization>(quantizer: Q, vectors: FixedSizeListArray) -> Q::Storage {
    let schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("vector", vectors.data_type().clone(), true),
    ]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(stored_row_ids()), Arc::new(vectors)]).unwrap();
    StorageBuilder::new("vector".to_owned(), DistanceType::L2, quantizer)
        .unwrap()
        .build(vec![batch])
        .unwrap()
}

fn as_batch(storage: &impl VectorStore) -> RecordBatch {
    let batches: Vec<RecordBatch> = storage.to_batches().unwrap().collect();
    concat_batches(storage.schema(), &batches).unwrap()
}

/// Read `storage`'s partition, as an index file stores it, through the
/// script, and compare it with the rewrite of `storage` through the same
/// translation.
async fn assert_translated_read_matches_rewrite<S: QuantizerStorage>(storage: S) {
    let stored = as_batch(&storage);
    let (remapper, script) = script().into_remapper();
    let row_id_idx = stored.schema().index_of(ROW_ID).unwrap();
    let translated = translate_preserving_layout(&remapper, stored, row_id_idx)
        .await
        .unwrap();
    assert!(
        script.batches() < script.addresses(),
        "{} batches for {} addresses: the read translates row by row",
        script.batches(),
        script.addresses()
    );
    let read = drop_tombstones(
        S::try_from_batch(translated, storage.metadata(), storage.distance_type()).unwrap(),
    )
    .unwrap();

    let rewritten = storage.remap(&RowAddrRemap::direct(moves())).unwrap();
    assert_eq!(as_batch(&read), as_batch(&rewritten));

    let mut row_ids: Vec<u64> = read.row_ids().copied().collect();
    row_ids.sort_unstable();
    let mut expected: Vec<u64> = moves().into_values().flatten().collect();
    expected.extend((0..UNTOUCHED).map(|offset| address(2, offset)));
    expected.sort_unstable();
    assert_eq!(row_ids, expected);
}

#[tokio::test]
async fn flat() {
    let quantizer = FlatQuantizer::new(DIM, DistanceType::L2);
    assert_translated_read_matches_rewrite(built(quantizer, vectors())).await;
}

#[tokio::test]
async fn flat_bin() {
    let rows = (REWRITTEN + UNTOUCHED) as usize;
    let codes = UInt8Array::from_iter_values((0..rows * DIM).map(|i| (i * 31 % 251) as u8));
    let vectors = FixedSizeListArray::try_new_from_values(codes, DIM as i32).unwrap();
    let quantizer = FlatBinQuantizer::new(DIM, DistanceType::Hamming);
    assert_translated_read_matches_rewrite(built(quantizer, vectors)).await;
}

/// Product quantization codes are stored transposed: a row's codes are not
/// in its own slot of the code column.
#[tokio::test]
async fn product() {
    let quantizer =
        ProductQuantizer::build(&vectors(), DistanceType::L2, &PQBuildParams::new(4, 8)).unwrap();
    assert_translated_read_matches_rewrite(built(quantizer, vectors())).await;
}

#[tokio::test]
async fn scalar() {
    let quantizer =
        ScalarQuantizer::build(&vectors(), DistanceType::L2, &SQBuildParams::default()).unwrap();
    assert_translated_read_matches_rewrite(built(quantizer, vectors())).await;
}

/// RaBitQ codes are stored packed in blocks of rows.
#[tokio::test]
async fn rabit() {
    let quantizer =
        RabitQuantizer::new_with_rotation::<Float32Type>(1, DIM as i32, RQRotationType::Fast);
    let codes = quantizer.quantize(&vectors()).unwrap();
    let rows = (REWRITTEN + UNTOUCHED) as usize;
    let factors = |shift: f32| {
        Arc::new(Float32Array::from_iter_values(
            (0..rows).map(|row| row as f32 + shift),
        )) as ArrayRef
    };
    let batch = RecordBatch::try_from_iter(vec![
        (ROW_ID, Arc::new(stored_row_ids()) as ArrayRef),
        (RABIT_CODE_COLUMN, codes),
        (ADD_FACTORS_COLUMN, factors(0.0)),
        (SCALE_FACTORS_COLUMN, factors(0.5)),
        (ERROR_FACTORS_COLUMN, factors(0.25)),
    ])
    .unwrap();
    let storage = RabitQuantizationStorage::try_from_batch(
        batch,
        &quantizer.metadata(None),
        DistanceType::L2,
    )
    .unwrap();
    assert_translated_read_matches_rewrite(storage).await;
}
