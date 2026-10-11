// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Generator for the `zonemap_signed_nan` test fixture.
//!
//! Writes a dataset whose float columns put every NaN/null shape into its own
//! ZoneMap zone, builds a seeded ZoneMap index on each float column, then
//! appends a second fragment so the appended data file carries a write seed.
//! The writer predates `negative_nan_count`: any NaN in a zone is written as
//! `max = +NaN`, and neither the index nor the seeds say which sign the NaNs had.
//!
//! Standalone crate — not part of the parent workspace because it must compile
//! against that older writer (pinned in `Cargo.toml`).
//!
//! ```bash
//! cargo run --release \
//!     --manifest-path test_data/v14.0.0-beta.10/zonemap_signed_nan/datagen/Cargo.toml -- \
//!     test_data/v14.0.0-beta.10/zonemap_signed_nan/dataset.lance
//! ```

use std::sync::Arc;

use arrow_array::{
    Float16Array, Float32Array, Float64Array, Int64Array, RecordBatch, RecordBatchIterator,
};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::DatasetIndexExt;
use lance_index::IndexType;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};

const ROWS_PER_ZONE: usize = 8;

/// One zone's worth of values. Written identically into every float column
/// (cast per type), so each zone of each column has a known NaN/null shape.
#[derive(Clone, Copy)]
enum V {
    Num(f64),
    /// Positive NaN; the payload makes bit patterns differ within a zone.
    PosNan(u8),
    /// Negative NaN.
    NegNan(u8),
    Null,
}

/// Zones in fragment order. Each inner array is exactly one zone.
fn zones() -> Vec<[V; ROWS_PER_ZONE]> {
    use V::*;
    vec![
        // 0: ordinary values only, including signed zeros and infinities
        [
            Num(-0.0),
            Num(0.0),
            Num(f64::NEG_INFINITY),
            Num(f64::INFINITY),
            Num(5.0),
            Num(8.0),
            Num(-3.0),
            Num(1.0),
        ],
        // 1: negative NaN with ordinary values
        [
            NegNan(0),
            NegNan(1),
            Num(5.0),
            Num(8.0),
            Num(1.0),
            Num(-3.0),
            Num(0.0),
            Num(2.0),
        ],
        // 2: positive NaN with ordinary values
        [
            PosNan(0),
            PosNan(1),
            Num(5.0),
            Num(8.0),
            Num(1.0),
            Num(-3.0),
            Num(0.0),
            Num(2.0),
        ],
        // 3: both NaN signs, ordinary values and a null
        [
            NegNan(0),
            PosNan(0),
            Num(5.0),
            Num(8.0),
            Num(1.0),
            Num(-3.0),
            Null,
            Num(2.0),
        ],
        // 4: only positive NaN
        [
            PosNan(0),
            PosNan(1),
            PosNan(2),
            PosNan(3),
            PosNan(0),
            PosNan(1),
            PosNan(2),
            PosNan(3),
        ],
        // 5: only negative NaN
        [
            NegNan(0),
            NegNan(1),
            NegNan(2),
            NegNan(3),
            NegNan(0),
            NegNan(1),
            NegNan(2),
            NegNan(3),
        ],
        // 6: only null
        [Null, Null, Null, Null, Null, Null, Null, Null],
        // 7: null and negative NaN, no ordinary values
        [
            Null,
            NegNan(0),
            Null,
            NegNan(1),
            Null,
            NegNan(0),
            Null,
            NegNan(1),
        ],
        // 8: null and positive NaN, no ordinary values
        [
            Null,
            PosNan(0),
            Null,
            PosNan(1),
            Null,
            PosNan(0),
            Null,
            PosNan(1),
        ],
        // 9: ordinary values and null
        [
            Null,
            Num(5.0),
            Null,
            Num(8.0),
            Null,
            Num(-3.0),
            Null,
            Num(1.0),
        ],
    ]
}

fn f64_of(v: V) -> Option<f64> {
    match v {
        V::Num(x) => Some(x),
        V::PosNan(p) => Some(f64::from_bits(0x7ff8_0000_0000_0000 | p as u64)),
        V::NegNan(p) => Some(f64::from_bits(0xfff8_0000_0000_0000 | p as u64)),
        V::Null => None,
    }
}

fn f32_of(v: V) -> Option<f32> {
    match v {
        V::Num(x) => Some(x as f32),
        V::PosNan(p) => Some(f32::from_bits(0x7fc0_0000 | p as u32)),
        V::NegNan(p) => Some(f32::from_bits(0xffc0_0000 | p as u32)),
        V::Null => None,
    }
}

fn f16_of(v: V) -> Option<half::f16> {
    match v {
        V::Num(x) => Some(half::f16::from_f64(x)),
        V::PosNan(p) => Some(half::f16::from_bits(0x7e00 | p as u16)),
        V::NegNan(p) => Some(half::f16::from_bits(0xfe00 | p as u16)),
        V::Null => None,
    }
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("f16", DataType::Float16, true),
        Field::new("f32", DataType::Float32, true),
        Field::new("f64", DataType::Float64, true),
    ]))
}

fn fragment_batch(first_id: i64) -> RecordBatch {
    let values: Vec<V> = zones().into_iter().flatten().collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from_iter_values(
                first_id..first_id + values.len() as i64,
            )),
            Arc::new(Float16Array::from_iter(values.iter().copied().map(f16_of))),
            Arc::new(Float32Array::from_iter(values.iter().copied().map(f32_of))),
            Arc::new(Float64Array::from_iter(values.iter().copied().map(f64_of))),
        ],
    )
    .unwrap()
}

#[tokio::main]
async fn main() {
    let output = std::env::args()
        .nth(1)
        .expect("usage: zonemap-signed-nan-datagen <output dir>");
    let _ = std::fs::remove_dir_all(&output);

    let rows = fragment_batch(0);
    let num_rows = rows.num_rows();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(rows)], schema()),
        &output,
        Some(WriteParams {
            max_rows_per_file: num_rows,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let params = ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap)
        .with_params(&serde_json::json!({"rows_per_zone": ROWS_PER_ZONE, "use_seeds": true}));
    for column in ["f16", "f32", "f64"] {
        dataset
            .create_index(&[column], IndexType::ZoneMap, None, &params, false)
            .await
            .unwrap();
    }

    // Appended after the indices exist, so this fragment's data file carries a
    // write seed for each float column.
    dataset
        .append(
            RecordBatchIterator::new([Ok(fragment_batch(num_rows as i64))], schema()),
            None,
        )
        .await
        .unwrap();

    println!(
        "wrote {} rows in {} fragments to {}",
        dataset.count_rows(None).await.unwrap(),
        dataset.get_fragments().len(),
        output
    );
}
