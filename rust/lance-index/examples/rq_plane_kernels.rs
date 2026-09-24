// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Non-production single-thread kernel POC. Every timing cell has an untimed warm-up.
use lance_index::vector::bq::ex_dot::{
    blocked_ex_code_bytes, ex_dot_kernel_variants, ex_dot_prefix_kernel_variants, pack_blocked_row,
};
use lance_index::vector::bq::layered::RQLayout;
use std::hint::black_box;
use std::time::Instant;

const RESIDENT_ROWS: usize = 256;

fn measure(name: &str, dim: usize, run: impl Fn(usize) -> f32) {
    for row in 0..500 {
        black_box(run(row % RESIDENT_ROWS));
    }
    let mut timings = Vec::new();
    for _ in 0..7 {
        let started = Instant::now();
        for row in 0..200_000 {
            black_box(run(row % RESIDENT_ROWS));
        }
        timings.push(started.elapsed().as_nanos() as f64 / 200_000.);
    }
    timings.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"kernel":name,"dim":dim,"median_ns_row":timings[3],"rounds_ns_row":timings,"rows_per_round":200_000,"resident_rows":RESIDENT_ROWS})
    );
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for dim in [64, 128, 1024, 1536, 2048] {
        let q: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.13).sin()).collect();
        for bits in [2, 4, 6, 8] {
            let values: Vec<u8> = (0..dim)
                .map(|i| ((i * 71 + 13) & ((1 << bits) - 1)) as u8)
                .collect();
            let mut packed = vec![0; blocked_ex_code_bytes(dim, bits)];
            pack_blocked_row(&values, bits, &mut packed);
            let row_bytes = packed.len();
            let packed = packed.repeat(RESIDENT_ROWS);
            for (variant, kernel) in ex_dot_kernel_variants(bits) {
                measure(&format!("native_u{bits}_{variant}"), dim, |row| {
                    kernel(
                        black_box(&q),
                        black_box(&packed[row * row_bytes..(row + 1) * row_bytes]),
                    )
                });
            }
        }
        for bits in [5, 7, 9] {
            let layout = RQLayout::try_new(bits)?;
            let values: Vec<u8> = (0..dim)
                .map(|i| ((i * 71 + 13) & ((1 << (bits - 1)) - 1)) as u8)
                .collect();
            let high: Vec<u8> = values.iter().map(|v| v >> layout.low_bits).collect();
            let low: Vec<u8> = values
                .iter()
                .map(|v| v & ((1 << layout.low_bits) - 1))
                .collect();
            let mut hi = vec![0; blocked_ex_code_bytes(dim, layout.high_bits)];
            let mut lo = vec![0; blocked_ex_code_bytes(dim, layout.low_bits)];
            pack_blocked_row(&high, layout.high_bits, &mut hi);
            pack_blocked_row(&low, layout.low_bits, &mut lo);
            let hi_bytes = hi.len();
            let lo_bytes = lo.len();
            let hi = hi.repeat(RESIDENT_ROWS);
            let lo = lo.repeat(RESIDENT_ROWS);
            for (name, kernel) in ex_dot_prefix_kernel_variants(bits) {
                measure(&format!("layered_{bits}_{name}"), dim, |row| {
                    kernel(
                        black_box(&q),
                        black_box(&hi[row * hi_bytes..(row + 1) * hi_bytes]),
                        black_box(&lo[row * lo_bytes..(row + 1) * lo_bytes]),
                    )
                });
            }
        }
    }
    Ok(())
}
