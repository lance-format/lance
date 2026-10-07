// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Deterministic fragment metadata for tests. Data files are not written.

use std::num::NonZero;

use lance_file::version::ConcreteFileVersion;

use crate::format::{DataFile, Fragment};

const FILE_VERSION: ConcreteFileVersion = ConcreteFileVersion::V2_0;

/// A realistic Lance data-file path `data/<50 chars>.lance` derived
/// deterministically from `(id, salt)`. Matches Lance's real naming: first 3 of
/// 16 UUID bytes → 24 binary chars, remaining 13 → 26 hex chars.
#[cfg_attr(coverage, coverage(off))]
pub fn data_file_path(id: u64, salt: u64) -> String {
    // splitmix64-fill 16 pseudo-random-but-deterministic bytes (a synthetic UUID).
    let mut bytes = [0u8; 16];
    let mut state = id
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(salt.wrapping_mul(0xD1B5_4A32_D192_ED03));
    for chunk in bytes.chunks_mut(8) {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let n = chunk.len();
        chunk.copy_from_slice(&z.to_le_bytes()[..n]);
    }

    let mut stem = String::with_capacity(50);
    for &b in &bytes[..3] {
        for i in (0..8).rev() {
            stem.push(if (b >> i) & 1 == 1 { '1' } else { '0' });
        }
    }
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &b in &bytes[3..] {
        stem.push(HEX[(b >> 4) as usize] as char);
        stem.push(HEX[(b & 0xf) as usize] as char);
    }
    format!("data/{stem}.lance")
}

/// A base 1-row fragment with one two-column data file — the bootstrap table of
/// N tiny fragments.
#[cfg_attr(coverage, coverage(off))]
pub fn make_fragment(id: u64) -> Fragment {
    let mut fragment = Fragment::new(id).with_file(
        data_file_path(id, 0),
        vec![0, 1],
        vec![0, 1],
        FILE_VERSION,
        NonZero::new(1024),
    );
    fragment.physical_rows = Some(1);
    fragment
}

/// The data file that backfill round `col` attaches to `frag_id` (add-column):
/// one new field in a new data file, as an embedding backfill would produce.
#[cfg_attr(coverage, coverage(off))]
pub fn make_backfill_data_file(frag_id: u64, col: u32) -> DataFile {
    DataFile::new(
        data_file_path(frag_id, col as u64 + 1),
        vec![2 + col as i32],
        vec![0],
        FILE_VERSION,
        NonZero::new(4096),
        None,
    )
}

/// The data file that data-replacement round `round` writes for `frag_id`:
/// the same base columns in a fresh file, as an update or compaction rewrite
/// would produce. Path salts start above every backfill salt so replacement
/// paths never collide with add-column paths.
#[cfg_attr(coverage, coverage(off))]
pub fn make_replacement_data_file(frag_id: u64, round: u32) -> DataFile {
    const REPLACEMENT_SALT_BASE: u64 = 1 << 32;
    DataFile::new(
        data_file_path(frag_id, REPLACEMENT_SALT_BASE + round as u64),
        vec![0, 1],
        vec![0, 1],
        FILE_VERSION,
        NonZero::new(1024),
        None,
    )
}

/// A fragment containing one base file and additional single-column files.
#[cfg_attr(coverage, coverage(off))]
pub fn make_fragment_with_files(id: u64, num_files: u32) -> Fragment {
    let mut fragment = make_fragment(id);
    for col in 0..num_files.saturating_sub(1) {
        fragment.files.push(make_backfill_data_file(id, col));
    }
    fragment
}
