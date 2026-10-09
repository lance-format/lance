// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::BTreeSet, ops::Range, sync::Arc};

use lance_core::{Error, Result};

use crate::{
    data::FixedWidthDataBlock,
    encodings::physical::{checked_fixed_values, dictionary::MAX_BLOCK_DICTIONARY_ITEMS},
};

const GENERAL_SAMPLE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone)]
pub(super) struct SequenceStats {
    pub(super) bits_per_value: u64,
    pub(super) len: u64,
    pub(super) first: Option<u64>,
    pub(super) min: u64,
    pub(super) max: u64,
    pub(super) arithmetic_step: Option<u64>,
    pub(super) run_count: u64,
    pub(super) sample: Arc<[u8]>,
}

impl SequenceStats {
    pub(super) fn raw_bytes(&self) -> u64 {
        self.len.saturating_mul(self.bits_per_value / 8)
    }

    #[cfg(feature = "bitpacking")]
    pub(super) fn required_bits(&self) -> u64 {
        u64::from((u64::BITS - self.max.leading_zeros()).max(1))
    }
}

#[derive(Debug)]
pub(super) struct MemberStats {
    pub(super) values: SequenceStats,
    pub(super) deltas: Option<SequenceStats>,
    pub(super) run_values: SequenceStats,
    pub(super) run_lengths: SequenceStats,
}

#[derive(Debug)]
pub(super) struct FamilyStats {
    pub(super) members: Vec<MemberStats>,
    pub(super) dictionary_items: Option<Arc<[u64]>>,
}

#[derive(Debug)]
struct SequenceStatsBuilder {
    bits_per_value: u64,
    len: u64,
    first: Option<u64>,
    previous: Option<u64>,
    min: u64,
    max: u64,
    arithmetic_step: Option<u64>,
    is_arithmetic: bool,
    run_count: u64,
    sample: Vec<u8>,
    sample_limit: usize,
}

impl SequenceStatsBuilder {
    fn new(bits_per_value: u64, sample_limit: usize) -> Self {
        Self {
            bits_per_value,
            len: 0,
            first: None,
            previous: None,
            min: u64::MAX,
            max: 0,
            arithmetic_step: None,
            is_arithmetic: true,
            run_count: 0,
            sample: Vec::new(),
            sample_limit,
        }
    }

    fn push(&mut self, value: u64) {
        if self.first.is_none() {
            self.first = Some(value);
            self.run_count = 1;
        }
        if let Some(previous) = self.previous {
            if value != previous {
                self.run_count += 1;
            }
            let step = value.checked_sub(previous);
            match (self.arithmetic_step, step, self.len) {
                (None, Some(step), 1) => self.arithmetic_step = Some(step),
                (Some(expected), Some(actual), _) if expected == actual => {}
                _ => self.is_arithmetic = false,
            }
        }
        self.previous = Some(value);
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.len += 1;

        if self.sample.len() < self.sample_limit {
            let bytes = value.to_le_bytes();
            let value_bytes = (self.bits_per_value / 8) as usize;
            let remaining = self.sample_limit - self.sample.len();
            self.sample
                .extend_from_slice(&bytes[..value_bytes.min(remaining)]);
        }
    }

    fn finish(self) -> SequenceStats {
        SequenceStats {
            bits_per_value: self.bits_per_value,
            len: self.len,
            first: self.first,
            min: if self.len == 0 { 0 } else { self.min },
            max: self.max,
            arithmetic_step: (self.len >= 2 && self.is_arithmetic)
                .then_some(self.arithmetic_step)
                .flatten(),
            run_count: self.run_count,
            sample: Arc::from(self.sample),
        }
    }
}

pub(super) fn analyze_family(
    data: &FixedWidthDataBlock,
    ranges: &[Range<usize>],
    collect_general_sample: bool,
) -> Result<FamilyStats> {
    if ranges.is_empty() {
        return Err(Error::invalid_input(
            "Offset compression requires at least one chunk",
        ));
    }
    let sample_limit = if collect_general_sample {
        GENERAL_SAMPLE_BYTES / ranges.len()
    } else {
        0
    };
    match data.bits_per_value {
        32 => {
            let values = checked_fixed_values::<u32>(data, "Offset family")?;
            analyze_typed_family(&values, ranges, 32, sample_limit, true, |value| {
                u64::from(value)
            })
        }
        64 => {
            let values = checked_fixed_values::<u64>(data, "Offset family")?;
            analyze_typed_family(&values, ranges, 64, sample_limit, true, |value| value)
        }
        bits_per_value => Err(Error::invalid_input(format!(
            "Offset compression only supports 32 or 64-bit values, got {bits_per_value}"
        ))),
    }
}

fn analyze_typed_family<T: Copy + Eq>(
    values: &[T],
    ranges: &[Range<usize>],
    bits_per_value: u64,
    sample_limit: usize,
    collect_dictionary: bool,
    to_u64: impl Fn(T) -> u64 + Copy,
) -> Result<FamilyStats> {
    let mut dictionary = collect_dictionary.then(BoundedDistinct::default);
    let members = ranges
        .iter()
        .enumerate()
        .map(|(member_index, range)| {
            let member = values.get(range.clone()).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Offset chunk {member_index} range {}..{} exceeds {} values",
                    range.start,
                    range.end,
                    values.len()
                ))
            })?;
            let base = member.first().copied().map(to_u64).ok_or_else(|| {
                Error::invalid_input(format!("Offset chunk {member_index} is empty"))
            })?;
            let mut value_stats = SequenceStatsBuilder::new(bits_per_value, sample_limit);
            let mut delta_stats = SequenceStatsBuilder::new(bits_per_value, 0);
            let mut run_value_stats = SequenceStatsBuilder::new(bits_per_value, 0);
            let mut run_length_stats = SequenceStatsBuilder::new(32, 0);
            let mut previous = None;
            let mut run_value = None;
            let mut run_length = 0_u64;

            for raw_value in member.iter().copied().map(to_u64) {
                let value = raw_value.checked_sub(base).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Offset chunk {member_index} contains a value below its first offset"
                    ))
                })?;
                if let Some(previous) = previous {
                    let delta = value.checked_sub(previous).ok_or_else(|| {
                        Error::invalid_input(format!(
                            "Offset chunk {member_index} must be non-decreasing"
                        ))
                    })?;
                    delta_stats.push(delta);
                }
                previous = Some(value);
                value_stats.push(value);
                if let Some(dictionary) = dictionary.as_mut() {
                    dictionary.push(value);
                }

                match run_value {
                    Some(current) if current == value => run_length += 1,
                    Some(current) => {
                        run_value_stats.push(current);
                        run_length_stats.push(run_length);
                        run_value = Some(value);
                        run_length = 1;
                    }
                    None => {
                        run_value = Some(value);
                        run_length = 1;
                    }
                }
            }
            run_value_stats.push(run_value.expect("non-empty offset chunk has one run"));
            run_length_stats.push(run_length);
            let values = value_stats.finish();
            Ok(MemberStats {
                deltas: (values.len >= 2).then(|| delta_stats.finish()),
                values,
                run_values: run_value_stats.finish(),
                run_lengths: run_length_stats.finish(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(FamilyStats {
        members,
        dictionary_items: dictionary.and_then(BoundedDistinct::finish),
    })
}

#[derive(Debug)]
struct BoundedDistinct {
    values: Option<BTreeSet<u64>>,
}

impl Default for BoundedDistinct {
    fn default() -> Self {
        Self {
            values: Some(BTreeSet::new()),
        }
    }
}

impl BoundedDistinct {
    fn push(&mut self, value: u64) {
        let Some(values) = self.values.as_mut() else {
            return;
        };
        values.insert(value);
        if values.len() > MAX_BLOCK_DICTIONARY_ITEMS {
            self.values = None;
        }
    }

    fn finish(self) -> Option<Arc<[u64]>> {
        self.values
            .map(|values| Arc::from(values.into_iter().collect::<Vec<_>>()))
    }
}

pub(super) fn combine(stats: &[&SequenceStats]) -> Result<SequenceStats> {
    let bits_per_value = stats
        .first()
        .ok_or_else(|| Error::invalid_input("Cannot combine an empty stats family"))?
        .bits_per_value;
    let mut len = 0_u64;
    let mut run_count = 0_u64;
    let mut min = u64::MAX;
    let mut max = 0_u64;
    let mut sample = Vec::new();
    for member in stats {
        if member.bits_per_value != bits_per_value {
            return Err(Error::invalid_input(
                "Cannot combine sequences with different value widths",
            ));
        }
        len = len
            .checked_add(member.len)
            .ok_or_else(|| Error::invalid_input("Combined sequence length overflows u64"))?;
        run_count = run_count
            .checked_add(member.run_count)
            .ok_or_else(|| Error::invalid_input("Combined run count overflows u64"))?;
        if member.len > 0 {
            min = min.min(member.min);
            max = max.max(member.max);
        }
        let remaining = GENERAL_SAMPLE_BYTES.saturating_sub(sample.len());
        sample.extend_from_slice(&member.sample[..member.sample.len().min(remaining)]);
    }
    Ok(SequenceStats {
        bits_per_value,
        len,
        first: stats.first().and_then(|stats| stats.first),
        min: if len == 0 { 0 } else { min },
        max,
        arithmetic_step: None,
        run_count,
        sample: Arc::from(sample),
    })
}
