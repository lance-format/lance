// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::encodings::physical::sequence::{SequenceDecoder, SequenceMetadata};
use crate::format::pb23;
use arrow_buffer::ArrowNativeType;

fn invalid_enum<T>(
    value: std::result::Result<T, prost::UnknownEnumValue>,
    label: &str,
) -> Result<T> {
    value.map_err(|error| {
        Error::invalid_input_source(
            format!("Sparse structural {label} has an invalid enum value: {error}").into(),
        )
    })
}

pub(super) mod dictionary;
pub(super) mod planar;
pub(super) mod writer;

fn usize_from_u64(value: u64, label: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        Error::invalid_input_source(
            format!("Sparse structural {label} {value} exceeds usize::MAX").into(),
        )
    })
}

/// Native sparse structural representation used by the 2.3 sparse layout.
///
/// Layers are stored from outer-most to inner-most, matching the order Arrow structural
/// encoders record offsets and validity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseStructuralPlan {
    pub(crate) layers: Vec<SparseStructuralLayerPlan>,
    pub(crate) num_visible_items: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparsePositionPlan {
    Empty,
    All { len: u64 },
    Range { start: u64, len: u64 },
    Arithmetic { start: u64, step: u64, len: u64 },
    Explicit(Vec<u64>),
}

impl SparsePositionPlan {
    pub(crate) fn from_positions(
        positions: Vec<u64>,
        domain_len: u64,
        label: &str,
    ) -> Result<Self> {
        if positions.is_empty() {
            return Ok(Self::Empty);
        }
        for window in positions.windows(2) {
            let [previous, current] = window else {
                continue;
            };
            if previous >= current {
                return Err(Error::invalid_input_source(
                    format!("Sparse structural {label} positions must be strictly increasing")
                        .into(),
                ));
            }
        }
        if let Some(position) = positions.iter().find(|position| **position >= domain_len) {
            return Err(Error::invalid_input_source(
                format!(
                    "Sparse structural {label} position {} is outside layer with {} slots",
                    position, domain_len
                )
                .into(),
            ));
        }

        let len = u64::try_from(positions.len()).map_err(|_| {
            Error::invalid_input_source(
                format!("Sparse structural {label} position count exceeds u64::MAX").into(),
            )
        })?;
        let first = positions.first().copied().ok_or_else(|| {
            Error::invalid_input_source(
                format!("Sparse structural {label} positions are unexpectedly empty").into(),
            )
        })?;
        let last = positions.last().copied().ok_or_else(|| {
            Error::invalid_input_source(
                format!("Sparse structural {label} positions are unexpectedly empty").into(),
            )
        })?;
        if first == 0 && len == domain_len && domain_len > 0 && last == domain_len - 1 {
            return Ok(Self::All { len: domain_len });
        }
        if last - first + 1 == len {
            return Ok(Self::Range { start: first, len });
        }
        if len >= 2 {
            let step = positions[1] - first;
            if positions.windows(2).all(|pair| pair[1] - pair[0] == step) {
                return Ok(Self::Arithmetic {
                    start: first,
                    step,
                    len,
                });
            }
        }
        Ok(Self::Explicit(positions))
    }

    pub(crate) fn empty() -> Self {
        Self::Empty
    }

    pub(crate) fn all(len: u64) -> Self {
        if len == 0 {
            Self::Empty
        } else {
            Self::All { len }
        }
    }

    pub(crate) fn range(start: u64, len: u64) -> Self {
        if len == 0 {
            Self::Empty
        } else {
            Self::Range { start, len }
        }
    }

    pub(crate) fn len(&self) -> u64 {
        match self {
            Self::Empty => 0,
            Self::All { len } | Self::Range { len, .. } | Self::Arithmetic { len, .. } => *len,
            Self::Explicit(positions) => positions.len() as u64,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn deep_size(&self) -> usize {
        match self {
            Self::Explicit(positions) => positions.len() * std::mem::size_of::<u64>(),
            Self::Empty | Self::All { .. } | Self::Range { .. } | Self::Arithmetic { .. } => 0,
        }
    }

    pub(crate) fn materialize(&self) -> Result<Vec<u64>> {
        match self {
            Self::Empty => Ok(Vec::new()),
            Self::All { len } => Self::materialize_range(0, *len),
            Self::Range { start, len } => Self::materialize_range(*start, *len),
            Self::Arithmetic { start, step, len } => {
                let mut values = crate::encodings::physical::try_vec_with_capacity::<u64>(
                    *len,
                    "Sparse positions",
                )?;
                values.extend((0..*len).map(|i| start + step * i));
                Ok(values)
            }
            Self::Explicit(positions) => Ok(positions.clone()),
        }
    }

    fn materialize_range(start: u64, len: u64) -> Result<Vec<u64>> {
        let len_usize = usize::try_from(len).map_err(|_| {
            Error::invalid_input_source(
                format!(
                    "Sparse structural position range length {} exceeds usize::MAX",
                    len
                )
                .into(),
            )
        })?;
        let end = start.checked_add(len).ok_or_else(|| {
            Error::invalid_input_source("Sparse structural position range overflows".into())
        })?;
        let mut positions = Vec::with_capacity(len_usize);
        positions.extend(start..end);
        Ok(positions)
    }

    fn contains(&self, position: u64) -> bool {
        match self {
            Self::Empty => false,
            Self::All { len } => position < *len,
            Self::Range { start, len } => {
                position >= *start && position < start.saturating_add(*len)
            }
            Self::Arithmetic { start, step, len } => {
                position >= *start
                    && (position - start).is_multiple_of(*step)
                    && (position - start) / step < *len
            }
            Self::Explicit(positions) => positions.binary_search(&position).is_ok(),
        }
    }

    fn progression(&self) -> Option<(u64, u64, u64)> {
        match self {
            Self::All { len } => Some((0, 1, *len)),
            Self::Range { start, len } => Some((*start, 1, *len)),
            Self::Arithmetic { start, step, len } => Some((*start, *step, *len)),
            _ => None,
        }
    }

    fn rank(&self, position: u64) -> u64 {
        if let Some((start, step, len)) = self.progression() {
            position.saturating_sub(start).div_ceil(step).min(len)
        } else if let Self::Explicit(values) = self {
            values.partition_point(|value| *value < position) as u64
        } else {
            0
        }
    }

    fn is_subset_of(&self, other: &Self, domain_len: u64) -> Result<bool> {
        self.validate_domain(domain_len, "subset")?;
        other.validate_domain(domain_len, "superset")?;
        if self.is_empty() {
            return Ok(true);
        }
        if self.len() > other.len() {
            return Ok(false);
        }
        if let Some((start, 1, len)) = self.progression() {
            return Ok(other.rank(start + len) - other.rank(start) == len);
        }
        if let (Some((start, step, len)), Some((_, other_step, _))) =
            (self.progression(), other.progression())
        {
            return Ok(other.contains(start)
                && other.contains(start + step * (len - 1))
                && (len == 1 || step.is_multiple_of(other_step)));
        }
        Ok(match self {
            Self::Explicit(values) => values.iter().all(|value| other.contains(*value)),
            _ => {
                let (start, step, len) = self.progression().expect("non-empty metadata positions");
                (0..len).all(|i| other.contains(start + step * i))
            }
        })
    }

    fn is_disjoint(&self, other: &Self, domain_len: u64) -> Result<bool> {
        self.validate_domain(domain_len, "first disjoint set")?;
        other.validate_domain(domain_len, "second disjoint set")?;
        if self.is_empty() || other.is_empty() {
            return Ok(true);
        }
        if let Some((start, 1, len)) = self.progression() {
            return Ok(other.rank(start + len) == other.rank(start));
        }
        if let Some((start, 1, len)) = other.progression() {
            return Ok(self.rank(start + len) == self.rank(start));
        }
        if let (Self::Explicit(left), Self::Explicit(right)) = (self, other) {
            let (smaller, larger) = if left.len() <= right.len() {
                (left, right)
            } else {
                (right, left)
            };
            return Ok(smaller
                .iter()
                .all(|value| larger.binary_search(value).is_err()));
        }
        if let Self::Explicit(values) = self {
            return Ok(values.iter().all(|value| !other.contains(*value)));
        }
        if let Self::Explicit(values) = other {
            return Ok(values.iter().all(|value| !self.contains(*value)));
        }
        let (a, s, n) = self.progression().expect("non-empty metadata positions");
        let (b, t, m) = other.progression().expect("non-empty metadata positions");
        Ok(!progressions_intersect(a, s, n, b, t, m))
    }
}

// Solve a + s*k = b (mod t), then test the first common value in both domains.
fn progressions_intersect(a: u64, s: u64, n: u64, b: u64, t: u64, m: u64) -> bool {
    let (mut r, mut next_r) = (i128::from(s), i128::from(t));
    let (mut inverse, mut next_inverse) = (1_i128, 0_i128);
    while next_r != 0 {
        let q = r / next_r;
        (r, next_r) = (next_r, r - q * next_r);
        (inverse, next_inverse) = (next_inverse, inverse - q * next_inverse);
    }
    let difference = i128::from(b) - i128::from(a);
    if difference % r != 0 {
        return false;
    }
    let modulus = i128::from(t) / r;
    let k = ((difference / r).rem_euclid(modulus) as u128 * inverse.rem_euclid(modulus) as u128)
        % modulus as u128;
    let mut first = u128::from(a) + u128::from(s) * k;
    let lower = u128::from(a.max(b));
    let period = u128::from(s) * (u128::from(t) / r as u128);
    if first < lower {
        first += (lower - first).div_ceil(period) * period;
    }
    first <= u128::from((a + s * (n - 1)).min(b + t * (m - 1)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseValidityMeaning {
    NullPositions,
    ValidPositions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseValiditySet {
    pub(crate) meaning: SparseValidityMeaning,
    pub(crate) positions: SparsePositionPlan,
}

impl SparseValiditySet {
    pub(crate) fn deep_size(&self) -> usize {
        self.positions.deep_size()
    }

    fn contains_only_valid_positions(
        &self,
        positions: &SparsePositionPlan,
        num_slots: u64,
    ) -> Result<bool> {
        match self.meaning {
            SparseValidityMeaning::NullPositions => {
                positions.is_disjoint(&self.positions, num_slots)
            }
            SparseValidityMeaning::ValidPositions => {
                positions.is_subset_of(&self.positions, num_slots)
            }
        }
    }

    fn append_to(&self, validity: &mut BooleanBufferBuilder, num_slots: u64) -> Result<()> {
        self.positions.validate_domain(num_slots, "validity")?;
        let num_slots_usize = usize_from_u64(num_slots, "validity slot count")?;
        match (self.meaning, &self.positions) {
            (SparseValidityMeaning::NullPositions, SparsePositionPlan::Empty) => {
                validity.append_n(num_slots_usize, true);
            }
            (SparseValidityMeaning::ValidPositions, SparsePositionPlan::Empty) => {
                validity.append_n(num_slots_usize, false);
            }
            (SparseValidityMeaning::NullPositions, SparsePositionPlan::All { .. }) => {
                validity.append_n(num_slots_usize, false);
            }
            (SparseValidityMeaning::ValidPositions, SparsePositionPlan::All { .. }) => {
                validity.append_n(num_slots_usize, true);
            }
            (meaning, SparsePositionPlan::Range { start, len }) => {
                let range_end = start.checked_add(*len).ok_or_else(|| {
                    Error::invalid_input_source("Sparse structural validity range overflows".into())
                })?;
                let default_valid = matches!(meaning, SparseValidityMeaning::NullPositions);
                let range_valid = !default_valid;
                validity.append_n(
                    usize_from_u64(*start, "validity range start")?,
                    default_valid,
                );
                validity.append_n(usize_from_u64(*len, "validity range length")?, range_valid);
                validity.append_n(
                    usize_from_u64(num_slots - range_end, "validity range tail")?,
                    default_valid,
                );
            }
            (_, SparsePositionPlan::Explicit(_) | SparsePositionPlan::Arithmetic { .. }) => {
                let mut cursor = SparseValidityCursor::new(self, num_slots, "validity")?;
                for slot in 0..num_slots {
                    validity.append(cursor.is_valid(slot)?);
                }
                cursor.finish()?;
            }
        }
        Ok(())
    }
}

impl SparsePositionPlan {
    fn validate_domain(&self, domain_len: u64, label: &str) -> Result<()> {
        match self {
            Self::Empty => {}
            Self::All { len } => {
                if *len != domain_len {
                    return Err(Error::invalid_input_source(
                        format!(
                            "Sparse structural {label} all set length {} does not match domain {}",
                            len, domain_len
                        )
                        .into(),
                    ));
                }
            }
            Self::Range { start, len } => {
                let end = start.checked_add(*len).ok_or_else(|| {
                    Error::invalid_input_source(
                        format!("Sparse structural {label} range overflows").into(),
                    )
                })?;
                if end > domain_len {
                    return Err(Error::invalid_input_source(
                        format!(
                            "Sparse structural {label} range {}..{} is outside domain {}",
                            start, end, domain_len
                        )
                        .into(),
                    ));
                }
            }
            Self::Arithmetic { start, step, len } => {
                let last =
                    crate::encodings::physical::range::checked_range_last(64, *start, *step, *len)?;
                if last >= domain_len {
                    return Err(Error::invalid_input(format!(
                        "Sparse structural {label} position {last} is outside layer with {domain_len} slots"
                    )));
                }
            }
            Self::Explicit(positions) => {
                for window in positions.windows(2) {
                    let [previous, current] = window else {
                        continue;
                    };
                    if previous >= current {
                        return Err(Error::invalid_input_source(
                            format!(
                                "Sparse structural {label} positions must be strictly increasing"
                            )
                            .into(),
                        ));
                    }
                }
                if let Some(position) = positions.last()
                    && *position >= domain_len
                {
                    return Err(Error::invalid_input_source(
                        format!(
                            "Sparse structural {label} position {} is outside layer with {} slots",
                            position, domain_len
                        )
                        .into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

struct SparsePositionCursor<'a> {
    set: &'a SparsePositionPlan,
    explicit: Option<std::iter::Peekable<std::slice::Iter<'a, u64>>>,
}

impl<'a> SparsePositionCursor<'a> {
    fn new(set: &'a SparsePositionPlan, domain_len: u64, label: &str) -> Result<Self> {
        set.validate_domain(domain_len, label)?;
        let explicit = match set {
            SparsePositionPlan::Explicit(positions) => Some(positions.iter().peekable()),
            _ => None,
        };
        Ok(Self { set, explicit })
    }

    fn contains(&mut self, slot: u64) -> Result<bool> {
        Ok(match self.set {
            SparsePositionPlan::Empty => false,
            SparsePositionPlan::All { .. } => true,
            SparsePositionPlan::Range { start, len } => {
                slot >= *start && slot < start.saturating_add(*len)
            }
            SparsePositionPlan::Arithmetic { .. } => self.set.contains(slot),
            SparsePositionPlan::Explicit(_) => {
                let iter = self.explicit.as_mut().ok_or_else(|| {
                    Error::internal("Sparse structural explicit cursor is missing".to_string())
                })?;
                if let Some(position) = iter.peek()
                    && **position < slot
                {
                    return Err(Error::invalid_input_source(
                        format!(
                            "Sparse structural explicit position {} was skipped before slot {}",
                            **position, slot
                        )
                        .into(),
                    ));
                }
                if iter.peek().is_some_and(|position| **position == slot) {
                    iter.next();
                    true
                } else {
                    false
                }
            }
        })
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(iter) = self.explicit.as_mut()
            && let Some(position) = iter.next()
        {
            return Err(Error::invalid_input_source(
                format!(
                    "Sparse structural explicit position {} was not consumed",
                    position
                )
                .into(),
            ));
        }
        Ok(())
    }
}

struct SparseValidityCursor<'a> {
    meaning: SparseValidityMeaning,
    positions: SparsePositionCursor<'a>,
}

impl<'a> SparseValidityCursor<'a> {
    fn new(validity: &'a SparseValiditySet, domain_len: u64, label: &str) -> Result<Self> {
        Ok(Self {
            meaning: validity.meaning,
            positions: SparsePositionCursor::new(&validity.positions, domain_len, label)?,
        })
    }

    fn is_valid(&mut self, slot: u64) -> Result<bool> {
        let is_stored = self.positions.contains(slot)?;
        Ok(match self.meaning {
            SparseValidityMeaning::NullPositions => !is_stored,
            SparseValidityMeaning::ValidPositions => is_stored,
        })
    }

    fn finish(&mut self) -> Result<()> {
        self.positions.finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseCountPlan {
    Empty,
    Constant {
        value: u64,
        len: u64,
    },
    Explicit {
        counts: Arc<[u64]>,
        offsets: Arc<[u64]>,
    },
}

impl SparseCountPlan {
    pub(crate) fn from_counts(counts: Vec<u64>) -> Result<Self> {
        if counts.is_empty() {
            return Ok(Self::Empty);
        }
        if let Some(first) = counts.first().copied()
            && counts.iter().all(|count| *count == first)
        {
            return Ok(Self::Constant {
                value: first,
                len: counts.len() as u64,
            });
        }
        let offsets = offsets_from_counts(&counts)?;
        Ok(Self::Explicit {
            counts: counts.into(),
            offsets: offsets.into(),
        })
    }

    pub(crate) fn constant(value: u64, len: u64) -> Self {
        if len == 0 {
            Self::Empty
        } else {
            Self::Constant { value, len }
        }
    }

    pub(crate) fn len(&self) -> u64 {
        match self {
            Self::Empty => 0,
            Self::Constant { len, .. } => *len,
            Self::Explicit { counts, .. } => counts.len() as u64,
        }
    }

    pub(crate) fn deep_size(&self) -> usize {
        match self {
            Self::Explicit { counts, offsets } => {
                counts.len() * std::mem::size_of::<u64>()
                    + offsets.len() * std::mem::size_of::<u64>()
            }
            Self::Empty | Self::Constant { .. } => 0,
        }
    }

    pub(crate) fn materialize(&self) -> Result<Vec<u64>> {
        match self {
            Self::Empty => Ok(Vec::new()),
            Self::Constant { value, len } => {
                let len = usize::try_from(*len).map_err(|_| {
                    Error::invalid_input_source(
                        "Sparse structural count set length exceeds usize::MAX".into(),
                    )
                })?;
                Ok(vec![*value; len])
            }
            Self::Explicit { counts, .. } => Ok(counts.to_vec()),
        }
    }

    pub(crate) fn sum(&self) -> Result<u64> {
        match self {
            Self::Empty => Ok(0),
            Self::Constant { value, len } => value.checked_mul(*len).ok_or_else(|| {
                Error::invalid_input_source(
                    format!(
                        "Sparse structural constant count sum overflows: value={}, len={}",
                        value, len
                    )
                    .into(),
                )
            }),
            Self::Explicit { offsets, .. } => offsets.last().copied().ok_or_else(|| {
                Error::invalid_input_source(
                    "Sparse structural explicit count offsets are empty".into(),
                )
            }),
        }
    }

    fn validate_positive(&self) -> Result<()> {
        let has_zero = match self {
            Self::Empty => false,
            Self::Constant { value, .. } => *value == 0,
            Self::Explicit { counts, .. } => counts.contains(&0),
        };
        if has_zero {
            return Err(Error::invalid_input_source(
                "Sparse structural non-empty list count is zero".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseStructuralLayerPlan {
    Validity {
        num_slots: u64,
        validity: SparseValiditySet,
    },
    List {
        num_slots: u64,
        num_child_slots: u64,
        non_empty_positions: SparsePositionPlan,
        counts: SparseCountPlan,
        validity: SparseValiditySet,
    },
    FixedSizeList {
        num_slots: u64,
        dimension: u64,
        validity: SparseValiditySet,
    },
}

impl SparseStructuralPlan {
    fn validate(&self, row_domain: u64) -> Result<()> {
        usize_from_u64(self.num_visible_items, "visible item count")?;
        let mut expected_slots = row_domain;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let (num_slots, num_child_slots, validity) = match layer {
                SparseStructuralLayerPlan::Validity {
                    num_slots,
                    validity,
                } => (*num_slots, *num_slots, validity),
                SparseStructuralLayerPlan::List {
                    num_slots,
                    num_child_slots,
                    non_empty_positions,
                    counts,
                    validity,
                } => {
                    counts.validate_positive()?;
                    if non_empty_positions.len() != counts.len() {
                        return Err(Error::invalid_input_source(
                            format!(
                                "Sparse structural list layer {} has {} non-empty positions but {} counts",
                                layer_index,
                                non_empty_positions.len(),
                                counts.len()
                            )
                            .into(),
                        ));
                    }
                    if counts.sum()? != *num_child_slots {
                        return Err(Error::invalid_input_source(
                            format!(
                                "Sparse structural list layer {} count sum does not match {} child slots",
                                layer_index, num_child_slots
                            )
                            .into(),
                        ));
                    }
                    if !validity.contains_only_valid_positions(non_empty_positions, *num_slots)? {
                        return Err(Error::invalid_input_source(
                            format!(
                                "Sparse structural list layer {} contains a non-empty null slot",
                                layer_index
                            )
                            .into(),
                        ));
                    }
                    (*num_slots, *num_child_slots, validity)
                }
                SparseStructuralLayerPlan::FixedSizeList {
                    num_slots,
                    dimension,
                    validity,
                } => {
                    if *dimension == 0 {
                        return Err(Error::invalid_input_source(
                            format!(
                                "Sparse structural fixed-size-list layer {} has dimension zero",
                                layer_index
                            )
                            .into(),
                        ));
                    }
                    let num_child_slots = num_slots.checked_mul(*dimension).ok_or_else(|| {
                        Error::invalid_input_source(
                            "Sparse structural fixed-size-list child domain overflows".into(),
                        )
                    })?;
                    (*num_slots, num_child_slots, validity)
                }
            };
            usize_from_u64(num_slots, "layer slot count")?;
            usize_from_u64(num_child_slots, "layer child slot count")?;
            if num_slots != expected_slots {
                return Err(Error::invalid_input_source(
                    format!(
                        "Sparse structural layer {} has {} slots, expected {}",
                        layer_index, num_slots, expected_slots
                    )
                    .into(),
                ));
            }
            validity.positions.validate_domain(num_slots, "validity")?;
            expected_slots = num_child_slots;
        }
        if expected_slots != self.num_visible_items {
            return Err(Error::invalid_input_source(
                format!(
                    "Sparse structural terminal domain has {} slots, expected {} visible items",
                    expected_slots, self.num_visible_items
                )
                .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct SparseStructuralUnraveler {
    layers: Vec<SparseStructuralLayerPlan>,
    next_layer: usize,
    pending_fixed_size_list: bool,
}

impl SparseStructuralUnraveler {
    pub(crate) fn new(plan: SparseStructuralPlan) -> Self {
        let next_layer = plan.layers.len();
        Self {
            layers: plan.layers,
            next_layer,
            pending_fixed_size_list: false,
        }
    }

    fn current_layer(&self) -> Option<&SparseStructuralLayerPlan> {
        self.next_layer
            .checked_sub(1)
            .and_then(|idx| self.layers.get(idx))
    }

    fn consume_current_layer(&mut self) -> Result<()> {
        self.next_layer = self.next_layer.checked_sub(1).ok_or_else(|| {
            Error::invalid_input_source(
                "Sparse structural metadata has fewer layers than the Arrow schema".into(),
            )
        })?;
        Ok(())
    }

    pub(crate) fn ensure_exhausted(&self) -> Result<()> {
        if self.pending_fixed_size_list {
            return Err(Error::invalid_input_source(
                "Sparse structural metadata has an unconsumed fixed-size-list layer".into(),
            ));
        }
        if self.next_layer != 0 {
            return Err(Error::invalid_input_source(
                format!(
                    "Sparse structural metadata has {} unconsumed layer(s)",
                    self.next_layer
                )
                .into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn is_all_valid(&self) -> bool {
        match self.current_layer() {
            Some(SparseStructuralLayerPlan::Validity {
                num_slots,
                validity,
            })
            | Some(SparseStructuralLayerPlan::FixedSizeList {
                num_slots,
                validity,
                ..
            })
            | Some(SparseStructuralLayerPlan::List {
                num_slots,
                validity,
                ..
            }) => match validity.meaning {
                SparseValidityMeaning::NullPositions => validity.positions.is_empty(),
                SparseValidityMeaning::ValidPositions => validity.positions.len() == *num_slots,
            },
            None => true,
        }
    }

    pub(crate) fn max_lists(&self) -> Result<usize> {
        match self.current_layer() {
            Some(SparseStructuralLayerPlan::List { num_slots, .. }) => {
                usize_from_u64(*num_slots, "list slot count")
            }
            _ => Ok(0),
        }
    }

    pub(crate) fn skip_validity(&mut self) -> Result<()> {
        match self.current_layer() {
            Some(SparseStructuralLayerPlan::Validity { .. }) => {
                if self.pending_fixed_size_list {
                    return Err(Error::invalid_input_source(
                        "Sparse fixed-size-list schema does not match a validity layer".into(),
                    ));
                }
                self.consume_current_layer()?;
            }
            Some(SparseStructuralLayerPlan::FixedSizeList { .. }) => {
                if !self.pending_fixed_size_list {
                    return Err(Error::invalid_input_source(
                        "Sparse fixed-size-list layer does not match the Arrow schema".into(),
                    ));
                }
                self.pending_fixed_size_list = false;
                self.consume_current_layer()?;
            }
            None => {
                return Err(Error::invalid_input_source(
                    "Sparse structural metadata has fewer layers than the Arrow schema".into(),
                ));
            }
            Some(SparseStructuralLayerPlan::List { .. }) => {
                return Err(Error::invalid_input_source(
                    "Sparse structural list layer does not match an Arrow validity layer".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn unravel_validity(&mut self, validity: &mut BooleanBufferBuilder) -> Result<()> {
        match self.current_layer() {
            Some(SparseStructuralLayerPlan::Validity {
                num_slots,
                validity: layer_validity,
            }) => {
                if self.pending_fixed_size_list {
                    return Err(Error::invalid_input_source(
                        "Sparse fixed-size-list schema does not match a validity layer".into(),
                    ));
                }
                layer_validity.append_to(validity, *num_slots)?;
                self.consume_current_layer()?;
            }
            Some(SparseStructuralLayerPlan::FixedSizeList {
                num_slots,
                validity: layer_validity,
                ..
            }) => {
                if !self.pending_fixed_size_list {
                    return Err(Error::invalid_input_source(
                        "Sparse fixed-size-list layer does not match the Arrow schema".into(),
                    ));
                }
                layer_validity.append_to(validity, *num_slots)?;
                self.pending_fixed_size_list = false;
                self.consume_current_layer()?;
            }
            None => {
                return Err(Error::invalid_input_source(
                    "Sparse structural metadata has fewer layers than the Arrow schema".into(),
                ));
            }
            Some(SparseStructuralLayerPlan::List { .. }) => {
                return Err(Error::invalid_input_source(
                    "Sparse structural list layer does not match an Arrow validity layer".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn decimate(&mut self, dimension: usize) -> Result<()> {
        if self.pending_fixed_size_list {
            return Err(Error::invalid_input_source(
                "Sparse fixed-size-list layer was decimated more than once".into(),
            ));
        }
        let Some(SparseStructuralLayerPlan::FixedSizeList {
            dimension: actual_dimension,
            ..
        }) = self.current_layer()
        else {
            return Err(Error::invalid_input_source(
                "Sparse structural layer does not match an Arrow fixed-size-list layer".into(),
            ));
        };
        if usize_from_u64(*actual_dimension, "fixed-size-list dimension")? != dimension {
            return Err(Error::invalid_input_source(
                format!(
                    "Sparse structural fixed-size-list dimension {} does not match Arrow dimension {}",
                    actual_dimension, dimension
                )
                .into(),
            ));
        }
        self.pending_fixed_size_list = true;
        Ok(())
    }

    fn to_offset<T: ArrowNativeType>(value: u64) -> Result<T> {
        let value = usize::try_from(value).map_err(|_| {
            Error::invalid_input_source(
                format!("Sparse structural offset {} exceeds usize::MAX", value).into(),
            )
        })?;
        T::from_usize(value).ok_or_else(|| {
            Error::invalid_input_source(
                "Sparse structural offset does not fit the Arrow offset type".into(),
            )
        })
    }

    pub(crate) fn unravel_offsets<T: ArrowNativeType>(
        &mut self,
        offsets: &mut Vec<T>,
        validity: Option<&mut BooleanBufferBuilder>,
    ) -> Result<()> {
        let Some(SparseStructuralLayerPlan::List {
            num_slots,
            num_child_slots,
            non_empty_positions,
            counts,
            validity: layer_validity,
            ..
        }) = self.current_layer()
        else {
            return Err(Error::invalid_input_source(
                "Sparse structural layer does not match an Arrow list layer".into(),
            ));
        };
        if self.pending_fixed_size_list {
            return Err(Error::invalid_input_source(
                "Sparse fixed-size-list schema does not match an Arrow list layer".into(),
            ));
        }

        if non_empty_positions.len() != counts.len() {
            return Err(Error::invalid_input_source(
                format!(
                    "Sparse structural list has {} non-empty positions but {} counts",
                    non_empty_positions.len(),
                    counts.len()
                )
                .into(),
            ));
        }
        let actual_child_slots = counts.sum()?;
        if actual_child_slots != *num_child_slots {
            return Err(Error::invalid_input_source(
                format!(
                    "Sparse structural list count sum {} does not match child slots {}",
                    actual_child_slots, num_child_slots
                )
                .into(),
            ));
        }

        let mut current_offset = offsets
            .last()
            .map(|offset| offset.as_usize() as u64)
            .unwrap_or(0);
        if offsets.is_empty() {
            offsets.push(Self::to_offset(current_offset)?);
        }

        if non_empty_positions.is_empty() {
            if let Some(validity) = validity {
                layer_validity.append_to(validity, *num_slots)?;
            }
            let offset = Self::to_offset(current_offset)?;
            let new_len = offsets
                .len()
                .checked_add(usize_from_u64(*num_slots, "list slot count")?)
                .ok_or_else(|| {
                    Error::invalid_input_source(
                        "Sparse structural list offset length overflows usize".into(),
                    )
                })?;
            offsets.resize(new_len, offset);
            self.consume_current_layer()?;
            return Ok(());
        }

        let non_empty_positions = non_empty_positions.materialize()?;
        let counts = counts.materialize()?;
        let mut non_empty_iter = non_empty_positions
            .iter()
            .copied()
            .zip(counts.iter().copied())
            .peekable();
        let mut validity_cursor =
            SparseValidityCursor::new(layer_validity, *num_slots, "list validity")?;
        let mut validity = validity;
        for slot in 0..*num_slots {
            let is_valid = validity_cursor.is_valid(slot)?;
            if let Some(validity) = validity.as_mut() {
                validity.append(is_valid);
            }

            if non_empty_iter
                .peek()
                .is_some_and(|(non_empty_pos, _)| *non_empty_pos == slot)
            {
                let (_, count) = non_empty_iter.next().ok_or_else(|| {
                    Error::invalid_input_source(
                        "Sparse structural list position is missing its child count".into(),
                    )
                })?;
                if !is_valid {
                    return Err(Error::invalid_input_source(
                        format!(
                            "Sparse structural list slot {} is both invalid and non-empty",
                            slot
                        )
                        .into(),
                    ));
                }
                current_offset = current_offset.checked_add(count).ok_or_else(|| {
                    Error::invalid_input_source(
                        "Sparse structural list offsets overflow u64".into(),
                    )
                })?;
            }
            offsets.push(Self::to_offset(current_offset)?);
        }
        validity_cursor.finish()?;
        if let Some((extra_pos, _)) = non_empty_iter.next() {
            return Err(Error::invalid_input_source(
                format!(
                    "Sparse structural non-empty position {} is outside layer with {} slots",
                    extra_pos, num_slots
                )
                .into(),
            ));
        }

        self.consume_current_layer()?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
enum SparsePositionDecoder {
    Ready(SparsePositionPlan),
    Encoded {
        sequence: SequenceDecoder,
        domain_len: u64,
    },
}

#[derive(Debug, Clone)]
enum SparseCountDecoder {
    Ready(SparseCountPlan),
    Encoded(SequenceDecoder),
}

#[derive(Debug, Clone)]
enum SparseLayerDecompressors {
    Validity {
        num_slots: u64,
        validity: SparseValiditySetDecoder,
    },
    List {
        num_slots: u64,
        num_child_slots: u64,
        non_empty_positions: SparsePositionDecoder,
        counts: SparseCountDecoder,
        validity: SparseValiditySetDecoder,
    },
    FixedSizeList {
        num_slots: u64,
        num_child_slots: u64,
        dimension: u64,
        validity: SparseValiditySetDecoder,
    },
}

impl SparseLayerDecompressors {
    fn num_buffers(&self) -> usize {
        let position_buffers = |positions: &SparsePositionDecoder| match positions {
            SparsePositionDecoder::Ready(_) => 0,
            SparsePositionDecoder::Encoded { sequence, .. } => {
                usize::from(sequence.requires_payload())
            }
        };
        match self {
            Self::Validity { validity, .. } | Self::FixedSizeList { validity, .. } => {
                position_buffers(&validity.positions)
            }
            Self::List {
                non_empty_positions,
                counts,
                validity,
                ..
            } => {
                let count_buffers = match counts {
                    SparseCountDecoder::Ready(_) => 0,
                    SparseCountDecoder::Encoded(sequence) => {
                        usize::from(sequence.requires_payload())
                    }
                };
                position_buffers(non_empty_positions)
                    + count_buffers
                    + position_buffers(&validity.positions)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct SparseValiditySetDecoder {
    meaning: SparseValidityMeaning,
    positions: SparsePositionDecoder,
}

impl pb23::SparseStructure {
    /// Derive the logical page row count and fixed-size-list scheduling scale.
    ///
    /// Page metadata counts rows multiplied by every structural fixed-size-list
    /// dimension, including dimensions below variable-size lists.
    ///
    /// ```
    /// let layout = lance_encoding::format::pb23::SparseStructure::default();
    /// assert_eq!(layout.row_count_and_scale(5)?, (5, 1));
    /// # Ok::<(), lance_core::Error>(())
    /// ```
    pub fn row_count_and_scale(&self, encoded_rows: u64) -> Result<(u64, u64)> {
        let scale =
            self.layers
                .iter()
                .try_fold(1_u64, |scale, layer| match layer.layer.as_ref() {
                    Some(pb23::sparse_structural_layer::Layer::FixedSizeList(layer)) => {
                        if layer.dimension == 0 {
                            return Err(Error::invalid_input(
                                "Sparse structural fixed-size-list dimension is zero",
                            ));
                        }
                        scale.checked_mul(layer.dimension).ok_or_else(|| {
                            Error::invalid_input(
                                "Sparse structural fixed-size-list row scale overflows",
                            )
                        })
                    }
                    Some(_) => Ok(scale),
                    None => Err(Error::invalid_input(
                        "Sparse structural layer is missing its layer variant",
                    )),
                })?;
        if !encoded_rows.is_multiple_of(scale) {
            return Err(Error::invalid_input(format!(
                "Sparse encoded row domain {encoded_rows} is not divisible by fixed-size-list scale {scale}"
            )));
        }
        Ok((encoded_rows / scale, scale))
    }
}

fn require_layer(
    layer: &pb23::SparseStructuralLayer,
) -> Result<&pb23::sparse_structural_layer::Layer> {
    layer.layer.as_ref().ok_or_else(|| {
        Error::invalid_input_source("Sparse structural layer is missing its layer variant".into())
    })
}

fn child_domain(layer: &pb23::SparseStructuralLayer, num_slots: u64) -> Result<u64> {
    let child_slots = match require_layer(layer)? {
        pb23::sparse_structural_layer::Layer::Validity(_) => num_slots,
        pb23::sparse_structural_layer::Layer::List(layer) => layer.num_child_slots,
        pb23::sparse_structural_layer::Layer::FixedSizeList(layer) => {
            num_slots.checked_mul(layer.dimension).ok_or_else(|| {
                Error::invalid_input("Sparse structural fixed-size-list child slot count overflows")
            })?
        }
    };
    usize_from_u64(child_slots, "layer child slot count")?;
    Ok(child_slots)
}

fn checked_buffer_range(position: u64, size: u64, label: &str) -> Result<Range<u64>> {
    let end = position.checked_add(size).ok_or_else(|| {
        Error::invalid_input_source(
            format!("Sparse structural {label} buffer range overflows").into(),
        )
    })?;
    Ok(position..end)
}

fn require_sequence_encoding<'a>(
    encoding: &'a Option<CompressiveEncoding>,
    label: &str,
) -> Result<&'a CompressiveEncoding> {
    encoding.as_ref().ok_or_else(|| {
        Error::invalid_input(format!("Sparse structural {label} encoding is required"))
    })
}

fn require_validity_set<'a>(
    set: &'a Option<pb23::SparseValiditySet>,
    label: &str,
) -> Result<&'a pb23::SparseValiditySet> {
    set.as_ref().ok_or_else(|| {
        Error::invalid_input_source(
            format!("Sparse structural {label} validity set is required").into(),
        )
    })
}

fn validity_meaning(
    validity_set: &pb23::SparseValiditySet,
    label: &str,
) -> Result<SparseValidityMeaning> {
    match invalid_enum(
        pb23::sparse_validity_set::Meaning::try_from(validity_set.meaning),
        "validity meaning",
    )? {
        pb23::sparse_validity_set::Meaning::SparseValidityUnspecified => {
            Err(Error::invalid_input_source(
                format!("Sparse structural {label} meaning is unspecified").into(),
            ))
        }
        pb23::sparse_validity_set::Meaning::SparseValidityNullPositions => {
            Ok(SparseValidityMeaning::NullPositions)
        }
        pb23::sparse_validity_set::Meaning::SparseValidityValidPositions => {
            Ok(SparseValidityMeaning::ValidPositions)
        }
    }
}

fn position_decoder(
    encoding: &CompressiveEncoding,
    cardinality: u64,
    domain_len: u64,
    label: &str,
) -> Result<SparsePositionDecoder> {
    if cardinality > domain_len {
        return Err(Error::invalid_input(format!(
            "Sparse structural {label} cardinality {cardinality} exceeds domain {domain_len}"
        )));
    }
    let sequence = SequenceDecoder::try_new(encoding, 64, cardinality)?;
    let positions = match sequence.metadata() {
        Some(SequenceMetadata::Empty) => SparsePositionPlan::Empty,
        Some(SequenceMetadata::Constant(value)) => {
            if cardinality != 1 {
                return Err(Error::invalid_input(format!(
                    "Sparse structural {label} positions must be strictly increasing"
                )));
            }
            SparsePositionPlan::range(value, 1)
        }
        Some(SequenceMetadata::Range { start, step: 1 }) => {
            if start == 0 && cardinality == domain_len {
                SparsePositionPlan::all(cardinality)
            } else {
                SparsePositionPlan::range(start, cardinality)
            }
        }
        Some(SequenceMetadata::Range { start, step }) => SparsePositionPlan::Arithmetic {
            start,
            step,
            len: cardinality,
        },
        None => {
            return Ok(SparsePositionDecoder::Encoded {
                sequence,
                domain_len,
            });
        }
    };
    positions.validate_domain(domain_len, label)?;
    Ok(SparsePositionDecoder::Ready(positions))
}

fn validity_set_decoder(
    validity: &pb23::SparseValiditySet,
    domain_len: u64,
    label: &str,
) -> Result<(SparseValiditySetDecoder, u64)> {
    let meaning = validity_meaning(validity, label)?;
    let encoding = require_sequence_encoding(&validity.positions, label)?;
    let positions = position_decoder(encoding, validity.num_positions, domain_len, label)?;
    Ok((
        SparseValiditySetDecoder { meaning, positions },
        validity.num_positions,
    ))
}

fn num_valid_slots(
    meaning: SparseValidityMeaning,
    cardinality: u64,
    num_slots: u64,
    label: &str,
) -> Result<u64> {
    match meaning {
        SparseValidityMeaning::NullPositions => {
            num_slots.checked_sub(cardinality).ok_or_else(|| {
                Error::invalid_input_source(
                    format!(
                        "Sparse structural {label} null cardinality {} exceeds slots {}",
                        cardinality, num_slots
                    )
                    .into(),
                )
            })
        }
        SparseValidityMeaning::ValidPositions => Ok(cardinality),
    }
}

fn count_decoder(
    encoding: &CompressiveEncoding,
    cardinality: u64,
    label: &str,
) -> Result<SparseCountDecoder> {
    let sequence = SequenceDecoder::try_new(encoding, 64, cardinality)?;
    Ok(match sequence.metadata() {
        Some(SequenceMetadata::Empty) => SparseCountDecoder::Ready(SparseCountPlan::Empty),
        Some(SequenceMetadata::Constant(value)) => {
            if value == 0 {
                return Err(Error::invalid_input(format!(
                    "Sparse structural {label} constant count is zero"
                )));
            }
            value.checked_mul(cardinality).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Sparse structural {label} constant count sum overflows"
                ))
            })?;
            SparseCountDecoder::Ready(SparseCountPlan::constant(value, cardinality))
        }
        _ => SparseCountDecoder::Encoded(sequence),
    })
}

fn layer_decompressors(
    layer: &pb23::SparseStructuralLayer,
    num_slots: u64,
    num_child_slots: u64,
) -> Result<SparseLayerDecompressors> {
    Ok(match require_layer(layer)? {
        pb23::sparse_structural_layer::Layer::Validity(layer) => {
            let (validity, _) = validity_set_decoder(layer, num_slots, "validity positions")?;
            SparseLayerDecompressors::Validity {
                num_slots,
                validity,
            }
        }
        pb23::sparse_structural_layer::Layer::List(layer) => {
            let non_empty_positions =
                require_sequence_encoding(&layer.non_empty_positions, "list non-empty")?;
            let num_non_empty = layer.num_non_empty_positions;
            let non_empty_positions = position_decoder(
                non_empty_positions,
                layer.num_non_empty_positions,
                num_slots,
                "list non-empty positions",
            )?;
            let counts = count_decoder(
                require_sequence_encoding(&layer.counts, "list counts")?,
                num_non_empty,
                "list counts",
            )?;
            let (validity, validity_cardinality) = validity_set_decoder(
                require_validity_set(&layer.validity, "list")?,
                num_slots,
                "list validity positions",
            )?;
            let num_valid_slots = num_valid_slots(
                validity.meaning,
                validity_cardinality,
                num_slots,
                "list validity",
            )?;
            if num_non_empty > num_valid_slots {
                return Err(Error::invalid_input(format!(
                    "Sparse structural list has {num_non_empty} non-empty slots but only {num_valid_slots} valid slots"
                )));
            }
            if let SparseCountDecoder::Ready(counts) = &counts {
                let child_slots = counts.sum()?;
                if child_slots != layer.num_child_slots {
                    return Err(Error::invalid_input(format!(
                        "Sparse structural list count sum {child_slots} does not match child slots {}",
                        layer.num_child_slots
                    )));
                }
            }
            SparseLayerDecompressors::List {
                num_slots,
                num_child_slots,
                non_empty_positions,
                counts,
                validity,
            }
        }
        pb23::sparse_structural_layer::Layer::FixedSizeList(layer) => {
            let (validity, _) = validity_set_decoder(
                require_validity_set(&layer.validity, "fixed-size-list")?,
                num_slots,
                "fixed-size-list validity positions",
            )?;
            SparseLayerDecompressors::FixedSizeList {
                num_slots,
                num_child_slots,
                dimension: layer.dimension,
                validity,
            }
        }
    })
}

fn decode_u64_values(
    sequence: &SequenceDecoder,
    buffers: &mut impl Iterator<Item = Bytes>,
    label: &str,
) -> Result<Vec<u64>> {
    let payload = if sequence.requires_payload() {
        Some(LanceBuffer::from_bytes(
            next_structural_buffer(buffers, label)?,
            1,
        ))
    } else {
        None
    };
    let decoded = sequence.decode(payload)?;
    let fixed = decoded.as_fixed_width().ok_or_else(|| {
        Error::invalid_input(format!(
            "Sparse structural {label} did not decode to fixed width data"
        ))
    })?;
    Ok(crate::encodings::physical::checked_fixed_values::<u64>(&fixed, label)?.to_vec())
}

fn decode_position_set(
    decoder: &SparsePositionDecoder,
    buffers: &mut impl Iterator<Item = Bytes>,
    label: &str,
) -> Result<SparsePositionPlan> {
    match decoder {
        SparsePositionDecoder::Ready(positions) => Ok(positions.clone()),
        SparsePositionDecoder::Encoded {
            sequence,
            domain_len,
        } => {
            let values = decode_u64_values(sequence, buffers, label)?;
            SparsePositionPlan::from_positions(values, *domain_len, label)
        }
    }
}

fn decode_validity_set(
    decoder: &SparseValiditySetDecoder,
    buffers: &mut impl Iterator<Item = Bytes>,
    label: &str,
) -> Result<SparseValiditySet> {
    Ok(SparseValiditySet {
        meaning: decoder.meaning,
        positions: decode_position_set(&decoder.positions, buffers, label)?,
    })
}

fn decode_count_set(
    decoder: &SparseCountDecoder,
    buffers: &mut impl Iterator<Item = Bytes>,
    label: &str,
) -> Result<SparseCountPlan> {
    match decoder {
        SparseCountDecoder::Ready(counts) => Ok(counts.clone()),
        SparseCountDecoder::Encoded(sequence) => {
            SparseCountPlan::from_counts(decode_u64_values(sequence, buffers, label)?)
        }
    }
}

fn next_structural_buffer(buffers: &mut impl Iterator<Item = Bytes>, label: &str) -> Result<Bytes> {
    buffers.next().ok_or_else(|| {
        Error::invalid_input_source(
            format!("Sparse structural {label} is missing its buffer").into(),
        )
    })
}

fn decode_layer(
    layer: &SparseLayerDecompressors,
    buffers: &mut impl Iterator<Item = Bytes>,
) -> Result<SparseStructuralLayerPlan> {
    Ok(match layer {
        SparseLayerDecompressors::Validity {
            num_slots,
            validity,
        } => SparseStructuralLayerPlan::Validity {
            num_slots: *num_slots,
            validity: decode_validity_set(validity, buffers, "validity positions")?,
        },
        SparseLayerDecompressors::List {
            num_slots,
            num_child_slots,
            non_empty_positions,
            counts,
            validity,
        } => {
            let non_empty_positions =
                decode_position_set(non_empty_positions, buffers, "list non-empty positions")?;
            let counts = decode_count_set(counts, buffers, "list counts")?;
            let validity = decode_validity_set(validity, buffers, "list validity positions")?;
            let actual_child_slots = counts.sum()?;
            if actual_child_slots != *num_child_slots {
                return Err(Error::invalid_input_source(
                        format!(
                            "Sparse structural list count sum {} does not match declared child slots {}",
                            actual_child_slots, num_child_slots
                        )
                        .into(),
                    ));
            }
            SparseStructuralLayerPlan::List {
                num_slots: *num_slots,
                num_child_slots: *num_child_slots,
                non_empty_positions,
                counts,
                validity,
            }
        }
        SparseLayerDecompressors::FixedSizeList {
            num_slots,
            num_child_slots,
            dimension,
            validity,
        } => {
            let expected_child_slots = num_slots.checked_mul(*dimension).ok_or_else(|| {
                    Error::invalid_input_source(
                        format!(
                            "Sparse structural fixed-size-list child slot count overflows: slots={}, dimension={}",
                            num_slots, dimension
                        )
                        .into(),
                    )
                })?;
            if expected_child_slots != *num_child_slots {
                return Err(Error::invalid_input_source(
                        format!(
                            "Sparse structural fixed-size-list child slot count {} does not match slots {} * dimension {}",
                            num_child_slots, num_slots, dimension
                        )
                        .into(),
                    ));
            }
            SparseStructuralLayerPlan::FixedSizeList {
                num_slots: *num_slots,
                dimension: *dimension,
                validity: decode_validity_set(
                    validity,
                    buffers,
                    "fixed-size-list validity positions",
                )?,
            }
        }
    })
}

struct SparseStructuralSelection {
    plan: SparseStructuralPlan,
    leaf_ranges: Vec<Range<u64>>,
}

struct SparsePositionSelection {
    positions: SparsePositionPlan,
    ordinal_ranges: Vec<Range<u64>>,
}

fn validate_slice_ranges(ranges: &[Range<u64>], domain_len: u64, label: &str) -> Result<u64> {
    let mut total = 0_u64;
    for range in ranges {
        if range.start > range.end || range.end > domain_len {
            return Err(Error::invalid_input_source(
                format!(
                    "Sparse structural {label} slice {}..{} is outside domain {}",
                    range.start, range.end, domain_len
                )
                .into(),
            ));
        }
        total = total.checked_add(range.end - range.start).ok_or_else(|| {
            Error::invalid_input_source(
                format!("Sparse structural {label} slice length overflows").into(),
            )
        })?;
    }
    Ok(total)
}

fn push_coalesced_range(ranges: &mut Vec<Range<u64>>, range: Range<u64>) {
    if range.is_empty() {
        return;
    }
    if let Some(last) = ranges.last_mut()
        && last.end == range.start
    {
        last.end = range.end;
        return;
    }
    ranges.push(range);
}

fn position_segments_to_set(
    segments: Vec<Range<u64>>,
    output_domain: u64,
    label: &str,
) -> Result<SparsePositionPlan> {
    let segments = coalesce_ranges(segments);
    if segments.is_empty() {
        return Ok(SparsePositionPlan::empty());
    }
    if segments.len() == 1 {
        let segment = segments.first().ok_or_else(|| {
            Error::internal("Sparse structural segment unexpectedly missing".to_string())
        })?;
        if segment.start == 0 && segment.end == output_domain {
            return Ok(SparsePositionPlan::all(output_domain));
        }
        return Ok(SparsePositionPlan::range(
            segment.start,
            segment.end - segment.start,
        ));
    }

    let total_len = segments
        .iter()
        .map(|range| range.end - range.start)
        .try_fold(0_u64, |sum, len| {
            sum.checked_add(len).ok_or_else(|| {
                Error::invalid_input_source(
                    format!("Sparse structural {label} segment length overflows").into(),
                )
            })
        })?;
    let total_len = usize::try_from(total_len).map_err(|_| {
        Error::invalid_input_source(
            format!("Sparse structural {label} segment length exceeds usize::MAX").into(),
        )
    })?;
    let mut positions = Vec::with_capacity(total_len);
    for segment in segments {
        positions.extend(segment);
    }
    SparsePositionPlan::from_positions(positions, output_domain, label)
}

fn select_position_set(
    positions: &SparsePositionPlan,
    ranges: &[Range<u64>],
    domain_len: u64,
    label: &str,
) -> Result<SparsePositionSelection> {
    let output_domain = validate_slice_ranges(ranges, domain_len, label)?;
    let mut segments = Vec::new();
    let mut ordinal_ranges = Vec::new();
    let mut output_base = 0_u64;

    match positions {
        SparsePositionPlan::Empty => {}
        SparsePositionPlan::All { len } => {
            if *len != domain_len {
                return Err(Error::invalid_input_source(
                    format!(
                        "Sparse structural {label} all set length {} does not match domain {}",
                        len, domain_len
                    )
                    .into(),
                ));
            }
            for range in ranges {
                let range_len = range.end - range.start;
                let output_end = output_base.checked_add(range_len).ok_or_else(|| {
                    Error::invalid_input_source(
                        format!("Sparse structural {label} output domain overflows").into(),
                    )
                })?;
                push_coalesced_range(&mut segments, output_base..output_end);
                push_coalesced_range(&mut ordinal_ranges, range.clone());
                output_base = output_end;
            }
        }
        SparsePositionPlan::Range { start, len } => {
            let source_end = start.checked_add(*len).ok_or_else(|| {
                Error::invalid_input_source(
                    format!("Sparse structural {label} range overflows").into(),
                )
            })?;
            if source_end > domain_len {
                return Err(Error::invalid_input_source(
                    format!(
                        "Sparse structural {label} range {}..{} is outside domain {}",
                        start, source_end, domain_len
                    )
                    .into(),
                ));
            }
            for range in ranges {
                let intersect_start = range.start.max(*start);
                let intersect_end = range.end.min(source_end);
                if intersect_start < intersect_end {
                    let out_start = output_base
                        .checked_add(intersect_start - range.start)
                        .ok_or_else(|| {
                            Error::invalid_input_source(
                                format!("Sparse structural {label} output position overflows")
                                    .into(),
                            )
                        })?;
                    let out_end = output_base
                        .checked_add(intersect_end - range.start)
                        .ok_or_else(|| {
                            Error::invalid_input_source(
                                format!("Sparse structural {label} output position overflows")
                                    .into(),
                            )
                        })?;
                    push_coalesced_range(&mut segments, out_start..out_end);
                    push_coalesced_range(
                        &mut ordinal_ranges,
                        intersect_start - *start..intersect_end - *start,
                    );
                }
                output_base = output_base
                    .checked_add(range.end - range.start)
                    .ok_or_else(|| {
                        Error::invalid_input_source(
                            format!("Sparse structural {label} output domain overflows").into(),
                        )
                    })?;
            }
        }
        SparsePositionPlan::Arithmetic { start, step, .. } => {
            positions.validate_domain(domain_len, label)?;
            let mut selected = Vec::new();
            let mut single = None;
            for range in ranges {
                let first = positions.rank(range.start);
                let end = positions.rank(range.end);
                if first < end {
                    push_coalesced_range(&mut ordinal_ranges, first..end);
                    let out_start = output_base + (start + step * first - range.start);
                    if ranges.len() == 1 {
                        single = Some(SparsePositionPlan::Arithmetic {
                            start: out_start,
                            step: *step,
                            len: end - first,
                        });
                    } else {
                        selected.extend((0..end - first).map(|i| out_start + step * i));
                    }
                }
                output_base += range.end - range.start;
            }
            let positions =
                if let Some(SparsePositionPlan::Arithmetic { start, step, len }) = single {
                    if len == 1 {
                        SparsePositionPlan::range(start, 1)
                    } else {
                        SparsePositionPlan::Arithmetic { start, step, len }
                    }
                } else {
                    SparsePositionPlan::from_positions(selected, output_domain, label)?
                };
            return Ok(SparsePositionSelection {
                positions,
                ordinal_ranges,
            });
        }
        SparsePositionPlan::Explicit(source_positions) => {
            let mut out_positions = Vec::new();
            for range in ranges {
                let idx_start =
                    source_positions.partition_point(|position| *position < range.start);
                let idx_end = source_positions.partition_point(|position| *position < range.end);
                if idx_start < idx_end {
                    let selected_positions =
                        source_positions.get(idx_start..idx_end).ok_or_else(|| {
                            Error::invalid_input_source(
                                format!(
                                    "Sparse structural {label} explicit position slice is invalid"
                                )
                                .into(),
                            )
                        })?;
                    for position in selected_positions {
                        out_positions.push(
                            output_base
                                .checked_add(*position - range.start)
                                .ok_or_else(|| {
                                    Error::invalid_input_source(
                                        format!(
                                            "Sparse structural {label} output position overflows"
                                        )
                                        .into(),
                                    )
                                })?,
                        );
                    }
                    push_coalesced_range(&mut ordinal_ranges, idx_start as u64..idx_end as u64);
                }
                output_base = output_base
                    .checked_add(range.end - range.start)
                    .ok_or_else(|| {
                        Error::invalid_input_source(
                            format!("Sparse structural {label} output domain overflows").into(),
                        )
                    })?;
            }
            let positions =
                SparsePositionPlan::from_positions(out_positions, output_domain, label)?;
            return Ok(SparsePositionSelection {
                positions,
                ordinal_ranges,
            });
        }
    }

    Ok(SparsePositionSelection {
        positions: position_segments_to_set(segments, output_domain, label)?,
        ordinal_ranges,
    })
}

fn select_validity_set(
    validity: &SparseValiditySet,
    ranges: &[Range<u64>],
    domain_len: u64,
    label: &str,
) -> Result<SparseValiditySet> {
    Ok(SparseValiditySet {
        meaning: validity.meaning,
        positions: select_position_set(&validity.positions, ranges, domain_len, label)?.positions,
    })
}

fn offsets_from_counts(counts: &[u64]) -> Result<Vec<u64>> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut offset = 0_u64;
    offsets.push(offset);
    for count in counts {
        offset = offset.checked_add(*count).ok_or_else(|| {
            Error::invalid_input_source("Sparse structural list count offsets overflow".into())
        })?;
        offsets.push(offset);
    }
    Ok(offsets)
}

fn coalesce_ranges(ranges: Vec<Range<u64>>) -> Vec<Range<u64>> {
    let mut coalesced: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if range.is_empty() {
            continue;
        }
        if let Some(last) = coalesced.last_mut()
            && last.end == range.start
        {
            last.end = range.end;
            continue;
        }
        coalesced.push(range);
    }
    coalesced
}

fn slice_list_layer(
    num_slots: u64,
    num_child_slots: u64,
    non_empty_positions: &SparsePositionPlan,
    counts: &SparseCountPlan,
    validity: &SparseValiditySet,
    ranges: &[Range<u64>],
) -> Result<(SparseStructuralLayerPlan, Vec<Range<u64>>)> {
    if non_empty_positions.len() != counts.len() {
        return Err(Error::invalid_input_source(
            "Sparse structural list has mismatched non-empty positions and counts".into(),
        ));
    }
    let count_sum = counts.sum()?;
    if count_sum != num_child_slots {
        return Err(Error::invalid_input_source(
            format!(
                "Sparse structural list count sum {} does not match child slots {}",
                count_sum, num_child_slots
            )
            .into(),
        ));
    }

    let non_empty_selection =
        select_position_set(non_empty_positions, ranges, num_slots, "list non-empty")?;
    let out_counts = select_count_set(
        counts,
        &non_empty_selection.ordinal_ranges,
        non_empty_selection.positions.len(),
    )?;
    let child_ranges =
        child_ranges_from_counts(counts, num_child_slots, &non_empty_selection.ordinal_ranges)?;
    let out_validity = select_validity_set(validity, ranges, num_slots, "list validity")?;
    let out_num_slots = validate_slice_ranges(ranges, num_slots, "list")?;
    let out_num_child_slots = out_counts.sum()?;
    Ok((
        SparseStructuralLayerPlan::List {
            num_slots: out_num_slots,
            num_child_slots: out_num_child_slots,
            non_empty_positions: non_empty_selection.positions,
            counts: out_counts,
            validity: out_validity,
        },
        child_ranges,
    ))
}

fn select_count_set(
    counts: &SparseCountPlan,
    ordinal_ranges: &[Range<u64>],
    selected_len: u64,
) -> Result<SparseCountPlan> {
    match counts {
        SparseCountPlan::Empty => {
            if selected_len != 0 {
                return Err(Error::invalid_input_source(
                    "Sparse structural selected non-empty positions but counts are empty".into(),
                ));
            }
            Ok(SparseCountPlan::Empty)
        }
        SparseCountPlan::Constant { value, len } => {
            validate_ordinal_ranges(ordinal_ranges, *len, "constant list counts")?;
            Ok(SparseCountPlan::constant(*value, selected_len))
        }
        SparseCountPlan::Explicit {
            counts: source_counts,
            ..
        } => {
            validate_ordinal_ranges(
                ordinal_ranges,
                source_counts.len() as u64,
                "explicit list counts",
            )?;
            let selected_len = usize::try_from(selected_len).map_err(|_| {
                Error::invalid_input_source(
                    "Sparse structural selected count length exceeds usize::MAX".into(),
                )
            })?;
            let mut out_counts = Vec::with_capacity(selected_len);
            for range in ordinal_ranges {
                let start = usize::try_from(range.start).map_err(|_| {
                    Error::invalid_input_source(
                        "Sparse structural count ordinal exceeds usize::MAX".into(),
                    )
                })?;
                let end = usize::try_from(range.end).map_err(|_| {
                    Error::invalid_input_source(
                        "Sparse structural count ordinal exceeds usize::MAX".into(),
                    )
                })?;
                out_counts.extend_from_slice(source_counts.get(start..end).ok_or_else(|| {
                    Error::invalid_input_source(
                        "Sparse structural explicit count slice is invalid".into(),
                    )
                })?);
            }
            SparseCountPlan::from_counts(out_counts)
        }
    }
}

fn validate_ordinal_ranges(ranges: &[Range<u64>], len: u64, label: &str) -> Result<()> {
    for range in ranges {
        if range.start > range.end || range.end > len {
            return Err(Error::invalid_input_source(
                format!(
                    "Sparse structural {label} ordinal range {}..{} is outside {} values",
                    range.start, range.end, len
                )
                .into(),
            ));
        }
    }
    Ok(())
}

fn child_ranges_from_counts(
    counts: &SparseCountPlan,
    num_child_slots: u64,
    ordinal_ranges: &[Range<u64>],
) -> Result<Vec<Range<u64>>> {
    if ordinal_ranges.is_empty() {
        return Ok(Vec::new());
    }
    let child_ranges = match counts {
        SparseCountPlan::Empty => {
            return Err(Error::invalid_input_source(
                "Sparse structural selected non-empty positions but counts are empty".into(),
            ));
        }
        SparseCountPlan::Constant { value, len } => {
            let expected_child_slots = value.checked_mul(*len).ok_or_else(|| {
                Error::invalid_input_source(
                    "Sparse structural list constant count sum overflows child slots".into(),
                )
            })?;
            if expected_child_slots != num_child_slots {
                return Err(Error::invalid_input_source(
                    format!(
                        "Sparse structural list constant count sum {} does not match child slots {}",
                        expected_child_slots, num_child_slots
                    )
                    .into(),
                ));
            }
            validate_ordinal_ranges(ordinal_ranges, *len, "constant list counts")?;
            ordinal_ranges
                .iter()
                .map(|range| {
                    let start = range.start.checked_mul(*value).ok_or_else(|| {
                        Error::invalid_input_source(
                            "Sparse structural list child range start overflows".into(),
                        )
                    })?;
                    let end = range.end.checked_mul(*value).ok_or_else(|| {
                        Error::invalid_input_source(
                            "Sparse structural list child range end overflows".into(),
                        )
                    })?;
                    Ok(start..end)
                })
                .collect::<Result<Vec<_>>>()?
        }
        SparseCountPlan::Explicit {
            counts,
            offsets: value_offsets,
        } => {
            let last_offset = *value_offsets.last().ok_or_else(|| {
                Error::invalid_input_source(
                    "Sparse structural list count offsets are unexpectedly empty".into(),
                )
            })?;
            if last_offset != num_child_slots {
                return Err(Error::invalid_input_source(
                    format!(
                        "Sparse structural list count sum {} does not match child slots {}",
                        last_offset, num_child_slots
                    )
                    .into(),
                ));
            }
            validate_ordinal_ranges(ordinal_ranges, counts.len() as u64, "explicit list counts")?;
            ordinal_ranges
                .iter()
                .map(|range| {
                    let start = usize::try_from(range.start).map_err(|_| {
                        Error::invalid_input_source(
                            "Sparse structural count ordinal exceeds usize::MAX".into(),
                        )
                    })?;
                    let end = usize::try_from(range.end).map_err(|_| {
                        Error::invalid_input_source(
                            "Sparse structural count ordinal exceeds usize::MAX".into(),
                        )
                    })?;
                    let start_offset = *value_offsets.get(start).ok_or_else(|| {
                        Error::invalid_input_source(
                            "Sparse structural list start offset is missing".into(),
                        )
                    })?;
                    let end_offset = *value_offsets.get(end).ok_or_else(|| {
                        Error::invalid_input_source(
                            "Sparse structural list end offset is missing".into(),
                        )
                    })?;
                    Ok(start_offset..end_offset)
                })
                .collect::<Result<Vec<_>>>()?
        }
    };
    Ok(coalesce_ranges(child_ranges))
}

fn slice_sparse_plan(
    plan: &SparseStructuralPlan,
    row_ranges: &[Range<u64>],
    row_domain: u64,
) -> Result<SparseStructuralSelection> {
    // The immutable page plan is validated before it enters the page cache.
    // Validate each requested range below, without rescanning the whole page.
    let mut selected_ranges = row_ranges.to_vec();
    let mut selected_domain = row_domain;
    let mut sliced_layers = Vec::with_capacity(plan.layers.len());

    for layer in &plan.layers {
        match layer {
            SparseStructuralLayerPlan::Validity {
                num_slots,
                validity,
            } => {
                if selected_domain != *num_slots {
                    return Err(Error::invalid_input_source(
                        format!(
                            "Sparse structural validity slice domain {} does not match {} slots",
                            selected_domain, num_slots
                        )
                        .into(),
                    ));
                }
                let out_num_slots =
                    validate_slice_ranges(&selected_ranges, *num_slots, "validity")?;
                sliced_layers.push(SparseStructuralLayerPlan::Validity {
                    num_slots: out_num_slots,
                    validity: select_validity_set(
                        validity,
                        &selected_ranges,
                        *num_slots,
                        "validity",
                    )?,
                });
            }
            SparseStructuralLayerPlan::List {
                num_slots,
                num_child_slots,
                non_empty_positions,
                counts,
                validity,
                ..
            } => {
                if selected_domain != *num_slots {
                    return Err(Error::invalid_input_source(
                        format!(
                            "Sparse structural list slice domain {} does not match {} slots",
                            selected_domain, num_slots
                        )
                        .into(),
                    ));
                }
                let (sliced_layer, child_ranges) = slice_list_layer(
                    *num_slots,
                    *num_child_slots,
                    non_empty_positions,
                    counts,
                    validity,
                    &selected_ranges,
                )?;
                sliced_layers.push(sliced_layer);
                selected_ranges = child_ranges;
                selected_domain = *num_child_slots;
            }
            SparseStructuralLayerPlan::FixedSizeList {
                num_slots,
                dimension,
                validity,
            } => {
                if selected_domain != *num_slots {
                    return Err(Error::invalid_input_source(
                        format!(
                            "Sparse structural fixed-size-list slice domain {} does not match {} slots",
                            selected_domain, num_slots
                        )
                        .into(),
                    ));
                }
                let num_child_slots = num_slots.checked_mul(*dimension).ok_or_else(|| {
                    Error::invalid_input_source(
                        format!(
                            "Sparse structural fixed-size-list child slot count overflows: slots={}, dimension={}",
                            num_slots, dimension
                        )
                        .into(),
                    )
                })?;
                let out_num_slots =
                    validate_slice_ranges(&selected_ranges, *num_slots, "fixed-size-list")?;
                let child_ranges = selected_ranges
                    .iter()
                    .map(|range| {
                        let start = range.start.checked_mul(*dimension).ok_or_else(|| {
                            Error::invalid_input_source(
                                "Sparse structural fixed-size-list child range start overflows"
                                    .into(),
                            )
                        })?;
                        let end = range.end.checked_mul(*dimension).ok_or_else(|| {
                            Error::invalid_input_source(
                                "Sparse structural fixed-size-list child range end overflows"
                                    .into(),
                            )
                        })?;
                        Ok(start..end)
                    })
                    .collect::<Result<Vec<_>>>()?;
                sliced_layers.push(SparseStructuralLayerPlan::FixedSizeList {
                    num_slots: out_num_slots,
                    dimension: *dimension,
                    validity: select_validity_set(
                        validity,
                        &selected_ranges,
                        *num_slots,
                        "fixed-size-list validity",
                    )?,
                });
                selected_ranges = child_ranges;
                selected_domain = num_child_slots;
            }
        }
    }

    if selected_domain != plan.num_visible_items {
        return Err(Error::invalid_input_source(
            format!(
                "Sparse structural selected terminal domain {} does not match {} visible items",
                selected_domain, plan.num_visible_items
            )
            .into(),
        ));
    }
    let num_visible_items =
        validate_slice_ranges(&selected_ranges, plan.num_visible_items, "visible value")?;
    Ok(SparseStructuralSelection {
        plan: SparseStructuralPlan {
            layers: sliced_layers,
            num_visible_items,
        },
        leaf_ranges: coalesce_ranges(selected_ranges),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn assert_invalid_input_contains(err: Error, expected: &str) {
        let message = err.to_string();
        assert!(
            matches!(&err, Error::InvalidInput { .. }),
            "expected InvalidInput, got {err:?}"
        );
        assert!(
            message.contains(expected),
            "expected error to contain {expected:?}, got {message}"
        );
    }

    fn null_set(positions: SparsePositionPlan) -> SparseValiditySet {
        SparseValiditySet {
            meaning: SparseValidityMeaning::NullPositions,
            positions,
        }
    }

    #[test]
    fn generic_sparse_positions_remain_symbolic() {
        let len = 1_000_000_000_000;
        for encoding in [
            EncodingUtils::range(64, 3, 7),
            EncodingUtils::delta(
                64,
                3,
                EncodingUtils::constant(Some(Bytes::copy_from_slice(&7_u64.to_le_bytes()))),
            ),
        ] {
            let decoder = position_decoder(&encoding, len, 7 * len, "test").unwrap();
            let positions = decode_position_set(&decoder, &mut std::iter::empty(), "test").unwrap();
            assert_eq!(positions.deep_size(), 0);
            assert_eq!(positions.rank(17), 2);
            assert!(positions.contains(17));
            assert!(!positions.contains(18));
            let selected = select_position_set(&positions, &[10..32], 7 * len, "test").unwrap();
            assert_eq!(
                selected.positions.materialize().unwrap(),
                vec![0, 7, 14, 21]
            );
            assert_eq!(selected.ordinal_ranges, vec![1..5]);
            assert_eq!(selected.positions.deep_size(), 0);
            let opposite = SparsePositionPlan::Arithmetic {
                start: 4,
                step: 7,
                len,
            };
            assert!(positions.is_disjoint(&opposite, 7 * len).unwrap());
        }
    }

    #[test]
    fn generic_sparse_sequences_reject_invalid_metadata() {
        for (encoding, count, domain) in [
            (EncodingUtils::range(64, u64::MAX - 1, 2), 2, u64::MAX),
            (EncodingUtils::range(64, 0, 0), 2, 3),
            (EncodingUtils::range(64, 1, 1), 1, 3),
            (EncodingUtils::constant(None), 1, 3),
            (EncodingUtils::flat(64, None), 0, 3),
            (EncodingUtils::flat(32, None), 2, 3),
            (
                EncodingUtils::constant(Some(Bytes::copy_from_slice(&1_u64.to_le_bytes()))),
                2,
                3,
            ),
        ] {
            assert!(position_decoder(&encoding, count, domain, "test").is_err());
        }
        let zero = EncodingUtils::constant(Some(Bytes::copy_from_slice(&0_u64.to_le_bytes())));
        assert!(count_decoder(&zero, 2, "test").is_err());
        let huge = EncodingUtils::constant(Some(Bytes::copy_from_slice(&u64::MAX.to_le_bytes())));
        assert!(count_decoder(&huge, 2, "test").is_err());
    }

    #[test]
    fn arithmetic_set_operations_match_explicit_positions() {
        for a in 0..8 {
            for b in 0..8 {
                for s in 1..8 {
                    for t in 1..8 {
                        let first = SparsePositionPlan::Arithmetic {
                            start: a,
                            step: s,
                            len: 5,
                        };
                        let second = SparsePositionPlan::Arithmetic {
                            start: b,
                            step: t,
                            len: 4,
                        };
                        let left = first.materialize().unwrap();
                        let right = second.materialize().unwrap();
                        assert_eq!(
                            first.is_disjoint(&second, 64).unwrap(),
                            left.iter().all(|v| !right.contains(v))
                        );
                        assert_eq!(
                            first.is_subset_of(&second, 64).unwrap(),
                            left.iter().all(|v| right.contains(v))
                        );
                        assert_eq!(
                            second.is_subset_of(&first, 64).unwrap(),
                            right.iter().all(|v| left.contains(v))
                        );
                    }
                }
            }
        }
    }

    fn valid_set(positions: SparsePositionPlan) -> SparseValiditySet {
        SparseValiditySet {
            meaning: SparseValidityMeaning::ValidPositions,
            positions,
        }
    }

    #[test]
    fn semantic_position_and_count_sets_project_without_materializing_ranges() {
        let ranges = [1..4];
        assert_eq!(
            select_position_set(&SparsePositionPlan::Empty, &ranges, 5, "empty")
                .unwrap()
                .positions,
            SparsePositionPlan::Empty
        );
        assert_eq!(
            select_position_set(&SparsePositionPlan::all(5), &ranges, 5, "all")
                .unwrap()
                .positions,
            SparsePositionPlan::all(3)
        );
        assert_eq!(
            select_position_set(&SparsePositionPlan::range(1, 3), &ranges, 5, "range")
                .unwrap()
                .positions,
            SparsePositionPlan::all(3)
        );
        assert_eq!(
            select_position_set(
                &SparsePositionPlan::Explicit(vec![0, 2, 4]),
                &[0..1, 4..5],
                5,
                "explicit",
            )
            .unwrap()
            .positions,
            SparsePositionPlan::all(2)
        );

        assert_eq!(
            select_count_set(&SparseCountPlan::Empty, &[], 0).unwrap(),
            SparseCountPlan::Empty
        );
        assert_eq!(
            select_count_set(&SparseCountPlan::constant(2, 3), &[0..1, 2..3], 2,).unwrap(),
            SparseCountPlan::constant(2, 2)
        );
        assert_eq!(
            select_count_set(
                &SparseCountPlan::from_counts(vec![1, 2, 3]).unwrap(),
                &[0..1, 2..3],
                2,
            )
            .unwrap(),
            SparseCountPlan::from_counts(vec![1, 3]).unwrap()
        );
    }

    #[test]
    fn validity_polarities_rebuild_the_same_arrow_domain() {
        let mut null_builder = BooleanBufferBuilder::new(4);
        null_set(SparsePositionPlan::Explicit(vec![1, 3]))
            .append_to(&mut null_builder, 4)
            .unwrap();
        assert_eq!(
            null_builder.finish().iter().collect::<Vec<_>>(),
            vec![true, false, true, false]
        );

        let mut valid_builder = BooleanBufferBuilder::new(4);
        valid_set(SparsePositionPlan::range(1, 2))
            .append_to(&mut valid_builder, 4)
            .unwrap();
        assert_eq!(
            valid_builder.finish().iter().collect::<Vec<_>>(),
            vec![false, true, true, false]
        );
    }

    #[test]
    fn schema_layer_mismatches_are_invalid_input() {
        let mut missing = SparseStructuralUnraveler::new(SparseStructuralPlan {
            layers: Vec::new(),
            num_visible_items: 1,
        });
        let mut validity = BooleanBufferBuilder::new(1);
        let err = missing.unravel_validity(&mut validity).unwrap_err();
        assert_invalid_input_contains(err, "fewer layers than the Arrow schema");

        let mut fixed_size_list = SparseStructuralUnraveler::new(SparseStructuralPlan {
            layers: vec![SparseStructuralLayerPlan::FixedSizeList {
                num_slots: 2,
                dimension: 2,
                validity: null_set(SparsePositionPlan::empty()),
            }],
            num_visible_items: 4,
        });
        let mut validity = BooleanBufferBuilder::new(2);
        let err = fixed_size_list.unravel_validity(&mut validity).unwrap_err();
        assert_invalid_input_contains(err, "does not match the Arrow schema");

        let extra = SparseStructuralUnraveler::new(SparseStructuralPlan {
            layers: vec![SparseStructuralLayerPlan::Validity {
                num_slots: 1,
                validity: null_set(SparsePositionPlan::empty()),
            }],
            num_visible_items: 1,
        });
        let err = extra.ensure_exhausted().unwrap_err();
        assert_invalid_input_contains(err, "1 unconsumed layer");
    }

    fn nested_plan() -> SparseStructuralPlan {
        SparseStructuralPlan {
            layers: vec![
                SparseStructuralLayerPlan::Validity {
                    num_slots: 6,
                    validity: null_set(SparsePositionPlan::Explicit(vec![1, 4])),
                },
                SparseStructuralLayerPlan::List {
                    num_slots: 6,
                    num_child_slots: 5,
                    non_empty_positions: SparsePositionPlan::Explicit(vec![0, 2, 5]),
                    counts: SparseCountPlan::from_counts(vec![2, 1, 2]).unwrap(),
                    validity: null_set(SparsePositionPlan::Explicit(vec![1, 4])),
                },
                SparseStructuralLayerPlan::FixedSizeList {
                    num_slots: 5,
                    dimension: 2,
                    validity: valid_set(SparsePositionPlan::range(1, 3)),
                },
            ],
            num_visible_items: 10,
        }
    }

    #[test]
    fn discontiguous_projection_preserves_outer_to_inner_order() {
        let selection = slice_sparse_plan(&nested_plan(), &[5..6, 0..1], 6).unwrap();
        assert_eq!(selection.leaf_ranges, vec![6..10, 0..4]);
        assert_eq!(selection.plan.num_visible_items, 8);
        selection.plan.validate(2).unwrap();

        let SparseStructuralLayerPlan::List {
            non_empty_positions,
            counts,
            ..
        } = &selection.plan.layers[1]
        else {
            panic!("expected projected list layer");
        };
        assert_eq!(*non_empty_positions, SparsePositionPlan::all(2));
        assert_eq!(*counts, SparseCountPlan::constant(2, 2));
    }

    #[test]
    fn no_value_projection_keeps_empty_and_null_list_structure() {
        let selection = slice_sparse_plan(&nested_plan(), &[1..2, 3..4], 6).unwrap();
        assert!(selection.leaf_ranges.is_empty());
        assert_eq!(selection.plan.num_visible_items, 0);
        selection.plan.validate(2).unwrap();

        let SparseStructuralLayerPlan::List {
            num_slots,
            num_child_slots,
            non_empty_positions,
            counts,
            validity,
        } = &selection.plan.layers[1]
        else {
            panic!("expected projected list layer");
        };
        assert_eq!((*num_slots, *num_child_slots), (2, 0));
        assert_eq!(*non_empty_positions, SparsePositionPlan::Empty);
        assert_eq!(*counts, SparseCountPlan::Empty);
        assert_eq!(*validity, null_set(SparsePositionPlan::range(0, 1)));
    }
    #[rstest::rstest]
    #[case::zero_dimension(0, 4, "dimension is zero")]
    #[case::unaligned_rows(3, 4, "not divisible")]
    fn rejects_invalid_row_scale(#[case] dimension: u64, #[case] rows: u64, #[case] message: &str) {
        let structure = pb23::SparseStructure {
            layers: vec![pb23::SparseStructuralLayer {
                layer: Some(pb23::sparse_structural_layer::Layer::FixedSizeList(
                    pb23::SparseFixedSizeListLayer {
                        dimension,
                        validity: None,
                    },
                )),
            }],
        };
        assert_invalid_input_contains(structure.row_count_and_scale(rows).unwrap_err(), message);
    }

    #[test]
    fn rejects_derived_domain_overflow() {
        let layer = pb23::SparseStructuralLayer {
            layer: Some(pb23::sparse_structural_layer::Layer::FixedSizeList(
                pb23::SparseFixedSizeListLayer {
                    dimension: 2,
                    validity: None,
                },
            )),
        };
        assert_invalid_input_contains(child_domain(&layer, u64::MAX).unwrap_err(), "overflows");
        let structure = pb23::SparseStructure {
            layers: vec![layer; 64],
        };
        assert_invalid_input_contains(structure.row_count_and_scale(0).unwrap_err(), "overflows");
    }

    #[rstest::rstest]
    #[case::duplicate(vec![1, 1], "strictly increasing")]
    #[case::unordered(vec![3, 1], "strictly increasing")]
    #[case::out_of_domain(vec![0, 4], "outside")]
    fn rejects_invalid_decoded_positions(#[case] positions: Vec<u64>, #[case] message: &str) {
        let decoder = position_decoder(&EncodingUtils::flat(64, None), 2, 4, "test").unwrap();
        let bytes = Bytes::from(
            positions
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        let error =
            decode_position_set(&decoder, &mut vec![bytes].into_iter(), "test").unwrap_err();
        assert_invalid_input_contains(error, message);
    }

    #[test]
    fn rejects_malformed_compressed_structural_payload() {
        use crate::encodings::physical::block::{CompressionConfig, CompressionScheme};
        let encoding = EncodingUtils::wrapped(
            CompressionConfig::new(CompressionScheme::Lz4, None),
            EncodingUtils::flat(64, None),
        )
        .unwrap();
        let decoder = position_decoder(&encoding, 2, 4, "test").unwrap();
        let error = decode_position_set(
            &decoder,
            &mut vec![Bytes::from_static(&[0xff])].into_iter(),
            "test",
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    }
}
