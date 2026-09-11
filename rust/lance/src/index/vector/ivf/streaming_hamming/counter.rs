// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compact per-bit counters for Hamming coreset summaries.

use lance_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum CounterWidth {
    U8,
    U16,
    U32,
    U64,
}

impl CounterWidth {
    fn for_count(count: u64) -> Self {
        if u8::try_from(count).is_ok() {
            Self::U8
        } else if u16::try_from(count).is_ok() {
            Self::U16
        } else if u32::try_from(count).is_ok() {
            Self::U32
        } else {
            Self::U64
        }
    }
}

pub(super) trait CompactCounter: Copy {
    fn from_u64(value: u64) -> Option<Self>;

    fn to_u64(self) -> u64;
}

macro_rules! impl_compact_counter {
    ($type:ty) => {
        impl CompactCounter for $type {
            fn from_u64(value: u64) -> Option<Self> {
                Self::try_from(value).ok()
            }

            fn to_u64(self) -> u64 {
                u64::from(self)
            }
        }
    };
}

impl_compact_counter!(u8);
impl_compact_counter!(u16);
impl_compact_counter!(u32);

impl CompactCounter for u64 {
    fn from_u64(value: u64) -> Option<Self> {
        Some(value)
    }

    fn to_u64(self) -> u64 {
        self
    }
}

fn widen_vec<T, U>(values: Vec<T>, convert: impl Fn(T) -> U) -> Vec<U> {
    let mut widened = Vec::with_capacity(values.capacity());
    widened.extend(values.into_iter().map(convert));
    widened
}

fn add_counter_slices<T: CompactCounter, U: CompactCounter>(
    destination: &mut [T],
    source: &[U],
) -> Result<()> {
    debug_assert_eq!(destination.len(), source.len());
    for (bit_index, (sum, value)) in destination.iter_mut().zip(source).enumerate() {
        let next = sum.to_u64().checked_add(value.to_u64()).ok_or_else(|| {
            Error::invalid_input(format!(
                "Hamming one-count overflow while merging bit {bit_index}"
            ))
        })?;
        *sum = T::from_u64(next).ok_or_else(|| {
            Error::invalid_input(format!(
                "Hamming one-count {next} does not fit destination counter at bit {bit_index}"
            ))
        })?;
    }
    Ok(())
}

fn extend_counter_slice<T: CompactCounter, U: CompactCounter>(
    destination: &mut Vec<T>,
    source: &[U],
) -> Result<()> {
    for (bit_index, value) in source.iter().copied().enumerate() {
        let value = value.to_u64();
        destination.push(T::from_u64(value).ok_or_else(|| {
            Error::invalid_input(format!(
                "Hamming one-count {value} does not fit destination counter at bit {bit_index}"
            ))
        })?);
    }
    Ok(())
}

pub(super) fn add_vector_bits<T: CompactCounter>(ones: &mut [T], vector: &[u8]) -> Result<()> {
    debug_assert_eq!(ones.len(), vector.len() * 8);
    for (byte_index, byte) in vector.iter().copied().enumerate() {
        for bit in 0..8 {
            if byte & (1 << bit) != 0 {
                let bit_index = byte_index * 8 + bit;
                let next = ones[bit_index].to_u64().checked_add(1).ok_or_else(|| {
                    Error::invalid_input(format!("Hamming one-count overflow at bit {bit_index}"))
                })?;
                ones[bit_index] = T::from_u64(next).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Hamming one-count {next} does not fit counter at bit {bit_index}"
                    ))
                })?;
            }
        }
    }
    Ok(())
}

pub(super) fn mode_from_counts<T: CompactCounter>(
    count: u64,
    ones: &[T],
    dimension: usize,
) -> Vec<u8> {
    let mut mode = vec![0_u8; dimension];
    for (bit_index, one_count) in ones.iter().copied().enumerate() {
        let one_count = one_count.to_u64();
        if one_count > count - one_count {
            mode[bit_index / 8] |= 1 << (bit_index % 8);
        }
    }
    mode
}

pub(super) fn summary_cost<T: CompactCounter>(
    count: u64,
    ones: &[T],
    centroid: &[u8],
    dimension: usize,
) -> Result<u64> {
    if centroid.len() != dimension || ones.len() != dimension * 8 {
        return Err(Error::invalid_input(format!(
            "invalid Hamming summary dimensions: centroid={}, bit_counts={}, dimension={dimension}",
            centroid.len(),
            ones.len()
        )));
    }
    let mut cost = 0_u64;
    for (bit_index, one_count) in ones.iter().copied().enumerate() {
        let one_count = one_count.to_u64();
        let mismatch = if centroid[bit_index / 8] & (1 << (bit_index % 8)) == 0 {
            one_count
        } else {
            count - one_count
        };
        cost = cost.checked_add(mismatch).ok_or_else(|| {
            Error::invalid_input("Hamming coreset distance overflow during assignment")
        })?;
    }
    Ok(cost)
}

/// Contiguous per-bit counters that use the narrowest exact integer width.
///
/// One width is shared by the whole buffer so hot loops dispatch once instead
/// of storing and inspecting a tag for every counter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum CompactCounters {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl CompactCounters {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self::U8(Vec::with_capacity(capacity))
    }

    pub(super) fn zeros(len: usize) -> Self {
        Self::U8(vec![0; len])
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
        }
    }

    pub(super) fn width(&self) -> CounterWidth {
        match self {
            Self::U8(_) => CounterWidth::U8,
            Self::U16(_) => CounterWidth::U16,
            Self::U32(_) => CounterWidth::U32,
            Self::U64(_) => CounterWidth::U64,
        }
    }

    pub(super) fn ensure_width_for(&mut self, count: u64) {
        self.promote_to(CounterWidth::for_count(count));
    }

    pub(super) fn promote_to(&mut self, width: CounterWidth) {
        while self.width() < width {
            let current = std::mem::replace(self, Self::U8(Vec::new()));
            *self = match current {
                Self::U8(values) => Self::U16(widen_vec(values, u16::from)),
                Self::U16(values) => Self::U32(widen_vec(values, u32::from)),
                Self::U32(values) => Self::U64(widen_vec(values, u64::from)),
                Self::U64(values) => Self::U64(values),
            };
        }
    }

    pub(super) fn value(&self, index: usize) -> u64 {
        match self {
            Self::U8(values) => u64::from(values[index]),
            Self::U16(values) => u64::from(values[index]),
            Self::U32(values) => u64::from(values[index]),
            Self::U64(values) => values[index],
        }
    }

    #[cfg(test)]
    pub(super) fn extend_u64(&mut self, source: &[u64], max_count: u64) -> Result<()> {
        self.ensure_width_for(max_count);
        match self {
            Self::U8(destination) => extend_counter_slice(destination, source),
            Self::U16(destination) => extend_counter_slice(destination, source),
            Self::U32(destination) => extend_counter_slice(destination, source),
            Self::U64(destination) => extend_counter_slice(destination, source),
        }
    }

    pub(super) fn extend_from(
        &mut self,
        source: &Self,
        source_start: usize,
        len: usize,
        max_count: u64,
    ) -> Result<()> {
        self.ensure_width_for(max_count);
        let source_end = source_start.checked_add(len).ok_or_else(|| {
            Error::invalid_input("Hamming counter source range overflow while appending")
        })?;
        if source_end > source.len() {
            return Err(Error::invalid_input(format!(
                "Hamming counter source range {source_start}..{source_end} exceeds length {}",
                source.len()
            )));
        }

        macro_rules! extend {
            ($destination:expr, $source:expr) => {
                extend_counter_slice($destination, &$source[source_start..source_end])
            };
        }
        match (self, source) {
            (Self::U8(destination), Self::U8(source)) => extend!(destination, source),
            (Self::U8(destination), Self::U16(source)) => extend!(destination, source),
            (Self::U8(destination), Self::U32(source)) => extend!(destination, source),
            (Self::U8(destination), Self::U64(source)) => extend!(destination, source),
            (Self::U16(destination), Self::U8(source)) => extend!(destination, source),
            (Self::U16(destination), Self::U16(source)) => extend!(destination, source),
            (Self::U16(destination), Self::U32(source)) => extend!(destination, source),
            (Self::U16(destination), Self::U64(source)) => extend!(destination, source),
            (Self::U32(destination), Self::U8(source)) => extend!(destination, source),
            (Self::U32(destination), Self::U16(source)) => extend!(destination, source),
            (Self::U32(destination), Self::U32(source)) => extend!(destination, source),
            (Self::U32(destination), Self::U64(source)) => extend!(destination, source),
            (Self::U64(destination), Self::U8(source)) => extend!(destination, source),
            (Self::U64(destination), Self::U16(source)) => extend!(destination, source),
            (Self::U64(destination), Self::U32(source)) => extend!(destination, source),
            (Self::U64(destination), Self::U64(source)) => extend!(destination, source),
        }
    }

    pub(super) fn append(&mut self, mut other: Self) -> Result<()> {
        let width = self.width().max(other.width());
        self.promote_to(width);
        other.promote_to(width);
        match (self, other) {
            (Self::U8(destination), Self::U8(mut source)) => destination.append(&mut source),
            (Self::U16(destination), Self::U16(mut source)) => destination.append(&mut source),
            (Self::U32(destination), Self::U32(mut source)) => destination.append(&mut source),
            (Self::U64(destination), Self::U64(mut source)) => destination.append(&mut source),
            _ => {
                return Err(Error::invalid_input(
                    "Hamming counter widths differ after promotion",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn add_from(
        &mut self,
        destination_start: usize,
        source: &Self,
        source_start: usize,
        len: usize,
        max_count: u64,
    ) -> Result<()> {
        self.ensure_width_for(max_count);
        let destination_end = destination_start.checked_add(len).ok_or_else(|| {
            Error::invalid_input("Hamming counter destination range overflow while merging")
        })?;
        let source_end = source_start.checked_add(len).ok_or_else(|| {
            Error::invalid_input("Hamming counter source range overflow while merging")
        })?;
        if destination_end > self.len() || source_end > source.len() {
            return Err(Error::invalid_input(format!(
                "invalid Hamming counter merge ranges: destination={destination_start}..{destination_end} of {}, source={source_start}..{source_end} of {}",
                self.len(),
                source.len()
            )));
        }

        macro_rules! add {
            ($destination:expr, $source:expr) => {
                add_counter_slices(
                    &mut $destination[destination_start..destination_end],
                    &$source[source_start..source_end],
                )
            };
        }
        match (self, source) {
            (Self::U8(destination), Self::U8(source)) => add!(destination, source),
            (Self::U8(destination), Self::U16(source)) => add!(destination, source),
            (Self::U8(destination), Self::U32(source)) => add!(destination, source),
            (Self::U8(destination), Self::U64(source)) => add!(destination, source),
            (Self::U16(destination), Self::U8(source)) => add!(destination, source),
            (Self::U16(destination), Self::U16(source)) => add!(destination, source),
            (Self::U16(destination), Self::U32(source)) => add!(destination, source),
            (Self::U16(destination), Self::U64(source)) => add!(destination, source),
            (Self::U32(destination), Self::U8(source)) => add!(destination, source),
            (Self::U32(destination), Self::U16(source)) => add!(destination, source),
            (Self::U32(destination), Self::U32(source)) => add!(destination, source),
            (Self::U32(destination), Self::U64(source)) => add!(destination, source),
            (Self::U64(destination), Self::U8(source)) => add!(destination, source),
            (Self::U64(destination), Self::U16(source)) => add!(destination, source),
            (Self::U64(destination), Self::U32(source)) => add!(destination, source),
            (Self::U64(destination), Self::U64(source)) => add!(destination, source),
        }
    }

    pub(super) fn add_vector(
        &mut self,
        destination_start: usize,
        vector: &[u8],
        max_count: u64,
    ) -> Result<()> {
        self.ensure_width_for(max_count);
        let len = vector.len().checked_mul(8).ok_or_else(|| {
            Error::invalid_input("Hamming vector bit dimension overflow while accumulating")
        })?;
        let destination_end = destination_start.checked_add(len).ok_or_else(|| {
            Error::invalid_input("Hamming counter destination range overflow while accumulating")
        })?;
        if destination_end > self.len() {
            return Err(Error::invalid_input(format!(
                "Hamming vector counter range {destination_start}..{destination_end} exceeds length {}",
                self.len()
            )));
        }
        match self {
            Self::U8(values) => {
                add_vector_bits(&mut values[destination_start..destination_end], vector)
            }
            Self::U16(values) => {
                add_vector_bits(&mut values[destination_start..destination_end], vector)
            }
            Self::U32(values) => {
                add_vector_bits(&mut values[destination_start..destination_end], vector)
            }
            Self::U64(values) => {
                add_vector_bits(&mut values[destination_start..destination_end], vector)
            }
        }
    }

    pub(super) fn mode(&self, start: usize, count: u64, dimension: usize) -> Vec<u8> {
        let end = start + dimension * 8;
        match self {
            Self::U8(values) => mode_from_counts(count, &values[start..end], dimension),
            Self::U16(values) => mode_from_counts(count, &values[start..end], dimension),
            Self::U32(values) => mode_from_counts(count, &values[start..end], dimension),
            Self::U64(values) => mode_from_counts(count, &values[start..end], dimension),
        }
    }

    pub(super) fn cost(
        &self,
        start: usize,
        count: u64,
        centroid: &[u8],
        dimension: usize,
    ) -> Result<u64> {
        let end = start
            .checked_add(dimension.saturating_mul(8))
            .ok_or_else(|| {
                Error::invalid_input("Hamming counter range overflow while computing cost")
            })?;
        if end > self.len() {
            return Err(Error::invalid_input(format!(
                "Hamming counter cost range {start}..{end} exceeds length {}",
                self.len()
            )));
        }
        match self {
            Self::U8(values) => summary_cost(count, &values[start..end], centroid, dimension),
            Self::U16(values) => summary_cost(count, &values[start..end], centroid, dimension),
            Self::U32(values) => summary_cost(count, &values[start..end], centroid, dimension),
            Self::U64(values) => summary_cost(count, &values[start..end], centroid, dimension),
        }
    }

    #[cfg(test)]
    pub(super) fn to_u64_vec(&self) -> Vec<u64> {
        (0..self.len()).map(|index| self.value(index)).collect()
    }
}
