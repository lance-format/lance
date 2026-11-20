// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    encodings::physical::sequence::{SequenceDecoder, SequenceMetadata, WindowSequence},
    format::metadata::CompressiveEncoding,
};

mod dictionary;
mod rle;
use dictionary::DictionaryCounts;
use rle::RleCounts;

#[derive(Clone, Debug)]
pub(super) enum Counts {
    Metadata { first: u64, step: u64, len: u64 },
    Sequence(WindowSequence),
    Rle(Box<RleCounts>),
    Dictionary(Box<DictionaryCounts>),
    Selected { start: u64, prefixes: Buffer },
}

impl Counts {
    pub(super) fn parse<'a>(
        encoding: Option<&CompressiveEncoding>,
        len: u64,
        buffers: &mut impl Iterator<Item = &'a (u64, u64)>,
    ) -> Result<Self> {
        let encoding = encoding.ok_or_else(|| corrupt("missing unsigned counts descriptor"))?;
        let sequence = SequenceDecoder::try_new(encoding, 64, len)?;
        if matches!(
            encoding.compression,
            Some(crate::format::metadata::compressive_encoding::Compression::Dictionary(_))
        ) {
            return Ok(Self::Dictionary(Box::new(DictionaryCounts::parse(
                encoding,
                len,
                sequence.requires_payload(),
                buffers,
            )?)));
        }
        if matches!(
            encoding.compression,
            Some(crate::format::metadata::compressive_encoding::Compression::Rle(_))
        ) {
            return Ok(Self::Rle(Box::new(RleCounts::parse(
                encoding, len, buffers,
            )?)));
        }
        let (first, step) = match sequence.metadata().filter(|_| !sequence.requires_payload()) {
            Some(SequenceMetadata::Empty) => (0, 0),
            Some(SequenceMetadata::Constant(value)) => (value, 0),
            Some(SequenceMetadata::Range { start, step }) => (start, step),
            None => {
                let payload = if sequence.requires_payload() {
                    let &(start, size) = buffers
                        .next()
                        .ok_or_else(|| corrupt("missing counts buffer"))?;
                    let end = start
                        .checked_add(size)
                        .ok_or_else(|| corrupt("counts buffer range overflows"))?;
                    start..end
                } else {
                    0..0
                };
                return Ok(Self::Sequence(WindowSequence::try_new(
                    encoding, len, payload,
                )?));
            }
        };
        if len > 0 && first == 0 {
            return Err(corrupt("nonempty List counts must be positive"));
        }
        let counts = Self::Metadata { first, step, len };
        counts.prefix(len)?;
        Ok(counts)
    }

    pub(super) fn metadata_total(&self) -> Result<Option<u64>> {
        match self {
            Self::Metadata { len, .. } => self.prefix(*len).map(Some),
            _ => Ok(None),
        }
    }

    pub(super) fn prefix(&self, count: u64) -> Result<u64> {
        match self {
            Self::Metadata { first, step, len } => {
                if count > *len {
                    return Err(corrupt("count prefix exceeds descriptor cardinality"));
                }
                if count == 0 {
                    return Ok(0);
                }
                // Divide the even factor before multiplying to keep u64 sums in u128.
                let endpoints = u128::from(*first) * 2 + u128::from(*step) * u128::from(count - 1);
                let sum = if count.is_multiple_of(2) {
                    u128::from(count / 2) * endpoints
                } else {
                    u128::from(count) * (endpoints / 2)
                };
                u64::try_from(sum).map_err(|_| corrupt("List count sum overflows u64"))
            }
            Self::Selected { start, prefixes } => {
                let byte = count
                    .checked_sub(*start)
                    .and_then(|n| n.checked_mul(8))
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| corrupt("count prefix outside prepared selection"))?;
                let bytes = prefixes
                    .get(byte..)
                    .and_then(|b| b.first_chunk::<8>())
                    .ok_or_else(|| corrupt("count prefix outside prepared selection"))?;
                Ok(u64::from_le_bytes(*bytes))
            }
            Self::Sequence(_) | Self::Rle(_) | Self::Dictionary(_) => {
                Err(Error::internal("count sequence was not prepared"))
            }
        }
    }

    pub(super) async fn prepare<F, Fut>(
        &self,
        selected: Range<u64>,
        total: u64,
        work: &mut Reservation,
        read: &mut F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        if let Self::Rle(rle) = self {
            return rle.prepare(selected, total, work, read).await;
        }
        if let Self::Dictionary(dictionary) = self {
            return dictionary.prepare(selected, total, work, read).await;
        }
        let Self::Sequence(sequence) = self else {
            return Ok(Some(self.clone()));
        };
        let bytes = selected
            .end
            .checked_sub(selected.start)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| corrupt("selected count prefixes exceed address space"))?;
        if bytes
            .checked_add(sequence.scratch_bytes())
            .is_none_or(|needed| needed > work.len())
        {
            return Ok(None);
        }
        let mut prefixes = work
            .try_allocate(bytes)?
            .ok_or_else(|| Error::internal("count prefix preflight changed"))?;
        let mut index = 0_u64;
        let mut sum = 0_u64;
        let mut visit = |count: u64| {
            if (selected.start..=selected.end).contains(&index) {
                let offset = ((index - selected.start) * 8) as usize;
                prefixes.as_mut()[offset..offset + 8].copy_from_slice(&sum.to_le_bytes());
            }
            if count == 0 {
                return Err(corrupt("nonempty List counts must be positive"));
            }
            sum = sum
                .checked_add(count)
                .ok_or_else(|| corrupt("List count sum overflows u64"))?;
            index += 1;
            Ok(())
        };
        if sequence.visit(work, read, &mut visit).await?.is_none() {
            return Err(Error::internal("count scratch preflight changed"));
        }
        if sum != total || selected.end > index {
            return Err(corrupt("List count sum differs from child domain"));
        }
        if selected.end == index {
            prefixes.as_mut()[bytes - 8..].copy_from_slice(&sum.to_le_bytes());
        }
        Ok(Some(Self::Selected {
            start: selected.start,
            prefixes: Buffer::from(prefixes.freeze()),
        }))
    }
}
