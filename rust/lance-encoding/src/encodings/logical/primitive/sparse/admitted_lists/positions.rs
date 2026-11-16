// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{encodings::physical::sequence::WindowSequence, format::metadata::CompressiveEncoding};

#[derive(Clone, Debug)]
pub(super) enum Positions {
    Metadata(SparsePositionPlan),
    Sequence {
        sequence: WindowSequence,
        domain: u64,
        len: u64,
    },
    Selected {
        rows: Range<u64>,
        before: u64,
        values: Buffer,
    },
}

impl Positions {
    pub(super) fn parse<'a>(
        encoding: &CompressiveEncoding,
        len: u64,
        domain: u64,
        buffers: &mut impl Iterator<Item = &'a (u64, u64)>,
    ) -> Result<Self> {
        match position_decoder(encoding, len, domain, "admitted List nonempty")? {
            SparsePositionDecoder::Ready(plan) => Ok(Self::Metadata(plan)),
            SparsePositionDecoder::Encoded { sequence, .. } => {
                // Encoded includes transforms of metadata, such as Delta(Range).
                // Only the decoder's payload arity determines buffer consumption.
                let payload = if sequence.requires_payload() {
                    let &(start, size) = buffers
                        .next()
                        .ok_or_else(|| corrupt("missing nonempty positions buffer"))?;
                    let end = start
                        .checked_add(size)
                        .ok_or_else(|| corrupt("nonempty positions buffer range overflows"))?;
                    start..end
                } else {
                    0..0
                };
                Ok(Self::Sequence {
                    sequence: WindowSequence::try_new(encoding, len, payload)?,
                    domain,
                    len,
                })
            }
        }
    }

    pub(super) fn rank(&self, row: u64) -> Result<u64> {
        match self {
            Self::Metadata(plan) => Ok(plan.rank(row)),
            Self::Selected {
                rows,
                before,
                values,
            } => {
                if row < rows.start || row > rows.end {
                    return Err(corrupt("position rank outside prepared selection"));
                }
                // The selected buffer stores little-endian words. Avoid an aligned
                // copy or a per-row index; binary search the admitted bytes directly.
                let mut low = 0;
                let mut high = values.len() / 8;
                while low < high {
                    let mid = low + (high - low) / 2;
                    let bytes = values[mid * 8..]
                        .first_chunk::<8>()
                        .ok_or_else(|| Error::internal("invalid prepared position width"))?;
                    if u64::from_le_bytes(*bytes) < row {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                Ok(before + low as u64)
            }
            Self::Sequence { .. } => Err(Error::internal("position sequence was not prepared")),
        }
    }

    pub(super) async fn prepare<F, Fut>(
        &self,
        rows: Range<u64>,
        validity: &SparseValiditySet,
        work: &mut Reservation,
        read: &mut F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let Self::Sequence {
            sequence,
            domain,
            len,
        } = self
        else {
            return Ok(Some(self.clone()));
        };
        if rows.start > rows.end || rows.end > *domain {
            return Err(corrupt("position selection outside List domain"));
        }
        // One pass avoids a second read whose contents could disagree with the
        // sizing pass. Unused capacity remains charged through the buffer owner.
        let capacity = (rows.end - rows.start)
            .min(*len)
            .checked_mul(8)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| corrupt("selected positions exceed address space"))?;
        if capacity
            .checked_add(sequence.scratch_bytes())
            .is_none_or(|needed| needed > work.len())
        {
            return Ok(None);
        }
        let mut values = work
            .try_allocate(capacity)?
            .ok_or_else(|| Error::internal("position preflight changed"))?;
        let mut previous = None;
        let mut before = 0_u64;
        let mut used = 0;
        let is_null_set = matches!(validity.meaning, SparseValidityMeaning::NullPositions);
        let mut visit = |position| {
            if position >= *domain {
                return Err(corrupt("nonempty position exceeds List domain"));
            }
            if previous.is_some_and(|p| p >= position) {
                return Err(corrupt("nonempty positions must be strictly increasing"));
            }
            previous = Some(position);
            if validity.positions.contains(position) == is_null_set {
                return Err(corrupt("null List parent contains child slots"));
            }
            if position < rows.start {
                before += 1;
            } else if position < rows.end {
                let target = values
                    .as_mut()
                    .get_mut(used..used + 8)
                    .ok_or_else(|| Error::internal("selected position capacity underestimated"))?;
                target.copy_from_slice(&position.to_le_bytes());
                used += 8;
            }
            Ok(())
        };
        // Validate the entire domain, even when no selected parent is nonempty.
        if sequence.visit(work, read, &mut visit).await?.is_none() {
            return Err(Error::internal("position scratch preflight changed"));
        }
        Ok(Some(Self::Selected {
            rows,
            before,
            values: Buffer::from(values.freeze()).slice_with_length(0, used),
        }))
    }
}
