// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::encodings::physical::sequence::WindowCursor;

#[derive(Clone, Debug)]
pub(super) enum Validity {
    Metadata(SparseValiditySet),
    Sequence {
        meaning: SparseValidityMeaning,
        positions: Positions,
        domain: u64,
    },
    Selected {
        rows: Range<u64>,
        bitmap: Buffer,
        nulls: u64,
    },
}

impl Validity {
    pub(super) fn parse<'a>(
        descriptor: Option<&pb23::SparseValiditySet>,
        domain: u64,
        buffers: &mut impl Iterator<Item = &'a (u64, u64)>,
    ) -> Result<Self> {
        let descriptor = descriptor.ok_or_else(|| corrupt("missing parent validity"))?;
        let meaning = super::super::validity_meaning(descriptor, "admitted List validity")?;
        let encoding = descriptor
            .positions
            .as_ref()
            .ok_or_else(|| corrupt("missing validity positions"))?;
        let positions = Positions::parse(encoding, descriptor.num_positions, domain, buffers)?;
        Ok(match positions {
            Positions::Metadata(positions) => {
                Self::Metadata(SparseValiditySet { meaning, positions })
            }
            positions => Self::Sequence {
                meaning,
                positions,
                domain,
            },
        })
    }

    pub(super) fn null_count(&self, rows: &Range<u64>) -> Result<u64> {
        match self {
            Self::Metadata(set) => Ok(null_count(set, rows)),
            Self::Selected {
                rows: prepared,
                nulls,
                ..
            } if rows == prepared => Ok(*nulls),
            _ => Err(Error::internal("validity range was not prepared")),
        }
    }

    pub(super) fn is_valid(&self, row: u64) -> Result<bool> {
        match self {
            Self::Metadata(set) => Ok(set.positions.contains(row)
                != matches!(set.meaning, SparseValidityMeaning::NullPositions)),
            Self::Selected { rows, bitmap, .. } if rows.contains(&row) => {
                let index = (row - rows.start) as usize;
                Ok(bitmap[index / 8] & (1 << (index % 8)) != 0)
            }
            _ => Err(Error::internal("validity row was not prepared")),
        }
    }

    pub(super) async fn prepare<F, Fut>(
        &self,
        non_empty: &Positions,
        rows: Range<u64>,
        work: &mut Reservation,
        read: &mut F,
    ) -> Result<Option<(Positions, Self)>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        if let Self::Metadata(set) = self {
            return Ok(non_empty
                .prepare(rows, set, work, read)
                .await?
                .map(|p| (p, self.clone())));
        }
        let Self::Sequence {
            meaning,
            positions,
            domain,
        } = self
        else {
            return Err(Error::internal(
                "selected validity cannot prepare another range",
            ));
        };
        if rows.start > rows.end || rows.end > *domain {
            return Err(corrupt("validity selection exceeds domain"));
        }
        let bitmap_bytes = usize::try_from((rows.end - rows.start).div_ceil(8))
            .map_err(|_| corrupt("validity bitmap exceeds address space"))?;
        let capacity = non_empty.selected_capacity(&rows)?;
        let required = capacity
            .checked_add(bitmap_bytes)
            .and_then(|v| v.checked_add(non_empty.scratch_bytes()))
            .and_then(|v| v.checked_add(positions.scratch_bytes()));
        if required.is_none_or(|n| n > work.len()) {
            return Ok(None);
        }
        let mut values = work
            .try_allocate(capacity)?
            .ok_or_else(|| Error::internal("position merge preflight changed"))?;
        let mut bitmap = work
            .try_allocate(bitmap_bytes)?
            .ok_or_else(|| Error::internal("validity preflight changed"))?;
        let is_null_set = matches!(meaning, SparseValidityMeaning::NullPositions);
        bitmap.as_mut().fill(if is_null_set { 255 } else { 0 });
        let mut before = 0;
        let mut used = 0;
        let mut retain = |row: u64| -> Result<()> {
            if row < rows.start {
                before += 1;
            } else if row < rows.end && matches!(non_empty, Positions::Sequence { .. }) {
                let target = values
                    .as_mut()
                    .get_mut(used..used + 8)
                    .ok_or_else(|| Error::internal("selected positions underestimated"))?;
                target.copy_from_slice(&row.to_le_bytes());
                used += 8;
            }
            Ok(())
        };
        let mut parent = PositionCursor::new(non_empty, *domain)?;
        let mut validity = PositionCursor::new(positions, *domain)?;
        let mut next_parent = parent.next(work, read).await?;
        let mut matched = 0_u64;
        // Merge complete sorted domains. Checking only selected validity would miss
        // a null parent with child slots in an unselected prefix or suffix.
        while let Some(position) = validity.next(work, read).await? {
            while next_parent.is_some_and(|p| p < position) {
                let row = next_parent.ok_or_else(|| Error::internal("missing merge parent"))?;
                if !is_null_set {
                    return Err(corrupt("null List parent contains child slots"));
                }
                retain(row)?;
                next_parent = parent.next(work, read).await?;
            }
            if next_parent == Some(position) {
                if is_null_set {
                    return Err(corrupt("null List parent contains child slots"));
                }
                retain(position)?;
                next_parent = parent.next(work, read).await?;
            }
            if rows.contains(&position) {
                let index = (position - rows.start) as usize;
                if is_null_set {
                    bitmap.as_mut()[index / 8] &= !(1 << (index % 8));
                } else {
                    bitmap.as_mut()[index / 8] |= 1 << (index % 8);
                }
                matched += 1;
            }
        }
        while let Some(row) = next_parent {
            if !is_null_set {
                return Err(corrupt("null List parent contains child slots"));
            }
            retain(row)?;
            next_parent = parent.next(work, read).await?;
        }
        let non_empty = match non_empty {
            Positions::Metadata(_) => non_empty.clone(),
            _ => Positions::Selected {
                rows: rows.clone(),
                before,
                values: Buffer::from(values.freeze()).slice_with_length(0, used),
            },
        };
        let nulls = if is_null_set {
            matched
        } else {
            rows.end - rows.start - matched
        };
        Ok(Some((
            non_empty,
            Self::Selected {
                rows,
                bitmap: Buffer::from(bitmap.freeze()),
                nulls,
            },
        )))
    }
}

struct PositionCursor<'a> {
    positions: &'a Positions,
    sequence: Option<WindowCursor<'a>>,
    index: u64,
    previous: Option<u64>,
    domain: u64,
}

impl<'a> PositionCursor<'a> {
    fn new(positions: &'a Positions, domain: u64) -> Result<Self> {
        let sequence = match positions {
            Positions::Metadata(_) => None,
            Positions::Sequence { sequence, .. } => Some(WindowCursor::new(sequence)),
            Positions::Selected { .. } => {
                return Err(Error::internal(
                    "prepared positions cannot be streamed again",
                ));
            }
        };
        Ok(Self {
            positions,
            sequence,
            index: 0,
            previous: None,
            domain,
        })
    }

    async fn next<F, Fut>(&mut self, work: &mut Reservation, read: &mut F) -> Result<Option<u64>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let value = if let Some(sequence) = &mut self.sequence {
            sequence.next(work, read).await?
        } else if let Positions::Metadata(plan) = self.positions {
            if self.index >= plan.len() {
                return Ok(None);
            }
            if self.index > 0 && self.index.is_multiple_of(1024) {
                tokio::task::yield_now().await;
            }
            let value = if let Some((start, step, _)) = plan.progression() {
                start + step * self.index
            } else if let SparsePositionPlan::Explicit(values) = plan {
                values[self.index as usize]
            } else {
                return Err(Error::internal("invalid metadata position cursor"));
            };
            self.index += 1;
            Some(value)
        } else {
            return Err(Error::internal("missing position cursor"));
        };
        if let Some(value) = value {
            if value >= self.domain {
                return Err(corrupt("structural position exceeds List domain"));
            }
            if self.previous.is_some_and(|p| p >= value) {
                return Err(corrupt("structural positions must be strictly increasing"));
            }
            self.previous = Some(value);
        }
        Ok(value)
    }
}
