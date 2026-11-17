// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Two bounded passes: validate all offsets, then copy selected value ranges.

use super::*;

// Dictionary index, output row/destination, start and end. Sorting these charged
// records avoids either a whole dictionary table or one decompression per row.
const RECORD: usize = 32;

impl Dictionary {
    pub(super) async fn expand_lz4<F, Fut>(
        &self,
        indices: &FixedWidthDataBlock,
        output_bits: u8,
        work: &mut Reservation,
        read: &mut F,
    ) -> Result<Option<DataBlock>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let Some(plan) = lz4::Plan::read(&self.bytes, work, read).await? else {
            return Ok(None);
        };
        let values_start = usize::try_from((self.count + 3) * self.width as u64)
            .map_err(|_| corrupt("dictionary header exceeds usize"))?;
        if values_start > plan.decoded {
            return Err(corrupt("dictionary count/header exceeds LZ4 decoded size"));
        }
        let count = indices.num_values as usize; // checked by expand before dispatch
        let records_bytes = count
            .checked_mul(RECORD)
            .ok_or_else(|| corrupt("dictionary selected records overflow"))?;
        if records_bytes
            .checked_add(plan.scratch().max(4))
            .is_none_or(|n| n > work.len())
        {
            return Ok(None);
        }
        let mut records = work
            .try_allocate(records_bytes)?
            .ok_or_else(|| Error::internal("dictionary record preflight changed"))?;
        for (row, (index, record)) in indices
            .data
            .as_ref()
            .chunks_exact((indices.bits_per_value / 8) as usize)
            .zip(records.as_mut().as_chunks_mut::<RECORD>().0)
            .enumerate()
        {
            record[..8].copy_from_slice(&word(index).to_le_bytes());
            record[8..16].copy_from_slice(&(row as u64).to_le_bytes());
        }
        records
            .as_mut()
            .as_chunks_mut::<RECORD>()
            .0
            .sort_unstable_by_key(|r| word(&r[..8]));
        let mut offsets = Offsets::new(self.width, values_start, plan.decoded);
        plan.visit(&self.bytes, work, read, |at, chunk| {
            offsets.visit(
                at,
                chunk,
                records.as_mut().as_chunks_mut::<RECORD>().0,
                false,
            )
        })
        .await?;

        let mut total = 0_u64;
        for record in records.as_ref().as_chunks::<RECORD>().0 {
            total = total
                .checked_add(word(&record[24..]) - word(&record[16..24]))
                .ok_or_else(|| corrupt("dictionary expansion overflows"))?;
        }
        if total
            > if output_bits == 32 {
                i32::MAX as u64
            } else {
                i64::MAX as u64
            }
        {
            records.recycle(work)?;
            return Ok(None);
        }
        let output_width = usize::from(output_bits / 8);
        let offsets_bytes = count
            .checked_add(1)
            .and_then(|n| n.checked_mul(output_width))
            .ok_or_else(|| corrupt("dictionary output offsets overflow"))?;
        let output_bytes = usize::try_from(total)
            .ok()
            .and_then(|n| n.checked_add(offsets_bytes))
            .ok_or_else(|| corrupt("dictionary output size overflows"))?;
        let scratch = if total == 0 { 0 } else { plan.scratch().max(4) };
        if output_bytes
            .checked_add(scratch)
            .is_none_or(|n| n > work.len())
        {
            records.recycle(work)?;
            return Ok(None);
        }
        let mut output = work
            .try_allocate(output_bytes)?
            .ok_or_else(|| Error::internal("dictionary output preflight changed"))?;
        // Write lengths in caller order, prefix-sum in place, then attach each
        // sorted record's destination. No second routing allocation is needed.
        for record in records.as_ref().as_chunks::<RECORD>().0 {
            let row = word(&record[8..16]) as usize;
            let length = word(&record[24..]) - word(&record[16..24]);
            output.as_mut()[(row + 1) * output_width..(row + 2) * output_width]
                .copy_from_slice(&length.to_le_bytes()[..output_width]);
        }
        let mut prefix = 0_u64;
        for slot in output.as_mut()[output_width..offsets_bytes].chunks_exact_mut(output_width) {
            prefix += word(slot);
            slot.copy_from_slice(&prefix.to_le_bytes()[..output_width]);
        }
        for record in records.as_mut().as_chunks_mut::<RECORD>().0 {
            let row = word(&record[8..16]) as usize;
            let destination = word(&output.as_ref()[row * output_width..(row + 1) * output_width]);
            record[8..16].copy_from_slice(&destination.to_le_bytes());
        }
        if total != 0 {
            let mut offsets = Offsets::new(self.width, values_start, plan.decoded);
            let mut first = 0;
            plan.visit(&self.bytes, work, read, |at, chunk| {
                let records = records.as_mut().as_chunks_mut::<RECORD>().0;
                // Validate selected pairs again before copying any values. A
                // changed custom reader must not redirect copies using old offsets.
                offsets.visit(at, chunk, records, true)?;
                let start = at.max(values_start);
                let end = at + chunk.len();
                if start >= end {
                    return Ok(());
                }
                while first < records.len()
                    && word(&records[first][24..]) <= (start - values_start) as u64
                {
                    first += 1;
                }
                for record in &records[first..] {
                    let from = values_start + word(&record[16..24]) as usize;
                    let to = values_start + word(&record[24..]) as usize;
                    if from >= end {
                        break;
                    }
                    let copy_start = from.max(start);
                    let copy_end = to.min(end);
                    if copy_start < copy_end {
                        let destination =
                            offsets_bytes + word(&record[8..16]) as usize + copy_start - from;
                        output.as_mut()[destination..destination + copy_end - copy_start]
                            .copy_from_slice(&chunk[copy_start - at..copy_end - at]);
                    }
                }
                Ok(())
            })
            .await?;
        }
        records.recycle(work)?;
        let bytes = output.freeze();
        Ok(Some(DataBlock::VariableWidth(VariableWidthBlock {
            data: LanceBuffer::from_bytes(bytes.slice(offsets_bytes..), 1),
            offsets: LanceBuffer::from_bytes(bytes.slice(..offsets_bytes), output_width as u64),
            bits_per_offset: output_bits,
            num_values: indices.num_values,
            block_info: BlockInfo::new(),
        })))
    }
}

struct Offsets {
    width: usize,
    values_start: usize,
    decoded: usize,
    previous: u64,
    starts: usize,
    ends: usize,
}

impl Offsets {
    fn new(width: usize, values_start: usize, decoded: usize) -> Self {
        Self {
            width,
            values_start,
            decoded,
            previous: 0,
            starts: 0,
            ends: 0,
        }
    }

    fn visit(
        &mut self,
        at: usize,
        chunk: &[u8],
        records: &mut [[u8; RECORD]],
        verify: bool,
    ) -> Result<()> {
        if at >= self.values_start {
            return Ok(());
        }
        let len = chunk.len().min(self.values_start - at);
        // The visitor emits aligned 4 KiB spans except for its final tail; the
        // checked header must fit entirely before that tail's value payload.
        if !at.is_multiple_of(self.width) || !len.is_multiple_of(self.width) {
            return Err(corrupt("LZ4 dictionary offset window is truncated"));
        }
        for (i, bytes) in chunk[..len].chunks_exact(self.width).enumerate() {
            let number = at / self.width + i;
            let value = word(bytes);
            if number < 2 {
                let expected = if number == 0 {
                    (self.width * 8) as u64
                } else {
                    self.values_start as u64
                };
                if value != expected {
                    return Err(corrupt(
                        "dictionary header disagrees with offset width/count",
                    ));
                }
                continue;
            }
            let index = (number - 2) as u64;
            if (index == 0 && value != 0)
                || value < self.previous
                || value > (self.decoded - self.values_start) as u64
            {
                return Err(corrupt(
                    "dictionary offsets must start at zero and remain monotonic within values",
                ));
            }
            self.previous = value;
            while self.starts < records.len() && word(&records[self.starts][..8]) == index {
                Self::set(&mut records[self.starts][16..24], value, verify)?;
                self.starts += 1;
            }
            while self.ends < records.len() && word(&records[self.ends][..8]) + 1 == index {
                Self::set(&mut records[self.ends][24..], value, verify)?;
                self.ends += 1;
            }
        }
        Ok(())
    }

    fn set(slot: &mut [u8], value: u64, verify: bool) -> Result<()> {
        if verify && word(slot) != value {
            return Err(corrupt(
                "selected dictionary offsets changed between LZ4 passes",
            ));
        }
        slot.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }
}
