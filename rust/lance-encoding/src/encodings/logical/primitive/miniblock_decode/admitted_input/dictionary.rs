// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read selected raw dictionary values without retaining the complete dictionary.

use super::*;
use crate::{
    data::{BlockInfo, DataBlock, FixedWidthDataBlock, VariableWidthBlock},
    format::metadata::{CompressiveEncoding, compressive_encoding::Compression},
};

// Bounds both offset traversal and selected value-copy scratch per read.
const WINDOW: usize = 4096;

#[derive(Debug)]
pub(super) struct Dictionary {
    bytes: Range<u64>,
    count: u64,
    width: usize,
}

fn word(bytes: &[u8]) -> u64 {
    let mut value = [0; 8];
    value[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(value)
}

impl Dictionary {
    pub(super) fn try_new(
        encoding: &CompressiveEncoding,
        bytes: Range<u64>,
        count: u64,
    ) -> Result<Self> {
        let Some(Compression::Variable(variable)) = &encoding.compression else {
            return Err(Error::not_supported(
                "admitted dictionary requires raw variable-width values",
            ));
        };
        let Some(Compression::Flat(offsets)) = variable
            .offsets
            .as_ref()
            .and_then(|e| e.compression.as_ref())
        else {
            return Err(Error::not_supported(
                "admitted dictionary requires raw Flat offsets",
            ));
        };
        if variable.values.is_some()
            || offsets.data.is_some()
            || !matches!(offsets.bits_per_value, 32 | 64)
        {
            return Err(Error::not_supported(
                "admitted dictionary requires uncompressed 32/64-bit offsets and values",
            ));
        }
        let width = (offsets.bits_per_value / 8) as usize;
        let minimum = count
            .checked_add(3)
            .and_then(|n| n.checked_mul(width as u64));
        if count == 0 || minimum.is_none_or(|n| n > bytes.end - bytes.start) {
            return Err(corrupt("dictionary count/header exceeds its buffer"));
        }
        Ok(Self {
            bytes,
            count,
            width,
        })
    }

    async fn read<F, Fut>(
        range: Range<u64>,
        work: &mut Reservation,
        read: &mut F,
    ) -> Result<ReservedBuffer>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let len = usize::try_from(range.end - range.start)
            .map_err(|_| corrupt("dictionary read length overflows"))?;
        let out = work
            .try_allocate(len)?
            .ok_or_else(|| Error::internal("dictionary scratch preflight changed"))?;
        let out = read(range, out).await?;
        if out.as_ref().len() != len {
            return Err(corrupt("dictionary read length mismatch"));
        }
        Ok(out)
    }

    // Validate the entire offset domain, even when the selected indices omit its
    // corrupt suffix. The dictionary value payload itself is never read in full.
    async fn validate<F, Fut>(&self, work: &mut Reservation, read: &mut F) -> Result<u64>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let header_len = (2 * self.width) as u64;
        let values_start = self.bytes.start + (self.count + 3) * self.width as u64;
        let header =
            Self::read(self.bytes.start..self.bytes.start + header_len, work, read).await?;
        let bits = word(&header.as_ref()[..self.width]);
        let offset = word(&header.as_ref()[self.width..]);
        header.recycle(work)?;
        if bits != (self.width * 8) as u64 || offset != values_start - self.bytes.start {
            return Err(corrupt(
                "dictionary header disagrees with offset width/count",
            ));
        }
        let mut at = self.bytes.start + header_len;
        let mut previous = 0;
        let mut first = true;
        while at < values_start {
            let end = at + (values_start - at).min(WINDOW as u64);
            let scratch = Self::read(at..end, work, read).await?;
            let result = scratch.as_ref().chunks_exact(self.width).try_for_each(|bytes| {
                let value = word(bytes);
                if value < previous || value > self.bytes.end - values_start || (first && value != 0) {
                    return Err(corrupt("dictionary offsets must start at zero and remain monotonic within values"));
                }
                first = false;
                previous = value;
                Ok(())
            });
            scratch.recycle(work)?;
            result?;
            at = end;
            tokio::task::yield_now().await;
        }
        Ok(values_start)
    }

    pub(super) async fn expand<F, Fut>(
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
        if !matches!(indices.bits_per_value, 8 | 16 | 32 | 64) || !matches!(output_bits, 32 | 64) {
            return Err(corrupt("dictionary indices/output width is unsupported"));
        }
        let index_width = (indices.bits_per_value / 8) as usize;
        let count = usize::try_from(indices.num_values)
            .map_err(|_| corrupt("dictionary selection count exceeds usize"))?;
        if count.checked_mul(index_width) != Some(indices.data.len()) {
            return Err(corrupt(
                "dictionary index buffer length disagrees with count",
            ));
        }
        for value in indices.data.as_ref().chunks_exact(index_width) {
            if word(value) >= self.count {
                return Err(corrupt("dictionary index exceeds entry count"));
            }
        }
        let ranges_bytes = count
            .checked_mul(16)
            .ok_or_else(|| corrupt("dictionary selected ranges overflow"))?;
        let metadata_scratch = ((self.count + 1) * self.width as u64).min(WINDOW as u64) as usize;
        let scratch_bytes = metadata_scratch.max(2 * self.width);
        if ranges_bytes
            .checked_add(scratch_bytes)
            .is_none_or(|n| n > work.len())
        {
            return Ok(None);
        }
        let values_start = self.validate(work, read).await?;
        let mut ranges = work
            .try_allocate(ranges_bytes)?
            .ok_or_else(|| Error::internal("dictionary ranges preflight changed"))?;
        let mut total = 0_u64;
        let mut max_value = 0_u64;
        for (index, range) in indices
            .data
            .as_ref()
            .chunks_exact(index_width)
            .zip(ranges.as_mut().chunks_exact_mut(16))
        {
            let offset = self.bytes.start + (word(index) + 2) * self.width as u64;
            let pair = Self::read(offset..offset + (2 * self.width) as u64, work, read).await?;
            let start = word(&pair.as_ref()[..self.width]);
            let end = word(&pair.as_ref()[self.width..]);
            pair.recycle(work)?;
            // Revalidate selected pairs, including when a custom reader's bytes
            // differ between the domain validation and selected offset requests.
            if start > end || end > self.bytes.end - values_start {
                return Err(corrupt("selected dictionary offset pair exceeds values"));
            }
            total = total
                .checked_add(end - start)
                .ok_or_else(|| corrupt("dictionary expansion overflows"))?;
            max_value = max_value.max(end - start);
            range[..8].copy_from_slice(&start.to_le_bytes());
            range[8..].copy_from_slice(&end.to_le_bytes());
        }
        if total
            > if output_bits == 32 {
                i32::MAX as u64
            } else {
                i64::MAX as u64
            }
        {
            // A valid repeated selection can outgrow one Arrow array even when
            // the stored dictionary fits. Let the caller reduce its row prefix.
            ranges.recycle(work)?;
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
        let scratch_bytes = max_value.min(WINDOW as u64) as usize;
        if output_bytes
            .checked_add(scratch_bytes)
            .is_none_or(|n| n > work.len())
        {
            ranges.recycle(work)?;
            return Ok(None);
        }
        let mut output = work
            .try_allocate(output_bytes)?
            .ok_or_else(|| Error::internal("dictionary output preflight changed"))?;
        let mut written = 0_usize;
        for (i, pair) in ranges.as_ref().as_chunks::<16>().0.iter().enumerate() {
            let mut at = values_start + word(&pair[..8]);
            let end = values_start + word(&pair[8..]);
            while at < end {
                let len = (end - at).min(WINDOW as u64) as usize;
                let scratch = Self::read(at..at + len as u64, work, read).await?;
                output.as_mut()[offsets_bytes + written..offsets_bytes + written + len]
                    .copy_from_slice(scratch.as_ref());
                scratch.recycle(work)?;
                written += len;
                at += len as u64;
            }
            output.as_mut()[(i + 1) * output_width..(i + 2) * output_width]
                .copy_from_slice(&(written as u64).to_le_bytes()[..output_width]);
        }
        ranges.recycle(work)?;
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

#[cfg(test)]
mod tests;
