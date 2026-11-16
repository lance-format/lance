// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    compression::create_fixed_width_block_decompressor,
    encodings::physical::{
        rle::{BlockRleRunCount, infer_generic_run_count_from_sizes},
        sequence::WindowCursor,
    },
    format::metadata::compressive_encoding::Compression,
};

/// Frame metadata only. Child windows and retained prefixes use the caller's
/// original reservation; neither compressed nor expanded whole pages are held.
#[derive(Clone, Debug)]
pub struct RleCounts {
    values: CompressiveEncoding,
    lengths: CompressiveEncoding,
    values_payload: bool,
    lengths_payload: bool,
    source: BlockRleRunCount,
    payload: Range<u64>,
    count: u64,
}

impl RleCounts {
    pub(super) fn parse<'a>(
        encoding: &CompressiveEncoding,
        count: u64,
        buffers: &mut impl Iterator<Item = &'a (u64, u64)>,
    ) -> Result<Self> {
        // Counts::parse has already validated the current generic RLE grammar.
        let Some(Compression::Rle(rle)) = &encoding.compression else {
            return Err(Error::internal("expected RLE counts"));
        };
        let values = rle
            .values
            .as_deref()
            .ok_or_else(|| corrupt("missing RLE values"))?;
        let lengths = rle
            .run_lengths
            .as_deref()
            .ok_or_else(|| corrupt("missing RLE lengths"))?;
        let source = match lengths.compression.as_ref() {
            Some(Compression::Constant(c)) => {
                let value = c
                    .value
                    .as_ref()
                    .and_then(|v| v.first_chunk::<4>())
                    .ok_or_else(|| corrupt("invalid RLE constant length"))?;
                BlockRleRunCount::ConstantRunLength(u64::from(u32::from_le_bytes(*value)))
            }
            Some(Compression::Range(r)) => BlockRleRunCount::RangeRunLengths {
                start: r.start,
                step: r.step,
            },
            _ if matches!(&values.compression, Some(Compression::Flat(_))) => {
                BlockRleRunCount::FlatValues
            }
            _ if matches!(&lengths.compression, Some(Compression::Flat(_))) => {
                BlockRleRunCount::FlatRunLengths
            }
            _ => {
                return Err(Error::not_supported(
                    "RLE counts need an addressable cardinality child",
                ));
            }
        };
        let values_payload = create_fixed_width_block_decompressor(values, 64)?.requires_payload();
        let lengths_payload =
            create_fixed_width_block_decompressor(lengths, 32)?.requires_payload();
        let payload = if values_payload || lengths_payload {
            let &(start, bytes) = buffers
                .next()
                .ok_or_else(|| corrupt("missing RLE counts buffer"))?;
            let end = start
                .checked_add(bytes)
                .ok_or_else(|| corrupt("RLE frame range overflows"))?;
            if bytes < 8 {
                return Err(corrupt("RLE frame is shorter than its header"));
            }
            start..end
        } else {
            0..0
        };
        Ok(Self {
            values: values.clone(),
            lengths: lengths.clone(),
            values_payload,
            lengths_payload,
            source,
            payload,
            count,
        })
    }

    pub(super) async fn prepare<F, Fut>(
        &self,
        selected: Range<u64>,
        total: u64,
        work: &mut Reservation,
        read: &mut F,
    ) -> Result<Option<Counts>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        if selected.start > selected.end || selected.end > self.count {
            return Err(corrupt("RLE count selection exceeds cardinality"));
        }
        let (values_range, lengths_range) = if self.payload.is_empty() {
            (0..0, 0..0)
        } else {
            let Some(header) = work.try_allocate(8)? else {
                return Ok(None);
            };
            let header_end = self.payload.start + 8; // frame length checked at parse
            let header = read(self.payload.start..header_end, header).await?;
            let value_bytes = header
                .as_ref()
                .first_chunk::<8>()
                .filter(|_| header.as_ref().len() == 8)
                .ok_or_else(|| corrupt("short RLE header read"))?;
            let split = header_end
                .checked_add(u64::from_le_bytes(*value_bytes))
                .filter(|&end| end <= self.payload.end)
                .ok_or_else(|| corrupt("RLE value frame exceeds payload"))?;
            header.recycle(work)?;
            (header_end..split, split..self.payload.end)
        };
        let runs = infer_generic_run_count_from_sizes(
            self.source,
            self.values_payload
                .then_some(values_range.end - values_range.start),
            self.lengths_payload
                .then_some(lengths_range.end - lengths_range.start),
            64,
            self.count,
        )?;
        let values = WindowSequence::try_new(&self.values, runs, values_range)?;
        let lengths = WindowSequence::try_new_with_bits(&self.lengths, 32, runs, lengths_range)?;
        let bytes = (selected.end - selected.start)
            .checked_add(1)
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| corrupt("RLE selected prefixes exceed address space"))?;
        if bytes
            .checked_add(values.scratch_bytes())
            .and_then(|n| n.checked_add(lengths.scratch_bytes()))
            .is_none_or(|n| n > work.len())
        {
            return Ok(None);
        }
        let mut prefixes = work
            .try_allocate(bytes)?
            .ok_or_else(|| Error::internal("RLE prefix preflight changed"))?;
        let mut values = WindowCursor::new(&values);
        let mut lengths = WindowCursor::new(&lengths);
        let mut index = 0_u64;
        let mut sum = 0_u64;
        for run in 0..runs {
            let value = values
                .next(work, read)
                .await?
                .ok_or_else(|| corrupt("missing RLE run value"))?;
            let length = lengths
                .next(work, read)
                .await?
                .ok_or_else(|| corrupt("missing RLE run length"))?;
            if value == 0 || length == 0 {
                return Err(corrupt("RLE List counts and run lengths must be positive"));
            }
            let end = index
                .checked_add(length)
                .filter(|&v| v <= self.count)
                .ok_or_else(|| corrupt("RLE run length sum exceeds count cardinality"))?;
            let next_sum = value
                .checked_mul(length)
                .and_then(|v| sum.checked_add(v))
                .ok_or_else(|| corrupt("RLE List count sum overflows"))?;
            // Only selected prefixes are expanded. Entire runs outside selection
            // still participate in count/child-domain validation.
            for row in index.max(selected.start)..=end.min(selected.end) {
                let prefix = sum + (row - index) * value; // bounded by checked next_sum
                let offset = ((row - selected.start) * 8) as usize;
                prefixes.as_mut()[offset..offset + 8].copy_from_slice(&prefix.to_le_bytes());
                if (row - selected.start + 1).is_multiple_of(1024) {
                    tokio::task::yield_now().await;
                }
            }
            index = end;
            sum = next_sum;
            if (run + 1).is_multiple_of(1024) {
                tokio::task::yield_now().await;
            }
        }
        // Drain both EOFs to validate framing and recycle their final windows.
        let value_end = values.next(work, read).await?;
        let length_end = lengths.next(work, read).await?;
        if value_end.is_some() || length_end.is_some() || index != self.count || sum != total {
            return Err(corrupt("RLE counts disagree with count or child domain"));
        }
        Ok(Some(Counts::Selected {
            start: selected.start,
            prefixes: Buffer::from(prefixes.freeze()),
        }))
    }
}
