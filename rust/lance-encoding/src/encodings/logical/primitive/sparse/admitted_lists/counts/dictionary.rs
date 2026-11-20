// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::format::metadata::compressive_encoding::Compression;

/// The current generic codec bounds the lookup table to 4096 items. The table,
/// selected prefixes and input scratch still share the original reservation.
#[derive(Clone, Debug)]
pub struct DictionaryCounts {
    indices: CompressiveEncoding,
    items: CompressiveEncoding,
    item_count: u64,
    count: u64,
    payload: Range<u64>,
    indices_have_payload: bool,
    items_have_payload: bool,
}

impl DictionaryCounts {
    pub(super) fn parse<'a>(
        encoding: &CompressiveEncoding,
        count: u64,
        has_payload: bool,
        buffers: &mut impl Iterator<Item = &'a (u64, u64)>,
    ) -> Result<Self> {
        // Counts::parse validates the generic grammar and dictionary size first.
        let Some(Compression::Dictionary(dictionary)) = &encoding.compression else {
            return Err(Error::internal("expected Dictionary counts"));
        };
        let indices = dictionary
            .indices
            .as_deref()
            .ok_or_else(|| corrupt("missing Dictionary indices"))?;
        let items = dictionary
            .items
            .as_deref()
            .ok_or_else(|| corrupt("missing Dictionary items"))?;
        let payload = if has_payload {
            let &(start, bytes) = buffers
                .next()
                .ok_or_else(|| corrupt("missing Dictionary counts buffer"))?;
            let end = start
                .checked_add(bytes)
                .ok_or_else(|| corrupt("Dictionary frame range overflows"))?;
            if bytes < 16 {
                return Err(corrupt("Dictionary frame is shorter than its header"));
            }
            start..end
        } else {
            0..0
        };
        Ok(Self {
            indices: indices.clone(),
            items: items.clone(),
            item_count: u64::from(dictionary.num_dictionary_items),
            count,
            payload,
            indices_have_payload: SequenceDecoder::try_new(indices, 32, count)?.requires_payload(),
            items_have_payload: SequenceDecoder::try_new(
                items,
                64,
                u64::from(dictionary.num_dictionary_items),
            )?
            .requires_payload(),
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
            return Err(corrupt("Dictionary count selection exceeds cardinality"));
        }
        let (index_range, item_range) = if self.payload.is_empty() {
            (0..0, 0..0)
        } else {
            let Some(header) = work.try_allocate(16)? else {
                return Ok(None);
            };
            let start = self.payload.start + 16; // checked frame length at parse
            let header = read(self.payload.start..start, header).await?;
            let words = header
                .as_ref()
                .first_chunk::<16>()
                .filter(|_| header.as_ref().len() == 16)
                .ok_or_else(|| corrupt("short Dictionary header read"))?;
            let [indices_bytes, items_bytes] = words.as_chunks::<8>().0 else {
                return Err(Error::internal("invalid Dictionary header width"));
            };
            let split = start
                .checked_add(u64::from_le_bytes(*indices_bytes))
                .ok_or_else(|| corrupt("Dictionary index frame overflows"))?;
            let end = split
                .checked_add(u64::from_le_bytes(*items_bytes))
                .filter(|&end| end == self.payload.end)
                .ok_or_else(|| corrupt("Dictionary frame lengths disagree with payload"))?;
            header.recycle(work)?;
            (start..split, split..end)
        };
        if (!self.indices_have_payload && !index_range.is_empty())
            || (!self.items_have_payload && !item_range.is_empty())
        {
            return Err(corrupt(
                "metadata-only Dictionary child has framed payload bytes",
            ));
        }
        let indices =
            WindowSequence::try_new_with_bits(&self.indices, 32, self.count, index_range)?;
        let items = WindowSequence::try_new(&self.items, self.item_count, item_range)?;
        let prefix_bytes = (selected.end - selected.start)
            .checked_add(1)
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| corrupt("Dictionary selected prefixes exceed address space"))?;
        let table_bytes = usize::try_from(self.item_count)
            .ok()
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| corrupt("Dictionary lookup size exceeds address space"))?;
        // The two child streams run sequentially, sharing scratch. The lookup
        // table remains live throughout index traversal, then is recycled.
        if prefix_bytes
            .checked_add(table_bytes)
            .and_then(|n| n.checked_add(items.scratch_bytes().max(indices.scratch_bytes())))
            .is_none_or(|n| n > work.len())
        {
            return Ok(None);
        }
        let mut table = work
            .try_allocate(table_bytes)?
            .ok_or_else(|| Error::internal("Dictionary table preflight changed"))?;
        let mut prefixes = work
            .try_allocate(prefix_bytes)?
            .ok_or_else(|| Error::internal("Dictionary prefix preflight changed"))?;
        let mut item = 0_usize;
        if items
            .visit(work, read, &mut |value| {
                table
                    .as_mut()
                    .get_mut(item..item + 8)
                    .ok_or_else(|| corrupt("Dictionary item stream exceeds cardinality"))?
                    .copy_from_slice(&value.to_le_bytes());
                item += 8;
                Ok(())
            })
            .await?
            .is_none()
        {
            return Err(Error::internal("Dictionary item scratch preflight changed"));
        }
        let mut row = 0_u64;
        let mut sum = 0_u64;
        if indices
            .visit(work, read, &mut |index| {
                if index >= self.item_count {
                    return Err(corrupt("Dictionary index exceeds item count"));
                }
                let offset = index as usize * 8; // bounded by the validated dictionary
                let value = table.as_ref()[offset..offset + 8]
                    .first_chunk::<8>()
                    .ok_or_else(|| Error::internal("missing Dictionary lookup item"))?;
                let value = u64::from_le_bytes(*value);
                // Unreferenced dictionary items may be zero; referenced List
                // counts cannot. Validate even indices outside the selection.
                if value == 0 {
                    return Err(corrupt("nonempty List counts must be positive"));
                }
                if (selected.start..=selected.end).contains(&row) {
                    let offset = ((row - selected.start) * 8) as usize;
                    prefixes.as_mut()[offset..offset + 8].copy_from_slice(&sum.to_le_bytes());
                }
                sum = sum
                    .checked_add(value)
                    .ok_or_else(|| corrupt("Dictionary count sum overflows"))?;
                row += 1;
                Ok(())
            })
            .await?
            .is_none()
        {
            return Err(Error::internal(
                "Dictionary index scratch preflight changed",
            ));
        }
        if row != self.count || sum != total {
            return Err(corrupt("Dictionary count sum differs from child domain"));
        }
        if selected.end == row {
            prefixes.as_mut()[prefix_bytes - 8..].copy_from_slice(&sum.to_le_bytes());
        }
        table.recycle(work)?;
        Ok(Some(Counts::Selected {
            start: selected.start,
            prefixes: Buffer::from(prefixes.freeze()),
        }))
    }
}
