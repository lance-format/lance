// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read the row index alongside chunk metadata without materializing either page.

use super::*;

#[derive(Debug)]
pub(super) struct RepetitionSummary {
    starts: u64,
    has_preamble: bool,
    partial: u64,
}

impl MiniBlockArrayScheduler {
    pub(super) async fn nested_selections<F, Fut>(
        &self,
        rows: &Range<u64>,
        repetition: &Range<u64>,
        work: &mut Reservation,
        read: &mut F,
    ) -> Result<Option<Vec<Selection>>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let width = if self.decoder.has_large_chunk { 4 } else { 2 };
        let count = (self.metadata.end - self.metadata.start) / width;
        // At most 4 KiB of repetition index and 1 KiB of chunk metadata overlap.
        // Preflight both before either dependent read; never wait holding one.
        let window_chunks = count.min(256);
        if window_chunks * (16 + width) > work.len() as u64 {
            return Ok(None);
        }
        let available = work.len() as u64;
        let mut selected = Vec::new();
        let mut selected_bytes = 0_u64;
        let mut fits = true;
        let mut index = 0_u64;
        let mut item = 0_u64;
        let mut byte = self.data.start;
        let mut row = 0_u64;
        let mut has_preamble = false;
        while index < count {
            let chunks = (count - index).min(window_chunks);
            let meta_start = self.metadata.start + index * width;
            let rep_start = repetition.start + index * 16;
            let meta_len = (chunks * width) as usize;
            let rep_len = (chunks * 16) as usize;
            let meta = work
                .try_allocate(meta_len)?
                .ok_or_else(|| Error::internal("MiniBlock index preflight changed"))?;
            let meta = read(meta_start..meta_start + meta_len as u64, meta).await?;
            let rep = work
                .try_allocate(rep_len)?
                .ok_or_else(|| Error::internal("MiniBlock repetition preflight changed"))?;
            let rep = read(rep_start..rep_start + rep_len as u64, rep).await?;
            if meta.as_ref().len() != meta_len || rep.as_ref().len() != rep_len {
                return Err(corrupt("nested index read length mismatch"));
            }
            for (word, pair) in meta
                .as_ref()
                .chunks_exact(width as usize)
                .zip(rep.as_ref().as_chunks::<16>().0)
            {
                let mut word_bytes = [0; 4];
                word_bytes[..word.len()].copy_from_slice(word);
                let word = u32::from_le_bytes(word_bytes);
                let log = word & 15;
                let items = if index + 1 == count {
                    let remaining = self
                        .items
                        .checked_sub(item)
                        .filter(|n| *n > 0)
                        .ok_or_else(|| corrupt("chunk item counts exceed nested page items"))?;
                    if log != 0 && (1_u64 << log) != remaining {
                        return Err(corrupt(
                            "final chunk count disagrees with nested page items",
                        ));
                    }
                    remaining
                } else {
                    if log == 0 {
                        return Err(corrupt("non-final chunk has zero count exponent"));
                    }
                    1_u64 << log
                };
                let end_item = item
                    .checked_add(items)
                    .filter(|n| *n <= self.items)
                    .ok_or_else(|| corrupt("chunk item counts exceed nested page items"))?;
                let size = (u64::from(word >> 4) + 1) * MINIBLOCK_ALIGNMENT as u64;
                let end_byte = byte
                    .checked_add(size)
                    .filter(|n| *n <= self.data.end)
                    .ok_or_else(|| corrupt("chunk byte sizes exceed data buffer"))?;
                let ends = u64::from_le_bytes(
                    pair[..8]
                        .try_into()
                        .map_err(|_| corrupt("truncated row-end count"))?,
                );
                let partial = u64::from_le_bytes(
                    pair[8..]
                        .try_into()
                        .map_err(|_| corrupt("truncated partial count"))?,
                );
                let has_trailer = partial > 0;
                let starts = ends
                    .checked_add(u64::from(has_trailer))
                    .and_then(|n| n.checked_sub(u64::from(has_preamble)))
                    .ok_or_else(|| corrupt("repetition row starts overflow or underflow"))?;
                if (!has_preamble && starts == 0) || (index + 1 == count && has_trailer) {
                    return Err(corrupt(
                        "repetition index has an empty chunk or unfinished final row",
                    ));
                }
                let end_row = row
                    .checked_add(starts)
                    .filter(|n| *n <= self.rows)
                    .ok_or_else(|| corrupt("repetition index exceeds page rows"))?;
                let take_preamble = has_preamble && row > 0 && rows.contains(&(row - 1));
                let start = rows.start.max(row);
                let end = rows.end.min(end_row);
                if start < end || take_preamble {
                    selected_bytes = selected_bytes
                        .checked_add(size)
                        .ok_or_else(|| corrupt("selected input size overflows"))?;
                    if fits && selected_bytes > available {
                        selected.clear();
                        fits = false;
                    }
                    if fits {
                        selected.push(Selection {
                            index: usize::try_from(index)
                                .map_err(|_| corrupt("chunk index exceeds usize"))?,
                            bytes: byte..end_byte,
                            items,
                            rows: if start < end {
                                start - row..end - row
                            } else {
                                0..0
                            },
                            preamble: if take_preamble {
                                PreambleAction::Take
                            } else if has_preamble {
                                PreambleAction::Skip
                            } else {
                                PreambleAction::Absent
                            },
                            take_trailer: has_trailer
                                && end_row > 0
                                && rows.contains(&(end_row - 1)),
                            repetition: Some(RepetitionSummary {
                                starts,
                                has_preamble,
                                partial,
                            }),
                        });
                    }
                }
                row = end_row;
                item = end_item;
                byte = end_byte;
                has_preamble = has_trailer;
                index += 1;
            }
            meta.recycle(work)?;
            rep.recycle(work)?;
            tokio::task::yield_now().await;
        }
        if row != self.rows || item != self.items || byte != self.data.end {
            return Err(corrupt(
                "nested index does not cover the exact row/item/data domain",
            ));
        }
        Ok(fits.then_some(selected))
    }
}

impl EncodedArrayInput {
    // Called while the existing admitted decoder owns its level scratch, before
    // selection or value preparation. A mismatch takes its ordinary cleanup path.
    pub(super) fn validate_repetition(
        &self,
        instruction: usize,
        levels: Option<&[u16]>,
    ) -> Result<()> {
        let summary = self
            .repetition
            .get(instruction)
            .ok_or_else(|| corrupt("nested input summary count mismatch"))?;
        let values = levels.ok_or_else(|| corrupt("nested input has no repetition levels"))?;
        let max_rep = self.task.def_meaning.iter().filter(|m| m.is_list()).count() as u16;
        if max_rep == 0
            || values.is_empty()
            || values.iter().any(|value| *value > max_rep)
            || (values[0] != max_rep) != summary.has_preamble
            || values.iter().filter(|value| **value == max_rep).count() as u64 != summary.starts
        {
            return Err(corrupt(
                "repetition index disagrees with decoded row boundaries",
            ));
        }
        if summary.partial != 0 {
            let tail = values
                .iter()
                .rposition(|value| *value == max_rep)
                .map_or(values.len(), |position| values.len() - position);
            if summary.partial != tail as u64 {
                return Err(corrupt(
                    "repetition index disagrees with decoded trailer length",
                ));
            }
        }
        Ok(())
    }
}
