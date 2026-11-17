// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Visit a size-prefixed LZ4 block with bounded, caller-owned input and history.
//! Unlike the ordinary decoder this never materializes the complete output.

use super::*;

// A match can refer back at most 65535 bytes. A power-of-two history also keeps
// each emitted 4 KiB span contiguous across ring wrap.
const HISTORY: usize = 65536;

pub(super) struct Plan {
    pub(super) decoded: usize,
    input: usize,
}

impl Plan {
    pub(super) fn scratch(&self) -> usize {
        self.decoded.min(HISTORY) + self.input
    }

    pub(super) async fn read<F, Fut>(
        bytes: &Range<u64>,
        work: &mut Reservation,
        read: &mut F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        if bytes.end.checked_sub(bytes.start).is_none_or(|n| n < 5) {
            return Err(corrupt(
                "LZ4 dictionary is missing its size prefix or token",
            ));
        }
        if work.len() < 4 {
            return Ok(None);
        }
        let prefix = Dictionary::read(bytes.start..bytes.start + 4, work, read).await?;
        let decoded = word(prefix.as_ref()) as usize;
        prefix.recycle(work)?;
        if decoded as u64 > (bytes.end - bytes.start).saturating_mul(255)
            || ::lz4::block::compress_bound(decoded).is_err()
        {
            return Err(corrupt("LZ4 dictionary declares an invalid decoded size"));
        }
        Ok(Some(Self {
            decoded,
            input: (bytes.end - bytes.start - 4).min(WINDOW as u64) as usize,
        }))
    }

    pub(super) async fn visit<F, Fut, V>(
        &self,
        bytes: &Range<u64>,
        work: &mut Reservation,
        read: &mut F,
        mut visit: V,
    ) -> Result<()>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
        V: FnMut(usize, &[u8]) -> Result<()>,
    {
        // Recheck the size when a custom reader changes between the two passes.
        let current = Self::read(bytes, work, read)
            .await?
            .ok_or_else(|| Error::internal("LZ4 dictionary prefix preflight changed"))?;
        if current.decoded != self.decoded {
            return Err(corrupt("LZ4 dictionary size changed between reads"));
        }
        if self.scratch() > work.len() {
            return Err(Error::internal("LZ4 dictionary scratch preflight changed"));
        }
        let mut history = work
            .try_allocate(self.decoded.min(HISTORY))?
            .ok_or_else(|| Error::internal("LZ4 history preflight changed"))?;
        let mut input = Input {
            position: bytes.start + 4,
            end: bytes.end,
            window: None,
            index: 0,
        };
        let result = self
            .decode(&mut input, history.as_mut(), work, read, &mut visit)
            .await;
        input.recycle(work)?;
        history.recycle(work)?;
        result
    }

    async fn decode<F, Fut, V>(
        &self,
        input: &mut Input,
        history: &mut [u8],
        work: &mut Reservation,
        read: &mut F,
        visit: &mut V,
    ) -> Result<()>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
        V: FnMut(usize, &[u8]) -> Result<()>,
    {
        let mut produced = 0_usize;
        let mut last_match = None;
        loop {
            let token = input.byte(work, read).await?;
            let literals = input.length(usize::from(token >> 4), work, read).await?;
            if literals > self.decoded - produced || literals as u64 > input.end - input.position {
                return Err(corrupt("LZ4 literals exceed input or declared output"));
            }
            let mut left = literals;
            while left != 0 {
                input.fill(work, read).await?;
                let source = input.available()?;
                let len = left.min(source.len()).min(WINDOW - produced % WINDOW);
                let at = produced % history.len();
                history[at..at + len].copy_from_slice(&source[..len]);
                input.consume(len);
                produced += len;
                left -= len;
                if produced.is_multiple_of(WINDOW) {
                    emit(history, produced - WINDOW, WINDOW, visit)?;
                    tokio::task::yield_now().await;
                }
            }
            if input.position == input.end {
                if produced != self.decoded
                    || (self.decoded >= 5 && literals < 5)
                    || last_match.is_some_and(|at| self.decoded - at < 12)
                {
                    return Err(corrupt("LZ4 final literals or decoded length are invalid"));
                }
                let tail = produced % WINDOW;
                if tail != 0 {
                    emit(history, produced - tail, tail, visit)?;
                }
                return Ok(());
            }
            let low = input.byte(work, read).await?;
            let high = input.byte(work, read).await?;
            let offset = usize::from(u16::from_le_bytes([low, high]));
            if offset == 0 || offset > produced {
                return Err(corrupt("LZ4 match offset exceeds available history"));
            }
            let length = input
                .length(usize::from(token & 15), work, read)
                .await?
                .checked_add(4)
                .ok_or_else(|| corrupt("LZ4 match length overflows"))?;
            if length > self.decoded - produced {
                return Err(corrupt("LZ4 match exceeds declared output"));
            }
            last_match = Some(produced);
            let mut left = length;
            while left != 0 {
                let len = left.min(WINDOW - produced % WINDOW);
                // Forward copying is required: offset=1 repeats the last byte,
                // and a match can cross both the emitted span and ring boundary.
                for _ in 0..len {
                    let ring_len = history.len();
                    let byte = history[(produced - offset) % ring_len];
                    history[produced % ring_len] = byte;
                    produced += 1;
                }
                left -= len;
                if produced.is_multiple_of(WINDOW) {
                    emit(history, produced - WINDOW, WINDOW, visit)?;
                    tokio::task::yield_now().await;
                }
            }
        }
    }
}

fn emit<V>(history: &[u8], start: usize, len: usize, visit: &mut V) -> Result<()>
where
    V: FnMut(usize, &[u8]) -> Result<()>,
{
    let at = start % history.len();
    visit(start, &history[at..at + len])
}

struct Input {
    position: u64,
    end: u64,
    window: Option<ReservedBuffer>,
    index: usize,
}

impl Input {
    fn recycle(&mut self, work: &mut Reservation) -> Result<()> {
        if let Some(window) = self.window.take() {
            window.recycle(work)?;
        }
        Ok(())
    }

    fn available(&self) -> Result<&[u8]> {
        self.window
            .as_ref()
            .map(|window| &window.as_ref()[self.index..])
            .ok_or_else(|| Error::internal("LZ4 input window is absent"))
    }

    fn consume(&mut self, len: usize) {
        self.index += len;
        self.position += len as u64;
    }

    async fn fill<F, Fut>(&mut self, work: &mut Reservation, read: &mut F) -> Result<()>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        if self
            .window
            .as_ref()
            .is_some_and(|w| self.index < w.as_ref().len())
        {
            return Ok(());
        }
        self.recycle(work)?;
        if self.position == self.end {
            return Err(corrupt("truncated LZ4 token, length or offset"));
        }
        let end = self.position + (self.end - self.position).min(WINDOW as u64);
        self.window = Some(Dictionary::read(self.position..end, work, read).await?);
        self.index = 0;
        tokio::task::yield_now().await;
        Ok(())
    }

    async fn byte<F, Fut>(&mut self, work: &mut Reservation, read: &mut F) -> Result<u8>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        self.fill(work, read).await?;
        let value = self.available()?[0];
        self.consume(1);
        Ok(value)
    }

    async fn length<F, Fut>(
        &mut self,
        initial: usize,
        work: &mut Reservation,
        read: &mut F,
    ) -> Result<usize>
    where
        F: FnMut(Range<u64>, ReservedBuffer) -> Fut,
        Fut: Future<Output = Result<ReservedBuffer>>,
    {
        let mut length = initial;
        if initial == 15 {
            loop {
                let next = self.byte(work, read).await?;
                length = length
                    .checked_add(usize::from(next))
                    .ok_or_else(|| corrupt("LZ4 sequence length overflows"))?;
                if next != 255 {
                    break;
                }
            }
        }
        Ok(length)
    }
}

#[cfg(test)]
mod tests;
