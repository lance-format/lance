// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Request-owned speculation. No task is spawned for read-ahead.

use std::{future::Future, ops::Range, task::Poll};

use futures::{FutureExt, future::BoxFuture};
use lance_core::{Error, Result};

use super::PositionWindow;

struct PendingWindow<'a> {
    blocks: Range<usize>,
    future: Option<BoxFuture<'a, Result<PositionWindow>>>,
    ready: Option<Result<PositionWindow>>,
}

/// At most one extra window per distinct term. The caller admits its bytes
/// together with the term's current window before scheduling it.
pub(super) struct ReadAhead<'a> {
    slots: Vec<Option<PendingWindow<'a>>>,
}

impl<'a> ReadAhead<'a> {
    pub(super) fn new(terms: usize) -> Self {
        Self {
            slots: (0..terms).map(|_| None).collect(),
        }
    }

    pub(super) fn insert(
        &mut self,
        term: usize,
        blocks: Range<usize>,
        future: BoxFuture<'a, Result<PositionWindow>>,
    ) -> Result<()> {
        let slot = self
            .slots
            .get_mut(term)
            .ok_or_else(|| Error::internal("read-ahead term has no prepared position stream"))?;
        *slot = Some(PendingWindow {
            blocks,
            future: Some(future),
            ready: None,
        });
        Ok(())
    }

    /// Transfer a requested window, or drop stale speculation after a skip.
    /// Speculative errors affect the query only if that window is needed.
    pub(super) fn take(
        &mut self,
        term: usize,
        block: usize,
    ) -> Option<BoxFuture<'a, Result<PositionWindow>>> {
        let mut pending = self.slots.get_mut(term)?.take()?;
        if !pending.blocks.contains(&block) {
            return None;
        }
        Some(
            async move {
                match pending.ready.take() {
                    Some(result) => result,
                    None => {
                        pending
                            .future
                            .take()
                            .ok_or_else(|| Error::internal("read-ahead lost its position window"))?
                            .await
                    }
                }
            }
            .boxed(),
        )
    }

    /// Poll speculation only while essential CPU or I/O work is pending.
    /// Returning never waits for an unused speculative read to finish.
    pub(super) async fn drive<F: Future>(&mut self, future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        futures::future::poll_fn(|context| {
            let essential = future.as_mut().poll(context);
            if essential.is_pending() {
                for pending in self.slots.iter_mut().flatten() {
                    if let Some(future) = pending.future.as_mut()
                        && let Poll::Ready(result) = future.as_mut().poll(context)
                    {
                        pending.ready = Some(result);
                        pending.future = None;
                    }
                }
            }
            essential
        })
        .await
    }

    pub(super) fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
    }
}
