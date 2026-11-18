// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use arrow_array::RecordBatch;
use futures::{Stream, StreamExt, stream::BoxStream};
use lance_core::{
    Error, Result,
    utils::tokio::{JoinHandle, spawn},
};
use tokio::sync::mpsc::{self, Receiver};

use super::{AdmittedDatasetScan, AdmittedDatasetScanOutcome};

struct PrefetchedBatches {
    receiver: Receiver<Result<RecordBatch>>,
    producer: JoinHandle<()>,
    is_finished: bool,
}

impl Drop for PrefetchedBatches {
    fn drop(&mut self) {
        self.producer.abort();
    }
}

impl Stream for PrefetchedBatches {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.is_finished {
            return Poll::Ready(None);
        }
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(result)) => {
                if result.is_err() {
                    self.is_finished = true;
                    self.producer.abort();
                    self.receiver.close();
                }
                Poll::Ready(Some(result))
            }
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                // A closed channel alone cannot prove complete output: the
                // producer may have panicked while decoding the remaining rows.
                let result = std::task::ready!(Pin::new(&mut self.producer).poll(cx));
                self.is_finished = true;
                Poll::Ready(result.err().map(|error| {
                    Err(Error::internal(format!(
                        "admitted prefetch producer failed: {error}"
                    )))
                }))
            }
        }
    }
}

pub(super) fn start(mut scan: AdmittedDatasetScan) -> BoxStream<'static, Result<RecordBatch>> {
    let (sender, receiver) = mpsc::channel(1);
    let producer = spawn(async move {
        loop {
            // Reserve delivery capacity before doing I/O, not after retaining
            // another completed batch behind a full queue.
            let Ok(slot) = sender.reserve().await else {
                return;
            };
            match scan.next_when_available().await {
                Ok(AdmittedDatasetScanOutcome::Batch { batch, .. }) => {
                    slot.send(Ok(batch));
                }
                Ok(AdmittedDatasetScanOutcome::Finished) => return,
                Ok(AdmittedDatasetScanOutcome::NeedsCapacity) => {
                    slot.send(Err(Error::invalid_input(format!(
                        "minimal prefix at {:?} does not fit admitted prefetch workspace {} bytes",
                        scan.position(),
                        scan.workspace_bytes,
                    ))));
                    return;
                }
                Err(error) => {
                    slot.send(Err(error));
                    return;
                }
            }
        }
    });
    PrefetchedBatches {
        receiver,
        producer,
        is_finished: false,
    }
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn producer_failure_is_not_end_of_stream() {
        let (sender, receiver) = mpsc::channel(1);
        let producer = spawn(async move {
            let _sender = sender;
            panic!("injected producer failure");
        });
        let mut stream = PrefetchedBatches {
            receiver,
            producer,
            is_finished: false,
        };
        let error = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(
            error
                .to_string()
                .contains("admitted prefetch producer failed")
        );
        assert!(stream.next().await.is_none());
    }
}
