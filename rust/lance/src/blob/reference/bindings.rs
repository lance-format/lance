// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, LazyLock};

use arrow_array::{LargeBinaryArray, RecordBatch, UInt64Array};
use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use lance_core::{Error, Result};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use super::{BlobRefReadStream, BlobRefReader, BlobRefRequest, ReadBlobRef};

static RESULT_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("request_index", DataType::UInt64, false),
        Field::new("data", DataType::LargeBinary, true),
    ]))
});

/// A binding-friendly cancellable stream of Arrow result batches.
///
/// Each batch contains one completed request with `request_index: uint64` and
/// nullable `data: large_binary`. Completion order is unspecified. Payload buffers
/// share the native result owner without copying, so Arrow consumers retain their
/// materialization reservations until they release the arrays. Extra copies are
/// outside that budget. Release batches promptly to allow later work to proceed.
///
/// ```
/// # use lance::Result;
/// # use lance::blob::reference::{BlobRefReader, BlobRefRequest, BlobRefReadSession};
/// # async fn example(reader: &BlobRefReader, requests: Vec<BlobRefRequest>) -> Result<()> {
/// let session = BlobRefReadSession::try_new(reader, requests)?;
/// while let Some(batch) = session.next_batch().await.transpose()? {
///     assert_eq!(batch.num_rows(), 1);
/// }
/// session.close().await;
/// # Ok(())
/// # }
/// ```
pub struct BlobRefReadSession {
    stream: Mutex<Option<BlobRefReadStream>>,
    cancellation: CancellationToken,
}

impl BlobRefReadSession {
    /// Validate all known demands and create a lazily polled native read stream.
    pub fn try_new(reader: &BlobRefReader, requests: Vec<BlobRefRequest>) -> Result<Self> {
        let stream = reader.read(requests)?;
        Ok(Self {
            cancellation: stream.cancellation_token(),
            stream: Mutex::new(Some(stream)),
        })
    }

    /// Await one completed demand. EOF and errors release pending native resources.
    /// Calls are serialized; cancellation can be requested concurrently.
    pub async fn next_batch(&self) -> Option<Result<RecordBatch>> {
        let mut stream = self.stream.lock().await;
        let result = stream
            .as_mut()?
            .next()
            .await
            .map(|result| result.and_then(result_batch));
        if !matches!(result, Some(Ok(_))) {
            stream.take();
        }
        result
    }

    /// Cancel without waiting for a consumer or native budget admission.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Cancel and release pending buffers/readers; already yielded Arrow arrays stay valid.
    /// Repeated calls are harmless. Backend I/O already submitted may finish later.
    pub async fn close(&self) {
        self.cancel();
        self.stream.lock().await.take();
    }
}

fn result_batch(result: ReadBlobRef) -> Result<RecordBatch> {
    let data = match result.data {
        Some(bytes) => {
            let length = i64::try_from(bytes.len()).map_err(|_| {
                Error::invalid_input(
                    "Blob result length exceeds Arrow large_binary offset capacity",
                )
            })?;
            LargeBinaryArray::new(
                OffsetBuffer::new(ScalarBuffer::from(vec![0_i64, length])),
                Buffer::from(bytes),
                None,
            )
        }
        None => LargeBinaryArray::new_null(1),
    };
    Ok(RecordBatch::try_new(
        RESULT_SCHEMA.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![result.request_index])),
            Arc::new(data),
        ],
    )?)
}
