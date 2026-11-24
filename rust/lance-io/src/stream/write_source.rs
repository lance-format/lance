// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use futures::{Stream, StreamExt, TryStreamExt, stream};
use lance_core::{Error, Result, datatypes::Schema};

use super::background_iterator::BackgroundIterator;
use super::{RecordBatchStreamAdapter, SendableRecordBatchStream};

/// A source of Arrow batches for streaming writes.
///
/// Arrow readers run on background blocking threads. A Lance stream is used
/// directly.
///
/// ```
/// # use arrow_array::{RecordBatchIterator, record_batch};
/// # use lance_io::stream::StreamingWriteSource;
/// # async fn example() -> lance_core::Result<()> {
/// let batch = record_batch!(("id", Int32, [1, 2, 3]))?;
/// let schema = batch.schema();
/// let reader = RecordBatchIterator::new([Ok(batch)], schema);
/// let (stream, schema) = reader.into_stream_and_schema().await?;
/// # drop((stream, schema));
/// # Ok(())
/// # }
/// ```
#[async_trait]
pub trait StreamingWriteSource: Send {
    /// Infer and validate the Lance schema, including dictionaries from the
    /// first batch. The returned stream includes that batch.
    ///
    /// Compatibility with an existing dataset must be checked by the caller.
    async fn into_stream_and_schema(self) -> Result<(SendableRecordBatchStream, Schema)>
    where
        Self: Sized,
    {
        let (stream, arrow_schema, schema) =
            infer_write_schema(self.into_stream(), |stream| stream.schema()).await?;
        Ok((
            Box::pin(RecordBatchStreamAdapter::new(arrow_schema, stream)),
            schema,
        ))
    }

    /// Return the Arrow schema without consuming a batch.
    fn arrow_schema(&self) -> SchemaRef;

    /// Convert to a stream, reading synchronous sources on blocking threads.
    fn into_stream(self) -> SendableRecordBatchStream;
}

impl<R: RecordBatchReader + Send + 'static> StreamingWriteSource for R {
    fn arrow_schema(&self) -> SchemaRef {
        self.schema()
    }

    fn into_stream(self) -> SendableRecordBatchStream {
        reader_to_stream(self)
    }
}

impl StreamingWriteSource for SendableRecordBatchStream {
    fn arrow_schema(&self) -> SchemaRef {
        self.schema()
    }

    fn into_stream(self) -> SendableRecordBatchStream {
        self
    }
}

/// Stream an Arrow reader on background blocking threads.
///
/// ```
/// # use arrow_array::RecordBatchReader;
/// # use lance_io::stream::reader_to_stream;
/// # fn example(reader: Box<dyn RecordBatchReader + Send>) {
/// let stream = reader_to_stream(reader);
/// # drop(stream);
/// # }
/// ```
pub fn reader_to_stream(
    reader: impl RecordBatchReader + Send + 'static,
) -> SendableRecordBatchStream {
    let schema = reader.schema();
    let stream = BackgroundIterator::new(reader).fuse().map_err(Error::from);
    Box::pin(RecordBatchStreamAdapter::new(schema, stream))
}

/// Infer a validated Lance write schema and replay the first batch on the
/// returned stream. Errors from later batches keep the source stream's type.
///
/// The schema accessor and first batch are read in a spawned task.
///
/// ```
/// # use lance_io::stream::{SendableRecordBatchStream, infer_write_schema};
/// # async fn example(input: SendableRecordBatchStream) -> lance_core::Result<()> {
/// let (stream, arrow_schema, write_schema) =
///     infer_write_schema(input, |stream| stream.schema()).await?;
/// # drop((stream, arrow_schema, write_schema));
/// # Ok(())
/// # }
/// ```
pub async fn infer_write_schema<S, E>(
    mut stream: S,
    schema_of: fn(&S) -> SchemaRef,
) -> Result<(
    impl Stream<Item = std::result::Result<RecordBatch, E>> + Send,
    SchemaRef,
    Schema,
)>
where
    S: Stream<Item = std::result::Result<RecordBatch, E>> + Send + Unpin + 'static,
    E: Into<Error> + Send + 'static,
{
    let result = lance_core::utils::tokio::spawn(async move {
        let arrow_schema = schema_of(&stream);
        let mut schema = Schema::try_from(arrow_schema.as_ref())?;
        let first_batch = stream.try_next().await.map_err(Into::into)?;
        if let Some(batch) = &first_batch {
            schema.set_dictionary(batch)?;
        }
        let stream = stream::iter(first_batch.map(Ok)).chain(stream);
        Result::Ok((stream, arrow_schema, schema))
    })
    .await;
    let (stream, arrow_schema, schema) = match result {
        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
        result => result??,
    };
    schema.validate()?;
    Ok((stream, arrow_schema, schema))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::Int8Type;
    use arrow_array::{ArrayRef, DictionaryArray, RecordBatchIterator, record_batch};

    use super::*;

    #[tokio::test]
    async fn reader_write_source_infers_dictionary_and_replays_first_batch() {
        let dictionary: DictionaryArray<Int8Type> = ["red", "blue", "red"].into_iter().collect();
        let values = dictionary.values().clone();
        let batch =
            RecordBatch::try_from_iter([("label", Arc::new(dictionary) as ArrayRef)]).unwrap();
        let runtime_thread = std::thread::current().id();
        let batches = [Ok(batch.clone())].into_iter().inspect(move |_| {
            assert_ne!(std::thread::current().id(), runtime_thread);
        });
        let reader: Box<dyn RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(batches, batch.schema()));

        let (stream, schema) = reader.into_stream_and_schema().await.unwrap();
        let dictionary = schema.field("label").unwrap().dictionary.as_ref().unwrap();
        assert_eq!(dictionary.values.as_ref().unwrap(), &values);
        assert_eq!(stream.schema(), batch.schema());
        assert_eq!(stream.try_collect::<Vec<_>>().await.unwrap(), vec![batch]);
    }

    #[tokio::test]
    async fn stream_write_source_reports_first_batch_error() {
        let schema = record_batch!(("id", Int32, [1])).unwrap().schema();
        let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter([Err(Error::invalid_input("first batch failed"))]),
        ));

        let Err(error) = stream.into_stream_and_schema().await else {
            panic!("schema inference succeeded after a first-batch error");
        };
        assert!(matches!(error, Error::InvalidInput { source, .. }
            if source.to_string().contains("first batch failed")));
    }
}
