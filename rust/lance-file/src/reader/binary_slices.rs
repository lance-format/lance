// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;

use lance_core::{
    Error, Result,
    cache::{CacheKey, CacheKeySchema, KeyBuilder},
    deepsize::{Context, DeepSizeOf},
};
use lance_encoding::encodings::{
    logical::primitive::MiniBlockScheduler,
    physical::binary::{BinaryValueLocation, PreparedBinaryValue},
};

use super::FileReader;

#[derive(Debug)]
struct CachedBinaryLocation(Option<BinaryValueLocation>);

impl DeepSizeOf for CachedBinaryLocation {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        // Locations contain only scalar file offsets; no request or payload owners.
        0
    }
}

#[derive(Debug)]
struct BinaryLocationKey {
    column_index: usize,
    row: u64,
}

impl CacheKey for BinaryLocationKey {
    type ValueType = CachedBinaryLocation;

    fn key(&self) -> Cow<'_, str> {
        format!("binary_location/{}/{}", self.column_index, self.row).into()
    }

    fn type_name() -> &'static str {
        "BinaryValueLocation"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.file.binary-value-location", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.column_index as u64);
        builder.write_u64(self.row);
    }
}

impl FileReader {
    /// Prepare byte reads for one flat binary value without loading its payload.
    ///
    /// `column_index` is a physical column index and `row` is relative to that
    /// column, including ragged files. The handle exposes the exact value length
    /// and reuses validated metadata for repeated byte reads. Unsupported
    /// encodings return `None`, requesting ordinary decoding by the caller.
    /// Handles retain this reader's I/O scope; do not share request-specific
    /// handles through a global cache. Only detached locations use this reader's
    /// existing file-scoped metadata cache; each handle binds the current I/O
    /// scope and priority. No persisted format changes.
    ///
    /// ```
    /// # use lance_file::reader::FileReader;
    /// # async fn example(reader: &FileReader) -> lance_core::Result<()> {
    /// if let Some(value) = reader.try_prepare_binary_value(0, 2).await? {
    ///     let tail = value.read_slices(&[value.len().saturating_sub(8)..value.len()]).await?;
    ///     assert!(tail[0].len() <= 8);
    /// }
    /// # Ok(()) }
    /// ```
    pub async fn try_prepare_binary_value(
        &self,
        column_index: usize,
        row: u64,
    ) -> Result<Option<PreparedBinaryValue>> {
        let column = self
            .metadata
            .column_infos
            .get(column_index)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "binary byte read column index {column_index} exceeds {} physical columns",
                    self.metadata.column_infos.len()
                ))
            })?;
        let mut page_row = row;
        for page in column.page_infos.iter() {
            if page_row < page.num_rows {
                let location = self
                    .core
                    .cache
                    .get_or_insert_with_key(BinaryLocationKey { column_index, row }, || async {
                        let value = MiniBlockScheduler::try_prepare_binary_value(
                            page,
                            page_row,
                            &self.core.scheduler,
                        )
                        .await?;
                        Ok::<_, Error>(CachedBinaryLocation(value.map(|value| value.location())))
                    })
                    .await?;
                return Ok(location
                    .0
                    .as_ref()
                    .map(|location| location.with_io(self.core.scheduler.clone(), page.priority)));
            }
            page_row -= page.num_rows;
        }
        Err(Error::invalid_input(format!(
            "binary byte read row {row} is outside physical column {column_index}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };

    use arrow_array::{LargeBinaryArray, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use futures::future::BoxFuture;
    use lance_core::cache::LanceCache;
    use lance_encoding::{
        EncodingsIo,
        constants::{STRUCTURAL_ENCODING_META_KEY, STRUCTURAL_ENCODING_MINIBLOCK},
        decoder::DecoderPlugins,
    };
    use lance_io::utils::CachedFileSize;

    use super::*;
    use crate::{
        reader::FileReaderOptions,
        testing::{FsFixture, write_lance_file},
        version::ConcreteFileVersion,
        writer::FileWriterOptions,
    };

    #[derive(Debug)]
    struct ScopedIo {
        inner: Arc<dyn EncodingsIo>,
        reads: AtomicUsize,
        short_next: AtomicBool,
    }

    impl EncodingsIo for ScopedIo {
        fn submit_request(
            &self,
            ranges: Vec<std::ops::Range<u64>>,
            priority: u64,
        ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            let read = self.inner.submit_request(ranges, priority);
            if self.short_next.swap(false, Ordering::SeqCst) {
                Box::pin(async move {
                    let mut buffers = read.await?;
                    if let Some(bytes) = buffers.first_mut() {
                        bytes.truncate(bytes.len().saturating_sub(1));
                    }
                    Ok(buffers)
                })
            } else {
                read
            }
        }
    }

    #[tokio::test]
    async fn binary_slices_cached_location_rebinds_scope_and_separates_files_columns_rows() {
        let cache = LanceCache::with_capacity(1024 * 1024);
        for file in 0_u8..2 {
            let fs = FsFixture::default();
            let schema = Arc::new(Schema::new(["left", "right"].map(|name| {
                Field::new(name, DataType::LargeBinary, false).with_metadata(HashMap::from([(
                    STRUCTURAL_ENCODING_META_KEY.to_owned(),
                    STRUCTURAL_ENCODING_MINIBLOCK.to_owned(),
                )]))
            })));
            let values = [
                [
                    vec![b'a' + file; 8192 + usize::from(file) * 257],
                    vec![b'b' + file; 10240],
                ],
                [
                    vec![b'c' + file; 6144],
                    vec![b'd' + file; 9216 + usize::from(file) * 131],
                ],
            ];
            let batch = RecordBatch::try_new(
                schema.clone(),
                values
                    .iter()
                    .map(|column| Arc::new(LargeBinaryArray::from_iter_values(column)) as _)
                    .collect(),
            )
            .unwrap();
            write_lance_file(
                RecordBatchIterator::new([Ok(batch)], schema),
                &fs,
                ConcreteFileVersion::V2_3,
                FileWriterOptions::default(),
            )
            .await;
            let scheduler = fs
                .scheduler
                .open_file(&fs.tmp_path, &CachedFileSize::unknown())
                .await
                .unwrap();
            let mut reader = FileReader::try_open(
                scheduler,
                None,
                Arc::<DecoderPlugins>::default(),
                &cache,
                FileReaderOptions::default(),
            )
            .await
            .unwrap();
            let base_io = reader.core.scheduler.clone();
            let first_scope = Arc::new(ScopedIo {
                inner: base_io.clone(),
                reads: AtomicUsize::new(0),
                short_next: AtomicBool::new(true),
            });
            let weak = Arc::downgrade(&first_scope);
            reader.core.scheduler = first_scope.clone();
            let Err(error) = reader.try_prepare_binary_value(0, 0).await else {
                panic!("truncated preparation metadata was accepted");
            };
            assert!(
                matches!(error.find_root(), Error::CorruptFile { .. }),
                "{error:?}"
            );
            assert!(error.to_string().contains("truncated"));
            let value = reader
                .try_prepare_binary_value(0, 0)
                .await
                .unwrap()
                .unwrap();
            assert!(first_scope.reads.load(Ordering::SeqCst) > 0);
            drop(value);

            let next_scope = Arc::new(ScopedIo {
                inner: base_io,
                reads: AtomicUsize::new(0),
                short_next: AtomicBool::new(false),
            });
            let mut next_reader = reader.clone();
            next_reader.core.scheduler = next_scope.clone();
            drop(reader);
            drop(first_scope);
            assert!(
                weak.upgrade().is_none(),
                "cached locations must release the previous request scope"
            );
            let value = next_reader
                .try_prepare_binary_value(0, 0)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(
                next_scope.reads.load(Ordering::SeqCst),
                0,
                "cached preparation must not perform I/O"
            );
            assert_eq!(
                value.read_slices(&[3..11]).await.unwrap()[0].as_ref(),
                &values[0][0][3..11]
            );
            assert_eq!(
                next_scope.reads.load(Ordering::SeqCst),
                1,
                "payload reads must use the new scope"
            );

            for (column, rows) in values.iter().enumerate() {
                for (row, expected) in rows.iter().enumerate() {
                    let value = next_reader
                        .try_prepare_binary_value(column, row as u64)
                        .await
                        .unwrap()
                        .unwrap();
                    assert_eq!(value.len(), expected.len() as u64);
                    let tail = value
                        .read_slices(&[value.len() - 4..value.len()])
                        .await
                        .unwrap();
                    assert_eq!(tail[0].as_ref(), &expected[expected.len() - 4..]);
                }
            }
        }
    }
}
