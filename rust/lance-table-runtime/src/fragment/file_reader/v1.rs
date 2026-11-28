// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{RecordBatch, StructArray};
use arrow_schema::Schema as ArrowSchema;
use futures::future::{BoxFuture, try_join_all};
use futures::{FutureExt, StreamExt, TryFutureExt, join, stream};
use lance_arrow::{RecordBatchExt, SchemaExt};
use lance_core::datatypes::{OnMissing, OnTypeMismatch, Schema};
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, ROW_ADDR_FIELD, ROW_ID_FIELD, Result};
use lance_file::version::ConcreteFileVersion;
use lance_file::versions::v1::reader::{FileReader as V1FileReader, read_batch as v1_read_batch};
use lance_io::ReadBatchParams;
use lance_table::format::DataFile;
use lance_table::rowids::RowIdSequence;
use lance_table::utils::stream::{ReadBatchTask, ReadBatchTaskStream, RowIdAndDeletesConfig};

use super::GenericFileReader;
use crate::fragment::{FileFragment, FragReadConfig};
use crate::rowids::load_row_id_sequence;

impl FileFragment {
    /// Open the row-group reader for a fragment containing only legacy V1 files.
    #[doc(hidden)]
    pub async fn open_v1_fragment_reader(
        &self,
        projection: &Schema,
        read_config: &FragReadConfig,
    ) -> Result<V1FragmentReader> {
        for data_file in &self.metadata.files {
            let actual = data_file.file_version()?;
            if actual != ConcreteFileVersion::V1 {
                return Err(Error::invalid_input(format!(
                    "Cannot open file {} with the v1 reader because it has version {}",
                    data_file.path, actual
                )));
            }
        }
        let open_readers = async {
            let mut readers = Vec::new();
            for data_file in &self.metadata.files {
                if let Some(reader) = self.open_v1_reader(data_file, Some(projection)).await? {
                    readers.push(reader);
                }
            }
            Result::Ok(readers)
        };
        let deletion_vec_load = self.get_deletion_vector();
        let row_id_load = if self.table.manifest.uses_stable_row_ids() {
            futures::future::Either::Left(
                load_row_id_sequence(&self.table, &self.metadata).map_ok(Some),
            )
        } else {
            futures::future::Either::Right(futures::future::ready(Ok(None)))
        };
        let (readers, deletion_vec, row_id_sequence) =
            join!(open_readers, deletion_vec_load, row_id_load);
        let mut reader =
            V1FragmentReader::try_new(readers?, deletion_vec?, row_id_sequence?, self.id())?;
        if read_config.with_row_id {
            reader.with_row_id();
        }
        if read_config.with_row_address {
            reader.with_row_address();
        }
        Ok(reader)
    }

    fn get_field_id_offset(data_file: &DataFile) -> u32 {
        data_file.fields.first().copied().unwrap_or(0) as u32
    }

    pub(super) async fn open_v1_reader(
        &self,
        data_file: &DataFile,
        projection: Option<&Schema>,
    ) -> Result<Option<V1Reader>> {
        let full_schema = self.table.schema();
        let data_file_schema = Arc::new(data_file.schema(full_schema));
        let projection = projection.unwrap_or(full_schema);
        let schema_per_file =
            Arc::new(projection.intersection_ignore_types(data_file_schema.as_ref())?);
        if schema_per_file.fields.is_empty() {
            return Ok(None);
        }

        let max_field_id = data_file.fields.iter().max().ok_or_else(|| {
            Error::invalid_input(format!(
                "Legacy data file {} does not contain any fields",
                data_file.path
            ))
        })?;
        let path = self
            .table
            .data_file_dir(data_file)?
            .join(data_file.path.as_str());
        let object_store = self.table.object_store(data_file.base_id).await?;
        let field_id_offset = Self::get_field_id_offset(data_file);
        let reader = V1FileReader::try_new_with_fragment_id(
            &object_store,
            &path,
            self.schema().clone(),
            self.id() as u32,
            field_id_offset as i32,
            *max_field_id,
            Some(&self.table.state.metadata_cache.file_metadata_cache(&path)),
        )
        .await?;
        let initialized_schema = reader.schema().project_by_schema(
            schema_per_file.as_ref(),
            OnMissing::Error,
            OnTypeMismatch::Error,
        )?;
        Ok(Some(V1Reader::new(reader, Arc::new(initialized_schema))))
    }
}

fn ranges_to_tasks(
    reader: &V1FileReader,
    ranges: Vec<(i32, Range<usize>)>,
    projection: Arc<Schema>,
) -> ReadBatchTaskStream {
    let reader = reader.clone();
    stream::iter(ranges)
        .map(move |(batch_idx, range)| {
            let num_rows = range.end - range.start;
            let reader = reader.clone();
            let projection = projection.clone();
            let task = lance_core::utils::tokio::spawn(async move {
                v1_read_batch(
                    &reader,
                    &ReadBatchParams::Range(range.clone()),
                    &projection,
                    batch_idx,
                )
                .await
            })
            .map(|task_out| task_out.unwrap())
            .boxed();
            ReadBatchTask {
                task,
                num_rows: num_rows as u32,
            }
        })
        .boxed()
}

#[derive(Clone, Debug)]
pub(super) struct V1Reader {
    reader: V1FileReader,
    projection: Arc<Schema>,
}

impl V1Reader {
    fn new(reader: V1FileReader, projection: Arc<Schema>) -> Self {
        Self { reader, projection }
    }
}

impl GenericFileReader for V1Reader {
    /// Reads the requested range of rows from the file, returning as a stream
    fn read_range_tasks(
        &self,
        range: Range<u64>,
        batch_size: u32,
        projection: Arc<Schema>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>> {
        let mut to_skip = range.start as u32;
        let mut remaining = range.end as u32 - to_skip;
        let mut ranges = Vec::new();
        let mut batch_idx = 0;
        while remaining > 0 {
            let next_batch_len = self.reader.num_rows_in_batch(batch_idx) as u32;
            let next_batch_idx = batch_idx;
            batch_idx += 1;
            if to_skip >= next_batch_len {
                to_skip -= next_batch_len;
                continue;
            }
            let batch_start = to_skip;
            to_skip = 0;
            let batch_end = next_batch_len.min(batch_start + remaining);
            remaining -= batch_end - batch_start;
            for chunk_start in (batch_start..batch_end).step_by(batch_size as usize) {
                let chunk_end = (chunk_start + batch_size).min(batch_end);
                ranges.push((next_batch_idx, (chunk_start as usize..chunk_end as usize)));
            }
        }
        let stream = ranges_to_tasks(&self.reader, ranges, projection);
        async move { Ok(stream) }.boxed()
    }

    fn read_all_tasks(
        &self,
        batch_size: u32,
        projection: Arc<Schema>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>> {
        let ranges = (0..self.reader.num_batches())
            .flat_map(move |batch_idx| {
                let rows_in_batch = self.reader.num_rows_in_batch(batch_idx as i32);
                (0..rows_in_batch)
                    .step_by(batch_size as usize)
                    .map(move |start| {
                        let end = (start + batch_size as usize).min(rows_in_batch);
                        (batch_idx as i32, start..end)
                    })
            })
            .collect::<Vec<_>>();
        let stream = ranges_to_tasks(&self.reader, ranges, projection);
        async move { Ok(stream) }.boxed()
    }

    fn read_ranges_tasks(
        &self,
        _ranges: Arc<[Range<u64>]>,
        _batch_size: u32,
        _projection: Arc<Schema>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>> {
        async move {
            Err(Error::internal(
                "Attempt to perform FilteredRead on v1 files".to_string(),
            ))
        }
        .boxed()
    }

    fn take_all_tasks(
        &self,
        indices: &[u32],
        _batch_size: u32,
        projection: Arc<Schema>,
        _take_priority: Option<u32>,
    ) -> BoxFuture<'_, Result<ReadBatchTaskStream>> {
        // Keep stored row groups separate: combining otherwise readable groups
        // can exceed Arrow's offset limits before the caller can split them.
        let mut groups = Vec::new();
        let mut indices = indices.to_vec();
        indices.sort_unstable();
        let mut remaining = indices.as_slice();
        let mut batch_end = 0_u64;
        for batch_id in 0..self.reader.num_batches() {
            let batch_start = batch_end;
            batch_end += self.reader.num_rows_in_batch(batch_id as i32) as u64;
            let count = remaining.partition_point(|&index| u64::from(index) < batch_end);
            if count > 0 {
                let offsets = remaining[..count]
                    .iter()
                    .map(|&index| (u64::from(index) - batch_start) as u32)
                    .collect::<Vec<_>>();
                groups.push((Some(batch_id as i32), offsets));
                remaining = &remaining[count..];
            }
            if remaining.is_empty() {
                break;
            }
        }
        // Preserve the file reader's validation of out-of-bounds indices and
        // its empty-selection result.
        if !remaining.is_empty() || groups.is_empty() {
            groups.push((None, remaining.to_vec()));
        }
        let reader = self.reader.clone();
        async move {
            // In the new path the row id is added by the fragment and not the file
            Ok(stream::iter(groups)
                .map(move |(batch_id, indices)| {
                    let reader = reader.clone();
                    let projection = projection.clone();
                    let num_rows = indices.len() as u32;
                    let task = async move {
                        if let Some(batch_id) = batch_id {
                            reader
                                .read_batch(batch_id, indices.as_slice(), projection.as_ref())
                                .await
                        } else {
                            reader.take(&indices, projection.as_ref()).await
                        }
                    }
                    .boxed();
                    ReadBatchTask { task, num_rows }
                })
                .boxed())
        }
        .boxed()
    }

    fn projection(&self) -> &Arc<Schema> {
        &self.projection
    }

    /// Return the number of rows in the file
    fn len(&self) -> u32 {
        self.reader.len() as u32
    }

    fn storage_stats(&self) -> Result<Vec<(u32, u64)>> {
        // No-op for v1 files
        Ok(Vec::new())
    }

    fn clone_box(&self) -> Box<dyn GenericFileReader> {
        Box::new(self.clone())
    }
}

/// Typed v1-only read operations used by the legacy pushdown path.
///
/// Keeping the previous readers here avoids exposing legacy downcasts through
/// [`GenericFileReader`]. Modern readers never implement or simulate these
/// row-group and page-statistics operations.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct V1FragmentReader {
    readers: Vec<V1Reader>,
    deletion_vec: Option<Arc<DeletionVector>>,
    row_id_sequence: Option<Arc<RowIdSequence>>,
    fragment_id: usize,
    with_row_id: bool,
    with_row_addr: bool,
    make_deletions_null: bool,
}

impl V1FragmentReader {
    fn try_new(
        readers: Vec<V1Reader>,
        deletion_vec: Option<Arc<DeletionVector>>,
        row_id_sequence: Option<Arc<RowIdSequence>>,
        fragment_id: usize,
    ) -> Result<Self> {
        let first_reader = readers.first().ok_or_else(|| {
            Error::invalid_input("Cannot create a v1 fragment reader without data files")
        })?;
        let num_batches = first_reader.reader.num_batches();
        if readers
            .iter()
            .any(|reader| reader.reader.num_batches() != num_batches)
        {
            return Err(Error::invalid_input(
                "Cannot create a v1 fragment reader from data files with different numbers of batches"
                    .to_string(),
            ));
        }
        Ok(Self {
            readers,
            deletion_vec,
            row_id_sequence,
            fragment_id,
            with_row_id: false,
            with_row_addr: false,
            make_deletions_null: false,
        })
    }

    #[doc(hidden)]
    pub fn with_row_id(&mut self) -> &mut Self {
        self.with_row_id = true;
        self
    }

    #[doc(hidden)]
    pub fn with_row_address(&mut self) -> &mut Self {
        self.with_row_addr = true;
        self
    }

    #[doc(hidden)]
    pub fn with_make_deletions_null(&mut self) -> &mut Self {
        self.make_deletions_null = true;
        self
    }

    #[doc(hidden)]
    pub fn num_batches(&self) -> usize {
        self.readers[0].reader.num_batches()
    }

    #[doc(hidden)]
    pub fn num_rows_in_batch(&self, batch_id: u32) -> Option<u32> {
        let reader = &self.readers[0].reader;
        (batch_id < reader.num_batches() as u32)
            .then(|| reader.num_rows_in_batch(batch_id as i32) as u32)
    }

    #[doc(hidden)]
    pub async fn read_page_stats(
        &self,
        projection: Option<&Schema>,
    ) -> Result<Option<RecordBatch>> {
        let mut stats_batches = Vec::new();
        for reader in &self.readers {
            let schema = match projection {
                Some(projection) => Arc::new(reader.projection.intersection(projection)?),
                None => reader.projection.clone(),
            };
            if let Some(stats_batch) = reader.reader.read_page_stats(&schema.field_ids()).await? {
                stats_batches.push(stats_batch);
            }
        }

        if stats_batches.is_empty() {
            Ok(None)
        } else {
            Ok(Some(merge_batches(&stats_batches)?))
        }
    }

    #[doc(hidden)]
    pub async fn read_batch_projected(
        &self,
        batch_id: usize,
        params: impl Into<ReadBatchParams> + Clone,
        projection: &Schema,
    ) -> Result<RecordBatch> {
        let first_reader = &self.readers[0].reader;
        // All batches have the same size in v1, except for the last one.
        let batch_offset = batch_id * first_reader.num_rows_in_batch(0);
        let rows_in_batch = first_reader.num_rows_in_batch(batch_id as i32);

        let batches = if !projection.fields.is_empty() {
            let read_tasks = self.readers.iter().map(|reader| {
                let projection = reader.projection.intersection(projection);
                let params = params.clone();
                let reader = &reader.reader;

                async move {
                    let projection = projection?;
                    if projection.fields.is_empty() {
                        Result::Ok(None)
                    } else {
                        Ok(Some(
                            reader
                                .read_batch(batch_id as i32, params, &projection)
                                .await?,
                        ))
                    }
                }
            });
            try_join_all(read_tasks)
                .await?
                .into_iter()
                .flatten()
                .collect::<Vec<RecordBatch>>()
        } else {
            let expected_rows = params
                .clone()
                .into()
                .slice(0, rows_in_batch)
                .unwrap()
                .to_offsets()?
                .len();
            vec![RecordBatch::from(StructArray::new_empty_fields(
                expected_rows,
                None,
            ))]
        };

        let params = params.into();
        let result = merge_batches(&batches)?;
        let file_params = match params {
            ReadBatchParams::Indices(indices) => ReadBatchParams::Indices(
                indices
                    .values()
                    .iter()
                    .map(|i| *i + batch_offset as u32)
                    .collect(),
            ),
            ReadBatchParams::Ranges(_) => {
                return Err(Error::internal(
                    "ReadBatchParams::Ranges should not be used in v1 files".to_string(),
                ));
            }
            ReadBatchParams::RangeFull => {
                ReadBatchParams::Range(batch_offset..(batch_offset + rows_in_batch))
            }
            ReadBatchParams::RangeFrom(start) => {
                ReadBatchParams::Range((start.start + batch_offset)..(batch_offset + rows_in_batch))
            }
            ReadBatchParams::RangeTo(end) => {
                ReadBatchParams::Range(batch_offset..(end.end + batch_offset))
            }
            ReadBatchParams::Range(range) => {
                ReadBatchParams::Range((range.start + batch_offset)..(range.end + batch_offset))
            }
        };
        let result = lance_table::utils::stream::apply_row_id_and_deletes(
            result,
            0,
            self.fragment_id as u32,
            &RowIdAndDeletesConfig {
                params: file_params,
                deletion_vector: self.deletion_vec.clone(),
                row_id_sequence: self.row_id_sequence.clone(),
                with_row_id: self.with_row_id,
                with_row_addr: self.with_row_addr,
                with_row_last_updated_at_version: false,
                with_row_created_at_version: false,
                last_updated_at_sequence: None,
                created_at_sequence: None,
                make_deletions_null: self.make_deletions_null,
                total_num_rows: first_reader.len() as u32,
            },
        )?;

        let mut output_schema = ArrowSchema::from(projection);
        if self.with_row_id {
            output_schema = output_schema.try_with_column(ROW_ID_FIELD.clone())?;
        }
        if self.with_row_addr {
            output_schema = output_schema.try_with_column(ROW_ADDR_FIELD.clone())?;
        }
        Ok(result.project_by_schema(&output_schema)?)
    }
}

fn merge_batches(batches: &[RecordBatch]) -> Result<RecordBatch> {
    if batches.is_empty() {
        return Err(Error::invalid_input(
            "Cannot merge empty batches".to_string(),
        ));
    }

    let mut merged = batches[0].clone();
    for batch in batches.iter().skip(1) {
        merged = merged.merge(batch)?;
    }
    Ok(merged)
}
