// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The vectors of the rows a re-score scores, read out of the dataset's data
//! files: at their byte offsets where the page a row sits on stores its
//! values flat (see [`VectorLayout`]), and through take otherwise.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::num::NonZero;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock};

use arrow::compute::concat;
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, new_empty_array};
use arrow_schema::DataType;
use futures::{StreamExt, TryStreamExt, stream};
use lance_core::datatypes::parse_field_path;
use lance_core::utils::address::RowAddress;
use lance_file::version::ConcreteFileVersion;
use lance_io::local::to_local_path;
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{FileScheduler, IoStats, ScanScheduler, SchedulerConfig};
use lance_table::format::{DataFile, Fragment};
use object_store::path::Path;
use roaring::RoaringBitmap;

use super::layout::{CachedLayout, Decoded, LayoutKey, decode};
use super::local::{self, Descriptors, LocalReads, ReadContext, ReadNow};
use crate::dataset::fragment::FileFragment;
use crate::dataset::rowids::get_row_id_index;
use crate::dataset::{Dataset, ProjectionRequest};
use crate::{Error, Result};

/// The vectors of some of a dataset's rows, each beside the id of its row.
#[derive(Debug)]
pub struct Vectors {
    /// The live rows among those asked for, each once, in no particular
    /// order.
    pub row_ids: Vec<u64>,
    /// The vector of each row of `row_ids`, of the column's own type.
    pub vectors: ArrayRef,
    /// How many of the rows were read through take: those not read at their
    /// byte offsets, and those whose control word there was not zero.
    pub through_take: usize,
    /// How the reads of data files held open went (see [`local`]).
    pub local: LocalReads,
}

/// How [`read_vectors_with`] reads.
pub struct ReadOptions<'a> {
    /// The pool local data files are held open in.
    pub descriptors: &'a Arc<Descriptors>,
    /// How a range of a held file is read on the thread that asks.
    pub read_now: ReadNow,
    /// The most a call reads on the thread that asks, over all its files.
    pub in_place_budget: u64,
    /// Whether every data file is read through the scheduler, as remote ones
    /// are.
    pub through_scheduler: bool,
    /// Whether a read that finds its file not yet held waits until it is,
    /// so that the next read finds it held.
    pub await_open: bool,
}

impl Default for ReadOptions<'_> {
    fn default() -> Self {
        Self {
            descriptors: Descriptors::global(),
            read_now: local::read_now,
            in_place_budget: local::IN_PLACE_BYTES,
            through_scheduler: false,
            await_open: false,
        }
    }
}

/// The vectors `dataset` stores in `column` for the live rows among
/// `row_ids`.
///
/// A row is read at its byte offset when the column holds single-precision
/// vectors, the row's data file is of format 2.1 or later, its page stores
/// its values flat and no overlay gives it a newer value; every other row is
/// read through take, and so is a row whose control word says its vector, or
/// a struct around it, is null, which take reads as Lance does. Reads at byte
/// offsets are counted in `io_stats`: a local data file is read as [`local`]
/// reads one, any other through Lance's scheduler.
///
/// Rows that are deleted, or whose fragment is gone, are not read: a
/// compaction deletes rows without rewriting an index of stable row ids, and a
/// fragment can be deleted whole once it has left the index's bitmap.
pub async fn read_vectors(
    dataset: &Arc<Dataset>,
    column: &str,
    row_ids: &[u64],
    io_stats: Option<&IoStats>,
) -> Result<Vectors> {
    read_vectors_with(dataset, column, row_ids, io_stats, &ReadOptions::default()).await
}

/// [`read_vectors`], reading as `options` say.
pub async fn read_vectors_with(
    dataset: &Arc<Dataset>,
    column: &str,
    row_ids: &[u64],
    io_stats: Option<&IoStats>,
    options: &ReadOptions<'_>,
) -> Result<Vectors> {
    let field = dataset.schema().field(column).ok_or_else(|| {
        Error::invalid_input(format!(
            "IVF_VAMANA re-scores column {column}, which the dataset does not have"
        ))
    })?;
    let data_type = field.data_type();
    let mut row_ids = row_ids.to_vec();
    row_ids.sort_unstable();
    row_ids.dedup();
    let rows = live_rows(dataset, &row_ids).await?;
    let flat = match &data_type {
        DataType::FixedSizeList(item, dimension) if item.data_type() == &DataType::Float32 => {
            u32::try_from(*dimension)
                .ok()
                .map(|dimension| (item.clone(), dimension))
        }
        _ => None,
    };
    let mut read_ids = Vec::new();
    let mut left = Vec::new();
    let mut parts = Vec::with_capacity(2);
    let mut local = LocalReads::default();
    match flat {
        None => left.extend(rows.into_iter().map(|(row_id, _)| row_id)),
        Some((item, dimension)) => {
            let mut runs = Vec::new();
            for (fragment_id, run) in by_fragment(rows) {
                match FragmentFile::of(dataset, fragment_id, field.id)? {
                    Some(file) => runs.push((file, run)),
                    None => left.extend(run.into_iter().map(|(row_id, _)| row_id)),
                }
            }
            let mut bases = HashMap::new();
            for (file, _) in &runs {
                if let Entry::Vacant(entry) = bases.entry(file.data_file.base_id) {
                    let store = dataset.object_store_for_data_file(file.data_file).await?;
                    entry.insert(Base {
                        store,
                        scheduler: OnceLock::new(),
                    });
                }
            }
            let reading = Reading {
                dataset,
                dimension,
                io_stats,
                options,
                budget: AtomicU64::new(options.in_place_budget),
            };
            let reads = runs
                .into_iter()
                .map(|(file, run)| {
                    let base = &bases[&file.data_file.base_id];
                    read_run(&reading, base, file, run)
                })
                .collect::<Vec<_>>();
            let read = stream::iter(reads)
                .buffer_unordered(dataset.object_store.io_parallelism())
                .try_collect::<Vec<_>>()
                .await?;
            let mut values = Vec::new();
            for run in read {
                read_ids.extend(run.row_ids);
                values.extend(run.values);
                left.extend(run.left);
                local += run.local;
            }
            if !read_ids.is_empty() {
                let values = Arc::new(Float32Array::from(values));
                let vectors = FixedSizeListArray::try_new(item, dimension as i32, values, None)?;
                parts.push(Arc::new(vectors) as ArrayRef);
            }
        }
    }
    if !left.is_empty() {
        parts.push(take(dataset, column, &left).await?);
    }
    let through_take = left.len();
    read_ids.extend(left);
    let vectors = match parts.as_slice() {
        [] => new_empty_array(&data_type),
        [vectors] => vectors.clone(),
        _ => concat(&parts.iter().map(|part| part.as_ref()).collect::<Vec<_>>())?,
    };
    Ok(Vectors {
        row_ids: read_ids,
        vectors,
        through_take,
        local,
    })
}

/// A storage base whose data files a call reads: its store, and the
/// scheduler made for it the first time a file of it is read through one.
struct Base {
    store: Arc<ObjectStore>,
    scheduler: OnceLock<Arc<ScanScheduler>>,
}

impl Base {
    /// The data file `data_file` at `path`, opened through the base's
    /// scheduler, its reads counted in `io_stats`.
    async fn open(
        &self,
        path: &Path,
        data_file: &DataFile,
        io_stats: Option<&IoStats>,
    ) -> Result<FileScheduler> {
        let scheduler = self.scheduler.get_or_init(|| {
            ScanScheduler::new(
                self.store.clone(),
                SchedulerConfig::max_bandwidth(&self.store),
            )
        });
        let mut file_scheduler = scheduler
            .open_file(path, &data_file.file_size_bytes)
            .await?;
        if let Some(io_stats) = io_stats {
            file_scheduler = file_scheduler.with_io_stats(io_stats.recorder());
        }
        Ok(file_scheduler)
    }
}

/// What the reads of one call's fragments share.
struct Reading<'a> {
    dataset: &'a Arc<Dataset>,
    dimension: u32,
    io_stats: Option<&'a IoStats>,
    options: &'a ReadOptions<'a>,
    /// What the call may still read on the thread that asks.
    budget: AtomicU64,
}

/// The live rows among `row_ids`, each with its address.
async fn live_rows(dataset: &Dataset, row_ids: &[u64]) -> Result<Vec<(u64, RowAddress)>> {
    match get_row_id_index(dataset).await? {
        Some(index) => Ok(row_ids
            .iter()
            .zip(index.get_many(row_ids)?)
            .filter_map(|(row_id, address)| Some((*row_id, address?)))
            .collect()),
        None => Ok(dataset
            .filter_deleted_ids(row_ids)
            .await?
            .into_iter()
            .map(|row_id| (row_id, RowAddress::from(row_id)))
            .collect()),
    }
}

/// `rows` by fragment, each fragment's rows ascending by their offset in it.
fn by_fragment(mut rows: Vec<(u64, RowAddress)>) -> Vec<(u32, Vec<(u64, u32)>)> {
    rows.sort_unstable_by_key(|(row_id, address)| (u64::from(*address), *row_id));
    let mut runs: Vec<(u32, Vec<(u64, u32)>)> = Vec::new();
    for (row_id, address) in rows {
        let row = (row_id, address.row_offset());
        match runs.last_mut() {
            Some((fragment_id, run)) if *fragment_id == address.fragment_id() => run.push(row),
            _ => runs.push((address.fragment_id(), vec![row])),
        }
    }
    runs
}

/// What reading the vectors of one fragment at their byte offsets needs.
struct FragmentFile<'a> {
    fragment: &'a Fragment,
    /// The data file that holds the column.
    data_file: &'a DataFile,
    version: ConcreteFileVersion,
    /// The column's index among the data file's columns.
    column: u32,
    /// How many rows the data file holds.
    rows: u64,
    /// The rows an overlay gives a newer vector than the data file holds.
    overlaid: RoaringBitmap,
}

impl<'a> FragmentFile<'a> {
    /// How to read the column of field `field_id` out of fragment
    /// `fragment_id` at byte offsets, or `None` when its rows are read through
    /// take: its data file is of a format older than 2.1, or the manifest does
    /// not say how many rows it holds.
    fn of(dataset: &'a Dataset, fragment_id: u32, field_id: i32) -> Result<Option<Self>> {
        let fragment = dataset
            .find_fragment(u64::from(fragment_id))
            .ok_or_else(|| {
                Error::internal(format!(
                    "IVF_VAMANA re-scores a row of fragment {fragment_id}, which the dataset does \
                     not have"
                ))
            })?;
        let Some((data_file, position)) = fragment.files.iter().find_map(|file| {
            let position = file.fields.iter().position(|id| *id == field_id)?;
            Some((file, position))
        }) else {
            return Ok(None);
        };
        // Format 2.3 keeps its values on planar pages, which `layout` does not read.
        let Ok(version @ (ConcreteFileVersion::V2_1 | ConcreteFileVersion::V2_2)) =
            data_file.file_version()
        else {
            return Ok(None);
        };
        let column = data_file
            .column_indices
            .get(position)
            .and_then(|column| u32::try_from(*column).ok());
        let (Some(column), Some(rows)) = (column, fragment.physical_rows) else {
            return Ok(None);
        };
        let mut overlaid = RoaringBitmap::new();
        for overlay in &fragment.overlays {
            if let Some(position) = overlay
                .data_file
                .fields
                .iter()
                .position(|id| *id == field_id)
            {
                overlaid |= overlay.coverage_for_field(position)?.as_ref();
            }
        }
        Ok(Some(Self {
            fragment,
            data_file,
            version,
            column,
            rows: rows as u64,
            overlaid,
        }))
    }
}

/// The vectors of one fragment's rows, read at their byte offsets.
#[derive(Default)]
struct RunVectors {
    /// The rows whose vectors were read at their byte offsets.
    row_ids: Vec<u64>,
    /// Their vectors' values, one vector after another.
    values: Vec<f32>,
    /// The rows left to take: those an overlay covers, those on a page that
    /// does not store its values flat, those whose control word is not zero,
    /// or all of them when the data file's pages do not add up to its rows.
    left: Vec<u64>,
    /// How the reads went, if the data file was held open.
    local: LocalReads,
}

impl RunVectors {
    /// Keeps the vectors `decoded` read for `row_ids`, in their order, and
    /// leaves to take the rows whose control word is not zero.
    fn keep(&mut self, decoded: Decoded) -> Result<()> {
        if decoded.has_vector.len() != self.row_ids.len() {
            return Err(Error::internal(format!(
                "IVF_VAMANA decoded {} vectors for {} rows",
                decoded.has_vector.len(),
                self.row_ids.len()
            )));
        }
        let row_ids = std::mem::replace(
            &mut self.row_ids,
            Vec::with_capacity(decoded.has_vector.len()),
        );
        for (row_id, has_vector) in row_ids.into_iter().zip(decoded.has_vector) {
            if has_vector {
                self.row_ids.push(row_id);
            } else {
                self.left.push(row_id);
            }
        }
        self.values = decoded.values;
        Ok(())
    }
}

/// The vectors of `run`, rows of `file`'s fragment ascending by offset, read
/// at their byte offsets where they can be.
///
/// The column's layout comes from the session's cache, where the first read
/// of the file puts it, and is used only while the manifest agrees with what
/// it was built for. A local data file the pool holds is read as [`local`]
/// reads one; any other file is read through the base's scheduler, and a
/// local one is opened in the background to be held for the reads after.
async fn read_run(
    reading: &Reading<'_>,
    base: &Base,
    file: FragmentFile<'_>,
    run: Vec<(u64, u32)>,
) -> Result<RunVectors> {
    let dataset = reading.dataset;
    let path = dataset
        .data_file_dir(file.data_file)?
        .join(file.data_file.path.as_str());
    let layouts = dataset.metadata_cache.file_metadata_cache(&path);
    let key = LayoutKey {
        column: file.column,
    };
    let mut opened = None;
    let cached = match layouts.get_with_key(&key).await {
        Some(cached) => cached,
        None => {
            let file_scheduler = base.open(&path, file.data_file, reading.io_stats).await?;
            let metadata = FileFragment::new(dataset.clone(), file.fragment.clone())
                .get_file_metadata(&file_scheduler)
                .await?;
            let cached = Arc::new(CachedLayout::of(
                &metadata,
                file.column,
                reading.dimension,
                file.rows,
            ));
            layouts.insert_with_key(&key, cached.clone()).await;
            opened = Some(file_scheduler);
            cached
        }
    };
    let mut read = RunVectors::default();
    let Some(layout) = cached.layout_for(file.version, file.rows, reading.dimension) else {
        read.left = run.into_iter().map(|(row_id, _)| row_id).collect();
        return Ok(read);
    };
    let (overlaid, run): (Vec<_>, Vec<_>) = run
        .into_iter()
        .partition(|(_, offset)| file.overlaid.contains(*offset));
    read.left
        .extend(overlaid.into_iter().map(|(row_id, _)| row_id));
    let offsets = run.iter().map(|(_, offset)| *offset).collect::<Vec<_>>();
    let mut ranges = Vec::with_capacity(offsets.len());
    for ((row_id, _), range) in run.into_iter().zip(layout.locate(&offsets)?) {
        match range {
            Some(range) => {
                read.row_ids.push(row_id);
                ranges.push(range);
            }
            None => read.left.push(row_id),
        }
    }
    if ranges.is_empty() {
        return Ok(read);
    }
    let options = reading.options;
    let size = file.data_file.file_size_bytes.get().map(NonZero::get);
    // A file no larger than a block Lance reads whole in one request, and
    // counts so; such a file holds a few vectors at most.
    let small = size.is_some_and(|size| size <= base.store.block_size() as u64);
    if !options.through_scheduler && !small && local::reads_in_place(&base.store) {
        let local_path = to_local_path(&path);
        let descriptors = options.descriptors;
        match descriptors.get(&local_path) {
            // The size the manifest records is the file's for good, so a file
            // held under the path with another size is not the one it names.
            Some(held) if size.is_none_or(|size| size == held.size()) => {
                let context = ReadContext {
                    store: &base.store,
                    path: &path,
                    io_stats: reading.io_stats,
                    pool: descriptors,
                    read_now: options.read_now,
                    budget: &reading.budget,
                };
                let bytes = local::read(held, &ranges, &context).await?;
                read.keep(decode(bytes.chunks(), reading.dimension)?)?;
                read.local = bytes.reads;
                return Ok(read);
            }
            Some(_) => descriptors.forget(&local_path),
            None if descriptors.elsewhere(&local_path) => {}
            None => {
                if let Some(opening) = descriptors.open_in_background(local_path)
                    && options.await_open
                {
                    opening.await.map_err(|error| {
                        Error::io(format!(
                            "IVF_VAMANA failed to open data file {path} to hold it: {error}"
                        ))
                    })?;
                }
            }
        }
    }
    let file_scheduler = match opened {
        Some(file_scheduler) => file_scheduler,
        None => base.open(&path, file.data_file, reading.io_stats).await?,
    };
    let bytes = file_scheduler.submit_request(ranges, 0).await?;
    read.keep(decode(
        bytes.iter().map(|chunk| chunk.as_ref()),
        reading.dimension,
    )?)?;
    Ok(read)
}

/// The vectors of `row_ids` read through take, in their order.
async fn take(dataset: &Arc<Dataset>, column: &str, row_ids: &[u64]) -> Result<ArrayRef> {
    let projection = Arc::new(dataset.schema().project(&[column])?);
    let batch = dataset
        .take_builder(row_ids, ProjectionRequest::Schema(projection))?
        .execute()
        .await?;
    if batch.num_rows() != row_ids.len() {
        return Err(Error::internal(format!(
            "IVF_VAMANA read {} of the {} live candidates it rescores from column {column}",
            batch.num_rows(),
            row_ids.len()
        )));
    }
    // The column may be a path into a struct, its names quoted as a field
    // path quotes them.
    let path = parse_field_path(column)?;
    let mut vectors = path
        .first()
        .and_then(|root| batch.column_by_name(root))
        .cloned();
    for name in path.iter().skip(1) {
        vectors = vectors.and_then(|parent| {
            parent
                .as_struct_opt()
                .and_then(|parent| parent.column_by_name(name))
                .cloned()
        });
    }
    vectors.ok_or_else(|| Error::internal(format!("take returned no column {column}")))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use futures::FutureExt;

    use arrow_array::types::{Float32Type, Int32Type, UInt64Type};
    use arrow_array::{
        Array, Int32Array, RecordBatch, RecordBatchIterator, StructArray, UInt64Array,
    };
    use arrow_buffer::NullBuffer;
    use arrow_schema::Field;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_ADDR, ROW_ID};
    use lance_datagen::{BatchCount, RowCount, array, gen_batch};
    use lance_file::version::LanceFileVersion;
    use lance_file::writer::FileWriterOptions;
    use lance_table::format::BasePath;
    use lance_table::format::overlay::OverlayCoverage;
    use rstest::rstest;

    use super::*;
    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::transaction::Operation;
    use crate::dataset::{NewColumnTransform, WriteDestination, WriteParams};
    use crate::session::Session;

    /// Rows each test dataset holds.
    const ROWS: usize = 200;
    /// Rows of both fragments of a dataset of two, numbered by `id`, out of
    /// order.
    const ASKED: [i32; 6] = [150, 3, 99, 100, 64, 199];

    /// A test dataset of `ROWS` rows numbered by `id` and a column of random
    /// vectors, written in batches of 32 rows to a page cache that each fills
    /// at 128 values, so that such vectors are written 32 to a page.
    struct Fixture {
        /// The vector column's name.
        name: &'static str,
        dimension: u32,
        element: DataType,
        /// The row whose vector is null, if any.
        null: Option<usize>,
        rows_per_file: usize,
        params: WriteParams,
    }

    impl Default for Fixture {
        fn default() -> Self {
            Self {
                name: "vec",
                dimension: 128,
                element: DataType::Float32,
                null: None,
                rows_per_file: ROWS / 2,
                params: WriteParams::default(),
            }
        }
    }

    impl Fixture {
        async fn write(self, uri: &str) -> Dataset {
            let batch = gen_batch()
                .col("id", array::step::<Int32Type>())
                .col(
                    self.name,
                    array::rand_vec::<Float32Type>(self.dimension.into()),
                )
                .into_batch_rows(RowCount::from(ROWS as u64))
                .unwrap();
            let DataType::FixedSizeList(item, _) = batch[self.name].data_type() else {
                unreachable!()
            };
            let item = Arc::new(Field::new(item.name(), self.element, item.is_nullable()));
            let values = arrow::compute::cast(
                batch[self.name].as_fixed_size_list().values(),
                item.data_type(),
            )
            .unwrap();
            let nulls = self
                .null
                .map(|row| NullBuffer::from_iter((0..ROWS).map(|other| other != row)));
            let vectors =
                FixedSizeListArray::try_new(item, self.dimension as i32, values, nulls).unwrap();
            let batch = RecordBatch::try_from_iter([
                ("id", batch["id"].clone()),
                (self.name, Arc::new(vectors) as ArrayRef),
            ])
            .unwrap();
            let schema = batch.schema();
            let batches = (0..ROWS)
                .step_by(32)
                .map(|start| Ok(batch.slice(start, 32.min(ROWS - start))))
                .collect::<Vec<_>>();
            let params = WriteParams {
                max_rows_per_file: self.rows_per_file,
                file_writer_options: Some(FileWriterOptions {
                    data_cache_bytes: Some(2 * 16 * 1024),
                    ..Default::default()
                }),
                ..self.params
            };
            Dataset::write(RecordBatchIterator::new(batches, schema), uri, Some(params))
                .await
                .unwrap()
        }
    }

    /// The row ids of the rows numbered `ids`, in that order.
    async fn row_ids(dataset: &Dataset, ids: &[i32]) -> Vec<u64> {
        let mut scan = dataset.scan();
        scan.with_row_id().project(&["id"]).unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        let by_id = batch["id"]
            .as_primitive::<Int32Type>()
            .values()
            .iter()
            .copied()
            .zip(
                batch[ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .values()
                    .iter()
                    .copied(),
            )
            .collect::<HashMap<_, _>>();
        ids.iter().map(|id| by_id[id]).collect()
    }

    fn sorted(mut row_ids: Vec<u64>) -> Vec<u64> {
        row_ids.sort_unstable();
        row_ids
    }

    /// The rows [`read_vectors`] reads of `row_ids` out of `column`, sorted,
    /// and how many of them it read through take, each vector checked
    /// against what take reads for its row.
    async fn read_column(
        dataset: &Arc<Dataset>,
        column: &str,
        row_ids: &[u64],
    ) -> (Vec<u64>, usize) {
        let read = read_checked(dataset, column, row_ids, &ReadOptions::default()).await;
        (sorted(read.row_ids), read.through_take)
    }

    /// What [`read_vectors_with`] reads of `row_ids` out of `column` as
    /// `options` say, each vector checked against what take reads for its
    /// row.
    async fn read_checked(
        dataset: &Arc<Dataset>,
        column: &str,
        row_ids: &[u64],
        options: &ReadOptions<'_>,
    ) -> Vectors {
        let read = read_vectors_with(dataset, column, row_ids, None, options)
            .await
            .unwrap();
        assert_eq!(read.row_ids.len(), read.vectors.len());
        let projection = Arc::new(dataset.schema().project(&[column]).unwrap());
        let taken = dataset
            .take_builder(&read.row_ids, ProjectionRequest::Schema(projection))
            .unwrap()
            .execute()
            .await
            .unwrap();
        assert_eq!(read.vectors.to_data(), taken.column(0).to_data());
        read
    }

    /// A pool of its own for a test, which holds a file on any filesystem.
    fn pool() -> Arc<Descriptors> {
        fn open_any(path: &str) -> std::io::Result<Option<std::fs::File>> {
            std::fs::File::open(path).map(Some)
        }
        Descriptors::new(
            64,
            Duration::from_secs(60),
            Duration::from_secs(10),
            open_any,
        )
    }

    /// Reading through `pool`, a file not yet held is read only once it is,
    /// so that the next read finds it held.
    fn held_through(pool: &Arc<Descriptors>) -> ReadOptions<'_> {
        ReadOptions {
            descriptors: pool,
            await_open: true,
            ..Default::default()
        }
    }

    /// Whether this platform and environment read local data files through
    /// held files.
    fn holds_files() -> bool {
        cfg!(all(target_os = "linux", target_pointer_width = "64")) && local::switched_on()
    }

    /// [`read_column`] of the column `vec`.
    async fn read(dataset: &Arc<Dataset>, row_ids: &[u64]) -> (Vec<u64>, usize) {
        read_column(dataset, "vec", row_ids).await
    }

    /// How many pages the vectors of the dataset's first fragment span.
    async fn pages(dataset: &Arc<Dataset>) -> usize {
        let fragment = &dataset.get_fragments()[0];
        let file = &fragment.metadata().files[0];
        let path = dataset
            .data_file_dir(file)
            .unwrap()
            .join(file.path.as_str());
        let store = dataset.object_store_for_data_file(file).await.unwrap();
        let file_scheduler =
            ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store))
                .open_file(&path, &file.file_size_bytes)
                .await
                .unwrap();
        let metadata = fragment.get_file_metadata(&file_scheduler).await.unwrap();
        metadata.column_infos[1].page_infos.len()
    }

    async fn commit(dataset: Arc<Dataset>, operation: Operation) -> Arc<Dataset> {
        let read_version = dataset.version().version;
        Arc::new(
            Dataset::commit(
                WriteDestination::Dataset(dataset),
                operation,
                Some(read_version),
                None,
                None,
                Arc::new(Default::default()),
                false,
            )
            .await
            .unwrap(),
        )
    }

    /// Vectors on pages that store them flat are read at their byte offsets,
    /// out of order, across fragments and pages, a row asked for twice once,
    /// in every file format that lays them out so, with row ids stable or not,
    /// from memory and from local files, which the second read reads through
    /// the files held open.
    #[rstest]
    #[tokio::test]
    async fn vectors_on_flat_pages_are_read_at_their_offsets(
        #[values(LanceFileVersion::V2_1, LanceFileVersion::V2_2)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] enable_stable_row_ids: bool,
        #[values(false, true)] local: bool,
    ) {
        let params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids,
            ..Default::default()
        };
        let fixture = Fixture {
            params,
            ..Default::default()
        };
        let dir = TempStrDir::default();
        let uri = if local { dir.as_str() } else { "memory://" };
        let dataset = Arc::new(fixture.write(uri).await);
        assert!(pages(&dataset).await >= 3);
        let asked = row_ids(&dataset, &ASKED).await;
        let twice = [asked.as_slice(), &asked[..2]].concat();
        let pool = pool();
        for held in [false, local && holds_files()] {
            let read = read_checked(&dataset, "vec", &twice, &held_through(&pool)).await;
            assert_eq!(
                (sorted(read.row_ids), read.through_take),
                (sorted(asked.clone()), 0)
            );
            assert_eq!(read.local.in_place + read.local.handed_off > 0, held);
        }
        assert_eq!(read(&dataset, &[]).await, (vec![], 0));
    }

    /// A local data file read again is read in place, without a trip to the
    /// blocking pool, when the page cache serves the file; when it does not,
    /// on the blocking pool.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[tokio::test]
    async fn a_local_file_read_again_is_read_in_place() {
        let dir = TempStrDir::default();
        let dataset = Arc::new(Fixture::default().write(dir.as_str()).await);
        let asked = row_ids(&dataset, &ASKED).await;
        let pool = pool();
        let first = read_checked(&dataset, "vec", &asked, &held_through(&pool)).await;
        assert_eq!(first.local, LocalReads::default());
        // Asked before the read it judges, which would put back the pages a
        // refused read goes on to read.
        let served = page_cache_serves(&dataset);
        let again = read_checked(&dataset, "vec", &asked, &held_through(&pool)).await;
        if !holds_files() {
            assert_eq!(again.local, LocalReads::default());
            return;
        }
        // The six vectors lie apart, three in each file, each read alone.
        assert_eq!(again.local.in_place + again.local.handed_off, ASKED.len());
        assert!(again.local.trips <= 2, "{:?}", again.local);
        if served {
            assert_eq!(again.local.in_place, ASKED.len(), "{:?}", again.local);
        } else {
            assert!(again.local.handed_off > 0, "{:?}", again.local);
        }
    }

    /// Whether the page cache serves every data file of `dataset` whole,
    /// asked of the kernel directly.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    fn page_cache_serves(dataset: &Dataset) -> bool {
        dataset.get_fragments().iter().all(|fragment| {
            let file = &fragment.metadata().files[0];
            let path = dataset
                .data_file_dir(file)
                .unwrap()
                .join(file.path.as_str());
            let path = to_local_path(&path);
            let len = std::fs::metadata(&path).unwrap().len() as usize;
            local::page_cache_serves(&path, len)
        })
    }

    /// The budget for reading in place is the call's: once one file has
    /// read it up, the other file's reads go to the blocking pool too.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[tokio::test]
    async fn the_budget_for_reading_in_place_is_the_calls() {
        let dir = TempStrDir::default();
        let dataset = Arc::new(Fixture::default().write(dir.as_str()).await);
        let asked = row_ids(&dataset, &ASKED).await;
        let pool = pool();
        read_checked(&dataset, "vec", &asked, &held_through(&pool)).await;
        // One vector's worth: of the six vectors, each read alone, one fits.
        let options = ReadOptions {
            in_place_budget: 128 * 4,
            ..held_through(&pool)
        };
        let served = page_cache_serves(&dataset);
        let read = read_checked(&dataset, "vec", &asked, &options).await;
        if !holds_files() {
            assert_eq!(read.local, LocalReads::default());
        } else if served {
            let reads = LocalReads {
                in_place: 1,
                handed_off: ASKED.len() - 1,
                trips: 2,
            };
            assert_eq!(read.local, reads);
        } else {
            assert_eq!(read.local.in_place, 0, "{:?}", read.local);
        }
    }

    /// A local read once its files are held and their layouts cached waits
    /// for nothing: polled once outside any runtime, where an open through
    /// Lance, a scheduler or a trip to the blocking pool would panic, it is
    /// done.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[test]
    fn a_warm_local_read_waits_for_nothing() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let dir = TempStrDir::default();
        let pool = pool();
        let (dataset, asked) = runtime.block_on(async {
            let dataset = Arc::new(Fixture::default().write(dir.as_str()).await);
            let asked = row_ids(&dataset, &ASKED).await;
            read_checked(&dataset, "vec", &asked, &held_through(&pool)).await;
            (dataset, asked)
        });
        if !holds_files() || !page_cache_serves(&dataset) {
            return;
        }
        let read = read_vectors_with(&dataset, "vec", &asked, None, &held_through(&pool))
            .now_or_never()
            .expect("a warm local read waits for nothing")
            .unwrap();
        assert_eq!(
            (sorted(read.row_ids), read.through_take),
            (sorted(asked), 0)
        );
        assert_eq!(read.local.in_place, ASKED.len());
    }

    /// A local data file read through the files held open is counted as the
    /// scheduler counts it, in the query's statistics and in the store's
    /// tracker, rows read together and rows read alone.
    #[tokio::test]
    async fn local_reads_are_counted_as_the_scheduler_counts_them() {
        let dir = TempStrDir::default();
        Fixture::default().write(dir.as_str()).await;
        // A session of its own gives the dataset a store of its own, whose
        // tracker no other test reads through.
        let dataset = DatasetBuilder::from_uri(dir.as_str())
            .with_session(Arc::new(Session::default()))
            .load()
            .await
            .unwrap();
        let dataset = Arc::new(dataset);
        // Rows 3, 4 and 5 lie side by side and are read at once; rows 64 and
        // 150 are read alone, the last from the other fragment's file.
        let asked = row_ids(&dataset, &[3, 4, 5, 64, 150]).await;
        let pool = pool();
        let held = held_through(&pool);
        read_checked(&dataset, "vec", &asked, &held).await;
        let scheduled = ReadOptions {
            descriptors: &pool,
            through_scheduler: true,
            ..Default::default()
        };
        let store = dataset
            .object_store_for_data_file(&dataset.get_fragments()[0].metadata().files[0])
            .await
            .unwrap();
        let mut counts = Vec::new();
        for options in [&scheduled, &held] {
            store.io_tracker().incremental_stats();
            let io_stats = IoStats::new();
            let read = read_vectors_with(&dataset, "vec", &asked, Some(&io_stats), options)
                .await
                .unwrap();
            let tracked = store.io_tracker().incremental_stats();
            let stats = io_stats.snapshot();
            counts.push((
                (stats.requests, stats.iops, stats.bytes_read),
                (tracked.read_iops, tracked.read_bytes),
                read.local,
            ));
        }
        let (scheduled_stats, scheduled_tracked, scheduled_local) = counts[0];
        let (held_stats, held_tracked, held_local) = counts[1];
        assert_eq!(scheduled_stats, (2, 3, 5 * 128 * 4));
        assert_eq!(held_stats, scheduled_stats);
        assert_eq!(held_tracked, scheduled_tracked);
        assert_eq!(scheduled_tracked, (3, 5 * 128 * 4));
        assert_eq!(scheduled_local, LocalReads::default());
        assert_eq!(
            held_local.in_place + held_local.handed_off > 0,
            holds_files()
        );
    }

    /// A local data file none of whose pages stores its vectors flat is never
    /// opened to be held: all its rows are read through take.
    #[tokio::test]
    async fn a_local_file_no_page_stores_flat_is_never_held() {
        let dir = TempStrDir::default();
        let fixture = Fixture {
            dimension: 32,
            ..Default::default()
        };
        let dataset = Arc::new(fixture.write(dir.as_str()).await);
        let asked = row_ids(&dataset, &ASKED).await;
        let pool = pool();
        for _ in 0..2 {
            let read = read_checked(&dataset, "vec", &asked, &held_through(&pool)).await;
            assert_eq!(read.through_take, ASKED.len());
        }
        pool.settle();
        assert_eq!(pool.held_files(), 0);
    }

    /// The vectors of many fragments read at once each stay beside their
    /// own row.
    #[tokio::test]
    async fn vectors_of_many_fragments_stay_beside_their_rows() {
        let fixture = Fixture {
            rows_per_file: 13,
            ..Default::default()
        };
        let dataset = Arc::new(fixture.write("memory://").await);
        assert_eq!(dataset.get_fragments().len(), 16);
        let ids = (0..ROWS as i32).rev().collect::<Vec<_>>();
        let asked = row_ids(&dataset, &ids).await;
        assert_eq!(read(&dataset, &asked).await, (sorted(asked), 0));
    }

    /// Vectors on another base of the dataset are read at their byte offsets
    /// from that base, the second time through the files held open.
    #[tokio::test]
    async fn vectors_on_another_base_are_read_at_their_offsets() {
        let (primary, base) = (TempStrDir::default(), TempStrDir::default());
        let params = WriteParams {
            initial_bases: Some(vec![BasePath::new(1, base.to_string(), None, false)]),
            target_bases: Some(vec![1]),
            ..Default::default()
        };
        let fixture = Fixture {
            params,
            ..Default::default()
        };
        let dataset = Arc::new(fixture.write(primary.as_str()).await);
        assert!(
            dataset
                .get_fragments()
                .iter()
                .all(|fragment| fragment.metadata().files[0].base_id == Some(1))
        );
        let asked = row_ids(&dataset, &ASKED).await;
        let pool = pool();
        for held in [false, holds_files()] {
            let read = read_checked(&dataset, "vec", &asked, &held_through(&pool)).await;
            assert_eq!(
                (sorted(read.row_ids), read.through_take),
                (sorted(asked.clone()), 0)
            );
            assert_eq!(read.local.in_place + read.local.handed_off > 0, held);
        }
    }

    /// A vector column added after the rows were written lives in a data
    /// file of its own in each fragment, and is read from that file at byte
    /// offsets.
    #[tokio::test]
    async fn vectors_of_a_column_added_later_are_read_at_their_offsets() {
        let mut dataset = Fixture::default().write("memory://").await;
        let added = gen_batch()
            .col("added", array::rand_vec::<Float32Type>(128.into()))
            .into_reader_rows(RowCount::from(ROWS as u64), BatchCount::from(1));
        dataset
            .add_columns(NewColumnTransform::Reader(Box::new(added)), None, None)
            .await
            .unwrap();
        let dataset = Arc::new(dataset);
        assert!(
            dataset
                .get_fragments()
                .iter()
                .all(|fragment| fragment.metadata().files.len() == 2)
        );
        let asked = row_ids(&dataset, &ASKED).await;
        assert_eq!(
            read_column(&dataset, "added", &asked).await,
            (sorted(asked), 0)
        );
    }

    /// A column whose name a field path quotes is read from its quoted path,
    /// at byte offsets or through take.
    #[rstest]
    #[case::at_offsets(128, 0)]
    #[case::through_take(32, ASKED.len())]
    #[tokio::test]
    async fn a_column_named_in_quotes_is_read(#[case] dimension: u32, #[case] through_take: usize) {
        let fixture = Fixture {
            name: "my-vec",
            dimension,
            ..Default::default()
        };
        let dataset = Arc::new(fixture.write("memory://").await);
        let asked = row_ids(&dataset, &ASKED).await;
        assert_eq!(
            read_column(&dataset, "`my-vec`", &asked).await,
            (sorted(asked), through_take)
        );
    }

    /// Vectors no page stores flat are read through take: narrow ones, which
    /// Lance lays out in mini-blocks, half-precision ones, those of a file
    /// older than format 2.1, and those of format 2.3, which lays them out on
    /// planar pages. Only a data file of format 2.1 or 2.2 holding
    /// single-precision vectors is opened to find that out: on a first read,
    /// its metadata is read, and no other file's is.
    #[rstest]
    #[case::narrow(32, DataType::Float32, LanceFileVersion::V2_2, true)]
    #[case::half_precision(128, DataType::Float16, LanceFileVersion::V2_2, false)]
    #[case::format_2_3(128, DataType::Float32, LanceFileVersion::V2_3, false)]
    #[case::format_2_0(128, DataType::Float32, LanceFileVersion::V2_0, false)]
    #[case::legacy_format(128, DataType::Float32, LanceFileVersion::Legacy, false)]
    #[tokio::test]
    async fn vectors_no_page_stores_flat_are_read_through_take(
        #[case] dimension: u32,
        #[case] element: DataType,
        #[case] data_storage_version: LanceFileVersion,
        #[case] opened: bool,
    ) {
        let params = WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let fixture = Fixture {
            dimension,
            element,
            params,
            ..Default::default()
        };
        // Addressed rather than scanned for: a scan would read the files'
        // metadata ahead of the read counted here.
        let asked = ASKED
            .iter()
            .map(|&id| {
                let id = id as usize;
                let fragment_id = id / fixture.rows_per_file;
                let offset = id % fixture.rows_per_file;
                u64::from(RowAddress::new_from_parts(
                    fragment_id as u32,
                    offset as u32,
                ))
            })
            .collect::<Vec<_>>();
        let dataset = Arc::new(fixture.write("memory://").await);
        let io_stats = IoStats::new();
        read_vectors(&dataset, "vec", &asked, Some(&io_stats))
            .await
            .unwrap();
        assert_eq!(io_stats.snapshot().requests > 0, opened);
        assert_eq!(asked, row_ids(&dataset, &ASKED).await);
        assert_eq!(read(&dataset, &asked).await, (sorted(asked), ASKED.len()));
    }

    /// A file held under a data file's path with another size than the
    /// manifest records is not the file it names: it is let go, and the rows
    /// are read through the scheduler.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[tokio::test]
    async fn a_held_file_of_another_size_is_let_go() {
        fn open_another(_path: &str) -> std::io::Result<Option<std::fs::File>> {
            // procfs gives its files no size.
            std::fs::File::open("/proc/self/stat").map(Some)
        }
        let dir = TempStrDir::default();
        let dataset = Arc::new(Fixture::default().write(dir.as_str()).await);
        let asked = row_ids(&dataset, &ASKED).await;
        let pool = Descriptors::new(
            64,
            Duration::from_secs(60),
            Duration::from_secs(10),
            open_another,
        );
        for _ in 0..2 {
            let read = read_checked(&dataset, "vec", &asked, &held_through(&pool)).await;
            assert_eq!((read.through_take, read.local), (0, LocalReads::default()));
        }
        pool.settle();
        assert_eq!(pool.held_files(), 0);
    }

    /// A data file no larger than a block, which Lance reads whole in one
    /// request, is read through the scheduler and never held.
    #[tokio::test]
    async fn a_data_file_no_larger_than_a_block_is_never_held() {
        let dir = TempStrDir::default();
        let data = gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vec", array::rand_vec::<Float32Type>(128.into()))
            .into_reader_rows(RowCount::from(2), BatchCount::from(1));
        let dataset = Dataset::write(data, dir.as_str(), None).await.unwrap();
        let dataset = Arc::new(dataset);
        let block = dataset.object_store.block_size() as u64;
        let fragments = dataset.get_fragments();
        let file = &fragments[0].metadata().files[0];
        let size = file.file_size_bytes.get().map(NonZero::get);
        assert!(size.is_some_and(|size| size <= block), "{size:?}");
        let asked = row_ids(&dataset, &[0, 1]).await;
        let pool = pool();
        for _ in 0..2 {
            let read = read_checked(&dataset, "vec", &asked, &held_through(&pool)).await;
            assert_eq!((read.through_take, read.local), (0, LocalReads::default()));
        }
        pool.settle();
        assert_eq!(pool.held_files(), 0);
    }

    static OPENED_ELSEWHERE: AtomicUsize = AtomicUsize::new(0);

    /// Data files on a filesystem off the list are read through the
    /// scheduler, and their directory, once found there, is not opened again
    /// even after the back-off.
    #[tokio::test]
    async fn files_off_the_listed_filesystems_are_not_opened_again() {
        fn open_elsewhere(_path: &str) -> std::io::Result<Option<std::fs::File>> {
            OPENED_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
            Ok(None)
        }
        let dir = TempStrDir::default();
        let dataset = Arc::new(Fixture::default().write(dir.as_str()).await);
        let asked = row_ids(&dataset, &ASKED).await;
        let pool = Descriptors::new(
            64,
            Duration::from_secs(60),
            Duration::from_millis(1),
            open_elsewhere,
        );
        let read = read_checked(&dataset, "vec", &asked, &held_through(&pool)).await;
        assert_eq!(read.local, LocalReads::default());
        let opened = OPENED_ELSEWHERE.load(Ordering::Relaxed);
        assert_eq!(opened > 0, holds_files());
        std::thread::sleep(Duration::from_millis(20));
        for _ in 0..2 {
            let read = read_checked(&dataset, "vec", &asked, &held_through(&pool)).await;
            assert_eq!(read.local, LocalReads::default());
        }
        assert_eq!(OPENED_ELSEWHERE.load(Ordering::Relaxed), opened);
    }

    static GATE_OPEN: AtomicBool = AtomicBool::new(false);

    /// A query that misses on a file does not wait for the file to be opened
    /// to be held: it reads through the scheduler meanwhile.
    #[tokio::test]
    async fn a_query_does_not_wait_for_its_files_to_be_held() {
        fn open_behind_the_gate(path: &str) -> std::io::Result<Option<std::fs::File>> {
            let waited = std::time::Instant::now();
            while !GATE_OPEN.load(Ordering::Relaxed) && waited.elapsed() < Duration::from_secs(5) {
                std::thread::sleep(Duration::from_millis(5));
            }
            std::fs::File::open(path).map(Some)
        }
        let dir = TempStrDir::default();
        let dataset = Arc::new(Fixture::default().write(dir.as_str()).await);
        let asked = row_ids(&dataset, &ASKED).await;
        let pool = Descriptors::new(
            64,
            Duration::from_secs(60),
            Duration::from_secs(10),
            open_behind_the_gate,
        );
        let options = ReadOptions {
            descriptors: &pool,
            ..Default::default()
        };
        let read = tokio::time::timeout(
            Duration::from_millis(500),
            read_vectors_with(&dataset, "vec", &asked, None, &options),
        )
        .await;
        GATE_OPEN.store(true, Ordering::Relaxed);
        let read = read
            .expect("the query waited for its files to be held")
            .unwrap();
        assert_eq!(sorted(read.row_ids), sorted(asked));
    }

    /// Two vector columns of one data file are each read from their own
    /// pages, through the one file held for both.
    #[tokio::test]
    async fn two_vector_columns_of_one_file_are_each_read_from_their_own_pages() {
        let dir = TempStrDir::default();
        let data = gen_batch()
            .col("id", array::step::<Int32Type>())
            .col("vec", array::rand_vec::<Float32Type>(128.into()))
            .col("vec2", array::rand_vec::<Float32Type>(128.into()))
            .into_reader_rows(RowCount::from(100), BatchCount::from(1));
        let dataset = Dataset::write(data, dir.as_str(), None).await.unwrap();
        let dataset = Arc::new(dataset);
        assert_eq!(dataset.get_fragments()[0].metadata().files.len(), 1);
        let asked = row_ids(&dataset, &[3, 40, 99]).await;
        let pool = pool();
        for _ in 0..2 {
            let first = read_checked(&dataset, "vec", &asked, &held_through(&pool)).await;
            let second = read_checked(&dataset, "vec2", &asked, &held_through(&pool)).await;
            assert_eq!((first.through_take, second.through_take), (0, 0));
            assert_ne!(first.vectors.to_data(), second.vectors.to_data());
        }
        pool.settle();
        assert_eq!(pool.held_files(), u64::from(holds_files()));
    }

    /// A data file the manifest records as of another format than its footer
    /// says is read through take, which reads it as Lance always does, even
    /// once its layout is cached from a read under a manifest that agreed.
    #[tokio::test]
    async fn a_data_file_its_manifest_misdates_is_read_through_take() {
        let dataset = Arc::new(Fixture::default().write("memory://").await);
        let asked = row_ids(&dataset, &ASKED).await;
        assert_eq!(read(&dataset, &asked).await, (sorted(asked.clone()), 0));
        let mut fragments = dataset.fragments().as_ref().clone();
        for fragment in &mut fragments {
            for file in &mut fragment.files {
                file.file_minor_version = 1;
            }
        }
        let overwrite = Operation::Overwrite {
            fragments,
            schema: dataset.schema().clone(),
            config_upsert_values: None,
            initial_bases: None,
        };
        let dataset = commit(dataset, overwrite).await;
        assert!(dataset.get_fragments().iter().all(|fragment| {
            fragment.metadata().files[0].file_version().unwrap() == ConcreteFileVersion::V2_1
        }));
        // The overwrite numbers the fragments anew, over the same files.
        let asked = row_ids(&dataset, &ASKED).await;
        assert_eq!(read(&dataset, &asked).await, (sorted(asked), ASKED.len()));
    }

    /// The rows of a page that holds a null vector are read at their byte
    /// offsets with the control word in front of each, in every file format
    /// that lays them out so, from memory and from local files, which the
    /// second read reads through the files held open. The null vector itself
    /// is read through take, which gives it as Lance does. Every row of the
    /// fragment is asked for, so that the null's page is among those read
    /// wherever its bounds fall.
    #[rstest]
    #[tokio::test]
    async fn rows_of_a_page_with_a_null_are_read_at_their_offsets(
        #[values(LanceFileVersion::V2_1, LanceFileVersion::V2_2)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] local: bool,
    ) {
        let params = WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let fixture = Fixture {
            null: Some(5),
            params,
            ..Default::default()
        };
        let dir = TempStrDir::default();
        let uri = if local { dir.as_str() } else { "memory://" };
        let dataset = Arc::new(fixture.write(uri).await);
        // The first fragment's rows, and one of the second's.
        let ids = (0..ROWS as i32 / 2).chain([150]).collect::<Vec<_>>();
        let asked = row_ids(&dataset, &ids).await;
        let pool = pool();
        for held in [false, local && holds_files()] {
            let read = read_checked(&dataset, "vec", &asked, &held_through(&pool)).await;
            assert_eq!(read.row_ids.last(), Some(&asked[5]));
            assert_eq!(
                (sorted(read.row_ids), read.through_take),
                (sorted(asked.clone()), 1)
            );
            assert_eq!(read.local.in_place + read.local.handed_off > 0, held);
        }
    }

    /// A vector column inside a struct column records the nulls of both in
    /// the control word in front of each vector: the rows whose struct or
    /// vector is null are read through take, which gives them as Lance does,
    /// and every other row at its byte offset.
    #[tokio::test]
    async fn vectors_inside_a_struct_with_nulls_are_read_at_their_offsets() {
        const STRUCT_NULL: usize = 7;
        const VECTOR_NULL: usize = 9;
        let generated = gen_batch()
            .col("vec", array::rand_vec::<Float32Type>(128.into()))
            .into_batch_rows(RowCount::from(100))
            .unwrap();
        let generated = generated["vec"].as_fixed_size_list();
        let DataType::FixedSizeList(item, dimension) = generated.data_type() else {
            unreachable!()
        };
        let vectors = FixedSizeListArray::try_new(
            item.clone(),
            *dimension,
            generated.values().clone(),
            Some(NullBuffer::from_iter(
                (0..100).map(|row| row != VECTOR_NULL),
            )),
        )
        .unwrap();
        let outer = StructArray::try_new(
            vec![Arc::new(Field::new(
                "vec",
                vectors.data_type().clone(),
                true,
            ))]
            .into(),
            vec![Arc::new(vectors) as ArrayRef],
            Some(NullBuffer::from_iter(
                (0..100).map(|row| row != STRUCT_NULL),
            )),
        )
        .unwrap();
        let batch = RecordBatch::try_from_iter([("s", Arc::new(outer) as ArrayRef)]).unwrap();
        let schema = batch.schema();
        let dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            "memory://",
            None,
        )
        .await
        .unwrap();
        let dataset = Arc::new(dataset);
        let asked = (0..100).collect::<Vec<u64>>();
        let read = read_vectors(&dataset, "s.vec", &asked, None).await.unwrap();
        assert_eq!(read.through_take, 2);
        assert_eq!(
            read.row_ids[read.row_ids.len() - 2..],
            [STRUCT_NULL as u64, VECTOR_NULL as u64]
        );
        assert_eq!(sorted(read.row_ids.clone()), asked);
        let taken = take(&dataset, "s.vec", &read.row_ids).await.unwrap();
        assert_eq!(read.vectors.to_data(), taken.to_data());
        assert_eq!(read.vectors.null_count(), 2);
    }

    /// Rows an overlay gives newer vectors are read through take, which
    /// reads the overlay's vectors. The other rows of their fragment are read
    /// at their byte offsets, among them one whose id alone the overlay
    /// covers, with row ids stable or not, except a null vector, which take
    /// reads after the overlaid ones.
    #[rstest]
    #[tokio::test]
    async fn rows_an_overlay_covers_are_read_through_take(
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let params = WriteParams {
            enable_stable_row_ids,
            ..Default::default()
        };
        let fixture = Fixture {
            null: Some(105),
            params,
            ..Default::default()
        };
        let dataset = Arc::new(fixture.write("memory://").await);
        let DataType::FixedSizeList(item, _) = dataset.schema().field("vec").unwrap().data_type()
        else {
            unreachable!()
        };
        let newer = FixedSizeListArray::try_new(
            item,
            128,
            Arc::new(Float32Array::from(vec![0.5f32; 2 * 128])),
            None,
        )
        .unwrap();
        let address = |offset| u64::from(RowAddress::new_from_parts(1, offset));
        let vectors_and_ids = RecordBatch::try_from_iter([
            (
                ROW_ADDR,
                Arc::new(UInt64Array::from(vec![address(7), address(42)])) as ArrayRef,
            ),
            ("id", Arc::new(Int32Array::from(vec![107, 142])) as ArrayRef),
            ("vec", Arc::new(newer) as ArrayRef),
        ])
        .unwrap();
        let id_alone = RecordBatch::try_from_iter([
            (
                ROW_ADDR,
                Arc::new(UInt64Array::from(vec![address(50)])) as ArrayRef,
            ),
            ("id", Arc::new(Int32Array::from(vec![150])) as ArrayRef),
        ])
        .unwrap();
        let schema = dataset.schema().project(&["id", "vec"]).unwrap();
        let mut overlay = dataset
            .get_fragment(1)
            .unwrap()
            .write_overlay(&schema)
            .await
            .unwrap();
        overlay.write_batch(&vectors_and_ids).await.unwrap();
        overlay.write_batch(&id_alone).await.unwrap();
        let group = overlay.finish().await.unwrap().unwrap();
        let operation = Operation::DataOverlay {
            groups: vec![group],
        };
        let dataset = commit(dataset, operation).await;
        let fragments = dataset.get_fragments();
        let overlays = &fragments[1].metadata().overlays;
        assert!(
            matches!(overlays[..], [ref overlay] if matches!(overlay.coverage, OverlayCoverage::PerField(_))),
            "{overlays:?}"
        );
        let asked = row_ids(&dataset, &[107, 3, 142, 150, 105]).await;
        assert_eq!(read(&dataset, &asked).await, (sorted(asked.clone()), 3));
        let overlaid = read_vectors(&dataset, "vec", &[asked[0], asked[2]], None)
            .await
            .unwrap();
        assert!(
            overlaid
                .vectors
                .as_fixed_size_list()
                .values()
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .all(|value| *value == 0.5)
        );
    }

    /// Rows the dataset no longer holds are not read: deleted rows and the
    /// rows of a fragment deleted whole.
    #[rstest]
    #[tokio::test]
    async fn rows_the_dataset_no_longer_holds_are_not_read(
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let params = WriteParams {
            enable_stable_row_ids,
            ..Default::default()
        };
        let fixture = Fixture {
            params,
            ..Default::default()
        };
        let mut dataset = fixture.write("memory://").await;
        let asked = row_ids(&dataset, &[3, 64, 150, 199]).await;
        dataset.delete("id = 64 OR id >= 100").await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 1);
        let dataset = Arc::new(dataset);
        assert_eq!(read(&dataset, &asked).await, (vec![asked[0]], 0));
    }

    /// With stable row ids a compaction moves rows to new addresses without
    /// rewriting an index, which still holds the ids of the rows it deleted:
    /// moved rows are read where they landed, at their byte offsets, and
    /// deleted ones not at all.
    #[tokio::test]
    async fn rows_a_compaction_moved_are_read_where_they_landed() {
        let params = WriteParams {
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let fixture = Fixture {
            params,
            ..Default::default()
        };
        let mut dataset = fixture.write("memory://").await;
        let asked = row_ids(&dataset, &[3, 64, 150, 199]).await;
        dataset.delete("id = 3 OR id = 150").await.unwrap();
        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        assert!(
            dataset.get_fragments().iter().all(|fragment| {
                fragment.id() >= 2 && fragment.metadata().deletion_file.is_none()
            })
        );
        let dataset = Arc::new(dataset);
        assert_eq!(read(&dataset, &asked).await, (vec![asked[1], asked[3]], 0));
    }

    /// Reads at byte offsets are counted in the I/O statistics given: once a
    /// data file's metadata is cached, one request to each fragment's data
    /// file for at least its rows' vectors. Reads through take are not.
    #[rstest]
    #[case::at_offsets(128, 2, ASKED.len() * 128 * 4)]
    #[case::through_take(32, 0, 0)]
    #[tokio::test]
    async fn reads_at_byte_offsets_are_counted(
        #[case] dimension: u32,
        #[case] requests: u64,
        #[case] bytes_read: usize,
    ) {
        let fixture = Fixture {
            dimension,
            ..Default::default()
        };
        let dataset = Arc::new(fixture.write("memory://").await);
        let asked = row_ids(&dataset, &ASKED).await;
        let io_stats = IoStats::new();
        read_vectors(&dataset, "vec", &asked, Some(&io_stats))
            .await
            .unwrap();
        let cold = io_stats.snapshot();
        read_vectors(&dataset, "vec", &asked, Some(&io_stats))
            .await
            .unwrap();
        let warm = io_stats.snapshot();
        assert_eq!(warm.requests - cold.requests, requests);
        let read = warm.bytes_read - cold.bytes_read;
        assert!(read >= bytes_read as u64, "{read} bytes read");
        if requests == 0 {
            assert_eq!(read, 0);
        }
    }

    #[tokio::test]
    async fn a_column_the_dataset_does_not_have_is_refused() {
        let dataset = Arc::new(Fixture::default().write("memory://").await);
        let error = read_vectors(&dataset, "missing", &[0], None)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error
                .to_string()
                .contains("column missing, which the dataset does not have"),
            "{error}"
        );
    }
}
