// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The small columns of an IVF_RQ storage file, kept in memory.
//!
//! A read of a partition or plane from the file costs at least one request
//! per column, yet every column but the codes and the estimator bounds holds
//! a few bytes per row: row ids and factors. With those columns read once
//! per index, later reads fetch only the codes (and bounds) from the file:
//! on an object store a native partition miss takes 2 requests instead of
//! 8, and a layered one 3 instead of 13. Reads attach copies of the resident
//! rows, so cache entries hold, and are charged, what a file read returns.
//! Every open of a file in the process shares one store
//! ([`ResidentColumns::for_file`]).

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use arrow::compute::concat_batches;
use arrow_array::{Array, ArrayRef, UInt64Array, new_empty_array};
use arrow_schema::{Field, Schema};
use arrow_select::concat::concat;
use arrow_select::take::take;
use futures::TryStreamExt;
use lance_core::{Error, Result};
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::FileReader;
use lance_io::ReadBatchParams;
use lance_io::scheduler::IoStats;
use tokio::sync::OnceCell;

use super::layered::{FULL_BOUNDS_COLUMN, HIGH_BOUNDS_COLUMN};
use super::layered_stats;
use super::storage::{
    RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_BLOCKED_EX_CODE_LO_COLUMN, RABIT_CODE_COLUMN,
    RABIT_EX_CODE_COLUMN,
};
use crate::vector::storage::{IndexFileKey, WeakRegistry, shared_by_key};

/// Columns that reads always fetch from the file: the codes, which hold most
/// of its bytes, and the estimator bounds, which only some scans read.
const FILE_COLUMNS: [&str; 6] = [
    RABIT_CODE_COLUMN,
    RABIT_EX_CODE_COLUMN,
    RABIT_BLOCKED_EX_CODE_COLUMN,
    RABIT_BLOCKED_EX_CODE_LO_COLUMN,
    HIGH_BOUNDS_COLUMN,
    FULL_BOUNDS_COLUMN,
];

/// Byte width of `field`'s values when the store keeps the column: every
/// fixed-width primitive column but [`FILE_COLUMNS`].
fn resident_width(field: &Field) -> Option<usize> {
    if FILE_COLUMNS.contains(&field.name().as_str()) {
        return None;
    }
    field.data_type().primitive_width()
}

/// Whether the store keeps column `field` of an IVF_RQ storage file.
pub(crate) fn is_resident(field: &Field) -> bool {
    resident_width(field).is_some()
}

/// Bytes of values the resident store holds for an IVF_RQ storage file with
/// `schema` and `num_rows` rows: every row of every column it keeps, at the
/// column's fixed width. It reads nothing, so an embedder can set the memory
/// aside before the index opens; no cache budget charges the store. The
/// store's allocations can exceed it by a few bytes per column.
pub fn resident_columns_bytes(schema: &Schema, num_rows: u64) -> u64 {
    schema
        .fields()
        .iter()
        .filter_map(|field| resident_width(field))
        .map(|width| width as u64 * num_rows)
        .sum()
}

/// The resident stores of the open index files; see
/// [`ResidentColumns::for_file`].
static RESIDENT_STORES: LazyLock<WeakRegistry<IndexFileKey, OnceCell<ResidentColumnStore>>> =
    LazyLock::new(Default::default);

/// The resident columns of one IVF_RQ storage file, loaded on first use.
/// Clones share the store, so every reconstruction of a cached index uses
/// the one load, as they share the index's plane access history. A default
/// store is the caller's own; [`Self::for_file`] gives the one every open of
/// the file shares.
#[derive(Debug, Clone, Default)]
pub struct ResidentColumns(Arc<OnceCell<ResidentColumnStore>>);

impl ResidentColumns {
    /// The store of index file `file` that every open of it in the process
    /// shares: the live one while an index or a cached state holds it, else
    /// a new one, loaded on first use. Indexes of the file opened at once, a
    /// re-open while an older index still runs, and a state read back from a
    /// persistent cache tier thus read the file's small columns once, and
    /// hold one copy of them.
    pub fn for_file(file: &IndexFileKey) -> Self {
        Self(shared_by_key(&RESIDENT_STORES, file, || {
            Arc::new(OnceCell::new())
        }))
    }

    /// Bytes of values the store holds, `None` until it has loaded. Equals
    /// [`resident_columns_bytes`] of its file.
    pub fn loaded_bytes(&self) -> Option<u64> {
        self.0.get().map(|store| store.bytes)
    }

    /// The store of `reader`'s file, loaded on first use. Concurrent first
    /// callers share one load, whose I/O is added to the loading caller's
    /// `io_stats`; a failed or dropped load leaves the store for the next
    /// caller to load. The load runs once per index, so it is boxed rather
    /// than inlined into every read's future.
    pub(crate) async fn get_or_load(
        &self,
        reader: &FileReader,
        io_stats: Option<&IoStats>,
    ) -> Result<&ResidentColumnStore> {
        self.0
            .get_or_try_init(|| Box::pin(ResidentColumnStore::load(reader, io_stats)))
            .await
    }
}

/// Every row of the resident columns of one file.
pub(crate) struct ResidentColumnStore {
    columns: HashMap<String, ResidentColumn>,
    num_rows: u64,
    /// See [`ResidentColumns::loaded_bytes`].
    bytes: u64,
}

/// Names the columns instead of printing hundreds of megabytes of values.
impl std::fmt::Debug for ResidentColumnStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut columns: Vec<&String> = self.columns.keys().collect();
        columns.sort();
        f.debug_struct("ResidentColumnStore")
            .field("columns", &columns)
            .field("num_rows", &self.num_rows)
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl ResidentColumnStore {
    async fn load(reader: &FileReader, io_stats: Option<&IoStats>) -> Result<Self> {
        let started = Instant::now();
        let schema = Schema::from(reader.schema().as_ref());
        let fields: Vec<(&Field, usize)> = schema
            .fields()
            .iter()
            .filter_map(|field| Some((field.as_ref(), resident_width(field)?)))
            .collect();
        let num_rows = reader.num_rows();
        let load_stats = IoStats::new();
        let batch = if fields.is_empty() || num_rows == 0 {
            None
        } else {
            let names: Vec<&str> = fields
                .iter()
                .map(|(field, _)| field.name().as_str())
                .collect();
            let projection = lance_file::versions::reader_projection_from_column_names(
                reader.metadata().version(),
                reader.schema(),
                &names,
            )?;
            let projected = Arc::new(Schema::from(projection.schema.as_ref()));
            let stats_reader = reader.with_io_stats(load_stats.recorder());
            let batches = stats_reader
                .read_stream_projected(
                    ReadBatchParams::Range(0..num_rows as usize),
                    u32::MAX,
                    1,
                    projection,
                    FilterExpression::no_filter(),
                )
                .await?
                .try_collect::<Vec<_>>()
                .await?;
            Some(concat_batches(&projected, batches.iter())?)
        };
        let mut columns = HashMap::with_capacity(fields.len());
        let mut bytes = 0u64;
        // What the arrays hold, which a zero-copy slice of a larger decoded
        // or read buffer would push past `bytes`.
        let mut alloc_bytes = 0u64;
        for (field, width) in fields {
            let name = field.name();
            let values = match &batch {
                Some(batch) => batch.column_by_name(name).cloned().ok_or_else(|| {
                    Error::internal(format!("resident column {name} is missing from its read"))
                })?,
                None => new_empty_array(field.data_type()),
            };
            if values.len() as u64 != num_rows {
                return Err(Error::internal(format!(
                    "resident column {name} read {} of the file's {num_rows} rows",
                    values.len()
                )));
            }
            bytes += width as u64 * num_rows;
            alloc_bytes += values.get_buffer_memory_size() as u64;
            let page_ends = column_page_ends(reader, name)?;
            columns.insert(name.clone(), ResidentColumn { values, page_ends });
        }
        let loaded = load_stats.snapshot();
        let stats = layered_stats::counters();
        stats.resident_columns_bytes.add(bytes);
        stats.resident_columns_alloc_bytes.add(alloc_bytes);
        stats.resident_columns_load_requests.add(loaded.iops);
        stats.resident_columns_load_bytes.add(loaded.bytes_read);
        stats.resident_columns_load_ns.add_elapsed(started);
        if let Some(io_stats) = io_stats {
            io_stats.add_scan_stats(&loaded);
        }
        Ok(Self {
            columns,
            num_rows,
            bytes,
        })
    }

    /// Resident column `name`, `None` for a column reads fetch from the file.
    pub(crate) fn column(&self, name: &str) -> Option<&ResidentColumn> {
        self.columns.get(name)
    }

    /// Rows of the file.
    pub(crate) fn num_rows(&self) -> u64 {
        self.num_rows
    }
}

/// The row at which each page of column `name` of `reader`'s file ends.
fn column_page_ends(reader: &FileReader, name: &str) -> Result<Vec<u64>> {
    let metadata = reader.metadata();
    let projection = lance_file::versions::reader_projection_from_column_names(
        metadata.version(),
        reader.schema(),
        &[name],
    )?;
    let column = match projection.column_indices.as_slice() {
        [column] => metadata.column_infos.get(*column as usize),
        _ => None,
    }
    .ok_or_else(|| {
        Error::internal(format!(
            "resident column {name} is not one column of the file: {:?}",
            projection.column_indices
        ))
    })?;
    Ok(column
        .page_infos
        .iter()
        .scan(0u64, |end, page| {
            *end += page.num_rows;
            Some(*end)
        })
        .collect())
}

/// Every row of one resident column, with the column's pages in the file.
pub(crate) struct ResidentColumn {
    values: ArrayRef,
    /// The row at which each page of the column ends, ascending.
    page_ends: Vec<u64>,
}

impl ResidentColumn {
    /// A copy of the ascending file rows `rows`, laid out as a file read of
    /// them so that a cache entry holding it is charged the same bytes: the
    /// reader decodes each page a read touches into a buffer of exactly its
    /// rows and concatenates the buffers when the read touches several
    /// pages. A slice would instead keep, and be charged, the whole column.
    pub(crate) fn copy_rows(&self, rows: &UInt64Array) -> Result<ArrayRef> {
        let offsets = rows.values();
        // The page split below relies on the order.
        if !offsets.is_sorted() {
            return Err(Error::invalid_input(
                "rows read from resident columns must ascend",
            ));
        }
        if let Some(&last) = offsets.last()
            && last >= self.values.len() as u64
        {
            return Err(Error::invalid_input(format!(
                "resident row {last} is past the {} rows of the file",
                self.values.len()
            )));
        }
        let mut pieces = Vec::new();
        let mut start = 0;
        let mut page = 0;
        while start < offsets.len() {
            let first = offsets[start];
            page += self.page_ends[page..].partition_point(|&end| end <= first);
            let page_end = self.page_ends.get(page).copied().unwrap_or(u64::MAX);
            let len = offsets[start..].partition_point(|&row| row < page_end);
            let piece = take(self.values.as_ref(), &rows.slice(start, len), None)?;
            pieces.push(piece);
            start += len;
        }
        match pieces.len() {
            0 => Ok(new_empty_array(self.values.data_type())),
            1 => Ok(pieces.swap_remove(0)),
            _ => {
                let pieces: Vec<&dyn Array> = pieces.iter().map(|piece| piece.as_ref()).collect();
                Ok(concat(&pieces)?)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float32Array, cast::AsArray, types::Float32Type};
    use arrow_schema::DataType;

    use crate::vector::bq::layered::{HIGH_ADD_FACTORS_COLUMN, HIGH_SCALE_FACTORS_COLUMN};
    use crate::vector::bq::transform::{
        ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN,
        SCALE_FACTORS_COLUMN,
    };

    fn list_field(name: &str, item: DataType, width: i32) -> Field {
        Field::new(
            name,
            DataType::FixedSizeList(Arc::new(Field::new("item", item, true)), width),
            true,
        )
    }

    fn factor_field(name: &str) -> Field {
        Field::new(name, DataType::Float32, true)
    }

    #[test]
    fn resident_columns_bytes_counts_the_small_fixed_width_columns() {
        let native = vec![
            Field::new(lance_core::ROW_ID, DataType::UInt64, true),
            list_field(RABIT_CODE_COLUMN, DataType::UInt8, 16),
            factor_field(ADD_FACTORS_COLUMN),
            factor_field(SCALE_FACTORS_COLUMN),
            factor_field(ERROR_FACTORS_COLUMN),
            list_field(RABIT_BLOCKED_EX_CODE_COLUMN, DataType::UInt8, 96),
            factor_field(EX_ADD_FACTORS_COLUMN),
            factor_field(EX_SCALE_FACTORS_COLUMN),
        ];
        let mut layered = native.clone();
        layered.extend([
            list_field(RABIT_BLOCKED_EX_CODE_LO_COLUMN, DataType::UInt8, 64),
            factor_field(HIGH_ADD_FACTORS_COLUMN),
            factor_field(HIGH_SCALE_FACTORS_COLUMN),
            list_field(HIGH_BOUNDS_COLUMN, DataType::Float32, 3),
            list_field(FULL_BOUNDS_COLUMN, DataType::Float32, 3),
        ]);
        // Row ids (8 bytes) and four-byte factors: 3 for the sign codes and
        // 2 for each ex level.
        assert_eq!(
            resident_columns_bytes(&Schema::new(native.clone()), 1000),
            28_000
        );
        assert_eq!(resident_columns_bytes(&Schema::new(layered), 1000), 36_000);
        assert_eq!(resident_columns_bytes(&Schema::new(native.clone()), 0), 0);
        // Legacy ex codes and columns without a fixed width stay in the file.
        let mut other = native;
        other.push(list_field(RABIT_EX_CODE_COLUMN, DataType::UInt8, 96));
        other.push(Field::new("label", DataType::Utf8, true));
        assert_eq!(resident_columns_bytes(&Schema::new(other), 10), 280);
    }

    /// Every open of an index file shares one store while any holds it; the
    /// same path in another bucket is another file, and a store no open
    /// holds any more is made anew.
    #[test]
    fn for_file_shares_the_store_of_a_file_while_held() {
        let path = "t.lance/_indices/resident-test/auxiliary.idx";
        let file = IndexFileKey::new("resident-test", "s3$bucket", path);
        let store = ResidentColumns::for_file(&file);
        let shared = ResidentColumns::for_file(&file);
        assert!(Arc::ptr_eq(&store.0, &shared.0));
        let other = IndexFileKey::new("resident-test", "s3$other", path);
        assert!(!Arc::ptr_eq(&store.0, &ResidentColumns::for_file(&other).0));
        let dropped = Arc::downgrade(&store.0);
        drop((store, shared));
        assert!(dropped.upgrade().is_none());
        assert_eq!(ResidentColumns::for_file(&file).loaded_bytes(), None);
    }

    /// A 100-row column in pages ending at rows 40, 40 (an empty page) and 100.
    fn column() -> ResidentColumn {
        let values: Vec<f32> = (0..100).map(|row| row as f32).collect();
        ResidentColumn {
            values: Arc::new(Float32Array::from(values)),
            page_ends: vec![40, 40, 100],
        }
    }

    fn copy(column: &ResidentColumn, rows: &[u64]) -> ArrayRef {
        column.copy_rows(&UInt64Array::from(rows.to_vec())).unwrap()
    }

    fn values(array: &ArrayRef) -> Vec<f32> {
        array.as_primitive::<Float32Type>().values().to_vec()
    }

    fn as_values(rows: &[u64]) -> Vec<f32> {
        rows.iter().map(|&row| row as f32).collect()
    }

    #[test]
    fn copy_rows_takes_the_rows_into_their_own_buffers() {
        let column = column();
        let store = column.values.as_primitive::<Float32Type>();
        let store_values = store.values().as_ptr();
        let rows: Vec<u64> = (10..30).collect();
        let copied = copy(&column, &rows);
        assert_eq!(values(&copied), as_values(&rows));
        assert!(copied.nulls().is_none());
        // One page: a buffer of exactly the rows, apart from the store's.
        assert_eq!(copied.get_buffer_memory_size(), rows.len() * 4);
        let copied_values = copied.as_primitive::<Float32Type>().values().as_ptr();
        assert_ne!(copied_values, store_values);

        let sparse = copy(&column, &[3, 7, 8, 39]);
        assert_eq!(values(&sparse), as_values(&[3, 7, 8, 39]));
        assert_eq!(sparse.get_buffer_memory_size(), 4 * 4);
        assert_eq!(copy(&column, &[]).len(), 0);
        for rows in [vec![5, 100], vec![7, 5]] {
            let error = column.copy_rows(&UInt64Array::from(rows)).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        }
    }

    #[test]
    fn copy_rows_across_pages_matches_a_file_read() {
        let column = column();
        for rows in [
            (30..60).collect::<Vec<u64>>(),
            vec![5, 39, 40, 99],
            (0..100).collect(),
        ] {
            let copied = copy(&column, &rows);
            assert_eq!(values(&copied), as_values(&rows));
            assert!(copied.nulls().is_none());
            // The reader concatenates the pages it decodes, each into a
            // buffer of exactly its rows.
            let (first, second): (Vec<u64>, Vec<u64>) = rows.iter().partition(|&&row| row < 40);
            let pages = [first, second]
                .map(|page| take(column.values.as_ref(), &UInt64Array::from(page), None).unwrap());
            let read = concat(&[pages[0].as_ref(), pages[1].as_ref()]).unwrap();
            assert_eq!(copied.as_ref(), read.as_ref(), "{rows:?}");
            assert_eq!(
                copied.get_buffer_memory_size(),
                read.get_buffer_memory_size(),
                "{rows:?}"
            );
        }
    }
}
