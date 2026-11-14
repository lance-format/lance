// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::mem::MaybeUninit;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, RecordBatch, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use lance_core::{Error, ROW_ID, Result};
use lance_linalg::distance::{DistanceType, cosine_distance, dot_distance, l2_f32};

use super::graph::ScoredPoint;

/// Vector column name used by Lance's flat vector storage batches.
pub const FLAT_COLUMN: &str = "flat";

#[derive(Clone)]
struct StoredArrowBatch {
    /// Keeps the Arrow buffer alive while `values_ptr` is used by readers.
    _array: Arc<FixedSizeListArray>,
    values_ptr: *const f32,
}

#[derive(Copy, Clone)]
struct RowLookup {
    batch_idx: u32,
    offset: u32,
}

/// Read-only vector source used by the HNSW graph.
///
/// The graph owns neighbor connectivity only. Vector bytes and row ids stay in
/// the caller's storage so MemTables can avoid a second `rows * dim * 4`
/// allocation.
pub trait VectorSource: Send + Sync {
    /// Number of vectors visible through this source.
    fn len(&self) -> usize;

    /// Number of `f32` values per vector.
    fn dim(&self) -> usize;

    /// Distance metric to use for all graph operations.
    fn distance_type(&self) -> DistanceType;

    /// Lance row id or row position for a vector id.
    fn row_id(&self, id: u32) -> u64;

    /// Borrow the vector values for a vector id.
    fn vector(&self, id: u32) -> &[f32];

    /// Returns true if no vectors are visible.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Compute distance between an external query vector and a vector id.
    fn distance_to(&self, query: &[f32], id: u32) -> f32 {
        compute_f32_distance(query, self.vector(id), self.distance_type())
    }

    /// Compute distance between two vector ids.
    fn distance_between(&self, left: u32, right: u32) -> f32 {
        compute_f32_distance(self.vector(left), self.vector(right), self.distance_type())
    }

    /// HNSW neighbor diversity heuristic. A tie keeps the candidate, unless it
    /// repeats a selected neighbor's vector: pruning ties lets a node linked to
    /// its exact copy drop every other neighbor, leaving copies linked only to
    /// each other.
    fn prefers_candidate(&self, candidate: ScoredPoint, selected: &[ScoredPoint]) -> bool {
        selected.iter().all(|other| {
            let between = self.distance_between(candidate.id, other.id);
            candidate.distance < between
                || (candidate.distance == between
                    && self.vector(candidate.id) != self.vector(other.id))
        })
    }
}

/// Compute a distance between two f32 vectors using Lance's SIMD-backed
/// distance kernels.
pub fn compute_f32_distance(query: &[f32], vector: &[f32], distance_type: DistanceType) -> f32 {
    match distance_type {
        DistanceType::L2 => l2_f32(query, vector),
        DistanceType::Dot => dot_distance(query, vector),
        DistanceType::Cosine => cosine_distance(query, vector),
        DistanceType::Hamming => f32::INFINITY,
    }
}

/// Zero-copy Arrow-backed vector store for active MemTables.
///
/// The store is append-only and follows a single-writer / multi-reader model;
/// an internal lock serializes accidental concurrent appends, and readers never
/// take it. Appended vector batches are retained by `Arc`, while id-to-batch
/// lookup tables make `vector(id)` O(1).
pub struct ArrowFixedSizeListVectorStore {
    batches: *mut MaybeUninit<StoredArrowBatch>,
    row_to_batch: *mut MaybeUninit<RowLookup>,
    row_ids: *mut MaybeUninit<u64>,
    committed_batches: AtomicUsize,
    committed_len: AtomicUsize,
    /// Held by `append_batch` while it claims, fills and publishes slots, so
    /// two callers sharing the store cannot write the same slots.
    append_lock: Mutex<()>,
    capacity: usize,
    max_batches: usize,
    dim: usize,
    distance_type: DistanceType,
    schema: SchemaRef,
}

// SAFETY: the store owns its three slot allocations. A stored batch holds an
// `Arc` of the Arrow array and a pointer into that array's immutable values, so
// both can move to another thread together with the store.
unsafe impl Send for ArrowFixedSizeListVectorStore {}
// SAFETY: `append_batch` writes only slots past the published lengths, holds
// `append_lock` while it does, and publishes them with Release stores. Readers
// load the lengths with Acquire and read only slots below them (snapshot reads
// rely on the HNSW graph passing ids below the snapshot length), and those
// slots are never written again.
unsafe impl Sync for ArrowFixedSizeListVectorStore {}

impl Drop for ArrowFixedSizeListVectorStore {
    fn drop(&mut self) {
        // SAFETY: `drop` has exclusive access to the store. Only committed
        // batch slots contain initialized `Arc` values that need dropping.
        unsafe {
            let committed_batches = self.committed_batches.load(Ordering::Acquire);
            for idx in 0..committed_batches {
                std::ptr::drop_in_place(self.batches.add(idx).cast::<StoredArrowBatch>());
            }
            let _: Box<[MaybeUninit<StoredArrowBatch>]> = Box::from_raw(
                std::ptr::slice_from_raw_parts_mut(self.batches, self.max_batches),
            );
            let _: Box<[MaybeUninit<RowLookup>]> = Box::from_raw(
                std::ptr::slice_from_raw_parts_mut(self.row_to_batch, self.capacity),
            );
            let _: Box<[MaybeUninit<u64>]> = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                self.row_ids,
                self.capacity,
            ));
        }
    }
}

impl ArrowFixedSizeListVectorStore {
    /// Create an append-only vector store.
    pub fn try_new(
        capacity: usize,
        max_batches: usize,
        dim: usize,
        distance_type: DistanceType,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::invalid_input("capacity must be greater than 0"));
        }
        if max_batches == 0 {
            return Err(Error::invalid_input("max_batches must be greater than 0"));
        }
        if dim == 0 {
            return Err(Error::invalid_input("dim must be greater than 0"));
        }
        if capacity > u32::MAX as usize {
            return Err(Error::invalid_input(format!(
                "capacity must fit in u32, got {capacity}"
            )));
        }
        if max_batches > u32::MAX as usize {
            return Err(Error::invalid_input(format!(
                "max_batches must fit in u32, got {max_batches}"
            )));
        }
        if distance_type == DistanceType::Hamming {
            return Err(Error::invalid_input(
                "ArrowFixedSizeListVectorStore stores f32 vectors and does not support hamming distance",
            ));
        }

        let batches: Box<[MaybeUninit<StoredArrowBatch>]> = uninit_boxed_slice(max_batches);
        let row_to_batch: Box<[MaybeUninit<RowLookup>]> = uninit_boxed_slice(capacity);
        let row_ids: Box<[MaybeUninit<u64>]> = uninit_boxed_slice(capacity);
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(
                FLAT_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));

        Ok(Self {
            batches: Box::into_raw(batches) as *mut MaybeUninit<StoredArrowBatch>,
            row_to_batch: Box::into_raw(row_to_batch) as *mut MaybeUninit<RowLookup>,
            row_ids: Box::into_raw(row_ids) as *mut MaybeUninit<u64>,
            committed_batches: AtomicUsize::new(0),
            committed_len: AtomicUsize::new(0),
            append_lock: Mutex::new(()),
            capacity,
            max_batches,
            dim,
            distance_type,
            schema,
        })
    }

    /// Heap bytes of the store's own slabs, all sized from `capacity` and
    /// `max_batches` at construction.
    ///
    /// Excludes the vectors themselves: batches are held by reference, so their
    /// bytes belong to the MemTable's batch store and counting them here would
    /// double-count. That also makes this independent of `dim`.
    pub(crate) fn resident_bytes(&self) -> usize {
        Self::reserved_bytes(self.capacity, self.max_batches)
    }

    /// What [`Self::resident_bytes`] will report for a store of this shape,
    /// answerable before one exists — the slabs are sized from these two
    /// numbers alone, and `dim` never enters. Lets a memory ceiling charge for
    /// the store ahead of the first insert that allocates it.
    pub(crate) fn reserved_bytes(capacity: usize, max_batches: usize) -> usize {
        max_batches * std::mem::size_of::<StoredArrowBatch>()
            + capacity * (std::mem::size_of::<RowLookup>() + std::mem::size_of::<u64>())
    }

    /// Number of committed vectors.
    pub fn committed_len(&self) -> usize {
        self.committed_len.load(Ordering::Acquire)
    }

    /// Number of `f32` values per vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Append one Arrow `FixedSizeList<Float32>` batch by reference.
    ///
    /// Null-vector rows (e.g. tombstones, or rows written without an embedding)
    /// are skipped — they get no graph node — matching base-table vector index
    /// semantics. Surviving non-null rows are compacted into a contiguous node
    /// range, but each node's `row_ids` entry preserves its *original* offset in
    /// the input batch so search still resolves a hit back to the correct row.
    pub fn append_batch(
        &self,
        vectors: Arc<FixedSizeListArray>,
        row_id_start: u64,
    ) -> Result<Range<u32>> {
        if vectors.is_empty() {
            let start = self.committed_len.load(Ordering::Relaxed) as u32;
            return Ok(start..start);
        }
        if vectors.value_length() as usize != self.dim {
            return Err(Error::invalid_input(format!(
                "vector dimension mismatch: expected {}, got {}",
                self.dim,
                vectors.value_length()
            )));
        }

        // Compact away null-vector rows, remembering each survivor's original
        // offset. The all-non-null case keeps the input array as-is (offsets are
        // the identity) to avoid a copy on the hot path.
        let (stored_array, original_offsets): (Arc<FixedSizeListArray>, Option<Vec<u32>>) =
            if vectors.null_count() == 0 {
                (vectors, None)
            } else {
                let valid = vectors
                    .nulls()
                    .ok_or_else(|| Error::internal("null_count > 0 but no null buffer"))?;
                let mask = BooleanArray::new(valid.inner().clone(), None);
                let filtered = arrow_select::filter::filter(vectors.as_ref(), &mask)?;
                let fsl = filtered
                    .as_fixed_size_list_opt()
                    .ok_or_else(|| {
                        Error::internal("filtered vector column is not a FixedSizeList")
                    })?
                    .clone();
                let offsets: Vec<u32> = (0..vectors.len() as u32)
                    .filter(|&i| valid.is_valid(i as usize))
                    .collect();
                (Arc::new(fsl), Some(offsets))
            };

        let num_rows = stored_array.len();
        if num_rows == 0 {
            // All rows were null (e.g. an all-tombstone memtable): no graph node.
            let start = self.committed_len.load(Ordering::Relaxed) as u32;
            return Ok(start..start);
        }

        let values = stored_array.values();
        let Some(values_f32) = values.as_primitive_opt::<Float32Type>() else {
            return Err(Error::invalid_input(format!(
                "vector values must be Float32, got {:?}",
                values.data_type()
            )));
        };

        // Held through the publishing stores at the end, so two appends never
        // claim the same slots. Every store to the lengths happens under it,
        // which makes the Relaxed loads below enough. A panicked append
        // publishes nothing and leaves only `Copy` row slots behind, so a
        // poisoned lock is still safe to take.
        let _append_guard = self
            .append_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);

        // Exhaustion is a shard-construction bug (store sized below the memtable),
        // not caller input — hence `internal`, not `invalid_input`.
        let start = self.committed_len.load(Ordering::Relaxed);
        let end = start.checked_add(num_rows).ok_or_else(|| {
            Error::internal(format!(
                "vector count overflow: start={}, batch_len={}",
                start, num_rows
            ))
        })?;
        if end > self.capacity {
            return Err(Error::internal(format!(
                "HNSW vector store capacity {} exhausted: inserting rows [{}..{}); \
                 the store is sized below the memtable's row capacity",
                self.capacity, start, end
            )));
        }

        let batch_idx = self.committed_batches.load(Ordering::Relaxed);
        if batch_idx >= self.max_batches {
            return Err(Error::internal(format!(
                "HNSW vector store max_batches {} exhausted at batch_idx {} \
                 (inserting rows [{}..{})); the store is sized below the \
                 memtable's batch capacity",
                self.max_batches, batch_idx, start, end
            )));
        }

        // SAFETY: `append_lock` is held, so no other append can claim
        // `batch_idx` or `start..end`, and both were bounds-checked above.
        // Readers cannot see these slots until the Release stores below.
        unsafe {
            for offset in 0..num_rows {
                self.row_to_batch
                    .add(start + offset)
                    .write(MaybeUninit::new(RowLookup {
                        batch_idx: batch_idx as u32,
                        offset: offset as u32,
                    }));
                // The node maps back to the survivor's *original* row position,
                // so a compacted node still resolves to the right row.
                let original = match &original_offsets {
                    Some(offsets) => offsets[offset] as u64,
                    None => offset as u64,
                };
                self.row_ids
                    .add(start + offset)
                    .write(MaybeUninit::new(row_id_start + original));
            }
            // Written after the loop, which can panic on a row id overflow, so a
            // panicked append never leaves an `Arc` in a slot that is not dropped.
            self.batches
                .add(batch_idx)
                .write(MaybeUninit::new(StoredArrowBatch {
                    _array: stored_array.clone(),
                    values_ptr: values_f32.values().as_ptr(),
                }));
        }

        self.committed_batches
            .store(batch_idx + 1, Ordering::Release);
        self.committed_len.store(end, Ordering::Release);
        Ok(start as u32..end as u32)
    }

    /// Capture a stable visible prefix of the store.
    pub fn snapshot(self: &Arc<Self>) -> VectorStoreSnapshot {
        self.snapshot_after_visible_len(|| {})
    }

    fn snapshot_after_visible_len(
        self: &Arc<Self>,
        after_visible_len: impl FnOnce(),
    ) -> VectorStoreSnapshot {
        // Read the length first. If it observes a newly committed batch, the
        // Acquire load also makes the preceding committed_batches publication
        // visible to the later load below. If it observes the old length, the
        // snapshot remains a valid prefix even if a writer commits meanwhile.
        let visible_len = self.committed_len();
        after_visible_len();
        let committed_batches = self.committed_batches.load(Ordering::Acquire);
        let contiguous_values_addr = if committed_batches == 1 {
            // SAFETY: batch slot 0 is initialized before committed_batches is
            // published. The store Arc retained by the snapshot keeps it alive.
            unsafe { (*self.batches.cast::<StoredArrowBatch>()).values_ptr as usize }
        } else {
            0
        };
        VectorStoreSnapshot {
            store: self.clone(),
            visible_len,
            contiguous_values_addr,
        }
    }

    /// Materialize the visible vectors into Lance flat-storage format.
    pub fn to_record_batch(&self, total_rows: Option<u64>) -> Result<RecordBatch> {
        let visible_len = self.committed_len();
        self.to_record_batch_with_len(visible_len, total_rows)
    }

    fn to_record_batch_with_len(
        &self,
        visible_len: usize,
        total_rows: Option<u64>,
    ) -> Result<RecordBatch> {
        let mut row_ids = Vec::with_capacity(visible_len);
        let mut values = Vec::with_capacity(visible_len * self.dim);
        for id in 0..visible_len as u32 {
            let row_id = self.row_id_at(id);
            row_ids.push(match total_rows {
                Some(total_rows) => total_rows.checked_sub(row_id + 1).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "row id reversal underflow: total_rows={total_rows}, row_id={row_id}"
                    ))
                })?,
                None => row_id,
            });
            values.extend_from_slice(self.vector_at(id));
        }

        let row_ids = Arc::new(UInt64Array::from(row_ids)) as ArrayRef;
        let values = Arc::new(Float32Array::from(values)) as ArrayRef;
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let vectors = Arc::new(FixedSizeListArray::try_new(
            field,
            self.dim as i32,
            values,
            None,
        )?) as ArrayRef;
        Ok(RecordBatch::try_new(
            self.schema.clone(),
            vec![row_ids, vectors],
        )?)
    }

    fn row_id_at(&self, id: u32) -> u64 {
        debug_assert!((id as usize) < self.committed_len.load(Ordering::Acquire));
        // SAFETY: callers pass committed ids. Row ids are initialized before
        // `committed_len` is published and never overwritten.
        unsafe { self.row_ids.add(id as usize).read().assume_init() }
    }

    fn vector_at(&self, id: u32) -> &[f32] {
        debug_assert!((id as usize) < self.committed_len.load(Ordering::Acquire));
        // SAFETY: callers pass committed ids. The lookup and batch slot were
        // initialized before publication; the retained Arc keeps values alive.
        unsafe {
            let lookup = self.row_to_batch.add(id as usize).read().assume_init();
            let batch = &*self
                .batches
                .add(lookup.batch_idx as usize)
                .cast::<StoredArrowBatch>();
            let ptr = batch.values_ptr.add(lookup.offset as usize * self.dim);
            std::slice::from_raw_parts(ptr, self.dim)
        }
    }
}

/// Snapshot of an [`ArrowFixedSizeListVectorStore`].
#[derive(Clone)]
pub struct VectorStoreSnapshot {
    store: Arc<ArrowFixedSizeListVectorStore>,
    visible_len: usize,
    contiguous_values_addr: usize,
}

impl VectorStoreSnapshot {
    /// Materialize the snapshot into Lance flat-storage format.
    pub fn to_record_batch(&self, total_rows: Option<u64>) -> Result<RecordBatch> {
        self.store
            .to_record_batch_with_len(self.visible_len, total_rows)
    }
}

impl VectorSource for VectorStoreSnapshot {
    fn len(&self) -> usize {
        self.visible_len
    }

    fn dim(&self) -> usize {
        self.store.dim
    }

    fn distance_type(&self) -> DistanceType {
        self.store.distance_type
    }

    fn row_id(&self, id: u32) -> u64 {
        // HNSW only requests ids from its own graph, which is built from this
        // snapshot's visible prefix. Keep this as a debug-only contract check.
        debug_assert!(
            (id as usize) < self.visible_len,
            "vector id {id} is outside snapshot length {}",
            self.visible_len
        );
        self.store.row_id_at(id)
    }

    fn vector(&self, id: u32) -> &[f32] {
        debug_assert!(
            (id as usize) < self.visible_len,
            "vector id {id} is outside snapshot length {}",
            self.visible_len
        );
        if self.contiguous_values_addr != 0 {
            // SAFETY: the HNSW graph passes only ids below this snapshot's
            // length: search caps neighbor ids at it, and inserts check that it
            // covers the range they build (debug builds also check above). The
            // address is set only while batch 0 was the only published batch,
            // so the whole visible prefix lies in that batch's values, which
            // the store `Arc` held by this snapshot keeps alive.
            unsafe {
                let ptr =
                    (self.contiguous_values_addr as *const f32).add(id as usize * self.store.dim);
                return std::slice::from_raw_parts(ptr, self.store.dim);
            }
        }
        self.store.vector_at(id)
    }
}

fn uninit_boxed_slice<T>(len: usize) -> Box<[MaybeUninit<T>]> {
    (0..len)
        .map(|_| MaybeUninit::uninit())
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    fn fsl(values: Vec<f32>, dim: usize) -> Arc<FixedSizeListArray> {
        let values = Arc::new(Float32Array::from(values)) as ArrayRef;
        Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
                values,
                None,
            )
            .unwrap(),
        )
    }

    #[test]
    fn test_arrow_store_reuses_batches() {
        let store =
            Arc::new(ArrowFixedSizeListVectorStore::try_new(8, 2, 2, DistanceType::L2).unwrap());
        let first = fsl(vec![1.0, 2.0, 3.0, 4.0], 2);
        let second = fsl(vec![5.0, 6.0], 2);

        assert_eq!(store.append_batch(first, 10).unwrap(), 0..2);
        assert_eq!(store.append_batch(second, 12).unwrap(), 2..3);

        let snapshot = store.snapshot();
        assert_eq!(snapshot.len(), 3);
        assert_eq!(snapshot.row_id(2), 12);
        assert_eq!(snapshot.vector(1), &[3.0, 4.0]);
        assert_eq!(
            compute_f32_distance(snapshot.vector(0), snapshot.vector(1), DistanceType::L2),
            8.0
        );
    }

    #[test]
    fn test_snapshot_stays_with_visible_prefix_during_commit() {
        let store =
            Arc::new(ArrowFixedSizeListVectorStore::try_new(8, 2, 2, DistanceType::L2).unwrap());
        store
            .append_batch(fsl(vec![1.0, 2.0, 3.0, 4.0], 2), 10)
            .unwrap();

        let visible_len_loaded = Arc::new(Barrier::new(2));
        let continue_snapshot = Arc::new(Barrier::new(2));
        let snapshot_store = store.clone();
        let snapshot_visible_len_loaded = visible_len_loaded.clone();
        let snapshot_continue = continue_snapshot.clone();
        let snapshot_thread = std::thread::spawn(move || {
            snapshot_store.snapshot_after_visible_len(|| {
                snapshot_visible_len_loaded.wait();
                snapshot_continue.wait();
            })
        });

        visible_len_loaded.wait();
        store.append_batch(fsl(vec![5.0, 6.0], 2), 12).unwrap();
        continue_snapshot.wait();

        let snapshot = snapshot_thread.join().unwrap();
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot.row_id(1), 11);
        assert_eq!(snapshot.vector(1), &[3.0, 4.0]);
    }

    /// Build a `FixedSizeList<Float32>` where `None` rows are null at the list
    /// level (the representation a tombstone / embedding-less row produces).
    fn fsl_opt(rows: &[Option<Vec<f32>>], dim: usize) -> Arc<FixedSizeListArray> {
        let mut values: Vec<f32> = Vec::new();
        let mut validity: Vec<bool> = Vec::new();
        for r in rows {
            match r {
                Some(v) => {
                    assert_eq!(v.len(), dim);
                    values.extend_from_slice(v);
                    validity.push(true);
                }
                None => {
                    values.extend(std::iter::repeat_n(0.0f32, dim));
                    validity.push(false);
                }
            }
        }
        let values = Arc::new(Float32Array::from(values)) as ArrayRef;
        let nulls = arrow_buffer::NullBuffer::new(arrow_buffer::BooleanBuffer::from(validity));
        Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
                values,
                Some(nulls),
            )
            .unwrap(),
        )
    }

    #[test]
    fn test_append_skips_null_vectors_preserves_mapping() {
        // Mid-batch null: rows [(1,1), null, (3,3)] → two nodes that map back to
        // the *original* offsets 0 and 2 (no off-by-one after the skip).
        let store =
            Arc::new(ArrowFixedSizeListVectorStore::try_new(8, 2, 2, DistanceType::L2).unwrap());
        let batch = fsl_opt(&[Some(vec![1.0, 1.0]), None, Some(vec![3.0, 3.0])], 2);
        assert_eq!(
            store.append_batch(batch, 0).unwrap(),
            0..2,
            "two non-null survivors → two contiguous nodes"
        );
        assert_eq!(store.committed_len(), 2);

        // `to_record_batch` exposes each node's row id = its original offset.
        let rb = store.to_record_batch(None).unwrap();
        let row_ids = rb.column(0).as_any().downcast_ref::<UInt64Array>().unwrap();
        let got: Vec<u64> = (0..row_ids.len()).map(|i| row_ids.value(i)).collect();
        assert_eq!(got, vec![0, 2], "node→row mapping skips the null row");

        let snapshot = store.snapshot();
        assert_eq!(snapshot.vector(0), &[1.0, 1.0]);
        assert_eq!(snapshot.vector(1), &[3.0, 3.0]);
        assert_eq!(snapshot.row_id(0), 0);
        assert_eq!(snapshot.row_id(1), 2);
    }

    #[test]
    fn test_append_skips_null_vectors_offsets_with_row_id_start() {
        // The survivor's original offset is added to `row_id_start`.
        let store =
            Arc::new(ArrowFixedSizeListVectorStore::try_new(8, 2, 2, DistanceType::L2).unwrap());
        // rows at row_id_start=100: [null, (2,2)] → one node, row id 100+1.
        let batch = fsl_opt(&[None, Some(vec![2.0, 2.0])], 2);
        assert_eq!(store.append_batch(batch, 100).unwrap(), 0..1);
        let snapshot = store.snapshot();
        assert_eq!(snapshot.row_id(0), 101);
        assert_eq!(snapshot.vector(0), &[2.0, 2.0]);
    }

    #[test]
    fn test_append_all_null_batch_is_empty_range() {
        // All-null batch (e.g. an all-tombstone memtable) adds no node.
        let store =
            Arc::new(ArrowFixedSizeListVectorStore::try_new(8, 2, 2, DistanceType::L2).unwrap());
        let batch = fsl_opt(&[None, None], 2);
        assert!(
            store.append_batch(batch, 0).unwrap().is_empty(),
            "all-null batch yields an empty node range"
        );
        assert_eq!(store.committed_len(), 0);
    }

    #[test]
    fn test_concurrent_appends_are_serialized() {
        const WRITERS: usize = 8;
        const APPENDS_PER_WRITER: usize = 50;
        const ROWS: usize = WRITERS * APPENDS_PER_WRITER;
        let store = Arc::new(
            ArrowFixedSizeListVectorStore::try_new(ROWS, ROWS, 2, DistanceType::L2).unwrap(),
        );
        let start = Arc::new(Barrier::new(WRITERS));
        let writers: Vec<_> = (0..WRITERS)
            .map(|writer| {
                let store = store.clone();
                let start = start.clone();
                std::thread::spawn(move || {
                    start.wait();
                    (0..APPENDS_PER_WRITER)
                        .map(|append| {
                            let row_id = (writer * APPENDS_PER_WRITER + append) as u64;
                            let vector = vec![row_id as f32, -(row_id as f32)];
                            (row_id, store.append_batch(fsl(vector, 2), row_id).unwrap())
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let appended: Vec<_> = writers
            .into_iter()
            .flat_map(|writer| writer.join().unwrap())
            .collect();

        // Each append got its own one-row range, and the store holds what that
        // append wrote there.
        let snapshot = store.snapshot();
        assert_eq!(snapshot.len(), ROWS);
        for (row_id, ids) in appended {
            assert_eq!(ids.len(), 1);
            assert_eq!(snapshot.row_id(ids.start), row_id);
            assert_eq!(
                snapshot.vector(ids.start),
                &[row_id as f32, -(row_id as f32)]
            );
        }
    }

    // The row id overflow below panics only where overflow checks are on.
    #[cfg(debug_assertions)]
    #[test]
    fn test_append_after_a_panicked_append() {
        let store =
            Arc::new(ArrowFixedSizeListVectorStore::try_new(8, 2, 2, DistanceType::L2).unwrap());
        // The second row's id overflows after the first row's slots are written.
        let abandoned = fsl(vec![1.0, 2.0, 3.0, 4.0], 2);
        let append = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            store.append_batch(abandoned.clone(), u64::MAX)
        }));
        assert!(append.is_err());
        assert_eq!(store.committed_len(), 0);

        // The poisoned lock does not block the next append, which reuses the slots.
        assert_eq!(store.append_batch(fsl(vec![5.0, 6.0], 2), 7).unwrap(), 0..1);
        let snapshot = store.snapshot();
        assert_eq!(snapshot.row_id(0), 7);
        assert_eq!(snapshot.vector(0), &[5.0, 6.0]);

        drop(snapshot);
        drop(store);
        assert_eq!(
            Arc::strong_count(&abandoned),
            1,
            "the panicked append left a reference to its batch behind"
        );
    }
}
