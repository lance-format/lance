// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded staging and native code-to-code tile scoring of index codes.
//!
//! Distances are symmetric values between the index representations `x̂` (the
//! vectors that the stored codes reconstruct). They are computed directly from
//! the codes; no floating-point vector is reconstructed:
//!
//! | metric  | distance                                          |
//! |---------|---------------------------------------------------|
//! | l2      | squared L2 `‖x̂a − x̂b‖²`                           |
//! | cosine  | `1 − cos(x̂a, x̂b)`, clamped to `[0, 2]`            |
//! | dot     | `1 − x̂a·x̂b`                                       |
//! | hamming | number of differing bits (binary IVF_FLAT)        |
//!
//! Cosine renormalizes the reconstructions, as the metric (and exact or refined
//! search) defines it. It is evaluated as
//! `(‖x̂a − x̂b‖² − (‖x̂a‖ − ‖x̂b‖)²) / (2‖x̂a‖‖x̂b‖)` from norms staged once per
//! row, so identical codes score exactly zero. A zero-norm reconstruction has
//! no cosine distance (NaN). Unrefined ANN search over quantized cosine indices
//! reports `‖q̂ − x̂‖²` of normalized vectors instead, about twice this value.
//!
//! Each distance depends only on its two rows, with the lower storage position
//! as the anchor, so results are bit-identical for every batch, block, spill and
//! concurrency layout.

use super::{
    bq::pairwise::RQPairScorer, flat::pairwise::FlatPairScorer, pq::pairwise::PQPairScorer,
    quantizer::Quantizer, sq::pairwise::SQPairScorer,
};
use crate::scalar::RowIdRemapper;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef, ArrowPrimitiveType, RecordBatch, UInt64Array};
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::{DataType, Field, Schema};
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, ROW_ID, Result};
use lance_io::{
    spill::{Spill, SpillStore},
    traits::{Reader, Writer},
};
use lance_linalg::distance::DistanceType;
use std::{io::Cursor, ops::Range, sync::Arc};
use tokio::io::AsyncWriteExt;

/// Default budget for staged codes. Larger partitions use session spill storage.
pub const PAIRWISE_MEMORY_LIMIT: usize = 256 * 1024 * 1024;

/// Candidate payload scored per pass over an anchor block. Together with the
/// anchor rows it stays in L2 cache, so each candidate row is loaded from
/// memory once per block rather than once per anchor.
const CANDIDATE_CHUNK_BYTES: usize = 256 * 1024;
const MIN_CANDIDATE_CHUNK_ROWS: usize = 16;

/// Row IDs plus Arrow overhead in the staged size estimate.
const STAGED_ROW_OVERHEAD_BYTES: usize = 16;

/// Staged `‖x̂‖` (f64) per row, for cosine.
pub(crate) const NORM_COLUMN: &str = "__pairwise_norm";

/// One staged batch of index rows in storage order.
#[derive(Clone, Debug)]
pub struct PairwiseVectorBatch {
    /// Current row IDs; null where compaction remapping deleted the row.
    pub row_ids: UInt64Array,
    /// Quantizer-specific staged payload, including the row IDs.
    pub codes: RecordBatch,
    batch_id: usize,
    selected: Vec<bool>,
}

impl PairwiseVectorBatch {
    /// Position of this batch in the partition's storage order.
    pub fn batch_id(&self) -> usize {
        self.batch_id
    }

    pub fn num_rows(&self) -> usize {
        self.row_ids.len()
    }

    /// Per-row eligibility for pairing. Rows with null IDs are never eligible.
    pub fn selected(&self) -> &[bool] {
        &self.selected
    }

    /// Restrict pairing to rows whose current row ID satisfies `keep`,
    /// evaluated once per row rather than once per pair.
    pub fn filter_rows(&mut self, keep: impl Fn(u64) -> bool) {
        for (selected, id) in self.selected.iter_mut().zip(self.row_ids.iter()) {
            *selected = *selected && id.is_some_and(&keep);
        }
    }
}

/// Qualifying pairs of one scoring block, ordered by anchor row, then by
/// candidate row, in storage order.
#[derive(Debug, Default, PartialEq)]
pub struct PairwiseHits {
    pub row_id_a: Vec<u64>,
    pub row_id_b: Vec<u64>,
    pub distances: Vec<f32>,
}

/// Native distances from one anchor row to a range of candidate rows.
pub(crate) trait PairKernel {
    fn distances(&self, anchor_row: usize, candidates: Range<usize>, out: &mut [f32]);
}

/// A quantizer's staged layout and code-to-code kernel.
pub(crate) trait PairScorer {
    type Kernel<'a>: PairKernel
    where
        Self: 'a;

    /// Staged payload bytes per row, excluding row IDs.
    fn row_bytes(&self) -> usize;

    /// Stage `rows` of a source index batch as payload columns.
    fn stage(&self, source: &RecordBatch, rows: Range<usize>) -> Result<Vec<(Field, ArrayRef)>>;

    /// Bind the kernel to two staged batches.
    fn kernel<'a>(
        &'a self,
        anchor: &'a RecordBatch,
        candidates: &'a RecordBatch,
    ) -> Result<Self::Kernel<'a>>;
}

pub(crate) enum PairwiseScorer {
    Flat(FlatPairScorer),
    Product(PQPairScorer),
    Scalar(SQPairScorer),
    Rabit(RQPairScorer),
}

impl PairwiseScorer {
    /// `centroid` is the partition centroid; `source` is the index file schema.
    pub(crate) fn new(
        quantizer: &Quantizer,
        centroid: ArrayRef,
        metric: DistanceType,
        source: &Schema,
    ) -> Result<Self> {
        Ok(match quantizer {
            Quantizer::Flat(_) | Quantizer::FlatBin(_) => {
                Self::Flat(FlatPairScorer::new(quantizer.column(), source, metric)?)
            }
            Quantizer::Product(pq) => Self::Product(PQPairScorer::new(pq, &centroid, metric)?),
            Quantizer::Scalar(sq) => Self::Scalar(SQPairScorer::new(sq, metric)?),
            Quantizer::Rabit(rq) => Self::Rabit(RQPairScorer::new(rq, centroid, metric)?),
        })
    }

    /// Estimated staged bytes per row, including row IDs.
    pub(crate) fn row_bytes(&self) -> usize {
        STAGED_ROW_OVERHEAD_BYTES
            + match self {
                Self::Flat(scorer) => scorer.row_bytes(),
                Self::Product(scorer) => scorer.row_bytes(),
                Self::Scalar(scorer) => scorer.row_bytes(),
                Self::Rabit(scorer) => scorer.row_bytes(),
            }
    }

    /// Stage `rows` of a source index batch once: remapped row IDs followed by
    /// the quantizer payload. Replays never touch the source index again.
    pub(crate) fn stage(
        &self,
        source: &RecordBatch,
        rows: Range<usize>,
        remapper: Option<&dyn RowIdRemapper>,
    ) -> Result<RecordBatch> {
        let ids = source
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::internal("index batch missing row IDs"))?
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(|| Error::internal("index row IDs must be UInt64"))?
            .slice(rows.start, rows.len());
        let ids = match remapper {
            Some(remapper) => ids
                .iter()
                .map(|id| id.and_then(|id| remapper.remap_row_id(id)))
                .collect::<UInt64Array>(),
            None => ids,
        };
        let payload = match self {
            Self::Flat(scorer) => scorer.stage(source, rows)?,
            Self::Product(scorer) => scorer.stage(source, rows)?,
            Self::Scalar(scorer) => scorer.stage(source, rows)?,
            Self::Rabit(scorer) => scorer.stage(source, rows)?,
        };
        let mut fields = Vec::with_capacity(payload.len() + 1);
        let mut columns = Vec::with_capacity(payload.len() + 1);
        fields.push(Field::new(ROW_ID, DataType::UInt64, true));
        columns.push(Arc::new(ids) as ArrayRef);
        for (field, column) in payload {
            fields.push(field);
            columns.push(column);
        }
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            columns,
        )?)
    }

    fn score_block(
        &self,
        anchor: &PairwiseVectorBatch,
        anchor_rows: Range<usize>,
        candidates: &PairwiseVectorBatch,
        threshold: f32,
    ) -> Result<PairwiseHits> {
        fn score<S: PairScorer>(
            scorer: &S,
            anchor: &PairwiseVectorBatch,
            anchor_rows: Range<usize>,
            candidates: &PairwiseVectorBatch,
            threshold: f32,
        ) -> Result<PairwiseHits> {
            let kernel = scorer.kernel(&anchor.codes, &candidates.codes)?;
            let chunk_rows =
                (CANDIDATE_CHUNK_BYTES / scorer.row_bytes().max(1)).max(MIN_CANDIDATE_CHUNK_ROWS);
            Ok(collect_pairs(
                &kernel,
                chunk_rows,
                anchor,
                anchor_rows,
                candidates,
                threshold,
            ))
        }
        match self {
            Self::Flat(s) => score(s, anchor, anchor_rows, candidates, threshold),
            Self::Product(s) => score(s, anchor, anchor_rows, candidates, threshold),
            Self::Scalar(s) => score(s, anchor, anchor_rows, candidates, threshold),
            Self::Rabit(s) => score(s, anchor, anchor_rows, candidates, threshold),
        }
    }
}

/// Score anchor rows against every later candidate row. Candidates are processed
/// in cache-sized chunks with anchors in the inner loop; hits are buffered per
/// anchor row so the output stays row-major.
fn collect_pairs<K: PairKernel>(
    kernel: &K,
    chunk_rows: usize,
    anchor: &PairwiseVectorBatch,
    anchor_rows: Range<usize>,
    candidates: &PairwiseVectorBatch,
    threshold: f32,
) -> PairwiseHits {
    let is_diagonal = anchor.batch_id == candidates.batch_id;
    let num_candidates = candidates.num_rows();
    let first = if is_diagonal {
        anchor_rows.start + 1
    } else {
        0
    };
    let a_ids = anchor.row_ids.values();
    let b_ids = candidates.row_ids.values();
    let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); anchor_rows.len()];
    let mut scratch = vec![0.0f32; chunk_rows.min(num_candidates)];
    let mut chunk_start = first;
    while chunk_start < num_candidates {
        let chunk_end = chunk_start.saturating_add(chunk_rows).min(num_candidates);
        for (a, hits) in anchor_rows.clone().zip(rows.iter_mut()) {
            if !anchor.selected[a] {
                continue;
            }
            let start = if is_diagonal {
                chunk_start.max(a + 1)
            } else {
                chunk_start
            };
            if start >= chunk_end {
                continue;
            }
            let out = &mut scratch[..chunk_end - start];
            kernel.distances(a, start..chunk_end, out);
            let a_id = a_ids[a];
            for (b, &distance) in (start..chunk_end).zip(out.iter()) {
                // NaN fails the threshold test; infinite distances are dropped too.
                if distance <= threshold
                    && distance.is_finite()
                    && candidates.selected[b]
                    && b_ids[b] != a_id
                {
                    hits.push((b as u32, distance));
                }
            }
        }
        chunk_start = chunk_end;
    }
    let total = rows.iter().map(Vec::len).sum();
    let mut output = PairwiseHits {
        row_id_a: Vec::with_capacity(total),
        row_id_b: Vec::with_capacity(total),
        distances: Vec::with_capacity(total),
    };
    for (a, hits) in anchor_rows.zip(rows) {
        output
            .row_id_a
            .extend(std::iter::repeat_n(a_ids[a], hits.len()));
        for (b, distance) in hits {
            output.row_id_b.push(b_ids[b as usize]);
            output.distances.push(distance);
        }
    }
    output
}

/// `1 − cos` of two vectors from their squared L2 distance and norms.
/// Identical vectors give exactly zero; a zero norm gives NaN.
#[inline]
pub(crate) fn cosine_from_l2(l2: f32, anchor_norm: f64, candidate_norm: f64) -> f32 {
    let denominator = 2.0 * anchor_norm * candidate_norm;
    if denominator > 0.0 {
        let norm_gap = anchor_norm - candidate_norm;
        ((f64::from(l2) - norm_gap * norm_gap) / denominator).clamp(0.0, 2.0) as f32
    } else {
        f32::NAN
    }
}

/// Convert squared L2 distances to cosine distances in place.
pub(crate) fn l2_to_cosine(distances: &mut [f32], anchor_norm: f64, candidate_norms: &[f64]) {
    for (distance, &norm) in distances.iter_mut().zip(candidate_norms) {
        *distance = cosine_from_l2(*distance, anchor_norm, norm);
    }
}

pub(crate) fn norm_field() -> Field {
    Field::new(NORM_COLUMN, DataType::Float64, false)
}

/// Values of a primitive staged column.
pub(crate) fn column_values<'a, T: ArrowPrimitiveType>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a [T::Native]> {
    Ok(batch
        .column_by_name(name)
        .and_then(|column| column.as_primitive_opt::<T>())
        .ok_or_else(|| {
            Error::internal(format!(
                "pairwise batch missing {} column {name}",
                T::DATA_TYPE
            ))
        })?
        .values())
}

/// Flattened values of a fixed-size-list staged column.
pub(crate) fn list_values<'a, T: ArrowPrimitiveType>(
    batch: &'a RecordBatch,
    name: &str,
) -> Result<&'a [T::Native]> {
    Ok(batch
        .column_by_name(name)
        .and_then(|column| column.as_fixed_size_list_opt())
        .and_then(|list| list.values().as_primitive_opt::<T>())
        .ok_or_else(|| {
            Error::internal(format!(
                "pairwise batch missing {} list column {name}",
                T::DATA_TYPE
            ))
        })?
        .values())
}

pub(crate) enum EncodedPartition {
    Memory(Vec<RecordBatch>),
    Spilled(SpilledPartition),
}

pub(crate) struct SpilledPartition {
    ranges: Vec<Range<usize>>,
    reader: Box<dyn Reader>,
    // Keep the spill alive until its reader has been dropped.
    _spill: Box<dyn Spill>,
}

impl SpilledPartition {
    async fn read_batch(&self, batch_id: usize) -> Result<RecordBatch> {
        let range = self.ranges.get(batch_id).ok_or_else(|| {
            Error::invalid_input(format!("pairwise spill batch_id={batch_id} out of range"))
        })?;
        let bytes = self.reader.get_range(range.clone()).await?;
        spawn_cpu(move || {
            StreamReader::try_new(Cursor::new(bytes), None)?
                .next()
                .ok_or_else(|| Error::internal("empty pairwise spill batch"))?
                .map_err(Error::from)
        })
        .await
    }
}

/// Invocation-owned staged codes and quantizer state. Replays do not access
/// the original index or source-table vectors.
pub struct PairwisePartition {
    pub(crate) encoded: EncodedPartition,
    pub(crate) scorer: Arc<PairwiseScorer>,
    pub(crate) batch_size: usize,
    pub(crate) num_rows: usize,
}

impl PairwisePartition {
    /// Number of rows per staged batch, possibly smaller than the requested
    /// maximum to bound wide code arrays. The final batch can contain fewer rows.
    pub fn vector_batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn num_batches(&self) -> usize {
        self.num_rows.div_ceil(self.batch_size)
    }

    /// Estimated staged bytes per row, a proxy for per-pair scoring work.
    pub fn row_bytes(&self) -> usize {
        self.scorer.row_bytes()
    }

    /// Whether staged batches live in spill storage. Each spilled
    /// [`Self::read_vectors`] decodes a fresh copy of the batch, while
    /// in-memory reads share the staged buffers.
    pub fn is_spilled(&self) -> bool {
        matches!(self.encoded, EncodedPartition::Spilled(_))
    }

    /// Read one staged batch. All rows with non-null IDs start out selected.
    pub async fn read_vectors(&self, batch_id: usize) -> Result<PairwiseVectorBatch> {
        let codes = match &self.encoded {
            EncodedPartition::Memory(batches) => {
                batches.get(batch_id).cloned().ok_or_else(|| {
                    Error::invalid_input(format!("pairwise batch_id={batch_id} out of range"))
                })?
            }
            EncodedPartition::Spilled(spill) => spill.read_batch(batch_id).await?,
        };
        let row_ids = codes
            .column_by_name(ROW_ID)
            .and_then(|ids| ids.as_primitive_opt::<UInt64Type>())
            .ok_or_else(|| Error::internal("pairwise batch missing row IDs"))?
            .clone();
        let selected = (0..row_ids.len()).map(|i| row_ids.is_valid(i)).collect();
        Ok(PairwiseVectorBatch {
            row_ids,
            codes,
            batch_id,
            selected,
        })
    }

    /// Score `anchor_rows` of `anchor` against every later row of `candidates`
    /// (all of its rows for a later batch, rows after the anchor within the
    /// same batch) and keep selected pairs with distinct IDs and finite
    /// `distance <= threshold`. Hits are ordered by anchor row, then candidate row.
    ///
    /// ```
    /// # use lance_index::vector::pairwise::PairwisePartition;
    /// # async fn example(partition: &PairwisePartition) -> lance_core::Result<()> {
    /// let batch = partition.read_vectors(0).await?;
    /// let hits = partition.score_block(&batch, 0..batch.num_rows(), &batch, 0.05)?;
    /// assert!(hits.row_id_a.len() < batch.num_rows() * batch.num_rows());
    /// # Ok(()) }
    /// ```
    pub fn score_block(
        &self,
        anchor: &PairwiseVectorBatch,
        anchor_rows: Range<usize>,
        candidates: &PairwiseVectorBatch,
        threshold: f32,
    ) -> Result<PairwiseHits> {
        if candidates.batch_id < anchor.batch_id {
            return Err(Error::invalid_input(format!(
                "pairwise candidate batch_id={} precedes anchor batch_id={}",
                candidates.batch_id, anchor.batch_id
            )));
        }
        if anchor_rows.start > anchor_rows.end || anchor_rows.end > anchor.num_rows() {
            return Err(Error::invalid_input(format!(
                "pairwise anchor rows {anchor_rows:?} out of range for {} rows",
                anchor.num_rows()
            )));
        }
        self.scorer
            .score_block(anchor, anchor_rows, candidates, threshold)
    }
}

/// Each Arrow stream is independently readable, so replay needs one local
/// range read, without opening a file or scanning earlier batches.
pub(crate) struct PairwiseSpillWriter {
    writer: Box<dyn Writer>,
    spill: Box<dyn Spill>,
    ranges: Vec<Range<usize>>,
    offset: usize,
}

impl PairwiseSpillWriter {
    pub(crate) async fn new(store: &dyn SpillStore) -> Result<Self> {
        let (writer, spill) = store.new_spill().await?;
        Ok(Self {
            writer,
            spill,
            ranges: Vec::new(),
            offset: 0,
        })
    }

    pub(crate) async fn write(&mut self, batch: RecordBatch) -> Result<()> {
        let bytes = spawn_cpu(move || -> Result<Vec<u8>> {
            let mut writer = StreamWriter::try_new(Vec::new(), batch.schema_ref())?;
            writer.write(&batch)?;
            writer.finish()?;
            Ok(writer.into_inner()?)
        })
        .await?;
        let end = self
            .offset
            .checked_add(bytes.len())
            .ok_or_else(|| Error::invalid_input("pairwise spill offset overflow"))?;
        self.writer.write_all(&bytes).await?;
        self.ranges.push(self.offset..end);
        self.offset = end;
        Ok(())
    }

    pub(crate) async fn finish(mut self) -> Result<SpilledPartition> {
        Writer::shutdown(self.writer.as_mut()).await?;
        drop(self.writer);
        let reader = self.spill.reader().await?;
        Ok(SpilledPartition {
            ranges: self.ranges,
            reader,
            _spill: self.spill,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::flat::index::FlatQuantizer;
    use crate::vector::quantizer::Quantization;
    use arrow_array::types::{Float32Type, UInt64Type};
    use lance_datagen::{array, gen_batch};
    use lance_io::spill::LocalSpillStore;
    use rstest::rstest;

    fn flat_batch() -> RecordBatch {
        let quantizer = FlatQuantizer::new(2, DistanceType::L2);
        gen_batch()
            .col(ROW_ID, array::step::<UInt64Type>())
            .col(quantizer.column(), array::rand_vec::<Float32Type>(2.into()))
            .into_batch_rows(32.into())
            .unwrap()
    }

    fn flat_scorer() -> PairwiseScorer {
        let batch = flat_batch();
        PairwiseScorer::new(
            &Quantizer::Flat(FlatQuantizer::new(2, DistanceType::L2)),
            Arc::new(arrow_array::Float32Array::from(vec![0.0f32; 2])),
            DistanceType::L2,
            batch.schema_ref(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_pairwise_spill_releases_disk_budget() {
        let batch = flat_batch();
        let store = LocalSpillStore::default();
        let mut writer = PairwiseSpillWriter::new(&store).await.unwrap();
        writer.write(batch.clone()).await.unwrap();
        let encoded = writer.finish().await.unwrap();
        let bytes = encoded.reader.size().await.unwrap();
        let spill_path =
            std::path::PathBuf::from(lance_io::local::to_local_path(encoded.reader.path()));
        let spill_dir = spill_path.parent().unwrap().to_owned();
        drop(encoded);
        let mut aborted = PairwiseSpillWriter::new(&store).await.unwrap();
        aborted.write(batch.clone()).await.unwrap();
        drop(aborted);
        assert_eq!(
            std::fs::read_dir(spill_dir).unwrap().count(),
            0,
            "cancelled preparation must remove temporary files"
        );

        let capped = LocalSpillStore::with_cap(bytes as u64).unwrap();
        let mut writer = PairwiseSpillWriter::new(&capped).await.unwrap();
        writer.write(batch.clone()).await.unwrap();
        let held = writer.finish().await.unwrap();
        let mut blocked = PairwiseSpillWriter::new(&capped).await.unwrap();
        let err = blocked.write(batch.clone()).await.unwrap_err();
        assert!(matches!(err, Error::DiskCapExceeded { .. }), "{err}");
        assert!(err.to_string().contains("cap"));
        drop(blocked);
        drop(held);

        // Dropping the prepared data must release its disk reservation, so the
        // same session can prepare another partition without leaking quota.
        let mut replacement = PairwiseSpillWriter::new(&capped).await.unwrap();
        replacement.write(batch).await.unwrap();
        drop(replacement.finish().await.unwrap());
    }

    #[tokio::test]
    async fn test_pairwise_spill_roundtrip_and_bounds() {
        let scorer = flat_scorer();
        let batch = flat_batch();
        let staged = [
            scorer.stage(&batch, 0..32, None).unwrap(),
            scorer.stage(&batch, 0..1, None).unwrap(),
        ];
        let store = LocalSpillStore::default();
        let mut writer = PairwiseSpillWriter::new(&store).await.unwrap();
        for batch in &staged {
            writer.write(batch.clone()).await.unwrap();
        }
        let prepared = PairwisePartition {
            encoded: EncodedPartition::Spilled(writer.finish().await.unwrap()),
            scorer: Arc::new(scorer),
            batch_size: 32,
            num_rows: 33,
        };
        assert_eq!(prepared.num_batches(), 2);
        for _ in 0..3 {
            for (id, expected) in staged.iter().enumerate() {
                let codes = prepared.read_vectors(id).await.unwrap();
                assert_eq!(codes.batch_id(), id);
                assert_eq!(&codes.codes, expected);
                assert_eq!(
                    &codes.row_ids,
                    expected[ROW_ID].as_primitive::<UInt64Type>()
                );
            }
        }
        for id in [2, usize::MAX] {
            let err = prepared.read_vectors(id).await.unwrap_err();
            assert!(matches!(err, Error::InvalidInput { .. }));
            assert!(err.to_string().contains("batch_id"));
        }
    }

    /// Distance `|value_a - value_b|` over one scalar per row.
    struct AbsDiff<'a> {
        anchor: &'a [f32],
        candidates: &'a [f32],
    }

    impl PairKernel for AbsDiff<'_> {
        fn distances(&self, anchor_row: usize, candidates: Range<usize>, out: &mut [f32]) {
            let a = self.anchor[anchor_row];
            for (out, &b) in out.iter_mut().zip(&self.candidates[candidates]) {
                *out = (a - b).abs();
            }
        }
    }

    fn scalar_batch(batch_id: usize, ids: Vec<Option<u64>>) -> PairwiseVectorBatch {
        let row_ids = UInt64Array::from(ids);
        let codes = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                ROW_ID,
                DataType::UInt64,
                true,
            )])),
            vec![Arc::new(row_ids.clone())],
        )
        .unwrap();
        let selected = (0..row_ids.len()).map(|i| row_ids.is_valid(i)).collect();
        PairwiseVectorBatch {
            row_ids,
            codes,
            batch_id,
            selected,
        }
    }

    fn pairs(hits: &PairwiseHits) -> Vec<(u64, u64, f32)> {
        hits.row_id_a
            .iter()
            .zip(&hits.row_id_b)
            .zip(&hits.distances)
            .map(|((&a, &b), &d)| (a, b, d))
            .collect()
    }

    #[rstest]
    fn test_collect_pairs_tile_order(#[values(1, 2, 3, 64)] chunk_rows: usize) {
        let values_a = [0.0f32, 1.0, 1.0, 5.0, 2.0];
        let values_b = [1.0f32, 0.5, 9.0, 0.0];
        let mut anchor = scalar_batch(0, vec![Some(10), Some(11), None, Some(13), Some(14)]);
        // Row 14 is filtered out once per batch, as a deleted row would be.
        anchor.filter_rows(|id| id != 14);
        let candidates = scalar_batch(1, vec![Some(20), Some(21), Some(22), Some(10)]);
        let diagonal = AbsDiff {
            anchor: &values_a,
            candidates: &values_a,
        };
        // Diagonal tile: upper triangle only; null and filtered rows are skipped
        // both as anchors and candidates; the threshold is inclusive.
        let hits = collect_pairs(&diagonal, chunk_rows, &anchor, 0..5, &anchor, 1.0);
        assert_eq!(pairs(&hits), vec![(10, 11, 1.0)]);
        let hits = collect_pairs(&diagonal, chunk_rows, &anchor, 1..3, &anchor, 4.0);
        assert_eq!(pairs(&hits), vec![(11, 13, 4.0)]);

        // Off-diagonal tile: every candidate row, row-major, excluding the
        // anchor's own ID (a remapper can alias two rows to one ID).
        let tile = AbsDiff {
            anchor: &values_a,
            candidates: &values_b,
        };
        let hits = collect_pairs(&tile, chunk_rows, &anchor, 0..5, &candidates, 1.0);
        assert_eq!(
            pairs(&hits),
            vec![
                (10, 20, 1.0),
                (10, 21, 0.5),
                (11, 20, 0.0),
                (11, 21, 0.5),
                (11, 10, 1.0),
            ]
        );
    }

    #[test]
    fn test_collect_pairs_drops_non_finite() {
        let values = [0.0f32, f32::NAN, f32::NEG_INFINITY, 0.0];
        let batch = scalar_batch(0, (0..4).map(Some).collect());
        let kernel = AbsDiff {
            anchor: &values,
            candidates: &values,
        };
        let hits = collect_pairs(&kernel, 2, &batch, 0..4, &batch, f32::MAX);
        assert_eq!(pairs(&hits), vec![(0, 3, 0.0)]);
    }

    #[test]
    fn test_cosine_from_l2() {
        assert_eq!(cosine_from_l2(0.0, 3.0, 3.0), 0.0);
        // [1, 0] vs [0, 2]: orthogonal.
        assert_eq!(cosine_from_l2(5.0, 1.0, 2.0), 1.0);
        // [1, 0] vs [-2, 0]: opposite.
        assert_eq!(cosine_from_l2(9.0, 1.0, 2.0), 2.0);
        // Roundoff outside [0, 2] is clamped.
        assert_eq!(cosine_from_l2(9.5, 1.0, 2.0), 2.0);
        assert!(cosine_from_l2(1.0, 0.0, 1.0).is_nan());
    }

    #[tokio::test]
    async fn test_score_block_validates_tiles() {
        let scorer = flat_scorer();
        let batch = flat_batch();
        let prepared = PairwisePartition {
            encoded: EncodedPartition::Memory(vec![
                scorer.stage(&batch, 0..16, None).unwrap(),
                scorer.stage(&batch, 16..32, None).unwrap(),
            ]),
            scorer: Arc::new(scorer),
            batch_size: 16,
            num_rows: 32,
        };
        let first = prepared.read_vectors(0).await.unwrap();
        let second = prepared.read_vectors(1).await.unwrap();
        let err = prepared
            .score_block(&second, 0..1, &first, 1.0)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains("precedes"), "{err}");
        let err = prepared
            .score_block(&first, 0..17, &second, 1.0)
            .unwrap_err();
        assert!(err.to_string().contains("out of range"), "{err}");
        let all = prepared
            .score_block(&first, 0..16, &second, f32::MAX)
            .unwrap();
        assert_eq!(all.row_id_a.len(), 16 * 16);
        let diagonal = prepared
            .score_block(&first, 0..16, &first, f32::MAX)
            .unwrap();
        assert_eq!(diagonal.row_id_a.len(), 16 * 15 / 2);
    }
}
