// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded reconstruction of index representations for symmetric pair scoring.

use std::{collections::VecDeque, io::Cursor, ops::Range, sync::Arc};

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type, UInt8Type};
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float64Array, RecordBatch, UInt64Array};
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, ROW_ID, Result};
use lance_io::{
    spill::{Spill, SpillStore},
    traits::{Reader, Writer},
};
use lance_linalg::distance::DistanceType;
use tokio::io::AsyncWriteExt;
use tokio::sync::OnceCell;

use crate::scalar::RowIdRemapper;

use super::bq::ex_dot::{blocked_ex_code_bytes, unpack_blocked_row};
use super::bq::storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RabitQueryEstimator, unpack_codes};
use super::bq::transform::{EX_SCALE_FACTORS_COLUMN, SCALE_FACTORS_COLUMN};
use super::quantizer::{Quantization, Quantizer};

/// Internal index batch. Vectors may be reconstructed and, for RQ, rotated.
/// All batches of a partition use the same coordinate system.
#[derive(Clone, Debug)]
pub struct PairwiseVectorBatch {
    pub row_ids: UInt64Array,
    pub vectors: FixedSizeListArray,
}

/// Upper bound for an in-memory encoded partition. Larger partitions are
/// staged in the session's spill store, independently of decoded vector batches.
pub const PAIRWISE_MEMORY_LIMIT: usize = 16 * 1024 * 1024;

pub(crate) enum EncodedPartition {
    Memory(RecordBatch),
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

/// An invocation-owned, prepared partition. Reads replay compact index codes;
/// this object deliberately has no reader for the original index file.
pub struct PairwisePartition {
    pub(crate) encoded: EncodedPartition,
    pub(crate) quantizer: Arc<Quantizer>,
    pub(crate) centroid: ArrayRef,
    pub(crate) metric: DistanceType,
    pub(crate) remapper: Option<Arc<dyn RowIdRemapper>>,
    pub(crate) batch_size: usize,
    pub(crate) num_rows: usize,
    pub(crate) rotated_center: OnceCell<Arc<Vec<f32>>>,
}

impl PairwisePartition {
    /// Decode one vector batch in storage order. The batch size is fixed when
    /// preparing the partition so RQ packed groups remain aligned.
    pub async fn read_vectors(&self, batch_id: usize) -> Result<PairwiseVectorBatch> {
        let start = batch_id
            .checked_mul(self.batch_size)
            .filter(|&start| start < self.num_rows)
            .ok_or_else(|| {
                Error::invalid_input(format!("pairwise batch_id={batch_id} out of range"))
            })?;
        let len = self.batch_size.min(self.num_rows - start);
        let batch = match &self.encoded {
            EncodedPartition::Memory(batch) => batch.slice(start, len),
            EncodedPartition::Spilled(spill) => spill.read_batch(batch_id).await?,
        };
        let quantizer = self.quantizer.clone();
        let centroid = self.centroid.clone();
        let rotated_center = if matches!(quantizer.as_ref(), Quantizer::Rabit(_)) {
            Some(
                self.rotated_center
                    .get_or_try_init(|| async {
                        let quantizer = quantizer.clone();
                        let centroid = centroid.clone();
                        spawn_cpu(move || {
                            let Quantizer::Rabit(rq) = quantizer.as_ref() else {
                                return Err(Error::internal("expected RQ quantizer"));
                            };
                            let dim = centroid.len();
                            let centroid =
                                FixedSizeListArray::try_new_from_values(centroid, dim as i32)?;
                            rq.rotate_fsl_to_f32(&centroid).map(Arc::new)
                        })
                        .await
                    })
                    .await?
                    .clone(),
            )
        } else {
            None
        };
        let metric = self.metric;
        let remapper = self.remapper.clone();
        let is_flat = matches!(
            quantizer.as_ref(),
            Quantizer::Flat(_) | Quantizer::FlatBin(_)
        );
        let decode = move || {
            let vectors = reconstruct(&quantizer, &batch, centroid, rotated_center, metric)?;
            let ids = batch
                .column_by_name(ROW_ID)
                .ok_or_else(|| Error::internal("index batch missing row IDs"))?
                .as_primitive::<arrow_array::types::UInt64Type>();
            let row_ids = if let Some(remapper) = remapper {
                // Preserve physical positions even when compaction removed a row.
                UInt64Array::from(
                    ids.iter()
                        .map(|id| id.and_then(|id| remapper.remap_row_id(id)))
                        .collect::<Vec<_>>(),
                )
            } else {
                ids.clone()
            };
            Ok(PairwiseVectorBatch { row_ids, vectors })
        };
        if is_flat {
            // Flat batches only clone Arrow views and map at most one batch
            // of row IDs; offloading that work costs more than the work itself.
            decode()
        } else {
            spawn_cpu(decode).await
        }
    }

    /// Reconstruct every batch once, retaining decoded vectors within the given
    /// cache budget or writing independently readable decoded spill records.
    /// Cache the tail: upper-triangle traversal revisits later batches most.
    /// Consuming `self` releases the encoded staging once reconstruction finishes.
    /// The budget excludes one in-flight batch, models, and spill range metadata.
    pub async fn materialize(
        self,
        decoded_cache_size: usize,
        spill_store: &dyn SpillStore,
    ) -> Result<DecodedPairwisePartition> {
        let mut batches = VecDeque::<(PairwiseVectorBatch, usize)>::new();
        let mut retained_bytes = 0usize;
        let mut writer: Option<PairwiseSpillWriter> = None;
        for batch_id in 0..self.num_rows.div_ceil(self.batch_size) {
            let mut batch = self.read_vectors(batch_id).await?;
            // An IPC row-ID slice can pin the entire encoded message. Keep
            // owned IDs so decoded caching releases the quantized buffers.
            batch.row_ids = UInt64Array::from(batch.row_ids.iter().collect::<Vec<_>>());
            let bytes = batch
                .row_ids
                .get_array_memory_size()
                .checked_add(batch.vectors.get_array_memory_size())
                .ok_or_else(|| Error::invalid_input("decoded pairwise batch size overflow"))?;
            let next_bytes = retained_bytes.checked_add(bytes);
            if writer.is_none() && next_bytes.is_none_or(|size| size > decoded_cache_size) {
                writer = Some(PairwiseSpillWriter::new(spill_store).await?);
            }
            while !batches.is_empty()
                && retained_bytes
                    .checked_add(bytes)
                    .is_none_or(|size| size > decoded_cache_size)
            {
                let (cached, cached_bytes) = batches
                    .pop_front()
                    .ok_or_else(|| Error::internal("missing decoded cache entry"))?;
                retained_bytes -= cached_bytes;
                writer
                    .as_mut()
                    .ok_or_else(|| Error::internal("missing decoded spill writer"))?
                    .write(decoded_record_batch(cached)?)
                    .await?;
            }
            if bytes > decoded_cache_size {
                writer
                    .as_mut()
                    .ok_or_else(|| Error::internal("missing decoded spill writer"))?
                    .write(decoded_record_batch(batch)?)
                    .await?;
            } else {
                retained_bytes = retained_bytes.checked_add(bytes).ok_or_else(|| {
                    Error::invalid_input("decoded pairwise partition size overflow")
                })?;
                batches.push_back((batch, bytes));
            }
        }
        let cached = batches.into_iter().map(|(batch, _)| batch).collect();
        let storage = if let Some(writer) = writer {
            DecodedStorage::Spilled {
                prefix: writer.finish().await?,
                cached,
            }
        } else {
            DecodedStorage::Memory(cached)
        };
        Ok(DecodedPairwisePartition { storage })
    }
}

enum DecodedStorage {
    Memory(Vec<PairwiseVectorBatch>),
    Spilled {
        prefix: SpilledPartition,
        cached: Vec<PairwiseVectorBatch>,
    },
}

/// Read-only reconstructed vectors. Replay never invokes a quantizer or reads
/// the original index, even when decoded vectors exceed the memory budget.
pub struct DecodedPairwisePartition {
    storage: DecodedStorage,
}

impl DecodedPairwisePartition {
    /// Read a decoded batch by its zero-based position in the partition.
    pub async fn read_vectors(&self, batch_id: usize) -> Result<PairwiseVectorBatch> {
        match &self.storage {
            DecodedStorage::Memory(batches) => batches.get(batch_id).cloned().ok_or_else(|| {
                Error::invalid_input(format!("decoded pairwise batch_id={batch_id} out of range"))
            }),
            DecodedStorage::Spilled { prefix, cached } => {
                if let Some(cached_id) = batch_id.checked_sub(prefix.ranges.len()) {
                    return cached.get(cached_id).cloned().ok_or_else(|| {
                        Error::invalid_input(format!(
                            "decoded pairwise batch_id={batch_id} out of range"
                        ))
                    });
                }
                let batch = prefix.read_batch(batch_id).await?;
                let row_ids = batch
                    .column_by_name(ROW_ID)
                    .and_then(|array| array.as_primitive_opt::<arrow_array::types::UInt64Type>())
                    .ok_or_else(|| Error::internal("decoded pairwise spill missing row IDs"))?
                    .clone();
                let vectors = batch
                    .column_by_name("vector")
                    .and_then(|array| array.as_fixed_size_list_opt())
                    .ok_or_else(|| Error::internal("decoded pairwise spill missing vectors"))?
                    .clone();
                Ok(PairwiseVectorBatch { row_ids, vectors })
            }
        }
    }
}

fn decoded_record_batch(batch: PairwiseVectorBatch) -> Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(ROW_ID, DataType::UInt64, true),
        Field::new("vector", batch.vectors.data_type().clone(), true),
    ]));
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(batch.row_ids), Arc::new(batch.vectors)],
    )?)
}

/// Each Arrow stream is independently readable, so replay needs one local
/// range read, without opening a file or scanning earlier batches per anchor.
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

fn float_values(array: &dyn Array) -> Result<Vec<f64>> {
    match array.data_type() {
        DataType::Float16 => Ok(array
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .map(|v| v.to_f64())
            .collect()),
        DataType::Float32 => Ok(array
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .map(|&v| f64::from(v))
            .collect()),
        DataType::Float64 => Ok(array.as_primitive::<Float64Type>().values().to_vec()),
        other => Err(Error::not_supported(format!(
            "pair reconstruction requires floating-point values, got {other}"
        ))),
    }
}

fn codes<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a FixedSizeListArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_fixed_size_list_opt())
        .ok_or_else(|| Error::invalid_input(format!("missing fixed-size vector column {name}")))
}

/// Reconstruct only the supplied batch, never fetching source-table vectors.
/// Quantized pair distances are distances between these reconstructions, not
/// asymmetric query-to-code estimates. This gives identical codes zero L2.
pub(crate) fn reconstruct(
    quantizer: &Quantizer,
    batch: &RecordBatch,
    centroid: ArrayRef,
    rotated_center: Option<Arc<Vec<f32>>>,
    metric: DistanceType,
) -> Result<FixedSizeListArray> {
    let encoded = codes(batch, quantizer.column())?;
    if matches!(quantizer, Quantizer::Flat(_) | Quantizer::FlatBin(_)) {
        return Ok(encoded.clone());
    }
    let center = if matches!(quantizer, Quantizer::Product(_)) {
        float_values(centroid.as_ref())?
    } else {
        Vec::new()
    };
    let dim;
    let mut values = Vec::new();
    match quantizer {
        Quantizer::Product(pq) => {
            dim = pq.dimension;
            let width = dim / pq.num_sub_vectors;
            let codebook = float_values(pq.codebook.values().as_ref())?;
            let num_centroids = 1usize << pq.num_bits;
            let raw = encoded.values().as_primitive::<UInt8Type>();
            values.reserve(batch.num_rows() * dim);
            for row in 0..batch.num_rows() {
                let bytes = &raw.values()[row * encoded.value_length() as usize
                    ..(row + 1) * encoded.value_length() as usize];
                for sub in 0..pq.num_sub_vectors {
                    let code = if pq.num_bits == 4 {
                        (bytes[sub / 2] >> (4 * (sub % 2))) & 15
                    } else {
                        bytes[sub]
                    } as usize;
                    let start = (sub * num_centroids + code) * width;
                    for (offset, &value) in codebook[start..start + width].iter().enumerate() {
                        let c = if super::pq::ProductQuantizer::use_residual(metric) {
                            center[sub * width + offset]
                        } else {
                            0.0
                        };
                        values.push(value + c);
                    }
                }
            }
        }
        Quantizer::Scalar(sq) => {
            dim = encoded.value_length() as usize;
            let bounds = sq.bounds();
            let scale = (bounds.end - bounds.start) / 255.0;
            values = encoded
                .values()
                .as_primitive::<UInt8Type>()
                .values()
                .iter()
                .map(|&v| bounds.start + f64::from(v) * scale)
                .collect();
        }
        Quantizer::Rabit(rq) => {
            let meta = rq.metadata_ref();
            if meta.query_estimator != RabitQueryEstimator::RawQuery {
                return Err(Error::not_supported(
                    "pair enumeration requires a current RQ index; rebuild the index",
                ));
            }
            dim = meta.rotated_dim();
            let rotated_center = rotated_center.ok_or_else(|| {
                Error::internal("RQ pair reconstruction missing prepared centroid")
            })?;
            // FastScan packs groups of 32 rows. The reader aligns the batch to
            // those groups before calling this decoder.
            let signs = if meta.packed {
                unpack_codes(encoded)
            } else {
                encoded.clone()
            };
            let signs = signs.values().as_primitive::<UInt8Type>();
            let ex_bits = meta.num_bits - 1;
            let scales_name = if ex_bits == 0 {
                SCALE_FACTORS_COLUMN
            } else {
                EX_SCALE_FACTORS_COLUMN
            };
            let scales = batch
                .column_by_name(scales_name)
                .and_then(|a| a.as_primitive_opt::<Float32Type>())
                .ok_or_else(|| {
                    Error::invalid_input(format!("missing RQ scale column {scales_name}"))
                })?;
            let extended = if ex_bits == 0 {
                None
            } else {
                Some(codes(batch, RABIT_BLOCKED_EX_CODE_COLUMN)?)
            };
            values.reserve(batch.num_rows() * dim);
            for row in 0..batch.num_rows() {
                let ex = if let Some(extended) = extended {
                    let data = extended.value(row);
                    let raw = data.as_primitive::<UInt8Type>();
                    if raw.len() != blocked_ex_code_bytes(dim, ex_bits) {
                        return Err(Error::invalid_input("invalid RQ extended code width"));
                    }
                    unpack_blocked_row(raw.values(), ex_bits, dim)
                } else {
                    Vec::new()
                };
                let scale = -f64::from(scales.value(row))
                    / if metric == DistanceType::Dot {
                        1.0
                    } else {
                        2.0
                    };
                let bias = (1u32 << ex_bits) as f64 - 0.5;
                for d in 0..dim {
                    let sign = (signs.value(row * (dim / 8) + d / 8) >> (d % 8)) & 1;
                    let code = (u32::from(sign) << ex_bits)
                        + if ex_bits == 0 { 0 } else { u32::from(ex[d]) };
                    values.push(f64::from(rotated_center[d]) + scale * (f64::from(code) - bias));
                }
            }
        }
        Quantizer::Flat(_) | Quantizer::FlatBin(_) => return Ok(encoded.clone()),
    }
    Ok(FixedSizeListArray::try_new_from_values(
        Float64Array::from(values),
        dim as i32,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::flat::index::FlatQuantizer;
    use arrow_array::types::{Float32Type, UInt64Type};
    use lance_datagen::{array, gen_batch};
    use lance_io::spill::LocalSpillStore;

    fn flat_batch() -> RecordBatch {
        let quantizer = FlatQuantizer::new(2, DistanceType::L2);
        gen_batch()
            .col(ROW_ID, array::step::<UInt64Type>())
            .col(quantizer.column(), array::rand_vec::<Float32Type>(2.into()))
            .into_batch_rows(32.into())
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
        let batch = flat_batch();
        let store = LocalSpillStore::default();
        let mut writer = PairwiseSpillWriter::new(&store).await.unwrap();
        writer.write(batch.clone()).await.unwrap();
        writer.write(batch.slice(0, 1)).await.unwrap();
        let prepared = PairwisePartition {
            encoded: EncodedPartition::Spilled(writer.finish().await.unwrap()),
            quantizer: Arc::new(Quantizer::Flat(FlatQuantizer::new(2, DistanceType::L2))),
            centroid: Arc::new(arrow_array::Float32Array::from(vec![0.0, 0.0])),
            metric: DistanceType::L2,
            remapper: None,
            batch_size: 32,
            num_rows: 33,
            rotated_center: OnceCell::new(),
        };
        for _ in 0..3 {
            for (id, expected) in [batch.clone(), batch.slice(0, 1)].iter().enumerate() {
                let decoded = prepared.read_vectors(id).await.unwrap();
                assert_eq!(
                    &decoded.row_ids,
                    expected[ROW_ID].as_primitive::<UInt64Type>()
                );
                assert_eq!(
                    &decoded.vectors,
                    expected[prepared.quantizer.column()].as_fixed_size_list()
                );
            }
        }
        for id in [2, usize::MAX] {
            let err = prepared.read_vectors(id).await.unwrap_err();
            assert!(matches!(err, Error::InvalidInput { .. }));
            assert!(err.to_string().contains("batch_id"));
        }
    }

    #[rstest::rstest]
    #[case::spill(0)]
    #[case::spill_after_first_batch(1)]
    #[case::memory(usize::MAX)]
    #[tokio::test]
    async fn test_decoded_replay_releases_encoded_storage(#[case] cached_batches: usize) {
        let batch = flat_batch();
        let quantizer = FlatQuantizer::new(2, DistanceType::L2);
        let store = LocalSpillStore::default();
        let mut writer = PairwiseSpillWriter::new(&store).await.unwrap();
        writer.write(batch.clone()).await.unwrap();
        writer.write(batch.slice(0, 1)).await.unwrap();
        let encoded = writer.finish().await.unwrap();
        let encoded_path =
            std::path::PathBuf::from(lance_io::local::to_local_path(encoded.reader.path()));
        let spill_dir = encoded_path.parent().unwrap().to_owned();
        let prepared = PairwisePartition {
            encoded: EncodedPartition::Spilled(encoded),
            quantizer: Arc::new(Quantizer::Flat(quantizer.clone())),
            centroid: Arc::new(arrow_array::Float32Array::from(vec![0.0, 0.0])),
            metric: DistanceType::L2,
            remapper: None,
            batch_size: 32,
            num_rows: 33,
            rotated_center: OnceCell::new(),
        };
        let mut first = prepared.read_vectors(0).await.unwrap();
        first.row_ids = UInt64Array::from(first.row_ids.iter().collect::<Vec<_>>());
        let batch_bytes =
            first.row_ids.get_array_memory_size() + first.vectors.get_array_memory_size();
        let limit = cached_batches.saturating_mul(batch_bytes);
        let decoded = prepared.materialize(limit, &store).await.unwrap();
        assert!(
            !encoded_path.exists(),
            "encoded staging must be released before replay"
        );
        assert_eq!(
            matches!(decoded.storage, DecodedStorage::Memory(_)),
            cached_batches == usize::MAX
        );
        if let DecodedStorage::Spilled { prefix, cached } = &decoded.storage {
            assert_eq!(prefix.ranges.len(), if cached_batches == 0 { 2 } else { 1 });
            assert_eq!(cached.len(), if cached_batches == 0 { 0 } else { 1 });
        }
        for _ in 0..3 {
            for (id, expected) in [batch.clone(), batch.slice(0, 1)].iter().enumerate() {
                let actual = decoded.read_vectors(id).await.unwrap();
                assert_eq!(
                    &actual.row_ids,
                    expected[ROW_ID].as_primitive::<UInt64Type>()
                );
                assert_eq!(
                    &actual.vectors,
                    expected[quantizer.column()].as_fixed_size_list()
                );
            }
        }
        let error = decoded.read_vectors(2).await.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("batch_id=2"));
        drop(decoded);
        assert_eq!(std::fs::read_dir(spill_dir).unwrap().count(), 0);
    }
}
