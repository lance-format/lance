// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bounded staging and native quantizer batch scoring of index codes.

use super::{
    bq::pairwise::RQCodeDistance,
    pq::storage::PQCodeDistance,
    quantizer::{Quantization, Quantizer, QuantizerStorage},
    sq::{
        ScalarQuantizer,
        storage::{SQDistCalculator, ScalarQuantizationStorage},
    },
    storage::DistCalculator,
};
use crate::scalar::RowIdRemapper;
use arrow_array::cast::AsArray;
use arrow_array::types::{UInt8Type, UInt64Type};
use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, ROW_ID, Result};
use lance_io::{
    spill::{Spill, SpillStore},
    traits::{Reader, Writer},
};
use lance_linalg::distance::DistanceType;
use std::{io::Cursor, ops::Range, sync::Arc};
use tokio::io::AsyncWriteExt;

/// Index codes and row IDs in storage order; quantized vectors are never restored.
#[derive(Clone, Debug)]
pub struct PairwiseVectorBatch {
    pub row_ids: UInt64Array,
    pub codes: RecordBatch,
}

/// Default budget for compact codes. Larger partitions use session spill storage.
pub const PAIRWISE_MEMORY_LIMIT: usize = 256 * 1024 * 1024;

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

pub(crate) enum PairwiseScorer {
    Flat {
        column: &'static str,
        metric: DistanceType,
    },
    Product {
        column: &'static str,
        scorer: PQCodeDistance,
        cosine: bool,
    },
    Scalar {
        quantizer: ScalarQuantizer,
        metric: DistanceType,
    },
    Rabit {
        scorer: RQCodeDistance,
        cosine: bool,
    },
}
impl PairwiseScorer {
    pub(crate) fn new(
        quantizer: &Quantizer,
        centroid: ArrayRef,
        metric: DistanceType,
    ) -> Result<Self> {
        Ok(match quantizer {
            Quantizer::Flat(_) | Quantizer::FlatBin(_) => Self::Flat {
                column: quantizer.column(),
                metric,
            },
            Quantizer::Product(pq) => Self::Product {
                column: pq.column(),
                scorer: PQCodeDistance::new(pq, metric)?,
                cosine: metric == DistanceType::Cosine,
            },
            Quantizer::Scalar(sq) => Self::Scalar {
                quantizer: sq.clone(),
                metric,
            },
            Quantizer::Rabit(rq) => Self::Rabit {
                scorer: RQCodeDistance::new(rq, centroid, metric)?,
                cosine: metric == DistanceType::Cosine,
            },
        })
    }
    pub(crate) fn prepare(&self, batch: RecordBatch) -> Result<RecordBatch> {
        match self {
            Self::Rabit { scorer, .. } => scorer.prepare(batch),
            _ => Ok(batch),
        }
    }
    fn distance_batch(
        &self,
        anchor: &RecordBatch,
        row: usize,
        candidates: &RecordBatch,
    ) -> Result<Vec<f32>> {
        let codes =
            |batch: &RecordBatch, column: &str| -> Result<arrow_array::FixedSizeListArray> {
                Ok(batch
                    .column_by_name(column)
                    .and_then(|c| c.as_fixed_size_list_opt())
                    .ok_or_else(|| {
                        Error::invalid_input(format!("pairwise batch missing code column {column}"))
                    })?
                    .clone())
            };
        let (mut distances, cosine) = match self {
            Self::Flat { column, metric } => {
                let query = codes(anchor, column)?.value(row);
                let distances =
                    metric.arrow_batch_func()(query.as_ref(), &codes(candidates, column)?)?;
                return Ok(distances.values().to_vec());
            }
            Self::Product {
                column,
                scorer,
                cosine,
            } => {
                let query = codes(anchor, column)?.value(row);
                (
                    scorer.distance_batch(
                        query.as_primitive::<UInt8Type>().values(),
                        &codes(candidates, column)?,
                    ),
                    *cosine,
                )
            }
            Self::Scalar { quantizer, metric } => {
                let query = codes(anchor, quantizer.column())?.value(row);
                let storage = ScalarQuantizationStorage::try_from_batch(
                    candidates.clone(),
                    &quantizer.metadata(None),
                    *metric,
                    None,
                )?;
                (
                    SQDistCalculator::from_codes(
                        query.as_primitive::<UInt8Type>().values(),
                        &storage,
                    )
                    .distance_all(candidates.num_rows()),
                    *metric == DistanceType::Cosine,
                )
            }
            Self::Rabit { scorer, cosine } => {
                (scorer.distance_batch(anchor, row, candidates)?, *cosine)
            }
        };
        // Quantized cosine indices use normalized vectors and L2 internally,
        // just as search does. Do not renormalize a quantized representation.
        if cosine {
            distances.iter_mut().for_each(|d| *d *= 0.5);
        }
        Ok(distances)
    }
}

/// Invocation-owned compact codes and quantizer state. Replays do not access
/// the original index or source-table vectors.
pub struct PairwisePartition {
    pub(crate) encoded: EncodedPartition,
    pub(crate) scorer: Arc<PairwiseScorer>,
    pub(crate) remapper: Option<Arc<dyn RowIdRemapper>>,
    pub(crate) batch_size: usize,
    pub(crate) num_rows: usize,
}
impl PairwisePartition {
    /// Read one code batch. Batch boundaries preserve packed RQ group alignment.
    pub async fn read_vectors(&self, batch_id: usize) -> Result<PairwiseVectorBatch> {
        let start = batch_id
            .checked_mul(self.batch_size)
            .filter(|&start| start < self.num_rows)
            .ok_or_else(|| {
                Error::invalid_input(format!("pairwise batch_id={batch_id} out of range"))
            })?;
        let len = self.batch_size.min(self.num_rows - start);
        let codes = match &self.encoded {
            EncodedPartition::Memory(batch) => batch.slice(start, len),
            EncodedPartition::Spilled(spill) => spill.read_batch(batch_id).await?,
        };
        let ids = codes
            .column_by_name(ROW_ID)
            .ok_or_else(|| Error::internal("index batch missing row IDs"))?
            .as_primitive::<UInt64Type>();
        let row_ids = if let Some(remapper) = &self.remapper {
            UInt64Array::from(
                ids.iter()
                    .map(|id| id.and_then(|id| remapper.remap_row_id(id)))
                    .collect::<Vec<_>>(),
            )
        } else {
            ids.clone()
        };
        Ok(PairwiseVectorBatch { row_ids, codes })
    }
    /// Score one anchor against a candidate vector batch using the quantizer's
    /// native batch kernel. The result has one distance per candidate row.
    ///
    /// ```
    /// # use lance_index::vector::pairwise::PairwisePartition;
    /// # async fn example(partition: &PairwisePartition) -> lance_core::Result<()> {
    /// let batch = partition.read_vectors(0).await?;
    /// let distances = partition.distance_batch(&batch, 0, &batch)?;
    /// assert_eq!(distances.len(), batch.row_ids.len());
    /// # Ok(()) }
    /// ```
    pub fn distance_batch(
        &self,
        anchor: &PairwiseVectorBatch,
        row: usize,
        candidates: &PairwiseVectorBatch,
    ) -> Result<Vec<f32>> {
        if row >= anchor.row_ids.len() {
            return Err(Error::invalid_input(format!(
                "pairwise anchor row={row} out of range"
            )));
        }
        if candidates.row_ids.is_empty() {
            return Ok(Vec::new());
        }
        self.scorer
            .distance_batch(&anchor.codes, row, &candidates.codes)
    }
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
            scorer: Arc::new(PairwiseScorer::Flat {
                column: FlatQuantizer::new(2, DistanceType::L2).column(),
                metric: DistanceType::L2,
            }),
            remapper: None,
            batch_size: 32,
            num_rows: 33,
        };
        for _ in 0..3 {
            for (id, expected) in [batch.clone(), batch.slice(0, 1)].iter().enumerate() {
                let codes = prepared.read_vectors(id).await.unwrap();
                assert_eq!(
                    &codes.row_ids,
                    expected[ROW_ID].as_primitive::<UInt64Type>()
                );
                assert_eq!(
                    &codes.codes[FlatQuantizer::new(2, DistanceType::L2).column()],
                    &expected[FlatQuantizer::new(2, DistanceType::L2).column()]
                );
            }
        }
        for id in [2, usize::MAX] {
            let err = prepared.read_vectors(id).await.unwrap_err();
            assert!(matches!(err, Error::InvalidInput { .. }));
            assert!(err.to_string().contains("batch_id"));
        }
    }
}
