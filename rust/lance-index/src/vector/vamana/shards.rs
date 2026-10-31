// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The storage of the shards of an IVF_VAMANA_* index built apart, as a merge
//! of them unites it.
//!
//! Each shard of scalar codes trains its bounds over its own rows, and a
//! merged storage keeps one set of bounds. The merge re-encodes every shard's
//! codes to the union of the bounds of the shards whose rows it keeps, so
//! that the codes of one shard are not decoded with the bounds of another.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, UInt8Type, UInt64Type};
use arrow_array::{Array, FixedSizeListArray, RecordBatch, UInt8Array};
use arrow_schema::DataType;
use futures::StreamExt;
use lance_arrow::RecordBatchExt;
use lance_core::{Error, ROW_ID, Result};
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::FileReader;
use lance_file::versions::reader_projection_from_column_names;
use lance_file::writer::FileWriter;
use lance_io::ReadBatchParams;

use crate::scalar::OldIndexDataFilter;
use crate::vector::SQ_CODE_COLUMN;
use crate::vector::distributed::index_merger::filter_batch_to_owned_rows;
use crate::vector::sq::scale_to_u8;
use crate::vector::sq::storage::ScalarQuantizationMetadata;

/// The row ids [`keeps_rows`] reads at a time.
const ROW_ID_BATCH_SIZE: u32 = 8192;

/// How the scalar codes of one shard are re-encoded to the bounds every shard
/// of a merge shares: for each code, the code the same value takes under the
/// shared bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarRecode {
    codes: Vec<u8>,
}

impl ScalarRecode {
    /// The metadata a merge of `shards` writes, whose bounds are the union of
    /// the bounds of the shards it keeps rows of, and how each shard's codes
    /// are re-encoded to it: `None` for a shard already on those bounds or
    /// with no row kept. `None` in all when the merge keeps no row. Each
    /// shard comes, in the order of the merge, with whether the merge keeps
    /// any of its rows (see [`keeps_rows`]).
    pub fn plan(
        shards: &[(ScalarQuantizationMetadata, bool)],
    ) -> Result<Option<(ScalarQuantizationMetadata, Vec<Option<Self>>)>> {
        let mut shared: Option<ScalarQuantizationMetadata> = None;
        for (position, (metadata, _)) in shards.iter().enumerate().filter(|(_, (_, kept))| *kept) {
            if metadata.num_bits != 8 {
                return Err(Error::not_supported(format!(
                    "IVF_VAMANA_SQ merges 8-bit codes only, but the shard at position \
                     {position} of the merge holds {}-bit codes",
                    metadata.num_bits
                )));
            }
            // A shard trained on no vector keeps the empty initial range, and
            // its codes decode to nothing.
            if metadata.bounds.start > metadata.bounds.end {
                return Err(Error::index(format!(
                    "The shard at position {position} of an IVF_VAMANA_SQ merge holds rows under \
                     the empty bounds {:?}",
                    metadata.bounds
                )));
            }
            if let Some(shared) = shared.as_mut() {
                if metadata.dim != shared.dim {
                    return Err(Error::index(format!(
                        "The shard at position {position} of an IVF_VAMANA_SQ merge holds codes of \
                         {} values, the shards before it codes of {}",
                        metadata.dim, shared.dim
                    )));
                }
                shared.bounds.start = shared.bounds.start.min(metadata.bounds.start);
                shared.bounds.end = shared.bounds.end.max(metadata.bounds.end);
            } else {
                shared = Some(metadata.clone());
            }
        }
        let Some(shared) = shared else {
            return Ok(None);
        };
        let recodes = shards
            .iter()
            .map(|(metadata, kept)| {
                (*kept && metadata.bounds != shared.bounds)
                    .then(|| Self::between(&metadata.bounds, &shared.bounds))
            })
            .collect();
        Ok(Some((shared, recodes)))
    }

    /// Re-encoding from `from` to `to`. A code stands for the values that
    /// `scale_to_u8` sends to it: the middle of its step, or, for the last
    /// code, the upper bound, which is the only value inside the bounds it
    /// takes.
    fn between(from: &Range<f64>, to: &Range<f64>) -> Self {
        // Through `scale_to_u8` an upper bound can come back a code short of
        // its own: `x * 255 / x` rounds below 255 for some `x`.
        if from == to {
            return Self {
                codes: (0..=255).collect(),
            };
        }
        let step = (from.end - from.start) / 255.0;
        let values = (0..=255u8)
            .map(|code| match code {
                255 => from.end,
                code => from.start + (f64::from(code) + 0.5) * step,
            })
            .collect::<Vec<_>>();
        Self {
            codes: scale_to_u8::<Float64Type>(&values, to),
        }
    }

    /// `batch` with the codes of its [`SQ_CODE_COLUMN`] re-encoded.
    pub fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let column = batch.column_by_name(SQ_CODE_COLUMN).ok_or_else(|| {
            Error::index(format!("{SQ_CODE_COLUMN} missing from a scalar-code shard"))
        })?;
        let DataType::FixedSizeList(field, size) = column.data_type() else {
            return Err(Error::index(format!(
                "{SQ_CODE_COLUMN} of a scalar-code shard is not a fixed-size list: {}",
                column.data_type()
            )));
        };
        let codes = column.as_fixed_size_list();
        let values = codes
            .values()
            .as_primitive_opt::<UInt8Type>()
            .ok_or_else(|| {
                Error::index(format!(
                    "{SQ_CODE_COLUMN} of a scalar-code shard does not hold bytes: {}",
                    codes.value_type()
                ))
            })?;
        let recoded = UInt8Array::from_iter_values(
            values
                .values()
                .iter()
                .map(|code| self.codes[usize::from(*code)]),
        );
        let recoded = FixedSizeListArray::try_new(
            field.clone(),
            *size,
            Arc::new(recoded),
            codes.nulls().cloned(),
        )?;
        Ok(batch.replace_column_by_name(SQ_CODE_COLUMN, Arc::new(recoded))?)
    }
}

/// Whether a merge keeps any of the `rows` rows of a shard's storage: any at
/// all without a `row_filter`, else any whose row id the filter keeps, read
/// from the row ids alone until one is kept.
pub async fn keeps_rows(
    reader: &FileReader,
    rows: usize,
    row_filter: Option<&OldIndexDataFilter>,
) -> Result<bool> {
    let Some(row_filter) = row_filter.filter(|_| rows > 0) else {
        return Ok(rows > 0);
    };
    let projection =
        reader_projection_from_column_names(reader.version(), reader.schema(), &[ROW_ID])?;
    let mut stream = reader
        .read_stream_projected(
            ReadBatchParams::RangeFull,
            ROW_ID_BATCH_SIZE,
            1,
            projection,
            FilterExpression::no_filter(),
        )
        .await?;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let row_ids = batch
            .column(0)
            .as_primitive_opt::<UInt64Type>()
            .ok_or_else(|| {
                Error::index(format!(
                    "{ROW_ID} of a scalar-code shard is not UInt64: {}",
                    batch.column(0).data_type()
                ))
            })?;
        if row_filter.filter_row_ids(row_ids).has_true() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Write the rows of `range` in a shard's storage that `row_filter` keeps,
/// their codes re-encoded by `recode`, and return how many were written.
pub async fn write_recoded_partition_rows(
    reader: &FileReader,
    writer: &mut FileWriter,
    range: Range<usize>,
    row_filter: Option<&OldIndexDataFilter>,
    recode: &ScalarRecode,
) -> Result<usize> {
    let mut stream = reader
        .read_stream(
            ReadBatchParams::Range(range),
            u32::MAX,
            4,
            FilterExpression::no_filter(),
        )
        .await?;
    let mut written = 0usize;
    while let Some(batch) = stream.next().await {
        let mut batch = batch?;
        if let Some(row_filter) = row_filter {
            batch = filter_batch_to_owned_rows(&batch, row_filter)?;
        }
        if batch.num_rows() == 0 {
            continue;
        }
        written = written.checked_add(batch.num_rows()).ok_or_else(|| {
            Error::index("Re-encoded partition row count exceeds usize capacity".to_string())
        })?;
        writer.write_batch(&recode.apply(batch)?).await?;
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use arrow_array::types::Float32Type;
    use arrow_schema::{Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;

    use super::*;

    fn metadata(bounds: Range<f64>) -> ScalarQuantizationMetadata {
        ScalarQuantizationMetadata {
            dim: 2,
            num_bits: 8,
            bounds,
        }
    }

    /// Values spread over `bounds`, both ends included.
    fn values_over(bounds: &Range<f64>) -> Vec<f64> {
        (0..=1000)
            .map(|i| bounds.start + (bounds.end - bounds.start) * f64::from(i) / 1000.0)
            .collect()
    }

    /// A shard's code re-encoded is the code of its value under the shared
    /// bounds, give or take one: no more than one step of the shared bounds
    /// beyond the error the shard's own codes already had.
    #[test]
    fn a_recoded_code_is_within_one_of_the_values_own_code() {
        let shared = -1.0..4.0;
        for own in [0.0..1.0, -1.0..0.5, 2.0..4.0, 0.25..0.25] {
            let recode = ScalarRecode::between(&own, &shared);
            let values = values_over(&own);
            let codes = scale_to_u8::<Float64Type>(&values, &own);
            let direct = scale_to_u8::<Float64Type>(&values, &shared);
            for ((value, code), direct) in values.iter().zip(&codes).zip(&direct) {
                let recoded = recode.codes[usize::from(*code)];
                assert!(
                    recoded.abs_diff(*direct) <= 1,
                    "{value} in {own:?}: code {code} recoded {recoded}, directly {direct}"
                );
            }
        }
    }

    /// A shard's code re-encoded is the code most values of its step take
    /// under the shared bounds, the upper bound's own for the last code.
    #[test]
    fn a_recoded_code_is_the_code_most_values_of_its_step_take() {
        for (own, shared) in [
            (0.0..1.0, 0.0..1.5),
            (0.0..1.0, -1.0..4.0),
            (-1.0..0.5, -1.0..4.0),
            (0.0..1.0, 0.0..1.001),
        ] {
            let recode = ScalarRecode::between(&own, &shared);
            let steps = 255 * 64;
            let values = (0..=steps)
                .map(|i| own.start + (own.end - own.start) * f64::from(i) / f64::from(steps))
                .collect::<Vec<_>>();
            let codes = scale_to_u8::<Float64Type>(&values, &own);
            let direct = scale_to_u8::<Float64Type>(&values, &shared);
            let mut agree = [0usize; 256];
            let mut total = [0usize; 256];
            for (code, direct) in codes.iter().zip(&direct) {
                let code = usize::from(*code);
                total[code] += 1;
                agree[code] += usize::from(recode.codes[code] == *direct);
            }
            for code in 0..256 {
                assert!(
                    2 * agree[code] >= total[code],
                    "{own:?} to {shared:?}: code {code} recoded {} agrees with {} of {} values",
                    recode.codes[code],
                    agree[code],
                    total[code]
                );
            }
        }
    }

    /// Re-encoding to the bounds a shard already has changes no code, even
    /// where the upper bound itself would come back a code short.
    #[test]
    fn recoding_to_the_same_bounds_keeps_every_code() {
        assert_eq!(scale_to_u8::<Float64Type>(&[1.1], &(0.0..1.1)), [254]);
        for bounds in [0.0..1.0, -3.0..7.5, 0.0..1.1, 0.5..0.5] {
            let recode = ScalarRecode::between(&bounds, &bounds);
            assert_eq!(recode.codes, (0..=255u8).collect::<Vec<_>>(), "{bounds:?}");
        }
    }

    /// The shared bounds cover the shards whose rows the merge keeps and
    /// nothing else; a shard already on them, or with no row kept, is not
    /// re-encoded.
    #[test]
    fn a_plan_shares_the_bounds_of_the_shards_that_hold_rows() {
        let empty = metadata(f64::MAX..f64::MIN);
        let shards = [
            (empty.clone(), false),
            (metadata(0.0..1.0), true),
            (metadata(-2.0..3.0), true),
            (metadata(0.5..9.0), false),
        ];
        let (shared, recodes) = ScalarRecode::plan(&shards).unwrap().unwrap();
        assert_eq!(shared, metadata(-2.0..3.0));
        assert_eq!(
            recodes.iter().map(Option::is_some).collect::<Vec<_>>(),
            [false, true, false, false]
        );
        assert_eq!(
            recodes[1],
            Some(ScalarRecode::between(&(0.0..1.0), &(-2.0..3.0)))
        );
        assert_eq!(ScalarRecode::plan(&[(empty, false)]).unwrap(), None);
    }

    /// A plan refuses shards whose codes it cannot re-encode, naming the
    /// shard by its position in the merge.
    #[test]
    fn a_plan_refuses_codes_it_cannot_recode() {
        let mut four_bits = metadata(0.0..1.0);
        four_bits.num_bits = 4;
        let mut wider = metadata(0.0..1.0);
        wider.dim = 3;
        for (shards, unsupported, refusal) in [
            (
                vec![(metadata(0.0..1.0), true), (four_bits.clone(), true)],
                true,
                "8-bit codes only, but the shard at position 1 of the merge holds 4-bit codes",
            ),
            (
                vec![(four_bits, true)],
                true,
                "the shard at position 0 of the merge holds 4-bit codes",
            ),
            (
                vec![
                    (metadata(0.0..1.0), true),
                    (metadata(f64::MAX..f64::MIN), true),
                ],
                false,
                "The shard at position 1 of an IVF_VAMANA_SQ merge holds rows under the empty \
                 bounds",
            ),
            (
                vec![(metadata(0.0..1.0), true), (wider, true)],
                false,
                "The shard at position 1 of an IVF_VAMANA_SQ merge holds codes of 3 values, the \
                 shards before it codes of 2",
            ),
        ] {
            let error = ScalarRecode::plan(&shards).unwrap_err();
            if unsupported {
                assert!(matches!(error, Error::NotSupported { .. }), "{error:?}");
            } else {
                assert!(matches!(error, Error::Index { .. }), "{error:?}");
            }
            assert!(error.to_string().contains(refusal), "{error}");
        }
    }

    /// Applying a re-encoding rewrites the code column only.
    #[test]
    fn applying_a_recode_rewrites_only_the_codes() {
        let recode = ScalarRecode::between(&(0.0..1.0), &(0.0..2.0));
        let codes =
            FixedSizeListArray::try_new_from_values(UInt8Array::from(vec![0, 128, 255, 7]), 2)
                .unwrap();
        let other = arrow_array::Float32Array::from(vec![1.5, 2.5]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("other", DataType::Float32, false),
            Field::new(SQ_CODE_COLUMN, codes.data_type().clone(), true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(other), Arc::new(codes)]).unwrap();
        let recoded = recode.apply(batch).unwrap();
        assert_eq!(
            recoded["other"].as_primitive::<Float32Type>().values(),
            &[1.5, 2.5]
        );
        let codes = recoded[SQ_CODE_COLUMN].as_fixed_size_list();
        assert_eq!(
            codes.values().as_primitive::<UInt8Type>().values(),
            &[
                recode.codes[0],
                recode.codes[128],
                recode.codes[255],
                recode.codes[7]
            ]
        );
    }
}
