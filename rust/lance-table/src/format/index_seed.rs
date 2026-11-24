// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index seed metadata and buffers embedded in Lance data files.

use std::{io::Cursor, sync::Arc};

use arrow_array::{ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow_ipc::{reader::FileReader, writer::FileWriter};
use bytes::Bytes;
use lance_core::Result;
use roaring::RoaringBitmap;

/// Schema metadata key prefix for all seed buffers: `"lance.seed.<column_name>"`.
pub const SEED_META_KEY_PREFIX: &str = "lance.seed.";

/// A hook registered during data file writes that observes column values batch
/// by batch, accumulates compact statistics in memory, and serializes them to
/// a byte buffer that is embedded in the data file footer as a global buffer.
///
/// The buffer can later be read back during index updates to reconstruct index
/// statistics without re-scanning the column data.
pub trait IndexSeedWriter: Send + std::fmt::Debug {
    /// The column this writer is interested in.
    fn column_name(&self) -> &str;

    /// Observe a slice of column values as they are written to the current fragment.
    /// Called once per batch.
    fn observe_batch(&mut self, values: &ArrayRef) -> Result<()>;

    /// Serialize accumulated state to bytes and reset for the next fragment.
    /// Returns `None` if no data was observed (empty fragment).
    fn finish(&mut self) -> Result<Option<Bytes>>;

    /// Schema metadata key used to record that a seed buffer was written.
    /// Convention: `"lance.seed.<column_name>"`.
    fn schema_metadata_key(&self) -> String;

    /// Create a string to store in the file's schema metadata. This will normally
    /// contain the buffer index (provided by the caller after `add_global_buffer`)
    /// as well as any other information needed to validate or understand the seed
    /// (zone map seeds append the `rows_per_zone` they were collected with).
    fn schema_metadata_value(&self, buf_index: u32) -> String;

    /// The dataset feature flags a reader must support to interpret the seeds
    /// this writer produces, as bits of both `reader_feature_flags` and
    /// `writer_feature_flags`.
    ///
    /// The write path embeds a seed only when the dataset snapshot already
    /// carries these bits, so a manifest never references a seed its bits do
    /// not announce. Zero (the default) means the format needs no flag.
    fn required_feature_flags(&self) -> u64 {
        0
    }
}

/// The default number of rows collected into each ZoneMap seed record.
pub const ROWS_PER_ZONE_DEFAULT: u64 = 8192;

/// Seed schema-metadata key carrying the encoded null positions.
pub const ZONEMAP_SEED_NULL_BITMAP_META_KEY: &str = "seed_null_bitmap";
/// Seed schema-metadata key carrying the seed format version. Absent on
/// version 0 seeds (see `docs/src/format/index/scalar/zone_write_seeds.md`).
pub const ZONEMAP_SEED_VERSION_META_KEY: &str = "version";
/// The seed format version this writer emits.
pub const ZONEMAP_SEED_FORMAT_VERSION: &str = "1";

/// A zone write seed parsed and shape-validated for binary-copy carrying.
///
/// Binary-copy compaction moves whole data files' pages verbatim, so an
/// output file's correct seed is the concatenation of its inputs' zone
/// records in copy order. Parsing proves the buffer really is a zone map
/// seed of a supported version whose statistics satisfy that version's
/// layout contract; merging refuses anything it cannot reproduce exactly.
#[derive(Debug)]
pub struct MergeableSeed {
    /// `None` is version 0; `"1"` is version 1. Other versions classify as
    /// [`CarriedSeed::UnsupportedVersion`] before any statistic is read.
    version: Option<String>,
    /// The zone records under a metadata-free schema, so sources concatenate.
    batch: RecordBatch,
    /// Exact null offsets within the covered rows; `None` when the source
    /// predates the bitmap.
    null_bitmap: Option<RoaringBitmap>,
    /// Checked sum of `zone_length`: the rows this seed covers.
    total_rows: u64,
    /// The collection granularity the locator declares (write-time
    /// configuration, not a layout rule).
    rows_per_zone: u64,
}

impl MergeableSeed {
    /// The rows this seed covers; the caller must check it against the file's
    /// physical row count before treating the seed as complete coverage.
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// Whether this seed declares format version 1 (finite bounds plus exact
    /// NaN sign counts, locatable by zone length).
    pub fn is_version_1(&self) -> bool {
        self.version.as_deref() == Some(ZONEMAP_SEED_FORMAT_VERSION)
    }

    /// The collection granularity the source's locator declared.
    pub fn rows_per_zone(&self) -> u64 {
        self.rows_per_zone
    }

    /// Exact null offsets within the covered rows, when the source recorded them.
    pub fn null_bitmap(&self) -> Option<&RoaringBitmap> {
        self.null_bitmap.as_ref()
    }
}

/// What a seed buffer turned out to be when parsed for binary-copy carrying.
#[derive(Debug)]
pub enum CarriedSeed {
    /// A seed of a supported version whose statistics passed validation.
    Supported(MergeableSeed),
    /// A seed declaring a format version this build cannot interpret. It is
    /// a recognized seed (so it does not disqualify the file), but its column
    /// is copied without a seed: unknown statistics are never relabeled.
    UnsupportedVersion(String),
}

fn seed_parse_error(message: impl std::fmt::Display) -> lance_core::Error {
    lance_core::Error::invalid_input(format!("zone map seed: {}", message))
}

/// Parse and validate one seed buffer for binary-copy carrying.
///
/// `rows_per_zone` is the collection granularity the locator declares
/// (`<buffer_index>:<rows_per_zone>`); zones are located by their lengths,
/// never by it. A buffer whose `version` metadata this build does not
/// support classifies as [`CarriedSeed::UnsupportedVersion`] without its
/// statistics being read. An error means the buffer must be neither carried
/// nor silently dropped: an unrecognized shape, or statistics that
/// contradict the layout (zero-length zones, counts beyond the zone, a
/// bitmap outside the covered rows). The caller then refuses binary copy for
/// the file, preserving the existing protection against unknown global
/// buffers.
pub fn parse_seed_for_binary_copy(bytes: &Bytes, rows_per_zone: u64) -> Result<CarriedSeed> {
    if rows_per_zone == 0 {
        return Err(seed_parse_error("locator declares rows_per_zone 0"));
    }
    let mut reader = FileReader::try_new(Cursor::new(bytes.as_ref()), None)
        .map_err(|e| seed_parse_error(format!("not a readable IPC buffer: {}", e)))?;
    let schema = reader.schema();
    let version = schema.metadata.get(ZONEMAP_SEED_VERSION_META_KEY).cloned();
    let is_v1 = match version.as_deref() {
        None => false,
        Some(ZONEMAP_SEED_FORMAT_VERSION) => true,
        Some(other) => return Ok(CarriedSeed::UnsupportedVersion(other.to_string())),
    };
    let null_bitmap_hex = schema
        .metadata
        .get(ZONEMAP_SEED_NULL_BITMAP_META_KEY)
        .cloned();

    let batch = match reader.next() {
        Some(Ok(batch)) => batch,
        Some(Err(e)) => {
            return Err(seed_parse_error(format!("unreadable zone batch: {}", e)));
        }
        None => return Err(seed_parse_error("no zone batch")),
    };
    if reader.next().is_some() {
        return Err(seed_parse_error("more than one zone batch"));
    }

    let expected_columns = if is_v1 { 6 } else { 5 };
    if batch.num_columns() != expected_columns {
        return Err(seed_parse_error(format!(
            "expected {} columns, found {}",
            expected_columns,
            batch.num_columns()
        )));
    }
    let min_col = batch
        .column_by_name("min")
        .ok_or_else(|| seed_parse_error("missing 'min' column"))?;
    let max_col = batch
        .column_by_name("max")
        .ok_or_else(|| seed_parse_error("missing 'max' column"))?;
    if min_col.data_type() != max_col.data_type() {
        return Err(seed_parse_error("'min' and 'max' column types differ"));
    }
    let null_count_col = batch
        .column_by_name("null_count")
        .and_then(|col| col.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| seed_parse_error("missing or mistyped 'null_count' column"))?
        .clone();
    let nan_count_col = batch
        .column_by_name("nan_count")
        .and_then(|col| col.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| seed_parse_error("missing or mistyped 'nan_count' column"))?
        .clone();
    let zone_length_col = batch
        .column_by_name("zone_length")
        .and_then(|col| col.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| seed_parse_error("missing or mistyped 'zone_length' column"))?
        .clone();
    let negative_nan_count_col = if is_v1 {
        Some(
            batch
                .column_by_name("negative_nan_count")
                .and_then(|col| col.as_any().downcast_ref::<UInt32Array>())
                .ok_or_else(|| seed_parse_error("missing or mistyped 'negative_nan_count' column"))?
                .clone(),
        )
    } else {
        None
    };
    // The count columns are non-null by contract; `min`/`max` are nullable
    // (a null bound means unknown or no ordinary values).
    for name in [
        "null_count",
        "nan_count",
        "negative_nan_count",
        "zone_length",
    ] {
        if let Some(col) = batch.column_by_name(name)
            && col.null_count() > 0
        {
            return Err(seed_parse_error(format!(
                "'{}' column must be non-null",
                name
            )));
        }
    }

    let num_zones = batch.num_rows();
    if num_zones == 0 {
        return Err(seed_parse_error("zero zones"));
    }
    let mut total_rows: u64 = 0;
    for i in 0..num_zones {
        let zone_length = zone_length_col.value(i);
        if zone_length == 0 {
            return Err(seed_parse_error(format!("zone {} has zero length", i)));
        }
        let null_count = null_count_col.value(i);
        let nan_count = nan_count_col.value(i);
        if u64::from(null_count) + u64::from(nan_count) > zone_length {
            return Err(seed_parse_error(format!(
                "zone {} counts exceed zone_length: null_count={} nan_count={} zone_length={}",
                i, null_count, nan_count, zone_length
            )));
        }
        if let Some(negative) = negative_nan_count_col.as_ref().map(|col| col.value(i))
            && negative > nan_count
        {
            return Err(seed_parse_error(format!(
                "zone {} negative_nan_count {} exceeds nan_count {}",
                i, negative, nan_count
            )));
        }
        total_rows = total_rows
            .checked_add(zone_length)
            .ok_or_else(|| seed_parse_error("zone_length sum overflows u64"))?;
    }

    let null_bitmap = match null_bitmap_hex {
        None => None,
        Some(hex) => {
            let bitmap_bytes = decode_zonemap_null_bitmap_hex(&hex)
                .map_err(|e| seed_parse_error(format!("bad null bitmap hex: {}", e)))?;
            let bitmap = if bitmap_bytes.is_empty() {
                RoaringBitmap::default()
            } else {
                RoaringBitmap::deserialize_from(bitmap_bytes.as_slice())
                    .map_err(|e| seed_parse_error(format!("bad null bitmap: {}", e)))?
            };
            if let Some(max) = bitmap.max()
                && u64::from(max) >= total_rows
            {
                return Err(seed_parse_error(format!(
                    "null bitmap offset {} outside the {} covered rows",
                    max, total_rows
                )));
            }
            Some(bitmap)
        }
    };

    // A metadata-free schema, so batches from different sources concatenate.
    let canonical_schema = Arc::new(arrow_schema::Schema::new(batch.schema().fields().clone()));
    let batch = RecordBatch::try_new(canonical_schema, batch.columns().to_vec())?;

    Ok(CarriedSeed::Supported(MergeableSeed {
        version,
        batch,
        null_bitmap,
        total_rows,
        rows_per_zone,
    }))
}

/// Merge the seeds of the files an output binary-copies, in copy order, into
/// the output file's seed.
///
/// Every record keeps its original statistics; nothing is recomputed or
/// re-chunked. Zones are located by their lengths, so sources collected at
/// different granularities concatenate freely and short zones may land
/// anywhere in the output (the caller picks the output locator's
/// `rows_per_zone` from the current configuration). `Ok(None)` means the
/// sources cannot be carried: any version 0 source (no implicit format
/// conversion) or mismatched value types.
///
/// The null bitmap is the union of the sources' bitmaps shifted by each
/// source's cumulative row offset; any source without a bitmap leaves the
/// output without one (absence is "unknown", never zero).
pub fn merge_seeds_for_binary_copy(sources: &[&MergeableSeed]) -> Result<Option<Bytes>> {
    let Some(first) = sources.first() else {
        return Ok(None);
    };
    if sources
        .iter()
        .any(|source| !source.is_version_1() || source.batch.schema() != first.batch.schema())
    {
        return Ok(None);
    }

    let mut null_bitmap = sources
        .iter()
        .all(|source| source.null_bitmap.is_some())
        .then(RoaringBitmap::new);
    let mut offset: u64 = 0;
    for source in sources {
        if let (Some(merged), Some(bitmap)) = (&mut null_bitmap, &source.null_bitmap) {
            for value in bitmap {
                let shifted = offset
                    .checked_add(u64::from(value))
                    .and_then(|shifted| u32::try_from(shifted).ok());
                let Some(shifted) = shifted else {
                    // The offsets no longer fit the bitmap's 32-bit space.
                    return Ok(None);
                };
                merged.insert(shifted);
            }
        }
        offset = offset
            .checked_add(source.total_rows)
            .ok_or_else(|| seed_parse_error("merged row count overflows u64"))?;
    }

    let batches: Vec<&RecordBatch> = sources.iter().map(|source| &source.batch).collect();
    let merged = arrow::compute::concat_batches(&first.batch.schema(), batches)
        .map_err(|e| seed_parse_error(format!("failed to concatenate zone records: {}", e)))?;

    let mut schema = merged.schema().as_ref().clone();
    schema.metadata.insert(
        ZONEMAP_SEED_VERSION_META_KEY.to_string(),
        ZONEMAP_SEED_FORMAT_VERSION.to_string(),
    );
    if let Some(bitmap) = null_bitmap {
        let bitmap_hex = encode_zonemap_null_bitmap(bitmap)
            .map_err(|e| seed_parse_error(format!("failed to serialize null bitmap: {}", e)))?;
        schema
            .metadata
            .insert(ZONEMAP_SEED_NULL_BITMAP_META_KEY.to_string(), bitmap_hex);
    }
    let batch = RecordBatch::try_new(Arc::new(schema), merged.columns().to_vec())?;

    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = FileWriter::try_new(&mut buf, batch.schema_ref())
            .map_err(|e| seed_parse_error(format!("failed to create IPC writer: {}", e)))?;
        writer
            .write(&batch)
            .map_err(|e| seed_parse_error(format!("failed to write IPC batch: {}", e)))?;
        writer
            .finish()
            .map_err(|e| seed_parse_error(format!("failed to finish IPC writer: {}", e)))?;
    }
    Ok(Some(Bytes::from(buf.into_inner())))
}

/// Serialize exact null positions into the ZoneMap seed metadata value.
/// An empty bitmap is encoded as an empty string.
pub fn encode_zonemap_null_bitmap(mut bitmap: RoaringBitmap) -> std::io::Result<String> {
    if bitmap.is_empty() {
        return Ok(String::new());
    }
    // Run containers make contiguous null regions a few bytes each.
    bitmap.optimize();
    let mut bytes = Vec::with_capacity(bitmap.serialized_size());
    bitmap.serialize_into(&mut bytes)?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Decode the hexadecimal bytes of a ZoneMap seed's null bitmap.
/// An empty value represents an empty bitmap.
pub fn decode_zonemap_null_bitmap_hex(s: &str) -> std::result::Result<Vec<u8>, String> {
    if s.is_empty() {
        return Ok(Vec::default());
    }
    if !s.len().is_multiple_of(2) {
        return Err(format!("odd hex length: {}", s.len()));
    }
    if !s.is_ascii() {
        return Err("null bitmap hex must be ASCII".to_string());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| format!("invalid hex at {}: {}", i, e))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::record_batch;
    use lance_core::Error;

    #[test]
    fn rejects_non_ascii_null_bitmap_metadata() {
        let batch = record_batch!(
            ("min", Int32, [1]),
            ("max", Int32, [1]),
            ("null_count", UInt32, [0]),
            ("nan_count", UInt32, [0]),
            ("zone_length", UInt64, [1]),
            ("negative_nan_count", UInt32, [0])
        )
        .unwrap();
        let mut schema = batch.schema().as_ref().clone();
        schema
            .metadata
            .insert("version".to_string(), "1".to_string());
        // The UTF-8 code point straddles a two-byte hex boundary.
        schema
            .metadata
            .insert("seed_null_bitmap".to_string(), "aé0".to_string());
        let batch = RecordBatch::try_new(Arc::new(schema), batch.columns().to_vec()).unwrap();
        let mut bytes = Vec::new();
        let mut writer = FileWriter::try_new(&mut bytes, batch.schema_ref()).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        drop(writer);

        let error = parse_seed_for_binary_copy(&Bytes::from(bytes), 1).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("bad null bitmap hex"));
    }
}
