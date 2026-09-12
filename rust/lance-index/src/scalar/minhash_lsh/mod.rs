// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! MinHash LSH index for near-duplicate detection over text columns.
//!
//! The index answers "which rows have the highest estimated Jaccard similarity
//! (over token shingles) to this query text?" and is the building block for
//! fuzzy deduplication of training corpora, web crawls, and RAG ingestion.
//!
//! Every indexed row gets a signature of `num_hashes` MinHash values, computed
//! from the row's token shingles. The signature is split into `num_bands`
//! bands; rows that agree on at least one band become candidates and are then
//! refined by comparing their full signatures against the query signature.
//!
//! A segment stores two files:
//!
//! ```text
//! segment (_indices/{uuid}/)
//! ├── signatures.lance   one row per indexed document, row number = doc id
//! │     ├── _rowid     UInt64
//! │     └── signature  FixedSizeList<UInt16, num_hashes>   (full-zip, no compression)
//! └── bands.lance        one row per (band key, doc id), ascending
//!       ├── band_key   UInt64   band id (high 8 bits) | hash of band values (low 56 bits)
//!       └── doc_id     UInt32   row number in signatures.lance
//!       (schema metadata: details, doc count, page rows, page table buffer)
//!
//! Both files repeat the index details and the file format version in their
//! schema metadata; a segment is opened only when both files match the
//! details of the index metadata.
//! ```
//!
//! A bucket is the run of rows sharing a band key; it may span pages. Lookups
//! binary-search a resident page table (max band key per page), fetch the
//! bucket's pages through the cache, binary-search each page for the band
//! key, collect the bucket's doc ids into a candidate bitmap, and read the
//! candidate signatures with a scattered read (or a sequential scan when the
//! candidate set is dense).

mod builder;
mod index;
#[cfg(test)]
mod tests;

pub use builder::MinHashLshIndexBuilder;
pub use index::{MinHashLshIndex, merge_minhash_indices};

use builder::{RowIdTransform, SignatureSource};

use std::any::Any;
use std::collections::{BinaryHeap, HashMap};
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt16Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::execution::SendableRecordBatchStream;
use futures::stream::FuturesOrdered;
use futures::{Stream, StreamExt, TryStreamExt};
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache, WeakLanceCache};
use lance_core::datatypes::SchemaCompareOptions;
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::utils::tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS};
use lance_core::{Error, ROW_ID, Result};
use lance_encoding::constants::{
    COMPRESSION_META_KEY, STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY,
};
use lance_select::RowAddrMask;
use lance_tokenizer::TokenStream;
use prost::Message;
use rayon::slice::ParallelSliceMut;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use twox_hash::XxHash64;

use crate::scalar::expression::ScalarQueryParser;
use crate::scalar::inverted::tokenizer::InvertedIndexParams;
use crate::scalar::inverted::tokenizer::document_tokenizer::LanceTokenizer;
use crate::scalar::registry::{
    BasicTrainer, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
    VALUE_COLUMN_NAME,
};
use crate::scalar::{
    AnyQuery, BuiltinIndexType, CreatedIndex, IndexFile, IndexReader, IndexStore, IndexWriter,
    MetricsCollector, OldIndexDataFilter, RowIdRemapper, ScalarIndex, ScalarIndexParams,
    SearchResult, UpdateCriteria,
};
use crate::{Index, IndexType};
use crate::{pb, pbold};

/// On-disk format version of the index files. The format is unstable; bump
/// on any layout change instead of adding compatibility paths.
pub const MINHASH_LSH_INDEX_VERSION: u32 = 0;
/// Version of the signature generation behavior (the token streams of the
/// tokenizers that ship with Lance, shingle hashing, permutation family, the
/// 16-bit compression and the band keys); bumped whenever a
/// change would make old and new signatures incomparable.
pub const SIGNATURE_VERSION: u32 = 0;

/// Seed of the shingle hash and of the generator that derives the permutation
/// and compression coefficients. Part of the signature scheme: changing it
/// changes every signature and therefore [`SIGNATURE_VERSION`].
const SIGNATURE_SEED: u64 = 42;

/// A stored MinHash value: the 64-bit permuted minimum compressed to 16 bits
/// by multiply-shift hashing (b-bit MinHash). A minimum is not uniformly
/// distributed (it shrinks as the set grows), so its bits cannot be stored
/// directly; multiply-shift with a random odd multiplier is 2-universal, so
/// two unrelated documents agree on a stored value with probability at most
/// 2^-15 whatever their length, far below the 1/num_hashes resolution of the
/// Jaccard estimate.
pub type SignatureValue = u16;

pub const SIGNATURES_FILENAME: &str = "signatures.lance";
pub const BANDS_FILENAME: &str = "bands.lance";
pub const SIGNATURE_COL: &str = "signature";
pub const BAND_KEY_COL: &str = "band_key";
pub const DOC_ID_COL: &str = "doc_id";

const DETAILS_META_KEY: &str = "minhash_lsh_details";
const INDEX_VERSION_META_KEY: &str = "minhash_lsh_index_version";
const PAGE_ROWS_META_KEY: &str = "minhash_lsh_page_rows";
const PAGE_TABLE_BUFFER_META_KEY: &str = "minhash_lsh_page_table_buffer";
const NUM_DOCS_META_KEY: &str = "minhash_lsh_num_docs";

/// Number of band-key rows per logical page of `bands.lance`. The page table
/// keeps one u64 per page in memory, so 4096 rows per page keeps the table at
/// ~0.01% of the bands file while one page read stays small (tens of KiB for
/// typical postings).
const DEFAULT_PAGE_ROWS: usize = 4096;
/// Byte inserted between the tokens of one shingle so that different token
/// splits of the same characters hash differently.
const SHINGLE_SEPARATOR: u8 = 0x1F;
/// Bytes per read or write batch of an index file. Scattered reads,
/// sequential scans and the batches handed to the file writers all size
/// themselves from it by their row width: large enough to amortize a
/// request, small enough to bound the transient memory of a query or of a
/// build stage.
const IO_BATCH_BYTES: usize = 8 * 1024 * 1024;
/// Bytes per resident chunk of the signature table and per prewarm read of
/// band pages. Chunks are separate cache entries, so a prewarm keeps as many
/// as the cache holds and a query scores candidates in resident chunks from
/// memory.
const RESIDENT_CHUNK_BYTES: usize = 128 * 1024 * 1024;
/// Reads in flight while a scan or a merge group gathers its rows.
const READ_CONCURRENCY: usize = 4;
/// A candidate set covering more than this percentage of a segment is refined
/// with a sequential scan of the signature file instead of a scattered read.
const SPARSE_REFINE_READ_PERCENT: u64 = 10;

/// Default memory budget of the sort: the run being filled, the runs being
/// spilled and the merge groups together.
const DEFAULT_SORT_MEMORY_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Memory budget of the sort (bytes), the variable the DataFusion-backed
/// index builds honor as well.
const SORT_MEMORY_ENV: &str = "LANCE_MEM_POOL_SIZE";
/// Limit on the temporary disk space of one build (bytes), the variable the
/// DataFusion-backed builds honor as well.
const SPILL_LIMIT_ENV: &str = "LANCE_MAX_TEMP_DIRECTORY_SIZE";
const DEFAULT_SPILL_LIMIT_BYTES: u64 = 100 * 1024 * 1024 * 1024;
/// Bytes of one (band key, doc id) row, in the bands file and in spill files.
const BAND_ROW_BYTES: usize = std::mem::size_of::<u64>() + std::mem::size_of::<u32>();
/// Runs sorted and written concurrently with signing. Each holds a full run
/// in memory, so this bounds the backlog when signing outpaces the disk.
const MAX_INFLIGHT_SPILLS: usize = 4;
/// Signed batches queued between the driver that polls the signing stream
/// and the loop that consumes them.
const SIGNED_BATCH_QUEUE: usize = 16;
/// Every sorted run records the row at which each key-range partition
/// starts, so the merge can gather one partition from every run, sort it in
/// memory and write it, partitions processed independently and in key order.
/// A band gets `SPILL_PARTITIONS / num_bands` partitions split on the hash
/// bits below the band id, so a partition holds `num_bands / SPILL_PARTITIONS`
/// of all records: 1.5 MB per 10^9 rows at eight bands, 47 MB at 256.
const SPILL_PARTITIONS: usize = 1 << 16;
/// Upper bound on merge groups gathered and sorted concurrently; the actual
/// count is a quarter of the CPU pool.
const MERGE_GROUPS_IN_FLIGHT: usize = 16;
/// Highest band count representable in the 8-bit band id prefix of a band key.
const MAX_NUM_BANDS: u32 = 256;
/// Highest signature width. Beyond it the estimate gains nothing (the error
/// is already ~1.5%) while every derived size keeps growing; the bound keeps
/// them all small enough that no parameter combination can overflow or
/// allocate unboundedly before validation runs.
const MAX_NUM_HASHES: u32 = 4096;

/// Rows of `row_bytes` each that fit one IO batch.
fn rows_per_batch(row_bytes: usize) -> usize {
    (IO_BATCH_BYTES / row_bytes.max(1)).max(1)
}

/// Bytes of one row of the signature table: the row id and `num_hashes`
/// stored values.
fn signature_row_bytes(num_hashes: usize) -> usize {
    std::mem::size_of::<u64>() + num_hashes * std::mem::size_of::<SignatureValue>()
}

/// `bands.lance`: (band key, doc id) records in ascending order. Both columns
/// are fixed width and stored raw, so a page is one exact ranged read that
/// decodes without copying.
static BANDS_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new(BAND_KEY_COL, DataType::UInt64, false).with_metadata(HashMap::from([(
            COMPRESSION_META_KEY.to_string(),
            "none".to_string(),
        )])),
        Field::new(DOC_ID_COL, DataType::UInt32, false).with_metadata(HashMap::from([(
            COMPRESSION_META_KEY.to_string(),
            "none".to_string(),
        )])),
    ]))
});

fn signatures_schema(num_hashes: i32) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(ROW_ID, DataType::UInt64, false),
        Field::new(
            SIGNATURE_COL,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt16, false)),
                num_hashes,
            ),
            false,
        )
        .with_metadata(HashMap::from([
            (
                STRUCTURAL_ENCODING_META_KEY.to_string(),
                STRUCTURAL_ENCODING_FULLZIP.to_string(),
            ),
            (COMPRESSION_META_KEY.to_string(), "none".to_string()),
        ])),
    ]))
}

/// A MinHash similarity search: the rows of `column` whose token shingles have
/// the highest estimated Jaccard similarity to `text`.
///
/// The number of rows comes from the scan limit and the output carries a
/// `_distance` column equal to `1 - estimated Jaccard similarity`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinHashQuery {
    /// Column with a MinHash LSH index.
    pub column: String,
    /// Query text, tokenized and shingled exactly like the indexed rows.
    pub text: String,
}

impl MinHashQuery {
    pub fn new(text: impl Into<String>, column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            text: text.into(),
        }
    }
}

/// Parameters of a MinHash LSH index.
///
/// These are the only inputs to signature generation, so every segment of a
/// logical index must be built from identical parameters. They are persisted
/// in the index details and re-read on the query side.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MinHashLshIndexParams {
    /// Number of MinHash values per signature (k), in `[1, 4096]` and a
    /// multiple of `num_bands`. The Jaccard estimate has error ~1/sqrt(k).
    pub num_hashes: u32,
    /// Number of LSH bands (b), in `[1, 256]`. Two rows become candidates when
    /// all `num_hashes / num_bands` values of at least one band agree.
    pub num_bands: u32,
    /// Number of consecutive tokens joined into one shingle. Documents shorter
    /// than this produce a single shingle of all their tokens.
    pub shingle_size: u32,
    /// Tokenizer configuration, using the same keys as the inverted (full
    /// text search) index. Defaults to lower-casing without stemming or stop
    /// word removal: aggressive normalization inflates Jaccard similarity and
    /// turns "related" into "duplicate".
    pub tokenizer: InvertedIndexParams,
}

impl Default for MinHashLshIndexParams {
    fn default() -> Self {
        Self {
            num_hashes: 128,
            num_bands: 16,
            shingle_size: 3,
            tokenizer: InvertedIndexParams::default()
                .stem(false)
                .remove_stop_words(false),
        }
    }
}

/// User-facing JSON shape: every key optional, tokenizer keys layered over the
/// MinHash tokenizer defaults rather than the full text search defaults.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMinHashLshIndexParams {
    num_hashes: Option<u32>,
    num_bands: Option<u32>,
    shingle_size: Option<u32>,
    tokenizer: Option<serde_json::Map<String, serde_json::Value>>,
}

impl MinHashLshIndexParams {
    /// Parse user parameters from JSON, filling omitted keys with defaults and
    /// validating the result.
    pub fn from_json(params: &str) -> Result<Self> {
        let params = if params.trim().is_empty() {
            "{}"
        } else {
            params
        };
        let raw: RawMinHashLshIndexParams = serde_json::from_str(params).map_err(|err| {
            Error::invalid_input(format!(
                "invalid MinHash LSH index params {params:?}: {err}"
            ))
        })?;
        let mut resolved = Self::default();
        if let Some(num_hashes) = raw.num_hashes {
            resolved.num_hashes = num_hashes;
        }
        if let Some(num_bands) = raw.num_bands {
            resolved.num_bands = num_bands;
        }
        if let Some(shingle_size) = raw.shingle_size {
            resolved.shingle_size = shingle_size;
        }
        if let Some(tokenizer) = raw.tokenizer {
            // Presets (`analyzer`) and full text search defaults resolve inside
            // InvertedIndexParams; the dedup defaults are layered back for the
            // keys the user did not set.
            let has_stem = tokenizer.contains_key("stem");
            let has_stop_words = tokenizer.contains_key("remove_stop_words");
            let mut parsed: InvertedIndexParams =
                serde_json::from_value(serde_json::Value::Object(tokenizer)).map_err(|err| {
                    Error::invalid_input(format!("invalid MinHash LSH tokenizer params: {err}"))
                })?;
            if !has_stem {
                parsed.stem = false;
            }
            if !has_stop_words {
                parsed.remove_stop_words = false;
            }
            resolved.tokenizer = parsed;
        }
        resolved.validate()?;
        // Keep only what the index details can record, so parameters compare
        // equal wherever they were read from.
        resolved.tokenizer = InvertedIndexParams::try_from(
            &pbold::InvertedIndexDetails::try_from(&resolved.tokenizer)?,
        )?;
        Ok(resolved)
    }

    fn validate(&self) -> Result<()> {
        if self.num_hashes == 0 || self.num_hashes > MAX_NUM_HASHES {
            return Err(Error::invalid_input(format!(
                "MinHash LSH index requires 1 <= num_hashes <= {MAX_NUM_HASHES}, got num_hashes={}",
                self.num_hashes
            )));
        }
        if self.num_bands == 0 || self.num_bands > MAX_NUM_BANDS {
            return Err(Error::invalid_input(format!(
                "MinHash LSH index requires 1 <= num_bands <= {MAX_NUM_BANDS}, got num_bands={}",
                self.num_bands
            )));
        }
        if !self.num_hashes.is_multiple_of(self.num_bands) {
            return Err(Error::invalid_input(format!(
                "MinHash LSH index requires num_hashes to be a multiple of num_bands, got num_hashes={} num_bands={}",
                self.num_hashes, self.num_bands
            )));
        }
        if self.shingle_size == 0 {
            return Err(Error::invalid_input(
                "MinHash LSH index requires shingle_size > 0".to_string(),
            ));
        }
        self.validate_tokenizer()?;
        // Surface tokenizer configuration errors at index creation instead of
        // on the first batch.
        self.tokenizer.build()?;
        Ok(())
    }

    /// The index details must identify the tokenizer completely: a query is
    /// tokenized from the details alone, and a segment built elsewhere must
    /// tokenize the same way. Settings the details cannot record, and
    /// tokenizers whose data is not part of Lance, are therefore rejected
    /// rather than silently tokenizing queries differently from the index.
    fn validate_tokenizer(&self) -> Result<()> {
        let unsupported = |reason: &str| {
            Error::invalid_input(format!(
                "MinHash LSH index cannot identify its tokenizer from the index details: {reason}"
            ))
        };
        if self.tokenizer.custom_stop_words.is_some() {
            return Err(unsupported("custom_stop_words are not recorded"));
        }
        if self.tokenizer.lance_tokenizer.is_some() {
            return Err(unsupported("lance_tokenizer is not recorded"));
        }
        let base_tokenizer = self.tokenizer.base_tokenizer.as_str();
        if base_tokenizer == "jieba"
            || base_tokenizer.starts_with("jieba/")
            || base_tokenizer.starts_with("lindera/")
        {
            return Err(unsupported(&format!(
                "the {base_tokenizer} tokenizer loads a dictionary from LANCE_LANGUAGE_MODEL_HOME, outside Lance"
            )));
        }
        Ok(())
    }

    // The details are the persisted identity of the signature procedure:
    // stored as an `Any` in the index metadata and repeated, hex encoded, in
    // the schema metadata of both index files.

    fn to_details(&self) -> Result<pb::MinHashLshIndexDetails> {
        Ok(pb::MinHashLshIndexDetails {
            num_hashes: self.num_hashes,
            num_bands: self.num_bands,
            shingle_size: self.shingle_size,
            tokenizer: Some(pbold::InvertedIndexDetails::try_from(&self.tokenizer)?),
            signature_version: SIGNATURE_VERSION,
        })
    }

    fn from_details(details: &pb::MinHashLshIndexDetails) -> Result<Self> {
        if details.signature_version != SIGNATURE_VERSION {
            return Err(Error::not_supported(format!(
                "MinHash LSH index was built with signature_version {} but this version of Lance produces signature_version {}; rebuild the index",
                details.signature_version, SIGNATURE_VERSION
            )));
        }
        let tokenizer = details.tokenizer.as_ref().ok_or_else(|| {
            Error::invalid_input("MinHash LSH index details carry no tokenizer".to_string())
        })?;
        let params = Self {
            num_hashes: details.num_hashes,
            num_bands: details.num_bands,
            shingle_size: details.shingle_size,
            tokenizer: InvertedIndexParams::try_from(tokenizer)?,
        };
        params.validate()?;
        Ok(params)
    }

    fn details_any(&self) -> Result<prost_types::Any> {
        Ok(prost_types::Any::from_msg(&self.to_details()?)?)
    }

    /// Parse the details stored in the index metadata.
    pub fn from_details_any(details: &prost_types::Any) -> Result<Self> {
        Self::from_details(&details.to_msg::<pb::MinHashLshIndexDetails>()?)
    }

    fn details_hex(&self) -> Result<String> {
        Ok(hex::encode(self.to_details()?.encode_to_vec()))
    }

    fn from_details_hex(hex: &str) -> Result<Self> {
        let bytes = hex::decode(hex).map_err(|err| {
            Error::invalid_input(format!("{hex:?} is not a hex-encoded message: {err}"))
        })?;
        let details = pb::MinHashLshIndexDetails::decode(bytes.as_slice()).map_err(|err| {
            Error::invalid_input(format!("invalid MinHash LSH index details: {err}"))
        })?;
        Self::from_details(&details)
    }
}

/// Deterministic 64-bit generator (SplitMix64) used to derive the permutation
/// coefficients from the seed. Implemented inline so the sequence is a fixed
/// part of the signature contract rather than a dependency's implementation
/// detail.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Computes signatures and band keys. Cloned per batch during builds and per
/// query during searches; the build and query side must share this code path
/// so signatures stay comparable.
#[derive(Clone)]
pub struct SignatureGenerator {
    tokenizer: Box<dyn LanceTokenizer>,
    shingle_size: usize,
    num_bands: usize,
    /// Odd multipliers of the `num_hashes` permutations `h(x) = a * x + b`.
    multipliers: Vec<u64>,
    /// Increments of the permutations.
    increments: Vec<u64>,
    /// Odd multipliers compressing each 64-bit minimum to its stored 16 bits.
    compressors: Vec<u64>,
    /// Token bytes of the current document, separated by [`SHINGLE_SEPARATOR`],
    /// and the end offset of each token; a shingle is one contiguous slice.
    token_text: Vec<u8>,
    token_ends: Vec<usize>,
    /// 64-bit permuted minima of the current document, mixed into the signature.
    mins: Vec<u64>,
}

impl SignatureGenerator {
    pub fn try_new(params: &MinHashLshIndexParams) -> Result<Self> {
        params.validate()?;
        let num_hashes = params.num_hashes as usize;
        let mut state = SIGNATURE_SEED;
        let mut multipliers = Vec::with_capacity(num_hashes);
        let mut increments = Vec::with_capacity(num_hashes);
        for _ in 0..num_hashes {
            multipliers.push(splitmix64(&mut state) | 1);
            increments.push(splitmix64(&mut state));
        }
        let compressors = (0..num_hashes)
            .map(|_| splitmix64(&mut state) | 1)
            .collect();
        Ok(Self {
            tokenizer: params.tokenizer.build()?,
            shingle_size: params.shingle_size as usize,
            num_bands: params.num_bands as usize,
            multipliers,
            increments,
            compressors,
            token_text: Vec::new(),
            token_ends: Vec::new(),
            mins: vec![u64::MAX; num_hashes],
        })
    }

    pub fn num_hashes(&self) -> usize {
        self.multipliers.len()
    }

    pub fn num_bands(&self) -> usize {
        self.num_bands
    }

    /// Fill `signature` (length `num_hashes`) with the MinHash signature of
    /// `text`. Returns false, leaving `signature` unspecified, when the text
    /// has no tokens and therefore no signature.
    pub fn signature(&mut self, text: &str, signature: &mut [SignatureValue]) -> bool {
        debug_assert_eq!(
            signature.len(),
            self.multipliers.len(),
            "signature buffer must hold num_hashes values"
        );
        self.token_text.clear();
        self.token_ends.clear();
        {
            let mut stream = self.tokenizer.token_stream_for_doc(text);
            while stream.advance() {
                if !self.token_ends.is_empty() {
                    self.token_text.push(SHINGLE_SEPARATOR);
                }
                self.token_text
                    .extend_from_slice(stream.token().text.as_bytes());
                self.token_ends.push(self.token_text.len());
            }
        }
        let num_tokens = self.token_ends.len();
        if num_tokens == 0 {
            return false;
        }
        self.mins.fill(u64::MAX);
        let window = self.shingle_size.min(num_tokens);
        for start in 0..=(num_tokens - window) {
            // Tokens are stored back to back with separators, so the shingle
            // is the byte range from this token's start to the window's end.
            let from = if start == 0 {
                0
            } else {
                self.token_ends[start - 1] + 1
            };
            let to = self.token_ends[start + window - 1];
            let base = XxHash64::oneshot(SIGNATURE_SEED, &self.token_text[from..to]);
            for ((value, multiplier), increment) in self
                .mins
                .iter_mut()
                .zip(&self.multipliers)
                .zip(&self.increments)
            {
                // Wrapping arithmetic is the hash function itself, not an
                // overflowing counter. The full 64-bit value is compared: the
                // ordering is decided by its high bits, so the weak low bits
                // of `a * x + b` only break ties. Branch-free min keeps this
                // loop vectorizable.
                let permuted = multiplier.wrapping_mul(base).wrapping_add(*increment);
                *value = (*value).min(permuted);
            }
        }
        // A minimum shrinks with the number of shingles, so its raw bits are
        // not comparable across documents of different lengths. Multiply-shift
        // with a random odd multiplier is 2-universal for any input
        // distribution: distinct minima collide with probability <= 2^-15.
        for ((value, min), compressor) in
            signature.iter_mut().zip(&self.mins).zip(&self.compressors)
        {
            *value = (compressor.wrapping_mul(*min) >> 48) as SignatureValue;
        }
        true
    }

    /// Append the `num_bands` band keys of `signature` to `keys`.
    ///
    /// A band key is the band id in the high 8 bits and the low 56 bits of
    /// the hash of the band's signature values, so keys of one band are
    /// contiguous in the sorted bands file.
    pub fn band_keys(&self, signature: &[SignatureValue], keys: &mut Vec<u64>) {
        let band_width = signature.len() / self.num_bands;
        let mut band_bytes = Vec::with_capacity(band_width * std::mem::size_of::<SignatureValue>());
        for (band_id, band) in signature.chunks_exact(band_width).enumerate() {
            band_bytes.clear();
            for value in band {
                band_bytes.extend_from_slice(&value.to_le_bytes());
            }
            let hash = XxHash64::oneshot(SIGNATURE_SEED, &band_bytes);
            keys.push(((band_id as u64) << 56) | (hash & 0x00FF_FFFF_FFFF_FFFF));
        }
    }
}

/// The texts of a string array of any width; the types a MinHash index accepts.
pub fn text_values(array: &dyn Array) -> Result<Box<dyn Iterator<Item = Option<&str>> + '_>> {
    match array.data_type() {
        DataType::Utf8 => Ok(Box::new(array.as_string::<i32>().iter())),
        DataType::LargeUtf8 => Ok(Box::new(array.as_string::<i64>().iter())),
        DataType::Utf8View => Ok(Box::new(array.as_string_view().iter())),
        other => Err(Error::invalid_input(format!(
            "MinHash LSH index supports Utf8, LargeUtf8 and Utf8View columns, got {other}"
        ))),
    }
}

/// Estimated Jaccard similarity of two signatures: the fraction of positions
/// whose MinHash values agree.
pub fn estimate_jaccard(a: &[SignatureValue], b: &[SignatureValue]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "signatures must have the same width");
    let matches = a.iter().zip(b).filter(|(x, y)| x == y).count();
    matches as f32 / a.len() as f32
}

/// Convert a document index into a segment-local doc id.
///
/// This is the only place where a document count becomes a `u32`; a segment
/// that overflows it must be split rather than silently wrapped.
fn checked_doc_id(doc_index: u64) -> Result<u32> {
    u32::try_from(doc_index).map_err(|_| {
        Error::invalid_input(format!(
            "MinHash LSH index segment cannot hold more than {} rows (row index {} does not fit a u32 doc id); \
             build the index in pieces by calling create_index_uncommitted on disjoint fragment subsets \
             and committing the resulting segments together with commit_existing_index_segments",
            u32::MAX as u64 + 1,
            doc_index
        ))
    })
}

/// A query's signature and band keys, computed once and searched against
/// every segment of a logical index.
#[derive(Debug, Clone)]
pub struct QuerySignature {
    signature: Vec<SignatureValue>,
    band_keys: Vec<u64>,
}

impl QuerySignature {
    /// Compute the query signature with `generator`, or `None` when the text
    /// has no tokens.
    pub fn compute(generator: &mut SignatureGenerator, text: &str) -> Option<Self> {
        let mut signature = vec![SignatureValue::MAX; generator.num_hashes()];
        if !generator.signature(text, &mut signature) {
            return None;
        }
        let mut band_keys = Vec::with_capacity(generator.num_bands());
        generator.band_keys(&signature, &mut band_keys);
        Some(Self {
            signature,
            band_keys,
        })
    }

    pub fn signature(&self) -> &[SignatureValue] {
        &self.signature
    }

    /// Whether a row signature shares at least one band with the query: the
    /// candidate test the index applies, so rows scored without the index
    /// (unindexed fragments) follow the same rule.
    pub fn shares_band(
        &self,
        generator: &SignatureGenerator,
        signature: &[SignatureValue],
        scratch: &mut Vec<u64>,
    ) -> bool {
        scratch.clear();
        generator.band_keys(signature, scratch);
        scratch
            .iter()
            .zip(&self.band_keys)
            .any(|(row, query)| row == query)
    }
}

/// One search hit: a row id and its Jaccard distance (`1 - estimated Jaccard`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinHashHit {
    pub row_id: u64,
    pub distance: f32,
}

impl Eq for MinHashHit {}

impl PartialOrd for MinHashHit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinHashHit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.row_id.cmp(&other.row_id))
    }
}

/// Bounded collection of the `limit` best (smallest distance) hits.
pub struct TopHits {
    limit: usize,
    heap: BinaryHeap<MinHashHit>,
}

impl TopHits {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            heap: BinaryHeap::with_capacity(limit.min(1024)),
        }
    }

    pub fn push(&mut self, hit: MinHashHit) {
        if self.limit == 0 {
            return;
        }
        if self.heap.len() < self.limit {
            self.heap.push(hit);
        } else if self.heap.peek().is_some_and(|worst| hit < *worst) {
            self.heap.pop();
            self.heap.push(hit);
        }
    }

    /// Hits ordered by ascending distance, ties broken by row id.
    pub fn into_sorted(self) -> Vec<MinHashHit> {
        self.heap.into_sorted_vec()
    }
}

impl Extend<MinHashHit> for TopHits {
    fn extend<I: IntoIterator<Item = MinHashHit>>(&mut self, hits: I) {
        for hit in hits {
            self.push(hit);
        }
    }
}

#[derive(Debug, Default)]
pub struct MinHashLshIndexPlugin;

#[derive(Debug)]
pub struct MinHashLshTrainingRequest {
    pub params: MinHashLshIndexParams,
    criteria: TrainingCriteria,
}

impl MinHashLshTrainingRequest {
    pub fn new(params: MinHashLshIndexParams) -> Self {
        Self {
            params,
            criteria: TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
        }
    }
}

impl TrainingRequest for MinHashLshTrainingRequest {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[async_trait]
impl BasicTrainer for MinHashLshIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {}
            other => {
                return Err(Error::invalid_input(format!(
                    "MinHash LSH index can only be created on a string column (Utf8, LargeUtf8, Utf8View), field {} has type {other}",
                    field.name()
                )));
            }
        }
        let params = MinHashLshIndexParams::from_json(params)?;
        Ok(Box::new(MinHashLshTrainingRequest::new(params)))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        _fragment_ids: Option<Vec<u32>>,
        _progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let request = (request as Box<dyn Any>)
            .downcast::<MinHashLshTrainingRequest>()
            .map_err(|_| {
                Error::invalid_input(
                    "must provide training request created by new_training_request".to_string(),
                )
            })?;
        let builder = MinHashLshIndexBuilder::try_new(request.params.clone())?;
        let files = builder.train(data, index_store).await?;
        Ok(CreatedIndex {
            index_details: request.params.details_any()?,
            index_version: MINHASH_LSH_INDEX_VERSION,
            files,
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for MinHashLshIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn name(&self) -> &str {
        "MinHashLsh"
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

    fn version(&self) -> u32 {
        MINHASH_LSH_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        _index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        // Similarity search is planned by the MinHash search node, not by
        // filter expression rewriting.
        None
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(
            MinHashLshIndex::load(index_store, index_details, frag_reuse_index, cache).await?
                as Arc<dyn ScalarIndex>,
        )
    }

    fn validate_new_segments_against_existing(
        &self,
        existing: &[&prost_types::Any],
        incoming: &[&prost_types::Any],
    ) -> Result<()> {
        // Compare parsed parameters rather than bytes: a newer Lance may
        // serialize additional tokenizer keys with their default values, which
        // must not split a logical index.
        let mut all = existing
            .iter()
            .chain(incoming)
            .map(|details| MinHashLshIndexParams::from_details_any(details));
        let Some(first) = all.next().transpose()? else {
            return Ok(());
        };
        for params in all {
            let params = params?;
            if params != first {
                return Err(Error::invalid_input(format!(
                    "MinHash LSH index segments must share identical parameters (signatures are only comparable under the same hash functions and tokenizer); found {first:?} and {params:?}"
                )));
            }
        }
        Ok(())
    }

    fn details_as_json(&self, details: &prost_types::Any) -> Result<serde_json::Value> {
        let details = details.to_msg::<pb::MinHashLshIndexDetails>()?;
        let params = MinHashLshIndexParams::from_details(&details)?;
        let mut value = serde_json::to_value(&params)?;
        if let Some(object) = value.as_object_mut() {
            object.insert(
                "signature_version".to_string(),
                serde_json::Value::from(details.signature_version),
            );
        }
        Ok(value)
    }
}
