// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::fmt::{Debug, Display};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::{
    cmp::{Reverse, min},
    collections::BinaryHeap,
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    ops::Range,
    time::Instant,
};

use crate::metrics::NoOpMetricsCollector;
use crate::prefilter::NoFilter;
use crate::scalar::registry::{TrainingCriteria, TrainingOrdering};
use arrow::array::{FixedSizeListBuilder, Float32Builder};
use arrow::compute::concat_batches;
use arrow::datatypes::{self, Float32Type, Int32Type, UInt64Type};
use arrow::{
    array::{
        AsArray, LargeBinaryBuilder, ListBuilder, StringBuilder, UInt32Builder, UInt64Builder,
    },
    buffer::{Buffer, OffsetBuffer},
};
use arrow::{buffer::ScalarBuffer, datatypes::UInt32Type};
use arrow_array::{
    Array, ArrayRef, Float32Array, LargeBinaryArray, ListArray, OffsetSizeTrait, RecordBatch,
    UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::metrics::Time;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use fst::{Automaton, IntoStreamer, Streamer};
use futures::{FutureExt, Stream, StreamExt, TryStreamExt, stream};
use itertools::{Either, Itertools};
use lance_arrow::{RecordBatchExt, iter_str_array};
use lance_core::cache::{CacheCodec, CacheKey, LanceCache, WeakLanceCache};
use lance_core::deepsize::DeepSizeOf;
use lance_core::error::{DataFusionResult, LanceOptionExt};
use lance_core::utils::address::RowAddress;
use lance_core::utils::tokio::{get_num_compute_intensive_cpus, spawn_cpu};
use lance_core::utils::tracing::{IO_TYPE_LOAD_SCALAR_PART, TRACE_IO_EVENTS};
use lance_core::{Error, ROW_ID, ROW_ID_FIELD, Result};
use lance_select::{RowAddrMask, RowAddrTreeMap};
use roaring::RoaringBitmap;
use std::sync::LazyLock;
use tokio::{sync::OnceCell, task::spawn_blocking};
use tracing::{info, instrument};

use super::encoding::{
    PositionBlockBuilder, decode_group_starts, decode_position_stream_block,
    decompress_posting_block, decompress_posting_remainder,
};
use super::iter::PostingListIterator;
use super::lazy_docset::LazyDocSet;
use super::{InvertedIndexBuilder, InvertedIndexParams, wand::*};
use super::{
    builder::{
        BLOCK_SIZE, PostingGroupAccumulator, PostingGroupConfig, ScoredDoc, doc_file_path,
        inverted_list_schema_for_version, posting_blocks_file_path, posting_coarse_skip_file_path,
        posting_file_path, posting_positions_file_path, token_file_path,
    },
    iter::PlainPostingListIterator,
    query::*,
    scorer::{B, IndexBM25Scorer, K1, Scorer, idf},
};
use super::{
    builder::{InnerBuilder, PositionRecorder},
    iter::CompressedPostingListIterator,
};
use crate::pbold;
use crate::progress::IndexBuildProgress;
use crate::scalar::inverted::scorer::MemBM25Scorer;
use crate::scalar::inverted::tokenizer::document_tokenizer::LanceTokenizer;
use crate::scalar::{
    AnyQuery, BuiltinIndexType, CreatedIndex, IndexReader, IndexStore, MetricsCollector,
    OldIndexDataFilter, RowIdRemapper, ScalarIndex, ScalarIndexParams, SearchResult, TokenQuery,
    UpdateCriteria,
};
use crate::{FtsPrewarmOptions, Index};
use crate::{prefilter::PreFilter, scalar::inverted::iter::take_fst_keys};
use std::str::FromStr;

mod block_wand;

// Version 0: Arrow TokenSetFormat (legacy)
// Version 1: Fst TokenSetFormat with per-doc compressed positions
// Version 2: Fst TokenSetFormat with shared posting-list position streams.
// Version 3: Fst TokenSetFormat with one posting block per row.
pub const INVERTED_INDEX_VERSION_V1: u32 = 1;
pub const INVERTED_INDEX_VERSION_V2: u32 = 2;
pub const INVERTED_INDEX_VERSION_V3: u32 = 3;
pub const TOKENS_FILE: &str = "tokens.lance";
pub const INVERT_LIST_FILE: &str = "invert.lance";
pub const POSTING_BLOCKS_FILE: &str = "posting_blocks.lance";
pub const POSTING_POSITIONS_FILE: &str = "posting_positions.lance";
pub const POSTING_COARSE_SKIP_FILE: &str = "posting_coarse_skip.lance";
pub const DOCS_FILE: &str = "docs.lance";
pub const METADATA_FILE: &str = "metadata.lance";

pub const TOKEN_COL: &str = "_token";
pub const TOKEN_ID_COL: &str = "_token_id";
pub const TOKEN_FST_BYTES_COL: &str = "_token_fst_bytes";
pub const TOKEN_NEXT_ID_COL: &str = "_token_next_id";
pub const TOKEN_TOTAL_LENGTH_COL: &str = "_token_total_length";
pub const FREQUENCY_COL: &str = "_frequency";
pub const POSITION_COL: &str = "_position";
pub const COMPRESSED_POSITION_COL: &str = "_compressed_position";
pub const POSITION_BLOCK_OFFSET_COL: &str = "_position_block_offset";
pub const POSTING_COL: &str = "_posting";
pub const MAX_SCORE_COL: &str = "_max_score";
pub const LENGTH_COL: &str = "_length";
pub const BLOCK_MAX_SCORE_COL: &str = "_block_max_score";
pub const BLOCK_START_COL: &str = "_block_start";
pub const NUM_BLOCKS_COL: &str = "_num_blocks";
pub const COARSE_SKIP_GROUP_BLOCKS: usize = 128;
pub const FIRST_DOC_ID_COL: &str = "_first_doc_id";
pub const LAST_DOC_ID_COL: &str = "_last_doc_id";
pub const NUM_TOKEN_COL: &str = "_num_tokens";
pub const SCORE_COL: &str = "_score";
pub const TOKEN_SET_FORMAT_KEY: &str = "token_set_format";
pub const POSTING_TAIL_CODEC_KEY: &str = "posting_tail_codec";
pub const POSITIONS_LAYOUT_KEY: &str = "positions_layout";
pub const POSITIONS_CODEC_KEY: &str = "positions_codec";
pub const POSTING_BLOCK_LAYOUT_KEY: &str = "posting_block_layout";
pub const POSTING_BLOCK_LAYOUT_BLOCK_ROW_V3: &str = "block_row_v3";
/// Schema-metadata key holding the 1-indexed global-buffer id of the
/// varint-delta-encoded posting-list cache-group boundaries (issue #7040).
/// Absent on indexes written before grouping was introduced, which fall back
/// to the per-token cache path.
pub const POSTING_GROUP_OFFSETS_BUF_KEY: &str = "posting_group_offsets_buf";
pub const POSTING_TAIL_CODEC_FIXED32_V1: &str = "fixed32_v1";
pub const POSTING_TAIL_CODEC_VARINT_DELTA_V1: &str = "varint_delta_v1";
pub const POSITIONS_LAYOUT_SHARED_STREAM_V2: &str = "shared_stream_v2";
pub const POSITIONS_CODEC_VARINT_DOC_DELTA_V2: &str = "varint_doc_delta_v2";
pub const POSITIONS_CODEC_PACKED_DELTA_V1: &str = "packed_delta_v1";
pub const DELETED_FRAGMENTS_COL: &str = "deleted_fragments";

// Just a heuristic when we need to pre-allocate memory for tokens
pub const ESTIMATED_MAX_TOKENS_PER_ROW: usize = 4 * 1024;

pub static SCORE_FIELD: LazyLock<Field> =
    LazyLock::new(|| Field::new(SCORE_COL, DataType::Float32, true));
pub static FTS_SCHEMA: LazyLock<SchemaRef> =
    LazyLock::new(|| Arc::new(Schema::new(vec![ROW_ID_FIELD.clone(), SCORE_FIELD.clone()])));
static ROW_ID_SCHEMA: LazyLock<SchemaRef> =
    LazyLock::new(|| Arc::new(Schema::new(vec![ROW_ID_FIELD.clone()])));

pub fn resolve_fts_format_version(
    value: Option<&str>,
) -> std::result::Result<InvertedListFormatVersion, Error> {
    match value {
        Some(value) => value.parse(),
        None => Ok(default_fts_format_version()),
    }
}

pub fn default_fts_format_version() -> InvertedListFormatVersion {
    InvertedListFormatVersion::V2
}

pub fn current_fts_format_version() -> InvertedListFormatVersion {
    default_fts_format_version()
}

pub fn max_supported_fts_format_version() -> InvertedListFormatVersion {
    InvertedListFormatVersion::V3
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum InvertedListFormatVersion {
    V1,
    #[default]
    V2,
    V3,
}

impl InvertedListFormatVersion {
    pub fn from_posting_tail_codec(codec: PostingTailCodec) -> Self {
        match codec {
            PostingTailCodec::Fixed32 => Self::V1,
            PostingTailCodec::VarintDelta => Self::V2,
        }
    }

    pub fn index_version(self) -> u32 {
        match self {
            Self::V1 => INVERTED_INDEX_VERSION_V1,
            Self::V2 => INVERTED_INDEX_VERSION_V2,
            Self::V3 => INVERTED_INDEX_VERSION_V3,
        }
    }

    pub fn posting_tail_codec(self) -> PostingTailCodec {
        match self {
            Self::V1 => PostingTailCodec::Fixed32,
            Self::V2 | Self::V3 => PostingTailCodec::VarintDelta,
        }
    }

    pub fn position_codec(self) -> Option<PositionStreamCodec> {
        match self {
            Self::V1 => None,
            Self::V2 | Self::V3 => Some(PositionStreamCodec::PackedDelta),
        }
    }

    pub fn uses_shared_position_stream(self) -> bool {
        matches!(self, Self::V2 | Self::V3)
    }

    pub fn uses_block_row_layout(self) -> bool {
        matches!(self, Self::V3)
    }
}

impl FromStr for InvertedListFormatVersion {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim() {
            "1" | "v1" | "V1" => Ok(Self::V1),
            "2" | "v2" | "V2" => Ok(Self::V2),
            "3" | "v3" | "V3" => Ok(Self::V3),
            other => Err(Error::index(format!(
                "unsupported FTS format version {}, expected 1, 2, or 3",
                other
            ))),
        }
    }
}

#[derive(Debug)]
struct PartitionCandidates {
    tokens_by_position: Vec<String>,
    grouped_expansions: Vec<GroupedExpansionTerms>,
    candidates: Vec<DocCandidate>,
}

impl PartitionCandidates {
    fn empty() -> Self {
        Self {
            tokens_by_position: Vec::new(),
            grouped_expansions: Vec::new(),
            candidates: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct LoadedPostings {
    postings: Vec<PostingIterator>,
    grouped_expansions: Vec<GroupedExpansionTerms>,
}

impl LoadedPostings {
    fn empty() -> Self {
        Self {
            postings: Vec::new(),
            grouped_expansions: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct GroupedExpansionTerms {
    position: u32,
    terms: Vec<ExpansionTermFreqs>,
}

#[derive(Debug, Clone)]
struct V3TermPlan {
    token_id: u32,
    token: String,
    position: u32,
    term_index: u32,
    posting_len: u32,
    blocks: Vec<V3BlockMeta>,
}

#[derive(Debug, Clone, Copy)]
struct V3BlockMeta {
    term_idx: usize,
    block_idx: usize,
    block_row: usize,
    first_doc_id: u32,
    last_doc_id: u32,
    block_max_score: f32,
}

#[derive(Debug, Clone, DeepSizeOf)]
struct V3BlockSkipMeta {
    block_idx: usize,
    block_row: usize,
    first_doc_id: u32,
    last_doc_id: u32,
    block_max_score: f32,
}

#[derive(Debug, Clone, Copy)]
struct V3Window {
    first_doc_id: u32,
    last_doc_id: u32,
    upper_bound: f32,
}

#[derive(Debug, Clone)]
struct V3DecodedPostingBlock {
    doc_ids: Vec<u32>,
    freqs: Vec<u32>,
}

#[derive(Debug, Clone)]
struct V3DecodedPositionBlock {
    values: Vec<u32>,
    offsets: Vec<usize>,
}

#[derive(Debug, Clone)]
struct V3TermPositions {
    position_in_query: i32,
    positions: Vec<u32>,
}

#[derive(Debug)]
struct V3DocAccumulator {
    freqs: Vec<Option<u32>>,
    block_rows: Vec<Option<usize>>,
    doc_offsets: Vec<Option<usize>>,
}

impl V3DocAccumulator {
    fn new(num_terms: usize) -> Self {
        Self {
            freqs: vec![None; num_terms],
            block_rows: vec![None; num_terms],
            doc_offsets: vec![None; num_terms],
        }
    }

    fn add(&mut self, term_idx: usize, block_row: usize, doc_offset: usize, freq: u32) {
        self.freqs[term_idx] = Some(freq);
        self.block_rows[term_idx] = Some(block_row);
        self.doc_offsets[term_idx] = Some(doc_offset);
    }

    fn matches_operator(&self, operator: Operator) -> bool {
        match operator {
            Operator::And => self.freqs.iter().all(Option::is_some),
            Operator::Or => self.freqs.iter().any(Option::is_some),
        }
    }

    fn term_freqs(&self, term_plans: &[V3TermPlan]) -> Vec<(u32, u32)> {
        self.freqs
            .iter()
            .enumerate()
            .filter_map(|(term_idx, freq)| freq.map(|freq| (term_plans[term_idx].term_index, freq)))
            .collect()
    }
}

fn atomic_store_max_f32(slot: &AtomicU32, val: f32) {
    let mut cur = slot.load(Ordering::Relaxed);
    while val > f32::from_bits(cur) {
        match slot.compare_exchange_weak(cur, val.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

fn v3_phrase_positions_match(mut terms: Vec<V3TermPositions>, slop: i32) -> bool {
    if terms.is_empty() || terms.iter().any(|term| term.positions.is_empty()) {
        return false;
    }
    for term in &mut terms {
        term.positions.sort_unstable();
        term.positions.dedup();
    }
    if slop == 0 {
        return v3_exact_phrase_positions_match(terms);
    }

    terms.sort_unstable_by_key(|term| term.position_in_query);
    let first = &terms[0];
    for &position in &first.positions {
        let relative = position as i32 - first.position_in_query;
        if v3_sloppy_phrase_positions_match_from(&terms, 1, relative, slop) {
            return true;
        }
    }
    false
}

fn v3_exact_phrase_positions_match(mut terms: Vec<V3TermPositions>) -> bool {
    terms.sort_unstable_by_key(|term| term.positions.len());
    let lead = &terms[0];
    for &anchor in &lead.positions {
        let Some(base) = anchor.checked_sub(lead.position_in_query as u32) else {
            continue;
        };
        if terms.iter().skip(1).all(|term| {
            base.checked_add(term.position_in_query as u32)
                .is_some_and(|target| term.positions.binary_search(&target).is_ok())
        }) {
            return true;
        }
    }
    false
}

fn v3_sloppy_phrase_positions_match_from(
    terms: &[V3TermPositions],
    term_idx: usize,
    previous_relative: i32,
    slop: i32,
) -> bool {
    if term_idx == terms.len() {
        return true;
    }
    let term = &terms[term_idx];
    let lower = previous_relative + term.position_in_query;
    let upper = previous_relative + slop + term.position_in_query;
    for &position in &term.positions {
        let position = position as i32;
        if position < lower {
            continue;
        }
        if position > upper {
            break;
        }
        if v3_sloppy_phrase_positions_match_from(
            terms,
            term_idx + 1,
            position - term.position_in_query,
            slop,
        ) {
            return true;
        }
    }
    false
}

fn grouped_rescore_wand_limit(
    limit: Option<usize>,
    grouped_expansions: &[GroupedExpansionTerms],
) -> Option<usize> {
    let limit = limit?;
    // Grouped fuzzy AND rescoring needs a small candidate cushion because WAND
    // ranks by the unioned group posting first and the exact expansion IDF later.
    let expansion_terms = grouped_expansions
        .iter()
        .map(|group| group.terms.len())
        .sum::<usize>()
        .max(1);
    Some(limit.saturating_mul(expansion_terms))
}

#[derive(Debug)]
struct ExpansionTermFreqs {
    token: String,
    freqs_by_posting_doc_id: Vec<(u64, u32)>,
}

impl ExpansionTermFreqs {
    fn new(token: String, posting: &PostingList) -> Self {
        let freqs_by_posting_doc_id = posting
            .iter()
            .map(|(posting_doc_id, freq, _)| (posting_doc_id, freq))
            .collect();
        Self {
            token,
            freqs_by_posting_doc_id,
        }
    }

    fn frequency(&self, posting_doc_id: u64) -> Option<u32> {
        self.freqs_by_posting_doc_id
            .binary_search_by_key(&posting_doc_id, |(doc_id, _)| *doc_id)
            .ok()
            .map(|idx| self.freqs_by_posting_doc_id[idx].1)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub enum TokenSetFormat {
    Arrow,
    #[default]
    Fst,
}

impl Display for TokenSetFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Arrow => f.write_str("arrow"),
            Self::Fst => f.write_str("fst"),
        }
    }
}

impl FromStr for TokenSetFormat {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim() {
            "" => Ok(Self::Arrow),
            "arrow" => Ok(Self::Arrow),
            "fst" => Ok(Self::Fst),
            other => Err(Error::index(format!(
                "unsupported token set format {}",
                other
            ))),
        }
    }
}

impl DeepSizeOf for TokenSetFormat {
    fn deep_size_of_children(&self, _: &mut lance_core::deepsize::Context) -> usize {
        0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PositionStreamCodec {
    VarintDocDelta,
    #[default]
    PackedDelta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PostingTailCodec {
    Fixed32,
    #[default]
    VarintDelta,
}

impl PostingTailCodec {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed32 => POSTING_TAIL_CODEC_FIXED32_V1,
            Self::VarintDelta => POSTING_TAIL_CODEC_VARINT_DELTA_V1,
        }
    }

    fn from_metadata_value(value: &str) -> Result<Self> {
        match value.trim() {
            POSTING_TAIL_CODEC_FIXED32_V1 => Ok(Self::Fixed32),
            POSTING_TAIL_CODEC_VARINT_DELTA_V1 => Ok(Self::VarintDelta),
            other => Err(Error::index(format!(
                "unsupported posting tail codec {}",
                other
            ))),
        }
    }
}

pub(super) fn parse_posting_tail_codec(
    metadata: &HashMap<String, String>,
) -> Result<PostingTailCodec> {
    Ok(metadata
        .get(POSTING_TAIL_CODEC_KEY)
        .map(|codec| PostingTailCodec::from_metadata_value(codec))
        .transpose()?
        .unwrap_or(PostingTailCodec::Fixed32))
}

impl PositionStreamCodec {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::VarintDocDelta => POSITIONS_CODEC_VARINT_DOC_DELTA_V2,
            Self::PackedDelta => POSITIONS_CODEC_PACKED_DELTA_V1,
        }
    }

    fn from_metadata_value(value: &str) -> Result<Self> {
        match value.trim() {
            POSITIONS_CODEC_VARINT_DOC_DELTA_V2 => Ok(Self::VarintDocDelta),
            POSITIONS_CODEC_PACKED_DELTA_V1 => Ok(Self::PackedDelta),
            other => Err(Error::index(format!(
                "unsupported positions codec {}",
                other
            ))),
        }
    }
}

fn parse_shared_position_codec(metadata: &HashMap<String, String>) -> Result<PositionStreamCodec> {
    if let Some(codec) = metadata.get(POSITIONS_CODEC_KEY) {
        return PositionStreamCodec::from_metadata_value(codec);
    }

    match metadata
        .get(POSITIONS_LAYOUT_KEY)
        .map(|layout| layout.as_str())
    {
        Some(POSITIONS_LAYOUT_SHARED_STREAM_V2) => Ok(PositionStreamCodec::VarintDocDelta),
        _ => Ok(PositionStreamCodec::VarintDocDelta),
    }
}

pub(super) fn parse_format_version_from_metadata(
    metadata: &HashMap<String, String>,
) -> Result<InvertedListFormatVersion> {
    if matches!(
        metadata.get(POSTING_BLOCK_LAYOUT_KEY).map(String::as_str),
        Some(POSTING_BLOCK_LAYOUT_BLOCK_ROW_V3)
    ) {
        return Ok(InvertedListFormatVersion::V3);
    }
    if metadata.contains_key(POSITIONS_CODEC_KEY) || metadata.contains_key(POSITIONS_LAYOUT_KEY) {
        return Ok(InvertedListFormatVersion::V2);
    }
    if parse_posting_tail_codec(metadata)? == PostingTailCodec::VarintDelta {
        Ok(InvertedListFormatVersion::V2)
    } else {
        Ok(InvertedListFormatVersion::V1)
    }
}

fn checked_usize(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::index(format!("{name} {value} exceeds usize::MAX")))
}

#[derive(Clone)]
pub struct InvertedIndex {
    params: InvertedIndexParams,
    store: Arc<dyn IndexStore>,
    tokenizer: Box<dyn LanceTokenizer>,
    token_set_format: TokenSetFormat,
    format_version: InvertedListFormatVersion,
    pub(crate) partitions: Vec<Arc<InvertedPartition>>,
    corpus_stats: Arc<OnceCell<(u64, usize)>>,
    // Fragments which are contained in the index, but no longer in the dataset.
    // These should be pruned at search time since we don't prune them at update time.
    deleted_fragments: RoaringBitmap,
}

impl Debug for InvertedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvertedIndex")
            .field("params", &self.params)
            .field("token_set_format", &self.token_set_format)
            .field("format_version", &self.format_version)
            .field("partitions", &self.partitions)
            .field("deleted_fragments", &self.deleted_fragments)
            .finish()
    }
}

impl DeepSizeOf for InvertedIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.partitions.deep_size_of_children(context)
    }
}

/// Resolve any `Pending` candidates that wand emitted via the
/// deferred-row_id path. After this returns, every entry in
/// `candidates` carries a real row_id.
async fn resolve_deferred_candidates(
    docs: &LazyDocSet,
    candidates: &mut [DocCandidate],
) -> Result<()> {
    let pending: Vec<u32> = candidates
        .iter()
        .filter_map(|c| match c.addr {
            CandidateAddr::Pending(d) => Some(d),
            CandidateAddr::RowId(_) => None,
        })
        .collect();
    if pending.is_empty() {
        return Ok(());
    }
    let mut iter = docs.resolve_row_ids(&pending).await?.into_iter();
    for c in candidates {
        if matches!(c.addr, CandidateAddr::Pending(_)) {
            let r = iter.next().ok_or_else(|| {
                Error::internal("resolve_row_ids returned fewer items than requested")
            })?;
            c.addr = CandidateAddr::RowId(r);
        }
    }
    Ok(())
}

impl InvertedIndex {
    fn format_version(&self) -> InvertedListFormatVersion {
        self.format_version
    }

    fn index_version(&self) -> u32 {
        match self.token_set_format {
            TokenSetFormat::Arrow => 0,
            TokenSetFormat::Fst => self.format_version().index_version(),
        }
    }

    fn posting_tail_codec(&self) -> PostingTailCodec {
        self.partitions
            .first()
            .map(|partition| partition.inverted_list.posting_tail_codec())
            .unwrap_or_default()
    }

    fn to_builder(&self) -> InvertedIndexBuilder {
        self.to_builder_with_offset(None)
    }

    fn to_builder_with_offset(&self, fragment_mask: Option<u64>) -> InvertedIndexBuilder {
        if self.is_legacy() {
            // for legacy format, we re-create the index in the new format
            InvertedIndexBuilder::from_existing_index(
                self.params.clone(),
                None,
                Vec::new(),
                self.token_set_format,
                fragment_mask,
                self.deleted_fragments.clone(),
            )
            .with_posting_tail_codec(self.posting_tail_codec())
        } else {
            let partitions = match fragment_mask {
                Some(fragment_mask) => self
                    .partitions
                    .iter()
                    // Filter partitions that belong to the specified fragment
                    // The mask contains fragment_id in high 32 bits, we check if partition's
                    // fragment_id matches by comparing the masked result with the original mask
                    .filter(|part| part.belongs_to_fragment(fragment_mask))
                    .map(|part| part.id())
                    .collect(),
                None => self.partitions.iter().map(|part| part.id()).collect(),
            };

            InvertedIndexBuilder::from_existing_index(
                self.params.clone(),
                Some(self.store.clone()),
                partitions,
                self.token_set_format,
                fragment_mask,
                self.deleted_fragments.clone(),
            )
            .with_format_version(self.format_version())
        }
    }

    pub fn tokenizer(&self) -> Box<dyn LanceTokenizer> {
        self.tokenizer.clone()
    }

    pub fn params(&self) -> &InvertedIndexParams {
        &self.params
    }

    /// Returns the number of partitions in this inverted index.
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }
    /// Returns the set of fragments which are contained in the index, but no longer in the dataset.
    ///
    /// Most other indices remove data from deleted fragments when the index updates (copy-on-write).
    /// However, this would require an expensive copy of the FTS index.  Instead, we track the deleted
    /// fragments and prune them at search time (merge-on-read).
    pub fn deleted_fragments(&self) -> &RoaringBitmap {
        &self.deleted_fragments
    }

    pub async fn merge_segments(
        segments: &[Arc<Self>],
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        old_data_filter: Option<OldIndexDataFilter>,
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let Some(first) = segments.first() else {
            return Err(Error::invalid_input(
                "cannot merge inverted index without at least one source segment".to_string(),
            ));
        };

        for segment in segments.iter().skip(1) {
            if segment.params != first.params {
                return Err(Error::index(
                    "cannot merge inverted index segments with different parameters".to_string(),
                ));
            }
            if segment.token_set_format != first.token_set_format {
                return Err(Error::index(
                    "cannot merge inverted index segments with different token set formats"
                        .to_string(),
                ));
            }
            if segment.format_version() != first.format_version() {
                return Err(Error::index(
                    "cannot merge inverted index segments with different format versions"
                        .to_string(),
                ));
            }
            if segment.posting_tail_codec() != first.posting_tail_codec() {
                return Err(Error::index(
                    "cannot merge inverted index segments with different posting tail codecs"
                        .to_string(),
                ));
            }
        }

        let mut builder = InvertedIndexBuilder::new(first.params.clone()).with_progress(progress);
        builder = builder
            .with_token_set_format(first.token_set_format)
            .with_format_version(first.format_version())
            .with_posting_tail_codec(first.posting_tail_codec());
        let files = builder
            .update_from_segments(new_data, dest_store, segments, old_data_filter)
            .await?;

        let details = pbold::InvertedIndexDetails::try_from(&first.params)?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&details).unwrap(),
            index_version: first.index_version(),
            files,
        })
    }

    /// Build a single-segment [`MemBM25Scorer`] whose per-term IDF table
    /// covers every token that the per-partition scoring loop will look
    /// up. For fuzzy queries that means the union of Levenshtein
    /// expansions, not just the raw query tokens — otherwise
    /// `query_weight(expanded_token)` returns 0 and the BM25 contribution
    /// of every expanded match is discarded.
    pub async fn bm25_base_scorer(
        &self,
        query_tokens: &Tokens,
        params: &FtsSearchParams,
    ) -> Result<MemBM25Scorer> {
        if matches!(params.fuzziness, Some(n) if n != 0) {
            let expanded = self.expand_fuzzy_tokens(query_tokens, params)?;
            self.bm25_scorer_for_final_tokens(&expanded).await
        } else {
            self.bm25_scorer_for_final_tokens(query_tokens).await
        }
    }

    /// Scorer for a token list that needs no further fuzzy expansion: dedup
    /// the terms and pull their document frequencies. `bm25_search` calls
    /// this with the tokens it already expanded, so the expansion runs once
    /// per query rather than once for the scorer and once per partition.
    async fn bm25_scorer_for_final_tokens(&self, tokens: &Tokens) -> Result<MemBM25Scorer> {
        let (total_tokens, num_docs) = self.aggregate_corpus_stats().await?;
        let mut terms: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for token in tokens {
            if seen.insert(token.to_string()) {
                terms.push(token.to_string());
            }
        }
        let mut token_docs = HashMap::with_capacity(terms.len());
        for term in &terms {
            let df = self.df_for_term(term).await?;
            token_docs.insert(term.clone(), df);
        }
        Ok(MemBM25Scorer::new(total_tokens, num_docs, token_docs))
    }

    pub async fn bm25_stats_for_terms(&self, terms: &[String]) -> Result<(u64, usize, Vec<usize>)> {
        let (total_tokens, num_docs) = self.aggregate_corpus_stats().await?;
        let token_docs =
            futures::future::try_join_all(terms.iter().map(|term| self.df_for_term(term))).await?;
        Ok((total_tokens, num_docs, token_docs))
    }

    /// Aggregate per-partition `total_tokens` and `num_docs` across the
    /// index. `len` is cheap (no IO); `total_tokens_num` reads only the
    /// num_tokens column the first time per partition and caches it on
    /// `LazyDocSet`. Avoids materializing the full DocSet just to get
    /// these two scalars.
    async fn aggregate_corpus_stats(&self) -> Result<(u64, usize)> {
        self.corpus_stats
            .get_or_try_init(|| async {
                let io_parallelism = self.store.io_parallelism();
                let num_docs: usize = self.partitions.iter().map(|p| p.docs.len()).sum();
                let futures = self
                    .partitions
                    .iter()
                    .map(|p| {
                        let docs = p.docs.clone();
                        async move { docs.total_tokens_num().await }
                    })
                    .collect::<Vec<_>>();
                let totals: Vec<u64> = stream::iter(futures)
                    .buffer_unordered(io_parallelism)
                    .try_collect()
                    .await?;
                Ok((totals.into_iter().sum(), num_docs))
            })
            .await
            .copied()
    }

    /// Sum the posting-list length for `term` across this index's partitions
    /// via single-row reads, with partition lookups bounded by the store's
    /// `io_parallelism()`.
    async fn df_for_term(&self, term: &str) -> Result<usize> {
        let io_parallelism = self.store.io_parallelism();
        let futures = self
            .partitions
            .iter()
            .map(|part| {
                let part = part.clone();
                async move {
                    match part.tokens.get(term) {
                        Some(token_id) => part.inverted_list.posting_len_for_token(token_id).await,
                        None => Ok(0),
                    }
                }
            })
            .collect::<Vec<_>>();
        let dfs: Vec<usize> = stream::iter(futures)
            .buffer_unordered(io_parallelism)
            .try_collect()
            .await?;
        Ok(dfs.into_iter().sum())
    }

    /// Expand fuzzy query tokens against all partitions in this segment.
    ///
    /// `params.max_expansions` caps the whole query's expansion, not any
    /// single partition's: for each query token the per-partition candidates
    /// (each streamed in FST key order) merge into one lexicographically
    /// ordered set, and the remaining budget takes a prefix of it. The
    /// selected terms are a pure function of the segment's vocabulary, so
    /// splitting the same corpus into more partitions cannot change which
    /// terms a fuzzy query matches.
    pub fn expand_fuzzy_tokens(&self, tokens: &Tokens, params: &FtsSearchParams) -> Result<Tokens> {
        let mut expanded_tokens = Vec::new();
        let mut expanded_positions = Vec::new();
        let mut seen = HashSet::new();
        for token_idx in 0..tokens.len() {
            let remaining = params.max_expansions.saturating_sub(expanded_tokens.len());
            if remaining == 0 {
                break;
            }
            let token = tokens.get_token(token_idx);
            let position = tokens.position(token_idx);
            // Each partition contributes at most its `remaining`
            // lexicographically smallest candidates, so the global
            // lex-smallest `remaining` selection below is unaffected by the
            // per-partition truncation.
            let mut candidates = BTreeSet::new();
            let base_prefix_len = tokens.token_type().prefix_len(token) as u32;
            for partition in &self.partitions {
                partition.collect_fuzzy_candidates(
                    token,
                    base_prefix_len,
                    params,
                    remaining,
                    &mut candidates,
                )?;
            }
            for candidate in candidates {
                if expanded_tokens.len() >= params.max_expansions {
                    break;
                }
                if seen.insert((candidate.clone(), position)) {
                    expanded_tokens.push(candidate);
                    expanded_positions.push(position);
                }
            }
        }
        Ok(Tokens::with_positions(
            expanded_tokens,
            expanded_positions,
            tokens.token_type().clone(),
        ))
    }

    /// Search documents that match the query and return row ids sorted by BM25 score.
    ///
    /// When `base_scorer` is provided, search uses those corpus-level BM25 statistics
    /// instead of deriving them from this segment alone.
    #[instrument(level = "debug", skip_all)]
    pub async fn bm25_search(
        &self,
        tokens: Arc<Tokens>,
        params: Arc<FtsSearchParams>,
        operator: Operator,
        prefilter: Arc<dyn PreFilter>,
        metrics: Arc<dyn MetricsCollector>,
        base_scorer: Option<&MemBM25Scorer>,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        // Fuzzy expansion runs once here, with the global `max_expansions`
        // budget, instead of once per partition: partitions receive the
        // final token list, so the matched terms cannot depend on how the
        // corpus happens to be partitioned.
        let tokens = if matches!(params.fuzziness, Some(n) if n != 0) {
            let expanded = Arc::new(self.expand_fuzzy_tokens(tokens.as_ref(), params.as_ref())?);
            if operator == Operator::And || params.phrase_slop.is_some() {
                // AND/phrase semantics require every original token position
                // to keep at least one expansion; a position that expands to
                // nothing anywhere in the segment can never be matched.
                let surviving = (0..expanded.len())
                    .map(|idx| expanded.position(idx))
                    .collect::<HashSet<_>>();
                if (0..tokens.len()).any(|idx| !surviving.contains(&tokens.position(idx))) {
                    return Ok((Vec::new(), Vec::new()));
                }
            }
            expanded
        } else {
            tokens
        };

        // The wand only consults `scorer.doc_weight`, which is metadata-free.
        // The outer aggregation below consults `scorer.query_weight`, which
        // hits per-token `posting_len`; building a `MemBM25Scorer` with
        // precomputed per-term IDFs avoids the v2 bulk metadata pull.
        let local_scorer;
        let scorer: &dyn Scorer = if let Some(base_scorer) = base_scorer {
            base_scorer
        } else {
            local_scorer = self.bm25_scorer_for_final_tokens(tokens.as_ref()).await?;
            &local_scorer
        };

        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        fn push_scored_candidate(
            candidates: &mut BinaryHeap<Reverse<ScoredDoc>>,
            limit: usize,
            addr: CandidateAddr,
            score: f32,
        ) -> Result<()> {
            // resolve_deferred_candidates ran upstream, so every candidate
            // carries a real row_id at this point.
            let row_id = match addr {
                CandidateAddr::RowId(r) => r,
                CandidateAddr::Pending(_) => {
                    return Err(Error::internal(
                        "bm25_search post-condition: deferred candidate left unresolved",
                    ));
                }
            };

            if candidates.len() < limit {
                candidates.push(Reverse(ScoredDoc::new(row_id, score)));
            } else if candidates.peek().unwrap().0.score.0 < score {
                candidates.pop();
                candidates.push(Reverse(ScoredDoc::new(row_id, score)));
            }
            Ok(())
        }

        let mask = prefilter.mask();

        let mut candidates = BinaryHeap::new();
        // Shared top-k floor across this query's partitions. Seeded to -inf so
        // the first real score wins; each partition publishes its local k-th
        // and prunes against the running global k-th (a lower bound on the true
        // global k-th — see `Wand::shared_threshold`).
        let shared_threshold = Arc::new(AtomicU32::new(f32::NEG_INFINITY.to_bits()));
        let parts = self
            .partitions
            .iter()
            .map(|part| {
                let part = part.clone();
                let tokens = tokens.clone();
                let params = params.clone();
                let mask = mask.clone();
                let metrics = metrics.clone();
                let shared_threshold = shared_threshold.clone();
                async move {
                    if let Some(mut partition_result) = part
                        .try_bm25_search_v3_blocks(
                            tokens.as_ref(),
                            params.as_ref(),
                            operator,
                            mask.clone(),
                            metrics.clone(),
                            shared_threshold.clone(),
                        )
                        .await?
                    {
                        resolve_deferred_candidates(&part.docs, &mut partition_result.candidates)
                            .await?;
                        return Result::Ok(partition_result);
                    }

                    let loaded_postings = part
                        .load_posting_lists(
                            tokens.as_ref(),
                            params.as_ref(),
                            operator,
                            metrics.as_ref(),
                        )
                        .await?;
                    let LoadedPostings {
                        postings,
                        grouped_expansions,
                    } = loaded_postings;
                    if postings.is_empty() {
                        // No hits in this partition; its DocSet stays
                        // unloaded, so we never pay the per-doc
                        // row_id/num_tokens download for it.
                        return Result::Ok(PartitionCandidates::empty());
                    }
                    let docs_for_wand = part.docs.docs_for_wand(mask.as_ref()).await?;
                    let max_position = postings
                        .iter()
                        .map(|posting| posting.term_index() as usize)
                        .max()
                        .unwrap_or_default();
                    let mut tokens_by_position = vec![String::new(); max_position + 1];
                    for posting in &postings {
                        let idx = posting.term_index() as usize;
                        tokens_by_position[idx] = posting.token().to_owned();
                    }
                    let params = params.clone();
                    let mask = mask.clone();
                    let metrics = metrics.clone();
                    let part_for_wand = part.clone();
                    let has_grouped_expansions = !grouped_expansions.is_empty();
                    let wand_params = if has_grouped_expansions {
                        let mut rescoring_params = params.as_ref().clone();
                        rescoring_params.limit =
                            grouped_rescore_wand_limit(params.limit, &grouped_expansions);
                        Arc::new(rescoring_params)
                    } else {
                        params.clone()
                    };
                    let partition_threshold = if has_grouped_expansions {
                        Arc::new(AtomicU32::new(f32::NEG_INFINITY.to_bits()))
                    } else {
                        shared_threshold
                    };
                    let candidates = spawn_cpu(move || {
                        let candidates = part_for_wand.bm25_search(
                            docs_for_wand.as_ref(),
                            wand_params.as_ref(),
                            operator,
                            mask,
                            postings,
                            metrics.as_ref(),
                            partition_threshold,
                        )?;
                        std::result::Result::<_, Error>::Ok(candidates)
                    })
                    .await?;
                    let mut partition_result = PartitionCandidates {
                        tokens_by_position,
                        grouped_expansions,
                        candidates,
                    };
                    resolve_deferred_candidates(&part.docs, &mut partition_result.candidates)
                        .await?;
                    Result::Ok(partition_result)
                }
            })
            .collect::<Vec<_>>();
        let mut parts = stream::iter(parts).buffer_unordered(get_num_compute_intensive_cpus());
        let mut idf_cache: HashMap<String, f32> = HashMap::new();
        while let Some(res) = parts.try_next().await? {
            if res.candidates.is_empty() {
                continue;
            }
            let PartitionCandidates {
                tokens_by_position,
                grouped_expansions,
                candidates: part_candidates,
            } = res;
            let mut idf_by_position = Vec::with_capacity(tokens_by_position.len());
            for token in &tokens_by_position {
                let idf_weight = match idf_cache.get(token) {
                    Some(weight) => *weight,
                    None => {
                        let weight = scorer.query_weight(token);
                        idf_cache.insert(token.clone(), weight);
                        weight
                    }
                };
                idf_by_position.push(idf_weight);
            }

            if grouped_expansions.is_empty() {
                for DocCandidate {
                    addr,
                    freqs,
                    doc_length,
                    ..
                } in part_candidates
                {
                    let mut score = 0.0;
                    for (term_index, freq) in freqs.into_iter() {
                        debug_assert!((term_index as usize) < idf_by_position.len());
                        score += idf_by_position[term_index as usize]
                            * scorer.doc_weight(freq, doc_length);
                    }
                    push_scored_candidate(&mut candidates, limit, addr, score)?;
                }
            } else {
                let grouped_positions = grouped_expansions
                    .iter()
                    .map(|group| group.position)
                    .collect::<HashSet<_>>();
                for DocCandidate {
                    addr,
                    posting_doc_id,
                    freqs,
                    doc_length,
                } in part_candidates
                {
                    let mut score = 0.0;
                    for (term_index, freq) in freqs.into_iter() {
                        if grouped_positions.contains(&term_index) {
                            continue;
                        }
                        debug_assert!((term_index as usize) < idf_by_position.len());
                        score += idf_by_position[term_index as usize]
                            * scorer.doc_weight(freq, doc_length);
                    }
                    for group in &grouped_expansions {
                        for term in &group.terms {
                            let Some(freq) = term.frequency(posting_doc_id) else {
                                continue;
                            };
                            let idf_weight = match idf_cache.get(&term.token) {
                                Some(weight) => *weight,
                                None => {
                                    let weight = scorer.query_weight(&term.token);
                                    idf_cache.insert(term.token.clone(), weight);
                                    weight
                                }
                            };
                            score += idf_weight * scorer.doc_weight(freq, doc_length);
                        }
                    }
                    push_scored_candidate(&mut candidates, limit, addr, score)?;
                }
            }
        }

        Ok(candidates
            .into_sorted_vec()
            .into_iter()
            .map(|Reverse(doc)| (doc.row_id, doc.score.0))
            .unzip())
    }

    async fn load_legacy_index(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        log::warn!("loading legacy FTS index");
        let tokens_fut = tokio::spawn({
            let store = store.clone();
            async move {
                let token_reader = store.open_index_file(TOKENS_FILE).await?;
                let tokenizer = token_reader
                    .schema()
                    .metadata
                    .get("tokenizer")
                    .map(|s| serde_json::from_str::<InvertedIndexParams>(s))
                    .transpose()?
                    .unwrap_or_default();
                let tokens = TokenSet::load(token_reader, TokenSetFormat::Arrow).await?;
                Result::Ok((tokenizer, tokens))
            }
        });
        let invert_list_fut = tokio::spawn({
            let store = store.clone();
            let index_cache_clone = index_cache.clone();
            async move {
                let invert_list_reader = store.open_index_file(INVERT_LIST_FILE).await?;
                let invert_list =
                    PostingListReader::try_new(invert_list_reader, &index_cache_clone).await?;
                Result::Ok(Arc::new(invert_list))
            }
        });
        let docs_fut = tokio::spawn({
            let store = store.clone();
            async move {
                let docs_reader = store.open_index_file(DOCS_FILE).await?;
                let docs = DocSet::load(docs_reader, true, frag_reuse_index).await?;
                Result::Ok(docs)
            }
        });

        let (tokenizer_config, tokens) = tokens_fut.await??;
        let inverted_list = invert_list_fut.await??;
        let docs = docs_fut.await??;

        let tokenizer = tokenizer_config.build()?;

        Ok(Arc::new(Self {
            params: tokenizer_config,
            store: store.clone(),
            tokenizer,
            token_set_format: TokenSetFormat::Arrow,
            format_version: InvertedListFormatVersion::V1,
            partitions: vec![Arc::new(InvertedPartition {
                id: 0,
                store,
                tokens,
                inverted_list,
                docs: Arc::new(LazyDocSet::from_loaded(docs)),
                token_set_format: TokenSetFormat::Arrow,
            })],
            corpus_stats: Arc::new(OnceCell::new()),
            deleted_fragments: RoaringBitmap::new(),
        }))
    }

    pub fn is_legacy(&self) -> bool {
        self.partitions.len() == 1 && self.partitions[0].is_legacy()
    }

    pub async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        index_cache: &LanceCache,
    ) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        // for new index format, there is a metadata file and multiple partitions,
        // each partition is a separate index containing tokens, inverted list and docs.
        // for old index format, there is no metadata file, and it's just like a single partition

        match store.open_index_file(METADATA_FILE).await {
            Ok(reader) => {
                let params = reader
                    .schema()
                    .metadata
                    .get("params")
                    .ok_or(Error::index("params not found in metadata".to_owned()))?;
                let params = serde_json::from_str::<InvertedIndexParams>(params)?;
                let partitions = reader
                    .schema()
                    .metadata
                    .get("partitions")
                    .ok_or(Error::index("partitions not found in metadata".to_owned()))?;
                let partitions: Vec<u64> = serde_json::from_str(partitions)?;
                let token_set_format = reader
                    .schema()
                    .metadata
                    .get(TOKEN_SET_FORMAT_KEY)
                    .map(|name| TokenSetFormat::from_str(name))
                    .transpose()?
                    .unwrap_or(TokenSetFormat::Arrow);
                let format_version = parse_format_version_from_metadata(&reader.schema().metadata)?;

                // Load deleted_fragments if present (optional for backward compatibility)
                let deleted_fragments = if reader.num_rows() > 0 {
                    let metadata_batch = reader.read_range(0..1, None).await?;
                    if let Some(col) = metadata_batch.column_by_name(DELETED_FRAGMENTS_COL) {
                        let arr = col.as_binary_opt::<i32>().expect_ok()?;
                        RoaringBitmap::deserialize_from(arr.value(0))?
                    } else {
                        RoaringBitmap::new()
                    }
                } else {
                    RoaringBitmap::new()
                };

                let format = token_set_format;
                let partitions = partitions.into_iter().enumerate().map(|(priority, id)| {
                    let store = store.with_io_priority(priority as u64);
                    let frag_reuse_index_clone = frag_reuse_index.clone();
                    let index_cache_for_part =
                        index_cache.with_key_prefix(format!("part-{}", id).as_str());
                    let token_set_format = format;
                    async move {
                        Result::Ok(Arc::new(
                            InvertedPartition::load(
                                store,
                                id,
                                frag_reuse_index_clone,
                                &index_cache_for_part,
                                token_set_format,
                            )
                            .await?,
                        ))
                    }
                });
                let partitions = stream::iter(partitions)
                    .buffer_unordered(store.io_parallelism())
                    .try_collect::<Vec<_>>()
                    .await?;

                let tokenizer = params.build()?;
                Ok(Arc::new(Self {
                    params,
                    store,
                    tokenizer,
                    token_set_format,
                    format_version,
                    partitions,
                    corpus_stats: Arc::new(OnceCell::new()),
                    deleted_fragments,
                }))
            }
            Err(_) => {
                // old index format
                Self::load_legacy_index(store, frag_reuse_index, index_cache).await
            }
        }
    }
}

#[async_trait]
impl Index for InvertedIndex {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        let num_tokens = self
            .partitions
            .iter()
            .map(|part| part.tokens.len())
            .sum::<usize>();
        let num_docs = self
            .partitions
            .iter()
            .map(|part| part.docs.len())
            .sum::<usize>();
        Ok(serde_json::json!({
            "params": self.params,
            "num_tokens": num_tokens,
            "num_docs": num_docs,
        }))
    }

    async fn prewarm(&self) -> Result<()> {
        self.prewarm_with_options(&FtsPrewarmOptions::default())
            .await
    }

    fn index_type(&self) -> crate::IndexType {
        crate::IndexType::Inverted
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        unimplemented!()
    }
}

/// Target on-disk size of one prewarm chunk. Keep this large enough that cloud
/// stores do not spend prewarm time on thousands of tiny range reads, but still
/// bounded so one large partition is not materialized all at once.
const PREWARM_CHUNK_TARGET_BYTES: u64 = 128 << 20;

/// Cap on token rows per chunk, bounding the built `Vec` when posting lists are tiny.
const PREWARM_MAX_CHUNK_TOKENS: usize = 256 * 1024;

/// Floor on token rows per chunk, so a partition always makes progress.
const PREWARM_MIN_CHUNK_TOKENS: usize = 1;

const V3_BLOCK_ROW_CACHE_ENV: &str = "LANCE_FTS_V3_BLOCK_ROW_CACHE";
const V3_BYPASS_POSTING_LIST_CACHE_ENV: &str = "LANCE_FTS_V3_BYPASS_POSTING_LIST_CACHE";

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn v3_block_row_cache_enabled() -> bool {
    env_flag_enabled(V3_BLOCK_ROW_CACHE_ENV)
}

fn v3_bypass_posting_list_cache_enabled() -> bool {
    env_flag_enabled(V3_BYPASS_POSTING_LIST_CACHE_ENV)
}

/// Token rows per chunk: byte target / average bytes-per-token, clamped to `[MIN, MAX]`.
fn prewarm_chunk_tokens(token_count: usize, file_size_bytes: u64) -> usize {
    if token_count == 0 {
        return PREWARM_MIN_CHUNK_TOKENS;
    }
    let bytes_per_token = (file_size_bytes / token_count as u64).max(1); // >= 1: no div-by-zero
    let by_bytes = (PREWARM_CHUNK_TARGET_BYTES / bytes_per_token) as usize;
    by_bytes.clamp(PREWARM_MIN_CHUNK_TOKENS, PREWARM_MAX_CHUNK_TOKENS)
}

/// Snap a chunk's exclusive token end back to a posting-group boundary so no group
/// straddles chunks. Returns the largest group boundary in `(tok_start, desired_end]`,
/// or the next boundary past an oversized group so it runs as one solo chunk.
fn group_aligned_chunk_end(
    starts: &[u32],
    token_count: usize,
    tok_start: usize,
    desired_end: usize,
) -> usize {
    if desired_end >= token_count {
        return token_count;
    }

    let first_after_start = starts.partition_point(|&start| start as usize <= tok_start);
    let first_after_desired = starts.partition_point(|&start| start as usize <= desired_end);
    if first_after_desired > first_after_start {
        return starts[first_after_desired - 1] as usize;
    }

    // Oversized group: extend to its end so it runs as one chunk.
    starts
        .get(first_after_start)
        .map(|&start| start as usize)
        .unwrap_or(token_count)
}

fn prewarm_chunk_ranges(
    group_starts: Option<&[u32]>,
    token_count: usize,
    chunk_tokens: usize,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut tok_start = 0usize;
    while tok_start < token_count {
        let mut tok_end = (tok_start + chunk_tokens).min(token_count);
        // `tok_start` is always a group boundary; snap `tok_end` back to one too.
        if let Some(starts) = group_starts {
            tok_end = group_aligned_chunk_end(starts, token_count, tok_start, tok_end);
        }
        ranges.push((tok_start, tok_end));
        tok_start = tok_end;
    }
    ranges
}

fn group_start_indices_for_chunk(starts: &[u32], tok_start: usize, tok_end: usize) -> Range<usize> {
    let first = starts.partition_point(|&start| (start as usize) < tok_start);
    let end = starts.partition_point(|&start| (start as usize) < tok_end);
    first..end
}

fn group_range_for_start_index(starts: &[u32], token_count: usize, group_idx: usize) -> (u32, u32) {
    let start = starts[group_idx];
    let end = starts
        .get(group_idx + 1)
        .copied()
        .unwrap_or(token_count as u32);
    (start, end)
}

impl InvertedIndex {
    pub async fn prewarm_with_options(&self, options: &FtsPrewarmOptions) -> Result<()> {
        let with_position = options.with_position;
        let chunk_concurrency = self.store.io_parallelism().max(1);
        for part in &self.partitions {
            part.inverted_list
                .prewarm_posting_lists(with_position, chunk_concurrency)
                .await?;
            // Materialize the deferred DocSet too: prewarm's contract is
            // that subsequent queries do no IO, so the per-doc row_ids /
            // num_tokens must be resident, not lazily faulted in at query
            // time. `ensure_loaded` opens, reads, and drops the reader.
            part.docs.ensure_loaded().await?;
        }
        Ok(())
    }

    /// Cache the highest-scoring fraction of blocks for the supplied query terms.
    ///
    /// `percent` is a block-count percentage in 1..=100, not a byte budget.
    /// The session cache capacity still applies. Full posting lists are never
    /// materialized, and positions are cached only when requested.
    pub async fn prewarm_v3_top_blocks_for_queries(
        &self,
        queries: &[String],
        percent: usize,
        with_position: bool,
    ) -> Result<usize> {
        if !(1..=100).contains(&percent) {
            return Err(Error::invalid_input(format!(
                "top block percent must be in [1, 100], got {percent}"
            )));
        }
        let mut tokenizer = self.tokenizer.clone();
        let mut query_terms = HashSet::new();
        for query in queries {
            let tokens = collect_query_tokens(query, &mut tokenizer);
            for token_idx in 0..tokens.len() {
                query_terms.insert(tokens.get_token(token_idx).to_owned());
            }
        }

        let mut warmed_blocks = 0usize;
        for part in &self.partitions {
            let mut token_ids = query_terms
                .iter()
                .filter_map(|term| part.map(term))
                .collect::<Vec<_>>();
            token_ids.sort_unstable();
            token_ids.dedup();
            warmed_blocks += part
                .inverted_list
                .prewarm_v3_top_blocks(&token_ids, percent, with_position)
                .await?;
            part.docs.ensure_loaded().await?;
        }
        Ok(warmed_blocks)
    }
    /// Search docs match the input text.
    async fn do_search(&self, text: &str) -> Result<RecordBatch> {
        let params = FtsSearchParams::new();
        let mut tokenizer = self.tokenizer.clone();
        let tokens = collect_query_tokens(text, &mut tokenizer);

        let (doc_ids, _) = self
            .bm25_search(
                Arc::new(tokens),
                params.into(),
                Operator::And,
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
                None,
            )
            .boxed()
            .await?;

        Ok(RecordBatch::try_new(
            ROW_ID_SCHEMA.clone(),
            vec![Arc::new(UInt64Array::from(doc_ids))],
        )?)
    }
}

#[async_trait]
impl ScalarIndex for InvertedIndex {
    // return the row ids of the documents that contain the query
    #[instrument(level = "debug", skip_all)]
    async fn search(
        &self,
        query: &dyn AnyQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<TokenQuery>().unwrap();

        match query {
            TokenQuery::TokensContains(text) => {
                let records = self.do_search(text).await?;
                let row_ids = records
                    .column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                let row_ids = row_ids.iter().flatten().collect_vec();
                Ok(SearchResult::at_most(RowAddrTreeMap::from_iter(row_ids)))
            }
        }
    }

    fn can_remap(&self) -> bool {
        true
    }

    async fn remap(
        &self,
        mapping: &RowAddrRemap,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let files = self
            .to_builder()
            .remap(mapping, self.store.clone(), dest_store)
            .await?;

        let details = pbold::InvertedIndexDetails::try_from(&self.params)?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&details).unwrap(),
            index_version: self.index_version(),
            files,
        })
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        old_data_filter: Option<crate::scalar::OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        let files = self
            .to_builder()
            .update(new_data, dest_store, old_data_filter)
            .await?;

        let details = pbold::InvertedIndexDetails::try_from(&self.params)?;

        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&details).unwrap(),
            index_version: self.index_version(),
            files,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        let criteria = TrainingCriteria::new(TrainingOrdering::None).with_row_id();
        if self.is_legacy() {
            UpdateCriteria::requires_old_data(criteria)
        } else {
            UpdateCriteria::only_new_data(criteria)
        }
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        let mut params = self.params.clone();
        if params.base_tokenizer.is_empty() {
            // Empty tokenizer metadata only appears in legacy simple-tokenizer indexes.
            params.base_tokenizer = "simple".to_string();
        }
        params = params.format_version(self.format_version());

        let params_json = params.to_training_json()?.to_string();

        Ok(ScalarIndexParams {
            index_type: BuiltinIndexType::Inverted.as_str().to_string(),
            params: Some(params_json),
        })
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct InvertedPartition {
    // 0 for legacy format
    id: u64,
    store: Arc<dyn IndexStore>,
    pub(crate) tokens: TokenSet,
    pub(crate) inverted_list: Arc<PostingListReader>,
    /// Per-doc row_id + num_tokens. Wrapped in `LazyDocSet` so partitions
    /// that don't contribute hits to a query never pay the full-array
    /// download. Scoring paths call `ensure_loaded` before walking wand.
    pub(crate) docs: Arc<LazyDocSet>,
    token_set_format: TokenSetFormat,
}

impl InvertedPartition {
    /// Check if this partition belongs to the specified fragment.
    ///
    /// This method encapsulates the bit manipulation logic for fragment filtering
    /// in distributed indexing scenarios.
    ///
    /// # Arguments
    /// * `fragment_mask` - A mask with fragment_id in high 32 bits
    ///
    /// # Returns
    /// * `true` if the partition belongs to the fragment, `false` otherwise
    pub fn belongs_to_fragment(&self, fragment_mask: u64) -> bool {
        (self.id() & fragment_mask) == fragment_mask
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn store(&self) -> &dyn IndexStore {
        self.store.as_ref()
    }

    pub fn is_legacy(&self) -> bool {
        self.inverted_list.is_legacy_layout()
    }

    pub async fn load(
        store: Arc<dyn IndexStore>,
        id: u64,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        index_cache: &LanceCache,
        token_set_format: TokenSetFormat,
    ) -> Result<Self> {
        let token_file = store.open_index_file(&token_file_path(id)).await?;
        let tokens = TokenSet::load(token_file, token_set_format).await?;
        let invert_list_file = store.open_index_file(&posting_file_path(id)).await?;
        let format_version =
            parse_format_version_from_metadata(&invert_list_file.schema().metadata)?;
        let inverted_list = if format_version.uses_block_row_layout() {
            let block_file = store.open_index_file(&posting_blocks_file_path(id)).await?;
            let position_file = if invert_list_file
                .schema()
                .metadata
                .contains_key(POSITIONS_LAYOUT_KEY)
            {
                Some(
                    store
                        .open_index_file(&posting_positions_file_path(id))
                        .await?,
                )
            } else {
                None
            };
            let coarse_skip_file = store
                .open_index_file(&posting_coarse_skip_file_path(id))
                .await?;
            PostingListReader::try_new_v3(
                invert_list_file,
                block_file,
                position_file,
                coarse_skip_file,
                index_cache,
            )
            .await?
        } else {
            PostingListReader::try_new(invert_list_file, index_cache).await?
        };
        // Defer the per-doc row_id/num_tokens read. Construction reads only
        // the doc count (one footer read) and then drops the reader; the bulk
        // load happens on first scoring use, re-opening the docs file on
        // demand, and partitions that never score skip it entirely. Storing
        // the store + path instead of an open reader keeps a cached partition
        // from pinning a docs-file handle for its whole lifetime.
        let docs_path = doc_file_path(id);
        let num_docs = store.open_index_file(&docs_path).await?.num_rows();
        let docs = Arc::new(LazyDocSet::new(
            store.clone(),
            docs_path,
            num_docs,
            false,
            frag_reuse_index,
        ));

        Ok(Self {
            id,
            store,
            tokens,
            inverted_list: Arc::new(inverted_list),
            docs,
            token_set_format,
        })
    }

    fn map(&self, token: &str) -> Option<u32> {
        self.tokens.get(token)
    }

    pub fn expand_fuzzy(&self, tokens: &Tokens, params: &FtsSearchParams) -> Result<Tokens> {
        let mut new_tokens = Vec::with_capacity(min(tokens.len(), params.max_expansions));
        let mut new_positions = Vec::with_capacity(new_tokens.capacity());
        let mut seen = HashSet::new();
        for token_idx in 0..tokens.len() {
            let remaining = params.max_expansions.saturating_sub(new_tokens.len());
            if remaining == 0 {
                break;
            }
            let token = tokens.get_token(token_idx);
            let position = tokens.position(token_idx);
            let base_prefix_len = tokens.token_type().prefix_len(token) as u32;
            let mut candidates = BTreeSet::new();
            self.collect_fuzzy_candidates(
                token,
                base_prefix_len,
                params,
                remaining,
                &mut candidates,
            )?;
            for candidate in candidates {
                if new_tokens.len() >= params.max_expansions {
                    break;
                }
                if seen.insert((candidate.clone(), position)) {
                    new_tokens.push(candidate);
                    new_positions.push(position);
                }
            }
        }
        Ok(Tokens::with_positions(
            new_tokens,
            new_positions,
            tokens.token_type().clone(),
        ))
    }

    /// Collect up to `limit` fuzzy candidates for one query token from this
    /// partition's token FST, in key (lexicographic) order. Callers merge
    /// candidates across partitions and apply the query-wide
    /// `max_expansions` budget; truncating each partition at `limit` is
    /// lossless for that selection because any term among the merged
    /// lexicographically-smallest `limit` is also among its own partition's
    /// smallest `limit`.
    fn collect_fuzzy_candidates(
        &self,
        token: &str,
        base_prefix_len: u32,
        params: &FtsSearchParams,
        limit: usize,
        candidates: &mut BTreeSet<String>,
    ) -> Result<()> {
        let fuzziness = match params.fuzziness {
            Some(fuzziness) => fuzziness,
            None => MatchQuery::auto_fuzziness(token),
        };
        let lev = fst::automaton::Levenshtein::new(token, fuzziness)
            .map_err(|e| Error::index(format!("failed to construct the fuzzy query: {}", e)))?;

        if let TokenMap::Fst(ref map) = self.tokens.tokens {
            let mut expanded = Vec::new();
            match base_prefix_len + params.prefix_length {
                0 => take_fst_keys(map.search(lev), &mut expanded, limit),
                prefix_length => {
                    let prefix = &token[..min(prefix_length as usize, token.len())];
                    let prefix = fst::automaton::Str::new(prefix).starts_with();
                    take_fst_keys(map.search(lev.intersection(prefix)), &mut expanded, limit)
                }
            }
            candidates.extend(expanded);
            Ok(())
        } else {
            Err(Error::index(
                "tokens is not fst, which is not expected".to_owned(),
            ))
        }
    }

    fn union_plain_posting_lists(postings: Vec<PostingList>) -> Result<PostingList> {
        let mut freqs_by_row_id = BTreeMap::new();
        for posting in postings {
            for (row_id, freq, _) in posting.iter() {
                let entry = freqs_by_row_id.entry(row_id).or_insert(0u32);
                *entry = entry.checked_add(freq).ok_or_else(|| {
                    Error::index(format!("posting frequency overflow for row id {}", row_id))
                })?;
            }
        }
        let mut row_ids = Vec::with_capacity(freqs_by_row_id.len());
        let mut frequencies = Vec::with_capacity(freqs_by_row_id.len());
        for (row_id, freq) in freqs_by_row_id {
            row_ids.push(row_id);
            frequencies.push(freq as f32);
        }
        Ok(PostingList::Plain(PlainPostingList::new(
            ScalarBuffer::from(row_ids),
            ScalarBuffer::from(frequencies),
            None,
            None,
        )))
    }

    fn union_compressed_posting_lists(
        postings: Vec<PostingList>,
        docs: &DocSet,
    ) -> Result<PostingList> {
        let mut freqs_by_doc_id = BTreeMap::new();
        for posting in postings {
            for (doc_id, freq, _) in posting.iter() {
                let doc_id = u32::try_from(doc_id).map_err(|_| {
                    Error::index(format!(
                        "compressed posting doc id {} exceeds u32::MAX",
                        doc_id
                    ))
                })?;
                let entry = freqs_by_doc_id.entry(doc_id).or_insert(0u32);
                *entry = entry.checked_add(freq).ok_or_else(|| {
                    Error::index(format!("posting frequency overflow for doc id {}", doc_id))
                })?;
            }
        }
        if freqs_by_doc_id.is_empty() {
            return Ok(PostingList::Plain(PlainPostingList::new(
                ScalarBuffer::from(Vec::<u64>::new()),
                ScalarBuffer::from(Vec::<f32>::new()),
                None,
                None,
            )));
        }

        let mut builder = PostingListBuilder::new(false);
        let mut doc_ids = Vec::with_capacity(freqs_by_doc_id.len());
        let mut frequencies = Vec::with_capacity(freqs_by_doc_id.len());
        for (doc_id, freq) in freqs_by_doc_id {
            builder.add(doc_id, PositionRecorder::Count(freq));
            doc_ids.push(doc_id);
            frequencies.push(freq);
        }
        let block_max_scores = docs.calculate_block_max_scores(doc_ids.iter(), frequencies.iter());
        let batch = builder.to_batch(block_max_scores)?;
        let max_score = batch[MAX_SCORE_COL].as_primitive::<Float32Type>().value(0);
        let length = batch[LENGTH_COL].as_primitive::<UInt32Type>().value(0);
        PostingList::from_batch(&batch, Some(max_score), Some(length))
    }

    fn union_posting_lists(postings: Vec<PostingList>, docs: &DocSet) -> Result<PostingList> {
        let has_plain = postings
            .iter()
            .any(|posting| matches!(posting, PostingList::Plain(_)));
        let has_compressed = postings
            .iter()
            .any(|posting| matches!(posting, PostingList::Compressed(_)));
        match (has_plain, has_compressed) {
            (true, true) => Err(Error::index(
                "cannot union mixed plain and compressed posting lists".to_owned(),
            )),
            (true, false) => Self::union_plain_posting_lists(postings),
            (false, true) => Self::union_compressed_posting_lists(postings, docs),
            (false, false) => Ok(PostingList::Plain(PlainPostingList::new(
                ScalarBuffer::from(Vec::<u64>::new()),
                ScalarBuffer::from(Vec::<f32>::new()),
                None,
                None,
            ))),
        }
    }

    // search the documents that contain the query
    // return the doc info and the doc length
    // ref: https://en.wikipedia.org/wiki/Okapi_BM25
    #[instrument(level = "debug", skip_all)]
    async fn load_posting_lists(
        &self,
        tokens: &Tokens,
        params: &FtsSearchParams,
        operator: Operator,
        metrics: &dyn MetricsCollector,
    ) -> Result<LoadedPostings> {
        let is_fuzzy = matches!(params.fuzziness, Some(n) if n != 0);
        let is_phrase_query = params.phrase_slop.is_some();
        let is_and_query = operator == Operator::And;
        let required_positions = (is_and_query || is_phrase_query).then(|| {
            (0..tokens.len())
                .map(|index| tokens.position(index))
                .collect::<HashSet<_>>()
        });
        // Fuzzy expansion already ran once at the index level (see
        // `InvertedIndex::bm25_search`) under the global `max_expansions`
        // budget; the incoming tokens are final and `is_fuzzy` only drives
        // the grouped dedup/scoring semantics below.
        let tokens = tokens.clone();
        let token_positions = (0..tokens.len())
            .map(|index| tokens.position(index))
            .collect::<Vec<_>>();
        let mut token_ids = Vec::with_capacity(tokens.len());
        let mut matched_positions = required_positions.as_ref().map(|_| HashSet::new());
        for (index, token) in tokens.into_iter().enumerate() {
            let token_id = self.map(&token);
            if let Some(token_id) = token_id {
                let position = token_positions[index];
                if let Some(matched_positions) = matched_positions.as_mut() {
                    matched_positions.insert(position);
                }
                token_ids.push((token_id, token, position));
            } else if is_phrase_query || is_and_query {
                // if the token is not found, we can't do phrase or AND query
                return Ok(LoadedPostings::empty());
            }
        }
        if token_ids.is_empty() {
            return Ok(LoadedPostings::empty());
        }
        if let Some(required_positions) = required_positions.as_ref()
            && let Some(matched_positions) = matched_positions.as_ref()
            && !required_positions.is_subset(matched_positions)
        {
            return Ok(LoadedPostings::empty());
        }

        let is_fuzzy_and_query = is_fuzzy && is_and_query && !is_phrase_query;
        if !is_phrase_query {
            if is_fuzzy_and_query {
                token_ids.sort_unstable_by_key(|(token_id, _, position)| (*position, *token_id));
                token_ids.dedup_by(|lhs, rhs| lhs.0 == rhs.0 && lhs.2 == rhs.2);
            } else {
                token_ids.sort_unstable_by_key(|(token_id, _, _)| *token_id);
                token_ids.dedup_by_key(|(token_id, _, _)| *token_id);
            }
        }

        let num_docs = self.docs.len();
        let loaded_postings = stream::iter(token_ids)
            .map(|(token_id, token, position)| async move {
                let posting = self
                    .inverted_list
                    .posting_list(token_id, is_phrase_query, metrics)
                    .await?;

                Result::Ok((token_id, token, position, posting))
            })
            .buffered(self.store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await?;

        if (is_and_query || is_phrase_query)
            && !is_fuzzy_and_query
            && loaded_postings
                .iter()
                .any(|(_, _, _, posting)| posting.is_empty())
        {
            return Ok(LoadedPostings::empty());
        }

        if !is_fuzzy_and_query {
            return Ok(LoadedPostings {
                postings: loaded_postings
                    .into_iter()
                    .map(|(token_id, token, position, posting)| {
                        let query_weight = idf(posting.len(), num_docs);
                        PostingIterator::with_query_weight(
                            token,
                            token_id,
                            position,
                            query_weight,
                            posting,
                            num_docs,
                        )
                    })
                    .collect(),
                grouped_expansions: Vec::new(),
            });
        }

        let needs_union = loaded_postings
            .windows(2)
            .any(|window| window[0].2 == window[1].2);
        let docs_for_union = if needs_union {
            Some(self.docs.ensure_num_tokens_loaded().await?)
        } else {
            None
        };

        // WAND's AND mode treats every iterator as required, so expansions from
        // one original query position must be merged before scoring.
        let mut grouped_postings = Vec::new();
        let mut grouped_expansions = Vec::new();
        let mut iter = loaded_postings.into_iter().peekable();
        while let Some((token_id, token, position, posting)) = iter.next() {
            let mut group = vec![(token_id, token, posting)];
            while matches!(iter.peek(), Some((_, _, next_position, _)) if *next_position == position)
            {
                let (token_id, token, _, posting) = iter.next().expect("peeked item must exist");
                group.push((token_id, token, posting));
            }

            let (token_id, token, posting) = if group.len() == 1 {
                group.pop().expect("single-item group must exist")
            } else {
                let token_id = group[0].0;
                let token = group[0].1.clone();
                grouped_expansions.push(GroupedExpansionTerms {
                    position,
                    terms: group
                        .iter()
                        .map(|(_, token, posting)| ExpansionTermFreqs::new(token.clone(), posting))
                        .collect(),
                });
                let postings = group
                    .into_iter()
                    .map(|(_, _, posting)| posting)
                    .collect::<Vec<_>>();
                let posting = Self::union_posting_lists(
                    postings,
                    docs_for_union
                        .as_deref()
                        .expect("union docs must be loaded for grouped fuzzy AND"),
                )?;
                (token_id, token, posting)
            };
            if posting.is_empty() {
                return Ok(LoadedPostings::empty());
            }

            let query_weight = idf(posting.len(), num_docs);
            grouped_postings.push(PostingIterator::with_query_weight(
                token,
                token_id,
                position,
                query_weight,
                posting,
                num_docs,
            ));
        }

        Ok(LoadedPostings {
            postings: grouped_postings,
            grouped_expansions,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_bm25_search_v3_blocks(
        &self,
        tokens: &Tokens,
        params: &FtsSearchParams,
        operator: Operator,
        mask: Arc<RowAddrMask>,
        metrics: Arc<dyn MetricsCollector>,
        shared_threshold: Arc<AtomicU32>,
    ) -> Result<Option<PartitionCandidates>> {
        if !self.inverted_list.is_v3_block_row_layout()
            || matches!(params.fuzziness, Some(n) if n != 0)
        {
            return Ok(None);
        }

        let token_ids = self.v3_query_token_ids(tokens, params, operator)?;
        if token_ids.is_empty() {
            return Ok(Some(PartitionCandidates::empty()));
        }
        let cache_token_ids = token_ids
            .iter()
            .map(|(token_id, _, _)| *token_id)
            .collect::<Vec<_>>();
        if self
            .inverted_list
            .v3_posting_lists_cached(&cache_token_ids)
            .await
        {
            return Ok(None);
        }

        let term_plans = self.v3_term_plans(token_ids, metrics.as_ref()).await?;
        if term_plans.is_empty() {
            return Ok(Some(PartitionCandidates::empty()));
        }

        let max_position = term_plans
            .iter()
            .map(|term| term.term_index as usize)
            .max()
            .unwrap_or_default();
        let mut tokens_by_position = vec![String::new(); max_position + 1];
        for term in &term_plans {
            tokens_by_position[term.term_index as usize] = term.token.clone();
        }

        let docs_for_wand = self.docs.docs_for_wand(mask.as_ref()).await?;
        let candidates = self
            .bm25_search_v3_blocks(
                docs_for_wand.as_ref(),
                params,
                operator,
                mask,
                term_plans,
                metrics.as_ref(),
                shared_threshold,
            )
            .await?;
        Ok(Some(PartitionCandidates {
            tokens_by_position,
            grouped_expansions: Vec::new(),
            candidates,
        }))
    }

    fn v3_query_token_ids(
        &self,
        tokens: &Tokens,
        params: &FtsSearchParams,
        operator: Operator,
    ) -> Result<Vec<(u32, String, u32)>> {
        let is_phrase_query = params.phrase_slop.is_some();
        let is_and_query = operator == Operator::And;
        let required_positions = (is_and_query || is_phrase_query).then(|| {
            (0..tokens.len())
                .map(|index| tokens.position(index))
                .collect::<HashSet<_>>()
        });

        let token_positions = (0..tokens.len())
            .map(|index| tokens.position(index))
            .collect::<Vec<_>>();
        let mut token_ids = Vec::with_capacity(tokens.len());
        let mut matched_positions = required_positions.as_ref().map(|_| HashSet::new());
        for (index, token) in tokens.into_iter().enumerate() {
            if let Some(token_id) = self.map(token) {
                let position = token_positions[index];
                if let Some(matched_positions) = matched_positions.as_mut() {
                    matched_positions.insert(position);
                }
                token_ids.push((token_id, token.clone(), position));
            } else if is_phrase_query || is_and_query {
                return Ok(Vec::new());
            }
        }
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(required_positions) = required_positions.as_ref()
            && let Some(matched_positions) = matched_positions.as_ref()
            && !required_positions.is_subset(matched_positions)
        {
            return Ok(Vec::new());
        }

        if !is_phrase_query {
            token_ids.sort_unstable_by_key(|(token_id, _, _)| *token_id);
            token_ids.dedup_by_key(|(token_id, _, _)| *token_id);
        }

        Ok(token_ids)
    }

    async fn v3_term_plans(
        &self,
        token_ids: Vec<(u32, String, u32)>,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<V3TermPlan>> {
        stream::iter(token_ids.into_iter().enumerate())
            .map(|(term_idx, (token_id, token, position))| async move {
                let posting_len = self.inverted_list.posting_len_for_token(token_id).await? as u32;
                let blocks = self
                    .inverted_list
                    .v3_block_metadata(token_id, term_idx, metrics)
                    .await?;
                Ok(V3TermPlan {
                    token_id,
                    token,
                    position,
                    term_index: position,
                    posting_len,
                    blocks,
                })
            })
            .buffered(16)
            .try_collect()
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn bm25_search_v3_blocks(
        &self,
        docs: &DocSet,
        params: &FtsSearchParams,
        operator: Operator,
        mask: Arc<RowAddrMask>,
        term_plans: Vec<V3TermPlan>,
        metrics: &dyn MetricsCollector,
        shared_threshold: Arc<AtomicU32>,
    ) -> Result<Vec<DocCandidate>> {
        let limit = params.limit.unwrap_or(usize::MAX);
        if limit == 0 || term_plans.is_empty() {
            return Ok(Vec::new());
        }

        let phrase_slop = params.phrase_slop.map(|slop| slop as i32);
        let candidate_operator = if phrase_slop.is_some() {
            Operator::And
        } else {
            operator
        };
        let mut windows = block_wand::search_windows(&term_plans, candidate_operator);
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        windows.sort_unstable_by(|left, right| {
            right
                .upper_bound
                .total_cmp(&left.upper_bound)
                .then_with(|| left.first_doc_id.cmp(&right.first_doc_id))
                .then_with(|| left.last_doc_id.cmp(&right.last_doc_id))
        });

        let scorer = IndexBM25Scorer::new(std::iter::once(self));
        let docs_has_row_ids = docs.has_row_ids();
        let mut candidates = BinaryHeap::with_capacity(std::cmp::min(limit, BLOCK_SIZE * 10));
        let mut threshold = f32::NEG_INFINITY;
        let mut decoded_blocks = HashMap::<usize, V3DecodedPostingBlock>::new();
        let mut decoded_positions = HashMap::<usize, V3DecodedPositionBlock>::new();
        let mut visited_blocks = HashSet::new();
        let mut num_comparisons = 0usize;
        let mut windows = windows.into_iter().peekable();
        while windows.peek().is_some() {
            threshold =
                threshold.max(self.v3_shared_threshold(params.wand_factor, &shared_threshold));
            // Establish a threshold before widening I/O. Later batches amortize
            // remote request latency without materializing whole posting lists.
            let batch_size = if candidates.len() < limit { 1 } else { 16 };
            let mut batch_windows = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let Some(window) = windows.peek() else { break };
                if candidates.len() >= limit && window.upper_bound < threshold {
                    break;
                }
                if let Some(window) = windows.next() {
                    batch_windows.push(window);
                }
            }
            if batch_windows.is_empty() {
                break;
            }
            let block_rows = batch_windows
                .iter()
                .flat_map(|window| {
                    term_plans.iter().flat_map(move |term| {
                        term.overlapping_blocks(window.first_doc_id, window.last_doc_id)
                    })
                })
                .filter(|block| !decoded_blocks.contains_key(&block.block_row))
                .map(|block| block.block_row)
                .collect::<Vec<_>>();
            visited_blocks.extend(block_rows.iter().copied());
            let block_batches = self
                .inverted_list
                .v3_payload_block_batches(&block_rows, metrics)
                .await?;
            for window in batch_windows {
                if candidates.len() >= limit && window.upper_bound < threshold {
                    continue;
                }
                let window_blocks = term_plans
                    .iter()
                    .flat_map(|term| {
                        term.overlapping_blocks(window.first_doc_id, window.last_doc_id)
                    })
                    .collect::<Vec<_>>();
                for block in &window_blocks {
                    if decoded_blocks.contains_key(&block.block_row) {
                        continue;
                    }
                    let batch = block_batches.get(&block.block_row).ok_or_else(|| {
                        Error::internal(format!("missing V3 payload block row {}", block.block_row))
                    })?;
                    let decoded = self.inverted_list.decode_v3_payload_block(
                        term_plans[block.term_idx].token_id,
                        block.block_idx,
                        batch,
                    )?;
                    decoded_blocks.insert(block.block_row, decoded);
                }

                let mut docs_in_window = HashMap::<u32, V3DocAccumulator>::new();
                for block in &window_blocks {
                    let decoded = decoded_blocks.get(&block.block_row).ok_or_else(|| {
                        Error::internal(format!("missing decoded V3 block row {}", block.block_row))
                    })?;
                    for (doc_offset, (&doc_id, &freq)) in
                        decoded.doc_ids.iter().zip(decoded.freqs.iter()).enumerate()
                    {
                        if doc_id < window.first_doc_id || doc_id > window.last_doc_id {
                            continue;
                        }
                        docs_in_window
                            .entry(doc_id)
                            .or_insert_with(|| V3DocAccumulator::new(term_plans.len()))
                            .add(block.term_idx, block.block_row, doc_offset, freq);
                    }
                }

                let mut docs_in_window = docs_in_window.into_iter().collect::<Vec<_>>();
                docs_in_window.sort_unstable_by_key(|(doc_id, _)| *doc_id);
                for (doc_id, acc) in docs_in_window {
                    if !acc.matches_operator(candidate_operator) {
                        continue;
                    }
                    num_comparisons += 1;

                    let addr = if docs_has_row_ids {
                        let row_id = docs.row_id(doc_id);
                        if row_id == RowAddress::TOMBSTONE_ROW || !mask.selected(row_id) {
                            continue;
                        }
                        CandidateAddr::RowId(row_id)
                    } else {
                        CandidateAddr::Pending(doc_id)
                    };
                    let doc_length = docs.num_tokens(doc_id);
                    let score = self.v3_score_candidate(&acc, &term_plans, doc_length, &scorer);
                    if candidates.len() >= limit && score < threshold {
                        continue;
                    }
                    if let Some(slop) = phrase_slop
                        && !self
                            .v3_candidate_matches_phrase(
                                &term_plans,
                                &acc,
                                &decoded_blocks,
                                &mut decoded_positions,
                                metrics,
                                slop,
                            )
                            .await?
                    {
                        continue;
                    }

                    let row_id_slot = match &addr {
                        CandidateAddr::RowId(row_id) => *row_id,
                        CandidateAddr::Pending(doc_id) => *doc_id as u64,
                    };
                    let term_freqs = acc.term_freqs(&term_plans);
                    if candidates.len() < limit {
                        candidates.push(Reverse((
                            ScoredDoc::new(row_id_slot, score),
                            term_freqs,
                            doc_length,
                            doc_id,
                        )));
                        if candidates.len() == limit {
                            threshold = self.v3_update_threshold(
                                candidates.peek().unwrap().0.0.score.0,
                                params.wand_factor,
                                &shared_threshold,
                            );
                        }
                    } else if score > candidates.peek().unwrap().0.0.score.0 {
                        candidates.pop();
                        candidates.push(Reverse((
                            ScoredDoc::new(row_id_slot, score),
                            term_freqs,
                            doc_length,
                            doc_id,
                        )));
                        threshold = self.v3_update_threshold(
                            candidates.peek().unwrap().0.0.score.0,
                            params.wand_factor,
                            &shared_threshold,
                        );
                    }
                }
            }
        }
        metrics.record_comparisons(num_comparisons);
        let all_rows = term_plans
            .iter()
            .flat_map(|term| term.blocks.iter().map(|block| block.block_row))
            .collect::<HashSet<_>>();
        metrics.record_fts_blocks_pruned(all_rows.len().saturating_sub(visited_blocks.len()));

        Ok(candidates
            .into_iter()
            .map(
                |Reverse((doc, freqs, doc_length, posting_doc_id))| DocCandidate {
                    addr: if docs_has_row_ids {
                        CandidateAddr::RowId(doc.row_id)
                    } else {
                        CandidateAddr::Pending(doc.row_id as u32)
                    },
                    posting_doc_id: posting_doc_id as u64,
                    freqs,
                    doc_length,
                },
            )
            .collect())
    }

    fn v3_update_threshold(
        &self,
        local_kth: f32,
        wand_factor: f32,
        shared_threshold: &AtomicU32,
    ) -> f32 {
        atomic_store_max_f32(shared_threshold, local_kth);
        local_kth.max(f32::from_bits(shared_threshold.load(Ordering::Relaxed))) * wand_factor
    }

    fn v3_shared_threshold(&self, wand_factor: f32, shared_threshold: &AtomicU32) -> f32 {
        f32::from_bits(shared_threshold.load(Ordering::Relaxed)) * wand_factor
    }

    fn v3_score_candidate(
        &self,
        acc: &V3DocAccumulator,
        term_plans: &[V3TermPlan],
        doc_length: u32,
        scorer: &IndexBM25Scorer,
    ) -> f32 {
        acc.freqs
            .iter()
            .enumerate()
            .filter_map(|(term_idx, freq)| freq.map(|freq| (term_idx, freq)))
            .map(|(term_idx, freq)| {
                idf(term_plans[term_idx].posting_len as usize, self.docs.len())
                    * scorer.doc_weight(freq, doc_length)
            })
            .sum()
    }

    async fn v3_candidate_matches_phrase(
        &self,
        term_plans: &[V3TermPlan],
        acc: &V3DocAccumulator,
        decoded_blocks: &HashMap<usize, V3DecodedPostingBlock>,
        decoded_positions: &mut HashMap<usize, V3DecodedPositionBlock>,
        metrics: &dyn MetricsCollector,
        slop: i32,
    ) -> Result<bool> {
        let mut rows = Vec::with_capacity(term_plans.len());
        for term_idx in 0..term_plans.len() {
            rows.push(
                acc.block_rows[term_idx].ok_or_else(|| {
                    Error::internal("phrase candidate is missing a term block row")
                })?,
            );
        }

        let missing_rows = rows
            .iter()
            .filter(|row| !decoded_positions.contains_key(row))
            .copied()
            .collect::<Vec<_>>();
        if !missing_rows.is_empty() {
            let payloads = self
                .inverted_list
                .v3_position_block_payloads(&missing_rows, metrics)
                .await?;
            let codec = match self.inverted_list.positions_layout {
                PositionsLayout::SharedStream(codec) => codec,
                PositionsLayout::None | PositionsLayout::LegacyPerDoc => {
                    return Err(Error::invalid_input(
                        "position is not found but required for phrase queries, try recreating the index with position".to_owned(),
                    ));
                }
            };
            for row in missing_rows {
                let posting_block = decoded_blocks.get(&row).ok_or_else(|| {
                    Error::internal(format!(
                        "V3 phrase candidate has positions for block row {row} but no decoded payload"
                    ))
                })?;
                let payload = payloads.get(&row).ok_or_else(|| {
                    Error::internal(format!("missing V3 position payload block row {row}"))
                })?;
                let mut values = Vec::new();
                decode_position_stream_block(payload, &posting_block.freqs, codec, &mut values)?;
                let mut offsets = Vec::with_capacity(posting_block.freqs.len() + 1);
                offsets.push(0);
                for freq in &posting_block.freqs {
                    offsets.push(offsets.last().copied().unwrap_or(0) + *freq as usize);
                }
                decoded_positions.insert(row, V3DecodedPositionBlock { values, offsets });
            }
        }

        let mut term_positions = Vec::with_capacity(term_plans.len());
        for (term_idx, term) in term_plans.iter().enumerate() {
            let row = acc.block_rows[term_idx]
                .ok_or_else(|| Error::internal("phrase candidate is missing a term block row"))?;
            let doc_offset = acc.doc_offsets[term_idx]
                .ok_or_else(|| Error::internal("phrase candidate is missing a term doc offset"))?;
            let position_block = decoded_positions.get(&row).ok_or_else(|| {
                Error::internal(format!("missing decoded V3 positions for block row {row}"))
            })?;
            let start = position_block.offsets[doc_offset];
            let end = position_block.offsets[doc_offset + 1];
            term_positions.push(V3TermPositions {
                position_in_query: term.position as i32,
                positions: position_block.values[start..end].to_vec(),
            });
        }
        Ok(v3_phrase_positions_match(term_positions, slop))
    }

    #[instrument(level = "debug", skip_all)]
    // Deferred-DocSet adds the `docs` param (caller materializes it) on top of
    // the cross-partition `shared_threshold`, tipping this hot-path search fn
    // one over the limit. Bundling args isn't worth the churn here.
    #[allow(clippy::too_many_arguments)]
    pub fn bm25_search(
        &self,
        docs: &DocSet,
        params: &FtsSearchParams,
        operator: Operator,
        mask: Arc<RowAddrMask>,
        postings: Vec<PostingIterator>,
        metrics: &dyn MetricsCollector,
        shared_threshold: Arc<AtomicU32>,
    ) -> Result<Vec<DocCandidate>> {
        if postings.is_empty() {
            return Ok(Vec::new());
        }

        // Caller selects the DocSet shape via `LazyDocSet::docs_for_wand`
        // and passes it in here; wand uses `docs.has_row_ids()` to
        // handle the num_tokens-only case.
        let scorer = IndexBM25Scorer::new(std::iter::once(self));
        let mut wand = Wand::new(operator, postings.into_iter(), docs, scorer)
            .with_shared_threshold(shared_threshold);
        let hits = wand.search(params, mask, metrics)?;
        Ok(hits)
    }

    pub async fn into_builder(self) -> Result<InnerBuilder> {
        let format_version = if matches!(self.inverted_list.metadata, PostingMetadata::V3 { .. }) {
            InvertedListFormatVersion::V3
        } else {
            InvertedListFormatVersion::from_posting_tail_codec(
                self.inverted_list.posting_tail_codec(),
            )
        };
        let mut builder = InnerBuilder::new_with_format_version(
            self.id,
            self.inverted_list.has_positions(),
            self.token_set_format,
            format_version,
        );
        builder.tokens = self.tokens.into_mutable();
        // into_builder rewrites every doc, so materialize the full
        // DocSet now and clone it out of the Arc.
        let docs_arc = self.docs.ensure_loaded().await?;
        builder.docs = (*docs_arc).clone();

        builder
            .posting_lists
            .reserve_exact(self.inverted_list.len());
        for posting_list in self
            .inverted_list
            .read_all(self.inverted_list.has_positions())
            .await?
        {
            let posting_list = posting_list?;
            builder
                .posting_lists
                .push(posting_list.into_builder(&builder.docs));
        }
        Ok(builder)
    }
}

// at indexing, we use HashMap because we need it to be mutable,
// at searching, we use fst::Map because it's more efficient
#[derive(Debug, Clone)]
pub enum TokenMap {
    HashMap(HashMap<String, u32>),
    Fst(fst::Map<Vec<u8>>),
}

impl Default for TokenMap {
    fn default() -> Self {
        Self::HashMap(HashMap::new())
    }
}

impl DeepSizeOf for TokenMap {
    fn deep_size_of_children(&self, ctx: &mut lance_core::deepsize::Context) -> usize {
        match self {
            Self::HashMap(map) => map.deep_size_of_children(ctx),
            Self::Fst(map) => map.as_fst().size(),
        }
    }
}

impl TokenMap {
    pub fn len(&self) -> usize {
        match self {
            Self::HashMap(map) => map.len(),
            Self::Fst(map) => map.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// TokenSet is a mapping from tokens to token ids
#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct TokenSet {
    // token -> token_id
    pub(crate) tokens: TokenMap,
    pub(crate) next_id: u32,
    total_length: usize,
}

impl TokenSet {
    pub fn into_mut(self) -> Self {
        let tokens = match self.tokens {
            TokenMap::HashMap(map) => map,
            TokenMap::Fst(map) => {
                let mut new_map = HashMap::with_capacity(map.len());
                let mut stream = map.into_stream();
                while let Some((token, token_id)) = stream.next() {
                    new_map.insert(String::from_utf8_lossy(token).into_owned(), token_id as u32);
                }

                new_map
            }
        };

        Self {
            tokens: TokenMap::HashMap(tokens),
            next_id: self.next_id,
            total_length: self.total_length,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_batch(self, format: TokenSetFormat) -> Result<RecordBatch> {
        match format {
            TokenSetFormat::Arrow => self.into_arrow_batch(),
            TokenSetFormat::Fst => self.into_fst_batch(),
        }
    }

    fn into_arrow_batch(self) -> Result<RecordBatch> {
        let mut token_builder = StringBuilder::with_capacity(self.tokens.len(), self.total_length);
        let mut token_id_builder = UInt32Builder::with_capacity(self.tokens.len());

        match self.tokens {
            TokenMap::Fst(map) => {
                let mut stream = map.stream();
                while let Some((token, token_id)) = stream.next() {
                    token_builder.append_value(String::from_utf8_lossy(token));
                    token_id_builder.append_value(token_id as u32);
                }
            }
            TokenMap::HashMap(map) => {
                for (token, token_id) in map.into_iter().sorted_unstable() {
                    token_builder.append_value(token);
                    token_id_builder.append_value(token_id);
                }
            }
        }

        let token_col = token_builder.finish();
        let token_id_col = token_id_builder.finish();

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(TOKEN_COL, DataType::Utf8, false),
            arrow_schema::Field::new(TOKEN_ID_COL, DataType::UInt32, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(token_col) as ArrayRef,
                Arc::new(token_id_col) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    fn into_fst_batch(mut self) -> Result<RecordBatch> {
        let fst_map = match std::mem::take(&mut self.tokens) {
            TokenMap::Fst(map) => map,
            TokenMap::HashMap(map) => Self::build_fst_from_map(map)?,
        };
        let bytes = fst_map.into_fst().into_inner();

        let mut fst_builder = LargeBinaryBuilder::with_capacity(1, bytes.len());
        fst_builder.append_value(bytes);
        let fst_col = fst_builder.finish();

        let mut next_id_builder = UInt32Builder::with_capacity(1);
        next_id_builder.append_value(self.next_id);
        let next_id_col = next_id_builder.finish();

        let mut total_length_builder = UInt64Builder::with_capacity(1);
        total_length_builder.append_value(self.total_length as u64);
        let total_length_col = total_length_builder.finish();

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(TOKEN_FST_BYTES_COL, DataType::LargeBinary, false),
            arrow_schema::Field::new(TOKEN_NEXT_ID_COL, DataType::UInt32, false),
            arrow_schema::Field::new(TOKEN_TOTAL_LENGTH_COL, DataType::UInt64, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(fst_col) as ArrayRef,
                Arc::new(next_id_col) as ArrayRef,
                Arc::new(total_length_col) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    fn build_fst_from_map(map: HashMap<String, u32>) -> Result<fst::Map<Vec<u8>>> {
        let mut entries: Vec<_> = map.into_iter().collect();
        entries.sort_unstable_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
        let mut builder = fst::MapBuilder::memory();
        for (token, token_id) in entries {
            builder
                .insert(&token, token_id as u64)
                .map_err(|e| Error::index(format!("failed to insert token {}: {}", token, e)))?;
        }
        Ok(builder.into_map())
    }

    pub async fn load(reader: Arc<dyn IndexReader>, format: TokenSetFormat) -> Result<Self> {
        match format {
            TokenSetFormat::Arrow => Self::load_arrow(reader).await,
            TokenSetFormat::Fst => Self::load_fst(reader).await,
        }
    }

    async fn load_arrow(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let batch = reader.read_range(0..reader.num_rows(), None).await?;

        let (tokens, next_id, total_length) = spawn_blocking(move || {
            let mut next_id = 0;
            let mut total_length = 0;
            let mut tokens = fst::MapBuilder::memory();

            let token_col = batch[TOKEN_COL].as_string::<i32>();
            let token_id_col = batch[TOKEN_ID_COL].as_primitive::<datatypes::UInt32Type>();

            for (token, &token_id) in token_col.iter().zip(token_id_col.values().iter()) {
                let token =
                    token.ok_or(Error::index("found null token in token set".to_owned()))?;
                next_id = next_id.max(token_id + 1);
                total_length += token.len();
                tokens.insert(token, token_id as u64).map_err(|e| {
                    Error::index(format!("failed to insert token {}: {}", token, e))
                })?;
            }

            Ok::<_, Error>((tokens.into_map(), next_id, total_length))
        })
        .await
        .map_err(|err| Error::execution(format!("failed to spawn blocking task: {}", err)))??;

        Ok(Self {
            tokens: TokenMap::Fst(tokens),
            next_id,
            total_length,
        })
    }

    async fn load_fst(reader: Arc<dyn IndexReader>) -> Result<Self> {
        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        if batch.num_rows() == 0 {
            return Err(Error::index("token set batch is empty".to_owned()));
        }

        let fst_col = batch[TOKEN_FST_BYTES_COL].as_binary::<i64>();
        let bytes = fst_col.value(0);
        let map = fst::Map::new(bytes.to_vec())
            .map_err(|e| Error::index(format!("failed to load fst tokens: {}", e)))?;

        let total_length_col =
            batch[TOKEN_TOTAL_LENGTH_COL].as_primitive::<datatypes::UInt64Type>();

        // Token ids are dense `[0, len)`, so `next_id` must equal the token count. Recompute
        // it instead of trusting the persisted value, which writers before #7115 could leave
        // stale. Mirrors `load_arrow`.
        let next_id = map.len() as u32;

        let total_length = total_length_col
            .values()
            .first()
            .copied()
            .ok_or(Error::index(
                "token total length column is empty".to_owned(),
            ))?;

        Ok(Self {
            tokens: TokenMap::Fst(map),
            next_id,
            total_length: usize::try_from(total_length).map_err(|_| {
                Error::index(format!(
                    "token total length {} overflows usize",
                    total_length
                ))
            })?,
        })
    }

    pub fn add(&mut self, token: String) -> u32 {
        let next_id = self.next_id();
        let len = token.len();
        let token_id = match self.tokens {
            TokenMap::HashMap(ref mut map) => *map.entry(token).or_insert(next_id),
            _ => unreachable!("tokens must be HashMap while indexing"),
        };

        // add token if it doesn't exist
        if token_id == next_id {
            self.next_id += 1;
            self.total_length += len;
        }

        token_id
    }

    pub(crate) fn get_or_add(&mut self, token: &str) -> u32 {
        let next_id = self.next_id;
        match self.tokens {
            TokenMap::HashMap(ref mut map) => {
                if let Some(&token_id) = map.get(token) {
                    return token_id;
                }

                map.insert(token.to_owned(), next_id);
            }
            _ => unreachable!("tokens must be HashMap while indexing"),
        }

        self.next_id += 1;
        self.total_length += token.len();
        next_id
    }

    pub(crate) fn into_mutable(self) -> Self {
        let Self {
            tokens,
            next_id,
            total_length,
        } = self;
        match tokens {
            TokenMap::HashMap(_) => Self {
                tokens,
                next_id,
                total_length,
            },
            TokenMap::Fst(map) => {
                let mut mutable = HashMap::new();
                let mut stream = map.stream();
                while let Some((token, token_id)) = stream.next() {
                    mutable.insert(String::from_utf8_lossy(token).into_owned(), token_id as u32);
                }
                Self {
                    tokens: TokenMap::HashMap(mutable),
                    next_id,
                    total_length,
                }
            }
        }
    }

    pub fn get(&self, token: &str) -> Option<u32> {
        match self.tokens {
            TokenMap::HashMap(ref map) => map.get(token).copied(),
            TokenMap::Fst(ref map) => map.get(token).map(|id| id as u32),
        }
    }

    // the `removed_token_ids` must be sorted
    pub fn remap(&mut self, removed_token_ids: &[u32]) {
        if removed_token_ids.is_empty() {
            return;
        }

        let mut map = match std::mem::take(&mut self.tokens) {
            TokenMap::HashMap(map) => map,
            TokenMap::Fst(map) => {
                let mut new_map = HashMap::with_capacity(map.len());
                let mut stream = map.into_stream();
                while let Some((token, token_id)) = stream.next() {
                    new_map.insert(String::from_utf8_lossy(token).into_owned(), token_id as u32);
                }

                new_map
            }
        };

        let mut retained_length = 0;
        map.retain(
            |token, token_id| match removed_token_ids.binary_search(token_id) {
                Ok(_) => false,
                Err(index) => {
                    *token_id -= index as u32;
                    retained_length += token.len();
                    true
                }
            },
        );

        self.tokens = TokenMap::HashMap(map);

        // The retain above compacts the surviving token ids into a dense `[0, len)`
        // range, so `next_id` (handed to the next new token) must follow them down.
        // `total_length` likewise must drop the removed tokens' bytes; it is persisted
        // and feeds memory accounting, so a stale value drifts across remap/merge cycles.
        self.next_id = self.tokens.len() as u32;
        self.total_length = retained_length;
    }

    pub fn next_id(&self) -> u32 {
        self.next_id
    }

    pub(crate) fn memory_size(&self) -> usize {
        match &self.tokens {
            TokenMap::HashMap(map) => {
                self.total_length
                    + map.capacity()
                        * (std::mem::size_of::<String>()
                            + std::mem::size_of::<u32>()
                            + std::mem::size_of::<usize>())
            }
            TokenMap::Fst(map) => map.as_fst().size(),
        }
    }
}

pub struct PostingListReader {
    /// V1/V2: posting-list file. V3: terms file (one row per token).
    reader: Arc<dyn IndexReader>,
    block_reader: Option<Arc<dyn IndexReader>>,
    position_reader: Option<Arc<dyn IndexReader>>,
    coarse_skip_reader: Option<Arc<dyn IndexReader>>,

    /// Layout-specific metadata. V2 keeps its per-token max-score and
    /// length columns lazy so opening a partition doesn't drag O(num_tokens)
    /// bytes off cold storage when the caller only needs `df` for a few terms.
    metadata: PostingMetadata,

    has_position: bool,
    posting_tail_codec: PostingTailCodec,
    positions_layout: PositionsLayout,

    /// First row of each posting-list cache group, decoded at open from the
    /// global buffer named by [`POSTING_GROUP_OFFSETS_BUF_KEY`] (issue #7040).
    /// `None` for indexes written before grouping; those use the per-token
    /// cache path. Always present for grouped v2 indexes with `>0` rows.
    group_starts: Option<Vec<u32>>,

    index_cache: WeakLanceCache,
}

/// Per-token metadata (max_score, length) needed by the BM25 query and stats
/// paths. The legacy and v2 formats store this metadata in different
/// places, with very different cost profiles for cold-load: the variants
/// surface that asymmetry so callers can choose a per-token or bulk access
/// pattern.
enum PostingMetadata {
    /// Legacy v1: offsets and max_scores are encoded in the file's schema
    /// metadata, so they are already in memory by the time `try_new` returns.
    LegacyV1 {
        offsets: Vec<usize>,
        max_scores: Option<Vec<f32>>,
    },
    /// V2: per-token `max_score` and `length` live as columns in the
    /// posting file. The bulk vectors are filled lazily by
    /// `ensure_metadata_loaded`, and the stats path can also fetch a single
    /// token via `posting_len_for_token` without forcing the bulk load.
    V2 {
        metadata: OnceCell<LoadedPostingMetadata>,
    },
    /// V3: terms table rows are token ids. `block_starts` has a trailing
    /// sentinel derived from `block_start + num_blocks`; `lengths` /
    /// `max_scores` are indexed by token id.
    V3 {
        block_starts: Vec<u64>,
        max_scores: Vec<f32>,
        lengths: Vec<u32>,
    },
}

#[derive(Debug, Clone)]
struct LoadedPostingMetadata {
    max_scores: Vec<f32>,
    lengths: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PositionsLayout {
    None,
    LegacyPerDoc,
    SharedStream(PositionStreamCodec),
}

impl std::fmt::Debug for PostingListReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("InvertedListReader");
        match &self.metadata {
            PostingMetadata::LegacyV1 {
                offsets,
                max_scores,
            } => {
                s.field("layout", &"legacy_v1")
                    .field("offsets", offsets)
                    .field("max_scores", max_scores);
            }
            PostingMetadata::V2 { metadata } => {
                s.field("layout", &"v2")
                    .field("metadata_loaded", &metadata.initialized());
            }
            PostingMetadata::V3 {
                block_starts,
                lengths,
                ..
            } => {
                s.field("layout", &"v3")
                    .field("num_tokens", &lengths.len())
                    .field("num_block_offsets", &block_starts.len())
                    .field(
                        "num_block_rows",
                        &self.block_reader.as_ref().map(|reader| reader.num_rows()),
                    )
                    .field(
                        "num_position_rows",
                        &self
                            .position_reader
                            .as_ref()
                            .map(|reader| reader.num_rows()),
                    )
                    .field(
                        "num_coarse_skip_rows",
                        &self
                            .coarse_skip_reader
                            .as_ref()
                            .map(|reader| reader.num_rows()),
                    );
            }
        }
        s.finish()
    }
}

impl DeepSizeOf for PostingListReader {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        let metadata_size = match &self.metadata {
            PostingMetadata::LegacyV1 {
                offsets,
                max_scores,
            } => offsets.deep_size_of_children(context) + max_scores.deep_size_of_children(context),
            PostingMetadata::V2 { metadata } => metadata
                .get()
                .map(|loaded| {
                    loaded.max_scores.deep_size_of_children(context)
                        + loaded.lengths.deep_size_of_children(context)
                })
                .unwrap_or(0),
            PostingMetadata::V3 {
                block_starts,
                max_scores,
                lengths,
            } => {
                block_starts.deep_size_of_children(context)
                    + max_scores.deep_size_of_children(context)
                    + lengths.deep_size_of_children(context)
            }
        };
        metadata_size + self.group_starts.deep_size_of_children(context)
    }
}

impl PostingListReader {
    pub(crate) async fn try_new(
        reader: Arc<dyn IndexReader>,
        index_cache: &LanceCache,
    ) -> Result<Self> {
        if matches!(
            reader
                .schema()
                .metadata
                .get(POSTING_BLOCK_LAYOUT_KEY)
                .map(String::as_str),
            Some(POSTING_BLOCK_LAYOUT_BLOCK_ROW_V3)
        ) {
            return Err(Error::index(
                "V3 posting layout requires separate terms, blocks, positions, and coarse-skip readers; use PostingListReader::try_new_v3".to_owned(),
            ));
        }
        let positions_layout = if reader.schema().field(COMPRESSED_POSITION_COL).is_some() {
            PositionsLayout::SharedStream(parse_shared_position_codec(&reader.schema().metadata)?)
        } else if reader.schema().field(POSITION_COL).is_some() {
            PositionsLayout::LegacyPerDoc
        } else {
            PositionsLayout::None
        };
        let posting_tail_codec = parse_posting_tail_codec(&reader.schema().metadata)?;
        let has_position = positions_layout != PositionsLayout::None;
        let metadata = if reader.schema().field(POSTING_COL).is_none() {
            let (offsets, max_scores) = Self::load_metadata(reader.schema())?;
            PostingMetadata::LegacyV1 {
                offsets,
                max_scores,
            }
        } else {
            PostingMetadata::V2 {
                metadata: OnceCell::new(),
            }
        };

        let group_starts = Self::load_group_starts(reader.as_ref()).await?;

        Ok(Self {
            reader,
            block_reader: None,
            position_reader: None,
            coarse_skip_reader: None,
            metadata,
            has_position,
            posting_tail_codec,
            positions_layout,
            group_starts,
            index_cache: WeakLanceCache::from(index_cache),
        })
    }

    pub(crate) async fn try_new_v3(
        terms_reader: Arc<dyn IndexReader>,
        block_reader: Arc<dyn IndexReader>,
        position_reader: Option<Arc<dyn IndexReader>>,
        coarse_skip_reader: Arc<dyn IndexReader>,
        index_cache: &LanceCache,
    ) -> Result<Self> {
        if !matches!(
            terms_reader
                .schema()
                .metadata
                .get(POSTING_BLOCK_LAYOUT_KEY)
                .map(String::as_str),
            Some(POSTING_BLOCK_LAYOUT_BLOCK_ROW_V3)
        ) {
            return Err(Error::index(
                "V3 terms table is missing block-row layout metadata".to_owned(),
            ));
        }
        if block_reader.schema().field(POSTING_COL).is_none() {
            return Err(Error::index(
                "V3 blocks table is missing posting payload column".to_owned(),
            ));
        }
        if block_reader.schema().field(FIRST_DOC_ID_COL).is_none()
            || block_reader.schema().field(LAST_DOC_ID_COL).is_none()
            || block_reader.schema().field(BLOCK_MAX_SCORE_COL).is_none()
        {
            return Err(Error::index(
                "V3 blocks table is missing skip columns".to_owned(),
            ));
        }
        if coarse_skip_reader.schema().field(BLOCK_START_COL).is_none()
            || coarse_skip_reader.schema().field(NUM_BLOCKS_COL).is_none()
            || coarse_skip_reader
                .schema()
                .field(FIRST_DOC_ID_COL)
                .is_none()
            || coarse_skip_reader.schema().field(LAST_DOC_ID_COL).is_none()
            || coarse_skip_reader
                .schema()
                .field(BLOCK_MAX_SCORE_COL)
                .is_none()
        {
            return Err(Error::index(
                "V3 coarse-skip table is missing required skip columns".to_owned(),
            ));
        }
        let posting_tail_codec = parse_posting_tail_codec(&terms_reader.schema().metadata)?;
        let block_tail_codec = parse_posting_tail_codec(&block_reader.schema().metadata)?;
        if posting_tail_codec != block_tail_codec {
            return Err(Error::index(format!(
                "V3 terms posting tail codec {:?} does not match blocks codec {:?}",
                posting_tail_codec, block_tail_codec
            )));
        }
        let positions_layout = if let Some(reader) = &position_reader {
            if reader.schema().field(COMPRESSED_POSITION_COL).is_none() {
                return Err(Error::index(
                    "V3 positions table is missing compressed position column".to_owned(),
                ));
            }
            if reader.num_rows() != block_reader.num_rows() {
                return Err(Error::index(format!(
                    "V3 positions table has {} rows but blocks table has {} rows",
                    reader.num_rows(),
                    block_reader.num_rows()
                )));
            }
            PositionsLayout::SharedStream(parse_shared_position_codec(&reader.schema().metadata)?)
        } else {
            PositionsLayout::None
        };
        let terms_has_positions = terms_reader
            .schema()
            .metadata
            .contains_key(POSITIONS_LAYOUT_KEY);
        if terms_has_positions != (positions_layout != PositionsLayout::None) {
            return Err(Error::index(format!(
                "V3 terms positions metadata ({terms_has_positions}) does not match positions file presence ({})",
                positions_layout != PositionsLayout::None
            )));
        }
        let metadata =
            Self::load_v3_terms_metadata(terms_reader.as_ref(), block_reader.num_rows()).await?;

        Ok(Self {
            reader: terms_reader,
            block_reader: Some(block_reader),
            position_reader,
            coarse_skip_reader: Some(coarse_skip_reader),
            metadata,
            has_position: positions_layout != PositionsLayout::None,
            posting_tail_codec,
            positions_layout,
            group_starts: None,
            index_cache: WeakLanceCache::from(index_cache),
        })
    }

    /// Decode the posting-list cache-group boundaries from the global buffer
    /// recorded in schema metadata, if present (issue #7040). Returns `None`
    /// for indexes written before grouping was introduced.
    async fn load_group_starts(reader: &dyn IndexReader) -> Result<Option<Vec<u32>>> {
        let Some(buf_id) = reader.schema().metadata.get(POSTING_GROUP_OFFSETS_BUF_KEY) else {
            return Ok(None);
        };
        let buf_id: u32 = buf_id.parse().map_err(|e| {
            Error::index(format!(
                "invalid {POSTING_GROUP_OFFSETS_BUF_KEY} metadata value {buf_id:?}: {e}"
            ))
        })?;
        let bytes = reader.read_global_buffer(buf_id).await?;
        let group_starts = decode_group_starts(&bytes)?;
        Ok(Some(group_starts))
    }

    async fn load_v3_terms_metadata(
        terms_reader: &dyn IndexReader,
        block_row_count: usize,
    ) -> Result<PostingMetadata> {
        let token_count = terms_reader.num_rows();
        let batch = terms_reader
            .read_range(
                0..token_count,
                Some(&[LENGTH_COL, MAX_SCORE_COL, BLOCK_START_COL, NUM_BLOCKS_COL]),
            )
            .await?;
        if batch.num_rows() != token_count {
            return Err(Error::index(format!(
                "V3 terms table returned {} rows for {} tokens",
                batch.num_rows(),
                token_count
            )));
        }
        let lengths = batch[LENGTH_COL]
            .as_primitive::<UInt32Type>()
            .values()
            .to_vec();
        let max_scores = batch[MAX_SCORE_COL]
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let starts = batch[BLOCK_START_COL].as_primitive::<UInt64Type>();
        let num_blocks = batch[NUM_BLOCKS_COL].as_primitive::<UInt32Type>();

        let mut block_starts = Vec::with_capacity(token_count + 1);
        let mut expected_start = 0u64;
        for row in 0..token_count {
            let start = starts.value(row);
            if start != expected_start {
                return Err(Error::index(format!(
                    "V3 terms metadata is not contiguous at token {row}: block_start is {start}, expected {expected_start}"
                )));
            }
            block_starts.push(start);
            expected_start = start
                .checked_add(u64::from(num_blocks.value(row)))
                .ok_or_else(|| Error::index("V3 block range overflow".to_owned()))?;
        }
        block_starts.push(expected_start);
        if checked_usize(expected_start, "V3 final block start")? != block_row_count {
            return Err(Error::index(format!(
                "V3 terms metadata mismatch: final block start is {} but blocks file has {} rows",
                expected_start, block_row_count
            )));
        }

        Ok(PostingMetadata::V3 {
            block_starts,
            max_scores,
            lengths,
        })
    }

    // for legacy format
    // returns the offsets and max scores
    fn load_metadata(
        schema: &lance_core::datatypes::Schema,
    ) -> Result<(Vec<usize>, Option<Vec<f32>>)> {
        let offsets = schema
            .metadata
            .get("offsets")
            .ok_or(Error::index("offsets not found in metadata".to_owned()))?;
        let offsets = serde_json::from_str(offsets)?;

        let max_scores = schema
            .metadata
            .get("max_scores")
            .map(|max_scores| serde_json::from_str(max_scores))
            .transpose()?;
        Ok((offsets, max_scores))
    }

    // the number of posting lists
    pub fn len(&self) -> usize {
        match &self.metadata {
            PostingMetadata::LegacyV1 { offsets, .. } => offsets.len(),
            PostingMetadata::V2 { .. } => self.reader.num_rows(),
            PostingMetadata::V3 { lengths, .. } => lengths.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn has_positions(&self) -> bool {
        self.has_position
    }

    pub(crate) fn posting_tail_codec(&self) -> PostingTailCodec {
        self.posting_tail_codec
    }

    fn is_legacy_layout(&self) -> bool {
        matches!(self.metadata, PostingMetadata::LegacyV1 { .. })
    }

    fn is_v3_block_row_layout(&self) -> bool {
        matches!(self.metadata, PostingMetadata::V3 { .. })
    }

    /// Sync access to `posting_len`. Requires v2 metadata to already be
    /// loaded via [`ensure_metadata_loaded`]; the bm25 scoring path enforces
    /// that contract before kicking off wand. The stats path uses
    /// [`Self::posting_len_for_token`] instead, which avoids the bulk load.
    pub(crate) fn posting_len(&self, token_id: u32) -> usize {
        let token_id = token_id as usize;
        match &self.metadata {
            PostingMetadata::LegacyV1 { offsets, .. } => {
                let next_offset = offsets
                    .get(token_id + 1)
                    .copied()
                    .unwrap_or(self.reader.num_rows());
                next_offset - offsets[token_id]
            }
            PostingMetadata::V2 { metadata } => {
                let metadata = metadata
                    .get()
                    .expect("v2 posting metadata must be bulk-loaded before sync posting_len; call ensure_metadata_loaded first");
                metadata.lengths[token_id] as usize
            }
            PostingMetadata::V3 { lengths, .. } => lengths[token_id] as usize,
        }
    }

    /// Async access to a single token's posting list length. For v2
    /// indexes this reads one row of posting metadata if the bulk metadata has
    /// not been loaded yet, and never triggers the bulk load itself. The stats
    /// path uses this so a single-term `df` lookup costs O(1) bytes rather
    /// than O(num_unique_tokens).
    pub(crate) async fn posting_len_for_token(&self, token_id: u32) -> Result<usize> {
        match &self.metadata {
            PostingMetadata::LegacyV1 { .. } => Ok(self.posting_len(token_id)),
            PostingMetadata::V2 { metadata } => {
                if let Some(metadata) = metadata.get() {
                    return Ok(metadata.lengths[token_id as usize] as usize);
                }
                let (_, length) = self.posting_metadata_for_token(token_id).await?;
                length
                    .map(|len| len as usize)
                    .ok_or_else(|| Error::index("posting length metadata missing".to_string()))
            }
            PostingMetadata::V3 { lengths, .. } => Ok(lengths[token_id as usize] as usize),
        }
    }

    /// Async access to a single token's `(max_score, length)` pair. Mirrors
    /// [`Self::posting_len_for_token`] but covers both columns the scoring
    /// path needs, in one read. For v2 indexes that have not been
    /// bulk-loaded this issues one `read_range(token..token+1, [MAX_SCORE,
    /// LENGTH])`; for legacy v1 the values come from in-memory schema
    /// metadata.
    pub(crate) async fn posting_metadata_for_token(
        &self,
        token_id: u32,
    ) -> Result<(Option<f32>, Option<u32>)> {
        match &self.metadata {
            PostingMetadata::LegacyV1 { max_scores, .. } => {
                Ok((max_scores.as_ref().map(|m| m[token_id as usize]), None))
            }
            PostingMetadata::V2 { metadata } => {
                if let Some(loaded) = metadata.get() {
                    return Ok((
                        Some(loaded.max_scores[token_id as usize]),
                        Some(loaded.lengths[token_id as usize]),
                    ));
                }
                let metadata = self
                    .index_cache
                    .get_or_insert_with_key(PostingMetadataKey { token_id }, || async move {
                        let token_id = token_id as usize;
                        let batch = self
                            .reader
                            .read_range(token_id..token_id + 1, Some(&[MAX_SCORE_COL, LENGTH_COL]))
                            .await?;
                        let max_score = batch[MAX_SCORE_COL].as_primitive::<Float32Type>().value(0);
                        let length = batch[LENGTH_COL].as_primitive::<UInt32Type>().value(0);
                        Ok(PostingMetadataValue { max_score, length })
                    })
                    .await?;
                Ok((Some(metadata.max_score), Some(metadata.length)))
            }
            PostingMetadata::V3 {
                max_scores,
                lengths,
                ..
            } => Ok((
                Some(max_scores[token_id as usize]),
                Some(lengths[token_id as usize]),
            )),
        }
    }

    /// Force the v2 bulk metadata (`max_scores`, `lengths`) into
    /// memory. Cheap to call repeatedly; no-op for legacy v1 indexes whose
    /// metadata is already populated from schema metadata at `try_new` time.
    pub(crate) async fn ensure_metadata_loaded(&self) -> Result<()> {
        if let PostingMetadata::V2 { metadata } = &self.metadata {
            metadata
                .get_or_try_init(|| async {
                    let batch = self
                        .reader
                        .read_range(
                            0..self.reader.num_rows(),
                            Some(&[MAX_SCORE_COL, LENGTH_COL]),
                        )
                        .await?;
                    let max_scores = batch[MAX_SCORE_COL]
                        .as_primitive::<Float32Type>()
                        .values()
                        .to_vec();
                    let lengths = batch[LENGTH_COL]
                        .as_primitive::<UInt32Type>()
                        .values()
                        .to_vec();
                    Ok::<LoadedPostingMetadata, Error>(LoadedPostingMetadata {
                        max_scores,
                        lengths,
                    })
                })
                .await?;
        }
        Ok(())
    }

    pub(crate) async fn posting_batch(
        &self,
        token_id: u32,
        with_position: bool,
    ) -> Result<RecordBatch> {
        if self.is_legacy_layout() {
            self.posting_batch_legacy(token_id, with_position).await
        } else if matches!(self.metadata, PostingMetadata::V3 { .. }) {
            self.posting_batch_v3(token_id, with_position).await
        } else {
            let token_id = token_id as usize;
            let columns = if with_position {
                match self.positions_layout {
                    PositionsLayout::SharedStream(_) => {
                        vec![
                            POSTING_COL,
                            COMPRESSED_POSITION_COL,
                            POSITION_BLOCK_OFFSET_COL,
                        ]
                    }
                    PositionsLayout::LegacyPerDoc => vec![POSTING_COL, POSITION_COL],
                    PositionsLayout::None => vec![POSTING_COL],
                }
            } else {
                vec![POSTING_COL]
            };
            let batch = self
                .reader
                .read_range(token_id..token_id + 1, Some(&columns))
                .await?;
            Ok(batch)
        }
    }

    async fn posting_batch_v3(&self, token_id: u32, with_position: bool) -> Result<RecordBatch> {
        let block_reader = self.v3_block_reader()?;
        let block_range = self.posting_list_range(token_id);
        let skip_batch = block_reader
            .read_range(
                block_range.clone(),
                Some(&[FIRST_DOC_ID_COL, LAST_DOC_ID_COL, BLOCK_MAX_SCORE_COL]),
            )
            .await?;
        let payload_ranges = self.v3_payload_ranges_from_skip(block_range, &skip_batch)?;
        if v3_block_row_cache_enabled() {
            return self
                .posting_batch_v3_cached(&payload_ranges, with_position)
                .await;
        }
        self.read_v3_payload_ranges(&payload_ranges, with_position)
            .await
    }

    async fn read_v3_payload_ranges(
        &self,
        payload_ranges: &[Range<usize>],
        with_position: bool,
    ) -> Result<RecordBatch> {
        let block_reader = self.v3_block_reader()?;
        let mut batch = block_reader
            .read_ranges(
                payload_ranges,
                Some(&[POSTING_COL, FIRST_DOC_ID_COL, BLOCK_MAX_SCORE_COL]),
            )
            .await?;
        if with_position && matches!(self.positions_layout, PositionsLayout::SharedStream(_)) {
            let position_batch = self
                .v3_position_reader()?
                .read_ranges(payload_ranges, Some(&[COMPRESSED_POSITION_COL]))
                .await
                .map_err(|e| match e {
                    Error::Schema { .. } => Error::invalid_input("position is not found but required for phrase queries, try recreating the index with position".to_owned()),
                    e => e,
                })?;
            batch = append_v3_position_column(batch, position_batch)?;
        }
        Ok(batch)
    }

    async fn posting_batch_v3_cached(
        &self,
        payload_ranges: &[Range<usize>],
        with_position: bool,
    ) -> Result<RecordBatch> {
        let row_ids = row_ids_from_ranges(payload_ranges);
        if row_ids.is_empty() {
            return self
                .read_v3_payload_ranges(payload_ranges, with_position)
                .await;
        }
        let mut row_batches = Vec::with_capacity(row_ids.len());
        let mut missing_rows = Vec::new();
        for row_id in &row_ids {
            let key = V3PostingBlockKey {
                block_row: *row_id,
                with_position,
            };
            if let Some(batch) = self.index_cache.get_with_key(&key).await {
                row_batches.push(Some((*batch).clone()));
            } else {
                missing_rows.push(*row_id);
                row_batches.push(None);
            }
        }

        if !missing_rows.is_empty() {
            let cache_missed_rows = !v3_bypass_posting_list_cache_enabled();
            let row_to_slot = row_ids
                .iter()
                .enumerate()
                .map(|(slot, row_id)| (*row_id, slot))
                .collect::<HashMap<_, _>>();
            let missing_ranges = ranges_from_sorted_rows(&missing_rows);
            let missing_batch = self
                .read_v3_payload_ranges(&missing_ranges, with_position)
                .await?;
            let mut offset = 0;
            for range in missing_ranges {
                for row_id in range {
                    let row_batch = missing_batch.slice(offset, 1).shrink_to_fit()?;
                    offset += 1;
                    if cache_missed_rows {
                        self.index_cache
                            .insert_with_key(
                                &V3PostingBlockKey {
                                    block_row: row_id,
                                    with_position,
                                },
                                Arc::new(row_batch.clone()),
                            )
                            .await;
                    }
                    let slot = *row_to_slot.get(&row_id).ok_or_else(|| {
                        Error::internal(format!("missing V3 block row {row_id} has no output slot"))
                    })?;
                    row_batches[slot] = Some(row_batch);
                }
            }
        }

        let batches = row_batches
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::internal("V3 block-row cache assembly missed rows"))?;
        concat_batches(batches[0].schema_ref(), batches.iter()).map_err(Error::from)
    }

    async fn v3_posting_lists_cached(&self, token_ids: &[u32]) -> bool {
        if !self.is_v3_block_row_layout() || v3_bypass_posting_list_cache_enabled() {
            return false;
        }
        for &token_id in token_ids {
            if self
                .index_cache
                .get_with_key(&PostingListKey { token_id })
                .await
                .is_none()
            {
                return false;
            }
        }
        true
    }

    async fn v3_block_metadata(
        &self,
        token_id: u32,
        term_idx: usize,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<V3BlockMeta>> {
        let skip = self
            .index_cache
            .get_or_insert_with_key(V3BlockMetadataKey { token_id }, || async move {
                let block_reader = self.v3_block_reader()?;
                let block_range = self.posting_list_range(token_id);
                let skip_batch = block_reader
                    .read_range(
                        block_range.clone(),
                        Some(&[FIRST_DOC_ID_COL, LAST_DOC_ID_COL, BLOCK_MAX_SCORE_COL]),
                    )
                    .await?;
                let expected_rows = block_range.end.saturating_sub(block_range.start);
                if skip_batch.num_rows() != expected_rows {
                    return Err(Error::index(format!(
                        "V3 skip metadata returned {} rows for block range {:?}",
                        skip_batch.num_rows(),
                        block_range
                    )));
                }
                metrics.record_fts_block_metadata_rows(skip_batch.num_rows());
                Self::v3_skip_metadata_from_batch(block_range, &skip_batch)
            })
            .await?;

        Ok(skip
            .iter()
            .map(|block| V3BlockMeta {
                term_idx,
                block_idx: block.block_idx,
                block_row: block.block_row,
                first_doc_id: block.first_doc_id,
                last_doc_id: block.last_doc_id,
                block_max_score: block.block_max_score,
            })
            .collect())
    }

    async fn v3_payload_block_batches(
        &self,
        rows: &[usize],
        metrics: &dyn MetricsCollector,
    ) -> Result<HashMap<usize, RecordBatch>> {
        let mut rows = rows.to_vec();
        rows.sort_unstable();
        rows.dedup();
        if rows.is_empty() {
            return Ok(HashMap::new());
        }

        let mut batches = HashMap::with_capacity(rows.len());
        let mut missing_rows = Vec::new();
        for row in rows {
            let key = V3PostingBlockKey {
                block_row: row,
                with_position: false,
            };
            if let Some(batch) = self.index_cache.get_with_key(&key).await {
                batches.insert(row, (*batch).clone());
            } else {
                missing_rows.push(row);
            }
        }

        if !missing_rows.is_empty() {
            let missing_ranges = ranges_from_sorted_rows(&missing_rows);
            metrics.record_fts_payload_blocks_read(missing_rows.len());
            metrics.record_fts_payload_read_ranges(missing_ranges.len());
            let missing_batch = self
                .v3_block_reader()?
                .read_ranges(
                    &missing_ranges,
                    Some(&[POSTING_COL, FIRST_DOC_ID_COL, BLOCK_MAX_SCORE_COL]),
                )
                .await?;
            let cache_missed_rows = !v3_bypass_posting_list_cache_enabled();
            let mut offset = 0;
            for range in missing_ranges {
                for row in range {
                    let row_batch = missing_batch.slice(offset, 1).shrink_to_fit()?;
                    offset += 1;
                    if cache_missed_rows {
                        self.index_cache
                            .insert_with_key(
                                &V3PostingBlockKey {
                                    block_row: row,
                                    with_position: false,
                                },
                                Arc::new(row_batch.clone()),
                            )
                            .await;
                    }
                    batches.insert(row, row_batch);
                }
            }
        }

        Ok(batches)
    }

    async fn v3_position_block_payloads(
        &self,
        rows: &[usize],
        metrics: &dyn MetricsCollector,
    ) -> Result<HashMap<usize, bytes::Bytes>> {
        let mut rows = rows.to_vec();
        rows.sort_unstable();
        rows.dedup();
        if rows.is_empty() {
            return Ok(HashMap::new());
        }

        let mut payloads = HashMap::with_capacity(rows.len());
        let mut missing_rows = Vec::new();
        for row in rows {
            let key = V3PositionBlockKey { block_row: row };
            if let Some(payload) = self.index_cache.get_with_key(&key).await {
                payloads.insert(row, (*payload).clone());
            } else {
                missing_rows.push(row);
            }
        }

        if !missing_rows.is_empty() {
            let missing_ranges = ranges_from_sorted_rows(&missing_rows);
            metrics.record_fts_position_blocks_read(missing_rows.len());
            let batch = self
                .v3_position_reader()?
                .read_ranges(&missing_ranges, Some(&[COMPRESSED_POSITION_COL]))
                .await
                .map_err(|e| match e {
                    Error::Schema { .. } => Error::invalid_input("position is not found but required for phrase queries, try recreating the index with position".to_owned()),
                    e => e,
                })?;
            let positions = batch[COMPRESSED_POSITION_COL].as_binary::<i64>();
            let mut offset = 0;
            for range in missing_ranges {
                for row in range {
                    let payload = bytes::Bytes::copy_from_slice(positions.value(offset));
                    offset += 1;
                    self.index_cache
                        .insert_with_key(
                            &V3PositionBlockKey { block_row: row },
                            Arc::new(payload.clone()),
                        )
                        .await;
                    payloads.insert(row, payload);
                }
            }
        }

        Ok(payloads)
    }

    fn decode_v3_payload_block(
        &self,
        token_id: u32,
        block_idx: usize,
        batch: &RecordBatch,
    ) -> Result<V3DecodedPostingBlock> {
        if batch.num_rows() != 1 {
            return Err(Error::index(format!(
                "V3 payload block decode expects 1 row, got {}",
                batch.num_rows()
            )));
        }
        let PostingMetadata::V3 {
            block_starts,
            lengths,
            ..
        } = &self.metadata
        else {
            return Err(Error::index(
                "V3 payload block decode requires block-row layout".to_owned(),
            ));
        };
        let token_idx = token_id as usize;
        let start = checked_usize(block_starts[token_idx], "V3 posting block start")?;
        let end = checked_usize(block_starts[token_idx + 1], "V3 posting block end")?;
        let num_blocks = end.saturating_sub(start);
        if block_idx >= num_blocks {
            return Err(Error::index(format!(
                "V3 block index {} is outside token {} block count {}",
                block_idx, token_id, num_blocks
            )));
        }
        let posting_len = lengths[token_idx] as usize;
        let docs_before_block = block_idx.saturating_mul(BLOCK_SIZE);
        let docs_in_block = if block_idx + 1 == num_blocks {
            posting_len.saturating_sub(docs_before_block)
        } else {
            BLOCK_SIZE
        };
        if docs_in_block == 0 {
            return Err(Error::index(format!(
                "V3 token {} block {} is empty",
                token_id, block_idx
            )));
        }

        let payload = batch[POSTING_COL].as_binary::<i64>().value(0);
        let first_doc_id = batch[FIRST_DOC_ID_COL]
            .as_primitive::<UInt32Type>()
            .value(0);
        let block_max_score = batch[BLOCK_MAX_SCORE_COL]
            .as_primitive::<Float32Type>()
            .value(0);
        let is_remainder_block = docs_in_block != BLOCK_SIZE;
        let mut block = Vec::with_capacity(payload.len() + 8);
        block.extend_from_slice(&block_max_score.to_le_bytes());
        if is_remainder_block && self.posting_tail_codec == PostingTailCodec::VarintDelta {
            append_varint_u32(first_doc_id, &mut block);
        } else {
            block.extend_from_slice(&first_doc_id.to_le_bytes());
        }
        block.extend_from_slice(payload);

        let mut doc_ids = Vec::with_capacity(docs_in_block);
        let mut freqs = Vec::with_capacity(docs_in_block);
        if is_remainder_block {
            decompress_posting_remainder(
                &block,
                docs_in_block,
                self.posting_tail_codec,
                &mut doc_ids,
                &mut freqs,
            );
        } else {
            let mut buffer = Box::new([0u32; BLOCK_SIZE]);
            decompress_posting_block(&block, &mut buffer, &mut doc_ids, &mut freqs);
        }
        if doc_ids.len() != docs_in_block || freqs.len() != docs_in_block {
            return Err(Error::index(format!(
                "V3 block decode returned {} docs and {} freqs, expected {}",
                doc_ids.len(),
                freqs.len(),
                docs_in_block
            )));
        }
        Ok(V3DecodedPostingBlock { doc_ids, freqs })
    }

    async fn posting_batch_legacy(
        &self,
        token_id: u32,
        with_position: bool,
    ) -> Result<RecordBatch> {
        let mut columns = vec![ROW_ID, FREQUENCY_COL];
        if with_position {
            columns.push(POSITION_COL);
        }

        let length = self.posting_len(token_id);
        let PostingMetadata::LegacyV1 { offsets, .. } = &self.metadata else {
            unreachable!("posting_batch_legacy is only reachable on legacy v1 layout");
        };
        let token_id = token_id as usize;
        let offset = offsets[token_id];
        let batch = self
            .reader
            .read_range(offset..offset + length, Some(&columns))
            .await?;
        Ok(batch)
    }

    fn v3_block_reader(&self) -> Result<&Arc<dyn IndexReader>> {
        self.block_reader.as_ref().ok_or_else(|| {
            Error::index("V3 posting reader is missing blocks table reader".to_owned())
        })
    }

    fn v3_position_reader(&self) -> Result<&Arc<dyn IndexReader>> {
        self.position_reader.as_ref().ok_or_else(|| {
            Error::invalid_input(
                "position is not found but required for phrase queries, try recreating the index with position".to_owned(),
            )
        })
    }

    fn posting_row_reader(&self) -> Result<&Arc<dyn IndexReader>> {
        if matches!(self.metadata, PostingMetadata::V3 { .. }) {
            self.v3_block_reader()
        } else {
            Ok(&self.reader)
        }
    }

    fn v3_payload_ranges_from_skip(
        &self,
        block_range: Range<usize>,
        skip_batch: &RecordBatch,
    ) -> Result<Vec<Range<usize>>> {
        let expected_rows = block_range.end.saturating_sub(block_range.start);
        if skip_batch.num_rows() != expected_rows {
            return Err(Error::index(format!(
                "V3 skip-first read returned {} rows for block range {:?}",
                skip_batch.num_rows(),
                block_range
            )));
        }
        let first_doc_ids = skip_batch[FIRST_DOC_ID_COL].as_primitive::<UInt32Type>();
        let last_doc_ids = skip_batch[LAST_DOC_ID_COL].as_primitive::<UInt32Type>();
        let max_scores = skip_batch[BLOCK_MAX_SCORE_COL].as_primitive::<Float32Type>();
        for row in 0..skip_batch.num_rows() {
            let first_doc_id = first_doc_ids.value(row);
            let last_doc_id = last_doc_ids.value(row);
            if first_doc_id > last_doc_id {
                return Err(Error::index(format!(
                    "V3 skip row {} has first_doc_id {} greater than last_doc_id {}",
                    block_range.start + row,
                    first_doc_id,
                    last_doc_id
                )));
            }
            if max_scores.value(row).is_nan() {
                return Err(Error::index(format!(
                    "V3 skip row {} has NaN block max score",
                    block_range.start + row
                )));
            }
        }
        if expected_rows == 0 {
            Ok(Vec::new())
        } else {
            Ok(vec![block_range])
        }
    }

    fn v3_skip_metadata_from_batch(
        block_range: Range<usize>,
        skip_batch: &RecordBatch,
    ) -> Result<Vec<V3BlockSkipMeta>> {
        let expected_rows = block_range.end.saturating_sub(block_range.start);
        if skip_batch.num_rows() != expected_rows {
            return Err(Error::index(format!(
                "V3 skip metadata returned {} rows for block range {:?}",
                skip_batch.num_rows(),
                block_range
            )));
        }

        let first_doc_ids = skip_batch[FIRST_DOC_ID_COL].as_primitive::<UInt32Type>();
        let last_doc_ids = skip_batch[LAST_DOC_ID_COL].as_primitive::<UInt32Type>();
        let max_scores = skip_batch[BLOCK_MAX_SCORE_COL].as_primitive::<Float32Type>();
        let mut blocks = Vec::with_capacity(skip_batch.num_rows());
        let mut previous_last = None;
        for row in 0..skip_batch.num_rows() {
            let first_doc_id = first_doc_ids.value(row);
            let last_doc_id = last_doc_ids.value(row);
            let block_max_score = max_scores.value(row);
            if first_doc_id > last_doc_id {
                return Err(Error::index(format!(
                    "V3 skip row {} has first_doc_id {} greater than last_doc_id {}",
                    block_range.start + row,
                    first_doc_id,
                    last_doc_id
                )));
            }
            if block_max_score.is_nan() {
                return Err(Error::index(format!(
                    "V3 skip row {} has NaN block max score",
                    block_range.start + row
                )));
            }
            if previous_last.is_some_and(|previous| previous >= first_doc_id) {
                return Err(Error::index(format!(
                    "V3 skip row {} overlaps or precedes the previous block",
                    block_range.start + row
                )));
            }
            previous_last = Some(last_doc_id);
            blocks.push(V3BlockSkipMeta {
                block_idx: row,
                block_row: block_range.start + row,
                first_doc_id,
                last_doc_id,
                block_max_score,
            });
        }
        Ok(blocks)
    }

    fn v3_top_payload_ranges_from_skip(
        &self,
        block_range: Range<usize>,
        skip_batch: &RecordBatch,
        percent: usize,
    ) -> Result<Vec<Range<usize>>> {
        let expected_rows = block_range.end.saturating_sub(block_range.start);
        let full_ranges = self.v3_payload_ranges_from_skip(block_range.clone(), skip_batch)?;
        if expected_rows == 0 || percent >= 100 {
            return Ok(full_ranges);
        }
        let max_scores = skip_batch[BLOCK_MAX_SCORE_COL].as_primitive::<Float32Type>();
        let keep = expected_rows.saturating_mul(percent).div_ceil(100).max(1);
        let mut scored_rows = (0..expected_rows)
            .map(|row| (max_scores.value(row), block_range.start + row))
            .collect::<Vec<_>>();
        scored_rows.sort_unstable_by(|(left_score, left_row), (right_score, right_row)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_row.cmp(right_row))
        });
        let mut selected_rows = scored_rows
            .into_iter()
            .take(keep)
            .map(|(_, row)| row)
            .collect::<Vec<_>>();
        selected_rows.sort_unstable();
        Ok(ranges_from_sorted_rows(&selected_rows))
    }

    #[instrument(level = "debug", skip(self, metrics))]
    pub(crate) async fn posting_list(
        &self,
        token_id: u32,
        is_phrase_query: bool,
        metrics: &dyn MetricsCollector,
    ) -> Result<PostingList> {
        let mut posting = match self.group_range_for_token(token_id) {
            // Grouped path (issue #7040): one cache entry covers rows
            // [start, end), so neighbouring rare terms share a single read.
            Some((start, end)) => {
                let group = self
                    .index_cache
                    .get_or_insert_with_key(PostingListGroupKey { start, end }, || async move {
                        metrics.record_part_load();
                        info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_SCALAR_PART, index_type="inverted", part_id=start);
                        self.load_posting_list_group(start, end).await
                    })
                    .await?;
                let slot = (token_id - start) as usize;
                group
                    .get(slot)
                    .ok_or_else(|| {
                        Error::index(format!(
                            "token {token_id} maps to slot {slot} outside posting group [{start}, {end})"
                        ))
                    })?
                    .clone()
            }
            // Fallback for indexes written before grouping: one cache entry
            // per token.
            None => {
                if matches!(self.metadata, PostingMetadata::V3 { .. })
                    && v3_bypass_posting_list_cache_enabled()
                {
                    metrics.record_part_load();
                    info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_SCALAR_PART, index_type="inverted", part_id=token_id);
                    let (batch, (max_score, length)) = futures::try_join!(
                        self.posting_batch(token_id, is_phrase_query),
                        self.posting_metadata_for_token(token_id),
                    )?;
                    self.posting_list_from_batch(&batch, max_score, length)?
                } else {
                    self.index_cache
                        .get_or_insert_with_key(PostingListKey { token_id }, || async move {
                            metrics.record_part_load();
                            info!(target: TRACE_IO_EVENTS, r#type=IO_TYPE_LOAD_SCALAR_PART, index_type="inverted", part_id=token_id);
                            // Fetch the posting batch and this token's (max_score,
                            // length) in parallel; for cold v2 partitions this is one
                            // single-row metadata read plus one posting-row read,
                            // instead of pulling the full per-token metadata table.
                            let (batch, (max_score, length)) = futures::try_join!(
                                self.posting_batch(token_id, false),
                                self.posting_metadata_for_token(token_id),
                            )?;
                            self.posting_list_from_batch(&batch, max_score, length)
                        })
                        .await?
                        .as_ref()
                        .clone()
                }
            }
        };

        if is_phrase_query && !posting.has_position() {
            // hit the cache and when the cache was populated, the positions column was not loaded
            let positions = self.read_positions(token_id).await?;
            posting.set_positions(positions);
        }

        Ok(posting)
    }

    /// Map a token id to its cache group's row range `[start, end)`, or `None`
    /// when grouping is not available (pre-grouping indexes) so the caller
    /// falls back to the per-token path. In v2 the token id is the row offset,
    /// so the group range is also the physical row range.
    fn group_range_for_token(&self, token_id: u32) -> Option<(u32, u32)> {
        if matches!(self.metadata, PostingMetadata::V3 { .. }) {
            return None;
        }
        let starts = self.group_starts.as_ref()?;
        // partition_point returns the count of group starts <= token_id, so the
        // owning group begins at index k - 1 and the next start (if any) is its
        // exclusive end.
        let k = starts.partition_point(|&s| s <= token_id);
        // k == 0 means token_id precedes the first group start, which cannot
        // happen for a valid token in a grouped index (the first group starts
        // at row 0); guard anyway and fall back to the per-token path.
        if k == 0 {
            return None;
        }
        let start = starts[k - 1];
        // The last group runs to the final posting list. `self.len()` is the
        // authoritative posting-list count (offsets length for v1, row count for
        // v2), and prewarm derives the same `end` from it — so warm- and
        // cold-cache group keys are identical by construction, not by the
        // incidental v2 `num_rows == token_count` equality.
        let end = starts.get(k).copied().unwrap_or(self.len() as u32);
        Some((start, end))
    }

    /// Read rows `[start, end)` of the posting file and decode them into a
    /// [`PostingListGroup`] cache value (issue #7040). Positions are excluded;
    /// phrase queries load them on demand via [`Self::read_positions`].
    async fn load_posting_list_group(&self, start: u32, end: u32) -> Result<PostingListGroup> {
        let batch = self
            .reader
            .read_range(
                start as usize..end as usize,
                Some(&[POSTING_COL, MAX_SCORE_COL, LENGTH_COL]),
            )
            .await?;
        let max_scores = batch[MAX_SCORE_COL].as_primitive::<Float32Type>();
        let lengths = batch[LENGTH_COL].as_primitive::<UInt32Type>();
        let mut posting_lists = Vec::with_capacity(batch.num_rows());
        for i in 0..batch.num_rows() {
            let row = batch.slice(i, 1);
            let posting = self.posting_list_from_batch(
                &row,
                Some(max_scores.value(i)),
                Some(lengths.value(i)),
            )?;
            posting_lists.push(posting);
        }
        Ok(PostingListGroup::new(posting_lists))
    }

    fn posting_list_from_batch_parts(
        batch: &RecordBatch,
        max_score: Option<f32>,
        length: Option<u32>,
        posting_tail_codec: PostingTailCodec,
        positions_layout: PositionsLayout,
    ) -> Result<PostingList> {
        let posting_list = PostingList::from_batch_with_tail_codec_and_positions_layout(
            batch,
            max_score,
            length,
            posting_tail_codec,
            positions_layout,
        )?;
        Ok(posting_list)
    }

    pub(crate) fn posting_list_from_batch(
        &self,
        batch: &RecordBatch,
        max_score: Option<f32>,
        length: Option<u32>,
    ) -> Result<PostingList> {
        Self::posting_list_from_batch_parts(
            batch,
            max_score,
            length,
            self.posting_tail_codec,
            self.positions_layout,
        )
    }

    /// Build posting lists for one chunk's token range from `chunk_batch`, rebasing
    /// global offsets to chunk-local rows. Returns `(global token_id, PostingList)`
    /// pairs identical to the whole-file path, only bounded to one chunk.
    fn build_prewarm_posting_lists_chunk(
        chunk_batch: RecordBatch,
        chunk: PrewarmChunk<'_>,
        ctx: &PrewarmBuildCtx<'_>,
    ) -> Result<Vec<(u32, PostingList)>> {
        let mut posting_lists = Vec::with_capacity(chunk.token_count);
        for local in 0..chunk.token_count {
            let global = chunk.tok_start + local;
            let row_batch = if let Some(chunk_offsets) = chunk.offsets {
                // Legacy v1: rebase global offsets to chunk row 0; the last token
                // ends at `chunk.end_row` (no trailing sentinel in chunk_offsets).
                let base = chunk_offsets[0];
                let start = chunk_offsets[local] - base;
                let end = if local + 1 < chunk_offsets.len() {
                    chunk_offsets[local + 1] - base
                } else {
                    chunk.end_row - base
                };
                chunk_batch.slice(start, end - start)
            } else {
                // V2: one posting row per token; row `local` within the chunk.
                chunk_batch.slice(local, 1)
            };
            let row_batch = row_batch.shrink_to_fit()?;
            let posting_list = Self::posting_list_from_batch_parts(
                &row_batch,
                ctx.max_scores.map(|scores| scores[global]),
                ctx.lengths.map(|lengths| lengths[global]),
                ctx.posting_tail_codec,
                ctx.positions_layout,
            )?;
            posting_lists.push((global as u32, posting_list));
        }

        Ok(posting_lists)
    }

    /// Read the posting rows for token ids `[tok_start, tok_end)` into one RecordBatch.
    /// For v2 the token range is the row range; for v1 it's derived from the offsets.
    async fn read_chunk_batch(
        &self,
        tok_start: usize,
        tok_end: usize,
        with_position: bool,
    ) -> Result<RecordBatch> {
        let columns = self.posting_columns(with_position);
        let row_range = match &self.metadata {
            PostingMetadata::LegacyV1 { offsets, .. } => {
                let start = offsets[tok_start];
                let end = offsets
                    .get(tok_end)
                    .copied()
                    .unwrap_or_else(|| self.reader.num_rows());
                start..end
            }
            PostingMetadata::V2 { .. } => tok_start..tok_end,
            PostingMetadata::V3 { block_starts, .. } => {
                checked_usize(block_starts[tok_start], "V3 posting block start")?
                    ..checked_usize(block_starts[tok_end], "V3 posting block end")?
            }
        };
        let mut batch = self
            .posting_row_reader()?
            .read_range(row_range.clone(), Some(&columns))
            .await?;
        if with_position && matches!(self.metadata, PostingMetadata::V3 { .. }) {
            let position_batch = self
                .v3_position_reader()?
                .read_range(row_range, Some(&[COMPRESSED_POSITION_COL]))
                .await
                .map_err(|e| match e {
                    Error::Schema { .. } => Error::invalid_input("position is not found but required for phrase queries, try recreating the index with position".to_owned()),
                    e => e,
                })?;
            batch = append_v3_position_column(batch, position_batch)?;
        }
        Ok(batch)
    }

    async fn prewarm_posting_lists(
        &self,
        with_position: bool,
        chunk_concurrency: usize,
    ) -> Result<()> {
        self.prewarm_posting_lists_chunked(with_position, None, chunk_concurrency)
            .await?;
        Ok(())
    }

    async fn prewarm_v3_top_blocks(
        &self,
        token_ids: &[u32],
        percent: usize,
        with_position: bool,
    ) -> Result<usize> {
        if token_ids.is_empty() {
            return Ok(0);
        }
        if !matches!(self.metadata, PostingMetadata::V3 { .. }) {
            return Err(Error::invalid_input(
                "top-block FTS prewarm is only supported for V3 block-row posting layout"
                    .to_owned(),
            ));
        }
        if with_position && !self.has_positions() {
            return Err(Error::invalid_input(
                "cannot prewarm positions for an inverted index that was built without positions; recreate the index with with_position=true".to_owned(),
            ));
        }

        let block_reader = self.v3_block_reader()?;
        let mut selected_rows = HashSet::new();
        for &token_id in token_ids {
            let block_range = self.posting_list_range(token_id);
            let skip_batch = block_reader
                .read_range(
                    block_range.clone(),
                    Some(&[FIRST_DOC_ID_COL, LAST_DOC_ID_COL, BLOCK_MAX_SCORE_COL]),
                )
                .await?;
            let skip_metadata =
                Self::v3_skip_metadata_from_batch(block_range.clone(), &skip_batch)?;
            self.index_cache
                .insert_with_key(&V3BlockMetadataKey { token_id }, Arc::new(skip_metadata))
                .await;
            let mut token_rows = Vec::new();
            for range in self.v3_top_payload_ranges_from_skip(block_range, &skip_batch, percent)? {
                token_rows.extend(range);
            }
            token_rows.sort_unstable();
            token_rows.dedup();
            selected_rows.extend(token_rows.iter().copied());
        }

        if selected_rows.is_empty() {
            return Ok(0);
        }
        let mut selected_rows = selected_rows.into_iter().collect::<Vec<_>>();
        selected_rows.sort_unstable();
        // Use the same cache entries as query execution, including positions.
        // Chunking bounds transient memory even when the cache is much smaller
        // than the selected blocks.
        for rows in selected_rows.chunks(4096) {
            self.v3_payload_block_batches(rows, &NoOpMetricsCollector)
                .await?;
            if with_position {
                self.v3_position_block_payloads(rows, &NoOpMetricsCollector)
                    .await?;
            }
        }
        Ok(selected_rows.len())
    }

    async fn prewarm_posting_lists_chunked(
        &self,
        with_position: bool,
        chunk_tokens_override: Option<usize>,
        chunk_concurrency: usize,
    ) -> Result<usize> {
        if with_position && !self.has_positions() {
            return Err(Error::invalid_input(
                "cannot prewarm positions for an inverted index that was built without positions; recreate the index with with_position=true".to_owned(),
            ));
        }

        // Make sure max_scores/lengths are populated before we clone them into
        // the blocking task; otherwise the v2 branch would unwrap empty
        // OnceCells.
        self.ensure_metadata_loaded().await?;

        let state = self.chunk_build_state();
        // With grouping the cache stores one entry per group, so a group's posting
        // lists must all be resident at once: align chunk boundaries to whole
        // groups. Without grouping, chunks are plain token ranges.
        let group_starts = self.group_starts.clone();
        let token_count = self.len();
        let posting_data_size_bytes = self.posting_data_size_bytes();
        let chunk_tokens = chunk_tokens_override
            .unwrap_or_else(|| prewarm_chunk_tokens(token_count, posting_data_size_bytes))
            .max(1);
        let chunk_ranges = prewarm_chunk_ranges(group_starts.as_deref(), token_count, chunk_tokens);
        let chunk_count = chunk_ranges.len();
        let chunk_concurrency = chunk_concurrency.max(1);

        let read_build_start = Instant::now();
        stream::iter(chunk_ranges)
            .map(|(tok_start, tok_end)| {
                let state = &state;
                let group_starts = group_starts.as_deref();
                async move {
                    let posting_lists = self
                        .build_chunk_postings(tok_start, tok_end, with_position, state)
                        .await?;
                    self.publish_chunk_postings(
                        posting_lists,
                        group_starts,
                        tok_start,
                        tok_end,
                        token_count,
                        with_position,
                    )
                    .await;
                    Result::Ok(())
                }
            })
            .buffer_unordered(chunk_concurrency)
            .try_collect::<()>()
            .await?;
        let read_build_elapsed = read_build_start.elapsed();

        info!(
            legacy_layout = self.is_legacy_layout(),
            with_position,
            token_count,
            chunk_count,
            chunk_tokens,
            chunk_concurrency,
            posting_data_size_bytes,
            read_build_ms = read_build_elapsed.as_secs_f64() * 1000.0,
            "posting list prewarm timing"
        );

        Ok(chunk_count)
    }

    /// Loop-invariant inputs shared by every chunk build: the metadata vecs
    /// (`Arc`d so chunks share them without re-cloning) plus codec/layout.
    fn chunk_build_state(&self) -> ChunkBuildState {
        let (offsets, max_scores, lengths) = match &self.metadata {
            PostingMetadata::LegacyV1 {
                offsets,
                max_scores,
            } => (Some(offsets.clone()), max_scores.clone(), None),
            PostingMetadata::V2 { metadata } => (
                None,
                metadata.get().map(|loaded| loaded.max_scores.clone()),
                metadata.get().map(|loaded| loaded.lengths.clone()),
            ),
            PostingMetadata::V3 {
                block_starts,
                max_scores,
                lengths,
            } => (
                Some(
                    block_starts
                        .iter()
                        .map(|offset| checked_usize(*offset, "V3 posting block start"))
                        .collect::<Result<Vec<_>>>()
                        .expect("V3 block starts were validated at load"),
                ),
                Some(max_scores.clone()),
                Some(lengths.clone()),
            ),
        };
        ChunkBuildState {
            offsets: offsets.map(Arc::new),
            max_scores: max_scores.map(Arc::new),
            lengths: lengths.map(Arc::new),
            posting_tail_codec: self.posting_tail_codec,
            positions_layout: self.positions_layout,
        }
    }

    /// Read one token-row chunk and build its posting lists off the runtime thread.
    /// The large batch is dropped inside the blocking task once built, bounding
    /// resident memory to one chunk.
    async fn build_chunk_postings(
        &self,
        tok_start: usize,
        tok_end: usize,
        with_position: bool,
        state: &ChunkBuildState,
    ) -> Result<Vec<(u32, PostingList)>> {
        let chunk_token_count = tok_end - tok_start;
        let chunk_batch = self
            .read_chunk_batch(tok_start, tok_end, with_position)
            .await?;

        let (chunk_offsets, chunk_end_row) = match state.offsets.as_ref() {
            Some(offsets) => {
                let end_row = offsets
                    .get(tok_end)
                    .copied()
                    .unwrap_or_else(|| self.reader.num_rows());
                (Some(offsets[tok_start..tok_end].to_vec()), end_row)
            }
            // V2 doesn't use chunk_end_row (one row per token); pass tok_end.
            None => (None, tok_end),
        };
        let max_scores = state.max_scores.clone();
        let lengths = state.lengths.clone();
        let posting_tail_codec = state.posting_tail_codec;
        let positions_layout = state.positions_layout;
        let posting_lists = spawn_blocking(move || {
            let ctx = PrewarmBuildCtx {
                max_scores: max_scores.as_deref().map(|v| v.as_slice()),
                lengths: lengths.as_deref().map(|v| v.as_slice()),
                posting_tail_codec,
                positions_layout,
            };
            let chunk = PrewarmChunk {
                tok_start,
                token_count: chunk_token_count,
                offsets: chunk_offsets.as_deref(),
                end_row: chunk_end_row,
            };
            Self::build_prewarm_posting_lists_chunk(chunk_batch, chunk, &ctx)
        })
        .await
        .map_err(|err| {
            Error::internal(format!(
                "Failed to build prewarm posting lists in blocking task: {err}"
            ))
        })??;
        // The chunk yields its token range as contiguous ascending ids from
        // `tok_start`; the group publish path relies on this to index the lists.
        debug_assert_eq!(posting_lists.len(), chunk_token_count);
        debug_assert!(
            posting_lists
                .iter()
                .enumerate()
                .all(|(i, (token_id, _))| *token_id as usize == tok_start + i)
        );
        Ok(posting_lists)
    }

    /// Strip positions into their own per-token cache entries (the posting cache
    /// holds positions-free lists), then populate the same cache keys the read
    /// path uses: grouped entries when grouping is active, per-token entries
    /// otherwise. Called once per chunk; the chunk's lists drop on return.
    async fn publish_chunk_postings(
        &self,
        posting_lists: Vec<(u32, PostingList)>,
        group_starts: Option<&[u32]>,
        tok_start: usize,
        tok_end: usize,
        token_count: usize,
        with_position: bool,
    ) {
        match group_starts {
            Some(starts) => {
                let mut chunk_postings = Vec::with_capacity(posting_lists.len());
                for (token_id, mut posting_list) in posting_lists {
                    self.cache_positions(&mut posting_list, token_id, with_position)
                        .await;
                    chunk_postings.push(posting_list);
                }
                // Chunk is group-aligned, so every group starting in it also ends
                // in it; `chunk_postings[i]` is token `tok_start + i`. The last
                // group's `end` derives from `token_count`, matching the read path
                // so both produce identical `PostingListGroupKey`s.
                for group_idx in group_start_indices_for_chunk(starts, tok_start, tok_end) {
                    let (start, end) = group_range_for_start_index(starts, token_count, group_idx);
                    let start_usize = start as usize;
                    let lo = start_usize - tok_start;
                    let hi = end as usize - tok_start;
                    let group = PostingListGroup::new(chunk_postings[lo..hi].to_vec());
                    self.index_cache
                        .insert_with_key(&PostingListGroupKey { start, end }, Arc::new(group))
                        .await;
                }
            }
            None => {
                for (token_id, mut posting_list) in posting_lists {
                    self.cache_positions(&mut posting_list, token_id, with_position)
                        .await;
                    self.index_cache
                        .insert_with_key(&PostingListKey { token_id }, Arc::new(posting_list))
                        .await;
                }
            }
        }
    }

    /// Move a posting list's positions (when present and requested) into the
    /// dedicated per-token position cache, leaving the posting list positions-free.
    async fn cache_positions(
        &self,
        posting_list: &mut PostingList,
        token_id: u32,
        with_position: bool,
    ) {
        if with_position && let Some(positions) = posting_list.take_positions() {
            self.index_cache
                .insert_with_key(&PositionKey { token_id }, Arc::new(Positions(positions)))
                .await;
        }
    }

    /// Cheap `invert.lance` size estimate (file length from object metadata, no
    /// data read), used only to size prewarm chunks. Falls back to a row-count
    /// proxy when the reader can't surface the length (legacy v1).
    pub(crate) fn posting_data_size_bytes(&self) -> u64 {
        let reader = if matches!(self.metadata, PostingMetadata::V3 { .. }) {
            self.block_reader.as_ref().unwrap_or(&self.reader)
        } else {
            &self.reader
        };
        if let Some(size) = reader.file_size_bytes() {
            return size;
        }
        // Fallback proxy for readers that don't cache their file length: just needs
        // to be monotonic in partition size.
        const ESTIMATED_BYTES_PER_ROW: u64 = 16;
        (reader.num_rows() as u64).saturating_mul(ESTIMATED_BYTES_PER_ROW)
    }

    pub(crate) async fn read_batch(&self, with_position: bool) -> Result<RecordBatch> {
        let columns = self.posting_columns(with_position);
        let reader = self.posting_row_reader()?;
        let mut batch = reader
            .read_range(0..reader.num_rows(), Some(&columns))
            .await?;
        if with_position && matches!(self.metadata, PostingMetadata::V3 { .. }) {
            let position_batch = self
                .v3_position_reader()?
                .read_range(0..reader.num_rows(), Some(&[COMPRESSED_POSITION_COL]))
                .await
                .map_err(|e| match e {
                    Error::Schema { .. } => Error::invalid_input("position is not found but required for phrase queries, try recreating the index with position".to_owned()),
                    e => e,
                })?;
            batch = append_v3_position_column(batch, position_batch)?;
        }
        Ok(batch)
    }

    pub(crate) async fn read_all(
        &self,
        with_position: bool,
    ) -> Result<impl Iterator<Item = Result<PostingList>> + '_> {
        // read_all walks every posting list; the bulk metadata is paid for
        // unconditionally, so just load it once up front and index into it
        // synchronously below.
        self.ensure_metadata_loaded().await?;
        let batch = self.read_batch(with_position).await?;
        Ok((0..self.len()).map(move |i| {
            let token_id = i as u32;
            let range = self.posting_list_range(token_id);
            let batch = batch.slice(range.start, range.end - range.start);
            let (max_score, length) = self.bulk_metadata_for_token(token_id);
            self.posting_list_from_batch(&batch, max_score, length)
        }))
    }

    /// Sync lookup of `(max_score, length)` from the bulk-loaded metadata.
    /// Only safe after [`Self::ensure_metadata_loaded`]; callers that hold
    /// the OnceCell-loaded reference (e.g. read_all, prewarm) use this to
    /// avoid the per-token IO path.
    fn bulk_metadata_for_token(&self, token_id: u32) -> (Option<f32>, Option<u32>) {
        match &self.metadata {
            PostingMetadata::LegacyV1 { max_scores, .. } => {
                (max_scores.as_ref().map(|m| m[token_id as usize]), None)
            }
            PostingMetadata::V2 { metadata } => {
                let loaded = metadata.get().expect(
                    "v2 metadata must be bulk-loaded before bulk_metadata_for_token; call ensure_metadata_loaded first",
                );
                (
                    Some(loaded.max_scores[token_id as usize]),
                    Some(loaded.lengths[token_id as usize]),
                )
            }
            PostingMetadata::V3 {
                max_scores,
                lengths,
                ..
            } => (
                Some(max_scores[token_id as usize]),
                Some(lengths[token_id as usize]),
            ),
        }
    }

    async fn read_positions(&self, token_id: u32) -> Result<CompressedPositionStorage> {
        let positions = self.index_cache.get_or_insert_with_key(PositionKey { token_id }, || async move {
            let positions = match self.positions_layout {
                PositionsLayout::None => {
                    return Err(Error::invalid_input(
                        "position is not found but required for phrase queries, try recreating the index with position".to_owned(),
                    ));
                }
                PositionsLayout::LegacyPerDoc => {
                    let batch = self
                        .reader
                        .read_range(self.posting_list_range(token_id), Some(&[POSITION_COL]))
                        .await
                        .map_err(|e| match e {
                            Error::Schema { .. } => Error::invalid_input("position is not found but required for phrase queries, try recreating the index with position".to_owned()),
                            e => e,
                        })?;
                    CompressedPositionStorage::LegacyPerDoc(
                        batch[POSITION_COL].as_list::<i32>().value(0).as_list::<i32>().clone(),
                    )
                }
                PositionsLayout::SharedStream(codec) => {
                    let batch = if matches!(self.metadata, PostingMetadata::V3 { .. }) {
                        self.v3_position_reader()?
                            .read_range(
                                self.posting_list_range(token_id),
                                Some(&[COMPRESSED_POSITION_COL]),
                            )
                            .await
                    } else {
                        self.reader
                            .read_range(
                                self.posting_list_range(token_id),
                                Some(&[COMPRESSED_POSITION_COL, POSITION_BLOCK_OFFSET_COL]),
                            )
                            .await
                    };
                    let batch = batch
                        .map_err(|e| match e {
                            Error::Schema { .. } => Error::invalid_input("position is not found but required for phrase queries, try recreating the index with position".to_owned()),
                            e => e,
                        })?;
                    let positions = batch[COMPRESSED_POSITION_COL].as_binary::<i64>();
                    let (block_offsets, bytes) =
                        if let Some(offset_col) = batch.column_by_name(POSITION_BLOCK_OFFSET_COL) {
                            (
                                offset_col
                                    .as_list::<i32>()
                                    .value(0)
                                    .as_primitive::<UInt32Type>()
                                    .values()
                                    .to_vec(),
                                bytes::Bytes::from(positions.value(0).to_vec()),
                            )
                        } else {
                            let mut block_offsets = Vec::with_capacity(positions.len());
                            let mut bytes = Vec::new();
                            for block_idx in 0..positions.len() {
                                block_offsets.push(bytes.len() as u32);
                                bytes.extend_from_slice(positions.value(block_idx));
                            }
                            (block_offsets, bytes::Bytes::from(bytes))
                        };
                    CompressedPositionStorage::SharedStream(SharedPositionStream::new(codec, block_offsets, bytes))
                }
            };
            Result::Ok(Positions(positions))
        }).await?;
        Ok(positions.0.clone())
    }

    fn posting_list_range(&self, token_id: u32) -> Range<usize> {
        match &self.metadata {
            PostingMetadata::LegacyV1 { offsets, .. } => {
                let offset = offsets[token_id as usize];
                let posting_len = self.posting_len(token_id);
                offset..offset + posting_len
            }
            PostingMetadata::V2 { .. } => {
                let token_id = token_id as usize;
                token_id..token_id + 1
            }
            PostingMetadata::V3 { block_starts, .. } => {
                let token_id = token_id as usize;
                checked_usize(block_starts[token_id], "V3 posting block start")
                    .expect("V3 block starts were validated at load")
                    ..checked_usize(block_starts[token_id + 1], "V3 posting block end")
                        .expect("V3 block starts were validated at load")
            }
        }
    }

    fn posting_columns(&self, with_position: bool) -> Vec<&'static str> {
        let mut base_columns = if self.is_legacy_layout() {
            vec![ROW_ID, FREQUENCY_COL]
        } else {
            vec![POSTING_COL]
        };
        if matches!(self.metadata, PostingMetadata::V3 { .. }) {
            base_columns.push(FIRST_DOC_ID_COL);
            base_columns.push(BLOCK_MAX_SCORE_COL);
        }
        if with_position {
            if matches!(self.metadata, PostingMetadata::V3 { .. }) {
                return base_columns;
            }
            match self.positions_layout {
                PositionsLayout::None => {}
                PositionsLayout::LegacyPerDoc => base_columns.push(POSITION_COL),
                PositionsLayout::SharedStream(_) => {
                    base_columns.push(COMPRESSED_POSITION_COL);
                    if !matches!(self.metadata, PostingMetadata::V3 { .. }) {
                        base_columns.push(POSITION_BLOCK_OFFSET_COL);
                    }
                }
            }
        }
        base_columns
    }
}

fn append_v3_position_column(
    block_batch: RecordBatch,
    position_batch: RecordBatch,
) -> Result<RecordBatch> {
    if block_batch.num_rows() != position_batch.num_rows() {
        return Err(Error::index(format!(
            "V3 positions table returned {} rows for {} posting block rows",
            position_batch.num_rows(),
            block_batch.num_rows()
        )));
    }
    let mut fields: Vec<Field> = block_batch
        .schema_ref()
        .fields()
        .iter()
        .map(|field| field.as_ref().clone())
        .collect();
    let (_, position_field) = position_batch
        .schema_ref()
        .column_with_name(COMPRESSED_POSITION_COL)
        .ok_or_else(|| {
            Error::index("V3 positions batch is missing compressed position column".to_owned())
        })?;
    fields.push(position_field.clone());

    let mut columns = block_batch.columns().to_vec();
    columns.push(position_batch[COMPRESSED_POSITION_COL].clone());

    let mut metadata = block_batch.schema_ref().metadata().clone();
    metadata.extend(position_batch.schema_ref().metadata().clone());
    RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(fields, metadata)),
        columns,
    )
    .map_err(Error::from)
}

fn row_ids_from_ranges(ranges: &[Range<usize>]) -> Vec<usize> {
    ranges
        .iter()
        .flat_map(|range| range.start..range.end)
        .collect()
}

fn ranges_from_sorted_rows(rows: &[usize]) -> Vec<Range<usize>> {
    let Some((&first, rest)) = rows.split_first() else {
        return Vec::new();
    };
    let mut ranges = Vec::new();
    let mut start = first;
    let mut end = first + 1;
    for &row in rest {
        if row == end {
            end += 1;
        } else {
            ranges.push(start..end);
            start = row;
            end = row + 1;
        }
    }
    ranges.push(start..end);
    ranges
}

/// Loop-invariant state for [`InvertedPartition::build_chunk_postings`]. The
/// metadata vecs are `Arc`d so each chunk's blocking build shares them cheaply.
struct ChunkBuildState {
    offsets: Option<Arc<Vec<usize>>>,
    max_scores: Option<Arc<Vec<f32>>>,
    lengths: Option<Arc<Vec<u32>>>,
    posting_tail_codec: PostingTailCodec,
    positions_layout: PositionsLayout,
}

/// Chunk-invariant inputs to [`InvertedPartition::build_prewarm_posting_lists_chunk`]:
/// the per-partition codec/layout and the (shared, whole-partition) metadata
/// slices indexed by global token id. These don't change across chunks.
struct PrewarmBuildCtx<'a> {
    max_scores: Option<&'a [f32]>,
    lengths: Option<&'a [u32]>,
    posting_tail_codec: PostingTailCodec,
    positions_layout: PositionsLayout,
}

/// Per-chunk inputs to [`InvertedPartition::build_prewarm_posting_lists_chunk`]:
/// the token sub-range `[tok_start, tok_start + token_count)` and, for legacy
/// v1, the rebased offset slice plus the chunk's end row.
struct PrewarmChunk<'a> {
    tok_start: usize,
    token_count: usize,
    /// Legacy v1 only: `offsets[tok_start..tok_start+token_count]` (no sentinel).
    offsets: Option<&'a [usize]>,
    /// Legacy v1 only: global row at which this chunk's posting rows end.
    end_row: usize,
}

/// New type just to allow Positions implement DeepSizeOf so it can be put
/// in the cache.
#[derive(Clone)]
pub struct Positions(pub(super) CompressedPositionStorage);

/// Slice-aware cache-size charge for the Arrow array shapes stored in posting
/// caches. [`Array::get_buffer_memory_size`] reports the full capacity of shared
/// backing buffers; cached posting lists often reference only a small slice of a
/// group read. Count the referenced span for the known posting-list types and
/// fall back to Arrow's full-buffer size for anything else.
fn sliced_cache_bytes(array: &dyn Array) -> usize {
    let validity_bytes = array
        .nulls()
        .map(|nulls| nulls.len().div_ceil(8))
        .unwrap_or(0);
    match array.data_type() {
        DataType::LargeBinary => {
            let array = array.as_binary::<i64>();
            let data_bytes = if array.is_empty() {
                0
            } else {
                let offsets = array.value_offsets();
                (offsets[array.len()] - offsets[0]) as usize
            };
            data_bytes + (array.len() + 1) * std::mem::size_of::<i64>() + validity_bytes
        }
        DataType::List(_) => {
            let array = array.as_list::<i32>();
            let (child_start, child_end) = if array.is_empty() {
                (0, 0)
            } else {
                let offsets = array.value_offsets();
                (offsets[0] as usize, offsets[array.len()] as usize)
            };
            let offset_bytes = (array.len() + 1) * std::mem::size_of::<i32>();
            let child = array.values().slice(child_start, child_end - child_start);
            offset_bytes + validity_bytes + sliced_cache_bytes(child.as_ref())
        }
        // Fixed-width primitives hold exactly `len * width` bytes regardless of
        // buffer capacity, so this is already slice-aware. Any other type falls
        // back to the full-buffer size.
        other => match other.primitive_width() {
            Some(width) => array.len() * width + validity_bytes,
            None => array.get_buffer_memory_size(),
        },
    }
}

impl DeepSizeOf for Positions {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

// Cache key implementations for type-safe cache access
#[derive(Debug, Clone)]
pub struct PostingListKey {
    pub token_id: u32,
}

impl CacheKey for PostingListKey {
    type ValueType = PostingList;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("postings-{}", self.token_id).into()
    }

    fn type_name() -> &'static str {
        "PostingList"
    }

    fn codec() -> Option<CacheCodec> {
        Some(CacheCodec::from_impl::<PostingList>())
    }
}

/// Cache key for a group of consecutive posting lists stored as a single
/// entry, covering rows `[start, end)` (issue #7040). The range, not a token
/// id, is the key so that a write-time config change that reshapes groups
/// simply misses old entries instead of serving a differently-shaped group.
#[derive(Debug, Clone)]
pub struct PostingListGroupKey {
    pub start: u32,
    pub end: u32,
}

impl CacheKey for PostingListGroupKey {
    type ValueType = PostingListGroup;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("postings-{}-{}", self.start, self.end).into()
    }

    fn type_name() -> &'static str {
        "PostingListGroup"
    }

    fn codec() -> Option<CacheCodec> {
        Some(CacheCodec::from_impl::<PostingListGroup>())
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
struct PostingMetadataValue {
    max_score: f32,
    length: u32,
}

#[derive(Debug, Clone)]
struct PostingMetadataKey {
    token_id: u32,
}

impl CacheKey for PostingMetadataKey {
    type ValueType = PostingMetadataValue;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("posting-metadata-{}", self.token_id).into()
    }

    fn type_name() -> &'static str {
        "PostingMetadata"
    }
}

#[derive(Debug, Clone)]
struct V3PostingBlockKey {
    block_row: usize,
    with_position: bool,
}

impl CacheKey for V3PostingBlockKey {
    type ValueType = RecordBatch;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!(
            "v3-posting-block-{}-{}",
            self.block_row,
            if self.with_position { "pos" } else { "nopos" }
        )
        .into()
    }

    fn type_name() -> &'static str {
        "V3PostingBlock"
    }
}

#[derive(Debug, Clone)]
struct V3BlockMetadataKey {
    token_id: u32,
}

impl CacheKey for V3BlockMetadataKey {
    type ValueType = Vec<V3BlockSkipMeta>;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("v3-block-metadata-{}", self.token_id).into()
    }

    fn type_name() -> &'static str {
        "V3BlockMetadata"
    }
}

#[derive(Debug, Clone)]
struct V3PositionBlockKey {
    block_row: usize,
}

impl CacheKey for V3PositionBlockKey {
    type ValueType = bytes::Bytes;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("v3-position-block-{}", self.block_row).into()
    }

    fn type_name() -> &'static str {
        "V3PositionBlock"
    }
}

#[derive(Debug, Clone)]
pub struct PositionKey {
    pub token_id: u32,
}

impl CacheKey for PositionKey {
    type ValueType = Positions;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("positions-{}", self.token_id).into()
    }

    fn type_name() -> &'static str {
        "Position"
    }

    fn codec() -> Option<CacheCodec> {
        Some(CacheCodec::from_impl::<Positions>())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompressedPositionStorage {
    LegacyPerDoc(ListArray),
    SharedStream(SharedPositionStream),
}

impl DeepSizeOf for CompressedPositionStorage {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        match self {
            Self::LegacyPerDoc(positions) => sliced_cache_bytes(positions),
            Self::SharedStream(stream) => stream.size(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SharedPositionStream {
    codec: PositionStreamCodec,
    block_offsets: Arc<[u32]>,
    // Stored with shared ownership so cache hits can clone position streams
    // without copying either offsets or bytes.
    bytes: bytes::Bytes,
}

impl SharedPositionStream {
    pub fn new(codec: PositionStreamCodec, block_offsets: Vec<u32>, bytes: bytes::Bytes) -> Self {
        Self {
            codec,
            block_offsets: Arc::from(block_offsets.into_boxed_slice()),
            bytes,
        }
    }

    pub fn codec(&self) -> PositionStreamCodec {
        self.codec
    }

    pub fn block_count(&self) -> usize {
        self.block_offsets.len()
    }

    pub fn block_range(&self, index: usize) -> Range<usize> {
        let start = self.block_offsets[index] as usize;
        let end = self
            .block_offsets
            .get(index + 1)
            .map(|offset| *offset as usize)
            .unwrap_or(self.bytes.len());
        start..end
    }

    pub fn block(&self, index: usize) -> &[u8] {
        let range = self.block_range(index);
        &self.bytes[range]
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn block_offsets(&self) -> &[u32] {
        self.block_offsets.as_ref()
    }

    pub fn size(&self) -> usize {
        self.block_offsets.len() * std::mem::size_of::<u32>() + self.bytes.len()
    }
}

/// A group of consecutive posting lists held in a single cache entry, in row
/// order (issue #7040). `posting_lists[i]` corresponds to row `start + i`,
/// where `start` is the group's first row from [`PostingListGroupKey`].
#[derive(Debug, Clone, DeepSizeOf)]
pub struct PostingListGroup {
    pub(super) posting_lists: Vec<PostingList>,
}

impl PostingListGroup {
    pub(super) fn new(posting_lists: Vec<PostingList>) -> Self {
        Self { posting_lists }
    }

    /// Borrow the posting list at offset `slot` within the group (i.e.
    /// `token_id - start`).
    pub(super) fn get(&self, slot: usize) -> Option<&PostingList> {
        self.posting_lists.get(slot)
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub enum PostingList {
    Plain(PlainPostingList),
    Compressed(CompressedPostingList),
}

impl PostingList {
    pub fn from_batch(
        batch: &RecordBatch,
        max_score: Option<f32>,
        length: Option<u32>,
    ) -> Result<Self> {
        let posting_tail_codec = parse_posting_tail_codec(batch.schema_ref().metadata())?;
        Self::from_batch_with_tail_codec(batch, max_score, length, posting_tail_codec)
    }

    pub fn from_batch_with_tail_codec(
        batch: &RecordBatch,
        max_score: Option<f32>,
        length: Option<u32>,
        posting_tail_codec: PostingTailCodec,
    ) -> Result<Self> {
        let positions_layout = if batch.column_by_name(COMPRESSED_POSITION_COL).is_some() {
            PositionsLayout::SharedStream(parse_shared_position_codec(
                batch.schema_ref().metadata(),
            )?)
        } else if batch.column_by_name(POSITION_COL).is_some() {
            PositionsLayout::LegacyPerDoc
        } else {
            PositionsLayout::None
        };
        Self::from_batch_with_tail_codec_and_positions_layout(
            batch,
            max_score,
            length,
            posting_tail_codec,
            positions_layout,
        )
    }

    fn from_batch_with_tail_codec_and_positions_layout(
        batch: &RecordBatch,
        max_score: Option<f32>,
        length: Option<u32>,
        posting_tail_codec: PostingTailCodec,
        positions_layout: PositionsLayout,
    ) -> Result<Self> {
        match batch.column_by_name(POSTING_COL) {
            Some(_) => {
                debug_assert!(max_score.is_some() && length.is_some());
                let shared_position_codec = match positions_layout {
                    PositionsLayout::SharedStream(codec) => Some(codec),
                    _ => None,
                };
                let posting = CompressedPostingList::from_batch(
                    batch,
                    max_score.unwrap(),
                    length.unwrap(),
                    posting_tail_codec,
                    shared_position_codec,
                );
                Ok(Self::Compressed(posting))
            }
            None => {
                let posting = PlainPostingList::from_batch(batch, max_score);
                Ok(Self::Plain(posting))
            }
        }
    }

    pub fn iter(&self) -> PostingListIterator<'_> {
        PostingListIterator::new(self)
    }

    pub fn has_position(&self) -> bool {
        match self {
            Self::Plain(posting) => posting.positions.is_some(),
            Self::Compressed(posting) => posting.positions.is_some(),
        }
    }

    pub fn set_positions(&mut self, positions: CompressedPositionStorage) {
        match self {
            Self::Plain(posting) => match positions {
                CompressedPositionStorage::LegacyPerDoc(positions) => {
                    posting.positions = Some(positions)
                }
                CompressedPositionStorage::SharedStream(_) => {
                    unreachable!("shared position stream is not supported for plain postings")
                }
            },
            Self::Compressed(posting) => {
                posting.positions = Some(positions);
            }
        }
    }

    pub fn take_positions(&mut self) -> Option<CompressedPositionStorage> {
        match self {
            Self::Plain(posting) => posting
                .positions
                .take()
                .map(CompressedPositionStorage::LegacyPerDoc),
            Self::Compressed(posting) => posting.positions.take(),
        }
    }

    pub fn max_score(&self) -> Option<f32> {
        match self {
            Self::Plain(posting) => posting.max_score,
            Self::Compressed(posting) => Some(posting.max_score),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Plain(posting) => posting.len(),
            Self::Compressed(posting) => posting.length as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn into_builder(self, docs: &DocSet) -> PostingListBuilder {
        let posting_tail_codec = match &self {
            Self::Plain(_) => PostingTailCodec::Fixed32,
            Self::Compressed(posting) => posting.posting_tail_codec,
        };
        let mut builder = PostingListBuilder::new_with_posting_tail_codec(
            self.has_position(),
            posting_tail_codec,
        );
        match self {
            // legacy format
            Self::Plain(posting) => {
                // convert the posting list to the new format:
                // 1. map row ids to doc ids
                // 2. sort the posting list by doc ids
                struct Item {
                    doc_id: u32,
                    positions: PositionRecorder,
                }
                let doc_ids = docs
                    .row_ids
                    .iter()
                    .enumerate()
                    .map(|(doc_id, row_id)| (*row_id, doc_id as u32))
                    .collect::<HashMap<_, _>>();
                let mut items = Vec::with_capacity(posting.len());
                for (row_id, freq, positions) in posting.iter() {
                    let freq = freq as u32;
                    let positions = match positions {
                        Some(positions) => {
                            PositionRecorder::Position(positions.collect::<Vec<_>>().into())
                        }
                        None => PositionRecorder::Count(freq),
                    };
                    items.push(Item {
                        doc_id: doc_ids[&row_id],
                        positions,
                    });
                }
                items.sort_unstable_by_key(|item| item.doc_id);
                for item in items {
                    builder.add(item.doc_id, item.positions);
                }
            }
            Self::Compressed(posting) => {
                posting.iter().for_each(|(doc_id, freq, positions)| {
                    let positions = match positions {
                        Some(positions) => {
                            PositionRecorder::Position(positions.collect::<Vec<_>>().into())
                        }
                        None => PositionRecorder::Count(freq),
                    };
                    builder.add(doc_id, positions);
                });
            }
        }
        builder
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct PlainPostingList {
    pub row_ids: ScalarBuffer<u64>,
    pub frequencies: ScalarBuffer<f32>,
    pub max_score: Option<f32>,
    pub positions: Option<ListArray>, // List of Int32
}

impl DeepSizeOf for PlainPostingList {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        self.row_ids.len() * std::mem::size_of::<u64>()
            + self.frequencies.len() * std::mem::size_of::<f32>()
            + self
                .positions
                .as_ref()
                .map(|positions| sliced_cache_bytes(positions))
                .unwrap_or(0)
    }
}

impl PlainPostingList {
    pub fn new(
        row_ids: ScalarBuffer<u64>,
        frequencies: ScalarBuffer<f32>,
        max_score: Option<f32>,
        positions: Option<ListArray>,
    ) -> Self {
        Self {
            row_ids,
            frequencies,
            max_score,
            positions,
        }
    }

    pub fn from_batch(batch: &RecordBatch, max_score: Option<f32>) -> Self {
        let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values().clone();
        let frequencies = batch[FREQUENCY_COL]
            .as_primitive::<Float32Type>()
            .values()
            .clone();
        let positions = batch
            .column_by_name(POSITION_COL)
            .map(|col| col.as_list::<i32>().clone());

        Self::new(row_ids, frequencies, max_score, positions)
    }

    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> PlainPostingListIterator<'_> {
        Box::new(
            self.row_ids
                .iter()
                .zip(self.frequencies.iter())
                .enumerate()
                .map(|(idx, (doc_id, freq))| {
                    (
                        *doc_id,
                        *freq,
                        self.positions.as_ref().map(|p| {
                            let start = p.value_offsets()[idx] as usize;
                            let end = p.value_offsets()[idx + 1] as usize;
                            Box::new(
                                p.values().as_primitive::<Int32Type>().values()[start..end]
                                    .iter()
                                    .map(|pos| *pos as u32),
                            ) as _
                        }),
                    )
                }),
        )
    }

    #[inline]
    pub fn doc(&self, i: usize) -> LocatedDocInfo {
        LocatedDocInfo::new(self.row_ids[i], self.frequencies[i])
    }

    pub fn positions(&self, index: usize) -> Option<Arc<dyn Array>> {
        self.positions
            .as_ref()
            .map(|positions| positions.value(index))
    }

    pub fn max_score(&self) -> Option<f32> {
        self.max_score
    }

    pub fn row_id(&self, i: usize) -> u64 {
        self.row_ids[i]
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct CompressedPostingList {
    pub max_score: f32,
    pub length: u32,
    // each binary is a block of compressed data
    // that contains `BLOCK_SIZE` doc ids and then `BLOCK_SIZE` frequencies
    pub blocks: LargeBinaryArray,
    pub posting_tail_codec: PostingTailCodec,
    pub positions: Option<CompressedPositionStorage>,
}

impl DeepSizeOf for CompressedPostingList {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        sliced_cache_bytes(&self.blocks)
            + self
                .positions
                .as_ref()
                .map(|positions| positions.deep_size_of_children(context))
                .unwrap_or(0)
    }
}

impl CompressedPostingList {
    pub fn new(
        blocks: LargeBinaryArray,
        max_score: f32,
        length: u32,
        posting_tail_codec: PostingTailCodec,
        positions: Option<CompressedPositionStorage>,
    ) -> Self {
        Self {
            max_score,
            length,
            blocks,
            posting_tail_codec,
            positions,
        }
    }

    pub fn from_batch(
        batch: &RecordBatch,
        max_score: f32,
        length: u32,
        posting_tail_codec: PostingTailCodec,
        shared_position_codec: Option<PositionStreamCodec>,
    ) -> Self {
        let blocks = match batch[POSTING_COL].data_type() {
            DataType::List(_) => {
                debug_assert_eq!(batch.num_rows(), 1);
                batch[POSTING_COL]
                    .as_list::<i32>()
                    .value(0)
                    .as_binary::<i64>()
                    .clone()
            }
            DataType::LargeBinary if batch.column_by_name(FIRST_DOC_ID_COL).is_some() => {
                rehydrate_block_row_postings(batch, length, posting_tail_codec)
            }
            DataType::LargeBinary => batch[POSTING_COL].as_binary::<i64>().clone(),
            data_type => {
                panic!("unsupported compressed posting column type {data_type}");
            }
        };
        let positions = if let Some(col) = batch.column_by_name(COMPRESSED_POSITION_COL) {
            let codec = shared_position_codec.unwrap_or_else(|| {
                parse_shared_position_codec(batch.schema_ref().metadata())
                    .expect("shared position stream codec metadata should be valid")
            });
            let positions = col.as_binary::<i64>();
            let (block_offsets, bytes) =
                if let Some(offset_col) = batch.column_by_name(POSITION_BLOCK_OFFSET_COL) {
                    debug_assert_eq!(batch.num_rows(), 1);
                    (
                        offset_col
                            .as_list::<i32>()
                            .value(0)
                            .as_primitive::<UInt32Type>()
                            .values()
                            .to_vec(),
                        bytes::Bytes::from(positions.value(0).to_vec()),
                    )
                } else {
                    let mut offsets = Vec::with_capacity(positions.len());
                    let mut bytes = Vec::new();
                    for block_idx in 0..positions.len() {
                        offsets.push(bytes.len() as u32);
                        bytes.extend_from_slice(positions.value(block_idx));
                    }
                    (offsets, bytes::Bytes::from(bytes))
                };
            Some(CompressedPositionStorage::SharedStream(
                SharedPositionStream::new(codec, block_offsets, bytes),
            ))
        } else {
            batch.column_by_name(POSITION_COL).map(|col| {
                CompressedPositionStorage::LegacyPerDoc(
                    col.as_list::<i32>().value(0).as_list::<i32>().clone(),
                )
            })
        };

        Self {
            max_score,
            length,
            blocks,
            posting_tail_codec,
            positions,
        }
    }

    pub fn iter(&self) -> CompressedPostingListIterator {
        CompressedPostingListIterator::new(
            self.length as usize,
            self.blocks.clone(),
            self.posting_tail_codec,
            self.positions.clone(),
        )
    }

    pub fn block_max_score(&self, block_idx: usize) -> f32 {
        let block = self.blocks.value(block_idx);
        block[0..4].try_into().map(f32::from_le_bytes).unwrap()
    }

    pub fn block_least_doc_id(&self, block_idx: usize) -> u32 {
        let block = self.blocks.value(block_idx);
        let remainder = self.length as usize % BLOCK_SIZE;
        let is_remainder_block = remainder > 0 && block_idx + 1 == self.blocks.len();
        if is_remainder_block {
            super::encoding::read_posting_tail_first_doc(block, self.posting_tail_codec)
        } else {
            block[4..8].try_into().map(u32::from_le_bytes).unwrap()
        }
    }
}

fn rehydrate_block_row_postings(
    batch: &RecordBatch,
    length: u32,
    posting_tail_codec: PostingTailCodec,
) -> LargeBinaryArray {
    let payloads = batch[POSTING_COL].as_binary::<i64>();
    let first_doc_ids = batch[FIRST_DOC_ID_COL].as_primitive::<UInt32Type>();
    let block_max_scores = batch[BLOCK_MAX_SCORE_COL].as_primitive::<Float32Type>();
    let mut builder = LargeBinaryBuilder::with_capacity(
        payloads.len(),
        payloads.value_data().len() + payloads.len() * 8,
    );
    let remainder = length as usize % BLOCK_SIZE;
    for block_idx in 0..payloads.len() {
        let payload = payloads.value(block_idx);
        let mut block = Vec::with_capacity(payload.len() + 8);
        block.extend_from_slice(&block_max_scores.value(block_idx).to_le_bytes());
        let is_remainder_block = remainder > 0 && block_idx + 1 == payloads.len();
        if is_remainder_block && posting_tail_codec == PostingTailCodec::VarintDelta {
            append_varint_u32(first_doc_ids.value(block_idx), &mut block);
        } else {
            block.extend_from_slice(&first_doc_ids.value(block_idx).to_le_bytes());
        }
        block.extend_from_slice(payload);
        builder.append_value(block);
    }
    builder.finish()
}

fn append_varint_u32(mut value: u32, output: &mut Vec<u8>) {
    while value >= 0x80 {
        output.push((value as u8) | 0x80);
        value >>= 7;
    }
    output.push(value as u8);
}

fn encoded_varint_u32_len(bytes: &[u8]) -> Result<usize> {
    for (idx, byte) in bytes.iter().enumerate() {
        if byte & 0x80 == 0 {
            return Ok(idx + 1);
        }
    }
    Err(Error::index(
        "posting remainder payload ended before first doc id varint completed".to_owned(),
    ))
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct EncodedBlocks {
    offsets: Vec<u32>,
    bytes: Vec<u8>,
}

impl EncodedBlocks {
    fn len(&self) -> usize {
        self.offsets.len()
    }

    fn size(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<u32>() + self.bytes.capacity()
    }

    fn push_full_block(&mut self, doc_ids: &[u32], frequencies: &[u32]) -> Result<usize> {
        let start = self.bytes.len();
        self.offsets.push(start as u32);
        super::encoding::encode_full_posting_block_into(doc_ids, frequencies, &mut self.bytes)?;
        Ok(self.bytes.len() - start)
    }

    fn block(&self, index: usize) -> &[u8] {
        let (start, end) = self.block_range(index);
        &self.bytes[start..end]
    }

    fn block_range(&self, index: usize) -> (usize, usize) {
        let start = self.offsets[index] as usize;
        let end = self
            .offsets
            .get(index + 1)
            .map(|offset| *offset as usize)
            .unwrap_or(self.bytes.len());
        (start, end)
    }

    fn set_block_score(&mut self, index: usize, score: f32) {
        let (start, _) = self.block_range(index);
        self.bytes[start..start + 4].copy_from_slice(&score.to_le_bytes());
    }

    fn append_remainder_block_with_codec(
        &mut self,
        doc_ids: &[u32],
        frequencies: &[u32],
        codec: PostingTailCodec,
    ) -> Result<()> {
        self.offsets.push(self.bytes.len() as u32);
        super::encoding::encode_remainder_posting_block_into(
            doc_ids,
            frequencies,
            codec,
            &mut self.bytes,
        )
    }

    fn into_array(mut self) -> LargeBinaryArray {
        let mut offsets = Vec::with_capacity(self.offsets.len() + 1);
        offsets.extend(self.offsets.into_iter().map(i64::from));
        offsets.push(self.bytes.len() as i64);
        LargeBinaryArray::new(
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Buffer::from_vec(std::mem::take(&mut self.bytes)),
            None,
        )
    }

    fn iter(&self) -> impl Iterator<Item = &[u8]> {
        (0..self.len()).map(|index| self.block(index))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct EncodedPositionBlocks {
    offsets: Vec<u32>,
    bytes: Vec<u8>,
}

impl EncodedPositionBlocks {
    fn size(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<u32>() + self.bytes.capacity()
    }

    fn block(&self, index: usize) -> &[u8] {
        let start = self.offsets[index] as usize;
        let end = self
            .offsets
            .get(index + 1)
            .map(|offset| *offset as usize)
            .unwrap_or(self.bytes.len());
        &self.bytes[start..end]
    }

    fn push_encoded_block(&mut self, block: &[u8]) -> usize {
        let start = self.bytes.len();
        self.offsets.push(start as u32);
        self.bytes.extend_from_slice(block);
        self.bytes.len() - start
    }

    fn into_stream(self) -> SharedPositionStream {
        SharedPositionStream::new(
            PositionStreamCodec::PackedDelta,
            self.offsets,
            bytes::Bytes::from(self.bytes),
        )
    }
}

#[derive(Debug)]
pub struct PostingListBuilder {
    with_positions: bool,
    posting_tail_codec: PostingTailCodec,
    encoded_blocks: Option<Box<EncodedBlocks>>,
    encoded_position_blocks: Option<Box<EncodedPositionBlocks>>,
    tail_entries: Vec<RawDocInfo>,
    tail_positions: PositionBlockBuilder,
    open_doc_id: Option<u32>,
    open_doc_frequency: u32,
    open_doc_last_position: Option<u32>,
    memory_size_bytes: u32,
    len: u32,
}

pub(super) enum PostingListBatchBuilder {
    TokenRows(TokenRowPostingListBatchBuilder),
}

pub(super) enum PostingWriteMetadata {
    TokenRows { group_starts: Vec<u32> },
}

pub(super) struct TokenRowPostingListBatchBuilder {
    schema: SchemaRef,
    postings: ListBuilder<LargeBinaryBuilder>,
    max_scores: Float32Builder,
    lengths: UInt32Builder,
    positions: BatchPositionsBuilder,
    len: usize,
    /// Tracks posting-list cache-group boundaries in row order across all
    /// batches this builder produces (issue #7040). Outlives `finish`, which
    /// only resets the per-batch column builders.
    group_accumulator: PostingGroupAccumulator,
}

pub(super) struct V3PostingLayoutBatch {
    pub(super) blocks: RecordBatch,
    pub(super) positions: Option<RecordBatch>,
    pub(super) coarse_skip: Option<RecordBatch>,
}

pub(super) struct V3PostingLayoutBatchBuilder {
    block_schema: SchemaRef,
    position_schema: Option<SchemaRef>,
    coarse_schema: SchemaRef,
    first_doc_ids: UInt32Builder,
    last_doc_ids: UInt32Builder,
    block_max_scores: Float32Builder,
    postings: LargeBinaryBuilder,
    position_payloads: Option<LargeBinaryBuilder>,
    coarse_block_starts: UInt64Builder,
    coarse_num_blocks: UInt32Builder,
    coarse_first_doc_ids: UInt32Builder,
    coarse_last_doc_ids: UInt32Builder,
    coarse_max_scores: Float32Builder,
    term_lengths: Vec<u32>,
    term_max_scores: Vec<f32>,
    term_block_starts: Vec<u64>,
    term_num_blocks: Vec<u32>,
    posting_tail_codec: PostingTailCodec,
    len: usize,
    coarse_len: usize,
    next_block_row: u64,
}

enum BatchPositionsBuilder {
    None,
    Legacy(ListBuilder<ListBuilder<LargeBinaryBuilder>>),
    Shared {
        bytes: LargeBinaryBuilder,
        block_offsets: ListBuilder<UInt32Builder>,
    },
}

struct PostingListParts<'a> {
    with_positions: bool,
    posting_tail_codec: PostingTailCodec,
    length: usize,
    encoded_blocks: EncodedBlocks,
    encoded_position_blocks: EncodedPositionBlocks,
    tail_entries: &'a [RawDocInfo],
    tail_position_block: Option<Vec<u8>>,
}

impl PostingListBatchBuilder {
    pub fn new(
        schema: SchemaRef,
        with_positions: bool,
        format_version: InvertedListFormatVersion,
        capacity: usize,
        group_config: PostingGroupConfig,
    ) -> Self {
        debug_assert!(!format_version.uses_block_row_layout());
        Self::TokenRows(TokenRowPostingListBatchBuilder::new(
            schema,
            with_positions,
            format_version,
            capacity,
            group_config,
        ))
    }

    pub fn len(&self) -> usize {
        match self {
            Self::TokenRows(builder) => builder.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn append(
        &mut self,
        compressed: LargeBinaryArray,
        max_score: f32,
        length: u32,
        positions: Option<&CompressedPositionStorage>,
    ) -> Result<()> {
        match self {
            Self::TokenRows(builder) => builder.append(compressed, max_score, length, positions),
        }
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        match self {
            Self::TokenRows(builder) => builder.finish(),
        }
    }

    pub fn into_write_metadata(self) -> PostingWriteMetadata {
        match self {
            Self::TokenRows(builder) => PostingWriteMetadata::TokenRows {
                group_starts: builder.into_group_starts(),
            },
        }
    }
}

impl TokenRowPostingListBatchBuilder {
    fn new(
        schema: SchemaRef,
        with_positions: bool,
        format_version: InvertedListFormatVersion,
        capacity: usize,
        group_config: PostingGroupConfig,
    ) -> Self {
        let positions = if !with_positions {
            BatchPositionsBuilder::None
        } else if format_version.uses_shared_position_stream() {
            BatchPositionsBuilder::Shared {
                bytes: LargeBinaryBuilder::with_capacity(capacity, 0),
                block_offsets: ListBuilder::with_capacity(UInt32Builder::new(), capacity),
            }
        } else {
            BatchPositionsBuilder::Legacy(ListBuilder::with_capacity(
                ListBuilder::new(LargeBinaryBuilder::new()),
                capacity,
            ))
        };
        Self {
            schema,
            postings: ListBuilder::with_capacity(LargeBinaryBuilder::new(), capacity),
            max_scores: Float32Builder::with_capacity(capacity),
            lengths: UInt32Builder::with_capacity(capacity),
            positions,
            len: 0,
            group_accumulator: PostingGroupAccumulator::new(group_config),
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn append(
        &mut self,
        compressed: LargeBinaryArray,
        max_score: f32,
        length: u32,
        positions: Option<&CompressedPositionStorage>,
    ) -> Result<()> {
        let posting_bytes = compressed.value_data().len();
        {
            let values = self.postings.values();
            for index in 0..compressed.len() {
                values.append_value(compressed.value(index));
            }
        }
        self.postings.append(true);
        self.group_accumulator.push(posting_bytes);
        self.max_scores.append_value(max_score);
        self.lengths.append_value(length);

        match &mut self.positions {
            BatchPositionsBuilder::None => {}
            BatchPositionsBuilder::Shared {
                bytes,
                block_offsets,
            } => {
                let positions = positions.ok_or_else(|| {
                    Error::index(format!(
                        "positions builder missing position data for posting length {}",
                        length
                    ))
                })?;
                let CompressedPositionStorage::SharedStream(positions) = positions else {
                    return Err(Error::index(
                        "shared positions builder received legacy positions".to_owned(),
                    ));
                };
                bytes.append_value(positions.bytes());
                let offsets_builder = block_offsets.values();
                for &offset in positions.block_offsets() {
                    offsets_builder.append_value(offset);
                }
                block_offsets.append(true);
            }
            BatchPositionsBuilder::Legacy(position_lists) => {
                let positions = positions.ok_or_else(|| {
                    Error::index(format!(
                        "positions builder missing position data for posting length {}",
                        length
                    ))
                })?;
                let CompressedPositionStorage::LegacyPerDoc(positions) = positions else {
                    return Err(Error::index(
                        "legacy positions builder received shared position stream".to_owned(),
                    ));
                };
                let docs_builder = position_lists.values();
                for doc_idx in 0..positions.len() {
                    let doc_positions = positions.value(doc_idx);
                    let compressed_positions = doc_positions.as_binary::<i64>();
                    for block_idx in 0..compressed_positions.len() {
                        docs_builder
                            .values()
                            .append_value(compressed_positions.value(block_idx));
                    }
                    docs_builder.append(true);
                }
                position_lists.append(true);
            }
        }

        self.len += 1;
        Ok(())
    }

    fn finish(&mut self) -> Result<RecordBatch> {
        let mut columns = vec![
            Arc::new(self.postings.finish()) as ArrayRef,
            Arc::new(self.max_scores.finish()) as ArrayRef,
            Arc::new(self.lengths.finish()) as ArrayRef,
        ];
        match &mut self.positions {
            BatchPositionsBuilder::None => {}
            BatchPositionsBuilder::Legacy(position_lists) => {
                columns.push(Arc::new(position_lists.finish()) as ArrayRef);
            }
            BatchPositionsBuilder::Shared {
                bytes,
                block_offsets,
            } => {
                columns.push(Arc::new(bytes.finish()) as ArrayRef);
                columns.push(Arc::new(block_offsets.finish()) as ArrayRef);
            }
        }
        self.len = 0;
        RecordBatch::try_new(self.schema.clone(), columns).map_err(Error::from)
    }

    /// Consume the builder and return the posting-list cache-group boundaries
    /// accumulated across all batches (issue #7040). Each entry is the first
    /// row of a group; the sequence is monotonically increasing.
    fn into_group_starts(self) -> Vec<u32> {
        self.group_accumulator.into_starts()
    }
}

fn stripped_posting_payload_start(
    block: &[u8],
    is_remainder_block: bool,
    posting_tail_codec: PostingTailCodec,
) -> Result<usize> {
    if is_remainder_block && posting_tail_codec == PostingTailCodec::VarintDelta {
        Ok(4 + encoded_varint_u32_len(&block[4..])?)
    } else {
        Ok(8)
    }
}

impl V3PostingLayoutBatchBuilder {
    pub(super) fn new(
        block_schema: SchemaRef,
        position_schema: Option<SchemaRef>,
        coarse_schema: SchemaRef,
        posting_tail_codec: PostingTailCodec,
        capacity: usize,
    ) -> Self {
        Self {
            block_schema,
            position_schema: position_schema.clone(),
            coarse_schema,
            first_doc_ids: UInt32Builder::with_capacity(capacity),
            last_doc_ids: UInt32Builder::with_capacity(capacity),
            block_max_scores: Float32Builder::with_capacity(capacity),
            postings: LargeBinaryBuilder::with_capacity(capacity, 0),
            position_payloads: position_schema
                .is_some()
                .then(|| LargeBinaryBuilder::with_capacity(capacity, 0)),
            coarse_block_starts: UInt64Builder::new(),
            coarse_num_blocks: UInt32Builder::new(),
            coarse_first_doc_ids: UInt32Builder::new(),
            coarse_last_doc_ids: UInt32Builder::new(),
            coarse_max_scores: Float32Builder::new(),
            term_lengths: Vec::new(),
            term_max_scores: Vec::new(),
            term_block_starts: Vec::new(),
            term_num_blocks: Vec::new(),
            posting_tail_codec,
            len: 0,
            coarse_len: 0,
            next_block_row: 0,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn append(
        &mut self,
        compressed: LargeBinaryArray,
        max_score: f32,
        length: u32,
        positions: Option<&CompressedPositionStorage>,
    ) -> Result<()> {
        let position_stream = match (&mut self.position_payloads, positions) {
            (None, _) => None,
            (Some(_), Some(CompressedPositionStorage::SharedStream(stream))) => Some(stream),
            (Some(_), Some(CompressedPositionStorage::LegacyPerDoc(_))) => {
                return Err(Error::index(
                    "V3 positions writer received legacy positions".to_owned(),
                ));
            }
            (Some(_), None) => {
                return Err(Error::index(format!(
                    "V3 positions writer missing position data for posting length {}",
                    length
                )));
            }
        };

        let term_block_start = self.next_block_row;
        let num_blocks = compressed.len();
        let remainder = length as usize % BLOCK_SIZE;
        let mut doc_ids = Vec::with_capacity(BLOCK_SIZE);
        let mut frequencies = Vec::with_capacity(BLOCK_SIZE);
        let mut buffer = [0u32; BLOCK_SIZE];
        let mut coarse_first_doc_id = 0;
        let mut coarse_last_doc_id = 0;
        let mut coarse_max_score = f32::MIN;
        let mut coarse_count = 0u32;
        let mut coarse_start = term_block_start;

        for block_idx in 0..num_blocks {
            let block = compressed.value(block_idx);
            let block_max_score = block[0..4].try_into().map(f32::from_le_bytes).unwrap();
            doc_ids.clear();
            frequencies.clear();
            let is_remainder_block = remainder > 0 && block_idx + 1 == num_blocks;
            if is_remainder_block {
                super::encoding::decompress_posting_remainder(
                    block,
                    remainder,
                    self.posting_tail_codec,
                    &mut doc_ids,
                    &mut frequencies,
                );
            } else {
                super::encoding::decompress_posting_block(
                    block,
                    &mut buffer,
                    &mut doc_ids,
                    &mut frequencies,
                );
            }
            let first_doc_id = doc_ids.first().copied().ok_or_else(|| {
                Error::index(format!(
                    "posting block {} for posting length {} decoded no docs",
                    block_idx, length
                ))
            })?;
            let last_doc_id = doc_ids.last().copied().expect("first_doc_id exists");
            self.first_doc_ids.append_value(first_doc_id);
            self.last_doc_ids.append_value(last_doc_id);
            self.block_max_scores.append_value(block_max_score);
            let payload_start =
                stripped_posting_payload_start(block, is_remainder_block, self.posting_tail_codec)?;
            self.postings.append_value(&block[payload_start..]);
            if let (Some(payloads), Some(stream)) = (&mut self.position_payloads, position_stream) {
                payloads.append_value(stream.block(block_idx));
            }

            if coarse_count == 0 {
                coarse_start = self.next_block_row;
                coarse_first_doc_id = first_doc_id;
                coarse_max_score = block_max_score;
            } else {
                coarse_max_score = coarse_max_score.max(block_max_score);
            }
            coarse_last_doc_id = last_doc_id;
            coarse_count += 1;
            if coarse_count as usize == COARSE_SKIP_GROUP_BLOCKS {
                self.append_coarse_skip(
                    coarse_start,
                    coarse_count,
                    coarse_first_doc_id,
                    coarse_last_doc_id,
                    coarse_max_score,
                );
                coarse_count = 0;
                coarse_max_score = f32::MIN;
            }

            self.len += 1;
            self.next_block_row += 1;
        }
        if coarse_count > 0 {
            self.append_coarse_skip(
                coarse_start,
                coarse_count,
                coarse_first_doc_id,
                coarse_last_doc_id,
                coarse_max_score,
            );
        }

        self.term_lengths.push(length);
        self.term_max_scores.push(max_score);
        self.term_block_starts.push(term_block_start);
        self.term_num_blocks.push(num_blocks as u32);
        Ok(())
    }

    fn append_coarse_skip(
        &mut self,
        block_start: u64,
        num_blocks: u32,
        first_doc_id: u32,
        last_doc_id: u32,
        max_score: f32,
    ) {
        self.coarse_block_starts.append_value(block_start);
        self.coarse_num_blocks.append_value(num_blocks);
        self.coarse_first_doc_ids.append_value(first_doc_id);
        self.coarse_last_doc_ids.append_value(last_doc_id);
        self.coarse_max_scores.append_value(max_score);
        self.coarse_len += 1;
    }

    pub(super) fn finish(&mut self) -> Result<V3PostingLayoutBatch> {
        let blocks = RecordBatch::try_new(
            self.block_schema.clone(),
            vec![
                Arc::new(self.first_doc_ids.finish()) as ArrayRef,
                Arc::new(self.last_doc_ids.finish()) as ArrayRef,
                Arc::new(self.block_max_scores.finish()) as ArrayRef,
                Arc::new(self.postings.finish()) as ArrayRef,
            ],
        )?;
        let positions = match (&self.position_schema, &mut self.position_payloads) {
            (Some(schema), Some(payloads)) => Some(RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(payloads.finish()) as ArrayRef],
            )?),
            _ => None,
        };
        let coarse_skip = if self.coarse_len > 0 {
            Some(RecordBatch::try_new(
                self.coarse_schema.clone(),
                vec![
                    Arc::new(self.coarse_block_starts.finish()) as ArrayRef,
                    Arc::new(self.coarse_num_blocks.finish()) as ArrayRef,
                    Arc::new(self.coarse_first_doc_ids.finish()) as ArrayRef,
                    Arc::new(self.coarse_last_doc_ids.finish()) as ArrayRef,
                    Arc::new(self.coarse_max_scores.finish()) as ArrayRef,
                ],
            )?)
        } else {
            None
        };
        self.len = 0;
        self.coarse_len = 0;
        Ok(V3PostingLayoutBatch {
            blocks,
            positions,
            coarse_skip,
        })
    }

    pub(super) fn finish_terms(self, terms_schema: SchemaRef) -> Result<RecordBatch> {
        RecordBatch::try_new(
            terms_schema,
            vec![
                Arc::new(UInt32Array::from(self.term_lengths)) as ArrayRef,
                Arc::new(Float32Array::from(self.term_max_scores)) as ArrayRef,
                Arc::new(UInt64Array::from(self.term_block_starts)) as ArrayRef,
                Arc::new(UInt32Array::from(self.term_num_blocks)) as ArrayRef,
            ],
        )
        .map_err(Error::from)
    }
}

impl PostingListBuilder {
    pub fn size(&self) -> u64 {
        self.memory_size_bytes as u64
    }

    pub fn has_positions(&self) -> bool {
        self.with_positions
    }

    pub fn new(with_position: bool) -> Self {
        Self::new_with_posting_tail_codec(
            with_position,
            current_fts_format_version().posting_tail_codec(),
        )
    }

    pub fn new_with_posting_tail_codec(
        with_position: bool,
        posting_tail_codec: PostingTailCodec,
    ) -> Self {
        Self {
            with_positions: with_position,
            posting_tail_codec,
            encoded_blocks: None,
            encoded_position_blocks: None,
            tail_entries: Vec::new(),
            tail_positions: PositionBlockBuilder::default(),
            open_doc_id: None,
            open_doc_frequency: 0,
            open_doc_last_position: None,
            len: 0,
            memory_size_bytes: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> std::vec::IntoIter<(u32, u32, Option<Vec<u32>>)> {
        self.collect_entries().into_iter()
    }

    pub fn for_each_entry<E>(
        &self,
        mut visit: impl FnMut(u32, u32, Option<Vec<u32>>) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        let mut doc_ids = Vec::with_capacity(BLOCK_SIZE);
        let mut frequencies = Vec::with_capacity(BLOCK_SIZE);
        let mut decoded_positions = Vec::new();
        let mut position_block_index = 0usize;

        if let Some(encoded_blocks) = self.encoded_blocks.as_deref() {
            for block in encoded_blocks.iter() {
                doc_ids.clear();
                frequencies.clear();
                super::encoding::decode_full_posting_block(block, &mut doc_ids, &mut frequencies);
                decoded_positions.clear();
                if self.with_positions {
                    let position_blocks = self
                        .encoded_position_blocks
                        .as_deref()
                        .expect("positions must exist for posting list");
                    super::encoding::decode_position_stream_block(
                        position_blocks.block(position_block_index),
                        &frequencies,
                        PositionStreamCodec::PackedDelta,
                        &mut decoded_positions,
                    )
                    .expect("position stream decoding should succeed");
                    position_block_index += 1;
                }
                let mut offset = 0usize;
                for (doc_id, frequency) in doc_ids.iter().copied().zip(frequencies.iter().copied())
                {
                    let positions = self.with_positions.then(|| {
                        let end = offset + frequency as usize;
                        let doc_positions = decoded_positions[offset..end].to_vec();
                        offset = end;
                        doc_positions
                    });
                    visit(doc_id, frequency, positions)?;
                }
            }
        }

        let mut decoded_tail_positions = Vec::new();
        if self.with_positions && !self.tail_entries.is_empty() {
            let tail_frequencies = self
                .tail_entries
                .iter()
                .map(|entry| entry.frequency)
                .collect::<Vec<_>>();
            self.tail_positions
                .decode_into(tail_frequencies.as_slice(), &mut decoded_tail_positions)
                .expect("tail position stream decoding should succeed");
        }
        let mut tail_offset = 0usize;
        for entry in &self.tail_entries {
            let positions = self.with_positions.then(|| {
                let end = tail_offset + entry.frequency as usize;
                let doc_positions = decoded_tail_positions[tail_offset..end].to_vec();
                tail_offset = end;
                doc_positions
            });
            visit(entry.doc_id, entry.frequency, positions)?;
        }

        Ok(())
    }

    pub fn add(&mut self, doc_id: u32, term_positions: PositionRecorder) {
        debug_assert!(
            self.open_doc_id.is_none(),
            "cannot add closed doc while a positions doc is still open"
        );
        let tail_entries_capacity_before = self.tail_entries.capacity();
        self.tail_entries
            .push(RawDocInfo::new(doc_id, term_positions.len()));
        let tail_entries_capacity_after = self.tail_entries.capacity();
        if tail_entries_capacity_after > tail_entries_capacity_before {
            self.add_memory_bytes(
                (tail_entries_capacity_after - tail_entries_capacity_before)
                    * std::mem::size_of::<RawDocInfo>(),
            );
        }
        if let PositionRecorder::Position(positions_in_doc) = term_positions {
            debug_assert!(self.with_positions);
            let old_size = self.tail_positions.size();
            self.tail_positions
                .append_doc_positions(positions_in_doc.as_slice())
                .expect("position stream encoding should succeed");
            self.adjust_tail_positions_size(old_size);
        }
        self.len += 1;

        if self.tail_entries.len() == BLOCK_SIZE {
            self.flush_tail_block()
                .expect("posting list block compression should succeed");
        }
    }

    pub fn add_occurrence(&mut self, doc_id: u32, position: u32) -> Result<bool> {
        if !self.with_positions {
            return Err(Error::index(
                "cannot append streamed positions to a posting list without positions".to_owned(),
            ));
        }

        match self.open_doc_id {
            Some(open_doc_id) if open_doc_id == doc_id => {
                let old_size = self.tail_positions.size();
                self.tail_positions
                    .append_position(position, self.open_doc_last_position)?;
                self.adjust_tail_positions_size(old_size);
                self.open_doc_frequency += 1;
                self.open_doc_last_position = Some(position);
                Ok(false)
            }
            Some(open_doc_id) => Err(Error::index(format!(
                "posting list received doc {} before finishing open doc {}",
                doc_id, open_doc_id
            ))),
            None => {
                let old_size = self.tail_positions.size();
                self.tail_positions.append_position(position, None)?;
                self.adjust_tail_positions_size(old_size);
                self.open_doc_id = Some(doc_id);
                self.open_doc_frequency = 1;
                self.open_doc_last_position = Some(position);
                self.len += 1;
                Ok(true)
            }
        }
    }

    pub fn finish_open_doc(&mut self, doc_id: u32) -> Result<()> {
        if !self.with_positions {
            return Ok(());
        }
        match self.open_doc_id {
            Some(open_doc_id) if open_doc_id == doc_id => {
                let tail_entries_capacity_before = self.tail_entries.capacity();
                self.tail_entries
                    .push(RawDocInfo::new(doc_id, self.open_doc_frequency));
                let tail_entries_capacity_after = self.tail_entries.capacity();
                if tail_entries_capacity_after > tail_entries_capacity_before {
                    self.add_memory_bytes(
                        (tail_entries_capacity_after - tail_entries_capacity_before)
                            * std::mem::size_of::<RawDocInfo>(),
                    );
                }
                self.open_doc_id = None;
                self.open_doc_frequency = 0;
                self.open_doc_last_position = None;
                if self.tail_entries.len() == BLOCK_SIZE {
                    self.flush_tail_block()?;
                }
                Ok(())
            }
            Some(open_doc_id) => Err(Error::index(format!(
                "attempted to finish doc {} while doc {} is still open",
                doc_id, open_doc_id
            ))),
            None => Ok(()),
        }
    }

    fn collect_entries(&self) -> Vec<(u32, u32, Option<Vec<u32>>)> {
        let mut entries = Vec::with_capacity(self.len());
        self.for_each_entry(|doc_id, frequency, positions| {
            entries.push((doc_id, frequency, positions));
            Ok::<(), ()>(())
        })
        .expect("collecting posting list entries should not fail");
        entries
    }

    fn encoded_blocks_mut(&mut self) -> &mut EncodedBlocks {
        if self.encoded_blocks.is_none() {
            self.encoded_blocks = Some(Box::default());
            self.add_memory_bytes(std::mem::size_of::<EncodedBlocks>());
        }
        self.encoded_blocks
            .as_deref_mut()
            .expect("encoded blocks must exist")
    }

    fn encoded_position_blocks_mut(&mut self) -> &mut EncodedPositionBlocks {
        if self.encoded_position_blocks.is_none() {
            self.encoded_position_blocks = Some(Box::default());
            self.add_memory_bytes(std::mem::size_of::<EncodedPositionBlocks>());
        }
        self.encoded_position_blocks
            .as_deref_mut()
            .expect("encoded position blocks must exist")
    }

    fn flush_tail_block(&mut self) -> Result<()> {
        if self.tail_entries.is_empty() {
            return Ok(());
        }
        debug_assert!(
            self.open_doc_id.is_none(),
            "cannot flush a posting block while a document is still open"
        );
        debug_assert_eq!(self.tail_entries.len(), BLOCK_SIZE);
        let mut doc_ids = [0u32; BLOCK_SIZE];
        let mut frequencies = [0u32; BLOCK_SIZE];
        for (index, entry) in self.tail_entries.iter().enumerate() {
            doc_ids[index] = entry.doc_id;
            frequencies[index] = entry.frequency;
        }
        let encoded_blocks_size_before = self
            .encoded_blocks
            .as_ref()
            .map(|encoded_blocks| encoded_blocks.size())
            .unwrap_or(0usize);
        self.encoded_blocks_mut()
            .push_full_block(&doc_ids, &frequencies)?;
        let encoded_blocks_size_after = self
            .encoded_blocks
            .as_ref()
            .map(|encoded_blocks| encoded_blocks.size())
            .unwrap_or(0usize);
        if encoded_blocks_size_after > encoded_blocks_size_before {
            self.add_memory_bytes(encoded_blocks_size_after - encoded_blocks_size_before);
        }
        if self.with_positions {
            let encoded_positions_size_before = self
                .encoded_position_blocks
                .as_ref()
                .map(|encoded| encoded.size())
                .unwrap_or(0usize);
            let released_tail_positions_bytes = self.tail_positions.size();
            let tail_position_block = std::mem::take(&mut self.tail_positions).finish();
            self.encoded_position_blocks_mut()
                .push_encoded_block(tail_position_block.as_slice());
            let encoded_positions_size_after = self
                .encoded_position_blocks
                .as_ref()
                .map(|encoded| encoded.size())
                .unwrap_or(0usize);
            if released_tail_positions_bytes > 0 {
                self.subtract_memory_bytes(released_tail_positions_bytes);
            }
            if encoded_positions_size_after > encoded_positions_size_before {
                self.add_memory_bytes(encoded_positions_size_after - encoded_positions_size_before);
            }
        }
        self.tail_entries.clear();
        Ok(())
    }

    fn adjust_tail_positions_size(&mut self, old_size: usize) {
        let new_size = self.tail_positions.size();
        if new_size > old_size {
            self.add_memory_bytes(new_size - old_size);
        } else if old_size > new_size {
            self.subtract_memory_bytes(old_size - new_size);
        }
    }

    fn add_memory_bytes(&mut self, bytes: usize) {
        self.memory_size_bytes = self
            .memory_size_bytes
            .checked_add(
                u32::try_from(bytes).expect("posting list memory size delta overflowed u32"),
            )
            .expect("posting list memory size overflowed u32");
    }

    fn subtract_memory_bytes(&mut self, bytes: usize) {
        self.memory_size_bytes = self
            .memory_size_bytes
            .checked_sub(
                u32::try_from(bytes).expect("posting list memory size delta overflowed u32"),
            )
            .expect("posting list memory size underflowed u32");
    }

    fn build_position_columns(
        positions: Option<CompressedPositionStorage>,
    ) -> Result<Vec<ArrayRef>> {
        let Some(positions) = positions else {
            return Ok(Vec::new());
        };
        match positions {
            CompressedPositionStorage::LegacyPerDoc(positions) => {
                Ok(vec![Arc::new(ListArray::try_new(
                    Arc::new(Field::new("item", positions.data_type().clone(), true)),
                    OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, positions.len() as i32])),
                    Arc::new(positions) as ArrayRef,
                    None,
                )?) as ArrayRef])
            }
            CompressedPositionStorage::SharedStream(positions) => {
                let mut columns = Vec::with_capacity(2);
                columns.push(
                    Arc::new(LargeBinaryArray::from(vec![Some(positions.bytes())])) as ArrayRef,
                );

                let mut offsets_builder = ListBuilder::new(UInt32Builder::new());
                for &offset in positions.block_offsets() {
                    offsets_builder.values().append_value(offset);
                }
                offsets_builder.append(true);
                columns.push(Arc::new(offsets_builder.finish()) as ArrayRef);
                Ok(columns)
            }
        }
    }

    fn build_batch(
        self,
        compressed: LargeBinaryArray,
        max_score: f32,
        schema: SchemaRef,
        positions: Option<CompressedPositionStorage>,
    ) -> Result<RecordBatch> {
        let length = self.len();
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, compressed.len() as i32]));
        let mut columns = vec![
            Arc::new(ListArray::try_new(
                Arc::new(Field::new("item", datatypes::DataType::LargeBinary, true)),
                offsets,
                Arc::new(compressed),
                None,
            )?) as ArrayRef,
            Arc::new(Float32Array::from_iter_values(std::iter::once(max_score))) as ArrayRef,
            Arc::new(UInt32Array::from_iter_values(std::iter::once(
                length as u32,
            ))) as ArrayRef,
        ];
        columns.extend(Self::build_position_columns(positions)?);

        let batch = RecordBatch::try_new(schema, columns)?;
        Ok(batch)
    }

    fn build_legacy_positions(&self) -> Result<ListArray> {
        let mut positions_builder = ListBuilder::new(LargeBinaryBuilder::new());
        self.for_each_entry(|_doc_id, frequency, positions| {
            let positions = positions.ok_or_else(|| {
                Error::index(format!(
                    "legacy position writer missing positions for frequency {}",
                    frequency
                ))
            })?;
            let compressed = super::encoding::compress_positions(positions.as_slice())?;
            for block_idx in 0..compressed.len() {
                positions_builder
                    .values()
                    .append_value(compressed.value(block_idx));
            }
            positions_builder.append(true);
            Ok::<(), Error>(())
        })?;
        Ok(positions_builder.finish())
    }

    pub(super) fn append_to_batch_with_docs(
        self,
        docs: &DocSet,
        batch_builder: &mut PostingListBatchBuilder,
        format_version: InvertedListFormatVersion,
    ) -> Result<()> {
        let legacy_positions =
            if self.with_positions && !format_version.uses_shared_position_stream() {
                Some(self.build_legacy_positions()?)
            } else {
                None
            };
        let Self {
            with_positions,
            posting_tail_codec,
            encoded_blocks,
            encoded_position_blocks,
            tail_entries,
            tail_positions,
            open_doc_id,
            open_doc_frequency,
            open_doc_last_position,
            len,
            ..
        } = self;
        debug_assert!(open_doc_id.is_none());
        debug_assert_eq!(open_doc_frequency, 0);
        debug_assert!(open_doc_last_position.is_none());
        let parts = PostingListParts {
            with_positions,
            posting_tail_codec,
            length: len as usize,
            encoded_blocks: encoded_blocks
                .map(|encoded_blocks| *encoded_blocks)
                .unwrap_or_default(),
            encoded_position_blocks: encoded_position_blocks
                .map(|encoded_positions| *encoded_positions)
                .unwrap_or_default(),
            tail_entries: tail_entries.as_slice(),
            tail_position_block: with_positions.then(|| tail_positions.finish()),
        };
        let (compressed, shared_positions, max_score) =
            Self::build_compressed_with_scores_from_parts(parts, docs)?;
        let positions = match legacy_positions {
            Some(positions) => Some(CompressedPositionStorage::LegacyPerDoc(positions)),
            None => shared_positions.map(CompressedPositionStorage::SharedStream),
        };
        batch_builder.append(compressed, max_score, len, positions.as_ref())
    }

    pub(super) fn append_to_v3_layout_with_docs(
        self,
        docs: &DocSet,
        batch_builder: &mut V3PostingLayoutBatchBuilder,
    ) -> Result<()> {
        let Self {
            with_positions,
            posting_tail_codec,
            encoded_blocks,
            encoded_position_blocks,
            tail_entries,
            tail_positions,
            open_doc_id,
            open_doc_frequency,
            open_doc_last_position,
            len,
            ..
        } = self;
        debug_assert!(open_doc_id.is_none());
        debug_assert_eq!(open_doc_frequency, 0);
        debug_assert!(open_doc_last_position.is_none());
        let parts = PostingListParts {
            with_positions,
            posting_tail_codec,
            length: len as usize,
            encoded_blocks: encoded_blocks
                .map(|encoded_blocks| *encoded_blocks)
                .unwrap_or_default(),
            encoded_position_blocks: encoded_position_blocks
                .map(|encoded_positions| *encoded_positions)
                .unwrap_or_default(),
            tail_entries: tail_entries.as_slice(),
            tail_position_block: with_positions.then(|| tail_positions.finish()),
        };
        let (compressed, shared_positions, max_score) =
            Self::build_compressed_with_scores_from_parts(parts, docs)?;
        let positions = shared_positions.map(CompressedPositionStorage::SharedStream);
        batch_builder.append(compressed, max_score, len, positions.as_ref())
    }

    fn extend_tail_components(
        tail_entries: &[RawDocInfo],
        doc_ids: &mut Vec<u32>,
        frequencies: &mut Vec<u32>,
    ) {
        doc_ids.clear();
        frequencies.clear();
        doc_ids.extend(tail_entries.iter().map(|entry| entry.doc_id));
        frequencies.extend(tail_entries.iter().map(|entry| entry.frequency));
    }

    fn build_compressed_with_scores_from_parts(
        parts: PostingListParts<'_>,
        docs: &DocSet,
    ) -> Result<(LargeBinaryArray, Option<SharedPositionStream>, f32)> {
        let PostingListParts {
            with_positions,
            posting_tail_codec,
            length,
            mut encoded_blocks,
            mut encoded_position_blocks,
            tail_entries,
            tail_position_block,
        } = parts;
        let avgdl = docs.average_length();
        let idf_scale = idf(length, docs.len()) * (K1 + 1.0);
        let mut max_score = f32::MIN;
        let mut doc_ids = Vec::with_capacity(BLOCK_SIZE);
        let mut frequencies = Vec::with_capacity(BLOCK_SIZE);

        for index in 0..encoded_blocks.len() {
            let block = encoded_blocks.block(index);
            doc_ids.clear();
            frequencies.clear();
            super::encoding::decode_full_posting_block(block, &mut doc_ids, &mut frequencies);
            let block_score = compute_block_score(
                docs,
                avgdl,
                idf_scale,
                doc_ids.iter().copied(),
                frequencies.iter().copied(),
            );
            max_score = max_score.max(block_score);
            encoded_blocks.set_block_score(index, block_score);
        }

        if !tail_entries.is_empty() {
            Self::extend_tail_components(tail_entries, &mut doc_ids, &mut frequencies);
            let block_score = compute_block_score(
                docs,
                avgdl,
                idf_scale,
                doc_ids.iter().copied(),
                frequencies.iter().copied(),
            );
            max_score = max_score.max(block_score);
            encoded_blocks.append_remainder_block_with_codec(
                doc_ids.as_slice(),
                frequencies.as_slice(),
                posting_tail_codec,
            )?;
            encoded_blocks.set_block_score(encoded_blocks.len() - 1, block_score);
            if with_positions {
                encoded_position_blocks.push_encoded_block(
                    tail_position_block
                        .as_deref()
                        .expect("tail position block must exist for postings with positions"),
                );
            }
        }

        Ok((
            encoded_blocks.into_array(),
            with_positions.then(|| encoded_position_blocks.into_stream()),
            max_score,
        ))
    }

    fn build_compressed_with_block_scores_from_parts(
        with_positions: bool,
        posting_tail_codec: PostingTailCodec,
        mut encoded_blocks: EncodedBlocks,
        mut encoded_position_blocks: EncodedPositionBlocks,
        tail_entries: &[RawDocInfo],
        tail_position_block: Option<Vec<u8>>,
        mut block_max_scores: impl Iterator<Item = f32>,
    ) -> Result<(LargeBinaryArray, Option<SharedPositionStream>, f32)> {
        let mut max_score = f32::MIN;
        let mut doc_ids = Vec::with_capacity(BLOCK_SIZE);
        let mut frequencies = Vec::with_capacity(BLOCK_SIZE);

        for index in 0..encoded_blocks.len() {
            let block_score = block_max_scores
                .next()
                .ok_or_else(|| Error::index("missing block max score".to_owned()))?;
            max_score = max_score.max(block_score);
            encoded_blocks.set_block_score(index, block_score);
        }

        if !tail_entries.is_empty() {
            let block_score = block_max_scores
                .next()
                .ok_or_else(|| Error::index("missing tail block max score".to_owned()))?;
            max_score = max_score.max(block_score);
            Self::extend_tail_components(tail_entries, &mut doc_ids, &mut frequencies);
            encoded_blocks.append_remainder_block_with_codec(
                doc_ids.as_slice(),
                frequencies.as_slice(),
                posting_tail_codec,
            )?;
            encoded_blocks.set_block_score(encoded_blocks.len() - 1, block_score);
            if with_positions {
                encoded_position_blocks.push_encoded_block(
                    tail_position_block
                        .as_deref()
                        .expect("tail position block must exist for postings with positions"),
                );
            }
        }

        Ok((
            encoded_blocks.into_array(),
            with_positions.then(|| encoded_position_blocks.into_stream()),
            max_score,
        ))
    }

    pub fn to_batch(self, block_max_scores: Vec<f32>) -> Result<RecordBatch> {
        let format_version = if self.posting_tail_codec == PostingTailCodec::Fixed32 {
            InvertedListFormatVersion::V1
        } else {
            InvertedListFormatVersion::V2
        };
        let schema = inverted_list_schema_for_version(self.has_positions(), format_version);
        let legacy_positions =
            if self.with_positions && !format_version.uses_shared_position_stream() {
                Some(self.build_legacy_positions()?)
            } else {
                None
            };
        let Self {
            with_positions,
            posting_tail_codec,
            encoded_blocks,
            encoded_position_blocks,
            tail_entries,
            tail_positions,
            open_doc_id,
            open_doc_frequency,
            open_doc_last_position,
            len,
            ..
        } = self;
        debug_assert!(open_doc_id.is_none());
        debug_assert_eq!(open_doc_frequency, 0);
        debug_assert!(open_doc_last_position.is_none());
        let (compressed, shared_positions, max_score) =
            Self::build_compressed_with_block_scores_from_parts(
                with_positions,
                posting_tail_codec,
                encoded_blocks
                    .map(|encoded_blocks| *encoded_blocks)
                    .unwrap_or_default(),
                encoded_position_blocks
                    .map(|encoded_positions| *encoded_positions)
                    .unwrap_or_default(),
                tail_entries.as_slice(),
                with_positions.then(|| tail_positions.finish()),
                block_max_scores.into_iter(),
            )?;
        let builder = Self {
            with_positions,
            posting_tail_codec,
            encoded_blocks: None,
            encoded_position_blocks: None,
            tail_entries: Vec::new(),
            tail_positions: PositionBlockBuilder::default(),
            open_doc_id: None,
            open_doc_frequency: 0,
            open_doc_last_position: None,
            memory_size_bytes: 0,
            len,
        };
        let positions = match legacy_positions {
            Some(positions) => Some(CompressedPositionStorage::LegacyPerDoc(positions)),
            None => shared_positions.map(CompressedPositionStorage::SharedStream),
        };
        builder.build_batch(compressed, max_score, schema, positions)
    }

    pub fn to_batch_with_docs(self, docs: &DocSet, schema: SchemaRef) -> Result<RecordBatch> {
        let format_version = if schema.column_with_name(POSITION_COL).is_some()
            && schema.column_with_name(COMPRESSED_POSITION_COL).is_none()
        {
            InvertedListFormatVersion::V1
        } else {
            InvertedListFormatVersion::V2
        };
        let legacy_positions =
            if self.with_positions && !format_version.uses_shared_position_stream() {
                Some(self.build_legacy_positions()?)
            } else {
                None
            };
        let Self {
            with_positions,
            posting_tail_codec,
            encoded_blocks,
            encoded_position_blocks,
            tail_entries,
            tail_positions,
            open_doc_id,
            open_doc_frequency,
            open_doc_last_position,
            len,
            ..
        } = self;
        debug_assert!(open_doc_id.is_none());
        debug_assert_eq!(open_doc_frequency, 0);
        debug_assert!(open_doc_last_position.is_none());
        let parts = PostingListParts {
            with_positions,
            posting_tail_codec,
            length: len as usize,
            encoded_blocks: encoded_blocks
                .map(|encoded_blocks| *encoded_blocks)
                .unwrap_or_default(),
            encoded_position_blocks: encoded_position_blocks
                .map(|encoded_positions| *encoded_positions)
                .unwrap_or_default(),
            tail_entries: tail_entries.as_slice(),
            tail_position_block: with_positions.then(|| tail_positions.finish()),
        };
        let (compressed, shared_positions, max_score) =
            Self::build_compressed_with_scores_from_parts(parts, docs)?;
        let builder = Self {
            with_positions,
            posting_tail_codec,
            encoded_blocks: None,
            encoded_position_blocks: None,
            tail_entries: Vec::new(),
            tail_positions: PositionBlockBuilder::default(),
            open_doc_id: None,
            open_doc_frequency: 0,
            open_doc_last_position: None,
            memory_size_bytes: 0,
            len,
        };
        let positions = match legacy_positions {
            Some(positions) => Some(CompressedPositionStorage::LegacyPerDoc(positions)),
            None => shared_positions.map(CompressedPositionStorage::SharedStream),
        };
        builder.build_batch(compressed, max_score, schema, positions)
    }

    pub fn remap(&mut self, removed: &[u32]) {
        let mut cursor = 0;
        let mut new_builder =
            Self::new_with_posting_tail_codec(self.has_positions(), self.posting_tail_codec);
        for (doc_id, freq, positions) in self.iter() {
            while cursor < removed.len() && removed[cursor] < doc_id {
                cursor += 1;
            }
            if cursor < removed.len() && removed[cursor] == doc_id {
                continue;
            }
            let positions = match positions {
                Some(positions) => PositionRecorder::Position(positions.into()),
                None => PositionRecorder::Count(freq),
            };
            new_builder.add(doc_id - cursor as u32, positions);
        }

        *self = new_builder;
    }
}

fn compute_block_score(
    docs: &DocSet,
    avgdl: f32,
    idf_scale: f32,
    doc_ids: impl Iterator<Item = u32>,
    frequencies: impl Iterator<Item = u32>,
) -> f32 {
    let mut block_max_score = f32::MIN;
    for (doc_id, freq) in doc_ids.zip(frequencies) {
        let doc_norm = K1 * (1.0 - B + B * docs.num_tokens(doc_id) as f32 / avgdl);
        let freq = freq as f32;
        let score = freq / (freq + doc_norm);
        block_max_score = block_max_score.max(score);
    }
    block_max_score * idf_scale
}

#[derive(Debug, Clone, DeepSizeOf, Copy)]
pub enum DocInfo {
    Located(LocatedDocInfo),
    Raw(RawDocInfo),
}

impl DocInfo {
    pub fn doc_id(&self) -> u64 {
        match self {
            Self::Raw(info) => info.doc_id as u64,
            Self::Located(info) => info.row_id,
        }
    }

    pub fn frequency(&self) -> u32 {
        match self {
            Self::Raw(info) => info.frequency,
            Self::Located(info) => info.frequency as u32,
        }
    }
}

impl Eq for DocInfo {}

impl PartialEq for DocInfo {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id() == other.doc_id()
    }
}

impl PartialOrd for DocInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DocInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.doc_id().cmp(&other.doc_id())
    }
}

#[derive(Debug, Clone, Default, DeepSizeOf, Copy)]
pub struct LocatedDocInfo {
    pub row_id: u64,
    pub frequency: f32,
}

impl LocatedDocInfo {
    pub fn new(row_id: u64, frequency: f32) -> Self {
        Self { row_id, frequency }
    }
}

impl Eq for LocatedDocInfo {}

impl PartialEq for LocatedDocInfo {
    fn eq(&self, other: &Self) -> bool {
        self.row_id == other.row_id
    }
}

impl PartialOrd for LocatedDocInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LocatedDocInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.row_id.cmp(&other.row_id)
    }
}

#[derive(Debug, Clone, Default, DeepSizeOf, Copy)]
pub struct RawDocInfo {
    pub doc_id: u32,
    pub frequency: u32,
}

impl RawDocInfo {
    pub fn new(doc_id: u32, frequency: u32) -> Self {
        Self { doc_id, frequency }
    }
}

impl Eq for RawDocInfo {}

impl PartialEq for RawDocInfo {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id
    }
}

impl PartialOrd for RawDocInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RawDocInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.doc_id.cmp(&other.doc_id)
    }
}

// DocSet is a mapping from row ids to the number of tokens in the document
// It's used to sort the documents by the bm25 score
#[derive(Debug, Clone, Default, DeepSizeOf)]
pub struct DocSet {
    row_ids: Vec<u64>,
    num_tokens: Vec<u32>,
    // (row_id, doc_id) pairs sorted by row_id
    inv: Vec<(u64, u32)>,

    total_tokens: u64,
}

impl DocSet {
    #[inline]
    pub fn len(&self) -> usize {
        // Use num_tokens instead of row_ids so the deferred-row_ids
        // scoring path (which constructs a DocSet via
        // [`Self::from_num_tokens_only`]) still reports the right doc
        // count.
        self.num_tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True iff the per-doc `row_id` array is populated. The
    /// deferred-row_id scoring path constructs DocSets with the array
    /// left empty so wand can skip the load; callers that need to do
    /// row_id lookups in the inner loop must check this and fall back
    /// to async resolution otherwise.
    #[inline]
    pub fn has_row_ids(&self) -> bool {
        !self.row_ids.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&u64, &u32)> {
        self.row_ids.iter().zip(self.num_tokens.iter())
    }

    pub fn row_id(&self, doc_id: u32) -> u64 {
        self.row_ids[doc_id as usize]
    }

    /// Resolve a `row_id` to every `doc_id` it owns.
    ///
    /// Modern indexes map each row to a single document. Older list indexes
    /// may have indexed each list element as its own document, so a single
    /// `row_id` can still own several `doc_id`s sharing that key in `inv`.
    /// The prefilter path (`flat_search`) walks an allow-list of row_ids and
    /// must evaluate all legacy documents for that row.
    pub fn doc_ids(&self, row_id: u64) -> impl Iterator<Item = u64> + '_ {
        if self.inv.is_empty() {
            // in legacy format, the row id is doc id (one document per row)
            let found = self.row_ids.binary_search(&row_id).is_ok();
            Either::Left(found.then_some(row_id).into_iter())
        } else {
            // `inv` is sorted by row_id, so the entries sharing this key form a
            // contiguous run; yield the doc_id of each.
            let lo = self.inv.partition_point(|entry| entry.0 < row_id);
            let hi = self.inv.partition_point(|entry| entry.0 <= row_id);
            Either::Right(self.inv[lo..hi].iter().map(|entry| entry.1 as u64))
        }
    }
    pub fn total_tokens_num(&self) -> u64 {
        self.total_tokens
    }

    #[inline]
    pub fn average_length(&self) -> f32 {
        self.total_tokens as f32 / self.len() as f32
    }

    pub fn calculate_block_max_scores<'a>(
        &self,
        doc_ids: impl Iterator<Item = &'a u32>,
        freqs: impl Iterator<Item = &'a u32>,
    ) -> Vec<f32> {
        let avgdl = self.average_length();
        let length = doc_ids.size_hint().0;
        let num_blocks = length.div_ceil(BLOCK_SIZE);
        let mut block_max_scores = Vec::with_capacity(num_blocks);
        let idf_scale = idf(length, self.len()) * (K1 + 1.0);
        let mut max_score = f32::MIN;
        for (i, (doc_id, freq)) in doc_ids.zip(freqs).enumerate() {
            let doc_norm = K1 * (1.0 - B + B * self.num_tokens(*doc_id) as f32 / avgdl);
            let freq = *freq as f32;
            let score = freq / (freq + doc_norm);
            if score > max_score {
                max_score = score;
            }
            if (i + 1) % BLOCK_SIZE == 0 {
                max_score *= idf_scale;
                block_max_scores.push(max_score);
                max_score = f32::MIN;
            }
        }
        if !length.is_multiple_of(BLOCK_SIZE) {
            max_score *= idf_scale;
            block_max_scores.push(max_score);
        }
        block_max_scores
    }

    pub fn to_batch(&self) -> Result<RecordBatch> {
        let row_id_col = UInt64Array::from_iter_values(self.row_ids.iter().cloned());
        let num_tokens_col = UInt32Array::from_iter_values(self.num_tokens.iter().cloned());

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(ROW_ID, DataType::UInt64, false),
            arrow_schema::Field::new(NUM_TOKEN_COL, DataType::UInt32, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(row_id_col) as ArrayRef,
                Arc::new(num_tokens_col) as ArrayRef,
            ],
        )?;
        Ok(batch)
    }

    pub async fn load(
        reader: Arc<dyn IndexReader>,
        is_legacy: bool,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Result<Self> {
        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
        let num_tokens_col = batch[NUM_TOKEN_COL].as_primitive::<datatypes::UInt32Type>();
        Self::from_columns(row_id_col, num_tokens_col, is_legacy, frag_reuse_index)
    }

    /// Build a `DocSet` carrying only the per-doc `num_tokens` array;
    /// `row_ids` and `inv` are left empty. Used by the deferred-row_id
    /// scoring path: wand checks `has_row_ids()` to skip `row_id` /
    /// `num_tokens_by_row_id` calls, and the per-partition caller
    /// resolves doc_id → row_id for the surviving top-K post-wand.
    pub fn from_num_tokens_only(num_tokens_col: &arrow_array::UInt32Array) -> Self {
        let num_tokens = num_tokens_col.values().to_vec();
        let total_tokens = num_tokens.iter().map(|&n| n as u64).sum();
        Self {
            row_ids: Vec::new(),
            num_tokens,
            inv: Vec::new(),
            total_tokens,
        }
    }

    /// Build a `DocSet` from already-loaded `row_id` and `num_tokens`
    /// arrow columns. Lets callers that have one column already in hand
    /// (e.g. `LazyDocSet` after `total_tokens_num` pre-fetched
    /// `num_tokens`) skip re-reading that column.
    pub fn from_columns(
        row_id_col: &UInt64Array,
        num_tokens_col: &arrow_array::UInt32Array,
        is_legacy: bool,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    ) -> Result<Self> {
        // for legacy format, the row id is doc id; sorting keeps binary search viable
        if is_legacy {
            let (row_ids, num_tokens): (Vec<_>, Vec<_>) = row_id_col
                .values()
                .iter()
                .filter_map(|id| {
                    if let Some(frag_reuse_index_ref) = frag_reuse_index.as_ref() {
                        frag_reuse_index_ref.remap_row_id(*id)
                    } else {
                        Some(*id)
                    }
                })
                .zip(num_tokens_col.values().iter())
                .sorted_unstable_by_key(|x| x.0)
                .unzip();

            let total_tokens = num_tokens.iter().map(|&x| x as u64).sum();
            return Ok(Self {
                row_ids,
                num_tokens,
                inv: Vec::new(),
                total_tokens,
            });
        }

        // If frag reuse happened, remap the row_ids through it. Crucially we
        // must NOT drop the rows the reuse index deleted, because the posting
        // lists reference doc_ids *positionally* (a doc_id is an index into
        // these arrays, fixed at build time). Dropping deleted rows would
        // renumber every later doc_id and desync the posting lists, so wand
        // would index `num_tokens`/`row_ids` out of bounds or score the wrong
        // doc. Instead we tombstone deleted rows in place: their slot survives
        // (so doc_ids stay aligned with the posting lists) carrying
        // `RowAddress::TOMBSTONE_ROW`, which wand skips, and they are left out
        // of `inv` so a row_id lookup never resolves to a deleted doc. The
        // heavyweight physical remap (`DocSet::remap`) is what actually
        // renumbers and compacts; this load-time path only has to stay
        // consistent until then.
        if let Some(frag_reuse_index_ref) = frag_reuse_index.as_ref() {
            let mut row_ids = Vec::with_capacity(row_id_col.len());
            let num_tokens = num_tokens_col.values().to_vec();
            let mut inv = Vec::with_capacity(row_id_col.len());
            for (doc_id, row_id) in row_id_col.values().iter().enumerate() {
                match frag_reuse_index_ref.remap_row_id(*row_id) {
                    Some(new_row_id) => {
                        row_ids.push(new_row_id);
                        inv.push((new_row_id, doc_id as u32));
                    }
                    None => {
                        // Deleted: keep the slot (doc_ids must not shift) but
                        // tombstone it and leave it out of `inv`.
                        row_ids.push(RowAddress::TOMBSTONE_ROW);
                    }
                }
            }
            inv.sort_unstable_by_key(|entry| entry.0);

            let total_tokens = num_tokens.iter().map(|&x| x as u64).sum();
            return Ok(Self {
                row_ids,
                num_tokens,
                inv,
                total_tokens,
            });
        }

        let row_ids = row_id_col.values().to_vec();
        let num_tokens = num_tokens_col.values().to_vec();
        let mut inv: Vec<(u64, u32)> = row_ids
            .iter()
            .enumerate()
            .map(|(doc_id, row_id)| (*row_id, doc_id as u32))
            .collect();
        if !row_ids.is_sorted() {
            inv.sort_unstable_by_key(|entry| entry.0);
        }
        let total_tokens = num_tokens.iter().map(|&x| x as u64).sum();
        Ok(Self {
            row_ids,
            num_tokens,
            inv,
            total_tokens,
        })
    }

    // remap the row ids to the new row ids
    // returns the removed doc ids
    pub fn remap(&mut self, mapping: &RowAddrRemap) -> Vec<u32> {
        let mut removed = Vec::new();
        let len = self.len();
        let row_ids = std::mem::replace(&mut self.row_ids, Vec::with_capacity(len));
        let num_tokens = std::mem::replace(&mut self.num_tokens, Vec::with_capacity(len));
        self.total_tokens = 0;
        for (doc_id, (row_id, num_token)) in std::iter::zip(row_ids, num_tokens).enumerate() {
            match mapping.get(row_id) {
                Some(Some(new_row_id)) => {
                    self.row_ids.push(new_row_id);
                    self.num_tokens.push(num_token);
                    self.total_tokens += num_token as u64;
                }
                Some(None) => {
                    removed.push(doc_id as u32);
                }
                None => {
                    self.row_ids.push(row_id);
                    self.num_tokens.push(num_token);
                    self.total_tokens += num_token as u64;
                }
            }
        }
        removed
    }

    #[inline]
    pub fn num_tokens(&self, doc_id: u32) -> u32 {
        self.num_tokens[doc_id as usize]
    }

    // this can be used only if it's a legacy format,
    // which store the sorted row ids so that we can use binary search
    #[inline]
    pub fn num_tokens_by_row_id(&self, row_id: u64) -> u32 {
        self.row_ids
            .binary_search(&row_id)
            .map(|idx| self.num_tokens[idx])
            .unwrap_or(0)
    }

    // append a document to the doc set
    // returns the doc_id (the number of documents before appending)
    pub fn append(&mut self, row_id: u64, num_tokens: u32) -> u32 {
        self.row_ids.push(row_id);
        self.num_tokens.push(num_tokens);
        self.total_tokens += num_tokens as u64;
        self.row_ids.len() as u32 - 1
    }

    pub(crate) fn memory_size(&self) -> usize {
        self.row_ids.capacity() * std::mem::size_of::<u64>()
            + self.num_tokens.capacity() * std::mem::size_of::<u32>()
            + self.inv.capacity() * std::mem::size_of::<(u64, u32)>()
    }
}

pub fn flat_full_text_search(
    batches: &[&RecordBatch],
    doc_col: &str,
    query: &str,
    tokenizer: Option<Box<dyn LanceTokenizer>>,
) -> Result<Vec<u64>> {
    if batches.is_empty() {
        return Ok(vec![]);
    }

    if is_phrase_query(query) {
        return Err(Error::invalid_input(
            "phrase query is not supported for flat full text search, try using FTS index",
        ));
    }

    match batches[0][doc_col].data_type() {
        DataType::Utf8 => do_flat_full_text_search::<i32>(batches, doc_col, query, tokenizer),
        DataType::LargeUtf8 => do_flat_full_text_search::<i64>(batches, doc_col, query, tokenizer),
        DataType::List(_) => {
            do_flat_full_text_search_list::<i32>(batches, doc_col, query, tokenizer)
        }
        DataType::LargeList(_) => {
            do_flat_full_text_search_list::<i64>(batches, doc_col, query, tokenizer)
        }
        data_type => Err(Error::invalid_input(format!(
            "unsupported data type {} for inverted index",
            data_type
        ))),
    }
}

fn do_flat_full_text_search<Offset: OffsetSizeTrait>(
    batches: &[&RecordBatch],
    doc_col: &str,
    query: &str,
    tokenizer: Option<Box<dyn LanceTokenizer>>,
) -> Result<Vec<u64>> {
    let mut results = Vec::new();
    let mut tokenizer =
        tokenizer.unwrap_or_else(|| InvertedIndexParams::default().build().unwrap());
    let query_tokens = collect_query_tokens(query, &mut tokenizer);

    for batch in batches {
        let row_id_array = batch[ROW_ID].as_primitive::<UInt64Type>();
        let doc_array = batch[doc_col].as_string::<Offset>();
        for i in 0..row_id_array.len() {
            let doc = doc_array.value(i);
            if has_query_token(doc, &mut tokenizer, &query_tokens) {
                results.push(row_id_array.value(i));
                // What is this assertion for?  Why would doc contain query?  Don't we reach
                // here only if they share at least one token?  Why is it not debug_assert?
                assert!(doc.contains(query));
            }
        }
    }

    Ok(results)
}

fn do_flat_full_text_search_list<ListOffset: OffsetSizeTrait>(
    batches: &[&RecordBatch],
    doc_col: &str,
    query: &str,
    tokenizer: Option<Box<dyn LanceTokenizer>>,
) -> Result<Vec<u64>> {
    let mut results = Vec::new();
    let mut tokenizer =
        tokenizer.unwrap_or_else(|| InvertedIndexParams::default().build().unwrap());
    let query_tokens = collect_query_tokens(query, &mut tokenizer);

    for batch in batches {
        let row_id_array = batch[ROW_ID].as_primitive::<UInt64Type>();
        let doc_array = batch[doc_col].as_list::<ListOffset>();
        match doc_array.value_type() {
            DataType::Utf8 | DataType::LargeUtf8 => {}
            data_type => {
                return Err(Error::invalid_input(format!(
                    "unsupported list item data type {} for inverted index",
                    data_type
                )));
            }
        }
        for i in 0..row_id_array.len() {
            if doc_array.is_null(i) {
                continue;
            }
            let elements = doc_array.value(i);
            if iter_str_array(elements.as_ref())
                .flatten()
                .any(|element| has_query_token(element, &mut tokenizer, &query_tokens))
            {
                results.push(row_id_array.value(i));
            }
        }
    }

    Ok(results)
}

const FLAT_ROW_ID_COL_IDX: usize = 0;
const FLAT_ALL_TOKENS_COL_IDX: usize = 1;
const FLAT_QUERY_TOKEN_COUNTS_COL_IDX: usize = 2;

/// If we accumulate this many bytes we warn the user they probably want to use an FTS index instead.
const BYTES_ACCUMULATED_WARNING_THRESHOLD: u64 = 1024 * 1024 * 1024; // 1GB

/// Consumes a stream of record batches and produces token counts
///
/// The resulting batch will have three columns:
/// - row_id: the row id of the document
/// - all_tokens: the total number of tokens in the document
/// - query_token_counts: a fixed size list of the count of each query token in the document
///
/// This is an unbounded accumulation, however, for most queries, the per-row
/// growth will be fairly small.  As a result we can process millions of tokens
/// with fairly modest memory usage.
///
/// However, it is unwise to do a flat search across billions of rows.  An FTS
/// index should be created instead.
async fn tokenize_and_count(
    input: impl Stream<Item = DataFusionResult<RecordBatch>> + Send,
    tokenizer: Box<dyn LanceTokenizer>,
    query_tokens: Arc<Tokens>,
    doc_col_idx: usize,
    elapsed_compute: Option<Time>,
) -> DataFusionResult<RecordBatch> {
    let output_schema = Arc::new(Schema::new(vec![
        ROW_ID_FIELD.clone(),
        Field::new("all_tokens", DataType::UInt64, false),
        Field::new(
            "query_token_counts",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::UInt64, true)),
                query_tokens.len() as i32,
            ),
            false,
        ),
    ]));
    let output_schema_clone = output_schema.clone();
    let bytes_accumulated = Arc::new(AtomicU64::new(0));
    let bytes_warning_emitted = Arc::new(AtomicBool::new(false));

    let batches = input
        .map(move |batch| {
            let mut tokenizer = tokenizer.box_clone();
            let output_schema = output_schema.clone();
            let query_tokens = query_tokens.clone();
            let bytes_accumulated = bytes_accumulated.clone();
            let bytes_warning_emitted = bytes_warning_emitted.clone();
            let elapsed_compute = elapsed_compute.clone();
            spawn_cpu(move || {
                // Time the per-batch CPU work so callers can attribute it to
                // `elapsed_compute` on a metric handle (the spawn_cpu worker
                // thread is invisible to the caller's poll timer otherwise).
                let start = std::time::Instant::now();
                let batch = batch?;
                let row_id_array = batch[ROW_ID].as_primitive::<UInt64Type>();
                let mut row_ids = UInt64Builder::with_capacity(batch.num_rows());
                let mut all_token_counts = UInt64Builder::with_capacity(batch.num_rows());
                let mut query_token_counts = FixedSizeListBuilder::with_capacity(
                    UInt64Builder::with_capacity(batch.num_rows() * query_tokens.len()),
                    query_tokens.len() as i32,
                    batch.num_rows(),
                );
                let mut temp_query_token_counts = Vec::with_capacity(query_tokens.len());
                let mut count_text = |doc: &str, temp_query_token_counts: &mut Vec<u64>| -> u64 {
                    let mut stream = tokenizer.token_stream_for_doc(doc);
                    let mut all_tokens = 0;
                    while let Some(token) = stream.next() {
                        all_tokens += 1;
                        if let Some(token_index) = query_tokens.token_index(&token.text) {
                            temp_query_token_counts[token_index] += 1;
                        }
                    }
                    all_tokens
                };
                let mut append_counts =
                    |row_id: u64, all_tokens: u64, temp_query_token_counts: &[u64]| {
                        row_ids.append_value(row_id);
                        all_token_counts.append_value(all_tokens);
                        for count in temp_query_token_counts.iter().copied() {
                            query_token_counts.values().append_value(count);
                        }
                        query_token_counts.append(true);
                    };
                match batch.column(doc_col_idx).data_type() {
                    DataType::Utf8 | DataType::LargeUtf8 => {
                        let doc_iter = iter_str_array(batch.column(doc_col_idx));
                        for (doc, row_id) in doc_iter.zip(row_id_array.values().iter()) {
                            temp_query_token_counts.clear();
                            temp_query_token_counts
                                .extend(std::iter::repeat_n(0, query_tokens.len()));

                            let Some(doc) = doc else {
                                append_counts(*row_id, 0, &temp_query_token_counts);
                                continue;
                            };

                            let all_tokens = count_text(doc, &mut temp_query_token_counts);
                            append_counts(*row_id, all_tokens, &temp_query_token_counts);
                        }
                    }
                    DataType::List(_) => {
                        tokenize_and_count_list::<i32>(
                            batch.column(doc_col_idx),
                            row_id_array,
                            &mut count_text,
                            &mut append_counts,
                            &mut temp_query_token_counts,
                            query_tokens.len(),
                        )?;
                    }
                    DataType::LargeList(_) => {
                        tokenize_and_count_list::<i64>(
                            batch.column(doc_col_idx),
                            row_id_array,
                            &mut count_text,
                            &mut append_counts,
                            &mut temp_query_token_counts,
                            query_tokens.len(),
                        )?;
                    }
                    data_type => {
                        return DataFusionResult::Err(datafusion_common::DataFusionError::Execution(
                            format!("unsupported data type {} for flat full text search", data_type),
                        ));
                    }
                }
                let row_ids = row_ids.finish();
                let all_token_counts = all_token_counts.finish();
                let query_token_counts = query_token_counts.finish();
                let result_batch = RecordBatch::try_new(
                    output_schema,
                    vec![
                        Arc::new(row_ids) as ArrayRef,
                        Arc::new(all_token_counts) as ArrayRef,
                        Arc::new(query_token_counts) as ArrayRef,
                    ],
                )?;
                let bytes_accumulated = bytes_accumulated.fetch_add(result_batch.get_array_memory_size() as u64, Ordering::Relaxed);
                if bytes_accumulated > BYTES_ACCUMULATED_WARNING_THRESHOLD && !bytes_warning_emitted.swap(true, Ordering::Relaxed) {
                    tracing::warn!("Flat full text search is accumulating a large number of bytes.  Consider using an FTS index instead.");
                }

                if let Some(t) = &elapsed_compute {
                    t.add_duration(start.elapsed());
                }
                DataFusionResult::Ok(result_batch)
            })
        })
        .buffered(get_num_compute_intensive_cpus())
        .try_collect::<Vec<_>>()
        .await?;

    Ok(arrow::compute::concat_batches(
        &output_schema_clone,
        &batches,
    )?)
}

fn tokenize_and_count_list<ListOffset: OffsetSizeTrait>(
    doc_col: &ArrayRef,
    row_id_array: &arrow_array::PrimitiveArray<UInt64Type>,
    count_text: &mut impl FnMut(&str, &mut Vec<u64>) -> u64,
    append_counts: &mut impl FnMut(u64, u64, &[u64]),
    temp_query_token_counts: &mut Vec<u64>,
    query_tokens_len: usize,
) -> DataFusionResult<()> {
    let doc_array = doc_col.as_list::<ListOffset>();
    match doc_array.value_type() {
        DataType::Utf8 | DataType::LargeUtf8 => {}
        data_type => {
            return Err(datafusion_common::DataFusionError::Execution(format!(
                "unsupported list item data type {} for flat full text search",
                data_type
            )));
        }
    }

    for i in 0..row_id_array.len() {
        if doc_array.is_null(i) {
            continue;
        }

        temp_query_token_counts.clear();
        temp_query_token_counts.extend(std::iter::repeat_n(0, query_tokens_len));

        let elements = doc_array.value(i);
        let mut all_tokens = 0;
        for element in iter_str_array(elements.as_ref()).flatten() {
            all_tokens += count_text(element, temp_query_token_counts);
        }

        if all_tokens > 0 {
            append_counts(row_id_array.value(i), all_tokens, temp_query_token_counts);
        }
    }

    Ok(())
}

/// Initialize the BM25 scorer
///
/// In order to calculate BM25 scores we need to know token counts for the entire corpus.  We extract these from the
/// counted input of the flat search combined with any counts recorded for the indexed portion.
fn initialize_scorer(
    base_scorer: Option<&MemBM25Scorer>,
    query_tokens: &Tokens,
    counted_input: &RecordBatch,
) -> MemBM25Scorer {
    let mut total_tokens = 0;
    let mut num_docs = 0;
    let mut all_token_counts = vec![0; query_tokens.len()];

    if let Some(base_scorer) = base_scorer {
        total_tokens += base_scorer.total_tokens;
        num_docs += base_scorer.num_docs;
        for (token_index, token) in query_tokens.into_iter().enumerate() {
            all_token_counts[token_index] = base_scorer.num_docs_containing_token(token) as u64;
        }
    }

    num_docs += counted_input.num_rows();
    total_tokens += arrow::compute::sum(
        counted_input
            .column(FLAT_ALL_TOKENS_COL_IDX)
            .as_primitive::<UInt64Type>(),
    )
    .unwrap_or_default();

    let mut input_token_counters = counted_input
        .column(FLAT_QUERY_TOKEN_COUNTS_COL_IDX)
        .as_fixed_size_list()
        .values()
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied();

    for _ in 0..counted_input.num_rows() {
        for token_count in all_token_counts.iter_mut() {
            if input_token_counters.next().unwrap_or_default() > 0 {
                *token_count += 1;
            }
        }
    }

    let token_counts_map = all_token_counts
        .into_iter()
        .enumerate()
        .map(|(token_index, count)| {
            (
                query_tokens.get_token(token_index).to_string(),
                count as usize,
            )
        })
        .collect::<HashMap<String, usize>>();
    MemBM25Scorer::new(total_tokens, num_docs, token_counts_map)
}

fn flat_bm25_score(
    query_tokens: &Tokens,
    counted_input: &RecordBatch,
    scorer: &MemBM25Scorer,
) -> Result<RecordBatch> {
    let mut row_ids_builder = UInt64Builder::with_capacity(counted_input.num_rows());
    let mut scores_builder = Float32Builder::with_capacity(counted_input.num_rows());

    let mut row_ids_iter = counted_input
        .column(FLAT_ROW_ID_COL_IDX)
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied();
    let mut all_token_counts_iter = counted_input
        .column(FLAT_ALL_TOKENS_COL_IDX)
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied();
    let mut query_token_counts_iter = counted_input
        .column(FLAT_QUERY_TOKEN_COUNTS_COL_IDX)
        .as_fixed_size_list()
        .values()
        .as_primitive::<UInt64Type>()
        .values()
        .iter()
        .copied();
    for _ in 0..counted_input.num_rows() {
        let num_tokens_in_doc = all_token_counts_iter.next().expect_ok()?;
        let row_id = row_ids_iter.next().expect_ok()?;
        if num_tokens_in_doc == 0 {
            for _ in query_tokens {
                query_token_counts_iter.next().expect_ok()?;
            }
            continue;
        }
        let doc_norm = K1 * (1.0 - B + B * num_tokens_in_doc as f32 / scorer.avg_doc_length());
        let mut score = 0.0;
        for token in query_tokens {
            let freq = query_token_counts_iter.next().expect_ok()? as f32;
            let idf = idf(scorer.num_docs_containing_token(token), scorer.num_docs());
            score += idf * (freq * (K1 + 1.0) / (freq + doc_norm));
        }
        if score > 0.0 {
            row_ids_builder.append_value(row_id);
            scores_builder.append_value(score);
        }
    }

    let row_ids = row_ids_builder.finish();
    let scores = scores_builder.finish();
    let batch = RecordBatch::try_new(
        FTS_SCHEMA.clone(),
        vec![Arc::new(row_ids) as ArrayRef, Arc::new(scores) as ArrayRef],
    )?;
    Ok(batch)
}

#[deprecated(
    note = "use `flat_bm25_search_stream_with_metrics` to record CPU compute \
            time on a metric handle; pass `None` for the old behavior"
)]
pub async fn flat_bm25_search_stream(
    input: SendableRecordBatchStream,
    doc_col: String,
    query: String,
    tokenizer: Box<dyn LanceTokenizer>,
    base_scorer: Option<MemBM25Scorer>,
    target_batch_size: usize,
) -> DataFusionResult<SendableRecordBatchStream> {
    flat_bm25_search_stream_with_metrics(
        input,
        doc_col,
        query,
        tokenizer,
        base_scorer,
        target_batch_size,
        None,
    )
    .await
}

/// Same as [`flat_bm25_search_stream`] but accepts an optional `Time` handle
/// that, if provided, will receive the CPU time spent in (a) per-batch
/// tokenization on the `spawn_cpu` worker threads and (b) the synchronous
/// scoring phase. This lets a calling `ExecutionPlan` report accurate
/// `elapsed_compute` without double-counting upstream poll time.
pub async fn flat_bm25_search_stream_with_metrics(
    input: SendableRecordBatchStream,
    doc_col: String,
    query: String,
    tokenizer: Box<dyn LanceTokenizer>,
    base_scorer: Option<MemBM25Scorer>,
    target_batch_size: usize,
    elapsed_compute: Option<Time>,
) -> DataFusionResult<SendableRecordBatchStream> {
    let mut tokenizer = tokenizer;

    // Pre-await synchronous work: query tokenization + chunk-stream setup.
    let pre_await_start = std::time::Instant::now();
    let query_tokens = Arc::new(collect_query_tokens(&query, &mut tokenizer));

    // A query that tokenizes to no terms (e.g. only stop words) has no
    // searchable content and matches nothing. Return early rather than
    // proceeding. This mirrors the indexed search path, which already
    // short-circuits on empty query tokens.
    if query_tokens.is_empty() {
        return Ok(Box::pin(RecordBatchStreamAdapter::new(
            FTS_SCHEMA.clone(),
            stream::empty::<DataFusionResult<RecordBatch>>(),
        )));
    }

    let input_schema = input.schema();
    let doc_col_idx = input_schema.index_of(&doc_col)?;

    // Accumulate small batches until this threshold before dispatching a task.
    const ACCUMULATE_BYTES: usize = 256 * 1024;
    // Slice oversized batches down to roughly this size.
    const SLICE_BYTES: usize = 512 * 1024;

    // Phase 1 - rechunk the input stream into appropriately sized chunks.  Tokenization is
    // fairly CPU-intensive, and we don't need too much data to justify a new thread task.
    let chunked = lance_arrow::stream::rechunk_stream_by_size(
        input,
        input_schema,
        ACCUMULATE_BYTES,
        SLICE_BYTES,
    );
    if let Some(t) = &elapsed_compute {
        t.add_duration(pre_await_start.elapsed());
    }

    // Phase 2 - For each row we need to know the total number of tokens and the count of each
    // of the query tokens.  For example, if the query is "book" and the row is "the book shop"
    // and we are tokenizing with a whitespace tokenizer, we need to know that there are 3 tokens
    // and the token book appears once.
    let counted_input = tokenize_and_count(
        chunked,
        tokenizer,
        query_tokens.clone(),
        doc_col_idx,
        elapsed_compute.clone(),
    )
    .await?;

    // Phase 3 - Calculate final scores (this is fairly cheap, probably don't need to parallelize).
    // All post-await work is synchronous; time the scorer + score + slicing loop together.
    let post_await_start = std::time::Instant::now();
    let scorer = initialize_scorer(base_scorer.as_ref(), query_tokens.as_ref(), &counted_input);
    let scores = flat_bm25_score(query_tokens.as_ref(), &counted_input, &scorer)?;

    // Finally we emit batches according to the target batch size
    let num_out_batches = scores.num_rows().div_ceil(target_batch_size);
    let mut batches = Vec::with_capacity(num_out_batches);
    for i in 0..num_out_batches {
        let start = i * target_batch_size;
        let len = (scores.num_rows() - start).min(target_batch_size);
        batches.push(Ok(scores.slice(start, len)));
    }
    if let Some(t) = &elapsed_compute {
        t.add_duration(post_await_start.elapsed());
    }
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        FTS_SCHEMA.clone(),
        stream::iter(batches),
    )))
}

pub fn is_phrase_query(query: &str) -> bool {
    query.starts_with('\"') && query.ends_with('\"')
}

#[cfg(test)]
mod tests {
    use crate::scalar::inverted::document_tokenizer::DocType;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_io::object_store::ObjectStore;

    use crate::metrics::{LocalMetricsCollector, NoOpMetricsCollector};
    use crate::prefilter::NoFilter;
    use crate::scalar::ScalarIndex;
    use crate::scalar::inverted::builder::{
        InnerBuilder, InvertedIndexBuilder, PositionRecorder, inverted_list_schema,
    };
    use crate::scalar::inverted::encoding::{
        compress_positions, compress_posting_list_with_tail_codec,
        decompress_posting_list_with_tail_codec, encode_position_stream_block_into,
    };
    use crate::scalar::inverted::query::{FtsSearchParams, Operator};
    use crate::scalar::lance_format::LanceIndexStore;
    use arrow::array::{
        AsArray, GenericListBuilder, GenericStringBuilder, Int32Builder, LargeBinaryBuilder,
        ListBuilder, UInt32Builder,
    };
    use arrow::datatypes::{Float32Type, UInt32Type};
    use arrow_array::{ArrayRef, Float32Array, RecordBatch, StringArray, UInt32Array, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use crate::scalar::inverted::tokenizer::document_tokenizer::TextTokenizer;
    use lance_tokenizer::{Language, SimpleTokenizer, StopWordFilter, TextAnalyzer};

    use super::*;

    async fn write_single_partition_index(
        store: Arc<LanceIndexStore>,
        params: InvertedIndexParams,
        token_set_format: TokenSetFormat,
        token: &str,
        row_id: u64,
    ) -> Result<Arc<InvertedIndex>> {
        let mut partition = InnerBuilder::new_with_format_version(
            0,
            false,
            token_set_format,
            InvertedListFormatVersion::V1,
        );
        partition.tokens.add(token.to_owned());
        let mut posting_list =
            PostingListBuilder::new_with_posting_tail_codec(false, PostingTailCodec::Fixed32);
        posting_list.add(0, PositionRecorder::Count(1));
        partition.posting_lists.push(posting_list);
        partition.docs.append(row_id, 1);
        partition.write(store.as_ref()).await?;

        let metadata = HashMap::from([
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0_u64]).unwrap(),
            ),
            ("params".to_owned(), serde_json::to_string(&params).unwrap()),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                token_set_format.to_string(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await?;
        writer.finish_with_metadata(metadata).await?;

        InvertedIndex::load(store, None, &LanceCache::no_cache()).await
    }

    fn empty_doc_stream() -> SendableRecordBatchStream {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc", DataType::Utf8, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(Vec::<datafusion::error::Result<RecordBatch>>::new()),
        ))
    }

    #[tokio::test]
    async fn test_posting_builder_remap() {
        let posting_tail_codec = PostingTailCodec::Fixed32;
        let mut builder =
            PostingListBuilder::new_with_posting_tail_codec(false, posting_tail_codec);
        let n = BLOCK_SIZE + 3;
        for i in 0..n {
            builder.add(i as u32, PositionRecorder::Count(1));
        }
        let removed = vec![5, 7];
        builder.remap(&removed);

        let mut expected =
            PostingListBuilder::new_with_posting_tail_codec(false, posting_tail_codec);
        for i in 0..n - removed.len() {
            expected.add(i as u32, PositionRecorder::Count(1));
        }
        let expected_entries = expected.iter().collect::<Vec<_>>();
        let actual_entries = builder.iter().collect::<Vec<_>>();
        assert_eq!(actual_entries, expected_entries);

        // BLOCK_SIZE + 3 elements should be reduced to BLOCK_SIZE + 1,
        // there are still 2 blocks.
        let batch = builder.to_batch(vec![1.0, 2.0]).unwrap();
        let (doc_ids, freqs) = decompress_posting_list_with_tail_codec(
            (n - removed.len()) as u32,
            batch[POSTING_COL]
                .as_list::<i32>()
                .value(0)
                .as_binary::<i64>(),
            posting_tail_codec,
        )
        .unwrap();
        assert!(
            doc_ids
                .iter()
                .zip(expected_entries.iter().map(|(doc_id, _, _)| doc_id))
                .all(|(a, b)| a == b)
        );
        assert!(
            freqs
                .iter()
                .zip(expected_entries.iter().map(|(_, freq, _)| freq))
                .all(|(a, b)| a == b)
        );
    }

    #[test]
    fn test_posting_builder_size_tracking_matches_structure() {
        fn tracked_memory_size(builder: &PostingListBuilder) -> u64 {
            let encoded_blocks_size = builder
                .encoded_blocks
                .iter()
                .map(|encoded_blocks| std::mem::size_of::<EncodedBlocks>() + encoded_blocks.size())
                .sum::<usize>();
            let encoded_positions_size = builder
                .encoded_position_blocks
                .as_ref()
                .map(|positions| std::mem::size_of::<EncodedPositionBlocks>() + positions.size())
                .unwrap_or(0usize);
            (encoded_blocks_size
                + builder.tail_entries.capacity() * std::mem::size_of::<RawDocInfo>()
                + builder.tail_positions.size()
                + encoded_positions_size) as u64
        }

        let mut builder = PostingListBuilder::new(true);
        for doc_id in 0..(BLOCK_SIZE + 5) as u32 {
            builder.add(
                doc_id,
                PositionRecorder::Position(smallvec::smallvec![1, 3, 5]),
            );
        }

        assert_eq!(builder.size(), tracked_memory_size(&builder));
    }

    #[test]
    fn test_posting_builder_flush_releases_tail_position_capacity() {
        let mut builder = PostingListBuilder::new(true);
        let positions = smallvec::SmallVec::<[u32; 2]>::from_vec((0..1024).collect());
        for doc_id in 0..BLOCK_SIZE as u32 {
            builder.add(doc_id, PositionRecorder::Position(positions.clone()));
        }

        assert_eq!(builder.tail_positions.size(), 0);
        assert_eq!(builder.size(), {
            let encoded_blocks_size = builder
                .encoded_blocks
                .iter()
                .map(|encoded_blocks| std::mem::size_of::<EncodedBlocks>() + encoded_blocks.size())
                .sum::<usize>();
            let encoded_positions_size = builder
                .encoded_position_blocks
                .as_ref()
                .map(|positions| std::mem::size_of::<EncodedPositionBlocks>() + positions.size())
                .unwrap_or(0usize);
            (encoded_blocks_size
                + builder.tail_entries.capacity() * std::mem::size_of::<RawDocInfo>()
                + builder.tail_positions.size()
                + encoded_positions_size) as u64
        });
    }

    #[test]
    fn test_posting_builder_streamed_positions_roundtrip() {
        let mut builder = PostingListBuilder::new(true);
        assert!(builder.add_occurrence(0, 1).unwrap());
        assert!(!builder.add_occurrence(0, 4).unwrap());
        assert!(!builder.add_occurrence(0, 9).unwrap());
        builder.finish_open_doc(0).unwrap();

        assert!(builder.add_occurrence(2, 3).unwrap());
        builder.finish_open_doc(2).unwrap();

        let entries = builder.iter().collect::<Vec<_>>();
        assert_eq!(
            entries,
            vec![
                (0_u32, 3_u32, Some(vec![1_u32, 4_u32, 9_u32])),
                (2_u32, 1_u32, Some(vec![3_u32])),
            ]
        );
    }

    #[test]
    fn test_shared_position_stream_clone_shares_block_offsets() {
        let stream = SharedPositionStream::new(
            PositionStreamCodec::PackedDelta,
            vec![0_u32, 4, 11],
            bytes::Bytes::from_static(b"shared position bytes"),
        );
        let original_offsets = stream.block_offsets().as_ptr();

        let cloned = stream.clone();

        assert_eq!(cloned.block_offsets(), stream.block_offsets());
        assert_eq!(cloned.block_offsets().as_ptr(), original_offsets);
    }

    #[test]
    fn test_posting_builder_roundtrip_shared_positions() {
        let entries = vec![
            (0_u32, vec![1_u32, 5]),
            (2, vec![0, 4, 9]),
            (4, vec![7]),
            (8, vec![3, 10]),
            (13, vec![2, 11, 30]),
        ];
        let mut builder =
            PostingListBuilder::new_with_posting_tail_codec(true, PostingTailCodec::VarintDelta);
        for (doc_id, positions) in &entries {
            builder.add(
                *doc_id,
                PositionRecorder::Position(positions.clone().into()),
            );
        }

        let batch = builder.to_batch(vec![1.0]).unwrap();
        assert!(batch.column_by_name(COMPRESSED_POSITION_COL).is_some());
        assert!(batch.column_by_name(POSITION_COL).is_none());
        assert_eq!(
            batch.schema_ref().metadata().get(POSTING_TAIL_CODEC_KEY),
            Some(&PostingTailCodec::VarintDelta.as_str().to_owned())
        );
        assert_eq!(
            batch.schema_ref().metadata().get(POSITIONS_LAYOUT_KEY),
            Some(&POSITIONS_LAYOUT_SHARED_STREAM_V2.to_owned())
        );
        assert_eq!(
            batch.schema_ref().metadata().get(POSITIONS_CODEC_KEY),
            Some(&PositionStreamCodec::PackedDelta.as_str().to_owned())
        );

        let posting =
            PostingList::from_batch(&batch, Some(1.0), Some(entries.len() as u32)).unwrap();
        let actual = posting
            .iter()
            .map(|(doc_id, freq, positions)| {
                (doc_id as u32, freq, positions.unwrap().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        let expected = entries
            .iter()
            .map(|(doc_id, positions)| (*doc_id, positions.len() as u32, positions.clone()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_posting_builder_roundtrip_legacy_positions() {
        let entries = vec![(0_u32, vec![1_u32, 5]), (2, vec![0, 4, 9]), (4, vec![7])];
        let mut builder =
            PostingListBuilder::new_with_posting_tail_codec(true, PostingTailCodec::Fixed32);
        for (doc_id, positions) in &entries {
            builder.add(
                *doc_id,
                PositionRecorder::Position(positions.clone().into()),
            );
        }

        let batch = builder.to_batch(vec![1.0]).unwrap();
        assert!(batch.column_by_name(POSITION_COL).is_some());
        assert!(batch.column_by_name(COMPRESSED_POSITION_COL).is_none());
        assert_eq!(
            batch.schema_ref().metadata().get(POSTING_TAIL_CODEC_KEY),
            None
        );
        assert_eq!(
            batch.schema_ref().metadata().get(POSITIONS_LAYOUT_KEY),
            None
        );
        assert_eq!(batch.schema_ref().metadata().get(POSITIONS_CODEC_KEY), None);

        let posting =
            PostingList::from_batch(&batch, Some(1.0), Some(entries.len() as u32)).unwrap();
        let actual = posting
            .iter()
            .map(|(doc_id, freq, positions)| {
                (doc_id as u32, freq, positions.unwrap().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        let expected = entries
            .iter()
            .map(|(doc_id, positions)| (*doc_id, positions.len() as u32, positions.clone()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_resolve_fts_format_version_defaults_to_v2() {
        assert_eq!(
            resolve_fts_format_version(None).unwrap(),
            InvertedListFormatVersion::V2
        );
        assert_eq!(
            resolve_fts_format_version(Some("2")).unwrap(),
            InvertedListFormatVersion::V2
        );
        assert_eq!(
            resolve_fts_format_version(Some("3")).unwrap(),
            InvertedListFormatVersion::V3
        );
    }

    #[tokio::test]
    async fn test_v3_block_row_layout_roundtrip_with_positions() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new_with_format_version(
            0,
            true,
            TokenSetFormat::default(),
            InvertedListFormatVersion::V3,
        );
        builder.tokens.add("alpha".to_owned());
        let mut posting = PostingListBuilder::new_with_posting_tail_codec(
            true,
            InvertedListFormatVersion::V3.posting_tail_codec(),
        );
        let mut expected = Vec::new();
        for doc_id in 0..(BLOCK_SIZE as u32 + 3) {
            let positions = vec![doc_id % 5, doc_id % 5 + 7];
            posting.add(doc_id, PositionRecorder::Position(positions.clone().into()));
            builder.docs.append(doc_id as u64, positions.len() as u32);
            expected.push((doc_id, positions.len() as u32, positions));
        }
        builder.posting_lists.push(posting);
        builder.write(store.as_ref()).await.unwrap();

        let terms_reader = store.open_index_file(&posting_file_path(0)).await.unwrap();
        assert!(terms_reader.schema().field(POSTING_COL).is_none());
        assert_eq!(
            terms_reader.schema().field(LENGTH_COL).unwrap().data_type(),
            DataType::UInt32
        );
        assert_eq!(
            terms_reader
                .schema()
                .field(MAX_SCORE_COL)
                .unwrap()
                .data_type(),
            DataType::Float32
        );
        assert_eq!(
            terms_reader
                .schema()
                .field(BLOCK_START_COL)
                .unwrap()
                .data_type(),
            DataType::UInt64
        );
        assert_eq!(
            terms_reader
                .schema()
                .field(NUM_BLOCKS_COL)
                .unwrap()
                .data_type(),
            DataType::UInt32
        );
        let terms_batch = terms_reader
            .read_range(
                0..terms_reader.num_rows(),
                Some(&[LENGTH_COL, MAX_SCORE_COL, BLOCK_START_COL, NUM_BLOCKS_COL]),
            )
            .await
            .unwrap();
        assert_eq!(terms_batch.num_rows(), 1);
        assert_eq!(
            terms_batch[LENGTH_COL]
                .as_primitive::<UInt32Type>()
                .value(0),
            (BLOCK_SIZE + 3) as u32
        );
        assert_eq!(
            terms_batch[BLOCK_START_COL]
                .as_primitive::<UInt64Type>()
                .value(0),
            0
        );
        assert_eq!(
            terms_batch[NUM_BLOCKS_COL]
                .as_primitive::<UInt32Type>()
                .value(0),
            2
        );

        let blocks_reader = store
            .open_index_file(&posting_blocks_file_path(0))
            .await
            .unwrap();
        assert_eq!(
            blocks_reader
                .schema()
                .field(POSTING_COL)
                .unwrap()
                .data_type(),
            DataType::LargeBinary
        );
        assert!(blocks_reader.schema().field(FIRST_DOC_ID_COL).is_some());
        assert!(blocks_reader.schema().field(LAST_DOC_ID_COL).is_some());
        assert!(blocks_reader.schema().field(BLOCK_MAX_SCORE_COL).is_some());
        assert!(
            blocks_reader
                .schema()
                .field(POSITION_BLOCK_OFFSET_COL)
                .is_none()
        );
        assert!(
            blocks_reader
                .schema()
                .field(COMPRESSED_POSITION_COL)
                .is_none()
        );

        let positions_reader = store
            .open_index_file(&posting_positions_file_path(0))
            .await
            .unwrap();
        assert_eq!(positions_reader.num_rows(), blocks_reader.num_rows());
        assert!(
            positions_reader
                .schema()
                .field(COMPRESSED_POSITION_COL)
                .is_some()
        );

        let coarse_reader = store
            .open_index_file(&posting_coarse_skip_file_path(0))
            .await
            .unwrap();
        assert_eq!(coarse_reader.num_rows(), 1);
        let coarse_batch = coarse_reader
            .read_range(
                0..coarse_reader.num_rows(),
                Some(&[
                    BLOCK_START_COL,
                    NUM_BLOCKS_COL,
                    FIRST_DOC_ID_COL,
                    LAST_DOC_ID_COL,
                ]),
            )
            .await
            .unwrap();
        assert_eq!(
            coarse_batch[BLOCK_START_COL]
                .as_primitive::<UInt64Type>()
                .value(0),
            0
        );
        assert_eq!(
            coarse_batch[NUM_BLOCKS_COL]
                .as_primitive::<UInt32Type>()
                .value(0),
            2
        );

        let skip_batch = blocks_reader
            .read_range(
                0..blocks_reader.num_rows(),
                Some(&[FIRST_DOC_ID_COL, LAST_DOC_ID_COL, BLOCK_MAX_SCORE_COL]),
            )
            .await
            .unwrap();
        assert_eq!(skip_batch.num_rows(), 2);
        assert!(skip_batch.column_by_name(POSTING_COL).is_none());
        assert_eq!(
            skip_batch[FIRST_DOC_ID_COL]
                .as_primitive::<UInt32Type>()
                .value(0),
            0
        );
        assert_eq!(
            skip_batch[LAST_DOC_ID_COL]
                .as_primitive::<UInt32Type>()
                .value(1),
            BLOCK_SIZE as u32 + 2
        );
        let payload_batch = blocks_reader
            .read_range(0..1, Some(&[POSTING_COL]))
            .await
            .unwrap();
        let on_disk_payload_len = payload_batch[POSTING_COL].as_binary::<i64>().value(0).len();

        let posting_reader = PostingListReader::try_new_v3(
            terms_reader,
            blocks_reader,
            Some(positions_reader),
            coarse_reader,
            &LanceCache::no_cache(),
        )
        .await
        .unwrap();
        assert_eq!(
            posting_reader.posting_len_for_token(0).await.unwrap(),
            BLOCK_SIZE + 3
        );
        let posting = posting_reader
            .posting_list(0, true, &NoOpMetricsCollector)
            .await
            .unwrap();
        let PostingList::Compressed(compressed) = &posting else {
            panic!("V3 posting list should be compressed");
        };
        assert_eq!(on_disk_payload_len + 8, compressed.blocks.value(0).len());
        let actual = posting
            .iter()
            .map(|(doc_id, freq, positions)| {
                (doc_id as u32, freq, positions.unwrap().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_legacy_compressed_positions_still_readable() {
        let doc_ids = [1_u32, 3_u32];
        let frequencies = [2_u32, 3_u32];
        let posting = compress_posting_list_with_tail_codec(
            doc_ids.len(),
            doc_ids.iter(),
            frequencies.iter(),
            std::iter::once(1.0_f32),
            PostingTailCodec::Fixed32,
        )
        .unwrap();

        let mut posting_builder = ListBuilder::new(LargeBinaryBuilder::new());
        for idx in 0..posting.len() {
            posting_builder.values().append_value(posting.value(idx));
        }
        posting_builder.append(true);

        let mut positions_builder = ListBuilder::new(ListBuilder::new(LargeBinaryBuilder::new()));
        for positions in [vec![1_u32, 5_u32], vec![0_u32, 4_u32, 9_u32]] {
            let compressed = compress_positions(&positions).unwrap();
            let doc_builder = positions_builder.values();
            for idx in 0..compressed.len() {
                doc_builder.values().append_value(compressed.value(idx));
            }
            doc_builder.append(true);
        }
        positions_builder.append(true);

        let schema = Arc::new(Schema::new(vec![
            Field::new(
                POSTING_COL,
                DataType::List(Arc::new(Field::new("item", DataType::LargeBinary, true))),
                false,
            ),
            Field::new(MAX_SCORE_COL, DataType::Float32, false),
            Field::new(LENGTH_COL, DataType::UInt32, false),
            Field::new(
                POSITION_COL,
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::List(Arc::new(Field::new("item", DataType::LargeBinary, true))),
                    true,
                ))),
                false,
            ),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(posting_builder.finish()) as ArrayRef,
                Arc::new(Float32Array::from(vec![1.0])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![doc_ids.len() as u32])) as ArrayRef,
                Arc::new(positions_builder.finish()) as ArrayRef,
            ],
        )
        .unwrap();

        let posting =
            PostingList::from_batch(&batch, Some(1.0), Some(doc_ids.len() as u32)).unwrap();
        let actual = posting
            .iter()
            .map(|(doc_id, freq, positions)| {
                (doc_id as u32, freq, positions.unwrap().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![(1, 2, vec![1, 5]), (3, 3, vec![0, 4, 9]),]);
    }

    #[test]
    fn test_shared_stream_v2_without_codec_still_readable() {
        let doc_ids = [1_u32, 3_u32];
        let frequencies = [2_u32, 3_u32];
        let posting = compress_posting_list_with_tail_codec(
            doc_ids.len(),
            doc_ids.iter(),
            frequencies.iter(),
            std::iter::once(1.0_f32),
            PostingTailCodec::Fixed32,
        )
        .unwrap();

        let mut posting_builder = ListBuilder::new(LargeBinaryBuilder::new());
        for idx in 0..posting.len() {
            posting_builder.values().append_value(posting.value(idx));
        }
        posting_builder.append(true);

        let positions = vec![1_u32, 5_u32, 0_u32, 4_u32, 9_u32];
        let mut encoded_positions = Vec::new();
        encode_position_stream_block_into(
            &positions,
            &frequencies,
            PositionStreamCodec::VarintDocDelta,
            &mut encoded_positions,
        )
        .unwrap();

        let mut position_offsets = ListBuilder::new(UInt32Builder::new());
        position_offsets.values().append_value(0);
        position_offsets.append(true);

        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new(
                    POSTING_COL,
                    DataType::List(Arc::new(Field::new("item", DataType::LargeBinary, true))),
                    false,
                ),
                Field::new(MAX_SCORE_COL, DataType::Float32, false),
                Field::new(LENGTH_COL, DataType::UInt32, false),
                Field::new(COMPRESSED_POSITION_COL, DataType::LargeBinary, false),
                Field::new(
                    POSITION_BLOCK_OFFSET_COL,
                    DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
                    false,
                ),
            ],
            HashMap::from([(
                POSITIONS_LAYOUT_KEY.to_owned(),
                POSITIONS_LAYOUT_SHARED_STREAM_V2.to_owned(),
            )]),
        ));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(posting_builder.finish()) as ArrayRef,
                Arc::new(Float32Array::from(vec![1.0])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![doc_ids.len() as u32])) as ArrayRef,
                Arc::new(arrow_array::LargeBinaryArray::from(vec![Some(
                    encoded_positions.as_slice(),
                )])) as ArrayRef,
                Arc::new(position_offsets.finish()) as ArrayRef,
            ],
        )
        .unwrap();

        let posting =
            PostingList::from_batch(&batch, Some(1.0), Some(doc_ids.len() as u32)).unwrap();
        let actual = posting
            .iter()
            .map(|(doc_id, freq, positions)| {
                (doc_id as u32, freq, positions.unwrap().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![(1, 2, vec![1, 5]), (3, 3, vec![0, 4, 9]),]);
    }

    #[test]
    fn test_shared_position_stream_is_smaller_for_sparse_positions() {
        let mut builder =
            PostingListBuilder::new_with_posting_tail_codec(true, PostingTailCodec::VarintDelta);
        let mut legacy_positions = Vec::with_capacity(BLOCK_SIZE * 4);
        for doc_id in 0..(BLOCK_SIZE * 4) as u32 {
            let mut positions = vec![doc_id * 3 + 1];
            if doc_id % 8 == 0 {
                positions.push(doc_id * 3 + 2);
            }
            builder.add(doc_id, PositionRecorder::Position(positions.clone().into()));
            legacy_positions.push(positions);
        }

        let batch = builder.to_batch(vec![1.0; 4]).unwrap();
        let shared_positions_size = batch[COMPRESSED_POSITION_COL].get_buffer_memory_size()
            + batch[POSITION_BLOCK_OFFSET_COL].get_buffer_memory_size();

        let mut positions_builder = ListBuilder::new(ListBuilder::new(LargeBinaryBuilder::new()));
        for positions in legacy_positions {
            let compressed = compress_positions(&positions).unwrap();
            let doc_builder = positions_builder.values();
            for idx in 0..compressed.len() {
                doc_builder.values().append_value(compressed.value(idx));
            }
            doc_builder.append(true);
        }
        positions_builder.append(true);
        let legacy_positions_size = positions_builder.finish().get_buffer_memory_size();

        assert!(
            shared_positions_size < legacy_positions_size,
            "expected shared position stream to be smaller than legacy per-doc storage, shared={shared_positions_size}, legacy={legacy_positions_size}",
        );
    }

    #[test]
    fn test_posting_list_batch_matches_docset_scoring() {
        let mut docs = DocSet::default();
        let num_docs = BLOCK_SIZE + 3;
        for doc_id in 0..num_docs as u32 {
            docs.append(doc_id as u64, doc_id % 7 + 1);
        }

        let doc_ids = (0..num_docs as u32).collect::<Vec<_>>();
        let freqs = doc_ids
            .iter()
            .map(|doc_id| doc_id % 5 + 1)
            .collect::<Vec<_>>();

        let mut builder_scores = PostingListBuilder::new(false);
        let mut builder_docs = PostingListBuilder::new(false);
        for (&doc_id, &freq) in doc_ids.iter().zip(freqs.iter()) {
            builder_scores.add(doc_id, PositionRecorder::Count(freq));
            builder_docs.add(doc_id, PositionRecorder::Count(freq));
        }

        let block_max_scores = docs.calculate_block_max_scores(doc_ids.iter(), freqs.iter());
        let batch_scores = builder_scores.to_batch(block_max_scores).unwrap();
        let batch_docs = builder_docs
            .to_batch_with_docs(&docs, inverted_list_schema(false))
            .unwrap();

        let scores_posting = batch_scores[POSTING_COL].as_list::<i32>().value(0);
        let scores_posting = scores_posting.as_binary::<i64>();
        let docs_posting = batch_docs[POSTING_COL].as_list::<i32>().value(0);
        let docs_posting = docs_posting.as_binary::<i64>();
        assert_eq!(scores_posting, docs_posting);

        let score_left = batch_scores[MAX_SCORE_COL]
            .as_primitive::<Float32Type>()
            .value(0);
        let score_right = batch_docs[MAX_SCORE_COL]
            .as_primitive::<Float32Type>()
            .value(0);
        assert!((score_left - score_right).abs() < 1e-6);

        let len_left = batch_scores[LENGTH_COL]
            .as_primitive::<UInt32Type>()
            .value(0);
        let len_right = batch_docs[LENGTH_COL].as_primitive::<UInt32Type>().value(0);
        assert_eq!(len_left, len_right);
    }

    #[tokio::test]
    async fn test_remap_to_empty_posting_list() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());

        // index of docs:
        // 0: lance
        // 1: lake lake
        // 2: lake lake lake
        builder.tokens.add("lance".to_owned());
        builder.tokens.add("lake".to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder.posting_lists[1].add(1, PositionRecorder::Count(2));
        builder.posting_lists[1].add(2, PositionRecorder::Count(3));
        builder.docs.append(0, 1);
        builder.docs.append(1, 1);
        builder.docs.append(2, 1);
        builder.write(store.as_ref()).await.unwrap();

        let index = InvertedPartition::load(
            store.clone(),
            0,
            None,
            &LanceCache::no_cache(),
            TokenSetFormat::default(),
        )
        .await
        .unwrap();
        let mut builder = index.into_builder().await.unwrap();

        let mapping = HashMap::from([(0, None), (2, Some(3))]);
        builder.remap(&RowAddrRemap::direct(mapping)).await.unwrap();

        // after remap, the doc 0 is removed, and the doc 2 is updated to 3
        assert_eq!(builder.tokens.len(), 1);
        assert_eq!(builder.tokens.get("lake"), Some(0));
        assert_eq!(builder.posting_lists.len(), 1);
        assert_eq!(builder.posting_lists[0].len(), 2);
        assert_eq!(builder.docs.len(), 2);
        assert_eq!(builder.docs.row_id(0), 1);
        assert_eq!(builder.docs.row_id(1), 3);

        builder.write(store.as_ref()).await.unwrap();

        // remap to delete all docs
        let mapping = HashMap::from([(1, None), (3, None)]);
        builder.remap(&RowAddrRemap::direct(mapping)).await.unwrap();

        assert_eq!(builder.tokens.len(), 0);
        assert_eq!(builder.posting_lists.len(), 0);
        assert_eq!(builder.docs.len(), 0);

        builder.write(store.as_ref()).await.unwrap();
    }

    #[tokio::test]
    async fn test_posting_cache_conflict_across_partitions() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Create first partition with one token and posting list length 1
        let mut builder1 = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder1.tokens.add("test".to_owned());
        builder1.posting_lists.push(PostingListBuilder::new(false));
        builder1.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder1.docs.append(100, 1); // row_id=100, num_tokens=1
        builder1.write(store.as_ref()).await.unwrap();

        // Create second partition with one token and posting list length 4
        let mut builder2 = InnerBuilder::new(1, false, TokenSetFormat::default());
        builder2.tokens.add("test".to_owned()); // Use same token to test cache prefix fix
        builder2.posting_lists.push(PostingListBuilder::new(false));
        builder2.posting_lists[0].add(0, PositionRecorder::Count(2));
        builder2.posting_lists[0].add(1, PositionRecorder::Count(1));
        builder2.posting_lists[0].add(2, PositionRecorder::Count(3));
        builder2.posting_lists[0].add(3, PositionRecorder::Count(1));
        builder2.docs.append(200, 2); // row_id=200, num_tokens=2
        builder2.docs.append(201, 1); // row_id=201, num_tokens=1
        builder2.docs.append(202, 3); // row_id=202, num_tokens=3
        builder2.docs.append(203, 1); // row_id=203, num_tokens=1
        builder2.write(store.as_ref()).await.unwrap();

        // Create metadata file with both partitions
        let metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0u64, 1u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        // Load the inverted index
        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        // Verify the index structure
        assert_eq!(index.partitions.len(), 2);
        assert_eq!(index.partitions[0].tokens.len(), 1);
        assert_eq!(index.partitions[1].tokens.len(), 1);

        // Verify the partitions were loaded correctly

        // Verify posting list lengths (note: partition order may differ from creation order).
        // `posting_len_for_token` works for both legacy and v2 layouts without
        // forcing the V2-only bulk metadata load.
        let pl_0_0 = index.partitions[0]
            .inverted_list
            .posting_len_for_token(0)
            .await
            .unwrap();
        let pl_1_0 = index.partitions[1]
            .inverted_list
            .posting_len_for_token(0)
            .await
            .unwrap();
        if index.partitions[0].id() == 0 {
            assert_eq!(pl_0_0, 1);
            assert_eq!(pl_1_0, 4);
            assert_eq!(index.partitions[0].docs.len(), 1);
            assert_eq!(index.partitions[1].docs.len(), 4);
        } else {
            assert_eq!(pl_0_0, 4);
            assert_eq!(pl_1_0, 1);
            assert_eq!(index.partitions[0].docs.len(), 4);
            assert_eq!(index.partitions[1].docs.len(), 1);
        }

        // Prewarm the inverted index (this loads posting lists into cache)
        index.prewarm().await.unwrap();

        let tokens = Arc::new(Tokens::new(vec!["test".to_string()], DocType::Text));
        let params = Arc::new(FtsSearchParams::new().with_limit(Some(10)));
        let prefilter = Arc::new(NoFilter);
        let metrics = Arc::new(NoOpMetricsCollector);

        let (row_ids, scores) = index
            .bm25_search(tokens, params, Operator::Or, prefilter, metrics, None)
            .await
            .unwrap();

        // Verify that we got search results
        // Expected to find 5 documents: 1 from first partition, 4 from second partition
        assert_eq!(row_ids.len(), 5, "row_ids: {:?}", row_ids);
        assert!(!row_ids.is_empty(), "Should find at least some documents");
        assert_eq!(row_ids.len(), scores.len());

        // All scores should be positive since all documents contain the search token
        for &score in &scores {
            assert!(score > 0.0, "All scores should be positive");
        }

        // Check that we got results from both partitions
        assert!(
            row_ids.contains(&100),
            "Should contain row_id from partition 0"
        );
        assert!(
            row_ids.iter().any(|&id| id >= 200),
            "Should contain row_id from partition 1"
        );
    }

    #[tokio::test]
    async fn test_modern_prewarm_shrinks_cached_posting_buffers() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.tokens.add("alpha".to_owned());
        builder.tokens.add("beta".to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder.posting_lists[0].add(1, PositionRecorder::Count(2));
        builder.posting_lists[1].add(2, PositionRecorder::Count(3));
        builder.posting_lists[1].add(3, PositionRecorder::Count(4));
        builder.docs.append(100, 1);
        builder.docs.append(101, 2);
        builder.docs.append(102, 3);
        builder.docs.append(103, 4);
        builder.write(store.as_ref()).await.unwrap();

        let metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();
        let inverted_list = &index.partitions[0].inverted_list;
        assert!(
            !inverted_list.is_legacy_layout(),
            "test should use modern posting layout"
        );

        inverted_list.prewarm_posting_lists(false, 2).await.unwrap();

        // The two tiny tokens land in a single cache group [0, 2) (issue
        // #7040); both postings are read out of that group entry.
        let (start, end) = inverted_list.group_range_for_token(0).unwrap();
        let group = inverted_list
            .index_cache
            .get_with_key(&PostingListGroupKey { start, end })
            .await
            .unwrap();

        let PostingList::Compressed(alpha) = group.get(0).unwrap() else {
            panic!("expected compressed posting list for token 0");
        };
        let PostingList::Compressed(beta) = group.get(1).unwrap() else {
            panic!("expected compressed posting list for token 1");
        };

        assert_ne!(
            alpha.blocks.values().as_ptr(),
            beta.blocks.values().as_ptr(),
            "prewarm should not leave cached posting lists sharing the same values buffer"
        );
    }

    #[test]
    fn test_group_aligned_chunk_end_boundary_cases() {
        let starts = [0, 3, 7, 10];
        let token_count = 13;

        assert_eq!(
            group_aligned_chunk_end(&starts, token_count, 0, 5),
            3,
            "chunk should snap back to the largest group boundary that fits"
        );
        assert_eq!(
            group_aligned_chunk_end(&starts, token_count, 3, 6),
            7,
            "oversized groups should run as one chunk"
        );
        assert_eq!(
            group_aligned_chunk_end(&starts, token_count, 7, 10),
            10,
            "an exact next group boundary should be selected"
        );
        assert_eq!(
            group_aligned_chunk_end(&starts, token_count, 10, 12),
            13,
            "the last group should extend to token_count"
        );
        assert_eq!(
            group_aligned_chunk_end(&starts, token_count, 7, 13),
            13,
            "token_count should act as the final boundary"
        );
    }

    #[test]
    fn test_group_start_indices_for_chunk_boundary_cases() {
        let starts = [0, 3, 7, 10];
        let token_count = 13;
        let ranges_for_chunk = |tok_start, tok_end| {
            group_start_indices_for_chunk(&starts, tok_start, tok_end)
                .map(|group_idx| group_range_for_start_index(&starts, token_count, group_idx))
                .collect::<Vec<_>>()
        };

        assert_eq!(
            ranges_for_chunk(0, 7),
            vec![(0, 3), (3, 7)],
            "publish should include only groups that start in the chunk"
        );
        assert_eq!(
            ranges_for_chunk(7, 13),
            vec![(7, 10), (10, 13)],
            "publish should include the final group ending at token_count"
        );
        assert_eq!(
            ranges_for_chunk(3, 10),
            vec![(3, 7), (7, 10)],
            "publish selection should work for an interior chunk"
        );
    }

    #[test]
    fn test_prewarm_chunk_ranges_preserve_group_boundaries() {
        let starts = [0, 3, 7, 10];
        assert_eq!(
            prewarm_chunk_ranges(Some(&starts), 13, 5),
            vec![(0, 3), (3, 7), (7, 10), (10, 13)],
            "grouped chunk ranges must never split a posting cache group"
        );
        assert_eq!(
            prewarm_chunk_ranges(None, 13, 5),
            vec![(0, 5), (5, 10), (10, 13)],
            "ungrouped chunk ranges should use plain token ranges"
        );
    }

    /// Prewarming a large partition in multiple chunks must end up holding exactly the
    /// same per-token posting lists (doc ids and frequencies) as the whole-file path.
    /// Parametrized over layout: the legacy-v1 chunk path rebases global offsets to
    /// chunk-local rows, which the v2 one-row-per-token path never exercises.
    #[rstest::rstest]
    #[case::v1(InvertedListFormatVersion::V1)]
    #[case::v2(InvertedListFormatVersion::V2)]
    #[case::v3(InvertedListFormatVersion::V3)]
    #[tokio::test]
    async fn test_prewarm_streams_in_chunks_preserves_content(
        #[case] format_version: InvertedListFormatVersion,
    ) {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // One partition with many tokens (so it spans many chunks) and several
        // docs per token (so each token is more than one posting row).
        const NUM_TOKENS: u32 = 20;
        const DOCS_PER_TOKEN: u32 = 3;
        let posting_tail_codec = format_version.posting_tail_codec();
        let mut builder = InnerBuilder::new_with_format_version(
            0,
            false,
            TokenSetFormat::default(),
            format_version,
        );
        // Small groups so the partition spans several; chunks snap to whole groups,
        // so several groups are needed to stream in more than one chunk.
        builder.group_config = PostingGroupConfig {
            target_bytes: 4096,
            max_tokens: 4,
        };
        // expected[token] = [(doc_id, frequency)] in stored (doc-id) order.
        let mut expected: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut doc_id = 0u64;
        for t in 0..NUM_TOKENS {
            builder.tokens.add(format!("tok_{t:03}"));
            let mut posting =
                PostingListBuilder::new_with_posting_tail_codec(false, posting_tail_codec);
            let mut docs = Vec::new();
            for _ in 0..DOCS_PER_TOKEN {
                posting.add(doc_id as u32, PositionRecorder::Count(1));
                builder.docs.append(doc_id, 1);
                docs.push((doc_id as u32, 1));
                doc_id += 1;
            }
            expected.push(docs);
            builder.posting_lists.push(posting);
        }
        builder.write(store.as_ref()).await.unwrap();

        let mut metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
            (
                POSTING_TAIL_CODEC_KEY.to_owned(),
                posting_tail_codec.as_str().to_owned(),
            ),
        ]);
        if format_version.uses_block_row_layout() {
            metadata.insert(
                POSTING_BLOCK_LAYOUT_KEY.to_owned(),
                POSTING_BLOCK_LAYOUT_BLOCK_ROW_V3.to_owned(),
            );
        }
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let cache = Arc::new(LanceCache::with_capacity(1 << 20));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();
        let inverted_list = &index.partitions[0].inverted_list;
        assert_eq!(inverted_list.len(), NUM_TOKENS as usize);

        // Force a small chunk so the partition deterministically splits; with
        // CHUNK_TOKENS < NUM_TOKENS each chunk is bounded below the whole partition.
        const CHUNK_TOKENS: usize = 6;
        let chunk_count = inverted_list
            .prewarm_posting_lists_chunked(false, Some(CHUNK_TOKENS), 2)
            .await
            .unwrap();

        // (1) The partition was streamed in multiple chunks. The exact count is
        // group-alignment-dependent (chunks snap to whole groups), so just
        // require more than one.
        assert!(
            chunk_count > 1,
            "single partition must be streamed in more than one chunk, got {chunk_count}"
        );

        // (2) Correctness: every token's posting list round-trips with exactly
        // the doc ids and frequencies of the whole-file path.
        for token_id in 0..NUM_TOKENS {
            let actual = inverted_list
                .posting_list(token_id, false, &NoOpMetricsCollector)
                .await
                .unwrap()
                .iter()
                .map(|(doc_id, freq, _positions)| (doc_id as u32, freq))
                .collect::<Vec<_>>();
            assert_eq!(
                actual, expected[token_id as usize],
                "token {token_id} posting list mismatch after chunked prewarm"
            );
        }
    }

    /// With positions, the chunked prewarm must strip positions into their own
    /// per-token cache entries (leaving the posting cache positions-free) and still
    /// round-trip exact doc ids, frequencies, and positions across chunk boundaries.
    #[rstest::rstest]
    #[case::v2(InvertedListFormatVersion::V2)]
    #[case::v3(InvertedListFormatVersion::V3)]
    #[tokio::test]
    async fn test_prewarm_streams_in_chunks_with_positions(
        #[case] format_version: InvertedListFormatVersion,
    ) {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let posting_tail_codec = format_version.posting_tail_codec();
        const NUM_TOKENS: u32 = 16;
        const DOCS_PER_TOKEN: u32 = 3;
        let mut builder = InnerBuilder::new_with_format_version(
            0,
            true,
            TokenSetFormat::default(),
            format_version,
        );
        builder.group_config = PostingGroupConfig {
            target_bytes: 4096,
            max_tokens: 4,
        };
        // expected[token] = [(doc_id, frequency, positions)].
        let mut expected: Vec<Vec<(u32, u32, Vec<u32>)>> = Vec::new();
        let mut doc_id = 0u64;
        for t in 0..NUM_TOKENS {
            builder.tokens.add(format!("tok_{t:03}"));
            let mut posting =
                PostingListBuilder::new_with_posting_tail_codec(true, posting_tail_codec);
            let mut docs = Vec::new();
            for _ in 0..DOCS_PER_TOKEN {
                let positions = vec![t % 3, t % 3 + 2, t % 3 + 5];
                posting.add(
                    doc_id as u32,
                    PositionRecorder::Position(positions.clone().into()),
                );
                builder.docs.append(doc_id, positions.len() as u32);
                docs.push((doc_id as u32, positions.len() as u32, positions));
                doc_id += 1;
            }
            expected.push(docs);
            builder.posting_lists.push(posting);
        }
        builder.write(store.as_ref()).await.unwrap();

        let mut metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default().with_position(true)).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
            (
                POSTING_TAIL_CODEC_KEY.to_owned(),
                posting_tail_codec.as_str().to_owned(),
            ),
            (
                POSITIONS_LAYOUT_KEY.to_owned(),
                POSITIONS_LAYOUT_SHARED_STREAM_V2.to_owned(),
            ),
            (
                POSITIONS_CODEC_KEY.to_owned(),
                PositionStreamCodec::PackedDelta.as_str().to_owned(),
            ),
        ]);
        if format_version.uses_block_row_layout() {
            metadata.insert(
                POSTING_BLOCK_LAYOUT_KEY.to_owned(),
                POSTING_BLOCK_LAYOUT_BLOCK_ROW_V3.to_owned(),
            );
        }
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let cache = Arc::new(LanceCache::with_capacity(1 << 20));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();
        let inverted_list = &index.partitions[0].inverted_list;

        const CHUNK_TOKENS: usize = 5;
        let chunk_count = inverted_list
            .prewarm_posting_lists_chunked(true, Some(CHUNK_TOKENS), 2)
            .await
            .unwrap();
        assert!(
            chunk_count > 1,
            "partition must be streamed in more than one chunk, got {chunk_count}"
        );

        for token_id in 0..NUM_TOKENS {
            // The prewarmed posting cache entry is positions-free.
            if let Some((start, end)) = inverted_list.group_range_for_token(token_id) {
                let group = inverted_list
                    .index_cache
                    .get_with_key(&PostingListGroupKey { start, end })
                    .await
                    .unwrap();
                let slot = (token_id - start) as usize;
                assert!(
                    !group.get(slot).unwrap().has_position(),
                    "token {token_id} posting cache entry must be positions-free after prewarm"
                );
            } else {
                let posting = inverted_list
                    .index_cache
                    .get_with_key(&PostingListKey { token_id })
                    .await
                    .unwrap();
                assert!(
                    !posting.has_position(),
                    "token {token_id} posting cache entry must be positions-free after prewarm"
                );
            }

            // Full content (doc ids, frequencies, positions) round-trips; the
            // positions come from the dedicated per-token cache prewarm populated.
            let actual = inverted_list
                .posting_list(token_id, true, &NoOpMetricsCollector)
                .await
                .unwrap()
                .iter()
                .map(|(doc_id, freq, positions)| {
                    (doc_id as u32, freq, positions.unwrap().collect::<Vec<_>>())
                })
                .collect::<Vec<_>>();
            assert_eq!(
                actual, expected[token_id as usize],
                "token {token_id} posting list / positions mismatch after chunked prewarm"
            );
        }
    }

    /// IO accounting for the IO-counting stats test below: tracks bytes
    /// pulled from the posting file so we can assert that the stats path is
    /// O(1) in num_unique_tokens.
    #[derive(Debug, Default)]
    struct PostingMetadataCounter {
        rows_read: std::sync::atomic::AtomicUsize,
        metadata_rows_read: std::sync::atomic::AtomicUsize,
        read_range_calls: std::sync::atomic::AtomicUsize,
    }

    impl PostingMetadataCounter {
        fn rows_read(&self) -> usize {
            self.rows_read.load(std::sync::atomic::Ordering::Relaxed)
        }
        fn metadata_rows_read(&self) -> usize {
            self.metadata_rows_read
                .load(std::sync::atomic::Ordering::Relaxed)
        }
        fn read_range_calls(&self) -> usize {
            self.read_range_calls
                .load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    struct CountingPostingReader {
        inner: Arc<dyn IndexReader>,
        counter: Arc<PostingMetadataCounter>,
    }

    #[async_trait]
    impl IndexReader for CountingPostingReader {
        async fn read_record_batch(&self, n: u64, batch_size: u64) -> Result<RecordBatch> {
            self.inner.read_record_batch(n, batch_size).await
        }
        async fn read_global_buffer(&self, index: u32) -> Result<bytes::Bytes> {
            self.inner.read_global_buffer(index).await
        }
        async fn read_range(
            &self,
            range: std::ops::Range<usize>,
            projection: Option<&[&str]>,
        ) -> Result<RecordBatch> {
            let n = range.end - range.start;
            self.counter
                .read_range_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.counter
                .rows_read
                .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
            let touches_metadata = projection
                .map(|cols| cols.contains(&MAX_SCORE_COL) || cols.contains(&LENGTH_COL))
                .unwrap_or(false);
            if touches_metadata {
                self.counter
                    .metadata_rows_read
                    .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
            }
            self.inner.read_range(range, projection).await
        }
        async fn num_batches(&self, batch_size: u64) -> u32 {
            self.inner.num_batches(batch_size).await
        }
        fn num_rows(&self) -> usize {
            self.inner.num_rows()
        }
        fn schema(&self) -> &lance_core::datatypes::Schema {
            self.inner.schema()
        }
    }

    #[derive(Debug)]
    struct CountingStore {
        inner: Arc<dyn IndexStore>,
        posting_file: String,
        counter: Arc<PostingMetadataCounter>,
    }

    impl DeepSizeOf for CountingStore {
        fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
            self.inner.deep_size_of_children(context)
        }
    }

    #[async_trait]
    impl IndexStore for CountingStore {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn clone_arc(&self) -> Arc<dyn IndexStore> {
            Arc::new(Self {
                inner: self.inner.clone(),
                posting_file: self.posting_file.clone(),
                counter: self.counter.clone(),
            })
        }
        fn io_parallelism(&self) -> usize {
            self.inner.io_parallelism()
        }
        fn with_io_priority(&self, io_priority: u64) -> Arc<dyn IndexStore> {
            Arc::new(Self {
                inner: self.inner.with_io_priority(io_priority),
                posting_file: self.posting_file.clone(),
                counter: self.counter.clone(),
            })
        }
        async fn new_index_file(
            &self,
            name: &str,
            schema: Arc<arrow_schema::Schema>,
        ) -> Result<Box<dyn crate::scalar::IndexWriter>> {
            self.inner.new_index_file(name, schema).await
        }
        async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>> {
            let reader = self.inner.open_index_file(name).await?;
            if name == self.posting_file {
                Ok(Arc::new(CountingPostingReader {
                    inner: reader,
                    counter: self.counter.clone(),
                }))
            } else {
                Ok(reader)
            }
        }
        async fn copy_index_file(
            &self,
            name: &str,
            dest_store: &dyn IndexStore,
        ) -> Result<crate::scalar::IndexFile> {
            self.inner.copy_index_file(name, dest_store).await
        }
        async fn copy_index_file_to(
            &self,
            name: &str,
            new_name: &str,
            dest_store: &dyn IndexStore,
        ) -> Result<crate::scalar::IndexFile> {
            self.inner
                .copy_index_file_to(name, new_name, dest_store)
                .await
        }
        async fn rename_index_file(
            &self,
            name: &str,
            new_name: &str,
        ) -> Result<crate::scalar::IndexFile> {
            self.inner.rename_index_file(name, new_name).await
        }
        async fn delete_index_file(&self, name: &str) -> Result<()> {
            self.inner.delete_index_file(name).await
        }
        async fn list_files_with_sizes(&self) -> Result<Vec<crate::scalar::IndexFile>> {
            self.inner.list_files_with_sizes().await
        }
    }

    // Returns the `TempObjDir` guard so callers keep the backing store alive
    // for the index's lifetime: the deferred DocSet re-opens the docs file on
    // demand (it does not pin an open handle), so the files must still exist
    // when the test exercises a scoring path.
    async fn load_counted_v2_index(
        num_tokens: usize,
        cache: LanceCache,
    ) -> (Arc<InvertedIndex>, Arc<PostingMetadataCounter>, TempObjDir) {
        let tmpdir = TempObjDir::default();
        let inner_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        for i in 0..num_tokens {
            builder.tokens.add(format!("t{}", i));
            let mut pl = PostingListBuilder::new(false);
            pl.add(i as u32, PositionRecorder::Count(1));
            builder.posting_lists.push(pl);
            builder.docs.append(i as u64, 1);
        }
        builder.write(inner_store.as_ref()).await.unwrap();

        let metadata = HashMap::from([
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
        ]);
        let mut writer = inner_store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let counter = Arc::new(PostingMetadataCounter::default());
        let counting_store: Arc<dyn IndexStore> = Arc::new(CountingStore {
            inner: inner_store,
            posting_file: posting_file_path(0),
            counter: counter.clone(),
        });
        let index = InvertedIndex::load(counting_store, None, &cache)
            .await
            .unwrap();
        (index, counter, tmpdir)
    }

    /// IO regression test for the lazy posting-metadata refactor. Builds a
    /// v2 InvertedIndex with `num_tokens` tokens in a single partition,
    /// wraps the IndexStore so reads against the posting file are counted,
    /// then asserts:
    ///
    /// * `InvertedIndex::load` does not touch the posting file at all
    ///   (`InvertedPartition::load` only needs the token file and docs file).
    /// * `bm25_stats_for_terms(["t0"])` reads exactly one metadata row from
    ///   the posting file for token 0 regardless of how many unique tokens the
    ///   partition has.
    ///
    /// Before this refactor, `PostingListReader::try_new` did
    /// `read_range(0..num_rows, [MAX_SCORE_COL, LENGTH_COL])`, so the
    /// `metadata_rows_read` figure scaled linearly with `num_tokens` even
    /// when nobody asked for those stats. The cases below exercise that
    /// scaling explicitly.
    #[rstest::rstest]
    #[case::tokens_10(10)]
    #[case::tokens_100(100)]
    #[case::tokens_1000(1000)]
    #[tokio::test]
    async fn test_bm25_stats_for_terms_is_lazy(#[case] num_tokens: usize) {
        let (index, counter, _tmpdir) =
            load_counted_v2_index(num_tokens, LanceCache::no_cache()).await;
        assert!(
            !index.partitions[0].inverted_list.is_legacy_layout(),
            "this test only proves the lazy path for v2 indexes",
        );

        // Opening the partition must not pull anything from the posting file.
        // Pre-fix, `PostingListReader::try_new` issued one read_range here for
        // [MAX_SCORE_COL, LENGTH_COL] covering every unique token.
        assert_eq!(
            counter.read_range_calls(),
            0,
            "InvertedIndex::load must not read the posting file (was {} calls)",
            counter.read_range_calls(),
        );
        assert_eq!(counter.rows_read(), 0);

        let (total_tokens, num_docs, dfs) = index
            .bm25_stats_for_terms(&["t0".to_string()])
            .await
            .unwrap();
        assert_eq!(total_tokens, num_tokens as u64);
        assert_eq!(num_docs, num_tokens);
        assert_eq!(dfs, vec![1]);

        // Stats must pull a constant number of metadata rows from the posting
        // file regardless of how many tokens the partition has. One term, one
        // partition, one row.
        assert_eq!(
            counter.metadata_rows_read(),
            1,
            "stats path should read exactly 1 metadata row per (term, partition); \
             got {} (read_range_calls={}, rows_read={}, num_tokens={})",
            counter.metadata_rows_read(),
            counter.read_range_calls(),
            counter.rows_read(),
            num_tokens,
        );
    }

    #[tokio::test]
    async fn test_bm25_stats_for_terms_reuses_posting_metadata_cache() {
        let cache = LanceCache::with_capacity(1024 * 1024);
        let (index, counter, _tmpdir) = load_counted_v2_index(100, cache.clone()).await;

        let terms = ["t0".to_string()];
        let first = index.bm25_stats_for_terms(&terms).await.unwrap();
        assert_eq!(first, (100, 100, vec![1]));
        assert_eq!(counter.metadata_rows_read(), 1);

        let second = index.bm25_stats_for_terms(&terms).await.unwrap();
        assert_eq!(second, first);
        assert_eq!(
            counter.metadata_rows_read(),
            1,
            "repeated stats for the same token should reuse cached posting metadata",
        );
    }

    #[tokio::test]
    async fn test_aggregate_corpus_stats_reuses_cached_value() {
        let (index, _counter, _tmpdir) = load_counted_v2_index(100, LanceCache::no_cache()).await;
        assert!(index.corpus_stats.get().is_none());

        let first = index.aggregate_corpus_stats().await.unwrap();
        assert_eq!(first, (100, 100));
        assert_eq!(index.corpus_stats.get().copied(), Some(first));

        let second = index.aggregate_corpus_stats().await.unwrap();
        assert_eq!(second, first);
    }

    #[tokio::test]
    async fn test_grouped_posting_lists_read_one_group_per_neighborhood() {
        // Cold-start scoring must not bulk-read the full `0..num_tokens`
        // metadata table. With small-posting grouping (issue #7040), scoring
        // K adjacent cold tokens shares a single group cache entry: one
        // read_range bounded by the group size, independent of the partition's
        // total token count.
        let num_tokens = 500;
        let queried_tokens: [u32; 4] = [0, 1, 2, 3];
        let (index, counter, _tmpdir) =
            load_counted_v2_index(num_tokens, LanceCache::no_cache()).await;
        let inverted_list = index.partitions[0].inverted_list.clone();
        assert!(
            !inverted_list.is_legacy_layout(),
            "this test only proves the lazy path for v2 indexes",
        );
        assert!(
            inverted_list.group_starts.is_some(),
            "freshly written v2 index should carry posting group offsets",
        );

        // This fixture uses a no-op cache, so each call re-reads; that isolates
        // the per-query read shape. Each posting_list call reads exactly its
        // own group — bounded by the group size, never the full token table.
        let metrics = Arc::new(NoOpMetricsCollector);
        for token_id in queried_tokens {
            inverted_list
                .posting_list(token_id, false, metrics.as_ref())
                .await
                .unwrap();
        }

        let (start, end) = inverted_list.group_range_for_token(0).unwrap();
        let group_len = (end - start) as usize;
        assert!(
            (queried_tokens.len()..num_tokens).contains(&group_len),
            "group [{start}, {end}) should cover the queried neighborhood but be \
             far smaller than the {num_tokens}-token table",
        );
        assert_eq!(
            counter.read_range_calls(),
            queried_tokens.len(),
            "each cold token should read exactly its own group, no bulk read",
        );
        assert_eq!(
            counter.metadata_rows_read(),
            queried_tokens.len() * group_len,
            "each query reads one group's metadata rows ({group_len}), not the \
             full {num_tokens}-row table",
        );
    }

    /// Build a single-partition v2 index where every token's posting list spans
    /// `docs_per_token` docs. Small `docs_per_token` yields tiny posting lists
    /// that the writer packs densely into shared cache groups.
    async fn load_v2_index_with_grouped_postings(
        num_tokens: usize,
        docs_per_token: usize,
    ) -> (Arc<InvertedIndex>, Arc<LanceCache>) {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let num_docs = num_tokens * docs_per_token;
        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        for token_id in 0..num_tokens {
            builder.tokens.add(format!("t{token_id}"));
            let mut pl = PostingListBuilder::new(false);
            for d in 0..docs_per_token {
                let doc_id = (token_id * docs_per_token + d) as u32;
                pl.add(doc_id, PositionRecorder::Count(1));
            }
            builder.posting_lists.push(pl);
        }
        for doc in 0..num_docs {
            builder.docs.append(doc as u64, 1);
        }
        builder.write(store.as_ref()).await.unwrap();

        let metadata = HashMap::from([
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        // The inverted list keeps only a `WeakLanceCache`, so the caller must
        // hold this `Arc<LanceCache>` alive for the cache to stay usable.
        let cache = Arc::new(LanceCache::with_capacity(1 << 30));
        let index = InvertedIndex::load(store, None, cache.as_ref())
            .await
            .unwrap();
        (index, cache)
    }

    /// The read path decodes a posting-list group by slicing one buffer read for
    /// the whole `[start, end)` row range, so every posting list in a cached
    /// group shares a single `blocks` buffer. `DeepSizeOf` must count each
    /// posting's slice of that buffer, not the whole buffer once per posting —
    /// otherwise a group of N postings reports ~N times its real footprint.
    #[rstest::rstest]
    #[case::single_doc_terms(512, 1)]
    #[case::small_terms(512, 4)]
    #[case::medium_terms(256, 32)]
    #[tokio::test]
    async fn test_read_path_group_size_counts_slices_not_shared_buffer(
        #[case] num_tokens: usize,
        #[case] docs_per_token: usize,
    ) {
        let (index, _cache) = load_v2_index_with_grouped_postings(num_tokens, docs_per_token).await;
        let inverted_list = index.partitions[0].inverted_list.clone();
        assert!(!inverted_list.is_legacy_layout(), "expected v2 layout");
        assert!(
            inverted_list.group_starts.is_some(),
            "expected grouped posting lists"
        );

        // Populate the group cache via the same path a query uses.
        inverted_list
            .posting_list(0, false, &NoOpMetricsCollector)
            .await
            .unwrap();
        let (start, end) = inverted_list.group_range_for_token(0).unwrap();
        let group = inverted_list
            .index_cache
            .get_with_key(&PostingListGroupKey { start, end })
            .await
            .unwrap();

        // Sum what counting the full backing buffer once per posting list would
        // charge, and confirm the postings really do share a single buffer.
        let mut distinct_buffers = std::collections::HashSet::new();
        let mut charged_if_counted_per_posting = 0usize;
        for posting in &group.posting_lists {
            let PostingList::Compressed(compressed) = posting else {
                panic!("expected compressed posting lists");
            };
            charged_if_counted_per_posting += compressed.blocks.get_buffer_memory_size();
            distinct_buffers.insert(compressed.blocks.values().as_ptr());
        }
        let posting_count = group.posting_lists.len();

        assert!(
            posting_count > 1,
            "default grouping should pack multiple tiny postings into one group"
        );
        assert_eq!(
            distinct_buffers.len(),
            1,
            "read-path postings in a group should share one backing buffer"
        );
        // With slice-aware accounting the shared buffer is counted ~once, so the
        // whole group costs far less than counting it once per posting list.
        let reported = group.deep_size_of();
        assert!(
            reported < charged_if_counted_per_posting / 2,
            "group deep_size_of {reported}B should not scale with the {posting_count}x-counted \
             shared buffer ({charged_if_counted_per_posting}B)"
        );
    }

    // ===========================================================================
    // Regression tests for index-cache size accounting of cached posting lists.
    //
    // A cached posting list is a *slice* of a buffer read for a whole posting-list
    // group, so its `DeepSizeOf` impl must charge only the bytes the slice
    // references, not the full shared backing buffer. These lock that in: each
    // builds an array that references a small slice of a much larger buffer and
    // asserts `deep_size_of()` tracks the slice, not the buffer.
    // ===========================================================================

    /// Build a `List<Int32>` of `num_sublists` x `ints_per_sublist`, then return
    /// the slice `[off, off + len)`. The returned array shares the full backing
    /// buffers, so `values().get_buffer_memory_size()` still reports the whole
    /// thing — the slicing-unaware over-count the fix targets.
    fn sliced_int32_list(
        num_sublists: usize,
        ints_per_sublist: usize,
        off: usize,
        len: usize,
    ) -> ListArray {
        let mut builder = ListBuilder::new(Int32Builder::new());
        for s in 0..num_sublists {
            for i in 0..ints_per_sublist {
                builder
                    .values()
                    .append_value((s * ints_per_sublist + i) as i32);
            }
            builder.append(true);
        }
        builder.finish().slice(off, len)
    }

    #[test]
    fn test_compressed_posting_deep_size_counts_only_referenced_blocks_slice() {
        const ELEM_BYTES: usize = 256;
        const TOTAL_ELEMS: usize = 64;
        const SLICE_OFF: usize = 10;
        const SLICE_LEN: usize = 2;

        let mut builder = LargeBinaryBuilder::new();
        for _ in 0..TOTAL_ELEMS {
            builder.append_value(vec![7u8; ELEM_BYTES]);
        }
        let full = builder.finish();
        let blocks = full.slice(SLICE_OFF, SLICE_LEN);

        let posting = CompressedPostingList::new(
            blocks,
            1.0,
            SLICE_LEN as u32,
            PostingTailCodec::Fixed32,
            None,
        );

        let full_backing = full.get_buffer_memory_size();
        let slice_bytes = SLICE_LEN * ELEM_BYTES;
        let reported = posting.deep_size_of();

        assert!(
            reported < full_backing / 4,
            "deep_size_of {reported}B must not count the {full_backing}B shared buffer"
        );
        assert!(
            reported <= slice_bytes * 2,
            "deep_size_of {reported}B should track the ~{slice_bytes}B referenced slice"
        );
    }

    #[test]
    fn test_plain_posting_deep_size_counts_only_referenced_positions_slice() {
        const SUBLISTS: usize = 64;
        const INTS: usize = 64;
        const SLICE_LEN: usize = 2;

        let positions = sliced_int32_list(SUBLISTS, INTS, 10, SLICE_LEN);
        let row_ids = ScalarBuffer::from(vec![0u64, 1]);
        let frequencies = ScalarBuffer::from(vec![1.0f32, 1.0]);
        let posting =
            PlainPostingList::new(row_ids, frequencies, Some(1.0), Some(positions.clone()));

        let full_backing = positions.values().get_buffer_memory_size();
        let slice_bytes = SLICE_LEN * INTS * std::mem::size_of::<i32>();
        let reported = posting.deep_size_of();

        assert!(
            reported < full_backing / 4,
            "deep_size_of {reported}B must not count the {full_backing}B shared positions buffer"
        );
        assert!(
            reported <= slice_bytes * 2 + 64,
            "deep_size_of {reported}B should track the ~{slice_bytes}B referenced slice"
        );
    }

    #[test]
    fn test_legacy_per_doc_positions_deep_size_counts_only_referenced_slice() {
        const SUBLISTS: usize = 64;
        const INTS: usize = 64;
        const SLICE_LEN: usize = 2;

        let positions = sliced_int32_list(SUBLISTS, INTS, 10, SLICE_LEN);
        let full_backing = positions.values().get_buffer_memory_size();
        let slice_bytes = SLICE_LEN * INTS * std::mem::size_of::<i32>();

        let storage = CompressedPositionStorage::LegacyPerDoc(positions);
        let reported = storage.deep_size_of();
        assert!(
            reported < full_backing / 4,
            "CompressedPositionStorage deep_size_of {reported}B must not count the \
             {full_backing}B shared buffer"
        );
        assert!(
            reported <= slice_bytes * 2 + 64,
            "deep_size_of {reported}B should track the ~{slice_bytes}B referenced slice"
        );

        // The `Positions` cache wrapper must report the same slice-aware size.
        let wrapped = Positions(storage).deep_size_of();
        assert!(
            wrapped < full_backing / 4,
            "Positions deep_size_of {wrapped}B must not count the {full_backing}B shared buffer"
        );
    }

    #[tokio::test]
    async fn test_prewarm_with_positions_populates_separate_position_cache() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new_with_format_version(
            0,
            true,
            TokenSetFormat::default(),
            InvertedListFormatVersion::V1,
        );
        builder.tokens.add("hello".to_owned());
        builder.tokens.add("world".to_owned());
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                true,
                PostingTailCodec::Fixed32,
            ));
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                true,
                PostingTailCodec::Fixed32,
            ));
        builder.posting_lists[0].add(0, PositionRecorder::Position(vec![0].into()));
        builder.posting_lists[1].add(0, PositionRecorder::Position(vec![1].into()));
        builder.posting_lists[0].add(1, PositionRecorder::Position(vec![0].into()));
        builder.posting_lists[1].add(1, PositionRecorder::Position(vec![2].into()));
        builder.docs.append(100, 2);
        builder.docs.append(101, 2);
        builder.write(store.as_ref()).await.unwrap();

        let metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0_u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default().with_position(true)).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        index
            .prewarm_with_options(&FtsPrewarmOptions::new().with_position(true))
            .await
            .unwrap();

        let inverted_list = &index.partitions[0].inverted_list;
        // The posting cache entry is grouped (issue #7040); the group holds
        // positions-free lists while positions live in their own per-token
        // entries.
        let (start, end) = inverted_list.group_range_for_token(0).unwrap();
        let group = inverted_list
            .index_cache
            .get_with_key(&PostingListGroupKey { start, end })
            .await
            .unwrap();
        assert!(
            !group.get(0).unwrap().has_position(),
            "posting cache should remain positions-free after prewarm"
        );

        let positions = inverted_list
            .index_cache
            .get_with_key(&PositionKey { token_id: 0 })
            .await
            .unwrap();
        assert!(
            matches!(
                positions.as_ref().0,
                CompressedPositionStorage::LegacyPerDoc(_)
            ),
            "positions should be stored in the dedicated position cache"
        );
    }

    #[tokio::test]
    async fn test_prewarm_with_v2_positions_preserves_shared_stream_codec() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let format_version = InvertedListFormatVersion::V2;
        let posting_tail_codec = format_version.posting_tail_codec();
        let mut builder = InnerBuilder::new_with_format_version(
            0,
            true,
            TokenSetFormat::default(),
            format_version,
        );
        builder.tokens.add("body".to_owned());

        let mut posting_list =
            PostingListBuilder::new_with_posting_tail_codec(true, posting_tail_codec);
        let expected = (0..(BLOCK_SIZE + 5) as u32)
            .map(|doc_id| {
                let positions = vec![doc_id % 3, doc_id % 3 + 2, doc_id % 3 + 5];
                posting_list.add(doc_id, PositionRecorder::Position(positions.clone().into()));
                builder.docs.append(30_000 + doc_id as u64, 20 + doc_id % 7);
                (doc_id, positions.len() as u32, positions)
            })
            .collect::<Vec<_>>();
        builder.posting_lists.push(posting_list);
        builder.write(store.as_ref()).await.unwrap();

        let metadata = HashMap::from([
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0_u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default().with_position(true)).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
            (
                POSTING_TAIL_CODEC_KEY.to_owned(),
                posting_tail_codec.as_str().to_owned(),
            ),
            (
                POSITIONS_LAYOUT_KEY.to_owned(),
                POSITIONS_LAYOUT_SHARED_STREAM_V2.to_owned(),
            ),
            (
                POSITIONS_CODEC_KEY.to_owned(),
                PositionStreamCodec::PackedDelta.as_str().to_owned(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store, None, cache.as_ref())
            .await
            .unwrap();
        index
            .prewarm_with_options(&FtsPrewarmOptions::new().with_position(true))
            .await
            .unwrap();

        let actual = index.partitions[0]
            .inverted_list
            .posting_list(0, true, &NoOpMetricsCollector)
            .await
            .unwrap()
            .iter()
            .map(|(doc_id, freq, positions)| {
                (doc_id as u32, freq, positions.unwrap().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_block_max_scores_capacity_matches_block_count() {
        let mut docs = DocSet::default();
        let num_docs = BLOCK_SIZE * 3 + 7;
        let doc_ids = (0..num_docs as u32).collect::<Vec<_>>();
        for doc_id in &doc_ids {
            docs.append(*doc_id as u64, 1);
        }

        let freqs = vec![1_u32; doc_ids.len()];
        let block_max_scores = docs.calculate_block_max_scores(doc_ids.iter(), freqs.iter());
        let expected_blocks = doc_ids.len().div_ceil(BLOCK_SIZE);

        assert_eq!(block_max_scores.len(), expected_blocks);
        assert_eq!(block_max_scores.capacity(), expected_blocks);
    }

    #[tokio::test]
    async fn test_bm25_search_uses_global_idf() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Partition 0: 3 docs, only one contains "alpha".
        let mut builder0 = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder0.tokens.add("alpha".to_owned());
        builder0.tokens.add("beta".to_owned());
        builder0.posting_lists.push(PostingListBuilder::new(false));
        builder0.posting_lists.push(PostingListBuilder::new(false));
        builder0.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder0.posting_lists[1].add(1, PositionRecorder::Count(1));
        builder0.posting_lists[1].add(2, PositionRecorder::Count(1));
        builder0.docs.append(100, 1);
        builder0.docs.append(101, 1);
        builder0.docs.append(102, 1);
        builder0.write(store.as_ref()).await.unwrap();

        // Partition 1: 1 doc, contains "alpha".
        let mut builder1 = InnerBuilder::new(1, false, TokenSetFormat::default());
        builder1.tokens.add("alpha".to_owned());
        builder1.posting_lists.push(PostingListBuilder::new(false));
        builder1.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder1.docs.append(200, 1);
        builder1.write(store.as_ref()).await.unwrap();

        let metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0u64, 1u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        let tokens = Arc::new(Tokens::new(vec!["alpha".to_string()], DocType::Text));
        let params = Arc::new(FtsSearchParams::new().with_limit(Some(10)));
        let prefilter = Arc::new(NoFilter);
        let metrics = Arc::new(NoOpMetricsCollector);

        let (row_ids, scores) = index
            .bm25_search(tokens, params, Operator::Or, prefilter, metrics, None)
            .await
            .unwrap();

        assert_eq!(row_ids.len(), 2);
        assert!(row_ids.contains(&100));
        assert!(row_ids.contains(&200));
        assert_eq!(row_ids.len(), scores.len());

        let expected_idf = idf(2, 4);
        for score in scores {
            assert!(
                (score - expected_idf).abs() < 1e-6,
                "score: {}, expected: {}",
                score,
                expected_idf
            );
        }
    }

    async fn write_test_metadata(
        store: &Arc<LanceIndexStore>,
        partition_ids: Vec<u64>,
        params: InvertedIndexParams,
    ) {
        let metadata = HashMap::from([
            (
                "partitions".to_owned(),
                serde_json::to_string(&partition_ids).unwrap(),
            ),
            ("params".to_owned(), serde_json::to_string(&params).unwrap()),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();
    }

    #[tokio::test]
    async fn test_v3_block_wand_prunes_without_payload_reads() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new_with_format_version(
            0,
            false,
            TokenSetFormat::default(),
            InvertedListFormatVersion::V3,
        );
        builder.tokens.add("alpha".to_owned());
        let mut posting = PostingListBuilder::new_with_posting_tail_codec(
            false,
            InvertedListFormatVersion::V3.posting_tail_codec(),
        );
        const NUM_BLOCKS: usize = 80;
        for doc_id in 0..(BLOCK_SIZE * NUM_BLOCKS) as u32 {
            let (freq, num_tokens) = if doc_id == 0 {
                (200, 1)
            } else if doc_id < BLOCK_SIZE as u32 {
                (10, 10)
            } else {
                (1, 1_000_000)
            };
            posting.add(doc_id, PositionRecorder::Count(freq));
            builder.docs.append(10_000 + doc_id as u64, num_tokens);
        }
        builder.posting_lists.push(posting);
        builder.write(store.as_ref()).await.unwrap();
        write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;

        let cache = Arc::new(LanceCache::no_cache());
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();
        let metrics = Arc::new(LocalMetricsCollector::default());
        let (row_ids, _scores) = index
            .bm25_search(
                Arc::new(Tokens::new(vec!["alpha".to_owned()], DocType::Text)),
                Arc::new(FtsSearchParams::new().with_limit(Some(1))),
                Operator::Or,
                Arc::new(NoFilter),
                metrics.clone(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(row_ids, vec![10_000]);
        assert_eq!(
            metrics
                .fts_block_metadata_rows
                .load(std::sync::atomic::Ordering::Relaxed),
            NUM_BLOCKS
        );
        assert_eq!(
            metrics
                .fts_payload_blocks_read
                .load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the low-score block should be pruned before payload read"
        );
        assert_eq!(
            metrics
                .fts_payload_read_ranges
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert!(
            metrics
                .fts_blocks_pruned
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 1
        );
        assert_eq!(
            metrics
                .fts_position_blocks_read
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[tokio::test]
    async fn test_v3_positions_are_lazy_until_phrase_candidate() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::with_capacity(1 << 20)),
        ));

        let mut builder = InnerBuilder::new_with_format_version(
            0,
            true,
            TokenSetFormat::default(),
            InvertedListFormatVersion::V3,
        );
        builder.tokens.add("hello".to_owned());
        builder.tokens.add("world".to_owned());
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                true,
                InvertedListFormatVersion::V3.posting_tail_codec(),
            ));
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                true,
                InvertedListFormatVersion::V3.posting_tail_codec(),
            ));
        builder.posting_lists[0].add(0, PositionRecorder::Position(vec![0].into()));
        builder.posting_lists[1].add(0, PositionRecorder::Position(vec![1].into()));
        builder.posting_lists[0].add(1, PositionRecorder::Position(vec![0].into()));
        builder.posting_lists[1].add(1, PositionRecorder::Position(vec![2].into()));
        builder.docs.append(100, 2);
        builder.docs.append(101, 2);
        builder.write(store.as_ref()).await.unwrap();
        write_test_metadata(
            &store,
            vec![0],
            InvertedIndexParams::default().with_position(true),
        )
        .await;

        let cache = Arc::new(LanceCache::with_capacity(1 << 20));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        let non_phrase_metrics = Arc::new(LocalMetricsCollector::default());
        let (mut non_phrase_row_ids, _) = index
            .bm25_search(
                Arc::new(Tokens::new(vec!["hello".to_owned()], DocType::Text)),
                Arc::new(FtsSearchParams::new().with_limit(Some(10))),
                Operator::Or,
                Arc::new(NoFilter),
                non_phrase_metrics.clone(),
                None,
            )
            .await
            .unwrap();
        non_phrase_row_ids.sort_unstable();
        assert_eq!(non_phrase_row_ids, vec![100, 101]);
        assert_eq!(
            non_phrase_metrics
                .fts_position_blocks_read
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "non-phrase queries must not read V3 positions"
        );

        let phrase_metrics = Arc::new(LocalMetricsCollector::default());
        let (row_ids, _) = index
            .bm25_search(
                Arc::new(Tokens::new(
                    vec!["hello".to_owned(), "world".to_owned()],
                    DocType::Text,
                )),
                Arc::new(
                    FtsSearchParams::new()
                        .with_limit(Some(10))
                        .with_phrase_slop(Some(0)),
                ),
                Operator::And,
                Arc::new(NoFilter),
                phrase_metrics.clone(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(row_ids, vec![100]);
        assert_eq!(
            phrase_metrics
                .fts_position_blocks_read
                .load(std::sync::atomic::Ordering::Relaxed),
            2,
            "phrase validation should read one positions block per matching term block"
        );
    }

    #[tokio::test]
    async fn test_v3_block_executor_matches_v2_and_v2_keeps_legacy_metrics() {
        async fn build_index(
            format_version: InvertedListFormatVersion,
        ) -> (Arc<InvertedIndex>, TempObjDir) {
            let tmpdir = TempObjDir::default();
            let store = Arc::new(LanceIndexStore::new(
                ObjectStore::local().into(),
                tmpdir.clone(),
                Arc::new(LanceCache::no_cache()),
            ));
            let mut builder = InnerBuilder::new_with_format_version(
                0,
                false,
                TokenSetFormat::default(),
                format_version,
            );
            for token in ["alpha", "beta", "gamma"] {
                builder.tokens.add(token.to_owned());
                builder
                    .posting_lists
                    .push(PostingListBuilder::new_with_posting_tail_codec(
                        false,
                        format_version.posting_tail_codec(),
                    ));
            }
            for doc_id in 0..6u32 {
                builder.docs.append(200 + doc_id as u64, 4 + doc_id);
            }
            for (doc_id, freq) in [(0, 2), (1, 1), (3, 4), (5, 1)] {
                builder.posting_lists[0].add(doc_id, PositionRecorder::Count(freq));
            }
            for (doc_id, freq) in [(1, 3), (2, 1), (3, 1)] {
                builder.posting_lists[1].add(doc_id, PositionRecorder::Count(freq));
            }
            builder.posting_lists[2].add(4, PositionRecorder::Count(1));
            builder.write(store.as_ref()).await.unwrap();
            write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
            let cache = Arc::new(LanceCache::no_cache());
            let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
                .await
                .unwrap();
            (index, tmpdir)
        }

        let (v2, _v2_tmpdir) = build_index(InvertedListFormatVersion::V2).await;
        let (v3, _v3_tmpdir) = build_index(InvertedListFormatVersion::V3).await;
        let tokens = Arc::new(Tokens::new(
            vec!["alpha".to_owned(), "beta".to_owned()],
            DocType::Text,
        ));
        let params = Arc::new(FtsSearchParams::new().with_limit(Some(10)));

        let v2_metrics = Arc::new(LocalMetricsCollector::default());
        let (mut v2_row_ids, _) = v2
            .bm25_search(
                tokens.clone(),
                params.clone(),
                Operator::Or,
                Arc::new(NoFilter),
                v2_metrics.clone(),
                None,
            )
            .await
            .unwrap();
        let v3_metrics = Arc::new(LocalMetricsCollector::default());
        let (mut v3_row_ids, _) = v3
            .bm25_search(
                tokens,
                params,
                Operator::Or,
                Arc::new(NoFilter),
                v3_metrics.clone(),
                None,
            )
            .await
            .unwrap();

        v2_row_ids.sort_unstable();
        v3_row_ids.sort_unstable();
        assert_eq!(v3_row_ids, v2_row_ids);
        assert_eq!(
            v2_metrics
                .fts_payload_blocks_read
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "V2 must stay on the legacy posting-list materialization path"
        );
        assert!(
            v3_metrics
                .fts_payload_blocks_read
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0,
            "V3 should use block payload reads"
        );
        assert!(
            v3_metrics
                .fts_block_metadata_rows
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0,
            "V3 should read block metadata before payload"
        );
    }

    #[tokio::test]
    async fn test_v3_top_block_cache_covers_large_multi_term_query() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new_with_format_version(
            0,
            false,
            TokenSetFormat::default(),
            InvertedListFormatVersion::V3,
        );
        builder.tokens.add("hot".to_owned());
        builder.tokens.add("topic".to_owned());
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                false,
                InvertedListFormatVersion::V3.posting_tail_codec(),
            ));
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                false,
                InvertedListFormatVersion::V3.posting_tail_codec(),
            ));

        const NUM_BLOCKS: usize = 80;
        for doc_id in 0..(BLOCK_SIZE * NUM_BLOCKS) as u32 {
            let (freq, num_tokens) = if doc_id == 0 {
                (200, 1)
            } else if doc_id < BLOCK_SIZE as u32 {
                (10, 10)
            } else {
                (1, 1_000_000)
            };
            builder.posting_lists[0].add(doc_id, PositionRecorder::Count(freq));
            if doc_id < BLOCK_SIZE as u32 {
                builder.posting_lists[1].add(doc_id, PositionRecorder::Count(1));
            }
            builder.docs.append(10_000 + doc_id as u64, num_tokens);
        }
        builder.write(store.as_ref()).await.unwrap();
        write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;

        let cache = Arc::new(LanceCache::with_capacity(1 << 20));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();
        let warmed = index.partitions[0]
            .inverted_list
            .prewarm_v3_top_blocks(&[0, 1], 10, false)
            .await
            .unwrap();
        assert!(warmed > 0);
        index.partitions[0].docs.ensure_loaded().await.unwrap();

        let metrics = Arc::new(LocalMetricsCollector::default());
        let (row_ids, _) = index
            .bm25_search(
                Arc::new(Tokens::new(
                    vec!["hot".to_owned(), "topic".to_owned()],
                    DocType::Text,
                )),
                Arc::new(FtsSearchParams::new().with_limit(Some(1))),
                Operator::Or,
                Arc::new(NoFilter),
                metrics.clone(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(row_ids, vec![10_000]);
        assert_eq!(
            metrics.parts_loaded.load(Ordering::Relaxed),
            0,
            "top-block cache should avoid legacy posting-list fallback"
        );
        assert_eq!(
            metrics.fts_block_metadata_rows.load(Ordering::Relaxed),
            0,
            "top-block prewarm should cache skip metadata"
        );
        assert_eq!(
            metrics.fts_payload_blocks_read.load(Ordering::Relaxed),
            0,
            "top-ranked payload blocks should be served from top-block cache"
        );
        assert!(
            metrics.fts_blocks_pruned.load(Ordering::Relaxed) > 0,
            "cached top blocks should establish a threshold that prunes cold blocks"
        );
    }

    #[tokio::test]
    async fn test_v3_top_block_cache_keeps_large_phrase_positions_lazy() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new_with_format_version(
            0,
            true,
            TokenSetFormat::default(),
            InvertedListFormatVersion::V3,
        );
        builder.tokens.add("hello".to_owned());
        builder.tokens.add("world".to_owned());
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                true,
                InvertedListFormatVersion::V3.posting_tail_codec(),
            ));
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                true,
                InvertedListFormatVersion::V3.posting_tail_codec(),
            ));

        const NUM_BLOCKS: usize = 80;
        for doc_id in 0..(BLOCK_SIZE * NUM_BLOCKS) as u32 {
            builder.posting_lists[0].add(doc_id, PositionRecorder::Position(vec![0].into()));
            builder.posting_lists[1].add(doc_id, PositionRecorder::Position(vec![1].into()));
            builder.docs.append(20_000 + doc_id as u64, 2);
        }
        builder.write(store.as_ref()).await.unwrap();
        write_test_metadata(
            &store,
            vec![0],
            InvertedIndexParams::default().with_position(true),
        )
        .await;

        let cache = Arc::new(LanceCache::with_capacity(1 << 20));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();
        let warmed = index.partitions[0]
            .inverted_list
            .prewarm_v3_top_blocks(&[0, 1], 10, false)
            .await
            .unwrap();
        assert!(warmed > 0);
        index.partitions[0].docs.ensure_loaded().await.unwrap();

        let metrics = Arc::new(LocalMetricsCollector::default());
        let (row_ids, _) = index
            .bm25_search(
                Arc::new(Tokens::new(
                    vec!["hello".to_owned(), "world".to_owned()],
                    DocType::Text,
                )),
                Arc::new(
                    FtsSearchParams::new()
                        .with_limit(Some(1))
                        .with_phrase_slop(Some(0)),
                ),
                Operator::And,
                Arc::new(NoFilter),
                metrics.clone(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(row_ids, vec![20_000]);
        assert_eq!(
            metrics.parts_loaded.load(Ordering::Relaxed),
            0,
            "top-block cache should keep large phrase queries on the block executor"
        );
        assert_eq!(
            metrics.fts_block_metadata_rows.load(Ordering::Relaxed),
            0,
            "phrase query should reuse prewarmed block metadata"
        );
        assert_eq!(
            metrics.fts_payload_blocks_read.load(Ordering::Relaxed),
            0,
            "top-cache phrase execution should not prefetch cold payload blocks"
        );
        assert_eq!(
            metrics.fts_position_blocks_read.load(Ordering::Relaxed),
            2,
            "positions should be read lazily only for the winning phrase candidate"
        );
    }

    #[tokio::test]
    async fn test_and_query_returns_empty_when_exact_term_missing() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.tokens.add("alpha".to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder.docs.append(100, 1);
        builder.write(store.as_ref()).await.unwrap();

        write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        let tokens = Arc::new(Tokens::new(
            vec!["alpha".to_owned(), "missing".to_owned()],
            DocType::Text,
        ));
        let params = Arc::new(FtsSearchParams::new().with_limit(Some(10)));
        let prefilter = Arc::new(NoFilter);
        let metrics = Arc::new(NoOpMetricsCollector);

        let (and_row_ids, _) = index
            .bm25_search(
                tokens.clone(),
                params.clone(),
                Operator::And,
                prefilter.clone(),
                metrics.clone(),
                None,
            )
            .await
            .unwrap();
        assert!(
            and_row_ids.is_empty(),
            "AND must not match when any required term is missing"
        );

        let (or_row_ids, _) = index
            .bm25_search(tokens, params, Operator::Or, prefilter, metrics, None)
            .await
            .unwrap();
        assert_eq!(
            or_row_ids,
            vec![100],
            "OR should still match the present term"
        );
    }

    // Enough distinct tokens that `write_posting_lists` emits several posting-list
    // batches (the default batch size is 256 rows), exercising the restructured
    // producer and async send path.
    const MANY_BATCH_TOKENS: u64 = 1000;
    const MANY_BATCH_ROW_ID_BASE: u64 = 1000;

    // Writes a single partition whose posting lists span many output batches. Each
    // token `tok{i:05}` maps to row id `MANY_BATCH_ROW_ID_BASE + i`.
    async fn write_partition_spanning_many_batches(store: &dyn IndexStore) {
        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        for i in 0..MANY_BATCH_TOKENS {
            // Zero-padded so tokens are inserted in sorted order, as the set expects.
            builder.tokens.add(format!("tok{i:05}"));
            let doc_id = builder.docs.append(MANY_BATCH_ROW_ID_BASE + i, 1);
            let mut posting_list = PostingListBuilder::new(false);
            posting_list.add(doc_id, PositionRecorder::Count(1));
            builder.posting_lists.push(posting_list);
        }
        builder
            .write(store)
            .await
            .expect("writing posting lists should succeed");
    }

    // Correctness guard for the restructured posting-list writer. The producer now
    // builds each batch in its own `spawn_cpu` call, handing the builder and the
    // remaining posting lists back out so state (the cross-batch cache-group
    // accumulator) is preserved, and dispatches with an async `send().await`. This
    // verifies that path over many batches by checking representative tokens after
    // they cross the producer/consumer boundary.
    //
    // Note: this does not reproduce the single-thread-pool deadlock the async send
    // fixes -- that requires a 1-thread CPU pool (a process-global singleton) plus
    // ~8MB of buffered posting data to trigger a consumer-side encoder flush, which
    // is impractical as a lightweight unit test.
    #[tokio::test]
    async fn test_write_many_posting_list_batches_preserves_all_batches() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        write_partition_spanning_many_batches(store.as_ref()).await;

        write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        // Probe tokens from the first, a middle, and the last batch to confirm they
        // remain queryable after crossing the producer/consumer boundary.
        for token_idx in [0u64, MANY_BATCH_TOKENS / 2, MANY_BATCH_TOKENS - 1] {
            let tokens = Arc::new(Tokens::new(
                vec![format!("tok{token_idx:05}")],
                DocType::Text,
            ));
            let params = Arc::new(FtsSearchParams::new().with_limit(Some(10)));
            let (row_ids, _) = index
                .bm25_search(
                    tokens,
                    params,
                    Operator::Or,
                    Arc::new(NoFilter),
                    Arc::new(NoOpMetricsCollector),
                    None,
                )
                .await
                .unwrap();
            assert_eq!(
                row_ids,
                vec![MANY_BATCH_ROW_ID_BASE + token_idx],
                "token tok{token_idx:05} should map to its single document"
            );
        }
    }

    #[tokio::test]
    async fn test_and_query_skips_partition_missing_required_term() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder0 = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder0.tokens.add("alpha".to_owned());
        builder0.posting_lists.push(PostingListBuilder::new(false));
        builder0.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder0.docs.append(100, 1);
        builder0.write(store.as_ref()).await.unwrap();

        let mut builder1 = InnerBuilder::new(1, false, TokenSetFormat::default());
        builder1.tokens.add("alpha".to_owned());
        builder1.tokens.add("beta".to_owned());
        builder1.posting_lists.push(PostingListBuilder::new(false));
        builder1.posting_lists.push(PostingListBuilder::new(false));
        builder1.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder1.posting_lists[1].add(0, PositionRecorder::Count(1));
        builder1.docs.append(200, 2);
        builder1.write(store.as_ref()).await.unwrap();

        write_test_metadata(&store, vec![0, 1], InvertedIndexParams::default()).await;
        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        let tokens = Arc::new(Tokens::new(
            vec!["alpha".to_owned(), "beta".to_owned()],
            DocType::Text,
        ));
        let params = Arc::new(FtsSearchParams::new().with_limit(Some(10)));
        let (mut row_ids, _) = index
            .bm25_search(
                tokens,
                params,
                Operator::And,
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
                None,
            )
            .await
            .unwrap();
        row_ids.sort_unstable();
        assert_eq!(
            row_ids,
            vec![200],
            "partition missing beta must not contribute alpha-only hits"
        );
    }

    #[tokio::test]
    async fn test_fuzzy_and_groups_expansions_by_original_position() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.tokens.add("alpha".to_owned());
        builder.tokens.add("alphi".to_owned());
        builder.tokens.add("beta".to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder.posting_lists[1].add(1, PositionRecorder::Count(1));
        builder.posting_lists[2].add(0, PositionRecorder::Count(1));
        builder.posting_lists[2].add(1, PositionRecorder::Count(1));
        builder.docs.append(100, 2);
        builder.docs.append(101, 2);
        builder.write(store.as_ref()).await.unwrap();

        write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();
        let params = Arc::new(
            FtsSearchParams::new()
                .with_limit(Some(10))
                .with_fuzziness(Some(1)),
        );

        let missing_position_tokens = Arc::new(Tokens::new(
            vec!["betx".to_owned(), "zzzzz".to_owned()],
            DocType::Text,
        ));
        let (missing_and_row_ids, _) = index
            .bm25_search(
                missing_position_tokens.clone(),
                params.clone(),
                Operator::And,
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
                None,
            )
            .await
            .unwrap();
        assert!(
            missing_and_row_ids.is_empty(),
            "fuzzy AND must require at least one expansion for every original position"
        );

        let (mut or_row_ids, _) = index
            .bm25_search(
                missing_position_tokens,
                params.clone(),
                Operator::Or,
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
                None,
            )
            .await
            .unwrap();
        or_row_ids.sort_unstable();
        assert_eq!(
            or_row_ids,
            vec![100, 101],
            "OR should still match present fuzzy expansions"
        );

        let grouped_tokens = Arc::new(Tokens::new(
            vec!["alphx".to_owned(), "betx".to_owned()],
            DocType::Text,
        ));
        let (mut grouped_row_ids, _) = index
            .bm25_search(
                grouped_tokens,
                params,
                Operator::And,
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
                None,
            )
            .await
            .unwrap();
        grouped_row_ids.sort_unstable();
        assert_eq!(
            grouped_row_ids,
            vec![100, 101],
            "each original fuzzy position should match any one of its expansions"
        );
    }

    #[tokio::test]
    async fn test_fuzzy_expansion_cap_applies_to_whole_query() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        for token in ["alpha", "alphi", "beta", "beti"] {
            builder.tokens.add(token.to_owned());
            builder.posting_lists.push(PostingListBuilder::new(false));
        }
        for token_id in 0..4 {
            builder.posting_lists[token_id].add(token_id as u32, PositionRecorder::Count(1));
            builder.docs.append(100 + token_id as u64, 1);
        }
        builder.write(store.as_ref()).await.unwrap();

        write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();
        let partition = index.partitions[0].clone();
        let params = FtsSearchParams::new()
            .with_fuzziness(Some(1))
            .with_max_expansions(3);
        let tokens = Tokens::new(vec!["alphx".to_owned(), "betx".to_owned()], DocType::Text);

        let expanded = partition.expand_fuzzy(&tokens, &params).unwrap();
        let expanded_terms = (0..expanded.len())
            .map(|idx| (expanded.get_token(idx).to_owned(), expanded.position(idx)))
            .collect::<Vec<_>>();

        assert_eq!(
            expanded_terms,
            vec![
                ("alpha".to_owned(), 0),
                ("alphi".to_owned(), 0),
                ("beta".to_owned(), 1),
            ],
            "max_expansions should cap the whole fuzzy query, not each token"
        );
    }

    /// Write one partition holding `variants` in order, with one
    /// single-token doc per variant taken from `row_ids`.
    async fn write_variant_partition(
        store: &Arc<LanceIndexStore>,
        partition_id: u64,
        variants: &[&str],
        row_ids: &[u64],
    ) {
        let mut builder = InnerBuilder::new(partition_id, false, TokenSetFormat::default());
        for token in variants {
            builder.tokens.add((*token).to_owned());
            builder.posting_lists.push(PostingListBuilder::new(false));
        }
        for (local_idx, row_id) in row_ids.iter().enumerate() {
            builder.posting_lists[local_idx].add(local_idx as u32, PositionRecorder::Count(1));
            builder.docs.append(*row_id, 1);
        }
        builder.write(store.as_ref()).await.unwrap();
    }

    #[tokio::test]
    async fn test_fuzzy_expansion_cap_is_global_across_partitions() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        write_variant_partition(&store, 0, &["alpha", "alphb"], &[100, 101]).await;
        write_variant_partition(&store, 1, &["alphc", "alphd"], &[102, 103]).await;
        write_test_metadata(&store, vec![0, 1], InvertedIndexParams::default()).await;
        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        let params = FtsSearchParams::new()
            .with_fuzziness(Some(1))
            .with_max_expansions(3);
        let tokens = Tokens::new(vec!["alphx".to_owned()], DocType::Text);

        let expanded = index.expand_fuzzy_tokens(&tokens, &params).unwrap();
        let expanded_terms = (0..expanded.len())
            .map(|idx| expanded.get_token(idx).to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            expanded_terms,
            vec!["alpha".to_owned(), "alphb".to_owned(), "alphc".to_owned()],
            "max_expansions must cap the whole query across partitions, \
             in lexicographic order"
        );
    }

    #[tokio::test]
    async fn test_fuzzy_results_independent_of_partition_shape() {
        // The same four single-variant docs, laid out as one partition and
        // as two. With a binding max_expansions the two shapes must still
        // match the same documents with the same scores.
        let single_dir = TempObjDir::default();
        let single_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            single_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        write_variant_partition(
            &single_store,
            0,
            &["alpha", "alphb", "alphc", "alphd"],
            &[100, 101, 102, 103],
        )
        .await;
        write_test_metadata(&single_store, vec![0], InvertedIndexParams::default()).await;

        let split_dir = TempObjDir::default();
        let split_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            split_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        write_variant_partition(&split_store, 0, &["alpha", "alphb"], &[100, 101]).await;
        write_variant_partition(&split_store, 1, &["alphc", "alphd"], &[102, 103]).await;
        write_test_metadata(&split_store, vec![0, 1], InvertedIndexParams::default()).await;

        let params = Arc::new(
            FtsSearchParams::new()
                .with_limit(Some(10))
                .with_fuzziness(Some(1))
                .with_max_expansions(3),
        );

        let mut results = Vec::new();
        for store in [single_store, split_store] {
            let cache = LanceCache::with_capacity(4096);
            let index = InvertedIndex::load(store, None, &cache).await.unwrap();
            let tokens = Arc::new(Tokens::new(vec!["alphx".to_owned()], DocType::Text));
            let (row_ids, scores) = index
                .bm25_search(
                    tokens,
                    params.clone(),
                    Operator::Or,
                    Arc::new(NoFilter),
                    Arc::new(NoOpMetricsCollector),
                    None,
                )
                .await
                .unwrap();
            let mut scored = row_ids.into_iter().zip(scores).collect::<Vec<_>>();
            scored.sort_unstable_by_key(|(row_id, _)| *row_id);
            results.push(scored);
        }

        assert_eq!(
            results[0]
                .iter()
                .map(|(row_id, _)| *row_id)
                .collect::<Vec<_>>(),
            vec![100, 101, 102],
            "a binding cap keeps the three lexicographically smallest variants"
        );
        assert_eq!(
            results[0], results[1],
            "fuzzy results must not depend on the partition shape"
        );
    }

    #[tokio::test]
    async fn test_fuzzy_and_scores_grouped_expansions_by_matched_token() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.tokens.add("alpha".to_owned());
        builder.tokens.add("alphi".to_owned());
        builder.tokens.add("beta".to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder.posting_lists[0].add(2, PositionRecorder::Count(1));
        builder.posting_lists[0].add(3, PositionRecorder::Count(1));
        builder.posting_lists[0].add(4, PositionRecorder::Count(1));
        builder.posting_lists[0].add(5, PositionRecorder::Count(1));
        builder.posting_lists[1].add(1, PositionRecorder::Count(1));
        builder.posting_lists[2].add(0, PositionRecorder::Count(1));
        builder.posting_lists[2].add(1, PositionRecorder::Count(1));
        builder.docs.append(100, 2);
        builder.docs.append(101, 2);
        builder.docs.append(102, 1);
        builder.docs.append(103, 1);
        builder.docs.append(104, 1);
        builder.docs.append(105, 1);
        builder.write(store.as_ref()).await.unwrap();

        write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        let tokens = Arc::new(Tokens::new(
            vec!["alphx".to_owned(), "betx".to_owned()],
            DocType::Text,
        ));
        let params = Arc::new(
            FtsSearchParams::new()
                .with_limit(Some(1))
                .with_fuzziness(Some(1)),
        );
        let (row_ids, _scores) = index
            .bm25_search(
                tokens,
                params,
                Operator::And,
                Arc::new(NoFilter),
                Arc::new(NoOpMetricsCollector),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            row_ids,
            vec![101],
            "the rare matched expansion should outrank the common expansion"
        );
    }

    #[tokio::test]
    async fn test_fuzzy_and_grouped_rescore_keeps_wand_limit_bounded() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let num_docs = BLOCK_SIZE * 2 + 4;
        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.tokens.add("alpha".to_owned());
        builder.tokens.add("alphi".to_owned());
        builder.tokens.add("beta".to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists.push(PostingListBuilder::new(false));

        builder.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder.posting_lists[1].add(1, PositionRecorder::Count(1));
        for doc_id in 0..num_docs {
            builder.posting_lists[2].add(doc_id as u32, PositionRecorder::Count(1));
            if doc_id >= 2 {
                builder.posting_lists[0].add(doc_id as u32, PositionRecorder::Count(1));
            }
            let num_tokens = if doc_id < 2 { 2 } else { 100 };
            builder.docs.append(100 + doc_id as u64, num_tokens);
        }
        builder.write(store.as_ref()).await.unwrap();

        write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        let tokens = Arc::new(Tokens::new(
            vec!["alphx".to_owned(), "betx".to_owned()],
            DocType::Text,
        ));
        let params = Arc::new(
            FtsSearchParams::new()
                .with_limit(Some(1))
                .with_fuzziness(Some(1)),
        );
        let metrics = Arc::new(LocalMetricsCollector::default());
        let (row_ids, _scores) = index
            .bm25_search(
                tokens,
                params,
                Operator::And,
                Arc::new(NoFilter),
                metrics.clone(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            row_ids,
            vec![101],
            "final rescoring should still rank by the matched expansion"
        );
        let comparisons = metrics.comparisons.load(Ordering::Relaxed);
        assert!(
            comparisons < num_docs,
            "grouped fuzzy AND should not clear the WAND top-k bound and scan every candidate; comparisons={comparisons}, num_docs={num_docs}"
        );
    }

    #[tokio::test]
    async fn test_phrase_query_reads_legacy_per_doc_positions() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new_with_format_version(
            0,
            true,
            TokenSetFormat::default(),
            InvertedListFormatVersion::V1,
        );
        builder.tokens.add("hello".to_owned());
        builder.tokens.add("world".to_owned());
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                true,
                PostingTailCodec::Fixed32,
            ));
        builder
            .posting_lists
            .push(PostingListBuilder::new_with_posting_tail_codec(
                true,
                PostingTailCodec::Fixed32,
            ));
        builder.posting_lists[0].add(0, PositionRecorder::Position(vec![0].into()));
        builder.posting_lists[1].add(0, PositionRecorder::Position(vec![1].into()));
        builder.posting_lists[0].add(1, PositionRecorder::Position(vec![0].into()));
        builder.posting_lists[1].add(1, PositionRecorder::Position(vec![2].into()));
        builder.docs.append(100, 2);
        builder.docs.append(101, 2);
        builder.write(store.as_ref()).await.unwrap();

        let metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0_u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default().with_position(true)).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let cache = Arc::new(LanceCache::with_capacity(4096));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        let tokens = Arc::new(Tokens::new(
            vec!["hello".to_owned(), "world".to_owned()],
            DocType::Text,
        ));
        let params = Arc::new(
            FtsSearchParams::new()
                .with_limit(Some(10))
                .with_phrase_slop(Some(0)),
        );
        let prefilter = Arc::new(NoFilter);
        let metrics = Arc::new(NoOpMetricsCollector);

        let (row_ids, _scores) = index
            .bm25_search(tokens, params, Operator::And, prefilter, metrics, None)
            .await
            .unwrap();

        assert_eq!(row_ids, vec![100]);
    }

    /// Build a multi-partition inverted index in `store` with `num_partitions`
    /// partitions, each carrying a handful of tokens/docs.
    async fn build_multi_partition_index(
        store: &Arc<LanceIndexStore>,
        num_partitions: u64,
    ) -> (Arc<InvertedIndex>, Arc<LanceCache>) {
        for id in 0..num_partitions {
            let mut builder = InnerBuilder::new_with_format_version(
                id,
                false,
                TokenSetFormat::default(),
                InvertedListFormatVersion::V1,
            );
            // A few distinct tokens per partition so each posting file has real
            // content to read and materialize during prewarm.
            for t in 0..4u32 {
                builder.tokens.add(format!("tok_{id}_{t}"));
                let mut posting = PostingListBuilder::new_with_posting_tail_codec(
                    false,
                    PostingTailCodec::Fixed32,
                );
                let base = id * 1000 + t as u64 * 10;
                for d in 0..5u32 {
                    posting.add(d, PositionRecorder::Count(1));
                    builder.docs.append(base + d as u64, 4);
                }
                builder.posting_lists.push(posting);
            }
            builder.write(store.as_ref()).await.unwrap();
        }

        let partition_ids: Vec<u64> = (0..num_partitions).collect();
        let metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&partition_ids).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        // Keep the cache alive and return it: the partition readers hold only a
        // WeakLanceCache, so the prewarmed entries vanish if this Arc is dropped.
        let cache = Arc::new(LanceCache::with_capacity(1 << 20));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();
        (index, cache)
    }

    /// The prewarm cost estimate must come from cheap object metadata (the
    /// posting file length) without reading the posting data, and must be
    /// monotonic in the partition's content.
    #[tokio::test]
    async fn test_posting_data_size_bytes_uses_file_length() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let (index, _cache) = build_multi_partition_index(&store, 3).await;
        for part in &index.partitions {
            // File length is reported by object metadata at open time; it must be
            // non-trivial for a partition that actually holds postings.
            let est = part.inverted_list.posting_data_size_bytes();
            assert!(
                est > 0,
                "expected a non-zero posting-data size estimate, got {est}"
            );
        }
    }

    /// Each partition must read through the shared scheduler at a distinct base
    /// priority. Tied priorities (every partition at 0) break the scheduler's
    /// backpressure deadlock-break — which admits the lowest-priority in-flight
    /// request — because there is no unique lowest request to advance, so a
    /// concurrent multi-partition read (e.g. prewarm) can wedge. Distinct
    /// per-partition priorities keep the in-flight set totally ordered.
    #[tokio::test]
    async fn test_partitions_load_with_distinct_priorities() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let (index, _cache) = build_multi_partition_index(&store, 5).await;

        let mut priorities: Vec<u64> = index
            .partitions
            .iter()
            .map(|part| {
                part.store
                    .as_any()
                    .downcast_ref::<LanceIndexStore>()
                    .expect("partition store should be a LanceIndexStore")
                    .io_priority()
            })
            .collect();

        // Distinct and dense (0..N): every partition reads at its own priority,
        // so the shared scheduler sees a total order across all partitions. The
        // partitions may finish loading in any order, so sort before comparing —
        // what matters is that the priorities form a contiguous, collision-free
        // set, not which partition ended up at which slot.
        priorities.sort_unstable();
        assert_eq!(
            priorities,
            (0..index.partitions.len() as u64).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_update_preserves_loaded_v2_format_version() -> Result<()> {
        let src_dir = TempObjDir::default();
        let dest_dir = TempObjDir::default();
        let src_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            src_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let dest_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            dest_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let format_version = InvertedListFormatVersion::V2;
        let posting_tail_codec = format_version.posting_tail_codec();
        let mut partition = InnerBuilder::new_with_format_version(
            0,
            false,
            TokenSetFormat::default(),
            format_version,
        );
        partition.tokens.add("hello".to_owned());
        let mut posting_list =
            PostingListBuilder::new_with_posting_tail_codec(false, posting_tail_codec);
        posting_list.add(0, PositionRecorder::Count(1));
        partition.posting_lists.push(posting_list);
        partition.docs.append(100, 1);
        partition.write(src_store.as_ref()).await?;

        let metadata = HashMap::from([
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0_u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
            (
                POSTING_TAIL_CODEC_KEY.to_owned(),
                posting_tail_codec.as_str().to_owned(),
            ),
        ]);
        let mut writer = src_store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let index = InvertedIndex::load(src_store, None, &LanceCache::no_cache()).await?;
        assert_eq!(index.index_version(), format_version.index_version());

        let schema = Arc::new(Schema::new(vec![
            Field::new("doc", DataType::Utf8, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let docs = Arc::new(StringArray::from(vec![Some("hello again")]));
        let row_ids = Arc::new(UInt64Array::from(vec![101u64]));
        let batch = RecordBatch::try_new(schema.clone(), vec![docs, row_ids])?;
        let stream = RecordBatchStreamAdapter::new(schema, stream::iter(vec![Ok(batch)]));
        let created = index
            .update(Box::pin(stream), dest_store.as_ref(), None)
            .await?;

        assert_eq!(created.index_version, format_version.index_version());

        let updated = InvertedIndex::load(dest_store, None, &LanceCache::no_cache()).await?;
        assert_eq!(updated.index_version(), format_version.index_version());
        assert_eq!(updated.partitions.len(), 2);
        for partition in &updated.partitions {
            assert_eq!(
                partition.inverted_list.posting_tail_codec(),
                posting_tail_codec
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_segments_preserves_arrow_token_set_format() -> Result<()> {
        let src_dir = TempObjDir::default();
        let dest_dir = TempObjDir::default();
        let src_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            src_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let dest_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            dest_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let index = write_single_partition_index(
            src_store,
            InvertedIndexParams::default(),
            TokenSetFormat::Arrow,
            "hello",
            100,
        )
        .await?;
        let created = InvertedIndex::merge_segments(
            &[index],
            empty_doc_stream(),
            dest_store.as_ref(),
            None,
            crate::progress::noop_progress(),
        )
        .await?;

        assert_eq!(created.index_version, 0);
        let merged = InvertedIndex::load(dest_store, None, &LanceCache::no_cache()).await?;
        assert_eq!(merged.token_set_format, TokenSetFormat::Arrow);

        let tokens = Arc::new(Tokens::new(vec!["hello".to_string()], DocType::Text));
        let params = Arc::new(FtsSearchParams::new().with_limit(Some(10)));
        let prefilter = Arc::new(NoFilter);
        let metrics = Arc::new(NoOpMetricsCollector);
        let (row_ids, _) = merged
            .bm25_search(tokens, params, Operator::Or, prefilter, metrics, None)
            .await?;
        assert_eq!(row_ids, vec![100]);

        Ok(())
    }

    #[tokio::test]
    async fn test_merge_segments_uses_memory_limit_for_old_partitions() -> Result<()> {
        let src_dir_1 = TempObjDir::default();
        let src_dir_2 = TempObjDir::default();
        let dest_dir = TempObjDir::default();
        let src_store_1 = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            src_dir_1.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let src_store_2 = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            src_dir_2.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let dest_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            dest_dir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let params = InvertedIndexParams::default().memory_limit_mb(0);
        let first = write_single_partition_index(
            src_store_1,
            params.clone(),
            TokenSetFormat::default(),
            "alpha",
            100,
        )
        .await?;
        let second = write_single_partition_index(
            src_store_2,
            params,
            TokenSetFormat::default(),
            "beta",
            200,
        )
        .await?;

        let mut builder =
            InvertedIndexBuilder::new(InvertedIndexParams::default().memory_limit_mb(0))
                .with_token_set_format(TokenSetFormat::default());
        builder
            .update_from_segments(
                empty_doc_stream(),
                dest_store.as_ref(),
                &[first, second],
                None,
            )
            .await?;

        let merged = InvertedIndex::load(dest_store, None, &LanceCache::no_cache()).await?;
        assert_eq!(merged.partitions.len(), 2);
        let mut partition_ids = merged
            .partitions
            .iter()
            .map(|partition| partition.id())
            .collect::<Vec<_>>();
        partition_ids.sort_unstable();
        assert_eq!(partition_ids, vec![0, 1]);

        Ok(())
    }

    #[tokio::test]
    async fn test_modern_index_without_deleted_col_has_empty_bitmap() {
        // An index created before the deleted_fragments feature was added
        // will have a metadata file with num_rows=0 (no record batch data).
        // The load path should gracefully handle this with an empty bitmap.
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.tokens.add("test".to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
        builder.posting_lists[0].add(0, PositionRecorder::Count(1));
        builder.docs.append(100, 1);
        builder.write(store.as_ref()).await.unwrap();

        // Write a metadata file WITHOUT the deleted_fragments column
        // (simulates an older index version)
        let metadata = std::collections::HashMap::from_iter(vec![
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let index = InvertedIndex::load(store, None, &LanceCache::no_cache())
            .await
            .unwrap();
        assert!(
            index.deleted_fragments().is_empty(),
            "index without deleted_fragments column should have empty bitmap"
        );
    }

    #[tokio::test]
    async fn flat_bm25_search_stream_with_metrics_records_elapsed_compute() {
        use crate::scalar::inverted::tokenizer::document_tokenizer::TextTokenizer;
        use arrow_array::{StringArray, UInt64Array};
        use lance_tokenizer::{SimpleTokenizer, TextAnalyzer};

        // Tiny stream of one batch containing the query term in two rows.
        let schema = Arc::new(Schema::new(vec![
            ROW_ID_FIELD.clone(),
            Field::new("text", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![0u64, 1, 2, 3])),
                Arc::new(StringArray::from(vec![
                    "the quick brown fox",
                    "lazy dog sleeps",
                    "the brown fox jumps over",
                    "completely unrelated text",
                ])),
            ],
        )
        .unwrap();

        let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![Ok(batch)]),
        ));

        let tokenizer: Box<dyn LanceTokenizer> = Box::new(TextTokenizer::new(
            TextAnalyzer::builder(SimpleTokenizer::default()).build(),
        ));

        let elapsed_compute = Time::default();
        let result_stream = flat_bm25_search_stream_with_metrics(
            input,
            "text".to_string(),
            "fox".to_string(),
            tokenizer,
            None,
            100,
            Some(elapsed_compute.clone()),
        )
        .await
        .unwrap();

        let batches: Vec<_> = result_stream.try_collect().await.unwrap();
        assert!(!batches.is_empty(), "expected at least one scored batch");

        // Both phase 1 (tokenize_and_count's spawn_cpu) and phase 2 (sync
        // scoring) call `add_duration` on the metric; verify the handle
        // was actually populated.
        assert!(
            elapsed_compute.value() > 0,
            "elapsed_compute should have been populated; got 0"
        );
    }

    #[tokio::test]
    async fn flat_bm25_search_uses_full_document_length_for_normalization() {
        let schema = Arc::new(Schema::new(vec![
            ROW_ID_FIELD.clone(),
            Field::new("text", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![0u64, 1])),
                Arc::new(StringArray::from(vec![
                    "alpha",
                    "alpha filler filler filler filler filler filler filler filler filler",
                ])),
            ],
        )
        .unwrap();

        let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![Ok(batch)]),
        ));
        let tokenizer: Box<dyn LanceTokenizer> = Box::new(TextTokenizer::new(
            TextAnalyzer::builder(SimpleTokenizer::default()).build(),
        ));

        let result_stream = flat_bm25_search_stream_with_metrics(
            input,
            "text".to_string(),
            "alpha".to_string(),
            tokenizer,
            None,
            100,
            None,
        )
        .await
        .unwrap();
        let batches: Vec<_> = result_stream.try_collect().await.unwrap();
        let scored = arrow::compute::concat_batches(&FTS_SCHEMA, &batches).unwrap();
        let row_ids = scored[ROW_ID].as_primitive::<UInt64Type>();
        let scores = scored[SCORE_COL].as_primitive::<Float32Type>();

        assert_eq!(row_ids.values(), &[0, 1]);
        assert!(
            scores.value(0) > scores.value(1),
            "same term frequency should score shorter document higher; short={}, long={}",
            scores.value(0),
            scores.value(1)
        );
    }

    #[tokio::test]
    async fn flat_bm25_search_treats_string_lists_as_row_documents() {
        let mut docs_builder =
            GenericListBuilder::<i32, _>::new(GenericStringBuilder::<i32>::new());
        docs_builder.values().append_value("alpha");
        docs_builder.values().append_value("alpha beta");
        docs_builder.append(true);
        docs_builder.values().append_value("beta");
        docs_builder.append(true);
        docs_builder.append(true);
        docs_builder.values().append_null();
        docs_builder.append(true);
        docs_builder.append(false);

        let docs = Arc::new(docs_builder.finish()) as ArrayRef;
        let schema = Arc::new(Schema::new(vec![
            ROW_ID_FIELD.clone(),
            Field::new("text", docs.data_type().clone(), true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![0u64, 1, 2, 3, 4])) as ArrayRef,
                docs,
            ],
        )
        .unwrap();

        let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![Ok(batch)]),
        ));
        let tokenizer: Box<dyn LanceTokenizer> = Box::new(TextTokenizer::new(
            TextAnalyzer::builder(SimpleTokenizer::default()).build(),
        ));

        let result_stream = flat_bm25_search_stream_with_metrics(
            input,
            "text".to_string(),
            "alpha".to_string(),
            tokenizer,
            None,
            100,
            None,
        )
        .await
        .unwrap();
        let batches: Vec<_> = result_stream.try_collect().await.unwrap();
        let scored = arrow::compute::concat_batches(&FTS_SCHEMA, &batches).unwrap();
        let row_ids = scored[ROW_ID].as_primitive::<UInt64Type>();

        assert_eq!(row_ids.values(), &[0]);
    }

    /// An [`IndexReader`] wrapper that hides the posting-group-offsets schema
    /// metadata key, so a [`PostingListReader`] opened on it takes the
    /// pre-grouping per-token fallback path (issue #7040).
    struct GroupKeyStrippingReader {
        inner: Arc<dyn IndexReader>,
        schema: lance_core::datatypes::Schema,
    }

    impl GroupKeyStrippingReader {
        fn new(inner: Arc<dyn IndexReader>) -> Self {
            let mut schema = inner.schema().clone();
            schema.metadata.remove(POSTING_GROUP_OFFSETS_BUF_KEY);
            Self { inner, schema }
        }
    }

    #[async_trait]
    impl IndexReader for GroupKeyStrippingReader {
        async fn read_record_batch(&self, n: u64, batch_size: u64) -> Result<RecordBatch> {
            self.inner.read_record_batch(n, batch_size).await
        }
        async fn read_global_buffer(&self, index: u32) -> Result<bytes::Bytes> {
            self.inner.read_global_buffer(index).await
        }
        async fn read_range(
            &self,
            range: std::ops::Range<usize>,
            projection: Option<&[&str]>,
        ) -> Result<RecordBatch> {
            self.inner.read_range(range, projection).await
        }
        async fn num_batches(&self, batch_size: u64) -> u32 {
            self.inner.num_batches(batch_size).await
        }
        fn num_rows(&self) -> usize {
            self.inner.num_rows()
        }
        fn schema(&self) -> &lance_core::datatypes::Schema {
            &self.schema
        }
    }

    fn posting_entries(posting: &PostingList) -> Vec<(u64, u32)> {
        posting.iter().map(|(doc, freq, _)| (doc, freq)).collect()
    }

    /// The grouped read path and the legacy per-token fallback must return
    /// identical posting lists for every token, including at group
    /// boundaries. Builds a single v2 partition that spans several groups,
    /// then reads it both with and without the group offsets present.
    #[tokio::test]
    async fn test_posting_list_fallback_matches_grouped() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // A small token cap forces several groups regardless of the default,
        // so the comparison exercises the partition_point math at group
        // boundaries.
        let num_tokens = 150u32;
        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.group_config = PostingGroupConfig {
            target_bytes: 4096,
            max_tokens: 32,
        };
        for t in 0..num_tokens {
            builder.tokens.add(format!("t{t}"));
            let mut pl = PostingListBuilder::new(false);
            pl.add(t, PositionRecorder::Count(1));
            builder.posting_lists.push(pl);
            builder.docs.append(1000 + t as u64, 1);
        }
        builder.write(store.as_ref()).await.unwrap();

        let reader = store.open_index_file(&posting_file_path(0)).await.unwrap();
        let cache = LanceCache::no_cache();
        let grouped = PostingListReader::try_new(reader.clone(), &cache)
            .await
            .unwrap();
        assert!(
            grouped.group_starts.as_ref().is_some_and(|s| s.len() > 1),
            "fixture should span multiple groups",
        );

        let stripped: Arc<dyn IndexReader> = Arc::new(GroupKeyStrippingReader::new(reader));
        let fallback = PostingListReader::try_new(stripped, &cache).await.unwrap();
        assert!(
            fallback.group_starts.is_none(),
            "stripped reader must take the per-token fallback path",
        );

        let metrics = NoOpMetricsCollector;
        for token in 0..num_tokens {
            let g = grouped.posting_list(token, false, &metrics).await.unwrap();
            let f = fallback.posting_list(token, false, &metrics).await.unwrap();
            assert_eq!(
                posting_entries(&g),
                posting_entries(&f),
                "grouped vs fallback mismatch for token {token}",
            );
            assert_eq!(g.len(), f.len(), "length mismatch for token {token}");
            assert_eq!(
                g.max_score(),
                f.max_score(),
                "max_score mismatch for token {token}",
            );
        }
    }

    /// Prewarm must populate exactly the `PostingListGroupKey`s the read path
    /// looks up — in particular the final group, whose `end` both paths derive
    /// from `self.len()`. If those derivations drifted (e.g. one used
    /// `num_rows()` and the other the loaded posting count), the last group's
    /// warm entry would be missing and prewarm silently wasted (issue #7040).
    #[tokio::test]
    async fn test_prewarm_group_keys_match_read_path() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // Small token cap so the partition spans several groups regardless of
        // the default, exercising every group boundary including the last.
        let num_tokens = 150u32;
        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.group_config = PostingGroupConfig {
            target_bytes: 4096,
            max_tokens: 32,
        };
        for t in 0..num_tokens {
            builder.tokens.add(format!("t{t}"));
            let mut pl = PostingListBuilder::new(false);
            pl.add(t, PositionRecorder::Count(1));
            builder.posting_lists.push(pl);
            builder.docs.append(1000 + t as u64, 1);
        }
        builder.write(store.as_ref()).await.unwrap();

        let reader = store.open_index_file(&posting_file_path(0)).await.unwrap();
        // A real (strong) cache must outlive the reader's weak handle so the
        // prewarmed entries are still resolvable below.
        let cache = LanceCache::with_capacity(1 << 20);
        let posting_reader = PostingListReader::try_new(reader, &cache).await.unwrap();
        assert!(
            posting_reader
                .group_starts
                .as_ref()
                .is_some_and(|s| s.len() > 1),
            "fixture should span multiple groups",
        );

        posting_reader
            .prewarm_posting_lists(false, 2)
            .await
            .unwrap();

        for token in 0..num_tokens {
            let (start, end) = posting_reader.group_range_for_token(token).unwrap();
            assert!(
                posting_reader
                    .index_cache
                    .get_with_key(&PostingListGroupKey { start, end })
                    .await
                    .is_some(),
                "prewarm did not populate group [{start}, {end}) that the read \
                 path requests for token {token}",
            );
        }

        let (_, last_end) = posting_reader
            .group_range_for_token(num_tokens - 1)
            .unwrap();
        assert_eq!(
            last_end, num_tokens,
            "the last group must end at the posting count ({num_tokens})",
        );
    }

    /// An empty partition writes no group-offsets buffer, so its reader takes
    /// the per-token fallback path (issue #7040).
    #[tokio::test]
    async fn test_empty_partition_has_no_group_offsets() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.write(store.as_ref()).await.unwrap();

        let reader = store.open_index_file(&posting_file_path(0)).await.unwrap();
        assert!(
            !reader
                .schema()
                .metadata
                .contains_key(POSTING_GROUP_OFFSETS_BUF_KEY),
            "empty partition must not write the group-offsets metadata key",
        );

        let posting_reader = PostingListReader::try_new(reader, &LanceCache::no_cache())
            .await
            .unwrap();
        assert!(
            posting_reader.group_starts.is_none(),
            "reader for an empty partition must use the per-token fallback path",
        );
        assert!(posting_reader.is_empty());
    }

    /// A posting list that alone exceeds the group target lands in its own
    /// `[t, t+1)` group (the clamp case) and reads back intact (issue #7040).
    #[tokio::test]
    async fn test_oversized_term_is_own_group_on_read() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // A tiny byte target so a modest posting trips the clamp without
        // needing a huge fixture; the surrounding tiny terms regroup after it.
        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.group_config = PostingGroupConfig {
            target_bytes: 50,
            max_tokens: 1000,
        };
        let big_docs = 30u32;
        builder.tokens.add("big".to_owned());
        let mut big = PostingListBuilder::new(false);
        for d in 0..big_docs {
            big.add(d, PositionRecorder::Count(1));
        }
        builder.posting_lists.push(big);
        for t in 1..5u32 {
            builder.tokens.add(format!("t{t}"));
            let mut pl = PostingListBuilder::new(false);
            pl.add(0, PositionRecorder::Count(1));
            builder.posting_lists.push(pl);
        }
        for d in 0..big_docs as u64 {
            builder.docs.append(1000 + d, 1);
        }
        builder.write(store.as_ref()).await.unwrap();

        let reader = store.open_index_file(&posting_file_path(0)).await.unwrap();
        let posting_reader = PostingListReader::try_new(reader, &LanceCache::no_cache())
            .await
            .unwrap();

        assert_eq!(
            posting_reader.group_range_for_token(0),
            Some((0, 1)),
            "an oversized term must occupy its own single-row group",
        );
        let big = posting_reader
            .posting_list(0, false, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(big.len(), big_docs as usize);
        // A trailing tiny term (in the next, multi-token group) still reads back.
        let tiny = posting_reader
            .posting_list(2, false, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(tiny.len(), 1);
    }

    /// When the group offsets are absent, prewarm populates per-token
    /// `PostingListKey` entries (the fallback path), matching what the read
    /// path then looks up (issue #7040).
    #[tokio::test]
    async fn test_prewarm_fallback_populates_per_token_entries() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        let num_tokens = 3u32;
        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        for t in 0..num_tokens {
            builder.tokens.add(format!("t{t}"));
            let mut pl = PostingListBuilder::new(false);
            pl.add(t, PositionRecorder::Count(1));
            builder.posting_lists.push(pl);
            builder.docs.append(1000 + t as u64, 1);
        }
        builder.write(store.as_ref()).await.unwrap();

        let reader = store.open_index_file(&posting_file_path(0)).await.unwrap();
        let stripped: Arc<dyn IndexReader> = Arc::new(GroupKeyStrippingReader::new(reader));
        let cache = LanceCache::with_capacity(1 << 20);
        let posting_reader = PostingListReader::try_new(stripped, &cache).await.unwrap();
        assert!(posting_reader.group_starts.is_none());

        posting_reader
            .prewarm_posting_lists(false, 2)
            .await
            .unwrap();

        for token_id in 0..num_tokens {
            assert!(
                posting_reader
                    .index_cache
                    .get_with_key(&PostingListKey { token_id })
                    .await
                    .is_some(),
                "fallback prewarm should populate per-token entry {token_id}",
            );
        }
    }

    /// End-to-end BM25 search over a grouped multi-group index must return the
    /// correct documents, and a warm-cache query must match the cold-cache
    /// result exactly (issue #7040).
    #[tokio::test]
    async fn test_grouped_bm25_search_correct_and_cache_stable() {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));

        // 130 rare tokens (one doc each) plus one common token in every doc; a
        // small token cap spreads them across several groups so scoring must
        // index into the right group slot.
        let num_rare = 130u32;
        let mut builder = InnerBuilder::new(0, false, TokenSetFormat::default());
        builder.group_config = PostingGroupConfig {
            target_bytes: 4096,
            max_tokens: 32,
        };
        for t in 0..num_rare {
            builder.tokens.add(format!("t{t}"));
            builder.posting_lists.push(PostingListBuilder::new(false));
        }
        let common_id = builder.tokens.add("common".to_owned());
        builder.posting_lists.push(PostingListBuilder::new(false));
        for d in 0..num_rare {
            builder.posting_lists[d as usize].add(d, PositionRecorder::Count(1));
            builder.posting_lists[common_id as usize].add(d, PositionRecorder::Count(1));
            builder.docs.append(1000 + d as u64, 2);
        }
        builder.write(store.as_ref()).await.unwrap();

        let metadata = HashMap::from([
            (
                "partitions".to_owned(),
                serde_json::to_string(&vec![0u64]).unwrap(),
            ),
            (
                "params".to_owned(),
                serde_json::to_string(&InvertedIndexParams::default()).unwrap(),
            ),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                TokenSetFormat::default().to_string(),
            ),
        ]);
        let mut writer = store
            .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
            .await
            .unwrap();
        writer.finish_with_metadata(metadata).await.unwrap();

        let cache = Arc::new(LanceCache::with_capacity(1 << 20));
        let index = InvertedIndex::load(store.clone(), None, cache.as_ref())
            .await
            .unwrap();

        // A rare token in the middle of a group must resolve to its one doc.
        let query = |term: &str| {
            let index = index.clone();
            let term = term.to_string();
            async move {
                index
                    .bm25_search(
                        Arc::new(Tokens::new(vec![term], DocType::Text)),
                        Arc::new(FtsSearchParams::new().with_limit(Some(200))),
                        Operator::Or,
                        Arc::new(NoFilter),
                        Arc::new(NoOpMetricsCollector),
                        None,
                    )
                    .await
                    .unwrap()
            }
        };

        let (rows_70, _) = query("t70").await;
        assert_eq!(rows_70, vec![1070], "rare token must map to its single doc");

        // Cold vs warm cache must agree for the common (large) token.
        let (cold_rows, cold_scores) = query("common").await;
        let (warm_rows, warm_scores) = query("common").await;
        assert_eq!(cold_rows.len(), num_rare as usize);
        assert_eq!(cold_rows, warm_rows, "warm-cache rows must match cold");
        assert_eq!(
            cold_scores, warm_scores,
            "warm-cache scores must match cold"
        );
    }

    #[tokio::test]
    async fn flat_bm25_search_stop_word_query_over_unindexed_rows_returns_empty() {
        let schema = Arc::new(Schema::new(vec![
            ROW_ID_FIELD.clone(),
            Field::new("text", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![0u64, 1, 2])),
                Arc::new(StringArray::from(vec![
                    "the quick brown fox",
                    "a lazy dog",
                    "for the win",
                ])),
            ],
        )
        .unwrap();

        let input: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![Ok(batch)]),
        ));

        // Analyzer with an English stop-word filter, so the query "the"
        // tokenizes to zero terms -- exactly the production trigger.
        let tokenizer: Box<dyn LanceTokenizer> = Box::new(TextTokenizer::new(
            TextAnalyzer::builder(SimpleTokenizer::default())
                .filter(StopWordFilter::new(Language::English).unwrap())
                .build(),
        ));

        let result_stream = flat_bm25_search_stream_with_metrics(
            input,
            "text".to_string(),
            "the".to_string(),
            tokenizer,
            None,
            100,
            None,
        )
        .await
        .unwrap();

        let batches: Vec<_> = result_stream.try_collect().await.unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total_rows, 0,
            "a stop-word-only query has no searchable terms and must match nothing"
        );
    }
}
