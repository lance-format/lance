// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use lance_core::cache::{CacheCodecImpl, CacheEntryReader, CacheEntryWriter};

pub(super) const DEFAULT_POSTING_GROUP_BYTES: usize = 32 * 1024;
const UNPLANNED_GROUP_TOKENS: u32 = 128;
const GROUP_BYTES_ENV: &str = "LANCE_FTS_POSTING_GROUP_MAX_BYTES";
const GROUP_TOKENS_ENV: &str = "LANCE_FTS_POSTING_GROUP_MAX_TOKENS";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PostingGroupPolicy {
    Bytes(usize),
    Tokens(u32),
}

impl PostingGroupPolicy {
    pub(super) fn parse(bytes: Option<&str>, tokens: Option<&str>) -> Result<Self> {
        if bytes.is_some() && tokens.is_some() {
            return Err(Error::invalid_input(format!(
                "{GROUP_BYTES_ENV} and {GROUP_TOKENS_ENV} are mutually exclusive"
            )));
        }
        let positive = |name: &str, value: &str| {
            value
                .parse::<usize>()
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "{name} must be a positive integer, got {value:?}"
                    ))
                })
        };
        if let Some(tokens) = tokens {
            let count = positive(GROUP_TOKENS_ENV, tokens)?;
            let count = u32::try_from(count).map_err(|_| {
                Error::invalid_input(format!("{GROUP_TOKENS_ENV}={count} exceeds {}", u32::MAX))
            })?;
            Ok(Self::Tokens(count))
        } else {
            Ok(Self::Bytes(
                bytes
                    .map(|value| positive(GROUP_BYTES_ENV, value))
                    .transpose()?
                    .unwrap_or(DEFAULT_POSTING_GROUP_BYTES),
            ))
        }
    }
}

static GROUP_POLICY: LazyLock<std::result::Result<PostingGroupPolicy, String>> =
    LazyLock::new(|| {
        let read = |name| {
            std::env::var(name).map(Some).or_else(|error| match error {
                std::env::VarError::NotPresent => Ok(None),
                error => Err(format!("invalid {name}: {error}")),
            })
        };
        let bytes = read(GROUP_BYTES_ENV)?;
        let tokens = read(GROUP_TOKENS_ENV)?;
        PostingGroupPolicy::parse(bytes.as_deref(), tokens.as_deref())
            .map_err(|error| error.to_string())
    });

/// Runtime cache ranges. Byte boundaries are learned during prewarm and stored
/// in the cache, never in the source index. Without that directory, retain the
/// existing bounded read-ahead policy instead of scanning the entire index or
/// multiplying cold fuzzy-query requests into singleton reads.
#[derive(Debug, Clone)]
pub(super) enum PostingGrouping {
    None,
    SyntheticFixed {
        group_size: u32,
    },
    Bytes {
        max_bytes: usize,
        boundaries: Arc<OnceLock<Arc<PostingGroupBoundaries>>>,
    },
}

impl DeepSizeOf for PostingGrouping {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        match self {
            Self::Bytes { boundaries, .. }
                if context.mark_seen(Arc::as_ptr(boundaries) as usize) =>
            {
                std::mem::size_of_val(boundaries.as_ref())
                    + boundaries
                        .get()
                        .map(|value| value.deep_size_of_children(context))
                        .unwrap_or(0)
            }
            _ => 0,
        }
    }
}

impl PostingGrouping {
    pub(super) async fn for_reader(
        is_legacy: bool,
        positions: PositionsLayout,
        token_count: usize,
        cache: &LanceCache,
    ) -> Result<Self> {
        if is_legacy || token_count == 0 {
            return Ok(Self::None);
        }
        u32::try_from(token_count).map_err(|_| {
            Error::index(format!(
                "posting token count {token_count} exceeds the u32 token ID range"
            ))
        })?;
        let policy = GROUP_POLICY
            .as_ref()
            .map_err(|error| Error::invalid_input(error.clone()))?;
        match *policy {
            PostingGroupPolicy::Tokens(group_size) => Ok(Self::SyntheticFixed { group_size }),
            // The old per-document position representation retains its existing
            // materialized prewarm path. Current writers use shared streams.
            PostingGroupPolicy::Bytes(_) if positions == PositionsLayout::LegacyPerDoc => {
                Ok(Self::SyntheticFixed { group_size: 128 })
            }
            PostingGroupPolicy::Bytes(max_bytes) => {
                let grouping = Self::byte_bounded(max_bytes);
                grouping.restore(token_count, cache).await;
                Ok(grouping)
            }
        }
    }

    pub(super) fn byte_bounded(max_bytes: usize) -> Self {
        Self::Bytes {
            max_bytes,
            boundaries: Arc::new(OnceLock::new()),
        }
    }

    pub(super) async fn restore(&self, token_count: usize, cache: &LanceCache) {
        if let Self::Bytes { max_bytes, .. } = self {
            let key = PostingGroupBoundariesKey {
                max_bytes: *max_bytes,
                token_count,
            };
            if let Some(bounds) = cache.get_with_key(&key).await
                && bounds.ends.last().copied().map(|end| end as usize) == Some(token_count)
            {
                self.publish(bounds);
            }
        }
    }

    pub(super) fn publish(&self, value: Arc<PostingGroupBoundaries>) {
        if let Self::Bytes { boundaries, .. } = self {
            // Concurrent prewarms derive the same ranges independently of read
            // chunk size, positions, or concurrency. Readers share this cell.
            let _ = boundaries.set(value);
        }
    }

    pub(super) fn is_grouped(&self) -> bool {
        !matches!(self, Self::None)
    }

    pub(super) fn range_for_token(&self, token_id: u32, token_count: usize) -> Option<(u32, u32)> {
        if token_id as usize >= token_count {
            return None;
        }
        match self {
            Self::None => None,
            Self::SyntheticFixed { group_size } => {
                let start = (token_id / *group_size) * *group_size;
                Some((
                    start,
                    start.saturating_add(*group_size).min(token_count as u32),
                ))
            }
            Self::Bytes { boundaries, .. } => Some(match boundaries.get() {
                Some(bounds) => {
                    let slot = bounds.ends.partition_point(|end| *end <= token_id);
                    (
                        slot.checked_sub(1)
                            .map(|slot| bounds.ends[slot])
                            .unwrap_or(0),
                        bounds.ends[slot],
                    )
                }
                None => {
                    let start = token_id / UNPLANNED_GROUP_TOKENS * UNPLANNED_GROUP_TOKENS;
                    (
                        start,
                        start
                            .saturating_add(UNPLANNED_GROUP_TOKENS)
                            .min(token_count as u32),
                    )
                }
            }),
        }
    }

    pub(super) fn aligned_chunk_end(
        &self,
        token_count: usize,
        tok_start: usize,
        desired_end: usize,
    ) -> usize {
        match self {
            Self::SyntheticFixed { group_size } => synthetic_group_aligned_chunk_end(
                *group_size as usize,
                token_count,
                tok_start,
                desired_end,
            ),
            // Byte prewarm carries the unfinished group across read chunks.
            _ => desired_end,
        }
    }

    pub(super) fn ranges_for_chunk(
        &self,
        tok_start: usize,
        tok_end: usize,
        token_count: usize,
    ) -> Vec<(u32, u32)> {
        match self {
            Self::None => Vec::new(),
            Self::SyntheticFixed { group_size } => synthetic_group_ranges_for_chunk(
                *group_size as usize,
                tok_start,
                tok_end,
                token_count,
            ),
            Self::Bytes { .. } => {
                let mut ranges = Vec::new();
                let mut start = tok_start as u32;
                while start < tok_end as u32 {
                    let Some(range) = self.range_for_token(start, token_count) else {
                        break;
                    };
                    ranges.push(range);
                    start = range.1;
                }
                ranges
            }
        }
    }
}

#[derive(Debug, DeepSizeOf)]
pub(super) struct PostingGroupBoundaries {
    pub(super) ends: Vec<u32>,
}

impl CacheCodecImpl for PostingGroupBoundaries {
    const TYPE_ID: &'static str = "lance.fts.PostingGroupBoundaries";
    const CURRENT_VERSION: u32 = 1;

    fn serialize(&self, writer: &mut CacheEntryWriter<'_>) -> Result<()> {
        let bytes: Vec<u8> = self.ends.iter().flat_map(|end| end.to_le_bytes()).collect();
        writer.write_raw(&bytes)
    }

    fn deserialize(reader: &mut CacheEntryReader<'_>) -> Result<Self> {
        if reader.version() != Self::CURRENT_VERSION {
            return Err(Error::io(format!(
                "unsupported posting group boundaries version {}",
                reader.version()
            )));
        }
        let bytes = reader.read_raw()?;
        if bytes.is_empty() || bytes.len() % 4 != 0 {
            return Err(Error::io(format!(
                "invalid posting group boundaries byte length {}",
                bytes.len()
            )));
        }
        let ends: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| u32::from_le_bytes(*value))
            .collect();
        if ends[0] == 0 || ends.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::io(
                "posting group boundaries must be strictly increasing and positive",
            ));
        }
        Ok(Self { ends })
    }
}

#[derive(Debug, Clone)]
pub(super) struct PostingGroupBoundariesKey {
    pub(super) max_bytes: usize,
    pub(super) token_count: usize,
}

impl CacheKey for PostingGroupBoundariesKey {
    type ValueType = PostingGroupBoundaries;
    fn type_name() -> &'static str {
        "PostingGroupBoundaries"
    }
    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!(
            "posting-group-boundaries-{}-{}",
            self.max_bytes, self.token_count
        )
        .into()
    }
    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.inverted.posting-group-boundaries", 1)
    }
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.max_bytes as u64);
        builder.write_u64(self.token_count as u64);
    }
    fn codec() -> Option<CacheCodec> {
        Some(CacheCodec::from_impl::<PostingGroupBoundaries>())
    }
}

/// Logical row-buffer weights used by the byte grouping policy. Account for
/// both list and binary offsets and the score/length scalars, even when the
/// latter are supplied by the reader. Positions have independent cache keys.
pub(super) fn posting_row_bytes(batch: &RecordBatch) -> Result<Vec<usize>> {
    let mut sizes = vec![8usize; batch.num_rows()];
    for name in [POSTING_COL, IMPACT_COL] {
        let Some(column) = batch.column_by_name(name) else {
            if name == POSTING_COL {
                return Err(Error::index(
                    "posting group batch is missing the posting column",
                ));
            }
            continue;
        };
        let lists = column.as_list_opt::<i32>().ok_or_else(|| {
            Error::index(format!(
                "posting group column {name} must be List<LargeBinary>"
            ))
        })?;
        let values = lists.values().as_binary_opt::<i64>().ok_or_else(|| {
            Error::index(format!(
                "posting group column {name} must contain LargeBinary values"
            ))
        })?;
        for (row, size) in sizes.iter_mut().enumerate() {
            let start = lists.value_offsets()[row] as usize;
            let end = lists.value_offsets()[row + 1] as usize;
            let payload = (values.value_offsets()[end] - values.value_offsets()[start]) as usize;
            *size = size
                .checked_add(payload)
                .and_then(|size| size.checked_add(8 * (end - start) + 4))
                .ok_or_else(|| {
                    Error::index(format!("posting row {row} buffer size overflows usize"))
                })?;
        }
    }
    Ok(sizes)
}

pub(super) type PostingGroupBatch = (u32, u32, RecordBatch);

/// Carries at most one incomplete group between reads, so grouping is greedy
/// over the partition rather than dependent on the prewarm read chunk size.
pub(super) struct ByteGroupPlanner {
    max_bytes: usize,
    next_token: u32,
    pending: Option<PostingGroupBatch>,
    pending_bytes: usize,
}

impl ByteGroupPlanner {
    pub(super) fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            next_token: 0,
            pending: None,
            pending_bytes: 0,
        }
    }

    pub(super) fn push(
        &mut self,
        batch: RecordBatch,
        last: bool,
    ) -> Result<Vec<PostingGroupBatch>> {
        let sizes = posting_row_bytes(&batch)?;
        let count = u32::try_from(batch.num_rows())
            .map_err(|_| Error::index("posting chunk row count exceeds u32"))?;
        let chunk_start = self.next_token;
        self.next_token = chunk_start
            .checked_add(count)
            .ok_or_else(|| Error::index("posting token count exceeds u32"))?;
        let mut groups = Vec::new();
        let mut local_start = 0;
        let mut bytes = self.pending_bytes;
        for (row, size) in sizes.into_iter().enumerate() {
            if (self.pending.is_some() || row > local_start)
                && size > self.max_bytes.saturating_sub(bytes)
            {
                groups.push(self.finish_group(&batch, chunk_start, local_start, row)?);
                local_start = row;
                bytes = 0;
            }
            bytes = bytes
                .checked_add(size)
                .ok_or_else(|| Error::index("posting group buffer size overflows usize"))?;
        }
        if batch.num_rows() > local_start || self.pending.is_some() {
            let (start, end, tail) =
                self.finish_group(&batch, chunk_start, local_start, batch.num_rows())?;
            if last || bytes >= self.max_bytes {
                // Oversized postings stay whole and never retain the following row.
                groups.push((start, end, tail));
                self.pending_bytes = 0;
            } else {
                self.pending = Some((start, end, tail.shrink_to_fit()?));
                self.pending_bytes = bytes;
            }
        }
        Ok(groups)
    }

    fn finish_group(
        &mut self,
        batch: &RecordBatch,
        chunk_start: u32,
        start: usize,
        end: usize,
    ) -> Result<PostingGroupBatch> {
        let tail = batch.slice(start, end - start);
        let end_token = chunk_start + end as u32;
        match self.pending.take() {
            Some((first, _, head)) if tail.num_rows() == 0 => Ok((first, end_token, head)),
            Some((first, _, head)) => {
                let combined =
                    arrow_select::concat::concat_batches(&batch.schema(), [&head, &tail])?;
                Ok((first, end_token, combined))
            }
            None => Ok((chunk_start + start as u32, end_token, tail)),
        }
    }
}
