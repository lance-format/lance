// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read and validate stable row IDs and per-row dataset versions.

mod spill;
mod take;
mod validate;

use std::sync::Arc;

use futures::{StreamExt, TryFutureExt, TryStreamExt};
use lance_core::cache::LanceCache;
use lance_core::utils::deletion::DeletionVector;
use lance_core::{Error, Result};
use lance_table::format::{
    Fragment, ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID,
    ROW_LAST_UPDATED_AT_VERSION_FIELD_ID, RowDatasetVersionMeta, RowDatasetVersionSequence,
    RowIdMeta,
};
use lance_table::rowids::version::{LoadedRowLineage, SpilledRowLineage};
use lance_table::rowids::{FragmentRowIdIndex, RowIdIndex, RowIdSequence, read_row_ids};

use crate::Table;
use crate::caches::{RowIdIndexKey, RowIdSequenceKey, RowVersionSequenceKey};
use crate::io::deletion::read_table_deletion_file;

#[doc(hidden)]
pub use spill::lineage_field;
pub use spill::{read_spilled_row_ids, read_spilled_versions};
pub(crate) use take::resolve_take_row_ids;
pub use validate::validate_stable_row_ids;

/// Load a row id sequence from the fragment's metadata and table files.
pub async fn load_row_id_sequence(
    table: &Table,
    fragment: &Fragment,
) -> Result<Arc<RowIdSequence>> {
    let Some(row_id_meta) = &fragment.row_id_meta else {
        return Err(Error::internal("Missing row id meta"));
    };
    let key = RowIdSequenceKey {
        fragment_id: fragment.id,
        row_id_meta,
        lineage_file: fragment.row_lineage_file(ROW_ID_FIELD_ID)?,
    };
    table
        .state
        .metadata_cache
        .get_or_insert_with_key(key, || read_row_id_sequence(table, fragment))
        .await
}

/// Decode the row id sequence of `fragment`, bypassing every cache.
async fn read_row_id_sequence(table: &Table, fragment: &Fragment) -> Result<RowIdSequence> {
    match &fragment.row_id_meta {
        None => Err(Error::internal("Missing row id meta")),
        Some(RowIdMeta::Inline(data)) => read_row_ids(data),
        Some(RowIdMeta::Column) => spill::read_spilled_row_ids(table, fragment).await,
    }
}

/// Which of a fragment's two per-row version sequences is meant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowVersionKind {
    /// The dataset version each row first appeared at.
    CreatedAt,
    /// The dataset version each row was last written at.
    LastUpdatedAt,
}

impl RowVersionKind {
    fn meta(self, fragment: &Fragment) -> Option<&RowDatasetVersionMeta> {
        match self {
            Self::CreatedAt => fragment.created_at_version_meta.as_ref(),
            Self::LastUpdatedAt => fragment.last_updated_at_version_meta.as_ref(),
        }
    }

    /// The reserved field id of the hidden column a spilled sequence lives in.
    pub fn field_id(self) -> i32 {
        match self {
            Self::CreatedAt => ROW_CREATED_AT_VERSION_FIELD_ID,
            Self::LastUpdatedAt => ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
        }
    }
}

/// Load one of `fragment`'s per-row version sequences, wherever it is stored.
///
/// Requires stable row ids on the table.
///
/// `None` when the fragment carries no such metadata, which readers treat as
/// every row being at version 1. A sequence spilled to a data file is cached
/// per fragment and file; an inline one is decoded from the manifest bytes.
pub async fn load_row_version_sequence(
    table: &Table,
    fragment: &Fragment,
    kind: RowVersionKind,
) -> Result<Option<Arc<RowDatasetVersionSequence>>> {
    if !table.manifest.uses_stable_row_ids() {
        return Err(Error::invalid_input(format!(
            "row version tracking requires stable row ids, version {} does not use them",
            table.manifest.version
        )));
    }
    let Some(meta) = kind.meta(fragment) else {
        return Ok(None);
    };
    match meta {
        RowDatasetVersionMeta::Column => {
            let data_file = fragment.row_lineage_file(kind.field_id())?.ok_or_else(|| {
                Error::corrupt_file(
                    table.state.base.clone(),
                    format!(
                        "fragment {} marks its {kind:?} versions as spilled but none of its \
                         data files carries field {}",
                        fragment.id,
                        kind.field_id()
                    ),
                )
            })?;
            let key = RowVersionSequenceKey {
                fragment_id: fragment.id,
                field_id: kind.field_id(),
                data_file,
            };
            table
                .state
                .metadata_cache
                .get_or_insert_with_key(key, || {
                    spill::read_spilled_versions(table, fragment, kind.field_id())
                })
                .await
                .map(Some)
        }
        RowDatasetVersionMeta::Inline(_) => meta
            .load_sequence()
            .map(|sequence| Some(Arc::new(sequence))),
    }
}

/// Read ahead every lineage sequence of `fragments` that lives outside the
/// manifest, for a commit that will need to consult them.
///
/// Building a manifest is synchronous and cannot read a data file, so the
/// commit path calls this first, over the whole manifest, and hands the result
/// over in `ManifestBuildConfig::spilled_row_lineage`. Only spilled sequences
/// are loaded; when none is, this returns an empty map without IO.
#[doc(hidden)]
pub async fn load_spilled_row_lineage<'a>(
    table: &Table,
    fragments: impl IntoIterator<Item = &'a Fragment>,
) -> Result<Arc<SpilledRowLineage>> {
    // A `for` loop rather than `map`: a closure returning a future that borrows
    // its argument trips the higher-ranked lifetime check on the outer future.
    let mut loads = Vec::new();
    for fragment in fragments {
        if fragment.has_spilled_row_lineage() {
            loads.push(load_fragment_spilled_lineage(table, fragment));
        }
    }
    let loaded: SpilledRowLineage = futures::stream::iter(loads)
        .buffer_unordered(table.state.object_store.io_parallelism())
        .try_collect()
        .await?;
    Ok(Arc::new(loaded))
}

/// The spilled sequences of one fragment, for [`load_spilled_row_lineage`].
async fn load_fragment_spilled_lineage(
    table: &Table,
    fragment: &Fragment,
) -> Result<(u64, LoadedRowLineage)> {
    let row_ids = match &fragment.row_id_meta {
        Some(RowIdMeta::Column) => Some(load_row_id_sequence(table, fragment).await?),
        _ => None,
    };
    let mut versions = [None, None];
    for (slot, kind) in versions
        .iter_mut()
        .zip([RowVersionKind::CreatedAt, RowVersionKind::LastUpdatedAt])
    {
        if let Some(RowDatasetVersionMeta::Column) = kind.meta(fragment) {
            *slot = load_row_version_sequence(table, fragment, kind).await?;
        }
    }
    let [created_at, last_updated_at] = versions;
    Ok((
        fragment.id,
        LoadedRowLineage {
            row_ids,
            created_at,
            last_updated_at,
        },
    ))
}

/// Load the stable row id index for this snapshot, if it uses stable row ids.
pub async fn get_row_id_index(table: &Table) -> Result<Option<Arc<RowIdIndex>>> {
    let cache = table
        .state
        .session
        .index_cache()
        .with_key_prefix(table.uri());
    get_row_id_index_with_cache(table, true, &cache).await
}

/// The row id index, built with or without the deletion vectors.
///
/// Without them the index maps a deleted row id to its address instead of
/// `None`, but reads no deletion file: on a table with many fragments that is
/// one request per fragment saved for a caller that already knows its ids are
/// live.
#[doc(hidden)]
pub async fn get_row_id_index_with_cache(
    table: &Table,
    with_deletions: bool,
    cache: &LanceCache,
) -> Result<Option<Arc<RowIdIndex>>> {
    if !table.manifest.uses_stable_row_ids() {
        return Ok(None);
    }
    // The cache is shared by every dataset opened at this URI, and one dropped
    // and recreated there restarts at version 1. Without a token for this
    // manifest generation a cached index could be the old generation's, so
    // build a private one instead of sharing.
    let Some(e_tag) = table.state.manifest_location.e_tag.as_deref() else {
        return Ok(Some(Arc::new(
            load_row_id_index(table, with_deletions).await?,
        )));
    };
    let key = RowIdIndexKey {
        version: table.manifest.version,
        with_deletions,
        e_tag: Some(e_tag),
    };
    let index = cache
        .get_or_insert_with_key(key, || load_row_id_index(table, with_deletions))
        .await?;
    Ok(Some(index))
}

/// Build the index from freshly decoded sequences. The index then owns their
/// memory alone, so its cache charge is exact and it stays cached whenever it
/// fits; reading them through the sequence cache instead would charge each
/// sequence twice and leave a large table's index too heavy to keep. A scan
/// that needs a sequence still caches its own copy under `RowIdSequenceKey`,
/// keyed by the fragment's content; the index is keyed by manifest generation
/// and cannot stand in for it.
async fn load_row_id_index(table: &Table, with_deletions: bool) -> Result<RowIdIndex> {
    load_row_id_index_for_fragments(table, &table.manifest.fragments, with_deletions).await
}

/// Build a row id index covering only `fragments`, not the whole table.
///
/// A caller that already knows the row ids it will look up were all read from
/// a known, small set of fragments -- compaction rewriting a handful of
/// fragments out of a much larger table, for instance -- gets no benefit from
/// [`get_row_id_index`]'s whole-dataset index and pays for reading every other
/// fragment's row id sequence and deletion vector for nothing. This builds an
/// index scoped to just those fragments instead.
async fn load_row_id_index_for_fragments(
    table: &Table,
    fragments: &[Fragment],
    with_deletions: bool,
) -> Result<RowIdIndex> {
    // A `for` loop rather than `map`: a closure returning a future that borrows
    // its argument trips the higher-ranked lifetime check on the outer future.
    let mut loads = Vec::with_capacity(fragments.len());
    for fragment in fragments.iter() {
        loads.push(read_row_id_sequence(table, fragment).map_ok(Arc::new));
    }
    let sequences: Vec<Arc<RowIdSequence>> = futures::stream::iter(loads)
        .buffered(table.state.object_store.io_parallelism())
        .try_collect()
        .await?;

    // Without deletions the index must still tell the copies of a row apart:
    // an update keeps the old copy under a deletion vector and writes the id
    // again elsewhere. Only fragments whose id spans overlap can hold such
    // copies, so only they need their deletion vector.
    let needs_deletions = if with_deletions {
        vec![true; fragments.len()]
    } else {
        overlapping_id_spans(&sequences)
    };

    let mut loads = Vec::with_capacity(fragments.len());
    for ((fragment, row_id_sequence), needs_deletions) in
        fragments.iter().zip(sequences).zip(needs_deletions)
    {
        loads.push(async move {
            let deletion_vector = match &fragment.deletion_file {
                Some(deletion_file) if needs_deletions => {
                    read_table_deletion_file(table, fragment.id, deletion_file).await?
                }
                _ => Arc::new(DeletionVector::default()),
            };
            Ok::<_, Error>(FragmentRowIdIndex {
                fragment_id: fragment.id as u32,
                row_id_sequence,
                deletion_vector,
            })
        });
    }
    let fragment_indices: Vec<FragmentRowIdIndex> = futures::stream::iter(loads)
        .buffer_unordered(table.state.object_store.io_parallelism())
        .try_collect()
        .await?;
    RowIdIndex::new(&fragment_indices)
}

/// For each sequence, whether its span of ids (lowest to highest) intersects
/// another sequence's span. An empty sequence overlaps nothing.
fn overlapping_id_spans(sequences: &[Arc<RowIdSequence>]) -> Vec<bool> {
    let mut spans: Vec<(u64, u64, usize)> = sequences
        .iter()
        .enumerate()
        .filter_map(|(index, sequence)| {
            let span = sequence.row_id_range()?;
            Some((*span.start(), *span.end(), index))
        })
        .collect();
    spans.sort_unstable();
    let mut overlapping = vec![false; sequences.len()];
    let mut reach: Option<(u64, usize)> = None;
    for &(start, end, index) in &spans {
        if let Some((reach_end, reach_index)) = reach
            && start <= reach_end
        {
            overlapping[index] = true;
            overlapping[reach_index] = true;
        }
        if reach.is_none_or(|(reach_end, _)| end > reach_end) {
            reach = Some((end, index));
        }
    }
    overlapping
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Containment, a chain of pairwise overlaps, a disjoint span and an empty
    /// sequence; the sweep must mark exactly the spans that touch another.
    #[test]
    fn test_overlapping_id_spans() {
        let seq = |ids: Vec<u64>| Arc::new(RowIdSequence::try_from_iter(ids).unwrap());
        let contained = vec![
            seq((0..=100).collect()),
            seq(vec![1, 2]),
            seq(vec![50, 60]),
            seq(vec![200, 201]),
            seq(vec![]),
        ];
        assert_eq!(
            overlapping_id_spans(&contained),
            vec![true, true, true, false, false]
        );
        // The last span overlaps only the middle one, not the one that set the
        // reach first.
        let chain = vec![seq(vec![0, 10]), seq(vec![20, 30]), seq(vec![5, 25])];
        assert_eq!(overlapping_id_spans(&chain), vec![true, true, true]);
        let disjoint = vec![seq(vec![0, 10]), seq(vec![11, 20]), seq(vec![21])];
        assert_eq!(overlapping_id_spans(&disjoint), vec![false, false, false]);
    }
}
