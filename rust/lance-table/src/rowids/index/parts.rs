// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cache ownership of prepared allocations, independent of snapshot routing.

use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::sync::{Arc, Weak};

use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};

use super::{FragmentEntry, FragmentIdentity, MergedChunk, MergedIndex, ProbeEntry, RowIdIndex};

type Chunks = Vec<(RangeInclusive<u64>, Arc<MergedChunk>)>;

#[derive(Clone, Debug)]
enum PartStorage {
    Probe(Arc<FragmentEntry>),
    Merged {
        sources: Arc<Vec<FragmentIdentity>>,
        chunks: Chunks,
    },
}

/// One independently owned prepared allocation group.
///
/// A cache can charge this value once while several weak snapshot views refer
/// to it. Probe fragments form separate parts. Merged chunks built together
/// form one part, including their shared source identity metadata.
#[derive(Clone, Debug)]
pub struct RowIdIndexPart(PartStorage);

impl RowIdIndexPart {
    /// In-process allocation identity: representation tag and allocation address.
    ///
    /// This is only an ownership/deduplication token while the part is alive.
    /// Never persist it or use it instead of verifying immutable source content.
    pub fn allocation_identity(&self) -> (bool, u64) {
        match &self.0 {
            PartStorage::Probe(fragment) => (false, Arc::as_ptr(fragment) as usize as u64),
            PartStorage::Merged { sources, .. } => (true, Arc::as_ptr(sources) as usize as u64),
        }
    }

    /// Physical fragment namespaces whose immutable sources identify this part.
    pub fn fragment_ids(&self) -> Vec<u32> {
        match &self.0 {
            PartStorage::Probe(fragment) => vec![fragment.fragment_id],
            PartStorage::Merged { sources, .. } => {
                sources.iter().map(|source| source.fragment_id).collect()
            }
        }
    }

    /// Preserve all chunks already owned for this allocation group.
    ///
    /// Another snapshot may use only some chunks of a shared group. Replacing
    /// its cache part with that subset would needlessly invalidate older views.
    /// Returns an error for different groups or inconsistent immutable chunks.
    pub fn try_union(&self, other: &Self) -> Result<Self> {
        if self.allocation_identity() != other.allocation_identity() {
            return Err(Error::invalid_input(
                "row id index parts have different allocation identities",
            ));
        }
        match (&self.0, &other.0) {
            (PartStorage::Probe(_), PartStorage::Probe(_)) => Ok(self.clone()),
            (
                PartStorage::Merged { sources, chunks },
                PartStorage::Merged { chunks: added, .. },
            ) => {
                let mut combined = chunks.clone();
                for (range, chunk) in added {
                    if let Some((existing_range, existing)) = combined
                        .iter()
                        .find(|(existing, _)| existing.start() == range.start())
                    {
                        if existing_range != range || !Arc::ptr_eq(existing, chunk) {
                            return Err(Error::internal(format!(
                                "prepared row id group changed immutable chunk {range:?}"
                            )));
                        }
                    } else {
                        combined.push((range.clone(), chunk.clone()));
                    }
                }
                combined.sort_unstable_by_key(|(range, _)| *range.start());
                if combined
                    .windows(2)
                    .any(|pair| pair[0].0.end() >= pair[1].0.start())
                {
                    return Err(Error::internal(
                        "prepared row id group contains overlapping immutable chunks",
                    ));
                }
                Ok(Self(PartStorage::Merged {
                    sources: sources.clone(),
                    chunks: combined,
                }))
            }
            _ => Err(Error::invalid_input(
                "row id index parts have different representations",
            )),
        }
    }
}

// Cache weights must include headers retained by Weak identities regardless
// of whether an external producer or an active graph still owns each source.
fn source_headers(sources: &[FragmentIdentity], context: &mut Context) -> usize {
    fn retained_header<T>(weak: &Weak<T>, context: &mut Context) -> usize {
        if context.mark_seen(weak.as_ptr() as usize) {
            std::mem::size_of::<T>()
        } else {
            0
        }
    }
    sources
        .iter()
        .map(|source| {
            retained_header(&source.sequence, context)
                + retained_header(&source.deletion_vector, context)
        })
        .sum()
}

impl DeepSizeOf for RowIdIndexPart {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match &self.0 {
            PartStorage::Probe(fragment) => fragment.deep_size_of_children(context),
            PartStorage::Merged { sources, chunks } => {
                let headers = source_headers(sources, context);
                sources.deep_size_of_children(context)
                    + headers
                    + chunks.capacity()
                        * std::mem::size_of::<(RangeInclusive<u64>, Arc<MergedChunk>)>()
                    + chunks
                        .iter()
                        .map(|(_, chunk)| chunk.deep_size_of_children(context))
                        .sum::<usize>()
            }
        }
    }
}

/// Snapshot routing that does not retain prepared graph allocations.
///
/// [`Self::upgrade`] returns an active index only while all required parts live.
/// Evicting cache-owned parts can invalidate a view without a hidden strong
/// ownership chain. The routing and source identity metadata have their own cost.
///
/// ```
/// # use std::sync::Arc;
/// # use lance_core::utils::deletion::DeletionVector;
/// # use lance_table::rowids::{FragmentRowIdIndex, RowIdIndex, RowIdSequence};
/// # fn example() -> lance_core::Result<()> {
/// let index = RowIdIndex::new(&[FragmentRowIdIndex {
///     fragment_id: 7,
///     row_id_sequence: Arc::new(RowIdSequence::from(0..3)),
///     deletion_vector: Arc::new(DeletionVector::default()),
/// }])?;
/// let parts = index.prepared_parts();
/// let view = index.downgrade();
/// drop(index);
/// assert!(view.upgrade().is_some());
/// drop(parts);
/// assert!(view.upgrade().is_none());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct RowIdIndexView {
    fragments: Vec<Weak<FragmentEntry>>,
    probes: Vec<ProbeEntry>,
    end_tree: Vec<u64>,
    merged: Option<Vec<(RangeInclusive<u64>, Weak<MergedChunk>)>>,
    sources: Vec<FragmentIdentity>,
}

impl RowIdIndexView {
    /// Resolve a batch while its required prepared allocations remain alive.
    ///
    /// `None` for the outer result means an allocation was evicted; `None` at
    /// an input position means that row ID is absent. Merged routing upgrades
    /// only the chunks covering the batch, without copying snapshot directories.
    pub fn get_many(&self, row_ids: &[u64]) -> Result<Option<Vec<Option<RowAddress>>>> {
        let Some(chunks) = &self.merged else {
            return self
                .upgrade()
                .map(|index| index.get_many(row_ids))
                .transpose();
        };
        let mut sorted: Vec<_> = row_ids.iter().copied().zip(0..row_ids.len()).collect();
        sorted.sort_unstable_by_key(|&(id, _)| id);
        let mut out = vec![None; row_ids.len()];
        let mut chunks = chunks.iter().peekable();
        let mut current: Option<Arc<MergedChunk>> = None;
        for (id, position) in sorted {
            while let Some((range, _)) = chunks.peek() {
                if *range.end() >= id {
                    break;
                }
                chunks.next();
                current = None;
            }
            let Some((range, weak)) = chunks.peek() else {
                break;
            };
            if id < *range.start() {
                continue;
            }
            if current.is_none() {
                current = weak.upgrade();
            }
            let Some(chunk) = &current else {
                return Ok(None);
            };
            if let Some(offset) = chunk.segments.0.position(id) {
                out[position] = chunk.segments.1.get(offset).map(RowAddress::from);
            }
        }
        Ok(Some(out))
    }

    /// Acquire prepared allocations for an active reader, or return `None` if
    /// eviction has removed any required part. Does not read or rebuild sources.
    pub fn upgrade(&self) -> Option<RowIdIndex> {
        let fragments = self
            .fragments
            .iter()
            .map(Weak::upgrade)
            .collect::<Option<Vec<_>>>()?;
        let merged = match &self.merged {
            Some(chunks) => Some(
                chunks
                    .iter()
                    .map(|(range, chunk)| Some((range.clone(), chunk.upgrade()?)))
                    .collect::<Option<MergedIndex>>()?,
            ),
            None => None,
        };
        Some(RowIdIndex {
            fragments,
            probes: self.probes.clone(),
            end_tree: self.end_tree.clone(),
            merged,
            sources: self.sources.clone(),
        })
    }
}

impl DeepSizeOf for RowIdIndexView {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        let headers = source_headers(&self.sources, context);
        // Weak allocations can retain their inline headers after graph eviction.
        // Include that small metadata cost without charging live graph children.
        self.fragments.capacity()
            * (std::mem::size_of::<Weak<FragmentEntry>>() + std::mem::size_of::<FragmentEntry>())
            + self.probes.capacity() * std::mem::size_of::<ProbeEntry>()
            + self.end_tree.deep_size_of_children(context)
            + self.sources.deep_size_of_children(context)
            + headers
            + self.merged.as_ref().map_or(0, |chunks| {
                chunks.capacity()
                    * (std::mem::size_of::<(RangeInclusive<u64>, Weak<MergedChunk>)>()
                        + std::mem::size_of::<MergedChunk>())
            })
    }
}

impl RowIdIndex {
    /// Share the independently chargeable prepared parts without copying graphs.
    /// See [`RowIdIndexView`] for cache ownership and reclamation.
    pub fn prepared_parts(&self) -> Vec<RowIdIndexPart> {
        let Some(merged) = &self.merged else {
            return self
                .fragments
                .iter()
                .map(|fragment| RowIdIndexPart(PartStorage::Probe(fragment.clone())))
                .collect();
        };
        let mut groups: HashMap<usize, (Arc<Vec<FragmentIdentity>>, Chunks)> = HashMap::new();
        for (range, chunk) in merged.iter() {
            groups
                .entry(Arc::as_ptr(&chunk.sources) as usize)
                .or_insert_with(|| (chunk.sources.clone(), Vec::new()))
                .1
                .push((range.clone(), chunk.clone()));
        }
        groups
            .into_values()
            .map(|(sources, chunks)| RowIdIndexPart(PartStorage::Merged { sources, chunks }))
            .collect()
    }

    /// Retain routing metadata and weak prepared references for this snapshot.
    /// See [`RowIdIndexView`] for the corresponding cache-owned parts.
    pub fn downgrade(&self) -> RowIdIndexView {
        RowIdIndexView {
            fragments: self.fragments.iter().map(Arc::downgrade).collect(),
            probes: self.probes.clone(),
            end_tree: self.end_tree.clone(),
            merged: self.merged.as_ref().map(|chunks| {
                chunks
                    .iter()
                    .map(|(range, chunk)| (range.clone(), Arc::downgrade(chunk)))
                    .collect()
            }),
            sources: self.sources.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rowids::{FragmentRowIdIndex, RowIdSequence};
    use lance_core::utils::{address::RowAddress, deletion::DeletionVector};
    use rstest::rstest;

    #[rstest]
    #[case::merged(false)]
    #[case::probe(true)]
    fn weak_views_release_graphs_and_preserve_physical_results(#[case] probe: bool) {
        let sequence = Arc::new(RowIdSequence::from(0..if probe { 4_000_000 } else { 7 }));
        let source = FragmentRowIdIndex {
            fragment_id: 7,
            row_id_sequence: sequence.clone(),
            deletion_vector: Arc::new(if probe {
                DeletionVector::Bitmap([1].into_iter().collect())
            } else {
                DeletionVector::default()
            }),
        };
        let index = RowIdIndex::new(&[source]).unwrap();
        assert_eq!(index.merged.is_none(), probe);
        let parts = index.prepared_parts();
        let view = index.downgrade();
        let routing_bytes = view.deep_size_of();
        let part_bytes = parts.deep_size_of();
        drop(sequence);
        assert_eq!(
            parts.deep_size_of(),
            part_bytes,
            "prepared part charge must not change after source producer release"
        );
        drop(index);
        let active = view.upgrade().unwrap();
        let expected = vec![
            Some(RowAddress::new_from_parts(7, 3)),
            None,
            Some(RowAddress::new_from_parts(7, 3)),
        ];
        assert_eq!(active.get_many(&[3, u64::MAX, 3]).unwrap(), expected);
        assert_eq!(view.get_many(&[3, u64::MAX, 3]).unwrap(), Some(expected));
        drop(active);
        drop(parts);
        assert_eq!(
            view.deep_size_of(),
            routing_bytes,
            "weak routing charge must not change after graph eviction"
        );
        assert!(view.upgrade().is_none());
        assert!(view.get_many(&[3]).unwrap().is_none());
    }

    #[test]
    fn source_group_union_keeps_chunks_needed_by_older_views() {
        let source = FragmentRowIdIndex {
            fragment_id: 7,
            row_id_sequence: Arc::new(RowIdSequence(vec![
                super::super::U64Segment::Range(0..2),
                super::super::U64Segment::Range(10..12),
                super::super::U64Segment::Range(20..22),
            ])),
            deletion_vector: Arc::new(DeletionVector::default()),
        };
        let index = RowIdIndex::new(&[source]).unwrap();
        let view = index.downgrade();
        let full = index.prepared_parts().pop().unwrap();
        let PartStorage::Merged { sources, chunks } = &full.0 else {
            panic!("small range fixture must use merged parts");
        };
        assert_eq!(chunks.len(), 3);
        let subset = RowIdIndexPart(PartStorage::Merged {
            sources: sources.clone(),
            chunks: chunks[..1].to_vec(),
        });
        let union = subset.try_union(&full).unwrap();
        drop(index);
        drop(full);
        drop(subset);
        assert_eq!(
            view.upgrade().unwrap().get(21).unwrap(),
            Some(RowAddress::new_from_parts(7, 5))
        );
        drop(union);
        assert!(view.upgrade().is_none());
    }
}
