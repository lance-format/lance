// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Ordered bulk materialization at any height. Keep one pending leaf across
//! routing boundaries so split/coalesce is decided before writing final leaves.

use std::collections::BTreeMap;

use bytes::Bytes;
use lance_core::Result;

use super::{IngestResult, WriteAcc};
use crate::format::{Fragment, pb};
use crate::fragment_metadata::node::{self, FragmentTreeConfig};
use crate::fragment_metadata::store::{LeafUploads, NodeStore, Uploaded};

fn with_leaves(mut output: WriteAcc, leaves: Uploaded) -> IngestResult {
    output.io_bytes = leaves.io_bytes;
    (leaves.children, output)
}

enum PendingLeaf {
    Unchanged {
        child: pb::FragmentTreeChild,
        end: u64,
    },
    Changed {
        fragments: Vec<Fragment>,
        encoded: Option<Bytes>,
    },
}

impl PendingLeaf {
    async fn bytes(&mut self, store: &NodeStore) -> Result<u64> {
        Ok(match self {
            Self::Unchanged { child, .. } => child.object_size,
            Self::Changed { fragments, encoded } => match encoded {
                Some(bytes) => bytes.len() as u64,
                None => {
                    let bytes = store.encode_leaf(fragments).await?;
                    let size = bytes.len() as u64;
                    *encoded = Some(bytes);
                    size
                }
            },
        })
    }

    async fn load(self, store: &NodeStore) -> Result<Vec<Fragment>> {
        match self {
            Self::Unchanged { child, end } => {
                Ok(store.read_leaf_shared(&child, end).await?.into_vec())
            }
            Self::Changed { fragments, .. } => Ok(fragments),
        }
    }

    async fn write(
        self,
        uploads: &mut LeafUploads<'_>,
        config: &FragmentTreeConfig,
        watermark: u64,
        output: &mut WriteAcc,
    ) -> Result<()> {
        match self {
            Self::Unchanged { child, .. } => uploads.reuse(child),
            Self::Changed { fragments, encoded } => {
                let leaves = uploads
                    .upload_leaves(&fragments, encoded, watermark, config)
                    .await?;
                output.splits += leaves.saturating_sub(1) as u64;
            }
        }
        Ok(())
    }

    /// Join `next` onto this leaf when either is small and the pair fits.
    /// Otherwise write this leaf. Returns the leaf left pending.
    async fn merge_or_write(
        mut self,
        mut next: Self,
        store: &NodeStore,
        uploads: &mut LeafUploads<'_>,
        config: &FragmentTreeConfig,
        watermark: u64,
        output: &mut WriteAcc,
    ) -> Result<Self> {
        let bytes = self.bytes(store).await?;
        let next_bytes = next.bytes(store).await?;
        let small = bytes <= config.leaf_merge_floor() || next_bytes <= config.leaf_merge_floor();
        let fits = bytes
            .checked_add(next_bytes)
            .is_some_and(|total| total <= config.leaf_coalesce_ceiling());
        if small && fits {
            let mut fragments = self.load(store).await?;
            fragments.extend(next.load(store).await?);
            output.merges += 1;
            return Ok(Self::Changed {
                fragments,
                encoded: None,
            });
        }
        self.write(uploads, config, watermark, output).await?;
        Ok(next)
    }
}

/// Visit old routing in order, carrying each ancestor's actions to its owning
/// leaf. Defer the last output leaf across parent boundaries so a later sparse
/// sibling cannot force a read/rewrite of an already emitted replacement.
/// Unchanged leaves are reused unless coalescing needs them. The caller builds
/// final routing over the returned leaf references, without intermediate nodes.
pub(super) async fn materialize(
    store: &NodeStore,
    config: &FragmentTreeConfig,
    children: Vec<pb::FragmentTreeChild>,
    buffer: Vec<pb::FragmentTreeMutation>,
    watermark: u64,
    next_fragment_id: u64,
) -> Result<IngestResult> {
    let mut output = WriteAcc::default();
    let mut uploads = LeafUploads::new(store);
    let mut pending: Option<PendingLeaf> = None;
    if children.is_empty() {
        let mut fragments = BTreeMap::new();
        output.materialized = buffer.len() as u64;
        store.apply_verified(&mut fragments, buffer)?;
        if !fragments.is_empty() {
            PendingLeaf::Changed {
                fragments: fragments.into_values().collect(),
                encoded: None,
            }
            .write(&mut uploads, config, watermark, &mut output)
            .await?;
        }
        return Ok(with_leaves(output, uploads.finish().await?));
    }
    let mut sequences = node::DecodedSequences::new(&buffer);
    let buckets = node::partition_buffer_by_child(&children, buffer);
    node::validate_routed(&children, &buckets, next_fragment_id)?;
    let mut stack: Vec<_> = node::with_exclusive_ends(children, next_fragment_id)
        .zip(buckets)
        .map(|((child, end), actions)| (child, actions, end, 0))
        .rev()
        .collect();
    while let Some((child, actions, end, depth)) = stack.pop() {
        if child.height > 0 {
            let mut interior = store.read_interior_in_range(&child, end).await?;
            if interior.children.is_empty() {
                return Err(super::super::validation::corrupt(format!(
                    "Interior {} has no children",
                    child.path
                )));
            }
            sequences.admit(&child.path, &interior.buffer)?;
            interior.buffer.extend(actions);
            let buckets = node::partition_buffer_by_child(&interior.children, interior.buffer);
            node::validate_routed(&interior.children, &buckets, end)?;
            stack.extend(
                node::with_exclusive_ends(interior.children, end)
                    .zip(buckets)
                    .map(|((child, end), actions)| (child, actions, end, depth + 1))
                    .rev(),
            );
            continue;
        }
        let current = if actions.is_empty() {
            PendingLeaf::Unchanged { child, end }
        } else {
            output.flushes += 1;
            output.max_flush_depth = output.max_flush_depth.max(depth);
            output.materialized += actions.len() as u64;
            let mut fragments: BTreeMap<_, _> = store
                .read_leaf_shared(&child, end)
                .await?
                .into_vec()
                .into_iter()
                .map(|f| (f.id, f))
                .collect();
            store.apply_verified(&mut fragments, actions)?;
            if fragments.is_empty() {
                continue;
            }
            PendingLeaf::Changed {
                fragments: fragments.into_values().collect(),
                encoded: None,
            }
        };
        pending = Some(match pending.take() {
            Some(previous) => {
                previous
                    .merge_or_write(current, store, &mut uploads, config, watermark, &mut output)
                    .await?
            }
            None => current,
        });
    }
    if let Some(leaf) = pending {
        leaf.write(&mut uploads, config, watermark, &mut output)
            .await?;
    }
    Ok(with_leaves(output, uploads.finish().await?))
}
