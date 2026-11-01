// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Leaf encoding and upload. Leaves are encoded on the caller's task while
//! their PUTs run behind the store's window.

use std::ops::Range;
use std::sync::Arc;

use futures::TryStreamExt;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{Error, Result};
use lance_file::version::ConcreteFileVersion;
use lance_file::writer::FileWriterOptions;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use object_store::{ObjectStore as OSObjectStore, PutOptions};
use tokio::sync::Semaphore;
use tokio::task::{JoinError, JoinSet};

use super::{NodeStore, Written, get_whole};
use crate::format::{Fragment, pb};
use crate::fragment_metadata::node;

struct EncodedLeaf {
    range: Range<usize>,
    bytes: bytes::Bytes,
}

/// Leaf references in key order, each backed by a durable object.
pub(in crate::fragment_metadata) struct Uploaded {
    pub(in crate::fragment_metadata) children: Vec<pb::FragmentTreeChild>,
    pub(in crate::fragment_metadata) io_bytes: u64,
}

impl Uploaded {
    /// Valid only when no child was reused, so each object size is the bytes
    /// this upload wrote.
    fn into_written(self) -> Vec<Written> {
        self.children
            .into_iter()
            .map(|child_ref| Written {
                io_bytes: child_ref.object_size,
                child_ref,
            })
            .collect()
    }
}

/// Leaf PUTs that run while the caller encodes the next leaf. A reference
/// depends only on the fragments and the encoded size, so it is built before
/// its PUT, but it leaves only through [`Self::finish`], once every PUT is
/// durable, so no parent can name a missing object. Dropping the queue aborts
/// PUTs still running. Any that landed are unreachable garbage.
pub(in crate::fragment_metadata) struct LeafUploads<'a> {
    store: &'a NodeStore,
    puts: JoinSet<Result<()>>,
    children: Vec<pb::FragmentTreeChild>,
    io_bytes: u64,
}

impl<'a> LeafUploads<'a> {
    pub(in crate::fragment_metadata) fn new(store: &'a NodeStore) -> Self {
        Self {
            store,
            puts: JoinSet::new(),
            children: Vec::new(),
            io_bytes: 0,
        }
    }

    /// Start the PUT of one encoded leaf once the store's window has room.
    pub(in crate::fragment_metadata) async fn upload(
        &mut self,
        fragments: &[Fragment],
        bytes: bytes::Bytes,
        materialized_through_action_sequence: u64,
    ) -> Result<()> {
        let permit = self
            .store
            .leaf_puts
            .clone()
            .acquire_owned()
            .await
            .map_err(|error| {
                Error::internal(format!("fragment tree leaf PUT window closed: {error}"))
            })?;
        // Stop producing at the first failed PUT instead of at finish.
        while let Some(put) = self.puts.try_join_next() {
            put_outcome(put)?;
        }
        let path = self.store.leaf_path();
        let size = bytes.len() as u64;
        let location = self.store.resolve_path(path.as_ref())?;
        self.children.push(node::leaf_ref(
            path.to_string(),
            fragments,
            size,
            materialized_through_action_sequence,
        )?);
        self.io_bytes += size;
        let inner = self.store.object_store.inner.clone();
        // A spawned task keeps the PUT moving while the caller encodes the
        // next leaf, which an unpolled future would not.
        self.puts.spawn(async move {
            let _permit = permit;
            inner
                .put_opts(&location, bytes.into(), PutOptions::default())
                .await?;
            Ok(())
        });
        Ok(())
    }

    /// Encode `fragments` as one or more leaves, split only when actual Lance
    /// bytes exceed the leaf limit, and upload each. Returns the leaf count.
    pub(in crate::fragment_metadata) async fn upload_leaves(
        &mut self,
        fragments: &[Fragment],
        bytes: Option<bytes::Bytes>,
        watermark: u64,
        config: &node::FragmentTreeConfig,
    ) -> Result<usize> {
        let store = self.store;
        let encoded = store.encoded_leaves(fragments, 0..fragments.len(), bytes, config);
        futures::pin_mut!(encoded);
        let mut leaves = 0;
        while let Some(leaf) = encoded.try_next().await? {
            self.upload(&fragments[leaf.range], leaf.bytes, watermark)
                .await?;
            leaves += 1;
        }
        Ok(leaves)
    }

    /// Keep an existing leaf in its key position.
    pub(in crate::fragment_metadata) fn reuse(&mut self, child: pb::FragmentTreeChild) {
        self.children.push(child);
    }

    pub(in crate::fragment_metadata) async fn finish(mut self) -> Result<Uploaded> {
        while let Some(put) = self.puts.join_next().await {
            put_outcome(put)?;
        }
        Ok(Uploaded {
            children: self.children,
            io_bytes: self.io_bytes,
        })
    }
}

fn put_outcome(put: std::result::Result<Result<()>, JoinError>) -> Result<()> {
    match put {
        Ok(outcome) => outcome,
        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
        Err(error) => Err(Error::internal(format!(
            "fragment tree leaf PUT task did not complete: {error}"
        ))),
    }
}

pub(super) fn leaf_put_window(object_store: &ObjectStore) -> Arc<Semaphore> {
    Arc::new(Semaphore::new(
        object_store
            .io_parallelism()
            .clamp(1, super::super::tree::MAX_CONCURRENT_LEAF_DRAINS),
    ))
}

impl NodeStore {
    /// Write sorted fragments as a columnar Lance leaf, one row per fragment.
    #[cfg(test)]
    pub(in crate::fragment_metadata) async fn write_leaf(
        &self,
        fragments: &[Fragment],
        materialized_through_action_sequence: u64,
    ) -> Result<Written> {
        let bytes = self.encode_leaf(fragments).await?;
        let mut uploads = LeafUploads::new(self);
        uploads
            .upload(fragments, bytes, materialized_through_action_sequence)
            .await?;
        Ok(uploads
            .finish()
            .await?
            .into_written()
            .pop()
            .expect("one upload yields one leaf"))
    }

    /// Encode before publishing, splitting only when actual Lance bytes exceed
    /// the leaf limit. A single fragment may exceed it, but never the hard
    /// object limit. Rejected encodings create no remote intermediate objects.
    pub(in crate::fragment_metadata) async fn write_leaves(
        &self,
        fragments: &[Fragment],
        bytes: Option<bytes::Bytes>,
        watermark: u64,
        config: &node::FragmentTreeConfig,
    ) -> Result<Vec<Written>> {
        let mut uploads = LeafUploads::new(self);
        uploads
            .upload_leaves(fragments, bytes, watermark, config)
            .await?;
        Ok(uploads.finish().await?.into_written())
    }

    /// Pack adjacent initial batches when their combined encoding fits. Keep
    /// at most three leaf limits of logical metadata in a candidate, except for an
    /// indivisible fragment. Normal mutation splits retain their own headroom.
    pub(in crate::fragment_metadata) async fn write_initial_leaves(
        &self,
        fragments: &[Fragment],
        config: &node::FragmentTreeConfig,
    ) -> Result<Vec<Written>> {
        let mut pending: Option<EncodedLeaf> = None;
        let mut uploads = LeafUploads::new(self);
        let mut start = 0;
        while start < fragments.len() {
            let mut end = start;
            let mut logical_bytes = 0;
            while end < fragments.len() && logical_bytes < config.max_leaf_bytes {
                logical_bytes += node::fragment_logical_bytes(&fragments[end]);
                end += 1;
            }
            let encoded = self.encoded_leaves(fragments, start..end, None, config);
            futures::pin_mut!(encoded);
            while let Some(leaf) = encoded.try_next().await? {
                if let Some(previous) = pending.take() {
                    let encoded_bytes = previous.bytes.len().checked_add(leaf.bytes.len());
                    let range = previous.range.start..leaf.range.end;
                    if encoded_bytes.is_some_and(|bytes| bytes as u64 <= config.max_leaf_bytes)
                        && node::leaf_logical_bytes(&fragments[range.clone()])
                            <= config.max_leaf_bytes * 3
                    {
                        let bytes = self.encode_leaf(&fragments[range.clone()]).await?;
                        // Compression is not additive. The estimate only chooses
                        // a candidate; its actual encoding decides admission.
                        if bytes.len() as u64 <= config.max_leaf_bytes {
                            pending = Some(EncodedLeaf { range, bytes });
                            continue;
                        }
                    }
                    uploads
                        .upload(&fragments[previous.range], previous.bytes, 0)
                        .await?;
                }
                pending = Some(leaf);
            }
            start = end;
        }
        if let Some(leaf) = pending {
            uploads
                .upload(&fragments[leaf.range], leaf.bytes, 0)
                .await?;
        }
        Ok(uploads.finish().await?.into_written())
    }

    pub(in crate::fragment_metadata) async fn encode_leaf(
        &self,
        fragments: &[Fragment],
    ) -> Result<bytes::Bytes> {
        #[cfg(test)]
        self.encoded_records
            .fetch_add(fragments.len(), std::sync::atomic::Ordering::Relaxed);
        let batch = super::super::leaf::encode(fragments)?;
        let arrow_schema = batch.schema();

        let lance_schema = LanceSchema::try_from(arrow_schema.as_ref())?;
        let memory = ObjectStore::memory();
        let path = Path::from("encoded-leaf");
        let writer = memory.create(&path).await?;
        let mut file_writer = lance_file::versions::create_writer(
            ConcreteFileVersion::V2_1,
            writer,
            lance_schema,
            FileWriterOptions::default(),
        )?;
        file_writer.write_batch(&batch).await?;
        file_writer.finish().await?;
        get_whole(&memory, &path).await
    }

    fn encoded_leaves<'a>(
        &'a self,
        fragments: &'a [Fragment],
        range: Range<usize>,
        bytes: Option<bytes::Bytes>,
        config: &'a node::FragmentTreeConfig,
    ) -> impl futures::Stream<Item = Result<EncodedLeaf>> + 'a {
        futures::stream::try_unfold(vec![(range, bytes)], move |mut pending| async move {
            while let Some((range, bytes)) = pending.pop() {
                if range.is_empty() {
                    continue;
                }
                let piece = &fragments[range.clone()];
                let bytes = match bytes {
                    Some(bytes) => bytes,
                    None => self.encode_leaf(piece).await?,
                };
                if bytes.len() as u64 > config.max_leaf_bytes && piece.len() > 1 {
                    let pieces =
                        split_ranges(fragments, range, bytes.len() as u64, config.max_leaf_bytes);
                    pending.extend(pieces.into_iter().rev().map(|piece| (piece, None)));
                    continue;
                }
                if bytes.len() as u64 > config.hard_capacity_bytes {
                    return Err(Error::invalid_input(format!(
                        "fragment {} requires an encoded leaf of {} bytes, exceeding hard_capacity_bytes={}",
                        piece[0].id,
                        bytes.len(),
                        config.hard_capacity_bytes
                    )));
                }
                return Ok(Some((EncodedLeaf { range, bytes }, pending)));
            }
            Ok(None)
        })
    }
}

/// Cut a range whose encoding exceeds the leaf limit into ordered, contiguous,
/// non-empty pieces in one step, so no intermediate half is encoded only to be
/// rejected. A piece that still encodes over the limit is cut again.
pub(super) fn split_ranges(
    fragments: &[Fragment],
    range: Range<usize>,
    encoded_bytes: u64,
    max_leaf_bytes: u64,
) -> Vec<Range<usize>> {
    // A power of two reproduces the leaf count that halving reaches on uniform
    // records. With exactly ceil(encoded / limit) pieces each would sit within
    // a fraction of a percent of the limit, and FSST sampling would push about
    // half of them over it into another cut.
    let piece_count = encoded_bytes
        .div_ceil(max_leaf_bytes)
        .next_power_of_two()
        .min(range.len() as u64);
    // Encoded size follows the data file count, so equal file weight keeps
    // pieces near equal bytes when a few fragments hold most of the files.
    // Weights only place cuts. Each piece's actual encoding decides admission.
    let weight = |fragment: &Fragment| 1 + fragment.files.len() as u64;
    let total_weight: u64 = fragments[range.clone()].iter().map(weight).sum();
    // A first piece of at least one fragment guarantees two pieces even when
    // the leading fragment outweighs every other cut target.
    let first_weight = weight(&fragments[range.start]);
    let mut pieces = Vec::new();
    let mut start = range.start;
    let mut end = range.start;
    let mut weight_before_end = 0;
    for piece in 1..=piece_count {
        let cut_weight = (total_weight * piece / piece_count).max(first_weight);
        while end < range.end && weight_before_end + weight(&fragments[end]) <= cut_weight {
            weight_before_end += weight(&fragments[end]);
            end += 1;
        }
        // A fragment heavier than one piece's share would leave an empty piece.
        // Dropping it keeps that fragment with its neighbors instead of
        // publishing one-fragment leaves beside it.
        if end > start {
            pieces.push(start..end);
            start = end;
        }
    }
    pieces
}
