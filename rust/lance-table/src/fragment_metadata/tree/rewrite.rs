// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Sequence validation across the immutable nodes one rewrite decodes.

use std::collections::BTreeSet;
use std::sync::Mutex;

use lance_core::{Error, Result};

use crate::format::pb;
use crate::fragment_metadata::node::{DecodedSequences, InternalNode};
use crate::fragment_metadata::store::{NodeLocation, NodeStore, Written};

pub(super) struct RewriteNodes {
    state: Mutex<State>,
}

struct State {
    sequences: DecodedSequences,
    admitted: BTreeSet<NodeLocation>,
}

impl RewriteNodes {
    pub(super) fn new(root_buffer: &[pb::FragmentTreeMutation]) -> Self {
        Self {
            state: Mutex::new(State {
                sequences: DecodedSequences::new(root_buffer),
                admitted: BTreeSet::new(),
            }),
        }
    }

    pub(super) async fn read_internal(
        &self,
        store: &NodeStore,
        child: &pb::FragmentTreeChild,
        end: u64,
    ) -> Result<InternalNode> {
        // Range validation still runs on every read, including a repeated
        // source reached through a different parent after a structural repair.
        let node = store.read_internal_in_range(child, end).await?;
        let location = store.child_location(child)?;
        let mut state = self.state.lock().map_err(|_| {
            Error::internal("fragment metadata rewrite sequence validation mutex poisoned")
        })?;
        if state.admitted.insert(location) {
            state.sequences.admit(&child.path, &node.buffer)?;
        }
        Ok(node)
    }

    pub(super) async fn write_internal(
        &self,
        store: &NodeStore,
        children: Vec<pb::FragmentTreeChild>,
        buffer: Vec<pb::FragmentTreeMutation>,
    ) -> Result<Written> {
        let written = store.write_internal(children, buffer).await?;
        let location = store.child_location(&written.child_ref)?;
        // Rewrites only move actions from the root or an admitted interior.
        // A later drain, merge, or root shrink must not admit those actions again.
        self.state
            .lock()
            .map_err(|_| {
                Error::internal("fragment metadata rewrite sequence validation mutex poisoned")
            })?
            .admitted
            .insert(location);
        Ok(written)
    }
}
