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
    /// The commit's frontier. It bounds the interiors this rewrite writes,
    /// while the store bounds every other interior by the root's frontier.
    next_action_sequence: u64,
    state: Mutex<State>,
}

struct State {
    sequences: DecodedSequences,
    /// Interiors of the source tree whose buffers this rewrite admitted.
    admitted: BTreeSet<NodeLocation>,
    /// Interiors this rewrite wrote. Their actions came from the root or an
    /// admitted interior, so a later drain, merge, or root shrink must not
    /// admit them again.
    written: BTreeSet<NodeLocation>,
}

impl RewriteNodes {
    pub(super) fn new(root_buffer: &[pb::FragmentTreeMutation], next_action_sequence: u64) -> Self {
        Self {
            next_action_sequence,
            state: Mutex::new(State {
                sequences: DecodedSequences::new(root_buffer),
                admitted: BTreeSet::new(),
                written: BTreeSet::new(),
            }),
        }
    }

    fn state(&self) -> Result<std::sync::MutexGuard<'_, State>> {
        self.state.lock().map_err(|_| {
            Error::internal("fragment metadata rewrite sequence validation mutex poisoned")
        })
    }

    pub(super) async fn read_internal(
        &self,
        store: &NodeStore,
        child: &pb::FragmentTreeChild,
        end: u64,
    ) -> Result<InternalNode> {
        let location = store.child_location(child)?;
        if self.state()?.written.contains(&location) {
            let mut own = store.clone();
            own.next_action_sequence = self.next_action_sequence;
            return own.read_internal_in_range(child, end).await;
        }
        // Range validation still runs on every read, including a repeated
        // source reached through a different parent after a structural repair.
        let node = store.read_internal_in_range(child, end).await?;
        let mut state = self.state()?;
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
        self.state()?.written.insert(location);
        Ok(written)
    }
}
