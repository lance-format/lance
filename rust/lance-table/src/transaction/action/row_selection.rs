// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-fragment row selections, shared by the actions that name rows.

use crate::format::pb_actions;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use std::collections::HashSet;

/// A selection of rows within one committed fragment.
#[derive(Debug, Clone, PartialEq)]
pub struct RowSelection {
    /// The committed fragment's id. `u32` like the wire message, which mirrors
    /// the legacy transaction's row selection, and like the fragment half of a
    /// row address, which is all a physical offset can be paired with.
    pub fragment_id: u32,
    /// Which of the fragment's rows.
    pub rows: SelectedRows,
}

/// Which rows of a fragment a [`RowSelection`] covers.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectedRows {
    /// Every row of the fragment.
    Full,
    /// The rows at these physical offsets.
    Offsets(RoaringBitmap),
}

impl DeepSizeOf for RowSelection {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        // RoaringBitmap does not expose its allocation size; its serialized
        // size is a close proxy.
        match &self.rows {
            SelectedRows::Full => 0,
            SelectedRows::Offsets(offsets) => offsets.serialized_size(),
        }
    }
}

impl From<&RowSelection> for pb_actions::RowSelection {
    fn from(selection: &RowSelection) -> Self {
        Self {
            fragment_id: selection.fragment_id,
            coverage: Some(match &selection.rows {
                SelectedRows::Full => pb_actions::row_selection::Coverage::Full(()),
                SelectedRows::Offsets(offsets) => {
                    let mut bytes = Vec::with_capacity(offsets.serialized_size());
                    // Writing into a Vec cannot fail.
                    offsets
                        .serialize_into(&mut bytes)
                        .expect("serializing a bitmap into memory");
                    pb_actions::row_selection::Coverage::Bitmap(bytes)
                }
            }),
        }
    }
}

/// Decode one action's row selections, rejecting a fragment selected twice, a
/// bitmap that does not deserialize, and a selection with no coverage.
/// `action` names the action in the errors.
pub(super) fn row_selections_from_proto(
    messages: Vec<pb_actions::RowSelection>,
    action: &str,
) -> Result<Vec<RowSelection>> {
    let mut seen = HashSet::with_capacity(messages.len());
    messages
        .into_iter()
        .map(|selection| {
            let fragment_id = selection.fragment_id;
            if !seen.insert(fragment_id) {
                return Err(Error::invalid_input(format!(
                    "{action} selects rows of fragment {fragment_id} more than once"
                )));
            }
            let rows = match selection.coverage {
                Some(pb_actions::row_selection::Coverage::Full(())) => SelectedRows::Full,
                Some(pb_actions::row_selection::Coverage::Bitmap(bytes)) => SelectedRows::Offsets(
                    RoaringBitmap::deserialize_from(&bytes[..]).map_err(|error| {
                        Error::invalid_input(format!(
                            "{action} selects rows of fragment {fragment_id} with a bitmap that \
                             is not a portable Roaring bitmap: {error}"
                        ))
                    })?,
                ),
                None => {
                    return Err(Error::invalid_input(format!(
                        "{action} selects rows of fragment {fragment_id} without saying which"
                    )));
                }
            };
            Ok(RowSelection { fragment_id, rows })
        })
        .collect()
}
