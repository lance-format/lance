// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;

use futures::{StreamExt, TryStreamExt};
use lance_core::utils::{address::RowAddress, deletion::OffsetMapper};
use lance_core::{Error, Result};
use lance_table::rowids::FragmentRowOffset;

use crate::{FileFragment, Table};

/// Convert logical row offsets into physical row addresses, in input order.
///
/// Offsets are zero-based across the supplied fragments in their given order.
/// Deletions do not count toward an offset. An offset at or beyond the end of
/// the live rows is mapped to [`RowAddress::TOMBSTONE_ROW`].
pub async fn row_offsets_to_row_addresses<'a>(
    fragments: impl IntoIterator<Item = &'a FileFragment>,
    row_indices: &[u64],
) -> Result<Vec<u64>> {
    if row_indices.is_empty() {
        return Ok(Vec::new());
    }

    let mut sorted_indices = (0..row_indices.len()).collect::<Vec<_>>();
    sorted_indices.sort_unstable_by_key(|&index| row_indices[index]);

    let mut fragments = fragments.into_iter();
    let mut fragment = fragments.next();
    let mut live_rows = match fragment {
        Some(fragment) => fragment.count_rows().await? as u64,
        None => 0,
    };
    let mut offset_mapper = None;
    let mut is_mapper_loaded = false;
    let mut fragment_start = 0;

    let mut addresses = vec![RowAddress::TOMBSTONE_ROW; row_indices.len()];
    for index in sorted_indices {
        let offset = row_indices[index];
        while fragment.is_some() && offset >= fragment_start + live_rows {
            fragment_start += live_rows;
            fragment = fragments.next();
            live_rows = match fragment {
                Some(fragment) => fragment.count_rows().await? as u64,
                None => 0,
            };
            offset_mapper = None;
            is_mapper_loaded = false;
        }
        let Some(fragment) = fragment else {
            break;
        };

        // Counts usually come from metadata. Only selected fragments need
        // deletion contents to translate logical offsets to physical addresses.
        if !is_mapper_loaded {
            offset_mapper = fragment.get_deletion_vector().await?.map(OffsetMapper::new);
            is_mapper_loaded = true;
        }

        let mut local_offset = (offset - fragment_start) as u32;
        if let Some(mapper) = &mut offset_mapper {
            local_offset = mapper.map_offset(local_offset);
        }
        addresses[index] = RowAddress::new_from_parts(fragment.id() as u32, local_offset).into();
    }

    Ok(addresses)
}

impl Table {
    /// Load the starting logical offsets and deletion vectors for this snapshot.
    #[doc(hidden)]
    pub async fn fragment_row_offsets(&self) -> Result<HashMap<u32, FragmentRowOffset>> {
        // Deletion reads can run concurrently, but cumulative offsets must be
        // computed in manifest order.
        let mut infos = futures::stream::iter(self.get_fragments())
            .map(|fragment| async move {
                let deletion_vector = fragment.get_deletion_vector().await?;
                let num_rows = fragment.count_rows().await?;
                Ok::<_, Error>((fragment.id() as u32, deletion_vector, num_rows))
            })
            .buffered(self.state.object_store.io_parallelism());

        let mut offsets = HashMap::with_capacity(self.count_fragments());
        let mut row_offset = 0;
        while let Some((fragment_id, deletion_vector, num_rows)) = infos.try_next().await? {
            offsets.insert(
                fragment_id,
                FragmentRowOffset {
                    row_offset,
                    deletion_vector,
                },
            );
            row_offset += num_rows as u64;
        }
        Ok(offsets)
    }
}
