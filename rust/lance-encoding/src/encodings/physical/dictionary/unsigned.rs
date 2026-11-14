// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Borrow dictionary children and look up items without expanded arrays.

use super::{BlockDictionaryDecompressor, split_payload};
use crate::buffer::LanceBuffer;
use lance_core::{Error, Result};

pub(super) fn visit(
    codec: &BlockDictionaryDecompressor,
    data: Option<&LanceBuffer>,
    bits: u64,
    count: u64,
    visitor: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<()> {
    if bits != codec.bits_per_value {
        return Err(Error::invalid_input(
            "unsigned dictionary visitor width mismatch",
        ));
    }
    let indices_have_payload = codec.indices.requires_payload();
    let items_have_payload = codec.items.requires_payload();
    let (indices, items) = split_payload(
        data.cloned(),
        indices_have_payload,
        items_have_payload,
        indices_have_payload || items_have_payload,
    )?;
    let item_count = u64::from(codec.num_dictionary_items);
    let mut seen_items = 0;
    codec
        .items
        .visit_unsigned(items.as_ref(), bits, item_count, &mut |_| {
            if seen_items >= item_count {
                return Err(Error::invalid_input(
                    "dictionary item visitor exceeds item count",
                ));
            }
            seen_items += 1;
            Ok(())
        })?;
    if seen_items != item_count {
        return Err(Error::invalid_input(
            "dictionary item visitor has too few items",
        ));
    }
    // Establish lookup support before calling the consumer, even for no indices.
    // The constructor guarantees a nonempty bounded dictionary.
    codec
        .items
        .unsigned_value_at(items.as_ref(), bits, item_count, 0)?;
    let mut seen_indices = 0;
    codec.indices.visit_unsigned(indices.as_ref(), 32, count, &mut |index| {
        if seen_indices >= count || index >= item_count {
            return Err(Error::invalid_input(format!(
                "Dictionary index {index} at position {seen_indices} exceeds {item_count} items or {count} values"
            )));
        }
        let value = codec.items.unsigned_value_at(items.as_ref(), bits, item_count, index)?;
        seen_indices += 1;
        visitor(value)
    })?;
    if seen_indices != count {
        return Err(Error::invalid_input(
            "dictionary index visitor has too few values",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
