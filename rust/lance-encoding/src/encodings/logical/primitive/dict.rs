// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

/// Bits per value for FixedWidth dictionary values (legacy default for 128-bit values)
pub const DICT_FIXED_WIDTH_BITS_PER_VALUE: u64 = 128;
/// Bits per index for dictionary indices (always i32)
pub const DICT_INDICES_BITS_PER_VALUE: u64 = 32;

use arrow_array::{
    Array, DictionaryArray, PrimitiveArray, UInt64Array,
    cast::AsArray,
    make_array,
    types::{
        ArrowDictionaryKeyType, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
        UInt32Type, UInt64Type,
    },
};
use arrow_buffer::ArrowNativeType;
use arrow_schema::DataType;
use arrow_select::take::TakeOptions;
use lance_core::{Error, Result, error::LanceOptionExt};

use crate::{
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock, VariableWidthBlock},
    statistics::{ComputeStat, GetStat, Stat},
};

// Helper function for normalize_dict_nulls
fn normalize_dict_nulls_impl<K: ArrowDictionaryKeyType>(
    array: Arc<dyn Array>,
) -> Result<Arc<dyn Array>> {
    // TODO: Fast path when there is only one null index? (common case)

    let dict_array = array.as_dictionary_opt::<K>().expect_ok()?;

    if dict_array.values().null_count() == 0 {
        return Ok(array);
    }

    let mut mapping = vec![None; dict_array.values().len()];
    let mut skipped = 0;
    let mut valid_indices = Vec::with_capacity(dict_array.values().len());
    for (old_idx, is_valid) in dict_array.values().nulls().expect_ok()?.iter().enumerate() {
        if is_valid {
            // Should be safe since we are only decreasing K values (e.g. won't overflow u8 keys into u16)
            mapping[old_idx] = Some(K::Native::from_usize(old_idx - skipped).expect_ok()?);
            valid_indices.push(old_idx as u64);
        } else {
            skipped += 1;
            mapping[old_idx] = None;
        }
    }

    let mut keys_builder = PrimitiveArray::<K>::builder(dict_array.keys().len());
    for key in dict_array.keys().iter() {
        if let Some(key) = key {
            if let Some(mapped) = mapping[key.to_usize().expect_ok()?] {
                // Valid item
                keys_builder.append_value(mapped);
            } else {
                // Null via values
                keys_builder.append_null();
            }
        } else {
            // Null via keys
            keys_builder.append_null();
        }
    }
    let keys = keys_builder.finish();

    let valid_indices = UInt64Array::from(valid_indices);
    let values = arrow_select::take::take(
        dict_array.values(),
        &valid_indices,
        Some(TakeOptions {
            check_bounds: false,
        }),
    )?;

    Ok(Arc::new(DictionaryArray::new(keys, values)) as Arc<dyn Array>)
}

/// In Arrow a dictionary array can have nulls in two different places:
/// 1. The keys can be null
/// 2. The values can be null
///
/// We want to normalize this so that all nulls are in the keys.  This way we can store
/// the nulls with the keys as rep-def values the same as any other array.
pub fn normalize_dict_nulls(array: Arc<dyn Array>) -> Result<Arc<dyn Array>> {
    match array.data_type() {
        DataType::Dictionary(key_type, _) => match key_type.as_ref() {
            DataType::UInt8 => normalize_dict_nulls_impl::<UInt8Type>(array),
            DataType::UInt16 => normalize_dict_nulls_impl::<UInt16Type>(array),
            DataType::UInt32 => normalize_dict_nulls_impl::<UInt32Type>(array),
            DataType::UInt64 => normalize_dict_nulls_impl::<UInt64Type>(array),
            DataType::Int8 => normalize_dict_nulls_impl::<Int8Type>(array),
            DataType::Int16 => normalize_dict_nulls_impl::<Int16Type>(array),
            DataType::Int32 => normalize_dict_nulls_impl::<Int32Type>(array),
            DataType::Int64 => normalize_dict_nulls_impl::<Int64Type>(array),
            _ => Err(Error::not_supported_source(
                format!("Unsupported dictionary key type: {}", key_type).into(),
            )),
        },
        _ => Err(Error::internal(format!(
            "Data type is not a dictionary: {}",
            array.data_type()
        ))),
    }
}

fn clear_out_of_range_null_keys_impl<K: ArrowDictionaryKeyType>(
    array: Arc<dyn Array>,
) -> Result<Arc<dyn Array>> {
    let dict_array = array.as_dictionary_opt::<K>().expect_ok()?;
    let num_values = dict_array.values().len();
    let Some(nulls) = dict_array.keys().nulls() else {
        return Ok(array);
    };

    // There is no valid replacement key for an empty dictionary, so that case
    // requires separate handling and must remain unchanged here.
    if num_values == 0 {
        return Ok(array);
    }

    let has_out_of_range_null_key = dict_array
        .keys()
        .values()
        .iter()
        .zip(nulls.iter())
        .any(|(key, is_valid)| !is_valid && key.to_usize().is_none_or(|key| key >= num_values));
    if !has_out_of_range_null_key {
        return Ok(array);
    }

    // Building from the logical iterator writes the default physical key into
    // every null slot while preserving the original validity bitmap.
    let keys = PrimitiveArray::<K>::from_iter(dict_array.keys().iter());
    let values = dict_array.values().clone();
    Ok(Arc::new(DictionaryArray::<K>::try_new(keys, values)?) as Arc<dyn Array>)
}

/// Replaces out-of-range physical keys in null dictionary slots with a valid key.
///
/// Arrow permits arbitrary keys in null slots, but the structural encoder removes
/// key validity after recording it as rep-def. The replacement keeps the array
/// valid when that null buffer is removed without changing its logical values.
pub(super) fn clear_out_of_range_null_keys(array: Arc<dyn Array>) -> Result<Arc<dyn Array>> {
    match array.data_type() {
        DataType::Dictionary(key_type, _) => match key_type.as_ref() {
            DataType::UInt8 => clear_out_of_range_null_keys_impl::<UInt8Type>(array),
            DataType::UInt16 => clear_out_of_range_null_keys_impl::<UInt16Type>(array),
            DataType::UInt32 => clear_out_of_range_null_keys_impl::<UInt32Type>(array),
            DataType::UInt64 => clear_out_of_range_null_keys_impl::<UInt64Type>(array),
            DataType::Int8 => clear_out_of_range_null_keys_impl::<Int8Type>(array),
            DataType::Int16 => clear_out_of_range_null_keys_impl::<Int16Type>(array),
            DataType::Int32 => clear_out_of_range_null_keys_impl::<Int32Type>(array),
            DataType::Int64 => clear_out_of_range_null_keys_impl::<Int64Type>(array),
            _ => Err(Error::not_supported_source(
                format!("Unsupported dictionary key type: {}", key_type).into(),
            )),
        },
        _ => Err(Error::internal(format!(
            "Data type is not a dictionary: {}",
            array.data_type()
        ))),
    }
}

/// Returns the values the structural encoder can store for any slice of one dictionary
/// chunk.
///
/// The encoder moves null values into the keys and drops the key validity, so the key
/// under a null slot selects a value as well.  When a slice holds a null slot whose key is
/// out of range, the encoder points all of that slice's null slots at key 0 instead.  The
/// result therefore holds the first value, the value of every valid key, and the value of
/// every in-range key under a null slot.  The first value is also where an all-null chunk
/// without values points its keys.  A chunk without values yields an empty array.
pub fn encoded_dictionary_values(array: Arc<dyn Array>) -> Result<Arc<dyn Array>> {
    let array = normalize_dict_nulls(array)?;
    let dictionary = array.as_any_dictionary_opt().expect_ok()?;
    let values = dictionary.values();
    if values.is_empty() {
        return Ok(values.clone());
    }
    // The physical key of every slot, null or not.  A key that is negative or does not fit
    // `i64` is out of range.
    let physical_keys = make_array(
        dictionary
            .keys()
            .to_data()
            .into_builder()
            .nulls(None)
            .build()?,
    );
    let physical_keys = arrow_cast::cast(&physical_keys, &DataType::Int64)?;
    let mut is_selected = vec![false; values.len()];
    is_selected[0] = true;
    for key in physical_keys.as_primitive::<Int64Type>().iter().flatten() {
        if let Some(is_selected) = usize::try_from(key)
            .ok()
            .and_then(|key| is_selected.get_mut(key))
        {
            *is_selected = true;
        }
    }
    let indices = is_selected
        .iter()
        .enumerate()
        .filter_map(|(index, is_selected)| is_selected.then_some(index as u64))
        .collect::<UInt64Array>();
    Ok(arrow_select::take::take(values, &indices, None)?)
}

const EMPTY_INTERNER_SLOT: u32 = u32::MAX;

#[inline]
fn mix_u64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[inline]
fn mix_u128(v: u128) -> u64 {
    let lo = v as u64;
    let hi = (v >> 64) as u64;
    mix_u64(lo ^ hi.wrapping_mul(0x9e3779b97f4a7c15))
}

struct CompactInterner {
    slots: Vec<u32>,
    mask: usize,
}

impl CompactInterner {
    fn new(max_dict_entries: u32, num_items: usize) -> Self {
        let init_entries = (max_dict_entries as usize).min(num_items).clamp(16, 1024);
        let table_len = (init_entries * 2).next_power_of_two();
        Self {
            slots: vec![EMPTY_INTERNER_SLOT; table_len],
            mask: table_len - 1,
        }
    }

    #[cold]
    fn grow(&mut self, count: usize, mut hash_of: impl FnMut(usize) -> u64) -> Option<()> {
        let new_len = self.slots.len().checked_mul(2)?;
        let new_mask = new_len - 1;
        let mut new_slots = vec![EMPTY_INTERNER_SLOT; new_len];
        for idx in 0..count {
            let mut slot = (hash_of(idx) as usize) & new_mask;
            while new_slots[slot] != EMPTY_INTERNER_SLOT {
                slot = (slot + 1) & new_mask;
            }
            new_slots[slot] = idx as u32;
        }
        self.slots = new_slots;
        self.mask = new_mask;
        Some(())
    }
}

fn dict_encode_variable_width<T>(
    variable_width_data_block: &VariableWidthBlock,
    bits_per_offset: u8,
    max_dict_entries: u32,
    max_encoded_size: usize,
    max_dict_bytes: usize,
) -> Option<(DataBlock, DataBlock)>
where
    T: ArrowNativeType,
    usize: TryFrom<T>,
{
    let offsets = variable_width_data_block
        .offsets
        .borrow_to_typed_slice::<T>();
    let offsets = offsets.as_ref();
    let num_items = offsets.len().saturating_sub(1);
    let bytes_per_offset = (bits_per_offset / 8) as usize;
    let total_indices_bytes = num_items.saturating_mul(DICT_INDICES_BITS_PER_VALUE as usize / 8);

    if num_items > 0
        && (max_dict_entries == 0
            || total_indices_bytes.saturating_add(bytes_per_offset * 2) > max_encoded_size)
    {
        return None;
    }

    let max_len = variable_width_data_block
        .get_stat(Stat::MaxLength)
        .expect("VariableWidth DataBlock should have valid `Stat::MaxLength` statistics");
    let max_len = max_len.as_primitive::<UInt64Type>().value(0);

    let max_dict_data_len = variable_width_data_block.data.len();
    let max_len: usize = max_len.try_into().unwrap_or(usize::MAX);
    let dict_data_capacity = max_len
        .saturating_mul(32)
        .max(1024)
        .min(max_dict_data_len)
        .min(max_encoded_size);

    let init_entries = (max_dict_entries as usize).min(num_items).clamp(16, 1024);
    let mut interner = CompactInterner::new(max_dict_entries, num_items);
    let mut hashes: Vec<u64> = Vec::with_capacity(init_entries);
    let mut dictionary_buffer: Vec<u8> = Vec::with_capacity(dict_data_capacity);
    let mut dictionary_offsets_buffer = Vec::with_capacity(init_entries + 1);
    dictionary_offsets_buffer.push(T::default());
    let mut curr_idx: i32 = 0;
    let mut indices_buffer = Vec::with_capacity(variable_width_data_block.num_values as usize);

    let raw_data = variable_width_data_block.data.as_ref();
    let mut prev: Option<(&[u8], i32)> = None;

    for window in offsets.windows(2) {
        let start = usize::try_from(window[0]).ok()?;
        let end = usize::try_from(window[1]).ok()?;
        if start > end || end > raw_data.len() {
            return None;
        }

        let key = &raw_data[start..end];
        if let Some((prev_key, prev_idx)) = prev
            && prev_key == key
        {
            indices_buffer.push(prev_idx);
            continue;
        }

        let hash = xxhash_rust::xxh3::xxh3_64(key);
        let mut slot = (hash as usize) & interner.mask;
        let idx = loop {
            let entry = interner.slots[slot];
            if entry == EMPTY_INTERNER_SLOT {
                if max_dict_entries == 0
                    || curr_idx as u32 >= max_dict_entries
                    || curr_idx == i32::MAX
                {
                    return None;
                }
                let next_dict_bytes = dictionary_buffer.len().checked_add(key.len())?;
                if next_dict_bytes > max_dict_bytes {
                    return None;
                }
                let next_offsets_bytes =
                    (dictionary_offsets_buffer.len() + 1).saturating_mul(bytes_per_offset);
                let projected_encoded_size = next_dict_bytes
                    .saturating_add(total_indices_bytes)
                    .saturating_add(next_offsets_bytes);
                if projected_encoded_size > max_encoded_size {
                    return None;
                }

                dictionary_buffer.extend_from_slice(key);
                let dict_offset = T::from_usize(dictionary_buffer.len())?;
                dictionary_offsets_buffer.push(dict_offset);
                hashes.push(hash);
                let assigned = curr_idx;
                interner.slots[slot] = assigned as u32;
                curr_idx += 1;
                if (curr_idx as usize) * 2 > interner.slots.len() {
                    interner.grow(curr_idx as usize, |i| hashes[i])?;
                }
                break assigned;
            }
            let e = entry as usize;
            if hashes[e] == hash {
                let d_start = usize::try_from(dictionary_offsets_buffer[e]).ok()?;
                let d_end = usize::try_from(dictionary_offsets_buffer[e + 1]).ok()?;
                if &dictionary_buffer[d_start..d_end] == key {
                    break entry as i32;
                }
            }
            slot = (slot + 1) & interner.mask;
        };

        prev = Some((key, idx));
        indices_buffer.push(idx);
    }

    let mut dictionary_data_block = DataBlock::VariableWidth(VariableWidthBlock {
        data: LanceBuffer::reinterpret_vec(dictionary_buffer),
        offsets: LanceBuffer::reinterpret_vec(dictionary_offsets_buffer),
        bits_per_offset,
        num_values: curr_idx as u64,
        block_info: BlockInfo::default(),
    });
    dictionary_data_block.compute_stat();

    let mut indices_data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(indices_buffer),
        bits_per_value: DICT_INDICES_BITS_PER_VALUE,
        num_values: variable_width_data_block.num_values,
        block_info: BlockInfo::default(),
    });
    indices_data_block.compute_stat();

    Some((indices_data_block, dictionary_data_block))
}

fn dict_encode_fixed_scalar<V>(
    values: &[V],
    num_values: u64,
    bits_per_value: u64,
    max_dict_entries: u32,
    max_encoded_size: usize,
    mut hash_fn: impl FnMut(V) -> u64,
) -> Option<(DataBlock, DataBlock)>
where
    V: ArrowNativeType + Copy + Eq,
{
    let bytes_per_value = std::mem::size_of::<V>();
    let total_indices_bytes = values
        .len()
        .saturating_mul(DICT_INDICES_BITS_PER_VALUE as usize / 8);
    if !values.is_empty()
        && (max_dict_entries == 0
            || total_indices_bytes.saturating_add(bytes_per_value) > max_encoded_size)
    {
        return None;
    }

    let init_entries = (max_dict_entries as usize)
        .min(values.len())
        .clamp(16, 1024);
    let mut interner = CompactInterner::new(max_dict_entries, values.len());
    let mut dictionary_buffer = Vec::with_capacity(init_entries);
    let mut indices_buffer = Vec::with_capacity(values.len());
    let mut curr_idx: i32 = 0;
    let mut prev: Option<(V, i32)> = None;

    for &value in values {
        if let Some((prev_val, prev_idx)) = prev
            && prev_val == value
        {
            indices_buffer.push(prev_idx);
            continue;
        }

        let hash = hash_fn(value);
        let mut slot = (hash as usize) & interner.mask;
        let idx = loop {
            let entry = interner.slots[slot];
            if entry == EMPTY_INTERNER_SLOT {
                if max_dict_entries == 0
                    || curr_idx as u32 >= max_dict_entries
                    || curr_idx == i32::MAX
                {
                    return None;
                }
                let next_dict_bytes = (dictionary_buffer.len() + 1).saturating_mul(bytes_per_value);
                if next_dict_bytes.saturating_add(total_indices_bytes) > max_encoded_size {
                    return None;
                }
                dictionary_buffer.push(value);
                let assigned = curr_idx;
                interner.slots[slot] = assigned as u32;
                curr_idx += 1;
                if (curr_idx as usize) * 2 > interner.slots.len() {
                    interner.grow(curr_idx as usize, |i| hash_fn(dictionary_buffer[i]))?;
                }
                break assigned;
            }
            if dictionary_buffer[entry as usize] == value {
                break entry as i32;
            }
            slot = (slot + 1) & interner.mask;
        };

        prev = Some((value, idx));
        indices_buffer.push(idx);
    }

    let mut dictionary_data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(dictionary_buffer),
        bits_per_value,
        num_values: curr_idx as u64,
        block_info: BlockInfo::default(),
    });
    dictionary_data_block.compute_stat();
    let mut indices_data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(indices_buffer),
        bits_per_value: DICT_INDICES_BITS_PER_VALUE,
        num_values,
        block_info: BlockInfo::default(),
    });
    indices_data_block.compute_stat();

    Some((indices_data_block, dictionary_data_block))
}

/// Dictionary encodes a data block
///
/// Currently only supported for some common cases (string / binary / 64-bit / 128-bit)
///
/// Gives up (returns `None`) once the dictionary would exceed `max_dict_entries` entries,
/// once dictionary plus indices would exceed `max_encoded_size`, or, for variable-width
/// values, once the dictionary values would exceed `max_dict_bytes`.
///
/// Returns a block of indices (will always be a fixed width data block) and a block of dictionary
pub fn dictionary_encode(
    data_block: &DataBlock,
    max_dict_entries: u32,
    max_encoded_size: usize,
    max_dict_bytes: usize,
) -> Option<(DataBlock, DataBlock)> {
    match data_block {
        DataBlock::FixedWidth(fixed_width_data_block) => {
            match fixed_width_data_block.bits_per_value {
                64 => {
                    let u64_slice = fixed_width_data_block.data.borrow_to_typed_slice::<u64>();
                    dict_encode_fixed_scalar(
                        u64_slice.as_ref(),
                        fixed_width_data_block.num_values,
                        64,
                        max_dict_entries,
                        max_encoded_size,
                        mix_u64,
                    )
                }
                128 => {
                    // `i128` implements `ArrowNativeType` and is bit-identical to `u128` for equality/hashing.
                    let i128_slice = fixed_width_data_block.data.borrow_to_typed_slice::<i128>();
                    dict_encode_fixed_scalar(
                        i128_slice.as_ref(),
                        fixed_width_data_block.num_values,
                        DICT_FIXED_WIDTH_BITS_PER_VALUE,
                        max_dict_entries,
                        max_encoded_size,
                        |v| mix_u128(v as u128),
                    )
                }
                _ => None,
            }
        }
        DataBlock::VariableWidth(variable_width_data_block) => {
            match variable_width_data_block.bits_per_offset {
                32 => dict_encode_variable_width::<u32>(
                    variable_width_data_block,
                    32,
                    max_dict_entries,
                    max_encoded_size,
                    max_dict_bytes,
                ),
                64 => dict_encode_variable_width::<u64>(
                    variable_width_data_block,
                    64,
                    max_dict_entries,
                    max_encoded_size,
                    max_dict_bytes,
                ),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Deduplicate byte-aligned rows, including the validity bytes of zipped vectors.
///
/// Keys borrow the input and compare the full row, so hash collisions cannot merge
/// distinct values. The original scalar dictionary paths retain their typed keys.
pub(super) fn encode_fixed_rows(
    data: &FixedWidthDataBlock,
    max_dict_entries: u32,
    max_encoded_size: usize,
) -> Option<(DataBlock, FixedWidthDataBlock)> {
    let width = usize::try_from(data.bits_per_value / 8).ok()?;
    let count = usize::try_from(data.num_values).ok()?;
    if width == 0
        || !data.bits_per_value.is_multiple_of(8)
        || count.checked_mul(width)? != data.data.len()
    {
        return None;
    }
    let indices_bytes = count.checked_mul(4)?;
    if indices_bytes > max_encoded_size {
        return None;
    }

    let max_entries_capped = max_dict_entries.min(i32::MAX as u32);
    let init_entries = (max_entries_capped as usize).min(count).clamp(16, 1024);
    let mut interner = CompactInterner::new(max_entries_capped, count);
    let mut hashes: Vec<u64> = Vec::with_capacity(init_entries);
    let mut items = Vec::with_capacity(init_entries.saturating_mul(width).min(max_encoded_size));
    let mut indices = Vec::with_capacity(count);
    let mut num_entries: u32 = 0;
    let mut prev: Option<(&[u8], u32)> = None;

    for value in data.data.chunks_exact(width) {
        if let Some((prev_val, prev_idx)) = prev
            && prev_val == value
        {
            indices.push(prev_idx);
            continue;
        }

        let hash = xxhash_rust::xxh3::xxh3_64(value);
        let mut slot = (hash as usize) & interner.mask;
        let index = loop {
            let entry = interner.slots[slot];
            if entry == EMPTY_INTERNER_SLOT {
                if num_entries >= max_entries_capped
                    || items.len().checked_add(width)?.checked_add(indices_bytes)?
                        > max_encoded_size
                {
                    return None;
                }
                items.extend_from_slice(value);
                hashes.push(hash);
                let assigned = num_entries;
                interner.slots[slot] = assigned;
                num_entries = num_entries.checked_add(1)?;
                if (num_entries as usize) * 2 > interner.slots.len() {
                    interner.grow(num_entries as usize, |i| hashes[i])?;
                }
                break assigned;
            }
            let e = entry as usize;
            if hashes[e] == hash && &items[e * width..(e + 1) * width] == value {
                break entry;
            }
            slot = (slot + 1) & interner.mask;
        };

        prev = Some((value, index));
        indices.push(index);
    }
    let mut indices = DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(indices),
        bits_per_value: 32,
        num_values: data.num_values,
        block_info: BlockInfo::new(),
    });
    indices.compute_stat();
    Some((
        indices,
        FixedWidthDataBlock {
            data: LanceBuffer::from(items),
            bits_per_value: data.bits_per_value,
            num_values: num_entries as u64,
            block_info: BlockInfo::new(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        buffer::LanceBuffer,
        data::{BlockInfo, FixedWidthDataBlock},
    };
    use arrow_array::{Array, Int8Array, StringArray};
    use arrow_buffer::{NullBuffer, ScalarBuffer};
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    // Index 3 is reached only through a null slot; index 100 is out of range and skipped.
    #[case::keys_under_nulls(
        vec![Some("a"), Some("b"), Some("c"), Some("d"), Some("e")],
        vec![1, 3, 100],
        vec![true, false, false],
        vec!["a", "b", "d"]
    )]
    // The null value moves into the keys, so only the valid values remain.
    #[case::null_values(
        vec![Some("a"), None, Some("c")],
        vec![0, 1, 2],
        vec![true, true, true],
        vec!["a", "c"]
    )]
    fn test_encoded_dictionary_values(
        #[case] values: Vec<Option<&str>>,
        #[case] keys: Vec<i8>,
        #[case] validity: Vec<bool>,
        #[case] expected: Vec<&str>,
    ) {
        let keys = Int8Array::new(ScalarBuffer::from(keys), Some(NullBuffer::from(validity)));
        let dictionary =
            DictionaryArray::<Int8Type>::try_new(keys, Arc::new(StringArray::from(values)))
                .unwrap();
        let encoded = encoded_dictionary_values(Arc::new(dictionary)).unwrap();
        assert_eq!(encoded.as_string::<i32>(), &StringArray::from(expected));
    }

    #[test]
    fn test_fixed_row_dictionary_preserves_float_bits_and_budgets() {
        let patterns = [0u32, 0x8000_0000, 0x7fc0_0001, 0x7fc0_0002];
        let data = FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(
                (0..128)
                    .flat_map(|i| [patterns[i % 4], 1.0f32.to_bits()])
                    .collect::<Vec<_>>(),
            ),
            bits_per_value: 64,
            num_values: 128,
            block_info: BlockInfo::new(),
        };
        let size = 128 * 4 + 4 * 8;
        assert!(encode_fixed_rows(&data, 3, size).is_none());
        assert!(encode_fixed_rows(&data, 4, size - 1).is_none());
        let (indices, items) = encode_fixed_rows(&data, 4, size).unwrap();
        assert_eq!(items.num_values, 4);
        let values = items.data.borrow_to_typed_slice::<u32>();
        for (i, pattern) in patterns.iter().enumerate() {
            assert_eq!(values[i * 2], *pattern);
        }
        let indices = indices
            .as_fixed_width_ref()
            .unwrap()
            .data
            .borrow_to_typed_slice::<u32>();
        for (i, index) in indices.iter().enumerate() {
            assert_eq!(*index, (i % 4) as u32);
        }
    }
    #[test]
    fn test_dictionary_encode_abort_fixed_width() {
        // Create a u128 block with very high cardinality where dict encoding
        // would result in larger data (dictionary overhead + indices > original)
        let num_values = 120u64;

        // Create actual data: each value is unique u128 so dictionary encode will not be helpful
        let mut data = Vec::with_capacity(num_values as usize);
        for i in 0..num_values {
            data.push(i as u128);
        }

        let mut data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: DICT_FIXED_WIDTH_BITS_PER_VALUE,
            data: LanceBuffer::reinterpret_vec(data),
            num_values,
            block_info: BlockInfo::default(),
        });

        // Compute stats naturally
        data_block.compute_stat();

        // Dictionary encoding should abort and return None
        let max_encoded_size = usize::try_from(data_block.data_size()).unwrap_or(usize::MAX);
        let result = dictionary_encode(&data_block, 1000, max_encoded_size, usize::MAX);
        assert!(
            result.is_none(),
            "Dictionary encoding should abort for high cardinality u128 data"
        );
    }

    #[test]
    fn test_dictionary_encode_success_fixed_width() {
        // Create a u128 block with low cardinality where dict encoding helps
        let num_values = 120u64;
        let cardinality = 3u64;

        // Create data with few unique u128 values
        let mut data = Vec::with_capacity(num_values as usize);
        for i in 0..num_values {
            data.push((i % cardinality) as u128);
        }

        let mut data_block = DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: DICT_FIXED_WIDTH_BITS_PER_VALUE,
            data: LanceBuffer::reinterpret_vec(data),
            num_values,
            block_info: BlockInfo::default(),
        });

        // Compute stats naturally
        data_block.compute_stat();

        // Dictionary encoding should succeed and return Some
        let max_encoded_size = usize::try_from(data_block.data_size()).unwrap_or(usize::MAX);
        let result = dictionary_encode(&data_block, 1000, max_encoded_size, usize::MAX);
        assert!(
            result.is_some(),
            "Dictionary encoding should succeed for low cardinality u128 data"
        );

        if let Some((indices, dictionary)) = result {
            // Verify indices block
            if let DataBlock::FixedWidth(indices_block) = indices {
                assert_eq!(indices_block.num_values, num_values);
                assert_eq!(indices_block.bits_per_value, DICT_INDICES_BITS_PER_VALUE);
            } else {
                panic!("Expected FixedWidth indices block");
            }

            // Verify dictionary block
            if let DataBlock::FixedWidth(dict_block) = dictionary {
                assert_eq!(dict_block.num_values, cardinality);
                assert_eq!(dict_block.bits_per_value, DICT_FIXED_WIDTH_BITS_PER_VALUE);
            } else {
                panic!("Expected FixedWidth dictionary block");
            }
        }
    }

    #[test]
    fn test_dictionary_encode_abort_variable_width() {
        // Create a variable-width block with high cardinality where dict encoding
        // won't provide sufficient benefit
        let num_values = 120u64;
        let mut values = Vec::with_capacity(num_values as usize);
        for i in 0..num_values {
            values.push(format!("unique_value_{:04}", i));
        }
        let array = StringArray::from(values);
        // from_array already computes stats
        let data_block = DataBlock::from_array(Arc::new(array) as Arc<dyn Array>);

        // Dictionary encoding should abort and return None
        let max_encoded_size = usize::try_from(data_block.data_size()).unwrap_or(usize::MAX);
        let result = dictionary_encode(&data_block, 10, max_encoded_size, usize::MAX);
        assert!(
            result.is_none(),
            "Dictionary encoding should abort for high cardinality string data"
        );
    }

    #[test]
    fn test_dictionary_encode_success_low_cardinality() {
        // Create a variable-width block with low cardinality where dict encoding helps
        let num_values = 120u64;
        let cardinality = 3u64;

        let mut values = Vec::with_capacity(num_values as usize);
        for i in 0..num_values {
            values.push(format!("value_{}", i % cardinality));
        }

        let array = StringArray::from(values);
        let data_block = DataBlock::from_array(Arc::new(array) as Arc<dyn Array>);

        // Dictionary encoding should succeed and return Some
        let max_encoded_size = usize::try_from(data_block.data_size()).unwrap_or(usize::MAX);
        let result = dictionary_encode(&data_block, 100, max_encoded_size, usize::MAX);
        assert!(
            result.is_some(),
            "Dictionary encoding should succeed for low cardinality data"
        );

        if let Some((indices, dictionary)) = result {
            // Verify indices block
            if let DataBlock::FixedWidth(indices_block) = indices {
                assert_eq!(indices_block.num_values, num_values);
                assert_eq!(indices_block.bits_per_value, DICT_INDICES_BITS_PER_VALUE);
            } else {
                panic!("Expected FixedWidth indices block");
            }

            // Verify dictionary block
            if let DataBlock::VariableWidth(dict_block) = dictionary {
                assert_eq!(dict_block.num_values, cardinality);
            } else {
                panic!("Expected VariableWidth dictionary block");
            }
        }
    }

    #[test]
    fn test_dictionary_encode_invalid_offset_width_returns_none() {
        let array = StringArray::from(vec!["a", "b", "c", "a"]);
        let data_block = DataBlock::from_array(Arc::new(array) as Arc<dyn Array>);
        let invalid_block = match data_block {
            DataBlock::VariableWidth(mut var) => {
                var.bits_per_offset = 16;
                DataBlock::VariableWidth(var)
            }
            other => panic!("Expected VariableWidth data block, got {:?}", other),
        };
        let max_encoded_size = usize::try_from(invalid_block.data_size()).unwrap_or(usize::MAX);
        assert!(dictionary_encode(&invalid_block, 100, max_encoded_size, usize::MAX).is_none());
    }

    #[test]
    fn test_dictionary_encode_respects_size_limit() {
        let num_values = 10_000u64;
        let cardinality = 50u64;

        let mut values = Vec::with_capacity(num_values as usize);
        for i in 0..num_values {
            values.push(format!("value_{:08}", i % cardinality));
        }

        let array = StringArray::from(values);
        let data_block = DataBlock::from_array(Arc::new(array) as Arc<dyn Array>);

        let full_size = usize::try_from(data_block.data_size()).unwrap_or(usize::MAX);
        let too_small_limit = full_size / 10;
        assert!(dictionary_encode(&data_block, 1000, too_small_limit, usize::MAX).is_none());
        assert!(dictionary_encode(&data_block, 1000, full_size, usize::MAX).is_some());
    }

    #[test]
    fn test_dictionary_encode_respects_entry_limit() {
        let num_values = 10_000u64;
        let cardinality = 200u64;

        let mut values = Vec::with_capacity(num_values as usize);
        for i in 0..num_values {
            values.push(format!("value_{:08}", i % cardinality));
        }

        let array = StringArray::from(values);
        let data_block = DataBlock::from_array(Arc::new(array) as Arc<dyn Array>);

        let max_encoded_size = usize::try_from(data_block.data_size()).unwrap_or(usize::MAX);
        assert!(dictionary_encode(&data_block, 10, max_encoded_size, usize::MAX).is_none());
        assert!(dictionary_encode(&data_block, 500, max_encoded_size, usize::MAX).is_some());
    }

    #[test]
    fn test_dictionary_encode_respects_dict_byte_limit() {
        // 300 values of 256 bytes with 100 distinct: the dictionary holds 25,600 bytes
        // while dictionary plus indices stay far below the raw size.
        let array = StringArray::from_iter_values((0..300).map(|i| format!("{:0256}", i % 100)));
        let data_block = DataBlock::from_array(Arc::new(array) as Arc<dyn Array>);

        let max_encoded_size = usize::try_from(data_block.data_size()).unwrap_or(usize::MAX);
        assert!(dictionary_encode(&data_block, 1000, max_encoded_size, 16 * 1024).is_none());
        let (_, dictionary) =
            dictionary_encode(&data_block, 1000, max_encoded_size, 25_600).unwrap();
        assert_eq!(dictionary.num_values(), 100);
    }

    #[test]
    fn test_compact_interner_growth_and_runs_preserve_first_seen_ids() {
        // 1,600 distinct strings (> 1,024 initial cap, triggering CompactInterner::grow)
        // interleaved with consecutive runs (exercising 1-entry MRU fast path).
        let cardinality = 1_600usize;
        let mut values = Vec::with_capacity(cardinality * 4);
        for i in 0..cardinality {
            let s = format!("tenant-tag-{i:05}-payload");
            values.push(s.clone());
            values.push(s);
        }
        for i in (0..cardinality).rev() {
            let s = format!("tenant-tag-{i:05}-payload");
            values.push(s.clone());
            values.push(s);
        }
        let array = StringArray::from(values.clone());
        let block = DataBlock::from_array(Arc::new(array) as Arc<dyn Array>);
        let (indices_blk, dict_blk) =
            dictionary_encode(&block, 4_096, usize::MAX, usize::MAX).unwrap();
        assert_eq!(dict_blk.num_values(), cardinality as u64);

        let reconstructed = crate::data::DictionaryDataBlock::from_parts(
            indices_blk.as_fixed_width().unwrap(),
            dict_blk,
        )
        .decode()
        .unwrap()
        .into_arrow(DataType::Utf8, true)
        .unwrap();
        let actual = make_array(reconstructed);
        assert_eq!(actual.as_ref(), &StringArray::from(values) as &dyn Array);
    }
}
