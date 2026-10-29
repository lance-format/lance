// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Data layouts to represent encoded data in a sub-Arrow format
//!
//! These [`DataBlock`] structures represent physical layouts.  They fill a gap somewhere
//! between [`arrow_data::ArrayData`] (which, as a collection of buffers, is too
//! generic because it doesn't give us enough information about what those buffers represent)
//! and [`arrow_array::array::Array`] (which is too specific, because it cares about the
//! logical data type).
//!
//! In addition, the layouts represented here are slightly stricter than Arrow's layout rules.
//! For example, offset buffers MUST start with 0.  These additional restrictions impose a
//! slight penalty on encode (to normalize arrow data) but make the development of encoders
//! and decoders easier (since they can rely on a normalized representation)

use std::{
    ops::Range,
    sync::{Arc, RwLock},
};

use arrow_array::{
    Array, ArrayRef, DictionaryArray, OffsetSizeTrait, UInt32Array, UInt64Array,
    cast::AsArray,
    make_array, new_empty_array, new_null_array,
    types::{
        ArrowDictionaryKeyType, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
        UInt32Type, UInt64Type,
    },
};
use arrow_buffer::{ArrowNativeType, BooleanBuffer, BooleanBufferBuilder, NullBuffer};
use arrow_data::{ArrayData, ArrayDataBuilder};
use arrow_schema::DataType;
use arrow_select::dictionary::garbage_collect_dictionary;
use lance_arrow::DataTypeExt;

use lance_core::{Error, Result};

use crate::{
    buffer::LanceBuffer,
    statistics::{ComputeStat, Stat},
};

/// A data block with no buffers where everything is null
///
/// Note: this data block should not be used for future work.  It will be deprecated
/// in the 2.1 version of the format where nullability will be handled by the structural
/// encoders.
#[derive(Debug, Clone)]
pub struct AllNullDataBlock {
    /// The number of values represented by this block
    pub num_values: u64,
}

impl AllNullDataBlock {
    fn into_arrow(self, data_type: DataType, _validate: bool) -> Result<ArrayData> {
        Ok(ArrayData::new_null(&data_type, self.num_values as usize))
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        vec![]
    }
}

use std::collections::HashMap;

// `BlockInfo` stores the statistics of this `DataBlock`, such as `NullCount` for `NullableDataBlock`,
// `BitWidth` for `FixedWidthDataBlock`, `Cardinality` for all `DataBlock`
#[derive(Debug, Clone)]
pub struct BlockInfo(pub Arc<RwLock<HashMap<Stat, Arc<dyn Array>>>>);

impl Default for BlockInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockInfo {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(HashMap::new())))
    }
}

impl PartialEq for BlockInfo {
    fn eq(&self, other: &Self) -> bool {
        let self_info = self.0.read().unwrap();
        let other_info = other.0.read().unwrap();
        *self_info == *other_info
    }
}

/// Wraps a data block and adds nullability information to it
///
/// Note: this data block should not be used for future work.  It will be deprecated
/// in the 2.1 version of the format where nullability will be handled by the structural
/// encoders.
#[derive(Debug, Clone)]
pub struct NullableDataBlock {
    /// The underlying data
    pub data: Box<DataBlock>,
    /// A bitmap of validity for each value
    pub nulls: LanceBuffer,

    pub block_info: BlockInfo,
}

impl NullableDataBlock {
    fn into_arrow(self, data_type: DataType, _validate: bool) -> Result<ArrayData> {
        if data_type == DataType::Null {
            // Every slot of a Null array is null, and arrow-rs rejects a validity buffer on it.
            return Ok(ArrayData::new_null(
                &data_type,
                self.data.num_values() as usize,
            ));
        }
        let nulls = self.nulls.into_buffer();
        let data = self.data.into_arrow_impl(data_type, true)?.into_builder();
        let data = data.null_bit_buffer(Some(nulls));
        Ok(data.build()?)
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        let mut buffers = vec![self.nulls];
        buffers.extend(self.data.into_buffers());
        buffers
    }

    pub fn data_size(&self) -> u64 {
        self.data.data_size() + self.nulls.len() as u64
    }
}

/// A block representing the same constant value repeated many times
#[derive(Debug, PartialEq, Clone)]
pub struct ConstantDataBlock {
    /// Data buffer containing the value
    pub data: LanceBuffer,
    /// The number of values
    pub num_values: u64,
}

impl ConstantDataBlock {
    fn into_buffers(self) -> Vec<LanceBuffer> {
        vec![self.data]
    }

    fn into_arrow(self, _data_type: DataType, _validate: bool) -> Result<ArrayData> {
        // We don't need this yet but if we come up with some way of serializing
        // scalars to/from bytes then we could implement it.
        todo!()
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.clone(),
            num_values: self.num_values,
        })
    }

    pub fn data_size(&self) -> u64 {
        self.data.len() as u64
    }
}

/// A data block for a single buffer of data where each element has a fixed number of bits
#[derive(Debug, PartialEq, Clone)]
pub struct FixedWidthDataBlock {
    /// The data buffer
    pub data: LanceBuffer,
    /// The number of bits per value
    pub bits_per_value: u64,
    /// The number of values represented by this block
    pub num_values: u64,

    pub block_info: BlockInfo,
}

impl FixedWidthDataBlock {
    fn do_into_arrow(
        self,
        data_type: DataType,
        num_values: u64,
        _validate: bool,
    ) -> Result<ArrayData> {
        // Booleans expanded for full-zip (bits_per_value==8, one byte each) need re-packing to
        // Arrow's bit-packed format.
        let data_buffer = if matches!(data_type, DataType::Boolean) && self.bits_per_value == 8 {
            let mut builder = BooleanBufferBuilder::new(num_values as usize);
            for &byte in self.data.as_ref().iter().take(num_values as usize) {
                builder.append(byte != 0);
            }
            builder.finish().into_inner()
        } else {
            self.data.into_buffer()
        };
        let builder = ArrayDataBuilder::new(data_type)
            .add_buffer(data_buffer)
            .len(num_values as usize)
            .null_count(0);
        Ok(builder.build()?)
    }

    /// Convert this block into Arrow data with full layout validation.
    ///
    /// The `validate` argument is retained for API compatibility. Conversion is
    /// always validated because callers can construct this public type directly.
    pub fn into_arrow(self, data_type: DataType, _validate: bool) -> Result<ArrayData> {
        let root_num_values = self.num_values;
        self.do_into_arrow(data_type, root_num_values, true)
    }

    pub fn into_buffers(self) -> Vec<LanceBuffer> {
        vec![self.data]
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            data: self.data.clone(),
            bits_per_value: self.bits_per_value,
            num_values: self.num_values,
            block_info: self.block_info.clone(),
        })
    }

    pub fn data_size(&self) -> u64 {
        self.data.len() as u64
    }
}

#[derive(Debug)]
struct VariableWidthDataBlockBuilder<T: OffsetSizeTrait> {
    offsets: Vec<T>,
    bytes: Vec<u8>,
}

impl<T: OffsetSizeTrait> VariableWidthDataBlockBuilder<T> {
    fn new(estimated_size_bytes: u64) -> Self {
        Self {
            offsets: vec![T::from_usize(0).unwrap()],
            bytes: Vec::with_capacity(estimated_size_bytes as usize),
        }
    }
}

impl<T: OffsetSizeTrait + bytemuck::Pod> DataBlockBuilderImpl for VariableWidthDataBlockBuilder<T> {
    fn validate_append(&self, data_block: &DataBlock, selection: &Range<u64>) -> Result<()> {
        let block = data_block.as_variable_width_ref().unwrap();
        block.validate_offsets_for_append::<T>(selection)
    }

    fn append_validated(&mut self, data_block: &DataBlock, selection: Range<u64>) -> Result<()> {
        let block = data_block.as_variable_width_ref().unwrap();
        debug_assert_eq!(block.bits_per_offset, T::get_byte_width() as u8 * 8);
        let offsets = block.offsets.borrow_to_typed_view::<T>();

        let start_offset = offsets[selection.start as usize];
        let end_offset = offsets[selection.end as usize];
        let selected_data_len = end_offset.as_usize() - start_offset.as_usize();
        let new_data_len = self
            .bytes
            .len()
            .checked_add(selected_data_len)
            .ok_or_else(|| {
                Error::not_supported_source(
                    "appending variable-width data would overflow usize".into(),
                )
            })?;
        if T::from_usize(new_data_len).is_none() {
            return Err(Error::not_supported_source(
                format!(
                    "appending variable-width data would require {} bytes, which exceeds the \
                     capacity of {}-bit offsets",
                    new_data_len,
                    T::get_byte_width() * 8
                )
                .into(),
            ));
        }
        let previous_len = self.bytes.len();

        self.bytes
            .extend_from_slice(&block.data[start_offset.as_usize()..end_offset.as_usize()]);

        self.offsets.extend(
            offsets[selection.start as usize + 1..=selection.end as usize]
                .iter()
                .map(|&offset| {
                    let rebased_offset =
                        previous_len + (offset.as_usize() - start_offset.as_usize());
                    T::from_usize(rebased_offset).unwrap()
                }),
        );
        Ok(())
    }

    fn finish(self: Box<Self>) -> DataBlock {
        let num_values = (self.offsets.len() - 1) as u64;
        DataBlock::VariableWidth(VariableWidthBlock {
            data: LanceBuffer::from(self.bytes),
            offsets: LanceBuffer::reinterpret_vec(self.offsets),
            bits_per_offset: T::get_byte_width() as u8 * 8,
            num_values,
            block_info: BlockInfo::new(),
        })
    }
}

#[derive(Debug)]
struct BitmapDataBlockBuilder {
    values: BooleanBufferBuilder,
}

impl BitmapDataBlockBuilder {
    fn new(estimated_size_bytes: u64) -> Self {
        Self {
            values: BooleanBufferBuilder::new(estimated_size_bytes as usize * 8),
        }
    }
}

impl DataBlockBuilderImpl for BitmapDataBlockBuilder {
    fn validate_append(&self, _data_block: &DataBlock, _selection: &Range<u64>) -> Result<()> {
        Ok(())
    }

    fn append_validated(&mut self, data_block: &DataBlock, selection: Range<u64>) -> Result<()> {
        let bitmap_blk = data_block.as_fixed_width_ref().unwrap();
        self.values.append_packed_range(
            selection.start as usize..selection.end as usize,
            &bitmap_blk.data,
        );
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> DataBlock {
        let bool_buf = self.values.finish();
        let num_values = bool_buf.len() as u64;
        let bits_buf = bool_buf.into_inner();
        DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bits_buf),
            bits_per_value: 1,
            num_values,
            block_info: BlockInfo::new(),
        })
    }
}

#[derive(Debug)]
struct FixedWidthDataBlockBuilder {
    bits_per_value: u64,
    bytes_per_value: u64,
    values: Vec<u8>,
}

impl FixedWidthDataBlockBuilder {
    fn new(bits_per_value: u64, estimated_size_bytes: u64) -> Self {
        assert!(bits_per_value.is_multiple_of(8));
        Self {
            bits_per_value,
            bytes_per_value: bits_per_value / 8,
            values: Vec::with_capacity(estimated_size_bytes as usize),
        }
    }
}

impl DataBlockBuilderImpl for FixedWidthDataBlockBuilder {
    fn validate_append(&self, _data_block: &DataBlock, _selection: &Range<u64>) -> Result<()> {
        Ok(())
    }

    fn append_validated(&mut self, data_block: &DataBlock, selection: Range<u64>) -> Result<()> {
        let block = data_block.as_fixed_width_ref().unwrap();
        assert_eq!(self.bits_per_value, block.bits_per_value);
        let start = selection.start as usize * self.bytes_per_value as usize;
        let end = selection.end as usize * self.bytes_per_value as usize;
        self.values.extend_from_slice(&block.data[start..end]);
        Ok(())
    }

    fn finish(self: Box<Self>) -> DataBlock {
        let num_values = (self.values.len() / self.bytes_per_value as usize) as u64;
        DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(self.values),
            bits_per_value: self.bits_per_value,
            num_values,
            block_info: BlockInfo::new(),
        })
    }
}

#[derive(Debug)]
struct StructDataBlockBuilder {
    children: Vec<Box<dyn DataBlockBuilderImpl>>,
}

impl StructDataBlockBuilder {
    fn new(children: Vec<Box<dyn DataBlockBuilderImpl>>) -> Self {
        Self { children }
    }
}

impl DataBlockBuilderImpl for StructDataBlockBuilder {
    fn validate_append(&self, data_block: &DataBlock, selection: &Range<u64>) -> Result<()> {
        let data_block = data_block.as_struct_ref().unwrap();
        for i in 0..self.children.len() {
            self.children[i].validate_append(&data_block.children[i], selection)?;
        }
        Ok(())
    }

    fn append_validated(&mut self, data_block: &DataBlock, selection: Range<u64>) -> Result<()> {
        let data_block = data_block.as_struct_ref().unwrap();
        for i in 0..self.children.len() {
            self.children[i].append_validated(&data_block.children[i], selection.clone())?;
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> DataBlock {
        let mut children_data_block = Vec::new();
        for child in self.children {
            let child_data_block = child.finish();
            children_data_block.push(child_data_block);
        }
        DataBlock::Struct(StructDataBlock {
            children: children_data_block,
            block_info: BlockInfo::new(),
            validity: None,
        })
    }
}

#[derive(Debug, Default)]
struct AllNullDataBlockBuilder {
    num_values: u64,
}

impl DataBlockBuilderImpl for AllNullDataBlockBuilder {
    fn validate_append(&self, _data_block: &DataBlock, _selection: &Range<u64>) -> Result<()> {
        Ok(())
    }

    fn append_validated(&mut self, _data_block: &DataBlock, selection: Range<u64>) -> Result<()> {
        self.num_values += selection.end - selection.start;
        Ok(())
    }

    fn finish(self: Box<Self>) -> DataBlock {
        DataBlock::AllNull(AllNullDataBlock {
            num_values: self.num_values,
        })
    }
}

/// A data block to represent a fixed size list
#[derive(Debug, Clone)]
pub struct FixedSizeListBlock {
    /// The child data block
    pub child: Box<DataBlock>,
    /// The number of items in each list
    pub dimension: u64,
}

impl FixedSizeListBlock {
    pub fn num_values(&self) -> u64 {
        self.child.num_values() / self.dimension
    }

    /// Try to flatten a FixedSizeListBlock into a FixedWidthDataBlock
    ///
    /// Returns None if any children are nullable
    pub fn try_into_flat(self) -> Option<FixedWidthDataBlock> {
        match *self.child {
            // Cannot flatten a nullable child
            DataBlock::Nullable(_) => None,
            DataBlock::FixedSizeList(inner) => {
                let mut flat = inner.try_into_flat()?;
                flat.bits_per_value *= self.dimension;
                flat.num_values /= self.dimension;
                Some(flat)
            }
            DataBlock::FixedWidth(mut inner) => {
                inner.bits_per_value *= self.dimension;
                inner.num_values /= self.dimension;
                Some(inner)
            }
            _ => panic!(
                "Expected FixedSizeList or FixedWidth data block but found {:?}",
                self
            ),
        }
    }

    pub fn flatten_as_fixed(&mut self) -> FixedWidthDataBlock {
        match self.child.as_mut() {
            DataBlock::FixedSizeList(fsl) => fsl.flatten_as_fixed(),
            DataBlock::FixedWidth(fw) => fw.clone(),
            _ => panic!("Expected FixedSizeList or FixedWidth data block"),
        }
    }

    /// Convert a flattened values block into a FixedSizeListBlock
    pub fn from_flat(data: FixedWidthDataBlock, data_type: &DataType) -> DataBlock {
        match data_type {
            DataType::FixedSizeList(child_field, dimension) => {
                let mut data = data;
                data.bits_per_value /= *dimension as u64;
                data.num_values *= *dimension as u64;
                let child_data = Self::from_flat(data, child_field.data_type());
                DataBlock::FixedSizeList(Self {
                    child: Box::new(child_data),
                    dimension: *dimension as u64,
                })
            }
            // Base case, we've hit a non-list type
            _ => DataBlock::FixedWidth(data),
        }
    }

    fn into_arrow(self, data_type: DataType, _validate: bool) -> Result<ArrayData> {
        let num_values = self.num_values();
        let builder = match &data_type {
            DataType::FixedSizeList(child_field, _) => {
                let child_data = self
                    .child
                    .into_arrow_impl(child_field.data_type().clone(), true)?;
                ArrayDataBuilder::new(data_type)
                    .add_child_data(child_data)
                    .len(num_values as usize)
                    .null_count(0)
            }
            _ => panic!("Expected FixedSizeList data type and got {:?}", data_type),
        };
        Ok(builder.build()?)
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        self.child.into_buffers()
    }

    fn data_size(&self) -> u64 {
        self.child.data_size()
    }
}

#[derive(Debug)]
struct FixedSizeListBlockBuilder {
    inner: Box<dyn DataBlockBuilderImpl>,
    dimension: u64,
}

impl FixedSizeListBlockBuilder {
    fn new(inner: Box<dyn DataBlockBuilderImpl>, dimension: u64) -> Self {
        Self { inner, dimension }
    }
}

impl DataBlockBuilderImpl for FixedSizeListBlockBuilder {
    fn validate_append(&self, data_block: &DataBlock, selection: &Range<u64>) -> Result<()> {
        let selection = selection.start * self.dimension..selection.end * self.dimension;
        let fsl = data_block.as_fixed_size_list_ref().unwrap();
        self.inner.validate_append(fsl.child.as_ref(), &selection)
    }

    fn append_validated(&mut self, data_block: &DataBlock, selection: Range<u64>) -> Result<()> {
        let selection = selection.start * self.dimension..selection.end * self.dimension;
        let fsl = data_block.as_fixed_size_list_ref().unwrap();
        self.inner.append_validated(fsl.child.as_ref(), selection)
    }

    fn finish(self: Box<Self>) -> DataBlock {
        let inner_block = self.inner.finish();
        DataBlock::FixedSizeList(FixedSizeListBlock {
            child: Box::new(inner_block),
            dimension: self.dimension,
        })
    }
}

#[derive(Debug)]
struct NullableDataBlockBuilder {
    inner: Box<dyn DataBlockBuilderImpl>,
    validity: BooleanBufferBuilder,
}

impl NullableDataBlockBuilder {
    fn new(inner: Box<dyn DataBlockBuilderImpl>, estimated_size_bytes: usize) -> Self {
        Self {
            inner,
            validity: BooleanBufferBuilder::new(estimated_size_bytes * 8),
        }
    }
}

impl DataBlockBuilderImpl for NullableDataBlockBuilder {
    fn validate_append(&self, data_block: &DataBlock, selection: &Range<u64>) -> Result<()> {
        let nullable = data_block.as_nullable_ref().unwrap();
        self.inner
            .validate_append(nullable.data.as_ref(), selection)
    }

    fn append_validated(&mut self, data_block: &DataBlock, selection: Range<u64>) -> Result<()> {
        let nullable = data_block.as_nullable_ref().unwrap();
        self.inner
            .append_validated(nullable.data.as_ref(), selection.clone())?;
        let bool_buf = BooleanBuffer::new(
            nullable.nulls.clone().into_buffer(),
            selection.start as usize,
            (selection.end - selection.start) as usize,
        );
        self.validity.append_buffer(&bool_buf);
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> DataBlock {
        let inner_block = self.inner.finish();
        DataBlock::Nullable(NullableDataBlock {
            data: Box::new(inner_block),
            nulls: LanceBuffer::from(self.validity.finish().into_inner()),
            block_info: BlockInfo::new(),
        })
    }
}

/// A data block with no regular structure.  There is no available spot to attach
/// validity / repdef information and it cannot be converted to Arrow without being
/// decoded
#[derive(Debug, Clone)]
pub struct OpaqueBlock {
    pub buffers: Vec<LanceBuffer>,
    pub num_values: u64,
    pub block_info: BlockInfo,
}

impl OpaqueBlock {
    pub fn data_size(&self) -> u64 {
        self.buffers.iter().map(|b| b.len() as u64).sum()
    }
}

/// A data block for variable-width data (e.g. strings, packed rows, etc.)
#[derive(Debug, Clone)]
pub struct VariableWidthBlock {
    /// The data buffer
    pub data: LanceBuffer,
    /// The offsets buffer (contains num_values + 1 offsets)
    ///
    /// Offsets MUST start at 0
    pub offsets: LanceBuffer,
    /// The number of bits per offset
    pub bits_per_offset: u8,
    /// The number of values represented by this block
    pub num_values: u64,

    pub block_info: BlockInfo,
}

/// Proof that a [`VariableWidthBlock`] satisfies the Arrow layout contract for
/// its target data type (offsets buffer long enough, offsets monotonic and
/// within the data buffer, values valid UTF-8 where required).
///
/// Only [`VariableWidthBlock::validate_layout`] can construct it, which ties the
/// unchecked Arrow build below to an actual validation pass instead of a
/// caller-controlled flag.
struct ValidVariableWidthLayout;

impl VariableWidthBlock {
    fn append_error(&self, selection: &Range<u64>, detail: impl std::fmt::Display) -> Error {
        Error::corrupt_file_named(
            "variable width data block",
            format!(
                "cannot append offsets for selection {}..{}: {} (num_values: {}, \
                 bits_per_offset: {}, offsets buffer size: {} bytes, data buffer size: {} bytes)",
                selection.start,
                selection.end,
                detail,
                self.num_values,
                self.bits_per_offset,
                self.offsets.len(),
                self.data.len(),
            ),
        )
    }

    fn validate_offsets_for_append<T>(&self, selection: &Range<u64>) -> Result<()>
    where
        T: OffsetSizeTrait + bytemuck::Pod,
    {
        let expected_bits_per_offset = T::get_byte_width() as u8 * 8;
        if self.bits_per_offset != expected_bits_per_offset {
            return Err(self.append_error(
                selection,
                format!(
                    "expected {}-bit offsets but found {}-bit offsets",
                    expected_bits_per_offset, self.bits_per_offset
                ),
            ));
        }
        let offset_size = std::mem::size_of::<T>();
        if !self.offsets.len().is_multiple_of(offset_size) {
            return Err(self.append_error(
                selection,
                format!(
                    "offsets buffer length {} is not a multiple of the {}-byte offset width",
                    self.offsets.len(),
                    offset_size
                ),
            ));
        }
        if selection.start > selection.end || selection.end > self.num_values {
            return Err(
                self.append_error(selection, "selection is outside the block's value range")
            );
        }
        let selection_start = usize::try_from(selection.start)
            .map_err(|_| self.append_error(selection, "selection start does not fit in usize"))?;
        let selection_end = usize::try_from(selection.end)
            .map_err(|_| self.append_error(selection, "selection end does not fit in usize"))?;
        let offsets = self.offsets.borrow_to_typed_view::<T>();
        if selection_end >= offsets.len() {
            return Err(self.append_error(
                selection,
                format!(
                    "selection requires offset {} but the buffer holds {} offsets",
                    selection_end,
                    offsets.len()
                ),
            ));
        }
        let selected_offsets = &offsets[selection_start..=selection_end];
        if let Some(detail) =
            Self::offset_violation_detail(selected_offsets, self.data.len(), selection_start)
        {
            return Err(self.append_error(selection, detail));
        }
        Ok(())
    }

    // The offsets buffer comes straight from file bytes, so an unchecked build would
    // let a corrupt file smuggle out-of-bounds offsets into an Arrow array whose
    // consumers then read (or crash on) memory outside the data buffer.  This
    // boundary therefore always validates the layout, ignoring the optional
    // `validate` flag.  Lance validates the common layouts itself (a branchless
    // scan, measurably cheaper than Arrow's element-wise checked build) and only
    // falls back to Arrow's checked build for the cold cases.
    fn into_arrow(self, data_type: DataType, _validate: bool) -> Result<ArrayData> {
        let Some(expected_bits_per_offset) = Self::expected_bits_per_offset(&data_type) else {
            // Not an [offsets, bytes] layout we know how to prove; let Arrow
            // check it.
            return self.into_arrow_checked(data_type);
        };
        if self.bits_per_offset != expected_bits_per_offset {
            return Err(self.layout_error(
                &data_type,
                format!(
                    "expected {}-bit offsets but got {}-bit offsets",
                    expected_bits_per_offset, self.bits_per_offset
                ),
            ));
        }
        if self.num_values == 0 {
            // Cold path; Arrow handles the empty-offsets special cases.
            return self.into_arrow_checked(data_type);
        }
        let proof = self.validate_layout(&data_type)?;
        Ok(self.into_arrow_unchecked(data_type, proof))
    }

    /// The offset width Arrow mandates for `data_type`, or `None` if the type
    /// does not use the `[offsets, bytes]` layout this block represents.
    fn expected_bits_per_offset(data_type: &DataType) -> Option<u8> {
        match data_type {
            DataType::Binary | DataType::Utf8 => Some(32),
            DataType::LargeBinary | DataType::LargeUtf8 => Some(64),
            _ => None,
        }
    }

    fn layout_error(&self, data_type: &DataType, detail: impl std::fmt::Display) -> Error {
        Self::format_layout_error(
            data_type,
            detail,
            self.num_values,
            self.bits_per_offset,
            self.offsets.len(),
            self.data.len(),
        )
    }

    fn format_layout_error(
        data_type: &DataType,
        detail: impl std::fmt::Display,
        num_values: u64,
        bits_per_offset: u8,
        offsets_size: usize,
        data_size: usize,
    ) -> Error {
        Error::corrupt_file_named(
            "variable width data block",
            format!(
                "invalid variable-width layout for {}: {} (num_values: {}, bits_per_offset: {}, \
                 offsets buffer size: {} bytes, data buffer size: {} bytes)",
                data_type, detail, num_values, bits_per_offset, offsets_size, data_size,
            ),
        )
    }

    fn validate_layout(&self, data_type: &DataType) -> Result<ValidVariableWidthLayout> {
        let bytes_per_offset = (self.bits_per_offset / 8) as u64;
        let required_bytes = self
            .num_values
            .checked_add(1)
            .and_then(|num_offsets| num_offsets.checked_mul(bytes_per_offset))
            .ok_or_else(|| self.layout_error(data_type, "offsets buffer size overflows"))?;
        if (self.offsets.len() as u64) < required_bytes {
            return Err(self.layout_error(
                data_type,
                format!(
                    "offsets buffer must hold at least {} offsets ({} bytes)",
                    self.num_values + 1,
                    required_bytes
                ),
            ));
        }
        let validate_utf8 = matches!(data_type, DataType::Utf8 | DataType::LargeUtf8);
        match self.bits_per_offset {
            32 => self.validate_offsets_and_values::<i32>(data_type, validate_utf8),
            64 => self.validate_offsets_and_values::<i64>(data_type, validate_utf8),
            other => Err(self.layout_error(
                data_type,
                format!("unsupported offset width: {} bits", other),
            )),
        }
    }

    fn validate_offsets_and_values<T: ArrowNativeType + Ord>(
        &self,
        data_type: &DataType,
        validate_utf8: bool,
    ) -> Result<ValidVariableWidthLayout> {
        let num_offsets = self.num_values as usize + 1;
        // Slice before borrowing: the buffer may carry padding that is not a
        // multiple of the offset width.
        let offsets = self
            .offsets
            .slice_with_length(0, num_offsets * std::mem::size_of::<T>());
        let offsets = offsets.borrow_to_typed_slice::<T>();
        let offsets: &[T] = offsets.as_ref();
        let data = self.data.as_ref();

        if let Some(detail) = Self::offset_violation_detail(offsets, data.len(), 0) {
            return Err(self.layout_error(data_type, detail));
        }

        if validate_utf8 {
            let (first, last) = (offsets[0].as_usize(), offsets[num_offsets - 1].as_usize());
            let values = std::str::from_utf8(&data[first..last])
                .map_err(|utf8_err| self.layout_error(data_type, utf8_err))?;
            let mut on_char_boundaries = true;
            for &offset in offsets {
                on_char_boundaries &= values.is_char_boundary(offset.as_usize() - first);
            }
            if !on_char_boundaries {
                // Cold path: rescan to pinpoint the offending offset.
                let position = offsets
                    .iter()
                    .position(|offset| !values.is_char_boundary(offset.as_usize() - first))
                    .expect("the fast scan found a non-boundary offset");
                return Err(self.layout_error(
                    data_type,
                    format!("offset at position {position} splits a UTF-8 character"),
                ));
            }
        }

        Ok(ValidVariableWidthLayout)
    }

    fn offset_violation_detail<T: ArrowNativeType + Ord>(
        offsets: &[T],
        data_size: usize,
        position_base: usize,
    ) -> Option<String> {
        // A monotonic sequence with a non-negative first offset and an
        // in-bounds last offset is entirely within [0, data_size].  Keep this
        // valid path branchless so it vectorizes, and only rescan on failure.
        let mut is_monotonic = true;
        for window in offsets.windows(2) {
            is_monotonic &= window[0] <= window[1];
        }
        let first = offsets[0];
        let last = offsets[offsets.len() - 1];
        let bounds_ok =
            first >= T::usize_as(0) && last.to_usize().is_some_and(|last| last <= data_size);
        if is_monotonic && bounds_ok {
            return None;
        }

        for (relative_position, window) in offsets.windows(2).enumerate() {
            if window[0] > window[1] {
                let position = position_base + relative_position + 1;
                return Some(format!(
                    "non-monotonic offset at position {}: {:?} decreases from {:?}",
                    position, window[1], window[0]
                ));
            }
        }
        for (relative_position, offset) in offsets.iter().enumerate() {
            let position = position_base + relative_position;
            match offset.to_usize() {
                None => {
                    return Some(format!(
                        "offset at position {} is negative: {:?}",
                        position, offset
                    ));
                }
                Some(offset) if offset > data_size => {
                    return Some(format!(
                        "offset at position {} is out of bounds: {} > {}",
                        position, offset, data_size
                    ));
                }
                Some(_) => {}
            }
        }
        Some("offsets failed validation".to_string())
    }

    fn into_arrow_checked(self, data_type: DataType) -> Result<ArrayData> {
        let num_values = self.num_values;
        let bits_per_offset = self.bits_per_offset;
        let offsets_size = self.offsets.len();
        let data_size = self.data.len();
        let builder = self.into_arrow_builder(data_type.clone());
        builder.build().map_err(|arrow_err| {
            Self::format_layout_error(
                &data_type,
                arrow_err,
                num_values,
                bits_per_offset,
                offsets_size,
                data_size,
            )
        })
    }

    fn into_arrow_unchecked(
        self,
        data_type: DataType,
        _proof: ValidVariableWidthLayout,
    ) -> ArrayData {
        let builder = self.into_arrow_builder(data_type);
        // SAFETY: `_proof` witnesses that `validate_layout` proved this block
        // satisfies the Arrow layout contract for `data_type`.
        unsafe { builder.build_unchecked() }
    }

    fn into_arrow_builder(self, data_type: DataType) -> ArrayDataBuilder {
        let num_values = self.num_values;
        let data_buffer = self.data.into_buffer();
        let offsets_buffer = self.offsets.into_buffer();
        ArrayDataBuilder::new(data_type)
            .add_buffer(offsets_buffer)
            .add_buffer(data_buffer)
            .len(num_values as usize)
            .null_count(0)
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        vec![self.offsets, self.data]
    }

    pub fn offsets_as_block(&mut self) -> DataBlock {
        let offsets = self.offsets.clone();
        DataBlock::FixedWidth(FixedWidthDataBlock {
            data: offsets,
            bits_per_value: self.bits_per_offset as u64,
            num_values: self.num_values + 1,
            block_info: BlockInfo::new(),
        })
    }

    pub fn data_size(&self) -> u64 {
        (self.data.len() + self.offsets.len()) as u64
    }
}

/// A data block representing a struct
#[derive(Debug, Clone)]
pub struct StructDataBlock {
    /// The child arrays
    pub children: Vec<DataBlock>,
    pub block_info: BlockInfo,
    /// The validity bitmap for the struct (None means all valid)
    pub validity: Option<NullBuffer>,
}

impl StructDataBlock {
    fn into_arrow(self, data_type: DataType, _validate: bool) -> Result<ArrayData> {
        if let DataType::Struct(fields) = &data_type {
            let mut builder = ArrayDataBuilder::new(DataType::Struct(fields.clone()));
            let mut num_rows = 0;
            for (field, child) in fields.iter().zip(self.children) {
                let child_data = child.into_arrow_impl(field.data_type().clone(), true)?;
                num_rows = child_data.len();
                builder = builder.add_child_data(child_data);
            }

            // Apply validity if present
            let builder = if let Some(validity) = self.validity {
                let null_count = validity.null_count();
                builder
                    .null_bit_buffer(Some(validity.into_inner().into_inner()))
                    .null_count(null_count)
            } else {
                builder.null_count(0)
            };

            let builder = builder.len(num_rows);
            Ok(builder.build()?)
        } else {
            Err(Error::internal(format!(
                "Expected Struct, got {:?}",
                data_type
            )))
        }
    }

    fn remove_outer_validity(self) -> Self {
        Self {
            children: self
                .children
                .into_iter()
                .map(|c| c.remove_outer_validity())
                .collect(),
            block_info: self.block_info,
            validity: None, // Remove the validity
        }
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        self.children
            .into_iter()
            .flat_map(|c| c.into_buffers())
            .collect()
    }

    pub fn has_variable_width_child(&self) -> bool {
        self.children
            .iter()
            .any(|child| !matches!(child, DataBlock::FixedWidth(_)))
    }

    pub fn data_size(&self) -> u64 {
        self.children
            .iter()
            .map(|data_block| data_block.data_size())
            .sum()
    }
}

/// A data block for dictionary encoded data
#[derive(Debug, Clone)]
pub struct DictionaryDataBlock {
    /// The indices buffer
    pub indices: FixedWidthDataBlock,
    /// The dictionary itself
    pub dictionary: Box<DataBlock>,
}

impl DictionaryDataBlock {
    fn decode_helper<K: ArrowDictionaryKeyType>(self) -> Result<DataBlock> {
        // Handle empty batch - this can happen when decoding a range that contains
        // only empty/null lists, or when reading sparse data
        if self.indices.num_values == 0 {
            return Ok(DataBlock::AllNull(AllNullDataBlock { num_values: 0 }));
        }

        // assume the indices are uniformly distributed.
        let estimated_size_bytes = self.dictionary.data_size()
            * (self.indices.num_values + self.dictionary.num_values() - 1)
            / self.dictionary.num_values();
        let mut data_builder = DataBlockBuilder::with_capacity_estimate(estimated_size_bytes);

        let indices = self.indices.data.borrow_to_typed_slice::<K::Native>();
        let indices = indices.as_ref();

        let selections = indices.iter().map(|idx| {
            let idx = idx.to_usize().unwrap() as u64;
            idx..idx + 1
        });
        data_builder.append_ranges(&self.dictionary, selections)?;

        Ok(data_builder.finish())
    }

    pub fn decode(self) -> Result<DataBlock> {
        match self.indices.bits_per_value {
            8 => self.decode_helper::<UInt8Type>(),
            16 => self.decode_helper::<UInt16Type>(),
            32 => self.decode_helper::<UInt32Type>(),
            64 => self.decode_helper::<UInt64Type>(),
            _ => Err(lance_core::Error::internal(format!(
                "Unsupported dictionary index bit width: {} bits",
                self.indices.bits_per_value
            ))),
        }
    }

    fn into_arrow_dict(
        self,
        key_type: Box<DataType>,
        value_type: Box<DataType>,
        _validate: bool,
    ) -> Result<ArrayData> {
        let declared_key_bits = key_type.byte_width() as u64 * 8;
        if self.indices.bits_per_value == 32 && declared_key_bits < 32 {
            return self.narrow_widened_indices(key_type, value_type);
        }
        if self.indices.bits_per_value != declared_key_bits {
            return Err(lance_core::Error::corrupt_file_named(
                "dictionary",
                format!(
                    "dictionary indices use {} bits but the declared {} key type uses {} bits",
                    self.indices.bits_per_value, key_type, declared_key_bits
                ),
            ));
        }
        let indices_num_values = self.indices.num_values;
        let indices = self
            .indices
            .do_into_arrow((*key_type).clone(), indices_num_values, true)?;
        let dictionary = self
            .dictionary
            .into_arrow_impl((*value_type).clone(), true)?;

        let builder = indices
            .into_builder()
            .add_child_data(dictionary)
            .data_type(DataType::Dictionary(key_type, value_type));

        Ok(builder.build()?)
    }

    /// Converts a decoded chunk whose indices are 32 bits wide although the field
    /// declares a narrower key type.
    ///
    /// Before #8220 the 2.0 writer stored such pages when it appended a null slot
    /// past the last key the declared type can address, or when the dictionaries of
    /// the chunks buffered into one page did not fit that type together.  The 2.1
    /// writer still stores them in the second case.  The indices still point at the
    /// right values, so the chunk keeps only the entries it references, re-interned
    /// when they do not fit the key type.
    fn narrow_widened_indices(
        self,
        key_type: Box<DataType>,
        value_type: Box<DataType>,
    ) -> Result<ArrayData> {
        let num_indices = self.indices.num_values;
        let indices = UInt32Array::from(self.indices.do_into_arrow(
            DataType::UInt32,
            num_indices,
            true,
        )?);
        let values = make_array(
            self.dictionary
                .into_arrow_impl((*value_type).clone(), true)?,
        );
        let dictionary = DictionaryArray::<UInt32Type>::try_new(indices, values).map_err(|err| {
            lance_core::Error::corrupt_file_named(
                "dictionary",
                format!(
                    "dictionary indices use 32 bits but the declared {key_type} key type uses {} bits, and they do not address the dictionary: {err}",
                    key_type.byte_width() * 8
                ),
            )
        })?;
        let dictionary = garbage_collect_dictionary(&dictionary)?;
        let max_keys = max_index_val(&key_type) + 1;
        let declared_type = DataType::Dictionary(key_type.clone(), value_type);
        if dictionary.values().len() as u64 <= max_keys {
            // The referenced entries fit the declared key type, so only the keys are
            // narrowed.  A null stored as a dictionary value stays a value, as it does
            // on a page whose indices match the key type.
            return Ok(arrow_cast::cast(&dictionary, &declared_type)?.to_data());
        }
        // The referenced entries can repeat a value or include the null slot, so intern
        // them exactly.  This turns a null dictionary value into a null key.
        let interned = match arrow_cast::cast(dictionary.values(), &declared_type) {
            Ok(interned) => interned,
            Err(arrow_schema::ArrowError::DictionaryKeyOverflowError) => {
                return Err(lance_core::Error::not_supported(format!(
                    "this batch references {} dictionary entries, more than the {max_keys} distinct values that {key_type} keys can address; read fewer rows per batch",
                    dictionary.values().len()
                )));
            }
            Err(err) => return Err(err.into()),
        };
        let interned = interned.as_any_dictionary();
        let keys = arrow_select::take::take(interned.keys(), dictionary.keys(), None)?;
        Ok(keys
            .to_data()
            .into_builder()
            .data_type(declared_type)
            .add_child_data(interned.values().to_data())
            .build()?)
    }

    fn into_arrow(self, data_type: DataType, validate: bool) -> Result<ArrayData> {
        if let DataType::Dictionary(key_type, value_type) = data_type {
            self.into_arrow_dict(key_type, value_type, validate)
        } else {
            self.decode()?.into_arrow_impl(data_type, validate)
        }
    }

    fn into_buffers(self) -> Vec<LanceBuffer> {
        let mut buffers = self.indices.into_buffers();
        buffers.extend(self.dictionary.into_buffers());
        buffers
    }

    pub fn into_parts(self) -> (DataBlock, DataBlock) {
        (DataBlock::FixedWidth(self.indices), *self.dictionary)
    }

    pub fn from_parts(indices: FixedWidthDataBlock, dictionary: DataBlock) -> Self {
        Self {
            indices,
            dictionary: Box::new(dictionary),
        }
    }
}

/// Limits recovery batches only when a stored dictionary exceeds its declared key capacity.
pub(crate) fn dictionary_batch_limit(data_type: &DataType, dictionary_len: u64) -> Option<u64> {
    let DataType::Dictionary(key_type, _) = data_type else {
        return None;
    };
    let capacity = match key_type.as_ref() {
        DataType::Int8 => i8::MAX as u64 + 1,
        DataType::UInt8 => u8::MAX as u64 + 1,
        DataType::Int16 => i16::MAX as u64 + 1,
        DataType::UInt16 => u16::MAX as u64 + 1,
        _ => return None,
    };
    (dictionary_len > capacity).then_some(capacity)
}

/// A DataBlock is a collection of buffers that represents an "array" of data in very generic terms
///
/// The output of each decoder is a DataBlock.  Decoders can be chained together to transform
/// one DataBlock into a different kind of DataBlock.
///
/// The DataBlock is somewhere in between Arrow's ArrayData and Array and represents a physical
/// layout of the data.
///
/// A DataBlock can be converted into an Arrow ArrayData (and then Array) for a given array type.
/// For example, a FixedWidthDataBlock can be converted into any primitive type or a fixed size
/// list of a primitive type.  This is a zero-copy operation.
///
/// In addition, a DataBlock can be created from an Arrow array or arrays.  This is not a zero-copy
/// operation as some normalization may be required.
#[derive(Debug, Clone)]
pub enum DataBlock {
    Empty(),
    Constant(ConstantDataBlock),
    AllNull(AllNullDataBlock),
    Nullable(NullableDataBlock),
    FixedWidth(FixedWidthDataBlock),
    FixedSizeList(FixedSizeListBlock),
    VariableWidth(VariableWidthBlock),
    Opaque(OpaqueBlock),
    Struct(StructDataBlock),
    Dictionary(DictionaryDataBlock),
}

impl DataBlock {
    /// Convert self into an Arrow ArrayData with full layout validation.
    ///
    /// The `validate` argument is retained for API compatibility. Conversion is
    /// always validated because callers can construct data blocks directly.
    pub fn into_arrow(self, data_type: DataType, _validate: bool) -> Result<ArrayData> {
        self.into_arrow_impl(data_type, true)
    }

    fn into_arrow_impl(self, data_type: DataType, validate: bool) -> Result<ArrayData> {
        match self {
            Self::Empty() => Ok(new_empty_array(&data_type).to_data()),
            Self::Constant(inner) => inner.into_arrow(data_type, validate),
            Self::AllNull(inner) => inner.into_arrow(data_type, validate),
            Self::Nullable(inner) => inner.into_arrow(data_type, validate),
            Self::FixedWidth(inner) => inner.into_arrow(data_type, validate),
            Self::FixedSizeList(inner) => inner.into_arrow(data_type, validate),
            Self::VariableWidth(inner) => inner.into_arrow(data_type, validate),
            Self::Struct(inner) => inner.into_arrow(data_type, validate),
            Self::Dictionary(inner) => inner.into_arrow(data_type, validate),
            Self::Opaque(_) => Err(Error::internal(
                "Cannot convert OpaqueBlock to Arrow".to_string(),
            )),
        }
    }

    /// Convert the data block into a collection of buffers for serialization
    ///
    /// The order matters and will be used to reconstruct the data block at read time.
    pub fn into_buffers(self) -> Vec<LanceBuffer> {
        match self {
            Self::Empty() => Vec::default(),
            Self::Constant(inner) => inner.into_buffers(),
            Self::AllNull(inner) => inner.into_buffers(),
            Self::Nullable(inner) => inner.into_buffers(),
            Self::FixedWidth(inner) => inner.into_buffers(),
            Self::FixedSizeList(inner) => inner.into_buffers(),
            Self::VariableWidth(inner) => inner.into_buffers(),
            Self::Struct(inner) => inner.into_buffers(),
            Self::Dictionary(inner) => inner.into_buffers(),
            Self::Opaque(inner) => inner.buffers,
        }
    }

    /// Converts the data buffers into borrowed mode and clones the block
    ///
    /// This is a zero-copy operation but requires a mutable reference to self and, afterwards,
    /// all buffers will be in Borrowed mode.
    /// Try and clone the block
    ///
    /// This will fail if any buffers are in owned mode.  You can call borrow_and_clone() to
    /// ensure that all buffers are in borrowed mode before calling this method.
    pub fn try_clone(&self) -> Result<Self> {
        match self {
            Self::Empty() => Ok(Self::Empty()),
            Self::Constant(inner) => Ok(Self::Constant(inner.clone())),
            Self::AllNull(inner) => Ok(Self::AllNull(inner.clone())),
            Self::Nullable(inner) => Ok(Self::Nullable(inner.clone())),
            Self::FixedWidth(inner) => Ok(Self::FixedWidth(inner.clone())),
            Self::FixedSizeList(inner) => Ok(Self::FixedSizeList(inner.clone())),
            Self::VariableWidth(inner) => Ok(Self::VariableWidth(inner.clone())),
            Self::Struct(inner) => Ok(Self::Struct(inner.clone())),
            Self::Dictionary(inner) => Ok(Self::Dictionary(inner.clone())),
            Self::Opaque(inner) => Ok(Self::Opaque(inner.clone())),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Constant(_) => "Constant",
            Self::Empty() => "Empty",
            Self::AllNull(_) => "AllNull",
            Self::Nullable(_) => "Nullable",
            Self::FixedWidth(_) => "FixedWidth",
            Self::FixedSizeList(_) => "FixedSizeList",
            Self::VariableWidth(_) => "VariableWidth",
            Self::Struct(_) => "Struct",
            Self::Dictionary(_) => "Dictionary",
            Self::Opaque(_) => "Opaque",
        }
    }

    pub fn is_variable(&self) -> bool {
        match self {
            Self::Constant(_) => false,
            Self::Empty() => false,
            Self::AllNull(_) => false,
            Self::Nullable(nullable) => nullable.data.is_variable(),
            Self::FixedWidth(_) => false,
            Self::FixedSizeList(fsl) => fsl.child.is_variable(),
            Self::VariableWidth(_) => true,
            Self::Struct(strct) => strct.children.iter().any(|c| c.is_variable()),
            Self::Dictionary(_) => {
                todo!("is_variable for DictionaryDataBlock is not implemented yet")
            }
            Self::Opaque(_) => panic!("Does not make sense to ask if an Opaque block is variable"),
        }
    }

    pub fn is_nullable(&self) -> bool {
        match self {
            Self::AllNull(_) => true,
            Self::Nullable(_) => true,
            Self::FixedSizeList(fsl) => fsl.child.is_nullable(),
            Self::Struct(strct) => strct.children.iter().any(|c| c.is_nullable()),
            Self::Dictionary(_) => {
                todo!("is_nullable for DictionaryDataBlock is not implemented yet")
            }
            Self::Opaque(_) => panic!("Does not make sense to ask if an Opaque block is nullable"),
            _ => false,
        }
    }

    /// The number of values in the block
    ///
    /// This function does not recurse into child blocks.  If this is a FSL then it will
    /// be the number of lists and not the number of items.
    pub fn num_values(&self) -> u64 {
        match self {
            Self::Empty() => 0,
            Self::Constant(inner) => inner.num_values,
            Self::AllNull(inner) => inner.num_values,
            Self::Nullable(inner) => inner.data.num_values(),
            Self::FixedWidth(inner) => inner.num_values,
            Self::FixedSizeList(inner) => inner.num_values(),
            Self::VariableWidth(inner) => inner.num_values,
            Self::Struct(inner) => inner.children[0].num_values(),
            Self::Dictionary(inner) => inner.indices.num_values,
            Self::Opaque(inner) => inner.num_values,
        }
    }

    /// The number of items in a single row
    ///
    /// This is always 1 unless there are layers of FSL
    pub fn items_per_row(&self) -> u64 {
        match self {
            Self::Empty() => todo!(),     // Leave undefined until needed
            Self::Constant(_) => todo!(), // Leave undefined until needed
            Self::AllNull(_) => todo!(),  // Leave undefined until needed
            Self::Nullable(nullable) => nullable.data.items_per_row(),
            Self::FixedWidth(_) => 1,
            Self::FixedSizeList(fsl) => fsl.dimension * fsl.child.items_per_row(),
            Self::VariableWidth(_) => 1,
            Self::Struct(_) => todo!(), // Leave undefined until needed
            Self::Dictionary(_) => 1,
            Self::Opaque(_) => 1,
        }
    }

    /// The number of bytes in the data block (including any child blocks)
    pub fn data_size(&self) -> u64 {
        match self {
            Self::Empty() => 0,
            Self::Constant(inner) => inner.data_size(),
            Self::AllNull(_) => 0,
            Self::Nullable(inner) => inner.data_size(),
            Self::FixedWidth(inner) => inner.data_size(),
            Self::FixedSizeList(inner) => inner.data_size(),
            Self::VariableWidth(inner) => inner.data_size(),
            Self::Struct(inner) => inner.children.iter().map(|child| child.data_size()).sum(),
            Self::Dictionary(inner) => inner.indices.data_size() + inner.dictionary.data_size(),
            Self::Opaque(inner) => inner.data_size(),
        }
    }

    /// Removes any validity information from the block
    ///
    /// This does not filter the block (e.g. remove rows).  It only removes
    /// the validity bitmaps (if present).  Any garbage masked by null bits
    /// will now appear as proper values.
    ///
    /// If `recurse` is true, then this will also remove validity from any child blocks.
    pub fn remove_outer_validity(self) -> Self {
        match self {
            Self::AllNull(_) => panic!("Cannot remove validity on all-null data"),
            Self::Nullable(inner) => *inner.data,
            Self::Struct(inner) => Self::Struct(inner.remove_outer_validity()),
            other => other,
        }
    }

    fn make_builder(&self, estimated_size_bytes: u64) -> Box<dyn DataBlockBuilderImpl> {
        match self {
            Self::FixedWidth(inner) => {
                if inner.bits_per_value == 1 {
                    Box::new(BitmapDataBlockBuilder::new(estimated_size_bytes))
                } else {
                    Box::new(FixedWidthDataBlockBuilder::new(
                        inner.bits_per_value,
                        estimated_size_bytes,
                    ))
                }
            }
            Self::VariableWidth(inner) => {
                if inner.bits_per_offset == 32 {
                    Box::new(VariableWidthDataBlockBuilder::<i32>::new(
                        estimated_size_bytes,
                    ))
                } else if inner.bits_per_offset == 64 {
                    Box::new(VariableWidthDataBlockBuilder::<i64>::new(
                        estimated_size_bytes,
                    ))
                } else {
                    todo!()
                }
            }
            Self::FixedSizeList(inner) => {
                let inner_builder = inner.child.make_builder(estimated_size_bytes);
                Box::new(FixedSizeListBlockBuilder::new(
                    inner_builder,
                    inner.dimension,
                ))
            }
            Self::Nullable(nullable) => {
                // There's no easy way to know what percentage of the data is in the valiidty buffer
                // but 1/16th seems like a reasonable guess.
                let estimated_validity_size_bytes = estimated_size_bytes / 16;
                let inner_builder = nullable
                    .data
                    .make_builder(estimated_size_bytes - estimated_validity_size_bytes);
                Box::new(NullableDataBlockBuilder::new(
                    inner_builder,
                    estimated_validity_size_bytes as usize,
                ))
            }
            Self::Struct(struct_data_block) => {
                let num_children = struct_data_block.children.len();
                let per_child_estimate = if num_children == 0 {
                    0
                } else {
                    estimated_size_bytes / num_children as u64
                };
                let child_builders = struct_data_block
                    .children
                    .iter()
                    .map(|child| child.make_builder(per_child_estimate))
                    .collect();
                Box::new(StructDataBlockBuilder::new(child_builders))
            }
            Self::AllNull(_) => Box::new(AllNullDataBlockBuilder::default()),
            _ => todo!("make_builder for {:?}", self),
        }
    }
}

macro_rules! as_type {
    ($fn_name:ident, $inner:tt, $inner_type:ident) => {
        pub fn $fn_name(self) -> Option<$inner_type> {
            match self {
                Self::$inner(inner) => Some(inner),
                _ => None,
            }
        }
    };
}

macro_rules! as_type_ref {
    ($fn_name:ident, $inner:tt, $inner_type:ident) => {
        pub fn $fn_name(&self) -> Option<&$inner_type> {
            match self {
                Self::$inner(inner) => Some(inner),
                _ => None,
            }
        }
    };
}

macro_rules! as_type_ref_mut {
    ($fn_name:ident, $inner:tt, $inner_type:ident) => {
        pub fn $fn_name(&mut self) -> Option<&mut $inner_type> {
            match self {
                Self::$inner(inner) => Some(inner),
                _ => None,
            }
        }
    };
}

// Cast implementations
impl DataBlock {
    as_type!(as_all_null, AllNull, AllNullDataBlock);
    as_type!(as_nullable, Nullable, NullableDataBlock);
    as_type!(as_fixed_width, FixedWidth, FixedWidthDataBlock);
    as_type!(as_fixed_size_list, FixedSizeList, FixedSizeListBlock);
    as_type!(as_variable_width, VariableWidth, VariableWidthBlock);
    as_type!(as_struct, Struct, StructDataBlock);
    as_type!(as_dictionary, Dictionary, DictionaryDataBlock);
    as_type_ref!(as_all_null_ref, AllNull, AllNullDataBlock);
    as_type_ref!(as_nullable_ref, Nullable, NullableDataBlock);
    as_type_ref!(as_fixed_width_ref, FixedWidth, FixedWidthDataBlock);
    as_type_ref!(as_fixed_size_list_ref, FixedSizeList, FixedSizeListBlock);
    as_type_ref!(as_variable_width_ref, VariableWidth, VariableWidthBlock);
    as_type_ref!(as_struct_ref, Struct, StructDataBlock);
    as_type_ref!(as_dictionary_ref, Dictionary, DictionaryDataBlock);
    as_type_ref_mut!(as_all_null_ref_mut, AllNull, AllNullDataBlock);
    as_type_ref_mut!(as_nullable_ref_mut, Nullable, NullableDataBlock);
    as_type_ref_mut!(as_fixed_width_ref_mut, FixedWidth, FixedWidthDataBlock);
    as_type_ref_mut!(
        as_fixed_size_list_ref_mut,
        FixedSizeList,
        FixedSizeListBlock
    );
    as_type_ref_mut!(as_variable_width_ref_mut, VariableWidth, VariableWidthBlock);
    as_type_ref_mut!(as_struct_ref_mut, Struct, StructDataBlock);
    as_type_ref_mut!(as_dictionary_ref_mut, Dictionary, DictionaryDataBlock);
}

// Methods to convert from Arrow -> DataBlock

fn get_byte_range<T: ArrowNativeType>(offsets: &mut LanceBuffer) -> Range<usize> {
    let offsets = offsets.borrow_to_typed_slice::<T>();
    if offsets.as_ref().is_empty() {
        0..0
    } else {
        offsets.as_ref().first().unwrap().as_usize()..offsets.as_ref().last().unwrap().as_usize()
    }
}

// Given multiple offsets arrays [0, 5, 10], [0, 3, 7], etc. stitch
// them together to get [0, 5, 10, 13, 20, ...]
//
// Also returns the data range referenced by each offset array (may
// not be 0..len if there is slicing involved)
fn stitch_offsets<T: ArrowNativeType + std::ops::Add<Output = T> + std::ops::Sub<Output = T>>(
    offsets: Vec<LanceBuffer>,
) -> (LanceBuffer, Vec<Range<usize>>) {
    if offsets.is_empty() {
        return (LanceBuffer::empty(), Vec::default());
    }
    let len = offsets.iter().map(|b| b.len()).sum::<usize>();
    // Note: we are making a copy here, even if there is only one input, because we want to
    // normalize that input if it doesn't start with zero.  This could be micro-optimized out
    // if needed.
    let mut dest = Vec::with_capacity(len);
    let mut byte_ranges = Vec::with_capacity(offsets.len());

    // We insert one leading 0 before processing any of the inputs
    dest.push(T::from_usize(0).unwrap());

    for mut o in offsets.into_iter() {
        if !o.is_empty() {
            let last_offset = *dest.last().unwrap();
            let o = o.borrow_to_typed_slice::<T>();
            let start = *o.as_ref().first().unwrap();
            // First, we skip the first offset
            // Then, we subtract that first offset from each remaining offset
            //
            // This gives us a 0-based offset array (minus the leading 0)
            //
            // Then we add the last offset from the previous array to each offset
            // which shifts our offset array to the correct position
            //
            // For example, let's assume the last offset from the previous array
            // was 10 and we are given [13, 17, 22].  This means we have two values with
            // length 4 (17 - 13) and 5 (22 - 17).  The output from this step will be
            // [14, 19].  Combined with our last offset of 10, this gives us [10, 14, 19]
            // which is our same two values of length 4 and 5.
            dest.extend(o.as_ref()[1..].iter().map(|&x| x + last_offset - start));
        }
        byte_ranges.push(get_byte_range::<T>(&mut o));
    }
    (LanceBuffer::reinterpret_vec(dest), byte_ranges)
}

fn arrow_binary_to_data_block(
    arrays: &[ArrayRef],
    num_values: u64,
    bits_per_offset: u8,
) -> DataBlock {
    let data_vec = arrays.iter().map(|arr| arr.to_data()).collect::<Vec<_>>();
    arrow_binary_array_data_to_data_block(&data_vec, num_values, bits_per_offset)
}

fn arrow_binary_array_data_to_data_block(
    data_vec: &[ArrayData],
    num_values: u64,
    bits_per_offset: u8,
) -> DataBlock {
    let bytes_per_offset = bits_per_offset as usize / 8;
    let offsets = data_vec
        .iter()
        .map(|d| {
            LanceBuffer::from(d.buffers()[0].slice_with_length(
                d.offset() * bytes_per_offset,
                (d.len() + 1) * bytes_per_offset,
            ))
        })
        .collect::<Vec<_>>();
    let (offsets, data_ranges) = if bits_per_offset == 32 {
        stitch_offsets::<i32>(offsets)
    } else {
        stitch_offsets::<i64>(offsets)
    };
    let data = data_vec
        .iter()
        .zip(data_ranges)
        .map(|(d, byte_range)| {
            LanceBuffer::from(
                d.buffers()[1]
                    .slice_with_length(byte_range.start, byte_range.end - byte_range.start),
            )
        })
        .collect::<Vec<_>>();
    let data = LanceBuffer::concat_into_one(data);
    DataBlock::VariableWidth(VariableWidthBlock {
        data,
        offsets,
        bits_per_offset,
        num_values,
        block_info: BlockInfo::new(),
    })
}

fn encode_flat_data(arrays: &[ArrayRef], num_values: u64) -> LanceBuffer {
    let bytes_per_value = arrays[0].data_type().byte_width();
    let mut buffer = Vec::with_capacity(num_values as usize * bytes_per_value);
    for arr in arrays {
        let data = arr.to_data();
        buffer.extend_from_slice(data.buffers()[0].as_slice());
    }
    LanceBuffer::from(buffer)
}

fn do_encode_bitmap_data(bitmaps: &[BooleanBuffer], num_values: u64) -> LanceBuffer {
    let mut builder = BooleanBufferBuilder::new(num_values as usize);

    for buf in bitmaps {
        builder.append_buffer(buf);
    }

    let buffer = builder.finish().into_inner();
    LanceBuffer::from(buffer)
}

fn encode_bitmap_data(arrays: &[ArrayRef], num_values: u64) -> LanceBuffer {
    let bitmaps = arrays
        .iter()
        .map(|arr| arr.as_boolean().values().clone())
        .collect::<Vec<_>>();
    do_encode_bitmap_data(&bitmaps, num_values)
}

// Concatenate dictionary arrays.  This is a bit tricky because we might overflow the
// index type.  If we do, we need to upscale the indices to a larger type.
fn concat_dict_arrays(arrays: &[ArrayRef]) -> ArrayRef {
    let value_type = arrays[0].as_any_dictionary().values().data_type();
    let array_refs = arrays.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>();
    // Arrow's concat reuses a dictionary only when every array shares the same one.  Otherwise
    // it appends the dictionaries one after another, unless together they hold at least as
    // many entries as there are rows or more than the keys can address.  Arrays that each
    // carry their own copy of the same values, as the write queue's copies of one input do,
    // would repeat them once per array, yet only need their keys concatenated.
    if !shares_one_dictionary(&array_refs)
        && let Some(concatenated) = concat_keys_of_equal_dictionaries(&array_refs)
    {
        return concatenated;
    }
    match arrow_select::concat::concat(&array_refs) {
        Ok(array) => array,
        Err(arrow_schema::ArrowError::DictionaryKeyOverflowError) => {
            // Arrow's concat merges dictionaries with a lossy, fixed-size
            // interner.  Hash collisions can inflate the merged dictionary
            // beyond the declared key type and report overflow even though the
            // exact, deduplicated union still fits.  Rebuild the dictionary by
            // re-encoding the logical values, which deduplicates exactly.
            if let Some(reencoded) = reencode_dict_arrays(&array_refs) {
                return reencoded;
            }
            // The merged values genuinely span more indices than the declared
            // key type can address.  Widen so that the writer's key-type check
            // can reject the mismatch instead of persisting a file whose
            // physical index width contradicts the schema.
            //
            // Slow, but hopefully a corner case.  Optimize later
            let upscaled = array_refs
                .iter()
                .map(|arr| {
                    match arrow_cast::cast(
                        *arr,
                        &DataType::Dictionary(
                            Box::new(DataType::UInt32),
                            Box::new(value_type.clone()),
                        ),
                    ) {
                        Ok(arr) => arr,
                        Err(arrow_schema::ArrowError::DictionaryKeyOverflowError) => {
                            // Technically I think this means the input type was u64 already
                            unimplemented!("Dictionary arrays with more than 2^32 unique values")
                        }
                        err => err.unwrap(),
                    }
                })
                .collect::<Vec<_>>();
            let array_refs = upscaled.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>();
            // Can still fail if concat pushes over u32 boundary
            match arrow_select::concat::concat(&array_refs) {
                Ok(array) => array,
                Err(arrow_schema::ArrowError::DictionaryKeyOverflowError) => {
                    unimplemented!("Dictionary arrays with more than 2^32 unique values")
                }
                err => err.unwrap(),
            }
        }
        // Shouldn't be any other possible errors in concat
        err => err.unwrap(),
    }
}

/// Whether every dictionary array uses the very same values (the same buffers, not merely
/// equal ones), which is when Arrow's concat keeps a single copy of them.
fn shares_one_dictionary(arrays: &[&dyn Array]) -> bool {
    let first = arrays[0].as_any_dictionary().values().to_data();
    arrays[1..]
        .iter()
        .all(|array| array.as_any_dictionary().values().to_data().ptr_eq(&first))
}

/// Concatenates the keys of dictionary arrays whose dictionaries hold equal values, keeping
/// one copy of the dictionary.  Returns `None` when the dictionaries differ, or when Arrow's
/// concat merges them itself (they hold at least as many entries as there are rows, or more
/// than the keys can address), which keeps only the distinct entries the keys use.
fn concat_keys_of_equal_dictionaries(arrays: &[&dyn Array]) -> Option<ArrayRef> {
    let DataType::Dictionary(key_type, _) = arrays[0].data_type() else {
        return None;
    };
    let values = arrays[0].as_any_dictionary().values();
    let entries = values.len().saturating_mul(arrays.len());
    let rows = arrays.iter().map(|array| array.len()).sum::<usize>();
    if entries >= rows || entries as u64 > max_index_val(key_type) {
        return None;
    }
    let values_data = values.to_data();
    if !arrays[1..]
        .iter()
        .all(|array| array.as_any_dictionary().values().to_data() == values_data)
    {
        return None;
    }
    let keys = arrays
        .iter()
        .map(|array| array.as_any_dictionary().keys())
        .collect::<Vec<_>>();
    let keys = arrow_select::concat::concat(&keys).ok()?;
    make_dictionary_array(keys, values.clone())
}

/// Concatenates the arrays decoded from consecutive pages of one column.
///
/// Arrow merges dictionaries with a lossy interner, so a batch that spans pages
/// whose dictionaries fit the key type together can still report an overflow.
/// On overflow the dictionaries are merged exactly instead.  A null stored as a
/// dictionary value (the 2.0 layout) becomes a null key in the merged array.
pub(crate) fn concat_decoded_arrays(arrays: &[&dyn Array]) -> Result<ArrayRef> {
    match arrow_select::concat::concat(arrays) {
        Err(arrow_schema::ArrowError::DictionaryKeyOverflowError) => {
            let DataType::Dictionary(key_type, _) = arrays[0].data_type() else {
                return Err(arrow_schema::ArrowError::DictionaryKeyOverflowError.into());
            };
            reencode_dict_arrays(arrays).ok_or_else(|| {
                let entries = arrays
                    .iter()
                    .map(|array| array.as_any_dictionary().values().len())
                    .sum::<usize>();
                Error::not_supported(format!(
                    "could not merge the dictionaries of the {} pages in this batch ({entries} entries) into {key_type} keys, which address at most {} distinct values; read fewer rows per batch",
                    arrays.len(),
                    max_index_val(key_type) + 1
                ))
            })
        }
        result => Ok(result?),
    }
}

/// Rebuilds one dictionary array from `arrays` while preserving their declared
/// dictionary key type.
///
/// Only the dictionary *entries* are expanded, never the rows: the referenced
/// entries are deduplicated once and each array's keys are remapped into the
/// merged dictionary.  Cost is therefore proportional to the number of entries
/// plus one index per row, not to the value payload per row.  Returns `None`
/// when the deduplicated entries do not fit the declared key type or the value
/// type is not supported by the cast, letting the caller fall back to widening.
fn reencode_dict_arrays(array_refs: &[&dyn Array]) -> Option<ArrayRef> {
    let DataType::Dictionary(key_type, value_type) = array_refs[0].data_type() else {
        return None;
    };

    // Drop entries the keys never reference first.  A backing dictionary can
    // carry unused entries, and keeping them could overflow the declared key
    // type for a merge whose referenced values actually fit.
    let collected = array_refs
        .iter()
        .map(|array| {
            arrow_select::dictionary::garbage_collect_any_dictionary(array.as_any_dictionary()).ok()
        })
        .collect::<Option<Vec<_>>>()?;
    let array_refs = collected
        .iter()
        .map(|array| array.as_ref())
        .collect::<Vec<_>>();

    // Concatenate the dictionary entries, capturing where each array's entries
    // start in the combined array.
    let mut entries = Vec::with_capacity(array_refs.len());
    let mut entry_offsets = Vec::with_capacity(array_refs.len());
    let mut entry_count = 0usize;
    for array in &array_refs {
        let dictionary = array.as_any_dictionary_opt()?;
        entry_offsets.push(entry_count);
        entry_count += dictionary.values().len();
        entries.push(dictionary.values().clone());
    }
    let entry_refs = entries
        .iter()
        .map(|entry| entry.as_ref())
        .collect::<Vec<_>>();
    let entries = arrow_select::concat::concat(&entry_refs).ok()?;

    // Exact entry deduplication: casting the entries to a dictionary interns
    // them and yields the mapping from each original entry to its unique index.
    let entry_map = arrow_cast::cast(
        entries.as_ref(),
        &DataType::Dictionary(key_type.clone(), value_type.clone()),
    )
    .ok()?;
    let entry_map = entry_map.as_any_dictionary();
    let merged_entries = entry_map.values().clone();
    let mapping = entry_map.keys();

    // Remap every key through its array's slice of the entry mapping.  This
    // costs one index per row, never one value per row.
    let mut remapped_keys = Vec::with_capacity(array_refs.len());
    for (array, &offset) in array_refs.iter().zip(entry_offsets.iter()) {
        let dictionary = array.as_any_dictionary_opt()?;
        let entry_slice = mapping.slice(offset, dictionary.values().len());
        let remapped = arrow_select::take::take(
            &entry_slice,
            dictionary.keys(),
            Some(arrow_select::take::TakeOptions {
                check_bounds: false,
            }),
        )
        .ok()?;
        remapped_keys.push(remapped);
    }
    let remapped_refs = remapped_keys
        .iter()
        .map(|keys| keys.as_ref())
        .collect::<Vec<_>>();
    let keys = arrow_select::concat::concat(&remapped_refs).ok()?;

    make_dictionary_array(keys, merged_entries)
}

/// Assembles a [`DictionaryArray`](arrow_array::DictionaryArray) from remapped
/// keys and deduplicated values.  Returns `None` if the key array is not a
/// supported dictionary key type or the keys do not fit the values.
fn make_dictionary_array(keys: ArrayRef, values: ArrayRef) -> Option<ArrayRef> {
    macro_rules! build {
        ($key:ty) => {{
            let keys = keys.as_primitive_opt::<$key>()?.clone();
            arrow_array::DictionaryArray::<$key>::try_new(keys, values)
                .ok()
                .map(|dictionary| Arc::new(dictionary) as ArrayRef)
        }};
    }
    match keys.data_type() {
        DataType::Int8 => build!(Int8Type),
        DataType::Int16 => build!(Int16Type),
        DataType::Int32 => build!(Int32Type),
        DataType::Int64 => build!(Int64Type),
        DataType::UInt8 => build!(UInt8Type),
        DataType::UInt16 => build!(UInt16Type),
        DataType::UInt32 => build!(UInt32Type),
        DataType::UInt64 => build!(UInt64Type),
        _ => None,
    }
}

fn max_index_val(index_type: &DataType) -> u64 {
    match index_type {
        DataType::Int8 => i8::MAX as u64,
        DataType::Int16 => i16::MAX as u64,
        DataType::Int32 => i32::MAX as u64,
        DataType::Int64 => i64::MAX as u64,
        DataType::UInt8 => u8::MAX as u64,
        DataType::UInt16 => u16::MAX as u64,
        DataType::UInt32 => u32::MAX as u64,
        DataType::UInt64 => u64::MAX,
        _ => panic!("Invalid dictionary index type"),
    }
}

// If we get multiple dictionary arrays and they don't all have the same dictionary
// then we need to normalize the indices.  Otherwise we might have something like:
//
// First chunk ["hello", "foo"], [0, 0, 1, 1, 1]
// Second chunk ["bar", "world"], [0, 1, 0, 1, 1]
//
// If we simply encode as ["hello", "foo", "bar", "world"], [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
// then we will get the wrong answer because the dictionaries were not merged and the indices
// were not remapped.
//
// A simple way to do this today is to just concatenate all the arrays.  This is because
// arrow's dictionary concatenation function already has the logic to merge dictionaries.
//
// TODO: If the dictionaries differ, we can possibly do something cheaper than concatenating
//
// In addition, we want to normalize the representation of nulls.  The cheapest thing to
// do (space-wise) is to put the nulls in the dictionary.
fn arrow_dictionary_to_data_block(arrays: &[ArrayRef], validity: Option<NullBuffer>) -> DataBlock {
    let array = concat_dict_arrays(arrays);
    let array_dict = array.as_any_dictionary();
    let mut indices = array_dict.keys();
    let num_values = indices.len() as u64;
    let mut values = array_dict.values().clone();

    // TODO: Should we just always normalize indices to u32?  That would make logic simpler
    // and we're going to bitpack them soon anyways

    let indices_block = if let Some(validity) = validity {
        // If there is validity then we find the first invalid index in the dictionary values, inserting
        // a new value if we need to.  Then we change all indices to point to that value.  This way we
        // never need to store nullability of the indices.
        let mut first_invalid_index = None;
        if let Some(values_validity) = values.nulls() {
            first_invalid_index = (!values_validity.inner()).set_indices().next();
        }
        let first_invalid_index = first_invalid_index.unwrap_or_else(|| {
            let null_arr = new_null_array(values.data_type(), 1);
            values = arrow_select::concat::concat(&[values.as_ref(), null_arr.as_ref()]).unwrap();
            values.len() - 1
        });
        let max_index_val = max_index_val(indices.data_type());
        let upcast = if first_invalid_index as u64 > max_index_val {
            // Widen the index type when the null dictionary value cannot be addressed by the
            // declared key type, whether the value already existed or was appended above.
            if max_index_val >= u32::MAX as u64 {
                unimplemented!("Dictionary arrays with 2^32 unique value (or more) and a null")
            }
            Some(arrow_cast::cast(indices, &DataType::UInt32).unwrap())
        } else {
            None
        };
        if let Some(upcast) = upcast.as_ref() {
            indices = upcast;
        }
        // This can't fail since we already checked for fit
        let null_index_arr = arrow_cast::cast(
            &UInt64Array::from(vec![first_invalid_index as u64]),
            indices.data_type(),
        )
        .unwrap();

        let bytes_per_index = indices.data_type().byte_width();
        let bits_per_index = bytes_per_index as u64 * 8;

        let null_index_arr = null_index_arr.into_data();
        let null_index_bytes = &null_index_arr.buffers()[0];
        // Need to make a copy here since indices isn't mutable, could be avoided in theory
        let mut indices_bytes = indices.to_data().buffers()[0].to_vec();
        for invalid_idx in (!validity.inner()).set_indices() {
            indices_bytes[invalid_idx * bytes_per_index..(invalid_idx + 1) * bytes_per_index]
                .copy_from_slice(null_index_bytes.as_slice());
        }
        FixedWidthDataBlock {
            data: LanceBuffer::from(indices_bytes),
            bits_per_value: bits_per_index,
            num_values,
            block_info: BlockInfo::new(),
        }
    } else {
        FixedWidthDataBlock {
            data: LanceBuffer::from(indices.to_data().buffers()[0].clone()),
            bits_per_value: indices.data_type().byte_width() as u64 * 8,
            num_values,
            block_info: BlockInfo::new(),
        }
    };

    let items = DataBlock::from(values);
    DataBlock::Dictionary(DictionaryDataBlock {
        indices: indices_block,
        dictionary: Box::new(items),
    })
}

enum Nullability {
    None,
    All,
    Some(NullBuffer),
}

impl Nullability {
    fn to_option(&self) -> Option<NullBuffer> {
        match self {
            Self::Some(nulls) => Some(nulls.clone()),
            _ => None,
        }
    }
}

fn extract_nulls(arrays: &[ArrayRef], num_values: u64) -> Nullability {
    let mut has_nulls = false;
    let nulls_and_lens = arrays
        .iter()
        .map(|arr| {
            let nulls = arr.logical_nulls();
            has_nulls |= nulls.is_some();
            (nulls, arr.len())
        })
        .collect::<Vec<_>>();
    if !has_nulls {
        return Nullability::None;
    }
    let mut builder = BooleanBufferBuilder::new(num_values as usize);
    let mut num_nulls = 0;
    for (null, len) in nulls_and_lens {
        if let Some(null) = null {
            num_nulls += null.null_count();
            builder.append_buffer(&null.into_inner());
        } else {
            builder.append_n(len, true);
        }
    }
    if num_nulls == num_values as usize {
        Nullability::All
    } else {
        Nullability::Some(NullBuffer::new(builder.finish()))
    }
}

impl DataBlock {
    fn validate_variable_width_offsets<T: ArrowNativeType + Ord>(
        array_data: &ArrayData,
    ) -> std::result::Result<(), String> {
        if array_data.is_empty() && array_data.buffers()[0].is_empty() {
            return Ok(());
        }
        let offset_size = std::mem::size_of::<T>();
        let offset_start = array_data.offset() * offset_size;
        let offset_len = (array_data.len() + 1) * offset_size;
        let offset_buffer =
            LanceBuffer::from(array_data.buffers()[0].slice_with_length(offset_start, offset_len));
        let offsets = offset_buffer.borrow_to_typed_slice::<T>();
        VariableWidthBlock::offset_violation_detail(
            offsets.as_ref(),
            array_data.buffers()[1].len(),
            0,
        )
        .map_or(Ok(()), Err)
    }

    // `validate_full` also rescans UTF-8 contents and character boundaries on every flush.
    // Encoding only needs a complete monotonicity and bounds proof before slicing offsets.
    fn validate_variable_width_layouts(array_data: &ArrayData) -> std::result::Result<(), String> {
        match array_data.data_type() {
            DataType::Binary | DataType::Utf8 => {
                Self::validate_variable_width_offsets::<i32>(array_data)?;
            }
            DataType::LargeBinary | DataType::LargeUtf8 => {
                Self::validate_variable_width_offsets::<i64>(array_data)?;
            }
            _ => {}
        }
        for child_data in array_data.child_data() {
            Self::validate_variable_width_layouts(child_data)?;
        }
        Ok(())
    }

    fn validate_array_data(
        array_data: &ArrayData,
        field_name: &str,
        array_index: usize,
    ) -> Result<()> {
        let validation = array_data
            .validate()
            .map_err(|error| error.to_string())
            .and_then(|_| Self::validate_variable_width_layouts(array_data));
        validation.map_err(|error| {
            Error::invalid_input_source(
                format!(
                    "Invalid Arrow array for field '{}' at buffered array {}: {}",
                    field_name, array_index, error
                )
                .into(),
            )
        })
    }

    pub(crate) fn validate_arrays(arrays: &[ArrayRef], field_name: &str) -> Result<()> {
        for (array_index, array) in arrays.iter().enumerate() {
            Self::validate_array_data(&array.to_data(), field_name, array_index)?;
        }
        Ok(())
    }

    pub fn from_arrays(arrays: &[ArrayRef], num_values: u64) -> Self {
        Self::from_arrays_impl(arrays, num_values, true)
    }

    /// Rewrap decoded Arrow buffers without collecting encoder statistics.
    pub(crate) fn from_decoded_array(array: ArrayRef) -> Self {
        let len = array.len() as u64;
        Self::from_arrays_impl(&[array], len, false)
    }

    fn from_arrays_impl(arrays: &[ArrayRef], num_values: u64, encoding: bool) -> Self {
        if arrays.is_empty() || num_values == 0 {
            return Self::AllNull(AllNullDataBlock { num_values: 0 });
        }

        let data_type = arrays[0].data_type();
        let nulls = extract_nulls(arrays, num_values);

        if let Nullability::All = nulls {
            return Self::AllNull(AllNullDataBlock { num_values });
        }

        let mut encoded = match data_type {
            DataType::Binary | DataType::Utf8 => arrow_binary_to_data_block(arrays, num_values, 32),
            // View types have no Lance disk representation; cast to the classic offset layout.
            DataType::Utf8View => {
                let casted: Vec<ArrayRef> = arrays
                    .iter()
                    .map(|a| {
                        arrow_cast::cast(a.as_ref(), &DataType::Utf8)
                            .expect("Utf8View to Utf8 cast is always valid")
                    })
                    .collect();
                arrow_binary_to_data_block(&casted, num_values, 32)
            }
            DataType::BinaryView => {
                let casted: Vec<ArrayRef> = arrays
                    .iter()
                    .map(|a| {
                        arrow_cast::cast(a.as_ref(), &DataType::Binary)
                            .expect("BinaryView to Binary cast is always valid")
                    })
                    .collect();
                arrow_binary_to_data_block(&casted, num_values, 32)
            }
            DataType::LargeBinary | DataType::LargeUtf8 => {
                arrow_binary_to_data_block(arrays, num_values, 64)
            }
            DataType::Boolean => {
                let data = encode_bitmap_data(arrays, num_values);
                Self::FixedWidth(FixedWidthDataBlock {
                    data,
                    bits_per_value: 1,
                    num_values,
                    block_info: BlockInfo::new(),
                })
            }
            DataType::Date32
            | DataType::Date64
            | DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Duration(_)
            | DataType::FixedSizeBinary(_)
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Int8
            | DataType::Interval(_)
            | DataType::Time32(_)
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::UInt8 => {
                let data = if !encoding && arrays.len() == 1 {
                    let data = arrays[0].to_data();
                    let width = data_type.byte_width();
                    LanceBuffer::from(
                        data.buffers()[0]
                            .slice_with_length(data.offset() * width, num_values as usize * width),
                    )
                } else {
                    encode_flat_data(arrays, num_values)
                };
                Self::FixedWidth(FixedWidthDataBlock {
                    data,
                    bits_per_value: data_type.byte_width() as u64 * 8,
                    num_values,
                    block_info: BlockInfo::new(),
                })
            }
            DataType::Null => Self::AllNull(AllNullDataBlock { num_values }),
            DataType::Dictionary(_, _) => arrow_dictionary_to_data_block(arrays, nulls.to_option()),
            DataType::Struct(fields) => {
                let structs = arrays.iter().map(|arr| arr.as_struct()).collect::<Vec<_>>();
                let mut children = Vec::with_capacity(fields.len());
                for child_idx in 0..fields.len() {
                    let child_vec = structs
                        .iter()
                        .map(|s| s.column(child_idx).clone())
                        .collect::<Vec<_>>();
                    children.push(Self::from_arrays_impl(&child_vec, num_values, encoding));
                }

                // Extract validity for the struct array
                let validity = match &nulls {
                    Nullability::None => None,
                    Nullability::Some(null_buffer) => Some(null_buffer.clone()),
                    Nullability::All => unreachable!("Should have returned AllNull earlier"),
                };

                Self::Struct(StructDataBlock {
                    children,
                    block_info: BlockInfo::default(),
                    validity,
                })
            }
            DataType::FixedSizeList(_, dim) => {
                let children = arrays
                    .iter()
                    .map(|arr| arr.as_fixed_size_list().values().clone())
                    .collect::<Vec<_>>();
                let child_block =
                    Self::from_arrays_impl(&children, num_values * *dim as u64, encoding);
                Self::FixedSizeList(FixedSizeListBlock {
                    child: Box::new(child_block),
                    dimension: *dim as u64,
                })
            }
            DataType::LargeList(_)
            | DataType::List(_)
            | DataType::ListView(_)
            | DataType::LargeListView(_)
            | DataType::Map(_, _)
            | DataType::RunEndEncoded(_, _)
            | DataType::Union(_, _) => {
                panic!(
                    "Field with data type {} cannot be converted to data block",
                    data_type
                )
            }
        };

        // compute statistics
        if encoding {
            encoded.compute_stat();
        }

        if !matches!(data_type, DataType::Dictionary(_, _)) {
            match nulls {
                Nullability::None => encoded,
                Nullability::Some(nulls) => Self::Nullable(NullableDataBlock {
                    data: Box::new(encoded),
                    nulls: LanceBuffer::from(nulls.into_inner().into_inner()),
                    block_info: BlockInfo::new(),
                }),
                _ => unreachable!(),
            }
        } else {
            // Dictionaries already insert the nulls into the dictionary items
            encoded
        }
    }

    pub fn from_array<T: Array + 'static>(array: T) -> Self {
        let num_values = array.len();
        Self::from_arrays(&[Arc::new(array)], num_values as u64)
    }
}

impl From<ArrayRef> for DataBlock {
    fn from(array: ArrayRef) -> Self {
        let num_values = array.len() as u64;
        Self::from_arrays(&[array], num_values)
    }
}

trait DataBlockBuilderImpl: std::fmt::Debug {
    fn validate_append(&self, data_block: &DataBlock, selection: &Range<u64>) -> Result<()>;

    fn append_validated(&mut self, data_block: &DataBlock, selection: Range<u64>) -> Result<()>;

    fn append(&mut self, data_block: &DataBlock, selection: Range<u64>) -> Result<()> {
        self.validate_append(data_block, &selection)?;
        self.append_validated(data_block, selection)
    }

    fn finish(self: Box<Self>) -> DataBlock;
}

#[derive(Debug)]
pub struct DataBlockBuilder {
    estimated_size_bytes: u64,
    builder: Option<Box<dyn DataBlockBuilderImpl>>,
}

impl DataBlockBuilder {
    pub fn with_capacity_estimate(estimated_size_bytes: u64) -> Self {
        Self {
            estimated_size_bytes,
            builder: None,
        }
    }

    fn get_builder(&mut self, block: &DataBlock) -> &mut dyn DataBlockBuilderImpl {
        if self.builder.is_none() {
            self.builder = Some(block.make_builder(self.estimated_size_bytes));
        }
        self.builder.as_mut().unwrap().as_mut()
    }

    pub fn append(&mut self, data_block: &DataBlock, selection: Range<u64>) -> Result<()> {
        self.get_builder(data_block).append(data_block, selection)
    }

    fn append_ranges(
        &mut self,
        data_block: &DataBlock,
        selections: impl IntoIterator<Item = Range<u64>>,
    ) -> Result<()> {
        let full_selection = 0..data_block.num_values();
        let builder = self.get_builder(data_block);
        builder.validate_append(data_block, &full_selection)?;
        for selection in selections {
            if selection.start > selection.end || selection.end > full_selection.end {
                return Err(Error::corrupt_file_named(
                    "data block",
                    format!(
                        "cannot append selection {}..{} from a block with {} values",
                        selection.start, selection.end, full_selection.end
                    ),
                ));
            }
            builder.append_validated(data_block, selection)?;
        }
        Ok(())
    }

    pub fn finish(self) -> DataBlock {
        let builder = self.builder.expect("DataBlockBuilder didn't see any data");
        builder.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BinaryArray, BinaryViewArray, DictionaryArray, Int8Array, LargeBinaryArray,
        LargeStringArray, StringArray, StringViewArray, UInt8Array, UInt16Array,
        cast::AsArray,
        make_array, new_null_array,
        types::{Int8Type, Int32Type, UInt16Type},
    };
    use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
    use arrow_data::ArrayData;

    use arrow_schema::{DataType, Field, Fields};
    use lance_core::Error;
    use lance_datagen::{ArrayGeneratorExt, DEFAULT_SEED, RowCount, array};
    use rand::SeedableRng;
    use rstest::rstest;

    use crate::buffer::LanceBuffer;

    use super::{
        AllNullDataBlock, BlockInfo, DataBlock, DataBlockBuilder, DictionaryDataBlock,
        FixedWidthDataBlock, VariableWidthBlock, concat_decoded_arrays, reencode_dict_arrays,
    };

    use arrow_array::Array;

    #[test]
    fn test_sliced_to_data_block() {
        let ints = UInt16Array::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let ints = ints.slice(2, 4);
        let data = DataBlock::from_array(ints);

        let fixed_data = data.as_fixed_width().unwrap();
        assert_eq!(fixed_data.num_values, 4);
        assert_eq!(fixed_data.data.len(), 8);

        let nullable_ints =
            UInt16Array::from(vec![Some(0), None, Some(2), None, Some(4), None, Some(6)]);
        let nullable_ints = nullable_ints.slice(1, 3);
        let data = DataBlock::from_array(nullable_ints);

        let nullable = data.as_nullable().unwrap();
        assert_eq!(nullable.nulls, LanceBuffer::from(vec![0b00000010]));
    }

    #[test]
    fn test_string_to_data_block() {
        // Converting string arrays that contain nulls to DataBlock
        let strings1 = StringArray::from(vec![Some("hello"), None, Some("world")]);
        let strings2 = StringArray::from(vec![Some("a"), Some("b")]);
        let strings3 = StringArray::from(vec![Option::<&'static str>::None, None]);

        let arrays = &[strings1, strings2, strings3]
            .iter()
            .map(|arr| Arc::new(arr.clone()) as ArrayRef)
            .collect::<Vec<_>>();

        let block = DataBlock::from_arrays(arrays, 7);

        assert_eq!(block.num_values(), 7);
        let block = block.as_nullable().unwrap();

        assert_eq!(block.nulls, LanceBuffer::from(vec![0b00011101]));

        let data = block.data.as_variable_width().unwrap();
        assert_eq!(
            data.offsets,
            LanceBuffer::reinterpret_vec(vec![0, 5, 5, 10, 11, 12, 12, 12])
        );

        assert_eq!(data.data, LanceBuffer::copy_slice(b"helloworldab"));

        // Converting string arrays that do not contain nulls to DataBlock
        let strings1 = StringArray::from(vec![Some("a"), Some("bc")]);
        let strings2 = StringArray::from(vec![Some("def")]);

        let arrays = &[strings1, strings2]
            .iter()
            .map(|arr| Arc::new(arr.clone()) as ArrayRef)
            .collect::<Vec<_>>();

        let block = DataBlock::from_arrays(arrays, 3);

        assert_eq!(block.num_values(), 3);
        // Should be no nullable wrapper
        let data = block.as_variable_width().unwrap();
        assert_eq!(data.offsets, LanceBuffer::reinterpret_vec(vec![0, 1, 3, 6]));
        assert_eq!(data.data, LanceBuffer::copy_slice(b"abcdef"));
    }

    #[test]
    fn test_string_view_to_data_block() {
        let views1 = StringViewArray::from(vec![Some("hello"), None, Some("world")]);
        let views2 = StringViewArray::from(vec![Some("a"), Some("b")]);
        let views3 = StringViewArray::from(vec![Option::<&'static str>::None, None]);

        let arrays = &[views1, views2, views3]
            .iter()
            .map(|arr| Arc::new(arr.clone()) as ArrayRef)
            .collect::<Vec<_>>();

        let block = DataBlock::from_arrays(arrays, 7);

        assert_eq!(block.num_values(), 7);
        let block = block.as_nullable().unwrap();
        assert_eq!(block.nulls, LanceBuffer::from(vec![0b00011101]));
        let data = block.data.as_variable_width().unwrap();
        assert_eq!(
            data.offsets,
            LanceBuffer::reinterpret_vec(vec![0, 5, 5, 10, 11, 12, 12, 12])
        );
        assert_eq!(data.data, LanceBuffer::copy_slice(b"helloworldab"));

        let views1 = StringViewArray::from(vec![Some("a"), Some("bc")]);
        let views2 = StringViewArray::from(vec![Some("def")]);

        let arrays = &[views1, views2]
            .iter()
            .map(|arr| Arc::new(arr.clone()) as ArrayRef)
            .collect::<Vec<_>>();

        let block = DataBlock::from_arrays(arrays, 3);

        assert_eq!(block.num_values(), 3);
        let data = block.as_variable_width().unwrap();
        assert_eq!(data.offsets, LanceBuffer::reinterpret_vec(vec![0, 1, 3, 6]));
        assert_eq!(data.data, LanceBuffer::copy_slice(b"abcdef"));
    }

    #[test]
    fn test_binary_view_to_data_block() {
        let arr: ArrayRef = Arc::new(BinaryViewArray::from(vec![
            Some(b"foo".as_slice()),
            None,
            Some(b"bar".as_slice()),
        ]));
        let block = DataBlock::from_arrays(&[arr], 3);
        let block = block.as_nullable().unwrap();
        let data = block.data.as_variable_width().unwrap();
        assert_eq!(data.data, LanceBuffer::copy_slice(b"foobar"));
    }

    #[test]
    fn test_string_sliced() {
        let check = |arr: Vec<StringArray>, expected_off: Vec<i32>, expected_data: &[u8]| {
            let arrs = arr
                .into_iter()
                .map(|a| Arc::new(a) as ArrayRef)
                .collect::<Vec<_>>();
            let num_rows = arrs.iter().map(|a| a.len()).sum::<usize>() as u64;
            let data = DataBlock::from_arrays(&arrs, num_rows);

            assert_eq!(data.num_values(), num_rows);

            let data = data.as_variable_width().unwrap();
            assert_eq!(data.offsets, LanceBuffer::reinterpret_vec(expected_off));
            assert_eq!(data.data, LanceBuffer::copy_slice(expected_data));
        };

        let string = StringArray::from(vec![Some("hello"), Some("world")]);
        check(vec![string.slice(1, 1)], vec![0, 5], b"world");
        check(vec![string.slice(0, 1)], vec![0, 5], b"hello");
        check(
            vec![string.slice(0, 1), string.slice(1, 1)],
            vec![0, 5, 10],
            b"helloworld",
        );

        let string2 = StringArray::from(vec![Some("foo"), Some("bar")]);
        check(
            vec![string.slice(0, 1), string2.slice(0, 1)],
            vec![0, 5, 8],
            b"hellofoo",
        );
    }

    #[rstest]
    #[case::utf8(
        DataType::Utf8,
        Buffer::from_slice_ref([0_i32, 5, 10]),
        32,
        LanceBuffer::reinterpret_vec(vec![0_i32, 5])
    )]
    #[case::large_utf8(
        DataType::LargeUtf8,
        Buffer::from_slice_ref([0_i64, 5, 10]),
        64,
        LanceBuffer::reinterpret_vec(vec![0_i64, 5])
    )]
    fn test_variable_width_array_data_offset(
        #[case] data_type: DataType,
        #[case] offsets: Buffer,
        #[case] bits_per_offset: u8,
        #[case] expected_offsets: LanceBuffer,
    ) {
        let array_data = ArrayData::builder(data_type)
            .len(1)
            .offset(1)
            .add_buffer(offsets)
            .add_buffer(Buffer::from(b"helloworld"))
            .build()
            .unwrap();

        DataBlock::validate_array_data(&array_data, "text", 0).unwrap();
        let data = super::arrow_binary_array_data_to_data_block(&[array_data], 1, bits_per_offset);

        let data = data.as_variable_width().unwrap();
        assert_eq!(data.offsets, expected_offsets);
        assert_eq!(data.data, LanceBuffer::copy_slice(b"world"));
    }

    #[rstest]
    #[case::utf8(DataType::Utf8, Buffer::from_slice_ref([0_i32, -1]))]
    #[case::large_utf8(DataType::LargeUtf8, Buffer::from_slice_ref([0_i64, -1]))]
    fn test_invalid_string_offsets_rejected_before_encoding(
        #[case] data_type: DataType,
        #[case] offsets: Buffer,
    ) {
        let array_data = unsafe {
            ArrayData::builder(data_type)
                .len(1)
                .add_buffer(offsets)
                .add_buffer(Buffer::from(b""))
                .build_unchecked()
        };

        let error = DataBlock::validate_array_data(&array_data, "text", 0).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        let message = error.to_string();
        assert!(
            message.contains("field 'text'"),
            "unexpected message: {message}"
        );
        assert!(
            message.contains("offset[1] (-1)"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn test_large() {
        let arr = LargeBinaryArray::from_vec(vec![b"hello", b"world"]);
        let data = DataBlock::from_array(arr);

        assert_eq!(data.num_values(), 2);
        let data = data.as_variable_width().unwrap();
        assert_eq!(data.bits_per_offset, 64);
        assert_eq!(data.num_values, 2);
        assert_eq!(data.data, LanceBuffer::copy_slice(b"helloworld"));
        assert_eq!(
            data.offsets,
            LanceBuffer::reinterpret_vec(vec![0_u64, 5, 10])
        );
    }

    #[test]
    fn test_dictionary_indices_normalized() {
        let arr1 = DictionaryArray::<Int8Type>::from_iter([Some("a"), Some("a"), Some("b")]);
        let arr2 = DictionaryArray::<Int8Type>::from_iter([Some("b"), Some("c")]);

        let data = DataBlock::from_arrays(&[Arc::new(arr1), Arc::new(arr2)], 5);

        assert_eq!(data.num_values(), 5);
        let data = data.as_dictionary().unwrap();
        let indices = data.indices;
        assert_eq!(indices.bits_per_value, 8);
        assert_eq!(indices.num_values, 5);
        assert_eq!(
            indices.data,
            // You might expect 0, 0, 1, 1, 2 but it seems that arrow's dictionary concat does
            // not actually collapse dictionaries.  This is an arrow problem however, and we don't
            // need to fix it here.
            LanceBuffer::reinterpret_vec::<i8>(vec![0, 0, 1, 2, 3])
        );

        let items = data.dictionary.as_variable_width().unwrap();
        assert_eq!(items.bits_per_offset, 32);
        assert_eq!(items.num_values, 4);
        assert_eq!(items.data, LanceBuffer::copy_slice(b"abbc"));
        assert_eq!(
            items.offsets,
            LanceBuffer::reinterpret_vec(vec![0, 1, 2, 3, 4],)
        );
    }

    #[test]
    fn test_dictionary_copies_of_one_dictionary_keep_one_copy() {
        // Two batches with equal but separately allocated dictionaries
        let arr1 = DictionaryArray::<Int8Type>::from_iter([Some("a"), Some("b"), Some("a")]);
        let arr2 = DictionaryArray::<Int8Type>::from_iter([Some("a"), Some("b")]);

        let data = DataBlock::from_arrays(&[Arc::new(arr1), Arc::new(arr2)], 5);

        let data = data.as_dictionary().unwrap();
        assert_eq!(
            data.indices.data,
            LanceBuffer::reinterpret_vec::<i8>(vec![0, 1, 0, 0, 1])
        );
        let items = data.dictionary.as_variable_width().unwrap();
        assert_eq!(items.data, LanceBuffer::copy_slice(b"ab"));

        // Where Arrow merges the dictionaries itself (128 entries for 128 rows) it also
        // collapses repeated values, which keeping one copy would not
        let make = || -> ArrayRef {
            let values = StringArray::from(vec!["same"; 64]);
            let keys = Int8Array::from((0..64).collect::<Vec<i8>>());
            Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
        };
        let data = DataBlock::from_arrays(&[make(), make()], 128);
        assert_eq!(data.as_dictionary().unwrap().dictionary.num_values(), 1);
    }

    fn all_int8_dictionary() -> DictionaryArray<Int8Type> {
        let values: Vec<u16> = (0..127u16).collect();
        let keys: Vec<i8> = (0..127)
            .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % 127) as i8)
            .collect();
        DictionaryArray::<Int8Type>::try_new(
            Int8Array::from(keys),
            Arc::new(UInt16Array::from(values)),
        )
        .unwrap()
    }

    #[test]
    fn test_dictionary_merge_keeps_declared_key_type() {
        // Two identical Dictionary(Int8, UInt16) chunks holding every value an
        // Int8 key can address.  Arrow's concat merges dictionaries with a
        // lossy interner that can over-count on hash collisions, which used to
        // widen the merged indices to UInt32 even though the exact union (127
        // values) fits.  The block must keep the declared key type.
        let a = Arc::new(all_int8_dictionary()) as ArrayRef;
        let b = Arc::new(all_int8_dictionary()) as ArrayRef;
        let data = DataBlock::from_arrays(&[a, b], 254);

        let dict = data.as_dictionary().unwrap();
        assert_eq!(dict.indices.bits_per_value, 8);
        assert_eq!(dict.dictionary.num_values(), 127);
    }

    #[test]
    fn test_dictionary_merge_keeps_declared_key_type_utf8() {
        // Same shape as the integer case but through the byte-value entry path:
        // the entries must be deduplicated without expanding them per row.
        let make = || -> ArrayRef {
            let values = StringArray::from(
                (0..127)
                    .map(|i| format!("value_{i:03}"))
                    .collect::<Vec<_>>(),
            );
            let keys = Int8Array::from(
                (0..127)
                    .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % 127) as i8)
                    .collect::<Vec<_>>(),
            );
            Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
        };
        let data = DataBlock::from_arrays(&[make(), make()], 254);

        let dict = data.as_dictionary().unwrap();
        assert_eq!(dict.indices.bits_per_value, 8);
        assert_eq!(dict.dictionary.num_values(), 127);
    }

    #[test]
    fn test_dictionary_merge_ignores_unused_entries() {
        // Each batch references only 0..127 but carries one extra entry the keys
        // never use.  The referenced union (127) fits Int8; the unused entries
        // must not force a wider key type.
        let make = |unused: u16| -> ArrayRef {
            let values = UInt16Array::from((0..127u16).chain([unused]).collect::<Vec<_>>());
            let keys = Int8Array::from(
                (0..127)
                    .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % 127) as i8)
                    .collect::<Vec<_>>(),
            );
            Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap())
        };
        let data = DataBlock::from_arrays(&[make(1000), make(1001)], 254);

        let dict = data.as_dictionary().unwrap();
        assert_eq!(dict.indices.bits_per_value, 8);
        assert_eq!(dict.dictionary.num_values(), 127);
    }

    fn uint16_dictionary(values: Vec<u16>, keys: Vec<Option<i8>>) -> ArrayRef {
        Arc::new(
            DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(keys),
                Arc::new(UInt16Array::from(values)),
            )
            .unwrap(),
        )
    }

    fn decoded_values(array: &dyn Array) -> Vec<Option<u16>> {
        arrow_cast::cast(array, &DataType::UInt16)
            .unwrap()
            .as_primitive::<UInt16Type>()
            .iter()
            .collect()
    }

    #[test]
    fn test_reencode_dict_arrays_remaps_each_array() {
        // The dictionaries overlap and list their values in different orders,
        // so every array's keys must be remapped through its own entries.
        let a = uint16_dictionary(vec![10, 20, 30], vec![Some(2), Some(0), None, Some(1)]);
        let b = uint16_dictionary(vec![30, 40, 10], vec![Some(0), Some(1), Some(2), None]);

        let merged = reencode_dict_arrays(&[a.as_ref(), b.as_ref()]).unwrap();

        assert_eq!(merged.data_type(), a.data_type());
        assert_eq!(merged.as_any_dictionary().values().len(), 4);
        assert_eq!(
            decoded_values(merged.as_ref()),
            [
                Some(30),
                Some(10),
                None,
                Some(20),
                Some(30),
                Some(40),
                Some(10),
                None
            ]
        );
    }

    #[test]
    fn test_concat_decoded_arrays_merges_dictionaries_exactly() {
        // Two pages that each carry the same 127-value dictionary: Arrow's lossy
        // merge overflows Int8, the exact merge fits.
        let a = Arc::new(all_int8_dictionary()) as ArrayRef;
        let b = Arc::new(all_int8_dictionary()) as ArrayRef;
        assert!(
            matches!(
                arrow_select::concat::concat(&[a.as_ref(), b.as_ref()]),
                Err(arrow_schema::ArrowError::DictionaryKeyOverflowError)
            ),
            "Arrow no longer overflows here, so the exact merge is not exercised"
        );

        let merged = concat_decoded_arrays(&[a.as_ref(), b.as_ref()]).unwrap();

        assert_eq!(merged.data_type(), a.data_type());
        assert_eq!(merged.as_any_dictionary().values().len(), 127);
        let expected = [decoded_values(a.as_ref()), decoded_values(b.as_ref())].concat();
        assert_eq!(decoded_values(merged.as_ref()), expected);
    }

    #[test]
    fn test_concat_decoded_arrays_rejects_dictionaries_beyond_key_type() {
        let a = uint16_dictionary((0..100).collect(), (0..100).map(Some).collect());
        let b = uint16_dictionary((1000..1100).collect(), (0..100).map(Some).collect());

        let error = concat_decoded_arrays(&[a.as_ref(), b.as_ref()]).unwrap_err();

        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(
            error.to_string().contains(
                "could not merge the dictionaries of the 2 pages in this batch (200 entries) into Int8 keys, which address at most 128 distinct values"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_dictionary_nulls() {
        // Test both ways of encoding nulls

        // By default, nulls get encoded into the indices
        let arr1 = DictionaryArray::<Int8Type>::from_iter([None, Some("a"), Some("b")]);
        let arr2 = DictionaryArray::<Int8Type>::from_iter([Some("c"), None]);

        let data = DataBlock::from_arrays(&[Arc::new(arr1), Arc::new(arr2)], 5);

        let check_common = |data: DataBlock| {
            assert_eq!(data.num_values(), 5);
            let dict = data.as_dictionary().unwrap();

            let nullable_items = dict.dictionary.as_nullable().unwrap();
            assert_eq!(nullable_items.nulls, LanceBuffer::from(vec![0b00000111]));
            assert_eq!(nullable_items.data.num_values(), 4);

            let items = nullable_items.data.as_variable_width().unwrap();
            assert_eq!(items.bits_per_offset, 32);
            assert_eq!(items.num_values, 4);
            assert_eq!(items.data, LanceBuffer::copy_slice(b"abc"));
            assert_eq!(
                items.offsets,
                LanceBuffer::reinterpret_vec(vec![0, 1, 2, 3, 3],)
            );

            let indices = dict.indices;
            assert_eq!(indices.bits_per_value, 8);
            assert_eq!(indices.num_values, 5);
            assert_eq!(
                indices.data,
                LanceBuffer::reinterpret_vec::<i8>(vec![3, 0, 1, 2, 3])
            );
        };
        check_common(data);

        // However, we can manually create a dictionary where nulls are in the dictionary
        let items = StringArray::from(vec![Some("a"), Some("b"), Some("c"), None]);
        let indices = Int8Array::from(vec![Some(3), Some(0), Some(1), Some(2), Some(3)]);
        let dict = DictionaryArray::new(indices, Arc::new(items));

        let data = DataBlock::from_array(dict);

        check_common(data);
    }

    #[test]
    fn test_dictionary_cannot_add_null() {
        // 256 unique strings
        let items = StringArray::from(
            (0..256)
                .map(|i| Some(String::from_utf8(vec![0; i]).unwrap()))
                .collect::<Vec<_>>(),
        );
        // 257 indices, covering the whole range, plus one null
        let indices = UInt8Array::from(
            (0..=256)
                .map(|i| if i == 256 { None } else { Some(i as u8) })
                .collect::<Vec<_>>(),
        );
        // We want to normalize this by pushing nulls into the dictionary, but we cannot because
        // the dictionary is too large for the index type
        let dict = DictionaryArray::new(indices, Arc::new(items));
        let data = DataBlock::from_array(dict.clone());

        assert_eq!(data.num_values(), 257);

        // The reader narrows the widened indices back to the declared UInt8 keys.
        let read = make_array(
            data.clone()
                .into_arrow(dict.data_type().clone(), false)
                .unwrap(),
        );
        assert_eq!(read.data_type(), dict.data_type());
        assert_eq!(
            arrow_cast::cast(&read, &DataType::Utf8).unwrap().as_ref(),
            arrow_cast::cast(&dict, &DataType::Utf8).unwrap().as_ref()
        );

        let dict = data.as_dictionary().unwrap();

        assert_eq!(dict.indices.bits_per_value, 32);
        assert_eq!(
            dict.indices.data,
            LanceBuffer::reinterpret_vec((0_u32..257).collect::<Vec<_>>())
        );

        let nullable_items = dict.dictionary.as_nullable().unwrap();
        let null_buffer = NullBuffer::new(BooleanBuffer::new(
            nullable_items.nulls.into_buffer(),
            0,
            257,
        ));
        for i in 0..256 {
            assert!(!null_buffer.is_null(i));
        }
        assert!(null_buffer.is_null(256));

        assert_eq!(
            nullable_items.data.as_variable_width().unwrap().data.len(),
            32640
        );
    }

    #[test]
    fn test_all_null() {
        for data_type in [
            DataType::UInt32,
            DataType::FixedSizeBinary(2),
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
            DataType::Struct(Fields::from(vec![Field::new("a", DataType::UInt32, true)])),
        ] {
            let block = DataBlock::AllNull(AllNullDataBlock { num_values: 10 });
            let arr = block.into_arrow(data_type.clone(), true).unwrap();
            let arr = make_array(arr);
            let expected = new_null_array(&data_type, 10);
            assert_eq!(&arr, &expected);
        }
    }

    #[test]
    fn test_dictionary_cannot_concatenate() {
        // 256 unique strings
        let items = StringArray::from(
            (0..256)
                .map(|i| Some(String::from_utf8(vec![0; i]).unwrap()))
                .collect::<Vec<_>>(),
        );
        // 256 different unique strings
        let other_items = StringArray::from(
            (0..256)
                .map(|i| Some(String::from_utf8(vec![1; i + 1]).unwrap()))
                .collect::<Vec<_>>(),
        );
        let indices = UInt8Array::from_iter_values(0..=255);
        let dict1 = DictionaryArray::new(indices.clone(), Arc::new(items));
        let dict2 = DictionaryArray::new(indices, Arc::new(other_items));
        let data = DataBlock::from_arrays(&[Arc::new(dict1), Arc::new(dict2)], 512);
        assert_eq!(data.num_values(), 512);

        // 512 distinct values cannot be read back under the declared UInt8 keys.
        let declared = DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8));
        let error = data.clone().into_arrow(declared, false).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("the 256 distinct values that UInt8 keys can address"),
            "unexpected error: {error}"
        );

        let dict = data.as_dictionary().unwrap();

        assert_eq!(dict.indices.bits_per_value, 32);
        assert_eq!(
            dict.indices.data,
            LanceBuffer::reinterpret_vec::<u32>((0..512).collect::<Vec<_>>())
        );
        // What fun: 0 + 1 + .. + 255 + 1 + 2 + .. + 256 = 2^16
        assert_eq!(
            dict.dictionary.as_variable_width().unwrap().data.len(),
            65536
        );
    }

    #[test]
    fn test_data_size() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(DEFAULT_SEED.0);
        // test data_size() when input has no nulls
        let mut genn = array::rand::<Int32Type>().with_nulls(&[false, false, false]);

        let arr = genn.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());
        assert!(block.data_size() == arr.get_buffer_memory_size() as u64);

        let arr = genn.generate(RowCount::from(400), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());
        assert!(block.data_size() == arr.get_buffer_memory_size() as u64);

        // test data_size() when input has nulls
        let mut genn = array::rand::<Int32Type>().with_nulls(&[false, true, false]);
        let arr = genn.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());

        let array_data = arr.to_data();
        let total_buffer_size: usize = array_data.buffers().iter().map(|buffer| buffer.len()).sum();
        // the NullBuffer.len() returns the length in bits so we divide_round_up by 8
        let array_nulls_size_in_bytes = arr.nulls().unwrap().len().div_ceil(8);
        assert!(block.data_size() == (total_buffer_size + array_nulls_size_in_bytes) as u64);

        let arr = genn.generate(RowCount::from(400), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());

        let array_data = arr.to_data();
        let total_buffer_size: usize = array_data.buffers().iter().map(|buffer| buffer.len()).sum();
        let array_nulls_size_in_bytes = arr.nulls().unwrap().len().div_ceil(8);
        assert!(block.data_size() == (total_buffer_size + array_nulls_size_in_bytes) as u64);

        let mut genn = array::rand::<Int32Type>().with_nulls(&[true, true, false]);
        let arr = genn.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());

        let array_data = arr.to_data();
        let total_buffer_size: usize = array_data.buffers().iter().map(|buffer| buffer.len()).sum();
        let array_nulls_size_in_bytes = arr.nulls().unwrap().len().div_ceil(8);
        assert!(block.data_size() == (total_buffer_size + array_nulls_size_in_bytes) as u64);

        let arr = genn.generate(RowCount::from(400), &mut rng).unwrap();
        let block = DataBlock::from_array(arr.clone());

        let array_data = arr.to_data();
        let total_buffer_size: usize = array_data.buffers().iter().map(|buffer| buffer.len()).sum();
        let array_nulls_size_in_bytes = arr.nulls().unwrap().len().div_ceil(8);
        assert!(block.data_size() == (total_buffer_size + array_nulls_size_in_bytes) as u64);

        let mut genn = array::rand::<Int32Type>().with_nulls(&[false, true, false]);
        let arr1 = genn.generate(RowCount::from(3), &mut rng).unwrap();
        let arr2 = genn.generate(RowCount::from(3), &mut rng).unwrap();
        let arr3 = genn.generate(RowCount::from(3), &mut rng).unwrap();
        let block = DataBlock::from_arrays(&[arr1.clone(), arr2.clone(), arr3.clone()], 9);

        let concatenated_array = arrow_select::concat::concat(&[
            &*Arc::new(arr1.clone()) as &dyn Array,
            &*Arc::new(arr2.clone()) as &dyn Array,
            &*Arc::new(arr3.clone()) as &dyn Array,
        ])
        .unwrap();
        let total_buffer_size: usize = concatenated_array
            .to_data()
            .buffers()
            .iter()
            .map(|buffer| buffer.len())
            .sum();

        let total_nulls_size_in_bytes = concatenated_array.nulls().unwrap().len().div_ceil(8);
        assert!(block.data_size() == (total_buffer_size + total_nulls_size_in_bytes) as u64);
    }

    #[test]
    fn variable_width_rejects_out_of_bounds_offsets_without_optional_validation() {
        let block = VariableWidthBlock {
            data: LanceBuffer::copy_slice(b"alphabetagamma"),
            offsets: LanceBuffer::reinterpret_vec(vec![0_i32, 5, 9, 100_000]),
            bits_per_offset: 32,
            num_values: 3,
            block_info: BlockInfo::new(),
        };

        let error = block
            .into_arrow(DataType::Binary, false)
            .expect_err("out-of-bounds offsets must be rejected");
        assert!(
            matches!(error, Error::CorruptFile { .. }),
            "expected CorruptFile, got: {error}"
        );
        let message = error.to_string();
        assert!(
            message.contains("100000") && message.contains("data buffer size: 14 bytes"),
            "error must report the offending offset and the data buffer size: {message}"
        );
    }

    #[test]
    fn public_fixed_width_conversion_always_validates_layout() {
        let block = FixedWidthDataBlock {
            data: LanceBuffer::from(vec![0_u8; 4]),
            bits_per_value: 32,
            num_values: 2,
            block_info: BlockInfo::new(),
        };

        for validate in [false, true] {
            block
                .clone()
                .into_arrow(DataType::Int32, validate)
                .expect_err("a short values buffer must be rejected");
            DataBlock::FixedWidth(block.clone())
                .into_arrow(DataType::Int32, validate)
                .expect_err("a short values buffer must be rejected");
        }
    }

    #[rstest]
    #[case::i32_decreasing(
        LanceBuffer::reinterpret_vec(vec![0_i32, 5, 2]),
        32,
        2,
        "decreases"
    )]
    #[case::i64_decreasing(
        LanceBuffer::reinterpret_vec(vec![0_i64, 5, 2]),
        64,
        2,
        "decreases"
    )]
    #[case::i32_out_of_bounds(
        LanceBuffer::reinterpret_vec(vec![0_i32, 6]),
        32,
        1,
        "out of bounds"
    )]
    fn variable_width_builder_rejects_malformed_offsets(
        #[case] offsets: LanceBuffer,
        #[case] bits_per_offset: u8,
        #[case] num_values: u64,
        #[case] expected_message: &str,
    ) {
        let block = DataBlock::VariableWidth(VariableWidthBlock {
            data: LanceBuffer::copy_slice(b"abcde"),
            offsets,
            bits_per_offset,
            num_values,
            block_info: BlockInfo::new(),
        });
        let mut builder = DataBlockBuilder::with_capacity_estimate(5);

        let error = builder
            .append(&block, 0..num_values)
            .expect_err("malformed offsets must fail concatenation");
        assert!(
            matches!(error, Error::CorruptFile { .. }),
            "expected CorruptFile, got: {error}"
        );
        assert!(
            error.to_string().contains(expected_message),
            "unexpected message: {error}"
        );
    }

    #[rstest]
    #[case::binary_i32_tail_out_of_bounds(
        DataType::Binary,
        LanceBuffer::reinterpret_vec(vec![0_i32, 5, 9, 100_000]),
        32,
        3,
        b"alphabetagamma".as_slice()
    )]
    #[case::utf8_i32_tail_out_of_bounds(
        DataType::Utf8,
        LanceBuffer::reinterpret_vec(vec![0_i32, 5, 9, 100_000]),
        32,
        3,
        b"alphabetagamma".as_slice()
    )]
    #[case::large_binary_i64_tail_out_of_bounds(
        DataType::LargeBinary,
        LanceBuffer::reinterpret_vec(vec![0_i64, 5, 9, 100_000]),
        64,
        3,
        b"alphabetagamma".as_slice()
    )]
    #[case::large_utf8_i64_tail_out_of_bounds(
        DataType::LargeUtf8,
        LanceBuffer::reinterpret_vec(vec![0_i64, 5, 9, 100_000]),
        64,
        3,
        b"alphabetagamma".as_slice()
    )]
    #[case::binary_negative_offset(
        DataType::Binary,
        LanceBuffer::reinterpret_vec(vec![0_i32, -1, 9, 14]),
        32,
        3,
        b"alphabetagamma".as_slice()
    )]
    #[case::binary_non_monotonic_offsets(
        DataType::Binary,
        LanceBuffer::reinterpret_vec(vec![0_i32, 9, 5, 14]),
        32,
        3,
        b"alphabetagamma".as_slice()
    )]
    #[case::binary_interior_offset_out_of_bounds(
        DataType::Binary,
        LanceBuffer::reinterpret_vec(vec![0_i32, 100_000, 100_000, 14]),
        32,
        3,
        b"alphabetagamma".as_slice()
    )]
    #[case::binary_offsets_buffer_too_short(
        DataType::Binary,
        LanceBuffer::reinterpret_vec(vec![0_i32, 5, 9]),
        32,
        3,
        b"alphabetagamma".as_slice()
    )]
    #[case::utf8_invalid_byte_sequence(
        DataType::Utf8,
        LanceBuffer::reinterpret_vec(vec![0_i32, 1, 2, 3]),
        32,
        3,
        &[b'a', 0xFF, b'b']
    )]
    #[case::utf8_offset_splits_multibyte_char(
        DataType::Utf8,
        LanceBuffer::reinterpret_vec(vec![0_i32, 1, 2]),
        32,
        2,
        "é".as_bytes()
    )]
    #[case::large_utf8_invalid_byte_sequence(
        DataType::LargeUtf8,
        LanceBuffer::reinterpret_vec(vec![0_i64, 1, 2, 3]),
        64,
        3,
        &[b'a', 0xFF, b'b']
    )]
    fn variable_width_rejects_malformed_layout(
        #[case] data_type: DataType,
        #[case] offsets: LanceBuffer,
        #[case] bits_per_offset: u8,
        #[case] num_values: u64,
        #[case] data: &[u8],
    ) {
        let block = VariableWidthBlock {
            data: LanceBuffer::copy_slice(data),
            offsets,
            bits_per_offset,
            num_values,
            block_info: BlockInfo::new(),
        };

        // The malformed layout must be rejected regardless of the optional
        // `validate` flag: the flag selects extra validation, not the memory
        // safety proof required to construct an Arrow array.
        for validate in [false, true] {
            let error = DataBlock::VariableWidth(block.clone())
                .into_arrow(data_type.clone(), validate)
                .expect_err("malformed variable-width layout must be rejected");
            assert!(
                matches!(error, Error::CorruptFile { .. }),
                "expected CorruptFile with validate={validate}, got: {error}"
            );
        }
    }

    #[test]
    fn dictionary_rejects_malformed_variable_width_values_without_optional_validation() {
        let values = VariableWidthBlock {
            data: LanceBuffer::copy_slice(b"alphabetagamma"),
            offsets: LanceBuffer::reinterpret_vec(vec![0_i32, 5, 9, 100_000]),
            bits_per_offset: 32,
            num_values: 3,
            block_info: BlockInfo::new(),
        };
        let dictionary = DataBlock::Dictionary(DictionaryDataBlock {
            indices: FixedWidthDataBlock {
                data: LanceBuffer::reinterpret_vec(vec![0_i32, 1, 2]),
                bits_per_value: 32,
                num_values: 3,
                block_info: BlockInfo::new(),
            },
            dictionary: Box::new(DataBlock::VariableWidth(values)),
        });

        let data_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Binary));
        let error = dictionary
            .into_arrow(data_type, false)
            .expect_err("dictionary with out-of-bounds value offsets must be rejected");
        assert!(
            matches!(error, Error::CorruptFile { .. }),
            "expected CorruptFile, got: {error}"
        );
    }

    #[test]
    fn dictionary_rejects_wide_indices_outside_dictionary() {
        let dictionary = DataBlock::Dictionary(DictionaryDataBlock {
            indices: FixedWidthDataBlock {
                data: LanceBuffer::reinterpret_vec(vec![0_u32, 1, 128]),
                bits_per_value: 32,
                num_values: 3,
                block_info: BlockInfo::new(),
            },
            dictionary: Box::new(DataBlock::from_array(StringArray::from(vec![
                Some("zero"),
                Some("one"),
                None,
            ]))),
        });

        let data_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let error = dictionary
            .into_arrow(data_type, false)
            .expect_err("indices outside the dictionary must be rejected");

        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(
            error.to_string().contains(
                "dictionary indices use 32 bits but the declared Int8 key type uses 8 bits, and they do not address the dictionary"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn dictionary_rejects_unsupported_index_width_mismatch() {
        let block = DataBlock::Dictionary(DictionaryDataBlock {
            indices: FixedWidthDataBlock {
                data: LanceBuffer::reinterpret_vec(vec![0_u16]),
                bits_per_value: 16,
                num_values: 1,
                block_info: BlockInfo::new(),
            },
            dictionary: Box::new(DataBlock::from_array(StringArray::from(vec!["value"]))),
        });
        let data_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let error = block.into_arrow(data_type, false).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("dictionary indices use 16 bits"));
    }

    fn widened_int8_dictionary(indices: Vec<u32>, values: Vec<Option<String>>) -> ArrayRef {
        let num_values = indices.len() as u64;
        let dictionary = DataBlock::Dictionary(DictionaryDataBlock {
            indices: FixedWidthDataBlock {
                data: LanceBuffer::reinterpret_vec(indices),
                bits_per_value: 32,
                num_values,
                block_info: BlockInfo::new(),
            },
            dictionary: Box::new(DataBlock::from_array(StringArray::from(values))),
        });
        let data_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        make_array(dictionary.into_arrow(data_type, false).unwrap())
    }

    fn decode_strings(array: &ArrayRef) -> Vec<Option<String>> {
        let strings = arrow_cast::cast(array, &DataType::Utf8).unwrap();
        strings
            .as_string::<i32>()
            .iter()
            .map(|value| value.map(str::to_owned))
            .collect()
    }

    #[test]
    fn dictionary_narrows_indices_widened_for_null_slot() {
        // The 2.0 writer appended a null slot at index 128, past the last Int8 key.
        let values = (0..128)
            .map(|i| Some(format!("v{i}")))
            .chain([None])
            .collect();
        let array = widened_int8_dictionary(vec![0, 128, 127, 128, 5], values);

        assert_eq!(
            array.data_type(),
            &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8))
        );
        assert_eq!(
            decode_strings(&array),
            vec![
                Some("v0".to_owned()),
                None,
                Some("v127".to_owned()),
                None,
                Some("v5".to_owned())
            ]
        );
        let dictionary = array.as_dictionary::<Int8Type>();
        // Only the four referenced entries remain, and the null slot stays a value.
        assert_eq!(dictionary.values().len(), 4);
        assert_eq!(dictionary.values().null_count(), 1);
    }

    #[test]
    fn dictionary_narrows_indices_widened_at_key_capacity() {
        // 127 values plus the null slot are exactly the 128 entries Int8 keys address.
        let values = (0..128)
            .map(|i| Some(format!("v{i}")))
            .chain([None])
            .collect();
        let indices = (1..128).chain([128]).collect::<Vec<u32>>();
        let array = widened_int8_dictionary(indices, values);

        let expected = (1..128)
            .map(|i| Some(format!("v{i}")))
            .chain([None])
            .collect::<Vec<_>>();
        assert_eq!(decode_strings(&array), expected);
        let dictionary = array.as_dictionary::<Int8Type>();
        assert_eq!(dictionary.values().len(), 128);
        assert_eq!(dictionary.values().null_count(), 1);
    }

    #[test]
    fn dictionary_narrows_indices_widened_for_merged_dictionaries() {
        // Two merged chunks of 100 values each; this batch references three of them.
        let values = (0..200).map(|i| Some(format!("v{i}"))).collect();
        let array = widened_int8_dictionary(vec![150, 3, 150, 199], values);

        assert_eq!(
            decode_strings(&array),
            ["v150", "v3", "v150", "v199"]
                .map(|value| Some(value.to_owned()))
                .to_vec()
        );
        assert_eq!(array.as_dictionary::<Int8Type>().values().len(), 3);
    }

    #[test]
    fn dictionary_narrows_indices_widened_with_repeated_entries() {
        // Merging chunks that share 90 values could repeat them in the merged
        // dictionary.  180 referenced entries do not fit Int8, but 90 distinct values do.
        let values = (0..180).map(|i| Some(format!("v{}", i % 90))).collect();
        let array = widened_int8_dictionary((0..180).rev().collect(), values);

        assert_eq!(
            decode_strings(&array),
            (0..180)
                .rev()
                .map(|i| Some(format!("v{}", i % 90)))
                .collect::<Vec<_>>()
        );
        assert_eq!(array.as_dictionary::<Int8Type>().values().len(), 90);
    }

    #[test]
    fn dictionary_rejects_widened_indices_beyond_declared_key_type() {
        let num_values = 200;
        let dictionary = DataBlock::Dictionary(DictionaryDataBlock {
            indices: FixedWidthDataBlock {
                data: LanceBuffer::reinterpret_vec((0..num_values).collect::<Vec<u32>>()),
                bits_per_value: 32,
                num_values: num_values as u64,
                block_info: BlockInfo::new(),
            },
            dictionary: Box::new(DataBlock::from_array(StringArray::from(
                (0..num_values).map(|i| format!("v{i}")).collect::<Vec<_>>(),
            ))),
        });

        let data_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let error = dictionary
            .into_arrow(data_type, false)
            .expect_err("200 distinct values cannot be addressed by Int8 keys");

        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(
            error.to_string().contains(
                "this batch references 200 dictionary entries, more than the 128 distinct values that Int8 keys can address"
            ),
            "unexpected error: {error}"
        );
    }

    #[rstest]
    #[case::fits_int8(DataType::Int8, 128, None)]
    #[case::overflow_int8(DataType::Int8, 129, Some(128))]
    #[case::fits_uint8(DataType::UInt8, 256, None)]
    #[case::overflow_uint8(DataType::UInt8, 257, Some(256))]
    #[case::fits_int16(DataType::Int16, 32768, None)]
    #[case::overflow_int16(DataType::Int16, 32769, Some(32768))]
    #[case::fits_uint16(DataType::UInt16, 65536, None)]
    #[case::overflow_uint16(DataType::UInt16, 65537, Some(65536))]
    fn dictionary_recovery_batch_limit(
        #[case] key_type: DataType,
        #[case] dictionary_len: u64,
        #[case] expected: Option<u64>,
    ) {
        let data_type = DataType::Dictionary(Box::new(key_type), Box::new(DataType::Utf8));
        assert_eq!(
            super::dictionary_batch_limit(&data_type, dictionary_len),
            expected
        );
    }

    #[rstest]
    #[case::int8(DataType::Int8)]
    #[case::uint8(DataType::UInt8)]
    #[case::int16(DataType::Int16)]
    #[case::uint16(DataType::UInt16)]
    fn dictionary_recovers_widened_indices(#[case] key_type: DataType) {
        // Duplicate values and a null beyond the declared key's range are both
        // emitted by historical writers. Only the selected values need to fit.
        let high_index = super::max_index_val(&key_type) as usize + 1;
        let mut values = vec![Some("duplicate"); high_index + 2];
        values[high_index] = Some("high");
        values[high_index + 1] = None;
        let block = DataBlock::Dictionary(DictionaryDataBlock {
            indices: FixedWidthDataBlock {
                data: LanceBuffer::reinterpret_vec(vec![
                    high_index as u32,
                    0,
                    high_index as u32 + 1,
                    1,
                ]),
                bits_per_value: 32,
                num_values: 4,
                block_info: BlockInfo::new(),
            },
            dictionary: Box::new(DataBlock::from_array(StringArray::from(values))),
        });
        let data_type = DataType::Dictionary(Box::new(key_type), Box::new(DataType::Utf8));
        for validate in [false, true] {
            let array = make_array(
                block
                    .clone()
                    .into_arrow(data_type.clone(), validate)
                    .unwrap(),
            );
            assert_eq!(array.data_type(), &data_type);
            let decoded = arrow_cast::cast(array.as_ref(), &DataType::Utf8).unwrap();
            assert_eq!(
                decoded.as_ref(),
                &StringArray::from(vec![
                    Some("high"),
                    Some("duplicate"),
                    None,
                    Some("duplicate")
                ])
            );
        }
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::all_null(vec![None, None])]
    fn dictionary_recovers_widened_nulls(#[case] values: Vec<Option<&str>>) {
        let block = DataBlock::Dictionary(DictionaryDataBlock {
            indices: FixedWidthDataBlock {
                data: LanceBuffer::reinterpret_vec(vec![0_u32; values.len()]),
                bits_per_value: 32,
                num_values: values.len() as u64,
                block_info: BlockInfo::new(),
            },
            dictionary: Box::new(DataBlock::from_array(StringArray::from(vec![None::<&str>]))),
        });
        let data_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let array = make_array(block.into_arrow(data_type, false).unwrap());
        let decoded = arrow_cast::cast(array.as_ref(), &DataType::Utf8).unwrap();
        assert_eq!(decoded.as_ref(), &StringArray::from(values));
    }

    #[test]
    fn dictionary_recovers_widened_duplicate_entries() {
        let values =
            StringArray::from((0..180).map(|i| format!("v{}", i % 90)).collect::<Vec<_>>());
        let block = DataBlock::Dictionary(DictionaryDataBlock {
            indices: FixedWidthDataBlock {
                data: LanceBuffer::reinterpret_vec((0_u32..180).collect()),
                bits_per_value: 32,
                num_values: 180,
                block_info: BlockInfo::new(),
            },
            dictionary: Box::new(DataBlock::from_array(values.clone())),
        });
        let data_type = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
        let array = make_array(block.into_arrow(data_type, false).unwrap());
        let decoded = arrow_cast::cast(array.as_ref(), &DataType::Utf8).unwrap();
        assert_eq!(decoded.as_ref(), &values);
    }

    #[rstest]
    #[case::binary(Arc::new(BinaryArray::from_vec(vec![b"alpha", b"", b"gamma"])) as ArrayRef)]
    #[case::large_binary(
        Arc::new(LargeBinaryArray::from_vec(vec![b"alpha", b"", b"gamma"])) as ArrayRef
    )]
    #[case::utf8(Arc::new(StringArray::from(vec!["héllo", "", "world"])) as ArrayRef)]
    #[case::large_utf8(Arc::new(LargeStringArray::from(vec!["héllo", "", "world"])) as ArrayRef)]
    fn variable_width_valid_data_survives_mandatory_validation(#[case] array: ArrayRef) {
        let block = DataBlock::from_array(array.clone());
        for validate in [false, true] {
            let round_tripped = make_array(
                block
                    .clone()
                    .into_arrow(array.data_type().clone(), validate)
                    .unwrap(),
            );
            assert_eq!(&round_tripped, &array);
        }
    }
}
