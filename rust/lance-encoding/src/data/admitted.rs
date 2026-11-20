// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Payload admission across a complete typed materialization tree.

use std::alloc::Layout;

use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, alloc::ALIGNMENT, bit_util};
use arrow_data::ArrayData;
use arrow_schema::DataType;
use lance_core::{Error, Result, utils::allocation::Reservation};

use super::DataBlock;

#[cfg(test)]
mod tests;

enum Payload<'a> {
    Zeroed(usize),
    BooleanBytes(&'a [u8]),
}

impl Payload<'_> {
    fn new_bytes(&self) -> usize {
        match self {
            Self::Zeroed(bytes) => *bytes,
            Self::BooleanBytes(values) => values.len().div_ceil(8),
        }
    }

    fn materialize(self, workspace: &mut Reservation) -> Result<Buffer> {
        let mut buffer = workspace.try_allocate(self.new_bytes())?.ok_or_else(|| {
            Error::internal("typed materialization exceeded its complete workspace plan")
        })?;
        if let Self::BooleanBytes(values) = self {
            for (index, value) in values.iter().enumerate() {
                if *value != 0 {
                    bit_util::set_bit(buffer.as_mut(), index);
                }
            }
        }
        Ok(Buffer::from(buffer.freeze()))
    }
}

enum Validity {
    Shared(NullBuffer),
    AllNull,
}

enum Plan<'a> {
    // Already validated, zero-copy layouts need no additional payload allocation.
    Existing(ArrayData),
    Build {
        data_type: DataType,
        len: usize,
        buffers: Vec<Payload<'a>>,
        children: Vec<Self>,
        validity: Option<Validity>,
    },
}

impl Plan<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Existing(data) => data.len(),
            Self::Build { len, .. } => *len,
        }
    }

    fn additional_bytes(&self) -> Result<usize> {
        match self {
            Self::Existing(_) => Ok(0),
            Self::Build {
                len,
                buffers,
                children,
                validity,
                ..
            } => {
                let mut bytes = if matches!(validity, Some(Validity::AllNull)) {
                    len.div_ceil(8)
                } else {
                    0
                };
                for payload in buffers {
                    let size = payload.new_bytes();
                    Layout::from_size_align(size, ALIGNMENT).map_err(|_| {
                        Error::invalid_input("typed payload exceeds addressable allocation size")
                    })?;
                    bytes = sum(bytes, size)?;
                }
                for child in children {
                    bytes = sum(bytes, child.additional_bytes()?)?;
                }
                Ok(bytes)
            }
        }
    }

    fn validation_scratch(&self) -> Result<usize> {
        let Self::Build {
            data_type,
            len,
            children,
            validity,
            ..
        } = self
        else {
            return Ok(0);
        };
        let has_nulls = match validity {
            Some(Validity::AllNull) => *len > 0,
            Some(Validity::Shared(nulls)) => nulls.null_count() > 0,
            None => false,
        };
        // Pinned Arrow validate_nulls expands the parent bitmap for non-nullable
        // FSL children. NullBuffer::expand owns an exact ceil(child_len / 8)
        // allocation, discarded before build returns. Child/parent validation is
        // sequential, so one shared maximum-sized scratch allowance suffices.
        let mut scratch = match data_type {
            DataType::FixedSizeList(field, size) if !field.is_nullable() && has_nulls => {
                product(*len, dimension(*size)?)?.div_ceil(8)
            }
            _ => 0,
        };
        for child in children {
            scratch = scratch.max(child.validation_scratch()?);
        }
        Ok(scratch)
    }

    fn with_validity(self, nulls: NullBuffer) -> Result<Self> {
        if nulls.len() != self.len() {
            return Err(Error::invalid_input(
                "typed validity length differs from its row count",
            ));
        }
        match self {
            Self::Existing(data) => Ok(Self::Existing(
                data.into_builder().nulls(Some(nulls)).build()?,
            )),
            Self::Build {
                data_type,
                len,
                buffers,
                children,
                ..
            } => Ok(Self::Build {
                data_type,
                len,
                buffers,
                children,
                validity: Some(Validity::Shared(nulls)),
            }),
        }
    }

    fn materialize(self, workspace: &mut Reservation) -> Result<ArrayData> {
        match self {
            Self::Existing(data) => Ok(data),
            Self::Build {
                data_type,
                len,
                buffers,
                children,
                validity,
            } => {
                let mut builder = ArrayData::builder(data_type).len(len);
                for payload in buffers {
                    builder = builder.add_buffer(payload.materialize(workspace)?);
                }
                for child in children {
                    builder = builder.add_child_data(child.materialize(workspace)?);
                }
                if let Some(validity) = validity {
                    let nulls = match validity {
                        Validity::Shared(nulls) => nulls,
                        Validity::AllNull => NullBuffer::new(BooleanBuffer::new(
                            Payload::Zeroed(len.div_ceil(8)).materialize(workspace)?,
                            0,
                            len,
                        )),
                    };
                    builder = builder.nulls(Some(nulls));
                }
                Ok(builder.build()?)
            }
        }
    }
}

fn rows(count: u64) -> Result<usize> {
    usize::try_from(count).map_err(|_| Error::invalid_input("typed row count exceeds usize"))
}

fn sum(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| Error::invalid_input("typed payload size overflows"))
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::invalid_input("typed payload size overflows"))
}

fn dimension(value: i32) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::invalid_input("typed dimension must not be negative"))
}

fn null_plan(data_type: &DataType, len: usize) -> Result<Plan<'static>> {
    let mut buffers = Vec::new();
    let mut children = Vec::new();
    if let Some(width) = data_type.primitive_width() {
        buffers.push(Payload::Zeroed(product(len, width)?));
    } else {
        match data_type {
            DataType::Null => (),
            DataType::Boolean => buffers.push(Payload::Zeroed(len.div_ceil(8))),
            DataType::FixedSizeBinary(width) => {
                buffers.push(Payload::Zeroed(product(len, dimension(*width)?)?))
            }
            DataType::Binary | DataType::Utf8 | DataType::LargeBinary | DataType::LargeUtf8 => {
                let width = if matches!(data_type, DataType::Binary | DataType::Utf8) {
                    4
                } else {
                    8
                };
                buffers.push(Payload::Zeroed(product(sum(len, 1)?, width)?));
                buffers.push(Payload::Zeroed(0));
            }
            DataType::List(field) | DataType::LargeList(field) => {
                let width = if matches!(data_type, DataType::List(_)) {
                    4
                } else {
                    8
                };
                buffers.push(Payload::Zeroed(product(sum(len, 1)?, width)?));
                children.push(null_plan(field.data_type(), 0)?);
            }
            DataType::FixedSizeList(field, size) => {
                children.push(null_plan(
                    field.data_type(),
                    product(len, dimension(*size)?)?,
                )?);
            }
            DataType::Struct(fields) => {
                for field in fields {
                    children.push(null_plan(field.data_type(), len)?);
                }
            }
            other => {
                return Err(Error::not_supported(format!(
                    "no admitted null materialization for {other}"
                )));
            }
        }
    }
    let validity = (data_type != &DataType::Null && len > 0).then_some(Validity::AllNull);
    Ok(Plan::Build {
        data_type: data_type.clone(),
        len,
        buffers,
        children,
        validity,
    })
}

fn plan<'a>(block: &'a DataBlock, data_type: &DataType) -> Result<Plan<'a>> {
    match block {
        DataBlock::Empty() => null_plan(data_type, 0),
        DataBlock::AllNull(block) => null_plan(data_type, rows(block.num_values)?),
        DataBlock::FixedWidth(block) => {
            let len = rows(block.num_values)?;
            if data_type == &DataType::Boolean && block.bits_per_value == 8 {
                let values = block.data.get(..len).ok_or_else(|| {
                    Error::invalid_input("expanded Boolean buffer is shorter than its row count")
                })?;
                return Ok(Plan::Build {
                    data_type: DataType::Boolean,
                    len,
                    buffers: vec![Payload::BooleanBytes(values)],
                    children: vec![],
                    validity: None,
                });
            }
            if data_type.primitive_width().is_none()
                && !matches!(data_type, DataType::Boolean | DataType::FixedSizeBinary(_))
            {
                return Err(Error::not_supported(
                    "admitted FixedWidth conversion needs a scalar Arrow layout",
                ));
            }
            if let DataType::FixedSizeBinary(width) = data_type {
                dimension(*width)?;
            }
            // These conversions only borrow buffers; Arrow validates layout and alignment.
            // The Boolean byte expansion above is the sole copying FixedWidth conversion.
            Ok(Plan::Existing(
                block.clone().into_arrow(data_type.clone(), true)?,
            ))
        }
        DataBlock::VariableWidth(block) => {
            if !matches!(
                data_type,
                DataType::Binary | DataType::LargeBinary | DataType::Utf8 | DataType::LargeUtf8
            ) {
                return Err(Error::not_supported(
                    "admitted VariableWidth conversion needs a binary or string Arrow layout",
                ));
            }
            Ok(Plan::Existing(
                block.clone().into_arrow(data_type.clone(), true)?,
            ))
        }
        DataBlock::Nullable(block) => {
            if data_type == &DataType::Null {
                return match block.data.as_ref() {
                    DataBlock::Empty() => null_plan(data_type, 0),
                    DataBlock::AllNull(inner) => null_plan(data_type, rows(inner.num_values)?),
                    _ => Err(Error::not_supported(
                        "Null materialization requires an all-null block",
                    )),
                };
            }
            let child = plan(&block.data, data_type)?;
            if block.nulls.len() < child.len().div_ceil(8) {
                return Err(Error::invalid_input(
                    "typed validity buffer is shorter than its row count",
                ));
            }
            let nulls = NullBuffer::new(BooleanBuffer::new(
                block.nulls.clone().into_buffer(),
                0,
                child.len(),
            ));
            child.with_validity(nulls)
        }
        DataBlock::FixedSizeList(block) => {
            let DataType::FixedSizeList(field, size) = data_type else {
                return Err(Error::invalid_input(
                    "FixedSizeList block needs a FixedSizeList Arrow type",
                ));
            };
            let size = dimension(*size)?;
            if size == 0 || block.dimension != size as u64 {
                return Err(Error::invalid_input(
                    "FixedSizeList block and Arrow dimensions must match and be positive",
                ));
            }
            let child = plan(&block.child, field.data_type())?;
            if !child.len().is_multiple_of(size) {
                return Err(Error::invalid_input(
                    "FixedSizeList child length is not divisible by its dimension",
                ));
            }
            Ok(Plan::Build {
                data_type: data_type.clone(),
                len: child.len() / size,
                buffers: vec![],
                children: vec![child],
                validity: None,
            })
        }
        DataBlock::Struct(block) => {
            let DataType::Struct(fields) = data_type else {
                return Err(Error::invalid_input(
                    "Struct block needs a Struct Arrow type",
                ));
            };
            if fields.len() != block.children.len() {
                return Err(Error::invalid_input("Struct field and child counts differ"));
            }
            let children = block
                .children
                .iter()
                .zip(fields)
                .map(|(child, field)| plan(child, field.data_type()))
                .collect::<Result<Vec<_>>>()?;
            let len = children.first().map(Plan::len).unwrap_or(0);
            if children.iter().any(|child| child.len() != len)
                || block
                    .validity
                    .as_ref()
                    .is_some_and(|nulls| nulls.len() != len)
            {
                return Err(Error::invalid_input(
                    "Struct child and validity row counts differ",
                ));
            }
            Ok(Plan::Build {
                data_type: data_type.clone(),
                len,
                buffers: vec![],
                children,
                validity: block.validity.clone().map(Validity::Shared),
            })
        }
        other => Err(Error::not_supported(format!(
            "no admitted typed materialization for {}",
            other.name()
        ))),
    }
}

impl DataBlock {
    /// Materialize supported Arrow layouts after admitting every new payload buffer.
    /// Existing input buffers must already carry their physical allocation charge.
    /// This borrows the block, so `None` preserves it and the workspace for drain/replan.
    /// All-null layouts and byte-to-bit Boolean conversion allocate through the workspace;
    /// ordinary compatible buffers remain shared. Unsupported layouts return an error
    /// instead of invoking an unaccounted conversion. Metadata and allocator overhead
    /// are excluded; this opt-in method does not change existing scan paths.
    /// Pinned Arrow validation scratch is admitted too; its allowance stays in the
    /// workspace for reuse after validation releases the temporary allocation.
    /// Constant, dictionary and opaque blocks remain unsupported. Null materialization
    /// supports scalar, binary/string, list, fixed-size-list and struct layouts.
    ///
    /// ```
    /// use arrow_schema::DataType;
    /// use lance_core::utils::allocation::AllocationBudget;
    /// use lance_encoding::data::{AllNullDataBlock, DataBlock};
    /// let budget = AllocationBudget::new(38); // Nine f32 values plus two validity bytes.
    /// let mut work = budget.try_reserve(38).unwrap();
    /// let block = DataBlock::AllNull(AllNullDataBlock { num_values: 9 });
    /// let array = block.try_to_arrow_in(&DataType::Float32, &mut work).unwrap().unwrap();
    /// assert_eq!(array.null_count(), 9);
    /// assert_eq!(budget.reserved(), 38);
    /// drop(array);
    /// assert_eq!(budget.reserved(), 0);
    /// ```
    pub fn try_to_arrow_in(
        &self,
        data_type: &DataType,
        workspace: &mut Reservation,
    ) -> Result<Option<ArrayData>> {
        // Planning validates and retains borrowed inputs but allocates no new payload.
        // A single preflight covers the whole tree, not one child at a time.
        let plan = plan(self, data_type)?;
        let required = sum(plan.additional_bytes()?, plan.validation_scratch()?)?;
        if required > workspace.len() {
            return Ok(None);
        }
        plan.materialize(workspace).map(Some)
    }
}
