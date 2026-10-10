// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The plane-row layout of IVF_RQ auxiliary files.
//!
//! The column layout stores each field of an IVF_RQ storage row (row id,
//! codes, factors, bounds) in a column of its own, so reading one plane of a
//! partition reads one byte range per field. The plane-row layout packs the
//! fields of each plane into one fixed-width row column instead, so a plane of
//! a partition, or a run of its rows, is one contiguous byte range. Both hold
//! the same values; readers unpack plane rows into the column layout's fields,
//! and nothing downstream depends on the layout. See the format
//! documentation's "RaBitQ plane-row layout" for the binding specification.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt8Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchOptions, UInt8Array,
    UInt64Array, new_empty_array,
};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use lance_core::{Error, ROW_ID, ROW_ID_FIELD, Result};
use lance_encoding::constants::{
    COMPRESSION_META_KEY, STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY,
};
use lance_file::version::ConcreteFileVersion;

use super::builder::rabit_storage_fields;
use super::layered::{
    FULL_BOUNDS_COLUMN, HIGH_ADD_FACTORS_COLUMN, HIGH_BOUNDS_COLUMN, HIGH_SCALE_FACTORS_COLUMN,
};
use super::layered_stats;
use super::storage::{
    RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_BLOCKED_EX_CODE_LO_COLUMN, RABIT_CODE_COLUMN,
    RABIT_EX_CODE_COLUMN, RabitQuantizationMetadata,
};
use super::transform::{
    ADD_FACTORS_COLUMN, ERROR_FACTORS_COLUMN, EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN,
    SCALE_FACTORS_COLUMN,
};

/// A native index's packed column: the whole row.
pub const RQ_ROWS_COLUMN: &str = "__rq_rows";
/// A layered index's sign plane: row id, sign codes and their factors.
pub const RQ_SIGN_ROWS_COLUMN: &str = "__rq_sign_rows";
/// A layered index's estimator bounds.
pub const RQ_BOUNDS_ROWS_COLUMN: &str = "__rq_bounds_rows";
/// A layered index's high ex plane.
pub const RQ_HIGH_ROWS_COLUMN: &str = "__rq_high_rows";
/// A layered index's low ex plane.
pub const RQ_LOW_ROWS_COLUMN: &str = "__rq_low_rows";

/// Every packed column name. A column-layout file has none of them.
pub const PACKED_COLUMNS: [&str; 5] = [
    RQ_ROWS_COLUMN,
    RQ_SIGN_ROWS_COLUMN,
    RQ_BOUNDS_ROWS_COLUMN,
    RQ_HIGH_ROWS_COLUMN,
    RQ_LOW_ROWS_COLUMN,
];

/// A packed column and the fields it packs, in their order within a row.
/// A field the index does not store is left out.
type PackedGroup = (&'static str, &'static [&'static str]);

/// A native index's one packed column.
const NATIVE_GROUPS: [PackedGroup; 1] = [(
    RQ_ROWS_COLUMN,
    &[
        ROW_ID,
        RABIT_CODE_COLUMN,
        ADD_FACTORS_COLUMN,
        SCALE_FACTORS_COLUMN,
        ERROR_FACTORS_COLUMN,
        RABIT_BLOCKED_EX_CODE_COLUMN,
        EX_ADD_FACTORS_COLUMN,
        EX_SCALE_FACTORS_COLUMN,
    ],
)];

/// A layered index's packed columns, one per plane, in file order.
const LAYERED_GROUPS: [PackedGroup; 4] = [
    (
        RQ_SIGN_ROWS_COLUMN,
        &[
            ROW_ID,
            RABIT_CODE_COLUMN,
            ADD_FACTORS_COLUMN,
            SCALE_FACTORS_COLUMN,
            ERROR_FACTORS_COLUMN,
        ],
    ),
    (
        RQ_BOUNDS_ROWS_COLUMN,
        &[HIGH_BOUNDS_COLUMN, FULL_BOUNDS_COLUMN],
    ),
    (
        RQ_HIGH_ROWS_COLUMN,
        &[
            RABIT_BLOCKED_EX_CODE_COLUMN,
            HIGH_ADD_FACTORS_COLUMN,
            HIGH_SCALE_FACTORS_COLUMN,
        ],
    ),
    (
        RQ_LOW_ROWS_COLUMN,
        &[
            RABIT_BLOCKED_EX_CODE_LO_COLUMN,
            EX_ADD_FACTORS_COLUMN,
            EX_SCALE_FACTORS_COLUMN,
        ],
    ),
];

/// Field metadata value that turns off general compression.
const NO_COMPRESSION: &str = "none";

/// Whether `name` is an internal column of an IVF_RQ storage file in either
/// layout, which a plane-row file cannot carry unpacked.
fn is_internal_column(name: &str) -> bool {
    name == RABIT_EX_CODE_COLUMN
        || PACKED_COLUMNS.contains(&name)
        || NATIVE_GROUPS
            .iter()
            .chain(&LAYERED_GROUPS)
            .any(|(_, fields)| fields.contains(&name))
}

/// Whether writers of Lance file `version` request the full-zip structural
/// encoding for packed columns: 2.1 and later, where rows narrower than the
/// full-zip cutoff would otherwise be written as mini-blocks, which are read
/// whole.
pub fn requests_fullzip(version: ConcreteFileVersion) -> bool {
    !matches!(version, ConcreteFileVersion::V1 | ConcreteFileVersion::V2_0)
}

/// How a packed field's values are encoded in a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueKind {
    /// A little-endian `uint64`.
    U64,
    /// A little-endian `float32`.
    F32,
    /// A `list<uint8>[n]`: its `n` bytes.
    U8List(usize),
    /// A `list<float32>[n]`: `n` little-endian `float32` values.
    F32List(usize),
}

impl ValueKind {
    fn of(field: &Field) -> Result<Self> {
        let kind = match field.data_type() {
            DataType::UInt64 => Some(Self::U64),
            DataType::Float32 => Some(Self::F32),
            DataType::FixedSizeList(item, size) => {
                let size = usize::try_from(*size).ok();
                match (item.data_type(), size) {
                    (DataType::UInt8, Some(size)) => Some(Self::U8List(size)),
                    (DataType::Float32, Some(size)) => Some(Self::F32List(size)),
                    _ => None,
                }
            }
            _ => None,
        };
        kind.ok_or_else(|| {
            Error::invalid_input(format!(
                "column {} of type {} has no plane-row form",
                field.name(),
                field.data_type()
            ))
        })
    }

    /// Bytes the value takes in a row.
    fn width(self) -> usize {
        match self {
            Self::U64 => size_of::<u64>(),
            Self::F32 => size_of::<f32>(),
            Self::U8List(size) => size,
            Self::F32List(size) => size * size_of::<f32>(),
        }
    }
}

/// One field of a packed column.
#[derive(Debug, Clone)]
struct PackedMember {
    /// The field as the column layout stores it.
    field: FieldRef,
    /// Where its value starts in a row.
    offset: usize,
    kind: ValueKind,
}

impl PackedMember {
    fn type_error(&self, array: &dyn Array) -> Error {
        Error::invalid_input(format!(
            "column {} is {}, which does not pack as {}",
            self.field.name(),
            array.data_type(),
            self.field.data_type()
        ))
    }

    /// Write `array`'s values into the rows of `out`, rows of `stride` bytes.
    fn pack_into(&self, array: &dyn Array, out: &mut [u8], stride: usize) -> Result<()> {
        let name = self.field.name();
        let rows = out.len() / stride;
        if array.len() != rows {
            return Err(Error::invalid_input(format!(
                "column {name} has {} rows, the batch {rows}",
                array.len()
            )));
        }
        if array.null_count() > 0 {
            return Err(Error::invalid_input(format!(
                "column {name} has nulls, which the plane-row layout cannot store"
            )));
        }
        let range = self.offset..self.offset + self.kind.width();
        let rows_out = out.chunks_exact_mut(stride);
        match self.kind {
            ValueKind::U64 => {
                let values = array
                    .as_primitive_opt::<UInt64Type>()
                    .ok_or_else(|| self.type_error(array))?;
                for (row, value) in rows_out.zip(values.values()) {
                    row[range.clone()].copy_from_slice(&value.to_le_bytes());
                }
            }
            ValueKind::F32 => {
                let values = array
                    .as_primitive_opt::<Float32Type>()
                    .ok_or_else(|| self.type_error(array))?;
                for (row, value) in rows_out.zip(values.values()) {
                    row[range.clone()].copy_from_slice(&value.to_le_bytes());
                }
            }
            ValueKind::U8List(size) => {
                let values = self.list_values(array, size)?;
                let values = values
                    .as_primitive_opt::<UInt8Type>()
                    .ok_or_else(|| self.type_error(array))?
                    .values();
                for (row, value) in rows_out.zip(values.chunks_exact(size)) {
                    row[range.clone()].copy_from_slice(value);
                }
            }
            ValueKind::F32List(size) => {
                let values = self.list_values(array, size)?;
                let values = values
                    .as_primitive_opt::<Float32Type>()
                    .ok_or_else(|| self.type_error(array))?
                    .values();
                for (row, value) in rows_out.zip(values.chunks_exact(size)) {
                    for (dst, item) in row[range.clone()]
                        .as_chunks_mut::<{ size_of::<f32>() }>()
                        .0
                        .iter_mut()
                        .zip(value)
                    {
                        dst.copy_from_slice(&item.to_le_bytes());
                    }
                }
            }
        }
        Ok(())
    }

    /// The items of list column `array`, `size` per row and without nulls.
    fn list_values<'a>(&self, array: &'a dyn Array, size: usize) -> Result<&'a ArrayRef> {
        let list = array
            .as_fixed_size_list_opt()
            .filter(|list| list.value_length() as usize == size)
            .ok_or_else(|| self.type_error(array))?;
        let values = list.values();
        if values.len() != list.len() * size || values.null_count() > 0 {
            return Err(Error::invalid_input(format!(
                "column {} has null or misaligned list items",
                self.field.name()
            )));
        }
        Ok(values)
    }

    /// The field's values in `packed`, slices of `rows` rows of `stride` bytes
    /// each, as the column layout stores them: in buffers of exactly their
    /// size.
    fn unpack(&self, packed: &[&[u8]], stride: usize, rows: usize) -> Result<ArrayRef> {
        let range = self.offset..self.offset + self.kind.width();
        let rows_in = || packed.iter().flat_map(|bytes| bytes.chunks_exact(stride));
        Ok(match self.kind {
            ValueKind::U64 => {
                let mut values = Vec::with_capacity(rows);
                values
                    .extend(rows_in().map(|row| u64::from_le_bytes(le_bytes(&row[range.clone()]))));
                Arc::new(UInt64Array::from(values))
            }
            ValueKind::F32 => {
                let mut values = Vec::with_capacity(rows);
                values
                    .extend(rows_in().map(|row| f32::from_le_bytes(le_bytes(&row[range.clone()]))));
                Arc::new(Float32Array::from(values))
            }
            ValueKind::U8List(size) => {
                let mut values = Vec::with_capacity(rows * size);
                for row in rows_in() {
                    values.extend_from_slice(&row[range.clone()]);
                }
                self.list_array(size, Arc::new(UInt8Array::from(values)))?
            }
            ValueKind::F32List(size) => {
                let mut values = Vec::with_capacity(rows * size);
                for row in rows_in() {
                    values.extend(
                        row[range.clone()]
                            .as_chunks::<{ size_of::<f32>() }>()
                            .0
                            .iter()
                            .copied()
                            .map(f32::from_le_bytes),
                    );
                }
                self.list_array(size, Arc::new(Float32Array::from(values)))?
            }
        })
    }

    /// A list column of this field's type over `values`.
    fn list_array(&self, size: usize, values: ArrayRef) -> Result<ArrayRef> {
        let DataType::FixedSizeList(item, _) = self.field.data_type() else {
            return Err(Error::internal(format!(
                "packed field {} is not a list",
                self.field.name()
            )));
        };
        Ok(Arc::new(FixedSizeListArray::try_new(
            item.clone(),
            size as i32,
            values,
            None,
        )?))
    }
}

/// Whether `read` is `expected` but for the nullability of a list's items,
/// which a Lance file's schema does not keep: the packed columns are written
/// with non-nullable items and read back with nullable ones, so reads reject
/// null items instead (see [`PackedColumn`]).
fn same_type_but_item_nullability(read: &DataType, expected: &DataType) -> bool {
    match (read, expected) {
        (DataType::FixedSizeList(read_item, read_size), DataType::FixedSizeList(item, size)) => {
            read_size == size && read_item.data_type() == item.data_type()
        }
        _ => read == expected,
    }
}

/// `bytes`, a slice of exactly `N` bytes, as an array.
fn le_bytes<const N: usize>(bytes: &[u8]) -> [u8; N] {
    let mut out = [0; N];
    out.copy_from_slice(bytes);
    out
}

/// One packed column of a plane-row file: `list<uint8>[stride]`, each row
/// the concatenation of its fields' values, without padding.
#[derive(Debug, Clone)]
pub struct PackedColumn {
    name: &'static str,
    stride: usize,
    members: Vec<PackedMember>,
}

impl PackedColumn {
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Bytes per row.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The fields it packs, in their order within a row.
    pub fn fields(&self) -> impl Iterator<Item = &FieldRef> {
        self.members.iter().map(|member| &member.field)
    }

    fn data_type(&self) -> DataType {
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            self.stride as i32,
        )
    }

    /// The column's field in a file. With `fullzip`, it requests the
    /// full-zip structural encoding and no general compression, so a page
    /// holds `rows * stride` bytes and any run of rows is one byte range.
    fn file_field(&self, fullzip: bool) -> Field {
        let field = Field::new(self.name, self.data_type(), false);
        if !fullzip {
            return field;
        }
        field.with_metadata(HashMap::from([
            (
                STRUCTURAL_ENCODING_META_KEY.to_string(),
                STRUCTURAL_ENCODING_FULLZIP.to_string(),
            ),
            (COMPRESSION_META_KEY.to_string(), NO_COMPRESSION.to_string()),
        ]))
    }

    /// The rows of this column in `batch`, `stride` bytes each.
    fn rows_of<'a>(&self, batch: &'a RecordBatch) -> Result<&'a [u8]> {
        let column = batch.column_by_name(self.name).ok_or_else(|| {
            Error::internal(format!("a plane-row read is missing column {}", self.name))
        })?;
        let rows = column
            .as_fixed_size_list_opt()
            .filter(|list| list.value_length() as usize == self.stride && list.null_count() == 0)
            .and_then(|list| list.values().as_primitive_opt::<UInt8Type>())
            .filter(|values| values.null_count() == 0)
            .ok_or_else(|| {
                Error::index(format!(
                    "plane-row column {} read as {}, expected non-null {}",
                    self.name,
                    column.data_type(),
                    self.data_type()
                ))
            })?
            .values();
        if rows.len() != batch.num_rows() * self.stride {
            return Err(Error::index(format!(
                "plane-row column {} holds {} bytes for {} rows of {} bytes",
                self.name,
                rows.len(),
                batch.num_rows(),
                self.stride
            )));
        }
        Ok(rows)
    }
}

/// Where a logical field is stored in a plane-row file.
#[derive(Debug, Clone, Copy)]
enum Location {
    /// Member `member` of packed column `column`.
    Packed { column: usize, member: usize },
    /// Carried column `index`, stored unpacked.
    Carried(usize),
}

/// The plane-row layout of one IVF_RQ storage file: its packed columns, the
/// column layout's fields each packs, and the columns it carries unpacked.
#[derive(Debug, Clone)]
pub struct PlaneRowsSpec {
    packed: Vec<PackedColumn>,
    carried: Vec<FieldRef>,
    /// The column layout's fields in its order, then the carried fields,
    /// with the file's schema metadata: what a read unpacks into.
    logical: SchemaRef,
    locations: HashMap<String, Location>,
}

impl PlaneRowsSpec {
    /// The plane rows of an IVF_RQ storage with `metadata`: the column
    /// layout's fields, the row id and [`rabit_storage_fields`], packed as
    /// the layout packs them, with no carried columns.
    pub fn try_new(metadata: &RabitQuantizationMetadata) -> Result<Self> {
        let mut fields = vec![Arc::new(ROW_ID_FIELD.clone())];
        fields.extend(rabit_storage_fields(metadata)?.into_iter().map(Arc::new));
        Self::from_parts(fields, metadata.layered, Vec::new(), HashMap::new())
    }

    /// The plane rows of a plane-row file with `metadata` and `file_schema`,
    /// which must start with the packed columns the metadata implies, in
    /// order and with their strides; every later column is carried. A file
    /// whose schema does not match is an invalid index.
    pub fn for_file(metadata: &RabitQuantizationMetadata, file_schema: &Schema) -> Result<Self> {
        let spec = Self::try_new(metadata)?;
        let packed = spec.packed.len();
        let carried: Vec<FieldRef> = file_schema.fields().iter().skip(packed).cloned().collect();
        let spec = Self::from_parts(
            spec.column_fields().cloned().collect(),
            metadata.layered,
            carried,
            file_schema.metadata().clone(),
        )
        .map_err(|error| Error::index(format!("invalid plane-row IVF_RQ file: {error}")))?;
        spec.validate_file_schema(file_schema)?;
        Ok(spec)
    }

    /// The plane rows of the column-layout file with `metadata` and
    /// `column_schema`, to convert it: the file must store the column
    /// layout's fields exactly as [`Self::try_new`] derives them, in that
    /// order; every other column is carried.
    pub fn for_column_file(
        metadata: &RabitQuantizationMetadata,
        column_schema: &Schema,
    ) -> Result<Self> {
        let spec = Self::try_new(metadata)?;
        let fields: Vec<&FieldRef> = spec.column_fields().collect();
        let stored: Vec<&FieldRef> = column_schema
            .fields()
            .iter()
            .filter(|field| spec.locations.contains_key(field.name()))
            .collect();
        if stored != fields {
            return Err(Error::invalid_input(format!(
                "the column layout's fields are {:?}, the plane-row layout packs {:?}",
                stored, fields
            )));
        }
        let carried = column_schema
            .fields()
            .iter()
            .filter(|field| !spec.locations.contains_key(field.name()))
            .cloned()
            .collect();
        Self::from_parts(
            fields.into_iter().cloned().collect(),
            metadata.layered,
            carried,
            column_schema.metadata().clone(),
        )
    }

    fn from_parts(
        column_fields: Vec<FieldRef>,
        layered: bool,
        carried: Vec<FieldRef>,
        metadata: HashMap<String, String>,
    ) -> Result<Self> {
        let groups: &[PackedGroup] = if layered {
            &LAYERED_GROUPS
        } else {
            &NATIVE_GROUPS
        };
        let mut locations = HashMap::new();
        let mut packed = Vec::with_capacity(groups.len());
        for (name, group) in groups {
            let mut members = Vec::new();
            let mut stride = 0;
            for &member in *group {
                let Some(field) = column_fields.iter().find(|field| field.name() == member) else {
                    continue;
                };
                let kind = ValueKind::of(field)?;
                locations.insert(
                    member.to_string(),
                    Location::Packed {
                        column: packed.len(),
                        member: members.len(),
                    },
                );
                members.push(PackedMember {
                    field: field.clone(),
                    offset: stride,
                    kind,
                });
                stride += kind.width();
            }
            if members.is_empty() {
                return Err(Error::invalid_input(format!(
                    "packed column {name} would pack no field"
                )));
            }
            packed.push(PackedColumn {
                name,
                stride,
                members,
            });
        }
        if let Some(field) = column_fields
            .iter()
            .find(|field| !locations.contains_key(field.name()))
        {
            return Err(Error::invalid_input(format!(
                "column {} has no plane-row form",
                field.name()
            )));
        }
        for (index, field) in carried.iter().enumerate() {
            if is_internal_column(field.name()) {
                return Err(Error::invalid_input(format!(
                    "column {} cannot be carried unpacked in the plane-row layout",
                    field.name()
                )));
            }
            if locations
                .insert(field.name().clone(), Location::Carried(index))
                .is_some()
            {
                return Err(Error::invalid_input(format!(
                    "column {} appears twice",
                    field.name()
                )));
            }
        }
        let logical = Arc::new(Schema::new_with_metadata(
            column_fields
                .into_iter()
                .chain(carried.iter().cloned())
                .collect::<Vec<_>>(),
            metadata,
        ));
        Ok(Self {
            packed,
            carried,
            logical,
            locations,
        })
    }

    /// The column layout's fields, without the carried ones.
    fn column_fields(&self) -> impl Iterator<Item = &FieldRef> {
        self.logical
            .fields()
            .iter()
            .take(self.logical.fields().len() - self.carried.len())
    }

    /// The packed columns, in file order.
    pub fn packed_columns(&self) -> &[PackedColumn] {
        &self.packed
    }

    /// The columns stored unpacked after the packed ones.
    pub fn carried_fields(&self) -> &[FieldRef] {
        &self.carried
    }

    /// What reads unpack into: the column layout's fields in its order, then
    /// the carried fields, with the file's schema metadata.
    pub fn logical_schema(&self) -> &SchemaRef {
        &self.logical
    }

    /// The fields of a plane-row file: the packed columns, requesting the
    /// full-zip encoding when `fullzip` (see [`requests_fullzip`]), then the
    /// carried columns.
    pub fn file_fields(&self, fullzip: bool) -> Vec<Field> {
        self.packed
            .iter()
            .map(|column| column.file_field(fullzip))
            .chain(self.carried.iter().map(|field| field.as_ref().clone()))
            .collect()
    }

    /// Check that `schema`, a file's, is this layout's: the packed columns,
    /// non-nullable lists of bytes of their strides, then the carried
    /// columns. Field metadata is not checked, as a version 2.0 file requests
    /// no encoding.
    pub fn validate_file_schema(&self, schema: &Schema) -> Result<()> {
        let expected = self.file_fields(false);
        let matches = schema.fields().len() == expected.len()
            && schema
                .fields()
                .iter()
                .zip(&expected)
                .all(|(field, expected)| {
                    field.name() == expected.name()
                        && same_type_but_item_nullability(field.data_type(), expected.data_type())
                        && field.is_nullable() == expected.is_nullable()
                });
        if matches {
            return Ok(());
        }
        let describe = |fields: Vec<(&String, &DataType, bool)>| {
            fields
                .into_iter()
                .map(|(name, data_type, nullable)| {
                    format!("{name}: {data_type}{}", if nullable { "?" } else { "" })
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        Err(Error::index(format!(
            "invalid plane-row IVF_RQ file: its columns are [{}], the layout needs [{}]",
            describe(
                schema
                    .fields()
                    .iter()
                    .map(|field| (field.name(), field.data_type(), field.is_nullable()))
                    .collect()
            ),
            describe(
                expected
                    .iter()
                    .map(|field| (field.name(), field.data_type(), field.is_nullable()))
                    .collect()
            ),
        )))
    }

    /// The file columns a read of the column layout's `columns` reads: the
    /// packed columns that hold them, in file order, then the carried ones.
    /// Every packed column read must be wanted whole, since the plane-row
    /// layout stores a plane only as all its fields.
    pub fn packed_for(&self, columns: &[&str]) -> Result<Vec<&str>> {
        let mut wanted = vec![0usize; self.packed.len()];
        let mut carried = Vec::new();
        let mut seen = HashSet::new();
        for &name in columns {
            if !seen.insert(name) {
                continue;
            }
            match self.location(name)? {
                Location::Packed { column, .. } => wanted[column] += 1,
                Location::Carried(index) => carried.push(index),
            }
        }
        let mut read = Vec::new();
        for (column, count) in self.packed.iter().zip(wanted) {
            if count == 0 {
                continue;
            }
            if count != column.members.len() {
                return Err(Error::invalid_input(format!(
                    "a read of {columns:?} wants only some of the fields packed in {}, which holds {:?}",
                    column.name,
                    column
                        .fields()
                        .map(|field| field.name())
                        .collect::<Vec<_>>()
                )));
            }
            read.push(column.name);
        }
        carried.sort_unstable();
        read.extend(
            carried
                .into_iter()
                .map(|index| self.carried[index].name().as_str()),
        );
        Ok(read)
    }

    /// The column layout's `columns` in that order, as a read of them
    /// returns them, with the file's schema metadata.
    pub fn projection(&self, columns: &[&str]) -> Result<SchemaRef> {
        let fields = columns
            .iter()
            .map(|name| {
                self.location(name)?;
                Ok(self.logical.field_with_name(name)?.clone())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(Schema::new_with_metadata(
            fields,
            self.logical.metadata().clone(),
        )))
    }

    fn location(&self, name: &str) -> Result<Location> {
        self.locations.get(name).copied().ok_or_else(|| {
            Error::invalid_input(format!("the plane-row layout has no column {name}"))
        })
    }

    /// `batch`, rows in the column layout, as plane rows: the packed
    /// columns, then the carried ones. The batch must hold exactly this
    /// layout's fields, by name, without nulls.
    pub fn pack(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        if let Some(field) = batch
            .schema()
            .fields()
            .iter()
            .find(|field| !self.locations.contains_key(field.name()))
        {
            return Err(Error::invalid_input(format!(
                "column {} has no place in the plane-row layout",
                field.name()
            )));
        }
        let rows = batch.num_rows();
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(self.packed.len() + self.carried.len());
        for column in &self.packed {
            let mut bytes = vec![0u8; rows * column.stride];
            for member in &column.members {
                let array = batch.column_by_name(member.field.name()).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "packing {}: the batch has no column {}",
                        column.name,
                        member.field.name()
                    ))
                })?;
                member.pack_into(array.as_ref(), &mut bytes, column.stride)?;
            }
            columns.push(Arc::new(FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::UInt8, false)),
                column.stride as i32,
                Arc::new(UInt8Array::from(bytes)),
                None,
            )?));
        }
        for field in &self.carried {
            let array = batch.column_by_name(field.name()).ok_or_else(|| {
                Error::invalid_input(format!("the batch has no carried column {}", field.name()))
            })?;
            columns.push(array.clone());
        }
        let schema = Arc::new(Schema::new(self.file_fields(false)));
        Ok(RecordBatch::try_new_with_options(
            schema,
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )?)
    }

    /// `schema`'s columns, fields of [`Self::logical_schema`], from
    /// `batches`, consecutive reads of the packed (and carried) columns
    /// that hold them ([`Self::packed_for`]): values in buffers of exactly
    /// their size, as the column layout returns them.
    pub fn unpack(&self, batches: &[RecordBatch], schema: &SchemaRef) -> Result<RecordBatch> {
        let started = Instant::now();
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        let mut unpacked = vec![false; self.packed.len()];
        let mut columns = Vec::with_capacity(schema.fields().len());
        for field in schema.fields() {
            match self.location(field.name())? {
                Location::Packed { column, member } => {
                    let packed = &self.packed[column];
                    let bytes = batches
                        .iter()
                        .map(|batch| packed.rows_of(batch))
                        .collect::<Result<Vec<_>>>()?;
                    columns.push(packed.members[member].unpack(&bytes, packed.stride, rows)?);
                    unpacked[column] = true;
                }
                Location::Carried(_) => {
                    let arrays = batches
                        .iter()
                        .map(|batch| {
                            batch.column_by_name(field.name()).ok_or_else(|| {
                                Error::internal(format!(
                                    "a plane-row read is missing carried column {}",
                                    field.name()
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    columns.push(match arrays.as_slice() {
                        [] => new_empty_array(field.data_type()),
                        [array] => (*array).clone(),
                        arrays => arrow_select::concat::concat(
                            &arrays
                                .iter()
                                .map(|array| array.as_ref())
                                .collect::<Vec<_>>(),
                        )?,
                    });
                }
            }
        }
        let batch = RecordBatch::try_new_with_options(
            schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )?;
        let stats = layered_stats::counters();
        stats.plane_rows_unpack_bytes.add(
            self.packed
                .iter()
                .zip(unpacked)
                .filter(|(_, unpacked)| *unpacked)
                .map(|(column, _)| (rows * column.stride) as u64)
                .sum(),
        );
        stats.plane_rows_unpack_ns.add_elapsed(started);
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::builder::Float32Builder;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rstest::rstest;

    use crate::vector::bq::RQRotationType;
    use crate::vector::bq::layered::{SignBounds, plane_columns};
    use crate::vector::bq::storage::{
        RQRowLayout, RabitQueryEstimator, pack_codes, take_packed_codes,
    };
    use lance_arrow::RecordBatchExt;
    use lance_core::deepsize::{Context, DeepSizeOf};

    fn metadata(
        num_bits: u8,
        dim: usize,
        layered: bool,
        query_estimator: RabitQueryEstimator,
    ) -> RabitQuantizationMetadata {
        RabitQuantizationMetadata {
            rotate_mat: None,
            rotate_mat_position: None,
            fast_rotation_signs: None,
            rotation_type: RQRotationType::Fast,
            code_dim: dim as u32,
            num_bits,
            packed: true,
            layered,
            row_layout: RQRowLayout::PlaneRows,
            query_estimator,
        }
    }

    /// `rows` rows of `schema` with random values, none null.
    fn random_batch(schema: &SchemaRef, rows: usize, seed: u64) -> RecordBatch {
        let mut rng = StdRng::seed_from_u64(seed);
        let columns = schema
            .fields()
            .iter()
            .map(|field| -> ArrayRef {
                match field.data_type() {
                    DataType::UInt64 => Arc::new(UInt64Array::from_iter_values(
                        (0..rows).map(|_| rng.random()),
                    )),
                    DataType::Float32 => Arc::new(Float32Array::from_iter_values(
                        (0..rows).map(|_| rng.random_range(-10.0f32..10.0)),
                    )),
                    DataType::FixedSizeList(item, size) => {
                        let len = rows * *size as usize;
                        let values: ArrayRef = match item.data_type() {
                            DataType::UInt8 => Arc::new(UInt8Array::from_iter_values(
                                (0..len).map(|_| rng.random()),
                            )),
                            DataType::Float32 => Arc::new(Float32Array::from_iter_values(
                                (0..len).map(|_| rng.random_range(-10.0f32..10.0)),
                            )),
                            other => panic!("unexpected item type {other}"),
                        };
                        Arc::new(
                            FixedSizeListArray::try_new(item.clone(), *size, values, None).unwrap(),
                        )
                    }
                    other => panic!("unexpected type {other}"),
                }
            })
            .collect();
        RecordBatch::try_new(schema.clone(), columns).unwrap()
    }

    /// Every stride as the format documentation gives it.
    fn documented_strides(metadata: &RabitQuantizationMetadata) -> Vec<(&'static str, usize)> {
        let dim = metadata.code_dim as usize;
        let sign = dim.div_ceil(8);
        let padded = 64 * dim.div_ceil(64);
        if metadata.layered {
            let layout = crate::vector::bq::layered::RQLayout::try_new(metadata.num_bits).unwrap();
            vec![
                (RQ_SIGN_ROWS_COLUMN, 20 + sign),
                (RQ_BOUNDS_ROWS_COLUMN, 24),
                (
                    RQ_HIGH_ROWS_COLUMN,
                    padded * layout.high_bits as usize / 8 + 8,
                ),
                (
                    RQ_LOW_ROWS_COLUMN,
                    padded * layout.low_bits as usize / 8 + 8,
                ),
            ]
        } else {
            let factors = match metadata.query_estimator {
                RabitQueryEstimator::RawQuery => 3,
                RabitQueryEstimator::ResidualQuery => 2,
            };
            let mut stride = 8 + sign + 4 * factors;
            if metadata.num_bits > 1 {
                stride += padded * (metadata.num_bits as usize - 1) / 8 + 8;
            }
            vec![(RQ_ROWS_COLUMN, stride)]
        }
    }

    fn all_layouts() -> Vec<RabitQuantizationMetadata> {
        let mut layouts = Vec::new();
        for dim in [64, 128, 1024] {
            for num_bits in [1, 5, 7, 9] {
                for estimator in [
                    RabitQueryEstimator::RawQuery,
                    RabitQueryEstimator::ResidualQuery,
                ] {
                    layouts.push(metadata(num_bits, dim, false, estimator));
                }
                if num_bits > 1 {
                    layouts.push(metadata(num_bits, dim, true, RabitQueryEstimator::RawQuery));
                }
            }
        }
        layouts
    }

    /// Packing then unpacking every native and layered layout gives the
    /// column layout's batch back, with buffers of exactly their size, in one
    /// batch or across reads of several, with the documented strides.
    #[test]
    fn plane_rows_round_trip_every_layout() {
        for metadata in all_layouts() {
            let context = format!(
                "num_bits={} dim={} layered={} {:?}",
                metadata.num_bits, metadata.code_dim, metadata.layered, metadata.query_estimator
            );
            let spec = PlaneRowsSpec::try_new(&metadata).unwrap();
            let strides: Vec<_> = spec
                .packed_columns()
                .iter()
                .map(|column| (column.name(), column.stride()))
                .collect();
            assert_eq!(strides, documented_strides(&metadata), "{context}");
            for rows in [0, 1, 37] {
                let batch = random_batch(spec.logical_schema(), rows, rows as u64);
                let packed = spec.pack(&batch).unwrap();
                assert_eq!(packed.num_rows(), rows, "{context}");
                let unpacked = spec
                    .unpack(std::slice::from_ref(&packed), spec.logical_schema())
                    .unwrap();
                assert_eq!(unpacked, batch, "{context}");
                // The arrays plus exactly the values' bytes: a packed row's
                // per row.
                let row_bytes: usize = strides.iter().map(|(_, stride)| stride).sum();
                let arrays = spec
                    .unpack(&[], spec.logical_schema())
                    .unwrap()
                    .deep_size_of_children(&mut Context::new());
                assert_eq!(
                    unpacked.deep_size_of_children(&mut Context::new()),
                    arrays + rows * row_bytes,
                    "{context}"
                );
                if rows > 2 {
                    let split = [packed.slice(0, 2), packed.slice(2, rows - 2)];
                    assert_eq!(
                        spec.unpack(&split, spec.logical_schema()).unwrap(),
                        batch,
                        "{context}"
                    );
                }
            }
        }
    }

    /// The documented example: d = 1024, 7 bits.
    #[test]
    fn plane_rows_strides_match_the_format_example() {
        let layered =
            PlaneRowsSpec::try_new(&metadata(7, 1024, true, RabitQueryEstimator::RawQuery))
                .unwrap();
        let strides: Vec<_> = layered
            .packed_columns()
            .iter()
            .map(|column| column.stride())
            .collect();
        assert_eq!(strides, [148, 24, 520, 264]);
        let native =
            PlaneRowsSpec::try_new(&metadata(7, 1024, false, RabitQueryEstimator::RawQuery))
                .unwrap();
        assert_eq!(native.packed_columns()[0].stride(), 924);
    }

    /// A packed row is its fields' little-endian values concatenated in
    /// order, without padding.
    #[test]
    fn plane_rows_bytes_equal_hand_built_rows() {
        let metadata = metadata(3, 64, false, RabitQueryEstimator::RawQuery);
        let spec = PlaneRowsSpec::try_new(&metadata).unwrap();
        let batch = random_batch(spec.logical_schema(), 2, 7);
        let packed = spec.pack(&batch).unwrap();
        let rows = packed[RQ_ROWS_COLUMN].as_fixed_size_list();
        let rows = rows.values().as_primitive::<UInt8Type>().values();
        let stride = spec.packed_columns()[0].stride();
        for row in 0..2 {
            let mut expected = Vec::new();
            expected.extend_from_slice(
                &batch[ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .value(row)
                    .to_le_bytes(),
            );
            let list_bytes = |name: &str| {
                batch[name]
                    .as_fixed_size_list()
                    .value(row)
                    .as_primitive::<UInt8Type>()
                    .values()
                    .to_vec()
            };
            let factor = |name: &str| {
                batch[name]
                    .as_primitive::<Float32Type>()
                    .value(row)
                    .to_le_bytes()
            };
            expected.extend(list_bytes(RABIT_CODE_COLUMN));
            for name in [
                ADD_FACTORS_COLUMN,
                SCALE_FACTORS_COLUMN,
                ERROR_FACTORS_COLUMN,
            ] {
                expected.extend(factor(name));
            }
            expected.extend(list_bytes(RABIT_BLOCKED_EX_CODE_COLUMN));
            for name in [EX_ADD_FACTORS_COLUMN, EX_SCALE_FACTORS_COLUMN] {
                expected.extend(factor(name));
            }
            assert_eq!(expected.len(), stride);
            assert_eq!(&rows[row * stride..(row + 1) * stride], expected.as_slice());
        }
        // The bounds pack as three little-endian floats each.
        let layered = PlaneRowsSpec::try_new(&metadata_layered()).unwrap();
        let batch = random_batch(layered.logical_schema(), 1, 3);
        let packed = layered.pack(&batch).unwrap();
        let bounds = packed[RQ_BOUNDS_ROWS_COLUMN].as_fixed_size_list();
        let bounds = bounds.values().as_primitive::<UInt8Type>().values();
        let expected: Vec<u8> = [HIGH_BOUNDS_COLUMN, FULL_BOUNDS_COLUMN]
            .iter()
            .flat_map(|name| {
                batch[*name]
                    .as_fixed_size_list()
                    .values()
                    .as_primitive::<Float32Type>()
                    .values()
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(bounds, expected.as_slice());
    }

    fn metadata_layered() -> RabitQuantizationMetadata {
        metadata(7, 128, true, RabitQueryEstimator::RawQuery)
    }

    /// Sign codes keep their partition-local transposition: the bytes of
    /// the packed buffer survive, and selecting rows from them still agrees.
    #[test]
    fn plane_rows_keep_transposed_sign_codes() {
        let spec = PlaneRowsSpec::try_new(&metadata_layered()).unwrap();
        let mut batch = random_batch(spec.logical_schema(), 45, 11);
        let transposed = pack_codes(batch[RABIT_CODE_COLUMN].as_fixed_size_list());
        batch = batch
            .replace_column_by_name(RABIT_CODE_COLUMN, Arc::new(transposed.clone()))
            .unwrap();
        let unpacked = spec
            .unpack(&[spec.pack(&batch).unwrap()], spec.logical_schema())
            .unwrap();
        let codes = unpacked[RABIT_CODE_COLUMN].as_fixed_size_list();
        assert_eq!(codes, &transposed);
        let rows = [0, 3, 31, 32, 44];
        assert_eq!(
            take_packed_codes(codes, &rows).unwrap(),
            take_packed_codes(&transposed, &rows).unwrap()
        );
    }

    #[test]
    fn plane_rows_reject_nulls_bad_widths_and_types() {
        let spec = PlaneRowsSpec::try_new(&metadata_layered()).unwrap();
        let batch = random_batch(spec.logical_schema(), 4, 5);

        let mut nulls = Float32Builder::new();
        nulls.append_value(1.0);
        nulls.append_null();
        nulls.append_value(1.0);
        nulls.append_value(1.0);
        let with_nulls = RecordBatch::try_new(
            Arc::new(Schema::new(
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.as_ref().clone().with_nullable(true))
                    .collect::<Vec<_>>(),
            )),
            batch.columns().to_vec(),
        )
        .unwrap()
        .replace_column_by_name(ADD_FACTORS_COLUMN, Arc::new(nulls.finish()))
        .unwrap();
        let err = spec.pack(&with_nulls).unwrap_err().to_string();
        assert!(err.contains("nulls"), "{err}");

        let code_bytes = metadata_layered().binary_code_bytes();
        let wide_codes = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::UInt8, true)),
            code_bytes as i32 + 1,
            Arc::new(UInt8Array::from(vec![0u8; 4 * (code_bytes + 1)])),
            None,
        )
        .unwrap();
        let mut fields: Vec<Field> = batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect();
        let mut columns = batch.columns().to_vec();
        let code_index = batch.schema().index_of(RABIT_CODE_COLUMN).unwrap();
        fields[code_index] = Field::new(RABIT_CODE_COLUMN, wide_codes.data_type().clone(), true);
        columns[code_index] = Arc::new(wide_codes);
        let wide =
            RecordBatch::try_new(Arc::new(Schema::new(fields.clone())), columns.clone()).unwrap();
        assert!(spec.pack(&wide).is_err());

        let row_index = batch.schema().index_of(ROW_ID).unwrap();
        fields[code_index] = batch.schema().field(code_index).clone();
        columns[code_index] = batch.column(code_index).clone();
        fields[row_index] = Field::new(ROW_ID, DataType::Int64, true);
        columns[row_index] = Arc::new(arrow_array::Int64Array::from(vec![1i64; 4]));
        let wrong_type = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        assert!(spec.pack(&wrong_type).is_err());

        let extra = batch
            .clone()
            .try_with_column(
                Field::new("other", DataType::UInt64, true),
                Arc::new(UInt64Array::from(vec![1u64; 4])),
            )
            .unwrap();
        assert!(spec.pack(&extra).is_err());
        assert!(
            spec.pack(&batch.drop_column(HIGH_BOUNDS_COLUMN).unwrap())
                .is_err()
        );

        // A field type without a plane-row form.
        let field = Field::new("utf8", DataType::Utf8, true);
        assert!(ValueKind::of(&field).is_err());
    }

    /// A read maps whole planes to their packed columns, and no read can
    /// take only part of a plane.
    #[test]
    fn plane_rows_packed_for_planes_and_rejects_partial_covers() {
        let spec = PlaneRowsSpec::try_new(&metadata_layered()).unwrap();
        let cases = [
            (0, SignBounds::Lazy, vec![RQ_SIGN_ROWS_COLUMN]),
            (
                0,
                SignBounds::Eager,
                vec![RQ_SIGN_ROWS_COLUMN, RQ_BOUNDS_ROWS_COLUMN],
            ),
            (1, SignBounds::Lazy, vec![RQ_HIGH_ROWS_COLUMN]),
            (2, SignBounds::Eager, vec![RQ_LOW_ROWS_COLUMN]),
            (3, SignBounds::Lazy, vec![RQ_BOUNDS_ROWS_COLUMN]),
        ];
        for (plane, bounds, expected) in cases {
            let columns = plane_columns(plane, bounds);
            assert_eq!(spec.packed_for(columns).unwrap(), expected);
            let projection = spec.projection(columns).unwrap();
            let names: Vec<_> = projection
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect();
            assert_eq!(names, columns);
        }
        let all: Vec<&str> = spec
            .logical_schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect();
        assert_eq!(spec.packed_for(&all).unwrap().len(), 4);

        let err = spec
            .packed_for(&[ROW_ID, RABIT_CODE_COLUMN])
            .unwrap_err()
            .to_string();
        assert!(err.contains(RQ_SIGN_ROWS_COLUMN), "{err}");
        assert!(spec.packed_for(&[RABIT_BLOCKED_EX_CODE_COLUMN]).is_err());
        assert!(spec.packed_for(&["missing"]).is_err());
        assert!(spec.projection(&["missing"]).is_err());
    }

    #[test]
    fn plane_rows_file_fields_request_fullzip() {
        let spec = PlaneRowsSpec::try_new(&metadata_layered()).unwrap();
        for field in spec.file_fields(true) {
            assert!(!field.is_nullable());
            let DataType::FixedSizeList(item, _) = field.data_type() else {
                panic!("{field:?}");
            };
            assert_eq!(item.data_type(), &DataType::UInt8);
            assert!(!item.is_nullable());
            assert_eq!(
                field.metadata().get(STRUCTURAL_ENCODING_META_KEY),
                Some(&STRUCTURAL_ENCODING_FULLZIP.to_string())
            );
            assert_eq!(
                field.metadata().get(COMPRESSION_META_KEY),
                Some(&NO_COMPRESSION.to_string())
            );
        }
        assert!(
            spec.file_fields(false)
                .iter()
                .all(|field| field.metadata().is_empty())
        );
        assert!(!requests_fullzip(ConcreteFileVersion::V2_0));
        assert!(requests_fullzip(ConcreteFileVersion::V2_1));
        assert!(requests_fullzip(ConcreteFileVersion::V2_2));
    }

    /// A plane-row file must have exactly the packed columns its metadata
    /// implies, then carried columns; anything else is an invalid index.
    #[rstest]
    #[case::native(false)]
    #[case::layered(true)]
    fn plane_rows_validate_file_schemas(#[case] layered: bool) {
        let metadata = metadata(7, 128, layered, RabitQueryEstimator::RawQuery);
        let spec = PlaneRowsSpec::try_new(&metadata).unwrap();
        let carried = Field::new("carried", DataType::Utf8, true);
        let mut fields = spec.file_fields(true);
        fields.push(carried.clone());
        let file = Schema::new_with_metadata(
            fields.clone(),
            HashMap::from([("key".to_string(), "value".to_string())]),
        );
        let read = PlaneRowsSpec::for_file(&metadata, &file).unwrap();
        // A Lance file reads list items back as nullable.
        let read_back: Vec<Field> = fields
            .iter()
            .map(|field| match field.data_type() {
                DataType::FixedSizeList(item, size) => Field::new(
                    field.name(),
                    DataType::FixedSizeList(
                        Arc::new(item.as_ref().clone().with_nullable(true)),
                        *size,
                    ),
                    field.is_nullable(),
                ),
                _ => field.clone(),
            })
            .collect();
        PlaneRowsSpec::for_file(&metadata, &Schema::new(read_back)).unwrap();
        assert_eq!(read.carried_fields().len(), 1);
        assert_eq!(read.logical_schema().metadata(), file.metadata());
        assert_eq!(
            read.logical_schema().fields().last().unwrap().as_ref(),
            &carried
        );
        assert_eq!(
            read.packed_for(&["carried"]).unwrap(),
            vec!["carried"],
            "a carried column reads alone"
        );

        let mut wrong_stride = fields.clone();
        wrong_stride[0] = Field::new(
            wrong_stride[0].name(),
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt8, false)), 3),
            false,
        );
        let mut nullable = fields.clone();
        nullable[0] = nullable[0].clone().with_nullable(true);
        let mut column_layout: Vec<Field> = spec
            .logical_schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect();
        column_layout.push(carried);
        let mut legacy = fields.clone();
        legacy.push(Field::new(RABIT_CODE_COLUMN, DataType::UInt8, true));
        for (case, fields) in [
            ("wrong stride", wrong_stride),
            ("nullable", nullable),
            ("column layout", column_layout),
            ("an internal column carried", legacy),
            ("no columns", Vec::new()),
        ] {
            let err = PlaneRowsSpec::for_file(&metadata, &Schema::new(fields)).unwrap_err();
            assert!(matches!(err, Error::Index { .. }), "{case}: {err:?}");
        }
    }

    /// Converting needs the column layout's fields exactly as the metadata
    /// implies them; other columns are carried, and the legacy sequential ex
    /// codes have no plane-row form.
    #[test]
    fn plane_rows_for_column_files() {
        let metadata = metadata(7, 128, false, RabitQueryEstimator::RawQuery);
        let spec = PlaneRowsSpec::try_new(&metadata).unwrap();
        let mut fields: Vec<Field> = spec
            .logical_schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect();
        let carried = Field::new("carried", DataType::Float32, true);
        fields.push(carried.clone());
        let read = PlaneRowsSpec::for_column_file(&metadata, &Schema::new(fields.clone())).unwrap();
        assert_eq!(read.carried_fields()[0].as_ref(), &carried);
        let batch = random_batch(read.logical_schema(), 5, 9);
        let packed = read.pack(&batch).unwrap();
        assert_eq!(packed.schema().fields().len(), 2);
        assert_eq!(
            read.unpack(&[packed], read.logical_schema()).unwrap(),
            batch
        );

        let mut legacy = fields.clone();
        let ex = legacy
            .iter()
            .position(|field| field.name() == RABIT_BLOCKED_EX_CODE_COLUMN)
            .unwrap();
        legacy[ex] = Field::new(RABIT_EX_CODE_COLUMN, legacy[ex].data_type().clone(), true);
        assert!(PlaneRowsSpec::for_column_file(&metadata, &Schema::new(legacy)).is_err());

        let mut retyped = fields.clone();
        retyped[0] = Field::new(ROW_ID, DataType::UInt64, false);
        assert!(PlaneRowsSpec::for_column_file(&metadata, &Schema::new(retyped)).is_err());

        let mut reordered = fields;
        reordered.swap(2, 3);
        assert!(PlaneRowsSpec::for_column_file(&metadata, &Schema::new(reordered)).is_err());
    }

    /// The unpack counters advance by the packed bytes each read unpacks.
    #[test]
    fn plane_rows_unpack_counts_packed_bytes() {
        let spec = PlaneRowsSpec::try_new(&metadata_layered()).unwrap();
        let batch = random_batch(spec.logical_schema(), 10, 1);
        let packed = spec.pack(&batch).unwrap();
        let before = layered_stats::counters().plane_rows_unpack_bytes.get();
        let columns = plane_columns(1, SignBounds::Lazy);
        spec.unpack(&[packed], &spec.projection(columns).unwrap())
            .unwrap();
        let high = spec.packed_columns()[2].stride() * 10;
        assert!(layered_stats::counters().plane_rows_unpack_bytes.get() >= before + high as u64);
    }
}
