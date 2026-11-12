// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One fragment per row, with ordered files and overlays and dictionary-encoded mappings.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, MutexGuard};

use arrow_array::builder::BinaryBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, BinaryArray, DictionaryArray, Int32Array, ListArray, RecordBatch, StringArray,
    StructArray, UInt32Array, UInt64Array,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, Fields, Schema};
use lance_core::{Error, Result};
use lance_io::utils::CachedFileSize;
use prost::Message;

use super::validation::{corrupt, fragment};
use crate::format::{DataFile, DataFileFieldInterner, Fragment, pb};

static FILE_FIELDS: LazyLock<Fields> = LazyLock::new(|| {
    let mapping = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Binary));
    vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("field_ids", mapping.clone(), false),
        Field::new("column_indices", mapping, false),
        Field::new("major_version", DataType::UInt32, false),
        Field::new("minor_version", DataType::UInt32, false),
        Field::new("file_size_bytes", DataType::UInt64, false),
        Field::new("base_id", DataType::UInt32, true),
    ]
    .into()
});

static OVERLAY_FIELDS: LazyLock<Fields> = LazyLock::new(|| {
    vec![
        Field::new("data_file", DataType::Struct(FILE_FIELDS.clone()), false),
        Field::new("overlay_meta", DataType::Binary, false),
    ]
    .into()
});

static LEAF_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("fragment_meta", DataType::Binary, false),
        Field::new(
            "files",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(FILE_FIELDS.clone()),
                false,
            ))),
            false,
        ),
        Field::new(
            "overlays",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(OVERLAY_FIELDS.clone()),
                false,
            ))),
            false,
        ),
    ]))
});

pub(super) fn schema() -> Arc<Schema> {
    Arc::clone(&LEAF_SCHEMA)
}

/// Dictionary-code field-id or column-index lists in order of first
/// appearance. Interned lists repeat the same allocation, so most files are
/// matched by pointer before their contents are compared.
fn encode_mappings<'a>(lists: impl ExactSizeIterator<Item = &'a Arc<[i32]>>) -> Result<ArrayRef> {
    let capacity = lists.len();
    let mut by_pointer = HashMap::<(*const i32, usize), i32>::new();
    let mut by_contents = HashMap::<&'a [i32], i32>::new();
    let mut values = BinaryBuilder::new();
    let mut keys = Vec::with_capacity(capacity);
    for list in lists {
        let pointer = (list.as_ptr(), list.len());
        let key = match by_pointer.get(&pointer) {
            Some(key) => *key,
            None => {
                let next = i32::try_from(by_contents.len())
                    .map_err(|_| corrupt("leaf has more than i32::MAX distinct mappings"))?;
                let key = *by_contents.entry(list.as_ref()).or_insert_with(|| {
                    values.append_value(
                        list.iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<_>>(),
                    );
                    next
                });
                by_pointer.insert(pointer, key);
                key
            }
        };
        keys.push(key);
    }
    Ok(Arc::new(DictionaryArray::try_new(
        Int32Array::from(keys),
        Arc::new(values.finish()),
    )?))
}

fn files_array(files: &[&DataFile]) -> Result<StructArray> {
    Ok(StructArray::try_new(
        FILE_FIELDS.clone(),
        vec![
            Arc::new(StringArray::from_iter_values(
                files.iter().map(|f| f.path.as_str()),
            )),
            encode_mappings(files.iter().map(|f| &f.fields))?,
            encode_mappings(files.iter().map(|f| &f.column_indices))?,
            Arc::new(UInt32Array::from_iter_values(
                files.iter().map(|f| f.file_major_version),
            )),
            Arc::new(UInt32Array::from_iter_values(
                files.iter().map(|f| f.file_minor_version),
            )),
            Arc::new(UInt64Array::from_iter_values(
                files
                    .iter()
                    .map(|f| f.file_size_bytes.get().map_or(0, |size| size.get())),
            )),
            Arc::new(UInt32Array::from(
                files.iter().map(|f| f.base_id).collect::<Vec<_>>(),
            )),
        ],
        None,
    )?)
}

pub(super) fn encode(fragments: &[Fragment]) -> Result<RecordBatch> {
    let mut headers = Vec::with_capacity(fragments.len());
    let mut files = Vec::new();
    let mut overlays = Vec::new();
    let mut overlay_headers = Vec::new();
    let mut file_lengths = Vec::with_capacity(fragments.len());
    let mut overlay_lengths = Vec::with_capacity(fragments.len());
    for fragment in fragments {
        // Files and overlays are stored as columns, so the header omits them
        // rather than converting them to protobuf only to discard them.
        let shell = Fragment {
            id: 0,
            ..fragment_scalars(fragment)
        };
        file_lengths.push(fragment.files.len());
        overlay_lengths.push(fragment.overlays.len());
        files.extend(&fragment.files);
        for overlay in &fragment.overlays {
            let mut header = pb::DataOverlayFile::from(overlay);
            header.data_file = None;
            overlays.push(&overlay.data_file);
            overlay_headers.push(header.encode_to_vec());
        }
        headers.push(pb::DataFragment::from(&shell).encode_to_vec());
    }
    let file_values = Arc::new(files_array(&files)?);
    let overlay_values = Arc::new(StructArray::try_new(
        OVERLAY_FIELDS.clone(),
        vec![
            Arc::new(files_array(&overlays)?),
            Arc::new(BinaryArray::from_iter_values(
                overlay_headers.iter().map(Vec::as_slice),
            )),
        ],
        None,
    )?);
    let lists = |values: ArrayRef, lengths: Vec<usize>| -> Result<ArrayRef> {
        Ok(Arc::new(ListArray::try_new(
            Arc::new(Field::new("item", values.data_type().clone(), false)),
            OffsetBuffer::from_lengths(lengths),
            values,
            None,
        )?))
    };
    Ok(RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(UInt64Array::from_iter_values(
                fragments.iter().map(|f| f.id),
            )),
            Arc::new(BinaryArray::from_iter_values(
                headers.iter().map(Vec::as_slice),
            )),
            lists(file_values, file_lengths)?,
            lists(overlay_values, overlay_lengths)?,
        ],
    )?)
}

/// Every field of `fragment` except its files and overlays.
fn fragment_scalars(fragment: &Fragment) -> Fragment {
    Fragment {
        id: fragment.id,
        files: Vec::new(),
        overlays: Vec::new(),
        deletion_file: fragment.deletion_file.clone(),
        row_id_meta: fragment.row_id_meta.clone(),
        physical_rows: fragment.physical_rows,
        last_updated_at_version_meta: fragment.last_updated_at_version_meta.clone(),
        created_at_version_meta: fragment.created_at_version_meta.clone(),
        clustering_id: fragment.clustering_id,
        metadata: fragment.metadata.clone(),
    }
}

/// Checks the leaf format invariants on `array` and its children. Non-nullable
/// fields have no nulls, mapping byte lengths divide by 4, and dictionary keys
/// are in bounds. Decoding indexes dictionaries and reads mappings without
/// repeating these checks.
fn validate_column(field: &Field, array: &dyn Array) -> Result<()> {
    if !field.is_nullable() && array.null_count() != 0 {
        return Err(corrupt(format!(
            "leaf field {} contains nulls",
            field.name()
        )));
    }
    match field.data_type() {
        DataType::Struct(fields) => {
            for (field, column) in fields.iter().zip(array.as_struct().columns()) {
                validate_column(field, column.as_ref())?;
            }
        }
        DataType::List(item) => validate_column(item, array.as_list::<i32>().values().as_ref())?,
        DataType::Dictionary(_, _) => {
            let dictionary = array.as_dictionary::<Int32Type>();
            if dictionary.values().null_count() != 0 {
                return Err(corrupt(format!(
                    "leaf dictionary {} contains null values",
                    field.name()
                )));
            }
            for value in dictionary.values().as_binary::<i32>().iter().flatten() {
                if !value.len().is_multiple_of(4) {
                    return Err(corrupt(format!(
                        "leaf mapping has {} bytes; expected a multiple of 4",
                        value.len()
                    )));
                }
            }
            for key in dictionary.keys().values() {
                if *key < 0 || *key as usize >= dictionary.values().len() {
                    return Err(corrupt(format!(
                        "leaf dictionary {} has invalid key {key}",
                        field.name()
                    )));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn decode_mapping_at(array: &dyn Array, row: usize) -> Vec<i32> {
    let dictionary = array.as_dictionary::<Int32Type>();
    let key = dictionary.keys().value(row);
    decode_mapping_bytes(dictionary.values().as_binary::<i32>().value(key as usize))
}

/// Width was checked by [`validate_column`], since `as_chunks` drops trailing bytes.
fn decode_mapping_bytes(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect()
}

/// The flattened data files of one leaf batch. Each distinct field-id and
/// column-index mapping is decoded and interned once, so files that share a
/// mapping share its allocation without a per-file copy or comparison.
struct FileColumns<'a> {
    paths: &'a StringArray,
    field_keys: &'a Int32Array,
    column_keys: &'a Int32Array,
    fields: Vec<Arc<[i32]>>,
    column_indices: Vec<Arc<[i32]>>,
    major_versions: &'a UInt32Array,
    minor_versions: &'a UInt32Array,
    sizes: &'a UInt64Array,
    bases: &'a UInt32Array,
}

impl<'a> FileColumns<'a> {
    fn new(files: &'a StructArray, interner: &mut DataFileFieldInterner) -> Self {
        let columns = files.columns();
        let field_dictionary = columns[1].as_dictionary::<Int32Type>();
        let column_dictionary = columns[2].as_dictionary::<Int32Type>();
        Self {
            paths: columns[0].as_string::<i32>(),
            field_keys: field_dictionary.keys(),
            column_keys: column_dictionary.keys(),
            fields: intern_mappings(field_dictionary, |ids| interner.intern_field_ids(ids)),
            column_indices: intern_mappings(column_dictionary, |ids| {
                interner.intern_column_indices(ids)
            }),
            major_versions: columns[3].as_primitive::<UInt32Type>(),
            minor_versions: columns[4].as_primitive::<UInt32Type>(),
            sizes: columns[5].as_primitive::<UInt64Type>(),
            bases: columns[6].as_primitive::<UInt32Type>(),
        }
    }

    /// Keys were range-checked against their dictionaries by [`validate_column`].
    fn data_file(&self, row: usize) -> DataFile {
        DataFile {
            path: self.paths.value(row).to_owned(),
            fields: self.fields[self.field_keys.value(row) as usize].clone(),
            column_indices: self.column_indices[self.column_keys.value(row) as usize].clone(),
            file_major_version: self.major_versions.value(row),
            file_minor_version: self.minor_versions.value(row),
            file_size_bytes: CachedFileSize::new(self.sizes.value(row)),
            // Leaves have no column for the hint yet, so it reads as unknown.
            file_metadata_size_bytes: None,
            base_id: (!self.bases.is_null(row)).then(|| self.bases.value(row)),
        }
    }
}

fn intern_mappings(
    dictionary: &DictionaryArray<Int32Type>,
    mut intern: impl FnMut(&[i32]) -> Arc<[i32]>,
) -> Vec<Arc<[i32]>> {
    let values = dictionary.values().as_binary::<i32>();
    (0..values.len())
        .map(|index| intern(&decode_mapping_bytes(values.value(index))))
        .collect()
}

fn file(array: &StructArray, row: usize) -> pb::DataFile {
    let columns = array.columns();
    let bases = columns[6].as_primitive::<UInt32Type>();
    pb::DataFile {
        path: columns[0].as_string::<i32>().value(row).to_owned(),
        fields: decode_mapping_at(columns[1].as_ref(), row),
        column_indices: decode_mapping_at(columns[2].as_ref(), row),
        file_major_version: columns[3].as_primitive::<UInt32Type>().value(row),
        file_minor_version: columns[4].as_primitive::<UInt32Type>().value(row),
        file_size_bytes: columns[5].as_primitive::<UInt64Type>().value(row),
        base_id: (!bases.is_null(row)).then(|| bases.value(row)),
        file_metadata_size_bytes: 0,
    }
}

fn decode_overlays(overlays: &ListArray, row: usize, id: u64) -> Result<Vec<pb::DataOverlayFile>> {
    if overlays.value_length(row) == 0 {
        return Ok(Vec::new());
    }
    let values = overlays.value(row);
    let values = values.as_struct();
    (0..values.len())
        .map(|i| {
            let bytes = values.columns()[1].as_binary::<i32>().value(i);
            let mut overlay = pb::DataOverlayFile::decode(bytes)
                .map_err(|e| corrupt(format!("invalid overlay_meta for fragment {id}: {e}")))?;
            if overlay.data_file.is_some() {
                return Err(corrupt("overlay_meta must not contain data_file"));
            }
            overlay.data_file = Some(file(values.columns()[0].as_struct(), i));
            Ok(overlay)
        })
        .collect()
}

/// `interner` is locked only while shared mappings are interned, so leaves
/// decoding on other threads do not wait for this batch.
pub(super) fn decode<'a>(
    batch: &RecordBatch,
    interner: impl Fn() -> MutexGuard<'a, DataFileFieldInterner>,
) -> Result<Vec<Fragment>> {
    if batch.schema().fields() != schema().fields() {
        return Err(corrupt(format!(
            "unexpected leaf schema: {:?}",
            batch.schema()
        )));
    }
    for (field, column) in batch.schema().fields().iter().zip(batch.columns()) {
        validate_column(field, column.as_ref())?;
    }
    let columns = batch.columns();
    let ids = columns[0].as_primitive::<UInt64Type>();
    let headers = columns[1].as_binary::<i32>();
    let files = columns[2].as_list::<i32>();
    let file_offsets = files.value_offsets();
    let file_columns = FileColumns::new(files.values().as_struct(), &mut interner());
    let overlays = columns[3].as_list::<i32>();
    let mut fragments = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let id = ids.value(row);
        if id > u32::MAX as u64 || (row > 0 && id <= ids.value(row - 1)) {
            return Err(corrupt(format!(
                "leaf fragment ID {id} is out of range or order"
            )));
        }
        let mut header = decode_header(id, headers.value(row))?;
        header.overlays = decode_overlays(overlays, row, id)?;
        let mut decoded = fragment(header)?;
        let (start, end) = (file_offsets[row] as usize, file_offsets[row + 1] as usize);
        decoded.files = (start..end).map(|i| file_columns.data_file(i)).collect();
        if !decoded.overlays.is_empty() {
            let mut interner = interner();
            for overlay in &mut decoded.overlays {
                let file = &mut overlay.data_file;
                file.fields = interner.intern_field_ids(&file.fields);
                file.column_indices = interner.intern_column_indices(&file.column_indices);
            }
        }
        // The object read here is the leaf, which the reader names, so the
        // data file moves from the error's path into its message.
        for file in decoded.referenced_lance_files() {
            file.validate(&object_store::path::Path::default())
                .map_err(|error| match error {
                    Error::CorruptFile { source, .. } => {
                        corrupt(format!("{}: {source}", file.path))
                    }
                    error => error,
                })?;
        }
        fragments.push(decoded);
    }
    Ok(fragments)
}

/// Decode a stored `fragment_meta`, which holds every field of a record except
/// its id, files and overlays.
fn decode_header(id: u64, bytes: &[u8]) -> Result<pb::DataFragment> {
    let mut header = pb::DataFragment::decode(bytes)
        .map_err(|e| corrupt(format!("invalid fragment_meta for fragment {id}: {e}")))?;
    if header.id != 0 || !header.files.is_empty() || !header.overlays.is_empty() {
        return Err(corrupt(format!(
            "fragment_meta for fragment {id} must have id 0 and empty files and overlays"
        )));
    }
    header.id = id;
    Ok(header)
}

/// The header-only projection of a leaf, read from its id and fragment_meta
/// columns and skipping files and overlays, along with the leaf's row totals.
pub(super) struct LeafHeaders<T> {
    pub(super) kept: Vec<T>,
    pub(super) first_id: Option<u64>,
    pub(super) total_rows: u64,
    pub(super) visible_rows: u64,
    pub(super) last_id: Option<u64>,
}

impl<T> LeafHeaders<T> {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            kept: Vec::with_capacity(capacity),
            first_id: None,
            total_rows: 0,
            visible_rows: 0,
            last_id: None,
        }
    }

    pub(super) fn extend(
        &mut self,
        batch: &RecordBatch,
        keep: impl Fn(Fragment) -> T,
    ) -> Result<()> {
        let leaf = schema();
        if batch.schema().fields()[..] != leaf.fields()[..2] {
            return Err(corrupt(format!(
                "unexpected leaf key schema: {:?}",
                batch.schema()
            )));
        }
        for (field, column) in leaf.fields().iter().zip(batch.columns()) {
            validate_column(field, column.as_ref())?;
        }
        let ids = batch.column(0).as_primitive::<UInt64Type>();
        let headers = batch.column(1).as_binary::<i32>();
        for row in 0..batch.num_rows() {
            let id = ids.value(row);
            if id > u64::from(u32::MAX) || self.last_id.is_some_and(|previous| previous >= id) {
                return Err(corrupt(format!(
                    "leaf fragment ID {id} is out of range or order"
                )));
            }
            // Count exactly as node::leaf_ref does, so a leaf that fails a
            // whole read also fails here.
            let header = fragment(decode_header(id, headers.value(row))?)?;
            self.total_rows = self
                .total_rows
                .checked_add(header.physical_rows.unwrap_or(0) as u64)
                .ok_or_else(|| corrupt(format!("leaf total_rows overflows at fragment {id}")))?;
            self.visible_rows = self
                .visible_rows
                .checked_add(header.num_rows().unwrap_or(0) as u64)
                .ok_or_else(|| corrupt(format!("leaf visible_rows overflows at fragment {id}")))?;
            self.first_id.get_or_insert(id);
            self.last_id = Some(id);
            self.kept.push(keep(header));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::BinaryDictionaryBuilder;
    use std::sync::Mutex;

    fn decode_leaf(batch: &RecordBatch) -> Result<Vec<Fragment>> {
        let interner = Mutex::default();
        decode(batch, || interner.lock().unwrap())
    }
    use crate::fragment_metadata::support::make_fragment;
    use rstest::rstest;

    /// Version decides overlay precedence and position only breaks ties, so
    /// a decoded leaf record lists overlays in the order flat decoding does.
    #[test]
    fn decoded_overlays_follow_flat_precedence_order() {
        let file = make_fragment(7).files[0].clone();
        let overlay = |committed_version: u64, path: &str| {
            let mut data_file = file.clone();
            data_file.path = path.into();
            crate::format::overlay::DataOverlayFile {
                data_file,
                coverage: crate::format::overlay::OverlayCoverage::dense(
                    roaring::RoaringBitmap::from_iter([0u32]),
                ),
                committed_version,
            }
        };
        // Another writer may store overlays in any order.
        let mut stored = make_fragment(7);
        stored.overlays = vec![
            overlay(3, "newer.lance"),
            overlay(1, "older.lance"),
            overlay(1, "older-tie.lance"),
        ];
        let paths = |record: &Fragment| {
            record
                .overlays
                .iter()
                .map(|overlay| overlay.data_file.path.clone())
                .collect::<Vec<_>>()
        };
        let flat = Fragment::try_from(pb::DataFragment::from(&stored)).unwrap();
        assert_eq!(
            paths(&flat),
            ["older.lance", "older-tie.lance", "newer.lance"]
        );
        let decoded = decode_leaf(&encode(&[stored]).unwrap()).unwrap();
        assert_eq!(paths(&decoded[0]), paths(&flat));
    }

    #[test]
    fn nested_leaf_preserves_file_order_overlays_mappings_and_object_metadata() {
        let mut value = pb::DataFragment::from(&make_fragment(7));
        value.clustering_id = Some(3);
        value
            .metadata
            .insert("org.example.cluster.bucket".into(), "[0, 100)".into());
        value.files[0].fields = vec![-1, 2];
        value.files[0].column_indices = vec![-1, 0];
        value.files[0].base_id = Some(0);
        value.files.push(value.files[0].clone());
        value.overlays.push(pb::DataOverlayFile::from(
            &crate::format::overlay::DataOverlayFile {
                data_file: crate::format::DataFile::try_from(value.files[0].clone()).unwrap(),
                coverage: crate::format::overlay::OverlayCoverage::dense(
                    roaring::RoaringBitmap::new(),
                ),
                committed_version: 3,
            },
        ));
        let mut tied = value.overlays[0].clone();
        let tied_file = tied.data_file.as_mut().unwrap();
        tied_file.path = "tied.lance".into();
        value.overlays.push(tied);
        let original = fragment(value).unwrap();
        // Overlays of one version keep their stored order.
        assert_eq!(original.overlays[1].data_file.path, "tied.lance");
        let batch = encode(std::slice::from_ref(&original)).unwrap();
        assert_eq!(batch.num_rows(), 1);
        let files = batch.column(2).as_list::<i32>().values().as_struct();
        let mappings = files.column(1).as_dictionary::<Int32Type>();
        assert_eq!(mappings.keys().values().as_ref(), &[0, 0]);
        assert_eq!(
            mappings.values().as_binary::<i32>().value(0),
            [(-1_i32).to_le_bytes(), 2_i32.to_le_bytes()].concat()
        );
        assert_eq!(decode_leaf(&batch).unwrap(), vec![original]);
    }

    #[test]
    fn file_and_overlay_metadata_size_hints_decode_as_unknown() {
        let mut stored = make_fragment(7);
        stored.files[0].file_metadata_size_bytes = std::num::NonZero::new(1_024);
        let mut data_file = stored.files[0].clone();
        data_file.path = "overlay.lance".into();
        stored.overlays = vec![crate::format::overlay::DataOverlayFile {
            data_file,
            coverage: crate::format::overlay::OverlayCoverage::dense(
                roaring::RoaringBitmap::from_iter([0u32]),
            ),
            committed_version: 3,
        }];
        let decoded = decode_leaf(&encode(std::slice::from_ref(&stored)).unwrap()).unwrap();
        stored.files[0].file_metadata_size_bytes = None;
        stored.overlays[0].data_file.file_metadata_size_bytes = None;
        assert_eq!(decoded, vec![stored]);
    }

    #[rstest]
    #[case::id("id")]
    #[case::files("files")]
    #[case::overlays("overlays")]
    fn rejects_duplicated_header_fields(#[case] populated: &str) {
        let original = make_fragment(7);
        let batch = encode(std::slice::from_ref(&original)).unwrap();
        let mut header = pb::DataFragment::default();
        match populated {
            "id" => header.id = 7,
            "files" => header.files.push(Default::default()),
            "overlays" => header.overlays.push(Default::default()),
            other => unreachable!("no field named {other}"),
        }
        let bytes = header.encode_to_vec();
        let mut columns = batch.columns().to_vec();
        columns[1] = Arc::new(BinaryArray::from_iter_values([bytes.as_slice()]));
        let malformed = RecordBatch::try_new(schema(), columns).unwrap();
        let whole = decode_leaf(&malformed).unwrap_err();
        assert!(matches!(whole, Error::CorruptFile { .. }), "{whole}");
        let projected = malformed.project(&[0, 1]).unwrap();
        let header = LeafHeaders::with_capacity(1)
            .extend(&projected, |header| header)
            .unwrap_err();
        assert_eq!(whole.to_string(), header.to_string());
    }

    #[rstest]
    #[case::short(vec![1, 2, 3])]
    #[case::empty(vec![])]
    fn mapping_values_have_exact_i32_width(#[case] value: Vec<u8>) {
        let mut builder = BinaryDictionaryBuilder::<Int32Type>::new();
        builder.append(&value).unwrap();
        let array = builder.finish();
        let field = Field::new("field_ids", array.data_type().clone(), false);
        let result = validate_column(&field, &array);
        if value.is_empty() {
            result.unwrap();
            assert!(decode_mapping_at(&array, 0).is_empty());
        } else {
            assert!(matches!(result, Err(Error::CorruptFile { .. })));
        }
    }

    /// `make_fragment`'s field ids as a leaf stores them.
    const STORED_FIELD_IDS: [u8; 8] = [0, 0, 0, 0, 1, 0, 0, 0];

    /// `as_chunks` drops trailing bytes and overlays decode only the mappings
    /// their rows reference, so only the validation in `decode` sees these.
    #[rstest]
    #[case::referenced_file_mapping(vec![vec![1, 2, 3]], vec![STORED_FIELD_IDS.to_vec()])]
    #[case::unreferenced_overlay_mapping(
        vec![STORED_FIELD_IDS.to_vec()],
        vec![STORED_FIELD_IDS.to_vec(), vec![1, 2, 3]],
    )]
    fn decode_rejects_mappings_not_divisible_by_4(
        #[case] file_mappings: Vec<Vec<u8>>,
        #[case] overlay_mappings: Vec<Vec<u8>>,
    ) {
        let mut stored = make_fragment(0);
        stored.overlays = vec![crate::format::overlay::DataOverlayFile {
            data_file: stored.files[0].clone(),
            coverage: crate::format::overlay::OverlayCoverage::dense(
                roaring::RoaringBitmap::from_iter([0u32]),
            ),
            committed_version: 3,
        }];
        let batch = encode(&[stored]).unwrap();
        // Every row keeps key 0, so later dictionary entries stay unreferenced.
        let with_field_mappings = |files: &StructArray, mappings: Vec<Vec<u8>>| {
            let keys = Int32Array::from(vec![0; files.len()]);
            let values = BinaryArray::from_iter_values(mappings);
            let mut columns = files.columns().to_vec();
            columns[1] = Arc::new(DictionaryArray::try_new(keys, Arc::new(values)).unwrap());
            StructArray::new(FILE_FIELDS.clone(), columns, None)
        };
        let relist = |list: &ListArray, values: StructArray| -> ArrayRef {
            let item = Arc::new(Field::new("item", values.data_type().clone(), false));
            Arc::new(ListArray::new(
                item,
                list.offsets().clone(),
                Arc::new(values),
                None,
            ))
        };
        let files = batch.column(2).as_list::<i32>();
        let overlays = batch.column(3).as_list::<i32>();
        let mut overlay_columns = overlays.values().as_struct().columns().to_vec();
        overlay_columns[0] = Arc::new(with_field_mappings(
            overlay_columns[0].as_struct(),
            overlay_mappings,
        ));
        let mut columns = batch.columns().to_vec();
        columns[2] = relist(
            files,
            with_field_mappings(files.values().as_struct(), file_mappings),
        );
        columns[3] = relist(
            overlays,
            StructArray::new(OVERLAY_FIELDS.clone(), overlay_columns, None),
        );
        let malformed = RecordBatch::try_new(schema(), columns).unwrap();
        let error = decode_leaf(&malformed).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains("multiple of 4"), "{error}");
    }

    #[rstest]
    #[case::past_the_values(1)]
    #[case::negative(-1)]
    fn rejects_dictionary_keys_outside_the_values(#[case] key: i32) {
        let values = BinaryArray::from_iter_values([[0_u8; 4].as_slice()]);
        // SAFETY: Only validate_column reads this array, and it checks each key
        // without following it into the values.
        let array = unsafe {
            DictionaryArray::<Int32Type>::new_unchecked(
                Int32Array::from(vec![key]),
                Arc::new(values),
            )
        };
        let field = Field::new("field_ids", array.data_type().clone(), false);
        let error = validate_column(&field, &array).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(error.to_string().contains("invalid key"), "{error}");
    }

    #[rstest]
    #[case::key(true)]
    #[case::value(false)]
    fn rejects_null_dictionary_entries(#[case] null_key: bool) {
        let keys = arrow_array::Int32Array::from(vec![if null_key { None } else { Some(0) }]);
        let values = BinaryArray::from(vec![None::<&[u8]>]);
        let array =
            arrow_array::DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();
        let field = Field::new("field_ids", array.data_type().clone(), false);
        let error = validate_column(&field, &array).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("null"));
    }

    #[test]
    fn rejects_incomplete_file_mapping() {
        let mut original = make_fragment(0);
        original.files[0].fields = Arc::from([0, 1]);
        original.files[0].column_indices = Arc::from([0]);
        let error = decode_leaf(&encode(&[original]).unwrap()).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("fewer column_indices"));
    }

    #[rstest]
    #[case::duplicate(vec![7, 7])]
    #[case::descending(vec![7, 6])]
    #[case::overflow(vec![u32::MAX as u64 + 1])]
    fn rejects_invalid_ids(#[case] ids: Vec<u64>) {
        let fragments = ids
            .iter()
            .map(|id| {
                let mut f = make_fragment(0);
                f.id = *id;
                f
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            decode_leaf(&encode(&fragments).unwrap()),
            Err(Error::CorruptFile { .. })
        ));
    }
}
