// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Keeps the dictionary columns of each data file within their declared key type.
//!
//! The 2.1+ encoder merges the dictionaries of the batches it buffers for a page, and
//! rejects a page whose merged dictionary needs more indices than the declared key type
//! addresses. Batches that each fit can still do that together, for example two
//! `Dictionary(Int8, _)` fragments of 100 values each that compaction rewrites into one
//! file. Cutting pages instead would not help: a read batch that spans two pages merges
//! their dictionaries too. The file reader decodes each data file on its own, so the
//! writer finishes a file before its dictionary columns would hold more distinct values
//! than their keys address, and continues in a new one.

use std::collections::{HashSet, VecDeque};

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_row::{RowConverter, SortField};
use arrow_schema::{DataType, Schema as ArrowSchema};
use lance_core::Result;
use lance_encoding::encodings::logical::primitive::dict::encoded_dictionary_values;

/// A 128-bit BLAKE3 digest of a value's row encoding.
///
/// The tracker keeps one digest per distinct value instead of the value itself, so its
/// memory is bounded by the key capacity (at most 65,536 digests of 16 bytes per column)
/// rather than by the size of the values. Two distinct values sharing a digest would be
/// counted once; the encoder's own check then rejects the page, as it does without the
/// tracker.
type ValueDigest = u128;

fn digest(row: &[u8]) -> ValueDigest {
    let mut bytes = [0; 16];
    bytes.copy_from_slice(&blake3::hash(row).as_bytes()[..16]);
    ValueDigest::from_le_bytes(bytes)
}

/// How to reach a nested dictionary array from its parent array.
#[derive(Debug, Clone)]
enum PathStep {
    /// A named child of a struct
    Child(String),
    /// The items of a list
    Items,
}

/// The distinct values one dictionary column holds in the current file.
struct TrackedDictionary {
    /// Name of the top-level column holding the dictionary
    column: String,
    /// Steps from that column to the dictionary array, empty for a top-level dictionary
    path: Vec<PathStep>,
    /// Number of distinct values the declared key type addresses
    capacity: usize,
    /// Declared value type, which `converter` encodes
    value_type: DataType,
    converter: RowConverter,
    /// Digests of the values written to the current file
    values: HashSet<ValueDigest>,
}

/// Rows of leading batches that fit the current file, see [`DictionaryValueBudget::admit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Admission {
    /// Rows the caller can write to the current file
    pub rows: usize,
    /// Whether a batch was turned away because the file's dictionaries are full
    pub is_full: bool,
}

/// Tracks the distinct values each narrow-key dictionary column holds in the data file
/// being written.
pub struct DictionaryValueBudget {
    dictionaries: Vec<TrackedDictionary>,
}

impl DictionaryValueBudget {
    /// Returns `None` when `schema` has no dictionary whose key type is narrower than
    /// 32 bits, since wider keys never overflow in practice.
    pub fn try_new(schema: &ArrowSchema) -> Result<Option<Self>> {
        let mut dictionaries = Vec::new();
        for field in schema.fields() {
            collect_dictionaries(
                field.name(),
                field.data_type(),
                &mut Vec::new(),
                &mut dictionaries,
            )?;
        }
        Ok((!dictionaries.is_empty()).then_some(Self { dictionaries }))
    }

    /// Admits the leading `batches`, up to `max_rows` rows, whose dictionary values still
    /// fit the current file, and records their values.
    ///
    /// Returns `max_rows` when every batch up to that limit fits. The first batch of an
    /// empty file is always admitted, so a writer that finishes the file whenever
    /// [`Admission::is_full`] is set always makes progress; if that batch alone does not
    /// fit, the encoder reports the column.
    pub fn admit(&mut self, batches: &VecDeque<RecordBatch>, max_rows: usize) -> Result<Admission> {
        let mut rows = 0;
        for batch in batches {
            if rows >= max_rows {
                break;
            }
            // The writer drops empty batches, so their values never reach a file.
            if batch.num_rows() == 0 {
                continue;
            }
            let new_values = self
                .dictionaries
                .iter()
                .map(|dictionary| dictionary.new_values(batch))
                .collect::<Result<Vec<_>>>()?;
            let fits = self
                .dictionaries
                .iter()
                .zip(&new_values)
                .all(|(dictionary, new)| {
                    dictionary.values.len() + new.len() <= dictionary.capacity
                });
            if !fits && (rows > 0 || !self.is_empty()) {
                return Ok(Admission {
                    rows,
                    is_full: true,
                });
            }
            for (dictionary, new) in self.dictionaries.iter_mut().zip(new_values) {
                dictionary.values.extend(new);
            }
            rows += batch.num_rows();
        }
        Ok(Admission {
            rows: max_rows,
            is_full: false,
        })
    }

    /// Forgets the values of the finished file.
    pub fn reset(&mut self) {
        for dictionary in &mut self.dictionaries {
            dictionary.values.clear();
        }
    }

    fn is_empty(&self) -> bool {
        self.dictionaries
            .iter()
            .all(|dictionary| dictionary.values.is_empty())
    }
}

/// Number of distinct values a dictionary key type addresses, for keys narrower than 32 bits.
fn key_capacity(key_type: &DataType) -> Option<usize> {
    match key_type {
        DataType::Int8 => Some(i8::MAX as usize + 1),
        DataType::UInt8 => Some(u8::MAX as usize + 1),
        DataType::Int16 => Some(i16::MAX as usize + 1),
        DataType::UInt16 => Some(u16::MAX as usize + 1),
        _ => None,
    }
}

/// Whether the encoder can merge dictionaries of this value type exactly, so a page fits
/// as long as its distinct values fit the key type.
///
/// Arrow merges the dictionaries of these types when concatenating a page's chunks would
/// overflow the key type, and the encoder deduplicates them exactly when that merge still
/// overflows. Columns of
/// other value types are left untracked and keep the encoder's own check; Lance schemas
/// reject dictionaries of most of them anyway.
fn is_merged_exactly(value_type: &DataType) -> bool {
    matches!(
        value_type,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
    )
}

fn collect_dictionaries(
    column: &str,
    data_type: &DataType,
    path: &mut Vec<PathStep>,
    dictionaries: &mut Vec<TrackedDictionary>,
) -> Result<()> {
    match data_type {
        DataType::Dictionary(key_type, value_type) => {
            let Some(capacity) = key_capacity(key_type) else {
                return Ok(());
            };
            if !is_merged_exactly(value_type) {
                return Ok(());
            }
            let converter = RowConverter::new(vec![SortField::new(value_type.as_ref().clone())])?;
            dictionaries.push(TrackedDictionary {
                column: column.to_string(),
                path: path.clone(),
                capacity,
                value_type: value_type.as_ref().clone(),
                converter,
                values: HashSet::new(),
            });
        }
        DataType::Struct(fields) => {
            for field in fields {
                path.push(PathStep::Child(field.name().clone()));
                collect_dictionaries(column, field.data_type(), path, dictionaries)?;
                path.pop();
            }
        }
        DataType::List(field) | DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            path.push(PathStep::Items);
            collect_dictionaries(column, field.data_type(), path, dictionaries)?;
            path.pop();
        }
        _ => {}
    }
    Ok(())
}

impl TrackedDictionary {
    /// Values of `batch` that the current file does not hold yet.
    fn new_values(&self, batch: &RecordBatch) -> Result<HashSet<ValueDigest>> {
        let Some(array) = self.locate(batch) else {
            return Ok(HashSet::new());
        };
        if array.as_any_dictionary_opt().is_none() {
            return Ok(HashSet::new());
        }
        let mut values = encoded_dictionary_values(array)?;
        // A batch can carry a view type the schema records as its offset counterpart.
        if values.data_type() != &self.value_type {
            values = arrow_cast::cast(&values, &self.value_type)?;
        }
        let rows = self.converter.convert_columns(&[values])?;
        Ok(rows
            .iter()
            .map(|row| digest(row.as_ref()))
            .filter(|digest| !self.values.contains(digest))
            .collect())
    }

    fn locate(&self, batch: &RecordBatch) -> Option<ArrayRef> {
        let mut array = batch.column_by_name(&self.column)?.clone();
        for step in &self.path {
            array = match step {
                PathStep::Child(name) => array.as_struct_opt()?.column_by_name(name)?.clone(),
                PathStep::Items => match array.data_type() {
                    DataType::List(_) => array.as_list_opt::<i32>()?.values().clone(),
                    DataType::LargeList(_) => array.as_list_opt::<i64>()?.values().clone(),
                    DataType::FixedSizeList(_, _) => {
                        array.as_fixed_size_list_opt()?.values().clone()
                    }
                    _ => return None,
                },
            };
        }
        Some(array)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::types::{Int8Type, Int16Type, Int32Type};
    use arrow_array::{
        DictionaryArray, Int8Array, Int16Array, Int32Array, ListArray, StringArray,
        StringViewArray, StructArray,
    };
    use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
    use arrow_schema::{Field, Fields};
    use rstest::rstest;

    use super::*;

    /// A batch whose top-level, struct-child and list-item dictionaries each reference
    /// the 100 values starting at `start`, plus an untracked Int32-key dictionary.
    fn nested_batch(start: usize) -> RecordBatch {
        let values =
            || StringArray::from_iter_values((start..start + 100).map(|i| format!("v{i}")));
        let dictionary = || {
            let keys = (0..100).collect::<Int8Array>();
            Arc::new(DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values())).unwrap())
                as ArrayRef
        };
        let data_type = dictionary().data_type().clone();
        let list = ListArray::new(
            Arc::new(Field::new("item", data_type.clone(), true)),
            OffsetBuffer::from_lengths(std::iter::repeat_n(1, 100)),
            dictionary(),
            None,
        );
        let child = Arc::new(Field::new("d", data_type, true));
        let nested = StructArray::new(Fields::from(vec![child]), vec![dictionary()], None);
        let wide_keys = (0..100).collect::<Int32Array>();
        let wide = DictionaryArray::<Int32Type>::try_new(wide_keys, Arc::new(values())).unwrap();
        RecordBatch::try_from_iter([
            ("top", dictionary()),
            ("nested", Arc::new(nested) as ArrayRef),
            ("list", Arc::new(list) as ArrayRef),
            ("wide", Arc::new(wide) as ArrayRef),
        ])
        .unwrap()
    }

    #[test]
    fn tracks_narrow_dictionaries_at_any_depth() {
        let schema = nested_batch(0).schema();
        let budget = DictionaryValueBudget::try_new(&schema).unwrap().unwrap();
        let tracked = budget
            .dictionaries
            .iter()
            .map(|dictionary| (dictionary.column.as_str(), dictionary.path.len()))
            .collect::<Vec<_>>();
        assert_eq!(tracked, [("top", 0), ("nested", 1), ("list", 1)]);

        let wide_only = ArrowSchema::new(vec![schema.field(3).clone()]);
        assert!(
            DictionaryValueBudget::try_new(&wide_only)
                .unwrap()
                .is_none()
        );
    }

    #[rstest]
    #[case::top_level(0)]
    #[case::struct_child(1)]
    #[case::list_items(2)]
    fn admits_batches_until_a_dictionary_is_full(#[case] column: usize) {
        let project = |batch: RecordBatch| batch.project(&[column]).unwrap();
        let first = project(nested_batch(0));
        let mut budget = DictionaryValueBudget::try_new(&first.schema())
            .unwrap()
            .unwrap();
        // The first two batches hold v0..v127, exactly the 128 values Int8 keys address;
        // the third would add 100 more.
        let batches =
            VecDeque::from([first, project(nested_batch(28)), project(nested_batch(128))]);
        let full = Admission {
            rows: 200,
            is_full: true,
        };
        assert_eq!(budget.admit(&batches, usize::MAX).unwrap(), full);

        budget.reset();
        let rest = VecDeque::from([project(nested_batch(128))]);
        let admitted = Admission {
            rows: usize::MAX,
            is_full: false,
        };
        assert_eq!(budget.admit(&rest, usize::MAX).unwrap(), admitted);
    }

    #[test]
    fn admits_up_to_max_rows() {
        let batches = VecDeque::from([nested_batch(0), nested_batch(28), nested_batch(128)]);
        let mut budget = DictionaryValueBudget::try_new(&nested_batch(0).schema())
            .unwrap()
            .unwrap();
        // The cap ends admission inside the second batch, before the third is looked at.
        let admitted = Admission {
            rows: 150,
            is_full: false,
        };
        assert_eq!(budget.admit(&batches, 150).unwrap(), admitted);
    }

    /// A single top-level Int8 dictionary column `d`.
    fn dictionary_batch(
        values: ArrayRef,
        keys: Vec<i8>,
        validity: Option<Vec<bool>>,
    ) -> RecordBatch {
        let keys = Int8Array::new(ScalarBuffer::from(keys), validity.map(NullBuffer::from));
        let dictionary = DictionaryArray::<Int8Type>::try_new(keys, values).unwrap();
        RecordBatch::try_from_iter([("d", Arc::new(dictionary) as ArrayRef)]).unwrap()
    }

    fn strings(prefix: &str, count: usize) -> ArrayRef {
        Arc::new(StringArray::from_iter_values(
            (0..count).map(|i| format!("{prefix}{i}")),
        ))
    }

    /// 65 new values, which together with 64 values already in the file overflow Int8.
    fn second_batch() -> RecordBatch {
        dictionary_batch(strings("b", 65), (0..65).collect(), None)
    }

    #[rstest]
    // The null slot's physical key 63 selects a value no valid key references; the encoder
    // drops key validity, so the page holds it.
    #[case::key_under_null(
        (0..64).collect(),
        Some((0..64).map(|key| key != 63).collect())
    )]
    // Index 0 is always counted: a neighboring all-null chunk without values can point its
    // keys at it.
    #[case::unreferenced_first_value((1..64).collect(), None)]
    fn counts_values_the_encoder_stores(
        #[case] keys: Vec<i8>,
        #[case] validity: Option<Vec<bool>>,
    ) {
        let first = dictionary_batch(strings("a", 64), keys, validity);
        let full = Admission {
            rows: first.num_rows(),
            is_full: true,
        };
        let mut budget = DictionaryValueBudget::try_new(&first.schema())
            .unwrap()
            .unwrap();
        let batches = VecDeque::from([first, second_batch()]);
        assert_eq!(budget.admit(&batches, usize::MAX).unwrap(), full);
    }

    #[test]
    fn skips_empty_batches() {
        let empty = dictionary_batch(strings("x", 1), vec![], None);
        let first = dictionary_batch(strings("a", 63), (0..63).collect(), None);
        let mut budget = DictionaryValueBudget::try_new(&first.schema())
            .unwrap()
            .unwrap();
        // Counting the empty batch's first value would leave room for only 64 more.
        let batches = VecDeque::from([empty, first, second_batch()]);
        let admitted = Admission {
            rows: usize::MAX,
            is_full: false,
        };
        assert_eq!(budget.admit(&batches, usize::MAX).unwrap(), admitted);
    }

    #[test]
    fn counts_view_values_as_their_declared_type() {
        let schema = ArrowSchema::new(vec![Field::new(
            "d",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            true,
        )]);
        let mut budget = DictionaryValueBudget::try_new(&schema).unwrap().unwrap();
        let views = Arc::new(StringViewArray::from_iter_values(
            (0..100).map(|i| format!("v{i}")),
        ));
        let batch = dictionary_batch(views, (0..100).collect(), None);
        // Equal values in separate batches count once.
        let batches = VecDeque::from([batch.clone(), batch]);
        let admitted = Admission {
            rows: usize::MAX,
            is_full: false,
        };
        assert_eq!(budget.admit(&batches, usize::MAX).unwrap(), admitted);
    }

    #[test]
    fn leaves_value_types_the_encoder_cannot_merge_exactly_untracked() {
        let schema = ArrowSchema::new(vec![Field::new(
            "d",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Boolean)),
            true,
        )]);
        assert!(DictionaryValueBudget::try_new(&schema).unwrap().is_none());
    }

    #[test]
    fn keeps_one_fixed_size_digest_per_value() {
        // Four batches of 64 values of 16 KiB each that differ only in their last characters.
        let batch = |batch_index: usize| {
            let values = StringArray::from_iter_values(
                (0..64).map(|entry| format!("{}{batch_index:04}:{entry:04}", "x".repeat(16_374))),
            );
            let keys = (0..64).collect::<Int16Array>();
            let dictionary = DictionaryArray::<Int16Type>::try_new(keys, Arc::new(values)).unwrap();
            RecordBatch::try_from_iter([("d", Arc::new(dictionary) as ArrayRef)]).unwrap()
        };
        let mut budget = DictionaryValueBudget::try_new(&batch(0).schema())
            .unwrap()
            .unwrap();
        for batch_index in 0..4 {
            let batches = VecDeque::from([batch(batch_index)]);
            assert!(!budget.admit(&batches, usize::MAX).unwrap().is_full);
        }
        // Every value is still told apart, but only its 16-byte digest is retained.
        let values: &HashSet<ValueDigest> = &budget.dictionaries[0].values;
        assert_eq!(values.len(), 256);
        assert_eq!(size_of::<ValueDigest>(), 16);
    }
}
