// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Key existence tracking for merge insert conflict detection.
//!
//! A merge insert records the join keys it inserted into a bloom filter, which
//! is carried in the transaction so a concurrent commit can detect whether it
//! inserted any of the same keys. The filter is serialized into the transaction
//! protobuf, so it lives at the table layer next to [`crate::format::pb`].

use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::format::pb_actions;
use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BinaryArray, LargeBinaryArray, LargeListArray, LargeStringArray, ListArray,
    RecordBatch, StringArray, StructArray,
};
use arrow_schema::DataType;
use lance_core::Result;
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::bloomfilter::sbbf::{Sbbf, SbbfBuilder};

// Default bloom filter config: 8192 items @ 0.00057 fpp -> 16KiB filter
pub const BLOOM_FILTER_DEFAULT_NUMBER_OF_ITEMS: u64 = 8192;
pub const BLOOM_FILTER_DEFAULT_PROBABILITY: f64 = 0.00057;

/// Key value for conflict detection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KeyValue {
    String(String),
    Int64(i64),
    UInt64(u64),
    Binary(Vec<u8>),
    List(Vec<Self>),
    Struct(Vec<Self>),
    Composite(Vec<Self>),
}

impl KeyValue {
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::String(s) => s.as_bytes().to_vec(),
            Self::Int64(i) => i.to_le_bytes().to_vec(),
            Self::UInt64(u) => u.to_le_bytes().to_vec(),
            Self::Binary(b) => b.clone(),
            Self::List(values) | Self::Struct(values) | Self::Composite(values) => {
                let mut result = Vec::new();
                for value in values {
                    result.extend_from_slice(&value.to_bytes());
                    result.push(0);
                }
                result
            }
        }
    }

    pub fn hash_value(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.to_bytes().hash(&mut hasher);
        hasher.finish()
    }
}

/// Builder for KeyExistenceFilter using Split Block Bloom Filter.
#[derive(Debug, Clone)]
pub struct KeyExistenceFilterBuilder {
    sbbf: Sbbf,
    field_ids: Vec<i32>,
    item_count: usize,
}

impl KeyExistenceFilterBuilder {
    pub fn new(field_ids: Vec<i32>) -> Self {
        let sbbf = SbbfBuilder::new()
            .expected_items(BLOOM_FILTER_DEFAULT_NUMBER_OF_ITEMS)
            .false_positive_probability(BLOOM_FILTER_DEFAULT_PROBABILITY)
            .build()
            .expect("Failed to build SBBF");
        Self {
            sbbf,
            field_ids,
            item_count: 0,
        }
    }

    pub fn insert(&mut self, key: KeyValue) -> Result<()> {
        self.sbbf.insert(&key.to_bytes()[..]);
        self.item_count += 1;
        Ok(())
    }

    pub fn contains(&self, key: &KeyValue) -> bool {
        self.sbbf.check(&key.to_bytes()[..])
    }

    pub fn might_intersect(&self, other: &Self) -> Result<bool> {
        self.sbbf
            .might_intersect(&other.sbbf)
            .map_err(|e| lance_core::Error::invalid_input(e.to_string()))
    }

    pub fn field_ids(&self) -> &[i32] {
        &self.field_ids
    }

    pub fn estimated_size_bytes(&self) -> usize {
        self.sbbf.size_bytes()
    }

    pub fn len(&self) -> usize {
        self.item_count
    }

    pub fn is_empty(&self) -> bool {
        self.item_count == 0
    }

    pub fn build(&self) -> KeyExistenceFilter {
        KeyExistenceFilter {
            field_ids: self.field_ids.clone(),
            filter: FilterType::Bloom {
                bitmap: self.sbbf.to_bytes(),
                num_bits: (self.sbbf.size_bytes() as u32) * 8,
                number_of_items: BLOOM_FILTER_DEFAULT_NUMBER_OF_ITEMS,
                probability: BLOOM_FILTER_DEFAULT_PROBABILITY,
            },
        }
    }
}

impl From<&KeyExistenceFilterBuilder> for pb_actions::KeyExistenceFilter {
    fn from(builder: &KeyExistenceFilterBuilder) -> Self {
        Self {
            field_ids: builder.field_ids.clone(),
            data: Some(pb_actions::key_existence_filter::Data::Bloom(
                pb_actions::BloomFilter {
                    bitmap: builder.sbbf.to_bytes(),
                    num_bits: (builder.sbbf.size_bytes() as u32) * 8,
                    number_of_items: BLOOM_FILTER_DEFAULT_NUMBER_OF_ITEMS,
                    probability: BLOOM_FILTER_DEFAULT_PROBABILITY,
                },
            )),
        }
    }
}

/// Filter type for key existence data.
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub enum FilterType {
    ExactSet(HashSet<u64>),
    Bloom {
        bitmap: Vec<u8>,
        num_bits: u32,
        number_of_items: u64,
        probability: f64,
    },
}

/// Tracks keys of inserted rows for conflict detection.
/// Only created when ON columns match the schema's unenforced primary key.
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct KeyExistenceFilter {
    pub field_ids: Vec<i32>,
    pub filter: FilterType,
}

enum ParsedKeyExistenceFilter<'a> {
    ExactSet(&'a HashSet<u64>),
    Bloom(Sbbf),
}

impl ParsedKeyExistenceFilter<'_> {
    fn might_contain(&self, key: &KeyValue) -> bool {
        match self {
            Self::ExactSet(hashes) => hashes.contains(&key.hash_value()),
            Self::Bloom(sbbf) => sbbf.check(&key.to_bytes()[..]),
        }
    }
}

impl KeyExistenceFilter {
    pub fn from_bloom_filter(bloom: &KeyExistenceFilterBuilder) -> Self {
        bloom.build()
    }

    /// Validate the filter data before exposing it through the Python bridge.
    pub fn validate(&self) -> Result<()> {
        self.parse().map(|_| ())
    }

    fn parse(&self) -> Result<ParsedKeyExistenceFilter<'_>> {
        match &self.filter {
            FilterType::ExactSet(hashes) => Ok(ParsedKeyExistenceFilter::ExactSet(hashes)),
            FilterType::Bloom { bitmap, .. } => {
                // `Sbbf::new` accepts an empty bitset, but `Sbbf::check` indexes
                // the block array, so an empty payload must fail here instead.
                if bitmap.is_empty() {
                    return Err(lance_core::Error::invalid_input(
                        "inserted_rows_filter bloom bitmap must not be empty",
                    ));
                }
                let sbbf = Sbbf::new(bitmap).map_err(|err| {
                    lance_core::Error::invalid_input(format!(
                        "inserted_rows_filter bloom bitmap is invalid: {err}"
                    ))
                })?;
                Ok(ParsedKeyExistenceFilter::Bloom(sbbf))
            }
        }
    }

    /// Check if two filters intersect. Returns (has_intersection, might_be_false_positive).
    /// Errors if bloom filter configs don't match.
    pub fn intersects(&self, other: &Self) -> Result<(bool, bool)> {
        match (&self.filter, &other.filter) {
            (FilterType::ExactSet(a), FilterType::ExactSet(b)) => {
                Ok((a.iter().any(|h| b.contains(h)), false))
            }
            (FilterType::ExactSet(_), FilterType::Bloom { .. })
            | (FilterType::Bloom { .. }, FilterType::ExactSet(_)) => {
                // Can't compare different hash schemes, assume intersection
                Ok((true, true))
            }
            (
                FilterType::Bloom {
                    bitmap: a_bits,
                    number_of_items: a_num_items,
                    probability: a_prob,
                    ..
                },
                FilterType::Bloom {
                    bitmap: b_bits,
                    number_of_items: b_num_items,
                    probability: b_prob,
                    ..
                },
            ) => {
                if a_num_items != b_num_items || (a_prob - b_prob).abs() > f64::EPSILON {
                    return Err(lance_core::Error::invalid_input(format!(
                        "Bloom filter config mismatch: ({}, {}) vs ({}, {})",
                        a_num_items, a_prob, b_num_items, b_prob
                    )));
                }
                let has = Sbbf::bytes_might_intersect(a_bits, b_bits)
                    .map_err(|e| lance_core::Error::invalid_input(e.to_string()))?;
                Ok((has, has))
            }
        }
    }

    /// Whether `key` might have been recorded in this filter.
    ///
    /// Exact sets compare hashed keys, so a different key with the same hash
    /// also matches. Bloom filters never return a false negative, but they can
    /// return a false positive.
    pub fn might_contain(&self, key: &KeyValue) -> Result<bool> {
        Ok(self.parse()?.might_contain(key))
    }

    /// Check one key per row. `columns` are in `field_ids` order.
    ///
    /// A null in any key column is reported as absent.
    pub fn might_contain_columns(&self, columns: &[ArrayRef]) -> Result<Vec<bool>> {
        if columns.is_empty() {
            return Err(lance_core::Error::invalid_input(
                "might_contain requires at least one key column".to_string(),
            ));
        }
        if columns.len() != self.field_ids.len() {
            return Err(lance_core::Error::invalid_input(format!(
                "might_contain expected {} key column(s) for field ids {:?}, got {}",
                self.field_ids.len(),
                self.field_ids,
                columns.len()
            )));
        }
        let num_rows = columns[0].len();
        if columns.iter().any(|column| column.len() != num_rows) {
            return Err(lance_core::Error::invalid_input(
                "might_contain columns must have the same number of rows".to_string(),
            ));
        }

        // Parse the bloom filter once for the whole batch instead of rebuilding
        // it for every row.
        let parsed = self.parse()?;
        let mut present = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            if columns.iter().any(|column| column.is_null(row_idx)) {
                present.push(false);
                continue;
            }
            let mut parts = Vec::with_capacity(columns.len());
            for column in columns {
                let Some(part) = extract_key_value(column.as_ref(), row_idx) else {
                    return Err(lance_core::Error::invalid_input(format!(
                        "might_contain: unsupported key type {} at row {row_idx}",
                        column.data_type()
                    )));
                };
                parts.push(part);
            }
            let key = if parts.len() == 1 {
                parts.swap_remove(0)
            } else {
                KeyValue::Composite(parts)
            };
            present.push(parsed.might_contain(&key));
        }
        Ok(present)
    }
}

impl From<&KeyExistenceFilter> for pb_actions::KeyExistenceFilter {
    fn from(filter: &KeyExistenceFilter) -> Self {
        match &filter.filter {
            FilterType::ExactSet(hashes) => Self {
                field_ids: filter.field_ids.clone(),
                data: Some(pb_actions::key_existence_filter::Data::Exact(
                    pb_actions::ExactKeySetFilter {
                        key_hashes: hashes.iter().copied().collect(),
                    },
                )),
            },
            FilterType::Bloom {
                bitmap,
                num_bits,
                number_of_items,
                probability,
            } => Self {
                field_ids: filter.field_ids.clone(),
                data: Some(pb_actions::key_existence_filter::Data::Bloom(
                    pb_actions::BloomFilter {
                        bitmap: bitmap.clone(),
                        num_bits: *num_bits,
                        number_of_items: *number_of_items,
                        probability: *probability,
                    },
                )),
            },
        }
    }
}

impl TryFrom<&pb_actions::KeyExistenceFilter> for KeyExistenceFilter {
    type Error = lance_core::Error;

    fn try_from(message: &pb_actions::KeyExistenceFilter) -> Result<Self> {
        let filter = match message.data.as_ref() {
            Some(pb_actions::key_existence_filter::Data::Exact(exact)) => {
                FilterType::ExactSet(exact.key_hashes.iter().copied().collect())
            }
            Some(pb_actions::key_existence_filter::Data::Bloom(b)) => {
                // Use defaults for backwards compatibility
                let number_of_items = if b.number_of_items == 0 {
                    BLOOM_FILTER_DEFAULT_NUMBER_OF_ITEMS
                } else {
                    b.number_of_items
                };
                let probability = if b.probability == 0.0 {
                    BLOOM_FILTER_DEFAULT_PROBABILITY
                } else {
                    b.probability
                };
                FilterType::Bloom {
                    bitmap: b.bitmap.clone(),
                    num_bits: b.num_bits,
                    number_of_items,
                    probability,
                }
            }
            None => FilterType::ExactSet(HashSet::new()),
        };
        Ok(Self {
            field_ids: message.field_ids.clone(),
            filter,
        })
    }
}

/// Extract key value from a batch row. Returns None if null or unsupported type.
pub fn extract_key_value_from_batch(
    batch: &RecordBatch,
    row_idx: usize,
    on_columns: &[String],
) -> Option<KeyValue> {
    let mut parts: Vec<KeyValue> = Vec::with_capacity(on_columns.len());

    for col_name in on_columns {
        let (col_idx, _) = batch.schema().column_with_name(col_name)?;
        let column = batch.column(col_idx);

        if column.is_null(row_idx) {
            return None;
        }

        let key_part = extract_key_value(column, row_idx)?;
        parts.push(key_part);
    }

    if parts.is_empty() {
        None
    } else if parts.len() == 1 {
        Some(parts.into_iter().next().unwrap())
    } else {
        Some(KeyValue::Composite(parts))
    }
}

fn extract_key_value(array: &dyn Array, row_idx: usize) -> Option<KeyValue> {
    let v = match array.data_type() {
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>()?;
            KeyValue::String(arr.value(row_idx).to_string())
        }
        DataType::LargeUtf8 => {
            let arr = array.as_any().downcast_ref::<LargeStringArray>()?;
            KeyValue::String(arr.value(row_idx).to_string())
        }
        DataType::UInt64 => {
            let arr = array.as_primitive::<arrow_array::types::UInt64Type>();
            KeyValue::UInt64(arr.value(row_idx))
        }
        DataType::Int64 => {
            let arr = array.as_primitive::<arrow_array::types::Int64Type>();
            KeyValue::Int64(arr.value(row_idx))
        }
        DataType::UInt32 => {
            let arr = array.as_primitive::<arrow_array::types::UInt32Type>();
            KeyValue::UInt64(arr.value(row_idx) as u64)
        }
        DataType::Int32 => {
            let arr = array.as_primitive::<arrow_array::types::Int32Type>();
            KeyValue::Int64(arr.value(row_idx) as i64)
        }
        DataType::Binary => {
            let arr = array.as_any().downcast_ref::<BinaryArray>()?;
            KeyValue::Binary(arr.value(row_idx).to_vec())
        }
        DataType::LargeBinary => {
            let arr = array.as_any().downcast_ref::<LargeBinaryArray>()?;
            KeyValue::Binary(arr.value(row_idx).to_vec())
        }
        DataType::FixedSizeBinary(_) => {
            let arr = array.as_fixed_size_binary();
            KeyValue::Binary(arr.value(row_idx).to_vec())
        }
        DataType::List(_) => {
            let list_array = array.as_any().downcast_ref::<ListArray>().unwrap();
            let values = list_array.value(row_idx);

            let mut elements = Vec::with_capacity(values.len());
            for i in 0..values.len() {
                if values.is_null(i) {
                    return None;
                }
                let element = extract_key_value(&values, i)?;
                elements.push(element);
            }
            KeyValue::List(elements)
        }
        DataType::LargeList(_) => {
            let list_array = array.as_any().downcast_ref::<LargeListArray>().unwrap();
            let values = list_array.value(row_idx);

            let mut elements = Vec::with_capacity(values.len());
            for i in 0..values.len() {
                if values.is_null(i) {
                    return None;
                }
                let element = extract_key_value(&values, i)?;
                elements.push(element);
            }
            KeyValue::List(elements)
        }
        DataType::Struct(_) => {
            let struct_array = array.as_any().downcast_ref::<StructArray>()?;
            let mut elements = Vec::with_capacity(struct_array.num_columns());
            for i in 0..struct_array.num_columns() {
                let child = struct_array.column(i);
                if child.is_null(row_idx) {
                    return None;
                }
                let field_value = extract_key_value(child.as_ref(), row_idx)?;
                elements.push(field_value);
            }
            KeyValue::Struct(elements)
        }
        _ => return None,
    };
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::builder::{Int32Builder, ListBuilder, StringBuilder};
    use arrow_array::{FixedSizeBinaryArray, Int32Array, RecordBatch, StringArray, StructArray};
    use arrow_schema::{Field, Schema};

    #[rstest::rstest]
    #[case::one_byte(1)]
    #[case::uuid(16)]
    #[case::content_hash(32)]
    fn test_extract_fixed_size_binary_key(#[case] size: i32) {
        let values = [
            Some(vec![0; size as usize]),
            Some(vec![1; size as usize]),
            None,
            Some(vec![2; size as usize]),
        ];
        let array =
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(values.into_iter(), size).unwrap();
        let batch = RecordBatch::try_from_iter([("id", Arc::new(array) as ArrayRef)])
            .unwrap()
            .slice(1, 3);
        let on_columns = vec!["id".to_string()];
        let key = KeyValue::Binary(vec![1; size as usize]);
        assert_eq!(
            extract_key_value_from_batch(&batch, 0, &on_columns),
            Some(key.clone())
        );
        assert_eq!(extract_key_value_from_batch(&batch, 1, &on_columns), None);
        assert_eq!(
            extract_key_value_from_batch(&batch, 2, &on_columns),
            Some(KeyValue::Binary(vec![2; size as usize]))
        );

        let mut builder = KeyExistenceFilterBuilder::new(vec![0]);
        builder.insert(key).unwrap();
        assert_eq!(
            builder
                .build()
                .might_contain_columns(batch.columns())
                .unwrap(),
            vec![true, false, false]
        );
    }

    #[test]
    fn test_extract_key_value_from_batch_list_int() {
        let values_builder = Int32Builder::new();
        let mut list_builder = ListBuilder::new(values_builder);

        list_builder.append_value([Some(1), Some(2)]);
        list_builder.append_value([Some(3), Some(4), Some(5)]);

        let list_array = list_builder.finish();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            list_array.data_type().clone(),
            false,
        )]));

        let batch = RecordBatch::try_new(schema, vec![Arc::new(list_array)])
            .expect("batch should be valid");

        let key0 = extract_key_value_from_batch(&batch, 0, &[String::from("id")])
            .expect("first row should produce a key");
        let key1 = extract_key_value_from_batch(&batch, 1, &[String::from("id")])
            .expect("second row should produce a key");

        match &key0 {
            KeyValue::List(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(values[0], KeyValue::Int64(1));
                assert_eq!(values[1], KeyValue::Int64(2));
            }
            other => panic!("expected list key, got {:?}", other),
        }

        match &key1 {
            KeyValue::List(values) => {
                assert_eq!(values.len(), 3);
                assert_eq!(values[0], KeyValue::Int64(3));
                assert_eq!(values[1], KeyValue::Int64(4));
                assert_eq!(values[2], KeyValue::Int64(5));
            }
            other => panic!("expected list key, got {:?}", other),
        }

        assert_ne!(
            key0.hash_value(),
            key1.hash_value(),
            "different list values should hash differently",
        );
    }

    #[test]
    fn test_extract_key_value_from_batch_empty_list() {
        let values_builder = Int32Builder::new();
        let mut list_builder = ListBuilder::new(values_builder);

        list_builder.append_value(std::iter::empty::<Option<i32>>());

        let list_array = list_builder.finish();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            list_array.data_type().clone(),
            false,
        )]));

        let batch = RecordBatch::try_new(schema, vec![Arc::new(list_array)])
            .expect("batch should be valid");

        let key = extract_key_value_from_batch(&batch, 0, &[String::from("id")])
            .expect("empty list should still produce a key");

        match key {
            KeyValue::List(values) => {
                assert!(values.is_empty(), "expected empty list");
            }
            other => panic!("expected list key, got {:?}", other),
        }
    }

    #[test]
    fn test_extract_key_value_from_batch_list_utf8() {
        let values_builder = StringBuilder::new();
        let mut list_builder = ListBuilder::new(values_builder);

        list_builder.append_value([Some("a"), Some("bc")]);
        list_builder.append_value([Some("de")]);

        let list_array = list_builder.finish();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            list_array.data_type().clone(),
            false,
        )]));

        let batch = RecordBatch::try_new(schema, vec![Arc::new(list_array)])
            .expect("batch should be valid");

        let key0 = extract_key_value_from_batch(&batch, 0, &[String::from("id")])
            .expect("first row should produce a key");
        let key1 = extract_key_value_from_batch(&batch, 1, &[String::from("id")])
            .expect("second row should produce a key");

        match &key0 {
            KeyValue::List(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(values[0], KeyValue::String("a".to_string()));
                assert_eq!(values[1], KeyValue::String("bc".to_string()));
            }
            other => panic!("expected list key, got {:?}", other),
        }

        match &key1 {
            KeyValue::List(values) => {
                assert_eq!(values.len(), 1);
                assert_eq!(values[0], KeyValue::String("de".to_string()));
            }
            other => panic!("expected list key, got {:?}", other),
        }

        assert_ne!(
            key0.hash_value(),
            key1.hash_value(),
            "different list values should hash differently",
        );
    }

    #[test]
    fn test_extract_key_value_from_batch_list_with_null_child() {
        let values_builder = Int32Builder::new();
        let mut list_builder = ListBuilder::new(values_builder);

        list_builder.append_value([Some(1), Some(2)]);
        list_builder.append_value([Some(3), None]);

        let list_array = list_builder.finish();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            list_array.data_type().clone(),
            false,
        )]));

        let batch = RecordBatch::try_new(schema, vec![Arc::new(list_array)])
            .expect("batch should be valid");

        let key0 = extract_key_value_from_batch(&batch, 0, &[String::from("id")])
            .expect("first row should produce a key");
        let key1 = extract_key_value_from_batch(&batch, 1, &[String::from("id")]);

        match &key0 {
            KeyValue::List(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(values[0], KeyValue::Int64(1));
                assert_eq!(values[1], KeyValue::Int64(2));
            }
            other => panic!("expected list key, got {:?}", other),
        }

        assert!(
            key1.is_none(),
            "list row with a null child should not produce a key",
        );
    }

    #[test]
    fn test_extract_key_value_from_batch_struct_int() {
        let a_values = Int32Array::from(vec![1, 3]);
        let b_values = Int32Array::from(vec![2, 4]);

        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("a", arrow_schema::DataType::Int32, false)),
                Arc::new(a_values) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("b", arrow_schema::DataType::Int32, false)),
                Arc::new(b_values) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            struct_array.data_type().clone(),
            false,
        )]));

        let batch = RecordBatch::try_new(schema, vec![Arc::new(struct_array)])
            .expect("batch should be valid");

        let key0 = extract_key_value_from_batch(&batch, 0, &[String::from("id")])
            .expect("first row should produce a key");
        let key1 = extract_key_value_from_batch(&batch, 1, &[String::from("id")])
            .expect("second row should produce a key");

        match &key0 {
            KeyValue::Struct(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(values[0], KeyValue::Int64(1));
                assert_eq!(values[1], KeyValue::Int64(2));
            }
            other => panic!("expected struct key, got {:?}", other),
        }

        match &key1 {
            KeyValue::Struct(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(values[0], KeyValue::Int64(3));
                assert_eq!(values[1], KeyValue::Int64(4));
            }
            other => panic!("expected struct key, got {:?}", other),
        }

        assert_ne!(
            key0.hash_value(),
            key1.hash_value(),
            "different struct values should hash differently",
        );
    }

    #[test]
    fn test_extract_key_value_from_batch_struct_utf8() {
        let first_names = StringArray::from(vec!["alice", "bob"]);
        let last_names = StringArray::from(vec!["smith", "jones"]);

        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("first", arrow_schema::DataType::Utf8, false)),
                Arc::new(first_names) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("last", arrow_schema::DataType::Utf8, false)),
                Arc::new(last_names) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            struct_array.data_type().clone(),
            false,
        )]));

        let batch = RecordBatch::try_new(schema, vec![Arc::new(struct_array)])
            .expect("batch should be valid");

        let key0 = extract_key_value_from_batch(&batch, 0, &[String::from("id")])
            .expect("first row should produce a key");
        let key1 = extract_key_value_from_batch(&batch, 1, &[String::from("id")])
            .expect("second row should produce a key");

        match &key0 {
            KeyValue::Struct(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(values[0], KeyValue::String("alice".to_string()));
                assert_eq!(values[1], KeyValue::String("smith".to_string()));
            }
            other => panic!("expected struct key, got {:?}", other),
        }

        match &key1 {
            KeyValue::Struct(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(values[0], KeyValue::String("bob".to_string()));
                assert_eq!(values[1], KeyValue::String("jones".to_string()));
            }
            other => panic!("expected struct key, got {:?}", other),
        }

        assert_ne!(
            key0.hash_value(),
            key1.hash_value(),
            "different struct values should hash differently",
        );
    }

    #[test]
    fn test_extract_key_value_from_batch_struct_with_null_child() {
        let a_values = Int32Array::from(vec![Some(1), None]);
        let b_values = Int32Array::from(vec![Some(2), Some(3)]);

        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("a", arrow_schema::DataType::Int32, true)),
                Arc::new(a_values) as Arc<dyn arrow_array::Array>,
            ),
            (
                Arc::new(Field::new("b", arrow_schema::DataType::Int32, true)),
                Arc::new(b_values) as Arc<dyn arrow_array::Array>,
            ),
        ]);

        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            struct_array.data_type().clone(),
            false,
        )]));

        let batch = RecordBatch::try_new(schema, vec![Arc::new(struct_array)])
            .expect("batch should be valid");

        let key0 = extract_key_value_from_batch(&batch, 0, &[String::from("id")])
            .expect("first row should produce a key");
        let key1 = extract_key_value_from_batch(&batch, 1, &[String::from("id")]);

        match &key0 {
            KeyValue::Struct(values) => {
                assert_eq!(values.len(), 2);
                assert_eq!(values[0], KeyValue::Int64(1));
                assert_eq!(values[1], KeyValue::Int64(2));
            }
            other => panic!("expected struct key, got {:?}", other),
        }

        assert!(
            key1.is_none(),
            "struct row with a null child should not produce a key",
        );
    }

    #[test]
    fn test_might_contain_exact_set_and_bloom_batches() {
        let present = KeyValue::Int64(4);
        let absent = KeyValue::Int64(99);
        let exact = KeyExistenceFilter {
            field_ids: vec![0],
            filter: FilterType::ExactSet(HashSet::from([present.hash_value()])),
        };
        assert!(exact.might_contain(&present).unwrap());
        assert!(!exact.might_contain(&absent).unwrap());

        let column: ArrayRef = Arc::new(Int32Array::from(vec![Some(4), None, Some(99)]));
        assert_eq!(
            exact.might_contain_columns(&[column]).unwrap(),
            vec![true, false, false]
        );

        let mut builder = KeyExistenceFilterBuilder::new(vec![0]);
        builder.insert(present.clone()).unwrap();
        builder.insert(KeyValue::Int64(8)).unwrap();
        let bloom = KeyExistenceFilter::from_bloom_filter(&builder);
        assert!(bloom.might_contain(&present).unwrap());
        let column: ArrayRef = Arc::new(Int32Array::from(vec![Some(4), Some(8), None]));
        assert_eq!(
            bloom.might_contain_columns(&[column]).unwrap(),
            vec![true, true, false]
        );
    }

    #[test]
    fn test_might_contain_rejects_invalid_bloom_payloads() {
        let column: ArrayRef = Arc::new(Int32Array::from(vec![Some(4)]));
        for (bitmap, expected) in [
            (Vec::new(), "must not be empty"),
            (vec![0; 33], "multiple of 32"),
        ] {
            let filter = KeyExistenceFilter {
                field_ids: vec![0],
                filter: FilterType::Bloom {
                    num_bits: (bitmap.len() as u32) * 8,
                    bitmap,
                    number_of_items: BLOOM_FILTER_DEFAULT_NUMBER_OF_ITEMS,
                    probability: BLOOM_FILTER_DEFAULT_PROBABILITY,
                },
            };
            let err = filter.might_contain(&KeyValue::Int64(4)).unwrap_err();
            assert!(
                err.to_string().contains(expected),
                "unexpected error for invalid bloom payload: {err}"
            );
            let err = filter
                .might_contain_columns(std::slice::from_ref(&column))
                .unwrap_err();
            assert!(
                err.to_string().contains(expected),
                "unexpected batch error for invalid bloom payload: {err}"
            );
            assert!(filter.validate().is_err());
        }
    }

    /// The conflict check concurrent merge-inserts run: disjoint inserted keys
    /// are not a conflict, and one shared key always is.
    #[test]
    fn test_intersects_only_on_a_shared_key() {
        fn filter(keys: impl IntoIterator<Item = i64>) -> KeyExistenceFilter {
            let mut builder = KeyExistenceFilterBuilder::new(vec![0]);
            for key in keys {
                builder.insert(KeyValue::Int64(key)).unwrap();
            }
            builder.build()
        }
        let a = filter(0..500);
        let b = filter(1_000_000..1_000_500);
        assert_eq!(a.intersects(&b).unwrap(), (false, false));
        let shared = filter((1_000_000..1_000_500).chain([17]));
        assert!(shared.intersects(&a).unwrap().0);
    }
}
