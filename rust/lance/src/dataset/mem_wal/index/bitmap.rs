// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! In-memory bitmap index for a memtable.
//!
//! A bitmap index answers the same queries as a BTree — equality, `IN`, range,
//! null — but earns its name on columns with few distinct values, where storing
//! one posting list per value costs far less than one entry per row.
//!
//! That difference decides the layout here. [`super::BTreeMemIndex`] keys its
//! skip list by `(value, row position)`, so a column of ten million rows holds
//! ten million nodes whatever its cardinality. This index keys by value alone
//! and hangs a [`RoaringTreemap`] off each one, so the same column costs one
//! entry per distinct value plus compressed positions — the shape the on-disk
//! [`lance_index::scalar::bitmap::BitmapIndex`] already uses.
//!
//! Ordering is kept because the on-disk index answers range queries from an
//! ordered map, and a memtable that could not would answer a narrower set of
//! queries than the index it is standing in for.

use std::collections::BTreeMap;
use std::sync::RwLock;

use arrow_array::{Array, RecordBatch};
use datafusion::common::ScalarValue;
use lance_core::{Error, Result};
use lance_index::scalar::btree::OrderableScalarValue;
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use roaring::RoaringTreemap;

use super::RowPosition;

/// Positions of the rows holding one value.
///
/// `RoaringTreemap` rather than `RoaringBitmap`: a [`RowPosition`] is a `u64`,
/// and a memtable large enough to exceed `u32` must not silently wrap.
type Positions = RoaringTreemap;

/// The map a reader and a writer share.
///
/// One lock over the whole map rather than a lock-free structure: inserts
/// arrive a batch at a time, so a writer takes it once per batch rather than
/// once per row, and the map it walks has one entry per distinct value. The
/// cost of the simpler structure is bounded by the cardinality this index is
/// chosen for.
#[derive(Default)]
struct Postings {
    /// Positions per distinct non-null value, ordered so range queries can walk
    /// a sub-range instead of the whole map.
    values: BTreeMap<OrderableScalarValue, Positions>,
    /// Positions whose value is null, kept apart because null matches only
    /// `IsNull` and never participates in ordering.
    nulls: Positions,
}

/// An in-memory bitmap index over one column of a memtable.
pub struct BitmapMemIndex {
    field_id: i32,
    column_name: String,
    postings: RwLock<Postings>,
}

impl std::fmt::Debug for BitmapMemIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitmapMemIndex")
            .field("field_id", &self.field_id)
            .field("column_name", &self.column_name)
            .field("distinct_values", &self.distinct_values())
            .finish()
    }
}

impl BitmapMemIndex {
    /// An empty index over `column_name`.
    pub fn new(field_id: i32, column_name: String) -> Self {
        Self {
            field_id,
            column_name,
            postings: RwLock::new(Postings::default()),
        }
    }

    pub fn field_id(&self) -> i32 {
        self.field_id
    }

    pub fn column_name(&self) -> &str {
        &self.column_name
    }

    /// Index every row of `batch`'s indexed column, numbering them from
    /// `row_offset`.
    pub fn insert(&self, batch: &RecordBatch, row_offset: u64) -> Result<()> {
        let Some(array) = batch.column_by_name(&self.column_name) else {
            // A batch without the column carries nothing to index. The BTree
            // path treats this the same way: a column may be absent from a
            // batch that predates it.
            return Ok(());
        };

        let mut postings = self
            .postings
            .write()
            .map_err(|_| Error::io("bitmap memtable index lock poisoned".to_string()))?;

        for row in 0..array.len() {
            let position = row_offset + row as u64;
            if array.is_null(row) {
                postings.nulls.insert(position);
                continue;
            }
            let value = ScalarValue::try_from_array(array, row)?;
            postings
                .values
                .entry(OrderableScalarValue(value))
                .or_default()
                .insert(position);
        }
        Ok(())
    }

    /// Positions holding `value`, ascending.
    pub fn get(&self, value: &ScalarValue) -> Vec<RowPosition> {
        let Ok(postings) = self.postings.read() else {
            return Vec::new();
        };
        if value.is_null() {
            return postings.nulls.iter().collect();
        }
        postings
            .values
            .get(&OrderableScalarValue(value.clone()))
            .map(|positions| positions.iter().collect())
            .unwrap_or_default()
    }

    /// Positions whose value is null, ascending.
    pub fn get_nulls(&self) -> Vec<RowPosition> {
        self.postings
            .read()
            .map(|postings| postings.nulls.iter().collect())
            .unwrap_or_default()
    }

    /// Positions whose value falls in `[start, end]`, ascending. An open bound
    /// is unbounded on that side.
    ///
    /// Walks only the matching sub-range, which is what the ordered map buys.
    pub fn range(
        &self,
        start: Option<&ScalarValue>,
        end: Option<&ScalarValue>,
    ) -> Vec<RowPosition> {
        use std::ops::Bound;

        let Ok(postings) = self.postings.read() else {
            return Vec::new();
        };
        let lower = match start {
            Some(value) => Bound::Included(OrderableScalarValue(value.clone())),
            None => Bound::Unbounded,
        };
        let upper = match end {
            Some(value) => Bound::Included(OrderableScalarValue(value.clone())),
            None => Bound::Unbounded,
        };

        let mut merged = Positions::new();
        for (_, positions) in postings.values.range((lower, upper)) {
            merged |= positions;
        }
        merged.iter().collect()
    }

    /// How many distinct non-null values are indexed.
    pub fn distinct_values(&self) -> usize {
        self.postings
            .read()
            .map(|postings| postings.values.len())
            .unwrap_or(0)
    }

    /// How many rows are indexed, nulls included.
    pub fn len(&self) -> usize {
        let Ok(postings) = self.postings.read() else {
            return 0;
        };
        let values: u64 = postings
            .values
            .values()
            .map(|positions| positions.len())
            .sum();
        (values + postings.nulls.len()) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Heap this index holds, for the memtable's memory budget.
    ///
    /// Roaring containers size themselves to their contents, so this asks them
    /// rather than multiplying a row count by a width.
    pub fn resident_bytes(&self) -> usize {
        let Ok(postings) = self.postings.read() else {
            return 0;
        };
        let mut bytes = postings.nulls.serialized_size();
        for (value, positions) in &postings.values {
            bytes += positions.serialized_size();
            bytes += value.0.size();
        }
        bytes
    }

    /// The `(value, row id)` rows an on-disk index is trained from, in
    /// `(value, position)` order, nulls first.
    ///
    /// Same schema the BTree memtable emits, because both feed the same
    /// builder through `preprocessed_data`.
    pub fn to_training_batches(&self, batch_size: usize) -> Result<Vec<RecordBatch>> {
        use arrow_schema::{DataType, Field, Schema};
        use lance_core::ROW_ID;
        use std::sync::Arc;

        let postings = self
            .postings
            .read()
            .map_err(|_| Error::io("bitmap memtable index lock poisoned".to_string()))?;
        let Some(data_type) = postings
            .values
            .keys()
            .next()
            .map(|value| value.0.data_type())
            .or_else(|| {
                // Null-only: the column still has a type, and a training batch
                // needs one to build a null array of.
                (!postings.nulls.is_empty()).then_some(DataType::Null)
            })
        else {
            return Ok(vec![]);
        };

        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, data_type.clone(), true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));

        let mut batches = Vec::new();
        let mut values: Vec<ScalarValue> = Vec::with_capacity(batch_size);
        let mut row_ids: Vec<u64> = Vec::with_capacity(batch_size);
        let null_value = ScalarValue::try_from(&data_type)?;

        let flush_if_full = |values: &mut Vec<ScalarValue>,
                             row_ids: &mut Vec<u64>,
                             batches: &mut Vec<RecordBatch>|
         -> Result<()> {
            if values.len() >= batch_size {
                batches.push(build_training_batch(&schema, values, row_ids)?);
                values.clear();
                row_ids.clear();
            }
            Ok(())
        };

        for position in &postings.nulls {
            values.push(null_value.clone());
            row_ids.push(position);
            flush_if_full(&mut values, &mut row_ids, &mut batches)?;
        }
        for (value, positions) in &postings.values {
            for position in positions {
                values.push(value.0.clone());
                row_ids.push(position);
                flush_if_full(&mut values, &mut row_ids, &mut batches)?;
            }
        }
        if !values.is_empty() {
            batches.push(build_training_batch(&schema, &values, &row_ids)?);
        }
        Ok(batches)
    }
}

fn build_training_batch(
    schema: &std::sync::Arc<arrow_schema::Schema>,
    values: &[ScalarValue],
    row_ids: &[u64],
) -> Result<RecordBatch> {
    use arrow_array::UInt64Array;
    use std::sync::Arc;

    let value_array = ScalarValue::iter_to_array(values.iter().cloned())?;
    let row_id_array = Arc::new(UInt64Array::from(row_ids.to_vec()));
    RecordBatch::try_new(schema.clone(), vec![value_array, row_id_array])
        .map_err(|e| Error::io(format!("Failed to create training batch: {}", e)))
}

/// Configuration for a bitmap scalar index in a memtable.
#[derive(Debug, Clone)]
pub struct BitmapIndexConfig {
    /// Index name, matching the base-table index it maintains.
    pub name: String,
    /// Field id of the indexed column.
    pub field_id: i32,
    /// Name of the indexed column.
    pub column: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    /// A colour column with `rows` rows cycling through `distinct` values, plus
    /// a null every seventh row.
    fn batch(rows: usize, distinct: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "colour",
            DataType::Utf8,
            true,
        )]));
        let values: Vec<Option<String>> = (0..rows)
            .map(|row| {
                if row % 7 == 6 {
                    None
                } else {
                    Some(format!("c{}", row % distinct))
                }
            })
            .collect();
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values))]).unwrap()
    }

    fn index_of(rows: usize, distinct: usize) -> BitmapMemIndex {
        let index = BitmapMemIndex::new(0, "colour".to_string());
        index.insert(&batch(rows, distinct), 0).unwrap();
        index
    }

    #[test]
    fn equality_returns_every_position_holding_the_value() {
        let index = index_of(20, 3);
        let positions = index.get(&ScalarValue::Utf8(Some("c0".to_string())));
        let expected: Vec<u64> = (0..20)
            .filter(|row| row % 7 != 6 && row % 3 == 0)
            .map(|row| row as u64)
            .collect();
        assert_eq!(positions, expected);
    }

    #[test]
    fn nulls_answer_only_the_null_query() {
        let index = index_of(20, 3);
        let nulls: Vec<u64> = (0..20)
            .filter(|row| row % 7 == 6)
            .map(|r| r as u64)
            .collect();
        assert_eq!(index.get_nulls(), nulls);
        // A null never lands in a value's posting list.
        for value in ["c0", "c1", "c2"] {
            let positions = index.get(&ScalarValue::Utf8(Some(value.to_string())));
            assert!(positions.iter().all(|position| !nulls.contains(position)));
        }
    }

    #[test]
    fn range_walks_only_the_matching_values() {
        let index = index_of(30, 5);
        let from_c1_to_c3 = index.range(
            Some(&ScalarValue::Utf8(Some("c1".to_string()))),
            Some(&ScalarValue::Utf8(Some("c3".to_string()))),
        );
        let expected: Vec<u64> = (0..30)
            .filter(|row| row % 7 != 6 && (1..=3).contains(&(row % 5)))
            .map(|row| row as u64)
            .collect();
        assert_eq!(from_c1_to_c3, expected);
    }

    #[test]
    fn an_open_bound_is_unbounded_on_that_side() {
        let index = index_of(30, 5);
        let all = index.range(None, None);
        let non_null: Vec<u64> = (0..30)
            .filter(|row| row % 7 != 6)
            .map(|r| r as u64)
            .collect();
        assert_eq!(all, non_null);
    }

    /// The reason this index keys by value rather than by row: its footprint
    /// follows the number of distinct values, not the number of rows.
    #[test]
    fn footprint_follows_cardinality_not_row_count() {
        let few_values = index_of(4096, 4);
        let many_values = index_of(4096, 1024);
        assert_eq!(few_values.len(), many_values.len());
        assert_eq!(few_values.distinct_values(), 4);
        assert_eq!(many_values.distinct_values(), 1024);
        assert!(
            few_values.resident_bytes() * 4 < many_values.resident_bytes(),
            "4 values held {} bytes, 1024 held {} — the low-cardinality case must be far cheaper",
            few_values.resident_bytes(),
            many_values.resident_bytes()
        );
    }

    #[test]
    fn positions_continue_across_batches() {
        let index = BitmapMemIndex::new(0, "colour".to_string());
        index.insert(&batch(10, 2), 0).unwrap();
        index.insert(&batch(10, 2), 10).unwrap();
        assert_eq!(index.len(), 20);
        let c0 = index.get(&ScalarValue::Utf8(Some("c0".to_string())));
        assert!(
            c0.iter().any(|position| *position >= 10),
            "second batch must be indexed"
        );
        assert!(
            c0.windows(2).all(|pair| pair[0] < pair[1]),
            "positions must ascend"
        );
    }

    #[test]
    fn training_batches_carry_every_row_once() {
        let index = index_of(100, 4);
        let batches = index.to_training_batches(16).unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total,
            index.len(),
            "every indexed row trains exactly one entry"
        );
        assert!(
            batches.iter().all(|b| b.num_rows() <= 16),
            "batch_size must be respected"
        );
        let schema = batches[0].schema();
        assert_eq!(schema.field(0).name(), VALUE_COLUMN_NAME);
        assert_eq!(schema.field(1).name(), lance_core::ROW_ID);
    }

    #[test]
    fn an_empty_index_trains_nothing() {
        let index = BitmapMemIndex::new(0, "colour".to_string());
        assert!(index.is_empty());
        assert!(index.to_training_batches(16).unwrap().is_empty());
    }

    #[test]
    fn a_batch_without_the_column_indexes_nothing() {
        let index = BitmapMemIndex::new(0, "colour".to_string());
        let other = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        index.insert(&other, 0).unwrap();
        assert!(index.is_empty());
    }
}
