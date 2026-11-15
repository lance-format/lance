// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A scalar index over the elements of a `list<struct>` column.
//!
//! [`ListElementIndex`] answers the per-element predicate
//!
//! ```sql
//! any_element(detections, 'label = ''car'' AND confidence > 0.9')
//! ```
//!
//! (see [`lance_datafusion::udf::ANY_ELEMENT`]) from an index, keeping one row
//! per list. It holds one entry per list element: the key is a tuple of the
//! element's `keys` fields, here `(label, confidence)`, encoded with Arrow's row
//! format (ascending, nulls first), whose bytes compare like the tuples they
//! encode; the posting is the row address of the list that holds the element.
//! The entries live in an ordinary B-tree over the binary keys, which this
//! index wraps the way the JSON index wraps its target.
//!
//! # Queries
//!
//! The predicate is parsed against the element's fields and planned with the
//! tuple planner ([`plan_tuple_predicate`]) over the key fields. An equality
//! (or `IN`) prefix and a range on the next key become seeks in the B-tree's
//! key space, one per prefix tuple and range; an equality on the first key
//! alone is a prefix range. Every key tuple comes from one element, so all
//! conditions of a seek hold on the same element. Two matching elements of one
//! list give the same row address, which the row-address set keeps once.
//!
//! - Every conjunct planned: the answer is exact.
//! - Some conjunct the tuple query does not express (a non-key field such as
//!   `area`, or a key condition it cannot seek): the answer is a superset and
//!   the scan rechecks it with `any_element` itself.
//! - No conjunct fixes the first key: the index declines, and the filter is
//!   evaluated by `any_element` on every row.
//!
//! # Nulls
//!
//! `any_element` is NULL for a NULL list, and for a list with no matching
//! element but some element on which the predicate is UNKNOWN: no conjunct is
//! FALSE and at least one is UNKNOWN. Those rows matter only under `NOT`, and
//! the index reports them when asked to track nulls. A NULL list is stored as a
//! NULL key. When every conjunct is a key condition that is UNKNOWN exactly
//! when its key is NULL, the predicate is UNKNOWN on an element exactly when a
//! non-empty set of the constrained keys is NULL and the rest match, so one
//! seek per such combination finds those elements. A NULL element is stored
//! as NULL in every key. Otherwise (a residual conjunct, a NULL literal, a
//! null-sensitive test, or too many combinations) the answer is not exact, so
//! the scan rechecks and `NOT` falls back to evaluating `any_element`.
//!
//! # Maintenance
//!
//! Update, remap and fragment coverage delegate to the B-tree. New rows are
//! exploded and encoded the same way as at training.

use std::{collections::HashSet, ops::Bound, sync::Arc};

use arrow::compute::{cast, concat, is_null, nullif, take};
use arrow::row::{RowConverter, SortField};
use arrow_array::{
    Array, ArrayRef, GenericListArray, OffsetSizeTrait, RecordBatch, StructArray, UInt32Array,
    UInt64Array, cast::AsArray, new_null_array,
};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, SortOptions};
use async_trait::async_trait;
use datafusion_common::{DataFusionError, ScalarValue, tree_node::TreeNode};
use datafusion_execution::SendableRecordBatchStream;
use datafusion_expr::{Expr, Operator, ScalarUDF, expr::ScalarFunction, utils::split_conjunction};
use datafusion_physical_expr::{PhysicalSortExpr, expressions::Column};
use datafusion_physical_plan::{
    ExecutionPlan, sorts::sort::SortExec, stream::RecordBatchStreamAdapter,
};
use futures::StreamExt;
use lance_core::{
    Error, ROW_ADDR, Result, cache::LanceCache, datatypes::LogicalType, deepsize::DeepSizeOf,
    utils::row_addr_remap::RowAddrRemap,
};
use lance_datafusion::udf::{ANY_ELEMENT, any_element_predicate, plan_element_predicate};
use lance_datafusion_exec::exec::{LanceExecutionOptions, OneShotExec, execute_plan};
use lance_index_core::remapping::RowAddrTranslator;
use lance_select::{NullableRowAddrSet, RowAddrTreeMap, RowSetOps};
use prost::Message;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use crate::{
    Index, IndexType,
    metrics::MetricsCollector,
    pb::{ListElementIndexDetails, ListElementKeyEncoding},
    scalar::{
        AnyQuery, CreatedIndex, IndexStore, OldIndexDataFilter, RowIdRemapper, SargableQuery,
        ScalarIndex, ScalarIndexParams, SearchOptions, SearchResult, UpdateCriteria,
        btree::BTreeIndexPlugin,
        expression::{IndexedExpression, ScalarQueryParser},
        registry::{
            BasicTrainer, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
            VALUE_COLUMN_NAME,
        },
        tuple::{KeyRange, KeyValues, TupleQuery, planner::plan_tuple_predicate},
    },
};

/// The format version of a list-element index segment, as recorded in
/// `IndexMetadata::index_version`. Every version stores row addresses.
pub const LIST_ELEMENT_INDEX_VERSION: u32 = 0;

/// The most seeks the NULL rows of one query may cost; beyond it the answer is
/// not exact (and `NOT` falls back to the scan).
const MAX_NULL_SEEKS: usize = 8192;

/// The plugin's display name. The registry derives the name users pass to
/// [`ScalarIndexParams`], `listelement`, from the details message instead.
const INDEX_TYPE_NAME: &str = "ListElement";

/// Training parameters, as JSON: `{"keys": ["label", "confidence"]}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ListElementIndexParams {
    /// Fields of the list's struct elements that form the key, in key order.
    pub keys: Vec<String>,
    /// Rows per B-tree page; the B-tree's default when unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zone_size: Option<u64>,
}

// ---------------------------------------------------------------------------
// Key encoding
// ---------------------------------------------------------------------------

/// The key fields of one index and their Arrow row encoding.
#[derive(Debug, Clone, PartialEq)]
struct KeyCodec {
    /// Key names and types, all nullable. The names are the ones the column
    /// has now: Lance resolves `field_ids` to them before handing the details
    /// to the plugin.
    fields: Fields,
    /// The Lance field IDs of the keys, parallel to `fields`. Empty in a
    /// segment the plugin has just trained, until Lance records them from its
    /// schema (the plugin only sees Arrow fields, which carry no IDs).
    field_ids: Vec<i32>,
}

/// One seek: values of the leading keys (NULL included), then optionally a
/// range on the next key.
#[derive(Debug, Clone)]
struct Seek {
    values: Vec<ScalarValue>,
    range: Option<KeyRange>,
}

type BinaryRange = (Bound<ScalarValue>, Bound<ScalarValue>);

/// The smallest byte string greater than every string that starts with
/// `prefix`, or `None` when no such string exists (all `0xFF`).
fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut out = prefix.to_vec();
    while let Some(last) = out.pop() {
        if last < u8::MAX {
            out.push(last + 1);
            return Some(out);
        }
    }
    None
}

fn binary(bytes: Vec<u8>) -> ScalarValue {
    ScalarValue::Binary(Some(bytes))
}

/// The type a key field of `data_type` is encoded as, or `None` when it cannot
/// be a key.
///
/// These are the types the `arrow_row` version 1 key encoding admits. A view
/// type encodes as its non-view counterpart, which gives the same bytes.
fn key_type(data_type: &DataType) -> Option<DataType> {
    match data_type {
        DataType::Utf8View => Some(DataType::Utf8),
        DataType::BinaryView => Some(DataType::Binary),
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Date32
        | DataType::Date64
        | DataType::Timestamp(_, _)
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Binary
        | DataType::LargeBinary => Some(data_type.clone()),
        _ => None,
    }
}

impl KeyCodec {
    fn try_new(names: &[String], types: &[DataType]) -> Result<Self> {
        if names.is_empty() || names.len() != types.len() {
            return Err(Error::invalid_input(format!(
                "a list-element index needs at least one key, each with a type; got keys \
                 {names:?} with types {types:?}"
            )));
        }
        Ok(Self {
            fields: names
                .iter()
                .zip(types)
                .map(|(name, data_type)| Field::new(name, data_type.clone(), true))
                .collect(),
            field_ids: Vec::new(),
        })
    }

    fn from_details(details: &ListElementIndexDetails) -> Result<Self> {
        if details.key_encoding() != ListElementKeyEncoding::ArrowRowV1 {
            return Err(Error::not_supported(format!(
                "list-element index key encoding {} is not supported by this build",
                details.key_encoding
            )));
        }
        let types = details
            .key_types
            .iter()
            .map(|logical_type| {
                DataType::try_from(&LogicalType::from(logical_type.as_str()))
                    .ok()
                    .filter(|data_type| key_type(data_type).as_ref() == Some(data_type))
                    .ok_or_else(|| {
                        Error::corrupt_file_named(
                            "ListElementIndexDetails",
                            format!(
                                "list-element index key type '{logical_type}' is not a key type"
                            ),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if !details.key_field_ids.is_empty() && details.key_field_ids.len() != details.keys.len() {
            return Err(Error::corrupt_file_named(
                "ListElementIndexDetails",
                format!(
                    "list-element index has {} key field IDs for {} keys",
                    details.key_field_ids.len(),
                    details.keys.len()
                ),
            ));
        }
        Ok(Self {
            field_ids: details.key_field_ids.clone(),
            ..Self::try_new(&details.keys, &types)?
        })
    }

    /// Whether Lance has recorded the key fields' IDs, which a query needs to
    /// be sure the keys are still the fields the index encoded.
    fn has_field_ids(&self) -> bool {
        self.field_ids.len() == self.fields.len()
    }

    fn names(&self) -> Vec<String> {
        self.fields.iter().map(|f| f.name().clone()).collect()
    }

    /// The Lance logical type of each key, as recorded in the details.
    fn logical_types(&self) -> Result<Vec<String>> {
        self.fields
            .iter()
            .map(|f| Ok(LogicalType::try_from(f.data_type())?.to_string()))
            .collect()
    }

    fn converter(&self, keys: usize) -> Result<RowConverter> {
        let sort = SortOptions {
            descending: false,
            nulls_first: true,
        };
        Ok(RowConverter::new(
            self.fields
                .iter()
                .take(keys)
                .map(|f| SortField::new_with_options(f.data_type().clone(), sort))
                .collect(),
        )?)
    }

    /// Encode the leading `columns.len()` keys of every row.
    fn encode_columns(&self, columns: &[ArrayRef]) -> Result<ArrayRef> {
        let rows = self.converter(columns.len())?.convert_columns(columns)?;
        Ok(Arc::new(rows.try_into_binary()?))
    }

    /// Encode one tuple of leading key values (typed NULLs included).
    fn encode_tuple(&self, values: &[ScalarValue]) -> Result<Vec<u8>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }
        let columns = values
            .iter()
            .zip(self.fields.iter())
            .map(|(value, field)| {
                let value = if value.data_type() == *field.data_type() {
                    value.clone()
                } else {
                    value.cast_to(field.data_type())?
                };
                Ok(value.to_array_of_size(1)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let rows = self.converter(values.len())?.convert_columns(&columns)?;
        Ok(rows.row(0).as_ref().to_vec())
    }

    fn null(&self, key: usize) -> Result<ScalarValue> {
        let field = self.fields.get(key).ok_or_else(|| {
            Error::internal(format!(
                "list-element query constrains key {key}, but the index has {} keys",
                self.fields.len()
            ))
        })?;
        Ok(ScalarValue::try_new_null(field.data_type())?)
    }

    /// The byte range of the keys a seek selects, or `None` when it is empty.
    fn seek_range(&self, seek: &Seek) -> Result<Option<BinaryRange>> {
        let prefix = self.encode_tuple(&seek.values)?;
        let Some((lower, upper)) = &seek.range else {
            let lower = if prefix.is_empty() {
                Bound::Unbounded
            } else {
                Bound::Included(binary(prefix.clone()))
            };
            let upper = match prefix_successor(&prefix) {
                Some(next) if !prefix.is_empty() => Bound::Excluded(binary(next)),
                _ => Bound::Unbounded,
            };
            return Ok(Some((lower, upper)));
        };
        let key = seek.values.len();
        let with = |value: ScalarValue| {
            let mut values = seek.values.clone();
            values.push(value);
            self.encode_tuple(&values)
        };
        // A range on a key never holds for its NULLs, which sort first.
        let lower = match lower {
            Bound::Included(value) => Bound::Included(binary(with(value.clone())?)),
            Bound::Excluded(value) => match prefix_successor(&with(value.clone())?) {
                Some(next) => Bound::Included(binary(next)),
                None => return Ok(None),
            },
            Bound::Unbounded => match prefix_successor(&with(self.null(key)?)?) {
                Some(next) => Bound::Included(binary(next)),
                None => return Ok(None),
            },
        };
        let upper = match upper {
            Bound::Included(value) => match prefix_successor(&with(value.clone())?) {
                Some(next) => Bound::Excluded(binary(next)),
                None => Bound::Unbounded,
            },
            Bound::Excluded(value) => Bound::Excluded(binary(with(value.clone())?)),
            Bound::Unbounded => match prefix_successor(&prefix) {
                Some(next) if !prefix.is_empty() => Bound::Excluded(binary(next)),
                _ => Bound::Unbounded,
            },
        };
        if let (
            Bound::Included(ScalarValue::Binary(Some(low))),
            Bound::Excluded(ScalarValue::Binary(Some(high))),
        ) = (&lower, &upper)
            && low >= high
        {
            return Ok(None);
        }
        Ok(Some((lower, upper)))
    }

    /// Byte ranges of the keys on which `query` is TRUE.
    fn match_ranges(&self, query: &TupleQuery) -> Result<Vec<BinaryRange>> {
        let mut slots = Vec::new();
        for (key, values) in query.prefix().iter().enumerate() {
            slots.push(match values {
                KeyValues::Null => vec![(Alt::Value(self.null(key)?), false)],
                KeyValues::In(values) => values
                    .iter()
                    .cloned()
                    .map(|v| (Alt::Value(v), false))
                    .collect(),
            });
        }
        if let Some(ranges) = query.ranges() {
            slots.push(
                ranges
                    .iter()
                    .cloned()
                    .map(|r| (Alt::Range(r), false))
                    .collect(),
            );
        }
        self.ranges_of(&slots, false)
    }

    /// Byte ranges of the keys on which `query`'s conjunction is UNKNOWN: a
    /// non-empty set of the constrained keys is NULL and every other one
    /// matches. `IS NULL` on a key is never UNKNOWN, so it only matches NULL.
    /// Holds when every conjunct has plain NULL semantics (see
    /// [`has_plain_null_semantics`]).
    fn null_ranges(&self, query: &TupleQuery) -> Result<Vec<BinaryRange>> {
        self.ranges_of(&self.null_slots(query)?, true)
    }

    fn null_slots(&self, query: &TupleQuery) -> Result<Vec<Vec<(Alt, bool)>>> {
        let mut slots = Vec::new();
        for (key, values) in query.prefix().iter().enumerate() {
            slots.push(match values {
                KeyValues::Null => vec![(Alt::Value(self.null(key)?), false)],
                KeyValues::In(values) => values
                    .iter()
                    .cloned()
                    .map(|v| (Alt::Value(v), false))
                    .chain(std::iter::once((Alt::Value(self.null(key)?), true)))
                    .collect(),
            });
        }
        if let Some(ranges) = query.ranges() {
            let key = query.prefix().len();
            slots.push(
                ranges
                    .iter()
                    .cloned()
                    .map(|r| (Alt::Range(r), false))
                    .chain(std::iter::once((Alt::Value(self.null(key)?), true)))
                    .collect(),
            );
        }
        Ok(slots)
    }

    fn null_seek_count(&self, query: &TupleQuery) -> Result<usize> {
        let slots = self.null_slots(query)?;
        let all = slots
            .iter()
            .try_fold(1_usize, |n, slot| n.checked_mul(slot.len()));
        let matching = slots.iter().try_fold(1_usize, |n, slot| {
            n.checked_mul(slot.iter().filter(|(_, is_null)| !is_null).count())
        });
        Ok(match (all, matching) {
            (Some(all), Some(matching)) => all - matching,
            _ => usize::MAX,
        })
    }

    /// Every combination of one alternative per slot, as byte ranges; with
    /// `need_null`, only combinations that pick at least one NULL alternative.
    fn ranges_of(&self, slots: &[Vec<(Alt, bool)>], need_null: bool) -> Result<Vec<BinaryRange>> {
        let mut out = Vec::new();
        let mut picks = vec![0_usize; slots.len()];
        if slots.iter().any(|slot| slot.is_empty()) {
            return Ok(out);
        }
        loop {
            let chosen = picks
                .iter()
                .zip(slots)
                .map(|(pick, slot)| &slot[*pick])
                .collect::<Vec<_>>();
            if !need_null || chosen.iter().any(|(_, is_null)| *is_null) {
                let mut seek = Seek {
                    values: Vec::new(),
                    range: None,
                };
                for (alt, _) in chosen {
                    match alt {
                        Alt::Value(value) => seek.values.push(value.clone()),
                        Alt::Range(range) => seek.range = Some(range.clone()),
                    }
                }
                if let Some(range) = self.seek_range(&seek)? {
                    out.push(range);
                }
            }
            // Advance the odometer.
            let mut slot = slots.len();
            loop {
                if slot == 0 {
                    return Ok(out);
                }
                slot -= 1;
                picks[slot] += 1;
                if picks[slot] < slots[slot].len() {
                    break;
                }
                picks[slot] = 0;
            }
        }
    }
}

/// One alternative of a seek slot: a fixed key value, or a range (last slot).
#[derive(Debug, Clone)]
enum Alt {
    Value(ScalarValue),
    Range(KeyRange),
}

// ---------------------------------------------------------------------------
// Training data: one row per element
// ---------------------------------------------------------------------------

fn training_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(VALUE_COLUMN_NAME, DataType::Binary, true),
        Field::new(ROW_ADDR, DataType::UInt64, false),
    ]))
}

/// Element positions (into the list's values) with their list's row, and the
/// rows whose list is NULL.
fn explode_offsets<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
) -> (UInt64Array, UInt32Array, UInt32Array) {
    let offsets = list.value_offsets();
    let mut elements = Vec::with_capacity(offsets[list.len()].as_usize() - offsets[0].as_usize());
    let mut parents = Vec::with_capacity(elements.capacity());
    let mut null_rows = Vec::new();
    for row in 0..list.len() {
        if list.is_null(row) {
            null_rows.push(row as u32);
            continue;
        }
        for element in offsets[row].as_usize()..offsets[row + 1].as_usize() {
            elements.push(element as u64);
            parents.push(row as u32);
        }
    }
    (
        UInt64Array::from(elements),
        UInt32Array::from(parents),
        UInt32Array::from(null_rows),
    )
}

/// `(list<struct>, _rowaddr)` rows to `(encoded key, _rowaddr)` rows: one per
/// element, plus a NULL key for each NULL list.
fn explode_batch(batch: &RecordBatch, codec: &KeyCodec) -> Result<RecordBatch> {
    let values = batch
        .column_by_name(VALUE_COLUMN_NAME)
        .unwrap_or_else(|| batch.column(0));
    let row_addrs = batch
        .column_by_name(ROW_ADDR)
        .ok_or_else(|| Error::internal("list-element training data has no _rowaddr column"))?;
    let (list_values, (elements, parents, null_rows)) = match values.data_type() {
        DataType::List(_) => {
            let list = values.as_list::<i32>();
            (list.values().clone(), explode_offsets(list))
        }
        DataType::LargeList(_) => {
            let list = values.as_list::<i64>();
            (list.values().clone(), explode_offsets(list))
        }
        other => {
            return Err(Error::invalid_input(format!(
                "a list-element index needs a list<struct> column, got {other}"
            )));
        }
    };
    let structs: &StructArray = list_values
        .as_struct_opt()
        .ok_or_else(|| Error::invalid_input("a list-element index needs a list of structs"))?;
    // A NULL element reads as NULL in every key.
    let element_is_null = if structs.null_count() > 0 {
        Some(take(&is_null(structs)?, &elements, None)?)
    } else {
        None
    };
    let key_columns = codec
        .fields
        .iter()
        .map(|field| {
            let child = structs.column_by_name(field.name()).ok_or_else(|| {
                Error::invalid_input(format!(
                    "list-element key '{}' is not a field of the list's elements",
                    field.name()
                ))
            })?;
            let mut child = take(child.as_ref(), &elements, None)?;
            if child.data_type() != field.data_type() {
                child = cast(&child, field.data_type())?;
            }
            if let Some(mask) = &element_is_null {
                child = nullif(&child, mask.as_boolean())?;
            }
            Ok(child)
        })
        .collect::<Result<Vec<_>>>()?;
    let keys = codec.encode_columns(&key_columns)?;
    let null_keys = new_null_array(&DataType::Binary, null_rows.len());
    let value = concat(&[keys.as_ref(), null_keys.as_ref()])?;
    let addrs = concat(&[
        take(row_addrs.as_ref(), &parents, None)?.as_ref(),
        take(row_addrs.as_ref(), &null_rows, None)?.as_ref(),
    ])?;
    Ok(RecordBatch::try_new(training_schema(), vec![value, addrs])?)
}

fn explode_stream(data: SendableRecordBatchStream, codec: KeyCodec) -> SendableRecordBatchStream {
    let stream =
        data.map(move |batch| explode_batch(&batch?, &codec).map_err(DataFusionError::from));
    Box::pin(RecordBatchStreamAdapter::new(training_schema(), stream))
}

/// Sort the exploded rows by key, nulls first, spilling as needed: the B-tree
/// trains on value-ordered input.
///
/// This uses the default spill pool, as the JSON index's sort does. A list
/// holds many elements, so this sort sees several times more rows than the
/// column has.
// TODO: size the pool for the exploded rows. A prototype build over 100M
// elements exhausted the default pool while merging spills and needed
// `LANCE_MEM_POOL_SIZE=8GiB`.
async fn sort_by_key(data: SendableRecordBatchStream) -> Result<SendableRecordBatchStream> {
    let input = Arc::new(OneShotExec::new(data));
    let value_idx = input.schema().index_of(VALUE_COLUMN_NAME)?;
    let sort_expr = PhysicalSortExpr {
        expr: Arc::new(Column::new(VALUE_COLUMN_NAME, value_idx)),
        options: SortOptions {
            descending: false,
            nulls_first: true,
        },
    };
    let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new([sort_expr].into(), input));
    execute_plan(
        plan,
        LanceExecutionOptions {
            use_spilling: true,
            ..Default::default()
        },
    )
}

/// The types the key fields `keys` of the elements of `list_type` are encoded
/// as (see [`key_type`]).
fn key_fields_of(list_type: &DataType, keys: &[String]) -> Result<Vec<DataType>> {
    let element = match list_type {
        DataType::List(field) | DataType::LargeList(field) => field.data_type(),
        other => {
            return Err(Error::invalid_input(format!(
                "a list-element index needs a list<struct> column, got {other}"
            )));
        }
    };
    let DataType::Struct(fields) = element else {
        return Err(Error::invalid_input(format!(
            "a list-element index needs a list<struct> column, got a list of {element}"
        )));
    };
    keys.iter()
        .map(|key| {
            let (_, field) = fields.find(key).ok_or_else(|| {
                Error::invalid_input(format!(
                    "list-element key '{key}' is not a field of the elements ({})",
                    fields
                        .iter()
                        .map(|f| f.name().as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?;
            key_type(field.data_type()).ok_or_else(|| {
                Error::invalid_input(format!(
                    "list-element key '{key}' has type {}, which cannot be a key; keys may be \
                     booleans, integers, floats, dates, timestamps, strings or binaries",
                    field.data_type()
                ))
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Query and parser
// ---------------------------------------------------------------------------

/// A search of a [`ListElementIndex`]: the element key tuples to find.
#[derive(Debug, Clone, PartialEq)]
pub struct ListElementQuery {
    tuple: TupleQuery,
    /// Whether the tuple query is the whole predicate.
    exact: bool,
    /// The `any_element(...)` call, used to recheck an inexact answer.
    call: Expr,
}

impl ListElementQuery {
    /// The key tuples the query seeks.
    pub fn tuple(&self) -> &TupleQuery {
        &self.tuple
    }

    /// Whether the index answer is exact (no recheck).
    pub fn is_exact(&self) -> bool {
        self.exact
    }
}

impl AnyQuery for ListElementQuery {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn format(&self, col: &str) -> String {
        format!(
            "ListElement({col}: {}{})",
            self.tuple,
            if self.exact { "" } else { ", recheck" }
        )
    }

    fn to_expr(&self, _col: String) -> Expr {
        self.call.clone()
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other == self)
    }
}

/// Whether a planned conjunct is UNKNOWN exactly when its key is NULL.
///
/// [`KeyCodec::null_ranges`] relies on this for every conjunct: the predicate
/// is then UNKNOWN on an element exactly when a non-empty set of the
/// constrained keys is NULL and every other one matches. A NULL literal breaks
/// it (`label IN ('car', NULL)` is UNKNOWN for `'dog'`; the tuple planner keeps
/// only the non-null values), and so do `IS NOT NULL`, the
/// `IS [NOT] TRUE/FALSE/UNKNOWN` tests and `IS [NOT] DISTINCT FROM`.
fn has_plain_null_semantics(conjunct: &Expr) -> bool {
    !conjunct
        .exists(|e| {
            Ok(matches!(
                e,
                Expr::IsNotNull(_)
                    | Expr::IsTrue(_)
                    | Expr::IsFalse(_)
                    | Expr::IsUnknown(_)
                    | Expr::IsNotTrue(_)
                    | Expr::IsNotFalse(_)
                    | Expr::IsNotUnknown(_)
            ) || matches!(e, Expr::Literal(value, _) if value.is_null())
                || matches!(e, Expr::BinaryExpr(b) if b.op == Operator::IsDistinctFrom || b.op == Operator::IsNotDistinctFrom))
        })
        .unwrap_or(true)
}

/// Plan an `any_element` predicate (parsed over the element's fields) against
/// the index keys. `None` when no conjunct fixes the first key.
///
/// The answer is exact only when both the TRUE rows and the NULL rows are:
/// every conjunct is a key condition the seeks express, each is UNKNOWN exactly
/// when its key is NULL, and the NULL seeks stay within [`MAX_NULL_SEEKS`]. One
/// flag covers both, because the recheck it controls is also what makes
/// `NOT any_element(...)` fall back to a scan; an inexact NULL set must never
/// reach `NOT` as an exact answer.
fn plan_predicate(predicate: &Expr, codec: &KeyCodec, call: Expr) -> Option<ListElementQuery> {
    let conjuncts = split_conjunction(predicate)
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    let plan = plan_tuple_predicate(&conjuncts, &codec.fields)?;
    // A residual conjunct may be UNKNOWN on an element whose keys all match.
    let is_fully_planned = plan.key_residual.is_empty() && plan.row_residual.is_empty();
    let nulls_follow_keys = plan.consumed.iter().all(|c| has_plain_null_semantics(c));
    let tuple = plan.query;
    let exact = is_fully_planned
        && nulls_follow_keys
        && codec.null_seek_count(&tuple).ok()? <= MAX_NULL_SEEKS;
    Some(ListElementQuery { tuple, exact, call })
}

/// Routes `any_element(<indexed column>, '<predicate>')` to the index.
#[derive(Debug)]
pub struct ListElementQueryParser {
    index_name: String,
    codec: KeyCodec,
}

impl ListElementQueryParser {
    fn plan(
        &self,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<ListElementQuery> {
        if func.name() != ANY_ELEMENT {
            return None;
        }
        let [_, predicate] = args else {
            return None;
        };
        let predicate = any_element_predicate(predicate)?;
        // The column's key fields must be the ones the index encoded.
        let types = key_fields_of(data_type, &self.codec.names()).ok()?;
        if types
            .iter()
            .zip(self.codec.fields.iter())
            .any(|(column_type, field)| column_type != field.data_type())
        {
            return None;
        }
        // A predicate that does not parse is left to `any_element`, which
        // reports the same error.
        let predicate = plan_element_predicate(data_type, predicate).ok()?;
        let call = Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(func.clone()),
            args.to_vec(),
        ));
        plan_predicate(&predicate, &self.codec, call)
    }
}

impl ScalarQueryParser for ListElementQueryParser {
    fn visit_between(
        &self,
        _: &str,
        _: &Bound<ScalarValue>,
        _: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_in_list(&self, _: &str, _: &[ScalarValue]) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_bool(&self, _: &str, _: bool) -> Option<IndexedExpression> {
        None
    }

    fn visit_is_null(&self, _: &str) -> Option<IndexedExpression> {
        None
    }

    fn visit_comparison(
        &self,
        _: &str,
        _: &ScalarValue,
        _: &Operator,
    ) -> Option<IndexedExpression> {
        None
    }

    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        let query = self.plan(data_type, func, args)?;
        let needs_recheck = !query.exact;
        Some(IndexedExpression::index_query_with_recheck(
            column.to_string(),
            self.index_name.clone(),
            INDEX_TYPE_NAME.to_string(),
            Arc::new(query),
            needs_recheck,
        ))
    }
}

// ---------------------------------------------------------------------------
// The index
// ---------------------------------------------------------------------------

/// A list-element index segment: a B-tree over encoded element keys.
#[derive(Debug)]
pub struct ListElementIndex {
    target: Arc<dyn ScalarIndex>,
    codec: KeyCodec,
    zone_size: Option<u64>,
}

impl DeepSizeOf for ListElementIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.target.deep_size_of_children(context)
            + self
                .codec
                .names()
                .iter()
                .map(|n| n.capacity())
                .sum::<usize>()
    }
}

impl ListElementIndex {
    /// The names this index reads its keys by, in key order: the names the
    /// key fields had when the index was opened.
    pub fn key_names(&self) -> Vec<String> {
        self.codec.names()
    }

    fn wrap_target(&self, target: CreatedIndex) -> Result<CreatedIndex> {
        wrap_target(&self.codec, target)
    }

    /// TRUE rows of a B-tree search, ignoring the B-tree's own NULL tracking.
    async fn search_target(
        &self,
        query: SargableQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<(RowAddrTreeMap, bool)> {
        let result = self
            .target
            .search_with_options(
                &query,
                SearchOptions::default().with_track_nulls(false),
                metrics,
            )
            .await?;
        let exact = result.is_exact();
        Ok((result.row_addrs().true_rows(), exact))
    }

    async fn search_ranges(
        &self,
        ranges: Vec<BinaryRange>,
        metrics: &dyn MetricsCollector,
    ) -> Result<(RowAddrTreeMap, bool)> {
        let results =
            futures::future::try_join_all(ranges.into_iter().map(|(lower, upper)| {
                self.search_target(SargableQuery::Range(lower, upper), metrics)
            }))
            .await?;
        let exact = results.iter().all(|(_, exact)| *exact);
        let rows =
            RowAddrTreeMap::union_all(&results.iter().map(|(rows, _)| rows).collect::<Vec<_>>());
        Ok((rows, exact))
    }
}

fn wrap_target(codec: &KeyCodec, target: CreatedIndex) -> Result<CreatedIndex> {
    let details = ListElementIndexDetails {
        key_field_ids: codec.field_ids.clone(),
        keys: codec.names(),
        key_types: codec.logical_types()?,
        key_encoding: ListElementKeyEncoding::ArrowRowV1.into(),
        target_details: Some(target.index_details),
        target_index_version: target.index_version,
    };
    Ok(CreatedIndex {
        index_details: prost_types::Any::from_msg(&details)?,
        index_version: LIST_ELEMENT_INDEX_VERSION,
        files: target.files,
    })
}

#[async_trait]
impl Index for ListElementIndex {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn index_type(&self) -> IndexType {
        IndexType::Scalar
    }

    async fn prewarm(&self) -> Result<()> {
        self.target.prewarm().await
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "keys": self.codec.names(),
            "key_types": self.codec.logical_types()?,
            "target": self.target.statistics()?,
        }))
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        self.target.calculate_included_frags().await
    }
}

#[async_trait]
impl ScalarIndex for ListElementIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        self.search_with_options(query, SearchOptions::default(), metrics)
            .await
    }

    async fn search_with_options(
        &self,
        query: &dyn AnyQuery,
        options: SearchOptions,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query
            .as_any()
            .downcast_ref::<ListElementQuery>()
            .ok_or_else(|| Error::invalid_input("a list-element index needs a ListElementQuery"))?;
        let (true_rows, target_exact) = self
            .search_ranges(self.codec.match_ranges(&query.tuple)?, metrics)
            .await?;
        if !query.exact || !target_exact {
            return Ok(SearchResult::AtMost(NullableRowAddrSet::new(
                true_rows,
                RowAddrTreeMap::new(),
            )));
        }
        if !options.track_nulls() {
            return Ok(SearchResult::Exact(NullableRowAddrSet::new(
                true_rows,
                RowAddrTreeMap::new(),
            )));
        }
        // NULL rows: NULL lists, and lists with an element on which the
        // predicate is NULL, unless another element makes them TRUE.
        let (mut null_rows, _) = self.search_target(SargableQuery::IsNull(), metrics).await?;
        let (null_elements, _) = self
            .search_ranges(self.codec.null_ranges(&query.tuple)?, metrics)
            .await?;
        null_rows |= null_elements;
        null_rows -= &true_rows;
        Ok(SearchResult::Exact(NullableRowAddrSet::new(
            true_rows, null_rows,
        )))
    }

    fn results_are_row_addresses(&self) -> bool {
        self.target.results_are_row_addresses()
    }

    #[allow(deprecated)]
    fn can_remap(&self) -> bool {
        self.target.can_remap()
    }

    async fn remap(
        &self,
        mapping: &RowAddrRemap,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let target = self.target.remap(mapping, dest_store).await?;
        self.wrap_target(target)
    }

    async fn remap_streaming(
        &self,
        translator: &RowAddrTranslator,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let target = self.target.remap_streaming(translator, dest_store).await?;
        self.wrap_target(target)
    }

    fn stored_fragments(&self) -> Option<RoaringBitmap> {
        self.target.stored_fragments()
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        old_data_filter: Option<OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        let new_data = sort_by_key(explode_stream(new_data, self.codec.clone())).await?;
        let target = self
            .target
            .update(new_data, dest_store, old_data_filter)
            .await?;
        self.wrap_target(target)
    }

    fn update_criteria(&self) -> UpdateCriteria {
        let target = self.target.update_criteria();
        UpdateCriteria {
            requires_old_data: target.requires_old_data,
            // The scan cannot order by the encoded key; `update` sorts.
            data_criteria: TrainingCriteria::new(TrainingOrdering::None).with_row_addr(),
        }
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(
            ScalarIndexParams::new("listelement".to_string()).with_params(
                &ListElementIndexParams {
                    keys: self.codec.names(),
                    zone_size: self.zone_size,
                },
            ),
        )
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

struct ListElementTrainingRequest {
    params: ListElementIndexParams,
    codec: KeyCodec,
    criteria: TrainingCriteria,
}

impl TrainingRequest for ListElementTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

/// The plugin for [`ListElementIndex`], created with the index type
/// `listelement` and parameters such as `{"keys": ["label", "confidence"]}`.
///
/// ```
/// # use lance_index::scalar::ScalarIndexParams;
/// let params = ScalarIndexParams::new("listelement".to_string())
///     .with_params(&serde_json::json!({"keys": ["label", "confidence"]}));
/// ```
#[derive(Debug, Default)]
pub struct ListElementIndexPlugin;

impl ListElementIndexPlugin {
    fn btree_params(params: &ListElementIndexParams) -> String {
        match params.zone_size {
            Some(zone_size) => serde_json::json!({ "zone_size": zone_size }).to_string(),
            None => "{}".to_string(),
        }
    }
}

#[async_trait]
impl BasicTrainer for ListElementIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        let params = serde_json::from_str::<ListElementIndexParams>(params).map_err(|err| {
            Error::invalid_input(format!(
                "list-element index parameters must look like {{\"keys\": [\"label\", \"confidence\"]}}: {err}"
            ))
        })?;
        if params.keys.is_empty() {
            return Err(Error::invalid_input(
                "a list-element index needs at least one key",
            ));
        }
        let mut seen = HashSet::new();
        if let Some(dup) = params.keys.iter().find(|k| !seen.insert(k.as_str())) {
            return Err(Error::invalid_input(format!(
                "list-element key '{dup}' is repeated"
            )));
        }
        let types = key_fields_of(field.data_type(), &params.keys)?;
        let codec = KeyCodec::try_new(&params.keys, &types)?;
        // Validate the B-tree parameters now rather than after the scan.
        BTreeIndexPlugin.new_training_request(
            &Self::btree_params(&params),
            &Field::new(VALUE_COLUMN_NAME, DataType::Binary, true),
        )?;
        Ok(Box::new(ListElementTrainingRequest {
            params,
            codec,
            criteria: TrainingCriteria::new(TrainingOrdering::None).with_row_addr(),
        }))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
        progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let request = (request as Box<dyn std::any::Any>)
            .downcast::<ListElementTrainingRequest>()
            .map_err(|_| Error::internal("list-element training got a foreign request"))?;
        let sorted = sort_by_key(explode_stream(data, request.codec.clone())).await?;
        let target_request = BTreeIndexPlugin.new_training_request(
            &Self::btree_params(&request.params),
            &Field::new(VALUE_COLUMN_NAME, DataType::Binary, true),
        )?;
        let target = BTreeIndexPlugin
            .train_index(sorted, index_store, target_request, fragment_ids, progress)
            .await?;
        wrap_target(&request.codec, target)
    }
}

fn corrupt_details(message: impl Into<String>) -> Error {
    Error::corrupt_file_named("ListElementIndexDetails", message)
}

fn decode_details(details: &prost_types::Any) -> Result<ListElementIndexDetails> {
    ListElementIndexDetails::decode(details.value.as_slice())
        .map_err(|err| corrupt_details(format!("cannot decode list-element index details: {err}")))
}

/// The B-tree details inside `details`, which must be present and name a B-tree.
fn target_details(details: &ListElementIndexDetails) -> Result<&prost_types::Any> {
    let target = details
        .target_details
        .as_ref()
        .ok_or_else(|| corrupt_details("list-element index details have no target details"))?;
    if !target.type_url.ends_with("BTreeIndexDetails") {
        return Err(corrupt_details(format!(
            "list-element index target must be a B-tree, got {}",
            target.type_url
        )));
    }
    Ok(target)
}

#[async_trait]
impl ScalarIndexPlugin for ListElementIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn name(&self) -> &str {
        INDEX_TYPE_NAME
    }

    fn provides_exact_answer(&self) -> bool {
        // It cannot serve equality on the list column itself (merge-insert joins).
        false
    }

    fn version(&self) -> u32 {
        LIST_ELEMENT_INDEX_VERSION
    }

    fn can_remap(&self, _details: &prost_types::Any) -> Result<bool> {
        Ok(true)
    }

    fn new_query_parser(
        &self,
        index_name: String,
        index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        let details = decode_details(index_details).ok()?;
        let codec = KeyCodec::from_details(&details).ok()?;
        // Without the key fields' IDs, the names could belong to other fields
        // after a rename.
        if !codec.has_field_ids() {
            return None;
        }
        Some(Box::new(ListElementQueryParser { index_name, codec }))
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        _index_version: u32,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let details = decode_details(index_details)?;
        let codec = KeyCodec::from_details(&details)?;
        let target = BTreeIndexPlugin
            .load_index(
                index_store,
                target_details(&details)?,
                details.target_index_version,
                frag_reuse_index,
                &cache.with_key_prefix("element_target"),
            )
            .await?;
        let zone_size = target
            .derive_index_params()
            .ok()
            .and_then(|p| p.params)
            .and_then(|p| serde_json::from_str::<serde_json::Value>(&p).ok())
            .and_then(|p| p.get("zone_size").and_then(|z| z.as_u64()));
        Ok(Arc::new(ListElementIndex {
            target,
            codec,
            zone_size,
        }))
    }

    fn details_as_json(&self, details: &prost_types::Any) -> Result<serde_json::Value> {
        let details = decode_details(details)?;
        Ok(serde_json::json!({
            "key_field_ids": details.key_field_ids,
            "keys": details.keys,
            "key_types": details.key_types,
            "key_encoding": ListElementKeyEncoding::try_from(details.key_encoding)
                .map_or_else(|_| format!("unknown({})", details.key_encoding), |encoding| {
                    encoding.as_str_name().to_string()
                }),
        }))
    }

    /// Segments of one index must encode the same keys the same way, or a
    /// query planned against one segment's keys would misread another's.
    fn validate_new_segments_against_existing(
        &self,
        existing: &[&prost_types::Any],
        incoming: &[&prost_types::Any],
    ) -> Result<()> {
        let mut segments = existing.iter().chain(incoming);
        let Some(first) = segments.next() else {
            return Ok(());
        };
        let first = decode_details(first)?;
        for segment in segments {
            let segment = decode_details(segment)?;
            // Names may differ after a rename; the field IDs identify the keys.
            if (
                &segment.key_field_ids,
                &segment.key_types,
                segment.key_encoding,
            ) != (&first.key_field_ids, &first.key_types, first.key_encoding)
            {
                return Err(Error::invalid_input(format!(
                    "list-element index segments must share their keys: one has key fields \
                     {:?} with types {:?}, another {:?} with types {:?}",
                    first.key_field_ids, first.key_types, segment.key_field_ids, segment.key_types
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::metrics::NoOpMetricsCollector;
    use crate::progress::noop_progress;
    use crate::scalar::lance_format::LanceIndexStore;
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow_array::{Float32Array, ListArray, StringArray};
    use futures::stream;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_datafusion::planner::Planner;
    use lance_io::object_store::ObjectStore;

    /// label, confidence
    type Det = (Option<&'static str>, Option<f32>);

    fn det_fields() -> Fields {
        Fields::from(vec![
            Field::new("label", DataType::Utf8, true),
            Field::new("confidence", DataType::Float32, true),
            Field::new("area", DataType::Float32, true),
        ])
    }

    fn list_type() -> DataType {
        DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(det_fields()),
            true,
        )))
    }

    fn codec() -> KeyCodec {
        KeyCodec::try_new(
            &["label".to_string(), "confidence".to_string()],
            &[DataType::Utf8, DataType::Float32],
        )
        .unwrap()
    }

    fn list(rows: &[Option<Vec<Option<Det>>>]) -> ArrayRef {
        let (mut labels, mut confs, mut areas, mut valid, mut lengths, mut list_valid) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        for dets in rows {
            list_valid.push(dets.is_some());
            let dets = dets.clone().unwrap_or_default();
            lengths.push(dets.len());
            for det in dets {
                valid.push(det.is_some());
                let (l, c) = det.unwrap_or((Some("car"), Some(0.99)));
                labels.push(l);
                confs.push(c);
                areas.push(Some(1.0_f32));
            }
        }
        let elements = StructArray::try_new(
            det_fields(),
            vec![
                Arc::new(StringArray::from(labels)) as ArrayRef,
                Arc::new(Float32Array::from(confs)),
                Arc::new(Float32Array::from(areas)),
            ],
            Some(NullBuffer::from(valid)),
        )
        .unwrap();
        Arc::new(
            ListArray::try_new(
                Arc::new(Field::new("item", DataType::Struct(det_fields()), true)),
                OffsetBuffer::from_lengths(lengths),
                Arc::new(elements),
                Some(NullBuffer::from(list_valid)),
            )
            .unwrap(),
        )
    }

    fn car(c: f32) -> Option<Det> {
        Some((Some("car"), Some(c)))
    }

    fn person(c: f32) -> Option<Det> {
        Some((Some("person"), Some(c)))
    }

    /// Row i has row address i.
    fn rows() -> Vec<Option<Vec<Option<Det>>>> {
        vec![
            Some(vec![car(0.30), person(0.95)]),            // 0 trap
            Some(vec![car(0.95)]),                          // 1
            Some(vec![]),                                   // 2 empty
            None,                                           // 3 null list
            Some(vec![None, car(0.95)]),                    // 4 null element + match
            Some(vec![None]),                               // 5 only a null element
            Some(vec![Some((Some("car"), None))]),          // 6 null confidence
            Some(vec![car(0.90)]),                          // 7 boundary
            Some(vec![Some((Some("carrot"), Some(0.99)))]), // 8 carrot
            Some(vec![Some((None, Some(0.99)))]),           // 9 null label
            Some(vec![car(0.95), car(0.97)]),               // 10 two matches
            Some(vec![car(0.5)]),                           // 11
            Some(vec![Some((None, None))]),                 // 12 both keys null
        ]
    }

    async fn trained() -> (Arc<dyn ScalarIndex>, TempObjDir) {
        trained_on(rows()).await
    }

    async fn trained_on(rows: Vec<Option<Vec<Option<Det>>>>) -> (Arc<dyn ScalarIndex>, TempObjDir) {
        let tmp = TempObjDir::default();
        let store: Arc<dyn IndexStore> = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmp.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let plugin = ListElementIndexPlugin;
        let request = plugin
            .new_training_request(
                r#"{"keys": ["label", "confidence"], "zone_size": 4}"#,
                &Field::new(VALUE_COLUMN_NAME, list_type(), true),
            )
            .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, list_type(), true),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                list(&rows),
                Arc::new(UInt64Array::from_iter_values(0..rows.len() as u64)),
            ],
        )
        .unwrap();
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![Ok(batch)]),
        ));
        let created = plugin
            .train_index(data, store.as_ref(), request, None, noop_progress())
            .await
            .unwrap();
        let index = plugin
            .load_index(
                store,
                &created.index_details,
                created.index_version,
                None,
                &LanceCache::no_cache(),
            )
            .await
            .unwrap();
        (index, tmp)
    }

    fn parse(filter: &str) -> Option<ListElementQuery> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("width", DataType::Int32, false),
            Field::new("detections", list_type(), true),
        ]));
        let planner = Planner::new(schema);
        let expr = planner
            .optimize_expr(planner.parse_filter(filter).unwrap())
            .unwrap();
        let Expr::ScalarFunction(call) = &expr else {
            panic!("{filter} planned as {expr}");
        };
        let parser = ListElementQueryParser {
            index_name: "det_idx".to_string(),
            codec: codec(),
        };
        parser.plan(&list_type(), &call.func, &call.args)
    }

    async fn search(
        index: &dyn ScalarIndex,
        query: &ListElementQuery,
        track_nulls: bool,
    ) -> (Vec<u64>, Vec<u64>, bool) {
        let result = index
            .search_with_options(
                query,
                SearchOptions::default().with_track_nulls(track_nulls),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        let exact = result.is_exact();
        let rows = result.row_addrs();
        let mut t = rows
            .true_rows()
            .row_addrs()
            .unwrap()
            .map(u64::from)
            .collect::<Vec<_>>();
        let mut n = rows
            .null_rows()
            .row_addrs()
            .unwrap()
            .map(u64::from)
            .collect::<Vec<_>>();
        t.sort();
        n.sort();
        (t, n, exact)
    }

    fn q(pred: &str) -> String {
        format!("any_element(detections, '{}')", pred.replace('\'', "''"))
    }

    // Spill-enabled sorts share the cached DataFusion memory pool.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_same_element_and_ranges() {
        let (index, _tmp) = trained().await;
        let cases: &[(&str, &[u64], bool)] = &[
            // The trap (row 0) must not match; two matches dedupe (row 10).
            ("label = 'car' AND confidence > 0.9", &[1, 4, 10], true),
            ("label = 'car' AND confidence >= 0.9", &[1, 4, 7, 10], true),
            ("label = 'car' AND confidence < 0.9", &[0, 11], true),
            ("label = 'car' AND confidence <= 0.9", &[0, 7, 11], true),
            (
                "label = 'car' AND confidence BETWEEN 0.5 AND 0.95",
                &[1, 4, 7, 10, 11],
                true,
            ),
            // Label-only: a prefix range, including a NULL confidence.
            ("label = 'car'", &[0, 1, 4, 6, 7, 10, 11], true),
            (
                "label IN ('car', 'carrot') AND confidence > 0.98",
                &[8],
                true,
            ),
            ("label = 'person' AND confidence > 0.9", &[0], true),
            ("label IS NULL", &[4, 5, 9, 12], true),
            // A non-key condition: candidates, recheck.
            (
                "label = 'car' AND confidence > 0.9 AND area > 0",
                &[1, 4, 10],
                false,
            ),
        ];
        for (pred, want, exact) in cases {
            let filter = q(pred);
            let query = parse(&filter).unwrap_or_else(|| panic!("{filter} not planned"));
            assert_eq!(query.is_exact(), *exact, "{filter}: {query:?}");
            let (got, _, result_exact) = search(index.as_ref(), &query, false).await;
            assert_eq!(&got, want, "{filter}: {}", query.format("detections"));
            assert_eq!(result_exact, *exact, "{filter}");
        }
    }

    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_null_rows_for_not() {
        let (index, _tmp) = trained().await;
        let query = parse(&q("label = 'car' AND confidence > 0.9")).unwrap();
        let (t, n, exact) = search(index.as_ref(), &query, true).await;
        assert!(exact);
        assert_eq!(t, vec![1, 4, 10]);
        // 3 null list, 5 null element, 6 car with null confidence, 9 null label
        // with a high confidence, 12 both keys null (every conjunct UNKNOWN);
        // row 4 is TRUE despite its null element.
        assert_eq!(n, vec![3, 5, 6, 9, 12]);
        let query = parse(&q("label = 'car'")).unwrap();
        let (_, n, _) = search(index.as_ref(), &query, true).await;
        assert_eq!(n, vec![3, 5, 9, 12]);
    }

    #[test]
    fn test_unanswerable_shapes_decline() {
        // No condition on the first key.
        assert!(parse(&q("confidence > 0.9")).is_none());
        // OR across keys.
        assert!(parse(&q("label = 'car' OR confidence > 0.9")).is_none());
        // Not the indexed function.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "detections",
            list_type(),
            true,
        )]));
        let planner = Planner::new(schema);
        let expr = planner
            .optimize_expr(planner.parse_filter("detections IS NULL").unwrap())
            .unwrap();
        assert!(!matches!(expr, Expr::ScalarFunction(_)));
        // Shapes whose NULL rows do not follow the keys: not exact, so the
        // scan rechecks and NOT falls back to the scan.
        for pred in [
            "label = 'car' AND confidence IS NOT NULL",
            "label IN ('car', NULL)",
            "label = 'car' AND confidence > 0.9 AND area > 0",
        ] {
            let query = parse(&q(pred)).unwrap_or_else(|| panic!("{pred} not planned"));
            assert!(!query.is_exact(), "{pred}: {query:?}");
        }
    }

    #[test]
    fn test_details_round_trip() {
        let codec = codec();
        let details = ListElementIndexDetails {
            key_field_ids: vec![2, 3],
            keys: codec.names(),
            key_types: codec.logical_types().unwrap(),
            key_encoding: ListElementKeyEncoding::ArrowRowV1.into(),
            target_details: None,
            target_index_version: 0,
        };
        assert_eq!(details.key_types, vec!["string", "float"]);
        let decoded = KeyCodec::from_details(&details).unwrap();
        assert_eq!(decoded.fields, codec.fields);
        assert_eq!(decoded.field_ids, vec![2, 3]);
        assert!(decoded.has_field_ids());
        assert!(
            !codec.has_field_ids(),
            "a freshly trained codec has no IDs yet"
        );
        assert_eq!(prefix_successor(&[1, 2, 255]), Some(vec![1, 3]));
        assert_eq!(prefix_successor(&[255, 255]), None);

        let unspecified = ListElementIndexDetails {
            key_encoding: ListElementKeyEncoding::Unspecified.into(),
            ..details.clone()
        };
        let err = KeyCodec::from_details(&unspecified).unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");

        let missing_id = ListElementIndexDetails {
            key_field_ids: vec![2],
            ..details.clone()
        };
        let err = KeyCodec::from_details(&missing_id).unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");

        let not_a_key = ListElementIndexDetails {
            key_types: vec!["string".to_string(), "list".to_string()],
            ..details
        };
        let err = KeyCodec::from_details(&not_a_key).unwrap_err();
        assert!(matches!(err, Error::CorruptFile { .. }), "{err:?}");
        assert!(err.to_string().contains("'list'"), "{err}");
    }

    /// A segment whose key field IDs Lance has not recorded cannot plan
    /// queries: its names alone may point at other fields after a rename.
    #[test]
    fn test_query_parser_needs_field_ids() {
        let details = |key_field_ids: Vec<i32>| {
            prost_types::Any::from_msg(&ListElementIndexDetails {
                key_field_ids,
                keys: vec!["label".to_string()],
                key_types: vec!["string".to_string()],
                key_encoding: ListElementKeyEncoding::ArrowRowV1.into(),
                target_details: None,
                target_index_version: 0,
            })
            .unwrap()
        };
        let plugin = ListElementIndexPlugin;
        assert!(
            plugin
                .new_query_parser("idx".to_string(), &details(vec![]))
                .is_none()
        );
        assert!(
            plugin
                .new_query_parser("idx".to_string(), &details(vec![2]))
                .is_some()
        );
    }

    /// The bytes the format specification gives as its example.
    #[test]
    fn test_key_encoding_matches_spec() {
        let codec = codec();
        let car_half = codec
            .encode_tuple(&[ScalarValue::from("car"), ScalarValue::Float32(Some(0.5))])
            .unwrap();
        assert_eq!(
            car_half,
            [
                0x02, 0x63, 0x61, 0x72, 0, 0, 0, 0, 0, 0x03, 0x01, 0xBF, 0, 0, 0
            ]
        );
        let null_label = codec.encode_tuple(&[ScalarValue::Utf8(None)]).unwrap();
        assert_eq!(null_label, [0x00]);
        let empty_label = codec.encode_tuple(&[ScalarValue::from("")]).unwrap();
        assert_eq!(empty_label, [0x01]);
    }

    fn training_field() -> Field {
        let mut fields = det_fields().to_vec();
        fields.push(Arc::new(Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        )));
        Field::new(
            VALUE_COLUMN_NAME,
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(Fields::from(fields)),
                true,
            ))),
            true,
        )
    }

    #[rstest::rstest]
    #[case::no_keys(r#"{"keys": []}"#, "at least one key")]
    #[case::unknown_key(r#"{"keys": ["label", "speed"]}"#, "'speed' is not a field")]
    #[case::repeated_key(r#"{"keys": ["label", "label"]}"#, "'label' is repeated")]
    #[case::nested_key(r#"{"keys": ["tags"]}"#, "cannot be a key")]
    #[case::unknown_parameter(r#"{"keys": ["label"], "key": "x"}"#, "unknown field")]
    fn test_invalid_training_parameters(#[case] params: &str, #[case] reason: &str) {
        let err = ListElementIndexPlugin
            .new_training_request(params, &training_field())
            .err()
            .unwrap_or_else(|| panic!("{params} should be rejected"));
        assert!(matches!(err, Error::InvalidInput { .. }), "{err:?}");
        assert!(err.to_string().contains(reason), "{params}: {err}");
    }

    #[test]
    fn test_requires_a_list_of_structs() {
        let field = Field::new(
            VALUE_COLUMN_NAME,
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        );
        let err = ListElementIndexPlugin
            .new_training_request(r#"{"keys": ["label"]}"#, &field)
            .err()
            .unwrap();
        assert!(err.to_string().contains("list<struct>"), "{err}");
    }

    #[test]
    fn test_segments_must_share_keys() {
        let details = |ids: &[i32], keys: &[&str], key_types: &[&str]| {
            prost_types::Any::from_msg(&ListElementIndexDetails {
                key_field_ids: ids.to_vec(),
                keys: keys.iter().map(|k| k.to_string()).collect(),
                key_types: key_types.iter().map(|t| t.to_string()).collect(),
                key_encoding: ListElementKeyEncoding::ArrowRowV1.into(),
                target_details: None,
                target_index_version: 0,
            })
            .unwrap()
        };
        let label_confidence = details(&[2, 3], &["label", "confidence"], &["string", "float"]);
        // The same fields after a rename.
        let renamed = details(&[2, 3], &["class", "score"], &["string", "float"]);
        let label = details(&[2], &["label"], &["string"]);
        let other_field = details(&[4, 3], &["label", "confidence"], &["string", "float"]);
        let label_double = details(&[2, 3], &["label", "confidence"], &["string", "double"]);
        let plugin = ListElementIndexPlugin;
        for same in [&label_confidence, &renamed] {
            plugin
                .validate_new_segments_against_existing(&[&label_confidence], &[same])
                .unwrap();
        }
        plugin
            .validate_new_segments_against_existing(&[], &[&label])
            .unwrap();
        for other in [&label, &other_field, &label_double] {
            let err = plugin
                .validate_new_segments_against_existing(&[&label_confidence], &[other])
                .unwrap_err();
            assert!(err.to_string().contains("must share their keys"), "{err}");
        }
    }

    /// A column of NULL lists indexes only NULL keys: nothing matches, and
    /// every row is NULL under `NOT`.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_all_null_lists() {
        let (index, _tmp) = trained_on(vec![None, None, None]).await;
        let query = parse(&q("label = 'car' AND confidence > 0.9")).unwrap();
        let (t, n, exact) = search(index.as_ref(), &query, true).await;
        assert!(exact);
        assert!(t.is_empty(), "{t:?}");
        assert_eq!(n, vec![0, 1, 2]);
    }
}
