// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bringing a batch written under one schema to the schema in force now.
//!
//! A MemWAL holds rows written under whatever schema the table had at the time,
//! and they are read and replayed against the schema it has now. Resolving one
//! to the other is done once, here, and the result drives both: replay applies
//! it to a WAL entry, and a scan applies it to a generation's batches.
//!
//! Columns are matched by **field id**. A rename changes a field's name and
//! keeps its id, so a name is not identity: a source column of the same name
//! under a different id is a different column, and reading it would answer with
//! values the table no longer has. A name is matched only where identity is
//! absent, as in a batch a caller has just handed in.
//!
//! Nothing here has to tell a cast from a column dropped and replaced, which no
//! rule can: a table with a MemWAL refuses to change a column's type, so the
//! only way for an id to disappear is a drop.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, GenericListArray, RecordBatch,
    RecordBatchOptions, StructArray,
};
use arrow::array::ArrayData;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use lance_core::datatypes::LANCE_FIELD_ID_KEY;
use lance_core::{Error, Result};

use super::TOMBSTONE;

/// The lance field id an Arrow field carries, if it carries one.
pub(crate) fn field_id_of(field: &ArrowField) -> Option<i32> {
    field
        .metadata()
        .get(LANCE_FIELD_ID_KEY)
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|id| *id >= 0)
}

/// `schema` without the field ids, for comparing against what a caller sends.
///
/// Ids belong to the stored schema, where identity has to survive a rename. A
/// caller's batch carries none, and Arrow compares a struct's children by their
/// full field — metadata included — so a stamped schema would reject it.
/// One stored column under the type the caller declared.
///
/// A field id lives inside a nested column's own Arrow type, so the memtable's
/// id-carrying storage schema and the table's plain one describe the same
/// values as two different types. Only the labels differ, so this relabels the
/// array rather than converting it.
pub(crate) fn relabel_to(column: &ArrayRef, data_type: &DataType) -> Result<ArrayRef> {
    if column.data_type() == data_type {
        return Ok(column.clone());
    }
    relabel_data(&column.to_data(), data_type).map(arrow_array::make_array)
}

/// [`relabel_to`] over one array's data, and its children's in turn. Arrow
/// validates a container against the child types its own type declares, so a
/// label that moved at any depth has to move at every level below it.
fn relabel_data(data: &ArrayData, data_type: &DataType) -> Result<ArrayData> {
    let children = children_of(data_type);
    let child_data = match children {
        Some(fields) if fields.len() == data.child_data().len() => data
            .child_data()
            .iter()
            .zip(fields.iter())
            .map(|(child, field)| relabel_data(child, field.data_type()))
            .collect::<Result<Vec<_>>>()?,
        _ => data.child_data().to_vec(),
    };
    data.clone()
        .into_builder()
        .data_type(data_type.clone())
        .child_data(child_data)
        .build()
        .map_err(|e| {
            Error::invalid_input(format!(
                "a {} column cannot be read as {data_type}: {e}",
                data.data_type()
            ))
        })
}

/// [`without_field_ids`] for a type rather than a schema, for the nested types a
/// reconciliation builds.
pub(crate) fn without_field_ids_in(data_type: &DataType) -> DataType {
    let one = ArrowSchema::new(vec![ArrowField::new("", data_type.clone(), true)]);
    without_field_ids(&one).field(0).data_type().clone()
}

pub(crate) fn without_field_ids(schema: &ArrowSchema) -> ArrowSchema {
    fn strip(field: &ArrowField) -> ArrowField {
        let mut metadata = field.metadata().clone();
        metadata.remove(LANCE_FIELD_ID_KEY);
        let field = field.clone().with_metadata(metadata);
        match field.data_type() {
            DataType::Struct(children) => {
                let children: Vec<ArrowField> = children.iter().map(|c| strip(c)).collect();
                field.with_data_type(DataType::Struct(children.into()))
            }
            DataType::List(element) => field
                .clone()
                .with_data_type(DataType::List(Arc::new(strip(element)))),
            DataType::LargeList(element) => field
                .clone()
                .with_data_type(DataType::LargeList(Arc::new(strip(element)))),
            DataType::FixedSizeList(element, size) => {
                let size = *size;
                field
                    .clone()
                    .with_data_type(DataType::FixedSizeList(Arc::new(strip(element)), size))
            }
            _ => field,
        }
    }
    let fields: Vec<ArrowField> = schema.fields().iter().map(|f| strip(f)).collect();
    ArrowSchema::new_with_metadata(fields, schema.metadata().clone())
}

/// Where one target column's values come from.
#[derive(Debug, Clone)]
enum Source {
    /// The source column at this index, as it stands.
    Take(usize),
    /// The source column at this index, whose struct children need their own
    /// resolution.
    Nested(usize, Vec<Source>, DataType),
    /// The source does not have this column: rows written before it existed
    /// hold no value for it.
    Null(DataType),
    /// `_tombstone`, which a generation written before deletes existed does not
    /// carry. Its rows are all live.
    Live,
}

/// One resolution of a source schema against a target schema.
pub struct Plan {
    target: SchemaRef,
    sources: Vec<Source>,
    /// Whether the source already is the target, so applying changes nothing.
    identity: bool,
}

impl Plan {
    /// The same plan emitting the table's plain Arrow schema.
    ///
    /// Resolution needs field ids on the target — a rename keeps the id and
    /// moves the name — but a reader is handed the table's schema, which does
    /// not carry them. They live inside a nested column's own type, so a struct
    /// built to the id-carrying target is a different Arrow type from the one
    /// the caller declared. Only replay, which writes back into the memtable's
    /// id-carrying storage schema, keeps them.
    pub(crate) fn emitting_plain_schema(mut self) -> Self {
        fn strip(source: &mut Source) {
            match source {
                Source::Nested(_, children, data_type) => {
                    *data_type = without_field_ids_in(data_type);
                    children.iter_mut().for_each(strip);
                }
                Source::Null(data_type) => *data_type = without_field_ids_in(data_type),
                _ => {}
            }
        }
        self.sources.iter_mut().for_each(strip);
        self.target = Arc::new(without_field_ids(&self.target));
        self
    }

    /// Resolve `source` against `target`, or say why it cannot be done.
    ///
    /// `pk_columns` may not be filled with nulls: a row with no primary key
    /// cannot be placed, so an absent one is an error rather than a null.
    pub(crate) fn resolve(
        source: &ArrowSchema,
        target: &SchemaRef,
        pk_columns: &[String],
    ) -> Result<Self> {
        // An id match takes its source column; a name match may then only take
        // one nothing has claimed. A rename frees a name for another column to
        // use, and it is the id that says which column is really which.
        let claimed = claimed_by_id(source, target);
        let sources = target
            .fields()
            .iter()
            .map(|field| resolve_field(field, source.fields(), &claimed, pk_columns))
            .collect::<Result<Vec<_>>>()?;
        let identity = source.fields() == target.fields();
        Ok(Self {
            target: Arc::clone(target),
            sources,
            identity,
        })
    }

    /// The schema a batch has after [`Self::apply`].
    pub(crate) fn target(&self) -> &SchemaRef {
        &self.target
    }

    /// Whether the source schema already is the target, so applying this plan
    /// would produce the batch it was given.
    pub(crate) fn is_identity(&self) -> bool {
        self.identity
    }

    /// `batch` under the target schema.
    pub(crate) fn apply(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let rows = batch.num_rows();
        let columns = self
            .sources
            .iter()
            .zip(self.target.fields())
            .map(|(source, field)| take_column(source, batch.columns(), rows, field.name()))
            .collect::<Result<Vec<_>>>()?;
        RecordBatch::try_new_with_options(
            Arc::clone(&self.target),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .map_err(|e| Error::invalid_input(format!("reconcile a batch to the schema: {e}")))
    }
}

/// Source columns an id match has taken, which a name match may not take again.
fn claimed_by_id(source: &ArrowSchema, target: &SchemaRef) -> Vec<bool> {
    let by_id: HashMap<i32, usize> = source
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(i, f)| field_id_of(f).map(|id| (id, i)))
        .collect();
    let mut claimed = vec![false; source.fields().len()];
    for field in target.fields() {
        if let Some(i) = field_id_of(field).and_then(|id| by_id.get(&id)) {
            claimed[*i] = true;
        }
    }
    claimed
}

fn resolve_field(
    field: &ArrowField,
    source_fields: &arrow_schema::Fields,
    claimed: &[bool],
    pk_columns: &[String],
) -> Result<Source> {
    let name = field.name();
    let by_id = field_id_of(field).and_then(|id| {
        source_fields
            .iter()
            .position(|f| field_id_of(f) == Some(id))
    });
    // Identity is the field id where both sides carry one. A name is not: a
    // rename moves the name and leaves the id, so a source column of the same
    // name under a *different* id is a different column — one dropped and
    // another added under its name, whose values the table no longer has.
    //
    // A name match is right only where identity is absent: a batch a caller has
    // just handed in carries no ids, and neither does a schema supplied by a
    // caller who has none to give.
    let by_name = || {
        source_fields
            .iter()
            .position(|f| f.name() == name)
            .filter(|i| !claimed[*i])
    };
    let index = match field_id_of(field) {
        // The target names an identity: only that identity answers for it,
        // unless the source has none to be matched on.
        Some(_) => by_id.or_else(|| {
            source_fields
                .iter()
                .all(|f| field_id_of(f).is_none())
                .then(by_name)
                .flatten()
        }),
        None => by_name(),
    };

    let Some(index) = index else {
        if name == TOMBSTONE {
            return Ok(Source::Live);
        }
        if pk_columns.iter().any(|c| c == name) {
            return Err(Error::invalid_input(format!(
                "batch is missing primary key column `{name}` declared by the schema"
            )));
        }
        return Ok(Source::Null(field.data_type().clone()));
    };

    let source = &source_fields[index];
    // A nested column's children are part of the array's own type, metadata
    // included, so the array is rebuilt under the target's children even when
    // nothing about them moved — otherwise the batch disagrees with the schema
    // it is built under. The leaves are reused, so it costs a pointer copy.
    if is_nested(field.data_type()) {
        return Ok(Source::Nested(
            index,
            resolve_children(source, field)?,
            field.data_type().clone(),
        ));
    }
    if source.data_type() == field.data_type() {
        return Ok(Source::Take(index));
    }
    // The same column under a different scalar type. A table with a MemWAL
    // refuses a cast, so this is a disagreement to surface rather than paper
    // over; a struct differing only in its children is handled below.
    if !matches!(
        (source.data_type(), field.data_type()),
        (DataType::Struct(_), DataType::Struct(_))
    ) {
        return Err(Error::invalid_input(format!(
            "column `{name}` is stored as {} and the schema declares {}; a column's type \
             cannot change on a table with a MemWAL",
            source.data_type(),
            field.data_type()
        )));
    }
    unreachable!("a struct is resolved above and any other mismatch is rejected")
}

/// Whether this type carries its children inside its own type, so an array of
/// it has to be rebuilt rather than taken as it stands.
fn is_nested(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Struct(_)
            | DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(_, _)
    )
}

/// The child fields of a nested type, if it has them.
fn children_of(data_type: &DataType) -> Option<arrow_schema::Fields> {
    match data_type {
        DataType::Struct(children) => Some(children.clone()),
        // A list has exactly one child: its element. Its name is part of the
        // type, so it is resolved like any other.
        DataType::List(element) | DataType::LargeList(element) => {
            Some(vec![element.as_ref().clone()].into())
        }
        DataType::FixedSizeList(element, _) => Some(vec![element.as_ref().clone()].into()),
        _ => None,
    }
}

/// How each of `field`'s children is produced from `source`'s.
fn resolve_children(source: &ArrowField, field: &ArrowField) -> Result<Vec<Source>> {
    let (Some(source_children), Some(target_children)) = (
        children_of(source.data_type()),
        children_of(field.data_type()),
    ) else {
        return Err(Error::invalid_input(format!(
            "column `{}` is stored as {} and the schema declares {}; a column's type \
             cannot change on a table with a MemWAL",
            field.name(),
            source.data_type(),
            field.data_type()
        )));
    };
    let claimed = claimed_children(&source_children, &target_children);
    target_children
        .iter()
        .map(|child| resolve_field(child, &source_children, &claimed, &[]))
        .collect()
}

fn claimed_children(source: &arrow_schema::Fields, target: &arrow_schema::Fields) -> Vec<bool> {
    let by_id: HashMap<i32, usize> = source
        .iter()
        .enumerate()
        .filter_map(|(i, f)| field_id_of(f).map(|id| (id, i)))
        .collect();
    let mut claimed = vec![false; source.len()];
    for field in target {
        if let Some(i) = field_id_of(field).and_then(|id| by_id.get(&id)) {
            claimed[*i] = true;
        }
    }
    claimed
}

/// One list column rebuilt around a reconciled element, for either offset width.
fn rebuild_list<O: arrow_array::OffsetSizeTrait>(
    column: &ArrayRef,
    element_source: &Source,
    target: &DataType,
    name: &str,
) -> Result<ArrayRef> {
    let list = column
        .as_any()
        .downcast_ref::<GenericListArray<O>>()
        .ok_or_else(|| Error::invalid_input(format!("column `{name}` is not a list")))?;
    let Some(element) = children_of(target).and_then(|c| c.first().cloned()) else {
        unreachable!("a list target has an element");
    };
    let child = take_column(
        element_source,
        std::slice::from_ref(list.values()),
        list.values().len(),
        element.name(),
    )?;
    Ok(Arc::new(
        GenericListArray::<O>::try_new(
            element,
            list.offsets().clone(),
            child,
            list.nulls().cloned(),
        )
        .map_err(|e| Error::invalid_input(format!("rebuild list column `{name}`: {e}")))?,
    ))
}

fn take_column(source: &Source, columns: &[ArrayRef], rows: usize, name: &str) -> Result<ArrayRef> {
    match source {
        Source::Take(i) => Ok(Arc::clone(&columns[*i])),
        // A list is rebuilt around its element: the offsets and the validity say
        // which rows hold what, and only the element's own type moves. The two
        // offset widths are different array types and neither downcasts to the
        // other.
        Source::Nested(i, children, to @ DataType::List(_)) => {
            rebuild_list::<i32>(&columns[*i], &children[0], to, name)
        }
        Source::Nested(i, children, to @ DataType::LargeList(_)) => {
            rebuild_list::<i64>(&columns[*i], &children[0], to, name)
        }
        // A fixed-size list is rebuilt the same way, keeping its width.
        Source::Nested(i, children, to @ DataType::FixedSizeList(_, _)) => {
            let DataType::FixedSizeList(_, size) = to else {
                unreachable!("matched above");
            };
            let list = columns[*i]
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    Error::invalid_input(format!("column `{name}` is not a fixed-size list"))
                })?;
            let Some(element) = children_of(to).and_then(|c| c.first().cloned()) else {
                unreachable!("a fixed-size list target has an element");
            };
            let child = take_column(
                &children[0],
                std::slice::from_ref(list.values()),
                list.values().len(),
                element.name(),
            )?;
            Ok(Arc::new(
                FixedSizeListArray::try_new(element, *size, child, list.nulls().cloned()).map_err(
                    |e| {
                        Error::invalid_input(format!(
                            "rebuild fixed-size list column `{name}`: {e}"
                        ))
                    },
                )?,
            ))
        }
        Source::Nested(i, children, to) => {
            let DataType::Struct(target_children) = to else {
                unreachable!("Nested is only built for a struct or list target");
            };
            let struct_array = columns[*i]
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| Error::invalid_input(format!("column `{name}` is not a struct")))?;
            let built = children
                .iter()
                .zip(target_children)
                .map(|(child, field)| {
                    take_column(child, struct_array.columns(), rows, field.name())
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(
                StructArray::try_new(
                    target_children.clone(),
                    built,
                    struct_array.nulls().cloned(),
                )
                .map_err(|e| {
                    Error::invalid_input(format!("rebuild struct column `{name}`: {e}"))
                })?,
            ))
        }
        Source::Null(ty) => Ok(arrow_array::new_null_array(ty, rows)),
        Source::Live => Ok(Arc::new(BooleanArray::from(vec![false; rows]))),
    }
}

#[cfg(test)]
mod relabel_tests {
    use super::*;
    use arrow_array::{Int64Array, StructArray};
    use arrow_schema::Fields;

    fn stamped(name: &str, data_type: DataType, id: i32) -> ArrowField {
        ArrowField::new(name, data_type, true).with_metadata(
            [(LANCE_FIELD_ID_KEY.to_string(), id.to_string())]
                .into_iter()
                .collect(),
        )
    }

    /// A field id lives inside a nested column's type at every level, so the
    /// relabel has to reach all of them: Arrow validates a struct's children
    /// against the child types its own type declares.
    #[test]
    fn relabel_reaches_a_nested_child() {
        let inner_stamped = stamped("b", DataType::Int64, 3);
        let middle_stamped = stamped(
            "inner",
            DataType::Struct(Fields::from(vec![inner_stamped.clone()])),
            2,
        );
        let outer_stamped = DataType::Struct(Fields::from(vec![middle_stamped.clone()]));

        let leaf = Arc::new(Int64Array::from(vec![Some(7)])) as ArrayRef;
        let middle = StructArray::new(
            Fields::from(vec![inner_stamped]),
            vec![Arc::clone(&leaf)],
            None,
        );
        let outer = Arc::new(StructArray::new(
            Fields::from(vec![middle_stamped]),
            vec![Arc::new(middle) as ArrayRef],
            None,
        )) as ArrayRef;
        assert_eq!(outer.data_type(), &outer_stamped);

        let plain = without_field_ids_in(&outer_stamped);
        let relabelled = relabel_to(&outer, &plain).expect("relabel a nested column");
        assert_eq!(relabelled.data_type(), &plain, "every level is relabelled");

        // The values have to survive, not just the type.
        let as_struct = relabelled
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("a struct");
        let middle = as_struct
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("a nested struct");
        let values = middle
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("the leaf");
        assert_eq!(values.value(0), 7);
    }
}

#[cfg(test)]
mod nested_relabel_tests {
    use super::*;
    use arrow_array::{
        Array, FixedSizeListArray, Int64Array, LargeListArray, ListArray, StructArray,
    };
    use arrow_buffer::{NullBuffer, OffsetBuffer};
    use arrow_schema::Fields;

    fn stamped(name: &str, data_type: DataType, id: i32) -> ArrowField {
        ArrowField::new(name, data_type, true).with_metadata(
            [(LANCE_FIELD_ID_KEY.to_string(), id.to_string())]
                .into_iter()
                .collect(),
        )
    }

    /// Relabel `column` to its own type with the field ids stripped, and check
    /// that nothing but the labels moved.
    fn strip_and_check(column: ArrayRef) -> ArrayRef {
        let plain = without_field_ids_in(column.data_type());
        let out = relabel_to(&column, &plain).expect("relabel");
        assert_eq!(out.data_type(), &plain, "every level is relabelled");
        assert_eq!(out.len(), column.len(), "row count is preserved");
        assert_eq!(
            out.null_count(),
            column.null_count(),
            "validity is preserved"
        );
        out
    }

    /// A struct whose parent is null at one row, and whose child is null at
    /// another: both levels of validity have to survive the relabel.
    #[test]
    fn a_null_parent_and_a_null_child_both_survive() {
        let child = stamped("b", DataType::Int64, 3);
        let values = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])) as ArrayRef;
        let column = Arc::new(StructArray::new(
            Fields::from(vec![child]),
            vec![values],
            Some(NullBuffer::from(vec![true, true, false])),
        )) as ArrayRef;

        let out = strip_and_check(column);
        let out = out.as_any().downcast_ref::<StructArray>().expect("struct");
        assert!(out.is_null(2), "the null parent stays null");
        let inner = out
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("the child");
        assert_eq!(inner.value(0), 1);
        assert!(inner.is_null(1), "the null child stays null");
    }

    /// A list carries offsets and its own validity, and the element carries
    /// values: an empty list, a null list and a null element in one column.
    #[test]
    fn a_lists_offsets_and_validity_survive() {
        let element = stamped("item", DataType::Struct(Fields::from(vec![stamped("b", DataType::Int64, 4)])), 3);
        let leaf = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])) as ArrayRef;
        let inner = Arc::new(StructArray::new(
            Fields::from(vec![stamped("b", DataType::Int64, 4)]),
            vec![leaf],
            None,
        )) as ArrayRef;
        // Rows: [two elements], [], null.
        let column = Arc::new(ListArray::new(
            Arc::new(element),
            OffsetBuffer::new(vec![0, 2, 2, 3].into()),
            inner,
            Some(NullBuffer::from(vec![true, true, false])),
        )) as ArrayRef;

        let out = strip_and_check(column);
        let out = out.as_any().downcast_ref::<ListArray>().expect("list");
        assert_eq!(out.value_length(0), 2, "the first row keeps two elements");
        assert_eq!(out.value_length(1), 0, "the empty list stays empty");
        assert!(out.is_null(2), "the null list stays null");
    }

    /// `LargeList` is a different offset width, and neither array downcasts to
    /// the other.
    #[test]
    fn a_large_lists_offsets_survive() {
        let element = stamped("item", DataType::Int64, 3);
        let values = Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef;
        let column = Arc::new(LargeListArray::new(
            Arc::new(element),
            OffsetBuffer::new(vec![0i64, 2, 3].into()),
            values,
            None,
        )) as ArrayRef;

        let out = strip_and_check(column);
        let out = out
            .as_any()
            .downcast_ref::<LargeListArray>()
            .expect("a large list, not a list");
        assert_eq!(out.value_length(0), 2);
        assert_eq!(out.value_length(1), 1);
    }

    /// A fixed-size list's width lives in its type, so a relabel must carry it.
    #[test]
    fn a_fixed_size_lists_width_survives() {
        let element = stamped("item", DataType::Int64, 3);
        let values = Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef;
        let column =
            Arc::new(FixedSizeListArray::new(Arc::new(element), 2, values, None)) as ArrayRef;

        let out = strip_and_check(column);
        assert!(
            matches!(out.data_type(), DataType::FixedSizeList(_, 2)),
            "the width is part of the type, got {:?}",
            out.data_type()
        );
    }

    /// A sliced array carries a non-zero offset into its buffers. Relabelling
    /// must not reinterpret that as a full array.
    #[test]
    fn a_slice_keeps_its_offset() {
        let child = stamped("b", DataType::Int64, 3);
        let values = Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef;
        let whole = StructArray::new(Fields::from(vec![child]), vec![values], None);
        let column = Arc::new(whole.slice(2, 2)) as ArrayRef;

        let out = strip_and_check(column);
        let out = out.as_any().downcast_ref::<StructArray>().expect("struct");
        let inner = out
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("the child");
        assert_eq!(
            (0..out.len()).map(|i| inner.value(i)).collect::<Vec<_>>(),
            vec![3, 4],
            "the slice reads its own rows, not the array's first ones"
        );
    }

    /// An empty batch has no values to check, so the schema is the whole
    /// contract.
    #[test]
    fn an_empty_column_is_still_relabelled() {
        let child = stamped("b", DataType::Int64, 3);
        let column = Arc::new(StructArray::new(
            Fields::from(vec![child]),
            vec![Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef],
            None,
        )) as ArrayRef;
        let out = strip_and_check(column);
        assert_eq!(out.len(), 0);
    }

    /// Nothing to change is the fast path, and it has to return the same
    /// arrays rather than a rebuilt approximation of them.
    #[test]
    fn a_column_already_in_the_target_shape_is_returned_as_it_stands() {
        let plain = ArrowField::new("b", DataType::Int64, true);
        let column = Arc::new(StructArray::new(
            Fields::from(vec![plain]),
            vec![Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef],
            None,
        )) as ArrayRef;
        let out = relabel_to(&column, column.data_type()).expect("relabel");
        assert_eq!(out.data_type(), column.data_type());
        assert_eq!(out.len(), 2);
    }
}
