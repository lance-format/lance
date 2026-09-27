// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bringing a batch written under one schema to the schema in force now.
//!
//! A MemWAL holds rows written under older schemas, read and replayed against
//! the current one. That resolution happens once, here, and drives both replay
//! and scans.
//!
//! Columns match by **field id**. A rename keeps the id and changes the name, so
//! the same name under a different id is a different column. Names are used only
//! where no id exists, as in a batch a caller just handed in.
//!
//! A table with a MemWAL refuses type changes, so an id that disappears always
//! means the column was dropped.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::ArrayData;
use arrow_array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, GenericListArray, RecordBatch,
    RecordBatchOptions, StructArray,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef};
use lance_core::datatypes::LANCE_FIELD_ID_KEY;
use lance_core::{Error, Result};

use super::TOMBSTONE;

/// The lance field id an Arrow field carries, if it carries one.
///
/// Lance writes `-1` for a field it has not assigned an id to yet. Treating
/// that as an id would pair every unassigned column with every other, so a
/// negative id counts as none at all.
pub(super) fn field_id_of(field: &ArrowField) -> Option<i32> {
    field
        .metadata()
        .get(LANCE_FIELD_ID_KEY)
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|id| *id >= 0)
}

/// `column` under `data_type`, which differs from its own only in the field
/// ids it carries.
///
/// The two describe the same values as two different Arrow types, because the
/// ids sit inside the type. Only the labels differ, so this relabels the array
/// rather than converting it.
pub(super) fn relabel_to(column: &ArrayRef, data_type: &DataType) -> Result<ArrayRef> {
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
pub(super) fn without_field_ids_in(data_type: &DataType) -> DataType {
    let one = ArrowSchema::new(vec![ArrowField::new("", data_type.clone(), true)]);
    without_field_ids(&one).field(0).data_type().clone()
}

/// `field` without its Lance field id, keeping the rest of its metadata.
pub(super) fn without_field_id(field: &ArrowField) -> ArrowField {
    let mut field = field.clone();
    field.metadata_mut().remove(LANCE_FIELD_ID_KEY);
    field
}

/// `schema` without the field ids, for comparing against what a caller sends.
///
/// Ids belong to the stored schema, where identity has to survive a rename. A
/// caller's batch carries none, and Arrow compares a struct's children by their
/// full field — metadata included — so a stamped schema would reject it.
pub(super) fn without_field_ids(schema: &ArrowSchema) -> ArrowSchema {
    fn strip(field: &ArrowField) -> ArrowField {
        let field = without_field_id(field);
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
            DataType::Map(entries, sorted) => {
                let sorted = *sorted;
                field
                    .clone()
                    .with_data_type(DataType::Map(Arc::new(strip(entries)), sorted))
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
    Nested(usize, Vec<Self>, DataType),
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
    /// Resolution needs ids on the target, but a reader is handed the table's
    /// plain schema. Since the ids sit inside a nested column's type, a struct
    /// built to the id-carrying target is a different Arrow type from the one
    /// the caller declared. Only replay keeps them, writing back into the
    /// memtable's id-carrying storage schema.
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
        let claimed = claimed_by_id(source.fields(), target.fields());
        let sources = target
            .fields()
            .iter()
            .map(|field| resolve_field(field, source.fields(), &claimed, pk_columns, true))
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

fn resolve_field(
    field: &ArrowField,
    source_fields: &arrow_schema::Fields,
    claimed: &[bool],
    pk_columns: &[String],
    // `_tombstone` is a column of the batch, not a name that means anything
    // inside one. A struct child may legitimately be called that, and at depth
    // it is an ordinary field: absent from the source it is null, like any
    // other. `pk_columns` is emptied at depth for the same reason.
    top_level: bool,
) -> Result<Source> {
    let name = field.name();
    let by_id = field_id_of(field).and_then(|id| {
        source_fields
            .iter()
            .position(|f| field_id_of(f) == Some(id))
    });
    // Ids first. The name fallback is asymmetric on purpose: it fires for a
    // target field carrying no id, and for one whose id no source field has --
    // an id-bearing target against an unstamped source still matches by name.
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
        if top_level && name == TOMBSTONE {
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
    children_of(data_type).is_some()
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
        // A map's child is its entries struct, which carries the key and value
        // as children of its own.
        DataType::Map(entries, _) => Some(vec![entries.as_ref().clone()].into()),
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
    let claimed = claimed_by_id(&source_children, &target_children);
    target_children
        .iter()
        .map(|child| resolve_field(child, &source_children, &claimed, &[], false))
        .collect()
}

/// Source columns an id match has taken, which a name match may not take again.
fn claimed_by_id(source: &arrow_schema::Fields, target: &arrow_schema::Fields) -> Vec<bool> {
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

/// One stored column under the type the caller declared.
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
        // A map is its entries struct behind offsets. The sortedness flag is
        // part of the type, so it comes from the target with the rest of it.
        Source::Nested(i, children, to @ DataType::Map(_, sorted)) => {
            let map = columns[*i]
                .as_any()
                .downcast_ref::<arrow_array::MapArray>()
                .ok_or_else(|| Error::invalid_input(format!("column `{name}` is not a map")))?;
            let Some(entries) = children_of(to).and_then(|c| c.first().cloned()) else {
                unreachable!("a map target has an entries field");
            };
            let stored_entries: ArrayRef = Arc::new(map.entries().clone());
            let rebuilt = take_column(
                &children[0],
                std::slice::from_ref(&stored_entries),
                map.entries().len(),
                entries.name(),
            )?;
            let rebuilt = rebuilt
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    Error::invalid_input(format!("map column `{name}` entries are not a struct"))
                })?
                .clone();
            Ok(Arc::new(
                arrow_array::MapArray::try_new(
                    entries,
                    map.offsets().clone(),
                    rebuilt,
                    map.nulls().cloned(),
                    *sorted,
                )
                .map_err(|e| Error::invalid_input(format!("rebuild map column `{name}`: {e}")))?,
            ))
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
    use arrow_array::{
        Array, FixedSizeListArray, Int64Array, LargeListArray, ListArray, StructArray,
    };
    use arrow_buffer::{NullBuffer, OffsetBuffer};
    use arrow_schema::Fields;

    /// A nested column resolves by the field ids Lance puts on its children.
    /// If Lance starts giving children to a type [`is_nested`] does not know,
    /// those ids go unrestored and the column falls back to matching by name --
    /// the failure this module exists to prevent. So every type Lance gives
    /// children to has to be one we recurse into.
    #[test]
    fn every_type_lance_gives_children_to_is_one_we_recurse_into() {
        let item = || Arc::new(ArrowField::new("item", DataType::Int64, true));
        let a_struct = || Fields::from(vec![ArrowField::new("a", DataType::Int64, true)]);
        let entries = Arc::new(ArrowField::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                ArrowField::new("key", DataType::Int64, false),
                ArrowField::new("value", DataType::Int64, true),
            ])),
            false,
        ));
        let candidates = [
            DataType::Int64,
            DataType::Struct(a_struct()),
            DataType::List(item()),
            DataType::LargeList(item()),
            DataType::FixedSizeList(item(), 2),
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Struct(a_struct()), true)),
                2,
            ),
            DataType::Map(entries, false),
            DataType::ListView(item()),
            DataType::LargeListView(item()),
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            DataType::RunEndEncoded(
                Arc::new(ArrowField::new("run_ends", DataType::Int32, false)),
                item(),
            ),
        ];
        for data_type in candidates {
            let field = ArrowField::new("c", data_type.clone(), true);
            // A type Lance refuses outright can never reach a MemWAL.
            let Ok(lance) = lance_core::datatypes::Field::try_from(&field) else {
                continue;
            };
            if !lance.children.is_empty() {
                assert!(
                    is_nested(&data_type),
                    "Lance gives {data_type:?} children, so reconcile must recurse into it"
                );
            }
        }
    }

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

    /// Arrow validates a struct against the child types its own type declares,
    /// so a relabel that stops at the outer level produces a rejected array.
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

    /// A map is a nested container like any other: its entries carry field ids
    /// of their own, so a rename inside one has to resolve by id rather than
    /// read as a changed type.
    #[test]
    fn a_renamed_field_inside_a_map_resolves_by_id() {
        let entry = |value: &str| {
            ArrowField::new(
                "entries",
                DataType::Struct(Fields::from(vec![
                    stamped("key", DataType::Int64, 2),
                    stamped(value, DataType::Int64, 3),
                ])),
                false,
            )
        };
        let keys = Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef;
        let values = Arc::new(Int64Array::from(vec![10, 20])) as ArrayRef;
        let entries = StructArray::new(
            Fields::from(vec![
                stamped("key", DataType::Int64, 2),
                stamped("old", DataType::Int64, 3),
            ]),
            vec![keys, values],
            None,
        );
        let column = Arc::new(
            arrow_array::MapArray::try_new(
                Arc::new(entry("old")),
                arrow_buffer::OffsetBuffer::new(vec![0, 2].into()),
                entries,
                None,
                false,
            )
            .unwrap(),
        ) as ArrayRef;

        let source = ArrowSchema::new(vec![stamped(
            "m",
            DataType::Map(Arc::new(entry("old")), false),
            1,
        )]);
        let target: SchemaRef = Arc::new(ArrowSchema::new(vec![stamped(
            "m",
            DataType::Map(Arc::new(entry("new")), false),
            1,
        )]));
        let batch = RecordBatch::try_new(Arc::new(source.clone()), vec![column]).unwrap();

        let plan = Plan::resolve(&source, &target, &[]).expect("a rename inside a map resolves");
        let out = plan.apply(&batch).expect("apply");
        let map = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::MapArray>()
            .expect("a map");
        assert_eq!(
            map.entries().column_names(),
            vec!["key", "new"],
            "the renamed entry arrives under its new name"
        );
        let vals = map
            .entries()
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("values");
        assert_eq!(
            (vals.value(0), vals.value(1)),
            (10, 20),
            "carrying its values"
        );
        assert_eq!(map.value_length(0), 2, "and its offsets");
    }

    /// A struct whose parent is null at one row, and whose child is null at
    /// another: both levels of validity have to survive the relabel.
    /// `_tombstone` names a column of the batch, not a field inside one. A
    /// struct child may legitimately carry that name, and at depth it is an
    /// ordinary field: absent from the source it is null like any other, not
    /// the live-row marker. Treating it as the marker also fails outright when
    /// the child is not Boolean.
    #[test]
    fn a_struct_child_named_like_the_tombstone_is_an_ordinary_field() {
        let kept = stamped("kept", DataType::Int64, 2);
        let source = ArrowSchema::new(vec![
            stamped("id", DataType::Int64, 0),
            stamped(
                "info",
                DataType::Struct(Fields::from(vec![kept.clone()])),
                1,
            ),
        ]);
        // The table declares a child with the tombstone's name that the
        // generation never stored -- and typed Int64, which the live marker
        // could not be built as.
        let target = Arc::new(ArrowSchema::new(vec![
            stamped("id", DataType::Int64, 0),
            stamped(
                "info",
                DataType::Struct(Fields::from(vec![
                    kept.clone(),
                    stamped(TOMBSTONE, DataType::Int64, 3),
                ])),
                1,
            ),
        ]));
        let batch = RecordBatch::try_new(
            Arc::new(source.clone()),
            vec![
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                Arc::new(StructArray::new(
                    Fields::from(vec![kept]),
                    vec![Arc::new(Int64Array::from(vec![7])) as ArrayRef],
                    None,
                )),
            ],
        )
        .expect("a batch under the source schema");

        let out = Plan::resolve(&source, &target, &["id".to_string()])
            .expect("a nested field named like the tombstone resolves")
            .emitting_plain_schema()
            .apply(&batch)
            .expect("and applies");
        let info = out
            .column_by_name("info")
            .expect("info")
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("a struct");
        assert!(
            info.column_by_name(TOMBSTONE)
                .expect("the child")
                .is_null(0),
            "a child the generation never stored is null, whatever it is called"
        );
    }

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
        let element = stamped(
            "item",
            DataType::Struct(Fields::from(vec![stamped("b", DataType::Int64, 4)])),
            3,
        );
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
