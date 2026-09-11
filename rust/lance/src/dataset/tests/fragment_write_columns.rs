// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-fragment column writes: staging columns' data as a standalone file with
//! `FileFragment::write_columns`, and committing it as a `DataReplacement`
//! whose coverage may not line up with any single file -- the case a computed
//! column reaches once compaction folds it into a shared base file.

use std::sync::Arc;

use arrow::array::AsArray;
use arrow_array::types::{Int32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Int32Array, ListArray, MapArray, RecordBatch,
    RecordBatchIterator, StringArray, StructArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};
use futures::{StreamExt, TryStreamExt, stream};
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::utils::tempfile::TempStrDir;
use lance_core::{Error, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};
use lance_encoding::constants::PACKED_STRUCT_META_KEY;
use lance_file::version::LanceFileVersion;
use rstest::rstest;

use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::schema_evolution::NewColumnTransform;
use crate::dataset::transaction::{DataReplacementGroup, Operation};
use crate::dataset::write::WriteParams;
use crate::dataset::{WriteDestination, fragment::FileFragment};
use crate::{Dataset, Result};

fn batch_of(fields: Vec<ArrowField>, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).unwrap()
}

fn ints(values: Vec<i32>) -> ArrayRef {
    Arc::new(Int32Array::from(values)) as ArrayRef
}

async fn dataset_of(batch: RecordBatch, version: Option<LanceFileVersion>) -> Dataset {
    let schema = batch.schema();
    let params = version.map(|data_storage_version| WriteParams {
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    });
    Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        params,
    )
    .await
    .unwrap()
}

/// A one-fragment dataset holding a single non-null `id` column of `[1, 2]`.
async fn id_dataset() -> Dataset {
    id_dataset_of(2, 1024).await
}

fn only_fragment(dataset: &Dataset) -> FileFragment {
    dataset.get_fragments().into_iter().next().unwrap()
}

/// Lance schema for a column the dataset does not define, with a fresh id.
fn new_column_schema(dataset: &Dataset, name: &str) -> LanceSchema {
    let mut schema = LanceSchema::try_from(&ArrowSchema::new(vec![ArrowField::new(
        name,
        DataType::Int32,
        true,
    )]))
    .unwrap();
    schema.fields[0].id = dataset.manifest.max_field_id() + 1;
    schema
}

/// Lance schema naming just the declared column `name`.
fn declared_schema(dataset: &Dataset, name: &str) -> LanceSchema {
    LanceSchema {
        fields: vec![dataset.schema().field(name).unwrap().clone()],
        metadata: Default::default(),
    }
}

async fn stage(
    dataset: &Dataset,
    batch: RecordBatch,
    schema: &LanceSchema,
) -> Result<DataReplacementGroup> {
    only_fragment(dataset)
        .write_columns(stream::iter([Ok(batch)]), schema)
        .await
}

async fn commit(dataset: &Dataset, replacements: Vec<DataReplacementGroup>) -> Result<Dataset> {
    let read_version = dataset.manifest.version;
    Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset.clone())),
        Operation::DataReplacement { replacements },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
}

/// A multi-fragment dataset of `rows` sequential ids, with stable row ids so
/// replacements can be checked against row lineage.
async fn id_dataset_of(rows: i32, max_rows_per_file: usize) -> Dataset {
    let batch = batch_of(
        vec![ArrowField::new("id", DataType::Int32, false)],
        vec![ints((1..=rows).collect())],
    );
    let schema = batch.schema();
    Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file,
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

async fn declare_all_null(dataset: &mut Dataset, name: &str) {
    let arrow = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        name,
        DataType::Int32,
        true,
    )]));
    dataset
        .add_columns(NewColumnTransform::AllNulls(arrow), None, None)
        .await
        .unwrap();
}

#[tokio::test]
async fn test_compact_metadata_only_all_null_dictionary() {
    let batch = batch_of(
        vec![ArrowField::new("id", DataType::Int32, false)],
        vec![ints(vec![1, 2])],
    );
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_3),
            max_rows_per_file: 1,
            enable_stable_row_ids: false,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let dictionary_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    dataset
        .add_columns(
            NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "category",
                dictionary_type.clone(),
                true,
            )]))),
            None,
            None,
        )
        .await
        .unwrap();

    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch["category"].data_type(), &dictionary_type);
    assert_eq!(batch["category"].null_count(), 2);
}

/// Stage `values` for an existing `column` of one fragment.
async fn stage_column(
    dataset: &Dataset,
    fragment_id: u64,
    column: &str,
    values: Vec<i32>,
) -> DataReplacementGroup {
    let schema = declared_schema(dataset, column);
    let batch = batch_of(
        vec![ArrowField::new(column, DataType::Int32, true)],
        vec![ints(values)],
    );
    dataset
        .get_fragments()
        .into_iter()
        .find(|fragment| fragment.id() as u64 == fragment_id)
        .expect("fragment to stage for")
        .write_columns(stream::iter([Ok(batch)]), &schema)
        .await
        .unwrap()
}

fn values(batch: &RecordBatch, name: &str) -> Vec<Option<i32>> {
    let col = batch[name].as_primitive::<Int32Type>();
    (0..batch.num_rows())
        .map(|i| (!col.is_null(i)).then(|| col.value(i)))
        .collect()
}

/// A `point` struct of two non-null Int32 children, packed or not.
fn point_schema(packed: bool) -> Arc<ArrowSchema> {
    let mut point = ArrowField::new("point", DataType::Struct(point_children()), false);
    if packed {
        point.set_metadata([(PACKED_STRUCT_META_KEY.to_string(), "true".to_string())].into());
    }
    Arc::new(ArrowSchema::new(vec![point]))
}

fn point_children() -> Fields {
    Fields::from(vec![
        ArrowField::new("x", DataType::Int32, false),
        ArrowField::new("y", DataType::Int32, false),
    ])
}

/// `xs` and `ys` are matched to `schema`'s children by name, so a schema that
/// orders them y-then-x still receives each child's own values.
fn points(schema: &Arc<ArrowSchema>, xs: [i32; 2], ys: [i32; 2]) -> RecordBatch {
    let DataType::Struct(children) = schema.field(0).data_type().clone() else {
        unreachable!("point schema is a struct")
    };
    let columns = children
        .iter()
        .map(|child| ints(if child.name() == "x" { xs } else { ys }.to_vec()))
        .collect();
    RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StructArray::new(children, columns, None)) as ArrayRef],
    )
    .unwrap()
}

/// Commit `group` and read the `point` column back as its two children.
async fn committed_points(dataset: &Dataset, group: DataReplacementGroup) -> (Vec<i32>, Vec<i32>) {
    let batch = commit(dataset, vec![group])
        .await
        .unwrap()
        .scan()
        .try_into_batch()
        .await
        .unwrap();
    let child = |i: usize| {
        batch
            .column(0)
            .as_struct()
            .column(i)
            .as_primitive::<Int32Type>()
            .values()
            .to_vec()
    };
    (child(0), child(1))
}

#[rstest]
#[tokio::test]
async fn test_records_writer_layout(
    #[values(LanceFileVersion::V2_0, LanceFileVersion::V2_1)] version: LanceFileVersion,
) {
    let mut dataset = dataset_of(
        arrow_array::record_batch!(("id", Int32, [1, 2])).unwrap(),
        Some(version),
    )
    .await;
    declare_all_null(&mut dataset, "value").await;
    let schema = declared_schema(&dataset, "value");
    let fragment = only_fragment(&dataset);

    // Streamed as two batches: the DataFile must record the writer's
    // field/column layout and the dataset's file version.
    let DataReplacementGroup(replaced, data_file) = fragment
        .write_columns(
            stream::iter([
                Ok(arrow_array::record_batch!(("value", Int32, [1])).unwrap()),
                Ok(arrow_array::record_batch!(("value", Int32, [2])).unwrap()),
            ]),
            &schema,
        )
        .await
        .unwrap();

    assert_eq!(replaced, fragment.id() as u64);
    assert_eq!(data_file.fields.as_ref(), &[schema.fields[0].id]);
    assert_eq!(data_file.fields.len(), data_file.column_indices.len());
    assert!(data_file.path.ends_with(".lance"));
    assert_eq!(
        (data_file.file_major_version, data_file.file_minor_version),
        version.resolve().to_data_file_numbers()
    );
}

/// Input `write_columns` turns down before anything can be committed. The
/// container cases matter twice over: projection reorders by name but downcasts
/// by shape, so an unchecked batch is dropped silently or panics.
#[rstest]
#[case::too_few_rows("short", "physical rows")]
#[case::too_many_rows("long", "physical rows")]
#[case::unrequested_column("extra", "unexpected=[unrequested]")]
#[case::wrong_container("struct", "should have type int32 but type was struct")]
#[case::reserved_system_name("rowid", "reserved column")]
// The commit publishes data files, never schema, so a field the manifest does
// not define would commit as coverage no live field answers for.
#[case::undeclared_field("undeclared", "does not define")]
// The reader takes type, nullability and nested layout from the manifest, so a
// staged field reusing an id but differing in any of them would be decoded as
// the manifest's version rather than rejected -- `validate()` would not notice.
#[case::field_type_mismatch("wrong_type", "does not match dataset field id")]
#[case::field_nullability_mismatch("wrong_nullability", "does not match dataset field id")]
// Projection picks children by name, so a duplicate makes the choice arbitrary.
// The schema check compares name sets and cannot see one.
#[case::duplicate_column("duplicate", "appears twice")]
#[tokio::test]
async fn test_rejects_bad_input(#[case] shape: &str, #[case] expected: &str) {
    let mut dataset = id_dataset().await;
    declare_all_null(&mut dataset, "value").await;
    let value = ArrowField::new("value", DataType::Int32, true);
    let mut schema = declared_schema(&dataset, "value");

    let values = match shape {
        "short" => batch_of(vec![value], vec![ints(vec![7])]),
        "long" => batch_of(vec![value], vec![ints(vec![7, 8, 9])]),
        "extra" => batch_of(
            vec![value, ArrowField::new("unrequested", DataType::Int32, true)],
            vec![ints(vec![1, 2]), ints(vec![3, 4])],
        ),
        "struct" => {
            let inner = Fields::from(vec![ArrowField::new("x", DataType::Int32, true)]);
            batch_of(
                vec![ArrowField::new(
                    "value",
                    DataType::Struct(inner.clone()),
                    true,
                )],
                vec![Arc::new(StructArray::new(inner, vec![ints(vec![1, 2])], None)) as ArrayRef],
            )
        }
        "rowid" => {
            schema = new_column_schema(&dataset, ROW_ID);
            batch_of(
                vec![ArrowField::new(ROW_ID, DataType::Int32, true)],
                vec![ints(vec![1, 2])],
            )
        }
        "undeclared" => {
            schema = new_column_schema(&dataset, "novel");
            batch_of(
                vec![ArrowField::new("novel", DataType::Int32, true)],
                vec![ints(vec![1, 2])],
            )
        }
        "duplicate" => batch_of(
            vec![value.clone(), value],
            vec![ints(vec![1, 2]), ints(vec![3, 4])],
        ),
        "wrong_type" | "wrong_nullability" => {
            let existing = dataset.schema().field("id").unwrap();
            let staged = if shape == "wrong_type" {
                ArrowField::new("id", DataType::Float32, existing.nullable)
            } else {
                ArrowField::new("id", DataType::Int32, !existing.nullable)
            };
            schema = LanceSchema::try_from(&ArrowSchema::new(vec![staged])).unwrap();
            schema.fields[0].id = existing.id;
            batch_of(
                vec![ArrowField::new("id", DataType::Int32, true)],
                vec![ints(vec![1, 2])],
            )
        }
        other => unreachable!("unknown case {other}"),
    };

    let err = stage(&dataset, values, &schema).await.unwrap_err();
    assert!(
        err.to_string().contains(expected),
        "expected '{expected}' in error, got: {err}"
    );
}

/// Nested containers the projector would otherwise reshape or unwind on: a
/// fixed-size list of the wrong width silently becomes a different row count,
/// and nulls under a required item panic instead of erroring.
#[rstest]
#[case::fixed_size_list_reshape(
    true,
    "fixed_size_list:int32:2 but type was fixed_size_list:int32:4"
)]
#[case::nulls_under_required_item(false, "non-null")]
#[tokio::test]
async fn test_rejects_bad_nested_input(#[case] reshape: bool, #[case] expected: &str) {
    let item = |nullable| Arc::new(ArrowField::new("item", DataType::Int32, nullable));
    let nest = |kind: DataType, values: ArrayRef| {
        batch_of(vec![ArrowField::new("v", kind, true)], vec![values])
    };
    // Fixed-size list: the same eight values, four rows of two against two of
    // four. List: four values, one of them null under a required item.
    let fsl = |width: i32| {
        let array = FixedSizeListArray::new(item(true), width, ints((1..=8).collect()), None);
        nest(
            DataType::FixedSizeList(item(true), width),
            Arc::new(array) as ArrayRef,
        )
    };
    let list = |values: Vec<Option<i32>>, nullable| {
        let array = ListArray::new(
            item(nullable),
            OffsetBuffer::new(vec![0, 2, 4].into()),
            Arc::new(Int32Array::from(values)) as ArrayRef,
            None,
        );
        nest(DataType::List(item(nullable)), Arc::new(array) as ArrayRef)
    };
    let (seed, staged) = if reshape {
        (fsl(2), fsl(4))
    } else {
        (
            list(vec![Some(1), Some(2), Some(3), Some(4)], false),
            list(vec![Some(10), None, Some(30), Some(40)], true),
        )
    };

    let dataset = dataset_of(seed, None).await;
    let schema = dataset.schema().clone();
    let err = stage(&dataset, staged, &schema).await.unwrap_err();
    assert!(
        err.to_string().contains(expected),
        "expected '{expected}' in error, got: {err}"
    );
}

/// Requesting the same declared field twice must be rejected inside the
/// staging contract: the per-field identity check cannot see it, and the
/// set-based batch comparison would match one column against both copies.
#[tokio::test]
async fn test_rejects_duplicate_requested_field() {
    let mut dataset = id_dataset().await;
    declare_all_null(&mut dataset, "value").await;
    let mut schema = declared_schema(&dataset, "value");
    schema.fields.push(schema.fields[0].clone());

    let batch = batch_of(
        vec![ArrowField::new("value", DataType::Int32, true)],
        vec![ints(vec![1, 2])],
    );
    let before = count_files(&dataset).await;
    let err = stage(&dataset, batch, &schema).await.unwrap_err();
    assert!(err.to_string().contains("more than once"), "got: {err}");
    assert_eq!(count_files(&dataset).await, before);
}

/// An empty stream fails the row-count gate and leaves nothing staged,
/// including the footer-only file the eagerly-opened writer creates.
#[tokio::test]
async fn test_rejects_empty_stream() {
    let mut dataset = id_dataset().await;
    declare_all_null(&mut dataset, "value").await;
    let schema = declared_schema(&dataset, "value");

    let before = count_files(&dataset).await;
    let err = only_fragment(&dataset)
        .write_columns(stream::iter(Vec::<Result<RecordBatch>>::new()), &schema)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("physical rows"), "got: {err}");
    assert_eq!(count_files(&dataset).await, before);
}

/// A visible null under a required child -- the parent is valid there, so the
/// slot is a value of the field -- is still rejected at the writer.
#[tokio::test]
async fn test_rejects_visible_null_under_required_child() {
    let dataset = dataset_of(points(&point_schema(false), [1, 2], [10, 20]), None).await;

    // The batch declares its children nullable, which staging tolerates; the
    // manifest's non-null rule is enforced against the data instead.
    let staged_children = Fields::from(vec![
        ArrowField::new("x", DataType::Int32, true),
        ArrowField::new("y", DataType::Int32, true),
    ]);
    let staged = batch_of(
        vec![ArrowField::new(
            "point",
            DataType::Struct(staged_children.clone()),
            false,
        )],
        vec![Arc::new(StructArray::new(
            staged_children,
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None])) as ArrayRef,
                ints(vec![10, 20]),
            ],
            None,
        )) as ArrayRef],
    );

    let schema = dataset.schema().clone();
    let err = stage(&dataset, staged, &schema).await.unwrap_err();
    assert!(
        err.to_string().contains("non-null"),
        "expected a nullability rejection, got: {err}"
    );
}

/// Field metadata decides physical layout -- a packed struct is one column, an
/// unpacked one a column per child -- so a caller's metadata must not be able to
/// stage a file whose coverage describes a different field set.
#[tokio::test]
async fn test_takes_layout_from_manifest() {
    let arrow_schema = point_schema(true);
    let dataset = dataset_of(
        points(&arrow_schema, [1, 2], [10, 20]),
        Some(LanceFileVersion::V2_1),
    )
    .await;
    let packed_field_id = dataset.schema().field("point").unwrap().id;

    // Identical to the manifest field but for the packed marker, which the
    // field-identity comparison does not look at.
    let mut staged_schema = dataset.schema().clone();
    staged_schema.fields[0]
        .metadata
        .remove(PACKED_STRUCT_META_KEY);
    assert!(!staged_schema.fields[0].is_packed_struct());

    let group = stage(
        &dataset,
        points(&arrow_schema, [3, 4], [30, 40]),
        &staged_schema,
    )
    .await
    .unwrap();
    // Unpacked, the file would cover x and y instead, and DataReplacement would
    // see coverage the packed field never had.
    assert_eq!(group.1.fields.as_ref(), &[packed_field_id]);

    assert_eq!(
        committed_points(&dataset, group).await,
        (vec![3, 4], vec![30, 40])
    );
}

/// Struct encoders consume children positionally, so a batch whose children are
/// ordered differently from the manifest would be written under the wrong field
/// ids. Batches are matched by name at every level instead.
#[tokio::test]
async fn test_reorders_struct_children_by_name() {
    let dataset = dataset_of(points(&point_schema(false), [1, 2], [10, 20]), None).await;

    // Names its children y-then-x: written positionally, y's values land in x.
    let reordered = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "point",
        DataType::Struct(point_children().into_iter().rev().cloned().collect()),
        false,
    )]));
    let schema = dataset.schema().clone();
    let group = stage(&dataset, points(&reordered, [30, 40], [300, 400]), &schema)
        .await
        .unwrap();

    assert_eq!(
        committed_points(&dataset, group).await,
        (vec![30, 40], vec![300, 400]),
        "each child keeps its own values"
    );
}

/// A Map is projected as a whole: its entries field carries metadata Lance
/// preserves in the schema, and a struct value's children may arrive
/// name-reordered. Projection must rebuild the map -- entries metadata intact,
/// children reordered by name -- rather than reject it.
#[tokio::test]
async fn test_stages_map_with_reordered_value_children() {
    let value_children = Fields::from(vec![
        ArrowField::new("a", DataType::Int32, true),
        ArrowField::new("b", DataType::Int32, true),
    ]);
    let entry_fields = |value_children: &Fields| {
        Fields::from(vec![
            ArrowField::new("key", DataType::Utf8, false),
            ArrowField::new("value", DataType::Struct(value_children.clone()), true),
        ])
    };
    let entries_field = |value_children: &Fields| {
        ArrowField::new(
            "entries",
            DataType::Struct(entry_fields(value_children)),
            false,
        )
        .with_metadata([("entry-semantic".to_string(), "kept".to_string())].into())
    };
    let map_batch = |value_children: &Fields,
                     a: [i32; 2],
                     b: [i32; 2],
                     offsets: Vec<i32>,
                     nulls: Option<NullBuffer>| {
        let children: Vec<ArrayRef> = value_children
            .iter()
            .map(|child| ints(if child.name() == "a" { a } else { b }.to_vec()))
            .collect();
        let value = StructArray::new(value_children.clone(), children, None);
        let entries = StructArray::new(
            entry_fields(value_children),
            vec![
                Arc::new(StringArray::from(vec!["k0", "k1"])) as ArrayRef,
                Arc::new(value) as ArrayRef,
            ],
            None,
        );
        let map = MapArray::new(
            Arc::new(entries_field(value_children)),
            OffsetBuffer::new(offsets.into()),
            entries,
            nulls,
            false,
        );
        batch_of(
            vec![ArrowField::new(
                "m",
                DataType::Map(Arc::new(entries_field(value_children)), false),
                true,
            )],
            vec![Arc::new(map) as ArrayRef],
        )
    };

    // Pinned to 2.2, the first version whose encoders accept Map.
    let dataset = dataset_of(
        map_batch(&value_children, [1, 2], [10, 20], vec![0, 1, 2], None),
        Some(LanceFileVersion::V2_2),
    )
    .await;

    // The staged batch orders the value's children b-then-a, holds both
    // entries in slot 0, and leaves slot 1 null: map validity has to
    // survive the rebuild alongside the reordering.
    let reordered: Fields = value_children.iter().rev().cloned().collect();
    let schema = dataset.schema().clone();
    let group = stage(
        &dataset,
        map_batch(
            &reordered,
            [3, 4],
            [30, 40],
            vec![0, 2, 2],
            Some(NullBuffer::from(vec![true, false])),
        ),
        &schema,
    )
    .await
    .unwrap();
    let dataset = commit(&dataset, vec![group]).await.unwrap();

    let batch = dataset.scan().try_into_batch().await.unwrap();
    let map = batch.column(0).as_map();
    assert!(map.is_valid(0), "slot 0 keeps its entries");
    assert!(map.is_null(1), "slot 1 stays null");
    let value = map.entries().column(1).as_struct();
    let child = |name: &str| {
        value
            .column_by_name(name)
            .unwrap()
            .as_primitive::<Int32Type>()
            .values()
            .to_vec()
    };
    assert_eq!(child("a"), vec![3, 4], "each child keeps its own values");
    assert_eq!(child("b"), vec![30, 40]);
}

/// Blob columns arrive logical and must be prepared into sidecars and
/// descriptors before the V2.2+ structural encoders accept them, which is what
/// the per-version update writer does.
#[tokio::test]
async fn test_stages_blob_column() {
    use crate::blob::{BlobArrayBuilder, blob_field};

    let arrow_schema = Arc::new(ArrowSchema::new(vec![blob_field("blob", true)]));
    let blobs = |values: [&[u8]; 2]| {
        let mut builder = BlobArrayBuilder::new(2);
        for value in values {
            builder.push_bytes(value).unwrap();
        }
        RecordBatch::try_new(arrow_schema.clone(), vec![builder.finish().unwrap()]).unwrap()
    };
    let dataset = dataset_of(blobs([b"one", b"two"]), Some(LanceFileVersion::V2_2)).await;

    let schema = dataset.schema().clone();
    let group = stage(&dataset, blobs([b"three", b"four"]), &schema)
        .await
        .unwrap();
    assert!(
        !group.1.fields.as_ref().is_empty(),
        "staged file must cover the blob field"
    );
}

/// The computed-column lifecycle: declare all null, backfill, compact, and
/// refresh again. The refresh after compaction is the case that previously
/// failed with "no changes were made".
#[tokio::test]
async fn test_replacement_survives_compaction() {
    let mut dataset = id_dataset_of(4, 2).await;
    declare_all_null(&mut dataset, "v").await;
    let v_id = dataset.schema().field("v").unwrap().id;

    let frag_ids: Vec<u64> = dataset
        .get_fragments()
        .iter()
        .map(|f| f.id() as u64)
        .collect();
    let mut replacements = Vec::new();
    for (i, frag_id) in frag_ids.iter().enumerate() {
        let base = i as i32 * 100;
        replacements.push(stage_column(&dataset, *frag_id, "v", vec![base + 1, base + 2]).await);
    }
    let mut dataset = commit(&dataset, replacements).await.unwrap();

    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    let files = dataset.get_fragments()[0].metadata().files.clone();
    assert_eq!(files.len(), 1, "compaction folded the column into one file");
    assert!(files[0].fields.len() > 1);

    // Refresh the compacted fragment, repeatedly: every round must land its own
    // values and reuse the appended file rather than stacking another one.
    let fragment_id = dataset.get_fragments()[0].id() as u64;
    let rows = dataset.get_fragments()[0].physical_rows().await.unwrap();
    for round in 0..3i32 {
        let refreshed: Vec<i32> = (0..rows as i32).map(|r| round * 1000 + r).collect();
        let replacement = stage_column(&dataset, fragment_id, "v", refreshed.clone()).await;
        dataset = commit(&dataset, vec![replacement]).await.unwrap();
        dataset.validate().await.unwrap();

        let batch = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(
            values(&batch, "v"),
            refreshed.iter().map(|v| Some(*v)).collect::<Vec<_>>()
        );
        assert_eq!(
            values(&batch, "id"),
            (1..=rows as i32).map(Some).collect::<Vec<_>>(),
            "round {round} disturbed a sibling column of the tombstoned file"
        );
        assert_eq!(
            dataset.get_fragments()[0].metadata().files.len(),
            2,
            "round {round} changed the file count"
        );
    }
    assert_eq!(
        dataset.schema().field("v").unwrap().id,
        v_id,
        "field id preserved"
    );

    let files = dataset.get_fragments()[0].metadata().files.clone();
    let covering: Vec<&[i32]> = files
        .iter()
        .filter(|f| f.fields.contains(&v_id))
        .map(|f| f.fields.as_ref())
        .collect();
    assert_eq!(covering.as_slice(), &[[v_id].as_slice()]);

    // Tombstoning into a wider file has to advance row lineage like any other
    // replacement, or a delta consumer never learns the refresh happened.
    let version = dataset.version().version;
    let batch = dataset
        .scan()
        .project(&["v", ROW_LAST_UPDATED_AT_VERSION])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(
        batch[ROW_LAST_UPDATED_AT_VERSION]
            .as_primitive::<UInt64Type>()
            .values(),
        vec![version; rows].as_slice()
    );
}

/// The existing uncovered (all-null backfill) and exact-match paths must be
/// unchanged.
#[tokio::test]
async fn test_existing_paths_unchanged() {
    // Uncovered -> push.
    let mut dataset = id_dataset_of(2, 1024).await;
    declare_all_null(&mut dataset, "v").await;
    let frag_id = dataset.get_fragments()[0].id() as u64;
    let r = stage_column(&dataset, frag_id, "v", vec![10, 20]).await;
    let dataset = commit(&dataset, vec![r]).await.unwrap();
    dataset.validate().await.unwrap();
    assert_eq!(
        values(&dataset.scan().try_into_batch().await.unwrap(), "v"),
        vec![Some(10), Some(20)]
    );
    let files_after_first = dataset.get_fragments()[0].metadata().files.len();

    // Exact match -> in-place swap, no new file.
    let r = stage_column(&dataset, frag_id, "v", vec![30, 40]).await;
    let dataset = commit(&dataset, vec![r]).await.unwrap();
    dataset.validate().await.unwrap();
    assert_eq!(
        values(&dataset.scan().try_into_batch().await.unwrap(), "v"),
        vec![Some(30), Some(40)]
    );
    assert_eq!(
        dataset.get_fragments()[0].metadata().files.len(),
        files_after_first,
        "exact-match replacement swaps in place rather than appending"
    );
}

/// Dropping a sibling that shares the wider file must not leave that file
/// answering for dead ids only: every data file has to share at least one
/// field with the dataset schema, or validate() reports it as corrupt and
/// cleanup can never collect it.
#[tokio::test]
async fn test_replacement_after_sibling_drop_stays_valid() {
    let batch = batch_of(
        vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("v", DataType::Int32, true),
        ],
        vec![ints(vec![1, 2]), ints(vec![10, 20])],
    );
    let mut dataset = dataset_of(batch, None).await;
    dataset.drop_columns(&["a"]).await.unwrap();

    let frag_id = dataset.get_fragments()[0].id() as u64;
    let r = stage_column(&dataset, frag_id, "v", vec![30, 40]).await;
    let dataset = commit(&dataset, vec![r]).await.unwrap();

    dataset.validate().await.unwrap();
    assert_eq!(
        values(&dataset.scan().try_into_batch().await.unwrap(), "v"),
        vec![Some(30), Some(40)]
    );
}

/// The mirror ordering: a stale handle drops the column after the
/// replacement has committed. The projection rebases over the replacement,
/// wins by commit order, and its pruning must leave no file behind that
/// answers only for the dropped field.
#[tokio::test]
async fn test_stale_column_drop_prunes_committed_replacement() {
    let mut dataset = id_dataset_of(2, 1024).await;
    declare_all_null(&mut dataset, "v").await;
    let frag_id = dataset.get_fragments()[0].id() as u64;
    let r = stage_column(&dataset, frag_id, "v", vec![10, 20]).await;
    commit(&dataset, vec![r]).await.unwrap();

    // The stale handle predates the replacement; its commit rebases over it.
    dataset.drop_columns(&["v"]).await.unwrap();

    dataset.validate().await.unwrap();
    assert!(dataset.schema().field("v").is_none());
    let live_ids: Vec<i32> = dataset.schema().fields.iter().map(|f| f.id).collect();
    for file in &dataset.get_fragments()[0].metadata().files {
        assert!(
            file.fields.iter().any(|f| live_ids.contains(f)),
            "file {} answers for no live field: {:?}",
            file.path,
            file.fields
        );
    }
}

/// The staged file is positionally aligned with physical rows, so a fragment
/// with deletions takes a value for every physical slot and the deletion
/// vector keeps masking the deleted ones afterwards.
#[tokio::test]
async fn test_replacement_preserves_deletions() {
    let mut dataset = id_dataset_of(4, 1024).await;
    declare_all_null(&mut dataset, "v").await;
    dataset.delete("id = 2").await.unwrap();
    let frag_id = dataset.get_fragments()[0].id() as u64;

    // Physical row count is still 4: the staged data covers deleted slots too.
    let r = stage_column(&dataset, frag_id, "v", vec![10, 20, 30, 40]).await;
    let dataset = commit(&dataset, vec![r]).await.unwrap();
    dataset.validate().await.unwrap();

    let batch = dataset.scan().try_into_batch().await.unwrap();
    assert_eq!(values(&batch, "id"), vec![Some(1), Some(3), Some(4)]);
    assert_eq!(values(&batch, "v"), vec![Some(10), Some(30), Some(40)]);
}

/// A projection that drops the staged column between staging and commit must
/// fail the commit: rebased over the drop, the staged file would answer for no
/// live schema field.
#[tokio::test]
async fn test_concurrent_column_drop_fails_commit() {
    let mut dataset = id_dataset_of(2, 1024).await;
    declare_all_null(&mut dataset, "v").await;
    let frag_id = dataset.get_fragments()[0].id() as u64;
    let staged = stage_column(&dataset, frag_id, "v", vec![10, 20]).await;

    // Lands after our snapshot: at commit time the field is gone.
    let mut dropper = dataset.clone();
    dropper.drop_columns(&["v"]).await.unwrap();

    let err = commit(&dataset, vec![staged]).await.unwrap_err();
    assert!(
        err.to_string().contains("dropped by concurrent"),
        "expected a field-dropped conflict, got: {err}"
    );
}

/// The legacy reader pairs a fragment's files by batch boundary, so a staged
/// file chunked to the caller's batches would leave the fragment unreadable.
#[tokio::test]
async fn test_rejects_legacy_format() {
    let dataset = dataset_of(
        arrow_array::record_batch!(("id", Int32, [1, 2])).unwrap(),
        Some(LanceFileVersion::Legacy),
    )
    .await;
    let schema = declared_schema(&dataset, "id");
    let batch = batch_of(
        vec![ArrowField::new("id", DataType::Int32, true)],
        vec![ints(vec![1, 2])],
    );
    let err = stage(&dataset, batch, &schema).await.unwrap_err();
    assert!(
        err.to_string().contains("legacy file format"),
        "expected a legacy-format rejection, got: {err}"
    );
}

/// Blob v2 spills sidecars into `data/<file-stem>/`; a rejected stage that
/// leaves them behind orphans arbitrarily large objects.
#[tokio::test]
async fn test_discards_blob_sidecars_on_failure() {
    use crate::blob::{BlobArrayBuilder, blob_field};

    let arrow_schema = Arc::new(ArrowSchema::new(vec![blob_field("blob", true)]));
    let blobs = |count: usize| {
        let mut builder = BlobArrayBuilder::new(count);
        for _ in 0..count {
            builder.push_bytes(vec![7u8; 128 * 1024]).unwrap();
        }
        RecordBatch::try_new(arrow_schema.clone(), vec![builder.finish().unwrap()]).unwrap()
    };
    let test_uri = TempStrDir::default();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(blobs(2))], arrow_schema.clone()),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let before = count_files(&dataset).await;
    // Three rows against a two-row fragment: rejected only after the sidecars
    // have been spilled.
    let schema = dataset.schema().clone();
    stage(&dataset, blobs(3), &schema).await.unwrap_err();
    assert_eq!(
        count_files(&dataset).await,
        before,
        "a rejected stage must not leave sidecars behind"
    );
}

/// A stream error after a batch was already written exits through the same
/// cleanup as a rejected batch: the staged data file and any Blob sidecars a
/// finished pack already spilled are discarded, not orphaned. The pack-file
/// threshold is pinned to one blob's size so the first batch finalizes packs
/// before the error arrives.
#[tokio::test]
async fn test_discards_staged_artifacts_on_stream_error() {
    use crate::blob::{BlobArrayBuilder, blob_field};
    use lance_arrow::BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY;

    let field = blob_field("blob", true);
    let mut metadata = field.metadata().clone();
    metadata.insert(
        BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY.to_string(),
        (128 * 1024).to_string(),
    );
    let arrow_schema = Arc::new(ArrowSchema::new(vec![field.with_metadata(metadata)]));
    let blobs = |count: usize| {
        let mut builder = BlobArrayBuilder::new(count);
        for _ in 0..count {
            builder.push_bytes(vec![7u8; 128 * 1024]).unwrap();
        }
        RecordBatch::try_new(arrow_schema.clone(), vec![builder.finish().unwrap()]).unwrap()
    };
    let test_uri = TempStrDir::default();
    let dataset = Dataset::write(
        RecordBatchIterator::new([Ok(blobs(2))], arrow_schema.clone()),
        &test_uri,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let before = count_files(&dataset).await;
    let schema = dataset.schema().clone();
    let err = only_fragment(&dataset)
        .write_columns(
            stream::iter([
                Ok(blobs(2)),
                Err(Error::invalid_input("stream failed".to_string())),
            ]),
            &schema,
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("stream failed"), "got: {err}");
    assert_eq!(
        count_files(&dataset).await,
        before,
        "a stream error must not leave staged artifacts behind"
    );
}

#[tokio::test]
async fn test_physical_slice_read_preserves_deleted_positions() {
    let mut dataset = id_dataset_of(4, 1024).await;
    dataset.delete("id = 2").await.unwrap();
    let fragment = only_fragment(&dataset);
    let schema = dataset.schema().clone();
    let batches = fragment
        .read_physical_slice(0..4, &schema, 2)
        .await
        .unwrap()
        .buffered(1)
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let batch =
        arrow::compute::concat_batches(&Arc::new(ArrowSchema::from(&schema)), &batches).unwrap();
    assert_eq!(
        batch["id"].as_primitive::<Int32Type>().values(),
        &[1, 2, 3, 4]
    );
}

async fn count_files(dataset: &Dataset) -> usize {
    dataset
        .object_store
        .read_dir_all(&dataset.data_dir(), None)
        .try_fold(0usize, |count, _| async move { Ok(count + 1) })
        .await
        .unwrap()
}
