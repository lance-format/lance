// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{Int32Type, Int64Type};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, RecordBatch,
    RecordBatchIterator, StringArray, StructArray,
};
use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema, SchemaRef};
use arrow_select::concat::concat_batches;
use futures::TryStreamExt;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::datatypes::BlobHandling;
use lance_core::utils::tempfile::TempStrDir;
use lance_core::{ROW_ADDR, ROW_ID};
use lance_datagen::{
    ArrayGeneratorExt, BatchCount, ByteCount, Dimension, RowCount, array, gen_batch,
};
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::scalar::inverted::SCORE_COL;
use lance_index::scalar::inverted::query::{MatchQuery, PhraseQuery};
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::{FullTextSearchQuery, ScalarIndexParams};
use lance_index::vector::DIST_COL;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_io::assert_io_gt;
use lance_io::stream::RecordBatchStream;
use lance_linalg::distance::{DistanceType, MetricType};
use rstest::rstest;

use crate::blob::{BlobArrayBuilder, blob_field};
use crate::dataset::scanner::MaterializationStyle;
use crate::dataset::scanner::test_dataset::{
    TestVectorDataset, assert_plan_equals, assert_query_index_field,
};
use crate::dataset::{Dataset, WriteParams};
use crate::index::DatasetIndexExt;
use crate::index::vector::{StageParams, VectorIndexParams};
use crate::io::exec::Planner;
use crate::io::exec::filtered_read::filter_read_columns;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
use crate::{Error, Result};

/// A physical deleted-row scan cannot late-materialize: the take would
/// silently drop the tombstone rows (null row id), so the row-stream
/// read must reject the forwarded flag at plan time
#[tokio::test]
async fn test_include_deleted_rows_rejects_late_materialization() {
    let data = gen_batch()
        .col("i", array::step::<Int32Type>())
        .col("payload", array::step::<Int64Type>())
        .into_reader_rows(RowCount::from(100), BatchCount::from(1));
    let mut dataset = Dataset::write(data, "memory://test", None).await.unwrap();
    dataset.delete("i = 5").await.unwrap();

    let mut scan = dataset.scan();
    scan.project(&["payload"])
        .unwrap()
        .filter("i > 2")
        .unwrap()
        .with_row_id()
        .include_deleted_rows()
        .materialization_style(MaterializationStyle::AllLate);
    let err = scan.create_plan().await.unwrap_err();
    assert!(err.to_string().contains("with_deleted_rows"), "{err}");
}

#[tokio::test]
async fn test_null_filter_on_wide_nested_column_reads_narrow_leaf() {
    // A structural column carries its validity in every leaf's definition levels, so a
    // null check on `cells` is served by the narrow `idx` leaf without reading `data`.
    let struct_fields = Fields::from(vec![
        ArrowField::new("idx", DataType::Int32, true),
        ArrowField::new("data", DataType::LargeBinary, true),
    ]);
    let item_field = Arc::new(ArrowField::new(
        "item",
        DataType::Struct(struct_fields.clone()),
        true,
    ));
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, true),
        ArrowField::new("cells", DataType::List(item_field.clone()), true),
    ]));
    // Rows cycle through: null list, empty list, one null item, one item with a null idx,
    // one plain item.  Only the first shape is `cells IS NULL`.
    let num_rows = 200usize;
    let value_len = 32 * 1024;
    let mut offsets = vec![0i32];
    let mut list_valid = Vec::with_capacity(num_rows);
    let mut item_valid = vec![];
    let mut idx = vec![];
    let mut data: Vec<Vec<u8>> = vec![];
    // Incompressible payload, so bytes read track bytes on disk.
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    for row in 0..num_rows {
        let kind = row % 5;
        list_valid.push(kind != 0);
        if kind >= 2 {
            item_valid.push(kind != 2);
            idx.push((kind != 3).then_some(row as i32));
            data.push(
                std::iter::repeat_with(|| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    rng as u8
                })
                .take(value_len)
                .collect(),
            );
        }
        offsets.push(idx.len() as i32);
    }
    // Values under a null struct item are not written, so only count the rest.
    let data_bytes = (item_valid.iter().filter(|valid| **valid).count() * value_len) as u64;
    let items = StructArray::new(
        struct_fields,
        vec![
            Arc::new(Int32Array::from(idx)) as ArrayRef,
            Arc::new(arrow_array::LargeBinaryArray::from_iter_values(data.iter())),
        ],
        Some(arrow_buffer::NullBuffer::from(item_valid)),
    );
    let cells = arrow_array::ListArray::new(
        item_field,
        arrow_buffer::OffsetBuffer::new(offsets.into()),
        Arc::new(items),
        Some(arrow_buffer::NullBuffer::from(list_valid)),
    );
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..num_rows as i32)),
            Arc::new(cells),
        ],
    )
    .unwrap();
    let test_dir = TempStrDir::default();
    let params = WriteParams {
        max_rows_per_file: 50,
        data_storage_version: Some(LanceFileVersion::V2_2),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    Dataset::write(reader, &test_dir, Some(params))
        .await
        .unwrap();

    // Baseline: materializing the column costs at least its payload.
    let dataset = Dataset::open(&test_dir).await.unwrap();
    dataset.object_store.io_stats_incremental();
    let full = dataset
        .scan()
        .project(&["cells"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(full.num_rows(), num_rows);
    let full_bytes = dataset.object_store.io_stats_incremental().read_bytes;
    assert!(full_bytes >= data_bytes, "{full_bytes} < {data_bytes}");

    for (filter, wants_null) in [("cells IS NULL", true), ("cells IS NOT NULL", false)] {
        let dataset = Dataset::open(&test_dir).await.unwrap();
        let mut scan = dataset.scan();
        scan.project(&["id"]).unwrap().filter(filter).unwrap();
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(plan.contains("projection=[id, cells.item.idx]"), "{plan}");
        dataset.object_store.io_stats_incremental();
        let batch = scan.try_into_batch().await.unwrap();
        let read_bytes = dataset.object_store.io_stats_incremental().read_bytes;
        assert_eq!(batch.schema().fields().len(), 1, "{:?}", batch.schema());
        let mut ids = batch
            .column(0)
            .as_primitive::<Int32Type>()
            .values()
            .to_vec();
        ids.sort_unstable();
        let expected = (0..num_rows as i32)
            .filter(|id| (id % 5 == 0) == wants_null)
            .collect::<Vec<_>>();
        assert_eq!(ids, expected, "{filter}");
        assert!(
            read_bytes * 10 < full_bytes,
            "{filter}: read {read_bytes} bytes of a {full_bytes} byte column"
        );
    }
}

/// Columns whose validity sits above a fixed-width leaf, with the rows that are null.
/// `null_structs` is off for formats that cannot encode struct validity (2.0).
fn null_check_shape_columns(
    num_rows: usize,
    null_structs: bool,
) -> Vec<(ArrowField, ArrayRef, Vec<i32>)> {
    let payload = |row: usize| vec![row as u8; 64];
    let rows = 0..num_rows;
    let valid = |pred: &dyn Fn(usize) -> bool| {
        Some(arrow_buffer::NullBuffer::from(
            rows.clone()
                .map(|r| !null_structs || pred(r))
                .collect::<Vec<_>>(),
        ))
    };
    let binary = |rows: &[usize]| {
        Arc::new(arrow_array::LargeBinaryArray::from_iter_values(
            rows.iter().map(|r| payload(*r)),
        )) as ArrayRef
    };
    let all_rows = rows.clone().collect::<Vec<_>>();

    // list<struct<idx, data>>: null list, empty list, null item, null idx, plain item
    let item_fields = Fields::from(vec![
        ArrowField::new("idx", DataType::Int32, true),
        ArrowField::new("data", DataType::LargeBinary, true),
    ]);
    let item_field = Arc::new(ArrowField::new(
        "item",
        DataType::Struct(item_fields.clone()),
        true,
    ));
    let mut offsets = vec![0i32];
    let mut list_valid = vec![];
    let mut item_valid = vec![];
    let mut idx = vec![];
    let mut item_rows = vec![];
    for row in rows.clone() {
        let kind = row % 5;
        list_valid.push(kind != 0);
        if kind >= 2 {
            item_valid.push(!null_structs || kind != 2);
            idx.push((kind != 3).then_some(row as i32));
            item_rows.push(row);
        }
        offsets.push(idx.len() as i32);
    }
    let items = StructArray::new(
        item_fields,
        vec![Arc::new(Int32Array::from(idx)), binary(&item_rows)],
        Some(arrow_buffer::NullBuffer::from(item_valid)),
    );
    let los = arrow_array::ListArray::new(
        item_field.clone(),
        arrow_buffer::OffsetBuffer::new(offsets.into()),
        Arc::new(items),
        Some(arrow_buffer::NullBuffer::from(list_valid)),
    );

    // struct<inner: struct<idx, data>, wide>: null outer, null inner, null idx, plain
    let inner_fields = Fields::from(vec![
        ArrowField::new("idx", DataType::Int32, true),
        ArrowField::new("data", DataType::LargeBinary, true),
    ]);
    let outer_fields = Fields::from(vec![
        ArrowField::new("inner", DataType::Struct(inner_fields.clone()), true),
        ArrowField::new("wide", DataType::LargeBinary, true),
    ]);
    let inner = StructArray::new(
        inner_fields,
        vec![
            Arc::new(Int32Array::from_iter(
                rows.clone().map(|r| (r % 4 != 2).then_some(r as i32)),
            )),
            binary(&all_rows),
        ],
        valid(&|r| r % 4 != 1),
    );
    let sos = StructArray::new(
        outer_fields,
        vec![Arc::new(inner), binary(&all_rows)],
        valid(&|r| r % 4 != 0),
    );

    // struct<idx NOT NULL, data>: a null struct whose non-nullable child has values
    let nn_fields = Fields::from(vec![
        ArrowField::new("idx", DataType::Int32, false),
        ArrowField::new("data", DataType::LargeBinary, true),
    ]);
    let nn = StructArray::new(
        nn_fields,
        vec![
            Arc::new(Int32Array::from_iter_values(rows.clone().map(|r| r as i32))),
            binary(&all_rows),
        ],
        valid(&|r| r % 3 != 0),
    );

    // list<struct<idx, data> NOT NULL>: nullable list, non-nullable item
    let lnn_item_fields = Fields::from(vec![
        ArrowField::new("idx", DataType::Int32, true),
        ArrowField::new("data", DataType::LargeBinary, true),
    ]);
    let lnn_item_field = Arc::new(ArrowField::new(
        "item",
        DataType::Struct(lnn_item_fields.clone()),
        false,
    ));
    let mut offsets = vec![0i32];
    let mut list_valid = vec![];
    let mut idx = vec![];
    let mut item_rows = vec![];
    for row in rows.clone() {
        let kind = row % 3;
        list_valid.push(kind != 0);
        if kind == 2 {
            idx.push(Some(row as i32));
            item_rows.push(row);
        }
        offsets.push(idx.len() as i32);
    }
    let lnn_items = StructArray::new(
        lnn_item_fields,
        vec![Arc::new(Int32Array::from(idx)), binary(&item_rows)],
        None,
    );
    let lnn = arrow_array::ListArray::new(
        lnn_item_field,
        arrow_buffer::OffsetBuffer::new(offsets.into()),
        Arc::new(lnn_items),
        Some(arrow_buffer::NullBuffer::from(list_valid)),
    );

    let nulls = |pred: &dyn Fn(usize) -> bool| {
        rows.clone()
            .filter(|r| pred(*r))
            .map(|r| r as i32)
            .collect::<Vec<_>>()
    };
    let struct_nulls = |pred: &dyn Fn(usize) -> bool| {
        if null_structs { nulls(pred) } else { vec![] }
    };
    vec![
        (
            ArrowField::new("los", DataType::List(item_field), true),
            Arc::new(los) as ArrayRef,
            nulls(&|r| r % 5 == 0),
        ),
        (
            ArrowField::new("sos", sos.data_type().clone(), true),
            Arc::new(sos),
            struct_nulls(&|r| r % 4 == 0),
        ),
        (
            ArrowField::new("nn", nn.data_type().clone(), true),
            Arc::new(nn),
            struct_nulls(&|r| r % 3 == 0),
        ),
        (
            ArrowField::new("lnn", lnn.data_type().clone(), true),
            Arc::new(lnn),
            nulls(&|r| r % 3 == 0),
        ),
    ]
}

/// Sorted `id`s matching `filter`, asserting the batch carries nothing but `id`.
async fn ids_matching(dataset: &Dataset, filter: &str) -> Vec<i32> {
    let batch = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .filter(filter)
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(
        batch.schema().fields().len(),
        1,
        "{filter}: {:?}",
        batch.schema()
    );
    let mut ids = batch
        .column(0)
        .as_primitive::<Int32Type>()
        .values()
        .to_vec();
    ids.sort_unstable();
    ids
}

fn null_check_read_columns(dataset: &Arc<Dataset>, filter: &str) -> Vec<String> {
    let planner = Planner::new(Arc::new(dataset.schema().into()));
    let expr = planner.parse_filter(filter).unwrap();
    filter_read_columns(&expr, &dataset.empty_projection())
}

#[rstest]
#[tokio::test]
async fn test_null_check_narrow_leaf_across_formats(
    #[values(
        LanceFileVersion::Legacy,
        LanceFileVersion::V2_0,
        LanceFileVersion::V2_1,
        LanceFileVersion::V2_2
    )]
    version: LanceFileVersion,
) {
    // The narrow leaf must answer the null check exactly as the whole column does.  From
    // 2.1 on every validity layer is stored, so both also match the written data.
    let num_rows = 60usize;
    let stores_struct_validity =
        !matches!(version, LanceFileVersion::Legacy | LanceFileVersion::V2_0);
    let shapes = null_check_shape_columns(num_rows, version != LanceFileVersion::V2_0);
    let mut fields = vec![ArrowField::new("id", DataType::Int32, true)];
    let mut columns = vec![Arc::new(Int32Array::from_iter_values(0..num_rows as i32)) as ArrayRef];
    for (field, array, _) in &shapes {
        fields.push(field.clone());
        columns.push(array.clone());
    }
    let schema = Arc::new(ArrowSchema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let test_dir = TempStrDir::default();
    let params = WriteParams {
        max_rows_per_file: 25,
        max_rows_per_group: 10,
        data_storage_version: Some(version),
        ..Default::default()
    };
    Dataset::write(reader, &test_dir, Some(params))
        .await
        .unwrap();
    let dataset = Arc::new(Dataset::open(&test_dir).await.unwrap());

    for (field, _, written_nulls) in &shapes {
        let name = field.name();
        if version == LanceFileVersion::Legacy && name == "lnn" {
            // The v1 reader cannot read back a list with a non-nullable item at all.
            continue;
        }
        let read_columns = null_check_read_columns(&dataset, &format!("{name} IS NULL"));
        assert_eq!(read_columns.len(), 1, "{name}: {read_columns:?}");
        assert!(
            read_columns[0].ends_with(".idx"),
            "{name}: {read_columns:?}"
        );

        let full = dataset
            .scan()
            .project(&["id", name])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let full_col = full.column_by_name(name).unwrap();
        let full_ids = full.column(0).as_primitive::<Int32Type>();
        let mut nulls = (0..full.num_rows())
            .filter(|i| full_col.is_null(*i))
            .map(|i| full_ids.value(i))
            .collect::<Vec<_>>();
        nulls.sort_unstable();
        if stores_struct_validity {
            assert_eq!(
                &nulls, written_nulls,
                "{version:?} {name}: full column read"
            );
        }
        let not_nulls = (0..num_rows as i32)
            .filter(|id| !nulls.contains(id))
            .collect::<Vec<_>>();
        assert_eq!(
            ids_matching(&dataset, &format!("{name} IS NULL")).await,
            nulls,
            "{version:?} {name} IS NULL"
        );
        assert_eq!(
            ids_matching(&dataset, &format!("{name} IS NOT NULL")).await,
            not_nulls,
            "{version:?} {name} IS NOT NULL"
        );
    }
}

#[tokio::test]
async fn test_null_check_atomic_columns_read_whole() {
    // Blobs, packed structs and maps are one physical column each, so a null check on
    // them is not rewritten and still answers correctly.
    let num_rows = 30usize;
    let packed_fields = Fields::from(vec![
        ArrowField::new("a", DataType::Int32, true),
        ArrowField::new("b", DataType::LargeBinary, true),
    ]);
    let packed = StructArray::new(
        packed_fields.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..num_rows as i32)) as ArrayRef,
            Arc::new(arrow_array::LargeBinaryArray::from_iter_values(
                (0..num_rows).map(|r| vec![r as u8; 16]),
            )),
        ],
        Some(arrow_buffer::NullBuffer::from(
            (0..num_rows).map(|r| r % 3 != 0).collect::<Vec<_>>(),
        )),
    );
    let mut blobs = BlobArrayBuilder::new(num_rows);
    for r in 0..num_rows {
        if r % 3 == 0 {
            blobs.push_null().unwrap();
        } else {
            blobs.push_bytes(vec![r as u8; 16]).unwrap();
        }
    }
    let legacy_blob = arrow_array::LargeBinaryArray::from_iter(
        (0..num_rows).map(|r| (r % 3 != 0).then(|| vec![r as u8; 16])),
    );
    let map_keys = (0..num_rows).map(|r| r.to_string()).collect::<Vec<_>>();
    let map_values = Int32Array::from_iter_values(0..num_rows as i32);
    let map = arrow_array::MapArray::new_from_strings(
        map_keys.iter().map(|s| s.as_str()),
        &map_values,
        &(0..=num_rows as u32).collect::<Vec<_>>(),
    )
    .unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, true),
        ArrowField::new("ps", DataType::Struct(packed_fields), true).with_metadata(HashMap::from(
            [("lance-encoding:packed".to_string(), "true".to_string())],
        )),
        blob_field("blobs", true),
        ArrowField::new("lb", DataType::LargeBinary, true).with_metadata(HashMap::from([(
            "lance-encoding:blob".to_string(),
            "true".to_string(),
        )])),
        ArrowField::new("m", map.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..num_rows as i32)),
            Arc::new(packed),
            blobs.finish().unwrap(),
            Arc::new(legacy_blob),
            Arc::new(map),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let test_dir = TempStrDir::default();
    Dataset::write(reader, &test_dir, None).await.unwrap();
    let dataset = Arc::new(Dataset::open(&test_dir).await.unwrap());

    let nulls = (0..num_rows as i32)
        .filter(|r| r % 3 == 0)
        .collect::<Vec<_>>();
    for (name, expected) in [
        ("ps", &nulls),
        ("blobs", &nulls),
        ("lb", &nulls),
        ("m", &vec![]),
    ] {
        let filter = format!("{name} IS NULL");
        assert_eq!(null_check_read_columns(&dataset, &filter), vec![name]);
        assert_eq!(&ids_matching(&dataset, &filter).await, expected, "{name}");
    }
}

#[tokio::test]
async fn test_null_check_with_late_materialized_column() {
    // The null-checked column is also projected: the narrow leaf is read eagerly, the
    // rest is taken after the filter, and the two halves merge back into the column.
    let num_rows = 60usize;
    let (field, array, nulls) = null_check_shape_columns(num_rows, true).remove(0);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, true),
        field,
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..num_rows as i32)),
            array.clone(),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let test_dir = TempStrDir::default();
    let params = WriteParams {
        max_rows_per_file: 25,
        ..Default::default()
    };
    Dataset::write(reader, &test_dir, Some(params))
        .await
        .unwrap();
    let dataset = Dataset::open(&test_dir).await.unwrap();

    for (filter, wants_null) in [
        ("los IS NULL", true),
        ("los IS NOT NULL", false),
        ("los IS NOT NULL AND id > 10", false),
    ] {
        let mut scan = dataset.scan();
        scan.project(&["id", "los"])
            .unwrap()
            .filter(filter)
            .unwrap();
        let plan = scan.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("projection=[id, los.item.idx]")
                && plan.contains("projection=[los], source=stream(_rowid)"),
            "{filter}:\n{plan}"
        );
        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch.column(0).as_primitive::<Int32Type>();
        let los = batch.column(1);
        assert_eq!(los.data_type(), array.data_type(), "{filter}");
        let expected_rows = (0..num_rows as i32)
            .filter(|id| nulls.contains(id) == wants_null && (!filter.contains("id >") || *id > 10))
            .count();
        assert_eq!(batch.num_rows(), expected_rows, "{filter}");
        for i in 0..batch.num_rows() {
            let id = ids.value(i) as usize;
            assert_eq!(
                los.slice(i, 1).as_ref(),
                array.slice(id, 1).as_ref(),
                "{filter} row {id}"
            );
        }
    }
}

#[rstest]
#[tokio::test]
async fn test_late_materialization(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    use lance_io::assert_io_lt;
    // Create a large dataset with a scalar indexed column and a sorted but not scalar
    // indexed column
    use lance_table::io::commit::RenameCommitHandler;
    let data = gen_batch()
        .col(
            "vector",
            array::rand_vec::<Float32Type>(Dimension::from(32)),
        )
        .col("indexed", array::step::<Int32Type>())
        .col("not_indexed", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(1000), BatchCount::from(20));

    let mut dataset = Dataset::write(
        data,
        "memory://test",
        Some(WriteParams {
            commit_handler: Some(Arc::new(RenameCommitHandler)),
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["indexed"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    // First run a full scan to get a baseline
    let _ = dataset.object_store.as_ref().io_stats_incremental(); // reset
    dataset.scan().try_into_batch().await.unwrap();
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    let full_scan_bytes = io_stats.read_bytes;

    // Next do a scan without pushdown, we should still see a benefit from late materialization
    dataset
        .scan()
        .use_stats(false)
        .filter("not_indexed = 50")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_lt!(io_stats, read_bytes, full_scan_bytes);
    let filtered_scan_bytes = io_stats.read_bytes;

    // Now do a scan with pushdown, the benefit should be even greater
    // Pushdown only works with the legacy format for now.
    if data_storage_version == LanceFileVersion::Legacy {
        dataset
            .scan()
            .filter("not_indexed = 50")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let io_stats = dataset.object_store.as_ref().io_stats_incremental();
        assert_io_lt!(io_stats, read_bytes, filtered_scan_bytes);
    }

    // Now do a scalar index scan, this should be better than a
    // full scan but since we have to load the index might be more
    // expensive than late / pushdown scan
    dataset
        .scan()
        .filter("indexed = 50")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_lt!(io_stats, read_bytes, full_scan_bytes);
    let index_scan_bytes = io_stats.read_bytes;

    // A second scalar index scan should be cheaper than the first
    // since we should have the index in cache
    dataset
        .scan()
        .filter("indexed = 50")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_lt!(io_stats, read_bytes, index_scan_bytes);
}

#[tokio::test]
async fn test_blob_all_binary_late_materialization() {
    // A selective filter that projects a blob column with `blob_handling=all_binary`
    // must late-materialize the blob (take only the matched rows) rather than eagerly
    // reading the whole column. Blobs returned as descriptions stay eager (they are
    // tiny), but full binary values should follow the width-based heuristic like any
    // other wide column.
    use lance_io::assert_io_lt;
    use lance_table::io::commit::RenameCommitHandler;

    // 8KB stays under the 64KB inline threshold, so the blob is a normal column in
    // the data file rather than a dedicated blob file.
    let blob_meta = HashMap::from([("lance-encoding:blob".to_string(), "true".to_string())]);
    let blobs = array::rand_fixedbin(ByteCount::from(8 * 1024), true).with_metadata(blob_meta);
    let data = gen_batch()
        .col("filterme", array::step::<UInt64Type>())
        .col("blobs", blobs)
        .into_reader_rows(RowCount::from(500), BatchCount::from(8));

    let dataset = Dataset::write(
        data,
        "memory://test",
        Some(WriteParams {
            commit_handler: Some(Arc::new(RenameCommitHandler)),
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Baseline: read the blob column as binary for the whole table.
    let _ = dataset.object_store.as_ref().io_stats_incremental(); // reset
    dataset
        .scan()
        .project(&["blobs"])
        .unwrap()
        .blob_handling(BlobHandling::AllBinary)
        .try_into_batch()
        .await
        .unwrap();
    let full_scan_bytes = dataset
        .object_store
        .as_ref()
        .io_stats_incremental()
        .read_bytes;

    // A filter matching a single row out of 4000 should read far less than the whole
    // column: only the filter leaf plus the one materialized blob.
    dataset
        .scan()
        .project(&["blobs"])
        .unwrap()
        .blob_handling(BlobHandling::AllBinary)
        .filter("filterme = 100")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_lt!(io_stats, read_bytes, full_scan_bytes);
}

#[tokio::test]
async fn test_nested_blob_all_binary_late_materialization() {
    // Same as above, but the blob is a leaf *inside* a struct and the filter is on a
    // sibling leaf. Materialization is decided per leaf (fields_pre_order), so the
    // nested blob must late-materialize under `all_binary` just like a top-level one.
    use lance_io::assert_io_lt;
    use lance_table::io::commit::RenameCommitHandler;

    let blob_meta = HashMap::from([("lance-encoding:blob".to_string(), "true".to_string())]);
    let a_field = ArrowField::new("a", DataType::Int32, false);
    let blob_field = ArrowField::new("blob", DataType::LargeBinary, false).with_metadata(blob_meta);
    let struct_fields: Fields = vec![a_field, blob_field].into();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        DataType::Struct(struct_fields.clone()),
        false,
    )]));

    let rows_per_batch = 500usize;
    let batches: Vec<RecordBatch> = (0..8)
        .map(|b| {
            let base = (b * rows_per_batch) as i32;
            let a = Arc::new(Int32Array::from_iter_values(
                base..base + rows_per_batch as i32,
            ));
            // Vary the payload per row so it does not collapse under compression.
            let blobs: Vec<Vec<u8>> = (0..rows_per_batch)
                .map(|r| {
                    let seed = (base as usize + r).wrapping_mul(2654435761);
                    (0usize..8 * 1024)
                        .map(|i| (i.wrapping_mul(31).wrapping_add(seed) & 0xff) as u8)
                        .collect()
                })
                .collect();
            let blob = Arc::new(arrow_array::LargeBinaryArray::from_iter_values(
                blobs.iter().map(|v| v.as_slice()),
            ));
            let s = StructArray::new(struct_fields.clone(), vec![a, blob as ArrayRef], None);
            RecordBatch::try_new(schema.clone(), vec![Arc::new(s)]).unwrap()
        })
        .collect();

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write(
        reader,
        "memory://test",
        Some(WriteParams {
            commit_handler: Some(Arc::new(RenameCommitHandler)),
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    let _ = dataset.object_store.as_ref().io_stats_incremental(); // reset
    dataset
        .scan()
        .project(&["s"])
        .unwrap()
        .blob_handling(BlobHandling::AllBinary)
        .try_into_batch()
        .await
        .unwrap();
    let full_scan_bytes = dataset
        .object_store
        .as_ref()
        .io_stats_incremental()
        .read_bytes;

    dataset
        .scan()
        .project(&["s"])
        .unwrap()
        .blob_handling(BlobHandling::AllBinary)
        .filter("s.a = 100")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_lt!(io_stats, read_bytes, full_scan_bytes);
}

#[rstest]
#[tokio::test]
async fn test_project_nested(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) -> Result<()> {
    let struct_i_field = ArrowField::new("i", DataType::Int32, true);
    let struct_o_field = ArrowField::new("o", DataType::Utf8, true);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(
            "struct",
            DataType::Struct(vec![struct_i_field.clone(), struct_o_field.clone()].into()),
            true,
        ),
        ArrowField::new("s", DataType::Utf8, true),
    ]));

    let input_batches: Vec<RecordBatch> = (0..5)
        .map(|i| {
            let struct_i_arr: Arc<Int32Array> =
                Arc::new(Int32Array::from_iter_values(i * 20..(i + 1) * 20));
            let struct_o_arr: Arc<StringArray> = Arc::new(StringArray::from_iter_values(
                (i * 20..(i + 1) * 20).map(|v| format!("o-{:02}", v)),
            ));
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StructArray::from(vec![
                        (Arc::new(struct_i_field.clone()), struct_i_arr as ArrayRef),
                        (Arc::new(struct_o_field.clone()), struct_o_arr as ArrayRef),
                    ])),
                    Arc::new(StringArray::from_iter_values(
                        (i * 20..(i + 1) * 20).map(|v| format!("s-{}", v)),
                    )),
                ],
            )
            .unwrap()
        })
        .collect();
    let batches =
        RecordBatchIterator::new(input_batches.clone().into_iter().map(Ok), schema.clone());
    let test_dir = TempStrDir::default();
    let test_uri = &test_dir;
    let write_params = WriteParams {
        max_rows_per_file: 40,
        max_rows_per_group: 10,
        data_storage_version: Some(data_storage_version),
        ..Default::default()
    };
    Dataset::write(batches, test_uri, Some(write_params))
        .await
        .unwrap();

    let dataset = Dataset::open(test_uri).await.unwrap();

    let batches = dataset
        .scan()
        .project(&["struct.i"])
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let batch = concat_batches(&batches[0].schema(), &batches).unwrap();
    assert!(batch.column_by_name("struct.i").is_some());
    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_plans(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
    #[values(false, true)] stable_row_id: bool,
) -> Result<()> {
    // Create a vector dataset

    use lance_index::scalar::inverted::query::BoostQuery;
    let dim = 256;
    let mut dataset =
        TestVectorDataset::new_with_dimension(data_storage_version, stable_row_id, dim).await?;
    let lance_schema = dataset.dataset.schema();

    // Scans
    // ---------------------------------------------------------------------
    // V2 writer does not use LancePushdownScan
    if data_storage_version == LanceFileVersion::Legacy {
        log::info!("Test case: Pushdown scan");
        assert_plan_equals(
            &dataset.dataset,
            |scan| scan.project(&["s"])?.filter("i > 10 and i < 20"),
            "LancePushdownScan: uri=..., projection=[s], predicate=i > Int32(10) AND i < Int32(20), row_id=false, row_addr=false, ordered=true"
        ).await?;
    }

    log::info!("Test case: Project and filter");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[s@2 as s]
  Take: columns=\"i, _rowid, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: i@0 > 10 AND i@0 < 20
        LanceScan: uri..., projection=[i], row_id=true, row_addr=false, ordered=true, range=None"
    } else {
        "ProjectionExec: expr=[s@2 as s]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    LanceRead: ..., projection=[i], num_fragments=2, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10) AND i < Int32(20), refine_filter=i > Int32(10) AND i < Int32(20)"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.use_stats(false)
                .project(&["s"])?
                .filter("i > 10 and i < 20")
        },
        expected,
    )
    .await?;

    // Integer fields will be eagerly materialized while string/vec fields
    // are not.
    log::info!("Test case: Late materialization");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@0 as i, s@1 as s, vec@3 as vec]
            Take: columns=\"i, s, _rowid, (vec)\"
              CoalesceBatchesExec: target_batch_size=8192
                FilterExec: s@1 IS NOT NULL
                  LanceScan: uri..., projection=[i, s], row_id=true, row_addr=false, ordered=true, range=None"
    } else {
        "ProjectionExec: expr=[i@0 as i, s@1 as s, vec@3 as vec]
  LanceRead: uri=..., projection=[vec], source=stream(_rowid)
    LanceRead: uri=..., projection=[i, s], num_fragments=2, range_before=None, range_after=None, \
    row_id=true, row_addr=false, full_filter=s IS NOT NULL, refine_filter=s IS NOT NULL"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| scan.use_stats(false).filter("s IS NOT NULL"),
        expected,
    )
    .await?;

    // Custom materialization
    log::info!("Test case: Custom materialization (all early)");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@0 as i, s@1 as s, vec@2 as vec]
  FilterExec: s@1 IS NOT NULL
    LanceScan: uri..., projection=[i, s, vec], row_id=true, row_addr=false, ordered=true, range=None"
    } else {
        "ProjectionExec: expr=[i@0 as i, s@1 as s, vec@2 as vec]
  LanceRead: uri=..., projection=[i, s, vec], num_fragments=2, range_before=None, \
  range_after=None, row_id=true, row_addr=false, full_filter=s IS NOT NULL, refine_filter=s IS NOT NULL"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.use_stats(false)
                .materialization_style(MaterializationStyle::AllEarly)
                .filter("s IS NOT NULL")
        },
        expected,
    )
    .await?;

    log::info!("Test case: Custom materialization 2 (all late)");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@2 as i, s@0 as s, vec@3 as vec]
  Take: columns=\"s, _rowid, (i), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: s@0 IS NOT NULL
        LanceScan: uri..., projection=[s], row_id=true, row_addr=false, ordered=true, range=None"
    } else {
        "ProjectionExec: expr=[i@2 as i, s@0 as s, vec@3 as vec]
  LanceRead: uri=..., projection=[i, vec], source=stream(_rowid)
    LanceRead: uri=..., projection=[s], num_fragments=2, range_before=None, \
    range_after=None, row_id=true, row_addr=false, full_filter=s IS NOT NULL, refine_filter=s IS NOT NULL"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.use_stats(false)
                .materialization_style(MaterializationStyle::AllLate)
                .filter("s IS NOT NULL")
        },
        expected,
    )
    .await?;

    log::info!("Test case: Custom materialization 3 (mixed)");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@3 as i, s@0 as s, vec@1 as vec]
  Take: columns=\"s, vec, _rowid, (i)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: s@0 IS NOT NULL
        LanceScan: uri..., projection=[s, vec], row_id=true, row_addr=false, ordered=true, range=None"
    } else {
        "ProjectionExec: expr=[i@3 as i, s@0 as s, vec@1 as vec]
  LanceRead: uri=..., projection=[i], source=stream(_rowid)
    LanceRead: uri=..., projection=[s, vec], num_fragments=2, range_before=None, range_after=None, \
    row_id=true, row_addr=false, full_filter=s IS NOT NULL, refine_filter=s IS NOT NULL"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.use_stats(false)
                .materialization_style(
                    MaterializationStyle::all_early_except(&["i"], lance_schema).unwrap(),
                )
                .filter("s IS NOT NULL")
        },
        expected,
    )
    .await?;

    log::info!("Test case: Scan out of order");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "LanceScan: uri=..., projection=[s], row_id=true, row_addr=false, ordered=false, range=None"
    } else {
        "LanceRead: uri=..., projection=[s], num_fragments=2, range_before=None, range_after=None, row_id=true, \
            row_addr=false, full_filter=--, refine_filter=--"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| Ok(scan.project(&["s"])?.with_row_id().scan_in_order(false)),
        expected,
    )
    .await?;

    // KNN
    // ---------------------------------------------------------------------
    let q: Float32Array = (32..32 + dim).map(|v| v as f32).collect();
    log::info!("Test case: Basic KNN");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@2 IS NOT NULL
        SortExec: TopK(fetch=5), expr=...
          KNNVectorDistance: metric=l2
            LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None"
    } else {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  LanceRead: uri=..., projection=[i, s], source=stream(_rowid)
    FilterExec: _distance@2 IS NOT NULL
      SortExec: TopK(fetch=5), expr=...
        KNNVectorDistance: metric=l2
          LanceRead: uri=..., projection=[vec], num_fragments=2, range_before=None, range_after=None, \
          row_id=true, row_addr=false, full_filter=--, refine_filter=--"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| scan.nearest("vec", &q, 5),
        expected,
    )
    .await?;

    // KNN + Limit (arguably the user, or us, should fold the limit into the KNN but we don't today)
    // ---------------------------------------------------------------------
    let q: Float32Array = (32..32 + dim).map(|v| v as f32).collect();
    log::info!("Test case: KNN with extraneous limit");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      GlobalLimitExec: skip=0, fetch=1
        FilterExec: _distance@2 IS NOT NULL
          SortExec: TopK(fetch=5), expr=...
            KNNVectorDistance: metric=l2
              LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None"
    } else {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  LanceRead: uri=..., projection=[i, s], source=stream(_rowid)
    GlobalLimitExec: skip=0, fetch=1
      FilterExec: _distance@2 IS NOT NULL
        SortExec: TopK(fetch=5), expr=...
          KNNVectorDistance: metric=l2
            LanceRead: uri=..., projection=[vec], num_fragments=2, range_before=None, range_after=None, \
            row_id=true, row_addr=false, full_filter=--, refine_filter=--"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| scan.nearest("vec", &q, 5)?.limit(Some(1), None),
        expected,
    )
    .await?;

    // ANN
    // ---------------------------------------------------------------------
    dataset.make_vector_index().await?;
    log::info!("Test case: Basic ANN");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=42), expr=...
        ANNSubIndex: name=..., k=42, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1"
    } else {
        "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  LanceRead: uri=..., projection=[i, s, vec], source=stream(_rowid)
    SortExec: TopK(fetch=42), expr=...
      ANNSubIndex: name=..., k=42, deltas=1, metric=L2
        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| scan.nearest("vec", &q, 42),
        expected,
    )
    .await?;

    log::info!("Test case: ANN with refine");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=10), expr=...
          KNNVectorDistance: metric=l2
            Take: columns=\"_distance, _rowid, (vec)\"
              CoalesceBatchesExec: target_batch_size=8192
                SortExec: TopK(fetch=40), expr=...
                  ANNSubIndex: name=..., k=40, deltas=1, metric=L2
                    ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1"
    } else {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  LanceRead: uri=..., projection=[i, s], source=stream(_rowid)
    FilterExec: _distance@... IS NOT NULL
      SortExec: TopK(fetch=10), expr=...
        KNNVectorDistance: metric=l2
          LanceRead: uri=..., projection=[vec], source=stream(_rowid)
            SortExec: TopK(fetch=40), expr=...
              ANNSubIndex: name=..., k=40, deltas=1, metric=L2
                ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| Ok(scan.nearest("vec", &q, 10)?.refine(4)),
        expected,
    )
    .await?;

    // use_index = False -> same plan as KNN
    log::info!("Test case: ANN with index disabled");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  Take: columns=\"vec, _rowid, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=13), expr=...
          KNNVectorDistance: metric=l2
            LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None"
    } else {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@0 as vec, _distance@2 as _distance]
  LanceRead: uri=..., projection=[i, s], source=stream(_rowid)
    FilterExec: _distance@... IS NOT NULL
      SortExec: TopK(fetch=13), expr=...
        KNNVectorDistance: metric=l2
          LanceRead: uri=..., projection=[vec], num_fragments=2, range_before=None, range_after=None, \
          row_id=true, row_addr=false, full_filter=--, refine_filter=--"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| Ok(scan.nearest("vec", &q, 13)?.use_index(false)),
        expected,
    )
    .await?;

    log::info!("Test case: ANN with postfilter");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[s@3 as s, vec@4 as vec, _distance@0 as _distance, _rowid@1 as _rowid]
  Take: columns=\"_distance, _rowid, i, (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: i@2 > 10
        Take: columns=\"_distance, _rowid, (i)\"
          CoalesceBatchesExec: target_batch_size=8192
            SortExec: TopK(fetch=17), expr=...
              ANNSubIndex: name=..., k=17, deltas=1, metric=L2
                ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1"
    } else {
        "ProjectionExec: expr=[s@3 as s, vec@4 as vec, _distance@0 as _distance, _rowid@1 as _rowid]
  LanceRead: uri=..., projection=[s, vec], source=stream(_rowid)
    FilterExec: i@2 > 10
      LanceRead: uri=..., projection=[i], source=stream(_rowid)
        SortExec: TopK(fetch=17), expr=...
          ANNSubIndex: name=..., k=17, deltas=1, metric=L2
            ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            Ok(scan
                .nearest("vec", &q, 17)?
                .filter("i > 10")?
                .project(&["s", "vec"])?
                .with_row_id())
        },
        expected,
    )
    .await?;

    log::info!("Test case: ANN with prefilter");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=17), expr=...
        ANNSubIndex: name=..., k=17, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
          FilterExec: i@0 > 10
            LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None"
    } else {
        "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  LanceRead: uri=..., projection=[i, s, vec], source=stream(_rowid)
    SortExec: TopK(fetch=17), expr=...
      ANNSubIndex: name=..., k=17, deltas=1, metric=L2
        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
        LanceRead: uri=..., projection=[], num_fragments=2, range_before=None, range_after=None, \
        row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)
"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            Ok(scan
                .nearest("vec", &q, 17)?
                .filter("i > 10")?
                .prefilter(true))
        },
        expected,
    )
    .await?;

    dataset.append_new_data().await?;
    log::info!("Test case: Combined KNN/ANN");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=6), expr=...
          KNNVectorDistance: metric=l2
            CoalescePartitionsExec
              UnionExec
                ProjectionExec: expr=[_distance@2 as _distance, _rowid@1 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=6), expr=...
                      KNNVectorDistance: metric=l2
                        LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=6), expr=...
                      ANNSubIndex: name=..., k=6, deltas=1, metric=L2
                        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1"
    } else {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  LanceRead: uri=..., projection=[i, s], source=stream(_rowid)
    FilterExec: _distance@... IS NOT NULL
      SortExec: TopK(fetch=6), expr=...
        KNNVectorDistance: metric=l2
          CoalescePartitionsExec
            UnionExec
              ProjectionExec: expr=[_distance@2 as _distance, _rowid@1 as _rowid, vec@0 as vec]
                FilterExec: _distance@... IS NOT NULL
                  SortExec: TopK(fetch=6), expr=...
                    KNNVectorDistance: metric=l2
                      LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None
              LanceRead: uri=..., projection=[vec], source=stream(_rowid)
                SortExec: TopK(fetch=6), expr=...
                  ANNSubIndex: name=..., k=6, deltas=1, metric=L2
                    ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| scan.nearest("vec", &q, 6),
        // TODO: we could write an optimizer rule to eliminate the last Projection
        // by doing it as part of the last Take. This would likely have minimal impact though.
        expected,
    )
    .await?;

    // new data and with filter
    log::info!("Test case: Combined KNN/ANN with postfilter");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, i, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: i@3 > 10
        Take: columns=\"_rowid, vec, _distance, (i)\"
          CoalesceBatchesExec: target_batch_size=8192
            FilterExec: _distance@... IS NOT NULL
              SortExec: TopK(fetch=15), expr=...
                KNNVectorDistance: metric=l2
                  CoalescePartitionsExec
                    UnionExec
                      ProjectionExec: expr=[_distance@2 as _distance, _rowid@1 as _rowid, vec@0 as vec]
                        FilterExec: _distance@... IS NOT NULL
                          SortExec: TopK(fetch=15), expr=...
                            KNNVectorDistance: metric=l2
                              LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None
                      Take: columns=\"_distance, _rowid, (vec)\"
                        CoalesceBatchesExec: target_batch_size=8192
                          SortExec: TopK(fetch=15), expr=...
                            ANNSubIndex: name=..., k=15, deltas=1, metric=L2
                              ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1"
    } else {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    FilterExec: i@3 > 10
      LanceRead: uri=..., projection=[i], source=stream(_rowid)
        FilterExec: _distance@... IS NOT NULL
          SortExec: TopK(fetch=15), expr=...
            KNNVectorDistance: metric=l2
              CoalescePartitionsExec
                UnionExec
                  ProjectionExec: expr=[_distance@2 as _distance, _rowid@1 as _rowid, vec@0 as vec]
                    FilterExec: _distance@... IS NOT NULL
                      SortExec: TopK(fetch=15), expr=...
                        KNNVectorDistance: metric=l2
                          LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None
                  LanceRead: uri=..., projection=[vec], source=stream(_rowid)
                    SortExec: TopK(fetch=15), expr=...
                      ANNSubIndex: name=..., k=15, deltas=1, metric=L2
                        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| scan.nearest("vec", &q, 15)?.filter("i > 10"),
        expected,
    )
    .await?;

    // new data and with prefilter
    log::info!("Test case: Combined KNN/ANN with prefilter");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=5), expr=...
          KNNVectorDistance: metric=l2
            CoalescePartitionsExec
              UnionExec
                ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=5), expr=...
                      KNNVectorDistance: metric=l2
                        FilterExec: i@1 > 10
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=5), expr=...
                      ANNSubIndex: name=..., k=5, deltas=1, metric=L2
                        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
                        FilterExec: i@0 > 10
                          LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None"
    } else {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  LanceRead: uri=..., projection=[i, s], source=stream(_rowid)
    FilterExec: _distance@... IS NOT NULL
      SortExec: TopK(fetch=5), expr=...
        KNNVectorDistance: metric=l2
          CoalescePartitionsExec
            UnionExec
              ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                FilterExec: _distance@... IS NOT NULL
                  SortExec: TopK(fetch=5), expr=...
                    KNNVectorDistance: metric=l2
                      FilterExec: i@1 > 10
                        LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
              LanceRead: uri=..., projection=[vec], source=stream(_rowid)
                SortExec: TopK(fetch=5), expr=...
                  ANNSubIndex: name=..., k=5, deltas=1, metric=L2
                    ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
                    LanceRead: uri=..., projection=[], num_fragments=2, range_before=None, range_after=None, \
                      row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            Ok(scan
                .nearest("vec", &q, 5)?
                .filter("i > 10")?
                .prefilter(true))
        },
        // TODO: i is scanned on both sides but is projected away mid-plan
        // only to be taken again later. We should fix this.
        expected,
    )
    .await?;

    // ANN with scalar index
    // ---------------------------------------------------------------------
    // Make sure both indices are up-to-date to start
    dataset.make_vector_index().await?;
    dataset.make_scalar_index().await?;

    log::info!("Test case: ANN with scalar index");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=5), expr=...
        ANNSubIndex: name=..., k=5, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
          ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
    } else {
        "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  LanceRead: uri=..., projection=[i, s, vec], source=stream(_rowid)
    SortExec: TopK(fetch=5), expr=...
      ANNSubIndex: name=..., k=5, deltas=1, metric=L2
        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
        ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            Ok(scan
                .nearest("vec", &q, 5)?
                .filter("i > 10")?
                .prefilter(true))
        },
        expected,
    )
    .await?;

    log::info!("Test case: ANN with scalar index disabled");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  Take: columns=\"_distance, _rowid, (i), (s), (vec)\"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: TopK(fetch=5), expr=...
        ANNSubIndex: name=..., k=5, deltas=1, metric=L2
          ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
          FilterExec: i@0 > 10
            LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None"
    } else {
        "ProjectionExec: expr=[i@2 as i, s@3 as s, vec@4 as vec, _distance@0 as _distance]
  LanceRead: uri=..., projection=[i, s, vec], source=stream(_rowid)
    SortExec: TopK(fetch=5), expr=...
      ANNSubIndex: name=..., k=5, deltas=1, metric=L2
        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
        LanceRead: uri=..., projection=[], num_fragments=3, range_before=None, \
        range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            Ok(scan
                .nearest("vec", &q, 5)?
                .use_scalar_index(false)
                .filter("i > 10")?
                .prefilter(true))
        },
        expected,
    )
    .await?;

    dataset.append_new_data().await?;

    log::info!("Test case: Combined KNN/ANN with scalar index");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=8), expr=...
          KNNVectorDistance: metric=l2
            CoalescePartitionsExec
              UnionExec
                ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=8), expr=...
                      KNNVectorDistance: metric=l2
                        FilterExec: i@1 > 10
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=8), expr=...
                      ANNSubIndex: name=..., k=8, deltas=1, metric=L2
                        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
                        ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
    } else {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  LanceRead: uri=..., projection=[i, s], source=stream(_rowid)
    FilterExec: _distance@... IS NOT NULL
      SortExec: TopK(fetch=8), expr=...
        KNNVectorDistance: metric=l2
          CoalescePartitionsExec
            UnionExec
              ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                FilterExec: _distance@... IS NOT NULL
                  SortExec: TopK(fetch=8), expr=...
                    KNNVectorDistance: metric=l2
                      FilterExec: i@1 > 10
                        LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
              LanceRead: uri=..., projection=[vec], source=stream(_rowid)
                SortExec: TopK(fetch=8), expr=...
                  ANNSubIndex: name=..., k=8, deltas=1, metric=L2
                    ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
                    ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            Ok(scan
                .nearest("vec", &q, 8)?
                .filter("i > 10")?
                .prefilter(true))
        },
        expected,
    )
    .await?;

    // Update scalar index but not vector index
    log::info!("Test case: Combined KNN/ANN with updated scalar index and outdated vector index");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  Take: columns=\"_rowid, vec, _distance, (i), (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      FilterExec: _distance@... IS NOT NULL
        SortExec: TopK(fetch=11), expr=...
          KNNVectorDistance: metric=l2
            CoalescePartitionsExec
              UnionExec
                ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                  FilterExec: _distance@... IS NOT NULL
                    SortExec: TopK(fetch=11), expr=...
                      KNNVectorDistance: metric=l2
                        FilterExec: i@1 > 10
                          LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
                Take: columns=\"_distance, _rowid, (vec)\"
                  CoalesceBatchesExec: target_batch_size=8192
                    SortExec: TopK(fetch=11), expr=...
                      ANNSubIndex: name=..., k=11, deltas=1, metric=L2
                        ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
                        ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
    } else {
        "ProjectionExec: expr=[i@3 as i, s@4 as s, vec@1 as vec, _distance@2 as _distance]
  LanceRead: uri=..., projection=[i, s], source=stream(_rowid)
    FilterExec: _distance@... IS NOT NULL
      SortExec: TopK(fetch=11), expr=...
        KNNVectorDistance: metric=l2
          CoalescePartitionsExec
            UnionExec
              ProjectionExec: expr=[_distance@3 as _distance, _rowid@2 as _rowid, vec@0 as vec]
                FilterExec: _distance@... IS NOT NULL
                  SortExec: TopK(fetch=11), expr=...
                    KNNVectorDistance: metric=l2
                      FilterExec: i@1 > 10
                        LanceScan: uri=..., projection=[vec, i], row_id=true, row_addr=false, ordered=false, range=None
              LanceRead: uri=..., projection=[vec], source=stream(_rowid)
                SortExec: TopK(fetch=11), expr=...
                  ANNSubIndex: name=..., k=11, deltas=1, metric=L2
                    ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1
                    ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
    };
    dataset.make_scalar_index().await?;
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            Ok(scan
                .nearest("vec", &q, 11)?
                .filter("i > 10")?
                .prefilter(true))
        },
        expected,
    )
    .await?;

    // Scans with scalar index
    // ---------------------------------------------------------------------
    log::info!("Test case: Filtered read with scalar index");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[s@1 as s]
  Take: columns=\"_rowid, (s)\"
    CoalesceBatchesExec: target_batch_size=8192
      MaterializeIndex: query=[i > 10]@i_idx(BTree)"
    } else {
        "LanceRead: uri=..., projection=[s], num_fragments=4, range_before=None, \
            range_after=None, row_id=false, row_addr=false, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| scan.project(&["s"])?.filter("i > 10"),
        expected,
    )
    .await?;

    if data_storage_version != LanceFileVersion::Legacy {
        log::info!("Test case: Filtered read with scalar index disabled (late materialization)");
        assert_plan_equals(
            &dataset.dataset,
            |scan| {
                scan.project(&["s"])?
                    .use_scalar_index(false)
                    .filter("i > 10")
            },
            "ProjectionExec: expr=[s@2 as s]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    LanceRead: uri=..., projection=[i], num_fragments=4, range_before=None, \
    range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=i > Int32(10)",
        )
        .await?;
    }

    log::info!("Test case: Empty projection");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[_rowaddr@0 as _rowaddr]
  AddRowAddrExec
    MaterializeIndex: query=[i > 10]@i_idx(BTree)"
    } else {
        "LanceRead: uri=..., projection=[], num_fragments=4, range_before=None, \
            range_after=None, row_id=false, row_addr=true, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.filter("i > 10")
                .unwrap()
                .with_row_address()
                .project::<&str>(&[])
        },
        expected,
    )
    .await?;

    dataset.append_new_data().await?;
    log::info!("Test case: Combined Scalar/non-scalar filtered read");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[s@1 as s]
  UnionExec
    Take: columns=\"_rowid, (s)\"
      CoalesceBatchesExec: target_batch_size=8192
        MaterializeIndex: query=[i > 10]@i_idx(BTree)
    ProjectionExec: expr=[_rowid@2 as _rowid, s@1 as s]
      FilterExec: i@0 > 10
        LanceScan: uri=..., projection=[i, s], row_id=true, row_addr=false, ordered=false, range=None"
    } else {
        "LanceRead: uri=..., projection=[s], num_fragments=5, range_before=None, \
            range_after=None, row_id=false, row_addr=false, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| scan.project(&["s"])?.filter("i > 10"),
        expected,
    )
    .await?;

    log::info!("Test case: Combined Scalar/non-scalar filtered read with empty projection");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[_rowaddr@0 as _rowaddr]
  UnionExec
    AddRowAddrExec
      MaterializeIndex: query=[i > 10]@i_idx(BTree)
    ProjectionExec: expr=[_rowaddr@2 as _rowaddr, _rowid@1 as _rowid]
      FilterExec: i@0 > 10
        LanceScan: uri=..., projection=[i], row_id=true, row_addr=true, ordered=false, range=None"
    } else {
        "LanceRead: uri=..., projection=[], num_fragments=5, range_before=None, \
            range_after=None, row_id=false, row_addr=true, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.filter("i > 10")
                .unwrap()
                .with_row_address()
                .project::<&str>(&[])
        },
        expected,
    )
    .await?;

    // Scans with dynamic projection
    // When an expression is specified in the projection, the plan should include a ProjectionExec
    log::info!("Test case: Dynamic projection");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        "ProjectionExec: expr=[regexp_match(s@1, .*) as matches]
  UnionExec
    Take: columns=\"_rowid, (s)\"
      CoalesceBatchesExec: target_batch_size=8192
        MaterializeIndex: query=[i > 10]@i_idx(BTree)
    ProjectionExec: expr=[_rowid@2 as _rowid, s@1 as s]
      FilterExec: i@0 > 10
        LanceScan: uri=..., row_id=true, row_addr=false, ordered=false, range=None"
    } else {
        "ProjectionExec: expr=[regexp_match(s@0, .*) as matches]
  LanceRead: uri=..., projection=[s], num_fragments=5, range_before=None, \
  range_after=None, row_id=false, row_addr=false, full_filter=i > Int32(10), refine_filter=--
    ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.project_with_transform(&[("matches", "regexp_match(s, \".*\")")])?
                .filter("i > 10")
        },
        expected,
    )
    .await?;

    // FTS
    // ---------------------------------------------------------------------
    // All rows are indexed
    dataset.make_fts_index().await?;
    log::info!("Test case: Full text search (match query)");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      MatchQuery: column=s, query=[hello]"#
    } else {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    MatchQuery: column=s, query=[hello]"#
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.project(&["s"])?
                .with_row_id()
                .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
        },
        expected,
    )
    .await?;

    log::info!("Test case: Full text search (phrase query)");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      PhraseQuery: column=s, query=hello world"#
    } else {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    PhraseQuery: column=s, query=hello world"#
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            let query = PhraseQuery::new("hello world".to_owned());
            scan.project(&["s"])?
                .with_row_id()
                .full_text_search(FullTextSearchQuery::new_query(query.into()))
        },
        expected,
    )
    .await?;

    log::info!("Test case: Full text search (boost query)");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      CompoundFtsScorer: query=Boosting(positive=Match(MatchQuery { column: Some("s"), terms: "hello", ... }), negative=Match(MatchQuery { column: Some("s"), terms: "world", ... }), negative_boost=1)"#
    } else {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    CompoundFtsScorer: query=Boosting(positive=Match(MatchQuery { column: Some("s"), terms: "hello", ... }), negative=Match(MatchQuery { column: Some("s"), terms: "world", ... }), negative_boost=1)"#
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            let positive = MatchQuery::new("hello".to_owned()).with_column(Some("s".to_owned()));
            let negative = MatchQuery::new("world".to_owned()).with_column(Some("s".to_owned()));
            let query = BoostQuery::new(positive.into(), negative.into(), Some(1.0));
            scan.project(&["s"])?
                .with_row_id()
                .full_text_search(FullTextSearchQuery::new_query(query.into()))
        },
        expected,
    )
    .await?;

    log::info!("Test case: Full text search with prefilter");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      MatchQuery: column=s, query=[hello]
        CoalescePartitionsExec
          UnionExec
            MaterializeIndex: query=[i > 10]@i_idx(BTree)
            ProjectionExec: expr=[_rowid@1 as _rowid]
              FilterExec: i@0 > 10
                LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None"#
    } else {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    MatchQuery: column=s, query=[hello]
      LanceRead: uri=..., projection=[], num_fragments=5, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=--
        ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"#
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.project(&["s"])?
                .with_row_id()
                .filter("i > 10")?
                .prefilter(true)
                .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
        },
        expected,
    )
    .await?;

    log::info!("Test case: Full text search with unindexed rows");
    // The flat-FTS path now reads through `FilteredReadExec`, matching the
    // brute-force KNN path. With no prefilter the scan still produces no
    // pushdown, but the operator differs by storage version: legacy emits
    // a `LanceScan`, v2 emits a `LanceRead` with empty filters.
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
        CoalescePartitionsExec
          UnionExec
            MatchQuery: column=s, query=[hello]
            FlatMatchQuery: column=s, query=hello
              LanceScan: uri=..., projection=[s], row_id=true, row_addr=false, ordered=true, range=None"#
    } else {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
      CoalescePartitionsExec
        UnionExec
          MatchQuery: column=s, query=[hello]
          FlatMatchQuery: column=s, query=hello
            LanceRead: uri=..., projection=[s], num_fragments=1, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=--, refine_filter=--"#
    };
    dataset.append_new_data().await?;
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.project(&["s"])?
                .with_row_id()
                .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
        },
        expected,
    )
    .await?;

    log::info!("Test case: Full text search with unindexed rows and fast_search");
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      MatchQuery: column=s, query=[hello]"#
    } else {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    MatchQuery: column=s, query=[hello]"#
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            let scan = scan
                .project(&["s"])?
                .with_row_id()
                .full_text_search(FullTextSearchQuery::new("hello".to_owned()))?;
            scan.fast_search();
            Ok(scan)
        },
        expected,
    )
    .await?;

    log::info!("Test case: Full text search with unindexed rows and prefilter");
    // After routing flat FTS through `FilteredReadExec`, the BTree on `i`
    // pushes into the unindexed-fragment scan too — no more `FilterExec` on
    // top of an unfiltered `LanceScan`. Legacy uses the `MaterializeIndex`
    // shape, v2 uses `LanceRead` with `full_filter` set.
    let expected = if data_storage_version == LanceFileVersion::Legacy {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  Take: columns="_rowid, _score, (s)"
    CoalesceBatchesExec: target_batch_size=8192
      SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
        CoalescePartitionsExec
          UnionExec
            MatchQuery: column=s, query=[hello]
              CoalescePartitionsExec
                UnionExec
                  MaterializeIndex: query=[i > 10]@i_idx(BTree)
                  ProjectionExec: expr=[_rowid@1 as _rowid]
                    FilterExec: i@0 > 10
                      LanceScan: uri=..., projection=[i], row_id=true, row_addr=false, ordered=false, range=None
            FlatMatchQuery: column=s, query=hello
              CoalescePartitionsExec
                UnionExec
                  Take: columns="_rowid, (s)"
                    CoalesceBatchesExec: target_batch_size=8192
                      MaterializeIndex: query=[i > 10]@i_idx(BTree)
                  ProjectionExec: expr=[_rowid@2 as _rowid, s@1 as s]
                    FilterExec: i@0 > 10
                      LanceScan: uri=..., projection=[i, s], row_id=true, row_addr=false, ordered=false, range=None"#
    } else {
        r#"ProjectionExec: expr=[s@2 as s, _score@1 as _score, _rowid@0 as _rowid]
  LanceRead: uri=..., projection=[s], source=stream(_rowid)
    SortExec: expr=[_score@1 DESC NULLS LAST], preserve_partitioning=[false]
      CoalescePartitionsExec
        UnionExec
          MatchQuery: column=s, query=[hello]
            LanceRead: uri=..., projection=[], num_fragments=5, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx(BTree)
          FlatMatchQuery: column=s, query=hello
            LanceRead: uri=..., projection=[s], num_fragments=1, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=i > Int32(10), refine_filter=--
              ScalarIndexQuery: query=[i > 10]@i_idx(BTree)"#
    };
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.project(&["s"])?
                .with_row_id()
                .filter("i > 10")?
                .prefilter(true)
                .full_text_search(FullTextSearchQuery::new("hello".to_owned()))
        },
        expected,
    )
    .await?;

    Ok(())
}

#[tokio::test]
async fn test_fast_search_plan() {
    // Create a vector dataset
    let mut dataset = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    dataset.make_vector_index().await.unwrap();
    dataset.append_new_data().await.unwrap();

    let q: Float32Array = (32..64).map(|v| v as f32).collect();

    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.nearest("vec", &q, 32)?
                .fast_search()
                .project(&["_distance", "_rowid"])
        },
        "SortExec: TopK(fetch=32), expr=[_distance@0 ASC NULLS LAST, _rowid@1 ASC NULLS LAST]...
    ANNSubIndex: name=idx, k=32, deltas=1, metric=L2
      ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1",
    )
    .await
    .unwrap();

    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.nearest("vec", &q, 33)?
                .fast_search()
                .with_row_id()
                .project(&["_distance", "_rowid"])
        },
        "SortExec: TopK(fetch=33), expr=[_distance@0 ASC NULLS LAST, _rowid@1 ASC NULLS LAST]...
    ANNSubIndex: name=idx, k=33, deltas=1, metric=L2
      ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1",
    )
    .await
    .unwrap();

    // Not `fast_scan` case
    assert_plan_equals(
        &dataset.dataset,
        |scan| {
            scan.nearest("vec", &q, 34)?
                .with_row_id()
                .project(&["_distance", "_rowid"])
        },
        "ProjectionExec: expr=[_distance@2 as _distance, _rowid@0 as _rowid]
  FilterExec: _distance@2 IS NOT NULL
    SortExec: TopK(fetch=34), expr=[_distance@2 ASC NULLS LAST, _rowid@0 ASC NULLS LAST]...
      KNNVectorDistance: metric=l2
        CoalescePartitionsExec
          UnionExec
            ProjectionExec: expr=[_distance@2 as _distance, _rowid@1 as _rowid, vec@0 as vec]
              FilterExec: _distance@2 IS NOT NULL
                SortExec: TopK(fetch=34), expr=[_distance@2 ASC NULLS LAST, _rowid@1 ASC NULLS LAST]...
                  KNNVectorDistance: metric=l2
                    LanceScan: uri=..., projection=[vec], row_id=true, row_addr=false, ordered=false, range=None
            LanceRead: uri=..., projection=[vec], source=stream(_rowid)
              SortExec: TopK(fetch=34), expr=[_distance@0 ASC NULLS LAST, _rowid@1 ASC NULLS LAST]...
                ANNSubIndex: name=idx, k=34, deltas=1, metric=L2
                  ANNIvfPartition: uuid=..., minimum_nprobes=1, maximum_nprobes=None, deltas=1",
    )
    .await
    .unwrap();
}

#[rstest]
#[case::vector(IndexType::Vector, DIST_COL, vec![2, 1])]
#[case::fts(IndexType::Inverted, SCORE_COL, vec![0, 2])]
#[tokio::test]
async fn test_fast_search_projection(
    #[case] index_type: IndexType,
    #[case] scoring_column: &str,
    #[case] expected_ids: Vec<i32>,
    #[values(false, true)] has_projection: bool,
    #[values(false, true)] is_row_id_requested: bool,
    #[values(false, true)] has_stable_row_ids: bool,
) {
    let mut dataset = gen_batch()
        .col("id", array::step::<Int32Type>())
        .col("text", array::cycle_utf8_literals(&["quick", "slow"]))
        .col(
            "vec",
            array::cycle_vec(array::step::<Float32Type>(), Dimension::from(2)),
        )
        .into_ram_dataset_with_params(
            FragmentCount::from(2),
            FragmentRowCount::from(2),
            Some(WriteParams {
                max_rows_per_file: 2,
                enable_stable_row_ids: has_stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    if index_type == IndexType::Vector {
        let centroids = Arc::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0, 1.0]), 2).unwrap(),
        );
        let params = VectorIndexParams::with_ivf_flat_params(
            MetricType::L2,
            IvfBuildParams::try_with_centroids(1, centroids).unwrap(),
        );
        dataset
            .create_index(&["vec"], index_type, None, &params, true)
            .await
            .unwrap();
    } else {
        dataset
            .create_index(
                &["text"],
                index_type,
                None,
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();
    }

    let mut scanner = dataset.scan();
    if index_type == IndexType::Vector {
        scanner
            .nearest("vec", &Float32Array::from(vec![4.0, 5.0]), 2)
            .unwrap();
    } else {
        scanner
            .full_text_search(FullTextSearchQuery::new("quick".to_owned()))
            .unwrap();
    }
    let mut expected_columns = if has_projection {
        scanner.project(&["id"]).unwrap();
        vec!["id"]
    } else {
        vec!["id", "text", "vec"]
    };
    expected_columns.push(scoring_column);
    if is_row_id_requested {
        scanner.with_row_id();
        expected_columns.push(ROW_ID);
    }
    scanner.limit(Some(2), None).unwrap();
    let normal_batch = scanner.try_into_batch().await.unwrap();

    scanner.fast_search();
    let fast_batch = scanner.try_into_batch().await.unwrap();
    assert_eq!(
        fast_batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        expected_columns
    );
    assert_eq!(fast_batch, normal_batch);
    assert_eq!(fast_batch.num_rows(), 2);
    let recall = fast_batch["id"]
        .as_primitive::<Int32Type>()
        .values()
        .iter()
        .filter(|id| expected_ids.contains(id))
        .count() as f64
        / expected_ids.len() as f64;
    assert_eq!(recall, 1.0);

    scanner.project(&["id", ROW_ID]).unwrap();
    let explicit_row_id_batch = scanner.try_into_batch().await.unwrap();
    assert_eq!(
        explicit_row_id_batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        if is_row_id_requested {
            vec!["id", scoring_column, ROW_ID]
        } else {
            vec!["id", ROW_ID, scoring_column]
        }
    );
    assert_eq!(&explicit_row_id_batch["id"], &fast_batch["id"]);
    assert_eq!(explicit_row_id_batch[ROW_ID].null_count(), 0);
}

#[rstest]
#[tokio::test]
async fn test_fast_search_requires_row_id_for_deleted_rows(
    #[values(false, true)] is_row_id_requested: bool,
) {
    let mut dataset = gen_batch()
        .col("id", array::step::<Int32Type>())
        .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(2))
        .await
        .unwrap();
    dataset.delete("id = 1").await.unwrap();

    let mut scanner = dataset.scan();
    scanner
        .project(&["id"])
        .unwrap()
        .fast_search()
        .include_deleted_rows();
    if is_row_id_requested {
        scanner.with_row_id();
    }

    let result = scanner.try_into_batch().await;
    if is_row_id_requested {
        let batch = result.unwrap();
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch[ROW_ID].null_count(), 1);
    } else {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("include_deleted_rows is set but with_row_id is false")
        );
    }
}

#[tokio::test]
async fn test_fast_search_without_vector_index_returns_empty() {
    let dataset = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    let q: Float32Array = (32..64).map(|v| v as f32).collect();

    let mut scanner = dataset.dataset.scan();
    scanner.nearest("vec", &q, 10).unwrap();
    let normal_rows = scanner.try_into_batch().await.unwrap().num_rows();

    let mut scanner = dataset.dataset.scan();
    scanner.nearest("vec", &q, 10).unwrap().fast_search();
    let fast_rows = scanner.try_into_batch().await.unwrap().num_rows();

    assert_eq!(normal_rows, 10);
    assert_eq!(fast_rows, 0);
}

#[tokio::test]
async fn test_batch_fast_search_without_index_returns_empty_with_query_index() {
    let dataset = TestVectorDataset::new(LanceFileVersion::Stable, true)
        .await
        .unwrap();
    let query_values = (32..96).map(|v| v as f32).collect::<Vec<_>>();
    let queries =
        FixedSizeListArray::try_new_from_values(Float32Array::from(query_values), 32).unwrap();

    let mut scanner = dataset.dataset.scan();
    scanner.nearest("vec", &queries, 2).unwrap().fast_search();
    let batch = scanner.try_into_batch().await.unwrap();

    assert_eq!(batch.num_rows(), 0);
    assert_query_index_field(&batch);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_fts_multiple_unindexed_appends() {
    // An FTS query over an indexed dataset plus more than one unindexed append
    // must return matches from every unindexed fragment, not just the first. The
    // flat search over unindexed data reads only a single input partition, so when
    // the default parallelism splits the scan across partitions it used to drop
    // every append but the first.
    let mut test_ds = TestVectorDataset::new(LanceFileVersion::Stable, false)
        .await
        .unwrap();
    test_ds.make_fts_index().await.unwrap();
    // Two separate appends -> two unindexed fragments, each large enough that the
    // physical optimizer parallelizes the flat scan across partitions.
    test_ds.append_data_with_range(400, 5400).await.unwrap();
    test_ds.append_data_with_range(5400, 10400).await.unwrap();

    // Every row's `s` value contains the token "s", so FTS("s") matches all rows.
    let total = test_ds.dataset.count_rows(None).await.unwrap();
    let returned = test_ds
        .dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("s".to_owned()))
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .try_fold(
            0usize,
            |acc, batch| async move { Ok(acc + batch.num_rows()) },
        )
        .await
        .unwrap();
    assert_eq!(returned, total);
}

#[rstest]
#[tokio::test]
async fn test_fast_search_scalar_index_skips_unindexed_fragments(
    #[values(LanceFileVersion::Stable)] data_storage_version: LanceFileVersion,
) {
    let mut dataset = TestVectorDataset::new(data_storage_version, false)
        .await
        .unwrap();
    dataset.make_scalar_index().await.unwrap();
    dataset.append_new_data().await.unwrap();

    let mut scanner = dataset.dataset.scan();
    scanner.filter("i >= 395").unwrap().project(&["i"]).unwrap();
    let normal_batch = scanner.try_into_batch().await.unwrap();

    let mut scanner = dataset.dataset.scan();
    scanner
        .filter("i >= 395")
        .unwrap()
        .fast_search()
        .project(&["i"])
        .unwrap();
    let fast_batch = scanner.try_into_batch().await.unwrap();

    assert_eq!(normal_batch.num_rows(), 15);
    assert_eq!(fast_batch.num_rows(), 5);

    let mut project_first_scanner = dataset.dataset.scan();
    project_first_scanner
        .filter("i >= 395")
        .unwrap()
        .project(&["i"])
        .unwrap()
        .fast_search();
    assert_eq!(
        project_first_scanner.explain_plan(false).await.unwrap(),
        scanner.explain_plan(false).await.unwrap()
    );
    assert_eq!(
        project_first_scanner.try_into_batch().await.unwrap(),
        fast_batch
    );
}

fn make_scalar_filter_test_batch(schema: SchemaRef, start: i32, end: i32) -> RecordBatch {
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from_iter_values(start..end)),
            Arc::new(Int32Array::from_iter_values(start..end)),
        ],
    )
    .unwrap()
}

async fn make_scalar_filter_test_dataset(
    data_storage_version: LanceFileVersion,
) -> (TempStrDir, SchemaRef, Dataset) {
    let tmp_dir = TempStrDir::default();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", DataType::Int32, false),
        ArrowField::new("b", DataType::Int32, false),
    ]));
    let batch = make_scalar_filter_test_batch(schema.clone(), 0, 100);
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let dataset = Dataset::write(
        reader,
        &tmp_dir,
        Some(WriteParams {
            max_rows_per_file: 100,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    (tmp_dir, schema, dataset)
}

async fn append_scalar_filter_test_data(
    dataset: &mut Dataset,
    schema: SchemaRef,
    start: i32,
    end: i32,
) {
    let batch = make_scalar_filter_test_batch(schema.clone(), start, end);
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    dataset.append(reader, None).await.unwrap();
}

async fn create_scalar_index(dataset: &mut Dataset, column: &str) {
    dataset
        .create_index(
            &[column],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
}

async fn scan_count(dataset: &Dataset, filter: &str, fast_search: bool) -> usize {
    let mut scanner = dataset.scan();
    scanner
        .filter(filter)
        .unwrap()
        .project(&["a", "b"])
        .unwrap();
    if fast_search {
        scanner.fast_search();
    }
    scanner.try_into_batch().await.unwrap().num_rows()
}

#[rstest]
#[tokio::test]
async fn test_fast_search_scalar_index_filter_coverage_cases(
    #[values(LanceFileVersion::Stable)] data_storage_version: LanceFileVersion,
) {
    let (_tmp_dir, schema, mut dataset) =
        make_scalar_filter_test_dataset(data_storage_version).await;
    create_scalar_index(&mut dataset, "a").await;
    append_scalar_filter_test_data(&mut dataset, schema, 100, 110).await;

    // a is indexed and b is not. The indexed side finds candidates in covered
    // fragments and b is applied as a refine filter.
    assert_eq!(scan_count(&dataset, "a >= 95 AND b >= 95", false).await, 15);
    assert_eq!(scan_count(&dataset, "a >= 95 AND b >= 95", true).await, 5);

    // OR cannot safely skip unindexed fragments: a row in an unindexed fragment
    // may satisfy `b >= 105` even if `a` is not indexed there. Skipping it would
    // silently drop valid results, so fast_search has no effect on OR queries where
    // any branch lacks a scalar index.
    assert_eq!(scan_count(&dataset, "a >= 105 OR b >= 105", false).await, 5);
    assert_eq!(scan_count(&dataset, "a >= 105 OR b >= 105", true).await, 5);

    // A single-column indexed filter skips the appended fragment in fast mode.
    assert_eq!(scan_count(&dataset, "a >= 95", false).await, 15);
    assert_eq!(scan_count(&dataset, "a >= 95", true).await, 5);

    let (_tmp_dir, schema, mut dataset) =
        make_scalar_filter_test_dataset(data_storage_version).await;
    create_scalar_index(&mut dataset, "a").await;
    append_scalar_filter_test_data(&mut dataset, schema, 100, 110).await;
    create_scalar_index(&mut dataset, "b").await;

    // a and b are both indexed, but a only covers the original fragment while b
    // covers both fragments. Fast search only reads the shared indexed coverage.
    assert_eq!(scan_count(&dataset, "a >= 95 AND b >= 95", false).await, 15);
    assert_eq!(scan_count(&dataset, "a >= 95 AND b >= 95", true).await, 5);

    let (_tmp_dir, schema, mut dataset) =
        make_scalar_filter_test_dataset(data_storage_version).await;
    append_scalar_filter_test_data(&mut dataset, schema, 100, 110).await;

    // With no scalar index query, fast_search must not enable indexed-fragment
    // pruning. This guards against treating any ordinary filter as indexed.
    assert_eq!(scan_count(&dataset, "a >= 95", false).await, 15);
    assert_eq!(scan_count(&dataset, "a >= 95", true).await, 15);
}

#[rstest]
#[tokio::test]
pub async fn test_scan_planning_io(
    #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
    data_storage_version: LanceFileVersion,
) {
    // Create a large dataset with a scalar indexed column and a sorted but not scalar
    // indexed column

    use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
    use lance_io::assert_io_eq;
    let data = gen_batch()
        .col(
            "vector",
            array::rand_vec::<Float32Type>(Dimension::from(32)),
        )
        .col("text", array::rand_utf8(ByteCount::from(4), false))
        .col("indexed", array::step::<Int32Type>())
        .col("not_indexed", array::step::<Int32Type>())
        .into_reader_rows(RowCount::from(100), BatchCount::from(5));

    let mut dataset = Dataset::write(
        data,
        "memory://test",
        Some(WriteParams {
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["indexed"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            None,
            &VectorIndexParams {
                metric_type: DistanceType::L2,
                stages: vec![
                    StageParams::Ivf(IvfBuildParams {
                        max_iters: 2,
                        num_partitions: Some(2),
                        sample_rate: 2,
                        ..Default::default()
                    }),
                    StageParams::PQ(PQBuildParams {
                        max_iters: 2,
                        num_sub_vectors: 2,
                        ..Default::default()
                    }),
                ],
                version: crate::index::vector::IndexFileVersion::Legacy,
                skip_transpose: false,
                runtime_hints: Default::default(),
            },
            false,
        )
        .await
        .unwrap();

    // First planning cycle needs to do some I/O to determine what scalar indices are available
    dataset
        .scan()
        .prefilter(true)
        .filter("indexed > 10")
        .unwrap()
        .explain_plan(true)
        .await
        .unwrap();

    // First pass will need to perform some IOPs to determine what scalar indices are available
    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_gt!(io_stats, read_iops, 0);

    // Second planning cycle should not perform any I/O
    dataset
        .scan()
        .prefilter(true)
        .filter("indexed > 10")
        .unwrap()
        .explain_plan(true)
        .await
        .unwrap();

    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0);

    dataset
        .scan()
        .prefilter(true)
        .filter("true")
        .unwrap()
        .explain_plan(true)
        .await
        .unwrap();

    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0);

    dataset
        .scan()
        .prefilter(true)
        .materialization_style(MaterializationStyle::AllEarly)
        .filter("true")
        .unwrap()
        .explain_plan(true)
        .await
        .unwrap();

    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0);

    dataset
        .scan()
        .prefilter(true)
        .materialization_style(MaterializationStyle::AllLate)
        .filter("true")
        .unwrap()
        .explain_plan(true)
        .await
        .unwrap();

    let io_stats = dataset.object_store.as_ref().io_stats_incremental();
    assert_io_eq!(io_stats, read_iops, 0);
}

#[rstest]
#[tokio::test]
pub async fn test_row_meta_columns(
    #[values(
        (true, false),  // Test row_id only
        (false, true),  // Test row_address only
        (true, true)    // Test both
    )]
    columns: (bool, bool),
) {
    let (with_row_id, with_row_address) = columns;
    let test_dir = TempStrDir::default();
    let uri = &test_dir;

    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("data_item_id", arrow_schema::DataType::Int32, false),
        arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
    ]));

    let data = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1001, 1002, 1003])),
            Arc::new(Int32Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();

    let dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(data)], schema.clone()),
        uri,
        None,
    )
    .await
    .unwrap();

    // Test explicit projection
    let mut scanner = dataset.scan();

    let mut projection = vec!["data_item_id".to_string()];
    if with_row_id {
        scanner.with_row_id();
        projection.push(ROW_ID.to_string());
    }
    if with_row_address {
        scanner.with_row_address();
        projection.push(ROW_ADDR.to_string());
    }

    scanner.project(&projection).unwrap();
    let stream = scanner.try_into_stream().await.unwrap();
    let batch = stream.try_collect::<Vec<_>>().await.unwrap().pop().unwrap();

    // Verify column existence and data type
    if with_row_id {
        let column = batch.column_by_name(ROW_ID).unwrap();
        assert_eq!(column.data_type(), &DataType::UInt64);
    }
    if with_row_address {
        let column = batch.column_by_name(ROW_ADDR).unwrap();
        assert_eq!(column.data_type(), &DataType::UInt64);
    }

    // Test implicit inclusion
    let mut scanner = dataset.scan();
    if with_row_id {
        scanner.with_row_id();
    }
    if with_row_address {
        scanner.with_row_address();
    }
    scanner.project(&["data_item_id"]).unwrap();
    let stream = scanner.try_into_stream().await.unwrap();
    let batch = stream.try_collect::<Vec<_>>().await.unwrap().pop().unwrap();
    let meta_column = batch.column_by_name(if with_row_id { ROW_ID } else { ROW_ADDR });
    assert!(meta_column.is_some());

    // Test error case
    let mut scanner = dataset.scan();
    if with_row_id {
        scanner.project(&[ROW_ID]).unwrap();
    } else {
        scanner.project(&[ROW_ADDR]).unwrap();
    };
    let stream = scanner.try_into_stream().await.unwrap();
    assert_eq!(stream.schema().fields().len(), 1);
    if with_row_id {
        assert!(stream.schema().field_with_name(ROW_ID).is_ok());
    } else {
        assert!(stream.schema().field_with_name(ROW_ADDR).is_ok());
    }
}
