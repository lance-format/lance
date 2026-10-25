// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::vec;

use crate::Dataset;
use crate::dataset::tests::dataset_transactions::execute_sql;
use crate::dataset::transaction::{Operation, Transaction, UpdateMode};
use crate::dataset::{
    MergeInsertBuilder, UpdateBuilder, WhenMatched, WhenNotMatched, WhenNotMatchedBySource,
    WriteParams,
};

use crate::index::DatasetIndexExt;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchIterator;
use arrow_array::cast::AsArray;
use arrow_array::record_batch;
use arrow_array::types::{Float64Type, Int32Type, Int64Type};
use arrow_array::{Float64Array, Int64Array, StringArray};
use arrow_schema::{DataType, Field};
use datafusion::common::{assert_contains, assert_not_contains};
use geo_types::{Rect, coord, line_string};
use geoarrow_array::{
    GeoArrowArray,
    builder::{LineStringBuilder, PointBuilder, PolygonBuilder},
};
use geoarrow_schema::{Dimension, LineStringType, PointType, PolygonType};
use lance_arrow::RecordBatchExt;
use lance_core::ROW_ID;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
use rstest::rstest;

#[tokio::test]
async fn test_geo_types() {
    // 1. Creates arrow table with spatial data.
    let point_type = PointType::new(Dimension::XY, Default::default());
    let line_string_type = LineStringType::new(Dimension::XY, Default::default());
    let polygon_type = PolygonType::new(Dimension::XY, Default::default());

    let schema = arrow_schema::Schema::new(vec![
        point_type.clone().to_field("point", true),
        line_string_type.clone().to_field("linestring", true),
        polygon_type.clone().to_field("polygon", true),
    ]);
    let schema = Arc::new(schema) as arrow_schema::SchemaRef;

    let mut point_builder = PointBuilder::new(point_type.clone());
    point_builder.push_point(Some(&geo_types::point!(x: -72.1235, y: 42.3521)));
    let point_arr = point_builder.finish();

    let mut line_string_builder = LineStringBuilder::new(line_string_type.clone());
    line_string_builder
        .push_line_string(Some(&line_string![
        (x: -72.1260, y: 42.45),
        (x: -72.123, y: 42.1546),
        (x: -73.123, y: 43.1546),
        ]))
        .unwrap();
    let line_arr = line_string_builder.finish();

    let mut polygon_builder = PolygonBuilder::new(polygon_type.clone());
    let rect = Rect::new(
        coord! { x: -72.123, y: 42.146 },
        coord! { x: -72.126, y: 42.45 },
    );
    polygon_builder.push_rect(Some(&rect)).unwrap();
    let polygon_arr = polygon_builder.finish();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            point_arr.to_array_ref(),
            line_arr.to_array_ref(),
            polygon_arr.to_array_ref(),
        ],
    )
    .unwrap();

    // 2. Write to lance
    let lance_path = TempStrDir::default();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write(reader, &lance_path, Some(Default::default()))
        .await
        .unwrap();

    // 3. Verifies that the schema fields and extension metadata are preserved
    assert_eq!(dataset.schema().fields.len(), 3);
    let fields = &dataset.schema().fields;
    assert_eq!(
        fields.first().unwrap().metadata.get("ARROW:extension:name"),
        Some(&"geoarrow.point".to_owned())
    );
    assert_eq!(
        fields.get(1).unwrap().metadata.get("ARROW:extension:name"),
        Some(&"geoarrow.linestring".to_owned())
    );
    assert_eq!(
        fields.get(2).unwrap().metadata.get("ARROW:extension:name"),
        Some(&"geoarrow.polygon".to_owned())
    );
}

#[tokio::test]
async fn test_geo_sql() {
    // 1. Creates arrow table with point and linestring spatial data
    let point_type = PointType::new(Dimension::XY, Default::default());
    let line_string_type = LineStringType::new(Dimension::XY, Default::default());

    let schema = arrow_schema::Schema::new(vec![
        point_type.clone().to_field("point", true),
        line_string_type.clone().to_field("linestring", true),
    ]);
    let schema = Arc::new(schema) as arrow_schema::SchemaRef;

    let mut point_builder = PointBuilder::new(point_type.clone());
    point_builder.push_point(Some(&geo_types::point!(x: -72.1235, y: 42.3521)));
    let point_arr = point_builder.finish();

    let mut line_string_builder = LineStringBuilder::new(line_string_type.clone());
    line_string_builder
        .push_line_string(Some(&line_string![
        (x: -72.1260, y: 42.45),
        (x: -72.123, y: 42.1546),
        (x: -73.123, y: 43.1546),
        ]))
        .unwrap();
    let line_arr = line_string_builder.finish();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![point_arr.to_array_ref(), line_arr.to_array_ref()],
    )
    .unwrap();

    // 2. Write to lance
    let lance_path = TempStrDir::default();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write(reader, &lance_path, Some(Default::default()))
        .await
        .unwrap();

    // 3. Executes a SQL query with St_Distance function
    let batches = execute_sql(
        "SELECT ST_Distance(point, linestring) AS dist FROM dataset",
        "dataset".to_owned(),
        Arc::new(dataset.clone()),
    )
    .await
    .unwrap();
    assert_eq!(batches.len(), 1);
    let batch = batches.first().unwrap();
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(batch.num_rows(), 1);
    approx::assert_relative_eq!(
        batch.column(0).as_primitive::<Float64Type>().value(0),
        0.0015056772638228177
    );
}

#[tokio::test]
async fn test_geo_rtree_index() {
    // 1. Creates arrow table linestring spatial data
    let line_string_type = LineStringType::new(Dimension::XY, Default::default());

    let schema =
        arrow_schema::Schema::new(vec![line_string_type.clone().to_field("linestring", true)]);
    let schema = Arc::new(schema) as arrow_schema::SchemaRef;

    let num_rows = 10000;
    let mut line_string_builder = LineStringBuilder::new(line_string_type.clone());
    for i in 0..num_rows {
        let i = i as f64;
        line_string_builder
            .push_line_string(Some(&line_string![
                (x: i, y: i),
                (x: i + 1.0, y: i + 1.0)
            ]))
            .unwrap();
    }
    let line_arr = line_string_builder.finish();

    let batch = RecordBatch::try_new(schema.clone(), vec![line_arr.to_array_ref()]).unwrap();

    // 2. Write to lance
    let lance_path = TempStrDir::default();
    let reader = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema.clone());
    let mut dataset = Dataset::write(reader, &lance_path, Some(Default::default()))
        .await
        .unwrap();

    async fn assert_intersects_sql(dataset: &mut Dataset, has_index: bool) {
        // Executes a SQL query with St_Distance function
        let sql = "SELECT linestring from dataset where St_Intersects(linestring, ST_GeomFromText('LINESTRING ( 2 0, 0 2 )'))";
        let batches = dataset
            .sql(sql)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();

        let mut num_rows = 0;
        for b in batches {
            num_rows += b.num_rows();
        }
        assert_eq!(2, num_rows);

        let batches = dataset
            .sql(&format!("Explain {}", sql))
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let plan = format!("{:?}", batches);
        if has_index {
            assert_contains!(&plan, "ScalarIndexQuery");
        } else {
            assert_not_contains!(&plan, "ScalarIndexQuery");
        }
    }

    assert_intersects_sql(&mut dataset, false).await;

    dataset
        .create_index(
            &["linestring"],
            IndexType::RTree,
            Some("rtree_index".to_string()),
            &ScalarIndexParams::new("RTree".to_string()),
            true,
        )
        .await
        .unwrap();

    assert_intersects_sql(&mut dataset, true).await;
}

#[rstest]
#[case::stale_unspecified(false, None)]
#[case::latest_unspecified(true, None)]
#[case::stale_rewrite_columns(false, Some(UpdateMode::RewriteColumns))]
#[case::latest_rewrite_columns(true, Some(UpdateMode::RewriteColumns))]
#[tokio::test]
async fn test_staged_rtree_after_rewrite_columns(
    #[case] checkout_latest: bool,
    #[case] update_mode: Option<UpdateMode>,
) {
    let point_type = PointType::new(Dimension::XY, Default::default());
    let mut points = PointBuilder::new(point_type.clone());
    for point in [
        Some(geo_types::point!(x: 0.0, y: 0.0)),
        Some(geo_types::point!(x: 1.0, y: 1.0)),
        None,
        Some(geo_types::point!(x: 10.0, y: 10.0)),
    ] {
        points.push_point(point.as_ref());
    }
    let batch = record_batch!(("id", Int32, [0, 1, 2, 3]))
        .unwrap()
        .try_with_column(
            point_type.to_field("point", true),
            points.finish().to_array_ref(),
        )
        .unwrap();
    let schema = batch.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    let segment = dataset
        .create_index_builder(
            &["point"],
            IndexType::RTree,
            &ScalarIndexParams::new("RTree".to_string()),
        )
        .name("point_rtree".to_string())
        .fragments(vec![0, 1])
        .execute_uncommitted()
        .await
        .unwrap();

    let mut points = PointBuilder::new(point_type.clone());
    points.push_point(Some(&geo_types::point!(x: 10.0, y: 10.0)));
    let update = record_batch!((ROW_ID, UInt64, [0]))
        .unwrap()
        .try_with_column(
            point_type.to_field("point", true),
            points.finish().to_array_ref(),
        )
        .unwrap();
    let update_schema = update.schema();
    let (fragment, fields_modified) = dataset
        .get_fragment(0)
        .unwrap()
        .update_columns(
            RecordBatchIterator::new([Ok(update)], update_schema),
            ROW_ID,
            ROW_ID,
        )
        .await
        .unwrap();
    let point_field_id = dataset.schema().field("point").unwrap().id as u32;
    let id_field_id = dataset.schema().field("id").unwrap().id as u32;
    assert!(fields_modified.contains(&point_field_id));
    assert!(!fields_modified.contains(&id_field_id));
    let mut updated = dataset.clone();
    updated
        .apply_commit(
            Transaction::new(
                dataset.version().version,
                Operation::Update {
                    updated_fragments: vec![fragment],
                    fields_modified,
                    removed_fragment_ids: vec![],
                    new_fragments: vec![],
                    compacted_sstables: vec![],
                    fields_for_preserving_frag_bitmap: vec![],
                    update_mode,
                    inserted_rows_filter: None,
                    updated_fragment_offsets: None,
                },
                None,
            ),
            &Default::default(),
            &Default::default(),
        )
        .await
        .unwrap();
    if checkout_latest {
        dataset = updated;
    }
    dataset
        .commit_existing_index_segments("point_rtree", "point", vec![segment])
        .await
        .unwrap();

    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(
        indices[0].fragment_bitmap.as_ref().unwrap(),
        &roaring::RoaringBitmap::from_iter([1]),
    );
    for (filter, expected) in [
        (
            "ST_Intersects(point, ST_GeomFromText('POINT (10 10)'))",
            vec![0, 3],
        ),
        (
            "ST_Intersects(point, ST_GeomFromText('POINT (0 0)'))",
            vec![],
        ),
        (
            "ST_Intersects(point, ST_GeomFromText('POINT (1 1)'))",
            vec![1],
        ),
        ("point IS NULL", vec![2]),
    ] {
        for use_scalar_index in [false, true] {
            let mut scanner = dataset.scan();
            scanner
                .project(&["id"])
                .unwrap()
                .filter(filter)
                .unwrap()
                .use_scalar_index(use_scalar_index);
            if use_scalar_index && filter.starts_with("ST_Intersects") {
                assert_contains!(
                    scanner.explain_plan(false).await.unwrap(),
                    "ScalarIndexQuery"
                );
            }
            let batch = scanner.try_into_batch().await.unwrap();
            let mut ids = batch["id"].as_primitive::<Int32Type>().values().to_vec();
            ids.sort_unstable();
            assert_eq!(
                ids, expected,
                "{filter}, use_scalar_index={use_scalar_index}"
            );
        }
        let batches = dataset
            .sql(&format!(
                "SELECT id FROM dataset WHERE {filter} ORDER BY id"
            ))
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let ids = batches
            .iter()
            .flat_map(|batch| batch["id"].as_primitive::<Int32Type>().values().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(ids, expected, "SQL: {filter}");
    }
}

/// The square `(0 0)-(4 4)` as a constant geometry.
const REGION: &str = "ST_GeomFromText('POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))')";
const NULL_GEOMETRY: &str = "ST_GeomFromText(CAST(NULL AS VARCHAR))";

/// Point coordinates of rows 0..=6: inside [`REGION`] (0, 1), on its boundary
/// (2, 3), outside it (4, 5), and a NULL geometry (6) whose `x`/`y` lie outside.
const POINTS: [(f64, f64); 7] = [
    (1.0, 1.0),
    (2.0, 3.0),
    (0.0, 2.0),
    (4.0, 4.0),
    (5.0, 5.0),
    (-1.0, 2.0),
    (10.0, 10.0),
];
const NULL_GEOMETRY_ID: usize = 6;

/// Rows with a point `geom`, the same coordinates as plain `x`/`y` columns, and a
/// `label` for update tests, written over several fragments.
async fn points_around_region(has_index: bool) -> Dataset {
    let point_type = PointType::new(Dimension::XY, Default::default());
    let mut geom_builder = PointBuilder::new(point_type.clone());
    for (id, (x, y)) in POINTS.iter().enumerate() {
        let point = (id != NULL_GEOMETRY_ID).then(|| geo_types::point!(x: *x, y: *y));
        geom_builder.push_point(point.as_ref());
    }
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        point_type.to_field("geom", true),
        Field::new("label", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(0..POINTS.len() as i64)),
            Arc::new(Float64Array::from_iter_values(POINTS.iter().map(|p| p.0))),
            Arc::new(Float64Array::from_iter_values(POINTS.iter().map(|p| p.1))),
            geom_builder.finish().to_array_ref(),
            Arc::new(StringArray::from(vec!["original"; POINTS.len()])),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let params = WriteParams {
        max_rows_per_file: 3,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, "memory://", Some(params))
        .await
        .unwrap();
    assert!(dataset.get_fragments().len() > 1);
    if has_index {
        dataset
            .create_index(
                &["geom"],
                IndexType::RTree,
                None,
                &ScalarIndexParams::new("RTree".to_string()),
                true,
            )
            .await
            .unwrap();
    }
    dataset
}

async fn ids(dataset: &Dataset, filter: &str) -> Vec<i64> {
    let batch = dataset
        .scan()
        .filter(filter)
        .unwrap()
        .project(&["id"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let mut ids = batch["id"].as_primitive::<Int64Type>().values().to_vec();
    ids.sort_unstable();
    ids
}

fn sql_ids(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|batch| batch["id"].as_primitive::<Int64Type>().values().to_vec())
        .collect()
}

/// Every spatial predicate must select the same rows with the constant first as
/// its column-first converse does, through the scanner filter, `Dataset::sql`,
/// and a DataFusion session built with `register_functions`.
#[rstest]
#[case::contains("ST_Contains({c}, geom)", "ST_Within(geom, {c})", vec![0, 1], true)]
#[case::within("ST_Within({c}, geom)", "ST_Contains(geom, {c})", vec![], true)]
#[case::covers("ST_Covers({c}, geom)", "ST_CoveredBy(geom, {c})", vec![0, 1, 2, 3], true)]
#[case::covered_by("ST_CoveredBy({c}, geom)", "ST_Covers(geom, {c})", vec![], true)]
#[case::intersects(
    "ST_Intersects({c}, geom)",
    "ST_Intersects(geom, {c})",
    vec![0, 1, 2, 3],
    true
)]
#[case::touches("ST_Touches({c}, geom)", "ST_Touches(geom, {c})", vec![2, 3], true)]
#[case::disjoint("ST_Disjoint({c}, geom)", "ST_Disjoint(geom, {c})", vec![4, 5], false)]
#[case::expression_argument(
    "ST_Contains({c}, ST_Point(x, y))",
    "ST_Within(ST_Point(x, y), {c})",
    vec![0, 1],
    false
)]
#[tokio::test]
async fn test_geo_constant_first_predicates(
    #[case] constant_first: &str,
    #[case] column_first: &str,
    #[case] expected: Vec<i64>,
    #[case] is_indexable: bool,
    #[values(false, true)] has_index: bool,
) {
    let dataset = points_around_region(has_index).await;
    let constant_first = constant_first.replace("{c}", REGION);
    let column_first = column_first.replace("{c}", REGION);
    assert_eq!(ids(&dataset, &constant_first).await, expected);
    assert_eq!(ids(&dataset, &column_first).await, expected);

    let sql = format!("SELECT id FROM dataset WHERE {constant_first} ORDER BY id");
    let batches = dataset
        .sql(&sql)
        .build()
        .await
        .unwrap()
        .into_batch_records()
        .await
        .unwrap();
    assert_eq!(sql_ids(&batches), expected);
    let batches = execute_sql(&sql, "dataset".to_owned(), Arc::new(dataset.clone()))
        .await
        .unwrap();
    assert_eq!(sql_ids(&batches), expected);

    let plan = dataset
        .scan()
        .filter(&constant_first)
        .unwrap()
        .explain_plan(true)
        .await
        .unwrap();
    if has_index && is_indexable {
        assert_contains!(&plan, "ScalarIndexQuery");
    } else {
        assert_not_contains!(&plan, "ScalarIndexQuery");
    }
}

/// A NULL constant geometry makes every predicate NULL instead of panicking.
#[rstest]
#[tokio::test]
async fn test_geo_null_constant_geometry(#[values(false, true)] has_index: bool) {
    let mut dataset = points_around_region(has_index).await;
    for filter in [
        format!("ST_Contains({NULL_GEOMETRY}, geom)"),
        format!("ST_Within(geom, {NULL_GEOMETRY})"),
        format!("NOT ST_Contains({NULL_GEOMETRY}, geom)"),
        format!("ST_Intersects({NULL_GEOMETRY}, geom)"),
    ] {
        assert_eq!(ids(&dataset, &filter).await, Vec::<i64>::new(), "{filter}");
    }
    dataset
        .delete(&format!("NOT ST_Contains({NULL_GEOMETRY}, geom)"))
        .await
        .unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), POINTS.len());
}

/// Wrong-row mutation regression: deleting with a constant-first predicate must
/// remove exactly the rows the predicate selects. Under the argument reversal,
/// `NOT ST_Contains(<region>, geom)` deleted the points inside the region too.
#[rstest]
#[case::not_contains("NOT ST_Contains({c}, geom)", vec![0, 1, 6])]
#[case::contains("ST_Contains({c}, geom)", vec![2, 3, 4, 5, 6])]
#[case::not_covers("NOT ST_Covers({c}, geom)", vec![0, 1, 2, 3, 6])]
#[case::not_within("NOT ST_Within({c}, geom)", vec![6])]
#[case::not_covered_by("NOT ST_CoveredBy({c}, geom)", vec![6])]
#[tokio::test]
async fn test_geo_constant_first_delete(
    #[case] predicate: &str,
    #[case] surviving_ids: Vec<i64>,
    #[values(false, true)] has_index: bool,
) {
    let mut dataset = points_around_region(has_index).await;
    let predicate = predicate.replace("{c}", REGION);
    dataset.delete(&predicate).await.unwrap();
    assert_eq!(ids(&dataset, "true").await, surviving_ids, "{predicate}");
}

/// Updating with a constant-first predicate must change exactly the rows the
/// predicate selects.
#[rstest]
#[case::contains("ST_Contains({c}, geom)", vec![0, 1])]
#[case::not_contains("NOT ST_Contains({c}, geom)", vec![2, 3, 4, 5])]
#[case::covers("ST_Covers({c}, geom)", vec![0, 1, 2, 3])]
#[case::not_within("NOT ST_Within({c}, geom)", vec![0, 1, 2, 3, 4, 5])]
#[tokio::test]
async fn test_geo_constant_first_update(
    #[case] predicate: &str,
    #[case] updated_ids: Vec<i64>,
    #[values(false, true)] has_index: bool,
) {
    let dataset = points_around_region(has_index).await;
    let predicate = predicate.replace("{c}", REGION);
    let result = UpdateBuilder::new(Arc::new(dataset))
        .update_where(&predicate)
        .unwrap()
        .set("label", "'updated'")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();
    assert_eq!(result.rows_updated, updated_ids.len() as u64, "{predicate}");
    assert_eq!(
        ids(&result.new_dataset, "label = 'updated'").await,
        updated_ids,
        "{predicate}"
    );
}

/// `merge_insert` conditions are spatial filters too: replacing the rows inside a
/// region must delete only unmatched rows inside it.
#[rstest]
#[tokio::test]
async fn test_geo_constant_first_merge_insert_delete_if(#[values(false, true)] has_index: bool) {
    let dataset = Arc::new(points_around_region(has_index).await);
    // The source re-inserts row 0 only, so row 1 is the one unmatched row inside the region.
    let source = dataset
        .scan()
        .filter("id = 0")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let delete_if =
        WhenNotMatchedBySource::delete_if(&dataset, &format!("ST_Contains({REGION}, geom)"))
            .unwrap();
    let (merged, stats) = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .when_not_matched_by_source(delete_if)
        .try_build()
        .unwrap()
        .execute_reader(RecordBatchIterator::new(
            vec![Ok(source.clone())],
            source.schema(),
        ))
        .await
        .unwrap();
    assert_eq!(stats.num_deleted_rows, 1);
    assert_eq!(ids(&merged, "true").await, vec![0, 2, 3, 4, 5, 6]);
}
