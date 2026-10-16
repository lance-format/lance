// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::vec;

use crate::Dataset;
use crate::dataset::tests::dataset_transactions::execute_sql;
use crate::dataset::{UpdateBuilder, WriteParams};

use crate::index::DatasetIndexExt;
use arrow_array::RecordBatchIterator;
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Int64Array, RecordBatch};
use datafusion::common::{assert_contains, assert_not_contains};
use geo_types::{Rect, coord, line_string};
use geoarrow_array::{
    GeoArrowArray,
    builder::{LineStringBuilder, PointBuilder, PolygonBuilder},
};
use geoarrow_schema::{Dimension, LineStringType, PointType, PolygonType};
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
use rstest::rstest;

#[rstest]
#[case::contains("ST_Contains", 1)]
#[case::within("ST_Within", 2)]
#[case::covers("ST_Covers", 1)]
#[case::covered_by("ST_CoveredBy", 2)]
#[tokio::test]
async fn test_geo_constant_first(
    #[case] predicate: &str,
    #[case] expected_id: i64,
    #[values(false, true)] has_index: bool,
) {
    let polygon_type = PolygonType::new(Dimension::XY, Default::default());
    let mut builder = PolygonBuilder::new(polygon_type.clone());
    for (min, max) in [(1.0, 2.0), (-1.0, 5.0), (6.0, 7.0)] {
        builder
            .push_rect(Some(&Rect::new(
                coord! { x: min, y: min },
                coord! { x: max, y: max },
            )))
            .unwrap();
    }
    builder.push_rect(None::<&Rect>).unwrap();
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("id", arrow_schema::DataType::Int64, false),
        polygon_type.to_field("geom", true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            builder.finish().to_array_ref(),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    let mut dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
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
    let filter =
        format!("{predicate}(ST_GeomFromText('POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))'), geom)");
    let mut scan = dataset.scan();
    scan.project(&["id"]).unwrap().filter(&filter).unwrap();
    let plan = scan.explain_plan(false).await.unwrap();
    assert_eq!(plan.contains("ScalarIndexQuery"), has_index, "{plan}");
    let result = scan.try_into_batch().await.unwrap();
    assert_eq!(result["id"].as_ref(), &Int64Array::from(vec![expected_id]));

    let sql = format!("SELECT id FROM dataset WHERE {filter}");
    let batches = dataset
        .sql(&sql)
        .build()
        .await
        .unwrap()
        .into_batch_records()
        .await
        .unwrap();
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        1
    );
    assert_eq!(
        batches[0]["id"].as_ref(),
        &Int64Array::from(vec![expected_id])
    );

    let null_filter = "ST_Contains(ST_GeomFromText(CAST(NULL AS VARCHAR)), geom)";
    assert_eq!(
        dataset
            .scan()
            .filter(null_filter)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap()
            .num_rows(),
        0
    );

    let updated = UpdateBuilder::new(Arc::new(dataset))
        .update_where(&filter)
        .unwrap()
        .set("id", "id + 10")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();
    assert_eq!(updated.rows_updated, 1);
    let mut dataset = updated.new_dataset.as_ref().clone();
    dataset.delete(&format!("NOT ({filter})")).await.unwrap();
    let result = dataset
        .scan()
        .project(&["id"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let mut ids = result["id"]
        .as_primitive::<arrow_array::types::Int64Type>()
        .values()
        .to_vec();
    ids.sort_unstable();
    // SQL NOT NULL is NULL, so the NULL geometry survives deletion too.
    assert_eq!(ids, vec![4, expected_id + 10]);
}

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
