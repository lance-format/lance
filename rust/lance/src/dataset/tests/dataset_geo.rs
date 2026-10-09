// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::vec;

use crate::Dataset;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::tests::dataset_transactions::execute_sql;
use crate::dataset::{UpdateBuilder, WriteParams};

use crate::index::DatasetIndexExt;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchIterator;
use arrow_array::UInt32Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::types::UInt32Type;
use arrow_schema::{DataType, Field, Schema};
use datafusion::common::{assert_contains, assert_not_contains};
use geo_types::{Rect, coord, line_string};
use geoarrow_array::{
    GeoArrowArray,
    builder::{LineStringBuilder, PointBuilder, PolygonBuilder},
};
use geoarrow_schema::{Dimension, LineStringType, PointType, PolygonType};
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::ScalarIndexParams;

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

#[rstest::rstest]
#[case::row_addresses(false, false)]
#[case::stable_row_ids(true, false)]
#[case::fragment_reuse(false, true)]
#[tokio::test]
async fn test_rtree_optimize_drops_rewritten_rows(
    #[case] enable_stable_row_ids: bool,
    #[case] compact: bool,
) {
    let point_type = PointType::new(Dimension::XY, Default::default());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        point_type.clone().to_field("point", true),
        point_type.clone().to_field("replacement", true),
    ]));
    let mut points = PointBuilder::new(point_type.clone());
    let mut replacements = PointBuilder::new(point_type);
    for id in 0..8 {
        if id % 2 == 0 {
            points.push_null();
            replacements.push_point(Some(&geo_types::point!(x: 10.0, y: 10.0)));
        } else {
            points.push_point(Some(&geo_types::point!(x: 1.0, y: 1.0)));
            replacements.push_null();
        }
    }
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..8)),
            points.finish().to_array_ref(),
            replacements.finish().to_array_ref(),
        ],
    )
    .unwrap();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new([Ok(batch)], schema),
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 4,
            enable_stable_row_ids,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    dataset
        .create_index(
            &["point"],
            IndexType::RTree,
            Some("rtree_index".to_string()),
            &ScalarIndexParams::new("RTree".to_string()),
            true,
        )
        .await
        .unwrap();
    if compact {
        compact_files(
            &mut dataset,
            CompactionOptions {
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 1);
    }
    // Keep other rows in the source fragment live while replacing a NULL with
    // a point and a point with a NULL. Stable IDs are reused by the replacements.
    dataset = UpdateBuilder::new(Arc::new(dataset))
        .update_where("id < 2")
        .unwrap()
        .set("point", "replacement")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap()
        .new_dataset
        .as_ref()
        .clone();
    dataset
        .optimize_indices(&OptimizeOptions::merge(1))
        .await
        .unwrap();
    let indices = dataset.load_indices_by_name("rtree_index").await.unwrap();
    assert_eq!(indices.len(), 1);

    for (predicate, expected) in [
        ("point IS NULL", vec![1, 2, 4, 6]),
        (
            "ST_Intersects(point, ST_GeomFromText('POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))'))",
            vec![3, 5, 7],
        ),
        (
            "ST_Intersects(point, ST_GeomFromText('POLYGON ((9 9, 11 9, 11 11, 9 11, 9 9))'))",
            vec![0],
        ),
    ] {
        let mut indexed = dataset.scan();
        indexed.project(&["id"]).unwrap().filter(predicate).unwrap();
        assert_contains!(
            indexed.explain_plan(false).await.unwrap(),
            "ScalarIndexQuery"
        );
        let indexed = indexed.try_into_batch().await.unwrap();
        let scanned = dataset
            .scan()
            .project(&["id"])
            .unwrap()
            .filter(predicate)
            .unwrap()
            .use_scalar_index(false)
            .try_into_batch()
            .await
            .unwrap();
        let mut indexed_ids = indexed["id"].as_primitive::<UInt32Type>().values().to_vec();
        let mut scanned_ids = scanned["id"].as_primitive::<UInt32Type>().values().to_vec();
        indexed_ids.sort_unstable();
        scanned_ids.sort_unstable();
        assert_eq!(indexed_ids, expected, "indexed predicate: {predicate}");
        assert_eq!(indexed_ids, scanned_ids, "predicate: {predicate}");
    }
}
