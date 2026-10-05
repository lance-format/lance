# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("geoarrow.rust.core")
from geoarrow.rust.core import (
    linestring,
    linestrings,
    point,
    points,
    polygon,
    polygons,
)


def _query_point_ids(dataset: lance.LanceDataset, wkt: str) -> list[int]:
    sql = f"""
          SELECT id, point
          FROM dataset
          WHERE St_Intersects(point, ST_GeomFromText('{wkt}'))
          """
    return [
        value
        for batch in dataset.sql(sql).build().to_batch_records()
        for value in batch.column("id").to_pylist()
    ]


def test_geo_types(tmp_path: Path):
    uri = str(tmp_path / "test_geo_types.lance")
    # Points
    points_2d = points([np.random.rand(3), np.random.rand(3)])

    # LineStrings
    line_offsets = np.array([0, 2, 6, 10], dtype=np.int32)
    linestrings_2d = linestrings([np.random.rand(10), np.random.rand(10)], line_offsets)

    # Polygons
    ring_offsets = np.array([0, 3, 7, 12], dtype=np.int32)
    geom_offsets = np.array([0, 1, 2, 3], dtype=np.int32)
    polygons_2d = polygons(
        [np.random.rand(12), np.random.rand(12)],
        ring_offsets=ring_offsets,
        geom_offsets=geom_offsets,
    )

    schema = pa.schema(
        [
            pa.field(point("xy")).with_name("geometry_points"),
            pa.field(linestring("xy")).with_name("geometry_lines"),
            pa.field(polygon("xy")).with_name("geometry_polygons_2d"),
        ]
    )
    table = pa.Table.from_arrays(
        [points_2d, linestrings_2d, polygons_2d], schema=schema
    )
    lance.write_dataset(table, uri)
    ds = lance.dataset(uri)
    assert ds.schema.field(0) == table.schema.field(0)
    assert ds.schema.field(1) == table.schema.field(1)
    assert ds.schema.field(2) == table.schema.field(2)

    read_table = ds.to_table()
    assert read_table.schema.field(0) == table.schema.field(0)
    assert read_table.schema.field(1) == table.schema.field(1)
    assert read_table.schema.field(2) == table.schema.field(2)

    assert (
        read_table.schema.field(0).metadata[b"ARROW:extension:name"]
        == b"geoarrow.point"
    )
    assert (
        read_table.schema.field(1).metadata[b"ARROW:extension:name"]
        == b"geoarrow.linestring"
    )
    assert (
        read_table.schema.field(2).metadata[b"ARROW:extension:name"]
        == b"geoarrow.polygon"
    )

    assert read_table.num_rows == 3


def test_geo_sql(tmp_path: Path):
    # Points
    points_2d = points([np.array([1.0]), np.array([2.0])])

    # LineStrings
    line_offsets = np.array([0, 2], dtype=np.int32)
    linestrings_2d = linestrings(
        [np.array([3.0, 4.0]), np.array([5.0, 0.0])], line_offsets
    )

    schema = pa.schema(
        [
            pa.field(point("xy")).with_name("point"),
            pa.field(linestring("xy")).with_name("linestring"),
        ]
    )
    table = pa.Table.from_arrays([points_2d, linestrings_2d], schema=schema)
    ds = lance.write_dataset(table, str(tmp_path / "test_geo_udf_distance.lance"))

    batches = (
        ds.sql("SELECT St_Distance(point, linestring) as dist FROM dataset")
        .build()
        .to_batch_records()
    )
    assert len(batches) == 1
    result = batches[0].to_pydict()
    assert result["dist"]
    assert np.allclose(
        np.array(result["dist"]), np.array([2.5495097567963922]), atol=1e-8
    )


def test_rtree_index(tmp_path: Path):
    # LineStrings
    num_lines = 10000
    line_offsets = np.arange(num_lines + 1, dtype=np.int32) * 2
    linestrings_2d = linestrings(
        [np.random.randn(num_lines * 2) * 100, np.random.randn(num_lines * 2) * 100],
        line_offsets,
    )
    assert len(linestrings_2d) == num_lines

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(linestring("xy")).with_name("linestring"),
        ]
    )
    table = pa.Table.from_arrays(
        [np.arange(num_lines, dtype=np.int64), linestrings_2d], schema=schema
    )
    ds = lance.write_dataset(table, str(tmp_path / "test_rtree_index.lance"))

    def query(ds: lance.LanceDataset, has_index=False):
        sql = """
              SELECT `id`, linestring
              FROM dataset
              WHERE
              St_Intersects(linestring, ST_GeomFromText('LINESTRING ( 2 0, 0 2 )'))
              """

        batches = ds.sql("EXPLAIN ANALYZE " + sql).build().to_batch_records()
        explain = pa.Table.from_batches(batches).to_pandas().to_string()

        if has_index:
            assert "ScalarIndexQuery" in explain
        else:
            assert "ScalarIndexQuery" not in explain

        batches = ds.sql(sql).build().to_batch_records()
        return pa.Table.from_batches(batches)

    table_without_index = query(ds)

    ds.create_scalar_index("linestring", "RTREE")

    table_with_index = query(ds, has_index=True)

    assert table_with_index == table_without_index


def test_rtree_segment_merge_and_commit(tmp_path: Path):
    num_points = 120
    points_2d = points(
        [
            np.arange(num_points, dtype=np.float64),
            np.arange(num_points, dtype=np.float64),
        ]
    )
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(point("xy")).with_name("point"),
        ]
    )
    table = pa.Table.from_arrays(
        [np.arange(num_points, dtype=np.int64), points_2d], schema=schema
    )
    ds = lance.write_dataset(
        table,
        str(tmp_path / "segmented_rtree.lance"),
        max_rows_per_file=40,
    )
    fragments = ds.get_fragments()
    assert len(fragments) == 3
    segments = [
        ds.create_index_uncommitted(
            column="point",
            index_type="RTREE",
            name="point_rtree",
            fragment_ids=[fragment.fragment_id],
        )
        for fragment in fragments
    ]

    merged = ds.merge_existing_index_segments(segments)
    assert set(merged.fragment_ids) == {fragment.fragment_id for fragment in fragments}
    ds = ds.commit_existing_index_segments("point_rtree", "point", [merged])

    sql = """
          SELECT id, point
          FROM dataset
          WHERE St_Intersects(point, ST_GeomFromText('LINESTRING (10 10, 110 110)'))
          """
    indexed = pa.Table.from_batches(ds.sql(sql).build().to_batch_records())
    assert indexed["id"].to_pylist() == list(range(10, 111))
    explain = (
        pa.Table.from_batches(
            ds.sql("EXPLAIN ANALYZE " + sql).build().to_batch_records()
        )
        .to_pandas()
        .to_string()
    )
    assert "ScalarIndexQuery" in explain


def test_staged_rtree_after_rewrite_columns(tmp_path: Path):
    uri = str(tmp_path / "stale_rtree.lance")
    point_type = point("xy")
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(point_type).with_name("point"),
        ]
    )
    dataset = lance.write_dataset(
        pa.Table.from_arrays(
            [
                pa.array([0, 1], type=pa.int64()),
                points([np.array([0.0, 1.0]), np.array([0.0, 1.0])]),
            ],
            schema=schema,
        ),
        uri,
    )
    segment = dataset.create_index_uncommitted(
        column="point",
        index_type="RTREE",
        name="point_rtree",
        fragment_ids=[0],
    )

    update_schema = pa.schema(
        [
            pa.field("_rowid", pa.uint64()),
            pa.field(point_type).with_name("point"),
        ]
    )
    update = pa.Table.from_arrays(
        [
            pa.array([0], type=pa.uint64()),
            points([np.array([10.0]), np.array([10.0])]),
        ],
        schema=update_schema,
    )
    fragment, fields = dataset.get_fragment(0).update_columns(update)
    updated = lance.LanceDataset.commit(
        uri,
        lance.LanceOperation.Update(
            updated_fragments=[fragment],
            fields_modified=fields,
        ),
        read_version=dataset.version,
    )

    assert _query_point_ids(updated, "POINT (10 10)") == [0]
    committed = dataset.commit_existing_index_segments(
        "point_rtree",
        "point",
        [segment],
    )
    assert committed.describe_indices()[0].segments[0].fragment_ids == set()
    assert _query_point_ids(committed, "POINT (10 10)") == [0]


def test_rtree_rejects_distributed_uuid_reuse(tmp_path: Path):
    uri = str(tmp_path / "uuid_reuse.lance")
    num_points = 120
    point_type = point("xy")
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(point_type).with_name("point"),
        ]
    )
    dataset = lance.write_dataset(
        pa.Table.from_arrays(
            [
                pa.array(range(num_points), type=pa.int64()),
                points(
                    [
                        np.arange(num_points, dtype=np.float64),
                        np.arange(num_points, dtype=np.float64),
                    ]
                ),
            ],
            schema=schema,
        ),
        uri,
        max_rows_per_file=40,
    )
    dataset.create_scalar_index("point", "RTREE")
    index_uuid = dataset.describe_indices()[0].segments[0].uuid

    with pytest.raises(
        ValueError,
        match="index_uuid is no longer accepted for RTree distributed index builds",
    ):
        dataset.create_index_uncommitted(
            column="point",
            index_type="RTREE",
            name="point_rtree_reuse",
            fragment_ids=[0],
            index_uuid=index_uuid,
        )

    assert _query_point_ids(
        lance.dataset(uri),
        "LINESTRING (100 100, 110 110)",
    ) == list(range(100, 111))


def test_rtree_merge_all_deleted_stable_row_ids(tmp_path: Path):
    uri = str(tmp_path / "all_deleted.lance")
    point_type = point("xy")
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(point_type).with_name("point"),
        ]
    )
    dataset = lance.write_dataset(
        pa.Table.from_arrays(
            [
                pa.array([0, 1], type=pa.int64()),
                points([np.array([0.0, 1.0]), np.array([0.0, 1.0])]),
            ],
            schema=schema,
        ),
        uri,
        enable_stable_row_ids=True,
    )
    segment = dataset.create_index_uncommitted(
        column="point",
        index_type="RTREE",
        name="point_rtree",
        fragment_ids=[0],
    )

    dataset.delete("true")
    merged = dataset.merge_existing_index_segments([segment])
    assert merged.fragment_ids == set()
    committed = dataset.commit_existing_index_segments(
        "point_rtree",
        "point",
        [merged],
    )
    assert _query_point_ids(committed, "POINT (0 0)") == []


def test_rtree_merge_preserves_newer_fragment_coverage(tmp_path: Path):
    uri = str(tmp_path / "mixed_versions.lance")
    point_type = point("xy")
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(point_type).with_name("point"),
        ]
    )

    def batch(start: int, stop: int) -> pa.Table:
        values = np.arange(start, stop, dtype=np.float64)
        return pa.Table.from_arrays(
            [
                pa.array(range(start, stop), type=pa.int64()),
                points([values, values]),
            ],
            schema=schema,
        )

    dataset = lance.write_dataset(batch(0, 40), uri)
    first = dataset.create_index_uncommitted(
        column="point",
        index_type="RTREE",
        name="point_rtree",
        fragment_ids=[0],
    )
    dataset = lance.write_dataset(batch(40, 80), uri, mode="append")
    second = dataset.create_index_uncommitted(
        column="point",
        index_type="RTREE",
        name="point_rtree",
        fragment_ids=[1],
    )

    merged = dataset.merge_existing_index_segments([first, second])
    assert merged.fragment_ids == {0, 1}
    assert merged.dataset_version == dataset.version

    update_schema = pa.schema(
        [
            pa.field("_rowid", pa.uint64()),
            pa.field(point_type).with_name("point"),
        ]
    )
    update = pa.Table.from_arrays(
        [
            pa.array([1 << 32], type=pa.uint64()),
            points([np.array([100.0]), np.array([100.0])]),
        ],
        schema=update_schema,
    )
    fragment, fields = dataset.get_fragment(1).update_columns(update)
    dataset = lance.LanceDataset.commit(
        uri,
        lance.LanceOperation.Update(
            updated_fragments=[fragment],
            fields_modified=fields,
        ),
        read_version=dataset.version,
    )

    committed = dataset.commit_existing_index_segments(
        "point_rtree",
        "point",
        [merged],
    )
    assert committed.describe_indices()[0].segments[0].fragment_ids == {0}
    assert _query_point_ids(
        committed,
        "POINT (100 100)",
    ) == [40]
