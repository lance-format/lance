# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import lance
import pyarrow as pa
import pytest


@pytest.mark.parametrize("data_storage_version", ["legacy", "2.0"])
def test_nullability(tmp_path, data_storage_version):
    nullable_schema = pa.schema([pa.field("a", pa.string(), nullable=True)])
    non_nullable_schema = pa.schema([pa.field("a", pa.string(), nullable=False)])

    def can_write(data, dataset, schema=None):
        lance.write_dataset(pa.table(data, schema=schema), dataset.uri, mode="append")

    def cannot_write(data, dataset, schema=None):
        with pytest.raises(Exception, match=r"contain(ed|s) null values"):
            can_write(data, dataset, schema)

    nullable_dataset = lance.write_dataset(
        [],
        str(tmp_path / "nullable.lance"),
        schema=nullable_schema,
        data_storage_version=data_storage_version,
    )
    non_nullable_dataset = lance.write_dataset(
        [],
        str(tmp_path / "non_nullable.lance"),
        schema=non_nullable_schema,
        data_storage_version=data_storage_version,
    )

    # Can write nullable data into nullable table
    can_write({"a": ["7", None]}, nullable_dataset, nullable_schema)
    # Can write non-nullable data into nullable table
    can_write({"a": ["7"]}, nullable_dataset, non_nullable_schema)
    # Can write non-nullable data into non-nullable table
    can_write({"a": ["7"]}, non_nullable_dataset, non_nullable_schema)
    # Can write nullable data into non-nullable table if
    # the data does not contain nulls
    can_write({"a": ["7"]}, non_nullable_dataset, nullable_schema)
    # If the data contains nulls, it will raise an error
    cannot_write({"a": [None]}, non_nullable_dataset, nullable_schema)
    # This is true even if the data lies and claims it is non-nullable
    cannot_write({"a": [None]}, non_nullable_dataset, non_nullable_schema)

    # Verify nested nullabilities too
    outer_nullable = pa.schema(
        [
            pa.field(
                "point",
                pa.struct([pa.field("x", pa.string(), nullable=False)]),
                nullable=True,
            )
        ]
    )
    dataset = lance.write_dataset(
        [],
        str(tmp_path / "outer_struct_nullable.lance"),
        schema=outer_nullable,
        data_storage_version=data_storage_version,
    )
    if data_storage_version == "2.0":
        with pytest.raises(OSError, match="does not encode struct validity"):
            can_write({"point": [None]}, dataset, outer_nullable)
    else:
        can_write({"point": [None]}, dataset, outer_nullable)
    cannot_write({"point": [{"x": None}]}, dataset, outer_nullable)

    inner_nullable = pa.schema(
        [
            pa.field(
                "point",
                pa.struct([pa.field("x", pa.string(), nullable=True)]),
                nullable=False,
            )
        ]
    )
    dataset = lance.write_dataset(
        [],
        str(tmp_path / "inner_struct_nullable.lance"),
        schema=inner_nullable,
        data_storage_version=data_storage_version,
    )
    cannot_write({"point": [None]}, dataset, inner_nullable)
    can_write({"point": [{"x": None}]}, dataset, inner_nullable)

    outer_nullable = pa.schema(
        [
            pa.field(
                "list",
                pa.list_(pa.field("item", pa.string(), nullable=False)),
                nullable=True,
            )
        ]
    )
    dataset = lance.write_dataset(
        [],
        str(tmp_path / "outer_list_nullable.lance"),
        schema=outer_nullable,
        data_storage_version=data_storage_version,
    )
    can_write({"list": [None]}, dataset, outer_nullable)
    cannot_write({"list": [[None]]}, dataset, outer_nullable)

    inner_nullable = pa.schema(
        [
            pa.field(
                "list",
                pa.list_(pa.field("item", pa.string(), nullable=True)),
                nullable=False,
            )
        ]
    )
    dataset = lance.write_dataset(
        [],
        str(tmp_path / "inner_list_nullable.lance"),
        schema=inner_nullable,
        data_storage_version=data_storage_version,
    )
    cannot_write({"list": [None]}, dataset, inner_nullable)
    can_write({"list": [[None]]}, dataset, inner_nullable)
