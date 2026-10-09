# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import lance
import pyarrow as pa
import pytest  # pyright: ignore[reportMissingImports]
from lance.schema import LanceSchema

if TYPE_CHECKING:
    from typing import assert_type

    from lance.lance.schema import LanceField

    def _check_field_lookup_types(schema: LanceSchema) -> None:
        """Static-only guard: both lookups return an optional field.

        ``LanceField`` exists only as a stub type -- ``lance.lance`` is a
        compiled extension that re-exports ``LanceSchema`` alone -- so these
        assertions cannot run, but pyright checks them.
        """
        assert_type(schema.field("x"), Optional[LanceField])
        assert_type(schema.field_case_insensitive("x"), Optional[LanceField])


def test_lance_schema(tmp_path: Path):
    # Include nested fields to test the reconstruction of the schema
    data = pa.table(
        {
            "x": range(2),
            "s": [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}],
            "y": [[1.0, 2.0], [3.0, 4.0]],
        }
    )
    dataset = lance.write_dataset(data, tmp_path)

    schema = dataset.lance_schema

    assert repr(schema).startswith("Schema {")

    dumped = pickle.dumps(schema)
    loaded = pickle.loads(dumped)
    assert schema == loaded

    assert schema.to_pyarrow() == data.schema
    assert LanceSchema.from_pyarrow(data.schema) == schema

    fields = schema.fields()
    assert len(fields) == 3
    assert fields[0].name() == "x"
    assert fields[0].id() == 0
    assert fields[1].name() == "s"
    assert fields[1].id() == 1

    s_children = fields[1].children()
    assert len(s_children) == 2
    assert s_children[0].name() == "a"
    assert s_children[0].id() == 2
    assert s_children[1].name() == "b"
    assert s_children[1].id() == 3

    assert fields[2].name() == "y"
    assert fields[2].id() == 4

    l_children = fields[2].children()
    assert len(l_children) == 1
    assert l_children[0].name() == "item"
    assert l_children[0].id() == 5

    # Changing column name does not change the id
    # alter_columns is variadic, but its parameter is annotated
    # Iterable[AlterColumn] rather than AlterColumn, so a single alteration
    # does not type check. Unrelated to this file; suppressed rather than
    # fixed here to keep the change focused.
    dataset.alter_columns(
        {"path": "s.a", "name": "new_name"}  # pyright: ignore[reportArgumentType]
    )
    schema = dataset.lance_schema
    fields = schema.fields()
    s_fields = fields[1].children()
    assert s_fields[0].name() == "new_name"
    assert s_fields[0].id() == 2


def test_lance_schema_from_protos_rejects_missing_parent():
    # name (field 2): child; id (field 3): 7; parent_id (field 4): 42;
    # logical_type (field 5): int32.
    field_proto = b"\x12\x05child\x18\x07\x20\x2a\x2a\x05int32"

    with pytest.raises(
        ValueError,
        match="Field 'child' \\(id=7\\) references parent id 42",
    ):
        LanceSchema._from_protos("{}", field_proto)


def test_lance_schema_field_lookup(tmp_path: Path):
    dataset = lance.write_dataset(
        pa.table({"x": range(2), "s": [{"a": 1}, {"a": 2}]}), tmp_path
    )
    schema = dataset.lance_schema

    field = schema.field("x")
    assert field is not None
    assert field.name() == "x"

    # Dotted paths address nested fields; a miss returns None rather than
    # raising, which is what the Optional return type encodes.
    nested = schema.field("s.a")
    assert nested is not None
    assert nested.name() == "a"
    assert schema.field("does_not_exist") is None


def test_lance_schema_field_case_insensitive(tmp_path: Path):
    dataset = lance.write_dataset(pa.table({"MixedCase": range(2)}), tmp_path)
    schema = dataset.lance_schema

    exact = schema.field_case_insensitive("MixedCase")
    assert exact is not None
    assert exact.name() == "MixedCase"

    # Falls back to a case-insensitive match, preserving the original casing.
    relaxed = schema.field_case_insensitive("mixedcase")
    assert relaxed is not None
    assert relaxed.name() == "MixedCase"

    assert schema.field_case_insensitive("does_not_exist") is None
