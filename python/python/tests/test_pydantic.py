# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from enum import Enum
from typing import List, Optional

import pyarrow as pa
import pytest

BaseModel = pytest.importorskip("pydantic").BaseModel
Field = pytest.importorskip("pydantic").Field

from lance.pydantic import (  # noqa: E402
    MultiVector,
    Vector,
    is_nullable,
    pydantic_to_schema,
)


def test_scalar_and_optional_and_list_unchanged():
    class Simple(BaseModel):
        name: str
        score: float
        tag: Optional[str] = None
        values: List[int]

    schema = pydantic_to_schema(Simple)
    assert schema == pa.schema(
        [
            pa.field("name", pa.utf8(), False),
            pa.field("score", pa.float64(), False),
            pa.field("tag", pa.utf8(), True),
            pa.field("values", pa.list_(pa.int64()), False),
        ]
    )


def test_nested_model_becomes_struct():
    class Address(BaseModel):
        city: str
        zip_code: str

    class Person(BaseModel):
        name: str
        address: Address

    schema = pydantic_to_schema(Person)
    address_type = pa.struct(
        [
            pa.field("city", pa.utf8(), False),
            pa.field("zip_code", pa.utf8(), False),
        ]
    )
    assert schema == pa.schema(
        [
            pa.field("name", pa.utf8(), False),
            pa.field("address", address_type, False),
        ]
    )


def test_string_enum_is_dictionary_encoded():
    class Color(str, Enum):
        RED = "red"
        GREEN = "green"

    class Item(BaseModel):
        color: Color

    schema = pydantic_to_schema(Item)
    assert schema.field("color").type == pa.dictionary(pa.int32(), pa.utf8())


def test_int_enum_uses_native_type():
    class Priority(int, Enum):
        LOW = 0
        HIGH = 1

    class Task(BaseModel):
        priority: Priority

    schema = pydantic_to_schema(Task)
    assert schema.field("priority").type == pa.int64()


def test_vector_field():
    class Doc(BaseModel):
        id: int
        embedding: Vector(8)

    schema = pydantic_to_schema(Doc)
    assert schema.field("embedding").type == pa.list_(pa.float32(), 8)
    assert schema.field("embedding").nullable is True


def test_vector_field_not_nullable():
    class Doc(BaseModel):
        id: int
        embedding: Vector(8, nullable=False)

    schema = pydantic_to_schema(Doc)
    assert schema.field("embedding").nullable is False


def test_multi_vector_field():
    class Doc(BaseModel):
        id: int
        embeddings: MultiVector(4)

    schema = pydantic_to_schema(Doc)
    assert schema.field("embeddings").type == pa.list_(pa.list_(pa.float32(), 4))


def test_tz_aware_datetime_field():
    from datetime import datetime

    class Event(BaseModel):
        occurred_at: datetime = Field(json_schema_extra={"tz": "UTC"})

    schema = pydantic_to_schema(Event)
    assert schema.field("occurred_at").type == pa.timestamp("us", tz="UTC")


def test_naive_datetime_field():
    from datetime import datetime

    class Event(BaseModel):
        occurred_at: datetime

    schema = pydantic_to_schema(Event)
    assert schema.field("occurred_at").type == pa.timestamp("us", tz=None)


def test_defaulted_non_optional_field_is_not_nullable():
    """A plain default value (without Optional) should not make a field
    nullable -- this matches lancedb's behavior, tightened from lance's
    previous default-implies-nullable inference."""

    class Counter(BaseModel):
        count: int = 0

    schema = pydantic_to_schema(Counter)
    assert schema.field("count").nullable is False

    field_info = Counter.model_fields["count"]
    assert is_nullable(field_info) is False


def test_optional_field_is_nullable():
    class Counter(BaseModel):
        count: Optional[int] = None

    schema = pydantic_to_schema(Counter)
    assert schema.field("count").nullable is True


def test_v1_style_field_nullable_vector_overrides_allow_none():
    """Simulates a Pydantic v1 ModelField (no `.annotation` attribute; type
    is read off `.outer_type_`, nullability off `.allow_none`) to verify a
    nullable Vector/MultiVector annotation overrides `allow_none=False`
    under v1, matching the v2 behavior tested in test_vector_field above."""

    class FakeV1Field:
        def __init__(self, outer_type_, allow_none):
            self.outer_type_ = outer_type_
            self.allow_none = allow_none

    nullable_vector = Vector(8, nullable=True)
    assert is_nullable(FakeV1Field(nullable_vector, allow_none=False)) is True

    non_nullable_vector = Vector(8, nullable=False)
    assert is_nullable(FakeV1Field(non_nullable_vector, allow_none=False)) is False

    # Plain (non-Vector) types still fall back to `allow_none`.
    assert is_nullable(FakeV1Field(str, allow_none=True)) is True
    assert is_nullable(FakeV1Field(str, allow_none=False)) is False
