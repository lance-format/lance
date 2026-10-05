# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
#
# Pydantic-to-Arrow conversion, ported from
# https://github.com/lancedb/lancedb/blob/f8dc2f78ee3219084d457a647e230b23e2f391b0/python/python/lancedb/pydantic.py
#
# Embedding-function-specific code (LanceModel, parse_embedding_functions,
# EmbeddingFunctionRegistry) is intentionally excluded -- lance has no concept
# of embedding functions, that stays in lancedb. Version detection uses
# hasattr checks against the field/model objects instead of branching on
# pydantic's version number, since pydantic is an optional dependency here.

"""Pydantic (v1 / v2) to Arrow schema conversion"""

from __future__ import annotations

import inspect
import sys
import types
from abc import ABC, abstractmethod
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Type, Union, _GenericAlias

import pyarrow as pa
import pydantic

from .dependencies import numpy as np

_PYDANTIC_V2 = hasattr(pydantic, "GetCoreSchemaHandler")

try:
    from pydantic_core import CoreSchema, core_schema
except ImportError:
    if _PYDANTIC_V2:
        raise


class FixedSizeListMixin(ABC):
    @staticmethod
    @abstractmethod
    def dim() -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def value_arrow_type() -> pa.DataType:
        raise NotImplementedError


def Vector(
    dim: int, value_type: pa.DataType = pa.float32(), nullable: bool = True
) -> Type[FixedSizeListMixin]:
    """Pydantic type for a fixed-size-list embedding vector column.

    Examples
    --------
    >>> import pydantic
    >>> from lance.pydantic import Vector
    ...
    >>> class MyModel(pydantic.BaseModel):
    ...     id: int
    ...     embedding: Vector(768)
    """

    class FixedSizeList(list, FixedSizeListMixin):
        def __repr__(self):
            return f"FixedSizeList(dim={dim})"

        @staticmethod
        def nullable() -> bool:
            return nullable

        @staticmethod
        def dim() -> int:
            return dim

        @staticmethod
        def value_arrow_type() -> pa.DataType:
            return value_type

        @classmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: Any, _handler: pydantic.GetCoreSchemaHandler
        ) -> CoreSchema:
            return core_schema.no_info_after_validator_function(
                cls,
                core_schema.list_schema(
                    min_length=dim,
                    max_length=dim,
                    items_schema=core_schema.float_schema(),
                ),
            )

        @classmethod
        def __get_validators__(cls) -> Generator[Callable, None, None]:
            yield cls.validate

        # For pydantic v1
        @classmethod
        def validate(cls, v):
            if not isinstance(v, (list, range, np.ndarray)) or len(v) != dim:
                raise TypeError("A list of numbers or numpy.ndarray is needed")
            return cls(v)

        if not _PYDANTIC_V2:

            @classmethod
            def __modify_schema__(cls, field_schema: Dict[str, Any]):
                field_schema["items"] = {"type": "number"}
                field_schema["maxItems"] = dim
                field_schema["minItems"] = dim

    return FixedSizeList


def MultiVector(
    dim: int, value_type: pa.DataType = pa.float32(), nullable: bool = True
) -> Type[FixedSizeListMixin]:
    """Pydantic type for a list of fixed-size-list embedding vectors.

    Useful for models that produce multiple embeddings per input (e.g.
    ColPali-style multi-vector embeddings).

    Examples
    --------
    >>> import pydantic
    >>> from lance.pydantic import MultiVector
    ...
    >>> class MyModel(pydantic.BaseModel):
    ...     id: int
    ...     embeddings: MultiVector(128)
    """

    class MultiVectorList(list, FixedSizeListMixin):
        def __repr__(self):
            return f"MultiVector(dim={dim})"

        @staticmethod
        def nullable() -> bool:
            return nullable

        @staticmethod
        def dim() -> int:
            return dim

        @staticmethod
        def value_arrow_type() -> pa.DataType:
            return value_type

        @staticmethod
        def is_multi_vector() -> bool:
            return True

        @classmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: Any, _handler: pydantic.GetCoreSchemaHandler
        ) -> CoreSchema:
            return core_schema.no_info_after_validator_function(
                cls,
                core_schema.list_schema(
                    items_schema=core_schema.list_schema(
                        min_length=dim,
                        max_length=dim,
                        items_schema=core_schema.float_schema(),
                    ),
                ),
            )

        @classmethod
        def __get_validators__(cls) -> Generator[Callable, None, None]:
            yield cls.validate

        # For pydantic v1
        @classmethod
        def validate(cls, v):
            if not isinstance(v, (list, range)):
                raise TypeError("A list of vectors is needed")
            for vec in v:
                if not isinstance(vec, (list, range, np.ndarray)) or len(vec) != dim:
                    raise TypeError(f"Each vector must be a list of {dim} numbers")
            return cls(v)

        if not _PYDANTIC_V2:

            @classmethod
            def __modify_schema__(cls, field_schema: Dict[str, Any]):
                field_schema["items"] = {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": dim,
                    "maxItems": dim,
                }

    return MultiVectorList


def _field_annotation(field: Any) -> Any:
    """Get the type annotation off a pydantic v1 ModelField or v2 FieldInfo."""
    if hasattr(field, "annotation"):
        # Pydantic v2 FieldInfo
        return field.annotation
    # Pydantic v1 ModelField -- Optional-ness is tracked via `allow_none`,
    # not folded into `outer_type_`.
    return field.outer_type_


def get_extras(field: Any, key: str) -> Any:
    """Get extra metadata (from `json_schema_extra`) off a pydantic field."""
    if hasattr(field, "json_schema_extra"):
        # Pydantic v2 FieldInfo
        return (field.json_schema_extra or {}).get(key)
    # Pydantic v1 ModelField
    return (field.field_info.extra or {}).get("json_schema_extra", {}).get(key)


def _py_type_to_arrow_type(py_type: Type[Any], field: Any) -> pa.DataType:
    """Convert a field with native Python type to Arrow data type.

    Raises
    ------
    TypeError
        If the type is not supported.
    """
    if py_type is int:
        return pa.int64()
    elif py_type is float:
        return pa.float64()
    elif py_type is str:
        return pa.utf8()
    elif py_type is bool:
        return pa.bool_()
    elif py_type is bytes:
        return pa.binary()
    elif py_type is date:
        return pa.date32()
    elif py_type is datetime:
        tz = get_extras(field, "tz")
        return pa.timestamp("us", tz=tz)
    elif getattr(py_type, "__origin__", None) in (list, tuple):
        # A bare, unparameterised ``typing.List`` / ``typing.Tuple`` matches
        # this branch (its ``__origin__`` is ``list`` / ``tuple``) but has no
        # ``__args__``, so we cannot infer the element type. Raise a clear
        # ``TypeError`` instead of crashing with an opaque ``AttributeError``.
        args = getattr(py_type, "__args__", None)
        if not args:
            raise TypeError(
                "Converting Pydantic type to Arrow Type: unsupported type "
                f"{py_type}. Specify the element type, e.g. List[int] instead "
                "of a bare List."
            )
        child = args[0]
        return _pydantic_list_child_to_arrow(child, field)
    raise TypeError(
        f"Converting Pydantic type to Arrow Type: unsupported type {py_type}."
    )


def _pydantic_model_to_fields(model: Type[pydantic.BaseModel]) -> List[pa.Field]:
    if hasattr(model, "model_fields"):
        # Pydantic v2
        return [
            _pydantic_to_field(name, field)
            for name, field in model.model_fields.items()
        ]
    # Pydantic v1
    return [_pydantic_to_field(name, field) for name, field in model.__fields__.items()]


def _pydantic_type_to_arrow_type(tp: Any, field: Any) -> pa.DataType:
    def _safe_issubclass(candidate: Any, base: type) -> bool:
        try:
            return issubclass(candidate, base)
        except TypeError:
            return False

    if inspect.isclass(tp):
        if _safe_issubclass(tp, pydantic.BaseModel):
            # Struct
            fields = _pydantic_model_to_fields(tp)
            return pa.struct(fields)
        if _safe_issubclass(tp, FixedSizeListMixin):
            if getattr(tp, "is_multi_vector", lambda: False)():
                return pa.list_(pa.list_(tp.value_arrow_type(), tp.dim()))
            # For regular Vector
            return pa.list_(tp.value_arrow_type(), tp.dim())
        if _safe_issubclass(tp, Enum):
            # Map Enum to the Arrow type of its value.
            # For string-valued enums, use dictionary encoding for efficiency.
            # For integer enums, use the native type.
            # Fall back to utf8 for mixed-type or empty enums.
            value_types = {type(m.value) for m in tp}
            if len(value_types) == 1:
                value_type = value_types.pop()
                if value_type is str:
                    # Use dictionary encoding for string enums
                    return pa.dictionary(pa.int32(), pa.utf8())
                return _py_type_to_arrow_type(value_type, field)
            return pa.utf8()
    return _py_type_to_arrow_type(tp, field)


def _pydantic_list_child_to_arrow(child: Any, field: Any) -> pa.DataType:
    unwrapped = _unwrap_optional_annotation(child)
    if unwrapped is not None:
        return pa.list_(
            pa.field("item", _pydantic_type_to_arrow_type(unwrapped, field), True)
        )
    return pa.list_(_pydantic_type_to_arrow_type(child, field))


def _unwrap_optional_annotation(annotation: Any) -> Any | None:
    if isinstance(annotation, (_GenericAlias, types.GenericAlias)):
        origin = annotation.__origin__
        args = annotation.__args__
        if origin == Union:
            non_none = [arg for arg in args if arg is not type(None)]
            if len(non_none) == 1 and len(non_none) != len(args):
                return non_none[0]
    elif sys.version_info >= (3, 10) and isinstance(annotation, types.UnionType):
        args = annotation.__args__
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1 and len(non_none) != len(args):
            return non_none[0]
    return None


def _pydantic_to_arrow_type(field: Any) -> pa.DataType:
    """Convert a pydantic field (v1 ModelField or v2 FieldInfo) to Arrow DataType"""
    annotation = _field_annotation(field)
    unwrapped = _unwrap_optional_annotation(annotation)
    if unwrapped is not None:
        return _pydantic_type_to_arrow_type(unwrapped, field)
    if isinstance(annotation, (_GenericAlias, types.GenericAlias)):
        origin = annotation.__origin__
        args = annotation.__args__

        if origin is list:
            child = args[0]
            return _pydantic_list_child_to_arrow(child, field)
    return _pydantic_type_to_arrow_type(annotation, field)


def is_nullable(field: Any) -> bool:
    """Check if a pydantic field (v1 ModelField or v2 FieldInfo) is nullable.

    Only a true ``Optional``/``Union[..., None]`` annotation (or a nullable
    ``Vector``/``MultiVector``) makes a field nullable -- a field with a
    plain default value but a non-Optional type is not.
    """
    if not hasattr(field, "annotation"):
        # Pydantic v1 ModelField: Optional-ness is tracked via `allow_none`
        # directly, since `outer_type_` already has Optional stripped. A
        # nullable Vector/MultiVector still overrides this, same as v2.
        v1_type = _field_annotation(field)
        if inspect.isclass(v1_type):
            try:
                if issubclass(v1_type, FixedSizeListMixin):
                    return v1_type.nullable()
            except TypeError:
                pass
        return bool(field.allow_none)

    annotation = field.annotation
    if _unwrap_optional_annotation(annotation) is not None:
        return True
    if isinstance(annotation, (_GenericAlias, types.GenericAlias)):
        origin = annotation.__origin__
        args = annotation.__args__
        if origin == Union:
            if any(typ is type(None) for typ in args):
                return True
    elif sys.version_info >= (3, 10) and isinstance(annotation, types.UnionType):
        args = annotation.__args__
        for typ in args:
            if typ is type(None):
                return True
    elif inspect.isclass(annotation):
        try:
            if issubclass(annotation, FixedSizeListMixin):
                return annotation.nullable()
        except TypeError:
            return False
    return False


def _pydantic_to_field(name: str, field: Any) -> pa.Field:
    """Convert a pydantic field (v1 ModelField or v2 FieldInfo) to a PyArrow Field."""
    dt = _pydantic_to_arrow_type(field)
    return pa.field(name, dt, is_nullable(field))


def pydantic_to_schema(model: Type[pydantic.BaseModel]) -> pa.Schema:
    """Convert a [Pydantic Model][pydantic.BaseModel] to a
       [PyArrow Schema][pyarrow.Schema].

    Supports nested ``BaseModel`` fields (-> struct), ``Enum`` fields
    (string-valued enums are dictionary-encoded), timezone-aware
    ``datetime`` fields (via ``Field(json_schema_extra={"tz": ...})``), and
    the ``Vector``/``MultiVector`` fixed-size-list types, in addition to
    plain scalar/``Optional``/``List`` types.

    Parameters
    ----------
    model : Type[pydantic.BaseModel]
        The Pydantic BaseModel to convert to Arrow Schema.

    Returns
    -------
    pyarrow.Schema
        The Arrow Schema

    Examples
    --------

    >>> from typing import List, Optional
    >>> import pydantic
    >>> from lance.pydantic import pydantic_to_schema, Vector
    >>> class FooModel(pydantic.BaseModel):
    ...     id: int
    ...     s: str
    ...     vec: Vector(1536)  # fixed_size_list<item: float32>[1536]
    ...     li: List[int]
    ...
    >>> schema = pydantic_to_schema(FooModel)
    >>> assert schema == pa.schema([
    ...     pa.field("id", pa.int64(), False),
    ...     pa.field("s", pa.utf8(), False),
    ...     pa.field("vec", pa.list_(pa.float32(), 1536)),
    ...     pa.field("li", pa.list_(pa.int64()), False),
    ... ])
    """
    fields = _pydantic_model_to_fields(model)
    return pa.schema(fields)
