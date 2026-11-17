# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for _coerce_query_vector to ensure invalid input types
raise TypeError instead of falling into the numpy conversion branch."""

import numpy as np
import pyarrow as pa
import pytest
from lance._dataset.scanner import _coerce_query_vector


class TestCoerceQueryVectorInvalidTypes:
    """Non-vector inputs should raise TypeError, not numpy ValueError."""

    def test_string_raises_typeerror(self):
        with pytest.raises(TypeError, match="query vector must be list-like"):
            _coerce_query_vector("not a vector")

    def test_integer_raises_typeerror(self):
        with pytest.raises(
            TypeError, match="Query vectors should be an array of floats"
        ):
            _coerce_query_vector(42)

    def test_object_raises_typeerror(self):
        """A random object that is not array-like should raise TypeError."""

        class NotAVector:
            pass

        with pytest.raises(
            TypeError, match="Query vectors should be an array of floats"
        ):
            _coerce_query_vector(NotAVector())

    def test_none_raises_typeerror(self):
        with pytest.raises(
            TypeError, match="Query vectors should be an array of floats"
        ):
            _coerce_query_vector(None)


@pytest.mark.parametrize(
    "value",
    [
        [1.0, 2.0, 3.0],
        (1.0, 2.0, 3.0),
        np.array([1.0, 2.0, 3.0]),
        pa.array([1.0, 2.0, 3.0]),
        pa.array([1, 2, 3]),
        pa.chunked_array([[1.0, 2.0, 3.0]]),
    ],
    ids=["list", "tuple", "numpy", "pa_float", "pa_int", "pa_chunked"],
)
def test_valid_vector_inputs(value):
    result, dim = _coerce_query_vector(value)
    assert isinstance(result, pa.FloatingPointArray)
    assert dim == 3
