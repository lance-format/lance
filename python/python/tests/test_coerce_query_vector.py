# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for _coerce_query_vector to ensure invalid input types
raise TypeError instead of falling into the numpy conversion branch."""

import numpy as np
import pyarrow as pa
import pytest
from lance.dataset import _coerce_query_vector


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


class TestCoerceQueryVectorValidTypes:
    """Valid vector inputs should be coerced successfully."""

    def test_list_of_floats(self):
        result, dim = _coerce_query_vector([1.0, 2.0, 3.0])
        assert isinstance(result, pa.FloatingPointArray)
        assert dim == 3

    def test_tuple_of_floats(self):
        result, dim = _coerce_query_vector((1.0, 2.0, 3.0))
        assert isinstance(result, pa.FloatingPointArray)
        assert dim == 3

    def test_numpy_array(self):
        result, dim = _coerce_query_vector(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, pa.FloatingPointArray)
        assert dim == 3

    def test_pa_float_array(self):
        result, dim = _coerce_query_vector(pa.array([1.0, 2.0, 3.0]))
        assert isinstance(result, pa.FloatingPointArray)
        assert dim == 3

    def test_pa_int_array_cast_to_float(self):
        result, dim = _coerce_query_vector(pa.array([1, 2, 3]))
        assert isinstance(result, pa.FloatingPointArray)
        assert dim == 3

    def test_pa_chunked_array(self):
        chunked = pa.chunked_array([[1.0, 2.0, 3.0]])
        result, dim = _coerce_query_vector(chunked)
        assert isinstance(result, pa.FloatingPointArray)
        assert dim == 3
