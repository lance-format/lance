# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for query vector coercion and scanner input validation."""

import lance
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


QUERIES = [[1, 0], [0, 1]]


@pytest.fixture(
    params=[
        pytest.param(pa.list_(pa.int32()), id="list"),
        pytest.param(pa.large_list(pa.float32()), id="large_list"),
        pytest.param(pa.list_(pa.float64(), 2), id="fixed_size_list"),
    ]
)
def query_type(request):
    return request.param


@pytest.fixture(params=["array", "slice", "chunked"])
def arrow_queries(query_type, request):
    if request.param == "array":
        return pa.array(QUERIES, type=query_type)

    padded = pa.array([[9, 9], *QUERIES, [8, None]], type=query_type)
    if request.param == "slice":
        return padded.slice(1, 2)
    return pa.chunked_array([padded.slice(1, 1), padded.slice(2, 1)])


def test_coerce_arrow_query_vectors(arrow_queries):
    result, dim = _coerce_query_vector(arrow_queries)
    assert result.type.value_type == pa.float32()
    assert type(result.type) is type(arrow_queries.type)
    assert result.to_pylist() == QUERIES
    assert dim == 2


@pytest.fixture
def vector_dataset(tmp_path):
    vector_type = pa.list_(pa.float32(), 2)
    return lance.write_dataset(
        pa.table(
            {
                "id": [1, 2, 3, 4],
                "vector": pa.array([[1, 0], [2, 0], [0, 2], [0, 1]], type=vector_type),
                "refine": pa.array([[2, 0], [1, 0], [0, 1], [0, 2]], type=vector_type),
            }
        ),
        tmp_path / "vectors",
        max_rows_per_file=2,
    )


def test_batch_search_with_arrow_queries(vector_dataset, arrow_queries):
    options = {"column": "vector", "k": 2, "use_index": False}
    expected = vector_dataset.scanner(
        columns=["id"], nearest={**options, "q": QUERIES}
    ).to_table()
    actual = vector_dataset.scanner(
        columns=["id"], nearest={**options, "q": arrow_queries}
    ).to_table()
    assert actual.equals(expected)
    assert actual["query_index"].to_pylist() == [0, 0, 1, 1]
    assert actual["id"].to_pylist() == [1, 2, 4, 3]
    assert actual["_distance"].to_pylist() == [0.0, 1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "query_type",
    [pa.list_(pa.float64()), pa.large_list(pa.float64()), pa.list_(pa.float64(), 2)],
    ids=["list", "large_list", "fixed_size_list"],
)
def test_arrow_query_batches_preserve_float64_precision(tmp_path, query_type):
    # These two coordinates round to the same Float32 value.
    nearby = 1.0 + 2**-26
    dataset = lance.write_dataset(
        pa.table(
            {
                "id": [1, 2],
                "vector": pa.array(
                    [[1.0, 0.0], [nearby, 0.0]], type=pa.list_(pa.float64(), 2)
                ),
            }
        ),
        tmp_path / "float64_vectors",
        max_rows_per_file=1,
    )
    queries = pa.array([[nearby, 0.0], [1.0, 0.0]], type=query_type)
    options = {"column": "vector", "k": 1, "use_index": False}
    batch = dataset.scanner(
        columns=["id"],
        nearest={
            **options,
            "q": pa.chunked_array([queries.slice(0, 1), queries.slice(1, 1)]),
        },
    ).to_table()
    individual = pa.concat_tables(
        [
            dataset.scanner(
                columns=["id"], nearest={**options, "q": query.values}
            ).to_table()
            for query in queries
        ]
    )
    assert batch["query_index"].to_pylist() == [0, 1]
    assert batch["id"].to_pylist() == [2, 1]
    assert batch.drop_columns(["query_index"]).equals(individual)


def test_batch_binary_search_with_large_list_queries(tmp_path):
    vectors = pa.array([[1, 0], [3, 0], [0, 3], [0, 1]], type=pa.list_(pa.uint8(), 2))
    dataset = lance.write_dataset(
        pa.table({"id": [1, 2, 3, 4], "vector": vectors}),
        tmp_path / "binary_vectors",
        max_rows_per_file=2,
    )
    query = pa.array(QUERIES, type=pa.large_list(pa.uint8()))
    options = {"column": "vector", "k": 2, "metric": "hamming", "use_index": False}
    expected = dataset.scanner(
        columns=["id"], nearest={**options, "q": QUERIES}
    ).to_table()
    actual = dataset.scanner(columns=["id"], nearest={**options, "q": query}).to_table()
    assert actual.equals(expected)
    assert actual["id"].to_pylist() == [1, 2, 4, 3]


def test_multivector_search_with_arrow_queries(tmp_path, arrow_queries):
    dataset = lance.write_dataset(
        pa.table(
            {
                "id": [1, 2],
                "vector": pa.array(
                    [[[1, 0], [0, 1]], [[-1, 0], [0, -1]]],
                    type=pa.list_(pa.list_(pa.float32(), 2)),
                ),
            }
        ),
        tmp_path / "multivectors",
        max_rows_per_file=1,
    )
    options = {"column": "vector", "k": 1, "metric": "cosine", "use_index": False}
    expected = dataset.scanner(
        columns=["id"], nearest={**options, "q": QUERIES}
    ).to_table()
    actual = dataset.scanner(
        columns=["id"], nearest={**options, "q": arrow_queries}
    ).to_table()
    assert actual.equals(expected)
    assert actual.column_names == ["id", "_distance"]
    assert actual["id"].to_pylist() == [1]


def test_refine_with_arrow_queries(vector_dataset, arrow_queries):
    options = {
        "column": "vector",
        "q": QUERIES,
        "k": 1,
        "use_index": False,
        "refine_column": "refine",
        "refine_metric": "l2",
        "refine_factor": 2,
    }
    expected = vector_dataset.scanner(
        columns=["id"], nearest={**options, "refine_q": QUERIES}
    ).to_table()
    actual = vector_dataset.scanner(
        columns=["id"], nearest={**options, "refine_q": arrow_queries}
    ).to_table()
    assert actual.equals(expected)
    assert actual["query_index"].to_pylist() == [0, 1]
    assert actual["id"].to_pylist() == [2, 3]


@pytest.mark.parametrize(
    "query,message",
    [
        pytest.param(
            pa.array([], type=pa.list_(pa.float32())),
            "must have non-zero length",
            id="empty",
        ),
        pytest.param(
            pa.array([None, [1, 0]], type=pa.list_(pa.float32(), 2)),
            "vectors must be non-null",
            id="null_vector",
        ),
        pytest.param(
            pa.array([[1, None]], type=pa.large_list(pa.float32())),
            "values must be non-null",
            id="null_coordinate",
        ),
        pytest.param(
            pa.array([[1, 0], [1]], type=pa.large_list(pa.float32())),
            r"query dim\(1\).*vector dim\(2\) at query index 1",
            id="ragged",
        ),
        pytest.param(
            pa.array([[1, 0, 0]], type=pa.list_(pa.float32(), 3)),
            r"dim\(3\).*dim\(2\)",
            id="wrong_dimension",
        ),
    ],
)
def test_rejects_invalid_arrow_query_vectors(vector_dataset, query, message):
    nearest = {"column": "vector", "q": query, "k": 1, "use_index": False}
    with pytest.raises(ValueError, match=message):
        vector_dataset.scanner(nearest=nearest)
    with pytest.raises(ValueError, match=message):
        vector_dataset._ds.scanner(nearest=nearest)
