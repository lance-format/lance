# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import lance.arrow
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pytest
from lance.util import validate_vector_index
from lance.vector import vec_to_table


def test_table_roundtrip(tmp_path, provide_pandas):
    uri = tmp_path

    df = pd.DataFrame({"a": range(100), "b": range(100)})
    tbl = pa.Table.from_pandas(df)
    lance.write_dataset(tbl, uri)

    dataset = lance.dataset(uri)
    assert dataset.uri == str(uri.absolute())
    assert tbl.schema == dataset.schema
    assert tbl == dataset.to_table()

    one_col = dataset.to_table(columns=["a"])
    assert one_col == tbl.select(["a"])

    table = dataset.to_table(columns=["a"], limit=20)
    assert len(table) == 20
    with_offset = dataset.to_table(columns=["a"], offset=10, limit=10)
    assert with_offset == table[10:]


def test_input_types(tmp_path, provide_pandas):
    # test all input types for write_dataset
    uri = tmp_path

    df = pd.DataFrame({"a": range(100), "b": range(100)})
    tbl = pa.Table.from_pandas(df)

    lance.write_dataset(df, str(uri / "pandas.lance"), schema=tbl.schema)
    assert tbl == lance.dataset(str(uri / "pandas.lance")).to_table()

    _check_roundtrip(tbl, uri / "table.lance", tbl)

    parquet_uri = str(uri / "dataset.parquet")
    pa_ds.write_dataset(tbl, parquet_uri, format="parquet")
    ds = pa_ds.dataset(parquet_uri)
    _check_roundtrip(ds, uri / "ds.lance", tbl)

    scanner = pa_ds.Scanner.from_dataset(ds)
    _check_roundtrip(scanner, uri / "scanner.lance", tbl)

    reader = scanner.to_reader()
    _check_roundtrip(reader, uri / "reader.lance", tbl)

    # TODO allow Dataset::create to take both async and also RecordBatchReader
    # lance_dataset = lance.dataset(uri / "table.lance")
    # _check_roundtrip(lance_dataset, uri / "lance_dataset.lance", tbl)

    # lance_scanner = lance_dataset.scanner()
    # _check_roundtrip(lance_scanner, uri / "lance_scanner.lance", tbl)


def _check_roundtrip(data_obj, uri, expected):
    lance.write_dataset(data_obj, uri)
    assert expected == lance.dataset(uri).to_table()


def test_nearest(tmp_path):
    uri = tmp_path

    schema = pa.schema([pa.field("emb", pa.list_(pa.float32(), 32), False)])
    npvals = np.random.rand(100, 32)
    npvals /= np.sqrt((npvals**2).sum(axis=1))[:, None]
    values = pa.array(npvals.ravel(), type=pa.float32())
    arr = pa.FixedSizeListArray.from_arrays(values, 32)
    tbl = pa.Table.from_arrays([arr], schema=schema)
    lance.write_dataset(tbl, uri)

    dataset = lance.dataset(uri)
    top10 = dataset.to_table(
        nearest={"column": "emb", "q": arr[0].values, "k": 10, "nprobes": 10}
    )
    distances = l2sq(arr[0].values, npvals.reshape((100, 32)))
    indices = np.argsort(distances)
    assert tbl.take(indices[:10]).to_pandas().equals(top10.to_pandas()[["emb"]])
    assert np.allclose(distances[indices[:10]], top10.to_pandas()["_distance"].values)


def l2sq(vec, mat):
    return np.sum((mat - vec) ** 2, axis=1)


def test_nearest_cosine(tmp_path):
    tbl = vec_to_table([[2.5, 6], [7.4, 3.3]])
    lance.write_dataset(tbl, tmp_path)

    q = [483.5, 1384.5]
    dataset = lance.dataset(tmp_path)
    rs = dataset.to_table(
        nearest={"column": "vector", "q": q, "k": 10, "metric": "cosine"}
    ).to_pandas()
    for i in range(len(rs)):
        assert rs["_distance"][i] == pytest.approx(
            cosine_distance(rs.vector[i], q), abs=1e-6
        )
        assert 0 <= rs["_distance"][i] <= 1


def cosine_distance(vec1, vec2):
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def test_count_rows(tmp_path, provide_pandas):
    df = pd.DataFrame({"values": range(100)})
    tbl = pa.Table.from_pandas(df)
    dataset = lance.write_dataset(tbl, tmp_path)
    assert dataset.count_rows() == 100


def test_create_index(tmp_path):
    dataset = _create_dataset(str(tmp_path / "test.lance"))

    # Check required args
    with pytest.raises(ValueError):
        dataset.create_index("emb", "IVF_PQ")
    with pytest.raises(ValueError):
        dataset.create_index("emb", "IVF_PQ", num_partitions=5)
    with pytest.raises(KeyError):
        dataset.create_index("foo", "IVF_PQ", num_partitions=5, num_sub_vectors=16)
    with pytest.raises(NotImplementedError):
        dataset.create_index("emb", "foo", num_partitions=5, num_sub_vectors=16)

    # all good
    dataset.create_index("emb", "IVF_PQ", num_partitions=16, num_sub_vectors=4)


def test_create_index_shuffle_params(tmp_path):
    dataset = _create_dataset(str(tmp_path / "test.lance"), num_batches=10)

    dataset = dataset.create_index(
        "emb",
        "IVF_PQ",
        num_partitions=16,
        num_sub_vectors=4,
        shuffle_partition_batches=1,
        shuffle_partition_concurrency=10,
    )

    validate_vector_index(dataset, "emb", sample_size=100)


def _create_dataset(uri, num_batches=1):
    schema = pa.schema([pa.field("emb", pa.list_(pa.float32(), 32), False)])
    npvals = np.random.rand(1000, 32)
    npvals /= np.sqrt((npvals**2).sum(axis=1))[:, None]
    values = pa.array(npvals.ravel(), type=pa.float32())
    arr = pa.FixedSizeListArray.from_arrays(values, 32)
    tbl = pa.Table.from_arrays([arr], schema=schema)
    lance.write_dataset(tbl, uri)

    for _ in range(num_batches - 1):
        lance.write_dataset(tbl, uri, mode="append")

    return lance.dataset(uri)


def test_schema_to_json():
    schema = pa.schema(
        [
            pa.field("embedding", pa.list_(pa.float32(), 32), False),
            pa.field("id", pa.int64(), True),
        ]
    )
    json_schema = lance.schema_to_json(schema)
    assert json_schema == {
        "fields": [
            {
                "name": "embedding",
                "nullable": False,
                "type": {
                    "type": "fixed_size_list",
                    "fields": [
                        {"name": "item", "nullable": True, "type": {"type": "float"}}
                    ],
                    "length": 32,
                },
            },
            {
                "name": "id",
                "type": {
                    "type": "int64",
                },
                "nullable": True,
            },
        ],
    }
    actual_schema = lance.json_to_schema(json_schema)
    assert actual_schema == schema


@pytest.fixture
def sample_data_all_types():
    nrows = 10

    tensor_type = pa.fixed_shape_tensor(pa.float32(), [2, 3])
    inner = pa.array([float(x) for x in range(60)], pa.float32())
    storage = pa.FixedSizeListArray.from_arrays(inner, 6)
    tensor_array = pa.ExtensionArray.from_storage(tensor_type, storage)

    return pa.table(
        {
            # TODO: add remaining types
            "str": pa.array([str(i) for i in range(nrows)]),
            "float16": pa.array(
                [np.float16(1.0 + i / 10) for i in range(nrows)], pa.float16()
            ),
            "bfloat16": lance.arrow.bfloat16_array(
                [1.0 + i / 10 for i in range(nrows)]
            ),
            "tensor": tensor_array,
        }
    )


def test_roundtrip_types(tmp_path, sample_data_all_types):
    dataset = lance.write_dataset(sample_data_all_types, tmp_path)
    roundtripped = dataset.to_table()
    assert roundtripped == sample_data_all_types


def test_roundtrip_schema(tmp_path):
    schema = pa.schema([pa.field("a", pa.float64())], metadata={"key": "value"})
    data = pa.table({"a": [1.0, 2.0]}).to_batches()
    dataset = lance.write_dataset(data, tmp_path, schema=schema)
    assert dataset.schema == schema


def test_io_counters(tmp_path):
    starting_iops = lance.iops_counter()
    starting_bytes = lance.bytes_read_counter()
    dataset = lance.write_dataset(pa.table({"a": [1, 2, 3]}), tmp_path)
    dataset.to_table()
    assert lance.iops_counter() > starting_iops
    assert lance.bytes_read_counter() > starting_bytes


def test_simd_info():
    info = lance.simd_info()
    assert info["tier"] in (
        "none",
        "sse",
        "avx",
        "avx_fma",
        "avx2",
        "avx512",
        "avx512_fp16",
        "neon",
        "lsx",
        "lasx",
    )
    assert isinstance(info["target_arch"], str) and info["target_arch"]
    if info["target_arch"] == "x86_64":
        # The x86_64 ABI mandates SSE2.
        assert "sse2" in info["host_features"]
    else:
        assert info["host_features"] == []


@pytest.mark.parametrize(
    ("query", "options", "expected"),
    [
        ("the cats and dogs", {}, [("cat", 0), ("dog", 2)]),
        (
            "the skip alpha",
            {"stem": False, "custom_stop_words": ["skip"]},
            [("the", 0), ("alpha", 2)],
        ),
        (
            "getUserName",
            {"analyzer": "code", "split_identifiers": True},
            [
                ("getusername", 0),
                ("get", 0),
                ("user", 1),
                ("name", 2),
            ],
        ),
        (
            "a::b",
            {"analyzer": "code", "index_operators": True},
            [("a", 0), ("::", 1), ("b", 2)],
        ),
    ],
)
def test_tokenize_fts_query(query, options, expected):
    tokens = lance.tokenize(query, **options)
    assert [(token.text, token.position) for token in tokens] == expected


def test_tokenize_fts_query_validates_analyzer_options():
    with pytest.raises(ValueError, match="code analyzer flags require analyzer='code'"):
        lance.tokenize("getUserName", split_identifiers=True)

    with pytest.raises(ValueError, match="unknown base tokenizer"):
        lance.tokenize("hello", base_tokenizer="unknown")


def test_tokenize_fts_query_can_disable_max_token_length():
    long_token = "x" * 41
    assert lance.tokenize(long_token, stem=False) == []
    unlimited_tokens = lance.tokenize(long_token, max_token_length=None, stem=False)
    assert [token.text for token in unlimited_tokens] == [long_token]


@pytest.mark.parametrize(
    "row_param, column_name",
    [("with_row_id", "_rowid"), ("with_row_address", "_rowaddr")],
)
def test_with_row_id(tmp_path, row_param, column_name):
    path = tmp_path / "test_with_inner_column.lance"
    data = pa.table(
        {
            "data_item_id": pa.array([1001, 1002, 1003]),
            "a": pa.array([1, 2, 3]),
        }
    )

    lance.write_dataset(data, path)
    lance_ds = lance.dataset(path)

    scanner_args = {row_param: True, "columns": ["data_item_id", column_name]}
    table = lance_ds.scanner(**scanner_args).to_table()
    assert table.column(column_name).type == pa.uint64()

    scanner_args = {row_param: True, "columns": ["data_item_id"]}
    table = lance_ds.scanner(**scanner_args).to_table()
    assert column_name in table.column_names
    assert table.column(column_name).type == pa.uint64()

    table = lance_ds.scanner(
        **{row_param: False, "columns": ["data_item_id", column_name]}
    ).to_table()
    assert column_name in table.column_names
