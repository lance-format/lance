# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from lance import FFILanceTableProvider, LanceDataset


def normalize(batches: list[pa.RecordBatch]) -> pa.RecordBatch:
    normalized = pa.Table.from_batches(batches).combine_chunks()
    batches = normalized.to_batches()
    assert len(batches) == 1
    return batches[0]


def test_table_loading():
    pytest.importorskip("datafusion")
    from datafusion import SessionContext, col

    lancedb_temp_path = "/tmp/test.lance"
    shutil.rmtree(lancedb_temp_path, ignore_errors=True)

    big_str = ("a" * 1000000).encode("utf-8")
    # Need a good amount of data to expose certain I/O patterns (if the file is too
    # small we trigger the small-files path which avoids I/O entirely during scan.)
    df = pa.table(
        {
            "col1": range(1000000),
            "col2": [str(i) for i in range(1000000)],
            "col3": [float(i) for i in range(1000000)],
        },
        schema=pa.schema(
            [
                pa.field("col1", pa.int64()),
                pa.field("col2", pa.string()),
                pa.field("col3", pa.float64()),
            ],
            metadata={
                b"big_str": big_str,
            },
        ),
    )
    dataset: LanceDataset = lance.write_dataset(df, lancedb_temp_path)
    dataset.create_scalar_index("col1", "BTREE")

    # We remake the context each time.  This ensures we are not skipping over any
    # I/O because we just happen to have data in the cache.
    def make_ctx():
        ctx = SessionContext()

        dataset = lance.dataset(lancedb_temp_path)
        ffi_lance_table = FFILanceTableProvider(
            dataset, with_row_id=True, with_row_addr=True
        )
        ctx.register_table("ffi_lance_table", ffi_lance_table)
        return ctx

    ctx = make_ctx()
    result = normalize(ctx.table("ffi_lance_table").collect())

    assert len(result) == 1000000
    assert result.num_columns == 5

    expected = pd.DataFrame(
        {
            "col1": np.array(range(1000000), dtype=np.int64),
            "col2": [str(i) for i in range(1000000)],
            "col3": np.array([float(i) for i in range(1000000)], dtype=np.float64),
            "_rowid": np.array(range(1000000), dtype=np.uint64),
            "_rowaddr": np.array(range(1000000), dtype=np.uint64),
        }
    )

    pd.testing.assert_frame_equal(result.to_pandas(), expected)

    ctx = make_ctx()
    result = normalize(ctx.table("ffi_lance_table").filter(col("col1") == 4).collect())
    assert len(result) == 1

    ctx = make_ctx()
    result = normalize(ctx.table("ffi_lance_table").limit(1).collect())
    assert len(result) == 1
    assert result["col1"][0].as_py() == 0

    ctx = make_ctx()
    result = normalize(ctx.table("ffi_lance_table").limit(1, offset=1).collect())
    assert len(result) == 1
    assert result["col1"][0].as_py() == 1


def test_custom_udf_filter(tmp_path):
    pytest.importorskip("datafusion")
    from datafusion import SessionContext, udf

    def is_even(values: pa.Array) -> pa.Array:
        return pa.array([value.as_py() % 2 == 0 for value in values], type=pa.bool_())

    is_even_udf = udf(
        is_even,
        input_fields=[pa.int64()],
        return_field=pa.bool_(),
        volatility="stable",
        name="is_even",
    )

    dataset = lance.write_dataset(pa.table({"i": [1, 2, 3, 4]}), str(tmp_path))
    provider = FFILanceTableProvider(dataset, with_row_id=True, with_row_addr=True)

    ctx = SessionContext()
    ctx.register_table("numbers", provider)
    ctx.register_udf(is_even_udf)

    result = normalize(
        ctx.sql("SELECT i FROM numbers WHERE i = 2 AND is_even(i)").collect()
    )
    assert result["i"].to_pylist() == [2]
