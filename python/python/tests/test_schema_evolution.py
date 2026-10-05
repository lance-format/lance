# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import uuid
from pathlib import Path

import lance
import lance.dependencies as dependencies
import lance.udf as udf_module
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from lance import LanceDataset
from lance.file import LanceFileReader, LanceFileWriter
from lance.fragment import write_fragments


def test_drop_columns(tmp_path: Path):
    dims = 32
    nrows = 512
    values = pc.random(nrows * dims).cast("float32")
    table = pa.table(
        {
            "a": pa.FixedSizeListArray.from_arrays(values, dims),
            "b": range(nrows),
            "c": range(nrows),
        }
    )
    dataset = lance.write_dataset(table, tmp_path)
    dataset.create_index("a", "IVF_PQ", num_partitions=2, num_sub_vectors=1)

    # Drop a column, index is kept
    dataset.drop_columns(["b"])
    assert dataset.schema == pa.schema(
        {
            "a": pa.list_(pa.float32(), dims),
            "c": pa.int64(),
        }
    )
    assert len(dataset.describe_indices()) == 1

    # Drop vector column, index is dropped
    dataset.drop_columns(["a"])
    assert dataset.schema == pa.schema({"c": pa.int64()})
    assert len(dataset.describe_indices()) == 0

    # Can't drop all columns
    with pytest.raises(ValueError):
        dataset.drop_columns(["c"])


# The LanceDataset.add_columns and LanceFragment.merge_columns should be mostly the
# same.  Tests that test these methods can use this fixture to test both methods.
def check_add_columns(
    dataset: LanceDataset, expected: pa.Table, use_fragments: bool, *args, **kwargs
):
    if use_fragments:
        # Ensure we are working with latest dataset version
        dataset = lance.dataset(dataset.uri)
        new_frags = []
        for fragment in dataset.get_fragments():
            # the parameter name is different in `merge_columns` (backwards compat.)
            if "read_columns" in kwargs:
                kwargs["columns"] = kwargs.pop("read_columns")
            new_frag, schema = fragment.merge_columns(*args, **kwargs)
            new_frags.append(new_frag)
        op = lance.LanceOperation.Merge(new_frags, schema)
        dataset = LanceDataset.commit(dataset.uri, op, read_version=dataset.version)
        assert dataset.to_table() == expected
    else:
        dataset.add_columns(*args, **kwargs)
        assert dataset.to_table() == expected


def check_add_columns_fails(
    dataset: LanceDataset,
    use_fragments: bool,
    expected_exception: any,
    match: str,
    *args,
    **kwargs,
):
    if use_fragments:
        with pytest.raises(expected_exception, match=match):
            frag = dataset.get_fragments()[0]
            frag.merge_columns(*args, **kwargs)
    else:
        with pytest.raises(expected_exception, match=match):
            dataset.add_columns(*args, **kwargs)


@pytest.mark.parametrize("use_fragments", [False, True])
def test_add_columns_udf(tmp_path, use_fragments):
    tab = pa.table({"a": range(100), "b": range(100)})
    dataset = lance.write_dataset(tab, tmp_path, max_rows_per_file=25)

    @lance.batch_udf(
        output_schema=pa.schema([pa.field("double_a", pa.int64())]),
    )
    def double_a(batch):
        assert batch.schema.names == ["a"]
        return pa.record_batch(
            [pa.array([2 * x.as_py() for x in batch["a"]])], ["double_a"]
        )

    expected = tab.append_column("double_a", pa.array([2 * x for x in range(100)]))
    check_add_columns(dataset, expected, use_fragments, double_a, read_columns=["a"])

    # Check: errors if produces inconsistent schema
    @lance.batch_udf()
    def make_new_col(batch):
        col_name = str(uuid.uuid4())
        return pa.record_batch([batch["a"]], [col_name])

    check_add_columns_fails(
        dataset,
        use_fragments,
        Exception,
        "Output schema of function does not match the expected schema",
        make_new_col,
    )

    # Schema inference and Pandas conversion
    @lance.batch_udf()
    def triple_a(batch):
        return pd.DataFrame({"triple_a": [3 * x.as_py() for x in batch["a"]]})

    expected = expected.append_column("triple_a", pa.array([3 * x for x in range(100)]))
    check_add_columns(dataset, expected, use_fragments, triple_a, read_columns=["a"])


def test_add_columns_from_rbr(tmp_path):
    tab = pa.table({"a": range(100), "b": range(100)})
    dataset = lance.write_dataset(tab, tmp_path / "dataset", max_rows_per_file=25)

    # New data in smaller chunks than old data
    def gen_data():
        for i in range(34):
            num_rows = 3
            if i == 33:
                num_rows = 1
            yield pa.record_batch(
                [pa.array(range(num_rows)), pa.array(range(num_rows))], ["c", "d"]
            )

    dataset.add_columns(
        gen_data(),
        reader_schema=pa.schema([pa.field("c", pa.int64()), pa.field("d", pa.int64())]),
    )

    expected = tab.append_column(
        "c", pa.array([i % 3 for i in range(100)])
    ).append_column("d", pa.array([i % 3 for i in range(100)]))

    assert expected == dataset.to_table()

    # New data in larger chunks than old data
    def gen_data():
        for i in range(3):
            num_rows = 40
            if i == 2:
                num_rows = 20
            yield pa.record_batch([pa.array(range(num_rows))], ["e"])

    dataset.add_columns(
        gen_data(),
        reader_schema=pa.schema([pa.field("e", pa.int64())]),
    )

    expected = expected.append_column("e", pa.array([i % 40 for i in range(100)]))

    assert expected == dataset.to_table()

    # Insufficient number of rows

    def gen_data():
        yield pa.record_batch([pa.array(range(50))], ["f"])

    with pytest.raises(
        OSError, match="Stream ended before producing values for all rows in dataset"
    ):
        dataset.add_columns(
            gen_data(),
            reader_schema=pa.schema([pa.field("f", pa.int64())]),
        )

    # Too many rows

    def gen_data():
        yield pa.record_batch([pa.array(range(101))], ["f"])

    with pytest.raises(
        OSError, match="Stream produced more values than expected for dataset"
    ):
        dataset.add_columns(
            gen_data(),
            reader_schema=pa.schema([pa.field("f", pa.int64())]),
        )


def test_add_columns_from_file(tmp_path):
    tab = pa.table({"a": range(100), "b": range(100)})
    dataset = lance.write_dataset(tab, tmp_path / "dataset", max_rows_per_file=25)

    tbl_one = dataset.to_table(columns={"double_b": "b * 2"}, limit=50)
    with LanceFileWriter(tmp_path / "tbl_0") as writer:
        writer.write_batch(tbl_one)

    tbl_two = dataset.to_table(columns={"double_b": "b * 2"}, offset=50, limit=50)
    with LanceFileWriter(tmp_path / "tbl_1") as writer:
        writer.write_batch(tbl_two)

    def data_gen():
        for i in range(2):
            reader = LanceFileReader(str(tmp_path / f"tbl_{i}"))
            for batch in reader.read_all().to_batches():
                yield batch

    reader_schema = pa.schema([pa.field("double_b", pa.int64())])

    dataset.add_columns(data_gen(), reader_schema=reader_schema)

    expected = tab.append_column("double_b", pa.array([2 * x for x in range(100)]))
    assert expected == dataset.to_table()


def test_add_columns_udf_caching(tmp_path):
    tab = pa.table(
        {
            "a": range(100),
            "b": range(100),
        }
    )
    dataset = lance.write_dataset(tab, tmp_path, max_rows_per_file=20)

    @lance.batch_udf(checkpoint_file=tmp_path / "cache.sqlite")
    def double_a(batch):
        if batch["a"][0].as_py() >= 50:
            raise RuntimeError("I failed")
        return pa.record_batch([pc.multiply(batch["a"], pa.scalar(2))], ["a_times_2"])

    with pytest.raises(Exception):
        dataset.add_columns(double_a, read_columns=["a"])

    assert dataset.version == 1
    assert "cache.sqlite" in os.listdir(tmp_path)

    @lance.batch_udf(checkpoint_file=tmp_path / "cache.sqlite")
    def double_a(batch):
        # We should skip these batches if they are cached
        # (It can be zero due to schema inference looking at the first batch.)
        assert batch["a"][0].as_py() == 0 or batch["a"][0].as_py() >= 50
        return pa.record_batch([pc.multiply(batch["a"], pa.scalar(2))], ["a_times_2"])

    dataset.add_columns(double_a, read_columns=["a"])
    assert dataset.schema.names == ["a", "b", "a_times_2"]

    assert "cache.sqlite" not in os.listdir(tmp_path)


@pytest.mark.parametrize("use_fragments", [False, True])
def test_add_columns_exprs(tmp_path, use_fragments):
    tab = pa.table({"a": range(100)})
    dataset = lance.write_dataset(tab, tmp_path)
    expected = pa.table({"a": range(100), "b": range(1, 101)})
    check_add_columns(dataset, expected, use_fragments, {"b": "a + 1"})


def test_add_many_columns(tmp_path: Path):
    table = pa.table([pa.array([1, 2, 3])], names=["0"])
    dataset = lance.write_dataset(table, tmp_path)
    dataset.add_columns(dict([(str(i), "0") for i in range(1, 1000)]))
    dataset = lance.dataset(tmp_path)
    assert dataset.to_table().num_rows == 3


@pytest.mark.parametrize("use_fragments", [False, True])
def test_add_columns_callable(tmp_path: Path, use_fragments):
    table = pa.table({"a": range(100)})
    dataset = lance.write_dataset(table, tmp_path)

    def mapper(batch: pa.RecordBatch):
        plus_one = pc.add(batch["a"], 1)
        return pa.record_batch([plus_one], names=["b"])

    expected = pa.table({"a": range(100), "b": range(1, 101)})
    check_add_columns(dataset, expected, use_fragments, mapper)


@pytest.mark.parametrize("use_batch_udf", [False, True])
def test_add_columns_arrow_udf_without_pandas_dependency(
    tmp_path: Path, monkeypatch, use_batch_udf
):
    table = pa.table({"caption": ["a Shutterstock photo", "clean"]})
    dataset = lance.write_dataset(table, tmp_path)

    def mapper(batch: pa.RecordBatch) -> pa.RecordBatch:
        flags = pc.match_substring_regex(
            pc.utf8_lower(batch["caption"]), "shutterstock"
        )
        return pa.record_batch([flags], names=["wm"])

    if use_batch_udf:
        mapper = lance.batch_udf()(mapper)

    # CI has pandas installed, so simulate Lance's no-pandas lazy proxy state.
    # Without the guard, accessing pd.DataFrame raises ModuleNotFoundError.
    monkeypatch.setattr(dependencies, "_PANDAS_AVAILABLE", False)
    monkeypatch.setattr(
        udf_module,
        "pd",
        dependencies._LazyModule("pandas", module_available=False),
    )

    dataset.add_columns(mapper, read_columns=["caption"], batch_size=64)

    expected = table.append_column("wm", pa.array([True, False]))
    assert dataset.to_table() == expected


def test_query_after_merge(tmp_path):
    # https://github.com/lancedb/lance/issues/1905
    tab = pa.table(
        {
            "id": range(100),
            "vec": pa.FixedShapeTensorArray.from_numpy_ndarray(
                np.random.rand(100, 128).astype("float32")
            ),
        }
    )
    tab2 = pa.table(
        {
            "id": range(100),
            "data": range(100, 200),
        }
    )
    dataset = lance.write_dataset(tab, tmp_path)

    dataset.merge(tab2, left_on="id")

    dataset.to_table(
        nearest=dict(column="vec", k=10, q=np.random.rand(128).astype("float32"))
    )


def test_alter_columns(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("a", pa.int64(), nullable=False),
            pa.field("b", pa.string(), nullable=False),
        ]
    )
    tab = pa.table(
        {"a": pa.array([1, 2, 1024]), "b": pa.array(["a", "b", "c"])}, schema=schema
    )

    dataset = lance.write_dataset(tab, tmp_path)

    dataset.alter_columns(
        {"path": "a", "name": "x", "nullable": True},
        {"path": "b", "name": "y"},
    )

    expected_schema = pa.schema(
        [
            pa.field("x", pa.int64()),
            pa.field("y", pa.string(), nullable=False),
        ]
    )
    assert dataset.schema == expected_schema

    expected_tab = pa.table(
        {"x": pa.array([1, 2, 1024]), "y": pa.array(["a", "b", "c"])},
        schema=expected_schema,
    )
    assert dataset.to_table() == expected_tab

    dataset.alter_columns(
        {"path": "x", "data_type": pa.int32()},
        {"path": "y", "data_type": pa.large_string()},
    )
    expected_schema = pa.schema(
        [
            pa.field("x", pa.int32()),
            pa.field("y", pa.large_string(), nullable=False),
        ]
    )
    assert dataset.schema == expected_schema

    expected_tab = pa.table(
        {"x": pa.array([1, 2, 1024], type=pa.int32()), "y": pa.array(["a", "b", "c"])},
        schema=expected_schema,
    )
    assert dataset.to_table() == expected_tab
    with pytest.raises(Exception, match="Can't cast value 1024 to type Int8"):
        dataset.alter_columns({"path": "x", "data_type": pa.int8()})

    with pytest.raises(Exception, match='Cannot cast column "x" from Int32 to Utf8'):
        dataset.alter_columns({"path": "x", "data_type": pa.string()})

    with pytest.raises(Exception, match='Column "q" does not exist'):
        dataset.alter_columns({"path": "q", "name": "z"})

    with pytest.raises(ValueError, match="Unknown key: type"):
        dataset.alter_columns({"path": "x", "type": "string"})

    with pytest.raises(
        ValueError,
        match="At least one of name, nullable, or data_type must be specified",
    ):
        dataset.alter_columns({"path": "x"})


def test_merge_columns(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    fragments = dataset.get_fragments()

    fragment = fragments[0]

    def adder(batch: pa.RecordBatch) -> pa.RecordBatch:
        c_array = pa.compute.multiply(batch.column(0), 2)
        return pa.RecordBatch.from_arrays([c_array], names=["c"])

    fragment_metadata, schema = fragment.merge_columns(adder, columns=["a"])

    operation = lance.LanceOperation.Overwrite(schema.to_pyarrow(), [fragment_metadata])
    dataset = lance.LanceDataset.commit(base_dir, operation)
    assert dataset.schema == schema.to_pyarrow()

    tbl = dataset.to_table()
    assert tbl == pa.Table.from_pydict(
        {
            "a": range(100),
            "b": range(100),
            "c": pa.array(range(0, 200, 2), pa.int64()),
        }
    )


def test_merge_columns_from_reader(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    fragments = dataset.get_fragments()

    fragment = fragments[0]

    with LanceFileWriter(tmp_path / "some_file") as writer:
        writer.write_batch(pa.table({"c": range(100), "d": range(100)}))

    def datareader():
        reader = LanceFileReader(str(tmp_path / "some_file"))
        for batch in reader.read_all(batch_size=10).to_batches():
            yield batch

    fragment_metadata, schema = fragment.merge_columns(
        datareader(),
        reader_schema=pa.schema([pa.field("c", pa.int64()), pa.field("d", pa.int64())]),
        batch_size=15,
    )

    operation = lance.LanceOperation.Overwrite(schema.to_pyarrow(), [fragment_metadata])
    dataset = lance.LanceDataset.commit(base_dir, operation)
    assert dataset.schema == schema.to_pyarrow()

    tbl = dataset.to_table()
    assert tbl == pa.Table.from_pydict(
        {
            "a": range(100),
            "b": range(100),
            "c": range(100),
            "d": range(100),
        }
    )


def test_merge_batch_size(tmp_path: Path):
    # Create dataset with 10 fragments with 100 rows each
    table = pa.table({"a": range(1000)})
    for batch_size in [1, 10, 100, 1000]:
        ds_path = str(tmp_path / str(batch_size))
        dataset = lance.write_dataset(table, ds_path, max_rows_per_file=100)
        fragments = []

        def mutate(batch):
            assert batch.num_rows <= batch_size
            return pa.RecordBatch.from_pydict({"b": batch.column("a")})

        for frag in dataset.get_fragments():
            merged, schema = frag.merge_columns(mutate, batch_size=batch_size)
            fragments.append(merged)

        merge = lance.LanceOperation.Merge(fragments, schema)
        dataset = lance.LanceDataset.commit(
            ds_path, merge, read_version=dataset.version
        )

        dataset.validate()
        tbl = dataset.to_table()
        expected = pa.table({"a": range(1000), "b": range(1000)})
        assert tbl == expected


def test_add_cols_batch_size(tmp_path: Path):
    # Same test as `test_merge_batch_size` but using LanceDataset.add_columns instead
    table = pa.table({"a": range(1000)})
    for batch_size in [1, 10, 100, 1000]:
        ds_path = str(tmp_path / str(batch_size))
        dataset = lance.write_dataset(table, ds_path, max_rows_per_file=100)

        def mutate(batch):
            assert batch.num_rows <= batch_size
            return pa.RecordBatch.from_pydict({"b": batch.column("a")})

        dataset.add_columns(mutate, batch_size=batch_size)

        dataset.validate()
        tbl = dataset.to_table()
        expected = pa.table({"a": range(1000), "b": range(1000)})
        assert tbl == expected


def test_no_checkpoint_merge_columns(tmp_path: Path):
    tab = pa.table(
        {
            "a": range(100),
            "b": range(100),
        }
    )
    dataset = lance.write_dataset(tab, tmp_path, max_rows_per_file=20)

    @lance.batch_udf(checkpoint_file=tmp_path / "cache.sqlite")
    def some_udf(batch):
        return batch

    frag = dataset.get_fragments()[0]

    with pytest.raises(ValueError, match="A checkpoint file cannot be used"):
        frag.merge_columns(some_udf, columns=["a"])


def test_add_cols_all_null_with_sql(tmp_path: Path):
    tab = pa.table(
        {
            "a": range(100),
        }
    )
    dataset = lance.write_dataset(
        tab, tmp_path, max_rows_per_file=50, data_storage_version="stable"
    )
    fragments_before = dataset.get_fragments()
    dataset.add_columns({"b": "CAST(NULL AS INT)"})
    fragments_after = dataset.get_fragments()

    # assert this was a metadata only operation and no data was written
    assert len(fragments_before) == len(fragments_after)
    for frag_before, frag_after in zip(fragments_before, fragments_after):
        assert frag_before.fragment_id == frag_after.fragment_id
        assert frag_before.data_files() == frag_after.data_files()

    # assert the schema is as expected
    assert dataset.schema == pa.schema(
        {
            "a": pa.int64(),
            "b": pa.int32(),
        }
    )


def test_merge_nullability_assertion(tmp_path: Path):
    tbl = pa.table({"value": pa.array([1, 2], pa.int32())})
    lance.write_dataset(tbl, tmp_path)
    written_at = lance.dataset(tmp_path).version

    # Stage an append against the original schema, then add a non-nullable
    # column. The merge claims non-null, and reading it back must preserve
    # the claim, or recommitting it would silently drop the barrier.
    fragments = write_fragments(
        pa.table({"value": pa.array([7], pa.int32())}), tmp_path, mode="append"
    )
    lance.dataset(tmp_path).add_columns({"one": "1"})
    txn = lance.dataset(tmp_path).read_transaction(2)
    assert txn is not None
    assert txn.operation.preserves_nullability is False

    # The stale append omits the required column, so its rows would read as
    # null under the merged schema; the claim refuses it.
    op = lance.LanceOperation.Append(fragments)
    with pytest.raises(Exception, match="preempted"):
        lance.LanceDataset.commit(tmp_path, op, read_version=written_at)

    # A nullable add preserves nullability and skips the barrier.
    lance.dataset(tmp_path).add_columns({"copied": "value"})
    txn = lance.dataset(tmp_path).read_transaction(3)
    assert txn is not None
    assert txn.operation.preserves_nullability is True


def test_project_nullability_assertion_round_trips(tmp_path: Path):
    tbl = pa.table({"value": pa.array([1, 2], pa.int32())})
    lance.write_dataset(tbl, tmp_path)
    lance.dataset(tmp_path).alter_columns({"path": "value", "nullable": False})

    # Reading the tightening back must preserve the claim, or recommitting it
    # would silently drop the concurrency barrier.
    txn = lance.dataset(tmp_path).read_transaction(2)
    assert txn is not None
    assert txn.operation.preserves_nullability is False

    # A Python-built non-assertion must reach the barrier: race an append.
    written_at = lance.dataset(tmp_path).version
    appended = lance.write_dataset(
        pa.table({"value": pa.array([7], pa.int32())}), tmp_path, mode="append"
    )
    assert appended.version > written_at
    relax = lance.LanceOperation.Project(
        schema=txn.operation.schema, preserves_nullability=False
    )
    with pytest.raises(Exception, match="preempted"):
        lance.LanceDataset.commit(tmp_path, relax, read_version=written_at)
