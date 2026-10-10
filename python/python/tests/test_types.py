# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from functools import singledispatch
from importlib import import_module
from typing import Optional

import lance
import lance.types as source_types
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import pytest
from lance.types import SourceStrategy, WriteSource, _coerce_reader, coerce_source


@pytest.mark.parametrize("kind", ["table", "batch", "dict", "dict_list"])
def test_materialized_sources(kind):
    table = pa.table({"id": [1, 2], "value": [10, 20]})
    inputs = {
        "table": table,
        "batch": table.to_batches()[0],
        "dict": table.to_pydict(),
        "dict_list": table.to_pylist(),
    }
    source = coerce_source(inputs[kind])
    assert source.strategy is SourceStrategy.MATERIALIZED
    assert source.schema == table.schema
    assert source.reader_factory().read_all() == table


@pytest.mark.parametrize("kind", ["reader", "list", "tuple", "iterator"])
def test_one_shot_sources(kind):
    table = pa.table({"id": [1, 2], "value": [10, 20]})
    batches = table.to_batches(max_chunksize=1)
    inputs = {
        "reader": table.to_reader(),
        "list": batches,
        "tuple": tuple(batches),
        "iterator": iter(batches),
    }
    source = coerce_source(inputs[kind], table.schema)
    assert source.strategy is SourceStrategy.ONE_SHOT
    assert source.schema == table.schema
    reader = source.reader_factory()
    assert reader.read_all() == table
    if kind == "reader":
        assert reader is inputs[kind]


@pytest.mark.parametrize("kind", ["table", "batch", "reader"])
def test_arrow_input_retains_its_schema(kind):
    table = pa.table({"value": pa.array([1, 2], pa.int64())})
    requested_schema = pa.schema([("value", pa.int32())])
    inputs = {
        "table": table,
        "batch": table.to_batches()[0],
        "reader": table.to_reader(),
    }
    result = _coerce_reader(inputs[kind], requested_schema).read_all()
    assert result == table


@pytest.mark.parametrize("data", [[], iter(())])
def test_empty_iterable_requires_schema(data):
    with pytest.raises(ValueError, match="Must provide schema"):
        coerce_source(data)


def test_empty_iterable_with_schema():
    schema = pa.schema([("id", pa.int64())])
    source = coerce_source([], schema)
    assert source.strategy is SourceStrategy.ONE_SHOT
    assert source.schema == schema
    assert source.reader_factory().read_all() == pa.Table.from_batches([], schema)


def test_batch_iterable_casts_to_requested_schema():
    batch = pa.record_batch({"id": pa.array([1, 2], pa.int32())})
    schema = pa.schema([("id", pa.int64())])
    reader = _coerce_reader(iter([batch]), schema)
    assert reader.read_all() == pa.table({"id": [1, 2]}, schema=schema)


def test_batch_iterable_rejects_invalid_item():
    schema = pa.schema([("id", pa.int64())])
    reader = _coerce_reader(iter([{"id": 1}]), schema)
    with pytest.raises(TypeError, match="Expected RecordBatch"):
        reader.read_all()


def test_batch_iterable_preserves_cast_error():
    batch = pa.record_batch({"id": ["not an integer"]})
    schema = pa.schema([("id", pa.int64())])
    reader = _coerce_reader(iter([batch]), schema)
    with pytest.raises(ValueError, match="does not match the expected schema"):
        reader.read_all()


def test_unknown_source():
    with pytest.raises(TypeError, match="Unknown data type"):
        coerce_source(object())


@pytest.mark.parametrize("kind", ["union_dataset", "scanner", "one_shot_scanner"])
def test_arrow_sources_remain_one_shot(kind):
    table = pa.table({"id": [1, 2], "value": [10, 20]})
    dataset = pa_ds.dataset(table)
    inputs = {
        "union_dataset": pa_ds.UnionDataset(table.schema, [dataset]),
        "scanner": dataset.scanner(),
        "one_shot_scanner": pa_ds.Scanner.from_batches(table.to_reader()),
    }
    source = coerce_source(inputs[kind])
    assert source.strategy is SourceStrategy.ONE_SHOT
    assert source.reader_factory().read_all() == table


@pytest.mark.parametrize("kind", ["memory", "filesystem"])
@pytest.mark.parametrize("filtered", [False, True])
def test_known_arrow_datasets_preserve_filters(tmp_path, kind, filtered):
    table = pa.table({"id": [0, 1, 2, 3], "value": [0, 10, 20, 30]})
    if kind == "memory":
        # Arrow 21 added RecordBatchReader inputs to InMemoryDataset.
        data = table.to_reader() if int(pa.__version__.split(".")[0]) >= 21 else table
        dataset = pa_ds.InMemoryDataset(data)
        strategy = SourceStrategy.MATERIALIZED
    else:
        pq.write_table(table.slice(0, 2), tmp_path / "first.parquet")
        pq.write_table(table.slice(2, 2), tmp_path / "second.parquet")
        dataset = pa_ds.dataset(tmp_path, format="parquet")
        strategy = SourceStrategy.RESCANNABLE
    expected = table
    if filtered:
        dataset = dataset.filter(pa_ds.field("id") > 0).filter(pa_ds.field("id") != 2)
        expected = table.take([1, 3])
        if kind == "filesystem":
            strategy = SourceStrategy.ONE_SHOT
    source = coerce_source(dataset)
    assert source.strategy is strategy
    assert source.schema == expected.schema
    first_scan = source.reader_factory().read_all()
    assert first_scan.sort_by("id") == expected
    if strategy is not SourceStrategy.ONE_SHOT:
        assert source.reader_factory().read_all() == first_scan

    # A caller-created Scanner remains one-shot, including its projection and
    # its combination of the dataset's filter with an additional scan filter.
    scanner = dataset.scanner(columns=["value"], filter=pa_ds.field("id") != 1)
    source = coerce_source(scanner)
    assert source.strategy is SourceStrategy.ONE_SHOT
    expected = (
        pa.table({"value": [30]}) if filtered else pa.table({"value": [0, 20, 30]})
    )
    assert source.schema == expected.schema
    assert source.reader_factory().read_all().sort_by("value") == expected


def test_arrow_volatile_filter_remains_one_shot(tmp_path):
    table = pa.table({"id": [0, 1, 2, 3]})
    pq.write_table(table, tmp_path / "source.parquet")
    dataset = pa_ds.dataset(tmp_path, format="parquet")
    dataset = dataset.filter(pc.Expression._call("random", []) > 0.5)
    source = coerce_source(dataset)
    assert source.strategy is SourceStrategy.ONE_SHOT
    assert source.schema == table.schema


@pytest.mark.parametrize(
    "source_kind", ["memory", "filesystem", "scanner", "filtered_filesystem"]
)
def test_arrow_source_strategy_on_commit_conflict(tmp_path, monkeypatch, source_kind):
    schema = pa.schema(
        [
            pa.field(
                "id",
                pa.int64(),
                nullable=False,
                metadata={b"lance-schema:unenforced-primary-key": b"true"},
            ),
            pa.field("value", pa.int64()),
        ]
    )
    target = lance.write_dataset(
        pa.table({"id": [0, 1], "value": [0, 0]}, schema=schema),
        tmp_path / "target",
        max_rows_per_file=1,
    )
    builder = (
        target.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
    )
    lance.write_dataset(
        pa.table({"id": [50], "value": [50]}, schema=schema),
        target.uri,
        mode="append",
    )
    table = pa.table({"id": [1, 2], "value": [10, 20]}, schema=schema)
    if source_kind == "memory":
        source = pa_ds.InMemoryDataset(table)
    else:
        paths = [tmp_path / "first.parquet", tmp_path / "second.parquet"]
        for index, path in enumerate(paths):
            pq.write_table(table.slice(index, 1), path)
        source = pa_ds.dataset(paths, format="parquet")
        if source_kind == "scanner":
            source = source.scanner()
        elif source_kind == "filtered_filesystem":
            source = source.filter(pa_ds.field("id") > 1)
    dataset_module = import_module("lance.dataset")
    original_coerce = dataset_module.coerce_source
    readers = []

    def counted_source(data_obj, schema=None):
        registered = original_coerce(data_obj, schema)

        def reader_factory():
            reader = registered.reader_factory()
            readers.append(reader)
            return reader

        return WriteSource(registered.strategy, registered.schema, reader_factory)

    monkeypatch.setattr(dataset_module, "coerce_source", counted_source)
    stats = builder.execute(source)
    assert len(readers) == (2 if source_kind == "filesystem" else 1)
    assert stats["num_updated_rows"] == (
        0 if source_kind == "filtered_filesystem" else 1
    )
    assert stats["num_inserted_rows"] == 1
    assert target.to_table().sort_by("id").to_pydict() == {
        "id": [0, 1, 2, 50],
        "value": [0, 0 if source_kind == "filtered_filesystem" else 10, 20, 50],
    }


@pytest.mark.parametrize("kind", ["dataset", "scanner"])
def test_lance_source_pins_snapshot(tmp_path, kind):
    table = pa.table({"id": [1, 2], "value": [10, 20]})
    dataset = lance.write_dataset(table, tmp_path, max_rows_per_file=1)
    data = dataset if kind == "dataset" else dataset.scanner()
    source = coerce_source(data)
    assert source.strategy is SourceStrategy.RESCANNABLE
    assert source.schema == table.schema
    assert source.reader_factory().read_all() == table

    # Mutate this same Python Dataset object, not just another handle or the URI.
    dataset.update({"value": "value + 100"})
    assert dataset.to_table().sort_by("id")["value"].to_pylist() == [110, 120]
    assert source.reader_factory().read_all() == table


def test_lance_scanner_preserves_projection_and_filter(tmp_path):
    table = pa.table({"id": [1, 2, 3], "value": [10, 20, 30]})
    dataset = lance.write_dataset(table, tmp_path, max_rows_per_file=1)
    scanner = dataset.scanner(columns=["value"], filter="id > 1", batch_size=1)
    source = coerce_source(scanner)
    expected = pa.table({"value": [20, 30]})
    assert source.schema == expected.schema
    for _ in range(2):
        assert source.reader_factory().read_all() == expected


@pytest.mark.parametrize(
    "scan_options, strategy",
    [
        pytest.param({}, SourceStrategy.RESCANNABLE, id="default"),
        pytest.param(
            {"columns": {"value": "id * 2"}, "filter": "id > 1"},
            SourceStrategy.RESCANNABLE,
            id="deterministic-expressions",
        ),
        pytest.param(
            {"columns": {"id": "id", "value": "random()"}},
            SourceStrategy.ONE_SHOT,
            id="volatile-projection",
        ),
        pytest.param(
            {"filter": "now() > TIMESTAMP '2000-01-01 00:00:00'"},
            SourceStrategy.ONE_SHOT,
            id="stable-filter",
        ),
        pytest.param(
            {"filter": "random() > 0.5"},
            SourceStrategy.ONE_SHOT,
            id="volatile-filter",
        ),
        pytest.param(
            {"scan_in_order": False, "limit": 2},
            SourceStrategy.ONE_SHOT,
            id="unordered-limit",
        ),
    ],
)
def test_lance_scanner_repeatability(tmp_path, scan_options, strategy):
    table = pa.table({"id": [1, 2, 3]})
    dataset = lance.write_dataset(table, tmp_path, max_rows_per_file=1)
    source = coerce_source(dataset.scanner(**scan_options))
    assert source.strategy is strategy
    assert source.reader_factory().schema == source.schema


def test_registered_source_works_with_reader_coercion():
    class Snapshot:
        def __init__(self, table):
            self.table = table
            self.reads = 0

        def to_reader(self):
            self.reads += 1
            return self.table.to_reader()

    @coerce_source.register(Snapshot)
    def snapshot_source(snapshot, schema=None):
        return WriteSource(
            SourceStrategy.RESCANNABLE,
            snapshot.table.schema,
            snapshot.to_reader,
        )

    table = pa.table({"id": [1, 2]})
    snapshot = Snapshot(table)
    source = coerce_source(snapshot)
    assert snapshot.reads == 0
    assert source.reader_factory().read_all() == table
    assert source.reader_factory().read_all() == table
    assert _coerce_reader(snapshot).read_all() == table
    assert snapshot.reads == 3


def test_pandas_source_uses_requested_schema():
    pd = pytest.importorskip("pandas")

    class Frame(pd.DataFrame):
        pass

    schema = pa.schema([("id", pa.int32())])
    source = coerce_source(Frame({"id": [1, 2]}), schema)
    assert source.strategy is SourceStrategy.MATERIALIZED
    assert source.reader_factory().read_all().cast(schema) == pa.table(
        {"id": [1, 2]}, schema=schema
    )
    assert source.schema.field("id").type == pa.int32()


def test_polars_source():
    pl = pytest.importorskip("polars")
    table = pa.table({"id": [1, 2]})
    source = coerce_source(pl.from_arrow(table))
    assert source.strategy is SourceStrategy.MATERIALIZED
    assert source.reader_factory().read_all() == table


def test_pydantic_source_keeps_schema_and_strategy():
    pydantic = pytest.importorskip("pydantic")

    class Row(pydantic.BaseModel):
        id: int
        value: Optional[int] = None

    source = coerce_source([Row(id=1), Row(id=2)])
    assert source.strategy is SourceStrategy.ONE_SHOT
    assert source.schema.field("id").nullable is False
    assert source.schema.field("value").type == pa.int64()
    assert source.reader_factory().read_all().to_pydict() == {
        "id": [1, 2],
        "value": [None, None],
    }


def test_huggingface_dataset_and_dict():
    hf = pytest.importorskip("datasets")
    table = pa.table({"id": [1, 2]})
    dataset = hf.Dataset.from_dict(table.to_pydict())
    # DatasetDict is a dict subclass. It must select its own adapter even if
    # this is the first HuggingFace object passed to the registry.
    with pytest.raises(ValueError, match="DatasetDict is not yet supported"):
        coerce_source(hf.DatasetDict({"train": dataset}))
    source = coerce_source(dataset)
    assert source.strategy is SourceStrategy.ONE_SHOT
    assert source.reader_factory().read_all() == table


def test_huggingface_iterable_dataset():
    hf = pytest.importorskip("datasets")
    schema = pa.schema([("id", pa.int64())])

    def rows():
        yield {"id": 1}
        yield {"id": 2}

    dataset = hf.IterableDataset.from_generator(
        rows, features=hf.Features.from_arrow_schema(schema)
    )
    source = coerce_source(dataset)
    assert source.strategy is SourceStrategy.ONE_SHOT
    assert source.reader_factory().read_all() == pa.table({"id": [1, 2]})


@pytest.mark.parametrize("package", ["pandas", "datasets"])
def test_lazy_registration_preserves_explicit_adapter(monkeypatch, package):
    dependency = pytest.importorskip(package)
    table = pa.table({"id": [1, 2]})
    if package == "pandas":
        source_type = dependency.DataFrame
        data = source_type({"id": [1, 2]})
        trigger = dependency.Series([1, 2])
    else:
        source_type = dependency.Dataset
        data = source_type.from_dict({"id": [1, 2]})

        def rows():
            yield {"id": 1}
            yield {"id": 2}

        trigger = dependency.IterableDataset.from_generator(
            rows, features=dependency.Features.from_arrow_schema(table.schema)
        )

    # Exercise first-use registration regardless of test order. Replacing the
    # dispatcher temporarily keeps custom handlers out of other tests, and
    # monkeypatch restores the original registry after this test.
    registry = singledispatch(source_types.coerce_source.__wrapped__)
    for registered_type, adapter in source_types.coerce_source.registry.items():
        if registered_type.__module__.split(".")[0] != package:
            registry.register(registered_type)(adapter)
    monkeypatch.setattr(source_types, "coerce_source", registry)
    expected = WriteSource(SourceStrategy.MATERIALIZED, table.schema, table.to_reader)

    @registry.register(source_type)
    def custom_adapter(data_obj, schema=None):
        return expected

    registry(trigger, table.schema)
    assert registry.dispatch(source_type) is custom_adapter
    assert registry(data) is expected
