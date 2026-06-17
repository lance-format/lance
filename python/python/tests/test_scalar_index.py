# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import json
import os
import random
import re
import shutil
import string
import uuid
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from conftest import ProgressRecorder, progress_event_tags, stage_progress_values
from lance.indices import IndexConfig
from lance.query import (
    BooleanQuery,
    BoostQuery,
    MatchQuery,
    MultiMatchQuery,
    Occur,
    PhraseQuery,
)
from lance.vector import vec_to_table


def create_table(nvec=1000, ndim=128):
    mat = np.random.randn(nvec, ndim)
    price = np.random.rand(nvec) * 100

    def gen_str(n, split="", char_set=string.ascii_letters + string.digits):
        return "".join(random.choices(char_set, k=n))

    meta = np.array([gen_str(100) for _ in range(nvec)])
    doc = [gen_str(10, " ", string.ascii_letters) for _ in range(nvec)]
    tbl = (
        vec_to_table(data=mat)
        .append_column("price", pa.array(price))
        .append_column("meta", pa.array(meta))
        .append_column("doc", pa.array(doc, pa.large_string()))
        .append_column("doc2", pa.array(doc, pa.string()))
        .append_column("id", pa.array(range(nvec)))
    )
    return tbl


def set_language_model_path():
    os.environ["LANCE_LANGUAGE_MODEL_HOME"] = os.path.join(
        os.path.dirname(__file__), "models"
    )


@pytest.fixture()
def lindera_ipadic():
    set_language_model_path()
    model_path = os.path.join(os.path.dirname(__file__), "models", "lindera", "ipadic")
    cwd = os.getcwd()
    try:
        os.chdir(model_path)
        with zipfile.ZipFile("main.zip", "r") as zip_ref:
            zip_ref.extractall()
        os.chdir(cwd)
        yield
    finally:
        shutil.rmtree(os.path.join(model_path, "main"))


@pytest.fixture()
def dataset(tmp_path):
    tbl = create_table()
    yield lance.write_dataset(tbl, tmp_path)


@pytest.fixture()
def indexed_dataset(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=2,
    )
    dataset.create_scalar_index("meta", index_type="BTREE")
    return dataset


@pytest.fixture()
def data_table(indexed_dataset: lance.LanceDataset):
    return indexed_dataset.scanner().to_table()


def _commit_segmented_btree_index(dataset, column, index_name):
    segments = [
        dataset.create_index_uncommitted(
            column=column,
            index_type="BTREE",
            name=index_name,
            fragment_ids=[fragment.fragment_id],
        )
        for fragment in dataset.get_fragments()
    ]
    return dataset.commit_existing_index_segments(index_name, column, segments)


def test_create_scalar_index_rejects_invalid_uuid(tmp_path):
    """Invalid UUID strings passed to create_scalar_index and merge_index_metadata
    must surface as a Python ValueError at the FFI boundary."""
    data = pa.table({"id": pa.array(range(100), type=pa.int64())})
    dataset = lance.write_dataset(data, tmp_path / "ds")

    with pytest.raises(ValueError, match="Invalid UUID"):
        dataset.create_scalar_index(
            column="id",
            index_type="BTREE",
            index_uuid="not-a-uuid",
        )

    with pytest.raises(ValueError, match="Invalid UUID"):
        dataset.merge_index_metadata("also-not-a-uuid", index_type="BTREE")


@pytest.fixture
def btree_comparison_datasets(tmp_path):
    """Setup datasets for B-tree comparison tests"""
    num_fragments = 3
    rows_per_fragment = 10000
    total_rows = num_fragments * rows_per_fragment

    fragment_ds = generate_multi_fragment_dataset(
        tmp_path / "fragment",
        num_fragments=num_fragments,
        rows_per_fragment=rows_per_fragment,
    )

    complete_ds = generate_multi_fragment_dataset(
        tmp_path / "complete",
        num_fragments=num_fragments,
        rows_per_fragment=rows_per_fragment,
    )

    fragment_ds_committed = _commit_segmented_btree_index(
        fragment_ds, "id", "fragment_btree_precise_test"
    )

    complete_index_name = f"complete_btree_{uuid.uuid4().hex[:8]}"
    complete_ds.create_scalar_index(
        column="id",
        index_type="BTREE",
        name=complete_index_name,
    )
    complete_ds = lance.dataset(complete_ds.uri)

    return {
        "fragment_ds": fragment_ds_committed,
        "complete_ds": complete_ds,
        "rows_per_fragment": rows_per_fragment,
        "total_rows": total_rows,
    }


def test_describe_indices_vector_and_scalar(indexed_dataset: lance.LanceDataset):
    indices = indexed_dataset.describe_indices()
    vec_idx = next(idx for idx in indices if "VectorIndex" in idx.type_url)
    scalar_idx = next(idx for idx in indices if idx.index_type == "BTree")
    assert vec_idx is not None
    assert scalar_idx is not None


def test_list_indices_characterization(indexed_dataset: lance.LanceDataset):
    """Lock down the backwards-compatible shape of the deprecated list_indices().

    list_indices() returns a list of plain dicts (one per index segment), not
    Index dataclasses. This characterization test guards the dict keys and
    values so the deprecated method stays backwards compatible.
    """
    with pytest.warns(DeprecationWarning):
        indices = indexed_dataset.list_indices()

    assert len(indices) == 2
    by_name = {idx["name"]: idx for idx in indices}
    assert set(by_name) == {"vector_idx", "meta_idx"}

    expected_keys = {
        "name",
        "type",
        "uuid",
        "fields",
        "version",
        "fragment_ids",
        "base_id",
    }
    for idx in indices:
        assert set(idx) == expected_keys
        assert isinstance(idx["uuid"], str) and len(idx["uuid"]) > 0
        assert isinstance(idx["fields"], list)
        assert isinstance(idx["fragment_ids"], set)
        assert isinstance(idx["version"], int)
        assert idx["type"] != "Unknown"
        assert idx["base_id"] is None

    vector_idx = by_name["vector_idx"]
    assert vector_idx["type"] == "IVF_PQ"
    assert vector_idx["fields"] == ["vector"]
    assert vector_idx["fragment_ids"] == {0}

    meta_idx = by_name["meta_idx"]
    assert meta_idx["type"] == "BTree"
    assert meta_idx["fields"] == ["meta"]
    assert meta_idx["fragment_ids"] == {0}


def test_list_indices_nested_field_path(tmp_path):
    """list_indices() reports nested fields as full dotted paths."""
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("meta", pa.struct([pa.field("lang", pa.string())])),
        ]
    )
    data = pa.table(
        {
            "id": [1, 2, 3],
            "meta": [{"lang": "en"}, {"lang": "fr"}, {"lang": "en"}],
        },
        schema=schema,
    )
    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index(column="meta.lang", index_type="BTREE")

    with pytest.warns(DeprecationWarning):
        indices = ds.list_indices()

    assert len(indices) == 1
    assert indices[0]["fields"] == ["meta.lang"]


def _commit_index(ds, index):
    """Commit a single raw Index entry via the CreateIndex operation."""
    return lance.LanceDataset.commit(
        ds.uri,
        lance.LanceOperation.CreateIndex(new_indices=[index], removed_indices=[]),
        read_version=ds.version,
    )


def test_list_indices_index_without_details(tmp_path):
    """An index whose manifest entry has no index details (e.g. committed by an
    older writer) is still reported on a best-effort basis: describe_indices()
    does not error, and the type is reported as "Unknown"."""
    from lance.dataset import Index

    data = pa.table({"id": range(100), "val": range(100)})
    ds = lance.write_dataset(data, tmp_path)

    field_id = ds.schema.get_field_index("id")
    fragment_ids = {f.fragment_id for f in ds.get_fragments()}
    ds = _commit_index(
        ds,
        Index(
            uuid=str(uuid.uuid4()),
            name="legacy_idx",
            fields=[field_id],
            dataset_version=ds.version,
            fragment_ids=fragment_ids,
            index_version=0,
        ),
    )

    described = ds.describe_indices()
    assert len(described) == 1
    assert described[0].name == "legacy_idx"
    assert described[0].index_type == "Unknown"
    assert described[0].type_url == ""

    with pytest.warns(DeprecationWarning):
        listed = ds.list_indices()
    assert len(listed) == 1
    assert listed[0]["name"] == "legacy_idx"
    assert listed[0]["type"] == "Unknown"


def test_list_indices_legacy_vector_index_without_details(tmp_path):
    """A legacy vector index predates VectorIndexDetails: it has no index
    details but stores a monolithic index file. Its type is recognized as
    "Vector" from the index file rather than reported as "Unknown"."""
    from lance.dataset import Index, IndexFile

    data = pa.table({"id": range(100), "val": range(100)})
    ds = lance.write_dataset(data, tmp_path)

    field_id = ds.schema.get_field_index("id")
    fragment_ids = {f.fragment_id for f in ds.get_fragments()}
    ds = _commit_index(
        ds,
        Index(
            uuid=str(uuid.uuid4()),
            name="legacy_vector_idx",
            fields=[field_id],
            dataset_version=ds.version,
            fragment_ids=fragment_ids,
            index_version=0,
            # "index.idx" is the legacy monolithic index file name; its presence
            # is how a pre-details vector index is recognized.
            files=[IndexFile(path="index.idx", size_bytes=0)],
        ),
    )

    described = ds.describe_indices()
    assert len(described) == 1
    assert described[0].index_type == "Vector"

    with pytest.warns(DeprecationWarning):
        listed = ds.list_indices()
    assert listed[0]["type"] == "Vector"


def test_indexed_scalar_scan(indexed_dataset: lance.LanceDataset, data_table: pa.Table):
    sample_meta = data_table["meta"][50]
    expected_price = data_table["price"][50]

    for filter in [f"meta='{sample_meta}'", f"price >= 0 AND meta='{sample_meta}'"]:
        scanner = indexed_dataset.scanner(
            columns=["price"], filter=filter, prefilter=True
        )

        assert (
            f"ScalarIndexQuery: query=[meta = {sample_meta}]@meta_idx"
            in scanner.explain_plan()
        )

        actual_data = scanner.to_table()
        assert actual_data.num_rows == 1
        assert actual_data.num_columns == 1

        actual_price = actual_data["price"][0]
        assert actual_price == expected_price


def test_indexed_between(tmp_path):
    dataset = lance.write_dataset(pa.table({"val": range(0, 10000)}), tmp_path)
    dataset.create_scalar_index("val", index_type="BTREE")

    scanner = dataset.scanner(filter="val BETWEEN 10 AND 20", prefilter=True)

    assert (
        "ScalarIndexQuery: query=[val >= 10 && val <= 20]@val_idx"
        in scanner.explain_plan()
    )

    actual_data = scanner.to_table()
    assert actual_data.num_rows == 11

    scanner = dataset.scanner(filter="val >= 10 AND val <= 20", prefilter=True)

    assert (
        "ScalarIndexQuery: query=[val >= 10 && val <= 20]@val_idx"
        in scanner.explain_plan()
    )

    actual_data = scanner.to_table()
    assert actual_data.num_rows == 11

    # The following cases are slightly ill-formed since end is before start
    # but we should handle them gracefully and simply return an empty result
    # (previously we panicked here)
    scanner = dataset.scanner(filter="val >= 5000 AND val <= 0", prefilter=True)

    assert (
        "ScalarIndexQuery: query=[val >= 5000 && val <= 0]@val_idx"
        in scanner.explain_plan()
    )

    actual_data = scanner.to_table()
    assert actual_data.num_rows == 0

    scanner = dataset.scanner(filter="val BETWEEN 5000 AND 0", prefilter=True)

    assert (
        "ScalarIndexQuery: query=[val >= 5000 && val <= 0]@val_idx"
        in scanner.explain_plan()
    )

    actual_data = scanner.to_table()
    assert actual_data.num_rows == 0


def test_create_inverted_index_progress_callback(tmp_path):
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=2, rows_per_fragment=75
    )
    progress_recorder = ProgressRecorder()

    ds.create_scalar_index(
        column="text",
        index_type="INVERTED",
        remove_stop_words=False,
        progress_callback=progress_recorder,
    )

    tags = progress_event_tags(progress_recorder.events)
    expected_order = [
        "start:load_data",
        "complete:load_data",
        "start:tokenize_docs",
        "complete:tokenize_docs",
        "start:copy_partitions",
        "complete:copy_partitions",
        "start:write_metadata",
        "complete:write_metadata",
    ]
    positions = [tags.index(tag) for tag in expected_order]
    assert positions == sorted(positions)

    tokenize_progress = stage_progress_values(progress_recorder.events, "tokenize_docs")
    assert tokenize_progress
    assert max(tokenize_progress) == ds.count_rows()

    assert "progress:copy_partitions" in tags
    assert "progress:write_metadata" in tags
    assert "start:merge_partitions" not in tags


def test_merge_index_metadata_inverted_index_progress_callback(tmp_path):
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=3, rows_per_fragment=60
    )

    inverted_index_id = str(uuid.uuid4())
    for fragment in ds.get_fragments():
        ds.create_scalar_index(
            column="text",
            index_type="INVERTED",
            name="text_inverted_progress_idx",
            replace=False,
            index_uuid=inverted_index_id,
            fragment_ids=[fragment.fragment_id],
            remove_stop_words=False,
        )

    progress_recorder = ProgressRecorder()
    ds.merge_index_metadata(
        inverted_index_id,
        index_type="INVERTED",
        progress_callback=progress_recorder,
    )

    tags = progress_event_tags(progress_recorder.events)
    expected_order = [
        "start:read_partition_metadata",
        "complete:read_partition_metadata",
        "start:remap_partition_files",
        "complete:remap_partition_files",
        "start:write_merged_metadata",
        "complete:write_merged_metadata",
    ]
    positions = [tags.index(tag) for tag in expected_order]
    assert positions == sorted(positions)

    metadata_progress = stage_progress_values(
        progress_recorder.events, "read_partition_metadata"
    )
    assert metadata_progress
    assert metadata_progress[-1] == len(ds.get_fragments())
    assert "progress:remap_partition_files" in tags
    assert "progress:write_merged_metadata" in tags


def test_create_inverted_index_progress_callback_error_after_completion_is_ignored(
    tmp_path,
):
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=2, rows_per_fragment=75
    )
    progress_recorder = ProgressRecorder(fail_on_tag="complete:write_metadata")

    ds.create_scalar_index(
        column="text",
        index_type="INVERTED",
        remove_stop_words=False,
        progress_callback=progress_recorder,
    )

    tags = progress_event_tags(progress_recorder.events)
    assert tags[-1] == "complete:write_metadata"
    assert any(idx.index_type == "Inverted" for idx in ds.describe_indices())


def test_index_combination(tmp_path):
    # This test regresses a bug in the index combination logic.

    colors = ["red", "green", "blue"]
    digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    data = [{"color": colors[i % 3], "digit": digits[i % 10]} for i in range(30000)]

    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("color", index_type="BTREE")
    ds.create_scalar_index("digit", index_type="BTREE")

    assert ds.count_rows("color = 'green' or digit <> 9") == 28000


def test_temporal_index(tmp_path):
    # Timestamps
    now = datetime.now()
    today = date.today()
    table = pa.Table.from_pydict(
        {
            "ts": [now - timedelta(days=i) for i in range(100)],
            "date": [today - timedelta(days=i) for i in range(100)],
            "time": pa.array([i for i in range(100)], type=pa.time32("s")),
            "id": [i for i in range(100)],
        }
    )
    dataset = lance.write_dataset(table, tmp_path)
    dataset.create_scalar_index("ts", index_type="BTREE")
    dataset.create_scalar_index("date", index_type="BTREE")
    dataset.create_scalar_index("time", index_type="BTREE")

    # Timestamp
    half_now = now - timedelta(days=50)
    scanner = dataset.scanner(filter=f"ts > timestamp '{half_now}'", scan_in_order=True)
    assert re.search(
        r"^.*ScalarIndexQuery: query=\[ts > .*\]@ts_idx.*$",
        scanner.explain_plan(True),
        re.MULTILINE,
    )
    assert scanner.to_table() == table.slice(0, 50)

    # Date
    half_toady = today - timedelta(days=50)
    scanner = dataset.scanner(filter=f"date > date '{half_toady}'", scan_in_order=True)
    assert re.search(
        r"^.*ScalarIndexQuery: query=\[date > .*\]@date_idx.*$",
        scanner.explain_plan(True),
        re.MULTILINE,
    )
    assert scanner.to_table() == table.slice(0, 50)


def test_indexed_vector_scan(indexed_dataset: lance.LanceDataset, data_table: pa.Table):
    # We query for row 25 but our prefilter only
    # considers row 50 so we should get row 50
    query_vec = data_table["vector"][25].as_py()
    sample_meta = data_table["meta"][50]
    expected_price = data_table["price"][50]

    def check_result(table: pa.Table):
        assert table.num_rows == 1
        assert table.num_columns == 1

        actual_price = table["price"][0]
        assert actual_price == expected_price

    scanner = indexed_dataset.scanner(
        nearest={"column": "vector", "q": query_vec, "k": 5, "nprobes": 4},
        columns=["price"],
        prefilter=True,
        filter=f"meta='{sample_meta}'",
    )

    assert "ScalarIndexQuery" in scanner.analyze_plan()

    check_result(scanner.to_table())

    scanner = indexed_dataset.scanner(
        nearest={"column": "vector", "q": query_vec, "k": 5, "nprobes": 4},
        columns=["price"],
        prefilter=True,
        filter=f"price >= 0 AND meta='{sample_meta}'",
    )

    assert (
        f"ScalarIndexQuery: query=[meta = {sample_meta}]@meta_idx"
        in scanner.explain_plan()
    )

    check_result(scanner.to_table())


def test_partly_indexed_prefiltered_search(tmp_path):
    # Regresses a case where the vector index is ahead of a scalar index.  The scalar
    # index wants to be used as a prefilter but we have to make sure to scan the
    # unindexed fragments and feed those into the prefilter

    # Create initial dataset
    table = pa.table(
        {
            "vec": pa.array([[i, i] for i in range(1000)], pa.list_(pa.float32(), 2)),
            "text": ["book" for _ in range(1000)],
            "id": range(1000),
        }
    )
    ds = lance.write_dataset(table, tmp_path)
    ds = ds.create_index(
        "vec",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=2,
    )
    ds.create_scalar_index("id", "BTREE")
    ds.create_scalar_index("text", index_type="INVERTED", with_position=False)

    def make_vec_search(ds):
        return ds.scanner(
            nearest={"column": "vec", "q": [5, 5], "k": 1000},
            prefilter=True,
            filter="id in (5, 10, 15, 20, 25, 30)",
        )

    def make_fts_search(ds):
        return ds.scanner(
            full_text_query="book",
            prefilter=True,
            filter="id in (5, 10, 15, 20, 25, 30)",
        )

    # Sanity test, no new data, should get 6 results
    plan = make_vec_search(ds).explain_plan()
    assert "ScalarIndexQuery" in plan
    assert "KNNVectorDistance" not in plan
    assert "LanceRead" not in plan
    assert make_vec_search(ds).to_table().num_rows == 6

    plan = make_fts_search(ds).explain_plan()
    assert "ScalarIndexQuery" in plan
    assert "KNNVectorDistance" not in plan
    assert "LanceRead" not in plan
    assert make_fts_search(ds).to_table().num_rows == 6

    # Add new data (including 6 more results)
    ds.insert(table)

    # Basic ann combined search, should get 12 results
    plan = make_vec_search(ds).explain_plan()
    assert "ScalarIndexQuery" in plan
    assert "MaterializeIndex" not in plan
    assert "KNNVectorDistance" in plan
    assert "LanceScan" in plan
    assert make_vec_search(ds).to_table().num_rows == 12

    plan = make_fts_search(ds).explain_plan()
    assert "ScalarIndexQuery" in plan
    assert "MaterializeIndex" not in plan
    assert "FlatMatchQuery" in plan
    assert "LanceScan" in plan
    assert make_fts_search(ds).to_table().num_rows == 12

    # Update vector index but NOT scalar index
    ds.optimize.optimize_indices(index_names=["vec_idx", "text_idx"])

    # Ann search but with combined prefilter, should get 12 results
    plan = make_vec_search(ds).explain_plan()
    assert "ScalarIndexQuery" in plan
    assert "LanceRead" in plan
    assert "KNNVectorDistance" not in plan
    assert "LanceScan" not in plan
    assert make_vec_search(ds).to_table().num_rows == 12

    plan = make_fts_search(ds).explain_plan()
    assert "ScalarIndexQuery" in plan
    assert "LanceRead" in plan
    assert "FlatMatchQuery" not in plan
    assert "LanceScan" not in plan
    assert make_fts_search(ds).to_table().num_rows == 12


# Post filtering does not use scalar indices.  This test merely confirms
# that post filtering is still being applied even if the query could be
# satisfied with scalar indices
def test_indexed_vector_scan_postfilter(
    indexed_dataset: lance.LanceDataset, data_table: pa.Table
):
    query_vec = data_table["vector"][25].as_py()
    sample_meta = data_table["meta"][50]

    scanner = indexed_dataset.scanner(
        nearest={"column": "vector", "q": query_vec, "k": 1, "nprobes": 4},
        columns=["price"],
        prefilter=False,
        filter=f"meta='{sample_meta}'",
    )

    assert scanner.to_table().num_rows == 0


def test_fixed_size_binary(tmp_path):
    arr = pa.array([b"0123012301230123", b"2345234523452345"], pa.uuid())

    ds = lance.write_dataset(pa.table({"uuid": arr}), tmp_path)

    ds.create_scalar_index("uuid", "BTREE")

    query = (
        "uuid = arrow_cast(0x32333435323334353233343532333435, 'FixedSizeBinary(16)')"
    )
    assert (
        "ScalarIndexQuery: query=[uuid = 32333435323334353233...]@uuid_idx"
        in ds.scanner(filter=query).explain_plan()
    )

    table = ds.scanner(filter=query).to_table()
    assert table.num_rows == 1
    assert table.column("uuid").to_pylist() == arr.slice(1, 1).to_pylist()


def test_index_take_batch_size(tmp_path):
    dataset = lance.write_dataset(
        pa.table({"ints": range(1024)}), tmp_path, max_rows_per_file=100
    )
    dataset.create_scalar_index("ints", index_type="BTREE")
    batches = dataset.scanner(
        with_row_id=True, filter="ints > 0", batch_size=50
    ).to_batches()
    batches = list(batches)
    assert len(batches) == 21

    dataset = lance.write_dataset(
        pa.table({"strings": [f"string-{i}" for i in range(1024)]}),
        tmp_path,
        max_rows_per_file=100,
        mode="overwrite",
    )
    dataset.create_scalar_index("strings", index_type="NGRAM")
    filter = "contains(strings, 'ing')"
    batches = dataset.scanner(
        with_row_id=True, filter=filter, batch_size=50, limit=1024
    ).to_batches()
    batches = list(batches)
    assert len(batches) == 21


def test_all_null_chunk(tmp_path):
    def gen_string(idx: int):
        if idx % 2 == 0:
            return None
        return f"string-{idx}"

    strings = pa.array([gen_string(i) for i in range(100 * 1024)], pa.string())
    table = pa.Table.from_arrays([strings], ["str"])
    dataset = lance.write_dataset(table, tmp_path)
    dataset.create_scalar_index("str", index_type="BTREE")
    scanner = dataset.scanner(prefilter=True, filter="str='string-501'")

    assert scanner.to_table().num_rows == 1


# We currently allow the memory pool default size to be configured with an
# environment variable.  This test ensures that the environment variable
# is respected.
def test_lance_mem_pool_env_var(tmp_path):
    ints = pa.array([i * 10 for i in range(100 * 1024)])
    table = pa.Table.from_arrays([ints], ["int"])
    dataset = lance.write_dataset(table, tmp_path)

    # Should succeed
    dataset.create_scalar_index("int", index_type="BTREE")

    try:
        # Should fail if we intentionally use a very small memory pool
        os.environ["LANCE_MEM_POOL_SIZE"] = "1024"
        with pytest.raises(Exception):
            dataset.create_scalar_index("int", index_type="BTREE", replace=True)

        # Should succeed again since bypassing spilling takes precedence
        os.environ["LANCE_BYPASS_SPILLING"] = "1"
        dataset.create_scalar_index("int", index_type="BTREE", replace=True)
    finally:
        del os.environ["LANCE_MEM_POOL_SIZE"]
        if "LANCE_BYPASS_SPILLING" in os.environ:
            del os.environ["LANCE_BYPASS_SPILLING"]


@pytest.mark.parametrize("with_position", [True, False])
@pytest.mark.parametrize("base_tokenizer", ["simple", "icu"])
def test_full_text_search(dataset, with_position, base_tokenizer):
    dataset.create_scalar_index(
        "doc",
        index_type="INVERTED",
        with_position=with_position,
        base_tokenizer=base_tokenizer,
    )
    row = dataset.take(indices=[0], columns=["doc"])
    query = row.column(0)[0].as_py()
    query = query.split(" ")[0]
    results = dataset.scanner(
        columns=["doc"],
        full_text_query=query,
    ).to_table()
    assert results.num_rows > 0
    results = results.column(0)
    for row in results:
        assert query in row.as_py()

    with pytest.raises(ValueError, match="Cannot include deleted rows"):
        dataset.to_table(
            with_row_id=True, full_text_query=query, include_deleted_rows=True
        )


def test_unindexed_full_text_search_on_empty_index(tmp_path):
    # Create fts index on empty table.
    schema = pa.schema({"text": pa.string()})
    ds = lance.write_dataset(pa.Table.from_pylist([], schema=schema), tmp_path)
    ds.create_scalar_index("text", "INVERTED")

    # Append unindexed data.
    ds.insert(pa.Table.from_pylist([{"text": "hello!"}], schema=schema))

    # Fts search.
    results = ds.scanner(
        columns=["text"],
        full_text_query="hello",
    ).to_table()
    assert results.num_rows == 1


def test_full_text_search_without_index(dataset):
    row = dataset.take(indices=[0], columns=["doc"])
    query_text = row.column(0)[0].as_py()
    query_text = query_text.split(" ")[0]
    query = MatchQuery(query_text, column="doc")
    results = dataset.scanner(
        columns=["doc"],
        full_text_query=query,
    ).to_table()
    assert results.num_rows > 0
    results = results.column(0)
    for row in results:
        assert query_text in row.as_py()


def test_fts_custom_stop_words(tmp_path):
    # Prepare dataset
    set_language_model_path()
    data = pa.table(
        {
            "text": ["他们拿着苹果手机", "他们穿着耐克阿迪"],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index(
        "text",
        "INVERTED",
        base_tokenizer="jieba/default",
        remove_stop_words=True,
        custom_stop_words=["他们"],
    )

    # Search
    results = ds.to_table(
        full_text_query="他们",
        prefilter=True,
        with_row_id=True,
    )
    assert len(results["_rowid"].to_pylist()) == 0

    results = ds.to_table(
        full_text_query="手机",
        prefilter=True,
        with_row_id=True,
    )
    assert len(results["_rowid"].to_pylist()) == 1


def test_rowid_order(dataset):
    dataset.create_scalar_index("doc", index_type="INVERTED", with_position=False)
    results = dataset.scanner(
        columns=["doc", "_score"],
        full_text_query="hello",
        with_row_id=True,
    ).to_table()

    assert results.schema[0].name == "doc"
    assert results.schema[1].name == "_score"
    assert results.schema[2].name == "_rowid"


def test_filter_with_fts_index(dataset):
    dataset.create_scalar_index("doc", index_type="INVERTED", with_position=False)
    row = dataset.take(indices=[0], columns=["doc"])
    query = row.column(0)[0].as_py()
    query = query.split(" ")[0]
    results = dataset.scanner(
        filter=f"doc = '{query}'",
        prefilter=True,
    ).to_table()
    assert results.num_rows > 0
    results = results["doc"]
    for row in results:
        assert query == row.as_py()


def test_create_scalar_index_fts_alias(dataset):
    dataset.create_scalar_index("doc", index_type="FTS", with_position=False)
    assert any(idx.index_type == "Inverted" for idx in dataset.describe_indices())


def test_multi_index_create(tmp_path):
    dataset = lance.write_dataset(
        pa.table({"ints": range(1024)}), tmp_path, max_rows_per_file=100
    )
    dataset.create_scalar_index("ints", index_type="BTREE")
    dataset.create_scalar_index(
        "ints", index_type="BITMAP", name="ints_bitmap_idx", replace=True
    )

    indices = dataset.describe_indices()
    assert len(indices) == 2

    idx_by_name = {idx.name: idx for idx in indices}
    assert idx_by_name["ints_idx"].index_type == "BTree"
    assert idx_by_name["ints_bitmap_idx"].index_type == "Bitmap"

    # Test that we can drop one of the indices
    dataset.drop_index("ints_idx")
    indices = dataset.describe_indices()
    assert len(indices) == 1
    assert indices[0].name == "ints_bitmap_idx"
    assert indices[0].index_type == "Bitmap"

    # Test that we can drop the last index
    dataset.drop_index("ints_bitmap_idx")
    indices = dataset.describe_indices()
    assert len(indices) == 0


def test_use_multi_index(tmp_path):
    dataset = lance.write_dataset(
        pa.table({"ints": range(1024)}), tmp_path, max_rows_per_file=100
    )
    dataset.create_scalar_index("ints", index_type="BTREE")
    dataset.create_scalar_index("ints", index_type="BITMAP", name="ints_bitmap_idx")

    # Test that we can use the index.  Multiple indices can be applied here.
    # One of them will be chosen (it is not deterministic which one is chosen)
    results = dataset.to_table(filter="ints = 0", prefilter=True)
    assert results.num_rows == 1

    assert (
        "ScalarIndexQuery: query=[ints = 0]@ints_idx"
        in dataset.scanner(filter="ints = 0", prefilter=True).explain_plan()
    )


def test_ngram_fts(tmp_path):
    dataset = lance.write_dataset(
        pa.table({"text": ["hello", "world", "hello world"]}),
        tmp_path,
    )
    dataset.create_scalar_index("text", index_type="INVERTED")
    dataset.create_scalar_index("text", name="text_ngram_idx", index_type="NGRAM")

    results = dataset.to_table(full_text_query="hello")
    assert results.num_rows == 2

    results = dataset.to_table(filter="contains(text, 'hello')")
    assert results.num_rows == 2

    assert (
        'ScalarIndexQuery: query=[contains(text, Utf8("hello"))]@text_ngram_idx'
        in dataset.scanner(
            filter="contains(text, 'hello')", prefilter=True
        ).explain_plan()
    )


@pytest.mark.parametrize("base_tokenizer", ["simple", "icu"])
def test_fts_fts(tmp_path, base_tokenizer):
    # Tests creating two FTS indices with the same name but different parameters
    dataset = lance.write_dataset(
        pa.table(
            {
                "text": [
                    "Frodo was a puppy",
                    "Frodo was a happy puppy",
                    "Frodo was a very happy puppy",
                ]
            }
        ),
        tmp_path,
    )
    dataset.create_scalar_index(
        "text",
        "INVERTED",
        with_position=True,
        remove_stop_words=False,
        base_tokenizer=base_tokenizer,
    )

    results = dataset.to_table(full_text_query='"was a puppy"', prefilter=True)
    assert results.num_rows == 1

    results = dataset.to_table(full_text_query="was a puppy", prefilter=True)
    assert results.num_rows == 3

    dataset.create_scalar_index(
        "text",
        "INVERTED",
        name="no_pos_idx",
        with_position=False,
        base_tokenizer=base_tokenizer,
    )

    # There is no way to currently specify which index to use.  Instead
    # it will always use the first index in the manifest.

    results = dataset.to_table(full_text_query='"was a puppy"', prefilter=True)
    assert results.num_rows == 1

    results = dataset.to_table(full_text_query="was a puppy", prefilter=True)
    assert results.num_rows == 3


def test_indexed_filter_with_fts_index(tmp_path):
    data = pa.table(
        {
            "text": [
                "Frodo was a puppy",
                "There were several kittens playing",
                "Frodo was a happy puppy",
                "Frodo was a very happy puppy",
            ],
            "sentiment": ["neutral", "neutral", "positive", "positive"],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED")
    ds.create_scalar_index("sentiment", "BITMAP")

    # append more data to test flat FTS
    data = pa.table(
        {
            "text": ["flat", "search"],
            "sentiment": ["positive", "positive"],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="append")

    results = ds.to_table(
        full_text_query="puppy",
        filter="sentiment='positive'",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [2, 3]


def test_fts_ngram_tokenizer(tmp_path):
    data = pa.table({"text": ["hello world", "lance database", "lance is cool"]})
    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", index_type="INVERTED", base_tokenizer="ngram")

    results = ds.to_table(full_text_query="lan")
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {"lance database", "lance is cool"}

    results = ds.to_table(full_text_query="nce")  # spellchecker:disable-line
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {"lance database", "lance is cool"}

    # the default min_ngram_length is 3, so "la" should not match
    results = ds.to_table(full_text_query="la")
    assert results.num_rows == 0

    # test setting min_ngram_length and prefix_only
    ds.create_scalar_index(
        "text",
        index_type="INVERTED",
        base_tokenizer="ngram",
        min_ngram_length=2,
        prefix_only=True,
    )

    results = ds.to_table(full_text_query="lan")
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {"lance database", "lance is cool"}

    results = ds.to_table(full_text_query="nce")  # spellchecker:disable-line
    assert results.num_rows == 0

    results = ds.to_table(full_text_query="la")
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {"lance database", "lance is cool"}


def test_fts_stats(dataset):
    dataset.create_scalar_index(
        "doc",
        index_type="INVERTED",
        with_position=False,
        remove_stop_words=True,
        memory_limit=4096,
        num_workers=2,
    )
    stats = dataset.stats.index_stats("doc_idx")
    assert stats["index_type"] == "Inverted"
    stats = stats["indices"][0]
    params = stats["params"]

    assert params["with_position"] is False
    assert params["base_tokenizer"] == "simple"
    assert params["language"] == "English"
    assert params["max_token_length"] == 40
    assert params["lower_case"] is True
    assert params["stem"] is True
    assert params["remove_stop_words"] is True
    assert params["ascii_folding"] is True
    assert "memory_limit" not in params
    assert "num_workers" not in params


def test_fts_optimize_num_indices_to_merge(tmp_path):
    def append_rows(rows):
        return lance.write_dataset(pa.table(rows), tmp_path, mode="append")

    def num_indices(ds):
        return ds.stats.index_stats("text_idx")["num_indices"]

    ds = lance.write_dataset(
        pa.table(
            {
                "id": [0, 1],
                "text": ["alpha base phrase", "beta base phrase"],
            }
        ),
        tmp_path,
    )
    ds.create_scalar_index("text", index_type="INVERTED", with_position=True)
    assert num_indices(ds) == 1

    ds = append_rows({"id": [2], "text": ["gamma delta phrase"]})
    ds.optimize.optimize_indices(num_indices_to_merge=0)
    assert num_indices(ds) == 2

    ds = append_rows({"id": [3], "text": ["epsilon zeta phrase"]})
    ds.optimize.optimize_indices(num_indices_to_merge=1)
    assert num_indices(ds) == 2
    assert ds.to_table(full_text_query="epsilon")["id"].to_pylist() == [3]

    ds = append_rows({"id": [4], "text": ["eta theta phrase"]})
    ds.optimize.optimize_indices(num_indices_to_merge=0)
    assert num_indices(ds) == 3

    ds.optimize.optimize_indices(num_indices_to_merge=2)
    assert num_indices(ds) == 2
    assert ds.to_table(full_text_query=PhraseQuery("eta theta", "text"))[
        "id"
    ].to_pylist() == [4]

    ds = append_rows({"id": [5], "text": ["iota kappa phrase"]})
    ds.optimize.optimize_indices(num_indices_to_merge=0)
    assert num_indices(ds) == 3

    ds.optimize.optimize_indices(num_indices_to_merge=10)
    assert num_indices(ds) == 1
    assert ds.to_table(full_text_query="alpha")["id"].to_pylist() == [0]
    assert ds.to_table(full_text_query="iota")["id"].to_pylist() == [5]


@pytest.mark.parametrize("base_tokenizer", ["simple", "icu"])
def test_fts_score(tmp_path, base_tokenizer):
    # the number of tokens matters for scoring,
    # make a table that all docs have the same number of tokens
    data = pa.table(
        {
            "id": [1, 2, 3],
            "text": ["lance database test", "full text search", "lance search text"],
        }
    )
    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", "INVERTED", base_tokenizer=base_tokenizer)

    results = ds.to_table(full_text_query="lance search text")
    assert results.num_rows == 3
    assert results["id"].to_pylist() == [3, 2, 1]

    # Assert distributed FTS produces the same BM25 scores as single-machine
    assert_distributed_fts_consistency(
        data,
        "text",
        "lance search text",
        tmp_path,
        index_params={"with_position": False, "base_tokenizer": base_tokenizer},
    )


def test_fts_with_filter(tmp_path):
    data = pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "text": [
                "lance database test",
                "full text search",
                "lance search text",
                "some other content",
                "some other content",
                "some other content",
                "some other content",
                "some other content",
                "some other content",
                "some other content",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("id", "BTREE")
    ds.create_scalar_index("text", "INVERTED")

    results = ds.to_table(full_text_query="lance search text")
    assert results.num_rows == 3
    assert results["id"].to_pylist() == [3, 2, 1]

    score_id1 = results.column("_score")[2].as_py()

    results = ds.to_table(
        full_text_query="lance search text",
        filter="id <= 1",
        prefilter=True,
    )
    assert results.num_rows == 1
    assert results["id"].to_pylist() == [1]
    assert results.column("_score")[0].as_py() == score_id1

    plan = ds.scanner(
        full_text_query="lance search text", filter="id <= 1", prefilter=True
    ).analyze_plan()
    assert "index_comparisons=1" in plan

    # Assert distributed FTS with prefilter produces consistent results
    assert_distributed_fts_consistency(
        data,
        "text",
        "lance search text",
        tmp_path,
        index_params={"with_position": False},
        query_params={"filter": "id <= 1", "prefilter": True},
    )


def test_fts_on_list(tmp_path):
    data = pa.table(
        {
            "text": [
                ["lance database", "the", "search"],
                ["lance database"],
                ["lance", "search"],
                ["database", "search"],
                ["unrelated", "doc"],
            ]
        }
    )
    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", "INVERTED", with_position=True)

    results = ds.to_table(full_text_query="lance")
    assert results.num_rows == 3

    results = ds.to_table(full_text_query=PhraseQuery("lance database", "text"))
    assert results.num_rows == 2

    # Append new data without fts index, then query.
    ds.insert(data)
    results = ds.to_table(full_text_query="lance")
    assert results.num_rows == 6


def test_fts_fuzzy_query(tmp_path):
    data = pa.table(
        {
            "text": [
                "fa",
                "fo",  # spellchecker:disable-line
                "fob",
                "focus",
                "foo",
                "food",
                "foul",
            ]
        }
    )

    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", "INVERTED")

    results = ds.to_table(
        full_text_query=MatchQuery("foo", "text", fuzziness=1),
    )
    assert results.num_rows == 4
    assert set(results["text"].to_pylist()) == {
        "foo",
        "fo",  # 1 deletion # spellchecker:disable-line
        "fob",  # 1 substitution
        "food",  # 1 insertion
    }

    results = ds.to_table(
        full_text_query=MatchQuery("foo", "text", fuzziness=1, max_expansions=3),
    )
    assert results.num_rows == 3

    # prefix matching
    results = ds.to_table(
        full_text_query=MatchQuery("foo", "text", fuzziness=1, prefix_length=3)
    )
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {
        "foo",
        "food",
    }

    # Assert distributed FTS produces consistent fuzzy query results
    assert_distributed_fts_consistency(
        data,
        "text",
        MatchQuery("foo", "text", fuzziness=1),
        tmp_path,
        index_params={"with_position": False},
    )


def test_fts_phrase_query(tmp_path):
    data = pa.table(
        {
            "text": [
                "frodo was a puppy",
                "frodo was a happy puppy",
                "frodo was a very happy puppy",
                "frodo was a puppy with a tail",
            ]
        }
    )

    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index(
        "text", "INVERTED", with_position=True, remove_stop_words=False
    )

    results = ds.to_table(
        full_text_query='"frodo was a puppy"',
    )
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
    }

    results = ds.to_table(
        full_text_query=PhraseQuery("frodo was a puppy", "text"),
    )
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
    }

    results = ds.to_table(full_text_query=PhraseQuery("frodo was happy", "text"))
    assert results.num_rows == 0

    results = ds.to_table(
        full_text_query=PhraseQuery("frodo was happy", "text", slop=1)
    )
    assert results.num_rows == 1

    results = ds.to_table(
        full_text_query=PhraseQuery("frodo was happy", "text", slop=2)
    )
    assert results.num_rows == 2

    # Assert distributed FTS produces consistent phrase query results
    assert_distributed_fts_consistency(
        data,
        "text",
        PhraseQuery("frodo was a puppy", "text"),
        tmp_path,
        index_params={"with_position": True, "remove_stop_words": False},
    )


def test_fts_boost_query(tmp_path):
    data = pa.table(
        {
            "text": [
                "frodo was a puppy",
                "frodo was a happy puppy",
                "frodo was a puppy with a tail",
            ]
        }
    )

    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", "INVERTED", with_position=True)
    results = ds.to_table(
        full_text_query=BoostQuery(
            MatchQuery("puppy", "text"),
            MatchQuery("happy", "text"),
            negative_boost=0.5,
        ),
    )
    assert results.num_rows == 3
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
        "frodo was a happy puppy",
    }

    # boost using phrase
    results = ds.to_table(
        full_text_query=BoostQuery(
            MatchQuery("puppy", "text"),
            PhraseQuery("a happy puppy", "text"),
            negative_boost=0.5,
        ),
    )

    assert results.num_rows == 3
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
        "frodo was a happy puppy",
    }


def test_fts_multi_match_query(tmp_path):
    data = pa.table(
        {
            "title": ["title common", "title hello", "title vector"],
            "content": ["content world", "content database", "content common"],
        }
    )

    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("title", "INVERTED")
    ds.create_scalar_index("content", "INVERTED")

    results = ds.to_table(
        full_text_query=MultiMatchQuery("common", ["title", "content"]),
    )
    assert set(results["title"].to_pylist()) == {"title common", "title vector"}
    assert set(results["content"].to_pylist()) == {"content world", "content common"}

    # Assert distributed FTS produces consistent multi-match query results
    assert_distributed_fts_consistency(
        data,
        ["title", "content"],
        MultiMatchQuery("common", ["title", "content"]),
        tmp_path,
        index_params={"with_position": False},
    )


def test_fts_boolean_query(tmp_path):
    data = pa.table(
        {
            "text": [
                "frodo was a puppy",
                "frodo was a happy puppy",
                "frodo was a puppy with a tail",
            ]
        }
    )

    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", "INVERTED", with_position=True)

    query = MatchQuery("puppy", "text") & MatchQuery("happy", "text")
    results = ds.to_table(
        full_text_query=query,
    )
    assert results.num_rows == 1
    assert results["text"][0].as_py() == "frodo was a happy puppy"

    query = MatchQuery("tail", "text") | MatchQuery("happy", "text")
    results = ds.to_table(
        full_text_query=query,
    )
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {
        "frodo was a happy puppy",
        "frodo was a puppy with a tail",
    }

    results = ds.to_table(
        full_text_query=BooleanQuery(
            [
                (Occur.MUST, MatchQuery("puppy", "text")),
                (Occur.MUST_NOT, MatchQuery("happy", "text")),
            ]
        ),
    )
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
    }

    results = ds.to_table(
        full_text_query=BooleanQuery(
            [
                (Occur.MUST, MatchQuery("puppy", "text")),
                (Occur.MUST_NOT, PhraseQuery("a happy puppy", "text")),
            ]
        ),
    )
    assert results.num_rows == 2
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
    }

    # Assert distributed FTS produces consistent boolean query results
    query_and = MatchQuery("puppy", "text") & MatchQuery("happy", "text")
    assert_distributed_fts_consistency(
        data, "text", query_and, tmp_path, index_params={"with_position": False}
    )


def test_fts_with_postfilter(tmp_path):
    tab = pa.table({"text": ["Frodo the puppy"] * 100, "id": range(100)})
    dataset = lance.write_dataset(tab, tmp_path)
    dataset.create_scalar_index("text", index_type="INVERTED", with_position=False)

    results = dataset.to_table(
        full_text_query="Frodo", filter="id = 7", prefilter=False
    )
    assert results.num_rows == 1

    dataset.create_scalar_index("id", index_type="BTREE")

    results = dataset.to_table(
        full_text_query="Frodo", filter="id = 7", prefilter=False
    )

    assert results.num_rows == 1


def test_fts_with_other_str_scalar_index(dataset):
    dataset.create_scalar_index("doc", index_type="INVERTED", with_position=False)
    dataset.create_scalar_index("doc2", index_type="BTREE")

    row = dataset.take(indices=[0], columns=["doc"])
    query = row.column(0)[0].as_py()
    query = query.split(" ")[0]

    assert dataset.to_table(full_text_query=query).num_rows > 0


def test_fts_all_deleted(dataset):
    dataset.create_scalar_index("doc", index_type="INVERTED", with_position=False)
    first_row_doc = dataset.take(indices=[0], columns=["doc"]).column(0)[0].as_py()
    dataset.delete(f"doc = '{first_row_doc}'")
    dataset.to_table(full_text_query=first_row_doc)


def test_fts_deleted_rows(tmp_path):
    data = pa.table({"text": ["lance is cool", "databases are cool", "search is neat"]})
    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", "INVERTED")
    ds.insert(
        pa.table({"text": ["lance is cool", "databases are cool", "search is neat"]})
    )

    ds.delete("text = 'lance is cool'")
    results = ds.to_table(full_text_query="cool")
    assert results.num_rows == 2


def test_fts_deleted_rows_with_stable_row_ids(tmp_path):
    # Regression test: stable-row-id prefiltering must not leak deleted rows.
    data = pa.table(
        {
            "text": [f"dup_{i}" for i in range(200)],
            "category": [["A", "B", "C", "D", "E"][i % 5] for i in range(200)],
        }
    )
    ds = lance.write_dataset(data, tmp_path, enable_stable_row_ids=True)
    ds.create_scalar_index("text", "INVERTED")

    assert ds.to_table(full_text_query="dup", prefilter=True).num_rows == 200

    ds.delete("category = 'A'")
    assert ds.to_table(full_text_query="dup", prefilter=True).num_rows == 160


def test_index_after_merge_insert(tmp_path):
    # This regresses a defect where a horizontal merge insert was not taking modified
    # fragments out of the index if the column is modified.
    dataset = lance.write_dataset(
        pa.table({"id": range(100), "payload": range(100), "other": range(100)}),
        tmp_path,
    )
    dataset.create_scalar_index("id", index_type="BTREE")
    dataset.create_scalar_index("payload", index_type="BTREE")

    assert dataset.to_table(filter="payload >= 30").num_rows == 70

    # Partial merge insert triggers horizontal merge insert
    dataset.merge_insert(
        "id"
    ).when_matched_update_all().when_not_matched_insert_all().execute(
        pa.table({"id": range(50, 150), "payload": [0] * 100})
    )

    assert dataset.to_table(filter="payload >= 30").num_rows == 20


def test_lindera_load_config_fallback(tmp_path, lindera_ipadic, monkeypatch):
    data = pa.table(
        {
            "text": [
                "成田国際空港",
                "東京国際空港",
                "羽田空港",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    with pytest.raises(OSError):
        ds.create_scalar_index(
            "text", "INVERTED", base_tokenizer="lindera/load_config_fallback"
        )

    config_path = os.path.join(
        os.path.dirname(__file__),
        "models/lindera/load_config_fallback/config_not_exists.yml",
    )
    monkeypatch.setenv("LINDERA_CONFIG_PATH", config_path)
    with pytest.raises(OSError):
        ds.create_scalar_index(
            "text", "INVERTED", base_tokenizer="lindera/load_config_fallback"
        )

    config_path = os.path.join(
        os.path.dirname(__file__), "models/lindera/load_config_fallback/config_env.yml"
    )
    monkeypatch.setenv("LINDERA_CONFIG_PATH", config_path)
    ds.create_scalar_index(
        "text", "INVERTED", base_tokenizer="lindera/load_config_fallback"
    )
    results = ds.to_table(
        full_text_query="成田",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]


def test_lindera_load_config_priority(tmp_path, lindera_ipadic, monkeypatch):
    data = pa.table(
        {
            "text": [
                "成田国際空港",
                "東京国際空港",
                "羽田空港",
            ],
        }
    )
    config_path = os.path.join(
        os.path.dirname(__file__), "models/lindera/load_config_priority/config_env.yml"
    )
    monkeypatch.setenv("LINDERA_CONFIG_PATH", config_path)
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index(
        "text", "INVERTED", base_tokenizer="lindera/load_config_priority"
    )
    results = ds.to_table(
        full_text_query="成田",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]

    results = ds.to_table(
        full_text_query="ほげほげ",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]


def test_indexed_filter_with_fts_index_with_lindera_ipadic_jp_tokenizer(
    tmp_path, lindera_ipadic
):
    data = pa.table(
        {
            "text": [
                "成田国際空港",
                "東京国際空港",
                "羽田空港",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED", base_tokenizer="lindera/ipadic")

    results = ds.to_table(
        full_text_query="成田",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]


def test_lindera_ipadic_jp_tokenizer_invalid_user_dict_path(tmp_path, lindera_ipadic):
    data = pa.table(
        {
            "text": [
                "成田国際空港",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    with pytest.raises(OSError):
        ds.create_scalar_index(
            "text", "INVERTED", base_tokenizer="lindera/invalid_dict"
        )


def test_lindera_ipadic_jp_tokenizer_csv_user_dict_without_type(
    tmp_path, lindera_ipadic
):
    data = pa.table(
        {
            "text": [
                "成田国際空港",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    with pytest.raises(OSError):
        ds.create_scalar_index(
            "text", "INVERTED", base_tokenizer="lindera/invalid_dict2"
        )


def test_lindera_ipadic_jp_tokenizer_csv_user_dict(tmp_path, lindera_ipadic):
    data = pa.table(
        {
            "text": [
                "成田国際空港",
                "東京国際空港",
                "羽田空港",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED", base_tokenizer="lindera/user_dict")
    results = ds.to_table(
        full_text_query="成田",
        prefilter=True,
        with_row_id=True,
    )
    assert len(results) == 0
    results = ds.to_table(
        full_text_query="成田国際空港",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]


def test_lindera_ipadic_jp_tokenizer_bin_user_dict(tmp_path, lindera_ipadic):
    data = pa.table(
        {
            "text": [
                "成田国際空港",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED", base_tokenizer="lindera/user_dict2")
    results = ds.to_table(
        full_text_query="成田",
        prefilter=True,
        with_row_id=True,
    )
    assert len(results) == 0
    results = ds.to_table(
        full_text_query="成田国際空港",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]


def test_jieba_tokenizer(tmp_path):
    set_language_model_path()
    data = pa.table(
        {
            "text": ["我们都有光明的前途", "光明的前途"],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED", base_tokenizer="jieba/default")
    results = ds.to_table(
        full_text_query="我们",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]


def test_icu_tokenizer(tmp_path):
    data = pa.table(
        {
            "text": ["Hello, こんにちは世界!", "Hello, こんにちは!"],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED", base_tokenizer="icu")
    results = ds.to_table(
        full_text_query="世界",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [0]


def test_jieba_invalid_user_dict_tokenizer(tmp_path):
    set_language_model_path()
    data = pa.table(
        {
            "text": [
                "我们都有光明的前途",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    with pytest.raises(OSError):
        ds.create_scalar_index("text", "INVERTED", base_tokenizer="jieba/invalid_dict")


def test_jieba_invalid_main_dict_tokenizer(tmp_path):
    set_language_model_path()
    data = pa.table(
        {
            "text": [
                "我们都有光明的前途",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    with pytest.raises(OSError):
        ds.create_scalar_index("text", "INVERTED", base_tokenizer="jieba/invalid_dict2")


def test_jieba_user_dict_tokenizer(tmp_path):
    set_language_model_path()
    data = pa.table(
        {
            "text": ["我们都有光明的前途", "光明的前途"],
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="overwrite")
    ds.create_scalar_index("text", "INVERTED", base_tokenizer="jieba/user_dict")
    results = ds.to_table(
        full_text_query="的前",
        prefilter=True,
        with_row_id=True,
    )
    assert len(results) == 0
    results = ds.to_table(
        full_text_query="光明的前途",
        prefilter=True,
        with_row_id=True,
    )
    assert results["_rowid"].to_pylist() == [1, 0]


def test_bitmap_index(tmp_path: Path):
    """Test create bitmap index"""
    tbl = pa.Table.from_arrays(
        [pa.array([["a", "b", "c"][i % 3] for i in range(100)])], names=["a"]
    )
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")
    dataset.create_scalar_index("a", index_type="BITMAP")
    indices = dataset.describe_indices()
    assert len(indices) == 1
    assert indices[0].index_type == "Bitmap"


def test_bitmap_empty_range(tmp_path: Path):
    data = pa.table({"c0": pa.array([1, 2, 3], type=pa.int64())})
    dataset = lance.write_dataset(data, tmp_path / "dataset")
    dataset.create_scalar_index("c0", index_type="BITMAP")
    filters = [
        "c0 BETWEEN 2 AND 1",
        "c0 > 2 AND c0 < 2",
        "c0 >= 2 AND c0 < 2",
        "c0 > 2 AND c0 <= 2",
    ]
    for filter_expr in filters:
        result = dataset.to_table(filter=filter_expr, use_scalar_index=True)
        assert result.num_rows == 0


def test_btree_remap_big_deletions(tmp_path: Path):
    # Write 15K rows in 3 fragments
    ds = lance.write_dataset(pa.table({"a": range(5000)}), tmp_path)
    ds = lance.write_dataset(
        pa.table({"a": range(5000, 10000)}), tmp_path, mode="append"
    )
    ds = lance.write_dataset(
        pa.table({"a": range(10000, 15000)}), tmp_path, mode="append"
    )

    # Create index (will have 4 pages)
    ds.create_scalar_index("a", index_type="BTREE")

    # Delete a lot of data (now there will only be two pages worth)
    ds.delete("a > 1000 AND a < 10000")

    # Run compaction (deletions will be materialized)
    ds.optimize.compact_files()

    # Reload dataset and ensure index still works
    ds = lance.dataset(tmp_path)

    for idx in [0, 500, 1000, 10000, 13000, 14000, 14999]:
        assert ds.to_table(filter=f"a = {idx}").num_rows == 1

    for idx in [1001, 5000, 8000, 9999]:
        assert ds.to_table(filter=f"a = {idx}").num_rows == 0


def test_bitmap_remap(tmp_path: Path):
    # Make one full fragment
    tbl = pa.Table.from_arrays(
        [pa.array([["a", "b"][i % 2] for i in range(10)])], names=["a"]
    )
    ds = lance.write_dataset(tbl, tmp_path, max_rows_per_file=10)

    # Make two half fragments
    tbl = pa.Table.from_arrays(
        [pa.array([["a", "b"][i % 2] for i in range(10)])], names=["a"]
    )
    ds = lance.write_dataset(tbl, tmp_path, max_rows_per_file=5, mode="append")

    # Create scalar index
    ds.create_scalar_index("a", index_type="BITMAP")

    # Run compaction (two partials will be remapped, full will not)
    compaction = ds.optimize.compact_files(target_rows_per_fragment=10)
    assert compaction.fragments_removed == 2

    for category in ["a", "b"]:
        # All rows should still be in index
        assert ds.count_rows(f"a = '{category}'") == 10


def test_ngram_index(tmp_path: Path):
    """Test create ngram index"""

    def test_with(tbl: pa.Table):
        dataset = lance.write_dataset(tbl, tmp_path / "dataset", mode="overwrite")
        dataset.create_scalar_index("words", index_type="NGRAM")
        indices = dataset.describe_indices()
        assert len(indices) == 1
        assert indices[0].index_type == "NGram"

        scan_plan = dataset.scanner(filter="contains(words, 'apple')").explain_plan(
            True
        )
        assert "ScalarIndexQuery: query=[contains(words" in scan_plan

        assert dataset.to_table(filter="contains(words, 'apple')").num_rows == 50
        assert dataset.to_table(filter="contains(words, 'banana')").num_rows == 25
        assert dataset.to_table(filter="contains(words, 'coconut')").num_rows == 25
        assert dataset.to_table(filter="contains(words, 'apples')").num_rows == 25
        assert (
            dataset.to_table(
                filter="contains(words, 'apple') AND contains(words, 'banana')"
            ).num_rows
            == 0
        )
        assert (
            dataset.to_table(
                filter="contains(words, 'apple') OR contains(words, 'banana')"
            ).num_rows
            == 75
        )

    tbl = pa.Table.from_arrays(
        [
            pa.array(
                [["apple", "apples", "banana", "coconut"][i % 4] for i in range(100)]
            )
        ],
        names=["words"],
    )
    test_with(tbl)

    # Test with large string
    tbl = pa.Table.from_arrays(
        [
            pa.array(
                [["apple", "apples", "banana", "coconut"][i % 4] for i in range(100)],
                type=pa.large_string(),
            )
        ],
        names=["words"],
    )
    test_with(tbl)


def test_zonemap_index(tmp_path: Path):
    """Test create zonemap index"""
    tbl = pa.Table.from_arrays([pa.array([i for i in range(8193)])], names=["values"])
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")
    dataset.create_scalar_index("values", index_type="ZONEMAP")
    indices = dataset.describe_indices()
    assert len(indices) == 1

    # Get detailed index statistics
    index_stats = dataset.stats.index_stats("values_idx")
    assert index_stats["index_type"] == "ZoneMap"
    assert "indices" in index_stats
    assert len(index_stats["indices"]) == 1

    # Verify zonemap statistics
    zonemap_stats = index_stats["indices"][0]
    assert zonemap_stats["rows_per_zone"] == 8192
    assert zonemap_stats["num_zones"] == 2  # Should have 2 zones (8192 rows + 1 row)

    # Test that the zonemap index is being used in the query plan
    scanner = dataset.scanner(filter="values > 50", prefilter=True)
    plan = scanner.explain_plan()
    assert "ScalarIndexQuery" in plan

    # Verify the query returns correct results
    result = scanner.to_table()
    assert result.num_rows == 8142  # 51..8192


def test_zonemap_zone_size(tmp_path: Path):
    ds = lance.write_dataset(pa.table({"x": range(64 * 1024)}), tmp_path)

    def get_bytes_read():
        scan_stats = None

        def scan_stats_callback(stats: lance.ScanStatistics):
            nonlocal scan_stats
            scan_stats = stats

        ds.scanner(filter="x = 7", scan_stats_callback=scan_stats_callback).to_table()

        return scan_stats.bytes_read

    ds.create_scalar_index(
        "x",
        IndexConfig(index_type="zonemap", parameters={"rows_per_zone": 64}),
    )

    small_bytes_read = get_bytes_read()

    ds.create_scalar_index(
        "x",
        IndexConfig(index_type="zonemap", parameters={"rows_per_zone": 16 * 1024}),
    )

    large_bytes_read = get_bytes_read()

    assert small_bytes_read < large_bytes_read


def test_zonemap_deletion_handling(tmp_path: Path):
    """Test zonemap deletion handling"""
    data = pa.table(
        {
            "id": range(10),
            "value": [True, False] * 5,
        }
    )
    ds = lance.write_dataset(data, "memory://", max_rows_per_group=5)
    ds.delete("NOT value")
    assert ds.to_table(filter="value = True").num_rows == 5
    assert ds.to_table(filter="value = False").num_rows == 0
    ids = ds.to_table(filter="value = True")["id"].to_pylist()
    assert ids == [0, 2, 4, 6, 8]

    ds.create_scalar_index("value", index_type="zonemap")
    ids = ds.to_table(filter="value = True")["id"].to_pylist()
    assert ids == [0, 2, 4, 6, 8]

    # now create the index before deletion
    ds = lance.write_dataset(data, "memory://", max_rows_per_group=5)
    ds.create_scalar_index("value", index_type="zonemap")
    ds.delete("NOT value")
    assert ds.to_table(filter="value = True").num_rows == 5
    assert ds.to_table(filter="value = False").num_rows == 0
    ids = ds.to_table(filter="value = True")["id"].to_pylist()
    assert ids == [0, 2, 4, 6, 8]


def test_zonemap_index_remapping(tmp_path: Path):
    """Test zonemap index remapping after compaction and optimization"""
    # Create a dataset with 5 fragments by writing data in chunks
    # Each fragment will have 1000 rows, so we need 5000 total rows
    tbl = pa.Table.from_arrays([pa.array(range(0, 5000))], names=["values"])
    dataset = lance.write_dataset(tbl, tmp_path / "dataset", max_rows_per_file=1000)

    fragments = dataset.get_fragments()
    assert len(fragments) == 5

    # Train a zone map index
    dataset.create_scalar_index("values", index_type="ZONEMAP")
    indices = dataset.describe_indices()
    assert len(indices) == 1
    assert indices[0].index_type == "ZoneMap"

    # Confirm the zone map index is used if you search the dataset
    scanner = dataset.scanner(filter="values > 2500", prefilter=True)
    plan = scanner.explain_plan()
    assert "ScalarIndexQuery" in plan

    # Verify the query returns correct results
    result = scanner.to_table()
    assert result.num_rows == 2499  # 2501..4999

    # Run compaction to merge fragments
    compaction = dataset.optimize.compact_files(target_rows_per_fragment=2000)
    assert compaction.fragments_removed == 5
    assert len(dataset.get_fragments()) == 3

    # Check if the zone map index is no longer being used
    scanner = dataset.scanner(filter="values > 2500", prefilter=True)
    plan = scanner.explain_plan()
    assert "ScalarIndexQuery" not in plan

    # Run optimize indices to rebuild the index
    dataset.optimize.optimize_indices()

    # Confirm the zone map index is used again after optimization
    scanner = dataset.scanner(filter="values > 2500", prefilter=True)
    plan = scanner.explain_plan()
    assert "ScalarIndexQuery" in plan

    # Verify the query returns correct results
    result = scanner.to_table()
    assert result.num_rows == 2499  # 2501..4999

    # Test with a different query to ensure index works properly
    scanner = dataset.scanner(filter="values BETWEEN 1000 AND 1500", prefilter=True)
    plan = scanner.explain_plan()
    assert "ScalarIndexQuery" in plan

    result = scanner.to_table()
    assert result.num_rows == 501  # 1000..1500 inclusive


def test_bloomfilter_index(tmp_path: Path):
    """Test create bloomfilter index"""
    tbl = pa.Table.from_arrays([pa.array([i for i in range(10000)])], names=["values"])
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")
    dataset.create_scalar_index("values", index_type="BLOOMFILTER")
    indices = dataset.describe_indices()
    assert len(indices) == 1

    # Get detailed index statistics
    index_stats = dataset.stats.index_stats("values_idx")
    assert index_stats["index_type"] == "BloomFilter"
    assert "indices" in index_stats
    assert len(index_stats["indices"]) == 1

    # Verify bloomfilter statistics
    bloom_stats = index_stats["indices"][0]
    assert "num_blocks" in bloom_stats
    assert bloom_stats["num_blocks"] == 2
    assert bloom_stats["number_of_items"] == 8192
    assert "probability" in bloom_stats
    assert bloom_stats["probability"] == 0.00057  # Default probability

    # Test that the bloomfilter index is being used in the query plan
    scanner = dataset.scanner(filter="values == 1234", prefilter=True)
    plan = scanner.explain_plan()
    assert "ScalarIndexQuery" in plan

    # Verify the query returns correct results
    result = scanner.to_table()
    assert result.num_rows == 1
    assert result["values"][0].as_py() == 1234


def test_bloomfilter_deletion_handling(tmp_path: Path):
    """Test bloomfilter deletion handling"""
    data = pa.table(
        {
            "id": range(10),
            "value": [1, 0] * 5,
        }
    )
    ds = lance.write_dataset(data, "memory://", max_rows_per_group=5)
    ds.delete("value = 0")
    assert ds.to_table(filter="value = 1").num_rows == 5
    assert ds.to_table(filter="value = 0").num_rows == 0
    ids = ds.to_table(filter="value = 1")["id"].to_pylist()
    assert ids == [0, 2, 4, 6, 8]

    ds.create_scalar_index("value", index_type="bloomfilter")
    ids = ds.to_table(filter="value = 1")["id"].to_pylist()
    assert ids == [0, 2, 4, 6, 8]

    # now create the index before deletion
    ds = lance.write_dataset(data, "memory://", max_rows_per_group=5)
    ds.create_scalar_index("value", index_type="bloomfilter")
    ds.delete("value = 0")
    assert ds.to_table(filter="value = 1").num_rows == 5
    assert ds.to_table(filter="value = 0").num_rows == 0
    ids = ds.to_table(filter="value = 1")["id"].to_pylist()
    assert ids == [0, 2, 4, 6, 8]


def test_json_index():
    vals = ['{"x": 7, "y": 10}', '{"x": 11, "y": 22}', '{"y": 0}', '{"x": 10}']
    tbl = pa.table({"jsons": pa.array(vals, pa.json_())})
    ds = lance.write_dataset(tbl, "memory://test")
    ds.create_scalar_index(
        "jsons",
        IndexConfig(
            index_type="json", parameters={"target_index_type": "btree", "path": "x"}
        ),
    )

    # TODO: I changed this into `json_get_int` for strong typed filter, should be
    # refactored into JSON Path compare.
    filter = "json_get_int(jsons, 'x') = 10"
    assert "ScalarIndexQuery" in ds.scanner(filter=filter).explain_plan()
    assert ds.to_table(filter=filter) == ds.to_table(
        filter=filter, use_scalar_index=False
    )


def test_null_handling():
    tbl = pa.table(
        {
            "x": [1, 2, None, 3],
            "y": ["a", "b", "c", None],
        }
    )
    dataset = lance.write_dataset(tbl, "memory://test")

    def check():
        assert dataset.to_table(filter="x IS NULL").num_rows == 1
        assert dataset.to_table(filter="x IS NOT NULL").num_rows == 3
        assert dataset.to_table(filter="x > 0").num_rows == 3
        assert dataset.to_table(filter="x < 5").num_rows == 3
        assert dataset.to_table(filter="x IN (1, 2)").num_rows == 2
        assert dataset.to_table(filter="x IN (1, 2, NULL)").num_rows == 2
        assert dataset.to_table(filter="x > 0 OR (y != 'a')").num_rows == 4
        assert dataset.to_table(filter="x > 0 AND (y != 'a')").num_rows == 1
        assert dataset.to_table(filter="y != 'a'").num_rows == 2
        # NOT should exclude nulls (issue #4756)
        assert dataset.to_table(filter="NOT (x < 2)").num_rows == 2
        assert dataset.to_table(filter="NOT (x IN (1, 2))").num_rows == 1
        # Double NOT
        assert dataset.to_table(filter="NOT (NOT (x < 2))").num_rows == 1

    check()
    dataset.create_scalar_index("x", index_type="BITMAP")
    check()
    dataset.create_scalar_index("y", index_type="BTREE")
    check()


def test_nan_handling(tmp_path: Path):
    tbl = pa.table(
        {
            "x": [
                1.0,
                float("-nan"),
                float("infinity"),
                float("-infinity"),
                2.0,
                float("nan"),
                3.0,
            ],
        }
    )
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")

    # There is no way, in DF, to query for NAN / INF, that I'm aware of.
    # So the best we can do here is make sure that the presence of NAN / INF
    # doesn't interfere with normal operation of the btree.
    def check(has_index: bool):
        assert dataset.to_table(filter="x IS NULL").num_rows == 0
        assert dataset.to_table(filter="x IS NOT NULL").num_rows == 7
        assert dataset.to_table(filter="x > 0").num_rows == 5
        assert dataset.to_table(filter="x < 5").num_rows == 5
        assert dataset.to_table(filter="x IN (1, 2)").num_rows == 2

    check(False)
    dataset.create_scalar_index("x", index_type="BITMAP")
    check(True)
    dataset.create_scalar_index("x", index_type="BTREE")
    check(True)


def test_scalar_index_with_nulls(tmp_path):
    # Create a test dataframe with 50% null values.
    test_table_size = 10_000
    test_table = pa.table(
        {
            "item_id": list(range(test_table_size)),
            "inner_id": list(range(test_table_size)),
            "category": ["a", None] * (test_table_size // 2),
            "numeric_int": [1, None] * (test_table_size // 2),
            "numeric_float": [0.1, None] * (test_table_size // 2),
            "boolean_col": [True, None] * (test_table_size // 2),
            "timestamp_col": [datetime(2023, 1, 1), None] * (test_table_size // 2),
            "ngram_col": ["apple", None] * (test_table_size // 2),
        }
    )
    ds = lance.write_dataset(test_table, tmp_path)
    ds.create_scalar_index("inner_id", index_type="BTREE")
    ds.create_scalar_index("category", index_type="BTREE")
    ds.create_scalar_index("boolean_col", index_type="BTREE")
    ds.create_scalar_index("timestamp_col", index_type="BTREE")
    ds.create_scalar_index("ngram_col", index_type="NGRAM")
    # Test querying with filters on columns with nulls.
    k = test_table_size // 2
    result = ds.to_table(filter="category = 'a'", limit=k)
    assert len(result) == k
    # Booleans should be stored as strings in the table for backwards compatibility.
    result = ds.to_table(filter="boolean_col IS TRUE", limit=k)
    assert len(result) == k
    result = ds.to_table(filter="timestamp_col IS NOT NULL", limit=k)
    assert len(result) == k

    # Ensure ngram index works with nulls
    result = ds.to_table(filter="ngram_col = 'apple'")
    assert len(result) == k
    result = ds.to_table(filter="ngram_col IS NULL")
    assert len(result) == k
    result = ds.to_table(filter="contains(ngram_col, 'appl')")
    assert len(result) == k


def test_label_list_index(tmp_path: Path):
    tags = pa.array(["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"])
    tag_list = pa.ListArray.from_arrays([0, 2, 4], tags)
    tbl = pa.Table.from_arrays([tag_list], names=["tags"])
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")
    dataset.create_scalar_index("tags", index_type="LABEL_LIST")
    indices = dataset.describe_indices()
    assert len(indices) == 1
    assert indices[0].index_type == "LabelList"


def test_label_list_index_array_contains(tmp_path: Path):
    # Include lists with NULL items to ensure NULL needle behavior matches
    # non-index execution.
    tbl = pa.table(
        {"labels": [["foo", "bar"], ["bar"], ["baz"], ["qux", None], [None], [], None]}
    )
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")
    expected_null_rows = dataset.to_table(
        filter="array_contains(labels, NULL)"
    ).num_rows

    dataset.create_scalar_index("labels", index_type="LABEL_LIST")

    result = dataset.to_table(filter="array_contains(labels, 'foo')")
    assert result.num_rows == 1

    result = dataset.to_table(filter="array_contains(labels, 'bar')")
    assert result.num_rows == 2

    result = dataset.to_table(filter="array_contains(labels, 'oof')")
    assert result.num_rows == 0

    explain = dataset.scanner(filter="array_contains(labels, 'foo')").explain_plan()
    assert "ScalarIndexQuery" in explain

    # NULL needle: preserve semantics (must match pre-index execution) and avoid
    # using the LABEL_LIST index.
    actual_null_rows = dataset.to_table(filter="array_contains(labels, NULL)").num_rows
    assert actual_null_rows == expected_null_rows
    explain = dataset.scanner(filter="array_contains(labels, NULL)").explain_plan()
    assert "ScalarIndexQuery" not in explain


def test_label_list_index_empty_list_filters(tmp_path: Path):
    """Empty list filters should not panic and should match pre-index results."""
    tbl = pa.table({"labels": [["foo"], ["bar"], ["foo", None], [None], [], None]})
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")

    filters = [
        "array_has_any(labels, [])",
        "array_has_all(labels, [])",
        "NOT array_has_all(labels, [])",
        "NOT array_has_any(labels, [])",
    ]
    expected = {f: dataset.to_table(filter=f).num_rows for f in filters}

    dataset.create_scalar_index("labels", index_type="LABEL_LIST")

    for f in filters:
        assert dataset.to_table(filter=f).num_rows == expected[f]


def test_label_list_index_null_element_match(tmp_path: Path):
    """Covers NULL elements inside non-NULL lists (list itself is never NULL)."""
    tbl = pa.table(
        {"labels": [["foo", None], ["foo"], ["bar", None], [None], ["bar"], []]}
    )
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")

    filters = [
        "array_has_any(labels, ['foo'])",
        "array_has_all(labels, ['foo'])",
        "array_contains(labels, 'foo')",
        "NOT array_has_any(labels, ['foo'])",
        "NOT array_has_all(labels, ['foo'])",
        "NOT array_contains(labels, 'foo')",
    ]
    expected = {
        f: dataset.to_table(filter=f).column("labels").to_pylist() for f in filters
    }

    dataset.create_scalar_index("labels", index_type="LABEL_LIST")

    actual = {
        f: dataset.to_table(filter=f).column("labels").to_pylist() for f in filters
    }
    assert actual == expected


def test_label_list_index_null_list_match(tmp_path: Path):
    """Covers NULL lists (list itself is NULL, elements are not NULL)."""
    tbl = pa.table({"labels": [["foo"], ["bar"], None, []]})
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")

    filters = [
        "array_has_any(labels, ['foo'])",
        "array_has_all(labels, ['foo'])",
        "array_contains(labels, 'foo')",
        "NOT array_has_any(labels, ['foo'])",
        "NOT array_has_all(labels, ['foo'])",
        "NOT array_contains(labels, 'foo')",
    ]
    expected = {
        f: dataset.to_table(filter=f).column("labels").to_pylist() for f in filters
    }

    dataset.create_scalar_index("labels", index_type="LABEL_LIST")

    actual = {
        f: dataset.to_table(filter=f).column("labels").to_pylist() for f in filters
    }
    assert actual == expected


def test_label_list_index_null_literal_filters(tmp_path: Path):
    """Ensure filters with NULL literal needles produce consistent results with scan."""
    tbl = pa.table(
        {"labels": [["foo", None], ["bar", None], [None], ["foo"], ["bar"], []]}
    )
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")

    filters = [
        "array_has_any(labels, [NULL])",
        "array_has_all(labels, [NULL])",
        "array_contains(labels, NULL)",
        "NOT array_has_any(labels, [NULL])",
        "NOT array_has_all(labels, [NULL])",
        "NOT array_contains(labels, NULL)",
    ]
    expected = {
        f: dataset.to_table(filter=f).column("labels").to_pylist() for f in filters
    }

    dataset.create_scalar_index("labels", index_type="LABEL_LIST")

    actual = {
        f: dataset.to_table(filter=f).column("labels").to_pylist() for f in filters
    }
    assert actual == expected


def test_label_list_index_explain_null_literals(tmp_path: Path):
    tbl = pa.table({"labels": [["foo", None], ["foo"]]})
    dataset = lance.write_dataset(tbl, tmp_path / "dataset")
    dataset.create_scalar_index("labels", index_type="LABEL_LIST")

    # explain_plan should not panic when list literals include NULLs.
    for expr in [
        "array_has_any(labels, [NULL])",
        "array_has_all(labels, [NULL])",
        "array_has_any(labels, ['foo', NULL])",
        "array_has_all(labels, ['foo', NULL])",
    ]:
        explain = dataset.scanner(filter=expr).explain_plan()
        assert isinstance(explain, str)


def test_create_index_empty_dataset(tmp_path: Path):
    # Creating an index on an empty dataset is (currently) not terribly useful but
    # we shouldn't return strange errors.
    schema = pa.schema(
        [
            pa.field("btree", pa.int32()),
            pa.field("bitmap", pa.int32()),
            pa.field("label_list", pa.list_(pa.string())),
            pa.field("inverted", pa.string()),
            pa.field("ngram", pa.string()),
        ]
    )
    ds = lance.write_dataset([], tmp_path, schema=schema)

    for index_type in ["BTREE", "BITMAP", "LABEL_LIST", "INVERTED", "NGRAM"]:
        ds.create_scalar_index(index_type.lower(), index_type=index_type)

    # Make sure the empty index doesn't cause searches to fail
    ds.insert(
        pa.table(
            {
                "btree": pa.array([1], pa.int32()),
                "bitmap": pa.array([1], pa.int32()),
                "label_list": [["foo", "bar"]],
                "inverted": ["blah"],
                "ngram": ["apple"],
            }
        )
    )

    def test_searches():
        assert ds.to_table(filter="btree = 1").num_rows == 1
        assert ds.to_table(filter="btree = 0").num_rows == 0
        assert ds.to_table(filter="bitmap = 1").num_rows == 1
        assert ds.to_table(filter="bitmap = 0").num_rows == 0
        assert ds.to_table(filter="array_has_any(label_list, ['foo'])").num_rows == 1
        assert ds.to_table(filter="array_has_any(label_list, ['oof'])").num_rows == 0
        assert ds.to_table(filter="inverted = 'blah'").num_rows == 1
        assert ds.to_table(filter="inverted = 'halb'").num_rows == 0
        assert ds.to_table(filter="contains(ngram, 'apple')").num_rows == 1
        assert ds.to_table(filter="contains(ngram, 'banana')").num_rows == 0
        assert ds.to_table(filter="ngram = 'apple'").num_rows == 1

    test_searches()

    # Make sure fetching index stats on empty index is ok
    for idx in ds.describe_indices():
        ds.stats.index_stats(idx.name)

    # Make sure updating empty indices is ok
    ds.optimize.optimize_indices()

    # Finally, make sure we can still search after updating
    test_searches()


def test_optimize_no_new_data(tmp_path: Path):
    tbl = pa.table(
        {
            "btree": pa.array([None], pa.int64()),
            "bitmap": pa.array([None], pa.int64()),
            "ngram": pa.array([None], pa.string()),
        }
    )
    dataset = lance.write_dataset(tbl, tmp_path)
    dataset.create_scalar_index("btree", index_type="BTREE")
    dataset.create_scalar_index("bitmap", index_type="BITMAP")
    dataset.create_scalar_index("ngram", index_type="NGRAM")

    assert dataset.to_table(filter="btree IS NULL").num_rows == 1
    assert dataset.to_table(filter="bitmap IS NULL").num_rows == 1
    assert dataset.to_table(filter="ngram IS NULL").num_rows == 1

    dataset.insert([], schema=tbl.schema)
    dataset.optimize.optimize_indices()

    assert dataset.to_table(filter="btree IS NULL").num_rows == 1
    assert dataset.to_table(filter="bitmap IS NULL").num_rows == 1
    assert dataset.to_table(filter="ngram IS NULL").num_rows == 1

    dataset.insert(pa.table({"btree": [2]}))
    dataset.optimize.optimize_indices()

    assert dataset.to_table(filter="btree IS NULL").num_rows == 1
    assert dataset.to_table(filter="bitmap IS NULL").num_rows == 2
    assert dataset.to_table(filter="ngram IS NULL").num_rows == 2

    dataset.insert(pa.table({"bitmap": [2]}))
    dataset.optimize.optimize_indices()

    assert dataset.to_table(filter="btree IS NULL").num_rows == 2
    assert dataset.to_table(filter="bitmap IS NULL").num_rows == 2
    assert dataset.to_table(filter="ngram IS NULL").num_rows == 3

    dataset.insert(pa.table({"ngram": ["apple"]}))

    assert dataset.to_table(filter="btree IS NULL").num_rows == 3
    assert dataset.to_table(filter="bitmap IS NULL").num_rows == 3
    assert dataset.to_table(filter="ngram IS NULL").num_rows == 3


def test_drop_index(tmp_path):
    test_table_size = 100
    test_table = pa.table(
        {
            "btree": list(range(test_table_size)),
            "bitmap": list(range(test_table_size)),
            "fts": ["a" for _ in range(test_table_size)],
            "ngram": ["a" for _ in range(test_table_size)],
        }
    )
    ds = lance.write_dataset(test_table, tmp_path)
    ds.create_scalar_index("btree", index_type="BTREE")
    ds.create_scalar_index("bitmap", index_type="BITMAP")
    ds.create_scalar_index("fts", index_type="INVERTED")
    ds.create_scalar_index("ngram", index_type="NGRAM")

    assert len(ds.describe_indices()) == 4

    # Attempt to drop index (name does not exist)
    with pytest.raises(RuntimeError, match="index not found"):
        ds.drop_index("nonexistent_name")

    for idx in ds.describe_indices():
        idx_name = idx.name
        ds.drop_index(idx_name)

    assert len(ds.describe_indices()) == 0

    # Ensure we can still search columns
    assert ds.to_table(filter="btree = 1").num_rows == 1
    assert ds.to_table(filter="bitmap = 1").num_rows == 1
    assert ds.to_table(filter="fts = 'a'").num_rows == test_table_size
    assert ds.to_table(filter="contains(ngram, 'a')").num_rows == test_table_size


def test_index_prewarm(tmp_path: Path):
    scan_stats = None

    def scan_stats_callback(stats: lance.ScanStatistics):
        nonlocal scan_stats
        scan_stats = stats

    test_table_size = 100
    test_table = pa.table(
        {
            "fts": ["word" for _ in range(test_table_size)],
        }
    )

    # Write index, cache should not be populated
    ds = lance.write_dataset(test_table, tmp_path)
    ds.create_scalar_index("fts", index_type="INVERTED")
    ds.scanner(
        scan_stats_callback=scan_stats_callback, full_text_query="word"
    ).to_table()
    assert scan_stats.parts_loaded > 0

    # Fresh load, no prewarm, cache should not be populated
    ds = lance.dataset(tmp_path)
    ds.scanner(
        scan_stats_callback=scan_stats_callback, full_text_query="word"
    ).to_table()
    assert scan_stats.parts_loaded > 0

    # Prewarm index, cache should be populated
    ds = lance.dataset(tmp_path)
    ds.prewarm_index("fts_idx")
    ds.scanner(
        scan_stats_callback=scan_stats_callback, full_text_query="word"
    ).to_table()
    assert scan_stats.parts_loaded == 0

    phrase_path = tmp_path / "phrase"
    phrase_table = pa.table(
        {
            "fts": ["word word" for _ in range(test_table_size)],
        }
    )
    ds = lance.write_dataset(phrase_table, phrase_path)
    ds.create_scalar_index("fts", index_type="INVERTED", with_position=True)

    ds = lance.dataset(phrase_path)
    ds.prewarm_index("fts_idx")
    cache_entries_after_prewarm = ds._ds.index_cache_entry_count()
    results = ds.to_table(full_text_query=PhraseQuery("word word", "fts"))
    assert results.num_rows == test_table_size
    cache_entries_after_query = ds._ds.index_cache_entry_count()
    assert cache_entries_after_query > cache_entries_after_prewarm

    ds = lance.dataset(phrase_path)
    ds.prewarm_index("fts_idx", with_position=True)
    cache_entries_after_prewarm = ds._ds.index_cache_entry_count()
    results = ds.to_table(full_text_query=PhraseQuery("word word", "fts"))
    assert results.num_rows == test_table_size
    cache_entries_after_query = ds._ds.index_cache_entry_count()
    assert cache_entries_after_query == cache_entries_after_prewarm

    with pytest.raises(
        TypeError,
        match="takes 2 positional arguments",
    ):
        ds.prewarm_index("fts_idx", True)


def test_btree_prewarm(tmp_path: Path):
    scan_stats = None

    def scan_stats_callback(stats: lance.ScanStatistics):
        nonlocal scan_stats
        scan_stats = stats

    test_table_size = 100
    test_table = pa.table(
        {
            "id": list(range(test_table_size)),
        }
    )
    ds = lance.write_dataset(test_table, tmp_path)
    ds.create_scalar_index("id", index_type="BTREE")
    ds.scanner(scan_stats_callback=scan_stats_callback, filter="id>0").to_table()
    assert scan_stats.parts_loaded > 0

    ds = lance.dataset(tmp_path)
    ds.scanner(scan_stats_callback=scan_stats_callback, filter="id>0").to_table()
    assert scan_stats.parts_loaded > 0

    ds = lance.dataset(tmp_path)
    ds.prewarm_index("id_idx")
    ds.scanner(scan_stats_callback=scan_stats_callback, filter="id>0").to_table()
    assert scan_stats.parts_loaded == 0


def test_fts_backward_v0_27_0(tmp_path: Path):
    path = (
        Path(__file__).parent.parent.parent.parent
        / "test_data"
        / "0.27.0"
        / "legacy_fts_index"
    )
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    ds = lance.dataset(tmp_path)

    # we can read the old index
    results = ds.to_table(
        full_text_query=BoostQuery(
            MatchQuery("puppy", "text"),
            MatchQuery("happy", "text"),
            negative_boost=0.5,
        ),
    )
    assert results.num_rows == 3
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
        "frodo was a happy puppy",
    }

    data = pa.table(
        {
            "text": [
                "new data",
            ]
        }
    )
    ds = lance.write_dataset(data, tmp_path, mode="append")
    ds.optimize.optimize_indices()
    results = ds.to_table(
        full_text_query=BoostQuery(
            MatchQuery("puppy", "text"),
            MatchQuery("happy", "text"),
            negative_boost=0.5,
        ),
    )
    assert results.num_rows == 3
    assert set(results["text"].to_pylist()) == {
        "frodo was a puppy",
        "frodo was a puppy with a tail",
        "frodo was a happy puppy",
    }
    res = ds.to_table(
        full_text_query="new",
    )
    assert res.num_rows == 1


# ============================================================================
# Distributed FTS Index Helper Functions
# ============================================================================


def build_distributed_fts_index(
    dataset: lance.LanceDataset, column: str, index_name: str = None, **index_params
) -> lance.LanceDataset:
    """
    Build FTS index in distributed way and return the committed dataset.

    This helper function builds the FTS index on individual fragments
    and then commits them as a single index, ensuring the distributed
    approach produces the same results as single-machine indexing.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset to build index on
    column : str
        The column name to build FTS index on
    index_name : str, optional
        Name for the index. If not provided, will use f"{column}_distributed_idx"
    **index_params
        Additional parameters to pass to create_scalar_index()
        (e.g., with_position, remove_stop_words, base_tokenizer, etc.)

    Returns
    -------
    lance.LanceDataset
        Dataset with committed distributed FTS index

    Examples
    --------
    >>> ds_distributed = build_distributed_fts_index(
    ...     dataset,
    ...     "text",
    ...     with_position=True,
    ...     remove_stop_words=False
    ... )
    >>> # Now compare with single-machine index results
    >>> results_distributed = ds_distributed.scanner(full_text_query="test").to_table()
    """
    import uuid

    from lance.dataset import Index

    # Generate unique index ID for distributed indexing
    index_id = str(uuid.uuid4())
    index_name = index_name or f"{column}_distributed_idx"

    # Get all fragments from the dataset
    fragments = dataset.get_fragments()
    fragment_ids = [fragment.fragment_id for fragment in fragments]

    # Build index on each fragment individually
    for fragment_id in fragment_ids:
        dataset.create_scalar_index(
            column=column,
            index_type="INVERTED",
            name=index_name,
            replace=False,
            index_uuid=index_id,
            fragment_ids=[fragment_id],
            **index_params,
        )

    # Merge the inverted index metadata
    dataset.merge_index_metadata(index_id, index_type="INVERTED")

    # Create Index object for commit
    field_id = dataset.schema.get_field_index(column)
    index = Index(
        uuid=index_id,
        name=index_name,
        fields=[field_id],
        dataset_version=dataset.version,
        fragment_ids=set(fragment_ids),
        index_version=0,
    )

    # Create and commit the index operation
    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )

    # Commit the distributed index
    committed_dataset = lance.LanceDataset.commit(
        dataset.uri,
        create_index_op,
        read_version=dataset.version,
    )

    return committed_dataset


def compare_fts_results(
    single_machine_results: pa.Table,
    distributed_results: pa.Table,
    tolerance: float = 1e-6,
) -> bool:
    """
    Compare FTS search results from single-machine and distributed indexing.

    Parameters
    ----------
    single_machine_results : pa.Table
        Results from single-machine FTS index
    distributed_results : pa.Table
        Results from distributed FTS index
    tolerance : float, default 1e-6
        Tolerance for floating point score comparison

    Returns
    -------
    bool
        True if results are equivalent, False otherwise

    Raises
    ------
    AssertionError
        If results don't match with detailed error message
    """
    # Check row count
    assert single_machine_results.num_rows == distributed_results.num_rows, (
        f"Row count mismatch: single={single_machine_results.num_rows}, "
        f"distributed={distributed_results.num_rows}"
    )

    # If no results, both should be empty
    if single_machine_results.num_rows == 0:
        return True

    # Convert to pandas for easier comparison
    single_df = single_machine_results.to_pandas()
    distributed_df = distributed_results.to_pandas()

    # Normalize row ordering for comparisons.
    #
    # FTS search results do not guarantee a stable order for tied scores and
    # different execution modes (single-machine vs distributed) may return rows
    # in different (but equivalent) orders.
    sort_cols = (
        ["_rowid"]
        if "_rowid" in single_df.columns
        else [c for c in single_df.columns if c != "_score"]
    )
    if sort_cols:
        single_df = single_df.sort_values(sort_cols, kind="mergesort").reset_index(
            drop=True
        )
        distributed_df = distributed_df.sort_values(
            sort_cols, kind="mergesort"
        ).reset_index(drop=True)

    # Compare row IDs (most important)
    if "_rowid" in single_df.columns:
        single_rowids = set(single_df["_rowid"])
        distributed_rowids = set(distributed_df["_rowid"])
        assert single_rowids == distributed_rowids, (
            f"Row ID mismatch: single={single_rowids}, distributed={distributed_rowids}"
        )

    # Compare scores with tolerance
    if "_score" in single_df.columns:
        single_scores = single_df["_score"].to_numpy(dtype=float)
        distributed_scores = distributed_df["_score"].to_numpy(dtype=float)
        score_diff = np.abs(single_scores - distributed_scores)
        max_diff = np.max(score_diff)
        assert max_diff <= tolerance, (
            f"Score difference exceeds tolerance: max_diff={max_diff}, "
            f"tolerance={tolerance}"
        )

    # Compare other columns (exact match for non-score columns)
    for col in single_df.columns:
        if col not in ["_score"]:  # Skip score column (already compared with tolerance)
            np.testing.assert_array_equal(
                single_df[col].to_numpy(dtype=object),
                distributed_df[col].to_numpy(dtype=object),
                err_msg=f"Column {col} values don't match",
            )

    return True


def validate_distributed_fts(
    dataset: lance.LanceDataset,
    columns_to_index,
    query,
    index_params: dict = None,
    query_params: dict = None,
    tolerance: float = 1e-6,
) -> dict:
    """
    Validate that distributed FTS indexing produces the same results as
    single-machine indexing.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset to test on
    columns_to_index : str or list of str
        The column name(s) to build FTS index on. Can be a single column name
        or a list of column names for multi-column indexing.
    query : str or query object
        The full text query to test with
    index_params : dict, optional
        Parameters for index creation
    query_params : dict, optional
        Parameters for query execution
    tolerance : float, default 1e-6
        Tolerance for score comparison

    Returns
    -------
    dict
        Dictionary with 'single_machine' and 'distributed' results

    Raises
    ------
    AssertionError
        If results don't match
    """
    index_params = index_params or {}
    query_params = query_params or {}

    # Normalize columns_to_index to a list
    if isinstance(columns_to_index, str):
        columns_list = [columns_to_index]
    else:
        columns_list = list(columns_to_index)

    # Build single-machine indices for all required columns
    for column in columns_list:
        dataset.create_scalar_index(
            column=column,
            index_type="INVERTED",
            name=f"{column}_single_idx",
            **index_params,
        )

    # Build distributed indices for all required columns
    distributed_ds = dataset  # Start with the original dataset
    for column in columns_list:
        distributed_ds = build_distributed_fts_index(
            distributed_ds,
            column,
            index_name=f"{column}_distributed_idx",
            **index_params,
        )

    # Execute queries
    single_results = dataset.scanner(full_text_query=query, **query_params).to_table()

    distributed_results = distributed_ds.scanner(
        full_text_query=query, **query_params
    ).to_table()

    # Compare results
    compare_fts_results(single_results, distributed_results, tolerance)

    return {"single_machine": single_results, "distributed": distributed_results}


def assert_distributed_fts_consistency(
    data: pa.Table,
    columns_to_index,
    query,
    tmp_path,
    index_params: dict = None,
    query_params: dict = None,
    tolerance: float = 1e-6,
):
    """
    Assert that distributed FTS indexing produces the same results as
    single-machine indexing.

    This is a streamlined version that eliminates repetitive dataset creation and
    try-catch-print patterns. Uses direct assertions instead of exception handling.

    Parameters
    ----------
    data : pa.Table
        The data to test with
    columns_to_index : str or list of str
        The column name(s) to build FTS index on. Can be a single column name
        or a list of column names for multi-column indexing.
    query : str or query object
        The full text query to test with
    tmp_path : Path
        Temporary path for datasets
    index_params : dict, optional
        Parameters for index creation
    query_params : dict, optional
        Parameters for query execution
    tolerance : float, default 1e-6
        Tolerance for score comparison

    Raises
    ------
    AssertionError
        If distributed and single-machine results don't match
    """
    index_params = index_params or {}
    query_params = query_params or {}

    # Normalize columns_to_index to a list
    if isinstance(columns_to_index, str):
        columns_list = [columns_to_index]
    else:
        columns_list = list(columns_to_index)

    # Create datasets for single-machine and distributed testing
    single_ds = lance.write_dataset(data, tmp_path / "single")
    distributed_ds = lance.write_dataset(data, tmp_path / "distributed")

    # Build single-machine indices for all required columns
    for column in columns_list:
        single_ds.create_scalar_index(
            column=column,
            index_type="INVERTED",
            name=f"{column}_single_idx",
            **index_params,
        )

    # Build distributed indices for all required columns
    distributed_ds_indexed = distributed_ds  # Start with the original dataset
    for column in columns_list:
        distributed_ds_indexed = build_distributed_fts_index(
            distributed_ds_indexed,
            column,
            index_name=f"{column}_distributed_idx",
            **index_params,
        )

    # Execute queries
    single_results = single_ds.scanner(full_text_query=query, **query_params).to_table()

    distributed_results = distributed_ds_indexed.scanner(
        full_text_query=query, **query_params
    ).to_table()

    # Assert results are identical
    compare_fts_results(single_results, distributed_results, tolerance)


def run_fts_distributed_validation_suite(
    test_functions: list, tmp_path_factory, verbose: bool = True
) -> dict:
    """
    Run distributed validation for a suite of FTS test functions.

    Parameters
    ----------
    test_functions : list
        List of FTS test functions to validate
    tmp_path_factory : pytest.TempPathFactory
        Pytest temp path factory for creating test directories
    verbose : bool, default True
        Whether to print detailed progress information

    Returns
    -------
    dict
        Dictionary mapping test function names to validation results
    """
    results = {}

    for test_func in test_functions:
        test_name = test_func.__name__
        if verbose:
            print(f"Running distributed validation for {test_name}...")

        try:
            tmp_path = tmp_path_factory.mktemp(f"distributed_{test_name}")

            # Run the test with distributed validation
            # Note: This is a simplified version - actual implementation would
            # need to extract the test logic and run both single-machine and
            # distributed versions
            test_func(tmp_path)

            results[test_name] = True
            if verbose:
                print(f"✓ {test_name} passed distributed validation")

        except Exception as e:
            results[test_name] = False
            if verbose:
                print(f"✗ {test_name} failed distributed validation: {e}")

    return results


# ============================================================================
# Test Data Generation Functions
# ============================================================================


def generate_multi_fragment_dataset(tmp_path, num_fragments=4, rows_per_fragment=250):
    """
    Generate a test dataset with multiple fragments for testing fragment-level indexing.
    Uses coherent English text similar to "frodo was a puppy" instead of random strings.

    Parameters
    ----------
    tmp_path : Path
        Temporary path for the dataset
    num_fragments : int, default 4
        Number of fragments to create
    rows_per_fragment : int, default 250
        Number of rows per fragment

    Returns
    -------
    lance.LanceDataset
        Dataset with multiple fragments
    """

    # Collection of coherent English sentences for text indexing tests
    # Based on the pattern used in other tests like "frodo was a puppy"
    coherent_sentences = [
        "frodo was a puppy",
        "frodo was a happy puppy",
        "frodo was a very happy puppy",
        "frodo was a puppy with a tail",
        "gandalf carried a staff",
        "gandalf was a wise wizard",
        "gandalf carried his wooden staff",
        "gandalf wore a grey robe",
        "aragorn became the king",
        "aragorn was a brave ranger",
        "aragorn carried his sword",
        "aragorn wore a crown",
        "legolas shot many arrows",
        "legolas was an elf archer",
        "legolas had keen eyes",
        "legolas climbed tall trees",
        "gimli swung his axe",
        "gimli was a dwarf warrior",
        "gimli had a long beard",
        "gimli loved precious gems",
        "sam cooked delicious meals",
        "sam was a loyal friend",
        "sam carried heavy bags",
        "sam tended the garden",
        "pippin played his flute",
        "pippin was very curious",
        "pippin loved second breakfast",
        "pippin climbed apple trees",
        "merry sang cheerful songs",
        "merry was quite clever",
        "merry rode a pony",
        "merry studied old maps",
        "boromir blew his horn",
        "boromir was a proud warrior",
        "boromir defended his city",
        "boromir carried a shield",
        "the ring was very powerful",
        "the ring glowed with fire",
        "the ring whispered secrets",
        "the ring corrupted minds",
        "the shire was peaceful",
        "the shire had green hills",
        "the shire grew fine crops",
        "the shire welcomed visitors",
        "eagles soared through clouds",
        "eagles had sharp talons",
        "eagles nested on peaks",
        "eagles watched the valleys",
        "dragons hoarded gold treasures",
        "dragons breathed hot flames",
        "dragons slept for centuries",
        "dragons guarded ancient caves",
    ]

    def generate_coherent_text():
        """Generate coherent English text by selecting from predefined sentences."""
        return random.choice(coherent_sentences)

    # Create first fragment
    first_data = pa.table(
        {
            "id": pa.array(range(rows_per_fragment)),
            "text": pa.array(
                [generate_coherent_text() for _ in range(rows_per_fragment)]
            ),
            "value": pa.array(np.random.rand(rows_per_fragment) * 100),
        }
    )

    ds = lance.write_dataset(first_data, tmp_path, max_rows_per_file=rows_per_fragment)

    # Add additional fragments
    for i in range(1, num_fragments):
        start_id = i * rows_per_fragment
        fragment_data = pa.table(
            {
                "id": pa.array(range(start_id, start_id + rows_per_fragment)),
                "text": pa.array(
                    [generate_coherent_text() for _ in range(rows_per_fragment)]
                ),
                "value": pa.array(np.random.rand(rows_per_fragment) * 100),
            }
        )
        ds = lance.write_dataset(
            fragment_data, tmp_path, mode="append", max_rows_per_file=rows_per_fragment
        )

    # Verify we have the expected number of fragments
    fragments = ds.get_fragments()
    assert len(fragments) == num_fragments, (
        f"Expected {num_fragments} fragments, got {len(fragments)}"
    )

    return ds


def generate_multi_fragment_bitmap_dataset(
    tmp_path, num_fragments=4, rows_per_fragment=40
):
    """
    Generate a multi-fragment dataset with a low-cardinality integer column
    suitable for distributed bitmap index tests.
    """

    def make_mock_bitmap_table(start_id: int) -> pa.Table:
        ids = list(range(start_id, start_id + rows_per_fragment))
        return pa.table(
            {
                "id": pa.array(ids, type=pa.int32()),
                "category": pa.array([row_id % 5 for row_id in ids], type=pa.int32()),
            }
        )

    ds = lance.write_dataset(
        make_mock_bitmap_table(0),
        tmp_path,
        max_rows_per_file=rows_per_fragment,
    )

    for fragment_idx in range(1, num_fragments):
        ds = lance.write_dataset(
            make_mock_bitmap_table(fragment_idx * rows_per_fragment),
            tmp_path,
            mode="append",
            max_rows_per_file=rows_per_fragment,
        )

    fragments = ds.get_fragments()
    assert len(fragments) == num_fragments, (
        f"Expected {num_fragments} fragments, got {len(fragments)}"
    )

    return ds


# ============================================================================
# Distributed FTS Index Unit Tests
# ============================================================================


def test_build_distributed_fts_index_basic(tmp_path):
    """
    Test basic functionality of build_distributed_fts_index helper function.
    """
    # Generate test dataset with multiple fragments
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=3, rows_per_fragment=100
    )

    # Build distributed FTS index
    distributed_ds = build_distributed_fts_index(
        ds, "text", with_position=False, remove_stop_words=False
    )

    # Verify the index was created
    index_name = "text_distributed_idx"
    stats = distributed_ds.stats.index_stats(index_name)
    assert stats["name"] == index_name
    assert stats["index_type"] == "Inverted"

    # Test that the index works for searching
    results = distributed_ds.scanner(
        full_text_query="frodo",
        columns=["id", "text"],
    ).to_table()

    assert results.num_rows > 0, "No results found for search term 'frodo'"


@pytest.mark.parametrize("index_type", ["INVERTED", "FTS"])
def test_segment_fts(tmp_path, index_type):
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=3, rows_per_fragment=100
    )

    index_name = f"text_{index_type.lower()}_segment_idx"
    segments = [
        ds.create_index_uncommitted(
            column="text",
            index_type=index_type,
            name=index_name,
            fragment_ids=[fragment.fragment_id],
            with_position=False,
            remove_stop_words=False,
        )
        for fragment in ds.get_fragments()
    ]
    committed_ds = ds.commit_existing_index_segments(index_name, "text", segments)

    query = MatchQuery("frodo", "text")
    results_without_index = committed_ds.scanner(
        full_text_query=query,
        columns=["id", "text"],
        use_scalar_index=False,
    ).to_table()
    results_with_index = committed_ds.scanner(
        full_text_query=query,
        columns=["id", "text"],
        use_scalar_index=True,
    ).to_table()

    compare_fts_results(results_without_index, results_with_index)
    assert any(
        idx.name == index_name and idx.index_type == "Inverted"
        for idx in committed_ds.describe_indices()
    )
    assert (
        "FlatMatchQuery"
        not in committed_ds.scanner(
            full_text_query=query,
            use_scalar_index=True,
        ).explain_plan()
    )


def test_compare_fts_results_identical(tmp_path):
    """
    Test compare_fts_results function with identical results.
    """
    # Create identical test results
    data = {
        "id": [1, 2, 3],
        "text": ["frodo was a puppy", "gandalf was wise", "aragorn became king"],
        "_score": [0.95, 0.85, 0.75],
        "_rowid": [0, 1, 2],
    }

    table1 = pa.table(data)
    table2 = pa.table(data)

    # Should return True for identical results
    result = compare_fts_results(table1, table2)
    assert result is True, "Identical results should be considered equal"


def test_compare_fts_results_different_scores(tmp_path):
    """
    Test compare_fts_results function with different scores (should fail).
    """
    data1 = {
        "id": [1, 2, 3],
        "text": ["frodo was a puppy", "gandalf was wise", "aragorn became king"],
        "_score": [0.95, 0.85, 0.75],
        "_rowid": [0, 1, 2],
    }

    data2 = {
        "id": [1, 2, 3],
        "text": ["frodo was a puppy", "gandalf was wise", "aragorn became king"],
        "_score": [0.90, 0.80, 0.70],  # Different scores
        "_rowid": [0, 1, 2],
    }

    table1 = pa.table(data1)
    table2 = pa.table(data2)

    # Should raise AssertionError for different scores
    with pytest.raises(AssertionError, match="Score difference exceeds tolerance"):
        compare_fts_results(table1, table2)


def test_compare_fts_results_different_rowids(tmp_path):
    """
    Test compare_fts_results function with different row IDs (should fail).
    """
    data1 = {
        "id": [1, 2, 3],
        "text": ["frodo was a puppy", "gandalf was wise", "aragorn became king"],
        "_score": [0.95, 0.85, 0.75],
        "_rowid": [0, 1, 2],
    }

    data2 = {
        "id": [1, 2, 3],
        "text": ["frodo was a puppy", "gandalf was wise", "aragorn became king"],
        "_score": [0.95, 0.85, 0.75],
        "_rowid": [0, 1, 3],  # Different row ID
    }

    table1 = pa.table(data1)
    table2 = pa.table(data2)

    # Should raise AssertionError for different row IDs
    with pytest.raises(AssertionError, match="Row ID mismatch"):
        compare_fts_results(table1, table2)


def test_validate_distributed_fts_basic_search(tmp_path):
    """
    Test validate_distributed_fts function with basic search.
    """
    # Generate test dataset with multiple fragments
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=3, rows_per_fragment=100
    )

    # Validate distributed FTS with basic search
    results = validate_distributed_fts(
        ds,
        "text",
        "frodo",
        index_params={"with_position": False, "remove_stop_words": False},
    )

    # Check that we got both results
    assert "single_machine" in results, "Missing single_machine results"
    assert "distributed" in results, "Missing distributed results"

    # Both should have the same number of rows
    single_rows = results["single_machine"].num_rows
    distributed_rows = results["distributed"].num_rows
    assert single_rows == distributed_rows, (
        f"Row count mismatch: {single_rows} vs {distributed_rows}"
    )

    # Should have found some results for 'frodo'
    assert single_rows > 0, "No results found for search term 'frodo'"


def test_validate_distributed_fts_score_consistency(tmp_path):
    """
    Test that distributed FTS produces consistent BM25 scores.
    """
    # Create a dataset with known content for scoring tests
    data = pa.table(
        {
            "id": [1, 2, 3],
            "text": ["lance database test", "full text search", "lance search text"],
        }
    )
    ds = lance.write_dataset(data, tmp_path)

    # Validate distributed FTS with scoring query
    results = validate_distributed_fts(
        ds, "text", "lance search text", index_params={"with_position": False}
    )

    # Check that scores are present and consistent
    single_results = results["single_machine"]
    distributed_results = results["distributed"]

    assert "_score" in single_results.column_names, (
        "Missing _score in single machine results"
    )
    assert "_score" in distributed_results.column_names, (
        "Missing _score in distributed results"
    )

    # Scores should be very close (within 1e-6 tolerance)
    single_scores = single_results.column("_score").to_pylist()
    distributed_scores = distributed_results.column("_score").to_pylist()

    for i, (s_score, d_score) in enumerate(zip(single_scores, distributed_scores)):
        score_diff = abs(s_score - d_score)
        assert score_diff <= 1e-6, f"Score difference at index {i}: {score_diff} > 1e-6"


def test_validate_distributed_fts_empty_results(tmp_path):
    """
    Test validate_distributed_fts function with query that returns no results.
    """
    # Generate test dataset
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=2, rows_per_fragment=50
    )

    # Search for something that doesn't exist
    results = validate_distributed_fts(
        ds, "text", "nonexistent_term_xyz", index_params={"with_position": False}
    )

    # Both should return empty results
    assert results["single_machine"].num_rows == 0, (
        "Single machine should return 0 results"
    )
    assert results["distributed"].num_rows == 0, "Distributed should return 0 results"


def test_validate_distributed_fts_large_dataset(tmp_path):
    """
    Test validate_distributed_fts function with larger dataset.
    """
    # Generate larger test dataset
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=5, rows_per_fragment=200
    )

    # Validate distributed FTS
    results = validate_distributed_fts(
        ds,
        "text",
        "gandalf",
        index_params={"with_position": False, "remove_stop_words": False},
    )

    # Should find results and they should match
    single_rows = results["single_machine"].num_rows
    distributed_rows = results["distributed"].num_rows

    assert single_rows > 0, "Should find results for 'gandalf'"
    assert single_rows == distributed_rows, (
        f"Row count mismatch: {single_rows} vs {distributed_rows}"
    )


# ============================================================================
# Advanced Query Tests for Distributed FTS
# ============================================================================


def test_distributed_fts_phrase_query_validation(tmp_path):
    """
    Test distributed FTS validation with phrase queries.
    """
    data = pa.table(
        {
            "text": [
                "frodo was a puppy",
                "frodo was a happy puppy",
                "frodo was a very happy puppy",
                "frodo was a puppy with a tail",
            ]
        }
    )
    ds = lance.write_dataset(data, tmp_path)

    # Test phrase query validation
    results = validate_distributed_fts(
        ds,
        "text",
        PhraseQuery("frodo was a puppy", "text"),
        index_params={"with_position": True, "remove_stop_words": False},
    )

    # Should find matching phrase results
    assert results["single_machine"].num_rows == results["distributed"].num_rows
    assert results["single_machine"].num_rows == 2  # Two exact matches


def test_distributed_fts_boolean_query_validation(tmp_path):
    """
    Test distributed FTS validation with boolean queries.
    """
    data = pa.table(
        {
            "text": [
                "frodo was a puppy",
                "frodo was a happy puppy",
                "frodo was a puppy with a tail",
            ]
        }
    )
    ds = lance.write_dataset(data, tmp_path)

    # Test boolean query validation
    query = MatchQuery("puppy", "text") & MatchQuery("happy", "text")
    results = validate_distributed_fts(
        ds, "text", query, index_params={"with_position": False}
    )

    # Should find only the happy puppy
    assert results["single_machine"].num_rows == results["distributed"].num_rows
    assert results["single_machine"].num_rows == 1


def test_distributed_fts_fuzzy_query_validation(tmp_path):
    """
    Test distributed FTS validation with fuzzy queries.
    """
    data = pa.table({"text": ["foo", "food", "fob", "focus", "foul"]})
    ds = lance.write_dataset(data, tmp_path)

    # Test fuzzy query validation
    results = validate_distributed_fts(
        ds,
        "text",
        MatchQuery("foo", "text", fuzziness=1),
        index_params={"with_position": False},
    )

    # Should find fuzzy matches
    assert results["single_machine"].num_rows == results["distributed"].num_rows
    assert results["single_machine"].num_rows > 1  # Multiple fuzzy matches


def test_distributed_fts_with_filter_validation(tmp_path):
    """
    Test distributed FTS validation with filters.
    """
    data = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "text": [
                "lance database test",
                "full text search",
                "lance search text",
                "some other content",
                "more lance content",
            ],
        }
    )
    ds = lance.write_dataset(data, tmp_path)

    # Create BTREE index for filtering
    ds.create_scalar_index("id", "BTREE")

    # Test FTS with filter validation
    results = validate_distributed_fts(
        ds,
        "text",
        "lance",
        index_params={"with_position": False},
        query_params={"filter": "id <= 3", "prefilter": True},
    )

    # Should find filtered results
    assert results["single_machine"].num_rows == results["distributed"].num_rows
    assert results["single_machine"].num_rows == 2  # Only id 1 and 3 have "lance"


def test_distributed_fts_multi_match_validation(tmp_path):
    """
    Test distributed FTS validation with multi-field matching.
    """
    data = pa.table(
        {
            "title": ["title common", "title hello", "title vector"],
            "content": ["content world", "content database", "content common"],
        }
    )
    ds = lance.write_dataset(data, tmp_path)

    # Test multi-match query validation
    results = validate_distributed_fts(
        ds,
        ["title", "content"],  # Index both fields for multi-match query
        MultiMatchQuery("common", ["title", "content"]),
        index_params={"with_position": False},
    )

    # Note: For multi-match, we need to create indices on both fields
    # This is a simplified test - in practice, we'd need more complex setup
    assert results["single_machine"].num_rows == results["distributed"].num_rows


# ============================================================================
# Integration with Existing High-Priority FTS Tests
# ============================================================================


def test_distribute_fts_index_build(tmp_path):
    """
    This test creates indices on individual fragments
    and then commits them as a single index.
    """
    # Generate test dataset with multiple fragments
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=4, rows_per_fragment=250
    )

    import uuid

    index_id = str(uuid.uuid4())
    index_name = "multiple_fragment_idx"

    fragments = ds.get_fragments()
    fragment_ids = [fragment.fragment_id for fragment in fragments]

    for fragment in ds.get_fragments():
        fragment_id = fragment.fragment_id

        # Use the new fragment_ids and index_uuid parameters
        ds.create_scalar_index(
            column="text",
            index_type="INVERTED",
            name=index_name,
            replace=False,
            index_uuid=index_id,
            fragment_ids=[fragment_id],
            remove_stop_words=False,
        )

        # For fragment-level indexing, we expect the method to return successfully
        # but not commit the index yet
        print(f"Fragment {fragment_id} index created successfully")

    # Merge the inverted index metadata
    ds.merge_index_metadata(index_id, index_type="INVERTED")

    # Create an Index object using the new dataclass format
    from lance.dataset import Index

    # Get the schema field for the indexed column
    # Only use for non nested struct schema
    field_id = ds.schema.get_field_index("text")

    index = Index(
        uuid=index_id,
        name=index_name,
        fields=[field_id],  # Use field index instead of field object
        dataset_version=ds.version,
        fragment_ids=set(fragment_ids),
        index_version=0,
    )

    # Create the index operation
    create_index_op = lance.LanceOperation.CreateIndex(
        new_indices=[index],
        removed_indices=[],
    )

    # Commit the index
    ds_committed = lance.LanceDataset.commit(
        ds.uri,
        create_index_op,
        read_version=ds.version,
    )

    # Verify the index was created and is functional
    stats = ds_committed.stats.index_stats(index_name)
    assert stats["name"] == index_name
    assert stats["index_type"] == "Inverted"

    # Test that the index works for searching
    # Get a sample text from the dataset to search for
    sample_data = ds_committed.take([0], columns=["text"])
    sample_text = sample_data.column(0)[0].as_py()
    search_word = sample_text.split()[0] if sample_text.split() else "test"

    # Perform a full-text search to verify the index works
    results = ds_committed.scanner(
        full_text_query=search_word,
        columns=["id", "text"],
    ).to_table()

    # We should get at least one result since we searched for a word from the dataset
    assert results.num_rows > 0, f"No results found for search term '{search_word}'"


def test_fragment_ids_parameter_validation(tmp_path):
    """
    Test validation of fragment_ids parameter.
    """
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=2, rows_per_fragment=100
    )

    # Test with valid fragment IDs
    fragments = ds.get_fragments()
    valid_fragment_id = fragments[0].fragment_id

    # This should work without errors
    ds.create_scalar_index(
        column="text",
        index_type="INVERTED",
        fragment_ids=[valid_fragment_id],
    )

    # Test with invalid fragment ID (should handle gracefully)
    # Note: The exact behavior for invalid fragment IDs may vary
    # This test ensures the parameter is properly passed through
    try:
        ds.create_scalar_index(
            column="text",
            index_type="INVERTED",
            fragment_ids=[999999],  # Non-existent fragment ID
        )
    except Exception as e:
        # It's acceptable for this to fail with an appropriate error
        print(f"Expected error for invalid fragment ID: {e}")


def test_backward_compatibility_no_fragment_ids(tmp_path):
    """
    Test that the API remains backward compatible when fragment_ids is not provided.
    """
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=2, rows_per_fragment=100
    )

    # This should work exactly as before (full dataset indexing)
    ds.create_scalar_index(
        column="text",
        index_type="INVERTED",
        name="full_dataset_idx",
        remove_stop_words=False,
    )

    # Verify the index was created
    indices = ds.describe_indices()
    assert len(indices) == 1
    assert indices[0].name == "full_dataset_idx"
    assert indices[0].index_type == "Inverted"

    # Test that the index works
    sample_data = ds.take([0], columns=["text"])
    sample_text = sample_data.column(0)[0].as_py()
    search_word = sample_text.split()[0] if sample_text.split() else "test"

    results = ds.scanner(full_text_query=search_word).to_table()
    assert results.num_rows > 0


def test_backward_compatibility_changed_index_protos(tmp_path):
    path = (
        Path(__file__).parent.parent.parent.parent
        / "test_data"
        / "0.36.0"
        / "btree_in_index_pkg.lance"
    )
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    ds = lance.dataset(tmp_path)

    indices = ds.describe_indices()
    assert len(indices) == 1
    assert indices[0].name == "x_idx"
    assert indices[0].index_type == "BTree"

    results = ds.scanner(filter="x = 100").to_table()
    assert results.num_rows == 1
    assert results.column("x").to_pylist() == [100]


def test_distribute_btree_index_build(tmp_path):
    """
    Test distributed B-tree index build with segmented index commit.
    This test creates B-tree segments on individual fragments and then
    commits them as a single logical index.
    """
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=4, rows_per_fragment=10000
    )

    index_name = "btree_multiple_fragment_idx"
    ds_committed = _commit_segmented_btree_index(ds, "id", index_name)

    stats = ds_committed.stats.index_stats(index_name)
    assert stats["name"] == index_name
    assert stats["index_type"] == "BTree"

    test_id = 100
    results = ds_committed.scanner(
        filter=f"id = {test_id}",
        columns=["id", "text"],
    ).to_table()

    assert results.num_rows == 1, f"No results found for id = {test_id}"

    results_range = ds_committed.scanner(
        filter="id >= 200 AND id < 800",
        columns=["id", "text"],
    ).to_table()

    assert results_range.num_rows > 0, "No results found for range query"

    reference_ds = generate_multi_fragment_dataset(
        tmp_path / "reference", num_fragments=4, rows_per_fragment=10000
    )

    reference_ds.create_scalar_index(
        column="id",
        index_type="BTREE",
        name="reference_btree_idx",
    )

    reference_results = reference_ds.scanner(
        filter=f"id = {test_id}",
        columns=["id", "text"],
    ).to_table()

    assert results.num_rows == reference_results.num_rows, (
        f"Distributed index returned {results.num_rows} results, "
        f"but complete index returned {reference_results.num_rows} results"
    )

    reference_range_results = reference_ds.scanner(
        filter="id >= 200 AND id < 800",
        columns=["id", "text"],
    ).to_table()

    assert results_range.num_rows == reference_range_results.num_rows, (
        f"Distributed index range query returned {results_range.num_rows} results, "
        f"but complete index returned {reference_range_results.num_rows} results"
    )


def test_bitmap_uncommitted_segments_can_be_committed_from_python(tmp_path):
    dataset_path = tmp_path / "bitmap_segments.lance"
    ds = generate_multi_fragment_bitmap_dataset(
        dataset_path, num_fragments=4, rows_per_fragment=40
    )

    index_name = "bitmap_segment_idx"
    fragment_ids = [fragment.fragment_id for fragment in ds.get_fragments()]
    fragment_groups = [
        fragment_ids[idx : idx + 2] for idx in range(0, len(fragment_ids), 2)
    ]
    assert len(fragment_groups) >= 2

    staged_segments = [
        ds.create_index_uncommitted(
            column="category",
            index_type="BITMAP",
            name=index_name,
            fragment_ids=fragment_group,
        )
        for fragment_group in fragment_groups
    ]

    assert len({segment.uuid for segment in staged_segments}) == len(staged_segments)
    for segment, fragment_group in zip(staged_segments, fragment_groups):
        assert segment.fragment_ids == set(fragment_group)
        assert any(file.path == "bitmap_page_lookup.lance" for file in segment.files)
        assert all(not file.path.startswith("part_") for file in segment.files)

    merged_segment = ds.merge_existing_index_segments(staged_segments)
    assert merged_segment.uuid not in {segment.uuid for segment in staged_segments}
    assert merged_segment.fragment_ids == set(fragment_ids)
    assert any(file.path == "bitmap_page_lookup.lance" for file in merged_segment.files)
    assert all(not file.path.startswith("part_") for file in merged_segment.files)

    ds = ds.commit_existing_index_segments(index_name, "category", [merged_segment])
    descriptions = {index.name: index for index in ds.describe_indices()}
    assert len(descriptions[index_name].segments) == 1

    filter_expr = "category = 3"
    without_index = ds.scanner(
        filter=filter_expr,
        columns=["id", "category"],
        use_scalar_index=False,
    ).to_table()
    with_index = ds.scanner(
        filter=filter_expr,
        columns=["id", "category"],
        use_scalar_index=True,
    ).to_table()

    assert with_index.num_rows == without_index.num_rows
    assert with_index["id"].to_pylist() == without_index["id"].to_pylist()
    assert set(with_index["category"].to_pylist()) == {3}
    assert (
        "ScalarIndexQuery"
        in ds.scanner(filter=filter_expr, use_scalar_index=True).explain_plan()
    )


def test_zonemap_fragment_ids_parameter_validation(tmp_path):
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=2, rows_per_fragment=100
    )

    fragment_ids = [fragment.fragment_id for fragment in ds.get_fragments()]
    with pytest.raises(ValueError, match="create_index_uncommitted"):
        ds.create_scalar_index(
            column="id",
            index_type="ZONEMAP",
            fragment_ids=[fragment_ids[0]],
        )


def test_zonemap_segment_merge_and_commit_from_python(tmp_path):
    rows_per_fragment = 20_000
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=4, rows_per_fragment=rows_per_fragment
    )

    index_name = "id_zonemap_segments"
    fragment_ids = [fragment.fragment_id for fragment in ds.get_fragments()]
    staged_segments = [
        ds.create_index_uncommitted(
            column="id",
            index_type="ZONEMAP",
            name=index_name,
            fragment_ids=[fragment_id],
        )
        for fragment_id in fragment_ids
    ]

    assert len({segment.uuid for segment in staged_segments}) == len(staged_segments)
    for segment, fragment_id in zip(staged_segments, fragment_ids):
        files = segment.files
        assert files is not None
        assert segment.fragment_ids == {fragment_id}
        assert any(file.path == "zonemap.lance" for file in files)
        assert all(not file.path.startswith("part_") for file in files)

    merged_segment = ds.merge_existing_index_segments(staged_segments)
    merged_files = merged_segment.files
    assert merged_files is not None
    assert merged_segment.uuid not in {segment.uuid for segment in staged_segments}
    assert merged_segment.fragment_ids == set(fragment_ids)
    assert any(file.path == "zonemap.lance" for file in merged_files)
    assert all(not file.path.startswith("part_") for file in merged_files)

    ds = ds.commit_existing_index_segments(index_name, "id", [merged_segment])
    descriptions = {index.name: index for index in ds.describe_indices()}
    assert descriptions[index_name].index_type == "ZoneMap"
    assert len(descriptions[index_name].segments) == 1

    filter_expr = "id >= 8200 AND id < 8300"
    without_index = ds.scanner(
        filter=filter_expr,
        columns=["id", "text"],
        use_scalar_index=False,
    ).to_table()
    with_index = ds.scanner(
        filter=filter_expr,
        columns=["id", "text"],
        use_scalar_index=True,
    ).to_table()

    assert with_index.num_rows == without_index.num_rows
    assert with_index["id"].to_pylist() == without_index["id"].to_pylist()
    assert (
        "ScalarIndexQuery"
        in ds.scanner(filter=filter_expr, use_scalar_index=True).explain_plan()
    )


def test_merge_index_metadata_btree_soft_break(tmp_path):
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=2, rows_per_fragment=100
    )

    with pytest.raises(ValueError, match="no longer supports merge_index_metadata"):
        ds.merge_index_metadata(str(uuid.uuid4()), index_type="BTREE")


def test_btree_fragment_ids_parameter_validation(tmp_path):
    """
    Test validation of fragment_ids parameter for B-tree indices.
    """
    ds = generate_multi_fragment_dataset(
        tmp_path, num_fragments=2, rows_per_fragment=10000
    )

    fragments = ds.get_fragments()
    valid_fragment_id = fragments[0].fragment_id

    # create_scalar_index no longer accepts fragment_ids for BTREE; distributed
    # builds must go through the segmented create_index_uncommitted path.
    with pytest.raises(ValueError, match="create_index_uncommitted"):
        ds.create_scalar_index(
            column="id",
            index_type="BTREE",
            fragment_ids=[valid_fragment_id],
        )

    with pytest.raises(ValueError, match="index_uuid is no longer accepted"):
        ds.create_index_uncommitted(
            column="id",
            index_type="BTREE",
            fragment_ids=[valid_fragment_id],
            index_uuid=str(uuid.uuid4()),
        )

    # Building one uncommitted segment for a valid fragment should work and
    # return the segment metadata without committing it.
    segment = ds.create_index_uncommitted(
        column="id",
        index_type="BTREE",
        fragment_ids=[valid_fragment_id],
    )
    assert segment.fragment_ids == {valid_fragment_id}


@pytest.mark.parametrize(
    "test_name,filter_expr",
    [
        # Test 1: Boundary values at fragment edges
        ("First value", "id = 0"),
        ("Fragment 0 last value", "id = 9999"),
        ("Fragment 1 first value", "id = 10000"),
        ("Fragment 1 last value", "id = 19999"),
        ("Fragment 2 first value", "id = 20000"),
        ("Last value", "id = 29999"),
        # Test 2: Values in the middle of fragments
        ("Fragment 0 middle", "id = 5000"),
        ("Fragment 1 middle", "id = 15000"),
        ("Fragment 2 middle", "id = 25000"),
        # Test 3: Range queries within single fragments
        ("Range within fragment 0", "id >= 10 AND id < 20"),
        ("Range within fragment 1", "id >= 10010 AND id < 10020"),
        ("Range within fragment 2", "id >= 20010 AND id < 20020"),
        # Test 4: Range queries spanning multiple fragments
        ("Cross fragment 0-1", "id >= 9995 AND id < 10005"),
        ("Cross fragment 1-2", "id >= 19995 AND id < 20005"),
        ("Cross all fragments", "id >= 5000 AND id < 25000"),
        # Test 5: Edge cases
        ("Non-existent small value", "id = -1"),
        ("Non-existent large value", "id = 30100"),
        ("Large range", "id >= 0 AND id < 30000"),
        # Test 6: Comparison operators
        ("Less than boundary", "id < 10000"),
        ("Greater than boundary", "id > 19999"),
        ("Less than or equal", "id <= 10050"),
        ("Greater than or equal", "id >= 10050"),
    ],
)
def test_btree_query_comparison_parametrized(
    btree_comparison_datasets, test_name, filter_expr
):
    """
    Parametrized B-tree index query comparison test.

    Compares segmented fragment-built BTree results with a complete BTree index.
    """
    fragment_ds = btree_comparison_datasets["fragment_ds"]
    complete_ds = btree_comparison_datasets["complete_ds"]

    fragment_results = fragment_ds.scanner(
        filter=filter_expr,
        columns=["id", "text"],
    ).to_table()

    complete_results = complete_ds.scanner(
        filter=filter_expr,
        columns=["id", "text"],
    ).to_table()

    assert fragment_results.num_rows == complete_results.num_rows, (
        f"Test '{test_name}' failed: Fragment index "
        f"returned {fragment_results.num_rows} rows, "
        f"but complete index returned {complete_results.num_rows}"
        f" rows for filter: {filter_expr}"
    )

    if fragment_results.num_rows > 0:
        fragment_ids = sorted(fragment_results.column("id").to_pylist())
        complete_ids = sorted(complete_results.column("id").to_pylist())

        assert fragment_ids == complete_ids, (
            f"Test '{test_name}' failed: Fragment index "
            f"and complete index returned different results for filter: {filter_expr}"
        )


def test_fts_flat_fallback_matches_wand(tmp_path):
    # Repro: when filter matches < 10%, FTS fell back to flat search and missed results.
    # Two-term query increases reproduction likelihood.
    # Compare default scan (prefilter=True) vs WAND-path (prefilter=False).

    # Deterministic data
    random.seed(123)
    np.random.seed(123)

    n = 2000
    # 5% category 'A' to trigger fallback (default threshold 10%)
    categories = np.where(np.random.rand(n) < 0.05, "A", "B")

    vocab = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
    ]
    needle1 = "needle"
    needle2 = "pin"

    texts = []
    a_indices = [i for i, c in enumerate(categories) if c == "A"]
    # Ensure many matches within the filtered set (both terms present)
    a_with_needle = set(
        random.sample(
            a_indices,
            max(1, len(a_indices) // 2),
        )
    )
    for i in range(n):
        toks = random.choices(vocab, k=random.randint(5, 12))
        if i in a_with_needle:
            # Add both terms for many A rows
            toks.append(needle1)
            toks.append(needle2)
        # Also sprinkle some needles in B rows to make ranking realistic
        elif categories[i] == "B" and random.random() < 0.03:
            # Randomly add one of the terms to some B rows
            toks.append(needle1 if random.random() < 0.5 else needle2)
        texts.append(" ".join(toks))

    tbl = pa.table(
        {
            "id": np.arange(n, dtype=np.int64),
            "category": pa.array(categories.astype(str)),
            "text": pa.array(texts, type=pa.large_string()),
        }
    )

    ds_path = tmp_path / "fts_fallback.lance"
    ds = lance.write_dataset(tbl, ds_path)
    ds.create_scalar_index("category", index_type="BTREE")
    ds.create_scalar_index("text", index_type="INVERTED")

    # Sanity: ensure there are hits in the filtered subset
    a_hit_count = sum((needle1 in texts[i]) or (needle2 in texts[i]) for i in a_indices)
    assert a_hit_count > 0

    # Two words query
    query = f"{needle1} {needle2}"
    filter_expr = "category = 'A'"
    limit = 10

    # flat-fallback path (prefilter=True)
    tbl_flat = ds.scanner(
        columns=["_rowid", "_score"],
        full_text_query=query,
        filter=filter_expr,
        limit=limit,
        prefilter=True,
    ).to_table()
    flat_pairs = list(
        zip(
            tbl_flat.column("_rowid").to_pylist(),
            tbl_flat.column("_score").to_pylist(),
        )
    )

    # WAND path (prefilter=False prevents building a mask, so no flat fallback)
    tbl_wand = (
        ds.scanner(
            columns=["_rowid", "_score"],
            full_text_query=query,
            filter=filter_expr,
            limit=n,
            prefilter=False,
        )
        .to_table()
        .slice(0, limit)
    )
    wand_pairs = list(
        zip(
            tbl_wand.column("_rowid").to_pylist(),
            tbl_wand.column("_score").to_pylist(),
        )
    )

    flat_scores = [s for _, s in flat_pairs]
    wand_scores = [s for _, s in wand_pairs]
    # we compare only scores because it's possible two rows have the same score
    assert flat_scores == wand_scores, (
        f"Flat FTS fallback differs from WAND (scores).\n"
        f"flat scores={flat_scores}\nwand scores={wand_scores}"
    )

    tbl_limited_wand = ds.scanner(
        columns=["_rowid", "_score"],
        full_text_query=query,
        limit=limit,
    ).to_table()

    tbl_full_wand = (
        ds.scanner(
            columns=["_rowid", "_score"],
            full_text_query=query,
        )
        .to_table()
        .slice(0, limit)
    )

    limited_wand_pairs = list(
        zip(
            tbl_limited_wand.column("_rowid").to_pylist(),
            tbl_limited_wand.column("_score").to_pylist(),
        )
    )
    full_wand_pairs = list(
        zip(
            tbl_full_wand.column("_rowid").to_pylist(),
            tbl_full_wand.column("_score").to_pylist(),
        )
    )
    limited_wand_scores = [s for _, s in limited_wand_pairs]
    full_wand_scores = [s for _, s in full_wand_pairs]
    assert limited_wand_scores == full_wand_scores, (
        f"Limited WAND scores differ from full WAND scores.\n"
        f"limited scores={limited_wand_scores}\nfull scores={full_wand_scores}"
    )


def test_scan_statistics_callback(tmp_path):
    """Test that scan_stats_callback receives all expected fields."""
    # Create a simple dataset
    table = pa.table(
        {
            "id": range(100),
            "value": np.random.randn(100),
        }
    )

    dataset = lance.write_dataset(table, tmp_path / "test_stats.lance")

    scan_stats = None

    def scan_stats_callback(stats: lance.ScanStatistics):
        nonlocal scan_stats
        scan_stats = stats

    result = dataset.scanner(scan_stats_callback=scan_stats_callback).to_table()
    assert result.num_rows == 100
    assert scan_stats is not None, "Callback should have been called"
    assert isinstance(scan_stats.iops, int)
    assert isinstance(scan_stats.requests, int)
    assert isinstance(scan_stats.bytes_read, int)
    assert isinstance(scan_stats.indices_loaded, int)
    assert isinstance(scan_stats.parts_loaded, int)
    assert isinstance(scan_stats.index_comparisons, int)
    assert isinstance(scan_stats.all_counts, dict)

    # Verify we got some I/O activity
    assert scan_stats.iops > 0, "Expected some I/O operations"
    assert scan_stats.bytes_read > 0, "Expected some bytes read"

    # Verify all_counts contains the standard metrics
    assert isinstance(scan_stats.all_counts, dict)
    for key, value in scan_stats.all_counts.items():
        assert isinstance(key, str)
        assert isinstance(value, int)


def test_nested_field_btree_index(tmp_path):
    """Test BTREE index creation and querying on nested fields"""
    # Create a dataset with nested structure
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(
                "meta",
                pa.struct(
                    [pa.field("lang", pa.string()), pa.field("version", pa.int32())]
                ),
            ),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "meta": [
                {"lang": "en", "version": 1},
                {"lang": "fr", "version": 2},
                {"lang": "en", "version": 1},
                {"lang": "es", "version": 3},
                {"lang": "fr", "version": 2},
            ],
        },
        schema=schema,
    )

    # Create dataset
    uri = tmp_path / "test_nested_btree"
    dataset = lance.write_dataset(data, uri)

    # Create BTREE index on nested string column
    dataset.create_scalar_index(column="meta.lang", index_type="BTREE")

    # Verify index was created
    indices = dataset.describe_indices()
    assert len(indices) == 1
    assert indices[0].field_names == ["meta.lang"]
    assert indices[0].index_type == "BTree"

    # Test query using the index - filter for English language
    result = dataset.scanner(filter="meta.lang = 'en'").to_table()
    assert len(result) == 2
    for i in range(len(result)):
        assert result["meta"][i]["lang"].as_py() == "en"

    # Test query for French language
    result = dataset.scanner(filter="meta.lang = 'fr'").to_table()
    assert len(result) == 2
    for i in range(len(result)):
        assert result["meta"][i]["lang"].as_py() == "fr"

    # Verify the index is being used
    plan = dataset.scanner(filter="meta.lang = 'en'").explain_plan()
    assert "ScalarIndexQuery" in plan

    # Write additional data to the dataset
    new_data = pa.table(
        {
            "id": [6, 7, 8],
            "meta": [
                {"lang": "de", "version": 4},
                {"lang": "en", "version": 2},
                {"lang": "de", "version": 4},
            ],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(new_data, uri, mode="append")

    # Verify query still works after appending data
    result = dataset.scanner(filter="meta.lang = 'en'").to_table()
    assert len(result) == 3, f"Expected 3 English records, got {len(result)}"
    for i in range(len(result)):
        assert result["meta"][i]["lang"].as_py() == "en"

    # Test query for new German language entries
    result = dataset.scanner(filter="meta.lang = 'de'").to_table()
    assert len(result) == 2
    for i in range(len(result)):
        assert result["meta"][i]["lang"].as_py() == "de"

    # Test optimize_indices with nested field BTREE index
    dataset.optimize.optimize_indices()

    # Verify query still works after optimization
    result = dataset.scanner(filter="meta.lang = 'en'").to_table()
    assert len(result) == 3
    result = dataset.scanner(filter="meta.lang = 'de'").to_table()
    assert len(result) == 2

    # Create BTREE index on nested integer column
    dataset.create_scalar_index(column="meta.version", index_type="BTREE", replace=True)

    # Test query using the version index
    result = dataset.scanner(filter="meta.version = 1").to_table()
    assert len(result) == 2
    for i in range(len(result)):
        assert result["meta"][i]["version"].as_py() == 1

    # Test query for version 4 (new data)
    result = dataset.scanner(filter="meta.version = 4").to_table()
    assert len(result) == 2
    for i in range(len(result)):
        assert result["meta"][i]["version"].as_py() == 4

    # Verify total row count
    total = dataset.count_rows()
    assert total == 8, f"Expected 8 total rows, got {total}"


def test_nested_field_fts_index(tmp_path):
    """Test FTS index creation and querying on nested fields"""
    # Create dataset with nested text field
    data = pa.table(
        {
            "id": range(100),
            "data": pa.StructArray.from_arrays(
                [
                    pa.array(
                        [f"document {i} about lance database" for i in range(100)]
                    ),
                    pa.array([f"label_{i}" for i in range(100)]),
                ],
                names=["text", "label"],
            ),
        }
    )

    ds = lance.write_dataset(data, tmp_path)

    # Create FTS index on nested field
    ds.create_scalar_index("data.text", index_type="INVERTED", with_position=False)

    # Verify index was created
    indices = ds.describe_indices()
    assert len(indices) == 1
    assert indices[0].field_names == ["data.text"]
    assert indices[0].index_type == "Inverted"

    # Test full text search on nested field
    results = ds.to_table(full_text_query="lance")
    assert results.num_rows == 100

    # Verify the results contain the expected text
    for i in range(results.num_rows):
        text = results["data"][i]["text"].as_py()
        assert "lance" in text

    # Test with prefilter using another nested field
    results = ds.to_table(
        full_text_query="database",
        filter="data.label = 'label_5'",
        prefilter=True,
    )
    assert results.num_rows == 1
    assert results["id"][0].as_py() == 5

    # Test optimize_indices with nested field FTS index
    # Append more data
    new_data = pa.table(
        {
            "id": range(100, 150),
            "data": pa.StructArray.from_arrays(
                [
                    pa.array(
                        [f"document {i} about lance search" for i in range(100, 150)]
                    ),
                    pa.array([f"label_{i}" for i in range(100, 150)]),
                ],
                names=["text", "label"],
            ),
        }
    )
    ds = lance.write_dataset(new_data, tmp_path, mode="append")

    # Optimize indices
    ds.optimize.optimize_indices()

    # Verify search still works after optimization
    results = ds.to_table(full_text_query="lance")
    assert results.num_rows == 150

    results = ds.to_table(full_text_query="search")
    assert results.num_rows == 50


def test_nested_field_bitmap_index(tmp_path):
    """Test BITMAP index creation and querying on nested fields"""
    # Create dataset with nested categorical field
    data = pa.table(
        {
            "id": range(100),
            "attributes": pa.StructArray.from_arrays(
                [
                    pa.array(["red", "green", "blue"][i % 3] for i in range(100)),
                    pa.array([f"size_{i % 5}" for i in range(100)]),
                ],
                names=["color", "size"],
            ),
        }
    )

    ds = lance.write_dataset(data, tmp_path)

    # Create BITMAP index on nested field
    ds.create_scalar_index("attributes.color", index_type="BITMAP")

    # Verify index was created
    indices = ds.describe_indices()
    assert len(indices) == 1
    assert indices[0].field_names == ["attributes.color"]
    assert indices[0].index_type == "Bitmap"

    # Test equality query
    results = ds.to_table(filter="attributes.color = 'red'", prefilter=True)
    assert results.num_rows == 34  # 0, 3, 6, 9, ... 99 (34 values)

    # Verify the index is being used
    plan = ds.scanner(filter="attributes.color = 'red'", prefilter=True).explain_plan()
    assert "ScalarIndexQuery" in plan

    # Test with different color
    results = ds.to_table(filter="attributes.color = 'green'", prefilter=True)
    assert results.num_rows == 33  # 1, 4, 7, 10, ... 97 (33 values)

    results = ds.to_table(filter="attributes.color = 'blue'", prefilter=True)
    assert results.num_rows == 33  # 2, 5, 8, 11, ... 98 (33 values)

    # Test optimize_indices with nested field BITMAP index
    new_data = pa.table(
        {
            "id": range(100, 150),
            "attributes": pa.StructArray.from_arrays(
                [
                    pa.array(["red", "green", "blue"][i % 3] for i in range(50)),
                    pa.array([f"size_{i % 5}" for i in range(50)]),
                ],
                names=["color", "size"],
            ),
        }
    )
    ds = lance.write_dataset(new_data, tmp_path, mode="append")

    # Optimize indices
    ds.optimize.optimize_indices()

    # Verify query still works after optimization
    results = ds.to_table(filter="attributes.color = 'red'", prefilter=True)
    assert results.num_rows == 51  # 34 + 17 from new data


def test_json_inverted_match_query(tmp_path):
    # Prepare dataset with JSON data
    json_data = [
        {
            "Title": "HarryPotter Chapter One",
            "Content": "Once upon a time, there was a boy named Harry.",
            "Author": "J.K. Rowling",
            "Price": 99,
            "Language": ["english", "french"],
        },
        {
            "Title": "HarryPotter Chapter Two",
            "Content": "Nearly ten years had passed since the Dursleys had woken up...",
            "Author": "J.K. Rowling",
            "Price": 128,
            "Language": ["english", "chinese"],
        },
        {
            "Title": "The Hobbit",
            "Content": "In a hole in the ground there lived a hobbit.",
            "Author": "J.R.R. Tolkien",
            "Price": 89,
            "Language": ["english"],
        },
    ]

    # Convert to JSON strings
    json_strings = pa.array([json.dumps(doc) for doc in json_data], type=pa.json_())
    table = pa.table({"json_col": json_strings, "id": range(len(json_data))})
    dataset = lance.write_dataset(table, tmp_path)

    # Create inverted index with JSON tokenizer
    dataset.create_scalar_index(
        "json_col",
        index_type="INVERTED",
        base_tokenizer="simple",
        max_token_length=10,
        stem=True,
        lower_case=True,
        remove_stop_words=True,
    )

    # Test match query with token exceeding max_token_length
    results = dataset.to_table(
        full_text_query=MatchQuery("Title,str,harrypotter", "json_col")
    )
    assert results.num_rows == 0

    # Test stemming
    results = dataset.to_table(
        full_text_query=MatchQuery("Content,str,onc", "json_col")
    )
    assert results.num_rows == 1

    # Test language match
    results = dataset.to_table(
        full_text_query=MatchQuery("Language,str,english", "json_col")
    )
    assert results.num_rows == 3

    # Test author match
    results = dataset.to_table(
        full_text_query=MatchQuery("Author,str,tolkien", "json_col")
    )
    assert results.num_rows == 1


@pytest.mark.parametrize("fts_format_version", ["1", "2"])
def test_describe_indices(tmp_path, monkeypatch, fts_format_version):
    monkeypatch.setenv("LANCE_FTS_FORMAT_VERSION", fts_format_version)
    data = pa.table(
        {
            "id": range(100),
            "text": [f"document {i} about lance database" for i in range(100)],
            "bitmap": range(100),
            "bloomfilter": range(100),
            "btree": range(100),
            "json": pa.array(
                [json.dumps({"key": f"value_{i}"}) for i in range(100)], pa.json_()
            ),
            "ngram": [f"document {i}" for i in range(100)],
            "zonemap": range(100),
        }
    )
    ds = lance.write_dataset(data, tmp_path)
    ds.create_scalar_index("text", index_type="INVERTED")
    indices = ds.describe_indices()
    assert len(indices) == 1

    assert indices[0].name == "text_idx"
    assert indices[0].type_url == "/lance.table.InvertedIndexDetails"
    assert indices[0].index_type == "Inverted"
    assert indices[0].num_rows_indexed == 100
    assert indices[0].fields == [1]
    assert indices[0].field_names == ["text"]
    assert len(indices[0].segments) == 1
    assert indices[0].segments[0].uuid is not None
    assert indices[0].segments[0].fragment_ids == {0}
    assert indices[0].segments[0].dataset_version_at_last_update == 1
    assert indices[0].segments[0].index_version == int(fts_format_version)
    assert indices[0].segments[0].created_at is not None
    assert isinstance(indices[0].segments[0].created_at, datetime)
    assert indices[0].segments[0].size_bytes is not None
    assert indices[0].segments[0].size_bytes > 0
    assert indices[0].total_size_bytes is not None
    assert indices[0].total_size_bytes > 0

    details = indices[0].details
    assert details is not None and len(details) > 0
    assert details["lance_tokenizer"] is None
    assert details["base_tokenizer"] == "simple"
    assert details["language"] == "English"
    assert not details["with_position"]
    assert details["max_token_length"] == 40
    assert details["lower_case"]
    assert details["stem"]
    assert details["remove_stop_words"]
    assert details["custom_stop_words"] is None
    assert details["ascii_folding"]
    assert details["min_ngram_length"] == 3
    assert details["max_ngram_length"] == 3
    assert not details["prefix_only"]

    ds.create_scalar_index("bitmap", index_type="BITMAP")
    ds.create_scalar_index("bloomfilter", index_type="BLOOMFILTER")
    ds.create_scalar_index("btree", index_type="BTREE")
    ds.create_scalar_index(
        "json",
        IndexConfig(
            index_type="json", parameters={"target_index_type": "btree", "path": "x"}
        ),
    )
    ds.create_scalar_index("ngram", index_type="NGRAM")
    ds.create_scalar_index("zonemap", index_type="ZONEMAP")

    indices = ds.describe_indices()
    # Skip text index since it is already asserted above
    indices = [index for index in indices if index.name != "text_idx"]
    indices.sort(key=lambda x: x.name)

    names = [
        "bitmap_idx",
        "bloomfilter_idx",
        "btree_idx",
        "json_idx",
        "ngram_idx",
        "zonemap_idx",
    ]
    types_urls = [
        "/lance.table.BitmapIndexDetails",
        "/lance.index.pb.BloomFilterIndexDetails",
        "/lance.table.BTreeIndexDetails",
        "/lance.index.pb.JsonIndexDetails",
        "/lance.table.NGramIndexDetails",
        "/lance.table.ZoneMapIndexDetails",
    ]
    index_types = [
        "Bitmap",
        "BloomFilter",
        "BTree",
        "Json",
        "NGram",
        "ZoneMap",
    ]
    details = [
        "{}",
        "{}",
        "{}",
        '{"path":"x","target_details":{}}',
        "{}",
        "{}",
    ]

    for i in range(len(indices)):
        assert indices[i].name == names[i]
        assert indices[i].type_url == types_urls[i]
        assert indices[i].index_type == index_types[i]
        assert indices[i].num_rows_indexed == 100
        assert indices[i].fields == [i + 2]
        assert indices[i].field_names == [data.column_names[i + 2]]
        assert len(indices[i].segments) == 1
        assert indices[i].segments[0].fragment_ids == {0}
        assert indices[i].segments[0].dataset_version_at_last_update == i + 2
        assert indices[i].segments[0].index_version == 0
        assert indices[i].segments[0].created_at is not None
        assert isinstance(indices[i].segments[0].created_at, datetime)
        assert indices[i].segments[0].size_bytes is not None
        assert indices[i].segments[0].size_bytes > 0
        assert indices[i].total_size_bytes is not None
        assert indices[i].total_size_bytes > 0
        assert indices[i].details == json.loads(details[i])

    ds.delete("id < 50")
    indices = ds.describe_indices()
    for index in indices:
        assert index.num_rows_indexed == 50


def test_vector_filter_fts_search(tmp_path):
    # Create test data
    ids = list(range(1, 301))
    vectors = [[float(i)] * 4 for i in ids]

    # Create text data:
    #   "text <i>" for ids 1-255, 299, 300,
    #   "noop <i>" for 256-298,
    texts = []
    for i in ids:
        if i <= 255:
            texts.append(f"text {i}")
        elif i <= 298:
            texts.append(f"noop {i}")
        else:
            texts.append(f"text {i}")

    categories = []
    for i in ids:
        if i % 3 == 1:
            categories.append("literature")
        elif i % 3 == 2:
            categories.append("science")
        else:
            categories.append("geography")

    table = pa.table(
        {
            "id": ids,
            "vector": pa.array(vectors, type=pa.list_(pa.float32(), 4)),
            "text": texts,
            "category": categories,
        }
    )

    # Write dataset and create indices
    ds = lance.write_dataset(table, tmp_path)

    ds = ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=2,
        num_sub_vectors=4,
    )
    ds.create_scalar_index("text", index_type="INVERTED", with_position=True)

    # Create vector_query
    vector_query = {
        "column": "vector",
        "q": np.array([300, 300, 300, 300], dtype=np.float32),
        "k": 5,
        "minimum_nprobes": 20,
        "use_index": True,
    }

    # Case 1: search with prefilter=true, query_filter=vector([300,300,300,300])
    scanner = ds.scanner(
        prefilter=False, nearest=vector_query, filter=MatchQuery("text", "text")
    )
    result = scanner.to_table()
    assert [300, 299] == result["id"].to_pylist()

    # Case 2: search with prefilter=true, search_filter=match("text"),
    #         filter="category='geography'"
    scanner = ds.scanner(
        prefilter=True,
        nearest=vector_query,
        filter={
            "expr_filter": "category='geography'",
            "search_filter": MatchQuery("text", "text"),
        },
    )
    result = scanner.to_table()
    assert [300, 255, 252, 249, 246] == result["id"].to_pylist()

    # Case 3: search with prefilter=false, search_filter=match("text")
    scanner = ds.scanner(
        prefilter=False,
        nearest=vector_query,
        filter=MatchQuery("text", "text"),
    )
    result = scanner.to_table()
    assert [300, 299] == result["id"].to_pylist()

    # Case 4: search with prefilter=false, search_filter=match("text"),
    #       filter="category='geography'"
    scanner = ds.scanner(
        prefilter=False,
        nearest=vector_query,
        filter={
            "expr_filter": "category='geography'",
            "search_filter": MatchQuery("text", "text"),
        },
    )
    result = scanner.to_table()
    assert [300] == result["id"].to_pylist()

    # Case 5: search with prefilter=false, search_filter=phrase("text")
    scanner = ds.scanner(
        prefilter=False,
        nearest=vector_query,
        filter=PhraseQuery("text", "text"),
    )
    with pytest.raises(ValueError):
        scanner.to_table()

    # Case 6: search with prefilter=false, search_filter=phrase("text")
    scanner = ds.scanner(
        prefilter=False,
        nearest=vector_query,
        filter={
            "expr_filter": "category='geography'",
            "search_filter": PhraseQuery("text", "text"),
        },
    )
    with pytest.raises(ValueError):
        scanner.to_table()
