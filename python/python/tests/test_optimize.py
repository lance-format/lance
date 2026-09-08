# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import json
import pickle
import random
import re
import threading
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.lance import Compaction
from lance.optimize import RewriteResult
from lance.vector import vec_to_table


def test_dataset_optimize(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = pa.table({"a": range(1000), "b": range(1000)})

    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=100)
    assert dataset.version == 1
    assert len(dataset.get_fragments()) == 10

    metrics = dataset.optimize.compact_files(
        target_rows_per_fragment=1000,
        materialize_deletions=False,
        num_threads=1,
        # Loose source budgets: all fragments still compact in one run.
        max_source_rows=100_000,
        max_source_bytes=1024 * 1024 * 1024,
    )

    assert metrics.fragments_removed == 10
    assert metrics.fragments_added == 1
    assert metrics.files_removed == 10
    assert metrics.files_added == 1

    assert dataset.version == 3


def test_dataset_optimize_excluded_fragment_ids(tmp_path: Path):
    dataset = lance.write_dataset(
        pa.table({"a": range(800)}),
        tmp_path / "dataset",
        max_rows_per_file=200,
    )
    fragments = dataset.get_fragments()

    metrics = dataset.optimize.compact_files(
        target_rows_per_fragment=400,
        excluded_fragment_ids=[1, 1, 999],
        num_threads=1,
    )

    assert metrics.fragments_removed == 2
    remaining_fragment_ids = {
        fragment.fragment_id for fragment in dataset.get_fragments()
    }
    assert fragments[1].fragment_id in remaining_fragment_ids


def test_compact_files_source_budgets(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = pa.table({"a": range(1000), "b": range(1000)})
    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=100)
    assert len(dataset.get_fragments()) == 10

    # A row budget of 250 admits the first two 100-row fragments only, so the
    # run is incremental instead of compacting all 10 at once.
    metrics = dataset.optimize.compact_files(
        target_rows_per_fragment=200,
        num_threads=1,
        max_source_rows=250,
    )
    assert metrics.fragments_removed == 2
    assert metrics.fragments_added == 1

    # The budgets are hard upper bounds: a budget smaller than a single task
    # produces an empty plan and compaction is a no-op.
    version_before = dataset.version
    metrics = dataset.optimize.compact_files(
        target_rows_per_fragment=200,
        num_threads=1,
        max_source_bytes=1,
    )
    assert metrics.fragments_removed == 0
    assert dataset.version == version_before

    with pytest.raises(OSError, match="must be greater than 0"):
        dataset.optimize.compact_files(max_source_rows=0)


def test_compact_files_max_source_fragments(tmp_path: Path):
    rows_per_fragment = 256 * 1024
    dataset = lance.write_dataset(
        pa.table({"a": pa.nulls(10 * rows_per_fragment)}),
        tmp_path / "dataset",
        max_rows_per_file=rows_per_fragment,
    )
    assert len(dataset.get_fragments()) == 10

    metrics = dataset.optimize.compact_files(
        max_source_fragments=4,
        num_threads=1,
    )

    assert metrics.fragments_removed == 4
    assert metrics.fragments_added == 1
    assert len(dataset.get_fragments()) == 7


def test_blob_compaction(tmp_path: Path):
    base_dir = tmp_path / "blob_dataset"
    blob_field = pa.field(
        "blob", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
    )
    schema = pa.schema([pa.field("id", pa.int32()), blob_field])
    blobs = [b"\x01\x02", b"\x03\x04\x05"]
    table = pa.table(
        {
            "id": pa.array([0, 1], type=pa.int32()),
            "blob": pa.array(blobs, type=pa.large_binary()),
        },
        schema=schema,
    )

    dataset = lance.write_dataset(
        table,
        base_dir,
        schema=schema,
        max_rows_per_file=1,
        data_storage_version="2.1",
    )
    assert len(dataset.get_fragments()) == 2

    dataset.optimize.compact_files(num_threads=1)
    assert len(dataset.get_fragments()) == 1

    blob_files = dataset.take_blobs("blob", indices=[0, 1])
    contents = [blob_files[0].readall(), blob_files[1].readall()]
    assert contents == blobs


def test_blob_compaction_with_nested_json_sibling(tmp_path: Path):
    dataset_uri = tmp_path / "nested_blob_json"
    info_fields = [lance.blob_field("blob"), pa.field("meta", pa.json_())]
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("info", pa.struct(info_fields)),
        ]
    )
    for index, row_id in enumerate([1, 2]):
        info = pa.StructArray.from_arrays(
            [
                lance.blob_array([f"blob-{row_id}".encode()]),
                pa.array([json.dumps({"row": row_id})], type=pa.json_()),
            ],
            fields=info_fields,
        )
        lance.write_dataset(
            pa.Table.from_arrays([pa.array([row_id]), info], schema=schema),
            dataset_uri,
            mode="create" if index == 0 else "append",
            data_storage_version="2.2",
        )

    dataset = lance.dataset(dataset_uri)
    dataset.optimize.compact_files(num_threads=1)

    assert len(dataset.get_fragments()) == 1
    assert [data for _, data in dataset.read_blobs("info.blob", indices=[0, 1])] == [
        b"blob-1",
        b"blob-2",
    ]
    assert [
        json.loads(value)
        for value in dataset.to_table(columns=["info.meta"])["info.meta"].to_pylist()
    ] == [{"row": 1}, {"row": 2}]


@pytest.mark.parametrize("storage_version", ["2.0", "2.1", "2.2"])
def test_blob_compaction_preserves_null_empty_and_read_parity(
    tmp_path: Path, storage_version: str
):
    base_dir = tmp_path / f"blob_dataset_{storage_version}"
    if storage_version == "2.2":
        blob_field = lance.blob_field("blob")
    else:
        blob_field = pa.field(
            "blob",
            pa.large_binary(),
            metadata={"lance-encoding:blob": "true"},
        )
    schema = pa.schema([pa.field("id", pa.int64()), blob_field])

    def make_blob_array(values):
        if storage_version == "2.2":
            return lance.blob_array(values)
        return pa.array(values, type=pa.large_binary())

    for index, (ids, values) in enumerate(
        [
            ([1, 2], [b"P1", None]),
            ([3, 4, 5], [b"P3", None, b""]),
        ]
    ):
        lance.write_dataset(
            pa.Table.from_arrays(
                [
                    pa.array(ids, type=pa.int64()),
                    make_blob_array(values),
                ],
                schema=schema,
            ),
            base_dir,
            mode="create" if index == 0 else "append",
            data_storage_version=storage_version,
        )

    dataset = lance.dataset(base_dir)
    dataset.delete("id = 2")
    dataset.optimize.compact_files(num_threads=1)

    expected = {1: b"P1", 3: b"P3", 4: None, 5: b""}
    binary_table = dataset.to_table(columns=["id", "blob"], blob_handling="all_binary")
    scan_values = dict(
        zip(
            binary_table.column("id").to_pylist(),
            binary_table.column("blob").to_pylist(),
        )
    )

    ids = dataset.to_table(columns=["id"]).column("id").to_pylist()
    blob_files = dataset.take_blobs("blob", indices=range(len(ids)))
    take_values = {
        row_id: None if blob_file is None else blob_file.readall()
        for row_id, blob_file in zip(ids, blob_files)
    }

    assert take_values == expected
    assert scan_values == take_values

    descriptor_table = dataset.to_table(columns=["id", "blob"])
    descriptions = dict(
        zip(
            descriptor_table.column("id").to_pylist(),
            descriptor_table.column("blob").to_pylist(),
        )
    )
    if storage_version == "2.0":
        assert descriptions[4] == {"position": 1, "size": 0}
    else:
        assert descriptions[4] is None
    assert descriptions[5] is not None
    assert descriptions[5]["size"] == 0


def test_optimize_max_bytes(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    arr = pa.array(range(4 * 1024 * 1024))
    arr = pa.FixedSizeListArray.from_arrays(arr, 1024)
    data = pa.table({"a": arr})

    # Write out 4K rows and 32MB of data
    dataset = lance.write_dataset(
        data, base_dir, max_rows_per_file=2 * 1024, data_storage_version="stable"
    )
    # We get 2 fragments
    assert len(dataset.get_fragments()) == 2

    # Now run compaction with a small max_bytes_per_file (1000 bytes) to get more
    # fragments.  The exact number is a bit tricky to calculate because we don't
    # split into a new fragment until we've actually written data and that depends
    # on how much the file format chooses to accumulate, but it should be more than 2
    metrics = dataset.optimize.compact_files(
        target_rows_per_fragment=100 * 1024,
        materialize_deletions=False,
        max_bytes_per_file=1000,
        batch_size=128,
    )
    assert metrics.fragments_removed == 2
    assert metrics.fragments_added > 2
    assert metrics.files_removed == 2
    assert metrics.files_added > 2

    num_frags = len(dataset.get_fragments())
    assert num_frags == metrics.fragments_added

    dataset = lance.write_dataset(
        data,
        base_dir,
        max_rows_per_file=2 * 1024,
        data_storage_version="stable",
        mode="overwrite",
    )

    # Same test but use Compaction.plan
    plan = Compaction.plan(
        dataset,
        options=dict(
            target_rows_per_fragment=100 * 1024, max_bytes_per_file=1000, batch_size=128
        ),
    )
    results = [task.execute(dataset) for task in plan.tasks]
    metrics = Compaction.commit(dataset, results)
    assert metrics.fragments_removed == 2
    assert metrics.fragments_added > 2
    assert metrics.files_removed == 2
    assert metrics.files_added > 2

    dataset = lance.write_dataset(
        data,
        base_dir,
        max_rows_per_file=2 * 1024,
        data_storage_version="stable",
        mode="overwrite",
    )

    # In this test max_bytes_per_file is still too small but the batch size
    # is so large we read the entire input in a single batch
    metrics = dataset.optimize.compact_files(
        target_rows_per_fragment=100 * 1024,
        materialize_deletions=False,
        max_bytes_per_file=1000,
        batch_size=2 * 1024,
    )

    assert metrics.fragments_removed == 2
    assert metrics.fragments_added == 2
    assert metrics.files_removed == 2
    assert metrics.files_added == 2

    num_frags = len(dataset.get_fragments())
    assert num_frags == 2


def create_table(min, max, nvec, ndim=8):
    mat = np.random.uniform(min, max, (nvec, ndim))
    tbl = vec_to_table(data=mat)
    # Add id column for filtering
    tbl = pa.Table.from_pydict(
        {
            "vector": tbl.column(0).chunk(0),
            "id": np.arange(0, nvec),
        }
    )
    return tbl


def test_compact_with_write(tmp_path: Path):
    # This test creates a dataset with a manifest containing fragments
    # that are not in sorted order (by id)
    #
    # We do this by running compaction concurrently with append
    #
    # This is because compaction first reserves a fragment id.  Then the
    # concurrent writes grab later ids and commit them.  Then the compaction
    # commits with its earlier id.
    #
    # In the next compaction we should detect this, and reorder the fragments
    # when writing the compacted file.
    base_dir = tmp_path / "dataset"
    NUM_FRAGS = 5
    ROWS_PER_FRAG = 300

    # First, create some data
    data = create_table(min=0, max=1, nvec=ROWS_PER_FRAG)
    dataset = lance.write_dataset(data, base_dir)
    for _ in range(NUM_FRAGS):
        lance.write_dataset(data, base_dir, mode="append")

    # Now, run compaction at the same time as creating new data
    def do_compaction():
        dataset = lance.dataset(base_dir)
        dataset.optimize.compact_files()

    compact_thread = threading.Thread(target=do_compaction)
    compact_thread.start()
    for _ in range(NUM_FRAGS):
        lance.write_dataset(data, base_dir, mode="append")
    compact_thread.join()

    # Now, run compaction again, this should succeed
    dataset = lance.dataset(base_dir)
    dataset.optimize.compact_files()

    assert dataset.to_table().num_rows == ROWS_PER_FRAG * (NUM_FRAGS * 2 + 1)


def test_index_remapping(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = create_table(min=0, max=1, nvec=300)

    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=150)
    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=2, num_sub_vectors=2
    )
    assert len(dataset.get_fragments()) == 2

    sample_query_indices = random.sample(range(300), 50)
    vecs = data.column("vector").chunk(0)
    sample_queries = [
        {"column": "vector", "q": vecs[i].values, "k": 5} for i in sample_query_indices
    ]

    def has_target(target, results):
        for item in results:
            if item.values == target:
                return True
        return False

    def check_index(has_knn_combined):
        for query in sample_queries:
            results = dataset.to_table(nearest=query).column("vector")
            assert has_target(query["q"], results)
            plan = dataset.scanner(nearest=query).explain_plan()
            assert ("KNNVectorDistance" in plan) == has_knn_combined

    # Original state is 2 indexed fragments of size 150.  This should not require
    # a combined scan
    check_index(has_knn_combined=False)

    # Compact the 2 fragments into 1.  Combined scan still not needed.
    dataset.optimize.compact_files()
    assert len(dataset.get_fragments()) == 1
    check_index(has_knn_combined=False)

    # Add a new fragment and recalculate the index
    extra_data = create_table(min=1000, max=1001, nvec=100)
    dataset = lance.write_dataset(
        extra_data, base_dir, mode="append", max_rows_per_file=100
    )
    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=2, num_sub_vectors=2, replace=True
    )

    # Combined scan should not be needed
    assert len(dataset.get_fragments()) == 2
    check_index(has_knn_combined=False)

    # Add a new unindexed fragment
    extra_data = create_table(min=1000, max=1001, nvec=100)
    dataset = lance.write_dataset(
        extra_data, base_dir, mode="append", max_rows_per_file=100
    )
    assert len(dataset.get_fragments()) == 3

    # Compaction should not combine the unindexed fragment with the indexed fragment
    dataset.optimize.compact_files()
    assert len(dataset.get_fragments()) == 2

    # Now a combined scan is required
    check_index(has_knn_combined=True)


def test_index_remapping_multiple_rewrite_tasks(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    ds = lance.write_dataset(
        create_table(min=0, max=1, nvec=300), base_dir, max_rows_per_file=150
    )
    ds = ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=2,
    )
    assert ds.has_index
    ds = lance.write_dataset(
        create_table(min=0, max=1, nvec=300),
        base_dir,
        mode="append",
        max_rows_per_file=150,
    )

    ds.delete("id % 4 == 0")
    fragments = list(ds.get_fragments())
    assert len(fragments) == 4

    # We have a dataset with 4 small fragments.  2 are indexed and
    # 2 are not.  The indexed fragments and the non-indexed fragments
    # cannot be combined and so we should end up with 2 fragments after
    # compaction
    ds.optimize.compact_files()

    fragments = list(ds.get_fragments())
    assert len(fragments) == 2

    index = ds.describe_indices()[0]
    index_frag_ids = list(index.segments[0].fragment_ids)
    frag_ids = [frag.fragment_id for frag in fragments]

    assert len(index_frag_ids) == 1
    assert index_frag_ids[0] in frag_ids


def test_defer_index_remap(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = pa.table({"i": range(6_000), "val": range(6_000)})
    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=1_000)
    dataset.create_scalar_index("i", "BTREE")
    options = dict(
        target_rows_per_fragment=2_000, defer_index_remap=True, num_threads=1
    )

    dataset.delete("i < 500")
    dataset.optimize.compact_files(**options)

    dataset = lance.dataset(base_dir)
    indices = dataset.describe_indices()
    assert any(idx.name == "__lance_frag_reuse" for idx in indices)


@pytest.mark.parametrize("use_commit_options", [True, False])
def test_defer_index_remap_via_commit_options(tmp_path: Path, use_commit_options: bool):
    """Compaction.commit respects defer_index_remap passed in options.

    When options={"defer_index_remap": True} is supplied to Compaction.commit
    the __lance_frag_reuse system index must appear in describe_indices().
    When the option is omitted (default) no such system index is written.
    """
    base_dir = tmp_path / f"dataset_commit_opts_{use_commit_options}"
    data = pa.table({"i": range(6_000), "val": range(6_000)})
    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=1_000)
    dataset.create_scalar_index("i", "BTREE")
    dataset.delete("i < 500")

    plan = Compaction.plan(
        dataset,
        options=dict(target_rows_per_fragment=2_000, num_threads=1),
    )
    rewrites = [task.execute(dataset) for task in plan.tasks]

    if use_commit_options:
        Compaction.commit(dataset, rewrites, options={"defer_index_remap": True})
    else:
        Compaction.commit(dataset, rewrites)

    dataset = lance.dataset(base_dir)
    indices = dataset.describe_indices()
    has_frag_reuse = any(idx.name == "__lance_frag_reuse" for idx in indices)

    if use_commit_options:
        assert has_frag_reuse, (
            "expected __lance_frag_reuse system index when defer_index_remap=True "
            "is passed to Compaction.commit"
        )
    else:
        assert not has_frag_reuse, (
            "did not expect __lance_frag_reuse system index when options is omitted "
            "from Compaction.commit"
        )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_describe_indices_matches_list_indices_for_frag_reuse(tmp_path: Path):
    """describe_indices() and list_indices() must agree on the index_type
    string for every index, including the __lance_frag_reuse system index
    that defer_index_remap produces.

    list_indices() is a wrapper over describe_indices(), so the two must stay
    in sync. System indices are identified by name via infer_system_index_type()
    in rust/lance/src/index.rs::IndexDescriptionImpl::try_new.
    """
    base_dir = tmp_path / "dataset"
    data = pa.table({"i": range(6_000), "val": range(6_000)})
    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=1_000)
    dataset.create_scalar_index("i", "BTREE")
    dataset.delete("i < 500")
    dataset.optimize.compact_files(
        target_rows_per_fragment=2_000, defer_index_remap=True, num_threads=1
    )

    dataset = lance.dataset(base_dir)
    described = {d.name: d.index_type for d in dataset.describe_indices()}
    listed = {idx["name"]: idx["type"] for idx in dataset.list_indices()}

    assert "__lance_frag_reuse" in listed, (
        "test precondition: defer_index_remap should produce a frag-reuse index"
    )
    assert described == listed, (
        "describe_indices and list_indices disagree on index_type:\n"
        f"  describe_indices: {described}\n"
        f"  list_indices:     {listed}"
    )


def test_dataset_distributed_optimize(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = pa.table({"a": range(800), "b": range(800)})

    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=200)
    fragments = dataset.get_fragments()
    assert len(fragments) == 4

    plan = Compaction.plan(
        dataset, options=dict(target_rows_per_fragment=400, num_threads=1)
    )
    assert plan.read_version == 1
    assert plan.num_tasks() == 2
    assert plan.tasks[0].fragments == [frag.metadata for frag in fragments[0:2]]
    assert plan.tasks[1].fragments == [frag.metadata for frag in fragments[2:4]]
    assert repr(plan) == "CompactionPlan(read_version=1, tasks=<2 compaction tasks>)"

    excluded_plan = Compaction.plan(
        dataset,
        options=dict(
            target_rows_per_fragment=400,
            excluded_fragment_ids=[1, 1, 999],
            num_threads=1,
        ),
    )
    assert excluded_plan.num_tasks() == 1
    assert excluded_plan.tasks[0].fragments == [
        frag.metadata for frag in fragments[2:4]
    ]
    assert pickle.loads(pickle.dumps(excluded_plan)) == excluded_plan

    none_plan = Compaction.plan(
        dataset,
        options=dict(
            target_rows_per_fragment=400,
            excluded_fragment_ids=None,
            num_threads=1,
        ),
    )
    assert none_plan == plan

    # Plan can be pickled
    assert pickle.loads(pickle.dumps(plan)) == plan

    pickled_task = pickle.dumps(plan.tasks[0])
    task = pickle.loads(pickled_task)
    assert task == plan.tasks[0]

    result1 = plan.tasks[0].execute(dataset)
    result1.metrics.fragments_removed == 2
    result1.metrics.fragments_added == 1

    pickled_result = pickle.dumps(result1)
    result = pickle.loads(pickled_result)
    assert isinstance(result, RewriteResult)
    assert result == result1
    assert re.match(
        r"RewriteResult\(read_version=1, new_fragments=\[.+\], old_fragments=\[.+\]\)",
        repr(result),
    )

    metrics = Compaction.commit(dataset, [result1])
    assert metrics.fragments_removed == 2
    assert metrics.fragments_added == 1
    # Compaction occurs in two transactions so it increments the version by 2.
    assert dataset.version == 3


def test_migration_via_fragment_apis(tmp_path):
    """
    This test is a regression of a case where we were using fragment APIs to migrate
    from v1 to v2 but that left the dataset in a state where it had v2 files but wasn't
    marked with the v2 writer flag.
    """
    data = pa.table({"a": range(800), "b": range(800)})

    # Create v1 dataset
    ds = lance.write_dataset(
        data, tmp_path / "dataset", max_rows_per_file=200, data_storage_version="legacy"
    )

    # Create empty v2 dataset
    lance.write_dataset(
        data_obj=[],
        uri=tmp_path / "dataset2",
        schema=ds.schema,
        data_storage_version="2.0",
    )

    # Add v2 files
    fragments = []
    for frag in ds.get_fragments():
        reader = ds.scanner(fragments=[frag])
        fragments.append(
            lance.LanceFragment.create(
                dataset_uri=tmp_path / "dataset2",
                data=reader,
                fragment_id=frag.fragment_id,
                data_storage_version="2.0",
            )
        )

    # Commit
    operation = lance.LanceOperation.Overwrite(ds.schema, fragments)
    ds2 = lance.LanceDataset.commit(tmp_path / "dataset2", operation)

    # Compact, dataset should still be v2
    ds2.optimize.compact_files()

    ds2 = lance.dataset(tmp_path / "dataset2")

    assert ds2.data_storage_version == "2.0"


def test_optimize_indices_second_call_is_noop(tmp_path: Path):
    """A second optimize_indices call when nothing has changed since the first
    must not write any new files to the dataset directory."""
    base_dir = tmp_path / "dataset"

    def assert_optimize_is_noop(dataset: lance.LanceDataset):
        version_before = dataset.version
        files_before = {
            path.relative_to(base_dir) for path in base_dir.rglob("*") if path.is_file()
        }

        dataset.optimize.optimize_indices()

        files_after = {
            path.relative_to(base_dir) for path in base_dir.rglob("*") if path.is_file()
        }
        assert dataset.version == version_before
        assert files_after == files_before

    n = 1024
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, 8)).astype(np.float32)
    table = pa.table(
        {
            "id": pa.array(range(n), type=pa.int64()),
            "category": pa.array([f"cat{i % 4}" for i in range(n)]),
            "tags": pa.array([[f"t{i % 3}", f"t{(i + 1) % 3}"] for i in range(n)]),
            "doc": pa.array([f"hello world document {i}" for i in range(n)]),
            "name": pa.array([f"name_{i:05d}" for i in range(n)]),
            "value": pa.array(range(n), type=pa.int64()),
            "bloom_val": pa.array(range(n), type=pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1), type=pa.float32()), 8
            ),
        }
    )
    dataset = lance.write_dataset(table, base_dir)

    dataset.create_scalar_index("id", index_type="BTREE")
    dataset.create_scalar_index("category", index_type="BITMAP")
    dataset.create_scalar_index("tags", index_type="LABEL_LIST")
    dataset.create_scalar_index("doc", index_type="INVERTED")
    dataset.create_scalar_index("name", index_type="NGRAM")
    dataset.create_scalar_index("value", index_type="ZONEMAP")
    dataset.create_scalar_index("bloom_val", index_type="BLOOMFILTER")
    # num_partitions=1 keeps this dataset balanced: the auto-rebalance check
    # in merge_indices only finds join candidates when num_partitions > 1, and
    # 1024 + 128 rows is well below the split threshold. Without this, the
    # rebalance heuristic would keep finding work on the small partitions.
    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=1, num_sub_vectors=2
    )

    extra_rows = 128
    extra_vectors = rng.standard_normal((extra_rows, 8)).astype(np.float32)
    extra = pa.table(
        {
            "id": pa.array(range(n, n + extra_rows), type=pa.int64()),
            "category": pa.array([f"cat{i % 4}" for i in range(extra_rows)]),
            "tags": pa.array([[f"t{i % 3}"] for i in range(extra_rows)]),
            "doc": pa.array([f"goodbye world document {i}" for i in range(extra_rows)]),
            "name": pa.array([f"add_{i:05d}" for i in range(extra_rows)]),
            "value": pa.array(range(n, n + extra_rows), type=pa.int64()),
            "bloom_val": pa.array(range(n, n + extra_rows), type=pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(extra_vectors.reshape(-1), type=pa.float32()), 8
            ),
        }
    )
    dataset = lance.write_dataset(extra, base_dir, mode="append")

    # First optimize: should pull the new fragment into each index.
    dataset.optimize.optimize_indices()
    assert_optimize_is_noop(dataset)

    # Consolidate the vector delta so every fragment has the same physical index
    # coverage; compaction intentionally avoids mixing fragments that do not.
    dataset.optimize.optimize_indices(num_indices_to_merge=10)

    # Compaction invalidates the old fragment coverage, so one optimize is needed
    # to rebuild each index. Further calls must return to the same steady state.
    compaction = dataset.optimize.compact_files(
        target_rows_per_fragment=2 * (n + extra_rows), num_threads=1
    )
    assert compaction.fragments_removed > 0
    dataset.optimize.optimize_indices()
    assert_optimize_is_noop(dataset)


def test_compaction_generates_rewrite_transaction(tmp_path: Path):
    # Create a small dataset with multiple fragments
    base_dir = tmp_path / "rewrite_txn"
    data = pa.table({"a": range(300), "b": range(300)})

    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=100)

    # Run compaction: should perform a rewrite (no deletions materialized)
    dataset.optimize.compact_files(materialize_deletions=False, num_threads=1)

    # Verify at least one transaction is a Rewrite; guard against None entries
    transactions = dataset.get_transactions()
    assert any(
        t is not None and t.operation.__class__.__name__ == "Rewrite"
        for t in transactions
    )


def test_remap_row_addrs(tmp_path: Path):
    # Dataset.remap_row_addrs follows rows across a compaction via the
    # fragment-reuse index: an address valid before the compaction maps to the
    # row's new address after it. None when there is no fragment-reuse index.
    base_dir = tmp_path / "dataset"
    data = pa.table({"id": range(1_000), "v": range(1_000)})
    ds = lance.write_dataset(data, base_dir, max_rows_per_file=100)  # 10 fragments

    # No fragment-reuse index yet -> None (nothing to remap against).
    addrs = pa.array([0, 1 << 32, (5 << 32) | 7], pa.uint64())
    assert ds.remap_row_addrs(addrs) is None

    before = ds.scanner(columns=["id"], with_row_address=True).to_table()
    old = dict(zip(before["id"].to_pylist(), before["_rowaddr"].to_pylist()))

    # A deferred-remap compaction records a fragment-reuse index only when it
    # rewrites data an index covers, so index a column first.
    ds.create_scalar_index("id", "BTREE")

    ds.optimize.compact_files(
        target_rows_per_fragment=1_000, defer_index_remap=True, num_threads=1
    )
    ds = lance.dataset(base_dir)
    assert any(idx.name == "__lance_frag_reuse" for idx in ds.describe_indices())

    after = ds.scanner(columns=["id"], with_row_address=True).to_table()
    new = dict(zip(after["id"].to_pylist(), after["_rowaddr"].to_pylist()))

    sample = [0, 137, 999]
    remapped = ds.remap_row_addrs(
        pa.array([old[i] for i in sample], pa.uint64())
    ).to_pylist()
    assert remapped == [new[i] for i in sample]
