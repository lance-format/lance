# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import math
import os
import time
import uuid

import lance
import pyarrow as pa
import pytest
from lance.mem_wal import (
    CompactedSsTable,
    LsmPointLookupPlanner,
    LsmScanner,
    LsmVectorSearchPlanner,
    ShardingField,
    ShardingSpec,
    ShardSnapshot,
    evaluate_sharding_spec,
)
from lance.schema import LanceSchema

_PK_META = {"lance-schema:unenforced-primary-key": "true"}
_LOOKUP_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False, metadata=_PK_META),
        pa.field("name", pa.utf8()),
    ]
)
_APPEND_ONLY_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("name", pa.utf8()),
    ]
)


def _lookup_table(ids, prefix: str) -> pa.Table:
    """Build a table where name = '{prefix}_{id}' for each id."""
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "name": pa.array([f"{prefix}_{i}" for i in ids], pa.utf8()),
        },
        schema=_LOOKUP_SCHEMA,
    )


def _append_only_table(ids, prefix: str) -> pa.Table:
    """Build a table without primary-key metadata."""
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "name": pa.array([f"{prefix}_{i}" for i in ids], pa.utf8()),
        },
        schema=_APPEND_ONLY_SCHEMA,
    )


def _write_sstable(base_path: str, shard_id: str, gen_folder: str, data: pa.Table):
    """Write an SSTable Lance dataset at the expected sub-path.

    The collector resolves SSTable paths as:
        {base_dataset_path}/_mem_wal/{shard_id}/{gen_folder}

    Production flush also writes a primary-key dedup sidecar (`_pk_index/`) that
    the LSM scanner opens to dedup across generations; stage it here too so the
    SSTable faithfully matches what flush produces.
    """
    from lance.lance import _write_pk_sidecar

    gen_path = os.path.join(base_path, "_mem_wal", shard_id, gen_folder)
    lance.write_dataset(data, gen_path, schema=_LOOKUP_SCHEMA)
    _write_pk_sidecar(gen_path, data, ["id"])


def test_mark_sstables_as_compacted(tmp_path):
    ds_path = str(tmp_path / "base")
    shard_id = str(uuid.uuid4())
    dataset = lance.write_dataset(
        _lookup_table([1, 2, 3], "base"), ds_path, schema=_LOOKUP_SCHEMA
    )
    dataset.initialize_mem_wal()

    (
        dataset.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .mark_sstables_as_compacted([CompactedSsTable(shard_id, 1)])
        .execute(_lookup_table([2, 4], "compacted"))
    )

    assert dataset.count_rows() == 4


def test_point_lookup_with_memtables(tmp_path):
    """
    Lookup against a base table that has one SSTable containing an
    update.  The SSTable version must win over the base table version.

    Setup
    -----
    base   : ids [1, 2, 3]  names ["base_1",  "base_2",  "base_3"]
    gen_1  : ids [2]        names ["gen1_2"]          ← update to id=2

    ShardSnapshot: sstable(gen=1, path="gen_1"), current_generation=2
    """
    ds_path = str(tmp_path / "base")
    shard_id = str(uuid.uuid4())

    # --- Base dataset ---
    base_ds = lance.write_dataset(
        _lookup_table([1, 2, 3], "base"), ds_path, schema=_LOOKUP_SCHEMA
    )
    base_ds.initialize_mem_wal()

    # --- SSTable: overwrites id=2 ---
    _write_sstable(ds_path, shard_id, "gen_1", _lookup_table([2], "gen1"))

    # --- ShardSnapshot describing the SSTable state ---
    snap = ShardSnapshot(shard_id).with_sstable(1, "gen_1").with_current_generation(2)

    planner = LsmPointLookupPlanner(base_ds, [snap])
    assert not hasattr(planner, "lookup")

    # id=2 must return the SSTable version
    plan = planner.plan_lookup(pa.array([2], type=pa.int64()))
    assert plan.schema.names == ["id", "name"]
    assert plan.dataset_schema.names == ["id", "name"]
    explained = plan.explain()
    assert "Take" in explained or "Scan" in explained or "LanceRead" in explained
    result = plan.to_table()
    assert len(result) == 1, "Expected exactly one row for id=2"
    assert result.column("name")[0].as_py() == "gen1_2", (
        "SSTable must win over base table"
    )

    # id=1 is only in the base table
    result_base = planner.plan_lookup(pa.array([1], type=pa.int64())).to_table()
    assert len(result_base) == 1
    assert result_base.column("name")[0].as_py() == "base_1"

    # id=99 does not exist anywhere
    result_miss = planner.plan_lookup(pa.array([99], type=pa.int64())).to_table()
    assert len(result_miss) == 0, "Non-existent key must return empty result"


def test_lsm_scanner_with_memtables(tmp_path):
    """
    Full-scan via LsmScanner.from_snapshots deduplicates rows across LSM
    levels: each primary key appears exactly once, from its newest level.

    base  : ids [1, 2, 3]  names ["base_1",  "base_2",  "base_3"]
    gen_1 : ids [2]        names ["gen1_2"]    ← overwrites id=2

    Expected result: 3 unique rows — id=2 from gen_1, id=1 and id=3 from base.
    """
    ds_path = str(tmp_path / "base")
    shard_id = str(uuid.uuid4())

    base_ds = lance.write_dataset(
        _lookup_table([1, 2, 3], "base"), ds_path, schema=_LOOKUP_SCHEMA
    )
    base_ds.initialize_mem_wal()

    _write_sstable(ds_path, shard_id, "gen_1", _lookup_table([2], "gen1"))

    snap = ShardSnapshot(shard_id).with_sstable(1, "gen_1").with_current_generation(2)

    scanner = LsmScanner.from_snapshots(base_ds, [snap])
    table = scanner.to_table()

    assert len(table) == 3, f"Expected 3 deduplicated rows, got {len(table)}"
    name_by_id = {row["id"]: row["name"] for row in table.to_pylist()}

    assert name_by_id[1] == "base_1"
    assert name_by_id[2] == "gen1_2", "SSTable gen must overwrite base for id=2"
    assert name_by_id[3] == "base_3"

    offset_table = (
        LsmScanner.from_snapshots(base_ds, [snap]).limit(None, offset=1).to_table()
    )
    assert len(offset_table) == 2, "Offset-only LSM scan should not require a limit"


def test_shard_writer_lsm_scanner_includes_own_sstables(tmp_path):
    ds_path = str(tmp_path / "base")
    shard_id = str(uuid.uuid4())
    ds = lance.write_dataset(_lookup_table([0], "base"), ds_path, schema=_LOOKUP_SCHEMA)
    ds.initialize_mem_wal()

    with ds.mem_wal_writer(
        shard_id,
        durable_write=True,
        max_wal_buffer_size=1,
        max_wal_flush_interval_ms=10,
        max_memtable_batches=1,
    ) as writer:
        writer.put(_lookup_table([1, 2], "writer"))

        deadline = time.time() + 5
        while True:
            table = writer.lsm_scanner().to_table()
            name_by_id = {row["id"]: row["name"] for row in table.to_pylist()}
            if name_by_id.get(1) == "writer_1" and name_by_id.get(2) == "writer_2":
                break
            if time.time() >= deadline:
                assert False, "writer.lsm_scanner() did not include SSTable writer rows"
            time.sleep(0.05)


def test_shard_writer_delete_binding_masks_base_row(tmp_path):
    ds_path = str(tmp_path / "base")
    shard_id = str(uuid.uuid4())
    ds = lance.write_dataset(
        _lookup_table([1, 2, 3], "base"), ds_path, schema=_LOOKUP_SCHEMA
    )
    ds.initialize_mem_wal()

    delete_keys = pa.table({"id": pa.array([2], type=pa.int64())})

    with ds.mem_wal_writer(
        shard_id,
        durable_write=True,
        max_wal_buffer_size=1,
        max_wal_flush_interval_ms=10,
    ) as writer:
        writer.put(_lookup_table([4], "writer"))
        writer.delete(delete_keys)
        table = writer.lsm_scanner().to_table()

    rows = {row["id"]: row["name"] for row in table.to_pylist()}
    assert rows[1] == "base_1"
    assert 2 not in rows, "deleted base row should be masked by the tombstone"
    assert rows[3] == "base_3"
    assert rows[4] == "writer_4"


_VDIM = 4  # matches Rust test fixture dimension


def _vector_search_schema():
    """Schema for vector-search tests: id (int32 PK) + vector column."""
    pk_meta = {"lance-schema:unenforced-primary-key": "true"}
    return pa.schema(
        [
            pa.field("id", pa.int32(), nullable=False, metadata=pk_meta),
            pa.field("vector", pa.list_(pa.float32(), _VDIM)),
        ]
    )


def _vector_search_table(ids):
    """Build a table with deterministic vectors matching the Rust fixture.

    For id=N the vector is [N*0.1, N*0.1+0.1, N*0.1+0.2, N*0.1+0.3].
    """
    flat = []
    for i in ids:
        base = i * 0.1
        flat.extend([base, base + 0.1, base + 0.2, base + 0.3])
    storage = pa.array(flat, type=pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(storage, _VDIM)
    return pa.table(
        {"id": pa.array(ids, pa.int32()), "vector": vectors},
        schema=_vector_search_schema(),
    )


def test_lsm_vector_search_filter_binding(tmp_path):
    ds_path = str(tmp_path / "vec")
    ds = lance.write_dataset(
        _vector_search_table(range(400)), ds_path, schema=_vector_search_schema()
    )
    ds.create_index(
        "vector",
        index_type="IVF_PQ",
        name="vector_idx",
        num_partitions=2,
        num_sub_vectors=2,
    )
    ds.initialize_mem_wal(maintained_indexes=["vector_idx"])

    planner = LsmVectorSearchPlanner(ds, [], "vector", filter="id >= 200")
    query_id = 250
    query = pa.array([25.0, 25.1, 25.2, 25.3], type=pa.float32())
    table = planner.plan_search(query, k=20, nprobes=2, columns=["id"]).to_table()

    ids = table.column("id").to_pylist()
    assert ids, "filtered vector search should return at least one row"
    assert min(ids) >= 200
    assert query_id in ids, f"filtered vector search recall missed id={query_id}: {ids}"

    with pytest.raises(ValueError, match="k must be positive"):
        planner.plan_search(query, k=0, nprobes=2, columns=["id"])
    with pytest.raises(ValueError, match="nprobes must be positive"):
        planner.plan_search(query, k=20, nprobes=0, columns=["id"])
    with pytest.raises(ValueError, match="overfetch_factor must be finite"):
        planner.plan_search(
            query, k=20, nprobes=2, columns=["id"], overfetch_factor=math.inf
        )


VECTOR_DIM = 32
ROWS_PER_BATCH = 50
NUM_WRITE_ROUNDS = 3
BATCHES_PER_ROUND = 3


def _e2e_schema():
    """Schema for the e2e test: id (PK), vector, text."""
    pk_meta = {"lance-schema:unenforced-primary-key": "true"}
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False, metadata=pk_meta),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
            pa.field("text", pa.utf8()),
        ]
    )


def _e2e_batch(schema, start_id: int, num_rows: int) -> pa.RecordBatch:
    """Deterministic RecordBatch for the e2e test."""
    ids = list(range(start_id, start_id + num_rows))
    flat_vecs = [
        math.sin((start_id + i) * 0.1 + d * 0.01)
        for i in range(num_rows)
        for d in range(VECTOR_DIM)
    ]
    storage = pa.array(flat_vecs, type=pa.float32())
    vector_array = pa.FixedSizeListArray.from_arrays(storage, VECTOR_DIM)
    texts = [f"Sample text for row {start_id + i}" for i in range(num_rows)]
    return pa.record_batch(
        {
            "id": pa.array(ids, pa.int64()),
            "vector": vector_array,
            "text": pa.array(texts, pa.utf8()),
        },
        schema=schema,
    )


def test_shard_writer_e2e_correctness(tmp_path):
    """
    End-to-end correctness test for ShardWriter covering:
    - Multi-round writes that trigger WAL and MemTable flushes
    - File-system layout verification (_mem_wal/<shard_id>/wal/ and manifest/)
    - SSTable data readable via LsmScanner
    - New writer created after close can write and scan correctly

    Mirrors Rust test: shard_writer_tests::test_shard_writer_e2e_correctness
    """
    schema = _e2e_schema()
    ds_path = str(tmp_path / "ds")

    # 500 seed rows so BTree index training succeeds
    initial_batch = _e2e_batch(schema, start_id=0, num_rows=500)
    ds = lance.write_dataset(
        pa.Table.from_batches([initial_batch]), ds_path, schema=schema
    )

    ds.create_scalar_index("id", "BTREE", name="id_btree")
    ds.initialize_mem_wal(maintained_indexes=["id_btree"])

    # Small buffers to trigger WAL and MemTable flushes during the test
    shard_id = str(uuid.uuid4())
    writer = ds.mem_wal_writer(
        shard_id,
        durable_write=True,
        max_wal_buffer_size=10 * 1024,  # 10 KB
        max_wal_flush_interval_ms=50,
        max_memtable_size=80,  # flush after ~80 rows
    )

    total_rows_written = 0
    for _round in range(NUM_WRITE_ROUNDS):
        start_id = 500 + total_rows_written
        for i in range(BATCHES_PER_ROUND):
            batch = _e2e_batch(schema, start_id + i * ROWS_PER_BATCH, ROWS_PER_BATCH)
            writer.put(pa.Table.from_batches([batch]))
        total_rows_written += BATCHES_PER_ROUND * ROWS_PER_BATCH
        time.sleep(0.15)  # allow async WAL/memtable flush

    writer.close()

    # === Stats ===
    stats = writer.stats()
    assert stats["put_count"] == NUM_WRITE_ROUNDS * BATCHES_PER_ROUND
    assert stats["wal_flush_count"] >= 1, "Expected at least one WAL flush"

    closed_memtable_stats = writer.memtable_stats()
    assert closed_memtable_stats["row_count"] == 0
    assert closed_memtable_stats["batch_count"] == 0
    assert closed_memtable_stats["generation"] >= 1
    assert "frozen_count" in closed_memtable_stats
    # close() flushes every frozen memtable, so nothing is still owed to flush.
    assert closed_memtable_stats["frozen_bytes"] == 0

    # === File-system layout ===
    mem_wal_dir = os.path.join(ds_path, "_mem_wal", shard_id)
    assert os.path.isdir(mem_wal_dir), f"MemWAL directory missing: {mem_wal_dir}"

    wal_dir = os.path.join(mem_wal_dir, "wal")
    assert os.path.isdir(wal_dir), "WAL sub-directory missing"
    wal_files = os.listdir(wal_dir)
    assert len(wal_files) >= 1
    assert all(f.endswith(".arrow") for f in wal_files), (
        f"All WAL files should have .arrow extension, got: {wal_files}"
    )

    manifest_dir = os.path.join(mem_wal_dir, "manifest")
    assert os.path.isdir(manifest_dir), "Manifest sub-directory missing"
    assert len(os.listdir(manifest_dir)) >= 1

    # === Generation counter advanced ===
    mt_stats = writer.memtable_stats()
    assert mt_stats["generation"] >= 1

    # === New writer: write and read back via active MemTable scanner ===
    ds2 = lance.dataset(ds_path)
    shard_id2 = str(uuid.uuid4())
    with ds2.mem_wal_writer(shard_id2, durable_write=False) as writer2:
        verify_batch = _e2e_batch(schema, start_id=10000, num_rows=10)
        writer2.put(pa.Table.from_batches([verify_batch]))
        result = writer2.lsm_scanner().to_table()

    assert len(result) >= 10
    new_ids = result.column("id").to_pylist()
    assert 10000 in new_ids
    assert 10009 in new_ids


# === initialize_mem_wal: sharding, maintained indexes, writer defaults ===


def _mem_wal_dataset(tmp_path, name: str = "ds"):
    """A base dataset with an unenforced primary key, ready for MemWAL init."""
    ds_path = str(tmp_path / name)
    return lance.write_dataset(
        _lookup_table([1, 2, 3], "base"), ds_path, schema=_LOOKUP_SCHEMA
    )


def test_initialize_mem_wal_manual(tmp_path):
    ds = _mem_wal_dataset(tmp_path)
    ds.initialize_mem_wal()

    details = ds.mem_wal_index_details()
    assert details is not None
    assert details["num_shards"] == 0
    assert details["sharding_specs"] == []
    assert details["maintained_indexes"] == []
    assert details["writer_config_defaults"] == {}


def test_initialize_mem_wal_unsharded(tmp_path):
    ds = _mem_wal_dataset(tmp_path)
    ds.initialize_mem_wal(unsharded=True)

    details = ds.mem_wal_index_details()
    assert details["num_shards"] == 1
    specs = details["sharding_specs"]
    assert len(specs) == 1
    assert specs[0]["fields"][0]["transform"] == "unsharded"


def test_initialize_mem_wal_bucket_sharding(tmp_path):
    ds = _mem_wal_dataset(tmp_path)
    ds.initialize_mem_wal(bucket_column="name", num_buckets=8)

    details = ds.mem_wal_index_details()
    assert details["num_shards"] == 8
    field = details["sharding_specs"][0]["fields"][0]
    assert field["transform"] == "bucket"
    assert "expression" in field
    assert field["parameters"]["num_buckets"] == "8"
    assert len(field["source_ids"]) == 1

    batch = _lookup_table([1, 2, 3], "base").to_batches()[0]
    result = evaluate_sharding_spec(
        batch, details["sharding_specs"][0], LanceSchema.from_pyarrow(batch.schema)
    )
    assert result.column_names == [field["field_id"]]
    assert result.num_rows == batch.num_rows


def test_initialize_mem_wal_bucket_sharding_without_primary_key(tmp_path):
    ds_path = str(tmp_path / "append_only")
    ds = lance.write_dataset(
        _append_only_table([1, 2, 3], "base"),
        ds_path,
        schema=_APPEND_ONLY_SCHEMA,
    )
    ds.initialize_mem_wal(bucket_column="id", num_buckets=8)

    details = ds.mem_wal_index_details()
    assert details["num_shards"] == 8
    field = details["sharding_specs"][0]["fields"][0]
    assert field["transform"] == "bucket"
    assert field["parameters"]["num_buckets"] == "8"


def test_initialize_mem_wal_identity_sharding(tmp_path):
    ds = _mem_wal_dataset(tmp_path)
    ds.initialize_mem_wal(identity_column="name")

    details = ds.mem_wal_index_details()
    # identity sharding has an open-ended shard count
    assert details["num_shards"] == 0
    field = details["sharding_specs"][0]["fields"][0]
    assert field["transform"] == "identity"
    assert field["result_type"] == "utf8"


def test_initialize_mem_wal_maintained_indexes(tmp_path):
    ds = _mem_wal_dataset(tmp_path)
    ds.create_scalar_index("id", "BTREE", name="id_btree")
    ds.initialize_mem_wal(maintained_indexes=["id_btree"])

    assert ds.mem_wal_index_details()["maintained_indexes"] == ["id_btree"]


def test_initialize_mem_wal_writer_config_defaults(tmp_path):
    ds = _mem_wal_dataset(tmp_path)
    ds.initialize_mem_wal(
        durable_write=False,
        max_memtable_size=8 * 1024 * 1024,
        max_wal_flush_interval_ms=250,
    )

    defaults = ds.mem_wal_index_details()["writer_config_defaults"]
    assert defaults["durable_write"] == "false"
    assert defaults["max_memtable_size"] == str(8 * 1024 * 1024)
    # Duration knobs are recorded in milliseconds with a `_ms` suffix.
    assert defaults["max_wal_flush_interval_ms"] == "250"
    # Every ShardWriterConfig tunable is recorded once any default is set.
    assert "enable_memtable" in defaults


def test_initialize_mem_wal_no_writer_config_defaults_when_unset(tmp_path):
    ds = _mem_wal_dataset(tmp_path)
    ds.initialize_mem_wal(bucket_column="id", num_buckets=4)
    assert ds.mem_wal_index_details()["writer_config_defaults"] == {}


def test_initialize_mem_wal_rejects_conflicting_sharding(tmp_path):
    ds = _mem_wal_dataset(tmp_path)
    with pytest.raises(ValueError, match="at most one"):
        ds.initialize_mem_wal(unsharded=True, bucket_column="id", num_buckets=8)


def test_initialize_mem_wal_rejects_partial_bucket(tmp_path):
    ds = _mem_wal_dataset(tmp_path)
    with pytest.raises(ValueError, match="num_buckets"):
        ds.initialize_mem_wal(bucket_column="id")


def test_evaluate_sharding_spec_python_binding():
    batch = pa.record_batch(
        [pa.array([1, 2, None, 3], type=pa.int32())],
        names=["id"],
    )
    spec = ShardingSpec(
        1,
        [
            ShardingField(
                field_id="bucket",
                source_ids=[0],
                transform="bucket",
                result_type="int32",
                parameters={"num_buckets": "8"},
            )
        ],
    )

    result = evaluate_sharding_spec(batch, spec, LanceSchema.from_pyarrow(batch.schema))

    assert result.column_names == ["bucket"]
    assert result.column(0).to_pylist() == [2, 7, 0, 1]


def test_evaluate_sharding_spec_python_binding_column_parameter():
    batch = pa.record_batch(
        [pa.array(["a", "b", None], type=pa.utf8())],
        names=["key"],
    )
    spec = {
        "spec_id": 1,
        "fields": [
            {
                "field_id": "key_bucket",
                "source_ids": [0],
                "transform": "bucket",
                "expression": None,
                "result_type": "int32",
                "parameters": {"num_buckets": "8"},
            }
        ],
    }

    result = evaluate_sharding_spec(batch, spec, LanceSchema.from_pyarrow(batch.schema))

    assert result.column_names == ["key_bucket"]
    assert result.column(0).to_pylist() == [1, 5, 0]


def test_mem_wal_index_details_none_before_init(tmp_path):
    ds = _mem_wal_dataset(tmp_path)
    assert ds.mem_wal_index_details() is None
