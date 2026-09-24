# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.sampler import ShardedBatchSampler, ShardedFixedBatchSampler, maybe_sample

TEST_CONFIG = {
    "total_rows": 1000,
    "batch_size": 250,
    "world_size": 4,
    "vec_dim": 32,
    "test_port": "29501",
    "master_addr": "127.0.0.1",
    "seed": 42,
    "test_shard_ratio": 0.5,
    "max_takes_factor": 0.1,
}


@pytest.fixture
def sample_dataset_path(tmp_path):
    data = pa.Table.from_arrays(
        [
            pa.array(range(TEST_CONFIG["total_rows"])),
            pa.array(np.random.rand(TEST_CONFIG["total_rows"])),
            pa.array([f"text_{i}" for i in range(TEST_CONFIG["total_rows"])]),
        ],
        names=["id", "value", "text"],
    )

    dataset_path = tmp_path / "test_dataset.lance"
    lance.write_dataset(data, dataset_path)
    return dataset_path


@pytest.fixture
def sample_dataset(sample_dataset_path) -> lance.LanceDataset:
    return lance.dataset(sample_dataset_path)


def test_consecutive_index_blocks():
    sampler = ShardedFixedBatchSampler(
        rank=0,
        world_size=TEST_CONFIG["world_size"],
        total_num_rows=TEST_CONFIG["total_rows"],
        batch_size=TEST_CONFIG["batch_size"],
    )

    batches = list(sampler)
    expected_size = TEST_CONFIG["total_rows"] // (
        TEST_CONFIG["world_size"] * TEST_CONFIG["batch_size"]
    )
    assert len(batches) == expected_size
    assert batches[0] == list(range(TEST_CONFIG["batch_size"]))


def _distributed_test_worker(rank, world_size, dataset_path):
    import os

    import torch

    os.environ.update(
        {
            "MASTER_ADDR": TEST_CONFIG["master_addr"],
            "MASTER_PORT": TEST_CONFIG["test_port"],
            "CUDA_VISIBLE_DEVICES": ",".join(
                map(str, range(torch.cuda.device_count()))
            ),
        }
    )

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(
            backend=backend, world_size=world_size, rank=rank
        )

        dataset = lance.dataset(dataset_path)
        assert len(dataset) == TEST_CONFIG["total_rows"]

        sampler = ShardedBatchSampler(
            rank=rank,
            world_size=world_size,
            total_num_rows=TEST_CONFIG["total_rows"],
            batch_size=TEST_CONFIG["batch_size"],
        )

        class DatasetAdapter(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __getitem__(self, index):
                return self.dataset.take([index], ["id", "value"]).to_pylist()[0]

            def __len__(self):
                return len(self.dataset)

        def collate_fn(batch):
            return {
                "ids": torch.tensor([x["id"] for x in batch], dtype=torch.long),
                "values": torch.tensor(
                    [x["value"] for x in batch], dtype=torch.float32
                ),
            }

        dataloader = torch.utils.data.DataLoader(
            DatasetAdapter(dataset),
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )

        total = 0
        for batch_indices, batch_data in zip(sampler, dataloader):
            current_size = batch_data["ids"].size(0)
            assert current_size == TEST_CONFIG["batch_size"]
            assert batch_data["ids"].tolist() == list(batch_indices)
            total += current_size

        expected_total = TEST_CONFIG["total_rows"] // world_size
        assert total == expected_total

    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.cuda
def test_pytorch_integration(sample_dataset_path):
    import torch

    test_world_sizes = [1, 2] if torch.cuda.device_count() >= 2 else [1]
    for ws in test_world_sizes:
        torch.multiprocessing.spawn(
            _distributed_test_worker,
            args=(ws, str(sample_dataset_path)),
            nprocs=ws,
            join=True,
        )


def test_data_stream_without_filter(sample_dataset):
    """Validate direct data loading without filters."""
    sampler = ShardedFixedBatchSampler(0, 4)
    batches = list(sampler(sample_dataset, batch_size=250, columns=["id", "value"]))

    # Data integrity checks
    batch = batches[0]
    assert batch.num_rows == 250, "Batch should contain 250 records"
    assert batch.column_names == ["id", "value"], "Should load specified columns"

    # Consecutive ID validation
    ids = batch["id"].to_numpy()
    assert np.array_equal(ids, np.arange(0, 250)), "IDs should be sequential 0-249"


def test_filtered_data_handling(sample_dataset):
    """Test filtered data processing with sharding."""
    # Apply ID filter and load data
    sampler = ShardedFixedBatchSampler(0, 4)
    batches = list(
        sampler(sample_dataset, batch_size=100, filter="id < 500", columns=["id"])
    )

    # Aggregated results validation
    all_ids = []
    for batch in batches:
        all_ids.extend(batch["id"].to_numpy().tolist())

    # Filter and sharding assertions
    assert all(id_val < 500 for id_val in all_ids), "Should respect ID filter"
    assert all(id_val % 4 == 0 for id_val in all_ids), "Should keep rank 0 shard"


def _filtered_ids(sampler, ds, filter):
    batches = list(sampler(ds, batch_size=1, columns=["id"], filter=filter))
    return [row for batch in batches for row in batch.column("id").to_pylist()]


@pytest.mark.parametrize("randomize", [False, True])
def test_sharded_batch_sampler_empty_filtered_shard(tmp_path, randomize):
    """Empty filtered shards must yield an empty stream, not crash.

    ShardedBatchSampler._shard_scan has two empty cases on the filtered path:

    1. Global zero-match: the filter matches no rows, so nothing accumulates and
       the final pa.Table.from_batches([]) raises ValueError.
    2. Empty per-rank shard: a sparse filter with world_size > 1 leaves some
       ranks with zero rows after round-robin, so batch.take([]) builds a
       null-typed index array and raises ArrowNotImplementedError.

    A rank with no matching rows must produce an empty iterator, matching the
    unfiltered (_sample_all) path's contract.
    """
    uri = str(tmp_path / "ds")
    lance.write_dataset(
        pa.table({"id": list(range(6)), "seg": [f"s{i % 2}" for i in range(6)]}),
        uri,
    )
    ds = lance.dataset(uri)

    # 3 rows match seg='s1' (ids 1,3,5); rank 3 of 4 gets none.
    sampler = ShardedBatchSampler(rank=3, world_size=4, randomize=randomize)
    assert _filtered_ids(sampler, ds, "seg IN ('s1')") == []

    # Filter matching nothing -> empty stream, no crash.
    sampler = ShardedBatchSampler(rank=0, world_size=1, randomize=randomize)
    assert _filtered_ids(sampler, ds, "seg IN ('nope')") == []

    # Non-empty shard unchanged: rank 0 of 4 from [1,3,5] -> [1].
    sampler = ShardedBatchSampler(rank=0, world_size=4, randomize=randomize)
    assert _filtered_ids(sampler, ds, "seg IN ('s1')") == [1]


def test_sharded_batch_sampler_filtered_carryover_across_fragments(tmp_path):
    """Round-robin offset must carry correctly across small filtered batches.

    When a filtered scan splits matches across several small batches/fragments,
    a rank's offset has to carry from one batch to the next. If the carry is
    computed from a sliced (truncated) batch instead of the full batch length,
    ranks desync and rows are dropped or double-assigned. Here every row matches
    and each fragment holds a single row, so the per-rank shards must partition
    the global sequence exactly.
    """
    uri = str(tmp_path / "ds")
    num_rows = 8
    for i in range(num_rows):
        lance.write_dataset(
            pa.table({"id": [i], "seg": ["s1"]}),
            uri,
            mode="overwrite" if i == 0 else "append",
        )
    ds = lance.dataset(uri)

    world_size = 4
    shards = {
        rank: _filtered_ids(
            ShardedBatchSampler(rank=rank, world_size=world_size),
            ds,
            "seg IN ('s1')",
        )
        for rank in range(world_size)
    }

    # Each rank owns global positions rank, rank + world_size, ...; ids equal
    # their global position here, so the shards are a disjoint, complete cover.
    for rank in range(world_size):
        assert shards[rank] == list(range(rank, num_rows, world_size))
    combined = sorted(i for ids in shards.values() for i in ids)
    assert combined == list(range(num_rows))


def test_randomization_effect():
    """Verify epoch-based randomization behavior."""
    # Initialize randomized sampler
    sampler = ShardedFixedBatchSampler(
        rank=0,
        world_size=4,
        total_num_rows=2000,
        batch_size=250,
        randomize=True,
        seed=42,
    )

    assert len(list(sampler)) > 1

    # Cross-epoch comparison
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    sampler.set_epoch(2)
    epoch2 = list(sampler)

    assert epoch1 != epoch2, "Different epochs should produce different orders"


def test_edge_cases():
    """Validate handling of partial batches and data boundaries."""

    sampler = ShardedFixedBatchSampler(
        rank=3, world_size=4, batch_size=250, total_num_rows=1000
    )
    batches = list(sampler)
    assert len(batches) == 1, "Should handle partial batch"
    assert batches[0] == list(range(750, 1000)), "Last rank should get 750-999"

    sampler = ShardedFixedBatchSampler(
        rank=0, world_size=2, batch_size=128, total_num_rows=500
    )
    batches = list(sampler)
    # rank 0: 0~249, rank 1: 250~499
    # rank 0: [0-127], [128-249]
    assert batches[0] == list(range(0, 128))
    assert batches[1] == list(range(128, 250))

    # total_num_rows < batch_size
    sampler = ShardedFixedBatchSampler(
        rank=0, world_size=1, batch_size=250, total_num_rows=100
    )
    batches = list(sampler)
    assert len(batches) == 1
    assert batches[0] == list(range(0, 100))

    # total_num_rows < world_size
    sampler = ShardedFixedBatchSampler(
        rank=2, world_size=4, batch_size=10, total_num_rows=2
    )
    batches = list(sampler)
    assert len(batches) == 0, "No data for this rank"

    # batch_size=1
    sampler = ShardedFixedBatchSampler(
        rank=0, world_size=2, batch_size=1, total_num_rows=4
    )
    batches = list(sampler)
    assert batches == [[0], [1]]

    # world_size=1
    sampler = ShardedFixedBatchSampler(
        rank=0, world_size=1, batch_size=3, total_num_rows=5
    )
    batches = list(sampler)
    assert batches == [list(range(0, 3)), list(range(3, 5))]


# We use + 97 to test case where num_rows and chunk_size aren't exactly aligned.
@pytest.mark.parametrize("nrows", [10, 10240, 10240 + 97, 10240 + 1024])
def test_sample_dataset(tmp_path: Path, nrows: int):
    # nrows of 32-d vectors.
    data = np.random.random(nrows * 32).astype("f")

    fsl = pa.FixedSizeListArray.from_arrays(data, 32)
    tbl = pa.Table.from_arrays([fsl], ["vec"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance")

    # Simple path
    simple_scan = list(maybe_sample(ds, 128, ["vec"]))
    assert len(simple_scan) == 1
    assert isinstance(simple_scan[0], pa.RecordBatch)
    assert simple_scan[0].schema == pa.schema([pa.field("vec", fsl.type)])
    assert simple_scan[0].num_rows == min(nrows, 128)

    # Random path.
    large_scan = list(maybe_sample(ds, 128, ["vec"], max_takes=32))
    assert len(large_scan) == 1
    assert isinstance(large_scan[0], pa.RecordBatch)
    assert large_scan[0].schema == pa.schema([pa.field("vec", fsl.type)])
    assert large_scan[0].num_rows == min(nrows, 128)
