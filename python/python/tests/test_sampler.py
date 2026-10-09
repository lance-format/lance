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


@pytest.mark.parametrize("n", [100, 3000, 4095])
def test_sample_dataset_returns_n_rows(tmp_path: Path, n: int):
    ds = lance.write_dataset(pa.table({"id": range(10000)}), tmp_path / "data.lance")

    batches = list(maybe_sample(ds, n, ["id"], max_takes=32))

    ids = pa.concat_arrays([batch["id"] for batch in batches]).to_pylist()
    assert len(ids) == n
    assert len(set(ids)) == n
    # Samples are spread over the whole dataset, not just its head.
    assert max(ids) >= 10000 - 10000 // 32
