# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import copy
import multiprocessing
import pickle
import shutil
import weakref
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

import lance
import numpy as np
import psutil
import pyarrow as pa
import pytest
from lance._dataset.cache import CachedDataset
from lance.sampler import ShardedBatchSampler, ShardedFragmentSampler

torch = pytest.importorskip("torch")
from lance.torch.data import (  # noqa: E402
    LanceDataset,
    SafeLanceDataset,
    _bf16_to_tensor,
)


def test_iter_over_dataset_fixed_shape_tensor(tmp_path):
    data = np.random.random((10240, 32)).astype("f")

    tensor_array = pa.FixedShapeTensorArray.from_numpy_ndarray(data)
    ids = pa.array(range(0, 10240), type=pa.int32())
    tbl = pa.Table.from_arrays([ids, tensor_array], ["ids", "vec"])

    lance.write_dataset(tbl, tmp_path / "data.lance")

    iter_over_dataset(tmp_path)


def test_iter_over_dataset_fixed_size_lists(tmp_path):
    # 10240 of 32-d vectors.
    data = np.random.random(10240 * 32).astype("f")

    fsl = pa.FixedSizeListArray.from_arrays(data, 32)
    ids = pa.array(range(0, 10240), type=pa.int32())
    tbl = pa.Table.from_arrays([ids, fsl], ["ids", "vec"])

    lance.write_dataset(tbl, tmp_path / "data.lance", max_rows_per_group=32)

    iter_over_dataset(tmp_path)


def iter_over_dataset(tmp_path):
    ds = lance.dataset(tmp_path / "data.lance")

    # test when sample size is smaller than max_takes
    torch_ds_small = LanceDataset(
        ds, batch_size=256, samples=1024, columns=["ids", "vec"], cache=True
    )

    total_rows = 0
    for batch in torch_ds_small:
        assert set(batch.keys()) == {"ids", "vec"}
        # row groups of 32 can be batched into 256 exactly.
        assert batch["vec"].shape[0] == 256
        total_rows += batch["vec"].shape[0]
        assert batch["ids"].dtype == torch.int32
        assert batch["vec"].shape[1] == 32
    assert total_rows == 1024

    # test when sample size is greater than max_takes
    torch_ds = LanceDataset(
        ds,
        batch_size=256,
        samples=4096,
        columns=["ids", "vec"],
        cache=True,
        batch_readahead=2,
    )

    total_rows = 0
    for batch in torch_ds:
        assert set(batch.keys()) == {"ids", "vec"}
        # row groups of 32 can be batched into 256 exactly.
        assert batch["vec"].shape[0] == 256
        total_rows += batch["vec"].shape[0]
        assert batch["ids"].dtype == torch.int32
        assert batch["vec"].shape[1] == 32
    assert total_rows == 4096

    shutil.rmtree(tmp_path / "data.lance")

    total_rows = 0
    # it should read from cache this time.
    for batch in torch_ds_small:
        assert set(batch.keys()) == {"ids", "vec"}
        assert batch["ids"].dtype == torch.int32
        total_rows += batch["vec"].shape[0]
        assert batch["vec"].shape[1] == 32
    assert total_rows == 1024

    total_rows = 0
    # it should read from cache this time.
    for batch in torch_ds:
        assert set(batch.keys()) == {"ids", "vec"}
        assert batch["ids"].dtype == torch.int32
        total_rows += batch["vec"].shape[0]
        assert batch["vec"].shape[1] == 32
    assert total_rows == 4096


@pytest.mark.parametrize("cache_location", ["temporary", "directory"])
def test_interrupted_cache_fill_restarts(tmp_path, cache_location, monkeypatch):
    source = tmp_path / "source.lance"
    lance.write_dataset(pa.table({"id": range(20)}), source, max_rows_per_file=5)
    cache_base = tmp_path / "cache"
    cache_base.mkdir()
    cache = True if cache_location == "temporary" else str(cache_base)
    cleanup = TemporaryDirectory.cleanup
    new_stream = pa.ipc.new_stream
    retained_writers = []

    def windows_cleanup(directory):
        opened = {Path(file.path).resolve() for file in psutil.Process().open_files()}
        if (Path(directory.name) / "cache.arrow").resolve() in opened:
            raise PermissionError("emulated Windows open-file deletion denial")
        cleanup(directory)

    def retain_writer(sink, schema):
        writer = new_stream(sink, schema)
        retained_writers.append((writer, sink))
        return writer

    monkeypatch.setattr(TemporaryDirectory, "cleanup", windows_cleanup)
    # Retain both objects so garbage collection cannot hide an open sink.
    monkeypatch.setattr(pa.ipc, "new_stream", retain_writer)
    dataset = LanceDataset(str(source), batch_size=1, cache=cache)
    try:
        batches = iter(dataset)
        assert next(batches).item() == 0
        partial = weakref.ref(dataset.cached_ds)
        partial_path = Path(dataset.cached_ds.cache_dir.name)
        batches.close()
        assert not dataset.cached_ds.finished_origin_stream
        assert dataset.cached_ds.cache_dir is None
        assert not partial_path.exists()
        assert retained_writers[0][1].closed

        assert [batch.item() for batch in dataset] == list(range(20))
        assert partial() is None
        completed = dataset.cached_ds
        assert completed.finished_origin_stream
        shutil.rmtree(source)
        batches = iter(dataset)
        assert next(batches).item() == 0
        batches.close()
        assert dataset.cached_ds is completed
        assert all(sink.closed for _, sink in retained_writers)
        assert [batch.item() for batch in dataset] == list(range(20))
        assert dataset.cached_ds is completed
    finally:
        if dataset.cached_ds is not None:
            dataset.cached_ds.close()
    assert list(cache_base.iterdir()) == []


def test_empty_cache_replays():
    class EmptyStream:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(())

    stream = EmptyStream()
    with CachedDataset(stream) as dataset:
        assert list(dataset) == []
        assert dataset.finished_origin_stream
        assert list(dataset) == []
    assert stream.iterations == 1


def test_cache_with_empty_result(tmp_path):
    lance.write_dataset(pa.table({"x": range(8)}), tmp_path / "data.lance")
    torch_ds = LanceDataset(
        tmp_path / "data.lance", batch_size=4, filter="x > 100", cache=True
    )
    assert list(torch_ds) == []
    # A second pass is also empty.
    assert list(torch_ds) == []


def test_iter_filter(tmp_path):
    arr = pa.array(range(1000))
    tbl = pa.Table.from_arrays([arr], ["ids"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance", max_rows_per_group=32)

    def check(dataset):
        total_rows = 0
        for batch in dataset:
            assert torch.where(batch >= 300, True, False).all()
            total_rows += batch.size(dim=0)
            assert batch.dtype == torch.int64
        assert total_rows == 700

    # No shard_grandularity
    check(
        LanceDataset(
            ds,
            batch_size=10,
            filter="ids >= 300",
            columns=["ids"],
        )
    )

    # shard_grandularity fragment ok
    check(
        LanceDataset(
            ds,
            batch_size=10,
            filter="ids >= 300",
            columns=["ids"],
            sampler=ShardedFragmentSampler(0, 1),
        )
    )

    # sampling with filter
    with pytest.raises(NotImplementedError):
        check(
            LanceDataset(
                ds,
                batch_size=10,
                filter="ids >= 300",
                samples=100,
                columns=["ids"],
            )
        )


@pytest.mark.parametrize("shard_granularity", ["bogus", "Batch"])
def test_invalid_shard_granularity(tmp_path: Path, shard_granularity):
    ds = lance.write_dataset(pa.table({"ids": range(100)}), tmp_path / "data.lance")
    dataset = LanceDataset(ds, batch_size=10, shard_granularity=shard_granularity)
    with pytest.raises(
        ValueError, match=f"Invalid shard_granularity: {shard_granularity}"
    ):
        next(iter(dataset))


def test_sample_fragments(tmp_path: Path):
    arr = pa.array(range(2000))
    tbl = pa.Table.from_arrays([arr], ["ids"])

    # Write 20 files
    lance.write_dataset(tbl, tmp_path, max_rows_per_file=100)

    ds = LanceDataset(
        tmp_path,
        batch_size=25,
        columns=["ids"],
        with_row_id=True,
        sampler=ShardedFragmentSampler(rank=1, world_size=2),
    )

    all_ids = list(chain.from_iterable([batch["ids"].cpu().numpy() for batch in ds]))
    assert all_ids == [i for i in range(2000) if i // 100 % 2 == 1]


def test_sample_batches(tmp_path: Path):
    arr = pa.array(range(2000))
    tbl = pa.Table.from_arrays([arr], ["ids"])

    # Write 20 files
    lance.write_dataset(tbl, tmp_path, max_rows_per_file=100)

    ds = LanceDataset(
        tmp_path,
        batch_size=25,
        columns=["ids"],
        with_row_id=True,
        sampler=ShardedBatchSampler(rank=1, world_size=2),
    )

    all_ids = list(chain.from_iterable([batch.cpu().numpy() for batch in ds]))
    assert all_ids == [i for i in range(2000) if i // 25 % 2 == 1]


@pytest.mark.parametrize("sampler_cls", [ShardedFragmentSampler, ShardedBatchSampler])
@pytest.mark.parametrize(
    "start_method, sharing_strategy",
    [
        ("spawn", None),
        # A parent on file_system must still reach a worker that starts on its
        # own default strategy (file_descriptor on Linux). forkserver is the
        # Linux default from Python 3.14. fork is left out: forking an
        # initialized lance runtime is unsupported.
        pytest.param(
            "forkserver",
            "file_system",
            marks=pytest.mark.skipif(
                "forkserver" not in multiprocessing.get_all_start_methods(),
                reason="forkserver is not available",
            ),
        ),
    ],
    ids=["spawn", "forkserver-file_system"],
)
def test_set_epoch_reaches_persistent_workers(
    tmp_path: Path, request, sampler_cls, start_method, sharing_strategy
):
    if sharing_strategy is not None:
        previous = torch.multiprocessing.get_sharing_strategy()
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)
        request.addfinalizer(
            lambda: torch.multiprocessing.set_sharing_strategy(previous)
        )
    lance.write_dataset(pa.table({"ids": range(400)}), tmp_path, max_rows_per_file=50)
    # Non-consecutive epochs: workers must see the value set_epoch stored,
    # not a count of how many epochs they have run.
    epochs = [0, 1, 5]

    def epoch_orders(loader, sampler):
        orders = []
        for epoch in epochs:
            sampler.set_epoch(epoch)
            orders.append([i for batch in loader for i in batch.tolist()])
        return orders

    def make(assign_after_init, **loader_kwargs):
        sampler = sampler_cls(rank=0, world_size=1, randomize=True, seed=42)
        if assign_after_init:
            dataset = LanceDataset(tmp_path, batch_size=25, columns=["ids"])
            dataset.sampler = sampler
        else:
            dataset = LanceDataset(
                tmp_path, batch_size=25, columns=["ids"], sampler=sampler
            )
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, **loader_kwargs)
        return loader, sampler

    expected = epoch_orders(*make(False, num_workers=0))
    assert len({tuple(order) for order in expected}) == len(epochs)

    # Assign the sampler after construction so the persistent run also pins
    # that later assignments go through the sharing property setter.
    persistent = epoch_orders(
        *make(
            True,
            num_workers=1,
            persistent_workers=True,
            multiprocessing_context=start_method,
        )
    )
    assert persistent == expected


@pytest.mark.parametrize("sampler_cls", [ShardedFragmentSampler, ShardedBatchSampler])
@pytest.mark.parametrize(
    "context",
    [lambda: torch.device("meta"), torch.inference_mode],
    ids=["meta", "inference"],
)
def test_set_epoch_ignores_torch_construction_context(
    tmp_path: Path, sampler_cls, context
):
    lance.write_dataset(pa.table({"ids": range(10)}), tmp_path)
    sampler = sampler_cls(rank=0, world_size=1, randomize=True)
    sampler.set_epoch(2)
    # Sharing under a default device or inference mode must still give a CPU
    # tensor set_epoch can update, and the epoch already set must survive.
    with context():
        LanceDataset(tmp_path, batch_size=5, sampler=sampler)
    # spawn and forkserver workers would get shared storage from torch's pickler
    # anyway; fork workers only see the epoch because it is shared up front.
    assert sampler._epoch_value.is_shared()
    assert sampler._epoch == 2
    sampler.set_epoch(3)
    assert sampler._epoch == 3
    # Subclasses that assign _epoch directly must update the shared storage too.
    sampler._epoch = 4
    assert sampler._epoch == 4
    assert isinstance(sampler._epoch_value, torch.Tensor)

    # A deep copy holds an unshared tensor; LanceDataset must share it again.
    clone = copy.deepcopy(sampler)
    assert not clone._epoch_value.is_shared()
    dataset = LanceDataset(tmp_path, batch_size=5, sampler=clone)
    assert clone._epoch_value.is_shared()

    # Copying the dataset itself bypasses the sampler setter.
    for copied in (copy.deepcopy(dataset), pickle.loads(pickle.dumps(dataset))):
        assert copied.sampler is not clone
        assert copied.sampler._epoch_value.is_shared()
        assert copied.sampler._epoch == 4
        copied.sampler.set_epoch(5)
        assert copied.sampler._epoch == 5
        assert clone._epoch == 4

    # A copy made inside inference mode must not leave an inference tensor that
    # set_epoch cannot update outside it.
    with torch.inference_mode():
        copied = copy.deepcopy(dataset)
        sampler_copy = copy.deepcopy(sampler)
    copied.sampler.set_epoch(5)
    assert copied.sampler._epoch == 5
    # A sampler copied on its own never passes through LanceDataset.
    sampler_copy.set_epoch(6)
    assert sampler_copy._epoch == 6
    # set_epoch inside inference mode still writes the shared storage.
    with torch.inference_mode():
        sampler.set_epoch(7)
    assert sampler._epoch_value.is_shared()
    sampler.set_epoch(8)
    assert sampler._epoch == 8


def test_set_epoch_keeps_storage_received_from_another_process(
    tmp_path: Path, monkeypatch
):
    # A spawn or forkserver worker unpickles the dataset with the epoch already
    # in shared memory. Sharing it again under a different strategy would move
    # it off the parent's storage. test_set_epoch_reaches_persistent_workers
    # covers this end to end where the strategies differ (Linux); this checks
    # it on platforms with a single strategy too.
    lance.write_dataset(pa.table({"ids": range(10)}), tmp_path)
    sampler = ShardedBatchSampler(rank=0, world_size=1, randomize=True)
    dataset = LanceDataset(tmp_path, batch_size=5, sampler=sampler)
    storage = sampler._epoch_value.data_ptr()
    current = torch.multiprocessing.get_sharing_strategy()
    other = "file_descriptor" if current == "file_system" else "file_system"
    monkeypatch.setattr(torch.multiprocessing, "get_sharing_strategy", lambda: other)
    dataset.sampler = sampler
    assert sampler._epoch_value.data_ptr() == storage


def test_set_epoch_rejects_meta_epoch(tmp_path: Path):
    lance.write_dataset(pa.table({"ids": range(10)}), tmp_path)
    sampler = ShardedBatchSampler(rank=0, world_size=1, randomize=True)
    sampler._epoch_value = torch.empty((), dtype=torch.int64, device="meta")
    with pytest.raises(ValueError, match="meta device"):
        LanceDataset(tmp_path, batch_size=5, sampler=sampler)


_ACCELERATORS = [
    device
    for device, available in (
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    )
    if available
]


@pytest.mark.skipif(not _ACCELERATORS, reason="needs a CUDA or MPS device")
def test_set_epoch_moves_epoch_loaded_onto_an_accelerator(tmp_path: Path):
    # torch.load(..., map_location="cuda") puts the epoch tensor on the GPU,
    # where share_memory_() does nothing.
    lance.write_dataset(pa.table({"ids": range(10)}), tmp_path)
    sampler = ShardedBatchSampler(rank=0, world_size=1, randomize=True)
    sampler._epoch_value = torch.tensor(3, device=_ACCELERATORS[0])
    LanceDataset(tmp_path, batch_size=5, sampler=sampler)
    assert sampler._epoch_value.device.type == "cpu"
    assert sampler._epoch_value.is_shared()
    assert sampler._epoch == 3


def test_set_epoch_survives_copies_of_subclasses_with_slots(tmp_path: Path):
    # With __slots__ the pickled state is a (__dict__, slots) tuple.
    class SlottedSampler(ShardedBatchSampler):
        __slots__ = ("extra",)

    class SlottedDataset(LanceDataset):
        __slots__ = ("extra",)

    lance.write_dataset(pa.table({"ids": range(10)}), tmp_path)
    sampler = SlottedSampler(rank=0, world_size=1, randomize=True)
    sampler.extra = 1
    dataset = SlottedDataset(tmp_path, batch_size=5, sampler=sampler)
    dataset.extra = 2
    sampler.set_epoch(3)
    for copied in (copy.copy(dataset), copy.deepcopy(dataset)):
        assert copied.extra == 2
        assert copied.sampler.extra == 1
        assert copied.sampler._epoch == 3
        assert copied.sampler._epoch_value.is_shared()


def test_copy_dataset_subclass_without_sampler():
    # A subclass that never runs LanceDataset.__init__ has no sampler to share.
    class BareDataset(LanceDataset):
        def __init__(self, n):
            self.n = n

    for copied in (copy.copy(BareDataset(4)), copy.deepcopy(BareDataset(4))):
        assert copied.n == 4


def test_set_epoch_reads_datasets_pickled_by_earlier_releases(tmp_path: Path):
    # Earlier releases stored LanceDataset.sampler as "sampler" and the sampler's
    # epoch as "_epoch"; both names are now shadowed by properties.
    lance.write_dataset(pa.table({"ids": range(10)}), tmp_path)
    sampler = ShardedFragmentSampler(rank=0, world_size=1, randomize=True)
    dataset = LanceDataset(tmp_path, batch_size=5, sampler=sampler)
    sampler_state = dict(vars(sampler))
    del sampler_state["_epoch_value"]
    sampler_state["_epoch"] = 3
    legacy_sampler = ShardedFragmentSampler.__new__(ShardedFragmentSampler)
    legacy_sampler.__setstate__(sampler_state)
    assert legacy_sampler._epoch == 3
    # Earlier releases also accepted float epochs.
    float_sampler = ShardedFragmentSampler.__new__(ShardedFragmentSampler)
    float_sampler.__setstate__({**sampler_state, "_epoch": 2.0})
    assert float_sampler._epoch == 2

    state = dict(vars(dataset))
    del state["_sampler"]
    state["sampler"] = legacy_sampler
    restored = LanceDataset.__new__(LanceDataset)
    restored.__setstate__(state)
    assert restored.sampler is legacy_sampler
    assert legacy_sampler._epoch == 3
    assert legacy_sampler._epoch_value.is_shared()


def test_set_epoch_normalizes_integer_epochs(tmp_path: Path):
    lance.write_dataset(pa.table({"ids": range(10)}), tmp_path)
    sampler = ShardedBatchSampler(rank=0, world_size=1, randomize=True)
    caller_epoch = torch.tensor(1)
    sampler.set_epoch(caller_epoch)
    sampler.set_epoch(np.int64(2))
    # The caller's tensor is not adopted as the epoch storage.
    assert caller_epoch.item() == 1

    LanceDataset(tmp_path, batch_size=5, sampler=sampler)
    assert sampler._epoch_value.is_shared()
    sampler.set_epoch(np.int64(3))
    assert sampler._epoch == 3

    # Earlier releases accepted float epochs; integral ones are still accepted.
    sampler.set_epoch(4.0)
    assert sampler._epoch == 4
    sampler.set_epoch(np.float64(3))
    assert sampler._epoch == 3

    for invalid in (1.5, float("nan"), None):
        with pytest.raises(TypeError, match="epoch must be an integer"):
            sampler.set_epoch(invalid)
    assert sampler._epoch == 3


def test_filtered_sampling_odd_batch_size(tmp_path: Path):
    tbl = pa.Table.from_pydict(
        {
            "vector": pa.array(
                [[1.0, 2.0, 3.0] for _ in range(10000)], pa.list_(pa.float32(), 3)
            ),
            "filterme": [i % 2 for i in range(10000)],
        }
    )

    lance.write_dataset(tbl, tmp_path, max_rows_per_file=200)

    ds = LanceDataset(
        tmp_path,
        batch_size=38,
        columns=["vector"],
        samples=38 * 256,
        filter="vector is not null",
    )

    x = next(iter(ds))

    assert x.shape[0] == 38
    assert x.shape[1] == 3


def test_sample_batches_with_filter(tmp_path: Path):
    NUM_ROWS = 10000
    tbl = pa.Table.from_pydict(
        {
            "id": range(NUM_ROWS),
            "filterme": [i % 2 for i in range(NUM_ROWS)],
        }
    )

    lance.write_dataset(tbl, tmp_path, max_rows_per_file=2000)

    ds = LanceDataset(
        tmp_path,
        batch_size=25,
        columns=["id"],
        with_row_id=True,
        filter="filterme == 0",
        sampler=ShardedBatchSampler(rank=3, world_size=5),
    )

    # The filtered sequence is 0, 2, 4, ...
    #
    # With rank 3 and world size 5 we should get
    #
    # - - - 6  -
    # - - - 16 -
    # - - - 26 -
    # ...
    all_ids = list(chain.from_iterable([batch.cpu().numpy() for batch in ds]))
    # Half of the data is filtered out, divided amongst 5 workers s
    # each should see 1/10th of the data
    assert len(all_ids) == 1000
    assert all_ids == [6 + (10 * i) for i in range(len(all_ids))]

    # Now test with random order
    ds = LanceDataset(
        tmp_path,
        batch_size=25,
        columns=["id"],
        with_row_id=True,
        filter="filterme == 0",
        sampler=ShardedBatchSampler(rank=3, world_size=5, randomize=True),
    )

    randomized_ids = list(chain.from_iterable([batch.cpu().numpy() for batch in ds]))
    assert randomized_ids != all_ids
    randomized_ids.sort()
    assert randomized_ids == all_ids


@pytest.mark.parametrize("dtype", [np.uint8, np.int64])
def test_convert_int_tensors(tmp_path: Path, dtype):
    data = np.random.randint(0, 256, size=128 * 32, dtype=dtype)
    fsl = pa.FixedSizeListArray.from_arrays(data, 32)
    ids = pa.array(range(0, 128), type=pa.int32())
    tbl = pa.Table.from_arrays([ids, fsl], ["ids", "vec"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance", max_rows_per_group=32)

    torch_ds = LanceDataset(
        ds,
        batch_size=4,
    )
    first = next(iter(torch_ds))
    assert first["vec"].dtype == torch.uint8 if dtype == np.uint8 else torch.int64
    assert first["vec"].shape == (4, 32)


def test_blob_v2_api(tmp_path: Path):
    payloads = [bytes([i]) * 64 for i in range(8)]
    tbl = pa.Table.from_arrays(
        [pa.array(range(8), pa.int64()), lance.blob_array(payloads)],
        schema=pa.schema([pa.field("int", pa.int64()), lance.blob_field("val")]),
    )
    dataset = lance.write_dataset(
        tbl, tmp_path / "data.lance", data_storage_version="2.2"
    )
    dataset = lance.dataset(tmp_path / "data.lance")
    blob_type = dataset.schema.field("val").type
    assert isinstance(blob_type, pa.ExtensionType)
    assert blob_type.extension_name == "lance.blob.v2"

    seen = []

    def to_tensor_fn(batch, *args, **kwargs):
        for blob in batch["val"]:
            assert isinstance(blob, lance.BlobFile)
            seen.append(blob.read())
        return {"int": torch.tensor(batch["int"].to_numpy())}

    torch_ds = LanceDataset(dataset, batch_size=4, to_tensor_fn=to_tensor_fn)
    assert len(list(torch_ds)) == 2
    assert seen == payloads


def test_blob_api(tmp_path: Path):
    ints = pa.array(range(100), type=pa.int64())
    vals = pa.array([b"0" * 1024 for _ in range(100)], pa.large_binary())
    schema = pa.schema(
        [
            pa.field("int", ints.type),
            pa.field(
                "val", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
            ),
        ]
    )
    tbl = pa.Table.from_arrays([ints, vals], schema=schema)

    uri = tmp_path / "data.lance"
    dataset = lance.write_dataset(tbl, uri, data_storage_version="2.1")

    torch_ds = LanceDataset(
        uri, batch_size=4, dataset_options={"version": dataset.version}
    )
    with pytest.raises(NotImplementedError):
        next(iter(torch_ds))

    def to_tensor_fn(batch, *args, **kwargs):
        ints = torch.tensor(batch["int"].to_numpy())
        vals = []
        for blob in batch["val"]:
            blob.seek(100)
            data = blob.read(100)
            tensor = torch.tensor(np.frombuffer(data, dtype=np.uint8))
            vals.append(tensor)

            # vals.append(torch.tensor(blob))
        vals = torch.stack(vals)
        return {"int": ints, "val": vals}

    torch_ds = LanceDataset(
        dataset,
        batch_size=4,
        to_tensor_fn=to_tensor_fn,
    )
    first = next(iter(torch_ds))
    assert first["int"].dtype == torch.int64
    assert first["int"].shape == (4,)
    assert first["val"].dtype == torch.uint8
    assert first["val"].shape == (4, 100)


def test_iter_over_dataset_bfloat16(tmp_path):
    """Test that bfloat16 vector columns convert to torch.bfloat16 tensors."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    from lance.arrow import BFloat16Array

    dim = 32
    num_rows = 128
    # Create random bfloat16 vectors via float32 → bfloat16 cast
    f32_data = np.random.random(num_rows * dim).astype("f")
    bf16_data = f32_data.astype(ml_dtypes.bfloat16)

    # Build a FixedSizeList<bf16> column
    inner = BFloat16Array.from_numpy(bf16_data)
    fsl = pa.FixedSizeListArray.from_arrays(inner, dim)
    ids = pa.array(range(num_rows), type=pa.int32())
    tbl = pa.Table.from_arrays([ids, fsl], ["ids", "vec"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance", max_rows_per_group=32)

    torch_ds = LanceDataset(ds, batch_size=16, columns=["ids", "vec"])

    total_rows = 0
    for batch in torch_ds:
        assert set(batch.keys()) == {"ids", "vec"}
        assert batch["vec"].dtype == torch.bfloat16
        assert batch["vec"].shape[1] == dim
        assert batch["ids"].dtype == torch.int32
        total_rows += batch["vec"].shape[0]
    assert total_rows == num_rows


def test_scalar_bfloat16_column(tmp_path):
    """Test that a scalar bfloat16 column converts to torch.bfloat16 tensor."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    from lance.arrow import BFloat16Array

    num_rows = 64
    f32_data = np.random.random(num_rows).astype("f")
    bf16_data = f32_data.astype(ml_dtypes.bfloat16)

    arr = BFloat16Array.from_numpy(bf16_data)
    tbl = pa.Table.from_arrays([arr], ["val"])

    ds = lance.write_dataset(tbl, tmp_path / "data.lance")

    torch_ds = LanceDataset(ds, batch_size=16, columns=["val"])

    total_rows = 0
    for batch in torch_ds:
        assert batch.dtype == torch.bfloat16
        total_rows += batch.shape[0]
    assert total_rows == num_rows


def test_bf16_to_tensor_zero_copy_without_nulls():
    """Non-null bf16 arrays should alias the Arrow data buffer."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    from lance.arrow import BFloat16Array

    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=ml_dtypes.bfloat16)
    arr = BFloat16Array.from_numpy(values).slice(1, 2)

    tensor = _bf16_to_tensor(arr)

    assert tensor.dtype == torch.bfloat16
    assert torch.equal(
        tensor.to(torch.float32),
        torch.tensor([2.0, 3.0], dtype=torch.float32),
    )
    assert (
        tensor.data_ptr() == arr.storage.buffers()[1].address + arr.storage.offset * 2
    )


def test_bf16_to_tensor_clones_when_nulls_present():
    """Null replacement requires a writable tensor, so the Arrow buffer is cloned."""
    from lance.arrow import bfloat16_array

    arr = bfloat16_array([1.0, None, 3.0])

    tensor = _bf16_to_tensor(arr)

    assert tensor.dtype == torch.bfloat16
    assert (
        tensor.data_ptr() != arr.storage.buffers()[1].address + arr.storage.offset * 2
    )
    assert tensor[0].to(torch.float32).item() == pytest.approx(1.0)
    assert torch.isnan(tensor[1])
    assert tensor[2].to(torch.float32).item() == pytest.approx(3.0)


def test_safe_lance_dataset_worker_uses_dataset_options(tmp_path: Path):
    """Worker processes must reopen the dataset with dataset_options.

    Regression test for: worker init called lance.dataset(uri) without
    dataset_options, silently dropping version, storage_options, etc.
    """
    tbl_v1 = pa.table({"id": pa.array([1, 2, 3], pa.int64())})
    ds = lance.write_dataset(tbl_v1, tmp_path / "data.lance")
    version_1 = ds.version

    # Write a second version with different data so we can distinguish them.
    tbl_v2 = pa.table({"id": pa.array([10, 20, 30], pa.int64())})
    lance.write_dataset(tbl_v2, tmp_path / "data.lance", mode="overwrite")

    # Pin to version 1 via dataset_options.
    safe_ds = SafeLanceDataset(
        str(tmp_path / "data.lance"),
        dataset_options={"version": version_1},
    )

    # Simulate worker-process state: _ds is None so __getitems__ must reopen.
    safe_ds._ds = None
    rows = safe_ds.__getitems__([0, 1, 2])

    assert [r["id"] for r in rows] == [1, 2, 3], (
        "Worker reopened dataset without dataset_options (got version 2 data)"
    )
