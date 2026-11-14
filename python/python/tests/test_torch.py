# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("torch")

from lance.torch.data import SafeLanceDataset, get_safe_loader  # noqa: E402


@pytest.fixture(scope="module")
def temp_lance_dataset(tmp_path_factory):
    """Create temporary Lance dataset for testing"""
    test_dir = tmp_path_factory.mktemp("lance_data")
    dataset_path = test_dir / "test_dataset.lance"

    # Generate test data with batch_size aligned sample count
    num_samples = 96  # 16 samples/batch * 6 batches
    data = pa.table(
        {
            "id": range(num_samples),
            "embedding": [
                np.random.rand(128).astype(np.float32).tobytes()
                for _ in range(num_samples)
            ],
        }
    )

    lance.write_dataset(data, dataset_path)
    yield str(dataset_path)


def test_dataset_initialization(temp_lance_dataset):
    """Verify dataset basic functionality"""
    ds = SafeLanceDataset(temp_lance_dataset)

    # Validate metadata
    assert len(ds) == 96, "Sample count should match configured size"

    # Validate single sample format
    sample = ds[0]
    assert isinstance(sample, dict), "Sample should be dictionary type"
    assert {"id", "embedding"}.issubset(sample.keys()), "Missing required fields"


@pytest.mark.parametrize("num_workers", [0, 2])
def test_multiprocess_loading(temp_lance_dataset, capsys, num_workers):
    """Verify single- and multi-worker data loading"""
    dataset = SafeLanceDataset(temp_lance_dataset)
    loader = get_safe_loader(
        dataset,
        num_workers=num_workers,
        batch_size=16,
        drop_last=False,  # Ensure full batches
    )

    total_samples = 0
    for batch in loader:
        assert batch["id"].shape == (16,), "Batch dimension mismatch"
        total_samples += batch["id"].shape[0]

    # Validate complete dataset loading
    assert total_samples == 96, "Should load all samples"
