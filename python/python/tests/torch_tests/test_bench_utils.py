# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")
from lance.torch.bench_utils import ground_truth, sort_tensors  # noqa: E402


def test_sort_tensor():
    ids = torch.tensor([[5, 7, 3, 9, 1], [10, 2, 4, 6, 8]], dtype=torch.float32)
    values = ids.clone()

    sorted_vals, sorted_ids = sort_tensors(values, ids, 3)

    assert torch.allclose(
        sorted_vals, torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.float32)
    )
    assert torch.allclose(
        sorted_ids, torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.float32)
    )


def test_ground_truth(tmp_path: Path):
    N = 1000
    NUM_QUERIES = 50
    DIM = 128
    K = 20

    device = "cpu"  # Github action friendly.
    # Keep the fixture independent of other tests that seed NumPy's global RNG.
    # This seed also keeps every top-20 boundary more than 8e-3 apart.
    rng = np.random.RandomState(4415)
    data = rng.rand(N, DIM).astype(np.float32)
    fsl = pa.FixedSizeListArray.from_arrays(data.reshape(-1), DIM)
    torch_data = torch.from_numpy(data).to(device)

    tbl = pa.Table.from_arrays([fsl], ["vec"])

    ds = lance.write_dataset(tbl, tmp_path)

    idx = rng.choice(N, NUM_QUERIES)
    keys = torch_data[idx, :]

    gt = ground_truth(ds, "vec", keys, k=K, batch_size=128, device=device)
    gt, _ = torch.sort(gt, dim=1)

    # Use direct float64 distances as an oracle, independent of pairwise_l2's
    # float32 matrix-multiplication reduction order.
    data64 = data.astype(np.float64)
    expected = []
    boundary_gaps = []
    for query in data64[idx]:
        distances = np.sum(np.square(data64 - query), axis=1)
        row_ids = np.argsort(distances, kind="stable")
        expected.append(np.sort(row_ids[:K]))
        boundary_gaps.append(distances[row_ids[K]] - distances[row_ids[K - 1]])

    assert min(boundary_gaps) > 8e-3, "fixture is too close to the top-k boundary"
    np.testing.assert_array_equal(np.stack(expected), gt.cpu().numpy())
