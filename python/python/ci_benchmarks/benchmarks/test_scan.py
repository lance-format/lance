# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import pytest
from ci_benchmarks.datasets import get_dataset_uri

DATASETS = ["tpch"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_full_scan(benchmark, dataset):
    dataset_uri = get_dataset_uri(dataset)

    def bench():
        ds = lance.dataset(dataset_uri)
        ds.to_table()

    benchmark.pedantic(bench, rounds=1, iterations=1)


@pytest.mark.parametrize("dataset", DATASETS)
def test_scan_slice(benchmark, dataset):
    dataset_uri = get_dataset_uri(dataset)

    ds = lance.dataset(dataset_uri)
    num_rows = ds.count_rows()

    def bench():
        ds = lance.dataset(dataset_uri)
        ds.to_table(offset=num_rows - 100, limit=50)

    # A single unwarmed round measures whatever page-cache state the preceding
    # benchmarks left behind (test_full_scan alone cycles the dataset plus a
    # full in-memory table through RAM), not the scan itself.  Warm up once and
    # take several rounds so the recorded value tracks the code path
    # deterministically (see issue #8289).
    benchmark.pedantic(bench, rounds=5, iterations=1, warmup_rounds=1)
