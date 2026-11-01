# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors


import lance
import pytest
from ci_benchmarks.datasets import get_dataset_uri, is_on_google
from ci_benchmarks.utils import wipe_os_cache

COLUMN_LABELS = ["bools", "normals"]
COLUMNS = [["bools"], ["normals"]]
FILTERS = [None, "bools IS TRUE"]


@pytest.mark.parametrize("columns", COLUMNS, ids=COLUMN_LABELS)
@pytest.mark.parametrize("filt", FILTERS)
@pytest.mark.skipif(not is_on_google(), reason="Not on Google Cloud")
def test_eda_search(benchmark, columns, filt):
    dataset_uri = get_dataset_uri("image_eda")

    batch_size = 32 if columns == ["image_data"] else None
    limit = None if filter is None else 100000
    frag_readahead = (
        4
        if (columns == ["image_data"] or columns == ["strings"]) and filter is None
        else None
    )

    def bench():
        ds = lance.dataset(dataset_uri)
        ds.to_table(
            columns=columns,
            filter=filt,
            batch_size=batch_size,
            fragment_readahead=frag_readahead,
            limit=limit,
        )

    benchmark.pedantic(bench, rounds=1, iterations=1)


LARGE_IN_FILTER = (
    "image_widths IN (" + ", ".join([str(i) for i in range(3990, 4100)]) + ")"
)

BTREE_FILTERS = [
    None,
    "image_widths = 3997",
    "image_widths >= 3990 AND image_widths <= 3997",
    "image_widths != 3997",
    LARGE_IN_FILTER,
]
BTREE_FILTER_LABELS = [
    None,
    "equal",
    "small_range",
    "not_equal",
    "large_in",
]


# These tests benchmark a variety of filtered read patterns
@pytest.mark.parametrize("filt", BTREE_FILTERS, ids=BTREE_FILTER_LABELS)
@pytest.mark.parametrize("payload", [None, "image_widths"], ids=["none", "integers"])
@pytest.mark.parametrize("use_cache", [True, False], ids=["cache", "no_cache"])
@pytest.mark.skipif(not is_on_google(), reason="Not on Google Cloud")
def test_eda_btree_search(
    benchmark, filt: str | None, payload: str | None, use_cache: bool
):
    dataset_uri = get_dataset_uri("image_eda")
    ds = lance.dataset(dataset_uri)

    batch_size = 1024 if payload == "strings" else 32 * 1024

    columns = []
    if payload is not None:
        columns = [payload]

    def bench():
        to_search = ds if use_cache else lance.dataset(dataset_uri)
        to_search.to_table(
            columns=columns,
            filter=filt,
            with_row_id=True,
            batch_size=batch_size,
        )

    iterations = 1
    if payload is None and filt != "image_widths != 3997":
        # These are fast queries against the index with no data load so we can
        # run them a lot more times
        iterations = 100

    # We warmup so we can test hot index performance
    warmup_rounds = 1 if use_cache else 0

    benchmark.pedantic(
        bench, warmup_rounds=warmup_rounds, rounds=1, iterations=iterations
    )


BASIC_LARGE_IN_FILTER = (
    "row_number IN (" + ", ".join([str(i) for i in range(100000, 100100)]) + ")"
)
# row_number runs 0..10M, so this selects 7M rows (70% of the dataset).  Both
# bounds land mid-page, so the scan has to walk the great majority of the
# btree's leaves and union their row id lists
BASIC_LARGE_RANGE_FILTER = "row_number >= 1000000 AND row_number < 8000000"

BASIC_BTREE_FILTERS = [
    None,
    "row_number = 100000",
    "row_number != 100000",
    "row_number >= 100000 AND row_number <= 100007",
    BASIC_LARGE_RANGE_FILTER,
    BASIC_LARGE_IN_FILTER,
]

BASIC_BTREE_FILTER_LABELS = [
    "none",
    "equal",
    "not_equal",
    "small_range",
    "large_range",
    "large_in",
]


def do_basic_search(benchmark, filt: str | None, payload: str | None, use_cache: bool):
    dataset_uri = get_dataset_uri("basic")
    ds = lance.dataset(dataset_uri)

    columns = []
    if payload is not None:
        columns = [payload]

    def clear_cache():
        wipe_os_cache(dataset_uri)

    def bench():
        to_search = ds if use_cache else lance.dataset(dataset_uri)
        to_search.to_table(
            columns=columns,
            filter=filt,
            with_row_id=True,
            batch_size=32 * 1024,
        )

    setup = None if use_cache else clear_cache

    warmup_rounds = 1 if use_cache else 0
    benchmark.pedantic(
        bench, warmup_rounds=warmup_rounds, rounds=10, iterations=1, setup=setup
    )


# Repeats the same test for the basic dataset which is easier to test with locally
# This benchmark is not part of the CI job as the EDA dataset is better for that
@pytest.mark.parametrize("filt", BASIC_BTREE_FILTERS, ids=BASIC_BTREE_FILTER_LABELS)
@pytest.mark.parametrize("payload", [None, "small_strings", "integers"])
@pytest.mark.parametrize("use_cache", [True, False], ids=["cache", "no_cache"])
def test_basic_btree_search(
    benchmark, filt: str | None, payload: str | None, use_cache: bool
):
    do_basic_search(benchmark, filt, payload, use_cache)


BASIC_LARGE_IN_FILTER_BITMAP = (
    "row_number_bitmap IN (" + ", ".join([str(i) for i in range(100000, 100100)]) + ")"
)
BASIC_BITMAP_FILTERS = [
    None,
    "row_number_bitmap = 100000",
    "row_number_bitmap != 100000",
    #     "row_number_bitmap >= 100000 AND row_number_bitmap <= 100007",
    #     BASIC_LARGE_IN_FILTER_BITMAP,
]

BASIC_BITMAP_FILTER_LABELS = [
    "none",
    "equal",
    "not_equal",
    #     "small_range",
    #     "large_in",
]


# Don't run the no_cache test on Google Cloud as it is way too expensive at the moment
def use_cache_param():
    if is_on_google():
        return [True]
    return [True, False]


def use_cache_ids():
    if is_on_google():
        return ["cache"]
    return ["cache", "no_cache"]


# Repeats the same test for the basic dataset which is easier to test with locally
# This benchmark is not part of the CI job as the EDA dataset is better for that
@pytest.mark.parametrize("filt", BASIC_BITMAP_FILTERS, ids=BASIC_BITMAP_FILTER_LABELS)
@pytest.mark.parametrize("payload", [None, "small_strings", "integers"])
@pytest.mark.parametrize("use_cache", use_cache_param(), ids=use_cache_ids())
def test_basic_bitmap_search(
    benchmark, filt: str | None, payload: str | None, use_cache: bool
):
    do_basic_search(benchmark, filt, payload, use_cache)


@pytest.mark.io_memory_benchmark()
@pytest.mark.parametrize("filt", BASIC_BTREE_FILTERS, ids=BASIC_BTREE_FILTER_LABELS)
@pytest.mark.parametrize("payload", ["small_strings", "integers"])
def test_io_mem_basic_btree_search(io_mem_benchmark, filt: str | None, payload: str):
    dataset_uri = get_dataset_uri("basic")
    ds = lance.dataset(dataset_uri)

    columns = []
    if payload is not None:
        columns = [payload]

    def bench(dataset):
        dataset.to_table(
            columns=columns,
            filter=filt,
            with_row_id=True,
            batch_size=32 * 1024,
        )

    io_mem_benchmark(bench, ds)
