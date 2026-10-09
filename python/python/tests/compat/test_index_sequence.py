# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Cross-version index maintenance-sequence search, wired as a compat test.

For each index kind, generate maintenance-op sequences from a small grammar, run the
setup half under an older writer ref and the exercise half under a newer reader, and
check after each that the reader did not panic and that an index query matches a full
scan. Discovers cross-version regressions (e.g. ENT-1662, under INVERTED) with no
hand-coded sequence.

Refs and max length are environment-driven so the suite can run between two refs
(versions, commits, or branches): COMPAT_FROM_REF / COMPAT_TO_REF / COMPAT_MAX_LENGTH /
COMPAT_VECTOR_MAX_LENGTH / COMPAT_KINDS (comma-separated subset of kinds) /
COMPAT_SHARDS (split each scalar/FTS kind's search into this many cases so pytest-xdist
(`-n auto`) parallelizes them across cores) / COMPAT_VECTOR_SHARDS (the bounded IVF_PQ
search uses fewer shards to avoid repeatedly training the same small index).
"""

import os
from itertools import product

import pytest

from .compat_decorator import pylance_stable_versions
from .compat_sequence import (
    ALL_KINDS,
    VECTOR_KIND,
    VECTOR_OPS,
    generate_vector,
    search,
)


def test_vector_sequence_generation_is_bounded_and_covers_high_risk_orders():
    cases = list(generate_vector(max_length=5))
    combined = [tuple(setup + exercise) for setup, exercise in cases]

    assert cases
    assert len(cases) < 128
    assert len(cases) == len(set((tuple(s), tuple(e)) for s, e in cases))
    assert all(exercise for _, exercise in cases)
    assert all(1 <= len(sequence) <= 5 for sequence in combined)

    split_cases = {(tuple(setup), tuple(exercise)) for setup, exercise in cases}
    for pair in product(VECTOR_OPS, repeat=2):
        assert ((), pair) in split_cases
        assert (pair[:1], pair[1:]) in split_cases

    two_delta_then_unindexed = ("W", "Ov", "W", "Ov", "W")
    assert {
        (tuple(setup), tuple(exercise))
        for setup, exercise in cases
        if tuple(setup + exercise) == two_delta_then_unindexed
    } == {
        (two_delta_then_unindexed[:split], two_delta_then_unindexed[split:])
        for split in range(len(two_delta_then_unindexed))
    }


def _default_refs():
    """The two most recent published stable releases (older -> newer)."""
    versions = pylance_stable_versions()
    if len(versions) >= 2:
        return str(versions[-2]), str(versions[-1])
    return "6.0.1", "7.0.0"  # fallback if PyPI is unreachable


_default_from, _default_to = _default_refs()
FROM_REF = os.environ.get("COMPAT_FROM_REF") or _default_from
TO_REF = os.environ.get("COMPAT_TO_REF") or _default_to
MAX_LENGTH = int(os.environ.get("COMPAT_MAX_LENGTH", "4"))
VECTOR_MAX_LENGTH = int(os.environ.get("COMPAT_VECTOR_MAX_LENGTH", str(MAX_LENGTH)))
KINDS = os.environ.get("COMPAT_KINDS", ",".join(ALL_KINDS)).split(",")
# Many small shards (default 4x cores) so xdist's dynamic scheduler keeps every worker
# busy and an oversubscribed `-n` has work to overlap.
NUM_SHARDS = int(os.environ.get("COMPAT_SHARDS", str((os.cpu_count() or 1) * 4)))
# Training IVF_PQ once per generic shard would multiply total work without expanding
# coverage. Four cases still parallelize the bounded vector search while retaining
# snapshot reuse within each case.
VECTOR_NUM_SHARDS = int(os.environ.get("COMPAT_VECTOR_SHARDS", str(min(NUM_SHARDS, 4))))


def _cases():
    """(kind, fts_version) cases. FTS runs under both on-disk formats (v1, v2); the
    scalar kinds are format-agnostic and run once."""
    cases = []
    for kind in KINDS:
        if kind == "INVERTED":
            cases.extend([("INVERTED", "1"), ("INVERTED", "2")])
        else:
            cases.append((kind, None))
    return cases


def _search_cases():
    cases = []
    for kind, fts_version in _cases():
        num_shards = VECTOR_NUM_SHARDS if kind == VECTOR_KIND else NUM_SHARDS
        kind_id = kind if fts_version is None else f"{kind}-fmtv{fts_version}"
        cases.extend(
            pytest.param(
                kind,
                fts_version,
                shard,
                num_shards,
                id=f"{kind_id}-shard{shard}",
            )
            for shard in range(num_shards)
        )
    return cases


SEARCH_CASES = _search_cases()


@pytest.mark.compat
@pytest.mark.parametrize("kind,fts_version,shard,num_shards", SEARCH_CASES)
def test_index_maintenance_sequence_search(
    venv_factory, tmp_path, kind, fts_version, shard, num_shards
):
    max_length = VECTOR_MAX_LENGTH if kind == VECTOR_KIND else MAX_LENGTH
    failures = search(
        venv_factory,
        FROM_REF,
        TO_REF,
        tmp_path,
        kind,
        max_length=max_length,
        shard=shard,
        num_shards=num_shards,
        fts_version=fts_version,
    )
    # First line is the failure itself so it shows in pytest's bottom summary; the rest
    # (if more than one) appears in the failure body.
    assert not failures, "\n".join(
        f"{f['sequence']}  ==>  {f['error']}" for f in failures
    )
