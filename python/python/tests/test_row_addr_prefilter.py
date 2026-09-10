# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""End-to-end tests for the external row-address prefilter.

``row_addr_allowlist`` / ``row_addr_blocklist`` restrict a scan to a set of row
addresses supplied by the caller, rather than to rows a filter expression
selects. The mask is applied before ranking, so a KNN or full-text search
computes top-k over the surviving rows instead of trimming the result
afterwards -- the two differ whenever k is smaller than the candidate set.

Each test asserts against ``_rowid`` ground truth so a mask that is silently
dropped (which would return every row) fails rather than passing by accident.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.dataset import ScannerBuilder, serialize_row_addrs
from lance.file import LanceFileWriter

if TYPE_CHECKING:
    from pathlib import Path

N = 256
DIM = 8


def _write(tmp_path: Path, with_index: bool = False) -> lance.LanceDataset:
    rng = np.random.default_rng(1234)
    vectors = rng.standard_normal((N, DIM)).astype(np.float32)
    tbl = pa.table(
        {
            "id": pa.array(range(N), pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1), pa.float32()), DIM
            ),
            "text": pa.array([f"row {i} lorem ipsum" for i in range(N)]),
        }
    )
    ds = lance.write_dataset(tbl, str(tmp_path / "t.lance"), mode="overwrite")
    if with_index:
        # IVF_FLAT with nprobes == num_partitions is exact, so the masked result
        # can be compared against brute force without recall slack.
        ds.create_index("vector", index_type="IVF_FLAT", num_partitions=4, metric="l2")
    return ds


def _rowids(ds: lance.LanceDataset) -> list[int]:
    return ds.to_table(with_row_id=True)["_rowid"].to_pylist()


def test_serialize_row_addrs_round_trips_through_a_scan(tmp_path: Path) -> None:
    ds = _write(tmp_path)
    addrs = _rowids(ds)
    want = addrs[3:9]

    got = ds.scanner(
        with_row_id=True, row_addr_allowlist=serialize_row_addrs(want)
    ).to_table()
    assert got["_rowid"].to_pylist() == want


def test_allowlist_and_blocklist_combine(tmp_path: Path) -> None:
    ds = _write(tmp_path)
    addrs = _rowids(ds)

    allow, block = addrs[:10], addrs[5:15]
    got = ds.scanner(
        with_row_id=True,
        row_addr_allowlist=serialize_row_addrs(allow),
        row_addr_blocklist=serialize_row_addrs(block),
    ).to_table()
    assert got["_rowid"].to_pylist() == addrs[:5]

    # Block alone excludes and leaves everything else.
    got = ds.scanner(
        with_row_id=True, row_addr_blocklist=serialize_row_addrs(addrs[:5])
    ).to_table()
    assert got["_rowid"].to_pylist() == addrs[5:]


def test_no_mask_reads_everything(tmp_path: Path) -> None:
    # Guards the "no mask" vs "empty mask" distinction: omitting both must not
    # be read as an allowlist of nothing.
    ds = _write(tmp_path)
    assert ds.scanner().to_table().num_rows == N


def test_empty_allowlist_selects_nothing(tmp_path: Path) -> None:
    ds = _write(tmp_path)
    got = ds.scanner(row_addr_allowlist=serialize_row_addrs([])).to_table()
    assert got.num_rows == 0


def test_mask_composes_with_a_filter(tmp_path: Path) -> None:
    ds = _write(tmp_path)
    addrs = _rowids(ds)
    got = ds.scanner(
        columns=["id"],
        filter="id % 2 == 0",
        row_addr_allowlist=serialize_row_addrs(addrs[:20]),
    ).to_table()
    assert got["id"].to_pylist() == [i for i in range(20) if i % 2 == 0]


def test_builder_setter_matches_the_kwarg(tmp_path: Path) -> None:
    ds = _write(tmp_path)
    blob = serialize_row_addrs(_rowids(ds)[2:7])
    from_kwarg = ds.scanner(with_row_id=True, row_addr_allowlist=blob).to_table()
    from_builder = (
        ScannerBuilder(ds)
        .with_row_id(True)
        .row_addr_prefilter(allowlist=blob)
        .to_scanner()
        .to_table()
    )
    assert from_kwarg["_rowid"].to_pylist() == from_builder["_rowid"].to_pylist()


@pytest.mark.parametrize("with_index", [False, True])
def test_knn_topk_is_computed_over_masked_rows(
    tmp_path: Path, with_index: bool
) -> None:
    # The point of a prefilter: with k=5 and a 10-row mask, post-filtering a
    # global top-5 would usually return fewer than 5 (often 0) rows.
    ds = _write(tmp_path, with_index=with_index)
    addrs = _rowids(ds)
    allowed = addrs[100:110]
    query = np.zeros(DIM, dtype=np.float32)

    got = ds.scanner(
        nearest={"column": "vector", "q": query, "k": 5, "nprobes": 4},
        with_row_id=True,
        row_addr_allowlist=serialize_row_addrs(allowed),
    ).to_table()

    assert got.num_rows == 5
    assert set(got["_rowid"].to_pylist()) <= set(allowed)

    # Exactly the 5 nearest *within* the mask, not the global 5 intersected.
    vectors = np.stack(
        [np.asarray(v) for v in ds.to_table(columns=["vector"])["vector"].to_pylist()]
    )
    by_addr = dict(zip(addrs, vectors))
    expect = sorted(allowed, key=lambda a: np.linalg.norm(by_addr[a] - query))[:5]
    assert sorted(got["_rowid"].to_pylist()) == sorted(expect)


def test_knn_blocklist_excludes_the_nearest(tmp_path: Path) -> None:
    ds = _write(tmp_path)
    query = np.zeros(DIM, dtype=np.float32)
    unmasked = (
        ds.scanner(nearest={"column": "vector", "q": query, "k": 3}, with_row_id=True)
        .to_table()["_rowid"]
        .to_pylist()
    )

    got = ds.scanner(
        nearest={"column": "vector", "q": query, "k": 3},
        with_row_id=True,
        row_addr_blocklist=serialize_row_addrs(unmasked[:1]),
    ).to_table()

    assert got.num_rows == 3  # refilled, not truncated
    assert unmasked[0] not in got["_rowid"].to_pylist()


def test_full_text_search_honors_the_mask(tmp_path: Path) -> None:
    ds = _write(tmp_path)
    ds.create_scalar_index("text", index_type="INVERTED")
    addrs = _rowids(ds)
    allowed = addrs[50:60]

    got = ds.scanner(
        full_text_query="lorem",
        with_row_id=True,
        row_addr_allowlist=serialize_row_addrs(allowed),
        limit=5,
    ).to_table()

    assert got.num_rows == 5
    assert set(got["_rowid"].to_pylist()) <= set(allowed)


def test_rejects_a_malformed_mask(tmp_path: Path) -> None:
    ds = _write(tmp_path)
    with pytest.raises(Exception, match="(?i)row address mask|invalid"):
        ds.scanner(row_addr_allowlist=b"not a treemap").to_table()


def _overlay(
    ds, base_dir: Path, name: str, batch: pa.Table, fields: list[int], offsets
):
    """Commit a data overlay covering `offsets` of fragment 0.

    An overlay committed after an index makes the indexed values stale, so the
    planner replays those rows through a separate take. That replay is a second
    row source, and it has to honor the caller's mask like every other one.
    """
    path = base_dir / "data" / name
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(batch)
    base_df = ds.get_fragments()[0].metadata.files[0]
    data_file = lance.fragment.DataFile(
        path=name,
        fields=fields,
        column_indices=list(range(len(fields))),
        file_major_version=base_df.file_major_version,
        file_minor_version=base_df.file_minor_version,
        file_size_bytes=os.path.getsize(path),
    )
    op = lance.LanceOperation.DataOverlay(
        [
            lance.LanceOperation.DataOverlayGroup(
                0, [lance.LanceOperation.DataOverlayFile(data_file, offsets=offsets)]
            )
        ]
    )
    return lance.LanceDataset.commit(ds, op, read_version=ds.version)


def test_overlay_stale_replay_scan_respects_mask(tmp_path: Path) -> None:
    base_dir = tmp_path / "ov_scan"
    ds = lance.write_dataset(
        pa.table(
            {
                "id": pa.array(range(10), pa.int32()),
                "val": pa.array([i * 10 for i in range(10)], pa.int32()),
            }
        ),
        base_dir,
    )
    # Index first, then overlay: offset 1 now reads 999 while the index still
    # says 10, so `val = 999` can only be answered by the stale replay.
    ds.create_scalar_index("val", index_type="BTREE")
    ds = _overlay(
        ds,
        base_dir,
        "ov.lance",
        pa.table({"val": pa.array([999], pa.int32())}),
        fields=[1],
        offsets=[1],
    )

    base = ds.scanner(filter="val = 999", with_row_id=True).to_table()
    assert base.num_rows == 1, "fixture did not produce a stale replay"
    stale_addr = base["_rowid"].to_pylist()[0]

    got = ds.scanner(
        filter="val = 999", row_addr_allowlist=serialize_row_addrs([])
    ).to_table()
    assert got.num_rows == 0, "stale replay must not return rows the mask excludes"

    got = ds.scanner(
        filter="val = 999",
        with_row_id=True,
        row_addr_allowlist=serialize_row_addrs([stale_addr]),
    ).to_table()
    assert got["_rowid"].to_pylist() == [stale_addr]


def test_overlay_stale_replay_ann_respects_mask(tmp_path: Path) -> None:
    base_dir = tmp_path / "ov_ann"
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((N, DIM)).astype(np.float32)
    ds = lance.write_dataset(
        pa.table(
            {
                "id": pa.array(range(N), pa.int64()),
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(vectors.reshape(-1), pa.float32()), DIM
                ),
            }
        ),
        base_dir,
    )
    ds.create_index("vector", index_type="IVF_FLAT", num_partitions=4, metric="l2")

    # Move two rows onto the query point after indexing. The ANN index still has
    # their old vectors, so they can only surface through the stale replay.
    query = np.zeros(DIM, dtype=np.float32)
    moved = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(2 * DIM, dtype=np.float32), pa.float32()), DIM
    )
    ds = _overlay(
        ds,
        base_dir,
        "ov_vec.lance",
        pa.table({"vector": moved}),
        fields=[1],
        offsets=[3, 7],
    )

    base = ds.scanner(
        nearest={"column": "vector", "q": query, "k": 5}, with_row_id=True
    ).to_table()
    assert base.num_rows > 0, "fixture did not produce ANN results"

    got = ds.scanner(
        nearest={"column": "vector", "q": query, "k": 5},
        row_addr_allowlist=serialize_row_addrs([]),
    ).to_table()
    assert got.num_rows == 0, (
        "the ANN stale replay must not return rows the mask excludes"
    )
