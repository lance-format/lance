# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Lay out a restored OSS-2221 corpus archive in this study's data format.

The archive keeps the frozen IVF_FLAT index that calibrated the existing L2 and
cosine profiles. Queries come from the archived oracle; their published top-4096
neighbors are stable row ids and are converted to source row positions.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from common import positions, row_ids_by_position, save

# Dataset version and index segment measured by the OSS-2221 campaign.
ARCHIVES = {
    "dino-10m": {
        "version": 2,
        "index_uuid": "a2449e25-2c49-41f9-b3e2-3a3969491f08",
        "queries": "unified-auto-metrics-20260908/dino10m_l2/oracle/queries.npz",
        "truth": "unified-auto-metrics-20260908/dino10m_l2/oracle/ground_truth.npz",
    },
    "laion-10m": {
        "version": 2,
        "index_uuid": "28448469-7d0e-49a7-931b-84f1e0467ad8",
        "queries": "unified-auto-metrics-20260908/laion10m_cosine/oracle/queries.npz",
        "truth": (
            "unified-auto-metrics-20260908/laion10m_cosine/oracle/ground_truth.npz"
        ),
    },
    "fineweb-10m": {
        "version": 3,
        "index_uuid": "a446f769-fa1d-4cdc-aba7-b8c978a433cb",
        "queries": (
            "global-auto-parameters-20260909/corpora/fineweb/prepared/"
            "queries-candidates.npz"
        ),
        "truth": None,
    },
}


def verify(directory):
    """Check every restored file against the archived inventory digests."""
    inventory = json.loads((directory / "inventory.json").read_text())
    files = inventory["files"]

    def digest(item):
        relative, expected = item
        path = directory / "archive.lance" / relative
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(64 * 1024 * 1024):
                hasher.update(chunk)
        assert path.stat().st_size == expected["bytes"], relative
        assert hasher.hexdigest() == expected["sha256"], relative
        return relative

    with ThreadPoolExecutor(16) as pool:
        verified = list(pool.map(digest, files.items()))
    return len(verified)


def import_archive(root, inputs, name):
    spec = ARCHIVES[name]
    directory = root / "data" / name
    base = directory / "base.lance"
    if not base.exists():
        verified_files = verify(directory)
        (directory / "archive.lance").rename(base)
        (directory / "VERIFIED").write_text(
            json.dumps({"inventory_files_sha256_verified": verified_files}) + "\n"
        )
    dataset = lance.dataset(base, version=spec["version"])
    indices = [index for index in dataset.list_indices() if index["type"] == "IVF_FLAT"]
    assert [index["uuid"] for index in indices] == [spec["index_uuid"]], indices
    loaded = np.load(inputs / spec["queries"])
    vectors = loaded["vectors"]
    assert vectors.dtype == np.float32 and np.isfinite(vectors).all()
    queries = pa.table(
        {
            "query_id": pa.array(np.arange(len(vectors), dtype=np.int64)),
            "source_query_id": pa.array(loaded["source_query_ids"]),
            "original_split": pa.array(loaded["split"]),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1)), vectors.shape[1]
            ),
        }
    )
    lance.write_dataset(queries, directory / "queries.lance", mode="overwrite")
    if spec["truth"] is not None:
        truth = np.load(inputs / spec["truth"])
        assert np.array_equal(truth["source_query_ids"], loaded["source_query_ids"])
        lookup = row_ids_by_position(dataset)
        neighbor_positions = positions(truth["ids"].reshape(-1), lookup).reshape(
            truth["ids"].shape
        )
        lance.write_dataset(
            pa.table(
                {
                    "query_id": pa.array(np.arange(len(vectors), dtype=np.int64)),
                    "neighbor_ids": pa.FixedSizeListArray.from_arrays(
                        pa.array(neighbor_positions.reshape(-1)),
                        neighbor_positions.shape[1],
                    ),
                }
            ),
            directory / "ground_truth.lance",
            mode="overwrite",
        )
    save(
        directory / "import.json",
        {
            "version": spec["version"],
            "index": indices[0],
            "queries": len(vectors),
            "query_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
            "original_split": {
                str(label): int(count)
                for label, count in zip(*np.unique(loaded["split"], return_counts=True))
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("inputs", type=Path, help="campaign reproduction-inputs")
    parser.add_argument("dataset", choices=sorted(ARCHIVES))
    args = parser.parse_args()
    import_archive(args.root, args.inputs, args.dataset)
