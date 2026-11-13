# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate a stable-row-ID dataset whose RTree index has two row-ID-domain segments.

Lance 14.0.2 is the last stable release before RTree moved to the row-address
domain (#9858), so its writer labels every RTree segment format version 0 and
stores logical row IDs in the leaf pages. The two fragments are indexed as two
separate segments through the distributed build API, which is how an index
accumulates multiple v0 segments before the migration; no deletion file is
written, so a bounded optimize that only selects the trailing segment leaves the
older one in place.

Run with: uv run --no-project --with pylance==14.0.2 datagen.py
"""

import shutil
import struct
from pathlib import Path

import lance
import pyarrow as pa

EXPECTED_LANCE_VERSION = "14.0.2"
assert lance.__version__ == EXPECTED_LANCE_VERSION, (
    f"Expected lance {EXPECTED_LANCE_VERSION}, got {lance.__version__}"
)

path = Path(__file__).parent / "rtree_row_id_domain.lance"
shutil.rmtree(path, ignore_errors=True)


def wkb_point(x: float, y: float) -> bytes:
    # Little-endian WKB point: byte order, geometry type, then the coordinates.
    # geoarrow infers the WKB type from a plain binary column.
    return struct.pack("<BIdd", 1, 1, x, y)


rows = 16
geometries = [
    None if i % 5 == 3 else wkb_point(float(i), float(i)) for i in range(rows)
]
table = pa.table(
    {
        "id": pa.array(range(rows), pa.int32()),
        "geometry": pa.array(geometries, pa.binary()),
    }
)
dataset = lance.write_dataset(
    table,
    path,
    enable_stable_row_ids=True,
    max_rows_per_file=8,
    max_rows_per_group=8,
)
assert len(dataset.get_fragments()) == 2

segments = [
    dataset.create_index_uncommitted(
        "geometry", "RTREE", name="geometry_idx", fragment_ids=[fragment.fragment_id]
    )
    for fragment in dataset.get_fragments()
]
assert len(segments) == 2
assert all(segment.index_version == 0 for segment in segments), segments
dataset.commit_existing_index_segments("geometry_idx", "geometry", segments)

dataset = lance.dataset(path)
assert dataset.count_rows() == rows
assert dataset.count_rows("geometry IS NULL") == 3
assert (
    dataset.count_rows("ST_Intersects(geometry, ST_GeomFromText('POINT (7 7)'))") == 1
)
