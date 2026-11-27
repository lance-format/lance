# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from datetime import timedelta

import lance
import pyarrow as pa
import pyarrow.compute as pc

# To generate the test file, we should be running this version of lance
assert lance.__version__ == "0.21.0"

data = pa.table(
    {
        "vector": pa.FixedSizeListArray.from_arrays(
            pc.random(16 * 256).cast(pa.float32()), 16
        )
    }
)
data2 = pa.table(
    {
        "vector": pa.FixedSizeListArray.from_arrays(
            pc.random(16 * 32).cast(pa.float32()), 16
        )
    }
)
for stable_ids, name in [
    (False, "bad_index_fragment_bitmap"),
    (True, "bad_index_fragment_bitmap_stable_row_ids"),
]:
    ds = lance.write_dataset(data, name, enable_move_stable_row_ids=stable_ids)
    ds.create_index("vector", index_type="IVF_PQ", num_partitions=1, num_sub_vectors=1)
    ds.insert(data2)
    ds.optimize.optimize_indices(num_indices_to_merge=0)
    ds.cleanup_old_versions(older_than=timedelta(0))

    indices = ds.list_indices()
    # The delta incorrectly includes fragment 0, which the first segment covers.
    assert [index["fragment_ids"] for index in indices] == [{0}, {0, 1}]
