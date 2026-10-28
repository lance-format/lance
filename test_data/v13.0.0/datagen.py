# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate the released row-ID-domain BTree compatibility fixture."""

from pathlib import Path

import lance
import pyarrow as pa

assert lance.__version__ == "13.0.0", lance.__version__

uri = Path(__file__).parent / "stable_row_id_btree.lance"
assert not uri.exists(), (
    f"Remove the existing generated fixture before rerunning: {uri}"
)
table = pa.table(
    {
        "key": pa.array(range(12), type=pa.int64()),
        "payload": [f"value-{key}" for key in range(12)],
    }
)
dataset = lance.write_dataset(
    table,
    uri,
    enable_stable_row_ids=True,
    max_rows_per_file=4,
)
dataset.create_scalar_index("key", "BTREE", name="legacy_key_idx")
dataset.delete("key IN (1, 9)")
dataset.update({"payload": "'updated'"}, where="key = 5")

rows = dataset.scanner(with_row_id=True).to_table().to_pylist()
assert len(rows) == 10
assert {row["key"] for row in rows} == set(range(12)) - {1, 9}
assert all(row["_rowid"] == row["key"] for row in rows)
assert next(row for row in rows if row["key"] == 5)["payload"] == "updated"
assert len(dataset.get_fragments()) > 1
