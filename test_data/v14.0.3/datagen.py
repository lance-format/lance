# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil
from datetime import timedelta
from pathlib import Path

import lance

# To generate the test file, we should be running this version of lance.
# Releases from 0.20.0 through 14.0.3 read a packed legacy 4-bit IVF_PQ page
# at one byte per sub-vector, so a compaction rewrites the index from
# misread codes and row ids.
assert lance.__version__ == "14.0.3"

name = "legacy_ivf_pq_4bit_misremapped"
source = Path(__file__).parent.parent / "v0.24.1" / "legacy_ivf_pq_4bit"

shutil.rmtree(name, ignore_errors=True)
shutil.copytree(source, name)

dataset = lance.dataset(name)
dataset.delete("id % 7 = 3")
dataset.optimize.compact_files()
dataset = lance.dataset(name)
dataset.cleanup_old_versions(older_than=timedelta(0), delete_unverified=True)

assert len(dataset.get_fragments()) == 1
assert len(dataset.list_indices()) == 1
