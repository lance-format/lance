# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# An ngram FTS index built with with_position=True, which released versions
# accept and current versions refuse to create. Existing indexes like this one
# must keep loading, optimizing and answering queries.

import shutil
from pathlib import Path

import lance
import pyarrow as pa

assert lance.__version__ == "13.0.0"

dataset_path = Path(__file__).parent / "ngram_with_position.lance"
shutil.rmtree(dataset_path, ignore_errors=True)

data = pa.table({"text": ["hello world", "world hello", "help yourself", "low"]})
dataset = lance.write_dataset(data, dataset_path)
dataset.create_scalar_index(
    "text",
    "INVERTED",
    base_tokenizer="ngram",
    with_position=True,
)
assert dataset.list_indices()[0]["type"] == "Inverted"
