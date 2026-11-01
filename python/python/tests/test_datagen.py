# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import math

import lance._datagen as datagen
import pyarrow as pa
import pytest

SCHEMA = pa.schema(
    [
        pa.field("int", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 128)),
    ]
)
BYTES_PER_ROW = 8 + 128 * 4


@pytest.mark.skipif(datagen.is_datagen_supported(), reason="datagen is supported")
def test_import_error():
    with pytest.raises(
        NotImplementedError, match="was not built with the datagen feature"
    ):
        datagen.rand_batches(None)


@pytest.mark.skipif(not datagen.is_datagen_supported(), reason="datagen not supported")
def test_rand_batches_by_bytes():
    reader = datagen.rand_batches(SCHEMA, batch_size_bytes=16 * 1024, num_batches=10)

    batches = list(reader)
    assert len(batches) == 10
    for batch in batches:
        assert batch.num_rows == math.ceil(16 * 1024 / BYTES_PER_ROW)
        assert batch.schema == SCHEMA


@pytest.mark.skipif(not datagen.is_datagen_supported(), reason="datagen not supported")
@pytest.mark.parametrize("rows_per_batch", [1, 100])
def test_rand_batches_by_rows(rows_per_batch):
    reader = datagen.rand_batches(SCHEMA, rows_per_batch=rows_per_batch, num_batches=3)

    batches = list(reader)
    assert len(batches) == 3
    for batch in batches:
        assert batch.num_rows == rows_per_batch
        assert batch.schema == SCHEMA


@pytest.mark.skipif(not datagen.is_datagen_supported(), reason="datagen not supported")
def test_rand_batches_rejects_both_sizes():
    with pytest.raises(ValueError, match="mutually exclusive"):
        datagen.rand_batches(SCHEMA, batch_size_bytes=16 * 1024, rows_per_batch=100)
