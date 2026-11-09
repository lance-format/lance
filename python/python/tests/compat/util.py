# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# Utilities shared by datagen.py
#
# Everything here must be runnable by older versions of Lance.


import pyarrow as pa


def safe_data_storage_version(version_str):
    """Return a data_storage_version safe for the given lance version.

    Versions before 0.38 use "2.0"; newer released versions use "2.1".

    Do not return a moving alias here: compatibility tests create data with the
    current Lance build and then read it with the requested older release.
    """
    parts = version_str.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    if major == 0 and minor < 38:
        return "2.0"
    return "2.1"


def build_basic_types():
    schema = pa.schema(
        [
            pa.field("int", pa.int64()),
            pa.field("float", pa.float32()),
            pa.field("str", pa.string()),
            pa.field("list_int", pa.list_(pa.int64())),
            pa.field("list_str", pa.list_(pa.string())),
            pa.field("struct", pa.struct([pa.field("a", pa.int64())])),
            pa.field("dict", pa.dictionary(pa.int16(), pa.string())),
            pa.field("str_as_dict", pa.string()),
        ]
    )

    return pa.table(
        [
            pa.array(range(1000)),
            pa.array(range(1000), pa.float32()),
            pa.array([str(i) for i in range(1000)]),
            pa.array([list(range(i)) for i in range(1000)]),
            pa.array([[str(i)] for i in range(1000)]),
            pa.array([{"a": i} for i in range(1000)]),
            pa.array(
                [str(i % 10) for i in range(1000)],
                pa.dictionary(pa.int16(), pa.string()),
            ),
            pa.array(["a"] * 500 + ["b"] * 500),
        ],
        schema=schema,
    )


def build_large():
    # ~40MB of vector embedding data (10K 1024-float32)
    fsl_data = pa.array(range(1024 * 1000 * 10), pa.float32())
    fsls = pa.FixedSizeListArray.from_arrays(fsl_data, 1024)
    # ~40 MiB of binary data (10k 4KiB chunks)
    bindata = pa.allocate_buffer(1024 * 1000 * 40)
    offsets = pa.array(
        range(0, (1024 * 1000 * 40) + 4 * 1024, 4 * 1024), pa.int32()
    ).buffers()[1]
    bins = pa.BinaryArray.from_buffers(pa.binary(), 10000, [None, offsets, bindata])

    schema = pa.schema(
        [
            pa.field("int", pa.int32()),
            pa.field("fsl", pa.list_(pa.float32())),
            pa.field("bin", pa.binary()),
        ]
    )

    return pa.table(
        [
            pa.array(range(10000), pa.int32()),
            fsls,
            bins,
        ],
        schema=schema,
    )
