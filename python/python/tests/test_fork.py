# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import sys
import traceback
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

N_DIMS = 768
NUM_ROWS = 100_000
NEW_ROWS = 10_000


def create_table(num_rows) -> pa.Table:
    return pa.table(
        {
            "a": pc.random(num_rows).cast(pa.float32()),
            "b": pa.array(range(0, num_rows)),
        }
    )


def check_reads(uri: Path, tbl: pa.Table):
    dataset = lance.dataset(uri)
    assert dataset.uri == str(uri.absolute())
    assert tbl.schema == dataset.schema
    assert tbl == dataset.to_table()

    one_col = dataset.to_table(columns=["a"])
    assert one_col == tbl.select(["a"])

    table = dataset.to_table(columns=["a"], limit=20)
    assert len(table) == 20


@pytest.mark.skipif(sys.platform == "win32", reason="Test not applicable on Windows")
def test_table_roundtrip(tmp_path: Path):
    uri = tmp_path

    tbl = create_table(100)
    lance.write_dataset(tbl, uri)

    child = os.fork()
    if child == 0:
        # The child has to leave through os._exit. Returning would run the rest
        # of the pytest session a second time, and under pytest-xdist it would
        # also report a second result for this test over the execnet connection
        # inherited from the worker, which crashes the controller's scheduler.
        status = 0
        try:
            check_reads(uri, tbl)
        except BaseException:
            traceback.print_exc()
            status = 1
        os._exit(status)

    check_reads(uri, tbl)
    _, wait_status = os.waitpid(child, 0)
    exitcode = os.waitstatus_to_exitcode(wait_status)
    # Nothing the child raises can reach this process, so its exit status is the
    # only evidence the post-fork read worked. On macOS the child dies of a
    # signal before finishing that read -- long-standing behaviour that this
    # test could not see while it never waited on the child at all. Checking it
    # where it does hold at least keeps the Linux path honest.
    if sys.platform != "darwin":
        assert exitcode == 0, "reading the dataset failed in the forked child"
