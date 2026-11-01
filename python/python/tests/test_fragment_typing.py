# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Type-checking regression tests for the fragment write APIs.

This module is part of the pyright target configured in ``pyproject.toml``, so
a regression in the annotations below fails the repository type check and not
only the runtime suite.

It is separate from ``test_fragment.py`` because that file currently carries
pre-existing pyright diagnostics unrelated to these APIs, so it cannot join the
type-check target without a much larger cleanup.

The results are bound to annotated locals on purpose: that pins the selected
overload, and pyright rejects the ``None`` argument if ``max_rows_per_group``
regresses to a plain ``int``.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List

import pyarrow as pa
from lance.fragment import FragmentMetadata, LanceFragment, write_fragments

if TYPE_CHECKING:
    from lance import Transaction


def test_write_fragments_accepts_none_max_rows_per_group(tmp_path: Path) -> None:
    table = pa.table({"a": range(8)})

    fragments: List[FragmentMetadata] = write_fragments(
        table, str(tmp_path / "fragments"), max_rows_per_group=None
    )
    assert len(fragments) == 1
    assert fragments[0].physical_rows == 8


def test_write_fragments_transaction_accepts_none_max_rows_per_group(
    tmp_path: Path,
) -> None:
    table = pa.table({"a": range(8)})

    transaction: "Transaction" = write_fragments(
        table,
        str(tmp_path / "transaction"),
        max_rows_per_group=None,
        return_transaction=True,
    )
    assert transaction.operation is not None


def test_fragment_create_accepts_none_max_rows_per_group(tmp_path: Path) -> None:
    table = pa.table({"a": range(8)})

    fragment: FragmentMetadata = LanceFragment.create(
        str(tmp_path / "create"), table, max_rows_per_group=None
    )
    assert fragment.physical_rows == 8
