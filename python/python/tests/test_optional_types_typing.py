# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Type-checking regressions for ``asof``; part of the pyright target.

Rejected cases use ``pyright: ignore``, so an overly permissive annotation fails
too. Like ``test_fragment_typing.py``, this module does not import ``pytest``.
"""

# pyright: reportUnnecessaryTypeIgnoreComment=true

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    import lance
    import pandas as pd
    from lance.util import sanitize_ts

    def _check_asof_types() -> None:
        _timestamp: datetime = sanitize_ts(pd.Timestamp("2026-01-01"))
        lance.dataset("memory://", asof=pd.Timestamp("2026-01-01"))
        lance.dataset("memory://", asof=pd.NaT)
        lance.dataset("memory://", asof=datetime(2026, 1, 1))
        lance.dataset("memory://", asof="2026-01-01")
        lance.dataset("memory://", asof=object())  # pyright: ignore[reportArgumentType]
