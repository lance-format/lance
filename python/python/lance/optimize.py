# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import Literal, Optional, TypedDict

# Re-exported from native module. See src/dataset/optimize.rs for implementation.
from .lance import Compaction as Compaction
from .lance import CompactionMetrics as CompactionMetrics
from .lance import CompactionPlan as CompactionPlan
from .lance import CompactionTask as CompactionTask
from .lance import RewriteResult as RewriteResult

# from .lance import CompactionPlan as CompactionPlan


class CompactionOptions(TypedDict):
    """Options for compaction."""

    target_rows_per_fragment: Optional[int]
    """
    The target number of rows per fragment. This is the number of rows
    that will be in each fragment after compaction. (default: 1024*1024)
    """
    max_rows_per_group: Optional[int]
    """
    Max number of rows per group. This does not affect which fragments
    need compaction, but does affect how they are re-written if selected.
    (default: 1024)
    """
    max_bytes_per_file: Optional[int]
    """
    Max number of bytes in a single file.  This does not affect which
    fragments need compaction, but does affect how they are re-written if
    selected.  If this value is too small you may end up with fragments
    that are smaller than `target_rows_per_fragment`.

    The default will use the default from ``write_dataset``.
    """
    materialize_deletions: Optional[bool]
    """
    Whether to compact fragments with soft deleted rows so they are no
    longer present in the file. (default: True)
    """
    materialize_deletions_threadhold: Optional[float]
    """
    The fraction of original rows that are soft deleted in a fragment
    before the fragment is a candidate for compaction.
    (default: 0.1 = 10%)
    """
    num_threads: Optional[int]
    """
    The number of threads to use when performing compaction. If not
    specified, defaults to the number of cores on the machine.
    """
    batch_size: Optional[int]
    """
    The batch size to use when scanning input fragments.  You may want
    to reduce this if you are running out of memory during compaction.

    The default will use the same default from ``scanner``.
    """
    compaction_mode: Optional[
        Literal["reencode", "try_binary_copy", "force_binary_copy"]
    ]
    """
    The compaction mode to use. Valid values:

    - ``"reencode"``: Decode and re-encode data (default).
    - ``"try_binary_copy"``: Try binary copy if fragments are compatible,
      fall back to reencode otherwise.
    - ``"force_binary_copy"``: Use binary copy or fail if fragments are
      not compatible.
    """
    binary_copy_read_batch_bytes: Optional[int]
    """
    The batch size in bytes for reading during binary copy operations.
    Controls how much data is read at once when performing binary copy.
    (default: 16MB)
    """
    defer_index_remap: Optional[bool]
    """
    Whether to defer index remapping during compaction (default: False).
    """
    max_source_fragments: Optional[int]
    """
    Maximum number of source fragments to compact in a single run. The
    planner stops collecting fragment metrics once this many fragments
    have been selected as compaction candidates, providing early
    termination to limit planning time and memory for datasets with
    many fragments.
    (default: None, no limit)
    """
    max_compaction_bytes: Optional[int]
    """
    Maximum total bytes of source data to compact in a single run. The
    planner stops collecting fragment metrics once the cumulative size
    of candidate fragments exceeds this limit. Fragments whose file size
    is unknown are treated as 0 bytes for budget purposes.
    (default: None, no limit)
    """
