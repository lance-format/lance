# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Compaction annotation regressions checked by the repository pyright target."""

from pathlib import Path
from typing import TYPE_CHECKING

import lance
import pyarrow as pa
from lance.optimize import Compaction, CompactionPlan, CompactionTask, RewriteResult

if TYPE_CHECKING:
    from lance.fragment import FragmentMetadata


def test_compaction_json_types(tmp_path: Path) -> None:
    dataset = lance.write_dataset(
        pa.table({"a": range(4)}), tmp_path / "dataset", max_rows_per_file=2
    )
    plan = Compaction.plan(dataset, {"target_rows_per_fragment": 4, "num_threads": 1})

    # Annotated locals pin the JSON helpers' return types in pyright.
    plan_json: str = plan.json()
    restored_plan: CompactionPlan = CompactionPlan.from_json(plan_json)
    assert restored_plan == plan

    task = restored_plan.tasks[0]
    task_json: str = task.json()
    restored_task: CompactionTask = CompactionTask.from_json(task_json)
    assert restored_task == task

    result = restored_task.execute(dataset)
    result_json: str = result.json()
    restored_result: RewriteResult = RewriteResult.from_json(result_json)
    assert restored_result == result

    original_fragments: list[FragmentMetadata] = restored_result.original_fragments
    assert original_fragments == [
        fragment.metadata for fragment in dataset.get_fragments()
    ]
