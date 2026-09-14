#  Copyright (c) 2023. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import List, Optional

from lance import LanceDataset
from lance.fragment import FragmentMetadata
from lance.optimize import CompactionOptions

class CompactionMetrics:
    fragments_removed: int
    fragments_added: int
    files_removed: int
    files_added: int

class RewriteResult:
    read_version: int
    metrics: CompactionMetrics
    old_fragments: List["FragmentMetadata"]
    new_fragments: List["FragmentMetadata"]

class CompactionTask:
    read_version: int
    fragments: List["FragmentMetadata"]

    def execute(self, dataset: "LanceDataset") -> RewriteResult: ...

class CompactionPlan:
    read_version: int
    tasks: List[CompactionTask]

    def num_tasks(self) -> int: ...

class Compaction:
    @staticmethod
    def execute(
        dataset: "LanceDataset", options: CompactionOptions
    ) -> CompactionMetrics: ...
    @staticmethod
    def plan(dataset: "LanceDataset", options: CompactionOptions) -> CompactionPlan: ...
    @staticmethod
    def commit(
        dataset: "LanceDataset",
        rewrites: List[RewriteResult],
        options: Optional[CompactionOptions] = None,
    ) -> CompactionMetrics: ...
