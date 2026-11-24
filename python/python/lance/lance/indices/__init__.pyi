#  Copyright (c) 2024. Lance Developers
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

from datetime import datetime
from typing import Any, Optional

import pyarrow as pa

from .. import _Fragment
from ..bitmap import Bitmap

class IndexConfig:
    index_type: str
    config: str

class IndexSegment:
    uuid: str
    fragment_ids: Bitmap
    index_version: int

    def __repr__(self) -> str: ...

def train_ivf_model(
    dataset,
    column: str,
    dimension: int,
    num_partitions: int,
    distance_type: str,
    sample_rate: int,
    max_iters: int,
    fragment_ids: Optional[list[int]] = None,
) -> pa.FixedSizeListArray: ...
def train_pq_model(
    dataset,
    column: str,
    dimension: int,
    num_subvectors: int,
    distance_type: str,
    sample_rate: int,
    max_iters: int,
    # Kept as the ``Array`` base type: callers pass ``IvfModel.centroids``,
    # which the public ``IvfModel`` constructor accepts as a plain ``pa.Array``.
    ivf_model: pa.Array,
    fragment_ids: Optional[list[int]] = None,
    num_bits: int = 8,
) -> pa.FixedSizeListArray: ...
def transform_vectors(
    dataset,
    column: str,
    dimension: int,
    num_subvectors: int,
    distance_type: str,
    ivf_centroids: pa.Array,
    pq_codebook: pa.Array,
    dst_uri: str,
    fragments: list[_Fragment],
    partitions_ds_uri: Optional[str] = None,
    num_bits: int = 8,
): ...
def build_rq_model(
    dimension: int,
    num_bits: int = 5,
    dtype: str = "float32",
) -> str: ...

class IndexSegmentDescription:
    uuid: str
    dataset_version_at_last_update: int
    fragment_ids: Bitmap
    index_version: int
    created_at: Optional[datetime]
    size_bytes: Optional[int]
    base_id: Optional[int]
    covering_fields: list[int]

    def __repr__(self) -> str: ...

class IndexDescription:
    name: str
    type_url: str
    index_type: str
    num_rows_indexed: int
    fields: list[int]
    field_names: list[str]
    segments: list[IndexSegmentDescription]
    details: dict
    total_size_bytes: Optional[int]

    def __repr__(self) -> str: ...

class IndexScanIvfStatistics:
    @property
    def indexed_vector_entries(self) -> int: ...
    @property
    def num_partitions(self) -> int: ...
    @property
    def partition_entry_counts(self) -> list[int]: ...

class IndexScanSegmentStatistics:
    @property
    def uuid(self) -> str: ...
    @property
    def index_version(self) -> int: ...
    @property
    def created_at(self) -> Optional[datetime]: ...
    @property
    def base_id(self) -> Optional[int]: ...
    @property
    def dataset_version_at_last_update(self) -> int: ...
    @property
    def size_bytes(self) -> Optional[int]: ...
    @property
    def index_type(self) -> Optional[str]: ...
    @property
    def metric_type(self) -> Optional[str]: ...
    @property
    def details(self) -> Any: ...
    @property
    def covered_fragment_ids(self) -> Optional[Bitmap]: ...
    @property
    def covered_fragment_live_rows(self) -> Optional[int]: ...
    @property
    def covered_fragment_deleted_rows(self) -> Optional[int]: ...
    @property
    def ivf(self) -> Optional[IndexScanIvfStatistics]: ...

class IndexScanCoverage:
    @property
    def covered_fragment_count(self) -> int: ...
    @property
    def uncovered_fragment_ids(self) -> Bitmap: ...
    @property
    def covered_fragment_live_rows(self) -> Optional[int]: ...
    @property
    def uncovered_fragment_live_rows(self) -> Optional[int]: ...
    @property
    def uncovered_data_bytes(self) -> Optional[int]: ...
    @property
    def covered_live_row_ratio(self) -> Optional[float]: ...
    @property
    def overlay_stale_row_slots(self) -> int: ...

class IndexScanStatistics:
    @property
    def dataset_version(self) -> int: ...
    @property
    def fragment_count(self) -> int: ...
    @property
    def live_rows(self) -> Optional[int]: ...
    @property
    def deleted_rows(self) -> Optional[int]: ...
    @property
    def total_data_bytes(self) -> Optional[int]: ...
    @property
    def index_name(self) -> str: ...
    @property
    def keyed_field_ids(self) -> list[int]: ...
    @property
    def covering_field_ids(self) -> list[int]: ...
    @property
    def type_url(self) -> Optional[str]: ...
    @property
    def index_type(self) -> Optional[str]: ...
    @property
    def metric_type(self) -> Optional[str]: ...
    @property
    def details(self) -> Any: ...
    @property
    def total_index_bytes(self) -> Optional[int]: ...
    @property
    def coverage(self) -> Optional[IndexScanCoverage]: ...
    @property
    def segments(self) -> list[IndexScanSegmentStatistics]: ...
