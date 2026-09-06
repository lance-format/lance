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
from typing import Optional

import pyarrow as pa

from .. import _Fragment

class IndexConfig:
    index_type: str
    config: str

class IndexSegment:
    uuid: str
    fragment_ids: set[int]
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
    fragment_ids: set[int]
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
    covering_fields: list[int]
    covering_field_names: list[str]
    segments: list[IndexSegmentDescription]
    details: dict
    total_size_bytes: Optional[int]

    def __repr__(self) -> str: ...
