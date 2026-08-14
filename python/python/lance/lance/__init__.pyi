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

from datetime import timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Self,
    Sequence,
    Tuple,
    Union,
)

import pyarrow as pa
from lance_namespace import LanceNamespace

from .._arrow.bf16 import BFloat16Array
from ..commit import CommitLock
from ..dataset import (
    AlterColumn,
    Branch,
    ExecuteResult,
    Index,
    LanceOperation,
    Tag,
    Transaction,
    UpdateResult,
    Version,
    VersionRef,
)
from ..fragment import (
    DataFile,
    FragmentMetadata,
)
from ..progress import FragmentWriteProgress as FragmentWriteProgress
from ..progress import IndexProgress as IndexProgress
from ..types import ReaderLike as ReaderLike
from ..udf import BatchUDF as BatchUDF
from .debug import format_fragment as format_fragment
from .debug import format_manifest as format_manifest
from .debug import format_schema as format_schema
from .debug import list_transactions as list_transactions
from .fragment import (
    DeletionFile as DeletionFile,
)
from .fragment import (
    RowIdMeta as RowIdMeta,
)
from .fragment import (
    RowIdSequence as RowIdSequence,
)
from .indices import IndexDescription as IndexDescription
from .indices import IndexSegment as IndexSegment
from .lance import PySearchFilter
from .optimize import (
    Compaction as Compaction,
)
from .optimize import (
    CompactionMetrics as CompactionMetrics,
)
from .optimize import (
    CompactionPlan as CompactionPlan,
)
from .optimize import (
    CompactionTask as CompactionTask,
)
from .optimize import (
    RewriteResult as RewriteResult,
)
from .schema import LanceSchema as LanceSchema
from .trace import TraceEvent as TraceEvent
from .trace import capture_trace_events as capture_trace_events
from .trace import shutdown_tracing as shutdown_tracing
from .trace import trace_to_chrome as trace_to_chrome

class MetricPoint:
    name: str
    kind: str
    attributes: Dict[str, str]
    value: Optional[float]
    buckets: Optional[List[Tuple[str, int]]]
    count: Optional[int]
    sum: Optional[float]

class MetricDescription:
    name: str
    kind: str
    unit: Optional[str]
    description: str

def register_lance_metrics_recorder() -> bool: ...
def lance_metrics_catalog() -> List[MetricDescription]: ...
def snapshot_lance_metrics() -> List[MetricPoint]: ...

class FtsToken:
    text: str
    position: int
    def __repr__(self) -> str: ...

def tokenize(
    query: str,
    *,
    analyzer: Optional[Literal["text", "code"]] = None,
    base_tokenizer: Optional[str] = None,
    language: Optional[str] = None,
    max_token_length: Optional[int] = 40,
    lower_case: Optional[bool] = None,
    stem: Optional[bool] = None,
    remove_stop_words: Optional[bool] = None,
    custom_stop_words: Optional[List[str]] = None,
    ascii_folding: Optional[bool] = None,
    min_ngram_length: Optional[int] = None,
    max_ngram_length: Optional[int] = None,
    prefix_only: Optional[bool] = None,
    split_identifiers: Optional[bool] = None,
    split_on_numerics: Optional[bool] = None,
    preserve_original: Optional[bool] = None,
    index_operators: Optional[bool] = None,
) -> List[FtsToken]:
    """Tokenize an FTS query without an index.

    ``max_token_length`` defaults to 40; pass ``None`` to disable the limit.
    """
    ...

class CleanupStats:
    bytes_removed: int
    old_versions: int
    data_files_removed: int
    transaction_files_removed: int
    index_files_removed: int
    deletion_files_removed: int

class CleanupCandidateFile:
    path: str
    kind: str
    unverified: bool
    size_bytes: int

class CleanupReferencedBranch:
    name: str
    referenced_version: int
    cleanup_candidate: bool

class CleanupExplanation:
    read_version: int
    stats: CleanupStats
    candidate_files: List[CleanupCandidateFile]
    candidate_files_truncated: bool
    candidate_file_limit: int
    referenced_branches: List[CleanupReferencedBranch]
    warnings: List[str]

class LanceFileWriteSummary:
    num_rows: int
    size_bytes: int

    def __repr__(self) -> str: ...

class LanceFileWriter:
    def __init__(
        self,
        path: str,
        schema: Optional[pa.Schema],
        data_cache_bytes: Optional[int],
        version: Optional[str],
        storage_options: Optional[Dict[str, str]],
        namespace_client: Optional[LanceNamespace],
        table_id: Optional[List[str]],
        keep_original_array: Optional[bool],
        max_page_bytes: Optional[int],
    ): ...
    def write_batch(self, batch: pa.RecordBatch) -> None: ...
    def finish(self) -> LanceFileWriteSummary: ...
    def add_schema_metadata(self, key: str, value: str) -> None: ...
    def add_global_buffer(self, data: bytes) -> int: ...

class BlobDescriptor:
    def __repr__(self) -> str: ...

class PackedBlobWriter:
    @property
    def blob_id(self) -> int: ...
    @property
    def path(self) -> str: ...
    @property
    def field(self) -> pa.Field: ...
    def write_blob(self, data: bytes) -> None: ...
    def write_blobs(
        self,
        payloads: Union[pa.BinaryArray, pa.LargeBinaryArray, pa.ChunkedArray],
    ) -> None: ...
    def finish(self) -> List[BlobDescriptor]: ...
    def finish_array(self, field_name: str) -> pa.StructArray: ...

class DedicatedBlobWriter:
    @property
    def blob_id(self) -> int: ...
    @property
    def path(self) -> str: ...
    def write(self, data: bytes) -> None: ...
    def finish(self) -> BlobDescriptor: ...

class BlobDescriptorArrayBuilder:
    def __init__(self, column: str): ...
    @property
    def field(self) -> pa.Field: ...
    def extend_packed(
        self, blob_id: int, offsets: List[int], sizes: List[int]
    ) -> None: ...
    def append_dedicated(self, blob_id: int, size: int) -> None: ...
    def append(self, value: BlobDescriptor) -> None: ...
    def extend(self, values: List[BlobDescriptor]) -> None: ...
    def append_inline(self, data: bytes) -> None: ...
    def append_null(self) -> None: ...
    def finish(self) -> pa.Array: ...

class LanceFileSession:
    def __init__(
        self,
        base_path: str,
        storage_options: Optional[Dict[str, str]] = None,
        namespace_client: Optional[LanceNamespace] = None,
        table_id: Optional[List[str]] = None,
    ): ...
    def open_reader(
        self, path: str, columns: Optional[List[str]] = None
    ) -> LanceFileReader: ...
    def open_writer(
        self,
        path: str,
        schema: Optional[pa.Schema] = None,
        data_cache_bytes: Optional[int] = None,
        version: Optional[str] = None,
        keep_original_array: Optional[bool] = None,
        max_page_bytes: Optional[int] = None,
    ) -> LanceFileWriter: ...
    def open_packed_blob_writer(self, path: str, blob_id: int) -> PackedBlobWriter: ...
    def open_dedicated_blob_writer(
        self, path: str, blob_id: int
    ) -> DedicatedBlobWriter: ...
    def contains(self, path: str) -> bool: ...
    def list(self, path: Optional[str] = None) -> List[str]: ...
    def list_with_delimiter(
        self, path: Optional[str] = None
    ) -> tuple[List[str], List[str]]: ...
    def read_range(self, path: str, offset: int, length: int) -> bytes: ...
    def delete_file(self, path: str) -> None: ...
    def upload_file(self, local_path: str, remote_path: str) -> None: ...
    def download_file(self, remote_path: str, local_path: str) -> None: ...

class LanceFileReader:
    def __init__(
        self,
        path: str,
        storage_options: Optional[Dict[str, str]],
        namespace_client: Optional[LanceNamespace],
        table_id: Optional[List[str]],
        columns: Optional[List[str]] = None,
    ): ...
    def read_all(
        self, batch_size: int, batch_readahead: int
    ) -> pa.RecordBatchReader: ...
    def read_range(
        self, start: int, num_rows: int, batch_size: int, batch_readahead: int
    ) -> pa.RecordBatchReader: ...
    def take_rows(
        self, indices: List[int], batch_size: int, batch_readahead: int
    ) -> pa.RecordBatchReader: ...
    def read_global_buffer(self, index: int) -> bytes: ...
    def metadata(self) -> LanceFileMetadata: ...
    def file_statistics(self) -> LanceFileStatistics: ...
    def num_rows(self): ...

class LanceBufferDescriptor:
    position: int
    size: int

class LancePageMetadata:
    buffers: List[LanceBufferDescriptor]
    encoding: str

class LanceColumnMetadata:
    column_buffers: List[LanceBufferDescriptor]
    pages: List[LancePageMetadata]

class LanceFileMetadata:
    schema: pa.Schema
    num_rows: int
    num_data_bytes: int
    num_column_metadata_bytes: int
    num_global_buffer_bytes: int
    global_buffers: List[LanceBufferDescriptor]
    columns: List[LanceColumnMetadata]

class LanceFileStatistics:
    columns: List[LanceColumnStatistics]

class LanceColumnStatistics:
    num_pages: int
    size_bytes: int

class _Session:
    def __init__(
        self,
        index_cache_size_bytes: Optional[int] = None,
        metadata_cache_size_bytes: Optional[int] = None,
        index_cache_backend: Optional[str | Dict[str, Any]] = None,
        metadata_cache_backend: Optional[str | Dict[str, Any]] = None,
    ) -> None:
        """Create a Lance session.

        Cache backends may be backend URI strings such as
        ``"moka://?capacity=1048576"`` or dictionaries such as
        ``{"kind": "moka", "options": {"capacity": "1048576"}}``.
        ``index_cache_backend`` is mutually exclusive with
        ``index_cache_size_bytes``. ``metadata_cache_backend`` is mutually
        exclusive with ``metadata_cache_size_bytes``.
        """
        ...
    def size_bytes(self) -> int: ...
    def index_cache_size_bytes(self) -> int: ...

class LanceBlobFile:
    def close(self): ...
    def is_closed(self) -> bool: ...
    def seek(self, offset: int): ...
    def tell(self) -> int: ...
    def size(self) -> int: ...
    def readall(self) -> bytes: ...
    def read_range(self, offset: int, length: int) -> bytes: ...
    def read_ranges(self, ranges: List[Tuple[int, int]]) -> List[bytes]: ...
    def read_into(self, b: bytearray) -> int: ...

class _Dataset:
    @property
    def uri(self) -> str: ...
    def __init__(
        self,
        uri: str,
        version: Optional[int | str] = None,
        block_size: Optional[int] = None,
        index_cache_size: Optional[int] = None,
        metadata_cache_size: Optional[int] = None,
        commit_handler: Optional[CommitLock] = None,
        storage_options: Optional[Dict[str, str]] = None,
        manifest: Optional[bytes] = None,
        metadata_cache_size_bytes: Optional[int] = None,
        index_cache_size_bytes: Optional[int] = None,
        **kwargs,
    ): ...
    @property
    def schema(self) -> pa.Schema: ...
    @property
    def lance_schema(self) -> LanceSchema: ...
    def replace_schema_metadata(self, metadata: Dict[str, str]): ...
    def replace_field_metadata(self, field_name: str, metadata: Dict[str, str]): ...
    @property
    def data_storage_version(self) -> str: ...
    @property
    def has_stable_row_ids(self) -> bool: ...
    def index_statistics(self, index_name: str) -> str: ...
    def serialized_manifest(self) -> bytes: ...
    def describe_indices(self) -> List[IndexDescription]: ...
    def remap_row_addrs(self, addrs: pa.Array) -> Optional[pa.Array]: ...
    def scanner(
        self,
        columns: Optional[List[str]] = None,
        columns_with_transform: Optional[List[Tuple[str, str]]] = None,
        filter: Optional[str] = None,
        search_filter: Optional[PySearchFilter] = None,
        prefilter: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[Dict] = None,
        batch_size: Optional[int] = None,
        io_buffer_size: Optional[int] = None,
        batch_readahead: Optional[int] = None,
        fragment_readahead: Optional[int] = None,
        scan_in_order: Optional[bool] = None,
        fragments: Optional[List[_Fragment]] = None,
        index_segments: Optional[List[str]] = None,
        with_row_id: Optional[bool] = None,
        with_row_address: Optional[bool] = None,
        use_stats: Optional[bool] = None,
        substrait_filter: Optional[bytes] = None,
        fast_search: Optional[bool] = None,
        full_text_query: Optional[dict] = None,
        late_materialization: Optional[bool | List[str]] = None,
        blob_handling: Optional[str] = None,
        use_scalar_index: Optional[bool] = None,
        include_deleted_rows: Optional[bool] = None,
        scan_stats_callback: Optional[Callable[[Any], None]] = None,
        strict_batch_size: Optional[bool] = None,
        order_by: Optional[List[Any]] = None,
        disable_scoring_autoprojection: Optional[bool] = None,
        substrait_aggregate: Optional[bytes] = None,
    ) -> _Scanner: ...
    def count_rows(self, filter: Optional[str] = None) -> int: ...
    def take(
        self,
        row_indices: List[int],
        columns: Optional[List[str]] = None,
        columns_with_transform: Optional[List[Tuple[str, str]]] = None,
    ) -> pa.RecordBatch: ...
    def take_rows(
        self,
        row_indices: List[int],
        columns: Optional[List[str]] = None,
        columns_with_transform: Optional[List[Tuple[str, str]]] = None,
    ) -> pa.RecordBatch: ...
    def take_blobs(
        self,
        row_ids: List[int],
        blob_column: str,
    ) -> List[Optional[LanceBlobFile]]: ...
    def take_blobs_by_addresses(
        self,
        row_addresses: List[int],
        blob_column: str,
    ) -> List[Optional[LanceBlobFile]]: ...
    def take_blobs_by_indices(
        self,
        row_indices: List[int],
        blob_column: str,
    ) -> List[Optional[LanceBlobFile]]: ...
    def read_blobs(
        self,
        row_ids: List[int],
        blob_column: str,
        io_buffer_size: Optional[int] = None,
        preserve_order: Optional[bool] = None,
    ) -> List[Tuple[int, Optional[bytes]]]: ...
    def read_blobs_by_addresses(
        self,
        row_addresses: List[int],
        blob_column: str,
        io_buffer_size: Optional[int] = None,
        preserve_order: Optional[bool] = None,
    ) -> List[Tuple[int, Optional[bytes]]]: ...
    def read_blobs_by_indices(
        self,
        row_indices: List[int],
        blob_column: str,
        io_buffer_size: Optional[int] = None,
        preserve_order: Optional[bool] = None,
    ) -> List[Tuple[int, Optional[bytes]]]: ...
    def read_blob_ranges(
        self,
        requests: List[Tuple[int, int, int]],
        blob_column: str,
        selector: Literal["ids", "addresses", "indices"],
        io_buffer_size: Optional[int] = None,
        preserve_order: Optional[bool] = None,
    ) -> List[Tuple[int, int, Optional[bytes]]]: ...
    def take_scan(
        self,
        row_slices: Iterable[Tuple[int, int]],
        columns: Optional[List[str]] = None,
        batch_readahead: int = 10,
    ) -> pa.RecordBatchReader: ...
    def alter_columns(self, alterations: List[AlterColumn]): ...
    def merge(self, reader: pa.RecordBatchReader, left_on: str, right_on: str): ...
    def delete(self, predicate: str): ...
    def update(
        self,
        updates: Dict[str, str],
        predicate: Optional[str] = None,
    ) -> UpdateResult: ...
    def count_deleted_rows(self) -> int: ...
    def versions(self) -> List[Version]: ...
    def version_refs(self) -> List[VersionRef]: ...
    def version(self) -> int: ...
    def latest_version(self) -> int: ...
    def checkout_version(
        self, version: int | str | Tuple[Optional[str], Optional[int]]
    ) -> _Dataset: ...
    def checkout_latest(self) -> _Dataset: ...
    def shallow_clone(
        self,
        target_path: str,
        reference: Optional[int | str | Tuple[Optional[str], Optional[int]]] = None,
        storage_options: Optional[Dict[str, str]] = None,
    ) -> _Dataset: ...
    def restore(self): ...
    def cleanup_old_versions(
        self,
        older_than_micros: Optional[int] = None,
        retain_versions: Optional[int] = None,
        delete_unverified: Optional[bool] = None,
        error_if_tagged_old_versions: Optional[bool] = None,
        delete_rate_limit: Optional[int] = None,
    ) -> CleanupStats: ...
    def explain_cleanup_old_versions(
        self,
        older_than_micros: Optional[int] = None,
        retain_versions: Optional[int] = None,
        delete_unverified: Optional[bool] = None,
        error_if_tagged_old_versions: Optional[bool] = None,
        delete_rate_limit: Optional[int] = None,
        include_files: bool = False,
        max_files: int = 1000,
    ) -> CleanupExplanation: ...
    def get_version(self, tag: str) -> int: ...
    # Tag operations
    def tags(self) -> Dict[str, Tag]: ...
    def tags_ordered(self, order: Optional[str]) -> List[Tuple[str, Tag]]: ...
    def create_tag(
        self,
        tag: str,
        reference: Optional[int | str | Tuple[Optional[str], Optional[int]]] = None,
    ) -> Tag: ...
    def delete_tag(self, tag: str): ...
    def update_tag(
        self,
        tag: str,
        reference: Optional[int | str | Tuple[Optional[str], Optional[int]]] = None,
    ): ...
    def replace_tag_metadata(
        self,
        tag: str,
        metadata: Dict[str, str],
    ) -> None: ...
    # Branch operations
    def branches(self) -> Dict[str, Branch]: ...
    def branches_ordered(self, order: Optional[str]) -> List[Tuple[str, Branch]]: ...
    def create_branch(
        self,
        branch: str,
        reference: Optional[int | str | Tuple[Optional[str], Optional[int]]] = None,
        storage_options: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> _Dataset: ...
    def delete_branch(self, branch: str) -> None: ...
    def replace_branch_metadata(
        self,
        branch: str,
        metadata: Dict[str, str],
    ) -> None: ...
    def optimize_indices(self, **kwargs): ...
    def create_index(
        self,
        columns: List[str],
        index_type: str,
        name: Optional[str] = None,
        replace: Optional[bool] = None,
        storage_options: Optional[Dict[str, str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ): ...
    def drop_index(self, name: str): ...
    def prewarm_index(
        self,
        name: str,
        *,
        with_position: bool = False,
        index_segments: Optional[List[str]] = None,
    ): ...
    def merge_index_metadata(
        self,
        index_uuid: str,
        index_type: str,
        batch_readhead: Optional[int] = None,
        progress_callback: Optional[Callable[[IndexProgress], None]] = None,
    ): ...
    def merge_existing_index_segments(self, segments: List[Index]) -> Index: ...
    def commit_existing_index_segments(
        self, index_name: str, column: str, segments: List[Union[IndexSegment, Index]]
    ) -> None: ...
    def count_fragments(self) -> int: ...
    def num_small_files(self, max_rows_per_group: int) -> int: ...
    def get_fragments(self) -> List[_Fragment]: ...
    def get_fragment(self, fragment_id: int) -> Optional[_Fragment]: ...
    def index_cache_entry_count(self) -> int: ...
    def index_cache_hit_rate(self) -> float: ...
    def session(self) -> _Session: ...
    @staticmethod
    def drop(
        dest: str,
        storage_options: Optional[Dict[str, str]] = None,
        ignore_not_found: Optional[bool] = None,
    ): ...
    @staticmethod
    def commit(
        dest: str | _Dataset,
        operation: LanceOperation.BaseOperation,
        read_version: Optional[int] = None,
        commit_lock: Optional[CommitLock] = None,
        storage_options: Optional[Dict[str, str]] = None,
        enable_v2_manifest_paths: Optional[bool] = None,
        detached: Optional[bool] = None,
        max_retries: Optional[int] = None,
        enable_stable_row_ids: Optional[bool] = None,
        commit_timeout: Optional[timedelta] = None,
        **kwargs,
    ) -> _Dataset: ...
    @staticmethod
    def commit_batch(
        dest: str | _Dataset,
        transactions: Sequence[Transaction],
        commit_lock: Optional[CommitLock] = None,
        storage_options: Optional[Dict[str, str]] = None,
        enable_v2_manifest_paths: Optional[bool] = None,
        detached: Optional[bool] = None,
        max_retries: Optional[int] = None,
        commit_timeout: Optional[timedelta] = None,
    ) -> Tuple[_Dataset, Transaction]: ...
    def validate(self): ...
    def migrate_manifest_paths_v2(self): ...
    def drop_columns(self, columns: List[str]): ...
    def add_columns_from_reader(
        self, reader: pa.RecordBatchReader, batch_size: Optional[int] = None
    ): ...
    def add_columns(
        self,
        transforms: Dict[str, str] | BatchUDF | ReaderLike,
        read_columns: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ): ...
    def add_columns_with_schema(self, schema: pa.Schema): ...
    def read_transaction(self, version: int) -> Optional[Transaction]: ...
    def get_transactions(
        self, recent_transactions=10
    ) -> List[Optional[Transaction]]: ...
    def hamming_clustering_for_ivf_partition(
        self,
        index_name: str,
        partition_id: int,
        hamming_threshold: int,
        index_segments: Optional[List[str]] = None,
    ) -> pa.RecordBatchReader: ...
    def get_ivf_partition_info(
        self,
        index_name: str,
        index_segments: Optional[List[str]] = None,
    ) -> List[dict]: ...
    def hamming_clustering_for_sample(
        self,
        column: str,
        sample_size: Optional[int],
        hamming_threshold: int,
    ) -> pa.RecordBatchReader: ...
    def hamming_clustering_for_range(
        self,
        column: str,
        fragment_id: int,
        start_row: int,
        num_rows: int,
        hamming_threshold: int,
    ) -> pa.RecordBatchReader: ...

class _MergeInsertBuilder:
    def __init__(self, dataset: _Dataset, on: str | Iterable[str]): ...
    def when_matched_update_all(self, condition: Optional[str] = None) -> Self: ...
    def when_matched_fail(self) -> Self: ...
    def when_not_matched_insert_all(self) -> Self: ...
    def when_not_matched_by_source_delete(self, expr: Optional[str] = None) -> Self: ...
    def target_bases(self, bases: list[str]) -> Self: ...
    def target_all_bases(self, include_primary: bool = True) -> Self: ...
    def execute(self, new_data: pa.RecordBatchReader) -> ExecuteResult: ...
    def execute_batches(self, new_data: pa.RecordBatchReader) -> ExecuteResult: ...
    def execute_uncommitted(
        self, new_data: pa.RecordBatchReader
    ) -> tuple[Transaction, ExecuteResult]: ...
    def execute_uncommitted_batches(
        self, new_data: pa.RecordBatchReader
    ) -> tuple[Transaction, ExecuteResult]: ...

class _Scanner:
    @property
    def schema(self) -> pa.Schema: ...
    def explain_plan(self, verbose: bool) -> str: ...
    def analyze_plan(self, count_rows: bool = False) -> str: ...
    def count_rows(self) -> int: ...
    def to_pyarrow(self) -> pa.RecordBatchReader: ...

class _Fragment:
    @staticmethod
    def create_from_file(
        filename: str,
        dataset: _Dataset,
        fragment_id: int,
    ) -> FragmentMetadata: ...
    @staticmethod
    def create(
        dataset_uri: str,
        fragment_id: Optional[int],
        reader: ReaderLike,
        **kwargs,
    ): ...
    def id(self) -> int: ...
    def metadata(self) -> FragmentMetadata: ...
    def count_rows(self, _filter: Optional[str] = None) -> int: ...
    def take(
        self,
        row_indices: List[int],
        columns: Optional[Union[List[str], Dict[str, str]]],
    ) -> pa.RecordBatch: ...
    def scanner(
        self,
        columns: Optional[List[str]] = None,
        columns_with_transform: Optional[List[Tuple[str, str]]] = None,
        batch_size: Optional[int] = None,
        filter: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        with_row_id: Optional[bool] = None,
        with_row_address: Optional[bool] = None,
        batch_readahead: Optional[int] = None,
        blob_handling: Optional[str] = None,
        order_by: Optional[List[Any]] = None,
    ) -> _Scanner: ...
    def add_columns_from_reader(
        self,
        reader: ReaderLike,
        batch_size: Optional[int],
    ) -> Tuple[FragmentMetadata, LanceSchema]: ...
    def add_columns(
        self,
        transforms: Dict[str, str] | BatchUDF | ReaderLike,
        read_columns: Optional[List[str]],
        batch_size: Optional[int],
    ) -> Tuple[FragmentMetadata, LanceSchema]: ...
    def delete(self, predicate: str) -> Optional[_Fragment]: ...
    def delete_rows(self, offsets: List[int]) -> Optional[_Fragment]: ...
    def schema(self) -> pa.Schema: ...
    def data_files(self) -> List[DataFile]: ...
    def deletion_file(self) -> Optional[str]: ...
    @property
    def physical_rows(self) -> int: ...
    @property
    def num_deletions(self) -> int: ...
    def validate(self) -> None: ...

def iops_counter() -> int: ...
def bytes_read_counter() -> int: ...
def stable_version() -> str: ...
def _write_dataset(
    reader: pa.RecordBatchReader, uri: str | Path | _Dataset, params: Dict[str, Any]
): ...
def _write_fragments(
    dataset_uri: str | Path | _Dataset,
    reader: ReaderLike,
    mode: str,
    max_rows_per_file: int,
    max_rows_per_group: int,
    max_bytes_per_file: int,
    progress: Optional[FragmentWriteProgress],
    data_storage_version: Optional[str],
    storage_options: Optional[Dict[str, str]],
    namespace_client: Optional[LanceNamespace],
    table_id: Optional[List[str]],
    enable_stable_row_ids: bool,
    target_bases: Optional[List[str]] = None,
    target_all_bases: Optional[bool] = None,
    initial_bases: Optional[List[Any]] = None,
    base_store_params: Optional[Dict[str, Dict[str, str]]] = None,
    external_blob_mode: Literal["reference", "ingest"] = "reference",
    allow_external_blob_outside_bases: bool = False,
    session: Optional[_Session] = None,
): ...
def _write_fragments_transaction(
    dataset_uri: str | Path | _Dataset,
    reader: ReaderLike,
    mode: str,
    max_rows_per_file: int,
    max_rows_per_group: int,
    max_bytes_per_file: int,
    progress: Optional[FragmentWriteProgress],
    data_storage_version: Optional[str],
    storage_options: Optional[Dict[str, str]],
    namespace_client: Optional[LanceNamespace],
    table_id: Optional[List[str]],
    enable_stable_row_ids: bool,
    target_bases: Optional[List[str]] = None,
    target_all_bases: Optional[bool] = None,
    initial_bases: Optional[List[Any]] = None,
    base_store_params: Optional[Dict[str, Dict[str, str]]] = None,
    external_blob_mode: Literal["reference", "ingest"] = "reference",
    allow_external_blob_outside_bases: bool = False,
    session: Optional[_Session] = None,
) -> Transaction: ...
def _json_to_schema(schema_json: str) -> pa.Schema: ...
def _schema_to_json(schema: pa.Schema) -> str: ...
def _parse_field_path(path: str) -> list[str]: ...
def _format_field_path(segments: list[str]) -> str: ...
def _evaluate_sharding_spec(
    batch: pa.RecordBatch,
    spec: Dict[str, Any],
    schema: LanceSchema,
) -> pa.RecordBatch: ...

class _CompactedSsTable:
    shard_id: str
    generation: int
    def __init__(self, shard_id: str, generation: int) -> None: ...

class _ShardSnapshot:
    shard_id: str
    def __init__(self, shard_id: str) -> None: ...
    def with_spec_id(self, spec_id: int) -> Self: ...
    def with_current_generation(self, generation: int) -> Self: ...
    def with_sstable(self, generation: int, path: str) -> Self: ...

class _ShardWriter:
    shard_id: str
    def put(self, data: Any) -> None: ...
    def delete(self, keys: Any) -> None: ...
    def close(self) -> None: ...
    def stats(self) -> Dict[str, Any]: ...
    def memtable_stats(self) -> Dict[str, Any]: ...
    def lsm_scanner(
        self, shard_snapshots: Optional[List[_ShardSnapshot]] = None
    ) -> _LsmScanner: ...

class _LsmScanner:
    @staticmethod
    def from_snapshots(
        dataset: _Dataset, shard_snapshots: List[_ShardSnapshot]
    ) -> _LsmScanner: ...
    def project(self, columns: List[str]) -> Self: ...
    def filter(self, expr: str) -> Self: ...
    def limit(self, n: int, offset: Optional[int] = None) -> Self: ...
    def with_row_address(self) -> Self: ...
    def with_memtable_gen(self) -> Self: ...
    def to_batch(self) -> pa.RecordBatch: ...
    def to_batches(self) -> List[pa.RecordBatch]: ...
    def count_rows(self) -> int: ...

class _ExecutionPlan:
    schema: pa.Schema
    dataset_schema: pa.Schema
    def explain(self) -> str: ...
    def to_reader(self) -> pa.RecordBatchReader: ...
    def to_batches(self) -> List[pa.RecordBatch]: ...

class _LsmPointLookupPlanner:
    def __init__(
        self,
        dataset: _Dataset,
        shard_snapshots: List[_ShardSnapshot],
        pk_columns: Optional[List[str]] = None,
    ) -> None: ...
    def plan_lookup(
        self, pk_value: pa.Array, columns: Optional[List[str]] = None
    ) -> _ExecutionPlan: ...

class _LsmVectorSearchPlanner:
    def __init__(
        self,
        dataset: _Dataset,
        shard_snapshots: List[_ShardSnapshot],
        vector_column: str,
        pk_columns: Optional[List[str]] = None,
        distance_type: Optional[str] = None,
    ) -> None: ...
    def plan_search(
        self,
        query: pa.Array,
        k: int = 10,
        nprobes: int = 20,
        columns: Optional[List[str]] = None,
    ) -> _ExecutionPlan: ...

class _Hnsw:
    @staticmethod
    def build(
        vectors_array: Iterator[pa.Array],
        max_level: int,
        m: int,
        ef_construction: int,
    ): ...
    def to_lance_file(self, file_path: str): ...
    def vectors(self) -> pa.Array: ...

class _KMeans:
    def __init__(
        self,
        k: int,
        metric_type: str,
        max_iters: int,
        centroids_arr: Optional[pa.FixedSizeListArray] = None,
    ): ...
    def fit(self, data: pa.FixedSizeListArray): ...
    def predict(self, data: pa.FixedSizeListArray) -> pa.UInt32Array: ...
    def centroids(
        self,
    ) -> Union[pa.FixedShapeTensorType, pa.FixedSizeListType | None]: ...

class BFloat16:
    def __init__(self, value: float) -> None: ...
    @classmethod
    def from_bytes(cls, bytes: bytes) -> BFloat16: ...
    def as_float(self) -> float: ...
    def __lt__(self, other: BFloat16) -> bool: ...
    def __le__(self, other: BFloat16) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __gt__(self, other: BFloat16) -> bool: ...
    def __ge__(self, other: BFloat16) -> bool: ...

def bfloat16_array(values: Sequence[float | None]) -> BFloat16Array: ...

class PyFullTextQuery:
    @staticmethod
    def match_query(
        column: str,
        query: str,
        boost: float = 1.0,
        fuzziness: Optional[int] = 0,
        max_expansions: int = 50,
        operator: str = "OR",
    ) -> PyFullTextQuery: ...
    @staticmethod
    def phrase_query(
        query: str,
        column: str,
    ) -> PyFullTextQuery: ...
    @staticmethod
    def boost_query(
        positive: PyFullTextQuery,
        negative: PyFullTextQuery,
        negative_boost: Optional[float],
    ) -> PyFullTextQuery: ...
    @staticmethod
    def multi_match_query(
        query: str,
        columns: List[str],
        boosts: Optional[List[float]] = None,
        operator: str = "OR",
    ) -> PyFullTextQuery: ...
    @staticmethod
    def combined_fields_query(
        query: str,
        columns: List[str],
        boosts: Optional[List[float]] = None,
        operator: str = "OR",
    ) -> PyFullTextQuery: ...

class ScanStatistics:
    """Statistics about a scan operation."""

    iops: int
    requests: int
    bytes_read: int
    indices_loaded: int
    parts_loaded: int
    index_comparisons: int
    index_cache_hits: int
    """Number of index cache page lookups where the loader was not executed
    in this scan. Counts both true cache hits on already-populated entries
    and coalesced concurrent loads (a follower attached to another caller's
    in-flight load).

    Instrumented boundaries in this release: BTree, IVF v2 (write-cache scan
    path), inverted posting list (grouped and per-token) and its per-token
    metadata, inverted phrase positions, bitmap (Equals / Range / IsIn),
    ngram, rtree.

    Caveats:

    * IVF v2 streaming scans and legacy v1 IVF partitions bypass the cache
      by design and are therefore reported as a miss on every call.
    * A cold posting-list lookup on the grouped inverted layout can record
      up to two misses (group + per-token metadata) for a single term.

    Uninstrumented paths (HNSW graph pages, quantizer codebooks) do not
    contribute to either counter."""
    index_cache_misses: int
    """Number of index cache page lookups where the loader ran (the page was
    not resident and had to be materialised, typically from storage). See
    the sibling ``index_cache_hits`` for the paired counter and the list of
    instrumented boundaries."""
    all_counts: Dict[
        str, int
    ]  # Additional metrics for debugging purposes. Subject to change.

class DatasetBasePath:
    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        is_dataset_root: bool = False,
        id: Optional[int] = None,
    ) -> None: ...

__version__: str
language_model_home: Callable[[], str]
