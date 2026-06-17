# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset Fragment"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import pyarrow as pa

from .lance import (
    DeletionFile as DeletionFile,
)
from .lance import (
    RowDatasetVersionMeta as RowDatasetVersionMeta,
)
from .lance import (
    RowIdMeta as RowIdMeta,
)
from .lance import _Fragment, _write_fragments, _write_fragments_transaction
from .progress import FragmentWriteProgress, NoopFragmentWriteProgress
from .types import _coerce_reader
from .udf import BatchUDF, normalize_transform

if TYPE_CHECKING:
    from .dataset import (
        ColumnOrdering,
        DatasetBasePath,
        LanceDataset,
        LanceScanner,
        ReaderLike,
        Transaction,
    )
    from .lance import LanceSchema
    from .namespace import LanceNamespace


DEFAULT_MAX_BYTES_PER_FILE = 90 * 1024 * 1024 * 1024


@dataclass
class FragmentMetadata:
    """Metadata for a fragment.

    Attributes
    ----------
    id : int
        The ID of the fragment.
    files : List[DataFile]
        The data files of the fragment. Each data file must have the same number
        of rows. Each file stores a different subset of the columns.
    physical_rows : int
        The number of rows originally in this fragment. This is the number of rows
        in the data files before deletions.
    deletion_file : Optional[DeletionFile]
        The deletion file, if any.
    row_id_meta : Optional[RowIdMeta]
        The row id metadata, if any.
    created_at_version_meta : Optional[RowDatasetVersionMeta]
        The row created at version metadata, if any.
    last_updated_at_version_meta : Optional[RowDatasetVersionMeta]
        The row last updated at version metadata, if any.
    """

    id: int
    files: List[DataFile]
    physical_rows: int
    deletion_file: Optional[DeletionFile] = None
    row_id_meta: Optional[RowIdMeta] = None
    created_at_version_meta: Optional[RowDatasetVersionMeta] = None
    last_updated_at_version_meta: Optional[RowDatasetVersionMeta] = None

    @property
    def num_deletions(self) -> int:
        """The number of rows that have been deleted from this fragment."""
        if self.deletion_file is None:
            return 0
        else:
            return self.deletion_file.num_deleted_rows

    @property
    def num_rows(self) -> int:
        """The number of rows in this fragment after deletions."""
        return self.physical_rows - self.num_deletions

    def data_files(self) -> List[DataFile]:
        warnings.warn(
            "FragmentMetadata.data_files is deprecated. Use .files instead.",
            DeprecationWarning,
        )
        return self.files

    def to_json(self) -> dict:
        """Get this as a simple JSON-serializable dictionary."""
        files = [asdict(f) for f in self.files]
        for f in files:
            f["path"] = f.pop("_path")
        return dict(
            id=self.id,
            files=files,
            physical_rows=self.physical_rows,
            deletion_file=(
                self.deletion_file.asdict() if self.deletion_file is not None else None
            ),
            row_id_meta=(
                self.row_id_meta.asdict() if self.row_id_meta is not None else None
            ),
            created_at_version_meta=(
                json.loads(self.created_at_version_meta.json())
                if self.created_at_version_meta is not None
                else None
            ),
            last_updated_at_version_meta=(
                json.loads(self.last_updated_at_version_meta.json())
                if self.last_updated_at_version_meta is not None
                else None
            ),
        )

    @staticmethod
    def from_json(json_data: str) -> FragmentMetadata:
        json_data = json.loads(json_data)

        deletion_file = json_data.get("deletion_file")
        if deletion_file is not None:
            deletion_file = DeletionFile(**deletion_file)

        row_id_meta = json_data.get("row_id_meta")
        if row_id_meta is not None:
            row_id_meta = RowIdMeta.from_dict(row_id_meta)

        created_at_version_meta = json_data.get("created_at_version_meta")
        if created_at_version_meta is not None:
            created_at_version_meta = RowDatasetVersionMeta.from_json(
                json.dumps(created_at_version_meta)
            )

        last_updated_at_version_meta = json_data.get("last_updated_at_version_meta")
        if last_updated_at_version_meta is not None:
            last_updated_at_version_meta = RowDatasetVersionMeta.from_json(
                json.dumps(last_updated_at_version_meta)
            )

        return FragmentMetadata(
            id=json_data["id"],
            files=[DataFile(**f) for f in json_data["files"]],
            physical_rows=json_data["physical_rows"],
            deletion_file=deletion_file,
            row_id_meta=row_id_meta,
            created_at_version_meta=created_at_version_meta,
            last_updated_at_version_meta=last_updated_at_version_meta,
        )


@dataclass
class DataFile:
    """
    A data file in a fragment.

    Attributes
    ----------
    path : str
        The path to the data file.
    fields : List[int]
        The field ids of the columns in this file.
    column_indices : List[int]
        The column indices where the fields are stored in the file.  Will  have
        the same length as `fields`.
    file_major_version : int
        The major version of the data storage format.
    file_minor_version : int
        The minor version of the data storage format.
    file_size_bytes : Optional[int]
        The size of the data file in bytes, if available.
    """

    _path: str
    fields: List[int]
    column_indices: List[int] = field(default_factory=list)
    file_major_version: int = 0
    file_minor_version: int = 0
    file_size_bytes: Optional[int] = None
    base_id: Optional[int] = None

    def __init__(
        self,
        path: str,
        fields: List[int],
        column_indices: List[int] = None,
        file_major_version: int = 0,
        file_minor_version: int = 0,
        file_size_bytes: Optional[int] = None,
        base_id: Optional[int] = None,
    ):
        # TODO: only we eliminate the path method, we can remove this
        self._path = path
        self.fields = fields
        self.column_indices = column_indices or []
        self.file_major_version = file_major_version
        self.file_minor_version = file_minor_version
        self.file_size_bytes = file_size_bytes
        self.base_id = base_id

    def __repr__(self):
        # pretend we have a 'path' attribute
        return (
            f"DataFile(path='{self._path}', fields={self.fields}, "
            f"column_indices={self.column_indices}, "
            f"file_major_version={self.file_major_version}, "
            f"file_minor_version={self.file_minor_version}, "
            f"file_size_bytes={self.file_size_bytes})"
        )

    @property
    def path(self) -> str:
        # path used to be a method. This is for backwards compatibility.
        class CallableStr(str):
            def __call__(self):
                warnings.warn(
                    "DataFile.path() is deprecated, use DataFile.path instead",
                    DeprecationWarning,
                )
                return self

            def __reduce__(self):
                return (str, (str(self),))

        return CallableStr(self._path)

    def field_ids(self) -> List[int]:
        warnings.warn(
            "DataFile.field_ids is deprecated, use DataFile.fields instead",
            DeprecationWarning,
        )
        return self.fields

    @classmethod
    def create(
        cls,
        dataset: "LanceDataset",
        path: str,
        *,
        base_id: Optional[int] = None,
    ) -> "DataFile":
        """Create a DataFile by reading metadata from an existing lance file.

        This is a convenience method for creating DataFile metadata needed
        for operations like DataReplacement. It opens the file, reads its
        schema and version information, matches columns to the dataset's
        schema to determine field IDs, and calculates column indices.

        Parameters
        ----------
        dataset : LanceDataset
            The dataset this file will belong to.
        path : str
            The path to the data file, relative to the dataset's data directory.
        base_id : int, optional
            The base path ID if the file is outside the dataset directory.

        Returns
        -------
        DataFile
        """
        return _Fragment.create_data_file(dataset._ds, path, base_id=base_id)


class LanceFragment(pa.dataset.Fragment):
    def __init__(
        self,
        dataset: "LanceDataset",
        fragment_id: Optional[int],
        *,
        fragment: Optional[_Fragment] = None,
    ):
        self._ds = dataset
        if fragment is None:
            if fragment_id is None:
                raise ValueError("Either fragment or fragment_id must be specified")
            fragment = dataset.get_fragment(fragment_id)._fragment
        self._fragment: _Fragment = fragment
        if self._fragment is None:
            raise ValueError(f"Fragment id does not exist: {fragment_id}")

    def __repr__(self):
        return self._fragment.__repr__()

    def __reduce__(self):
        from .dataset import LanceDataset

        ds = LanceDataset(self._ds.uri, self._ds.version)
        return LanceFragment, (ds, self.fragment_id)

    @staticmethod
    def create_from_file(
        filename: str,
        dataset: LanceDataset,
        fragment_id: int,
    ) -> FragmentMetadata:
        """Create a fragment from the given datafile uri.

        This can be used if the datafile is loss from dataset.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        filename: str
            The filename of the datafile.
        dataset: LanceDataset
            The dataset that the fragment belongs to.
        fragment_id: int
            The ID of the fragment.
        """
        return _Fragment.create_from_file(filename, dataset._ds, fragment_id)

    @staticmethod
    def create(
        dataset_uri: Union[str, Path],
        data: ReaderLike,
        fragment_id: Optional[int] = None,
        schema: Optional[pa.Schema] = None,
        max_rows_per_group: int = 1024,
        progress: Optional[FragmentWriteProgress] = None,
        mode: str = "append",
        *,
        data_storage_version: Optional[str] = None,
        use_legacy_format: Optional[bool] = None,
        storage_options: Optional[Dict[str, str]] = None,
        namespace_client: Optional["LanceNamespace"] = None,
        table_id: Optional[List[str]] = None,
    ) -> FragmentMetadata:
        """Create a :class:`FragmentMetadata` from the given data.

        This can be used if the dataset is not yet created.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        dataset_uri: str
            The URI of the dataset.
        fragment_id: int
            The ID of the fragment.
        data: pa.Table or pa.RecordBatchReader
            The data to be written to the fragment.
        schema: pa.Schema, optional
            The schema of the data. If not specified, the schema will be inferred
            from the data.
        max_rows_per_group: int, default 1024
            The maximum number of rows per group in the data file.
        progress: FragmentWriteProgress, optional
            *Experimental API*. Progress tracking for writing the fragment. Pass
            a custom class that defines hooks to be called when each fragment is
            starting to write and finishing writing.
        mode: str, default "append"
            The write mode. If "append" is specified, the data will be checked
            against the existing dataset's schema. Otherwise, pass "create" or
            "overwrite" to assign new field ids to the schema.
        data_storage_version: optional, str, default None
            The version of the data storage format to use. Newer versions are more
            efficient but require newer versions of lance to read.  The default (None)
            will use the latest stable version.  See the user guide for more details.
        use_legacy_format: bool, default None
            Deprecated parameter.  Use data_storage_version instead.
        storage_options : optional, dict
            Extra options that make sense for a particular storage connection. This is
            used to store connection parameters like credentials, endpoint, etc.
        namespace_client : optional, LanceNamespace
            A namespace client for automatic credential refresh. When provided with
            `table_id`, a storage options provider will be created automatically to
            refresh credentials via the namespace. Must be provided together with
            `table_id`. The caller should provide initial/merged storage options via
            the `storage_options` parameter.
        table_id : optional, List[str]
            The table identifier when using a namespace (e.g., ["my_table"]).
            Must be provided together with `namespace_client`.

        See Also
        --------
        lance.dataset.LanceOperation.Overwrite :
            The operation used to create a new dataset or overwrite one using
            fragments created with this API. See the doc page for an example of
            using this API.
        lance.dataset.LanceOperation.Append :
            The operation used to append fragments created with this API to an
            existing dataset. See the doc page for an example of using this API.

        Returns
        -------
        FragmentMetadata
        """
        # Validate namespace_client and table_id are provided together
        if namespace_client is not None and table_id is None:
            raise ValueError(
                "Both 'namespace_client' and 'table_id' must be provided together."
            )
        elif table_id is not None and namespace_client is None:
            raise ValueError(
                "Both 'namespace_client' and 'table_id' must be provided together."
            )

        if use_legacy_format is not None:
            warnings.warn(
                "use_legacy_format is deprecated, use data_storage_version instead",
                DeprecationWarning,
            )
            if use_legacy_format:
                data_storage_version = "legacy"
            else:
                data_storage_version = "stable"

        reader = _coerce_reader(data, schema)

        if isinstance(dataset_uri, Path):
            dataset_uri = str(dataset_uri)
        if progress is None:
            progress = NoopFragmentWriteProgress()

        return _Fragment.create(
            dataset_uri,
            fragment_id,
            reader,
            max_rows_per_group=max_rows_per_group,
            progress=progress,
            mode=mode,
            data_storage_version=data_storage_version,
            storage_options=storage_options,
            namespace_client=namespace_client,
            table_id=table_id,
        )

    @property
    def fragment_id(self):
        return self._fragment.id()

    def count_rows(
        self, filter: Optional[Union[pa.compute.Expression, str]] = None
    ) -> int:
        if isinstance(filter, pa.compute.Expression):
            return self.scanner(
                with_row_id=True, columns=[], filter=filter
            ).count_rows()
        return self._fragment.count_rows(filter)

    @property
    def num_deletions(self) -> int:
        """Return the number of deleted rows in this fragment."""
        return self._fragment.num_deletions

    @property
    def physical_rows(self) -> int:
        """
        Return the number of rows originally in this fragment.

        To get the number of rows after deletions, use
        :meth:`count_rows` instead.
        """
        return self._fragment.physical_rows

    @property
    def physical_schema(self) -> pa.Schema:
        # override the pyarrow super class method otherwise causes segfault
        raise NotImplementedError("Not implemented yet for LanceFragment")

    @property
    def partition_expression(self) -> pa.Schema:
        # override the pyarrow super class method otherwise causes segfault
        raise NotImplementedError("Not implemented yet for LanceFragment")

    def head(self, num_rows: int) -> pa.Table:
        return self.scanner(limit=num_rows).to_table()

    def scanner(
        self,
        *,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        batch_size: Optional[int] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        with_row_id: bool = False,
        with_row_address: bool = False,
        batch_readahead: int = 16,
        blob_handling: Optional[
            Literal["all_binary", "blobs_descriptions", "all_descriptions"]
        ] = None,
        order_by: Optional[List[ColumnOrdering]] = None,
    ) -> "LanceScanner":
        """See Dataset::scanner for details"""
        filter_str = str(filter) if filter is not None else None

        columns_arg = {}
        if isinstance(columns, dict):
            # convert to list of tuples
            columns_arg = {"columns_with_transform": list(columns.items())}
        elif isinstance(columns, list):
            columns_arg = {"columns": columns}

        s = self._fragment.scanner(
            batch_size=batch_size,
            filter=filter_str,
            limit=limit,
            offset=offset,
            with_row_id=with_row_id,
            with_row_address=with_row_address,
            batch_readahead=batch_readahead,
            blob_handling=blob_handling,
            order_by=order_by,
            **columns_arg,
        )
        from .dataset import LanceScanner

        snapshot = {
            "_limit": limit,
            "_filter": filter_str,
            "_search_filter": None,
            "_substrait_filter": None,
            "_prefilter": False,
            "_late_materialization": None,
            "_blob_handling": blob_handling,
            "_offset": offset,
            "_columns": tuple(columns) if isinstance(columns, list) else None,
            "_columns_with_transform": (
                tuple(columns.items()) if isinstance(columns, dict) else None
            ),
            "_nearest": None,
            "_batch_size": batch_size,
            "_io_buffer_size": None,
            "_batch_readahead": batch_readahead,
            "_fragment_readahead": None,
            "_scan_in_order": True,
            "_fragments": (self._fragment,),
            "_with_row_id": with_row_id,
            "_with_row_address": with_row_address,
            "_use_stats": True,
            "_fast_search": False,
            "_full_text_query": None,
            "_use_scalar_index": None,
            "_include_deleted_rows": None,
            "_scan_stats_callback": None,
            "_strict_batch_size": False,
            "_orderings": tuple(order_by) if order_by is not None else None,
            "_disable_scoring_autoprojection": None,
            "_substrait_aggregate": None,
        }
        return LanceScanner(s, self._ds, snapshot=snapshot)

    def open_session(
        self,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        with_row_address: Optional[bool] = None,
    ) -> "FragmentSession":
        """Open a FragmentSession, which manages a short-lived session of LanceFragment.
         This API works well for users making repeated requests over the same columns.

        Parameters
        ----------
        columns: list of str, or dict of str to str default None
            List of column names to be fetched.
            Or a dictionary of column names to SQL expressions.
            All columns are fetched if None or unspecified.
        with_row_address: enable returns with row addresses.

        Returns
        -------
        session : FragmentSession
        """
        return FragmentSession(self._fragment.open_session(columns, with_row_address))

    def take(
        self,
        indices,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
    ) -> pa.Table:
        return pa.Table.from_batches([self._fragment.take(indices, columns=columns)])

    def to_batches(
        self,
        *,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        batch_size: Optional[int] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        with_row_id: bool = False,
        with_row_address: bool = False,
        batch_readahead: int = 16,
        blob_handling: Optional[
            Literal["all_binary", "blobs_descriptions", "all_descriptions"]
        ] = None,
        order_by: Optional[List[ColumnOrdering]] = None,
    ) -> Iterator[pa.RecordBatch]:
        return self.scanner(
            columns=columns,
            batch_size=batch_size,
            filter=filter,
            limit=limit,
            offset=offset,
            with_row_id=with_row_id,
            with_row_address=with_row_address,
            batch_readahead=batch_readahead,
            blob_handling=blob_handling,
            order_by=order_by,
        ).to_batches()

    def to_table(
        self,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        with_row_id: bool = False,
        with_row_address: bool = False,
        blob_handling: Optional[
            Literal["all_binary", "blobs_descriptions", "all_descriptions"]
        ] = None,
        order_by: Optional[List[ColumnOrdering]] = None,
    ) -> pa.Table:
        return self.scanner(
            columns=columns,
            filter=filter,
            limit=limit,
            offset=offset,
            with_row_id=with_row_id,
            with_row_address=with_row_address,
            blob_handling=blob_handling,
            order_by=order_by,
        ).to_table()

    def to_pandas(
        self,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        batch_size: Optional[int] = None,
        with_row_id: bool = False,
        with_row_address: bool = False,
        blob_mode: str = "lazy",
        order_by: Optional[List[ColumnOrdering]] = None,
        **kwargs: Any,
    ) -> Any:
        """Read this fragment into a :py:class:`pandas.DataFrame`.

        Parameters are the same as :meth:`to_table`, except pandas export uses
        ``blob_mode`` instead of Arrow-facing ``blob_handling`` and accepts
        ``batch_size`` for scan batching.

        Parameters
        ----------
        blob_mode: str, default "lazy"
            Controls how blob columns are returned.

            - ``"lazy"``: return :class:`lance.BlobFile` objects
            - ``"bytes"``: return Python ``bytes``
            - ``"descriptions"``: preserve ``to_table().to_pandas()`` behavior
        batch_size: int, optional
            The maximum number of rows per scan batch.
        **kwargs
            Forwarded to :meth:`pyarrow.Table.to_pandas` for non-blob columns.
        """
        return self.scanner(
            columns=columns,
            filter=filter,
            limit=limit,
            offset=offset,
            batch_size=batch_size,
            with_row_id=with_row_id,
            with_row_address=with_row_address,
            order_by=order_by,
        ).to_pandas(blob_mode=blob_mode, **kwargs)

    def merge(
        self,
        data_obj: ReaderLike,
        left_on: str,
        right_on: Optional[str] = None,
        schema=None,
    ) -> Tuple[FragmentMetadata, LanceSchema]:
        """
        Merge another dataset into this fragment.

        Performs a left join, where the fragment is the left side and data_obj
        is the right side. Rows existing in the dataset but not on the left will
        be filled with null values, unless Lance doesn't support null values for
        some types, in which case an error will be raised.

        Parameters
        ----------
        data_obj: Reader-like
            The data to be merged. Acceptable types are:
            - Pandas DataFrame, Pyarrow Table, Dataset, Scanner,
            Iterator[RecordBatch], or RecordBatchReader
        left_on: str
            The name of the column in the dataset to join on.
        right_on: str or None
            The name of the column in data_obj to join on. If None, defaults to
            left_on.

        Examples
        --------

        >>> import lance
        >>> import pyarrow as pa
        >>> df = pa.table({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
        >>> dataset = lance.write_dataset(df, "dataset")
        >>> dataset.to_table().to_pandas()
           x  y
        0  1  a
        1  2  b
        2  3  c
        >>> fragments = dataset.get_fragments()
        >>> new_df = pa.table({'x': [1, 2, 3], 'z': ['d', 'e', 'f']})
        >>> merged = []
        >>> schema = None
        >>> for f in fragments:
        ...     f, schema = f.merge(new_df, 'x')
        ...     merged.append(f)
        >>> merge = lance.LanceOperation.Merge(merged, schema)
        >>> dataset = lance.LanceDataset.commit("dataset", merge, read_version=1)
        >>> dataset.to_table().to_pandas()
           x  y  z
        0  1  a  d
        1  2  b  e
        2  3  c  f

        See Also
        --------
        LanceDataset.merge_columns :
            Add columns to this Fragment.

        Returns
        -------
        Tuple[FragmentMetadata, LanceSchema]
            A new fragment with the merged column(s) and the final schema.
        """
        if right_on is None:
            right_on = left_on

        reader = _coerce_reader(data_obj, schema)
        max_field_id = self._ds.max_field_id
        metadata, schema = self._fragment.merge(reader, left_on, right_on, max_field_id)
        return metadata, schema

    def update_columns(
        self,
        data_obj: ReaderLike,
        left_on: str = "_rowid",
        right_on: Optional[str] = None,
        schema=None,
    ) -> Tuple[FragmentMetadata, List[int]]:
        """
        Update existing columns in this fragment.

        This operation performs a left-outer-hash-join with the right table (new data)
        on the column specified by left_on and right_on. For every row in the current
        fragment, the updated column value is:

        1. If no matched row on the right side, the column value of the left side row.
        2. If there is exactly one corresponding row on the right side, the column value
           of the matching row.
        3. If there are multiple corresponding rows, the column value of a random row.

        Parameters
        ----------
        data_obj: Reader-like
            The data to be used for updating. Acceptable types are:
            - Pandas DataFrame, Pyarrow Table, Dataset, Scanner,
            Iterator[RecordBatch], or RecordBatchReader
            The data must include the join column (right_on) and the columns to update.
        left_on: str, default "_rowid"
            The name of the column in the fragment to join on.
        right_on: str or None
            The name of the column in data_obj to join on. If None, defaults to left_on.
        schema: pa.Schema, optional
            The schema of the data. If not specified, the schema will be inferred.

        Returns
        -------
        Tuple[FragmentMetadata, List[int]]
            A tuple of:
            - FragmentMetadata: The updated fragment metadata
            - List[int]: The list of field IDs that were modified

        Examples
        --------

        >>> import lance
        >>> import pyarrow as pa
        >>> # Create initial dataset
        >>> df = pa.table({
        ...     'id': [1, 2],
        ...     'name': ['a', 'b'],
        ...     'value': [10, 20]
        ... })
        >>> dataset = lance.write_dataset(df, "dataset")
        >>> # Prepare update data with _rowid (must be UInt64)
        >>> update_data = pa.table({
        ...     'id': [1],
        ...     'name': ['x'],
        ...     'value': [100]
        ... })
        >>> # Update the fragment
        >>> fragment = dataset.get_fragment(0)
        >>> updated_fragment, fields_modified = fragment.update_columns(
        ...     update_data,
        ...     left_on="id",
        ...     right_on="id"
        ... )
        >>> # Commit the changes
        >>> from lance import LanceOperation
        >>> op = LanceOperation.Update(
        ...     updated_fragments=[updated_fragment],
        ...     fields_modified=fields_modified,
        ... )
        >>> dataset = lance.LanceDataset.commit(
        ...     "dataset",
        ...     op,
        ...     read_version=dataset.version
        ... )

        See Also
        --------
        LanceFragment.merge :
            Merge new columns into this fragment (adds columns).
        LanceDataset.update :
            Update rows in the entire dataset.

        Notes
        -----
        - The columns to update must already exist in the fragment
        - The join column (left_on/right_on) will not be updated
        - Metadata columns (_rowid, _rowaddr) cannot be updated
        - This is a low-level API; for most use cases, use Dataset.update() instead
        """
        if right_on is None:
            right_on = left_on

        reader = _coerce_reader(data_obj, schema)
        metadata, fields_modified = self._fragment.update_columns(
            reader, left_on, right_on
        )
        return metadata, fields_modified

    def merge_columns(
        self,
        value_func: (
            Dict[str, str]
            | BatchUDF
            | ReaderLike
            | Callable[[pa.RecordBatch], pa.RecordBatch]
        ),
        columns: Optional[list[str]] = None,
        batch_size: Optional[int] = None,
        reader_schema: Optional[pa.Schema] = None,
    ) -> Tuple[FragmentMetadata, LanceSchema]:
        """Add columns to this Fragment.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        The parameters and their interpretation are the same as in the
        :meth:`lance.dataset.LanceDataset.add_columns` operation.

        The only difference is that, instead of modifying the dataset, a new
        fragment is created.  The new schema of the fragment is returned as well.
        These can be used in a later operation to commit the changes to the dataset.

        See Also
        --------
        lance.dataset.LanceOperation.Merge :
            The operation used to commit these changes to the dataset. See the
            doc page for an example of using this API.

        Returns
        -------
        Tuple[FragmentMetadata, LanceSchema]
            A new fragment with the added column(s) and the final schema.
        """
        transforms = normalize_transform(value_func, self, columns, reader_schema)

        if isinstance(transforms, BatchUDF):
            if transforms.cache is not None:
                raise ValueError(
                    "A checkpoint file cannot be used when applying a UDF with "
                    "LanceFragment.merge_columns.  You must apply your own "
                    "checkpointing for fragment-level operations."
                )

        if isinstance(transforms, pa.RecordBatchReader):
            metadata, schema = self._fragment.add_columns_from_reader(
                transforms, batch_size
            )
        else:
            metadata, schema = self._fragment.add_columns(
                transforms, columns, batch_size
            )

        return metadata, schema

    def delete(self, predicate: str) -> FragmentMetadata | None:
        """Delete rows from this Fragment.

        This will add or update the deletion file of this fragment. It does not
        modify or delete the data files of this fragment. If no rows are left after
        the deletion, this method will return None.

        .. warning::

            Internal API. This method is not intended to be used by end users.

        Parameters
        ----------
        predicate: str
            A SQL predicate that specifies the rows to delete.

        Returns
        -------
        FragmentMetadata or None
            A new fragment containing the new deletion file, or None if no rows left.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> tab = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> dataset = lance.write_dataset(tab, "dataset")
        >>> frag = dataset.get_fragment(0)
        >>> frag.delete("a > 1")
        FragmentMetadata(id=0, files=[DataFile(path='...', fields=[0, 1], ...), ...)
        >>> frag.delete("a > 0") is None
        True

        See Also
        --------
        lance.dataset.LanceOperation.Delete :
            The operation used to commit these changes to a dataset. See the
            doc page for an example of using this API.
        """
        raw_fragment = self._fragment.delete(predicate)
        if raw_fragment is None:
            return None
        return raw_fragment.metadata()

    @property
    def schema(self) -> pa.Schema:
        """Return the schema of this fragment."""

        return self._fragment.schema()

    def data_files(self):
        """Return the data files of this fragment."""

        return self._fragment.data_files()

    def deletion_file(self):
        """Return the deletion file, if any"""
        return self._fragment.deletion_file()

    @property
    def metadata(self) -> FragmentMetadata:
        """Return the metadata of this fragment.

        Returns
        -------
        FragmentMetadata
        """
        return self._fragment.metadata()


if TYPE_CHECKING:

    @overload
    def write_fragments(
        data: ReaderLike,
        dataset_uri: Union[str, Path, LanceDataset],
        schema: Optional[pa.Schema] = None,
        *,
        return_transaction: Literal[True],
        mode: str = "append",
        max_rows_per_file: int = 1024 * 1024,
        max_rows_per_group: int = 1024,
        max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
        progress: Optional[FragmentWriteProgress] = None,
        data_storage_version: Optional[str] = None,
        use_legacy_format: Optional[bool] = None,
        storage_options: Optional[Dict[str, str]] = None,
        enable_stable_row_ids: bool = False,
        target_bases: Optional[List[str]] = None,
        initial_bases: Optional[List["DatasetBasePath"]] = None,
        base_store_params: Optional[Dict[str, Dict[str, str]]] = None,
        external_blob_mode: Literal["reference", "ingest"] = "reference",
        allow_external_blob_outside_bases: bool = False,
        namespace_client: Optional[LanceNamespace] = None,
        table_id: Optional[List[str]] = None,
    ) -> Transaction: ...

    @overload
    def write_fragments(
        data: ReaderLike,
        dataset_uri: Union[str, Path, LanceDataset],
        schema: Optional[pa.Schema] = None,
        *,
        return_transaction: Literal[False] = False,
        mode: str = "append",
        max_rows_per_file: int = 1024 * 1024,
        max_rows_per_group: int = 1024,
        max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
        progress: Optional[FragmentWriteProgress] = None,
        data_storage_version: Optional[str] = None,
        use_legacy_format: Optional[bool] = None,
        storage_options: Optional[Dict[str, str]] = None,
        enable_stable_row_ids: bool = False,
        target_bases: Optional[List[str]] = None,
        initial_bases: Optional[List["DatasetBasePath"]] = None,
        base_store_params: Optional[Dict[str, Dict[str, str]]] = None,
        external_blob_mode: Literal["reference", "ingest"] = "reference",
        allow_external_blob_outside_bases: bool = False,
        namespace_client: Optional[LanceNamespace] = None,
        table_id: Optional[List[str]] = None,
    ) -> List[FragmentMetadata]: ...


def write_fragments(
    data: ReaderLike,
    dataset_uri: Union[str, Path, LanceDataset],
    schema: Optional[pa.Schema] = None,
    *,
    return_transaction: bool = False,
    mode: str = "append",
    max_rows_per_file: int = 1024 * 1024,
    max_rows_per_group: int = 1024,
    max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
    progress: Optional[FragmentWriteProgress] = None,
    data_storage_version: Optional[str] = None,
    use_legacy_format: Optional[bool] = None,
    storage_options: Optional[Dict[str, str]] = None,
    enable_stable_row_ids: bool = False,
    target_bases: Optional[List[str]] = None,
    initial_bases: Optional[List["DatasetBasePath"]] = None,
    base_store_params: Optional[Dict[str, Dict[str, str]]] = None,
    external_blob_mode: Literal["reference", "ingest"] = "reference",
    allow_external_blob_outside_bases: bool = False,
    namespace_client: Optional[LanceNamespace] = None,
    table_id: Optional[List[str]] = None,
) -> List[FragmentMetadata] | Transaction:
    """
    Write data into one or more fragments.

    .. warning::

        This is a low-level API intended for manually implementing distributed
        writes. For most users, :func:`lance.write_dataset` is the recommended API.

    Parameters
    ----------
    data : pa.Table or pa.RecordBatchReader
        The data to be written to the fragment.
    dataset_uri : str, Path, or LanceDataset
        The URI of the dataset or the dataset object.
    schema : pa.Schema, optional
        The schema of the data. If not specified, the schema will be inferred
        from the data.
    return_transaction: bool, default False
        If it's true, the transaction will be returned.
    mode : str, default "append"
        The write mode. If "append" is specified, the data will be checked
        against the existing dataset's schema. Otherwise, pass "create" or
        "overwrite" to assign new field ids to the schema.
    max_rows_per_file : int, default 1024 * 1024
        The maximum number of rows per data file.
    max_rows_per_group : int, default 1024
        The maximum number of rows per group in the data file.
    max_bytes_per_file : int, default 90 * 1024 * 1024 * 1024
        The max number of bytes to write before starting a new file. This is a
        soft limit. This limit is checked after each group is written, which
        means larger groups may cause this to be overshot meaningfully. This
        defaults to 90 GB, since we have a hard limit of 100 GB per file on
        object stores.
    progress : FragmentWriteProgress, optional
        *Experimental API*. Progress tracking for writing the fragment. Pass
        a custom class that defines hooks to be called when each fragment is
        starting to write and finishing writing.
    data_storage_version: optional, str, default None
        The version of the data storage format to use. Newer versions are more
        efficient but require newer versions of lance to read.  The default (None)
        will use the 2.0 version.  See the user guide for more details.
    use_legacy_format : optional, bool, default None
        Deprecated method for setting the data storage version. Use the
        `data_storage_version` parameter instead.
    storage_options : Optional[Dict[str, str]]
        Extra options that make sense for a particular storage connection. This is
        used to store connection parameters like credentials, endpoint, etc.
    enable_stable_row_ids: bool
        Experimental: if set to true, the writer will use stable row ids.
        These row ids are stable after compaction operations, but not after updates.
        This makes compaction more efficient, since with stable row ids no
        secondary indices need to be updated to point to new row ids.
    target_bases : list of str, optional
        References to base paths where data should be written. Can be
        specified in all modes.

        Each string is resolved by trying to match:
        1. Base name (e.g., "primary", "archive") from registered bases
        2. Base path URI (e.g., "s3://bucket1/data")

        **CREATE mode**: References must match bases in `initial_bases`
        **APPEND/OVERWRITE modes**: References must match bases in the
        existing manifest
    initial_bases : list of DatasetBasePath, optional
        Base paths to register when creating a new dataset (CREATE mode only).

        This allows `target_bases` references to be resolved during fragment
        writing. Example:

        >>> from lance import DatasetBasePath
        >>> initial_bases = [DatasetBasePath(path="s3://bucket1/data", name="base1")]

        **Only valid in CREATE mode**. Will raise an error if used with
        APPEND/OVERWRITE modes.
    base_store_params : dict of str to dict, optional
        Runtime-only object store parameters keyed by exact base path URI.
        Each value is a dict of storage options for that base. These settings
        are not persisted to the manifest. When a base has no explicit entry,
        top-level ``storage_options`` is used as a fallback. If ``dataset_uri``
        is a LanceDataset and this is omitted, the dataset's base store params
        are inherited.
    external_blob_mode: {"reference", "ingest"}, default "reference"
        How external blob URIs are handled on write.

        - ``"reference"`` stores the URI as an external blob reference.
        - ``"ingest"`` reads the external bytes during write and stores them in
          Lance-managed storage using the normal inline / packed / dedicated
          thresholds.
    allow_external_blob_outside_bases: bool, default False
        If False, external blob URIs must map to a registered non-dataset-root
        base path. If True, external blob URIs outside registered bases are
        allowed. Only valid when ``external_blob_mode="reference"``. Setting
        this to True with ``"ingest"`` mode raises an error.
    namespace_client : optional, LanceNamespace
        A namespace client for automatic credential refresh. When provided with
        `table_id`, a storage options provider will be created automatically to
        refresh credentials via the namespace. Must be provided together with
        `table_id`. The caller should provide initial/merged storage options via
        the `storage_options` parameter.
    table_id : optional, List[str]
        The table identifier when using a namespace (e.g., ["my_table"]).
        Must be provided together with `namespace_client`.

    Returns
    -------
    List[FragmentMetadata] | Transaction
        If return_transaction is False:
            a list of :class:`FragmentMetadata` for the fragments written. The
            fragment ids are left as zero meaning they are not yet specified. They
            will be assigned when the fragments are committed to a dataset.

        If return_transaction is True:
            The write transaction. The type of transaction will correspond to
            the mode parameter specified. This transaction can be passed to
            :meth:`LanceDataset.commit`.

    """
    from .dataset import LanceDataset

    # Validate namespace_client and table_id are provided together
    if namespace_client is not None and table_id is None:
        raise ValueError(
            "Both 'namespace_client' and 'table_id' must be provided together."
        )
    elif table_id is not None and namespace_client is None:
        raise ValueError(
            "Both 'namespace_client' and 'table_id' must be provided together."
        )

    reader = _coerce_reader(data, schema)

    if isinstance(dataset_uri, Path):
        dataset_uri = str(dataset_uri)
    elif isinstance(dataset_uri, LanceDataset):
        if base_store_params is None:
            base_store_params = dataset_uri._base_store_params
        if storage_options is None:
            storage_options = dataset_uri._storage_options
        dataset_uri = dataset_uri._ds
    elif not isinstance(dataset_uri, str):
        raise TypeError(f"Unknown dataset_uri type {type(dataset_uri)}")

    if use_legacy_format is not None:
        warnings.warn(
            "use_legacy_format is deprecated, use data_storage_version instead",
            DeprecationWarning,
        )
        if use_legacy_format:
            data_storage_version = "legacy"
        else:
            data_storage_version = "stable"

    function = _write_fragments_transaction if return_transaction else _write_fragments

    return function(
        dataset_uri,
        reader,
        mode=mode,
        max_rows_per_file=max_rows_per_file,
        max_rows_per_group=max_rows_per_group,
        max_bytes_per_file=max_bytes_per_file,
        progress=progress,
        data_storage_version=data_storage_version,
        storage_options=storage_options,
        namespace_client=namespace_client,
        table_id=table_id,
        enable_stable_row_ids=enable_stable_row_ids,
        target_bases=target_bases,
        initial_bases=initial_bases,
        base_store_params=base_store_params,
        external_blob_mode=external_blob_mode,
        allow_external_blob_outside_bases=allow_external_blob_outside_bases,
    )


class FragmentSession:
    def __init__(self, session):
        self._session = session

    def take(self, indices):
        """
        Take rows from this fragment based on the offset in the fragment.

        Parameters
        ----------
        indices : Array or array-like
            indices of rows to select in the fragment.

        Returns
        -------
        table : pyarrow.Table
        """
        return pa.Table.from_batches([self._session.take(indices)])
