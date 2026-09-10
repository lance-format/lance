# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import pyarrow as pa

from .lance import (
    LanceBufferDescriptor,
    LanceColumnMetadata,
    LanceFileMetadata,
    LanceFileStatistics,
    LancePageMetadata,
    stable_version,
)
from .lance import (
    LanceFileReader as _LanceFileReader,
)
from .lance import (
    LanceFileSession as _LanceFileSession,
)
from .lance import (
    LanceFileWriter as _LanceFileWriter,
)

if TYPE_CHECKING:
    from .blob import DedicatedBlobWriter, PackedBlobWriter
    from .namespace import LanceNamespace


class ReaderResults:
    """
    Utility class for converting results from Lance's internal
    format (RecordBatchReader) to a desired format such
    as a pyarrow Table, etc.
    """

    def __init__(self, reader: pa.RecordBatchReader):
        """
        Creates a new instance, not meant for external use
        """
        self.reader = reader

    def to_batches(self) -> pa.RecordBatchReader:
        """
        Return the results as a pyarrow RecordBatchReader
        """
        return self.reader

    def to_table(self) -> pa.Table:
        """
        Return the results as a pyarrow Table
        """
        return self.reader.read_all()


class LanceFileReader:
    """
    A file reader for reading Lance files

    This class is used to read Lance data files, a low level structure
    optimized for storing multi-modal tabular data.  If you are working with
    Lance datasets then you should use the LanceDataset class instead.
    """

    def __init__(
        self,
        path: str,
        storage_options: Optional[Dict[str, str]] = None,
        columns: Optional[List[str]] = None,
        *,
        namespace_client: Optional["LanceNamespace"] = None,
        table_id: Optional[List[str]] = None,
        _inner_reader: Optional[_LanceFileReader] = None,
    ):
        """
        Creates a new file reader to read the given file

        Parameters
        ----------

        path: str
            The path to read, can be a pathname for local storage
            or a URI to read from cloud storage.
        storage_options : optional, dict
            Extra options to be used for a particular storage connection. This is
            used to store connection parameters like credentials, endpoint, etc.
        namespace_client : optional, LanceNamespace
            A namespace client for automatic credential refresh.
            Must be provided together with table_id.
        table_id : optional, List[str]
            The table identifier within the namespace.
            Must be provided together with namespace_client.
        columns: list of str, default None
            List of column names to be fetched.
            All columns are fetched if None or unspecified.
        """
        if _inner_reader is not None:
            self._reader = _inner_reader
        else:
            if isinstance(path, Path):
                path = str(path)
            self._reader = _LanceFileReader(
                path,
                storage_options=storage_options,
                namespace_client=namespace_client,
                table_id=table_id,
                columns=columns,
            )

    def read_all(self, *, batch_size: int = 1024, batch_readahead=16) -> ReaderResults:
        """
        Reads the entire file

        Parameters
        ----------
        batch_size: int, default 1024
            The file will be read in batches.  This parameter controls
            how many rows will be in each batch (except the final batch)

            Smaller batches will use less memory but might be slightly
            slower because there is more per-batch overhead
        """
        return ReaderResults(self._reader.read_all(batch_size, batch_readahead))

    def read_range(
        self, start: int, num_rows: int, *, batch_size: int = 1024, batch_readahead=16
    ) -> ReaderResults:
        """
        Read a range of rows from the file

        Parameters
        ----------
        start: int
            The offset of the first row to start reading
        num_rows: int
            The number of rows to read from the file
        batch_size: int, default 1024
            The file will be read in batches.  This parameter controls
            how many rows will be in each batch (except the final batch)

            Smaller batches will use less memory but might be slightly
            slower because there is more per-batch overhead
        """
        return ReaderResults(
            self._reader.read_range(start, num_rows, batch_size, batch_readahead)
        )

    def take_rows(
        self, indices, *, batch_size: int = 1024, batch_readahead=16
    ) -> ReaderResults:
        """
        Read a specific set of rows from the file

        Parameters
        ----------
        indices: List[int]
            The indices of the rows to read from the file in ascending order
        batch_size: int, default 1024
            The file will be read in batches.  This parameter controls
            how many rows will be in each batch (except the final batch)

            Smaller batches will use less memory but might be slightly
            slower because there is more per-batch overhead
        """
        for i in range(len(indices) - 1):
            if indices[i] > indices[i + 1]:
                raise ValueError(
                    f"Indices must be sorted in ascending order for \
                                 file API, got {indices[i]} > {indices[i + 1]}"
                )

        return ReaderResults(
            self._reader.take_rows(indices, batch_size, batch_readahead)
        )

    def metadata(self) -> LanceFileMetadata:
        """
        Return metadata describing the file contents
        """
        return self._reader.metadata()

    def file_statistics(self) -> LanceFileStatistics:
        """
        Return file statistics of the file
        """
        return self._reader.file_statistics()

    def read_global_buffer(self, index: int) -> bytes:
        """
        Read a global buffer from the file at a given index

        Parameters
        ----------
        index: int
            The index of the global buffer to read

        Returns
        -------
        bytes
            The contents of the global buffer
        """
        return self._reader.read_global_buffer(index)

    def num_rows(self) -> int:
        """Return the number of rows belonging to the data file."""
        return self._reader.num_rows()


@dataclass
class ListResult:
    """
    Result of a non-recursive, delimited list (see
    :meth:`LanceFileSession.list_with_delimiter`).

    Attributes
    ----------
    common_prefixes : List[str]
        The immediate child "directories" of the listed path, relative to the
        session's base path.
    objects : List[str]
        The immediate child files of the listed path, relative to the session's
        base path.
    """

    common_prefixes: List[str]
    objects: List[str]


class LanceFileSession:
    """
    A file session for reading and writing Lance files.

    If you plan on opening many readers or writers then creating a session first can
    be more efficient as it will share the underlying object_store configuration with
    all of the readers and writers.
    """

    def __init__(
        self,
        base_path: str,
        storage_options: Optional[Dict[str, str]] = None,
        namespace_client: Optional["LanceNamespace"] = None,
        table_id: Optional[List[str]] = None,
    ):
        """
        Creates a new file session

        Parameters
        ----------
        base_path: str
            The base path to read from.  Can be a pathname for local storage
            or a URI to read from cloud storage.  All readers will be opened relative
            to this base path.
        storage_options : optional, dict
            Extra options to be used for a particular storage connection. This is
            used to store connection parameters like credentials, endpoint, etc.
        namespace_client : optional, LanceNamespace
            A namespace client for automatic credential refresh.
            Must be provided together with table_id.
        table_id : optional, List[str]
            The table identifier within the namespace.
            Must be provided together with namespace_client.
        """
        if isinstance(base_path, Path):
            base_path = str(base_path)
        self._session = _LanceFileSession(
            base_path,
            storage_options=storage_options,
            namespace_client=namespace_client,
            table_id=table_id,
        )

    def open_reader(
        self, path: str, columns: Optional[List[str]] = None
    ) -> LanceFileReader:
        """
        Opens a new reader for the given path

        The path will be appended to the base path of the session.
        """
        return LanceFileReader(
            None,  # pyright: ignore[reportArgumentType]
            _inner_reader=self._session.open_reader(path, columns),
        )

    def open_writer(
        self,
        path: str,
        *,
        schema: Optional[pa.Schema] = None,
        data_cache_bytes: Optional[int] = None,
        version: Optional[str] = None,
        keep_original_array: Optional[bool] = None,
        max_page_bytes: Optional[int] = None,
    ) -> "LanceFileWriter":
        """
        Opens a new writer for the given path (relative to this session's base path),
        reusing the session's underlying object store.

        Parameters
        ----------
        path : str
            Path relative to `base_path` where the file will be written.
        schema : pyarrow.Schema, optional
            If provided, creates a schema-bound writer; otherwise a lazy writer is
            created.
        data_cache_bytes : int, optional
            Size of the row-group/page write cache in bytes.
        version : str, optional
            Lance file format version (e.g. "2"). Parsed by the Rust layer.
        keep_original_array : bool, optional
            If True, retain original arrays in the writer (advanced/diagnostic).
        max_page_bytes : int, optional
            Target max page size in bytes.

        Returns
        -------
        LanceFileWriter
        """
        inner = self._session.open_writer(
            path,
            schema,  # pyarrow.Schema or None
            data_cache_bytes,
            version,
            keep_original_array,
            max_page_bytes,
        )
        return LanceFileWriter(
            None,  # pyright: ignore[reportArgumentType]
            _inner_writer=inner,
        )

    def open_packed_blob_writer(self, path: str, blob_id: int) -> "PackedBlobWriter":
        """
        Opens a packed blob writer for the given data file path.

        The path will be appended to the base path of the session.
        """
        return self._session.open_packed_blob_writer(path, blob_id)

    def open_dedicated_blob_writer(
        self, path: str, blob_id: int
    ) -> "DedicatedBlobWriter":
        """
        Opens a dedicated blob writer for the given data file path.

        The path will be appended to the base path of the session.
        """
        return self._session.open_dedicated_blob_writer(path, blob_id)

    def contains(self, path: str) -> bool:
        """
        Check if a file exists at the given path (relative to this session's base path).

        Parameters
        ----------
        path : str
            Path relative to `base_path` to check for existence.

        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """
        return self._session.contains(path)

    def list(self, path: Optional[str] = None) -> List[str]:
        """
        List all files at the given path (relative to this session's base path).

        Parameters
        ----------
        path : str, optional
            Path relative to `base_path` to list files from. If None, lists files
            from the base path.

        Returns
        -------
        List[str]
            List of file paths.
        """
        return self._session.list(path)

    def list_with_delimiter(self, path: Optional[str] = None) -> ListResult:
        """
        Non-recursively list a single directory level (relative to this
        session's base path).

        Unlike :meth:`list`, which recurses into the entire subtree, this
        returns only the immediate children of ``path``: the child
        "directories" as ``common_prefixes`` and the direct child files as
        ``objects``.

        Parameters
        ----------
        path : str, optional
            Path relative to `base_path` to list. If None, lists the base path.

        Returns
        -------
        ListResult
            The immediate child prefixes and objects of `path`.
        """
        common_prefixes, objects = self._session.list_with_delimiter(path)
        return ListResult(common_prefixes=common_prefixes, objects=objects)

    def read_range(self, path: str, offset: int, length: int) -> bytes:
        """
        Read a byte range from a file (relative to this session's base path).

        Issues a single ranged read. Reading a missing object raises
        ``OSError``, consistent with ``download_file``.

        Parameters
        ----------
        path : str
            Path relative to `base_path` to read from.
        offset : int
            Byte offset at which to start reading.
        length : int
            Number of bytes to read.

        Returns
        -------
        bytes
            The requested byte range.
        """
        return self._session.read_range(path, offset, length)

    def delete_file(self, path: str) -> None:
        """
        Delete a file (relative to this session's base path).

        Deleting a path that does not exist raises ``OSError``, consistent with
        ``download_file``.

        Parameters
        ----------
        path : str
            Path relative to `base_path` to delete.
        """
        self._session.delete_file(path)

    def upload_file(self, local_path: Union[str, Path], remote_path: str) -> None:
        """
        Upload a file from local filesystem to the object store.

        Parameters
        ----------
        local_path : str or Path
            Local file path to upload.
        remote_path : str
            Remote path relative to session's base_path.
        """
        if isinstance(local_path, Path):
            local_path = str(local_path)
        self._session.upload_file(local_path, remote_path)

    def download_file(self, remote_path: str, local_path: Union[str, Path]) -> None:
        """
        Download a file from object store to local filesystem.

        Parameters
        ----------
        remote_path : str
            Remote path relative to session's base_path.
        local_path : str or Path
            Local file path where the file will be saved.
        """
        if isinstance(local_path, Path):
            local_path = str(local_path)
        self._session.download_file(remote_path, local_path)


class LanceFileWriter:
    """
    A file writer for writing Lance data files

    This class is used to write Lance data files, a low level structure
    optimized for storing multi-modal tabular data.  If you are working with
    Lance datasets then you should use the LanceDataset class instead.

    Attributes
    ----------
    size_bytes: Optional[int]
        The final size of the file in bytes.  This is None until `close` is
        called.
    """

    def __init__(
        self,
        path: str,
        schema: Optional[pa.Schema] = None,
        *,
        data_cache_bytes: Optional[int] = None,
        version: Optional[str] = None,
        storage_options: Optional[Dict[str, str]] = None,
        namespace_client: Optional["LanceNamespace"] = None,
        table_id: Optional[List[str]] = None,
        max_page_bytes: Optional[int] = None,
        _inner_writer: Optional[_LanceFileWriter] = None,
        **kwargs,
    ):
        """
        Create a new LanceFileWriter to write to the given path

        Parameters
        ----------
        path: str
            The path to write to.  Can be a pathname for local storage
            or a URI for remote storage.
        schema: pa.Schema
            The schema of data that will be written.  If not specified then
            the schema will be inferred from the first batch.  If the schema
            is not specified and no data is written then the write will fail.
        data_cache_bytes: int
            How many bytes (per column) to cache before writing a page.  The
            default is an appropriate value based on the filesystem.
        version: str
            The version of the file format to write.  If not specified then
            the latest stable version will be used.  Newer versions are more
            efficient but may not be readable by older versions of the software.
        storage_options : optional, dict
            Extra options to be used for a particular storage connection. This is
            used to store connection parameters like credentials, endpoint, etc.
        namespace_client : optional, LanceNamespace
            A namespace client for automatic credential refresh.
            Must be provided together with table_id.
        table_id : optional, List[str]
            The table identifier within the namespace.
            Must be provided together with namespace_client.
        max_page_bytes : optional, int
            The maximum size of a page in bytes, if a single array would create a
            page larger than this then it will be split into multiple pages. The
            default value is 32MB.
        """
        if _inner_writer is not None:
            self._writer = _inner_writer
        else:
            if isinstance(path, Path):
                path = str(path)
            self._writer = _LanceFileWriter(
                path,
                schema,
                data_cache_bytes=data_cache_bytes,
                version=version,
                storage_options=storage_options,
                namespace_client=namespace_client,
                table_id=table_id,
                max_page_bytes=max_page_bytes,
                **kwargs,
            )
        self.closed = False
        self.size_bytes: Optional[int] = None

    def write_batch(self, batch: Union[pa.RecordBatch, pa.Table]) -> None:
        """
        Write a batch of data to the file

        parameters
        ----------
        batch: Union[pa.RecordBatch, pa.Table]
            The data to write to the file
        """
        if isinstance(batch, pa.Table):
            for batch in batch.to_batches():
                self._writer.write_batch(batch)
        else:
            self._writer.write_batch(batch)

    def close(self) -> Optional[int]:
        """
        Write the file metadata and close the file

        Returns the number of rows written to the file

        After this returns, ``size_bytes`` holds the final size of the file.  This
        is reported by the writer itself, so it is available for object stores
        without issuing a separate metadata request.
        """
        if self.closed:
            return
        self.closed = True
        summary = self._writer.finish()
        self.size_bytes = summary.size_bytes
        return summary.num_rows

    def add_schema_metadata(self, key: str, value: str) -> None:
        """
        Add a metadata (key/value pair) entry to the schema. This method allows you to
        alter the schema metadata. It must be called before `close` is called.

        Parameters
        ----------
        key: str
            The key to add.
        value: str
            The value to add.
        """
        self._writer.add_schema_metadata(key, value)

    def add_global_buffer(self, data: bytes) -> int:
        """
        Add a global buffer to the file. The global buffer can contain any
        arbitrary bytes.

        Parameters
        ----------
        data: bytes
            The data to write to the file.

        Returns
        -------
        int
            The index of the global buffer. This will always start at 1
            and increment by 1 each time this method is called.
        """
        return self._writer.add_global_buffer(data)

    def __enter__(self) -> "LanceFileWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


__all__ = [
    "LanceFileReader",
    "LanceFileWriter",
    "LanceFileMetadata",
    "LanceColumnMetadata",
    "LancePageMetadata",
    "LanceBufferDescriptor",
    "LanceFileStatistics",
    "stable_version",
]
