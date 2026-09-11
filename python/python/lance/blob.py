# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import ctypes
import io
from dataclasses import dataclass
from typing import IO, Any, Iterator, Optional, Union

import pyarrow as pa

from .lance import (
    BlobDescriptor as BlobDescriptor,
)
from .lance import (
    BlobDescriptorArrayBuilder as BlobDescriptorArrayBuilder,
)
from .lance import (
    DedicatedBlobWriter as DedicatedBlobWriter,
)
from .lance import (
    LanceBlobFile,
)
from .lance import (
    PackedBlobWriter as PackedBlobWriter,
)

_BLOB_INLINE_SIZE_THRESHOLD_META_KEY = b"lance-encoding:blob-inline-size-threshold"
_BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY = (
    b"lance-encoding:blob-dedicated-size-threshold"
)
_BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY = (
    b"lance-encoding:blob-pack-file-size-threshold"
)
_MAX_RUST_USIZE = ctypes.c_size_t(-1).value


@dataclass(frozen=True)
class Blob:
    """
    A logical blob value for writing Lance blob columns.

    A blob can be represented as:
    - inline bytes
    - an external URI, optionally with a non-empty range

    Every blob must use exactly one representation. Use ``None`` for a null
    blob and :meth:`empty` for a valid empty blob.
    """

    data: Optional[bytes] = None
    uri: Optional[str] = None
    position: Optional[int] = None
    size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.data is not None and self.uri is not None:
            raise ValueError("Blob cannot have both data and uri")
        if self.uri == "":
            raise ValueError("Blob uri cannot be empty")
        if (self.position is not None or self.size is not None) and self.uri is None:
            raise ValueError("External packed blob must have a uri")
        if (self.position is None) != (self.size is None):
            raise ValueError(
                "External blob must set both position and size, or neither"
            )
        if self.data is not None and self.position is not None:
            raise ValueError(
                "Blob cannot have both inline data and external slice metadata"
            )
        if self.data is None and self.uri is None:
            raise ValueError("Blob must set `data` or `uri`; use None for a null blob")
        if self.size == 0:
            raise ValueError("External blob range size must be greater than zero")

    @staticmethod
    def from_bytes(data: Union[bytes, bytearray, memoryview]) -> "Blob":
        return Blob(data=bytes(data))

    @staticmethod
    def from_uri(
        uri: str, position: Optional[int] = None, size: Optional[int] = None
    ) -> "Blob":
        if uri == "":
            raise ValueError("Blob uri cannot be empty")
        if (position is not None and position < 0) or (size is not None and size < 0):
            raise ValueError("External blob position and size must be non-negative")
        return Blob(uri=uri, position=position, size=size)

    @staticmethod
    def empty() -> "Blob":
        return Blob(data=b"")


class BlobType(pa.ExtensionType):
    """
    A PyArrow extension type for Lance blob columns.

    This is the "logical" type users write. Lance will store it in a compact
    descriptor format, and reads will return descriptors by default. Its storage
    type defaults to ``Struct<data: LargeBinary?, uri: Utf8?, position: UInt64?,
    size: UInt64?>``. Arrow deserialization also preserves the accepted minimal
    ``Struct<data: LargeBinary?, uri: Utf8?>`` storage type. ``position`` and
    ``size`` select a range within an external ``uri`` and must either both be set
    or both be null. When set, ``size`` must be greater than zero. Every non-null
    value must set exactly one of ``data`` and ``uri``.
    """

    def __init__(self) -> None:
        storage_type = pa.struct(
            [
                pa.field("data", pa.large_binary(), nullable=True),
                pa.field("uri", pa.utf8(), nullable=True),
                pa.field("position", pa.uint64(), nullable=True),
                pa.field("size", pa.uint64(), nullable=True),
            ]
        )
        pa.ExtensionType.__init__(self, storage_type, "lance.blob.v2")

    def __arrow_ext_serialize__(self) -> bytes:
        return b""

    @staticmethod
    def _validate_storage_type(storage_type: pa.DataType) -> None:
        if not pa.types.is_struct(storage_type):
            raise TypeError("BlobType storage type must be a struct")

        fields = list(storage_type)
        if len(fields) not in (2, 4):
            raise TypeError(
                "BlobType storage struct must contain either data/uri or "
                "data/uri/position/size"
            )

        expected_fields = [
            ("data", pa.large_binary()),
            ("uri", pa.utf8()),
            ("position", pa.uint64()),
            ("size", pa.uint64()),
        ]
        for index, field in enumerate(fields):
            expected_name, expected_type = expected_fields[index]
            if field.name != expected_name or field.type != expected_type:
                raise TypeError(
                    "BlobType storage field "
                    f"{index} must be {expected_name}: {expected_type}, got "
                    f"{field.name}: {field.type}"
                )
            if index < 2 and not field.nullable:
                raise TypeError(f"BlobType storage field {field.name} must be nullable")

    @classmethod
    def _from_storage_type(cls, storage_type: pa.DataType) -> "BlobType":
        cls._validate_storage_type(storage_type)
        instance = cls.__new__(cls)
        pa.ExtensionType.__init__(instance, storage_type, "lance.blob.v2")
        return instance

    @classmethod
    def __arrow_ext_deserialize__(
        cls, storage_type: pa.DataType, serialized: bytes
    ) -> "BlobType":
        return cls._from_storage_type(storage_type)

    def __arrow_ext_class__(self):
        return BlobArray

    def __reduce__(self):
        # Workaround to ensure pickle works in earlier versions of PyArrow
        # https://github.com/apache/arrow/issues/35599
        return type(self).__arrow_ext_deserialize__, (
            self.storage_type,
            self.__arrow_ext_serialize__(),
        )


try:
    pa.register_extension_type(BlobType())
except pa.ArrowKeyError:
    # Already registered in this interpreter.
    pass


class BlobArray(pa.ExtensionArray):
    """
    A PyArrow extension array for Lance blob columns.

    Construct with :meth:`from_pylist` or use :func:`blob_array`.
    """

    @classmethod
    def from_pylist(cls, values: list[Any]) -> "BlobArray":
        data_values: list[Optional[bytes]] = []
        uri_values: list[Optional[str]] = []
        position_values: list[Optional[int]] = []
        size_values: list[Optional[int]] = []
        null_mask: list[bool] = []

        for v in values:
            if v is None:
                data_values.append(None)
                uri_values.append(None)
                position_values.append(None)
                size_values.append(None)
                null_mask.append(True)
                continue

            if isinstance(v, Blob):
                data_values.append(v.data)
                uri_values.append(v.uri)
                position_values.append(v.position)
                size_values.append(v.size)
                null_mask.append(False)
                continue

            if isinstance(v, str):
                if v == "":
                    raise ValueError("Blob uri cannot be empty")
                data_values.append(None)
                uri_values.append(v)
                position_values.append(None)
                size_values.append(None)
                null_mask.append(False)
                continue

            if isinstance(v, (bytes, bytearray, memoryview)):
                data_values.append(bytes(v))
                uri_values.append(None)
                position_values.append(None)
                size_values.append(None)
                null_mask.append(False)
                continue

            raise TypeError(
                "BlobArray values must be bytes-like, str (URI), Blob, or None; "
                f"got {type(v)}"
            )

        data_arr = pa.array(data_values, type=pa.large_binary())
        uri_arr = pa.array(uri_values, type=pa.utf8())
        position_arr = pa.array(position_values, type=pa.uint64())
        size_arr = pa.array(size_values, type=pa.uint64())
        mask_arr = pa.array(null_mask, type=pa.bool_())
        storage = pa.StructArray.from_arrays(
            [data_arr, uri_arr, position_arr, size_arr],
            names=["data", "uri", "position", "size"],
            mask=mask_arr,
        )
        return pa.ExtensionArray.from_storage(BlobType(), storage)  # type: ignore[return-value]


def blob_array(values: list[Any]) -> BlobArray:
    """
    Construct a blob array from Python values.

    Each value must be one of:
    - bytes-like: inline bytes
    - str: an external URI
    - Blob: explicit inline/uri/empty
    - None: null
    """

    return BlobArray.from_pylist(values)


def _validate_threshold(name: str, value: Optional[int], *, allow_zero: bool) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    elif value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > _MAX_RUST_USIZE:
        raise OverflowError(f"{name} must fit in a Rust usize")


def blob_field(
    name: str,
    *,
    nullable: bool = True,
    inline_size_threshold: Optional[int] = None,
    dedicated_size_threshold: Optional[int] = None,
    pack_file_size_threshold: Optional[int] = None,
) -> pa.Field:
    """
    Construct an Arrow field for a Lance blob column.

    The returned field uses the complete logical blob shape
    ``Struct<data: LargeBinary?, uri: Utf8?, position: UInt64?, size: UInt64?>``.
    Every non-null value must set exactly one of ``data`` and ``uri``. External
    ranges must set both ``position`` and a positive ``size``. Lance preserves
    this logical schema across create, append, and merge-insert writes while
    storing compact descriptors internally.

    Parameters
    ----------
    name : str
        Field name.
    nullable : bool, default True
        Whether the blob column accepts null values.
    inline_size_threshold : optional, int
        Maximum payload size in bytes to keep inline in the data file before
        using packed blob storage.
    dedicated_size_threshold : optional, int
        Maximum payload size in bytes to store in packed blob storage before
        using dedicated blob storage. This threshold is checked before
        ``inline_size_threshold``.
    pack_file_size_threshold : optional, int
        Maximum size in bytes of a single packed blob sidecar (``.pack``) file.
        Once a sidecar reaches this size a new one is started.
    """
    _validate_threshold("inline_size_threshold", inline_size_threshold, allow_zero=True)
    _validate_threshold(
        "dedicated_size_threshold", dedicated_size_threshold, allow_zero=False
    )
    _validate_threshold(
        "pack_file_size_threshold", pack_file_size_threshold, allow_zero=False
    )

    field = pa.field(name, BlobType(), nullable=nullable)
    if (
        inline_size_threshold is None
        and dedicated_size_threshold is None
        and pack_file_size_threshold is None
    ):
        return field

    metadata = dict(field.metadata or {})
    if inline_size_threshold is not None:
        metadata[_BLOB_INLINE_SIZE_THRESHOLD_META_KEY] = str(
            inline_size_threshold
        ).encode()
    if dedicated_size_threshold is not None:
        metadata[_BLOB_DEDICATED_SIZE_THRESHOLD_META_KEY] = str(
            dedicated_size_threshold
        ).encode()
    if pack_file_size_threshold is not None:
        metadata[_BLOB_PACK_FILE_SIZE_THRESHOLD_META_KEY] = str(
            pack_file_size_threshold
        ).encode()
    return field.with_metadata(metadata)


class BlobIterator:
    def __init__(self, binary_iter: Iterator[pa.BinaryScalar]):
        self.binary_iter = binary_iter

    def __next__(self) -> Optional[IO[bytes]]:
        value = next(self.binary_iter)
        if value is None:
            return None
        return io.BytesIO(value.as_py())


class BlobColumn:
    """
    A utility to wrap a Pyarrow binary column and iterate over the rows as
    file-like objects.

    This can be useful for working with medium-to-small binary objects that need
    to interface with APIs that expect file-like objects. For very large binary
    objects (4-8MB or more per value) you might be better off creating a blob
    column. Use :py:meth:`lance.Dataset.read_blobs` when you need complete blob
    bytes, or :py:meth:`lance.Dataset.take_blobs` when you need lazy file-like
    access.
    """

    def __init__(self, blob_column: Union[pa.Array, pa.ChunkedArray]):
        if not isinstance(blob_column, (pa.Array, pa.ChunkedArray)):
            raise ValueError(
                "Expected a pyarrow.Array or pyarrow.ChunkedArray, "
                f"got {type(blob_column)}"
            )

        if not pa.types.is_large_binary(blob_column.type) and not pa.types.is_binary(
            blob_column.type
        ):
            raise ValueError(f"Expected a binary array, got {blob_column.type}")

        self.blob_column = blob_column

    def __iter__(self) -> Iterator[IO[bytes]]:
        return BlobIterator(iter(self.blob_column))


class BlobFile(io.RawIOBase):
    """Represents a blob in a Lance dataset as a file-like object."""

    def __init__(self, inner: LanceBlobFile):
        """
        Internal only:  To obtain a BlobFile use
        :py:meth:`lance.dataset.Dataset.take_blobs`.
        """
        self.inner = inner

    ## Note: most methods undocumented since they are defined by
    ## the base class.
    def close(self) -> None:
        self.inner.close()

    @property
    def closed(self) -> bool:
        return self.inner.is_closed()

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.inner.seek(offset)
        elif whence == io.SEEK_CUR:
            self.inner.seek(self.inner.tell() + offset)
        elif whence == io.SEEK_END:
            self.inner.seek(self.inner.size() + offset)
        else:
            raise ValueError(f"Invalid whence: {whence}")

        return self.inner.tell()

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.inner.tell()

    def size(self) -> int:
        """
        Returns the size of the blob in bytes.
        """
        return self.inner.size()

    def readall(self) -> bytes:
        return self.inner.readall()

    def read_range(self, offset: int, length: int) -> bytes:
        """Read a blob-local byte range without changing the current cursor."""
        return self.inner.read_range(offset, length)

    def read_ranges(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        """
        Read multiple blob-local byte ranges without changing the current cursor.

        Each range is an ``(offset, length)`` pair, matching
        :py:meth:`read_range`. The underlying physical reads may be reordered,
        coalesced, or split for efficiency. For every range, offset plus length
        must fit in an unsigned 64-bit integer and must not extend beyond the
        blob size.

        Parameters
        ----------
        ranges : List[Tuple[int, int]]
            The ``(offset, length)`` byte ranges to read.

        Returns
        -------
        data : List[bytes]
            One payload per requested range, in input order.
        """
        return self.inner.read_ranges(ranges)

    def readinto(self, b: bytearray) -> int:
        return self.inner.read_into(b)

    def __repr__(self) -> str:
        return f"<BlobFile size={self.size()}>"
