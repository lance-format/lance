# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import Iterable, Iterator, Literal, Optional, Union

import pyarrow as pa

class DeletionFile:
    """
    Metadata for a deletion file.

    The deletion file contains the row ids that are tombstoned.

    Attributes
    ----------
    read_version : int
        The read version of the deletion file.
    id : int
        A unique identifier for the deletion file, used to prevent collisions.
    num_deleted_rows : int
        The number of rows that are deleted.
    file_type : str
        The type of deletion file. "array" is used for small sets, while
        "bitmap" is used for large sets.
    """

    read_version: int
    id: int
    num_deleted_rows: int
    file_type: Literal["array", "bitmap"]
    base_id: Optional[int]

    def __init__(
        self,
        read_version: int,
        id: int,
        file_type: Literal["array", "bitmap"],
        num_deleted_rows: int,
        base_id: Optional[int] = None,
    ): ...
    def asdict(self) -> dict:
        """Get a dictionary representation of the deletion file."""
        ...
    def path(self, fragment_id: int, base_uri: Optional[str] = None) -> str:
        """
        Get the path to the deletion file.

        Parameters
        ----------
        fragment_id : int
            The fragment id.
        base_uri : str, optional
            The base URI to use for the path. If not provided, a relative path
            is returned.

        Returns
        -------
        str
            The path to the deletion file.
        """
        ...

    def json(self) -> str:
        """Get a JSON representation of the deletion file.

        Returns
        -------
        str

        Warning
        -------
        The JSON representation is not guaranteed to be stable across versions.
        """
        ...

    @classmethod
    def from_json(json: str) -> DeletionFile:
        """
        Load a deletion file from a JSON representation.

        Parameters
        ----------
        json : str
            The JSON representation of the deletion file.

        Returns
        -------
        DeletionFile
        """
        ...

    def __reduce__(self) -> tuple: ...

class RowIdMeta:
    def json(self) -> str:
        """Get a JSON representation of the row id metadata.

        Returns
        -------
        str

        Warning
        -------
        The JSON representation is not guaranteed to be stable across versions.
        """
        ...

    @classmethod
    def from_json(json: str) -> RowIdMeta:
        """
        Load row id metadata from a JSON representation.

        Parameters
        ----------
        json : str
            The JSON representation of the row id metadata.

        Returns
        -------
        RowIdMeta
        """
        ...

    def __reduce__(self) -> tuple: ...

class RowIdSequence:
    """
    The stable row ids of the rows in a single fragment, in fragment order.

    Use this to attach pre-existing row ids to a fragment when assembling a
    transaction manually, so that rewritten rows keep their identity::

        sequence = RowIdSequence([7, 12])
        fragment = FragmentMetadata(..., row_id_meta=sequence.to_inline_metadata())

    The sequence may be shorter than the fragment's ``physical_rows``. The ids
    bind to the leading rows and the commit generates new ids for the remaining
    ones, which is how a fragment holding both rewritten and newly inserted rows
    is expressed: write the rewritten rows first and supply only their ids.
    Passing more ids than the fragment has rows is rejected.

    Warning
    -------
    Only duplicates within this sequence are rejected. Row ids must also be
    unique across the dataset, and Lance does not re-check that when
    committing, so the caller owns it. Supply only row ids that already exist
    and are being relocated by the same transaction, which must also remove
    every earlier occurrence of them. Do not generate ids for new rows yourself
    -- they come from a counter in the manifest that a concurrent commit can
    advance, so only the commit knows which values are free. Do not supply
    unused row ids either: a sequence covering all of a fragment's rows leaves
    the dataset's row id allocator untouched, so a later append will hand the
    same id out again.

    Parameters
    ----------
    row_ids : range | pa.Array | pa.ChunkedArray | Iterable[int]
        The row ids, in the order the corresponding rows appear in the
        fragment. A ``range`` with a step of one is stored compactly without
        materializing its values.
    """

    def __init__(
        self, row_ids: Union[range, pa.Array, pa.ChunkedArray, Iterable[int]]
    ) -> None: ...
    @staticmethod
    def from_inline_metadata(metadata: RowIdMeta) -> RowIdSequence:
        """
        Read back a sequence stored inline in fragment row id metadata.

        Parameters
        ----------
        metadata : RowIdMeta
            Row id metadata holding an inline sequence. Metadata pointing at an
            external file is not supported.

        Returns
        -------
        RowIdSequence
        """
        ...

    def to_inline_metadata(self) -> RowIdMeta:
        """
        Encode the sequence as row id metadata to store inline in the manifest.

        Returns
        -------
        RowIdMeta
            Suitable for the ``row_id_meta`` argument of
            :class:`lance.fragment.FragmentMetadata`.
        """
        ...

    def to_pyarrow(self) -> pa.UInt64Array:
        """
        Get the row ids as a ``uint64`` array, in sequence order.

        Returns
        -------
        pa.UInt64Array
        """
        ...

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def __reduce__(self) -> tuple: ...
