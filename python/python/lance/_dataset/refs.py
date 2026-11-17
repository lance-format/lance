# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset tag, branch, and version metadata objects."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, TypedDict

if TYPE_CHECKING:
    from ..lance import _Dataset


class Tag(TypedDict):
    branch: Optional[str]
    version: int
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    manifest_size: int
    metadata: Dict[str, str]


class Branch(TypedDict):
    parent_branch: Optional[str]
    branch_identifier: List[Tuple[int, str]]
    parent_version: int
    create_at: int
    manifest_size: int
    metadata: Dict[str, str]


class Version(TypedDict):
    version: int
    timestamp: int | datetime
    metadata: Dict[str, str]
    #: Size of this version's manifest file, in bytes, or ``None`` when the
    #: commit handler could not report the size. The manifest grows with the
    #: number of columns and fragments, so this is useful for observing
    #: metadata amplification in wide tables.
    manifest_size: int | None


def _convert_version_timestamp(version: Version) -> Version:
    # TODO: python datetime supports only microsecond precision. When a
    # separate Version object is implemented, expose the precise timestamp
    # (ns) to python.
    ts_nanos = version["timestamp"]
    version["timestamp"] = datetime.fromtimestamp(ts_nanos // 1e9) + timedelta(
        microseconds=(ts_nanos % 1e9) // 1e3
    )
    return version


class VersionRef(TypedDict):
    version: int


class Tags:
    """
    Dataset tag manager.
    """

    def __init__(self, dataset: _Dataset):
        self._ds = dataset

    def get(self, tag: str) -> Tag:
        """
        Get metadata for a dataset tag by name without listing all tags.

        Parameters
        ----------
        tag: str
            The name of the tag to retrieve.

        Returns
        -------
        Tag
            Tag metadata with the same fields as the values returned by
            :meth:`list`, including the referenced branch, version, timestamps,
            manifest size, and attached metadata.

        Raises
        ------
        ValueError
            If the tag does not exist or its name is invalid.

        Examples
        --------
        >>> import tempfile
        >>> import lance
        >>> import pyarrow as pa
        >>> with tempfile.TemporaryDirectory() as uri:
        ...     ds = lance.write_dataset(pa.table({"id": [1, 2]}), uri)
        ...     ds.tags.create("v1", 1)
        ...     ds.tags.get("v1")["version"]
        1
        """
        return self._ds.get_tag(tag)

    def list(self) -> dict[str, Tag]:
        """
        List all dataset tags.

        Returns
        -------
        dict[str, Tag]
            A dictionary mapping tag names to tag metadata, including the
            referenced branch, version, timestamps, manifest size, and any
            attached metadata.
        """
        return self._ds.tags()

    def get_version(self, tag: str) -> int:
        """
        Get the version of a specific tag by name.

        Parameters
        ----------
        tag: str
            The name of the tag to retrieve.

        Returns
        -------
        int
            The version number of the tag.

        Raises
        ------
        ValueError
            If the tag does not exist. Use :meth:`list` to check for presence.
        """
        return self._ds.get_version(tag)

    def list_ordered(self, order: Optional[str] = None) -> List[Tuple[str, Tag]]:
        """
        List all dataset tags.

        Parameters
        ----------
        order: str, optional
            The order in which to return the tags.
            "asc" or "desc" can be used to specify the order explicitly.
            default 'desc'.

        Returns
        -------
        List[Tuple[str, Tag]]
            An ordered list of tuples mapping tag names to its `Tag` metadata.
        """
        return self._ds.tags_ordered(order)

    def create(
        self,
        tag: str,
        reference: Optional[int | str | Tuple[Optional[str], Optional[int]]] = None,
    ) -> None:
        """
        Create a tag for a given dataset version.

        Parameters
        ----------
        tag: str,
            The name of the tag to create. This name must be unique among all tag
            names for the dataset.
        reference : int, str or Tuple[Optional[str], Optional[int]]
            An integer specifies a version number in the current branch; a string
            specifies a tag name; a Tuple[Optional[str], Optional[int]] specifies
            a version number in a specified branch. (None, None) means the latest
            version_number on the default branch. ``("main", version)`` is an
            explicit alias for the default branch in this reference context.
        """
        self._ds.create_tag(tag, reference)

    def delete(self, tag: str) -> None:
        """
        Delete tag from the dataset.

        Parameters
        ----------
        tag: str,
            The name of the tag to delete.

        """
        self._ds.delete_tag(tag)

    def update(
        self,
        tag: str,
        reference: Optional[int | str | Tuple[Optional[str], Optional[int]]] = None,
    ) -> None:
        """
        Update tag to a new version.

        Parameters
        ----------
        tag: str,
            The name of the tag to update.
        reference : int, str or Tuple[Optional[str], Optional[int]]
            An integer specifies a version number in the current branch; a string
            specifies a tag name; a Tuple[Optional[str], Optional[int]] specifies
            a version number in a specified branch. (None, None) means the latest
            version_number on the default branch. ``("main", version)`` is an
            explicit alias for the default branch in this reference context.
        """
        self._ds.update_tag(tag, reference)

    def replace_metadata(self, tag: str, metadata: Dict[str, str]) -> None:
        """
        Replace metadata for an existing tag.

        This replaces the entire metadata map instead of merging with existing
        keys. It does not change the tag reference, and it does not update
        `updated_at`. `updated_at` only changes when `update()` moves the tag
        to a different reference.
        """
        self._ds.replace_tag_metadata(tag, metadata)


class Branches:
    """
    Dataset branch manager.
    """

    def __init__(self, dataset: _Dataset):
        self._ds = dataset

    def list(self) -> dict[str, Branch]:
        """
        List all dataset branches.

        Returns
        -------
        dict[str, Branch]
            A dictionary mapping branch names to branch metadata.
        """
        return self._ds.branches()

    def list_ordered(self, order: Optional[str] = None) -> List[Tuple[str, Branch]]:
        """
        List all dataset branches ordered by parent version.
        """
        return self._ds.branches_ordered(order)

    def delete(self, branch: str) -> None:
        """
        Delete a branch.
        """
        self._ds.delete_branch(branch)

    def replace_metadata(self, branch: str, metadata: Dict[str, str]) -> None:
        """
        Replace metadata for a branch.
        """
        self._ds.replace_branch_metadata(branch, metadata)
