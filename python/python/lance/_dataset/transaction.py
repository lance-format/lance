# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset transaction and operation objects."""

from __future__ import annotations

import dataclasses
import uuid
import warnings
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, TypedDict, Union

import pyarrow as pa

from ..fragment import DataFile, FragmentMetadata

if TYPE_CHECKING:
    from .. import mem_wal
    from ..dataset import LanceDataset
    from ..lance import DatasetBasePath, InsertedRowsFilter, LanceSchema
    from .index import Index


class BulkCommitResult(TypedDict):
    dataset: LanceDataset
    merged: Transaction


@dataclass
class Transaction:
    """A dataset change based on one read version.

    Attributes
    ----------
    tag: str, optional
        Optional label stored with the transaction and returned by
        :meth:`LanceDataset.read_transaction`.
    """

    read_version: int
    operation: LanceOperation.BaseOperation
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    transaction_properties: Optional[Dict[str, str]] = dataclasses.field(
        default_factory=dict
    )
    tag: Optional[str] = None


class UpdateResult(TypedDict):
    num_rows_updated: int


class DeleteResult(TypedDict):
    num_deleted_rows: int


class AlterColumn(TypedDict):
    path: str
    name: Optional[str]
    nullable: Optional[bool]
    data_type: Optional[pa.DataType]


class ExecuteResult(TypedDict):
    num_inserted_rows: int
    num_updated_rows: int
    num_deleted_rows: int


# LanceOperation groups the transaction operations understood by the binding.


class LanceOperation:
    @staticmethod
    def _validate_fragments(fragments):
        if not isinstance(fragments, list):
            raise TypeError(
                f"fragments must be list[FragmentMetadata], got {type(fragments)}"
            )
        if len(fragments) > 0 and not all(
            isinstance(f, FragmentMetadata) for f in fragments
        ):
            raise TypeError(
                f"fragments must be list[FragmentMetadata], got {type(fragments[0])}"
            )

    class BaseOperation(ABC):
        """
        Base class for operations that can be applied to a dataset.

        See available operations under :class:`LanceOperation`.
        """

    @dataclass
    class Overwrite(BaseOperation):
        """
        Overwrite or create a new dataset.

        Attributes
        ----------
        new_schema: pyarrow.Schema
            The schema of the new dataset.
        fragments: list[FragmentMetadata]
            The newly written fragments that make up the new dataset. They are
            assigned fresh ids when the operation is committed, continuing from
            the highest id the dataset has ever used, so any id they carry is
            ignored. Since we reassign fragment ids, a fragment with a deletion
            file is rejected: use :class:`LanceOperation.Delete` to commit
            deletions, or :class:`LanceOperation.Merge` to change the schema of
            existing fragments.
        initial_bases: list[DatasetBasePath], optional
            Base paths to register when creating a new dataset (CREATE mode only).
            **Only valid in CREATE mode**. Will raise an error if used with
            OVERWRITE on existing dataset.

        Warning
        -------
        This is an advanced API for distributed operations. To overwrite or
        create new dataset on a single machine, use :func:`lance.write_dataset`.

        Examples
        --------

        To create or overwrite a dataset, first use
        :meth:`lance.fragment.LanceFragment.create` to create fragments. Then
        collect the fragment metadata into a list and pass it along with the
        schema to this class. Finally, pass the operation to the
        :meth:`LanceDataset.commit` method to create the new dataset.

        >>> import lance
        >>> import pyarrow as pa
        >>> tab1 = pa.table({"a": [1, 2], "b": ["a", "b"]})
        >>> tab2 = pa.table({"a": [3, 4], "b": ["c", "d"]})
        >>> fragment1 = lance.fragment.LanceFragment.create("example", tab1)
        >>> fragment2 = lance.fragment.LanceFragment.create("example", tab2)
        >>> fragments = [fragment1, fragment2]
        >>> operation = lance.LanceOperation.Overwrite(tab1.schema, fragments)
        >>> dataset = lance.LanceDataset.commit("example", operation)
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        """

        new_schema: LanceSchema | pa.Schema
        fragments: Iterable[FragmentMetadata]
        initial_bases: Optional[List[DatasetBasePath]] = None

        def __post_init__(self):
            LanceOperation._validate_fragments(self.fragments)

    @dataclass
    class Append(BaseOperation):
        """
        Append new rows to the dataset.

        Attributes
        ----------
        fragments: list[FragmentMetadata]
            The fragments that contain the new rows.

        Warning
        -------
        This is an advanced API for distributed operations. To append to a
        dataset on a single machine, use :func:`lance.write_dataset`.

        Examples
        --------

        To append new rows to a dataset, first use
        :meth:`lance.fragment.LanceFragment.create` to create fragments. Then
        collect the fragment metadata into a list and pass it to this class.
        Finally, pass the operation to the :meth:`LanceDataset.commit`
        method to create the new dataset.

        >>> import lance
        >>> import pyarrow as pa
        >>> tab1 = pa.table({"a": [1, 2], "b": ["a", "b"]})
        >>> dataset = lance.write_dataset(tab1, "example")
        >>> tab2 = pa.table({"a": [3, 4], "b": ["c", "d"]})
        >>> fragment = lance.fragment.LanceFragment.create("example", tab2)
        >>> operation = lance.LanceOperation.Append([fragment])
        >>> dataset = lance.LanceDataset.commit("example", operation,
        ...                                     read_version=dataset.version)
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        """

        fragments: Iterable[FragmentMetadata]

        def __post_init__(self):
            LanceOperation._validate_fragments(self.fragments)

    @dataclass
    class Delete(BaseOperation):
        """
        Remove fragments or rows from the dataset.

        Attributes
        ----------
        updated_fragments: list[FragmentMetadata]
            The fragments that have been updated with new deletion vectors.
        deleted_fragment_ids: list[int]
            The ids of the fragments that have been deleted entirely. These are
            the fragments where :meth:`LanceFragment.delete()` returned None.
        predicate: str
            The original SQL predicate used to select the rows to delete.

        Warning
        -------
        This is an advanced API for distributed operations. To delete rows from
        dataset on a single machine, use :meth:`lance.LanceDataset.delete`.

        Examples
        --------

        To delete rows from a dataset, call :meth:`lance.fragment.LanceFragment.delete`
        on each of the fragments. If that returns a new fragment, add that to
        the ``updated_fragments`` list. If it returns None, that means the whole
        fragment was deleted, so add the fragment id to the ``deleted_fragment_ids``.
        Finally, pass the operation to the :meth:`LanceDataset.commit` method to
        complete the deletion operation.

        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2], "b": ["a", "b"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> table = pa.table({"a": [3, 4], "b": ["c", "d"]})
        >>> dataset = lance.write_dataset(table, "example", mode="append")
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        >>> predicate = "a >= 2"
        >>> updated_fragments = []
        >>> deleted_fragment_ids = []
        >>> for fragment in dataset.get_fragments():
        ...     new_fragment = fragment.delete(predicate)
        ...     if new_fragment is not None:
        ...         updated_fragments.append(new_fragment)
        ...     else:
        ...         deleted_fragment_ids.append(fragment.fragment_id)
        >>> operation = lance.LanceOperation.Delete(updated_fragments,
        ...                                         deleted_fragment_ids,
        ...                                         predicate)
        >>> dataset = lance.LanceDataset.commit("example", operation,
        ...                                     read_version=dataset.version)
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        """

        updated_fragments: Iterable[FragmentMetadata]
        deleted_fragment_ids: Iterable[int]
        predicate: str

        def __post_init__(self):
            LanceOperation._validate_fragments(self.updated_fragments)

    @dataclass
    class Update(BaseOperation):
        """
        Operation that updates rows in the dataset.

        Attributes
        ----------
        removed_fragment_ids: list[int]
            The ids of the fragments that have been removed entirely.
        updated_fragments: list[FragmentMetadata]
            The fragments that have been updated with new deletion vectors.
            These are used as given, so pass back the metadata read from the
            dataset rather than a freshly constructed object, or the fragment
            loses its row id and version metadata.
        new_fragments: list[FragmentMetadata]
            The fragments that contain the new rows. On a dataset that uses
            stable row ids, set ``row_id_meta`` on these to carry the ids of
            rewritten rows over; see :class:`lance.fragment.RowIdSequence`. The
            created-at and last-updated-at version metadata are derived during
            the commit and should be left as None.
        fields_modified: list[int]
            If any fields are modified in updated_fragments, then they must be
            listed here so those fragments can be removed from indices covering
            those fields.
        fields_for_preserving_frag_bitmap: list[int]
            The fields that used to judge whether to preserve the new frag's id into
            the frag bitmap of the specified indices.
        updated_fragment_offsets: dict[int, bytes], optional
            Physical row offsets that matched the update, keyed by fragment id,
            each serialized in the portable RoaringBitmap format. Set on
            ``rewrite_columns`` updates over stable row ids so the commit
            refreshes row-level version metadata for the matched rows only.
        compacted_sstables: list[lance.mem_wal.CompactedSsTable]
            MemWAL SSTables to record as compacted. An empty list changes
            nothing. A non-empty list requires a MemWAL index on the table.
        mem_wal_dir_id: str, optional
            The ``dir_id`` of the MemWAL the SSTables belong to. Required when
            ``compacted_sstables`` is non-empty; the commit fails if the
            table's MemWAL has since been disabled or replaced.
        inserted_rows_filter: InsertedRowsFilter, optional
            Keys inserted by a merge-insert, used for concurrent conflict
            detection. ``might_contain`` compares stored key hashes for an
            exact set and may have false positives for a bloom filter.
        """

        removed_fragment_ids: List[int] = dataclasses.field(default_factory=list)
        updated_fragments: List[FragmentMetadata] = dataclasses.field(
            default_factory=list
        )
        new_fragments: List[FragmentMetadata] = dataclasses.field(default_factory=list)
        fields_modified: List[int] = dataclasses.field(default_factory=list)
        fields_for_preserving_frag_bitmap: List[int] = dataclasses.field(
            default_factory=list
        )
        update_mode: str = ""
        updated_fragment_offsets: Optional[Dict[int, bytes]] = None
        compacted_sstables: List["mem_wal.CompactedSsTable"] = dataclasses.field(
            default_factory=list
        )
        mem_wal_dir_id: Optional[str] = None
        inserted_rows_filter: Optional[InsertedRowsFilter] = None

        def __post_init__(self):
            LanceOperation._validate_fragments(self.updated_fragments)
            LanceOperation._validate_fragments(self.new_fragments)

    @dataclass
    class Merge(BaseOperation):
        """
        Operation that adds columns. Unlike Overwrite, this should not change
        the structure of the fragments, allowing existing indices to be kept.

        Attributes
        ----------
        fragments: iterable of FragmentMetadata
            The fragments that make up the new dataset.
        schema: LanceSchema or pyarrow.Schema
            The schema of the new dataset. Passing a LanceSchema is preferred,
            and passing a pyarrow.Schema is deprecated.
        preserves_nullability: bool
            True when this merge makes no nullability-affecting schema change:
            it introduces no field that data staged against an earlier schema
            could not safely omit. Without the assertion (the default) the
            merge conservatively conflicts with concurrent appends, whose
            fragments would omit new columns and read as null; that can only
            cause a retry. Pass True when every column this merge introduces
            is nullable to let concurrent appends commit without conflict.

        Warning
        -------
        This is an advanced API for distributed operations. To overwrite or
        create new dataset on a single machine, use :func:`lance.write_dataset`.

        Examples
        --------

        To add new columns to a dataset, first define a method that will create
        the new columns based on the existing columns. Then use
        :meth:`lance.fragment.LanceFragment.add_columns`

        >>> import lance
        >>> import pyarrow as pa
        >>> import pyarrow.compute as pc
        >>> table = pa.table({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        >>> def double_a(batch: pa.RecordBatch) -> pa.RecordBatch:
        ...     doubled = pc.multiply(batch["a"], 2)
        ...     return pa.record_batch([doubled], ["a_doubled"])
        >>> fragments = []
        >>> for fragment in dataset.get_fragments():
        ...     new_fragment, new_schema = fragment.merge_columns(double_a,
        ...                                                       columns=['a'])
        ...     fragments.append(new_fragment)
        >>> operation = lance.LanceOperation.Merge(fragments, new_schema)
        >>> dataset = lance.LanceDataset.commit("example", operation,
        ...                                     read_version=dataset.version)
        >>> dataset.to_table().to_pandas()
           a  b  a_doubled
        0  1  a          2
        1  2  b          4
        2  3  c          6
        3  4  d          8
        """

        fragments: Iterable[FragmentMetadata]
        schema: LanceSchema | pa.Schema
        preserves_nullability: bool = False

        def __post_init__(self):
            if isinstance(self.schema, pa.Schema):
                warnings.warn(
                    "Passing a pyarrow.Schema to Merge is deprecated. "
                    "Please use a LanceSchema instead.",
                    DeprecationWarning,
                )
            LanceOperation._validate_fragments(self.fragments)

    @dataclass
    class Restore(BaseOperation):
        """
        Operation that restores a previous version of the dataset.
        """

        version: int

    @dataclass
    class Clone(BaseOperation):
        """Operation that creates a clone or branch from a dataset reference.

        This operation is created internally by clone and branch APIs. It is
        exposed so transactions returned by :meth:`LanceDataset.get_transactions`
        can represent clone metadata without losing information.

        Only a shallow clone can be committed directly. A deep clone must copy
        the source files first, so committing one raises ``OSError``; use
        :meth:`LanceDataset.deep_clone` instead.

        Attributes
        ----------
        is_shallow: bool
            Whether data files are shared with the source dataset.
        ref_name: str, optional
            Source branch name, or ``None`` for the main branch.
        ref_version: int
            Source dataset version.
        ref_path: str
            Source dataset URI.
        branch_name: str, optional
            Destination branch name, when the clone creates a branch.
        """

        is_shallow: bool
        ref_name: Optional[str]
        ref_version: int
        ref_path: str
        branch_name: Optional[str]

    @dataclass
    class RewriteGroup:
        """
        Collection of rewritten files
        """

        old_fragments: Iterable[FragmentMetadata]
        new_fragments: Iterable[FragmentMetadata]

    @dataclass
    class RewrittenIndex:
        """
        An index that has been rewritten
        """

        old_id: str
        new_id: str
        new_details_type_url: str
        new_details_value: bytes
        new_index_version: int

    @dataclass
    class Rewrite(BaseOperation):
        """
        Operation that rewrites one or more files and indices into one
        or more files and indices.

        Attributes
        ----------
        groups: list[RewriteGroup]
            Groups of files that have been rewritten.
        rewritten_indices: list[RewrittenIndex]
            Indices that have been rewritten.
        frag_reuse_index: Index, optional
            Fragment reuse index entry to install with this rewrite. ``None``
            leaves the existing entry unchanged.

        Warning
        -------
        This is an advanced API not intended for general use.
        """

        groups: Iterable[LanceOperation.RewriteGroup]
        rewritten_indices: Iterable[LanceOperation.RewrittenIndex]
        frag_reuse_index: Optional[Index] = None

        def __post_init__(self):
            all_frags = [old for group in self.groups for old in group.old_fragments]
            all_frags += [new for group in self.groups for new in group.new_fragments]
            LanceOperation._validate_fragments(all_frags)

    @dataclass
    class CreateIndex(BaseOperation):
        """
        Operation that creates an index on the dataset.
        """

        new_indices: List[Index]
        removed_indices: List[Index]

    @dataclass
    class DataReplacementGroup:
        """
        Group of data replacements
        """

        fragment_id: int
        new_file: DataFile

    @dataclass
    class DataReplacement(BaseOperation):
        """
        Operation that replaces existing datafiles in the dataset.
        """

        replacements: List[LanceOperation.DataReplacementGroup]

    @dataclass
    class DataOverlayFile:
        """
        An overlay file supplying new values for a subset of
        ``(physical offset, field)`` cells of a fragment, resolved on read and
        layered over the base data without rewriting the base files.

        The overlay is dense or sparse depending on the shape of ``offsets``:
        pass a single iterable of ints (e.g. a :class:`~lance.bitmap.Bitmap`
        or a ``List[int]``) for a dense overlay (one offset set shared by
        every field in ``data_file``), or a list of int iterables for a
        sparse overlay (one offset set per field, in the order of the file's
        fields). Offsets are **physical** row offsets (positions in the base
        files, counting deleted rows), like deletion vectors.

        Attributes
        ----------
        data_file : DataFile
            The Lance data file storing the overlay's new cell values — one
            value column per covered field. The value at each covered offset is
            stored at the rank (0-based count of covered offsets below it) of
            that offset in the field's coverage.
        offsets : Iterable[int] | List[Iterable[int]]
            The covered physical row offsets. A single int iterable is dense
            coverage (shared by every field); a list of int iterables is
            sparse coverage (in field order). Offsets are always resolved in
            ascending order — the smallest covered offset maps to row 0 of
            ``data_file``, the next-smallest to row 1, and so on — regardless
            of the order values are given in, so a plain ``List[int]`` need
            not be pre-sorted. A repeated offset raises ``ValueError``: it
            would shift every later offset onto the wrong row.
        committed_version : Optional[int]
            The dataset version at which this overlay became effective. Leave as
            ``None`` when creating an overlay to commit — the commit stamps it.
            It is populated when reading an existing fragment's overlays so they
            round-trip through :class:`FragmentMetadata`.
        """

        data_file: DataFile
        offsets: Union[Iterable[int], List[Iterable[int]]]
        committed_version: Optional[int] = None

    @dataclass
    class DataOverlayGroup:
        """
        Overlay files to append to a single fragment.

        Attributes
        ----------
        fragment_id : int
            The id of the fragment the overlays apply to.
        overlays : List[LanceOperation.DataOverlayFile]
            The overlay files to append, ordered oldest-first (a later entry is
            newer and wins where coverage overlaps).
        """

        fragment_id: int
        overlays: List[LanceOperation.DataOverlayFile]

    @dataclass
    class DataOverlay(BaseOperation):
        """
        Operation that appends data overlay files to fragments.

        Overlays are appended to each fragment's existing overlays (overlays
        written by concurrent commits are preserved) and resolved on read
        over the base data without rewriting it.

        If multiple groups target the same data then the values in the
        latest group take precedence.
        """

        groups: List[LanceOperation.DataOverlayGroup]

    @dataclass
    class Project(BaseOperation):
        """
        Operation that project columns.
        Use this operator for drop column or rename/swap column.

        Attributes
        ----------
        schema: LanceSchema
            The lance schema of the new dataset.
        preserves_nullability: bool
            True when this projection makes no nullability-affecting schema
            change, as a rename or a drop does not. Without the assertion
            (the default) the projection conservatively conflicts with
            concurrent writes, which can only cause a retry.

        Examples
        --------
        Use the projece operator to swap column:

        >>> import lance
        >>> import pyarrow as pa
        >>> import pyarrow.compute as pc
        >>> from lance.schema import LanceSchema
        >>> table = pa.table({"a": [1, 2], "b": ["a", "b"], "b1": ["c", "d"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> dataset.to_table().to_pandas()
           a  b b1
        0  1  a  c
        1  2  b  d
        >>>
        >>> ## rename column `b` into `b0` and rename b1 into `b`
        >>> table = pa.table({"a": [3, 4], "b0": ["a", "b"], "b": ["c", "d"]})
        >>> lance_schema = LanceSchema.from_pyarrow(table.schema)
        >>> operation = lance.LanceOperation.Project(lance_schema)
        >>> dataset = lance.LanceDataset.commit("example", operation, read_version=1)
        >>> dataset.to_table().to_pandas()
           a b0  b
        0  1  a  c
        1  2  b  d
        """

        schema: LanceSchema
        preserves_nullability: bool = False

    @dataclass
    class UpdateMap:
        """
        Represents updates to a metadata map.

        Attributes
        ----------
        updates : Dict[str, Optional[str]]
            Dictionary of key-value pairs to update. None values mean delete the key.
        replace : bool
            If True, replace the entire map with the new entries.
            If False, merge the new entries with the existing map.
        """

        updates: Dict[str, Optional[str]]
        replace: bool = False

    @dataclass
    class UpdateConfig(BaseOperation):
        """
        Operation that updates dataset metadata.

        This operation can update various metadata levels:
        - Dataset configuration
        - Table metadata
        - Schema metadata
        - Field-specific metadata

        Attributes
        ----------
        config_updates : Optional[UpdateMap]
            Updates to the dataset configuration.
        table_metadata_updates : Optional[UpdateMap]
            Updates to the table metadata.
        schema_metadata_updates : Optional[UpdateMap]
            Updates to the schema metadata.
        field_metadata_updates : Optional[Dict[int, UpdateMap]]
            Updates to field metadata, keyed by field ID.
        """

        config_updates: Optional[LanceOperation.UpdateMap] = None
        table_metadata_updates: Optional[LanceOperation.UpdateMap] = None
        schema_metadata_updates: Optional[LanceOperation.UpdateMap] = None
        field_metadata_updates: Optional[Dict[int, LanceOperation.UpdateMap]] = None

    @dataclass
    class UpdateMemWalState(BaseOperation):
        """Record MemWAL SSTables compacted into the base table.

        An empty list commits as a no-op. A non-empty list requires a MemWAL
        index (``__lance_mem_wal``) and updates that index's compaction record.

        Attributes
        ----------
        compacted_sstables: list[lance.mem_wal.CompactedSsTable]
            SSTables to mark compacted, one entry per shard.
        dir_id: str
            The ``dir_id`` of the MemWAL this update was built for; empty for a
            MemWAL without one. The commit fails unless the table still carries
            that MemWAL.
        """

        compacted_sstables: List["mem_wal.CompactedSsTable"] = dataclasses.field(
            default_factory=list
        )
        dir_id: str = ""
