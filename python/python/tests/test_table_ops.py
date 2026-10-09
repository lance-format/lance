# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""
Tests for table operations such as conflict handling and raw commit operations.
"""

import uuid

import lance
import pyarrow as pa
import pytest
from lance.file import LanceFileWriter, stable_version
from lance.fragment import DataFile


def make_data_file(
    ds: lance.LanceDataset, fields: list[int], data: pa.Table
) -> DataFile:
    new_file_name = f"{uuid.uuid4()}.lance"
    new_file_path = f"{ds.uri}/data/{new_file_name}"
    with LanceFileWriter(new_file_path) as writer:
        writer.write_batch(data)

    return DataFile(
        path=new_file_name,
        fields=fields,
        column_indices=[i for i in range(len(fields))],
        file_major_version=int(stable_version().split(".")[0]),
        file_minor_version=int(stable_version().split(".")[1]),
    )


def test_index_after_replacement(tmp_path: str):
    """
    It should be possible to create an index on column X after a data replacement
    only if that replacement does not modify the column being indexed.
    """

    # Create a dataset with columns a and b in separate data files
    table = pa.Table.from_pydict({"a": range(100)})

    ds = lance.write_dataset(table, tmp_path)
    ds.add_columns({"b": "a + 1"})

    ds2 = lance.dataset(tmp_path)  # copies of the dataset
    ds3 = lance.dataset(tmp_path)  # from before the replacement

    # Replace column b with new data
    new_data_file = make_data_file(ds, [1], pa.table({"b": range(100, 200)}))

    ds.commit(
        ds.uri,
        lance.LanceOperation.DataReplacement(
            [lance.LanceOperation.DataReplacementGroup(0, new_data_file)]
        ),
        read_version=ds.version,
    )

    # Should be ok to create an index on column a
    ds2.create_scalar_index("a", "BTREE")

    # Creating an index on column b should conflict
    with pytest.raises(Exception, match="Retryable commit conflict for version 3"):
        ds3.create_scalar_index("b", "BTREE")

    # Should be ok to create an index when read version is higher than replacement
    lance.dataset(tmp_path).create_scalar_index("b", "BTREE")


def test_replacement_after_index(tmp_path: str):
    """
    It should be possible to replace data after an index has been created on the column
    only if the index was not covering the column being replaced.
    """
    table = pa.Table.from_pydict({"a": range(100)})

    ds = lance.write_dataset(table, tmp_path)
    ds.add_columns({"b": "a + 1"})

    ds2 = lance.dataset(tmp_path)  # copies of the dataset
    ds3 = lance.dataset(tmp_path)  # from before the index

    # Create an index on column a
    ds.create_scalar_index("a", "BTREE")

    # Replace column b with new data
    new_data_file = make_data_file(ds, [1], pa.table({"b": range(100, 200)}))

    # Should be ok (index was on column a, new data is on column b)
    ds2.commit(
        ds.uri,
        lance.LanceOperation.DataReplacement(
            [lance.LanceOperation.DataReplacementGroup(0, new_data_file)]
        ),
        read_version=ds2.version,
    )

    new_data_file = make_data_file(ds, [0], pa.table({"a": range(100, 200)}))

    # Should fail since index was on column a and new data is on column a
    with pytest.raises(Exception, match="Retryable commit conflict for version 3"):
        ds3.commit(
            ds.uri,
            lance.LanceOperation.DataReplacement(
                [lance.LanceOperation.DataReplacementGroup(0, new_data_file)]
            ),
            read_version=ds3.version,
        )


def test_data_file_create_basic(tmp_path: str):
    """DataFile.create should read file metadata and produce correct fields/indices."""
    table = pa.table({"a": range(10), "b": range(10, 20)})
    ds = lance.write_dataset(table, tmp_path)

    # Write a lance file with both columns
    new_file_name = f"{uuid.uuid4()}.lance"
    new_file_path = f"{tmp_path}/data/{new_file_name}"
    with LanceFileWriter(new_file_path) as writer:
        writer.write_batch(table)

    df = DataFile.create(ds, new_file_name)

    # Should have both field IDs from the dataset
    frag = ds.get_fragments()[0]
    expected_fields = frag.data_files()[0].fields
    assert df.fields == expected_fields
    assert df.column_indices == [0, 1]
    assert df.file_major_version == int(stable_version().split(".")[0])
    assert df.file_minor_version == int(stable_version().split(".")[1])
    assert df.file_size_bytes is not None and df.file_size_bytes > 0


def test_data_file_create_subset_columns(tmp_path: str):
    """DataFile.create should work for a file with a subset of dataset columns."""
    table = pa.table({"a": range(10), "b": range(10, 20)})
    ds = lance.write_dataset(table, tmp_path)
    ds.add_columns({"c": "a + b"})
    ds = lance.dataset(tmp_path)

    # Write a file with only column b
    new_file_name = f"{uuid.uuid4()}.lance"
    new_file_path = f"{tmp_path}/data/{new_file_name}"
    with LanceFileWriter(new_file_path, pa.schema([("b", pa.int64())])) as writer:
        writer.write_batch(pa.table({"b": range(100, 110)}))

    df = DataFile.create(ds, new_file_name)

    # Should only have b's field ID
    frag = ds.get_fragments()[0]
    all_fields = frag.data_files()[0].fields
    # b is the second field in the original data file
    b_field_id = all_fields[1]
    assert df.fields == [b_field_id]
    assert df.column_indices == [0]


def test_data_file_create_end_to_end(tmp_path: str):
    """DataFile.create should work end-to-end with DataReplacement."""
    table = pa.table({"a": range(100)})
    ds = lance.write_dataset(table, tmp_path)
    ds.add_columns({"b": "a + 1"})
    ds = lance.dataset(tmp_path)

    # Write a replacement file for column b
    new_file_name = f"{uuid.uuid4()}.lance"
    new_file_path = f"{tmp_path}/data/{new_file_name}"
    replacement_data = pa.table({"b": range(200, 300)})
    with LanceFileWriter(new_file_path, pa.schema([("b", pa.int64())])) as writer:
        writer.write_batch(replacement_data)

    # Use DataFile.create instead of manual construction
    df = DataFile.create(ds, new_file_name)

    ds.commit(
        ds.uri,
        lance.LanceOperation.DataReplacement(
            [lance.LanceOperation.DataReplacementGroup(0, df)]
        ),
        read_version=ds.version,
    )

    result = lance.dataset(tmp_path).to_table()
    assert result.column("b").to_pylist() == list(range(200, 300))
    assert result.column("a").to_pylist() == list(range(100))


def test_data_file_create_unknown_column(tmp_path: str):
    """DataFile.create should raise an error for a file with unknown columns."""
    table = pa.table({"a": range(10)})
    ds = lance.write_dataset(table, tmp_path)

    # Write a file with a column not in the dataset
    new_file_name = f"{uuid.uuid4()}.lance"
    new_file_path = f"{tmp_path}/data/{new_file_name}"
    with LanceFileWriter(new_file_path) as writer:
        writer.write_batch(pa.table({"z": range(10)}))

    with pytest.raises(Exception, match="z"):
        DataFile.create(ds, new_file_name)


def test_commit_conflict_raises_typed_error(tmp_path: str):
    """A losing commit should raise CommitConflictError, not a bare OSError."""
    from lance.commit import CommitConflictError

    table = pa.table({"a": range(100)})
    ds = lance.write_dataset(table, tmp_path)

    # Two commits based on the same version; the second must conflict.
    ds2 = lance.dataset(tmp_path)
    ds3 = lance.dataset(tmp_path)

    new_data_file = make_data_file(ds, [0], pa.table({"a": range(100, 200)}))
    ds2.commit(
        ds2.uri,
        lance.LanceOperation.DataReplacement(
            [lance.LanceOperation.DataReplacementGroup(0, new_data_file)]
        ),
        read_version=ds2.version,
    )

    new_data_file = make_data_file(ds, [0], pa.table({"a": range(200, 300)}))
    with pytest.raises(CommitConflictError) as exc_info:
        ds3.commit(
            ds3.uri,
            lance.LanceOperation.DataReplacement(
                [lance.LanceOperation.DataReplacementGroup(0, new_data_file)]
            ),
            read_version=ds3.version,
        )

    # It must be a CommitConflictError, still catchable as OSError, and retryable.
    assert isinstance(exc_info.value, OSError)
    assert exc_info.value.retryable is True


def test_incompatible_transaction_raises_non_retryable(tmp_path: str):
    """An incompatible transaction should raise a non-retryable CommitConflictError."""
    from lance.commit import CommitConflictError

    table = pa.table({"a": [1]})
    uri = tmp_path
    base = lance.write_dataset(table, uri)
    fragment = lance.fragment.LanceFragment.create(uri, table)
    stale_append = lance.LanceOperation.Append([fragment])

    # Overwrite the table, then try to append based on the stale version.
    lance.write_dataset(table, uri, mode="overwrite")
    with pytest.raises(CommitConflictError) as exc_info:
        lance.LanceDataset.commit(uri, stale_append, read_version=base.version)

    assert isinstance(exc_info.value, OSError)
    assert exc_info.value.retryable is False
