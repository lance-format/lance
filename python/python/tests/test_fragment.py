# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import json
import multiprocessing
import pickle
import uuid
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from helper import ProgressForTest
from lance import (
    FragmentMetadata,
    LanceDataset,
    LanceFragment,
    LanceOperation,
    write_dataset,
)
from lance.debug import format_fragment
from lance.file import LanceFileWriter
from lance.fragment import RowIdMeta, RowIdSequence, write_fragments
from lance.progress import FileSystemFragmentWriteProgress


def test_write_fragment(tmp_path: Path):
    with pytest.raises(OSError):
        LanceFragment.create(tmp_path, pd.DataFrame([]))
    with pytest.raises(OSError):
        LanceFragment.create(tmp_path, pd.DataFrame([{}]))

    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    frag = LanceFragment.create(tmp_path, df)

    assert len(frag.files) == 1
    assert frag.files[0].fields == [0]
    assert frag.physical_rows == 5
    assert frag.row_id_meta is None
    assert frag.deletion_file is None

    meta = frag.to_json()
    assert "id" in meta
    assert "files" in meta
    assert meta["files"][0]["fields"] == [0]


def test_write_fragment_two_phases(tmp_path: Path):
    num_files = 10
    json_array = []
    for i in range(num_files):
        df = pd.DataFrame({"a": [i * 10]})
        frag = LanceFragment.create(tmp_path, df)
        json_array.append(json.dumps(frag.to_json()))

    fragments = [FragmentMetadata.from_json(j) for j in json_array]

    schema = pa.schema([pa.field("a", pa.int64())])

    operation = LanceOperation.Overwrite(schema, fragments)
    dataset = LanceDataset.commit(tmp_path, operation)

    df = dataset.to_table().to_pandas()
    pd.testing.assert_frame_equal(
        df, pd.DataFrame({"a": [i * 10 for i in range(num_files)]})
    )


def test_write_legacy_fragment(tmp_path: Path):
    tab = pa.table({"a": range(1024)})
    frag = LanceFragment.create(tmp_path, tab, data_storage_version="legacy")
    assert "file_major_version=2" not in str(frag)

    tab = pa.table({"a": range(1024)})
    frag = LanceFragment.create(tmp_path, tab, data_storage_version="stable")
    assert "file_major_version=2" in str(frag)


def test_scan_fragment(tmp_path: Path):
    tab = pa.table({"a": range(100), "b": range(100, 200)})
    ds = write_dataset(tab, tmp_path)
    frag = ds.get_fragments()[0]

    actual = frag.to_table(
        columns=["b"],
        filter="a >= 2",
        offset=20,
    )
    expected = pa.table({"b": range(122, 200)})
    assert actual == expected


def test_scan_fragment_with_dynamic_projection(tmp_path: Path):
    tab = pa.table({"a": range(100), "b": range(100, 200)})
    ds = write_dataset(tab, tmp_path)
    frag = ds.get_fragments()[0]

    actual = frag.to_table(
        columns={"b_proj": "b"},
        filter="a >= 2",
        offset=20,
    )
    expected = pa.table({"b_proj": range(122, 200)})
    assert actual == expected


def test_fragment_session(tmp_path: Path):
    tab = pa.table({"a": range(100), "b": range(100, 200)})
    ds = write_dataset(tab, tmp_path)
    frag = ds.get_fragments()[0]

    session = frag.open_session(columns=["a", "b"], with_row_address=False)
    expected = frag.take(indices=range(1, 50), columns=["a", "b"])
    actual = session.take(range(1, 50))
    assert actual == expected

    session = frag.open_session(columns=["a", "b"], with_row_address=True)
    assert session.take(range(1, 5)).schema.names == ["a", "b", "_rowaddr"]


def test_write_fragments(tmp_path: Path):
    # Should result in two files since each batch is 8MB and max_bytes_per_file is small
    batches = pa.RecordBatchReader.from_batches(
        pa.schema([pa.field("a", pa.string())]),
        [
            pa.record_batch([pa.array(["0" * 1024] * 1024 * 8)], names=["a"]),
            pa.record_batch([pa.array(["0" * 1024] * 1024 * 8)], names=["a"]),
        ],
    )

    progress = ProgressForTest()
    fragments = write_fragments(
        batches,
        tmp_path,
        max_rows_per_group=512,
        max_bytes_per_file=1024,
        progress=progress,
    )
    assert len(fragments) == 2
    assert all(isinstance(f, FragmentMetadata) for f in fragments)
    # progress hook was called for each fragment
    assert progress.begin_called == 2
    assert progress.complete_called == 2


def test_write_fragments_schema_holes(tmp_path: Path):
    # Create table with 3 cols
    data = pa.table({"a": range(3)})
    dataset = write_dataset(data, tmp_path)
    dataset.add_columns({"b": "a + 1"})
    dataset.add_columns({"c": "a + 2"})
    # Delete the middle column to create a hole in the field ids
    dataset.drop_columns(["b"])

    def get_field_ids(fragment):
        return [id for f in fragment.files for id in f.fields]

    field_ids = get_field_ids(dataset.get_fragments()[0].metadata)

    data = pa.table({"a": range(3, 6), "c": range(5, 8)})
    fragment = LanceFragment.create(tmp_path, data)
    assert get_field_ids(fragment) == field_ids

    data = pa.table({"a": range(6, 9), "c": range(8, 11)})
    fragments = write_fragments(data, tmp_path)
    assert len(fragments) == 1
    assert get_field_ids(fragments[0]) == field_ids

    operation = LanceOperation.Append([fragment, *fragments])
    dataset = LanceDataset.commit(tmp_path, operation, read_version=dataset.version)

    assert dataset.to_table().equals(pa.table({"a": range(9), "c": range(2, 11)}))


def test_write_fragment_with_progress(tmp_path: Path):
    df = pd.DataFrame({"a": [10 * 10]})
    progress = ProgressForTest()
    LanceFragment.create(tmp_path, df, progress=progress)
    assert progress.begin_called == 1
    assert progress.complete_called == 1


def failing_write(progress_uri: str, dataset_uri: str):
    # re-create progress so we don't have to pickle it
    progress = FileSystemFragmentWriteProgress(
        progress_uri, metadata={"test_key": "test_value"}
    )
    arr = pa.array(range(100))
    batch = pa.record_batch([arr], names=["a"])

    def data():
        yield batch
        raise Exception("Something went wrong!")

    reader = pa.RecordBatchReader.from_batches(batch.schema, data())
    with pytest.raises(Exception):
        LanceFragment.create(
            dataset_uri,
            reader,
            fragment_id=1,
            progress=progress,
        )


def test_dataset_progress(tmp_path: Path):
    dataset_uri = tmp_path / "dataset"
    progress_uri = tmp_path / "progress"
    data = pa.table({"a": range(100)})
    progress = FileSystemFragmentWriteProgress(progress_uri)
    fragment = LanceFragment.create(
        dataset_uri,
        data,
        progress=progress,
    )

    # In-progress file should be deleted
    assert not (progress_uri / "fragment_0.in_progress").exists()

    # Metadata should be written
    with open(progress_uri / "fragment_0.json") as f:
        metadata = json.load(f)

    assert metadata["id"] == 0
    assert len(metadata["files"]) == 1
    # Fragments aren't exactly equal, because the file was written before
    # physical_rows was known.  However, the paths should be the same.
    assert len(fragment.files) == 1
    deserialized = FragmentMetadata.from_json(json.dumps(metadata))
    assert len(deserialized.files) == 1
    assert fragment.files[0].path == deserialized.files[0].path

    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=failing_write, args=(progress_uri, dataset_uri))
    p.start()
    try:
        p.join()
    except Exception:
        # Allow a crash to happen
        pass

    # In-progress file should be present
    with open(progress_uri / "fragment_1.in_progress") as f:
        progress_data = json.load(f)
    assert progress_data["fragment_id"] == 1
    # progress contains custom metadata
    assert progress_data["metadata"]["test_key"] == "test_value"

    # Metadata should be written
    with open(progress_uri / "fragment_1.json") as f:
        metadata = json.load(f)
    assert metadata["id"] == 1


def test_fragment_meta():
    # Intentionally leaving off column_indices / version fields to make sure
    # we can handle backwards compatibility (though not clear we need to)
    data = {
        "id": 0,
        "files": [
            {"path": "0.lance", "fields": [0], "file_size_bytes": 100},
            {"path": "1.lance", "fields": [1]},
        ],
        "deletion_file": None,
        "physical_rows": 100,
    }
    meta = FragmentMetadata.from_json(json.dumps(data))

    assert meta.id == 0
    assert len(meta.files) == 2
    with pytest.warns(DeprecationWarning):
        assert meta.files[0].path() == "0.lance"
    assert meta.files[1].path == "1.lance"

    assert repr(meta) == (
        "FragmentMetadata(id=0, files=[DataFile(path='0.lance', fields=[0], "
        "column_indices=[], file_major_version=0, file_minor_version=0, "
        "file_size_bytes=100), DataFile(path='1.lance', fields=[1], column_indices=[], "
        "file_major_version=0, file_minor_version=0, file_size_bytes=None)], "
        "physical_rows=100, deletion_file=None, row_id_meta=None, "
        "created_at_version_meta=None, last_updated_at_version_meta=None, overlays=[])"
    )


def test_fragment_v2(tmp_path):
    dataset_uri = tmp_path / "dataset"
    tab = pa.table(
        {
            "a": pa.array(range(1024)),
        }
    )
    lance.write_dataset([], dataset_uri, schema=tab.schema)
    fragments = write_fragments(
        tab,
        tmp_path,
        data_storage_version="stable",
    )
    assert len(fragments) == 1
    ds = lance.dataset(dataset_uri)
    assert "major_version: 2" in format_fragment(fragments[0], ds)


def test_mixed_fragment_versions(tmp_path):
    data = pa.table({"a": range(800), "b": range(800)})

    # Create empty v2 dataset
    ds = lance.write_dataset(
        data_obj=[],
        uri=tmp_path / "dataset2",
        schema=data.schema,
        data_storage_version="stable",
    )

    # Add one v1 file and one v2 file
    fragments = []
    fragments.append(
        lance.LanceFragment.create(ds.uri, data, data_storage_version="legacy")
    )
    fragments.append(
        lance.LanceFragment.create(ds.uri, data, data_storage_version="stable")
    )

    # Attempt to commit
    operation = lance.LanceOperation.Overwrite(ds.schema, fragments)
    with pytest.raises(OSError, match="All data files must have the same version"):
        lance.LanceDataset.commit(ds.uri, operation)


def test_create_from_file(tmp_path):
    data = pa.table({"a": range(800), "b": range(800)})
    dataset = lance.write_dataset(
        [], tmp_path, schema=data.schema, data_storage_version="stable"
    )

    # Append first file
    fragment_name = f"{uuid.uuid4()}.lance"
    with LanceFileWriter(str(tmp_path / "data" / fragment_name)) as writer:
        writer.write_batch(data)

    frag = LanceFragment.create_from_file(fragment_name, dataset, 0)
    op = LanceOperation.Append([frag])

    dataset = lance.LanceDataset.commit(dataset.uri, op, read_version=dataset.version)
    frag = dataset.get_fragments()[0]
    assert frag.fragment_id == 0

    assert dataset.count_rows() == 800

    # Append second file (fragment id shouldn't be 0 even though we pass in 0)
    fragment_name = f"{uuid.uuid4()}.lance"
    with LanceFileWriter(str(tmp_path / "data" / fragment_name)) as writer:
        writer.write_batch(data)

    frag = LanceFragment.create_from_file(fragment_name, dataset, 0)
    op = LanceOperation.Append([frag])

    dataset = lance.LanceDataset.commit(dataset.uri, op, read_version=dataset.version)
    frag = dataset.get_fragments()[1]
    assert frag.fragment_id == 1

    assert dataset.count_rows() == 1600

    # Simulate compaction
    compacted_name = f"{uuid.uuid4()}.lance"
    with LanceFileWriter(str(tmp_path / "data" / compacted_name)) as writer:
        for batch in dataset.to_batches():
            writer.write_batch(batch)

    frag = LanceFragment.create_from_file(compacted_name, dataset, 0)
    group = LanceOperation.RewriteGroup(
        old_fragments=[frag.metadata for frag in dataset.get_fragments()],
        new_fragments=[frag],
    )
    op = LanceOperation.Rewrite(groups=[group], rewritten_indices=[])
    dataset = lance.LanceDataset.commit(dataset.uri, op, read_version=dataset.version)

    assert dataset.count_rows() == 1600
    assert len(dataset.get_fragments()) == 1
    assert dataset.get_fragments()[0].fragment_id == 2


def test_fragment_merge(tmp_path):
    schema = pa.schema([pa.field("a", pa.string())])
    batches = pa.RecordBatchReader.from_batches(
        schema,
        [
            pa.record_batch([pa.array(["0" * 1024] * 1024 * 8)], names=["a"]),
            pa.record_batch([pa.array(["0" * 1024] * 1024 * 8)], names=["a"]),
        ],
    )

    progress = ProgressForTest()
    fragments = write_fragments(
        batches,
        tmp_path,
        max_rows_per_group=512,
        max_bytes_per_file=1024,
        progress=progress,
    )

    operation = lance.LanceOperation.Overwrite(schema, fragments)
    dataset = lance.LanceDataset.commit(tmp_path, operation)
    merged = []
    schema = None
    for fragment in dataset.get_fragments():
        table = fragment.scanner(with_row_id=True, columns=[]).to_table()
        table = table.add_column(0, "b", [[i for i in range(len(table))]])
        fragment, schema = fragment.merge(table, "_rowid")
        merged.append(fragment)

    merge = lance.LanceOperation.Merge(merged, schema)
    dataset = lance.LanceDataset.commit(
        tmp_path, merge, read_version=dataset.latest_version
    )

    merged = []
    schema = None
    for fragment in dataset.get_fragments():
        table = fragment.scanner(with_row_address=True, columns=[]).to_table()
        table = table.add_column(0, "c", [[i + 1 for i in range(len(table))]])
        fragment, schema = fragment.merge(table, "_rowaddr")
        merged.append(fragment)

    merge = lance.LanceOperation.Merge(merged, schema)
    dataset = lance.LanceDataset.commit(
        tmp_path, merge, read_version=dataset.latest_version
    )

    merged = []
    for fragment in dataset.get_fragments():
        table = fragment.scanner(columns=["b"]).to_table()
        table = table.add_column(0, "d", [[i + 2 for i in range(len(table))]])
        fragment, schema = fragment.merge(table, "b")
        merged.append(fragment)

    merge = lance.LanceOperation.Merge(merged, schema)
    dataset = lance.LanceDataset.commit(
        tmp_path, merge, read_version=dataset.latest_version
    )
    assert [f.name for f in dataset.schema] == ["a", "b", "c", "d"]


def test_fragment_count_rows(tmp_path: Path):
    data = pa.table({"a": range(800), "b": range(800)})
    ds = write_dataset(data, tmp_path)

    fragments = ds.get_fragments()
    assert len(fragments) == 1

    assert fragments[0].count_rows() == 800
    assert fragments[0].count_rows("a < 200") == 200
    assert fragments[0].count_rows(pc.field("a") < 200) == 200


@pytest.mark.parametrize("enable_stable_row_ids", [False, True])
def test_fragment_metadata_pickle(tmp_path: Path, enable_stable_row_ids: bool):
    ds = write_dataset(
        pa.table({"a": range(100)}),
        tmp_path,
        enable_stable_row_ids=enable_stable_row_ids,
    )
    # Create a deletion file
    ds.delete("a < 50")
    fragment = ds.get_fragments()[0]

    frag_meta = fragment.metadata

    assert frag_meta.deletion_file is not None
    if enable_stable_row_ids:
        assert frag_meta.row_id_meta is not None

    # Pickle and unpickle the fragment metadata
    round_trip = pickle.loads(pickle.dumps(frag_meta))
    assert frag_meta == round_trip

    # JSON round-trip
    json_data = frag_meta.to_json()
    json_round_trip = FragmentMetadata.from_json(json.dumps(json_data))
    assert frag_meta.id == json_round_trip.id
    assert frag_meta.physical_rows == json_round_trip.physical_rows
    if enable_stable_row_ids:
        assert json_round_trip.row_id_meta is not None


def test_deletion_file_with_base_id_serialization():
    """Test that DeletionFile with base_id serializes correctly."""
    from lance.fragment import DeletionFile, FragmentMetadata

    # Create a DeletionFile with base_id
    deletion_file = DeletionFile(
        read_version=1, id=123, file_type="array", num_deleted_rows=10, base_id=456
    )

    # Verify the base_id is set
    assert deletion_file.base_id == 456

    # Test asdict includes base_id
    deletion_dict = deletion_file.asdict()
    assert "base_id" in deletion_dict
    assert deletion_dict["base_id"] == 456

    # Create a FragmentMetadata with the deletion file
    metadata = FragmentMetadata(
        id=1, files=[], physical_rows=1000, deletion_file=deletion_file
    )

    # Test pickle serialization/deserialization
    pickled = pickle.dumps(metadata)
    unpickled = pickle.loads(pickled)

    # Verify the deletion file was correctly deserialized
    assert unpickled.deletion_file is not None
    assert unpickled.deletion_file.base_id == 456
    assert unpickled == metadata

    # Test JSON serialization/deserialization
    json_data = metadata.to_json()
    assert json_data["deletion_file"]["base_id"] == 456

    deserialized = FragmentMetadata.from_json(json.dumps(json_data))
    assert deserialized.deletion_file is not None
    assert deserialized.deletion_file.base_id == 456


def test_fragment_update_columns_basic(tmp_path):
    """Test basic fragment update columns functionality."""
    # Create initial dataset
    data = pa.table(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "value": [10, 20, 30, 40],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_basic"
    dataset = lance.write_dataset(data, dataset_uri)

    # Prepare update data with _rowid (must be UInt64 to match Lance's internal type)
    update_data = pa.table(
        {
            "_rowid": pa.array([0, 2], type=pa.uint64()),
            "name": ["Alice_Updated", "Charlie_Updated"],
            "value": [100, 300],
        }
    )

    # Get the fragment and update columns
    fragment = dataset.get_fragment(0)
    updated_fragment, fields_modified = fragment.update_columns(update_data)

    # Verify fields_modified is returned
    assert isinstance(fields_modified, list)
    assert len(fields_modified) > 0

    # Commit the changes using Update operation

    op = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    updated_dataset = lance.LanceDataset.commit(
        str(dataset_uri), op, read_version=dataset.version
    )

    # Verify the update
    result = updated_dataset.to_table().to_pydict()
    assert result["name"] == ["Alice_Updated", "Bob", "Charlie_Updated", "David"]
    assert result["value"] == [100, 20, 300, 40]
    assert result["id"] == [1, 2, 3, 4]  # id column should remain unchanged


def test_fragment_update_columns_with_custom_join_key(tmp_path):
    """Test fragment update columns with custom join key."""
    # Create initial dataset
    data = pa.table(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "score": [85, 90, 75, 80],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_custom_join_key"
    dataset = lance.write_dataset(data, dataset_uri)

    # Prepare update data using 'id' as join key
    # Note: We only update 'score', not 'id' itself
    update_data = pa.table(
        {
            "id": [1, 3],
            "name": ["Alan", "Chase"],
            "score": [95, 85],
        }
    )

    # Get the fragment and update columns
    fragment = dataset.get_fragment(0)
    updated_fragment, fields_modified = fragment.update_columns(
        update_data, left_on="id", right_on="id"
    )

    # Commit the changes

    op = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    updated_dataset = lance.LanceDataset.commit(
        str(dataset_uri), op, read_version=dataset.version
    )

    # Verify the update
    result = updated_dataset.to_table().to_pydict()
    assert result["score"][0] == 95  # id=1 should have score 95
    assert result["score"][2] == 85  # id=3 should have score 85
    assert result["name"][0] == "Alan"  # id=1 should have name Alan
    assert result["name"][2] == "Chase"  # id=3 should have name Chase


def test_fragment_update_columns_with_blob_v2(tmp_path):
    data = pa.table(
        {
            "id": pa.array([1, 2, 3, 4]),
            "payload": lance.blob_array([b"one", b"two", b"", None]),
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_blob_v2"
    dataset = lance.write_dataset(
        data,
        dataset_uri,
        data_storage_version="2.2",
    )

    fragment = dataset.get_fragment(0)
    updated_fragment, fields_modified = fragment.update_columns(
        pa.table(
            {
                "id": pa.array([2]),
                "payload": lance.blob_array([b"NEW"]),
            }
        ),
        left_on="id",
    )

    operation = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    updated_dataset = LanceDataset.commit(
        dataset_uri,
        operation,
        read_version=dataset.version,
    )

    result = updated_dataset.to_table(blob_handling="all_binary")
    assert result["id"].to_pylist() == [1, 2, 3, 4]
    assert result["payload"].to_pylist() == [b"one", b"NEW", b"", None]


def test_fragment_update_columns_with_nested_blob_v2(tmp_path):
    def info_array(names, payloads):
        fields = [pa.field("name", pa.string()), lance.blob_field("blob")]
        return pa.StructArray.from_arrays(
            [pa.array(names), lance.blob_array(payloads)], fields=fields
        )

    dataset_uri = tmp_path / "test_dataset_update_columns_nested_blob_v2"
    dataset = lance.write_dataset(
        pa.table(
            {
                "id": pa.array([1, 2]),
                "info": info_array(["a", "b"], [b"one", b"two"]),
            }
        ),
        dataset_uri,
        data_storage_version="2.2",
    )

    updated_fragment, fields_modified = dataset.get_fragment(0).update_columns(
        pa.table(
            {
                "id": pa.array([2]),
                "info": info_array(["B"], [b"NEW"]),
            }
        ),
        left_on="id",
    )
    updated_dataset = LanceDataset.commit(
        dataset_uri,
        LanceOperation.Update(
            updated_fragments=[updated_fragment],
            fields_modified=fields_modified,
        ),
        read_version=dataset.version,
    )

    info = updated_dataset.to_table(blob_handling="all_binary")["info"].combine_chunks()
    assert info.field("name").to_pylist() == ["a", "B"]
    assert info.field("blob").to_pylist() == [b"one", b"NEW"]


def test_fragment_update_columns_preserves_external_blob_v2(tmp_path):
    dataset_uri = tmp_path / "test_dataset_update_columns_external_blob_v2"
    external = tmp_path / "existing-payload.bin"
    external.write_bytes(b"outside")
    dataset = lance.write_dataset(
        pa.table(
            {
                "id": pa.array([1, 2]),
                "payload": lance.blob_array([external.as_uri(), b"two"]),
            }
        ),
        dataset_uri,
        data_storage_version="2.2",
        allow_external_blob_outside_bases=True,
    )

    updated_fragment, fields_modified = dataset.get_fragment(0).update_columns(
        pa.table(
            {
                "id": pa.array([2]),
                "payload": lance.blob_array([b"NEW"]),
            }
        ),
        left_on="id",
    )
    updated_dataset = LanceDataset.commit(
        dataset_uri,
        LanceOperation.Update(
            updated_fragments=[updated_fragment],
            fields_modified=fields_modified,
        ),
        read_version=dataset.version,
    )

    result = updated_dataset.to_table(blob_handling="all_binary")
    assert result["payload"].to_pylist() == [b"outside", b"NEW"]

    new_external = tmp_path / "new-payload.bin"
    new_external.write_bytes(b"new outside")
    with pytest.raises(ValueError, match="outside registered external bases"):
        updated_dataset.get_fragment(0).update_columns(
            pa.table(
                {
                    "id": pa.array([2]),
                    "payload": lance.blob_array([new_external.as_uri()]),
                }
            ),
            left_on="id",
        )


def test_fragment_update_columns_with_nulls(tmp_path):
    """Test fragment update columns with null values."""
    # Create initial dataset
    data = pa.table(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "optional_field": ["A", "B", "C", "D"],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_nulls"
    dataset = lance.write_dataset(data, dataset_uri)

    # Prepare update data with null values
    update_data = pa.table(
        {
            "_rowid": pa.array([1, 3], type=pa.uint64()),
            "optional_field": [None, "D_Updated"],
        }
    )

    # Get the fragment and update columns
    fragment = dataset.get_fragment(0)
    updated_fragment, fields_modified = fragment.update_columns(update_data)

    # Commit the changes

    op = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    updated_dataset = lance.LanceDataset.commit(
        str(dataset_uri), op, read_version=dataset.version
    )

    # Verify the update
    result = updated_dataset.to_table().to_pydict()
    assert result["optional_field"] == ["A", None, "C", "D_Updated"]


def test_fragment_update_columns_partial_update(tmp_path):
    """Test updating only some columns."""
    # Create initial dataset with multiple columns
    data = pa.table(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "SF"],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_partial_update"
    dataset = lance.write_dataset(data, dataset_uri)

    # Update only 'age' column, leaving 'name' and 'city' unchanged
    update_data = pa.table(
        {
            "_rowid": pa.array([0, 2], type=pa.uint64()),
            "age": [26, 36],
        }
    )

    # Get the fragment and update columns
    fragment = dataset.get_fragment(0)
    updated_fragment, fields_modified = fragment.update_columns(update_data)

    # Commit the changes

    op = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    updated_dataset = lance.LanceDataset.commit(
        str(dataset_uri), op, read_version=dataset.version
    )

    # Verify only age was updated
    result = updated_dataset.to_table().to_pydict()
    assert result["age"] == [26, 30, 36]
    assert result["name"] == ["Alice", "Bob", "Charlie"]  # Unchanged
    assert result["city"] == ["NYC", "LA", "SF"]  # Unchanged


def test_fragment_update_columns_with_offsets_rewrite_columns(tmp_path):
    """update_columns(with_offsets=True) returns matched row offsets that a
    rewrite_columns commit uses to refresh row version metadata for the
    matched rows only."""
    data = pa.table(
        {
            "id": list(range(8)),
            "value": [i * 10 for i in range(8)],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_with_offsets"
    dataset = lance.write_dataset(
        data, dataset_uri, max_rows_per_file=4, enable_stable_row_ids=True
    )
    assert len(dataset.get_fragments()) == 2

    # Update two rows of the second fragment, joining on a user column.
    fragment = dataset.get_fragments()[1]
    update_data = pa.table({"id": [5, 7], "value": [500, 700]})
    updated_fragment, fields_modified, matched_offsets = fragment.update_columns(
        update_data, left_on="id", with_offsets=True
    )
    # Portable RoaringBitmap serialization of {1, 3}: ids 5 and 7 sit at those
    # physical offsets within the second fragment.
    assert matched_offsets == (
        b"\x3a\x30\x00\x00\x01\x00\x00\x00\x00\x00\x01\x00"
        b"\x10\x00\x00\x00\x01\x00\x03\x00"
    )

    op = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
        update_mode="rewrite_columns",
        updated_fragment_offsets={updated_fragment.id: matched_offsets},
    )
    updated_dataset = lance.LanceDataset.commit(
        str(dataset_uri), op, read_version=dataset.version
    )

    result = updated_dataset.to_table(
        columns=["id", "value", "_row_last_updated_at_version"]
    ).sort_by("id")
    assert result["value"].to_pylist() == [0, 10, 20, 30, 40, 500, 60, 700]
    # Only the matched rows carry the commit's version.
    versions = result["_row_last_updated_at_version"].to_pylist()
    assert versions == [1, 1, 1, 1, 1, 2, 1, 2]


def test_fragment_update_columns_no_match(tmp_path):
    """Test update when no rows match the join condition."""
    # Create initial dataset
    data = pa.table(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_no_match"
    dataset = lance.write_dataset(data, dataset_uri)

    # Update data with non-existent _rowid
    update_data = pa.table(
        {
            "_rowid": pa.array(
                [100, 200], type=pa.uint64()
            ),  # These rowids don't exist
            "value": [999, 888],
        }
    )

    # Get the fragment and update columns
    fragment = dataset.get_fragment(0)
    updated_fragment, fields_modified, matched_offsets = fragment.update_columns(
        update_data, with_offsets=True
    )
    assert matched_offsets == b"\x3a\x30\x00\x00\x00\x00\x00\x00"

    # Commit the changes

    op = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    updated_dataset = lance.LanceDataset.commit(
        str(dataset_uri), op, read_version=dataset.version
    )

    # Verify nothing was updated (fallback to original values)
    result = updated_dataset.to_table().to_pydict()
    assert result["value"] == [10, 20, 30]  # Unchanged


def test_fragment_update_columns_error_on_nonexistent_column(tmp_path):
    """Test that updating a non-existent column raises an error."""
    # Create initial dataset
    data = pa.table(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_error_on_nonexistent_column"
    dataset = lance.write_dataset(data, dataset_uri)

    # Try to update a column that doesn't exist
    update_data = pa.table(
        {
            "_rowid": pa.array([0, 1], type=pa.uint64()),
            "nonexistent_column": [100, 200],
        }
    )

    fragment = dataset.get_fragment(0)

    # Should raise an error
    with pytest.raises(Exception) as exc_info:
        fragment.update_columns(update_data)

    assert "does not exist" in str(exc_info.value).lower()


def test_fragment_update_columns_error_on_metadata_column(tmp_path):
    """Test that updating metadata columns raises an error."""
    # Create initial dataset
    data = pa.table(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
        }
    )
    dataset_uri = tmp_path / "test_dataset_update_columns_error_on_metadata_column"
    dataset = lance.write_dataset(data, dataset_uri)

    # Try to update _rowid column (metadata column)
    update_data = pa.table(
        {
            "_rowid": pa.array([0, 1], type=pa.uint64()),
            "_rowaddr": pa.array([999, 888], type=pa.uint64()),  # This should fail
        }
    )

    fragment = dataset.get_fragment(0)

    # Should raise an error
    with pytest.raises(Exception) as exc_info:
        fragment.update_columns(update_data)

    assert (
        "metadata column" in str(exc_info.value).lower()
        or "cannot be updated" in str(exc_info.value).lower()
    )


def test_fragment_delete_rows(tmp_path: Path):
    # LanceFragment.delete_rows deletes by local row offset (not a predicate):
    # exactly the given rows are removed, the rest survive, and the fragment is
    # gone when every row is deleted.
    data = pa.table({"a": range(100), "b": range(100)})
    dataset = lance.write_dataset(data, tmp_path)
    frag = dataset.get_fragment(0)

    new_meta = frag.delete_rows([0, 5, 99])
    assert new_meta is not None
    op = LanceOperation.Delete(
        updated_fragments=[new_meta],
        deleted_fragment_ids=[],
        predicate="delete_rows([0, 5, 99])",
    )
    dataset = lance.LanceDataset.commit(dataset.uri, op, read_version=dataset.version)

    assert dataset.count_rows() == 97
    remaining = set(dataset.to_table(columns=["a"])["a"].to_pylist())
    assert {0, 5, 99}.isdisjoint(remaining)
    assert {1, 98}.issubset(remaining)

    # Deleting every row removes the fragment entirely (returns None).
    frag = lance.dataset(tmp_path).get_fragment(0)
    assert frag.delete_rows(range(100)) is None

    # Offsets outside the fragment's physical rows are rejected, not silently
    # written to the deletion file.
    frag = lance.dataset(tmp_path).get_fragment(0)
    with pytest.raises(ValueError, match="out of range"):
        frag.delete_rows([100])
    with pytest.raises(ValueError, match="out of range"):
        frag.delete_rows([0, 50, 1000])


def test_fragment_take_with_json_column(tmp_path):
    """Test that FileFragment.take returns JSON columns in Arrow JSON format."""
    json_type = pa.json_()
    data = pa.table(
        {
            "id": pa.array(range(10), type=pa.int64()),
            "meta": pa.array(
                [f'{{"val":{i}}}' for i in range(10)],
                type=json_type,
            ),
        }
    )
    dataset_uri = tmp_path / "test_frag_take_json"
    dataset = lance.write_dataset(data, dataset_uri)

    fragment = dataset.get_fragment(0)
    result = fragment.take([1, 4, 7])

    # Should return arrow.json type (Utf8), not lance.json (LargeBinary)
    meta_field = result.schema.field("meta")
    assert meta_field.type == pa.utf8() or meta_field.type == pa.json_(), (
        f"Expected arrow.json (Utf8), got {meta_field.type}"
    )

    metas = result.column("meta").to_pylist()
    assert metas[0] == '{"val":1}'
    assert metas[1] == '{"val":4}'
    assert metas[2] == '{"val":7}'


def test_fragment_create_with_json_column(tmp_path):
    """Test that LanceFragment.create works with Arrow JSON extension type.

    Previously the single-fragment create path skipped the Arrow JSON (Utf8) ->
    Lance JSON (JSONB LargeBinary) conversion that write_dataset/write_fragments
    perform, so the raw UTF-8 string bytes were written into a column whose schema
    declared JSONB. Reads then miss-decoded the bytes and returned garbage.
    """
    json_type = pa.json_()
    data = pa.table(
        {
            "uid": pa.array(["a", "b", "c", "d"], type=pa.utf8()),
            "payload": pa.array(
                ['{"x":1}', '{"x":2}', '{"y":3}', '{"y":4}'],
                type=json_type,
            ),
        }
    )

    frag = LanceFragment.create(tmp_path, data)
    operation = LanceOperation.Overwrite(data.schema, [frag])
    dataset = LanceDataset.commit(tmp_path, operation)

    result = dataset.to_table()
    assert result.column("uid").to_pylist() == ["a", "b", "c", "d"]
    payloads = result.column("payload").to_pylist()
    assert [json.loads(p) for p in payloads] == [
        {"x": 1},
        {"x": 2},
        {"y": 3},
        {"y": 4},
    ]


def test_fragment_update_columns_with_json_column(tmp_path):
    """Test that fragment update_columns works with Arrow JSON extension type.

    Previously this would fail with a type mismatch error because the
    HashJoiner didn't convert Arrow JSON (Utf8) to Lance JSON (LargeBinary).
    """
    # Create initial dataset with a JSON extension type column
    json_type = pa.json_()
    data = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
            "name": pa.array(["a", "b", "c", "d", "e"], type=pa.utf8()),
            "meta": pa.array(
                ['{"x":1}', '{"x":2}', '{"x":3}', '{"x":4}', '{"x":5}'],
                type=json_type,
            ),
        }
    )
    dataset_uri = tmp_path / "test_update_cols_json"
    dataset = lance.write_dataset(data, dataset_uri)

    # Prepare update data: update the JSON column for some rows
    update_data = pa.table(
        {
            "_rowid": pa.array([1, 3], type=pa.uint64()),
            "meta": pa.array(
                ['{"updated":true,"id":2}', '{"updated":true,"id":4}'],
                type=json_type,
            ),
        }
    )

    # This should NOT raise a type mismatch error
    fragment = dataset.get_fragment(0)
    updated_fragment, fields_modified = fragment.update_columns(update_data)

    assert len(fields_modified) > 0

    # Commit and verify
    op = LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    updated_dataset = lance.LanceDataset.commit(
        str(dataset_uri), op, read_version=dataset.version
    )

    result = updated_dataset.to_table()
    ids = result.column("id").to_pylist()
    metas = result.column("meta").to_pylist()

    for i, (id_val, meta_val) in enumerate(zip(ids, metas)):
        meta = json.loads(meta_val) if isinstance(meta_val, str) else meta_val
        if id_val == 2 or id_val == 4:
            assert "updated" in meta_val or meta.get("updated") is True, (
                f"id={id_val} should be updated, got {meta_val}"
            )
        else:
            assert "x" in meta_val or "x" in str(meta), (
                f"id={id_val} should have original value, got {meta_val}"
            )


def test_row_id_sequence_from_range():
    # A step-of-one range is the compact case and must not materialize its ids.
    sequence = RowIdSequence(range(10))

    assert len(sequence) == 10
    assert sequence.to_pyarrow() == pa.array(range(10), type=pa.uint64())
    assert sequence.to_pyarrow().type == pa.uint64()
    assert list(sequence) == list(range(10))


@pytest.mark.parametrize(
    "row_ids",
    [
        pytest.param(pa.array([1, 2, 3]), id="pyarrow_array"),
        pytest.param(pa.array([1, 2, 3], type=pa.uint64()), id="pyarrow_uint64_array"),
        pytest.param(pa.array([1, 2, 3], type=pa.int8()), id="pyarrow_int8_array"),
        pytest.param(pa.array([1, 2, 3], type=pa.uint16()), id="pyarrow_uint16_array"),
        # A slice carries an offset into a larger buffer; only the slice counts.
        pytest.param(pa.array([9, 1, 2, 3, 9]).slice(1, 3), id="pyarrow_sliced_array"),
        pytest.param(pa.chunked_array([[1], [2, 3]]), id="pyarrow_chunked_array"),
        pytest.param((x for x in [1, 2, 3]), id="generator"),
        pytest.param([1, 2, 3], id="list"),
        pytest.param(range(1, 4), id="range"),
        pytest.param(range(3, 0, -1), id="descending_range"),
    ],
)
def test_row_id_sequence_accepts_input_types(row_ids):
    sequence = RowIdSequence(row_ids)

    assert sorted(sequence) == [1, 2, 3]


@pytest.mark.parametrize(
    "row_ids",
    [
        pytest.param([], id="empty_list"),
        pytest.param(range(0), id="empty_range"),
        pytest.param(pa.array([], type=pa.uint64()), id="empty_array"),
    ],
)
def test_row_id_sequence_empty(row_ids):
    sequence = RowIdSequence(row_ids)

    assert len(sequence) == 0
    assert list(sequence) == []
    assert sequence.to_pyarrow() == pa.array([], type=pa.uint64())


@pytest.mark.parametrize(
    "row_ids",
    [
        pytest.param(list(range(4100)), id="contiguous"),
        pytest.param(list(range(0, 8200, 2)), id="gapped"),
        pytest.param(list(range(4100))[::-1], id="unsorted"),
    ],
)
def test_row_id_sequence_iterates_large_sequences(row_ids):
    # Each shape picks a different segment encoding, so all of them have to
    # round-trip through iteration.
    sequence = RowIdSequence(row_ids)

    assert list(sequence) == row_ids
    # Each call must hand back a fresh iterator rather than a spent one.
    assert list(sequence) == row_ids


def test_row_id_sequence_from_range_above_isize():
    # Range bounds are read as isize; beyond that the values are read one at a
    # time instead, which must still cover the whole uint64 row id domain.
    start = 2**63
    sequence = RowIdSequence(range(start, start + 3))

    assert list(sequence) == [start, start + 1, start + 2]


def test_row_id_sequence_unsorted_round_trips():
    sequence = RowIdSequence([12, 11, 10])

    assert list(sequence) == [12, 11, 10]
    assert sequence.to_pyarrow() == pa.array([12, 11, 10], type=pa.uint64())


@pytest.mark.parametrize(
    "row_ids",
    [
        pytest.param([1, 1, 2], id="adjacent"),
        pytest.param([1, 2, 3, 1], id="separated"),
        pytest.param(pa.array([5, 3, 5]), id="unsorted_array"),
    ],
)
def test_row_id_sequence_rejects_duplicates(row_ids):
    with pytest.raises(ValueError, match="Row ids must be unique"):
        RowIdSequence(row_ids)


@pytest.mark.parametrize(
    ("row_ids", "message"),
    [
        pytest.param(pa.array([1, None, 3]), "must not be null", id="null_in_array"),
        pytest.param(pa.array([1.5, 2.5]), "array of integers", id="float_array"),
        pytest.param(pa.array([-1, 2]), "uint64", id="negative_in_array"),
        pytest.param(range(-5, 5), "non-negative", id="negative_range"),
        pytest.param(5, "iterable of integers", id="not_iterable"),
    ],
)
def test_row_id_sequence_rejects_invalid_input(row_ids, message):
    with pytest.raises((ValueError, TypeError), match=message):
        RowIdSequence(row_ids)


def test_row_id_sequence_metadata_round_trip():
    sequence = RowIdSequence([7, 12, 3])

    metadata = sequence.to_inline_metadata()
    assert isinstance(metadata, RowIdMeta)
    assert RowIdSequence.from_inline_metadata(metadata) == sequence


def test_row_id_sequence_equality_and_repr():
    assert RowIdSequence(range(3)) == RowIdSequence([0, 1, 2])
    assert RowIdSequence(range(3)) != RowIdSequence([0, 1])
    # Comparing against an unrelated type is False rather than an error.
    assert RowIdSequence(range(3)) != "not a sequence"

    assert repr(RowIdSequence([1, 2])) == "RowIdSequence([1, 2])"
    assert repr(RowIdSequence(range(12))) == (
        "RowIdSequence([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...], len=12)"
    )


def test_row_id_sequence_pickle():
    sequence = RowIdSequence([7, 12, 3])

    assert pickle.loads(pickle.dumps(sequence)) == sequence


def _row_ids_by_id(dataset: LanceDataset) -> dict:
    table = dataset.to_table(columns=["id"], with_row_id=True)
    return dict(zip(table["id"].to_pylist(), table["_rowid"].to_pylist()))


def test_row_id_sequence_preserves_ids_in_manual_update(tmp_path: Path):
    # An update assembled externally (delete the old row, append a replacement)
    # keeps the row's identity only if the new fragment carries its row id.
    dataset = write_dataset(
        pa.table({"id": [1, 2, 3, 4], "v": [10, 20, 30, 40]}),
        tmp_path,
        max_rows_per_file=2,
        enable_stable_row_ids=True,
    )
    row_ids_before = _row_ids_by_id(dataset)

    updated_fragment = dataset.get_fragments()[0].delete("id = 2")
    (new_fragment,) = write_fragments(pa.table({"id": [2], "v": [99]}), tmp_path)
    new_fragment.row_id_meta = RowIdSequence([row_ids_before[2]]).to_inline_metadata()

    dataset = LanceDataset.commit(
        tmp_path,
        LanceOperation.Update(
            removed_fragment_ids=[],
            updated_fragments=[updated_fragment],
            new_fragments=[new_fragment],
            fields_modified=[],
        ),
        read_version=dataset.version,
    )

    assert _row_ids_by_id(dataset) == row_ids_before
    assert dataset.to_table().sort_by("id").to_pydict() == {
        "id": [1, 2, 3, 4],
        "v": [10, 99, 30, 40],
    }


def test_row_id_sequence_reads_back_fragment_metadata(tmp_path: Path):
    dataset = write_dataset(
        pa.table({"a": range(10)}),
        tmp_path,
        max_rows_per_file=5,
        enable_stable_row_ids=True,
    )

    sequences = [
        RowIdSequence.from_inline_metadata(fragment.metadata.row_id_meta)
        for fragment in dataset.get_fragments()
    ]

    assert [list(sequence) for sequence in sequences] == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ]


def test_fragment_validate(tmp_path: Path):
    dataset = write_dataset(
        pa.table({"a": range(100), "b": range(100)}),
        tmp_path,
        max_rows_per_file=50,
    )
    # A valid fragment validates without raising.
    for fragment in dataset.get_fragments():
        assert fragment.validate() is None


def test_fragment_validate_across_data_files(tmp_path: Path):
    # add_columns writes a second data file per fragment; validate must still
    # pass (field ids increasing and unique across a fragment's data files).
    dataset = write_dataset(pa.table({"a": range(100)}), tmp_path, max_rows_per_file=50)
    dataset.add_columns({"b": "a + 1"})
    for fragment in dataset.get_fragments():
        assert len(fragment.data_files()) > 1
        fragment.validate()


def test_fragment_validate_after_delete(tmp_path: Path):
    dataset = write_dataset(pa.table({"a": range(100)}), tmp_path, max_rows_per_file=50)
    dataset.delete("a < 10")
    # A fragment carrying a deletion vector still validates.
    for fragment in dataset.get_fragments():
        fragment.validate()


def _dataset_with_scalar_index(tmp_path: Path) -> LanceDataset:
    dataset = write_dataset(
        pa.table({"val": range(10000), "other": range(10000)}),
        tmp_path,
        max_rows_per_file=5000,
    )
    dataset.create_scalar_index("val", index_type="BTREE")
    return dataset


def test_fragment_scanner_use_scalar_index_disables_index_query(tmp_path: Path):
    # A filtered fragment scan on an indexed column plans a dataset-wide
    # ScalarIndexQuery (and caches its index pages) unless the scan opts out.
    dataset = _dataset_with_scalar_index(tmp_path)
    fragment = dataset.get_fragments()[0]
    filt = "val >= 10 AND val <= 20"

    default_plan = fragment.scanner(filter=filt, with_row_id=True).explain_plan(True)
    assert "ScalarIndexQuery" in default_plan

    opted_out_plan = fragment.scanner(
        filter=filt, with_row_id=True, use_scalar_index=False
    ).explain_plan(True)
    assert "ScalarIndexQuery" not in opted_out_plan


@pytest.mark.parametrize("use_scalar_index", [None, True, False])
def test_fragment_scanner_matches_dataset_scanner(tmp_path: Path, use_scalar_index):
    # The fragment scanner must build the same plan as the dataset scanner
    # restricted to that single fragment.
    dataset = _dataset_with_scalar_index(tmp_path)
    fragment = dataset.get_fragments()[0]
    filt = "val >= 10 AND val <= 20"

    frag_plan = fragment.scanner(
        filter=filt, with_row_id=True, use_scalar_index=use_scalar_index
    ).explain_plan(True)
    dataset_plan = dataset.scanner(
        fragments=[fragment],
        filter=filt,
        with_row_id=True,
        use_scalar_index=use_scalar_index,
    ).explain_plan(True)
    assert frag_plan == dataset_plan


def _fragment_with_deletions(tmp_path: Path) -> LanceFragment:
    dataset = write_dataset(pa.table({"a": range(20)}), tmp_path, max_rows_per_file=10)
    dataset.delete("a < 3")
    return dataset.get_fragments()[0]


def test_fragment_scanner_include_deleted_rows(tmp_path: Path):
    fragment = _fragment_with_deletions(tmp_path)
    assert fragment.physical_rows == 10
    assert fragment.num_deletions == 3

    # By default the deleted rows are omitted.
    default = fragment.to_table(with_row_id=True)
    assert default.num_rows == 7
    assert default["a"].to_pylist() == list(range(3, 10))

    # With include_deleted_rows the deleted rows are surfaced with a null _rowid.
    included = fragment.scanner(with_row_id=True, include_deleted_rows=True).to_table()
    assert included.num_rows == fragment.physical_rows
    assert included["a"].to_pylist() == list(range(10))
    assert included["_rowid"].null_count == fragment.num_deletions


def test_fragment_scanner_include_deleted_rows_requires_row_id(tmp_path: Path):
    fragment = _fragment_with_deletions(tmp_path)
    with pytest.raises(ValueError, match="with_row_id"):
        fragment.scanner(include_deleted_rows=True).to_table()


def test_fragment_scanner_include_deleted_rows_matches_dataset_scanner(tmp_path: Path):
    dataset = write_dataset(pa.table({"a": range(20)}), tmp_path, max_rows_per_file=10)
    dataset.delete("a < 3")
    fragment = dataset.get_fragments()[0]

    frag_plan = fragment.scanner(
        with_row_id=True, include_deleted_rows=True
    ).explain_plan(True)
    dataset_plan = dataset.scanner(
        fragments=[fragment], with_row_id=True, include_deleted_rows=True
    ).explain_plan(True)
    assert frag_plan == dataset_plan


@pytest.mark.parametrize(
    ("late_materialization", "is_late"),
    [
        pytest.param(None, False, id="default"),
        pytest.param(True, True, id="all_late"),
        pytest.param(False, False, id="all_early"),
        pytest.param(["values"], True, id="late_column"),
        pytest.param(["filter"], False, id="early_column"),
    ],
)
def test_fragment_scanner_late_materialization(
    tmp_path: Path, late_materialization, is_late
):
    # With no index, the plan shows whether `values` is fetched late (a take over
    # the row stream) or early (materialized in the scan projection).
    dataset = write_dataset(
        pa.table({"filter": range(2000), "values": range(2000)}),
        tmp_path,
        data_storage_version="stable",
    )
    fragment = dataset.get_fragments()[0]

    plan = fragment.scanner(
        filter="filter % 2 == 0", late_materialization=late_materialization
    ).explain_plan(True)

    if is_late:
        assert "projection=[values], source=stream" in plan
    else:
        assert "projection=[filter, values]" in plan


def test_fragment_scanner_rejects_invalid_late_materialization(tmp_path: Path):
    dataset = write_dataset(pa.table({"a": range(10)}), tmp_path)
    fragment = dataset.get_fragments()[0]

    with pytest.raises(
        ValueError, match="late_materialization must be a bool or a list of strings"
    ):
        fragment.scanner(late_materialization=123)


def test_fragment_scanner_io_buffer_size_forwarded(tmp_path: Path):
    # io_buffer_size has no plan-visible marker, so assert it is accepted through
    # both scan entry points and leaves results unchanged.
    dataset = write_dataset(pa.table({"val": range(1000)}), tmp_path)
    fragment = dataset.get_fragments()[0]
    filt = "val < 100"
    expected = fragment.to_table(filter=filt)

    assert fragment.to_table(filter=filt, io_buffer_size=4 * 1024 * 1024) == expected

    batched = pa.Table.from_batches(
        list(fragment.to_batches(filter=filt, io_buffer_size=4 * 1024 * 1024))
    )
    assert batched == expected


def test_fragment_scanner_strict_batch_size(tmp_path: Path):
    dataset = write_dataset(pa.table({"a": range(1000)}), tmp_path)
    fragment = dataset.get_fragments()[0]
    filt = "a % 3 == 0"

    # A filtered scan emits uneven, sub-batch_size batches by default.
    loose = [b.num_rows for b in fragment.to_batches(batch_size=100, filter=filt)]
    assert any(n < 100 for n in loose[:-1])

    # strict_batch_size coalesces to exactly batch_size (except the last batch).
    strict = [
        b.num_rows
        for b in fragment.to_batches(
            batch_size=100, filter=filt, strict_batch_size=True
        )
    ]
    assert all(n == 100 for n in strict[:-1])
    assert sum(strict) == sum(loose)


def test_fragment_scanner_batch_size_bytes(tmp_path: Path):
    # A small byte budget over wide rows forces many more batches than the
    # default, without changing the results.
    dataset = write_dataset(pa.table({"s": ["x" * 1024] * 2000}), tmp_path)
    fragment = dataset.get_fragments()[0]

    default_batches = list(fragment.to_batches())
    small_budget = list(fragment.to_batches(batch_size_bytes=64 * 1024))
    assert len(small_budget) > len(default_batches)
    assert pa.Table.from_batches(small_budget) == fragment.to_table()
