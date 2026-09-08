# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import threading

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from lance.file import LanceFileReader, LanceFileSession, LanceFileWriter


def test_file_read_projection(tmp_path):
    table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    path = tmp_path / "foo.lance"
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(table)

    reader = LanceFileReader(str(path), columns=["a"])
    assert reader.read_all().to_table() == table.select("a")


def test_file_writer(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    with LanceFileWriter(str(path), schema) as writer:
        writer.write_batch(pa.table({"a": [1, 2, 3]}))
    reader = LanceFileReader(str(path))
    metadata = reader.metadata()
    assert metadata.num_rows == 3


def test_write_no_schema(tmp_path):
    path = tmp_path / "foo.lance"
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(pa.table({"a": [1, 2, 3]}))
    reader = LanceFileReader(str(path))
    assert reader.read_all().to_table() == pa.table({"a": [1, 2, 3]})


def test_no_schema_no_data(tmp_path):
    path = tmp_path / "foo.lance"
    with pytest.raises(
        ValueError, match="Schema is unknown and file cannot be created"
    ):
        with LanceFileWriter(str(path)) as _:
            pass


def test_schema_only(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    with LanceFileWriter(str(path), schema=schema) as _:
        pass
    reader = LanceFileReader(str(path))
    assert reader.metadata().schema == schema


def test_write_with_max_page_bytes(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    for version in ["2.0", "2.1", "2.2", "2.3"]:
        with LanceFileWriter(
            str(path), schema, max_page_bytes=1, version=version
        ) as writer:
            writer.write_batch(pa.table({"a": [1, 2, 3]}))
        reader = LanceFileReader(str(path))
        # Only 2.0 splits large pages on write.   In 2.1+ we split on read.
        expected_pages = 3 if version == "2.0" else 1
        assert len(reader.metadata().columns[0].pages) == expected_pages


def test_aborted_write(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    writer = LanceFileWriter(str(path), schema)
    writer.write_batch(pa.table({"a": [1, 2, 3]}))
    del writer
    assert not path.exists()


def test_multiple_close(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    writer = LanceFileWriter(str(path), schema)
    writer.write_batch(pa.table({"a": [1, 2, 3]}))
    writer.close()
    size_bytes = writer.size_bytes
    # The second close is a no-op and must not clear the recorded size
    writer.close()
    assert writer.size_bytes == size_bytes


@pytest.mark.parametrize("num_rows", [0, 3])
def test_size_bytes(tmp_path, num_rows):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    writer = LanceFileWriter(str(path), schema)
    assert writer.size_bytes is None
    writer.write_batch(pa.table({"a": list(range(num_rows))}))
    assert writer.close() == num_rows
    # Even an empty file has a footer, so the size is always positive
    assert writer.size_bytes > 0
    assert writer.size_bytes == os.path.getsize(path)


def test_size_bytes_with_session(tmp_path):
    session = LanceFileSession(tmp_path)
    with session.open_writer("foo.lance") as writer:
        writer.write_batch(pa.table({"a": [1, 2, 3]}))
    assert writer.size_bytes == os.path.getsize(tmp_path / "foo.lance")


def test_version(tmp_path):
    schema = pa.schema([pa.field("a", pa.int64())])
    cases = [
        ("foo.lance", "2.0", (0, 3)),
        ("foo2.lance", "2.1", (2, 1)),
        ("foo3.lance", "2.2", (2, 2)),
        ("foo4.lance", "2.3", (2, 3)),
    ]

    for file_name, version, (major, minor) in cases:
        path = tmp_path / file_name
        with LanceFileWriter(str(path), schema, version=version) as writer:
            writer.write_batch(pa.table({"a": [1, 2, 3]}))
        reader = LanceFileReader(str(path))
        metadata = reader.metadata()
        assert metadata.major_version == major
        assert metadata.minor_version == minor


def test_take(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    writer = LanceFileWriter(str(path), schema)
    writer.write_batch(pa.table({"a": [i for i in range(100)]}))
    writer.close()

    reader = LanceFileReader(str(path))
    # Can't read out of range
    with pytest.raises(ValueError):
        reader.take_rows([0, 100]).to_table()

    table = reader.take_rows([0, 77, 83]).to_table()
    assert table == pa.table({"a": [0, 77, 83]})


def test_num_rows(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    writer = LanceFileWriter(str(path), schema)
    writer.write_batch(pa.table({"a": [i for i in range(100)]}))
    writer.close()

    reader = LanceFileReader(str(path))
    assert reader.num_rows() == 100


def check_round_trip(tmp_path, table):
    path = tmp_path / "foo.lance"
    with LanceFileWriter(str(path), table.schema) as writer:
        for batch in table.to_batches():
            writer.write_batch(pa.table(batch))
    reader = LanceFileReader(str(path))
    result = reader.read_all().to_table()
    assert result == table


def test_different_types(tmp_path):
    dict_values = pa.array(["foo", "bar", "baz"], pa.string())
    dict_indices = pa.array([2, 1, 0], pa.uint8())

    check_round_trip(
        tmp_path,
        pa.table(
            {
                "large_string": pa.array(["foo", "bar", "baz"], pa.large_string()),
                "large_binary": pa.array([b"foo", b"bar", b"baz"], pa.large_binary()),
                "dict_string": pa.DictionaryArray.from_arrays(
                    dict_indices, dict_values
                ),
            }
        ),
    )


def test_with_nulls(tmp_path):
    check_round_trip(
        tmp_path,
        pa.table(
            {
                "some_null_1": pa.array([1, 2, None], pa.int64()),
                "some_null_2": pa.array([None, None, 3], pa.int64()),
                "nullable_list": pa.array(
                    [[1, 2], None, [None, 3]], pa.list_(pa.int64())
                ),
                "nullable_fsl": pa.array(
                    [[1, 2], None, [None, 3]], pa.list_(pa.int64(), 2)
                ),
                "all_null": pa.array([None, None, None], pa.int64()),
                "null_strings": pa.array([None, "foo", None], pa.string()),
            }
        ),
    )


def test_batch_sizes(tmp_path):
    # Need a big string so there aren't too many rows per page because we
    # want to test different page sizes:
    #  - batch that spans multiple pages (including more than 2)
    #  - batch that is smaller than a page (including much smaller)
    my_str = b"0" * 299593

    data = [[my_str] for _ in range(1009)]
    tab = pa.table({"val": data})

    path = str(tmp_path / "foo.lance")
    with LanceFileWriter(path) as writer:
        writer.write_batch(tab)

    reader = LanceFileReader(path)

    for batch_size in range(10, 1050, 10):
        reader.read_all(batch_size=batch_size).to_table()


def test_round_trip(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    data = pa.table({"a": [1, 2, 3]})
    with LanceFileWriter(str(path), schema) as writer:
        writer.write_batch(data)
    reader = LanceFileReader(str(path))
    result = reader.read_all().to_table()
    assert result == data

    # TODO: Currently fails, need to fix reader
    # result = reader.read_range(1, 1).to_table()
    # assert result == pa.table({"a": [2]})

    # TODO: Test reading invalid ranges
    # TODO: Test invalid batch sizes


def test_metadata(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema([pa.field("a", pa.int64())])
    data = pa.table({"a": [1, 2, 3]})
    with LanceFileWriter(str(path), schema) as writer:
        writer.write_batch(data)
    reader = LanceFileReader(str(path))
    metadata = reader.metadata()

    assert metadata.schema == schema
    assert metadata.num_rows == 3
    assert metadata.num_global_buffer_bytes > 0
    assert metadata.num_column_metadata_bytes > 0
    assert metadata.num_data_bytes > 0
    assert len(metadata.columns) == 1

    column = metadata.columns[0]
    assert len(column.column_buffers) == 0
    assert len(column.pages) == 1

    page = column.pages[0]
    assert len(page.buffers) > 0
    for buffer in page.buffers:
        assert buffer.position % 64 == 0
        assert buffer.size > 0

    assert len(page.encoding) > 0


def test_file_stat(tmp_path):
    path = tmp_path / "foo.lance"
    schema = pa.schema(
        [pa.field("a", pa.int64()), pa.field("b", pa.list_(pa.float64(), 8))]
    )

    num_rows = 1_000_000

    data1 = pa.array(range(num_rows))

    # Create a fixed-size list of float64 with dimension 8
    fixed_size_list = [np.random.rand(8).tolist() for _ in range(num_rows)]
    data2 = pa.array(fixed_size_list, type=pa.list_(pa.float64(), 8))

    with LanceFileWriter(str(path), schema) as writer:
        writer.write_batch(pa.table({"a": data1, "b": data2}))
    reader = LanceFileReader(str(path))
    file_stat = reader.file_statistics()

    assert len(file_stat.columns) == 2

    assert file_stat.columns[0].num_pages == 1
    assert file_stat.columns[0].size_bytes <= 8_000_000

    # 2 pages on 2.0, 1 page in 2.1+
    assert file_stat.columns[1].num_pages <= 2
    # Slightly larger than 64MiB because of padding, chunk overhead, etc.
    assert file_stat.columns[1].size_bytes <= 64_200_000


def test_round_trip_parquet(tmp_path):
    pq_path = tmp_path / "foo.parquet"
    table = pa.table({"int": [1, 2], "list_str": [["x", "yz", "abc"], ["foo", "bar"]]})
    pq.write_table(table, str(pq_path))
    table = pq.read_table(str(pq_path))

    lance_path = tmp_path / "foo.lance"
    with LanceFileWriter(str(lance_path)) as writer:
        writer.write_batch(table)

    reader = LanceFileReader(str(lance_path))
    round_tripped = reader.read_all().to_table()
    assert round_tripped == table


def test_write_read_with_session(tmp_path):
    session = LanceFileSession(tmp_path)
    with session.open_writer("foo.lance") as writer:
        writer.write_batch(pa.table({"a": [1, 2, 3]}))

    with session.open_writer("bar.lance") as writer:
        writer.write_batch(pa.table({"a": [4, 5, 6]}))

    reader = session.open_reader("foo.lance")
    assert reader.read_all().to_table() == pa.table({"a": [1, 2, 3]})

    reader = session.open_reader("bar.lance")
    assert reader.read_all().to_table() == pa.table({"a": [4, 5, 6]})


def test_list_field_name(tmp_path):
    weird_field = pa.field("why does this name even exist", pa.string())
    weird_string_type = pa.list_(weird_field)
    schema = pa.schema([pa.field("list_str", weird_string_type)])
    table = pa.table({"list_str": [["x", "yz", "abc"], ["foo", "bar"]]}, schema=schema)

    path = tmp_path / "foo.lance"
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(table)

    reader = LanceFileReader(str(path))
    round_tripped = reader.read_all().to_table()

    assert round_tripped == table
    assert round_tripped.schema.field("list_str").type == weird_string_type


def test_field_meta(tmp_path):
    schema = pa.schema(
        [
            pa.field("primitive", pa.int64(), metadata={"foo": "bar"}),
            pa.field(
                "list",
                pa.list_(pa.field("item", pa.int64(), metadata={"list": "yes"})),
                metadata={"foo": "baz"},
            ),
            pa.field(
                "struct",
                pa.struct([pa.field("a", pa.int64(), metadata={"struct": "yes"})]),
                metadata={"foo": "qux"},
            ),
        ]
    )
    table = pa.table(
        {
            "primitive": [1, 2, 3],
            "list": [[1, 2], [3, 4], [5, 6]],
            "struct": [{"a": 1}, {"a": 2}, {"a": 3}],
        },
        schema=schema,
    )

    with LanceFileWriter(str(tmp_path / "foo.lance")) as writer:
        writer.write_batch(table)

    reader = LanceFileReader(str(tmp_path / "foo.lance"))
    round_tripped = reader.read_all().to_table()

    assert round_tripped == table


def test_dictionary(tmp_path):
    # Basic round trip
    dictionary = pa.array(["foo", "bar", "baz"], pa.string())
    indices = pa.array([0, 1, 2, 0, 1, 2], pa.int32())
    dict_arr = pa.DictionaryArray.from_arrays(indices, dictionary)

    def round_trip(arr):
        table = pa.table({"dict": arr})

        path = tmp_path / "foo.lance"
        with LanceFileWriter(str(path)) as writer:
            writer.write_batch(table)

        reader = LanceFileReader(str(path))
        table2 = reader.read_all().to_table()
        return table2.column("dict").chunk(0)

    round_tripped = round_trip(dict_arr)

    assert round_tripped == dict_arr
    assert round_tripped.type == dict_arr.type

    # Dictionary that doesn't use all values
    dictionary = pa.array(["foo", "bar", "baz"], pa.string())
    indices = pa.array([0, 0, 1, 1], pa.int32())
    dict_arr = pa.DictionaryArray.from_arrays(indices, dictionary)

    round_tripped = round_trip(dict_arr)

    assert round_tripped.dictionary == dictionary

    # different indices types
    dictionary = pa.array(["foo", "bar", "baz"], pa.string())
    for data_type in [
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
    ]:
        indices = pa.array([0, 1, 2, 0, 1, 2], data_type)
        dict_arr = pa.DictionaryArray.from_arrays(indices, dictionary)
        round_tripped = round_trip(dict_arr)
        assert round_tripped == dict_arr
        assert round_tripped.type == dict_arr.type


def test_empty_structs(tmp_path):
    schema = pa.schema([pa.field("empties", pa.struct([]))])
    table = pa.table({"empties": [{}] * 3}, schema=schema)
    path = tmp_path / "foo.lance"
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(table)
    reader = LanceFileReader(str(path))
    round_tripped = reader.read_all().to_table()
    assert round_tripped == table


def test_write_read_global_buffer(tmp_path):
    table = pa.table({"a": [1, 2, 3]})
    path = tmp_path / "foo.lance"
    global_buffer_text = "hello"
    global_buffer_bytes = bytes(global_buffer_text, "utf-8")
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(table)
        global_buffer_pos = writer.add_global_buffer(global_buffer_bytes)
    reader = LanceFileReader(str(path))
    assert reader.read_all().to_table() == table
    assert reader.metadata().global_buffers[global_buffer_pos].size == len(
        global_buffer_bytes
    )
    assert (
        bytes(reader.read_global_buffer(global_buffer_pos)).decode()
        == global_buffer_text
    )


def test_write_read_additional_schema_metadata(tmp_path):
    table = pa.table({"a": [1, 2, 3]})
    path = tmp_path / "foo.lance"
    schema_metadata_key = "foo"
    schema_metadata_value = "bar"
    with LanceFileWriter(str(path)) as writer:
        writer.write_batch(table)
        writer.add_schema_metadata(schema_metadata_key, schema_metadata_value)
    reader = LanceFileReader(str(path))
    assert reader.read_all().to_table() == table
    assert (
        reader.metadata().schema.metadata.get(schema_metadata_key.encode()).decode()
        == schema_metadata_value
    )


def test_writer_maintains_order(tmp_path):
    row_ids = list(range(8))
    payloads = ["0123456789abcdef" * (64 * 1024)] + [
        f"page-{row_id}" for row_id in row_ids[1:]
    ]
    table = pa.table({"payload": payloads, "row_id": row_ids})
    path = tmp_path / "ordered-pages.lance"

    # Before #2836, the seven cheap pages could finish encoding before the
    # expensive first page and be written out of order.
    with LanceFileWriter(
        str(path),
        table.schema,
        version="2.0",
        data_cache_bytes=1,
        max_page_bytes=64 * 1024,
    ) as writer:
        writer.write_batch(table)

    reader = LanceFileReader(str(path))
    assert [len(column.pages) for column in reader.metadata().columns] == [8, 1]
    assert reader.read_all().to_table() == table


def test_compression(tmp_path):
    # 10Ki strings, which should be highly compressible, but not eligible for dictionary
    compressible_strings = [f"compress_me_please-{i}" for i in range(10 * 1024)]
    table_default = pa.table({"compressible_strings": compressible_strings})

    schema_compress = pa.schema(
        [
            pa.field(
                "compressible_strings",
                pa.string(),
                metadata={"lance-encoding:compression": "zstd"},
            )
        ]
    )
    table_compress = pa.table(
        {"compressible_strings": compressible_strings}, schema=schema_compress
    )

    with LanceFileWriter(str(tmp_path / "default.lance")) as writer:
        writer.write_batch(table_default)

    with LanceFileWriter(str(tmp_path / "compress.lance"), schema_compress) as writer:
        writer.write_batch(table_compress)

    size_default = os.path.getsize(tmp_path / "default.lance")
    size_compress = os.path.getsize(tmp_path / "compress.lance")

    assert size_compress < size_default


def test_blob(tmp_path):
    # 100 1MiB values.  If we store as regular large_binary we end up
    # with several pages of values.  If we store as a blob we get a
    # single page
    expected = pa.table(
        {"val": pa.array([b"0" * (1024 * 1024)] * 100, pa.large_binary())}
    )
    schema_no_blob = pa.schema([pa.field("val", pa.large_binary())])
    schema_blob = pa.schema(
        [pa.field("val", pa.large_binary(), metadata={"lance-encoding:blob": "true"})]
    )

    path = tmp_path / "no_blob.lance"
    with LanceFileWriter(str(path), schema_no_blob) as writer:
        for _ in range(100):
            vals = pa.array([b"0" * (1024 * 1024)], pa.large_binary())
            writer.write_batch(pa.table({"val": vals}))

    reader = LanceFileReader(str(path))
    assert len(reader.metadata().columns[0].pages) > 1
    assert reader.read_all().to_table() == expected

    path = tmp_path / "blob.lance"
    with LanceFileWriter(str(path), schema_blob) as writer:
        for _ in range(100):
            vals = pa.array([b"0" * (1024 * 1024)], pa.large_binary())
            writer.write_batch(pa.table({"val": vals}))

    reader = LanceFileReader(str(path))
    assert len(reader.metadata().columns[0].pages) == 1

    actual = reader.read_all().to_table()

    assert actual.num_rows == expected.num_rows
    for row_num in range(expected.num_rows):
        actual_bytes = actual.column("val").chunk(0)[row_num].as_py()
        expected_bytes = expected.column("val").chunk(0)[row_num].as_py()
        assert len(actual_bytes) == len(expected_bytes)
        assert actual_bytes == expected_bytes


def test_multithreaded_writer(tmp_path):
    """Test concurrent multi-threaded writing to the same LanceFileWriter"""
    path = tmp_path / "multithreaded.lance"
    schema = pa.schema(
        [
            pa.field("thread_id", pa.int64()),
            pa.field("value", pa.int64()),
            pa.field("data", pa.string()),
        ]
    )

    # Used to store all written data for subsequent validation
    all_data = []
    data_lock = threading.Lock()

    def write_thread_data(thread_id, writer, num_records):
        """Function for individual thread to write data"""
        thread_data = []
        for i in range(num_records):
            record = {
                "thread_id": thread_id,
                "value": thread_id * 1000 + i,
                "data": f"thread_{thread_id}_record_{i}",
            }
            thread_data.append(record)

        # Create pyarrow table
        table = pa.table(
            {
                "thread_id": [r["thread_id"] for r in thread_data],
                "value": [r["value"] for r in thread_data],
                "data": [r["data"] for r in thread_data],
            }
        )

        # Write data
        writer.write_batch(table)

        # Record written data for validation
        with data_lock:
            all_data.extend(thread_data)

    # Test parameters
    num_threads = 5
    records_per_thread = 100

    # Create writer and start multi-threaded writing
    with LanceFileWriter(str(path), schema) as writer:
        threads = []

        # Start multiple threads
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=write_thread_data, args=(thread_id, writer, records_per_thread)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    # Validate written data
    reader = LanceFileReader(str(path))
    result_table = reader.read_all().to_table()

    # Check if total row count is correct
    expected_total_rows = num_threads * records_per_thread
    assert result_table.num_rows == expected_total_rows, (
        f"Expected {expected_total_rows} rows, got {result_table.num_rows}"
    )

    # Check data content correctness (order may differ, but data should be complete)
    # Convert results to dictionary list for comparison
    result_data = [
        {
            "thread_id": result_table.column("thread_id").chunk(0)[i].as_py(),
            "value": result_table.column("value").chunk(0)[i].as_py(),
            "data": result_table.column("data").chunk(0)[i].as_py(),
        }
        for i in range(result_table.num_rows)
    ]

    # Verify all data exists (order not considered)
    all_data_sorted = sorted(all_data, key=lambda x: (x["thread_id"], x["value"]))
    result_data_sorted = sorted(result_data, key=lambda x: (x["thread_id"], x["value"]))

    assert len(all_data_sorted) == len(result_data_sorted)

    # Compare data item by item
    for expected, actual in zip(all_data_sorted, result_data_sorted):
        assert expected == actual, f"Data mismatch: expected {expected}, got {actual}"

    # Verify data from each thread exists
    thread_ids_in_result = set(result_table.column("thread_id").chunk(0).to_pylist())
    expected_thread_ids = set(range(num_threads))
    assert thread_ids_in_result == expected_thread_ids

    # Verify record count for each thread
    for thread_id in range(num_threads):
        thread_rows = result_table.filter(
            pc.equal(result_table.column("thread_id"), thread_id)
        )
        assert thread_rows.num_rows == records_per_thread


def test_session_list_all_files(tmp_path):
    """Test that LanceFileSession.list() returns all files with relative paths"""
    session = LanceFileSession(str(tmp_path))
    schema = pa.schema([pa.field("x", pa.int64())])

    # Write files at different levels
    with session.open_writer("file1.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [1]}))

    with session.open_writer("file2.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [2]}))

    with session.open_writer("subdir/file3.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [3]}))

    with session.open_writer("subdir/file4.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [4]}))

    with session.open_writer("other/file5.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [5]}))

    # List all files
    files = sorted(session.list())

    # Verify relative paths (no absolute paths)
    assert files == [
        "file1.lance",
        "file2.lance",
        "other/file5.lance",
        "subdir/file3.lance",
        "subdir/file4.lance",
    ]

    # Verify no absolute paths
    for f in files:
        assert not f.startswith("/")
        assert str(tmp_path) not in f


def test_session_list_with_prefix(tmp_path):
    """Test that LanceFileSession.list() filters by prefix correctly"""
    session = LanceFileSession(str(tmp_path))
    schema = pa.schema([pa.field("x", pa.int64())])

    # Write files in different directories
    with session.open_writer("file1.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [1]}))

    with session.open_writer("subdir/file2.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [2]}))

    with session.open_writer("subdir/file3.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [3]}))

    with session.open_writer("other/file4.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [4]}))

    # List with prefix "subdir"
    subdir_files = sorted(session.list("subdir"))
    assert subdir_files == ["subdir/file2.lance", "subdir/file3.lance"]

    # List with prefix "other"
    other_files = sorted(session.list("other"))
    assert other_files == ["other/file4.lance"]

    # List with non-existent prefix
    empty = session.list("nonexistent")
    assert empty == []


def test_session_list_with_trailing_slash(tmp_path):
    """Test that LanceFileSession.list() handles trailing slashes correctly"""
    session = LanceFileSession(str(tmp_path))
    schema = pa.schema([pa.field("x", pa.int64())])

    with session.open_writer("dir/file.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [1]}))

    # Both with and without trailing slash should work
    files_no_slash = session.list("dir")
    files_with_slash = session.list("dir/")

    assert files_no_slash == files_with_slash
    assert files_no_slash == ["dir/file.lance"]


def test_session_list_with_delimiter(tmp_path):
    """Test that LanceFileSession.list_with_delimiter() is non-recursive."""
    session = LanceFileSession(str(tmp_path))
    schema = pa.schema([pa.field("x", pa.int64())])

    # Two top-level files and two nested subtrees.
    with session.open_writer("file1.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [1]}))
    with session.open_writer("file2.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [2]}))
    with session.open_writer("subdir/file3.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [3]}))
    with session.open_writer("subdir/nested/file4.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [4]}))
    with session.open_writer("other/file5.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [5]}))

    # Listing the base path returns only the immediate children: the two
    # top-level files and the two child directories (not their contents).
    result = session.list_with_delimiter()
    assert sorted(result.common_prefixes) == ["other", "subdir"]
    assert sorted(result.objects) == ["file1.lance", "file2.lance"]

    # Listing a subdirectory descends exactly one level: the direct file and
    # the nested directory, but not the file inside the nested directory.
    subdir = session.list_with_delimiter("subdir")
    assert subdir.common_prefixes == ["subdir/nested"]
    assert subdir.objects == ["subdir/file3.lance"]

    # Trailing slash behaves the same as no trailing slash.
    assert session.list_with_delimiter("subdir/") == subdir

    # A non-existent prefix yields empty results rather than erroring.
    empty = session.list_with_delimiter("nonexistent")
    assert empty.common_prefixes == []
    assert empty.objects == []


def test_session_contains(tmp_path):
    """Test that LanceFileSession.contains() works correctly"""
    session = LanceFileSession(str(tmp_path))
    schema = pa.schema([pa.field("x", pa.int64())])

    # File doesn't exist yet
    assert not session.contains("test.lance")

    # Write a file
    with session.open_writer("test.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [1]}))

    # File exists now
    assert session.contains("test.lance")

    # Nested file
    with session.open_writer("subdir/nested.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [2]}))

    assert session.contains("subdir/nested.lance")
    assert not session.contains("subdir/nonexistent.lance")


def test_session_read_range(tmp_path):
    """Test that LanceFileSession.read_range() returns the requested bytes."""
    session = LanceFileSession(str(tmp_path))

    payload = bytes(range(256))
    local = tmp_path / "src.bin"
    local.write_bytes(payload)
    session.upload_file(str(local), "data/file.bin")

    # A range in the middle of the file.
    assert session.read_range("data/file.bin", 10, 5) == payload[10:15]
    # From the start.
    assert session.read_range("data/file.bin", 0, 4) == payload[0:4]
    # Up to the end.
    assert session.read_range("data/file.bin", 250, 6) == payload[250:256]
    # A zero-length read yields empty bytes.
    assert session.read_range("data/file.bin", 100, 0) == b""

    # Reading a missing object raises OSError (consistent with download_file).
    with pytest.raises(OSError):
        session.read_range("data/missing.bin", 0, 4)


def test_session_delete_file(tmp_path):
    """Test that LanceFileSession.delete_file() removes files and is idempotent."""
    session = LanceFileSession(str(tmp_path))
    schema = pa.schema([pa.field("x", pa.int64())])

    with session.open_writer("test.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [1]}))
    with session.open_writer("subdir/nested.lance", schema=schema) as writer:
        writer.write_batch(pa.table({"x": [2]}))

    # Deleting an existing file removes it.
    assert session.contains("test.lance")
    session.delete_file("test.lance")
    assert not session.contains("test.lance")

    # Nested paths work too.
    assert session.contains("subdir/nested.lance")
    session.delete_file("subdir/nested.lance")
    assert not session.contains("subdir/nested.lance")

    # Deleting a missing path raises OSError (consistent with download_file).
    with pytest.raises(OSError):
        session.delete_file("test.lance")
    with pytest.raises(OSError):
        session.delete_file("never_existed.lance")


def test_struct_null_regression():
    import lance

    # Create struct array where 2nd element is null
    tag_array = pa.array(["valid", "null_struct", "valid", "valid"])
    struct_array = pa.StructArray.from_arrays(
        [tag_array],
        fields=[pa.field("tag", pa.string(), nullable=True)],
        mask=pa.array([True, False, True, True]),  # False = null struct element
    )

    # Create list containing these structs
    offsets = pa.array([0, 4], type=pa.int32())
    list_array = pa.ListArray.from_arrays(offsets, struct_array)
    batch = pa.record_batch([pa.array([0]), list_array], names=["id", "value"])

    ds = lance.write_dataset(batch, "memory://", data_storage_version="2.2")
    ds.to_table()
