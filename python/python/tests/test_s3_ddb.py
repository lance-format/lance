# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""
Integration tests with S3 and DynamoDB. Also used to test storage_options are
passed correctly.

See DEVELOPMENT.md under heading "Integration Tests" for more information.
"""

import copy
import time
import uuid
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import lance
import pyarrow as pa
import pytest
from lance.file import LanceFileReader, LanceFileSession, LanceFileWriter
from lance.fragment import write_fragments

# These are all keys that are accepted by storage_options
CONFIG = {
    "allow_http": "true",
    "aws_access_key_id": "ACCESS_KEY",
    "aws_secret_access_key": "SECRET_KEY",
    "aws_endpoint": "http://localhost:4566",
    "dynamodb_endpoint": "http://localhost:4566",
    "aws_region": "us-east-1",
}


def get_boto3_client(*args, **kwargs):
    import boto3

    return boto3.client(
        *args,
        region_name=CONFIG["aws_region"],
        aws_access_key_id=CONFIG["aws_access_key_id"],
        aws_secret_access_key=CONFIG["aws_secret_access_key"],
        **kwargs,
    )


@pytest.fixture(scope="module")
def s3_bucket():
    s3 = get_boto3_client("s3", endpoint_url=CONFIG["aws_endpoint"])
    bucket_name = "lance-integtest"
    # if bucket exists, delete it
    try:
        delete_bucket(s3, bucket_name)
    except s3.exceptions.NoSuchBucket:
        pass
    s3.create_bucket(Bucket=bucket_name)
    yield bucket_name

    delete_bucket(s3, bucket_name)


def delete_bucket(s3, bucket_name):
    # Delete all objects first
    for obj in s3.list_objects(Bucket=bucket_name).get("Contents", []):
        s3.delete_object(Bucket=bucket_name, Key=obj["Key"])
    s3.delete_bucket(Bucket=bucket_name)


@pytest.fixture(scope="module")
def ddb_table():
    dynamodb = get_boto3_client("dynamodb", endpoint_url=CONFIG["dynamodb_endpoint"])
    table_name = "lance-integtest"
    # if table exists, delete it
    try:
        dynamodb.delete_table(TableName=table_name)
        # smh dynamodb is async
        time.sleep(0.5)
    except dynamodb.exceptions.ResourceNotFoundException:
        pass
    dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "base_uri", "KeyType": "HASH"},
            {"AttributeName": "version", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "base_uri", "AttributeType": "S"},
            {"AttributeName": "version", "AttributeType": "N"},
        ],
        ProvisionedThroughput={"ReadCapacityUnits": 10, "WriteCapacityUnits": 10},
    )

    time.sleep(1)
    yield table_name
    dynamodb.delete_table(TableName=table_name)


@pytest.mark.integration
@pytest.mark.parametrize("use_env", [True, False])
def test_s3_ddb_create_and_append(
    s3_bucket: str, ddb_table: str, use_env: bool, monkeypatch
):
    # Test with and without environment variables, so we can validate it works
    # either way you provide them.
    storage_options = copy.deepcopy(CONFIG)
    if use_env:
        for key, value in storage_options.items():
            monkeypatch.setenv(key.upper(), value)
        storage_options = None

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_dir = f"s3+ddb://{s3_bucket}/{table_name}?ddbTableName={ddb_table}"
    ds = lance.write_dataset(table1, table_dir, storage_options=storage_options)
    assert len(ds.versions()) == 1

    table2 = pa.Table.from_pylist([{"a": 100, "b": 2000}])

    # can detect existing dataset
    with pytest.raises(OSError, match="Dataset already exists"):
        lance.write_dataset(table2, table_dir, storage_options=storage_options)

    ds = lance.write_dataset(
        table2, table_dir, mode="append", storage_options=storage_options
    )

    assert len(ds.versions()) == 2
    assert ds.count_rows() == 3

    # can checkout
    ds = ds.checkout_version(1)
    assert ds.count_rows() == 2
    ds = ds.checkout_version(2)
    assert ds.count_rows() == 3

    with pytest.raises(ValueError, match="Not found"):
        ds.checkout_version(3)


@pytest.mark.integration
def test_s3_ddb_concurrent_commit(
    s3_bucket: str,
    ddb_table: str,
    monkeypatch,
):
    for key, value in CONFIG.items():
        monkeypatch.setenv(key.upper(), value)

    table = pa.Table.from_pylist([{"a": -1}])
    table_name = uuid.uuid4().hex
    table_dir = f"s3+ddb://{s3_bucket}/{table_name}?ddbTableName={ddb_table}"
    lance.write_dataset(table, table_dir)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futs = [
            executor.submit(
                lance.write_dataset,
                pa.Table.from_pylist([{"a": i}]),
                table_dir,
                mode="append",
            )
            for i in range(5)
        ]
        # surface any errors -- shouldn't be any
        [result.result() for result in futures.as_completed(futs)]

    assert len(lance.dataset(table_dir).versions()) == 6
    assert lance.dataset(table_dir).count_rows() == 6

    assert sorted(
        [item["a"] for item in lance.dataset(table_dir).to_table().to_pylist()]
    ) == [-1, 0, 1, 2, 3, 4]


@pytest.mark.integration
def test_s3_ddb_concurrent_commit_more_than_five(
    s3_bucket: str, ddb_table: str, monkeypatch
):
    for key, value in CONFIG.items():
        monkeypatch.setenv(key.upper(), value)

    table = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_dir = f"s3+ddb://{s3_bucket}/{table_name}?ddbTableName={ddb_table}"
    lance.write_dataset(table, table_dir)

    failures = 0
    total_futures = 10

    # force the tests to start at the same time
    barrier = Barrier(10, timeout=5)

    def writh_dataset_with_start_barrier():
        barrier.wait()
        lance.write_dataset(table, table_dir, mode="append")

    with ThreadPoolExecutor(max_workers=6) as executor:
        futs = [
            executor.submit(writh_dataset_with_start_barrier)
            for _ in range(total_futures)
        ]
        for result in futures.as_completed(futs):
            try:
                result.result()
            except:  # noqa: E722,PERF203
                failures += 1

    assert failures > 0

    expected_version = total_futures - failures + 1

    assert len(lance.dataset(table_dir).versions()) == expected_version
    assert lance.dataset(table_dir).count_rows() == expected_version * 2


@pytest.mark.integration
def test_s3_ddb_branches(s3_bucket: str, ddb_table: str):
    """Branches on a table committed through the DynamoDB external manifest
    store.

    The DDB store keys version chains by base uri, so each branch chain must
    get its own entries via its branch-qualified path. Both chains are given
    the same version number with diverged data so a wrong-chain resolution
    cannot pass silently.
    """
    storage_options = copy.deepcopy(CONFIG)
    table_name = uuid.uuid4().hex
    table_dir = f"s3+ddb://{s3_bucket}/{table_name}?ddbTableName={ddb_table}"

    # main: v1 (a=1), v2 (a=2)
    lance.write_dataset(
        pa.Table.from_pylist([{"a": 1}]), table_dir, storage_options=storage_options
    )
    ds = lance.write_dataset(
        pa.Table.from_pylist([{"a": 2}]),
        table_dir,
        mode="append",
        storage_options=storage_options,
    )

    # Fork "dev" at v2 and commit on it, then diverge main to the same
    # version number.
    dev = ds.create_branch("dev", 2)
    dev = lance.write_dataset(
        pa.Table.from_pylist([{"a": 3}]),
        dev,
        mode="append",
        storage_options=storage_options,
    )
    ds = lance.write_dataset(
        pa.Table.from_pylist([{"a": 100}]),
        table_dir,
        mode="append",
        storage_options=storage_options,
    )

    assert sorted(dev.to_table()["a"].to_pylist()) == [1, 2, 3]
    assert sorted(ds.to_table()["a"].to_pylist()) == [1, 2, 100]

    # Cross-branch checkout at the overlapping version number resolves each
    # chain's own data.
    on_dev = ds.checkout_version(("dev", 3))
    assert sorted(on_dev.to_table()["a"].to_pylist()) == [1, 2, 3]
    back_on_main = dev.checkout_version(("main", None))
    assert sorted(back_on_main.to_table()["a"].to_pylist()) == [1, 2, 100]


@pytest.mark.integration
def test_s3_unsafe(s3_bucket: str):
    storage_options = copy.deepcopy(CONFIG)
    del storage_options["dynamodb_endpoint"]

    uri = f"s3://{s3_bucket}/test_unsafe"
    data = pa.table({"x": [1, 2, 3]})
    ds = lance.write_dataset(data, uri, storage_options=storage_options)

    assert len(ds.versions()) == 1
    assert ds.count_rows() == 3
    assert ds.to_table() == data


@pytest.mark.integration
def test_s3_ddb_distributed_commit(s3_bucket: str, ddb_table: str):
    table_name = uuid.uuid4().hex
    table_dir = f"s3+ddb://{s3_bucket}/{table_name}?ddbTableName={ddb_table}"

    schema = pa.schema([pa.field("a", pa.int64())])
    fragments = write_fragments(
        pa.table({"a": pa.array(range(1024))}),
        f"s3+ddb://{s3_bucket}/distributed_commit?ddbTableName={ddb_table}",
        storage_options=CONFIG,
    )
    operation = lance.LanceOperation.Overwrite(schema, fragments)
    ds = lance.LanceDataset.commit(table_dir, operation, storage_options=CONFIG)
    assert ds.count_rows() == 1024


@pytest.mark.integration
def test_file_writer_reader(s3_bucket: str):
    storage_options = copy.deepcopy(CONFIG)
    del storage_options["dynamodb_endpoint"]
    table = pa.table({"a": [1, 2, 3]})
    file_path = f"s3://{s3_bucket}/foo.lance"
    global_buffer_text = "hello"
    global_buffer_bytes = bytes(global_buffer_text, "utf-8")
    with LanceFileWriter(str(file_path), storage_options=storage_options) as writer:
        writer.write_batch(table)
        global_buffer_pos = writer.add_global_buffer(global_buffer_bytes)
    reader = LanceFileReader(str(file_path), storage_options=storage_options)
    assert reader.read_all().to_table() == table
    assert reader.metadata().global_buffers[global_buffer_pos].size == len(
        global_buffer_bytes
    )
    assert (
        bytes(reader.read_global_buffer(global_buffer_pos)).decode()
        == global_buffer_text
    )

    # The writer reports the size of the object it just wrote, so callers do not
    # need a second client to stat it.
    s3 = get_boto3_client("s3", endpoint_url=CONFIG["aws_endpoint"])
    head = s3.head_object(Bucket=s3_bucket, Key="foo.lance")
    assert writer.size_bytes == head["ContentLength"]


@pytest.mark.integration
def test_file_writer_size_bytes_multipart(s3_bucket: str):
    # Large enough to exceed the single-PUT threshold, so the write goes through
    # the multipart path where the reported size is filled in on completion.
    storage_options = copy.deepcopy(CONFIG)
    del storage_options["dynamodb_endpoint"]
    table = pa.table({"a": pa.array(range(4 * 1024 * 1024), type=pa.int64())})
    file_path = f"s3://{s3_bucket}/multipart.lance"
    with LanceFileWriter(str(file_path), storage_options=storage_options) as writer:
        writer.write_batch(table)

    s3 = get_boto3_client("s3", endpoint_url=CONFIG["aws_endpoint"])
    head = s3.head_object(Bucket=s3_bucket, Key="multipart.lance")
    assert head["ContentLength"] > 5 * 1024 * 1024
    assert writer.size_bytes == head["ContentLength"]


@pytest.mark.integration
def test_file_session_upload_download(s3_bucket: str, tmp_path):
    storage_options = copy.deepcopy(CONFIG)
    del storage_options["dynamodb_endpoint"]

    session_uri = f"s3://{s3_bucket}/test_session"
    session = LanceFileSession(session_uri, storage_options=storage_options)

    # Create a local file to upload
    local_file = tmp_path / "test_upload.txt"
    test_content = "Hello from LanceFileSession!"
    local_file.write_text(test_content)

    # Test upload_file
    session.upload_file(str(local_file), "uploaded.txt")

    # Test contains - file should exist after upload
    assert session.contains("uploaded.txt"), "File should exist after upload"
    assert not session.contains("nonexistent.txt"), "Nonexistent file should not exist"

    # Upload another file to test list
    local_file2 = tmp_path / "test_upload2.txt"
    local_file2.write_text("Second file")
    session.upload_file(str(local_file2), "subdir/nested.txt")

    # Test list - should see both files
    files = session.list()
    assert "uploaded.txt" in files, f"uploaded.txt should be in list: {files}"
    assert "subdir/nested.txt" in files, f"subdir/nested.txt should be in list: {files}"

    # Test list with prefix
    subdir_files = session.list("subdir")
    assert len(subdir_files) == 1, f"Should have 1 file in subdir: {subdir_files}"
    assert "subdir/nested.txt" in subdir_files

    # Test download_file
    download_path = tmp_path / "downloaded.txt"
    session.download_file("uploaded.txt", str(download_path))

    # Verify downloaded content matches
    assert download_path.read_text() == test_content, "Downloaded content should match"

    # Test downloading nested file
    download_nested = tmp_path / "downloaded_nested.txt"
    session.download_file("subdir/nested.txt", str(download_nested))
    assert download_nested.read_text() == "Second file"


@pytest.mark.integration
def test_append_fragment(s3_bucket: str):
    storage_options = copy.deepcopy(CONFIG)
    table = pa.table({"a": [1, 2], "b": ["a", "b"]})
    lance.fragment.LanceFragment.create(
        f"s3://{s3_bucket}/test_append.lance", table, storage_options=storage_options
    )


@pytest.mark.integration
def test_s3_drop(s3_bucket: str):
    storage_options = copy.deepcopy(CONFIG)
    table_name = uuid.uuid4().hex
    tmp_path = f"s3://{s3_bucket}/{table_name}.lance"
    table = pa.table({"x": [0]})
    dataset = lance.write_dataset(table, tmp_path, storage_options=storage_options)
    dataset.validate()
    lance.LanceDataset.drop(tmp_path, storage_options=storage_options)
