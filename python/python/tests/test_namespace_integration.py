# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Integration tests for Lance Namespace with S3 and credential refresh.

This test uses DirectoryNamespace with native ops_metrics and vend_input_storage_options
features to track API calls and test credential refresh mechanisms.

Tests are parameterized to run with both DirectoryNamespace and a CustomNamespace
wrapper to verify Python-Rust binding works correctly for custom implementations.

See DEVELOPMENT.md under heading "Integration Tests" for more information.
"""

import copy
import time
import uuid
from threading import Lock
from typing import Dict, Optional

import lance
import pyarrow as pa
import pytest
from lance.namespace import (
    DeclareTableRequest,
    DescribeTableRequest,
    DirectoryNamespace,
    LanceNamespace,
)
from lance_namespace import (
    CreateNamespaceRequest,
    CreateNamespaceResponse,
    CreateTableBranchRequest,
    CreateTableBranchResponse,
    CreateTableRequest,
    CreateTableResponse,
    CreateTableVersionRequest,
    CreateTableVersionResponse,
    DeclareTableResponse,
    DeregisterTableRequest,
    DeregisterTableResponse,
    DescribeNamespaceRequest,
    DescribeNamespaceResponse,
    DescribeTableResponse,
    DescribeTableVersionRequest,
    DescribeTableVersionResponse,
    DropNamespaceRequest,
    DropNamespaceResponse,
    DropTableRequest,
    DropTableResponse,
    ListNamespacesRequest,
    ListNamespacesResponse,
    ListTablesRequest,
    ListTablesResponse,
    ListTableVersionsRequest,
    ListTableVersionsResponse,
    NamespaceExistsRequest,
    NamespaceExistsResponse,
    RegisterTableRequest,
    RegisterTableResponse,
    TableExistsRequest,
    TableExistsResponse,
)


class CustomNamespace(LanceNamespace):
    """A custom namespace wrapper that delegates to DirectoryNamespace.

    This class verifies that the Python-Rust binding works correctly for
    custom namespace implementations that wrap the native DirectoryNamespace.
    All methods simply delegate to the underlying DirectoryNamespace instance.
    """

    def __init__(self, inner: DirectoryNamespace):
        self._inner = inner

    def namespace_id(self) -> str:
        return f"CustomNamespace[{self._inner.namespace_id()}]"

    def create_namespace(
        self, request: CreateNamespaceRequest
    ) -> CreateNamespaceResponse:
        return self._inner.create_namespace(request)

    def describe_namespace(
        self, request: DescribeNamespaceRequest
    ) -> DescribeNamespaceResponse:
        return self._inner.describe_namespace(request)

    def namespace_exists(
        self, request: NamespaceExistsRequest
    ) -> NamespaceExistsResponse:
        return self._inner.namespace_exists(request)

    def drop_namespace(self, request: DropNamespaceRequest) -> DropNamespaceResponse:
        return self._inner.drop_namespace(request)

    def list_namespaces(self, request: ListNamespacesRequest) -> ListNamespacesResponse:
        return self._inner.list_namespaces(request)

    def create_table(
        self, request: CreateTableRequest, data: bytes
    ) -> CreateTableResponse:
        return self._inner.create_table(request, data)

    def declare_table(self, request: DeclareTableRequest) -> DeclareTableResponse:
        return self._inner.declare_table(request)

    def describe_table(self, request: DescribeTableRequest) -> DescribeTableResponse:
        return self._inner.describe_table(request)

    def table_exists(self, request: TableExistsRequest) -> TableExistsResponse:
        return self._inner.table_exists(request)

    def drop_table(self, request: DropTableRequest) -> DropTableResponse:
        return self._inner.drop_table(request)

    def list_tables(self, request: ListTablesRequest) -> ListTablesResponse:
        return self._inner.list_tables(request)

    def register_table(self, request: RegisterTableRequest) -> RegisterTableResponse:
        return self._inner.register_table(request)

    def deregister_table(
        self, request: DeregisterTableRequest
    ) -> DeregisterTableResponse:
        return self._inner.deregister_table(request)

    def list_table_versions(
        self, request: ListTableVersionsRequest
    ) -> ListTableVersionsResponse:
        return self._inner.list_table_versions(request)

    def describe_table_version(
        self, request: DescribeTableVersionRequest
    ) -> DescribeTableVersionResponse:
        return self._inner.describe_table_version(request)

    def create_table_version(
        self, request: CreateTableVersionRequest
    ) -> CreateTableVersionResponse:
        return self._inner.create_table_version(request)

    def create_table_branch(
        self, request: CreateTableBranchRequest
    ) -> CreateTableBranchResponse:
        return self._inner.create_table_branch(request)

    def retrieve_ops_metrics(self) -> Optional[Dict[str, int]]:
        return self._inner.retrieve_ops_metrics()


def _wrap_if_custom(ns_client: DirectoryNamespace, use_custom: bool):
    """Wrap namespace client in CustomNamespace if use_custom is True."""
    if use_custom:
        return CustomNamespace(ns_client)
    return ns_client


# These are all keys that are accepted by storage_options
CONFIG = {
    "allow_http": "true",
    "aws_access_key_id": "ACCESS_KEY",
    "aws_secret_access_key": "SECRET_KEY",
    "aws_endpoint": "http://localhost:4566",
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
    bucket_name = "lance-namespace-integtest"
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
    try:
        for obj in s3.list_objects(Bucket=bucket_name).get("Contents", []):
            s3.delete_object(Bucket=bucket_name, Key=obj["Key"])
        s3.delete_bucket(Bucket=bucket_name)
    except Exception:
        pass


def create_tracking_namespace(
    bucket_name: str,
    storage_options: dict,
    credential_expires_in_seconds: int = 60,
    use_custom: bool = False,
    managed_versioning: bool = False,
):
    """Create a DirectoryNamespace with ops metrics and credential vending enabled.

    Uses native DirectoryNamespace features:
    - ops_metrics_enabled=true: Tracks API call counts via retrieve_ops_metrics()
    - vend_input_storage_options=true: Returns input storage options in responses
    - vend_input_storage_options_refresh_interval_millis: Adds expires_at_millis

    Args:
        bucket_name: S3 bucket name or local path
        storage_options: Storage options to pass through (credentials, endpoint, etc.)
        credential_expires_in_seconds: Interval in seconds for credential expiration
        use_custom: If True, wrap in CustomNamespace for testing custom implementations
        managed_versioning: If True, enable the manifest catalog so table versions
            are tracked by the namespace and commits route through
            create_table_version

    Returns:
        Tuple of (namespace_client, inner_namespace_client) where inner is always
        the DirectoryNamespace (used for metrics retrieval)
    """
    # Add refresh_offset_millis to storage options so that credentials are not
    # considered expired immediately. Set to 1 second (1000ms) so that refresh
    # checks work correctly with short-lived credentials in tests.
    storage_options_with_refresh = dict(storage_options)
    storage_options_with_refresh["refresh_offset_millis"] = "1000"

    dir_props = {f"storage.{k}": v for k, v in storage_options_with_refresh.items()}

    if bucket_name.startswith("/") or bucket_name.startswith("file://"):
        dir_props["root"] = f"{bucket_name}/namespace_root"
    else:
        dir_props["root"] = f"s3://{bucket_name}/namespace_root"

    # Enable ops metrics tracking
    dir_props["ops_metrics_enabled"] = "true"
    # Enable storage options vending
    dir_props["vend_input_storage_options"] = "true"
    # Set refresh interval in milliseconds
    dir_props["vend_input_storage_options_refresh_interval_millis"] = str(
        credential_expires_in_seconds * 1000
    )
    if managed_versioning:
        dir_props["manifest_enabled"] = "true"
        dir_props["table_version_tracking_enabled"] = "true"
        dir_props["table_version_storage_enabled"] = "true"

    inner_ns_client = DirectoryNamespace(**dir_props)
    ns_client = _wrap_if_custom(inner_ns_client, use_custom)
    return ns_client, inner_ns_client


def get_describe_call_count(namespace_client) -> int:
    """Get the number of describe_table calls made to the namespace client."""
    return namespace_client.retrieve_ops_metrics().get("describe_table", 0)


def get_declare_call_count(namespace_client) -> int:
    """Get the number of declare_table calls made to the namespace client."""
    return namespace_client.retrieve_ops_metrics().get("declare_table", 0)


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_namespace_open_dataset(s3_bucket: str, use_custom: bool):
    """Test creating and opening datasets through namespace with credential tracking."""
    storage_options = copy.deepcopy(CONFIG)

    ns_client, inner_ns_client = create_tracking_namespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
        use_custom=use_custom,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    assert get_declare_call_count(inner_ns_client) == 0
    assert get_describe_call_count(inner_ns_client) == 0

    ds = lance.write_dataset(
        table1,
        namespace_client=ns_client,
        table_id=table_id,
        mode="create",
        storage_options=storage_options,
    )
    assert len(ds.versions()) == 1
    assert ds.count_rows() == 2
    assert get_declare_call_count(inner_ns_client) == 1

    ds_from_ns_client = lance.dataset(
        namespace_client=ns_client,
        table_id=table_id,
        storage_options=storage_options,
    )

    # 1 describe call from lance.dataset() to get location
    assert get_describe_call_count(inner_ns_client) == 1
    assert ds_from_ns_client.count_rows() == 2
    result = ds_from_ns_client.to_table()
    assert result == table1

    # Test credential caching
    call_count_before_reads = get_describe_call_count(inner_ns_client)
    for _ in range(3):
        assert ds_from_ns_client.count_rows() == 2
    assert get_describe_call_count(inner_ns_client) == call_count_before_reads


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_namespace_with_refresh(s3_bucket: str, use_custom: bool):
    """Test credential refresh when credentials expire."""
    storage_options = copy.deepcopy(CONFIG)

    ns_client, inner_ns_client = create_tracking_namespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3,
        use_custom=use_custom,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    assert get_declare_call_count(inner_ns_client) == 0
    assert get_describe_call_count(inner_ns_client) == 0

    ds = lance.write_dataset(
        table1,
        namespace_client=ns_client,
        table_id=table_id,
        mode="create",
        storage_options=storage_options,
    )
    assert ds.count_rows() == 2
    assert get_declare_call_count(inner_ns_client) == 1

    ds_from_ns_client = lance.dataset(
        namespace_client=ns_client,
        table_id=table_id,
        storage_options=storage_options,
    )

    # 1 describe call from lance.dataset() to get location
    initial_call_count = get_describe_call_count(inner_ns_client)
    assert initial_call_count == 1
    assert ds_from_ns_client.count_rows() == 2
    result = ds_from_ns_client.to_table()
    assert result == table1

    call_count_after_initial_reads = get_describe_call_count(inner_ns_client)

    time.sleep(5)

    assert ds_from_ns_client.count_rows() == 2
    result2 = ds_from_ns_client.to_table()
    assert result2 == table1

    final_call_count = get_describe_call_count(inner_ns_client)
    assert final_call_count == call_count_after_initial_reads + 1


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_namespace_append_through_namespace(s3_bucket: str, use_custom: bool):
    """Test appending to dataset through namespace."""
    storage_options = copy.deepcopy(CONFIG)

    ns_client, inner_ns_client = create_tracking_namespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
        use_custom=use_custom,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    assert get_declare_call_count(inner_ns_client) == 0
    assert get_describe_call_count(inner_ns_client) == 0

    ds = lance.write_dataset(
        table1,
        namespace_client=ns_client,
        table_id=table_id,
        mode="create",
        storage_options=storage_options,
    )
    assert ds.count_rows() == 1
    assert len(ds.versions()) == 1
    assert get_declare_call_count(inner_ns_client) == 1
    initial_describe_count = get_describe_call_count(inner_ns_client)

    table2 = pa.Table.from_pylist([{"a": 10, "b": 20}])
    ds = lance.write_dataset(
        table2,
        namespace_client=ns_client,
        table_id=table_id,
        mode="append",
        storage_options=storage_options,
    )
    assert ds.count_rows() == 2
    assert len(ds.versions()) == 2
    assert get_declare_call_count(inner_ns_client) == 1
    assert get_describe_call_count(inner_ns_client) == initial_describe_count + 1

    ds_from_ns_client = lance.dataset(
        namespace_client=ns_client,
        table_id=table_id,
        storage_options=storage_options,
    )

    assert ds_from_ns_client.count_rows() == 2
    assert len(ds_from_ns_client.versions()) == 2
    # +1 for describe from lance.dataset()
    assert get_describe_call_count(inner_ns_client) == initial_describe_count + 2


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_namespace_write_create_mode(s3_bucket: str, use_custom: bool):
    """Test writing dataset through namespace in CREATE mode."""
    storage_options = copy.deepcopy(CONFIG)

    ns_client, inner_ns_client = create_tracking_namespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
        use_custom=use_custom,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex

    assert get_declare_call_count(inner_ns_client) == 0
    assert get_describe_call_count(inner_ns_client) == 0

    ds = lance.write_dataset(
        table1,
        namespace_client=ns_client,
        table_id=["test_ns", table_name],
        mode="create",
        storage_options=storage_options,
    )

    assert get_declare_call_count(inner_ns_client) == 1
    assert ds.count_rows() == 2
    assert len(ds.versions()) == 1
    result = ds.to_table()
    assert result == table1


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_namespace_write_append_mode(s3_bucket: str, use_custom: bool):
    """Test writing dataset through namespace in APPEND mode."""
    storage_options = copy.deepcopy(CONFIG)

    ns_client, inner_ns_client = create_tracking_namespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
        use_custom=use_custom,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    assert get_declare_call_count(inner_ns_client) == 0
    assert get_describe_call_count(inner_ns_client) == 0

    ds = lance.write_dataset(
        table1,
        namespace_client=ns_client,
        table_id=table_id,
        mode="create",
        storage_options=storage_options,
    )
    assert ds.count_rows() == 1
    assert get_declare_call_count(inner_ns_client) == 1
    assert get_describe_call_count(inner_ns_client) == 0

    table2 = pa.Table.from_pylist([{"a": 10, "b": 20}])

    ds = lance.write_dataset(
        table2,
        namespace_client=ns_client,
        table_id=table_id,
        mode="append",
        storage_options=storage_options,
    )

    assert get_declare_call_count(inner_ns_client) == 1
    describe_count_after_append = get_describe_call_count(inner_ns_client)
    assert describe_count_after_append == 1
    assert ds.count_rows() == 2
    assert len(ds.versions()) == 2

    call_count_before_reads = get_describe_call_count(inner_ns_client)
    for _ in range(3):
        assert ds.count_rows() == 2
    assert get_describe_call_count(inner_ns_client) == call_count_before_reads


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_namespace_write_overwrite_mode(s3_bucket: str, use_custom: bool):
    """Test writing dataset through namespace in OVERWRITE mode."""
    storage_options = copy.deepcopy(CONFIG)

    ns_client, inner_ns_client = create_tracking_namespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
        use_custom=use_custom,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    assert get_declare_call_count(inner_ns_client) == 0
    assert get_describe_call_count(inner_ns_client) == 0

    ds = lance.write_dataset(
        table1,
        namespace_client=ns_client,
        table_id=table_id,
        mode="create",
        storage_options=storage_options,
    )
    assert ds.count_rows() == 1
    assert get_declare_call_count(inner_ns_client) == 1
    assert get_describe_call_count(inner_ns_client) == 0

    table2 = pa.Table.from_pylist([{"a": 10, "b": 20}, {"a": 100, "b": 200}])

    ds = lance.write_dataset(
        table2,
        namespace_client=ns_client,
        table_id=table_id,
        mode="overwrite",
        storage_options=storage_options,
    )

    assert get_declare_call_count(inner_ns_client) == 1
    describe_count_after_overwrite = get_describe_call_count(inner_ns_client)
    assert describe_count_after_overwrite == 1
    assert ds.count_rows() == 2
    assert len(ds.versions()) == 2
    result = ds.to_table()
    assert result == table2

    call_count_before_reads = get_describe_call_count(inner_ns_client)
    for _ in range(3):
        assert ds.count_rows() == 2
    assert get_describe_call_count(inner_ns_client) == call_count_before_reads


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_namespace_managed_branches(s3_bucket: str, use_custom: bool):
    """Branches on a managed-versioning table over S3.

    Branch commits must route through the catalog (create_table_version) and
    leave main's chain untouched. A cross-branch checkout at an overlapping
    version number must resolve the requested chain: branch version numbers
    continue from the fork point, so the same number exists on both chains
    with different data.
    """
    storage_options = copy.deepcopy(CONFIG)

    ns_client, inner_ns_client = create_tracking_namespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
        use_custom=use_custom,
        managed_versioning=True,
    )

    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    def commit_count() -> int:
        return inner_ns_client.retrieve_ops_metrics().get("create_table_version", 0)

    lance.write_dataset(
        pa.Table.from_pylist([{"a": 1}]),
        namespace_client=ns_client,
        table_id=table_id,
        mode="create",
        storage_options=storage_options,
    )
    ds = lance.write_dataset(
        pa.Table.from_pylist([{"a": 2}]),
        namespace_client=ns_client,
        table_id=table_id,
        mode="append",
        storage_options=storage_options,
    )
    assert commit_count() >= 2

    ns_client.create_table_branch(
        CreateTableBranchRequest(id=table_id, name="dev", from_version=2)
    )

    dev = ds.checkout_version(("dev", None))
    commits_before_branch_append = commit_count()
    dev = lance.write_dataset(
        pa.Table.from_pylist([{"a": 3}]),
        dev,
        mode="append",
        storage_options=storage_options,
    )
    assert commit_count() == commits_before_branch_append + 1
    assert sorted(dev.to_table()["a"].to_pylist()) == [1, 2, 3]

    # Diverge main to the same version number as dev's tip.
    ds = lance.write_dataset(
        pa.Table.from_pylist([{"a": 100}]),
        namespace_client=ns_client,
        table_id=table_id,
        mode="append",
        storage_options=storage_options,
    )
    assert sorted(ds.to_table()["a"].to_pylist()) == [1, 2, 100]

    on_dev = ds.checkout_version(("dev", 3))
    assert sorted(on_dev.to_table()["a"].to_pylist()) == [1, 2, 3]
    back_on_main = dev.checkout_version(("main", None))
    assert sorted(back_on_main.to_table()["a"].to_pylist()) == [1, 2, 100]

    fresh = lance.dataset(
        namespace_client=ns_client,
        table_id=table_id,
        storage_options=storage_options,
    )
    assert sorted(fresh.to_table()["a"].to_pylist()) == [1, 2, 100]


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_namespace_distributed_write(s3_bucket: str, use_custom: bool):
    """Test distributed write pattern through namespace."""
    storage_options = copy.deepcopy(CONFIG)

    ns_client, inner_ns_client = create_tracking_namespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3600,
        use_custom=use_custom,
    )

    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    request = DeclareTableRequest(id=table_id, location=None)
    response = ns_client.declare_table(request)

    assert get_declare_call_count(inner_ns_client) == 1
    assert get_describe_call_count(inner_ns_client) == 0

    table_uri = response.location
    assert table_uri is not None

    ns_client_storage_options = response.storage_options
    assert ns_client_storage_options is not None

    merged_options = dict(storage_options)
    merged_options.update(ns_client_storage_options)

    from lance.fragment import write_fragments

    fragment1_data = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    fragment1 = write_fragments(
        fragment1_data,
        table_uri,
        storage_options=merged_options,
        namespace_client=ns_client,
        table_id=table_id,
    )

    fragment2_data = pa.Table.from_pylist([{"a": 10, "b": 20}, {"a": 30, "b": 40}])
    fragment2 = write_fragments(
        fragment2_data,
        table_uri,
        storage_options=merged_options,
        namespace_client=ns_client,
        table_id=table_id,
    )

    fragment3_data = pa.Table.from_pylist([{"a": 100, "b": 200}])
    fragment3 = write_fragments(
        fragment3_data,
        table_uri,
        storage_options=merged_options,
        namespace_client=ns_client,
        table_id=table_id,
    )

    all_fragments = fragment1 + fragment2 + fragment3

    operation = lance.LanceOperation.Overwrite(fragment1_data.schema, all_fragments)

    ds = lance.LanceDataset.commit(
        table_uri,
        operation,
        storage_options=merged_options,
        namespace_client=ns_client,
        table_id=table_id,
    )

    assert ds.count_rows() == 5
    assert len(ds.versions()) == 1

    result = ds.to_table().sort_by("a")
    expected = pa.Table.from_pylist(
        [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 10, "b": 20},
            {"a": 30, "b": 40},
            {"a": 100, "b": 200},
        ]
    )
    assert result == expected

    ds_from_ns_client = lance.dataset(
        namespace_client=ns_client,
        table_id=table_id,
        storage_options=storage_options,
    )
    assert ds_from_ns_client.count_rows() == 5


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_file_writer_with_namespace_client(s3_bucket: str, use_custom: bool):
    """Test LanceFileWriter with namespace_client and credential refresh."""
    from lance.file import LanceFileReader, LanceFileWriter

    storage_options = copy.deepcopy(CONFIG)

    ns_client, inner_ns_client = create_tracking_namespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3,
        use_custom=use_custom,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    assert get_declare_call_count(inner_ns_client) == 0
    assert get_describe_call_count(inner_ns_client) == 0

    ds = lance.write_dataset(
        table1,
        namespace_client=ns_client,
        table_id=table_id,
        mode="create",
        storage_options=storage_options,
    )
    assert ds.count_rows() == 2
    assert get_declare_call_count(inner_ns_client) == 1

    describe_response = ns_client.describe_table(
        DescribeTableRequest(id=table_id, version=None)
    )
    merged_options = dict(storage_options)
    if describe_response.storage_options:
        merged_options.update(describe_response.storage_options)

    initial_describe_count = get_describe_call_count(inner_ns_client)

    file_uri = f"s3://{s3_bucket}/{table_name}_file_test.lance"
    schema = pa.schema([pa.field("x", pa.int64()), pa.field("y", pa.int64())])

    writer = LanceFileWriter(
        file_uri,
        schema=schema,
        storage_options=merged_options,
        namespace_client=ns_client,
        table_id=table_id,
    )

    batch = pa.RecordBatch.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]}, schema=schema)
    writer.write_batch(batch)

    batch2 = pa.RecordBatch.from_pydict(
        {"x": [7, 8, 9], "y": [10, 11, 12]}, schema=schema
    )
    writer.write_batch(batch2)
    writer.close()

    describe_count_after_write = get_describe_call_count(inner_ns_client)
    assert describe_count_after_write == initial_describe_count

    reader = LanceFileReader(
        file_uri,
        storage_options=merged_options,
        namespace_client=ns_client,
        table_id=table_id,
    )
    result = reader.read_all(batch_size=1024)
    result_table = result.to_table()
    assert result_table.num_rows == 6
    assert result_table.schema == schema

    expected_table = pa.table(
        {"x": [1, 2, 3, 7, 8, 9], "y": [4, 5, 6, 10, 11, 12]}, schema=schema
    )
    assert result_table == expected_table

    time.sleep(5)

    file_uri2 = f"s3://{s3_bucket}/{table_name}_file_test2.lance"
    writer2 = LanceFileWriter(
        file_uri2,
        schema=schema,
        storage_options=merged_options,
        namespace_client=ns_client,
        table_id=table_id,
    )

    batch3 = pa.RecordBatch.from_pydict(
        {"x": [100, 200], "y": [300, 400]}, schema=schema
    )
    writer2.write_batch(batch3)
    writer2.close()

    final_describe_count = get_describe_call_count(inner_ns_client)
    assert final_describe_count == describe_count_after_write + 1

    reader2 = LanceFileReader(
        file_uri2,
        storage_options=merged_options,
        namespace_client=ns_client,
        table_id=table_id,
    )
    result2 = reader2.read_all(batch_size=1024)
    result_table2 = result2.to_table()
    assert result_table2.num_rows == 2
    expected_table2 = pa.table({"x": [100, 200], "y": [300, 400]}, schema=schema)
    assert result_table2 == expected_table2


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_file_reader_with_namespace_client(s3_bucket: str, use_custom: bool):
    """Test LanceFileReader with namespace_client and credential refresh."""
    from lance.file import LanceFileReader, LanceFileWriter

    storage_options = copy.deepcopy(CONFIG)

    ns_client, inner_ns_client = create_tracking_namespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3,
        use_custom=use_custom,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    ds = lance.write_dataset(
        table1,
        namespace_client=ns_client,
        table_id=table_id,
        mode="create",
        storage_options=storage_options,
    )
    assert ds.count_rows() == 2

    describe_response = ns_client.describe_table(
        DescribeTableRequest(id=table_id, version=None)
    )
    merged_options = dict(storage_options)
    if describe_response.storage_options:
        merged_options.update(describe_response.storage_options)

    file_uri = f"s3://{s3_bucket}/{table_name}_file_reader_test.lance"
    schema = pa.schema([pa.field("x", pa.int64()), pa.field("y", pa.int64())])

    # Write a file first (without namespace_client to keep it simple)
    writer = LanceFileWriter(
        file_uri,
        schema=schema,
        storage_options=merged_options,
    )
    batch = pa.RecordBatch.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]}, schema=schema)
    writer.write_batch(batch)
    writer.close()

    # Get fresh credentials for reading
    describe_response = ns_client.describe_table(
        DescribeTableRequest(id=table_id, version=None)
    )
    merged_options = dict(storage_options)
    if describe_response.storage_options:
        merged_options.update(describe_response.storage_options)

    initial_describe_count = get_describe_call_count(inner_ns_client)

    # First read should work without needing refresh
    reader = LanceFileReader(
        file_uri,
        storage_options=merged_options,
        namespace_client=ns_client,
        table_id=table_id,
    )
    result = reader.read_all(batch_size=1024)
    result_table = result.to_table()
    assert result_table.num_rows == 3
    assert result_table.schema == schema

    describe_count_after_first_read = get_describe_call_count(inner_ns_client)
    assert describe_count_after_first_read == initial_describe_count

    # Wait for credentials to expire
    time.sleep(5)

    # Write a second file
    file_uri2 = f"s3://{s3_bucket}/{table_name}_file_reader_test2.lance"
    writer2 = LanceFileWriter(
        file_uri2,
        schema=schema,
        storage_options=merged_options,
    )
    batch2 = pa.RecordBatch.from_pydict(
        {"x": [100, 200], "y": [300, 400]}, schema=schema
    )
    writer2.write_batch(batch2)
    writer2.close()

    # Second read should trigger credential refresh
    reader2 = LanceFileReader(
        file_uri2,
        storage_options=merged_options,
        namespace_client=ns_client,
        table_id=table_id,
    )
    result2 = reader2.read_all(batch_size=1024)
    result_table2 = result2.to_table()
    assert result_table2.num_rows == 2
    expected_table2 = pa.table({"x": [100, 200], "y": [300, 400]}, schema=schema)
    assert result_table2 == expected_table2

    final_describe_count = get_describe_call_count(inner_ns_client)
    assert final_describe_count == describe_count_after_first_read + 1


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_file_session_with_namespace_client(s3_bucket: str, use_custom: bool):
    """Test LanceFileSession with namespace_client and credential refresh."""
    from lance.file import LanceFileSession

    storage_options = copy.deepcopy(CONFIG)

    ns_client, inner_ns_client = create_tracking_namespace(
        bucket_name=s3_bucket,
        storage_options=storage_options,
        credential_expires_in_seconds=3,
        use_custom=use_custom,
    )

    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    table_name = uuid.uuid4().hex
    table_id = ["test_ns", table_name]

    ds = lance.write_dataset(
        table1,
        namespace_client=ns_client,
        table_id=table_id,
        mode="create",
        storage_options=storage_options,
    )
    assert ds.count_rows() == 2

    describe_response = ns_client.describe_table(
        DescribeTableRequest(id=table_id, version=None)
    )
    merged_options = dict(storage_options)
    if describe_response.storage_options:
        merged_options.update(describe_response.storage_options)

    initial_describe_count = get_describe_call_count(inner_ns_client)

    # Create session with namespace_client
    session = LanceFileSession(
        f"s3://{s3_bucket}/{table_name}_session",
        storage_options=merged_options,
        namespace_client=ns_client,
        table_id=table_id,
    )

    # Test contains method
    assert not session.contains("session_test.lance")

    # Test list method
    files = session.list()
    assert isinstance(files, list)

    schema = pa.schema([pa.field("x", pa.int64()), pa.field("y", pa.int64())])

    # Write using session - should not trigger credential refresh
    writer = session.open_writer(
        "session_test.lance",
        schema=schema,
    )
    batch = pa.RecordBatch.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]}, schema=schema)
    writer.write_batch(batch)
    writer.close()

    describe_count_after_first_write = get_describe_call_count(inner_ns_client)
    assert describe_count_after_first_write == initial_describe_count

    # Test contains method after write
    assert session.contains("session_test.lance")

    # Read using session - should not trigger credential refresh
    reader = session.open_reader("session_test.lance")
    result = reader.read_all(batch_size=1024)
    result_table = result.to_table()
    assert result_table.num_rows == 3
    assert result_table.schema == schema

    expected_table = pa.table({"x": [1, 2, 3], "y": [4, 5, 6]}, schema=schema)
    assert result_table == expected_table

    describe_count_after_first_read = get_describe_call_count(inner_ns_client)
    assert describe_count_after_first_read == describe_count_after_first_write

    # Wait for credentials to expire
    time.sleep(5)

    # Write again, should trigger credential refresh
    writer2 = session.open_writer(
        "session_test2.lance",
        schema=schema,
    )
    batch2 = pa.RecordBatch.from_pydict(
        {"x": [100, 200], "y": [300, 400]}, schema=schema
    )
    writer2.write_batch(batch2)
    writer2.close()

    describe_count_after_second_write = get_describe_call_count(inner_ns_client)
    assert describe_count_after_second_write == describe_count_after_first_read + 1

    # Read the second file - should not trigger another refresh since we just refreshed
    reader2 = session.open_reader("session_test2.lance")
    result2 = reader2.read_all(batch_size=1024)
    result_table2 = result2.to_table()
    assert result_table2.num_rows == 2
    expected_table2 = pa.table({"x": [100, 200], "y": [300, 400]}, schema=schema)
    assert result_table2 == expected_table2

    final_describe_count = get_describe_call_count(inner_ns_client)
    assert final_describe_count == describe_count_after_second_write


def create_test_table_data():
    """Create test PyArrow table data for concurrent tests."""
    return pa.Table.from_pylist(
        [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35},
        ]
    )


def table_to_ipc_bytes(table):
    """Convert PyArrow table to IPC bytes."""
    import io

    sink = io.BytesIO()
    with pa.ipc.RecordBatchStreamWriter(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_basic_create_and_drop_on_s3(s3_bucket: str, use_custom: bool):
    """Test basic create and drop table operations on S3.

    Mirrors Java: testBasicCreateAndDropOnS3
    """
    test_prefix = f"test-{uuid.uuid4().hex[:8]}"
    storage_options = copy.deepcopy(CONFIG)
    dir_props = {f"storage.{k}": v for k, v in storage_options.items()}
    dir_props["root"] = f"s3://{s3_bucket}/{test_prefix}"
    inner_ns_client = DirectoryNamespace(**dir_props)
    ns_client = _wrap_if_custom(inner_ns_client, use_custom)

    table_name = "basic_test_table"
    table_data = create_test_table_data()
    table_id = ["test_ns", table_name]

    # Create table using lance.write_dataset
    ds = lance.write_dataset(
        table_data,
        namespace_client=ns_client,
        table_id=table_id,
        mode="create",
        storage_options=storage_options,
    )
    assert ds is not None
    assert ds.count_rows() == 3

    # Drop table
    drop_req = DropTableRequest(id=table_id)
    drop_resp = ns_client.drop_table(drop_req)
    assert drop_resp is not None

    # Verify table no longer exists
    exists_req = TableExistsRequest(id=table_id)
    with pytest.raises(Exception):
        ns_client.table_exists(exists_req)


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_concurrent_create_and_drop_single_instance_on_s3(
    s3_bucket: str, use_custom: bool
):
    """Test concurrent create/drop with single namespace instance on S3."""
    import concurrent.futures

    test_prefix = f"test-{uuid.uuid4().hex[:8]}"
    storage_options = copy.deepcopy(CONFIG)
    dir_props = {f"storage.{k}": v for k, v in storage_options.items()}
    dir_props["root"] = f"s3://{s3_bucket}/{test_prefix}"
    # Very high retry count to guarantee all operations succeed
    dir_props["commit_retries"] = "2147483647"
    inner_ns_client = DirectoryNamespace(**dir_props)
    ns_client = _wrap_if_custom(inner_ns_client, use_custom)

    # Initialize namespace first - create parent namespace to ensure __manifest table
    # is created before concurrent operations
    create_ns_req = CreateNamespaceRequest(id=["test_ns"])
    ns_client.create_namespace(create_ns_req)

    num_tables = 10
    success_count = 0
    fail_count = 0
    lock = Lock()

    def create_and_drop_table(table_index):
        nonlocal success_count, fail_count
        try:
            table_name = f"s3_concurrent_table_{table_index}"
            table_data = create_test_table_data()
            table_id = ["test_ns", table_name]
            ipc_data = table_to_ipc_bytes(table_data)

            # Create table using atomic create_table API
            create_req = CreateTableRequest(id=table_id)
            ns_client.create_table(create_req, ipc_data)

            # Drop table
            drop_req = DropTableRequest(id=table_id)
            ns_client.drop_table(drop_req)

            with lock:
                success_count += 1
        except Exception:
            with lock:
                fail_count += 1
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tables) as executor:
        futures = [executor.submit(create_and_drop_table, i) for i in range(num_tables)]
        concurrent.futures.wait(futures)

    # All operations must succeed with very high retry count
    assert success_count == num_tables, (
        f"Expected {num_tables} successes, got {success_count}"
    )
    assert fail_count == 0, f"Expected 0 failures, got {fail_count}"


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_concurrent_create_and_drop_multiple_instances_on_s3(
    s3_bucket: str, use_custom: bool
):
    """Test concurrent create/drop with multiple namespace instances on S3."""
    import concurrent.futures

    test_prefix = f"test-{uuid.uuid4().hex[:8]}"
    storage_options = copy.deepcopy(CONFIG)
    base_dir_props = {f"storage.{k}": v for k, v in storage_options.items()}
    base_dir_props["root"] = f"s3://{s3_bucket}/{test_prefix}"
    # Very high retry count to guarantee all operations succeed
    base_dir_props["commit_retries"] = "2147483647"

    # Initialize namespace first with a single instance to ensure __manifest
    # table is created and parent namespace exists before concurrent operations
    inner_init_ns_client = DirectoryNamespace(**base_dir_props.copy())
    init_ns_client = _wrap_if_custom(inner_init_ns_client, use_custom)
    create_ns_req = CreateNamespaceRequest(id=["test_ns"])
    init_ns_client.create_namespace(create_ns_req)

    num_tables = 10
    success_count = 0
    fail_count = 0
    lock = Lock()

    def create_and_drop_table(table_index):
        nonlocal success_count, fail_count
        try:
            # Each thread creates its own namespace client instance
            inner_local_ns_client = DirectoryNamespace(**base_dir_props.copy())
            local_ns_client = _wrap_if_custom(inner_local_ns_client, use_custom)

            table_name = f"s3_multi_ns_table_{table_index}"
            table_data = create_test_table_data()
            table_id = ["test_ns", table_name]
            ipc_data = table_to_ipc_bytes(table_data)

            # Create table using atomic create_table API
            create_req = CreateTableRequest(id=table_id)
            local_ns_client.create_table(create_req, ipc_data)

            # Drop table
            drop_req = DropTableRequest(id=table_id)
            local_ns_client.drop_table(drop_req)

            with lock:
                success_count += 1
        except Exception:
            with lock:
                fail_count += 1
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tables) as executor:
        futures = [executor.submit(create_and_drop_table, i) for i in range(num_tables)]
        concurrent.futures.wait(futures)

    # All operations must succeed with very high retry count
    assert success_count == num_tables, (
        f"Expected {num_tables} successes, got {success_count}"
    )
    assert fail_count == 0, f"Expected 0 failures, got {fail_count}"

    # Verify remaining state is consistent (no corruption)
    inner_verify_ns_client = DirectoryNamespace(**base_dir_props)
    verify_ns_client = _wrap_if_custom(inner_verify_ns_client, use_custom)
    list_req = ListTablesRequest(id=["test_ns"])
    _ = verify_ns_client.list_tables(list_req)  # Should not error


@pytest.mark.integration
@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_concurrent_create_then_drop_from_different_instance_on_s3(
    s3_bucket: str, use_custom: bool
):
    """Test creating from one set of instances, dropping from different ones on S3."""
    import concurrent.futures

    test_prefix = f"test-{uuid.uuid4().hex[:8]}"
    storage_options = copy.deepcopy(CONFIG)
    base_dir_props = {f"storage.{k}": v for k, v in storage_options.items()}
    base_dir_props["root"] = f"s3://{s3_bucket}/{test_prefix}"
    # Very high retry count to guarantee all operations succeed
    base_dir_props["commit_retries"] = "2147483647"

    # Initialize namespace first with a single instance to ensure __manifest
    # table is created and parent namespace exists before concurrent operations
    inner_init_ns_client = DirectoryNamespace(**base_dir_props.copy())
    init_ns_client = _wrap_if_custom(inner_init_ns_client, use_custom)
    create_ns_req = CreateNamespaceRequest(id=["test_ns"])
    init_ns_client.create_namespace(create_ns_req)

    num_tables = 10

    # Phase 1: Create all tables concurrently using separate namespace client instances
    create_success_count = 0
    create_fail_count = 0
    create_lock = Lock()

    def create_table(table_index):
        nonlocal create_success_count, create_fail_count
        table_name = f"s3_cross_instance_table_{table_index}"
        try:
            inner_local_ns_client = DirectoryNamespace(**base_dir_props.copy())
            local_ns_client = _wrap_if_custom(inner_local_ns_client, use_custom)

            table_data = create_test_table_data()
            table_id = ["test_ns", table_name]
            ipc_data = table_to_ipc_bytes(table_data)

            # Create table using atomic create_table API
            create_req = CreateTableRequest(id=table_id)
            local_ns_client.create_table(create_req, ipc_data)

            with create_lock:
                create_success_count += 1
        except Exception:
            with create_lock:
                create_fail_count += 1
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tables) as executor:
        futures = [executor.submit(create_table, i) for i in range(num_tables)]
        concurrent.futures.wait(futures)

    # All creates must succeed with very high retry count
    assert create_success_count == num_tables, (
        f"Expected {num_tables} create successes, got {create_success_count}"
    )
    assert create_fail_count == 0, (
        f"Expected 0 create failures, got {create_fail_count}"
    )

    # Phase 2: Drop all tables using NEW namespace instances
    drop_success_count = 0
    drop_fail_count = 0
    drop_lock = Lock()

    def drop_table(table_index):
        nonlocal drop_success_count, drop_fail_count
        try:
            inner_local_ns_client = DirectoryNamespace(**base_dir_props.copy())
            local_ns_client = _wrap_if_custom(inner_local_ns_client, use_custom)

            table_name = f"s3_cross_instance_table_{table_index}"
            table_id = ["test_ns", table_name]

            drop_req = DropTableRequest(id=table_id)
            local_ns_client.drop_table(drop_req)

            with drop_lock:
                drop_success_count += 1
        except Exception:
            with drop_lock:
                drop_fail_count += 1
            raise

    # Drop all tables
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tables) as executor:
        futures = [executor.submit(drop_table, i) for i in range(num_tables)]
        concurrent.futures.wait(futures)

    # All drops must succeed with very high retry count
    assert drop_success_count == num_tables, (
        f"Expected {num_tables} drop successes, got {drop_success_count}"
    )
    assert drop_fail_count == 0, f"Expected 0 drop failures, got {drop_fail_count}"

    # Verify remaining state is consistent (no corruption)
    inner_verify_ns_client = DirectoryNamespace(**base_dir_props)
    verify_ns_client = _wrap_if_custom(inner_verify_ns_client, use_custom)
    list_req = ListTablesRequest(id=["test_ns"])
    _ = verify_ns_client.list_tables(list_req)  # Should not error
