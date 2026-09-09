# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for DirectoryNamespace implementation.

This module tests the DirectoryNamespace class which provides a directory-based
namespace implementation for organizing Lance tables and nested namespaces.

These tests mirror the Rust tests in rust/lance-namespace-impls/src/dir.rs

Additionally, tests are parameterized to run with both the native DirectoryNamespace
and a CustomNamespace wrapper to verify Python-Rust binding works correctly for
custom namespace implementations.
"""

import sys
import tempfile
import uuid
from threading import Lock
from typing import Dict, Optional

import lance
import lance.namespace
import pyarrow as pa
import pytest
from lance.namespace import LanceNamespace
from lance_namespace import (
    CountTableRowsRequest,
    CountTableRowsResponse,
    CreateNamespaceRequest,
    CreateNamespaceResponse,
    CreateTableBranchRequest,
    CreateTableBranchResponse,
    CreateTableIndexRequest,
    CreateTableIndexResponse,
    CreateTableRequest,
    CreateTableResponse,
    CreateTableVersionRequest,
    CreateTableVersionResponse,
    DeclareTableRequest,
    DeclareTableResponse,
    DeleteTableBranchRequest,
    DeleteTableBranchResponse,
    DeregisterTableRequest,
    DeregisterTableResponse,
    DescribeNamespaceRequest,
    DescribeNamespaceResponse,
    DescribeTableIndexStatsRequest,
    DescribeTableRequest,
    DescribeTableResponse,
    DescribeTableVersionRequest,
    DescribeTableVersionResponse,
    DropNamespaceRequest,
    DropNamespaceResponse,
    DropTableRequest,
    DropTableResponse,
    InsertIntoTableRequest,
    InsertIntoTableResponse,
    ListNamespacesRequest,
    ListNamespacesResponse,
    ListTableBranchesRequest,
    ListTableBranchesResponse,
    ListTableIndicesRequest,
    ListTableIndicesResponse,
    ListTablesRequest,
    ListTablesResponse,
    ListTableVersionsRequest,
    ListTableVersionsResponse,
    NamespaceExistsRequest,
    NamespaceExistsResponse,
    QueryTableRequest,
    QueryTableResponse,
    RegisterTableRequest,
    RegisterTableResponse,
    TableExistsRequest,
    TableExistsResponse,
    connect,
)
from lance_namespace.errors import (
    InvalidInputError,
    NamespaceNotEmptyError,
    NamespaceNotFoundError,
    TableBranchAlreadyExistsError,
    TableBranchNotFoundError,
    TableNotFoundError,
)


class CustomNamespace(LanceNamespace):
    """A custom namespace wrapper that delegates to DirectoryNamespace.

    This class verifies that the Python-Rust binding works correctly for
    custom namespace implementations that wrap the native DirectoryNamespace.
    All methods simply delegate to the underlying DirectoryNamespace instance.
    """

    def __init__(self, inner: lance.namespace.DirectoryNamespace):
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

    def list_table_branches(
        self, request: ListTableBranchesRequest
    ) -> ListTableBranchesResponse:
        return self._inner.list_table_branches(request)

    def delete_table_branch(
        self, request: DeleteTableBranchRequest
    ) -> DeleteTableBranchResponse:
        return self._inner.delete_table_branch(request)

    def create_table_index(
        self, request: CreateTableIndexRequest
    ) -> CreateTableIndexResponse:
        return self._inner.create_table_index(request)

    def list_table_indices(
        self, request: ListTableIndicesRequest
    ) -> ListTableIndicesResponse:
        return self._inner.list_table_indices(request)

    def count_table_rows(
        self, request: CountTableRowsRequest
    ) -> CountTableRowsResponse:
        return self._inner.count_table_rows(request)

    def insert_into_table(
        self, request: InsertIntoTableRequest, request_data: bytes
    ) -> InsertIntoTableResponse:
        return self._inner.insert_into_table(request, request_data)

    def query_table(self, request) -> QueryTableResponse:
        # Accept both QueryTableRequest and dict, like DirectoryNamespace does
        if hasattr(request, "model_dump"):
            request = request.model_dump()
        return self._inner.query_table(request)

    def retrieve_ops_metrics(self) -> Optional[Dict[str, int]]:
        return self._inner.retrieve_ops_metrics()


def create_test_data():
    """Create test PyArrow table data."""
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


def _wrap_if_custom(ns_client, use_custom: bool):
    """Wrap namespace client in CustomNamespace if use_custom is True."""
    if use_custom:
        return CustomNamespace(ns_client)
    return ns_client


@pytest.fixture(params=[False, True], ids=["DirectoryNamespace", "CustomNamespace"])
def temp_ns_client(request):
    """Create a temporary namespace client for testing.

    Parameterized to test both DirectoryNamespace and CustomNamespace wrapper.
    """
    use_custom = request.param
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use lance.namespace.connect() for consistency
        # Use high commit_retries for concurrent operation tests
        ns_client = connect(
            "dir", {"root": f"file://{tmpdir}", "commit_retries": "2147483647"}
        )
        yield _wrap_if_custom(ns_client, use_custom)


@pytest.fixture(params=[False, True], ids=["DirectoryNamespace", "CustomNamespace"])
def memory_ns_client(request):
    """Create a memory-based namespace client for testing.

    Parameterized to test both DirectoryNamespace and CustomNamespace wrapper.
    """
    use_custom = request.param
    unique_id = uuid.uuid4().hex[:8]
    # Use lance.namespace.connect() for consistency
    ns_client = connect("dir", {"root": f"memory://test_{unique_id}"})
    yield _wrap_if_custom(ns_client, use_custom)


class TestCreateTable:
    """Tests for create_table operation - mirrors Rust test_create_table."""

    def test_create_table(self, memory_ns_client):
        """Test creating a table with data."""
        # Create parent namespace first
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_ns_req)

        # Create table with data
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        create_req = CreateTableRequest(id=["workspace", "test_table"])
        response = memory_ns_client.create_table(create_req, ipc_data)

        assert response is not None
        assert response.location is not None
        # Location format varies based on manifest implementation
        # Just check that it contains the table name
        assert "test_table" in response.location
        assert response.version == 1

    def test_create_table_without_data(self, memory_ns_client):
        """Test creating a table without data should fail."""
        # Create parent namespace first
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_ns_req)

        create_req = CreateTableRequest(id=["workspace", "test_table"])

        with pytest.raises(InvalidInputError) as exc_info:
            memory_ns_client.create_table(create_req, b"")

        assert "Arrow IPC" in str(exc_info.value) or "required" in str(exc_info.value)

    def test_create_table_with_invalid_id(self, memory_ns_client):
        """Test creating a table with invalid ID should fail."""
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        # Test with empty ID
        create_req = CreateTableRequest(id=[])
        with pytest.raises(InvalidInputError):
            memory_ns_client.create_table(create_req, ipc_data)

    def test_create_table_in_child_namespace(self, memory_ns_client):
        """Test creating table in child namespace works with manifest enabled."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["test_namespace"])
        memory_ns_client.create_namespace(create_ns_req)

        # Create table in the namespace
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["test_namespace", "table"])
        response = memory_ns_client.create_table(create_req, ipc_data)

        # Should succeed with manifest enabled
        assert response is not None
        assert response.location is not None


class TestListTables:
    """Tests for list_tables operation - mirrors Rust test_list_tables."""

    def test_list_tables_empty(self, memory_ns_client):
        """Test listing tables in empty namespace."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_ns_req)

        # Initially, no tables
        list_req = ListTablesRequest(id=["workspace"])
        response = memory_ns_client.list_tables(list_req)
        assert len(response.tables) == 0

    def test_list_tables_with_tables(self, memory_ns_client):
        """Test listing tables after creating them."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_ns_req)

        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        # Create table1
        create_req = CreateTableRequest(id=["workspace", "table1"])
        memory_ns_client.create_table(create_req, ipc_data)

        # Create table2
        create_req = CreateTableRequest(id=["workspace", "table2"])
        memory_ns_client.create_table(create_req, ipc_data)

        # List tables should return both
        list_req = ListTablesRequest(id=["workspace"])
        response = memory_ns_client.list_tables(list_req)
        assert len(response.tables) == 2

        # List tables returns table names as strings
        assert "table1" in response.tables
        assert "table2" in response.tables

    def test_list_tables_with_namespace_id(self, memory_ns_client):
        """Test listing tables in a child namespace."""
        # Create child namespace
        create_ns_req = CreateNamespaceRequest(id=["test_namespace"])
        memory_ns_client.create_namespace(create_ns_req)

        # List tables in the child namespace
        list_req = ListTablesRequest(id=["test_namespace"])
        response = memory_ns_client.list_tables(list_req)

        # Should succeed and return empty list (no tables yet)
        assert len(response.tables) == 0


class TestDescribeTable:
    """Tests for describe_table operation - mirrors Rust test_describe_table."""

    def test_describe_table(self, memory_ns_client):
        """Test describing a table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_ns_req)

        # Create a table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        memory_ns_client.create_table(create_req, ipc_data)

        # Describe the table
        describe_req = DescribeTableRequest(id=["workspace", "test_table"])
        response = memory_ns_client.describe_table(describe_req)

        assert response is not None
        assert response.location is not None
        assert "test_table" in response.location

    def test_describe_nonexistent_table(self, memory_ns_client):
        """Test describing a table that doesn't exist."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_ns_req)

        describe_req = DescribeTableRequest(id=["workspace", "nonexistent"])

        with pytest.raises(TableNotFoundError):
            memory_ns_client.describe_table(describe_req)


class TestTableOperations:
    """Tests for various table operations."""

    def test_table_exists(self, memory_ns_client):
        """Test checking if a table exists."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_ns_req)

        # Create a table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        memory_ns_client.create_table(create_req, ipc_data)

        # Check it exists (should not raise)
        exists_req = TableExistsRequest(id=["workspace", "test_table"])
        memory_ns_client.table_exists(exists_req)

    def test_table_not_exists(self, memory_ns_client):
        """Test checking if a non-existent table exists raises TableNotFoundError."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_ns_req)

        exists_req = TableExistsRequest(id=["workspace", "nonexistent"])

        with pytest.raises(TableNotFoundError):
            memory_ns_client.table_exists(exists_req)

    def test_drop_table(self, memory_ns_client):
        """Test dropping a table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_ns_req)

        # Create table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        memory_ns_client.create_table(create_req, ipc_data)

        # Drop the table
        drop_req = DropTableRequest(id=["workspace", "test_table"])
        response = memory_ns_client.drop_table(drop_req)
        assert response is not None

        # Verify table no longer exists
        exists_req = TableExistsRequest(id=["workspace", "test_table"])
        with pytest.raises(TableNotFoundError):
            memory_ns_client.table_exists(exists_req)

    def test_deregister_table(self, temp_ns_client):
        """Test deregistering a table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        # Create table using lance directly
        table_data = create_test_data()
        # Get root path from namespace
        ns_id = temp_ns_client.namespace_id()
        import re

        match = re.search(r'root: "([^"]+)"', ns_id)
        assert match is not None
        root_path = match.group(1)

        # Create physical table
        table_uri = f"{root_path}/workspace/physical_table.lance"
        lance.write_dataset(table_data, table_uri)

        # Register the table with a relative location
        register_req = RegisterTableRequest(
            id=["workspace", "test_table"], location="workspace/physical_table.lance"
        )
        temp_ns_client.register_table(register_req)

        # Deregister it
        deregister_req = DeregisterTableRequest(id=["workspace", "test_table"])
        response = temp_ns_client.deregister_table(deregister_req)
        assert response is not None
        # Should return full URI to deregistered table
        # (use endswith to handle path canonicalization)
        assert response.location.endswith("/workspace/physical_table.lance"), (
            f"Expected location to end with '/workspace/physical_table.lance', "
            f"got {response.location}"
        )
        assert response.id == ["workspace", "test_table"]

    def test_register_table(self, temp_ns_client):
        """Test registering an existing table."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        # Create physical table using lance
        table_data = create_test_data()
        ns_id = temp_ns_client.namespace_id()
        import re

        match = re.search(r'root: "([^"]+)"', ns_id)
        assert match is not None
        root_path = match.group(1)

        # Create physical table
        table_uri = f"{root_path}/workspace/physical_table.lance"
        lance.write_dataset(table_data, table_uri)

        # Register with a different name using relative path
        register_req = RegisterTableRequest(
            id=["workspace", "registered_table"],
            location="workspace/physical_table.lance",
        )
        response = temp_ns_client.register_table(register_req)
        assert response is not None
        assert response.location == "workspace/physical_table.lance"

        # Verify table exists
        exists_req = TableExistsRequest(id=["workspace", "registered_table"])
        temp_ns_client.table_exists(exists_req)

        # Verify we can read from it
        describe_req = DescribeTableRequest(id=["workspace", "registered_table"])
        desc_response = temp_ns_client.describe_table(describe_req)
        assert desc_response is not None
        # Should point to the same physical location
        # (use endswith to handle path canonicalization)
        assert desc_response.location.endswith("/workspace/physical_table.lance"), (
            f"Expected location to end with '/workspace/physical_table.lance', "
            f"got {desc_response.location}"
        )

    def test_register_table_rejects_absolute_uri(self, temp_ns_client):
        """Test that register_table rejects absolute URIs."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        # Try to register with absolute URI - should fail
        register_req = RegisterTableRequest(
            id=["workspace", "test_table"], location="s3://bucket/table.lance"
        )
        with pytest.raises(InvalidInputError) as exc_info:
            temp_ns_client.register_table(register_req)
        assert "Absolute URIs are not allowed" in str(exc_info.value)

    def test_register_table_rejects_absolute_path(self, temp_ns_client):
        """Test that register_table rejects absolute paths."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        # Try to register with absolute path - should fail
        register_req = RegisterTableRequest(
            id=["workspace", "test_table"], location="/tmp/table.lance"
        )
        with pytest.raises(InvalidInputError) as exc_info:
            temp_ns_client.register_table(register_req)
        assert "Absolute paths are not allowed" in str(exc_info.value)

    def test_register_table_rejects_path_traversal(self, temp_ns_client):
        """Test that register_table rejects path traversal attempts."""
        # Create parent namespace
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        # Try to register with path traversal - should fail
        register_req = RegisterTableRequest(
            id=["workspace", "test_table"], location="../outside/table.lance"
        )
        with pytest.raises(InvalidInputError) as exc_info:
            temp_ns_client.register_table(register_req)
        assert "Path traversal is not allowed" in str(exc_info.value)


class TestTableBranchOperations:
    """Branch CRUD through the python bindings - mirrors the Rust branch
    CRUD tests."""

    def test_branch_crud_round_trip(self, temp_ns_client):
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)
        ipc_data = table_to_ipc_bytes(create_test_data())
        table_id = ["workspace", "branched_table"]
        temp_ns_client.create_table(CreateTableRequest(id=table_id), ipc_data)

        temp_ns_client.create_table_branch(
            CreateTableBranchRequest(id=table_id, name="dev")
        )
        listed = temp_ns_client.list_table_branches(
            ListTableBranchesRequest(id=table_id)
        )
        assert "dev" in listed.branches
        assert listed.branches["dev"].parent_version == 1

        # Duplicate creation and deleting a missing branch surface the typed
        # branch errors (codes 23 and 22), not InternalError.
        temp_ns_client.create_table_branch(
            CreateTableBranchRequest(id=table_id, name="dev2")
        )
        with pytest.raises(TableBranchAlreadyExistsError):
            temp_ns_client.create_table_branch(
                CreateTableBranchRequest(id=table_id, name="dev2")
            )

        temp_ns_client.delete_table_branch(
            DeleteTableBranchRequest(id=table_id, name="dev")
        )
        listed = temp_ns_client.list_table_branches(
            ListTableBranchesRequest(id=table_id)
        )
        assert "dev" not in listed.branches
        with pytest.raises(TableBranchNotFoundError):
            temp_ns_client.delete_table_branch(
                DeleteTableBranchRequest(id=table_id, name="dev")
            )

    def test_create_branch_from_other_branch(self, temp_ns_client):
        """Forking from a non-main source branch records the right parent."""
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)
        ipc_data = table_to_ipc_bytes(create_test_data())
        table_id = ["workspace", "fork_table"]
        temp_ns_client.create_table(CreateTableRequest(id=table_id), ipc_data)

        temp_ns_client.create_table_branch(
            CreateTableBranchRequest(id=table_id, name="dev")
        )
        temp_ns_client.create_table_branch(
            CreateTableBranchRequest(id=table_id, name="child", from_branch="dev")
        )
        listed = temp_ns_client.list_table_branches(
            ListTableBranchesRequest(id=table_id)
        )
        assert listed.branches["child"].parent_branch == "dev"


class _ForeignCodeError(Exception):
    """Not a LanceNamespaceError, but carries the same integer code as
    TABLE_NOT_FOUND."""

    code = 4


class _RaisingNamespace(LanceNamespace):
    """A namespace whose describe_table raises the configured exception."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def namespace_id(self) -> str:
        return "raising"

    def describe_table(self, request: DescribeTableRequest) -> DescribeTableResponse:
        raise self._exc


class TestPythonNamespaceErrorMapping:
    """The Rust adapter must trust the `code` attribute only on the
    lance_namespace exception hierarchy."""

    def test_namespace_error_identity_preserved(self):
        ns = _RaisingNamespace(TableNotFoundError("no such table"))
        with pytest.raises(TableNotFoundError, match="no such table"):
            lance.dataset(namespace_client=ns, table_id=["t"])

        # Branch error codes (22/23) survive the round trip too.
        ns = _RaisingNamespace(TableBranchNotFoundError("no such branch"))
        with pytest.raises(TableBranchNotFoundError, match="no such branch"):
            lance.dataset(namespace_client=ns, table_id=["t"])

    def test_foreign_code_attribute_not_trusted(self):
        # The foreign exception must surface as itself, not be reinterpreted
        # as a namespace error via its `code` attribute.
        ns = _RaisingNamespace(_ForeignCodeError("boom"))
        with pytest.raises(_ForeignCodeError, match="boom"):
            lance.dataset(namespace_client=ns, table_id=["t"])


class TestChildNamespaceOperations:
    """Tests for operations in child namespaces - mirrors Rust tests."""

    def test_create_table_in_child_namespace(self, memory_ns_client):
        """Test creating multiple tables in a child namespace."""
        # Create child namespace
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        memory_ns_client.create_namespace(create_ns_req)

        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        # Create three tables
        for i in range(1, 4):
            create_req = CreateTableRequest(id=["test_ns", f"table{i}"])
            memory_ns_client.create_table(create_req, ipc_data)

        # List tables
        list_req = ListTablesRequest(id=["test_ns"])
        response = memory_ns_client.list_tables(list_req)

        assert len(response.tables) == 3
        # List tables returns table names as strings
        assert "table1" in response.tables
        assert "table2" in response.tables
        assert "table3" in response.tables

    def test_drop_table_in_child_namespace(self, memory_ns_client):
        """Test dropping a table in a child namespace."""
        # Create child namespace
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        memory_ns_client.create_namespace(create_ns_req)

        # Create table
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["test_ns", "table1"])
        memory_ns_client.create_table(create_req, ipc_data)

        # Drop table
        drop_req = DropTableRequest(id=["test_ns", "table1"])
        memory_ns_client.drop_table(drop_req)

        # Verify table no longer exists
        exists_req = TableExistsRequest(id=["test_ns", "table1"])
        with pytest.raises(TableNotFoundError):
            memory_ns_client.table_exists(exists_req)

    def test_declared_table_in_child_namespace(self, memory_ns_client):
        """Test declaring a table in a child namespace."""
        # Create child namespace
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        memory_ns_client.create_namespace(create_ns_req)

        # Declare table
        declare_req = DeclareTableRequest(id=["test_ns", "declared_table"])
        memory_ns_client.declare_table(declare_req)

        # Verify table exists
        exists_req = TableExistsRequest(id=["test_ns", "declared_table"])
        memory_ns_client.table_exists(exists_req)

        describe_req = DescribeTableRequest(id=["test_ns", "declared_table"])
        describe_resp = memory_ns_client.describe_table(describe_req)
        assert describe_resp.is_only_declared is None

        describe_req = DescribeTableRequest(
            id=["test_ns", "declared_table"], check_declared=True
        )
        describe_resp = memory_ns_client.describe_table(describe_req)
        assert describe_resp.is_only_declared is True


class TestDeeplyNestedNamespaces:
    """Tests for deeply nested namespace hierarchies.

    Mirrors Rust test_deeply_nested_namespace.
    """

    def test_deeply_nested_namespace(self, memory_ns_client):
        """Test creating deeply nested namespace hierarchy."""
        # Create deeply nested namespace hierarchy
        memory_ns_client.create_namespace(CreateNamespaceRequest(id=["level1"]))
        memory_ns_client.create_namespace(
            CreateNamespaceRequest(id=["level1", "level2"])
        )
        memory_ns_client.create_namespace(
            CreateNamespaceRequest(id=["level1", "level2", "level3"])
        )

        # Create table in deeply nested namespace
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["level1", "level2", "level3", "table1"])
        memory_ns_client.create_table(create_req, ipc_data)

        # Verify table exists
        exists_req = TableExistsRequest(id=["level1", "level2", "level3", "table1"])
        memory_ns_client.table_exists(exists_req)


class TestNamespaceProperties:
    """Tests for namespace properties - mirrors Rust test_namespace_with_properties."""

    def test_namespace_with_properties(self, memory_ns_client):
        """Test creating a namespace with properties."""
        # Create namespace with properties
        properties = {
            "owner": "test_user",
            "description": "Test namespace",
        }

        create_req = CreateNamespaceRequest(id=["test_ns"], properties=properties)
        memory_ns_client.create_namespace(create_req)

        # Describe namespace and verify properties
        describe_req = DescribeNamespaceRequest(id=["test_ns"])
        response = memory_ns_client.describe_namespace(describe_req)

        assert response.properties is not None
        assert response.properties.get("owner") == "test_user"
        assert response.properties.get("description") == "Test namespace"


class TestNamespaceConstraints:
    """Tests for namespace constraints and isolation."""

    def test_cannot_drop_namespace_with_tables(self, memory_ns_client):
        """Test that dropping a namespace with tables should fail."""
        # Create namespace
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        memory_ns_client.create_namespace(create_ns_req)

        # Create table in namespace
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["test_ns", "table1"])
        memory_ns_client.create_table(create_req, ipc_data)

        # Try to drop namespace - should fail
        drop_req = DropNamespaceRequest(id=["test_ns"])
        with pytest.raises(NamespaceNotEmptyError):
            memory_ns_client.drop_namespace(drop_req)

    def test_isolation_between_namespaces(self, memory_ns_client):
        """Test that namespaces are isolated from each other."""
        # Create two namespaces
        memory_ns_client.create_namespace(CreateNamespaceRequest(id=["ns1"]))
        memory_ns_client.create_namespace(CreateNamespaceRequest(id=["ns2"]))

        # Create table with same name in both namespaces
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)

        create_req1 = CreateTableRequest(id=["ns1", "table1"])
        memory_ns_client.create_table(create_req1, ipc_data)

        create_req2 = CreateTableRequest(id=["ns2", "table1"])
        memory_ns_client.create_table(create_req2, ipc_data)

        # List tables in each namespace
        list_req = ListTablesRequest(id=["ns1"])
        response = memory_ns_client.list_tables(list_req)
        assert len(response.tables) == 1
        assert "table1" in response.tables

        list_req = ListTablesRequest(id=["ns2"])
        response = memory_ns_client.list_tables(list_req)
        assert len(response.tables) == 1
        assert "table1" in response.tables

        # Drop table in ns1 shouldn't affect ns2
        drop_req = DropTableRequest(id=["ns1", "table1"])
        memory_ns_client.drop_table(drop_req)

        # ns1 should have no tables
        list_req = ListTablesRequest(id=["ns1"])
        response = memory_ns_client.list_tables(list_req)
        assert len(response.tables) == 0

        # ns2 should still have the table
        list_req = ListTablesRequest(id=["ns2"])
        response = memory_ns_client.list_tables(list_req)
        assert len(response.tables) == 1


class TestBasicNamespaceOperations:
    """Tests for basic namespace CRUD operations."""

    def test_create_and_describe_namespace(self, memory_ns_client):
        """Test creating and describing a namespace."""
        # Create namespace
        create_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_req)

        # Describe it
        describe_req = DescribeNamespaceRequest(id=["workspace"])
        response = memory_ns_client.describe_namespace(describe_req)
        assert response is not None

    def test_namespace_exists(self, memory_ns_client):
        """Test checking if a namespace exists."""
        # Create namespace
        create_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_req)

        # Check it exists (should not raise)
        exists_req = NamespaceExistsRequest(id=["workspace"])
        memory_ns_client.namespace_exists(exists_req)

    def test_namespace_not_exists(self, memory_ns_client):
        """Test that a non-existent namespace raises NamespaceNotFoundError."""
        exists_req = NamespaceExistsRequest(id=["nonexistent"])

        with pytest.raises(NamespaceNotFoundError):
            memory_ns_client.namespace_exists(exists_req)

    def test_drop_empty_namespace(self, memory_ns_client):
        """Test dropping an empty namespace."""
        # Create namespace
        create_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_req)

        # Drop it
        drop_req = DropNamespaceRequest(id=["workspace"])
        response = memory_ns_client.drop_namespace(drop_req)
        assert response is not None

    def test_list_namespaces(self, memory_ns_client):
        """Test listing namespaces."""
        # Create some child namespaces under a parent
        memory_ns_client.create_namespace(CreateNamespaceRequest(id=["parent"]))
        memory_ns_client.create_namespace(
            CreateNamespaceRequest(id=["parent", "child1"])
        )
        memory_ns_client.create_namespace(
            CreateNamespaceRequest(id=["parent", "child2"])
        )

        # List namespaces under parent
        list_req = ListNamespacesRequest(id=["parent"])
        response = memory_ns_client.list_namespaces(list_req)

        assert response is not None
        # Should find the child namespaces
        assert len(response.namespaces) >= 2


class TestLanceNamespaceConnect:
    """Tests for lance.namespace.connect integration."""

    @pytest.mark.parametrize(
        "use_custom", [False, True], ids=["DirectoryNS", "CustomNS"]
    )
    def test_connect_with_properties(self, use_custom):
        """Test creating DirectoryNamespace via lance.namespace.connect()."""
        unique_id = uuid.uuid4().hex[:8]
        properties = {
            "root": f"memory://test_connect_{unique_id}",
            "manifest_enabled": "true",
            "dir_listing_enabled": "true",
            "dir_listing_to_manifest_migration_enabled": "true",
        }

        # Connect via lance.namespace.connect
        # should use lance.namespace.DirectoryNamespace
        inner_ns_client = connect("dir", properties)

        # Verify it's a DirectoryNamespace instance
        assert isinstance(inner_ns_client, lance.namespace.DirectoryNamespace)

        # Wrap if testing CustomNamespace
        ns_client = _wrap_if_custom(inner_ns_client, use_custom)

        # Verify it works
        create_req = CreateTableRequest(id=["test_table"])
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        response = ns_client.create_table(create_req, ipc_data)
        assert response is not None

        # Verify we can list the table
        list_req = ListTablesRequest(id=[])
        list_response = ns_client.list_tables(list_req)
        assert len(list_response.tables) == 1
        # tables is a list of strings
        assert list_response.tables[0] == "test_table"

    @pytest.mark.parametrize(
        "use_custom", [False, True], ids=["DirectoryNS", "CustomNS"]
    )
    def test_connect_with_storage_options(self, use_custom):
        """Test creating DirectoryNamespace with storage options via connect()."""
        unique_id = uuid.uuid4().hex[:8]
        properties = {
            "root": f"memory://test_storage_{unique_id}",
            "storage.some_option": "value",  # Test storage.* prefix
        }

        # This should work without errors
        inner_ns_client = connect("dir", properties)
        assert isinstance(inner_ns_client, lance.namespace.DirectoryNamespace)

        # Wrap if testing CustomNamespace
        ns_client = _wrap_if_custom(inner_ns_client, use_custom)

        # Verify it can be used for basic operations
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        ns_client.create_namespace(create_ns_req)


def _get_ops_metric(ns_client, metric_name: str) -> int:
    """Helper to get a specific metric count from namespace client ops metrics."""
    metrics = ns_client.retrieve_ops_metrics()
    return metrics.get(metric_name, 0)


@pytest.mark.parametrize("use_custom", [False, True], ids=["DirectoryNS", "CustomNS"])
def test_external_manifest_store_invokes_namespace_apis(use_custom):
    """Test that namespace APIs are invoked correctly for managed versioning.

    This test mirrors:
    - Rust: test_external_manifest_store_invokes_namespace_apis
    - Java: testExternalManifestStoreInvokesNamespaceApis

    It verifies:
    1. list_table_versions is called when opening dataset (latest version)
    2. create_table_version is called exactly once during append
    3. describe_table_version is called when opening specific version

    Uses native ops_metrics_enabled to track API calls instead of custom wrapper.
    The test is parameterized to verify both DirectoryNamespace and CustomNamespace.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use native namespace with ops metrics enabled
        inner_ns_client = lance.namespace.DirectoryNamespace(
            root=tmpdir,
            table_version_tracking_enabled="true",
            manifest_enabled="true",
            ops_metrics_enabled="true",
        )
        ns_client = _wrap_if_custom(inner_ns_client, use_custom)

        # Create parent namespace first (like Rust/Java tests)
        ns_client.create_namespace(CreateNamespaceRequest(id=["workspace"]))

        table_id = ["workspace", "test_table"]

        # Create initial table
        table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
        ds = lance.write_dataset(
            table1, namespace_client=ns_client, table_id=table_id, mode="create"
        )
        assert ds.count_rows() == 2
        assert len(ds.versions()) == 1

        # Verify describe_table returns managed_versioning=True
        describe_resp = ns_client.describe_table(DescribeTableRequest(id=table_id))
        assert describe_resp.managed_versioning is True, (
            f"Expected managed_versioning=True, got {describe_resp.managed_versioning}"
        )

        # Open dataset through namespace - should call list_table_versions for latest
        # Use inner_ns_client for metrics since CustomNamespace delegates to it
        initial_list_count = _get_ops_metric(inner_ns_client, "list_table_versions")
        ds_from_ns_client = lance.dataset(namespace_client=ns_client, table_id=table_id)
        assert ds_from_ns_client.count_rows() == 2
        assert ds_from_ns_client.version == 1
        assert (
            _get_ops_metric(inner_ns_client, "list_table_versions")
            == initial_list_count + 1
        ), "list_table_versions should be called once when opening latest version"

        # Verify create_table_version was called once during CREATE
        assert _get_ops_metric(inner_ns_client, "create_table_version") == 1, (
            "create_table_version should have been called once during CREATE"
        )

        # Append data - should call create_table_version again
        table2 = pa.Table.from_pylist([{"a": 100, "b": 200}, {"a": 1000, "b": 2000}])
        ds = lance.write_dataset(
            table2, namespace_client=ns_client, table_id=table_id, mode="append"
        )
        assert ds.count_rows() == 4
        assert len(ds.versions()) == 2

        assert _get_ops_metric(inner_ns_client, "create_table_version") == 2, (
            "create_table_version should be called twice (CREATE + APPEND)"
        )

        # Open latest version - should call list_table_versions
        list_count_before_latest = _get_ops_metric(
            inner_ns_client, "list_table_versions"
        )
        latest_ds = lance.dataset(namespace_client=ns_client, table_id=table_id)
        assert latest_ds.count_rows() == 4
        assert latest_ds.version == 2
        assert (
            _get_ops_metric(inner_ns_client, "list_table_versions")
            == list_count_before_latest + 1
        ), "list_table_versions should be called once when opening latest version"

        # Open specific version (v1) - should call describe_table_version
        describe_count_before_v1 = _get_ops_metric(
            inner_ns_client, "describe_table_version"
        )
        v1_ds = lance.dataset(namespace_client=ns_client, table_id=table_id, version=1)
        assert v1_ds.count_rows() == 2
        assert v1_ds.version == 1
        assert (
            _get_ops_metric(inner_ns_client, "describe_table_version")
            == describe_count_before_v1 + 1
        ), "describe_table_version should be called once when opening version 1"


def test_dataset_namespace_open_does_not_pass_version_to_describe_table():
    """Dataset versions are applied to dataset open, not namespace describe_table."""

    class VersionRejectingNamespace(CustomNamespace):
        def __init__(self, inner: lance.namespace.DirectoryNamespace):
            super().__init__(inner)
            self.describe_versions = []

        def describe_table(
            self, request: DescribeTableRequest
        ) -> DescribeTableResponse:
            self.describe_versions.append(request.version)
            assert request.version is None
            return super().describe_table(request)

    with tempfile.TemporaryDirectory() as tmpdir:
        inner_ns_client = lance.namespace.DirectoryNamespace(root=tmpdir)
        ns_client = VersionRejectingNamespace(inner_ns_client)
        table_id = ["test_table"]

        table1 = pa.Table.from_pylist([{"a": 1}, {"a": 2}])
        ds = lance.write_dataset(
            table1, namespace_client=ns_client, table_id=table_id, mode="create"
        )
        assert ds.count_rows() == 2
        assert ds.version == 1

        table2 = pa.Table.from_pylist([{"a": 3}])
        ds = lance.write_dataset(
            table2, namespace_client=ns_client, table_id=table_id, mode="append"
        )
        assert ds.count_rows() == 3
        assert ds.version == 2

        version_one = lance.dataset(
            namespace_client=ns_client, table_id=table_id, version=1
        )
        assert version_one.count_rows() == 2
        assert version_one.version == 1
        assert ns_client.describe_versions
        assert all(version is None for version in ns_client.describe_versions)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows file locking prevents reliable concurrent filesystem operations",
)
class TestConcurrentOperations:
    """Tests for concurrent table operations.

    These tests mirror the Rust and Java concurrent tests to ensure
    the DirectoryNamespace handles concurrent create/drop operations correctly.
    Tests are parameterized via temp_ns_client fixture to test both
    DirectoryNamespace and CustomNamespace.
    """

    def test_concurrent_create_and_drop_single_instance(self, temp_ns_client):
        """Test concurrent create/drop with single namespace instance."""
        import concurrent.futures

        # Initialize namespace first - create parent namespace to ensure __manifest
        # table is created before concurrent operations
        create_ns_req = CreateNamespaceRequest(id=["test_ns"])
        temp_ns_client.create_namespace(create_ns_req)

        num_tables = 10
        success_count = 0
        fail_count = 0
        lock = Lock()

        def create_and_drop_table(table_index):
            nonlocal success_count, fail_count
            try:
                table_name = f"concurrent_table_{table_index}"
                table_id = ["test_ns", table_name]
                table_data = create_test_data()
                ipc_data = table_to_ipc_bytes(table_data)

                # Create table
                create_req = CreateTableRequest(id=table_id)
                temp_ns_client.create_table(create_req, ipc_data)

                # Drop table
                drop_req = DropTableRequest(id=table_id)
                temp_ns_client.drop_table(drop_req)

                with lock:
                    success_count += 1
            except Exception as e:
                with lock:
                    fail_count += 1
                raise e

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_tables) as executor:
            futures = [
                executor.submit(create_and_drop_table, i) for i in range(num_tables)
            ]
            concurrent.futures.wait(futures)

        assert success_count == num_tables, (
            f"Expected {num_tables} successes, got {success_count}"
        )
        assert fail_count == 0, f"Expected 0 failures, got {fail_count}"

        # Verify all tables are dropped
        list_req = ListTablesRequest(id=["test_ns"])
        response = temp_ns_client.list_tables(list_req)
        assert len(response.tables) == 0, "All tables should be dropped"

    @pytest.mark.parametrize(
        "use_custom", [False, True], ids=["DirectoryNS", "CustomNS"]
    )
    def test_concurrent_create_and_drop_multiple_instances(self, use_custom):
        """Test concurrent create/drop with multiple namespace instances."""
        import concurrent.futures

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize namespace first with a single instance to ensure __manifest
            # table is created and parent namespace exists before concurrent operations
            init_ns_client = connect(
                "dir",
                {"root": f"file://{tmpdir}", "commit_retries": "2147483647"},
            )
            init_ns_client = _wrap_if_custom(init_ns_client, use_custom)
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
                    # Use high commit_retries to handle version collisions
                    local_ns_client = connect(
                        "dir",
                        {"root": f"file://{tmpdir}", "commit_retries": "2147483647"},
                    )
                    local_ns_client = _wrap_if_custom(local_ns_client, use_custom)

                    table_name = f"multi_ns_table_{table_index}"
                    table_id = ["test_ns", table_name]
                    table_data = create_test_data()
                    ipc_data = table_to_ipc_bytes(table_data)

                    # Create table
                    create_req = CreateTableRequest(id=table_id)
                    local_ns_client.create_table(create_req, ipc_data)

                    # Drop table
                    drop_req = DropTableRequest(id=table_id)
                    local_ns_client.drop_table(drop_req)

                    with lock:
                        success_count += 1
                except Exception as e:
                    with lock:
                        fail_count += 1
                    raise e

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_tables
            ) as executor:
                futures = [
                    executor.submit(create_and_drop_table, i) for i in range(num_tables)
                ]
                concurrent.futures.wait(futures)

            assert success_count == num_tables, (
                f"Expected {num_tables} successes, got {success_count}"
            )
            assert fail_count == 0, f"Expected 0 failures, got {fail_count}"

            # Verify with a fresh namespace client instance
            verify_ns_client = connect(
                "dir", {"root": f"file://{tmpdir}", "commit_retries": "2147483647"}
            )
            verify_ns_client = _wrap_if_custom(verify_ns_client, use_custom)
            list_req = ListTablesRequest(id=["test_ns"])
            response = verify_ns_client.list_tables(list_req)
            assert len(response.tables) == 0, "All tables should be dropped"

    @pytest.mark.parametrize(
        "use_custom", [False, True], ids=["DirectoryNS", "CustomNS"]
    )
    def test_concurrent_create_then_drop_from_different_instance(self, use_custom):
        """Test creating from one set of instances, dropping from different ones."""
        import concurrent.futures

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize namespace first with a single instance to ensure __manifest
            # table is created and parent namespace exists before concurrent operations
            init_ns_client = connect(
                "dir",
                {"root": f"file://{tmpdir}", "commit_retries": "2147483647"},
            )
            init_ns_client = _wrap_if_custom(init_ns_client, use_custom)
            create_ns_req = CreateNamespaceRequest(id=["test_ns"])
            init_ns_client.create_namespace(create_ns_req)

            num_tables = 10

            # Phase 1: Create all tables concurrently using separate namespace instances
            create_success_count = 0
            create_fail_count = 0
            create_lock = Lock()

            def create_table(table_index):
                nonlocal create_success_count, create_fail_count
                try:
                    # Use high commit_retries to handle version collisions
                    local_ns_client = connect(
                        "dir",
                        {"root": f"file://{tmpdir}", "commit_retries": "2147483647"},
                    )
                    local_ns_client = _wrap_if_custom(local_ns_client, use_custom)

                    table_name = f"cross_instance_table_{table_index}"
                    table_id = ["test_ns", table_name]
                    table_data = create_test_data()
                    ipc_data = table_to_ipc_bytes(table_data)

                    create_req = CreateTableRequest(id=table_id)
                    local_ns_client.create_table(create_req, ipc_data)

                    with create_lock:
                        create_success_count += 1
                except Exception as e:
                    with create_lock:
                        create_fail_count += 1
                    raise e

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_tables
            ) as executor:
                futures = [executor.submit(create_table, i) for i in range(num_tables)]
                concurrent.futures.wait(futures)

            assert create_success_count == num_tables, (
                f"All creates should succeed, got {create_success_count}"
            )

            # Phase 2: Drop all tables concurrently using NEW namespace instances
            drop_success_count = 0
            drop_fail_count = 0
            drop_lock = Lock()

            def drop_table(table_index):
                nonlocal drop_success_count, drop_fail_count
                try:
                    # Use high commit_retries to handle version collisions
                    local_ns_client = connect(
                        "dir",
                        {"root": f"file://{tmpdir}", "commit_retries": "2147483647"},
                    )
                    local_ns_client = _wrap_if_custom(local_ns_client, use_custom)

                    table_name = f"cross_instance_table_{table_index}"
                    table_id = ["test_ns", table_name]

                    drop_req = DropTableRequest(id=table_id)
                    local_ns_client.drop_table(drop_req)

                    with drop_lock:
                        drop_success_count += 1
                except Exception as e:
                    with drop_lock:
                        drop_fail_count += 1
                    raise e

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_tables
            ) as executor:
                futures = [executor.submit(drop_table, i) for i in range(num_tables)]
                concurrent.futures.wait(futures)

            assert drop_success_count == num_tables, (
                f"All drops should succeed, got {drop_success_count}"
            )
            assert drop_fail_count == 0, f"No drops should fail, got {drop_fail_count}"

            # Verify all tables are dropped
            verify_ns_client = connect(
                "dir", {"root": f"file://{tmpdir}", "commit_retries": "2147483647"}
            )
            verify_ns_client = _wrap_if_custom(verify_ns_client, use_custom)
            list_req = ListTablesRequest(id=["test_ns"])
            response = verify_ns_client.list_tables(list_req)
            assert len(response.tables) == 0, "All tables should be dropped"


class TestDataManipulation:
    """Tests for data manipulation operations."""

    def test_count_table_rows(self, temp_ns_client):
        """Test counting rows in a table."""
        # Create namespace and table
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        table_data = create_test_data()  # 3 rows
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        temp_ns_client.create_table(create_req, ipc_data)

        # Count rows
        count_req = CountTableRowsRequest(id=["workspace", "test_table"])
        count = temp_ns_client.count_table_rows(count_req).count
        assert count == 3

    def test_count_table_rows_with_filter(self, temp_ns_client):
        """Test counting rows with a filter predicate."""
        # Create namespace and table
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        table_data = create_test_data()  # 3 rows with ages 30, 25, 35
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        temp_ns_client.create_table(create_req, ipc_data)

        # Count rows with predicate
        count_req = CountTableRowsRequest(
            id=["workspace", "test_table"], predicate="age > 28"
        )
        count = temp_ns_client.count_table_rows(count_req).count
        assert count == 2  # Alice (30) and Charlie (35)

    def test_insert_into_table(self, temp_ns_client):
        """Test inserting data into a table."""
        # Create namespace and table
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        table_data = create_test_data()  # 3 rows
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        temp_ns_client.create_table(create_req, ipc_data)

        # Insert more data
        new_data = pa.Table.from_pylist(
            [
                {"id": 4, "name": "David", "age": 40},
                {"id": 5, "name": "Eve", "age": 22},
            ]
        )
        new_ipc_data = table_to_ipc_bytes(new_data)
        insert_req = InsertIntoTableRequest(
            id=["workspace", "test_table"], mode="append"
        )
        response = temp_ns_client.insert_into_table(insert_req, new_ipc_data)
        assert response is not None

        # Verify row count increased
        count_req = CountTableRowsRequest(id=["workspace", "test_table"])
        count = temp_ns_client.count_table_rows(count_req).count
        assert count == 5

    def test_query_table(self, temp_ns_client):
        """Test querying a table."""
        # Create namespace and table
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        table_data = create_test_data()  # 3 rows
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        temp_ns_client.create_table(create_req, ipc_data)

        # Query table with empty vector (for non-vector queries)
        query_req = QueryTableRequest(id=["workspace", "test_table"], k=10, vector={})
        result_bytes = temp_ns_client.query_table(query_req).data
        assert result_bytes is not None
        assert len(result_bytes) > 0

        # Parse the result
        reader = pa.ipc.open_file(pa.BufferReader(result_bytes))
        result_table = reader.read_all()
        assert result_table.num_rows == 3
        assert "id" in result_table.column_names
        assert "name" in result_table.column_names

    def test_query_table_with_filter(self, temp_ns_client):
        """Test querying a table with a filter."""
        # Create namespace and table
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        table_data = create_test_data()  # 3 rows
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        temp_ns_client.create_table(create_req, ipc_data)

        # Query with filter and empty vector
        query_req = QueryTableRequest(
            id=["workspace", "test_table"], filter="age >= 30", k=10, vector={}
        )
        result_bytes = temp_ns_client.query_table(query_req).data
        reader = pa.ipc.open_file(pa.BufferReader(result_bytes))
        result_table = reader.read_all()
        assert result_table.num_rows == 2  # Alice and Charlie


class TestTableVersions:
    """Tests for table version operations."""

    def test_list_table_versions(self, temp_ns_client):
        """Test listing table versions."""
        # Create namespace and table
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        temp_ns_client.create_table(create_req, ipc_data)

        # List versions
        list_req = ListTableVersionsRequest(id=["workspace", "test_table"])
        response = temp_ns_client.list_table_versions(list_req)
        assert response is not None
        assert len(response.versions) >= 1

    def test_describe_table_version(self, temp_ns_client):
        """Test describing a specific table version."""
        # Create namespace and table
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        temp_ns_client.create_table(create_req, ipc_data)

        # Describe version 1
        describe_req = DescribeTableVersionRequest(
            id=["workspace", "test_table"], version=1
        )
        response = temp_ns_client.describe_table_version(describe_req.model_dump())
        assert response is not None
        assert response.get("version") is not None

    def test_multiple_versions_via_insert(self, temp_ns_client):
        """Test that inserts create new versions."""
        # Create namespace and table
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        temp_ns_client.create_namespace(create_ns_req)

        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        temp_ns_client.create_table(create_req, ipc_data)

        # Insert more data to create version 2
        new_data = pa.Table.from_pylist([{"id": 4, "name": "David", "age": 40}])
        new_ipc_data = table_to_ipc_bytes(new_data)
        insert_req = InsertIntoTableRequest(
            id=["workspace", "test_table"], mode="append"
        )
        temp_ns_client.insert_into_table(insert_req, new_ipc_data)

        # List versions - should have at least 2
        list_req = ListTableVersionsRequest(id=["workspace", "test_table"])
        response = temp_ns_client.list_table_versions(list_req)
        assert len(response.versions) >= 2


class TestIndexOperations:
    """Tests for index operations."""

    def test_list_indices_empty(self, temp_ns_client):
        """Test listing indices on a table with no indices."""
        # Create table with a vector column
        import numpy as np

        vector_data = pa.Table.from_pydict(
            {
                "id": [1, 2, 3],
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(np.random.rand(12).astype(np.float32)), 4
                ),
            }
        )
        ipc_data = table_to_ipc_bytes(vector_data)
        create_req = CreateTableRequest(id=["vector_table"])
        temp_ns_client.create_table(create_req, ipc_data)

        # List indices (should be empty initially)
        list_req = ListTableIndicesRequest(id=["vector_table"])
        response = temp_ns_client.list_table_indices(list_req)
        assert response is not None
        # Initially no indices
        assert len(response.indexes) == 0

    def test_describe_table_index_stats(self, memory_ns_client):
        """Test describing index stats (even when no index exists)."""
        # Create namespace and table
        create_ns_req = CreateNamespaceRequest(id=["workspace"])
        memory_ns_client.create_namespace(create_ns_req)

        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["workspace", "test_table"])
        memory_ns_client.create_table(create_req, ipc_data)

        # Describe index stats for non-existent index should return empty/null stats
        describe_req = DescribeTableIndexStatsRequest(
            id=["workspace", "test_table"], index_name="nonexistent"
        )
        # This may raise an error or return empty stats depending on implementation
        try:
            response = memory_ns_client.describe_table_index_stats(describe_req)
            # If it succeeds, verify response structure
            assert response is not None
        except Exception:
            # Expected if index doesn't exist
            pass

    def test_create_scalar_index(self, temp_ns_client):
        """Test creating a scalar index."""
        # Create table at root level (without namespace)
        table_data = create_test_data()
        ipc_data = table_to_ipc_bytes(table_data)
        create_req = CreateTableRequest(id=["test_table"])
        temp_ns_client.create_table(create_req, ipc_data)

        # Create scalar index on 'id' column
        create_index_req = CreateTableIndexRequest(
            id=["test_table"],
            column="id",
            index_type="BTREE",
            name="id_idx",
        )
        response = temp_ns_client.create_table_index(create_index_req)
        assert response is not None

        # List indices to verify
        list_req = ListTableIndicesRequest(id=["test_table"])
        list_response = temp_ns_client.list_table_indices(list_req)
        assert len(list_response.indexes) == 1
        assert list_response.indexes[0].index_name == "id_idx"
        assert list_response.indexes[0].columns == ["id"]

    def test_create_vector_index(self, temp_ns_client):
        """Test creating a vector index."""
        import numpy as np

        # Create table with 256 rows of 8-dimensional vectors (enough for IVF)
        num_rows = 256
        dim = 8
        vector_data = pa.Table.from_pydict(
            {
                "id": list(range(num_rows)),
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(np.random.rand(num_rows * dim).astype(np.float32)), dim
                ),
            }
        )
        ipc_data = table_to_ipc_bytes(vector_data)
        create_req = CreateTableRequest(id=["vector_table"])
        temp_ns_client.create_table(create_req, ipc_data)

        # Create vector index using IVF_FLAT
        create_index_req = CreateTableIndexRequest(
            id=["vector_table"],
            column="vector",
            index_type="IVF_FLAT",
            name="vector_idx",
            distance_type="l2",
        )
        response = temp_ns_client.create_table_index(create_index_req)
        assert response is not None

        # List indices to verify
        list_req = ListTableIndicesRequest(id=["vector_table"])
        list_response = temp_ns_client.list_table_indices(list_req)
        assert len(list_response.indexes) == 1
        assert list_response.indexes[0].index_name == "vector_idx"
        assert list_response.indexes[0].columns == ["vector"]
