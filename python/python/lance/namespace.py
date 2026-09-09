# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""LanceNamespace storage options integration and implementations.

This module provides:
1. Native Rust-backed namespace implementations (DirectoryNamespace, RestNamespace)
2. Storage options integration with LanceNamespace for automatic credential refresh
3. Plugin registry for external namespace implementations
4. Dynamic context provider registry for per-request context injection

The LanceNamespace ABC interface is provided by the lance_namespace package.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from lance_namespace import (
    AlterTableAddColumnsRequest,
    AlterTableAddColumnsResponse,
    AlterTableAlterColumnsRequest,
    AlterTableAlterColumnsResponse,
    AlterTableBackfillColumnsRequest,
    AlterTableBackfillColumnsResponse,
    AlterTableDropColumnsRequest,
    AlterTableDropColumnsResponse,
    AlterTransactionRequest,
    AlterTransactionResponse,
    AnalyzeTableQueryPlanRequest,
    CountTableRowsRequest,
    CountTableRowsResponse,
    CreateMaterializedViewRequest,
    CreateMaterializedViewResponse,
    CreateNamespaceRequest,
    CreateNamespaceResponse,
    CreateTableBranchRequest,
    CreateTableBranchResponse,
    CreateTableIndexRequest,
    CreateTableIndexResponse,
    CreateTableRequest,
    CreateTableResponse,
    CreateTableTagRequest,
    CreateTableTagResponse,
    DeclareTableRequest,
    DeclareTableResponse,
    DeleteFromTableRequest,
    DeleteFromTableResponse,
    DeleteTableBranchRequest,
    DeleteTableBranchResponse,
    DeleteTableTagRequest,
    DeleteTableTagResponse,
    DeregisterTableRequest,
    DeregisterTableResponse,
    DescribeNamespaceRequest,
    DescribeNamespaceResponse,
    DescribeTableIndexStatsRequest,
    DescribeTableIndexStatsResponse,
    DescribeTableRequest,
    DescribeTableResponse,
    DescribeTransactionRequest,
    DescribeTransactionResponse,
    DropNamespaceRequest,
    DropNamespaceResponse,
    DropTableIndexRequest,
    DropTableIndexResponse,
    DropTableRequest,
    DropTableResponse,
    ExplainTableQueryPlanRequest,
    GetTableStatsRequest,
    GetTableStatsResponse,
    GetTableTagVersionRequest,
    GetTableTagVersionResponse,
    InsertIntoTableRequest,
    InsertIntoTableResponse,
    LanceNamespace,
    ListNamespacesRequest,
    ListNamespacesResponse,
    ListTableBranchesRequest,
    ListTableBranchesResponse,
    ListTableIndicesRequest,
    ListTableIndicesResponse,
    ListTablesRequest,
    ListTablesResponse,
    ListTableTagsRequest,
    ListTableTagsResponse,
    ListTableVersionsRequest,
    ListTableVersionsResponse,
    MergeInsertIntoTableRequest,
    MergeInsertIntoTableResponse,
    NamespaceExistsRequest,
    NamespaceExistsResponse,
    QueryTableResponse,
    RefreshMaterializedViewRequest,
    RefreshMaterializedViewResponse,
    RegisterTableRequest,
    RegisterTableResponse,
    RenameTableRequest,
    RenameTableResponse,
    RestoreTableRequest,
    RestoreTableResponse,
    TableExistsRequest,
    TableExistsResponse,
    UpdateTableRequest,
    UpdateTableResponse,
    UpdateTableSchemaMetadataRequest,
    UpdateTableSchemaMetadataResponse,
    UpdateTableTagRequest,
    UpdateTableTagResponse,
)

from .lance import PyDirectoryNamespace  # Low-level Rust binding

try:
    from .lance import PyRestNamespace  # Low-level Rust binding
except ImportError:
    PyRestNamespace = None

try:
    from .lance import PyRestAdapter  # Low-level Rust binding
except ImportError:
    PyRestAdapter = None

__all__ = [
    "DirectoryNamespace",
    "RestNamespace",
    "RestAdapter",
    "DynamicContextProvider",
]


# =============================================================================
# Dynamic Context Provider
# =============================================================================


class DynamicContextProvider(ABC):
    """Abstract base class for dynamic context providers.

    Implementations provide per-request context (e.g., authentication headers)
    based on the operation being performed. The provider is called synchronously
    before each namespace operation.

    For RestNamespace, context keys that start with `headers.` are converted to
    HTTP headers by stripping the prefix. For example, `{"headers.Authorization":
    "Bearer token"}` becomes the `Authorization: Bearer token` header.

    Example
    -------
    >>> # Define a provider class
    >>> class MyProvider(DynamicContextProvider):
    ...     def __init__(self, api_key: str):
    ...         self.api_key = api_key
    ...
    ...     def provide_context(self, info: dict) -> dict:
    ...         return {
    ...             "headers.Authorization": f"Bearer {self.api_key}",
    ...         }
    ...
    >>> # Create provider instance and use directly
    >>> provider = MyProvider(api_key="secret")
    >>> provider.provide_context({"operation": "list_tables", "object_id": "ns"})
    {'headers.Authorization': 'Bearer secret'}
    """

    @abstractmethod
    def provide_context(self, info: Dict[str, str]) -> Dict[str, str]:
        """Provide context for a namespace operation.

        Parameters
        ----------
        info : dict
            Information about the operation:
            - operation: The operation name (e.g., "list_tables", "describe_table")
            - object_id: The object identifier (namespace or table ID)

        Returns
        -------
        dict
            Context key-value pairs. For HTTP headers, use keys with the
            "headers." prefix (e.g., "headers.Authorization").
        """
        pass


def _create_context_provider_from_properties(
    properties: Dict[str, str],
) -> Optional[DynamicContextProvider]:
    """Create a context provider instance from properties.

    Extracts `dynamic_context_provider.*` properties and creates a provider
    instance by dynamically loading the class from the given class path.

    Parameters
    ----------
    properties : dict
        The full properties dict that may contain dynamic_context_provider.* keys.

    Returns
    -------
    DynamicContextProvider or None
        The created provider instance, or None if no provider is configured.

    Raises
    ------
    ValueError
        If dynamic_context_provider.impl is set but the class cannot be loaded.
    """
    import importlib

    prefix = "dynamic_context_provider."
    impl_key = "dynamic_context_provider.impl"

    impl_path = properties.get(impl_key)
    if not impl_path:
        return None

    # Parse the class path (e.g., "my_module.submodule.MyClass")
    if "." not in impl_path:
        raise ValueError(
            f"Invalid context provider class path '{impl_path}'. "
            f"Expected format: 'module.ClassName' (e.g., 'my_module.MyProvider')"
        )

    module_path, class_name = impl_path.rsplit(".", 1)

    try:
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Failed to import module '{module_path}' for context provider: {e}"
        ) from e
    except AttributeError as e:
        raise ValueError(
            f"Class '{class_name}' not found in module '{module_path}': {e}"
        ) from e

    # Extract provider-specific properties (strip prefix, exclude impl key)
    provider_props = {}
    for key, value in properties.items():
        if key.startswith(prefix) and key != impl_key:
            prop_name = key[len(prefix) :]
            provider_props[prop_name] = value

    # Create the provider instance
    return provider_class(**provider_props)


def _filter_context_provider_properties(properties: Dict[str, str]) -> Dict[str, str]:
    """Remove dynamic_context_provider.* properties from the dict.

    These properties are handled at the Python level and should not be
    passed to the Rust layer.

    Parameters
    ----------
    properties : dict
        The full properties dict.

    Returns
    -------
    dict
        Properties with dynamic_context_provider.* keys removed.
    """
    prefix = "dynamic_context_provider."
    return {k: v for k, v in properties.items() if not k.startswith(prefix)}


class DirectoryNamespace(LanceNamespace):
    """Directory-based Lance Namespace implementation backed by Rust.

    This namespace stores tables as Lance datasets in a filesystem directory structure.
    It uses a manifest table to track tables and nested namespaces efficiently.

    This is a Python wrapper around the Rust PyDirectoryNamespace implementation,
    providing compatibility with the LanceNamespace ABC interface.

    Parameters
    ----------
    session : Session, optional
        Lance session for sharing object store connections. If provided,
        this namespace will reuse the session's object store registry.
    **properties : dict
        Configuration properties as key-value pairs:
        - root (required): Root directory path or URI
        - manifest_enabled (optional): Enable manifest tracking (default: "true")
        - dir_listing_enabled (optional): Enable directory listing fallback
          (default: "true")
        - storage.* (optional): Storage options with "storage." prefix
          (e.g., storage.region="us-west-2" becomes region="us-west-2" in
          storage options)

        Credential vendor properties (vendor is auto-selected based on table location):
            When credential vendor properties are configured, describe_table() will
            return vended temporary credentials. The vendor type is auto-selected
            based on table location URI: s3:// for AWS, gs:// for GCP, az:// for
            Azure. Requires the corresponding credential-vendor-* feature.

            Common properties:
                - credential_vendor.enabled (required): Set to "true" to enable
                - credential_vendor.permission (optional): read, write, or admin

            AWS-specific properties (for s3:// locations):
                - credential_vendor.aws_role_arn (required): IAM role ARN to assume
                - credential_vendor.aws_external_id (optional): External ID
                - credential_vendor.aws_region (optional): AWS region
                - credential_vendor.aws_role_session_name (optional): Session name
                - credential_vendor.aws_duration_millis (optional): Duration in ms
                  (default: 3600000, range: 15min-12hrs)

            GCP-specific properties (for gs:// locations):
                - credential_vendor.gcp_service_account (optional): Service account
                  to impersonate using IAM Credentials API

                Note: GCP uses Application Default Credentials (ADC). To use a service
                account key file, set the GOOGLE_APPLICATION_CREDENTIALS environment
                variable before starting. GCP token duration cannot be configured;
                it's determined by the STS endpoint (typically 1 hour).

            Azure-specific properties (for az:// locations):
                - credential_vendor.azure_account_name (required): Azure storage
                  account name
                - credential_vendor.azure_tenant_id (optional): Azure tenant ID
                - credential_vendor.azure_duration_millis (optional): Duration in ms
                  (default: 3600000, up to 7 days)

        Testing properties:
            - ops_metrics_enabled (optional): "true" to enable operation metrics
              tracking. Use `retrieve_ops_metrics()` to get call counts.
            - vend_input_storage_options (optional): "true" to return input storage
              options in describe_table() when no credential vendor is configured.
              Useful for testing credential refresh.
            - vend_input_storage_options_refresh_interval_millis (optional): When set
              with vend_input_storage_options, adds expires_at_millis to storage
              options. Value is current_time_millis + this interval.

    Examples
    --------
    >>> import lance.namespace
    >>> # Create with properties dict
    >>> ns = lance.namespace.DirectoryNamespace(root="memory://test")
    >>>
    >>> # Using the connect() factory function from lance_namespace
    >>> import lance_namespace
    >>> ns = lance_namespace.connect("dir", {"root": "memory://test"})
    >>>
    >>> # With AWS credential vending (requires credential-vendor-aws feature)
    >>> # Use **dict to pass property names with dots
    >>> aws_properties = {
    ...     "root": "s3://my-bucket/data",
    ...     "credential_vendor.enabled": "true",
    ...     "credential_vendor.aws_role_arn": "arn:aws:iam::123456789012:role/MyRole",
    ...     "credential_vendor.aws_duration_millis": "3600000",
    ... }
    >>> # ns = lance.namespace.DirectoryNamespace(**aws_properties)

    With dynamic context provider:

    >>> import tempfile
    >>> class MyProvider(DynamicContextProvider):
    ...     def __init__(self, token: str):
    ...         self.token = token
    ...     def provide_context(self, info: dict) -> dict:
    ...         return {"headers.Authorization": f"Bearer {self.token}"}
    ...
    >>> provider = MyProvider(token="secret-token")
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     ns = lance.namespace.DirectoryNamespace(
    ...         root=tmpdir,
    ...         context_provider=provider,
    ...     )
    ...     _ = ns.namespace_id()  # verify it works
    """

    def __init__(self, session=None, context_provider=None, **properties):
        # Convert all values to strings as expected by Rust from_properties
        str_properties = {str(k): str(v) for k, v in properties.items()}

        # Create context provider from properties if configured
        if context_provider is None:
            context_provider = _create_context_provider_from_properties(str_properties)

        # Filter out dynamic_context_provider.* properties before passing to Rust
        filtered_properties = _filter_context_provider_properties(str_properties)

        # Create the underlying Rust namespace
        self._inner = PyDirectoryNamespace(
            session=session, context_provider=context_provider, **filtered_properties
        )

    def namespace_id(self) -> str:
        """Return a human-readable unique identifier for this namespace instance."""
        return self._inner.namespace_id()

    def __repr__(self) -> str:
        return f"DirectoryNamespace({self._inner.namespace_id()})"

    # Namespace operations

    def create_namespace(
        self, request: CreateNamespaceRequest
    ) -> CreateNamespaceResponse:
        response_dict = self._inner.create_namespace(request.model_dump())
        return CreateNamespaceResponse.from_dict(response_dict)

    def list_namespaces(self, request: ListNamespacesRequest) -> ListNamespacesResponse:
        response_dict = self._inner.list_namespaces(request.model_dump())
        return ListNamespacesResponse.from_dict(response_dict)

    def describe_namespace(
        self, request: DescribeNamespaceRequest
    ) -> DescribeNamespaceResponse:
        response_dict = self._inner.describe_namespace(request.model_dump())
        return DescribeNamespaceResponse.from_dict(response_dict)

    def drop_namespace(self, request: DropNamespaceRequest) -> DropNamespaceResponse:
        response_dict = self._inner.drop_namespace(request.model_dump())
        return DropNamespaceResponse.from_dict(response_dict)

    def namespace_exists(
        self, request: NamespaceExistsRequest
    ) -> NamespaceExistsResponse:
        self._inner.namespace_exists(request.model_dump())
        return NamespaceExistsResponse()

    # Table operations

    def list_tables(self, request: ListTablesRequest) -> ListTablesResponse:
        response_dict = self._inner.list_tables(request.model_dump())
        return ListTablesResponse.from_dict(response_dict)

    def describe_table(self, request: DescribeTableRequest) -> DescribeTableResponse:
        response_dict = self._inner.describe_table(request.model_dump())
        return DescribeTableResponse.from_dict(response_dict)

    def register_table(self, request: RegisterTableRequest) -> RegisterTableResponse:
        response_dict = self._inner.register_table(request.model_dump())
        return RegisterTableResponse.from_dict(response_dict)

    def table_exists(self, request: TableExistsRequest) -> TableExistsResponse:
        self._inner.table_exists(request.model_dump())
        return TableExistsResponse()

    def drop_table(self, request: DropTableRequest) -> DropTableResponse:
        response_dict = self._inner.drop_table(request.model_dump())
        return DropTableResponse.from_dict(response_dict)

    def deregister_table(
        self, request: DeregisterTableRequest
    ) -> DeregisterTableResponse:
        response_dict = self._inner.deregister_table(request.model_dump())
        return DeregisterTableResponse.from_dict(response_dict)

    def create_table(
        self, request: CreateTableRequest, request_data: bytes
    ) -> CreateTableResponse:
        response_dict = self._inner.create_table(request.model_dump(), request_data)
        return CreateTableResponse.from_dict(response_dict)

    def declare_table(self, request: DeclareTableRequest) -> DeclareTableResponse:
        response_dict = self._inner.declare_table(request.model_dump())
        return DeclareTableResponse.from_dict(response_dict)

    def rename_table(self, request: RenameTableRequest) -> RenameTableResponse:
        response_dict = self._inner.rename_table(request.model_dump())
        return RenameTableResponse.from_dict(response_dict)

    # Table version operations

    def list_table_versions(
        self, request: ListTableVersionsRequest
    ) -> ListTableVersionsResponse:
        response_dict = self._inner.list_table_versions(request.model_dump())
        return ListTableVersionsResponse.from_dict(response_dict)

    def create_table_version(self, request: dict) -> dict:
        """Create a table version (for external manifest store integration).

        Parameters
        ----------
        request : dict
            Request dictionary with keys:
            - id: List[str] - Table identifier
            - version: int - Version number to create
            - manifest_path: str - Path to staging manifest
            - manifest_size: int (optional) - Size in bytes
            - e_tag: str (optional) - ETag for optimistic concurrency

        Returns
        -------
        dict
            Response dictionary with optional transaction_id
        """
        return self._inner.create_table_version(request)

    def describe_table_version(self, request: dict) -> dict:
        """Describe a specific table version.

        Parameters
        ----------
        request : dict
            Request dictionary with keys:
            - id: List[str] - Table identifier
            - version: int (optional) - Version to describe (None = latest)

        Returns
        -------
        dict
            Response dictionary with version info:
            - version: dict with version, manifest_path, manifest_size, e_tag, timestamp
        """
        return self._inner.describe_table_version(request)

    def batch_delete_table_versions(self, request: dict) -> dict:
        """Delete multiple table versions in a single request.

        Parameters
        ----------
        request : dict
            Request dictionary with keys:
            - id: List[str] - Table identifier
            - versions: List[int] - List of version numbers to delete

        Returns
        -------
        dict
            Response dictionary with:
            - deleted_versions: List[int] - List of successfully deleted versions
        """
        return self._inner.batch_delete_table_versions(request)

    # Data manipulation operations

    def count_table_rows(
        self, request: CountTableRowsRequest
    ) -> CountTableRowsResponse:
        """Count the number of rows in a table, optionally filtered by a predicate.

        Parameters
        ----------
        request : CountTableRowsRequest
            Request with table id, optional version, and optional predicate filter

        Returns
        -------
        CountTableRowsResponse
            Response whose ``count`` is the number of rows matching the criteria
        """
        count = self._inner.count_table_rows(request.model_dump())
        return CountTableRowsResponse(count=count)

    def insert_into_table(
        self, request: InsertIntoTableRequest, request_data: bytes
    ) -> InsertIntoTableResponse:
        """Insert data into a table.

        Parameters
        ----------
        request : InsertIntoTableRequest
            Request with table id and write mode (append/overwrite)
        request_data : bytes
            Arrow IPC stream containing the data to insert

        Returns
        -------
        InsertIntoTableResponse
            Response with optional transaction_id
        """
        response_dict = self._inner.insert_into_table(
            request.model_dump(), request_data
        )
        return InsertIntoTableResponse.from_dict(response_dict)

    def merge_insert_into_table(
        self, request: MergeInsertIntoTableRequest, request_data: bytes
    ) -> MergeInsertIntoTableResponse:
        """Merge-insert data into a table (upsert operation).

        Parameters
        ----------
        request : MergeInsertIntoTableRequest
            Request with table id, merge keys, and merge behavior
        request_data : bytes
            Arrow IPC stream containing the data to merge

        Returns
        -------
        MergeInsertIntoTableResponse
            Response with optional transaction_id
        """
        response_dict = self._inner.merge_insert_into_table(
            request.model_dump(), request_data
        )
        return MergeInsertIntoTableResponse.from_dict(response_dict)

    def update_table(self, request: UpdateTableRequest) -> UpdateTableResponse:
        """Update rows in a table matching a filter.

        Parameters
        ----------
        request : UpdateTableRequest
            Request with table id, filter predicate, and column updates

        Returns
        -------
        UpdateTableResponse
            Response with optional transaction_id
        """
        response_dict = self._inner.update_table(request.model_dump())
        return UpdateTableResponse.from_dict(response_dict)

    def delete_from_table(
        self, request: DeleteFromTableRequest
    ) -> DeleteFromTableResponse:
        """Delete rows from a table matching a filter.

        Parameters
        ----------
        request : DeleteFromTableRequest
            Request with table id and filter predicate

        Returns
        -------
        DeleteFromTableResponse
            Response with optional transaction_id
        """
        response_dict = self._inner.delete_from_table(request.model_dump())
        return DeleteFromTableResponse.from_dict(response_dict)

    def query_table(self, request) -> QueryTableResponse:
        """Query a table and return results as Arrow IPC.

        Parameters
        ----------
        request : QueryTableRequest or dict
            Query request with table id, optional filter, columns, vector search
            parameters, limit/offset, etc.

        Returns
        -------
        QueryTableResponse
            Response whose ``data`` is the query results in Arrow IPC file format
        """
        if hasattr(request, "model_dump"):
            request = request.model_dump()
        data = self._inner.query_table(request)
        return QueryTableResponse(data=data)

    # Index operations

    def create_table_index(
        self, request: CreateTableIndexRequest
    ) -> CreateTableIndexResponse:
        """Create an index on a table.

        Parameters
        ----------
        request : CreateTableIndexRequest
            Request with table id, index name, column, and index configuration

        Returns
        -------
        CreateTableIndexResponse
            Response with optional transaction_id
        """
        response_dict = self._inner.create_table_index(request.model_dump())
        return CreateTableIndexResponse.from_dict(response_dict)

    def list_table_indices(
        self, request: ListTableIndicesRequest
    ) -> ListTableIndicesResponse:
        """List all indices on a table.

        Parameters
        ----------
        request : ListTableIndicesRequest
            Request with table id

        Returns
        -------
        ListTableIndicesResponse
            Response with list of index metadata
        """
        response_dict = self._inner.list_table_indices(request.model_dump())
        return ListTableIndicesResponse.from_dict(response_dict)

    def describe_table_index_stats(
        self, request: DescribeTableIndexStatsRequest
    ) -> DescribeTableIndexStatsResponse:
        """Get statistics for a specific index.

        Parameters
        ----------
        request : DescribeTableIndexStatsRequest
            Request with table id and index name

        Returns
        -------
        DescribeTableIndexStatsResponse
            Response with index statistics
        """
        response_dict = self._inner.describe_table_index_stats(request.model_dump())
        return DescribeTableIndexStatsResponse.from_dict(response_dict)

    # Transaction operations

    def describe_transaction(
        self, request: DescribeTransactionRequest
    ) -> DescribeTransactionResponse:
        """Describe a transaction.

        Parameters
        ----------
        request : DescribeTransactionRequest
            Request with transaction id

        Returns
        -------
        DescribeTransactionResponse
            Response with transaction details
        """
        response_dict = self._inner.describe_transaction(request.model_dump())
        return DescribeTransactionResponse.from_dict(response_dict)

    def alter_transaction(
        self, request: AlterTransactionRequest
    ) -> AlterTransactionResponse:
        """Alter a transaction (commit or abort).

        Parameters
        ----------
        request : AlterTransactionRequest
            Request with transaction id and action

        Returns
        -------
        AlterTransactionResponse
            Response confirming the action
        """
        response_dict = self._inner.alter_transaction(request.model_dump())
        return AlterTransactionResponse.from_dict(response_dict)

    # Additional index operations

    def create_table_scalar_index(
        self, request: CreateTableIndexRequest
    ) -> CreateTableIndexResponse:
        """Create a scalar index on a table column."""
        response_dict = self._inner.create_table_scalar_index(request.model_dump())
        return CreateTableIndexResponse.from_dict(response_dict)

    def drop_table_index(
        self, request: DropTableIndexRequest
    ) -> DropTableIndexResponse:
        """Drop an index from a table."""
        response_dict = self._inner.drop_table_index(request.model_dump())
        return DropTableIndexResponse.from_dict(response_dict)

    # Additional table operations

    def list_all_tables(self, request: ListTablesRequest) -> ListTablesResponse:
        """List all tables recursively across all namespaces."""
        response_dict = self._inner.list_all_tables(request.model_dump())
        return ListTablesResponse.from_dict(response_dict)

    def restore_table(self, request: RestoreTableRequest) -> RestoreTableResponse:
        """Restore a previously dropped table."""
        response_dict = self._inner.restore_table(request.model_dump())
        return RestoreTableResponse.from_dict(response_dict)

    def update_table_schema_metadata(
        self, request: UpdateTableSchemaMetadataRequest
    ) -> UpdateTableSchemaMetadataResponse:
        """Update the schema metadata of a table."""
        response_dict = self._inner.update_table_schema_metadata(request.model_dump())
        return UpdateTableSchemaMetadataResponse.from_dict(response_dict)

    def get_table_stats(self, request: GetTableStatsRequest) -> GetTableStatsResponse:
        """Get statistics for a table."""
        response_dict = self._inner.get_table_stats(request.model_dump())
        return GetTableStatsResponse.from_dict(response_dict)

    # Query plan operations

    def explain_table_query_plan(self, request: ExplainTableQueryPlanRequest) -> str:
        """Explain the query plan for a table query."""
        return self._inner.explain_table_query_plan(request.model_dump())

    def analyze_table_query_plan(self, request: AnalyzeTableQueryPlanRequest) -> str:
        """Analyze the query plan for a table query with execution statistics."""
        return self._inner.analyze_table_query_plan(request.model_dump())

    # Column alteration operations

    def alter_table_add_columns(
        self, request: AlterTableAddColumnsRequest
    ) -> AlterTableAddColumnsResponse:
        """Add columns to a table."""
        response_dict = self._inner.alter_table_add_columns(request.model_dump())
        return AlterTableAddColumnsResponse.from_dict(response_dict)

    def alter_table_alter_columns(
        self, request: AlterTableAlterColumnsRequest
    ) -> AlterTableAlterColumnsResponse:
        """Alter existing columns in a table."""
        response_dict = self._inner.alter_table_alter_columns(request.model_dump())
        return AlterTableAlterColumnsResponse.from_dict(response_dict)

    def alter_table_drop_columns(
        self, request: AlterTableDropColumnsRequest
    ) -> AlterTableDropColumnsResponse:
        """Drop columns from a table."""
        response_dict = self._inner.alter_table_drop_columns(request.model_dump())
        return AlterTableDropColumnsResponse.from_dict(response_dict)

    def alter_table_backfill_columns(
        self, request: AlterTableBackfillColumnsRequest
    ) -> AlterTableBackfillColumnsResponse:
        """Trigger an async backfill job for a computed column."""
        response_dict = self._inner.alter_table_backfill_columns(request.model_dump())
        return AlterTableBackfillColumnsResponse.from_dict(response_dict)

    def refresh_materialized_view(
        self, request: RefreshMaterializedViewRequest
    ) -> RefreshMaterializedViewResponse:
        """Trigger an async materialized view refresh."""
        response_dict = self._inner.refresh_materialized_view(request.model_dump())
        return RefreshMaterializedViewResponse.from_dict(response_dict)

    def create_materialized_view(
        self, request: CreateMaterializedViewRequest
    ) -> CreateMaterializedViewResponse:
        """Create a materialized view backed by an optional UDTF/chunker."""
        response_dict = self._inner.create_materialized_view(request.model_dump())
        return CreateMaterializedViewResponse.from_dict(response_dict)

    # Table tag operations

    def list_table_tags(self, request: ListTableTagsRequest) -> ListTableTagsResponse:
        """List all tags on a table."""
        response_dict = self._inner.list_table_tags(request.model_dump())
        return ListTableTagsResponse.from_dict(response_dict)

    def get_table_tag_version(
        self, request: GetTableTagVersionRequest
    ) -> GetTableTagVersionResponse:
        """Get the version associated with a table tag."""
        response_dict = self._inner.get_table_tag_version(request.model_dump())
        return GetTableTagVersionResponse.from_dict(response_dict)

    def create_table_tag(
        self, request: CreateTableTagRequest
    ) -> CreateTableTagResponse:
        """Create a new tag pointing to a table version."""
        response_dict = self._inner.create_table_tag(request.model_dump())
        return CreateTableTagResponse.from_dict(response_dict)

    def delete_table_tag(
        self, request: DeleteTableTagRequest
    ) -> DeleteTableTagResponse:
        """Delete a tag from a table."""
        response_dict = self._inner.delete_table_tag(request.model_dump())
        return DeleteTableTagResponse.from_dict(response_dict)

    def update_table_tag(
        self, request: UpdateTableTagRequest
    ) -> UpdateTableTagResponse:
        """Update a tag to point to a different version."""
        response_dict = self._inner.update_table_tag(request.model_dump())
        return UpdateTableTagResponse.from_dict(response_dict)

    def create_table_branch(
        self, request: CreateTableBranchRequest
    ) -> CreateTableBranchResponse:
        """Create a new branch forked from a table version."""
        response_dict = self._inner.create_table_branch(request.model_dump())
        return CreateTableBranchResponse.from_dict(response_dict)

    def list_table_branches(
        self, request: ListTableBranchesRequest
    ) -> ListTableBranchesResponse:
        """List all branches of a table."""
        response_dict = self._inner.list_table_branches(request.model_dump())
        return ListTableBranchesResponse.from_dict(response_dict)

    def delete_table_branch(
        self, request: DeleteTableBranchRequest
    ) -> DeleteTableBranchResponse:
        """Delete a branch from a table."""
        response_dict = self._inner.delete_table_branch(request.model_dump())
        return DeleteTableBranchResponse.from_dict(response_dict)

    # Operation metrics methods

    def retrieve_ops_metrics(self) -> Dict[str, int]:
        """Retrieve operation metrics as a dictionary.

        Returns a dict where keys are operation names (e.g., "list_tables",
        "describe_table") and values are the number of times each operation
        was called.

        Returns an empty dict if `ops_metrics_enabled` was false when creating
        the namespace.

        Returns
        -------
        Dict[str, int]
            Operation name to call count mapping
        """
        return self._inner.retrieve_ops_metrics()

    def reset_ops_metrics(self) -> None:
        """Reset all operation metrics counters to zero.

        Does nothing if `ops_metrics_enabled` was false when creating the
        namespace.
        """
        self._inner.reset_ops_metrics()


class RestNamespace(LanceNamespace):
    """REST-based Lance Namespace implementation backed by Rust.

    This namespace communicates with a Lance REST API server to manage
    namespaces and tables remotely.

    This is a Python wrapper around the Rust PyRestNamespace implementation,
    providing compatibility with the LanceNamespace ABC interface.

    Parameters
    ----------
    **properties : dict
        Configuration properties as key-value pairs:
        - uri (required): REST endpoint URI (e.g., "http://localhost:4099")
        - delimiter (optional): Namespace delimiter, default "$"
        - header.* (optional): HTTP headers with "header." prefix
          (e.g., header.Authorization="Bearer token" becomes
          Authorization="Bearer token" in HTTP headers)
        - tls.* (optional): TLS configuration with "tls." prefix

    Examples
    --------
    >>> import lance.namespace
    >>> # Create with properties dict
    >>> ns = lance.namespace.RestNamespace(uri="http://localhost:4099")
    >>>
    >>> # Using the connect() factory function from lance_namespace
    >>> import lance_namespace
    >>> ns = lance_namespace.connect("rest", {"uri": "http://localhost:4099"})

    With dynamic context provider:

    >>> class AuthProvider(DynamicContextProvider):
    ...     def __init__(self, api_key: str):
    ...         self.api_key = api_key
    ...     def provide_context(self, info: dict) -> dict:
    ...         return {"headers.Authorization": f"Bearer {self.api_key}"}
    ...
    >>> provider = AuthProvider(api_key="my-secret-key")
    >>> ns = lance.namespace.RestNamespace(
    ...     uri="http://localhost:4099",
    ...     context_provider=provider,
    ... )
    >>> ns.namespace_id()  # verify it works
    'RestNamespace { endpoint: "http://localhost:4099", delimiter: "$" }'
    """

    def __init__(self, context_provider=None, **properties):
        if PyRestNamespace is None:
            raise RuntimeError(
                "RestNamespace is not available. "
                "Lance was built without REST support. "
                "Please rebuild with the 'rest' feature enabled."
            )
        # Convert all values to strings as expected by Rust from_properties
        str_properties = {str(k): str(v) for k, v in properties.items()}

        # Create context provider from properties if configured
        if context_provider is None:
            context_provider = _create_context_provider_from_properties(str_properties)

        # Filter out dynamic_context_provider.* properties before passing to Rust
        filtered_properties = _filter_context_provider_properties(str_properties)

        # Create the underlying Rust namespace
        self._inner = PyRestNamespace(
            context_provider=context_provider, **filtered_properties
        )

    def namespace_id(self) -> str:
        """Return a human-readable unique identifier for this namespace instance."""
        return self._inner.namespace_id()

    def __repr__(self) -> str:
        return f"RestNamespace({self._inner.namespace_id()})"

    # Namespace operations

    def create_namespace(
        self, request: CreateNamespaceRequest
    ) -> CreateNamespaceResponse:
        response_dict = self._inner.create_namespace(request.model_dump())
        return CreateNamespaceResponse.from_dict(response_dict)

    def list_namespaces(self, request: ListNamespacesRequest) -> ListNamespacesResponse:
        response_dict = self._inner.list_namespaces(request.model_dump())
        return ListNamespacesResponse.from_dict(response_dict)

    def describe_namespace(
        self, request: DescribeNamespaceRequest
    ) -> DescribeNamespaceResponse:
        response_dict = self._inner.describe_namespace(request.model_dump())
        return DescribeNamespaceResponse.from_dict(response_dict)

    def drop_namespace(self, request: DropNamespaceRequest) -> DropNamespaceResponse:
        response_dict = self._inner.drop_namespace(request.model_dump())
        return DropNamespaceResponse.from_dict(response_dict)

    def namespace_exists(
        self, request: NamespaceExistsRequest
    ) -> NamespaceExistsResponse:
        self._inner.namespace_exists(request.model_dump())
        return NamespaceExistsResponse()

    # Table operations

    def list_tables(self, request: ListTablesRequest) -> ListTablesResponse:
        response_dict = self._inner.list_tables(request.model_dump())
        return ListTablesResponse.from_dict(response_dict)

    def describe_table(self, request: DescribeTableRequest) -> DescribeTableResponse:
        response_dict = self._inner.describe_table(request.model_dump())
        return DescribeTableResponse.from_dict(response_dict)

    def register_table(self, request: RegisterTableRequest) -> RegisterTableResponse:
        response_dict = self._inner.register_table(request.model_dump())
        return RegisterTableResponse.from_dict(response_dict)

    def table_exists(self, request: TableExistsRequest) -> TableExistsResponse:
        self._inner.table_exists(request.model_dump())
        return TableExistsResponse()

    def drop_table(self, request: DropTableRequest) -> DropTableResponse:
        response_dict = self._inner.drop_table(request.model_dump())
        return DropTableResponse.from_dict(response_dict)

    def deregister_table(
        self, request: DeregisterTableRequest
    ) -> DeregisterTableResponse:
        response_dict = self._inner.deregister_table(request.model_dump())
        return DeregisterTableResponse.from_dict(response_dict)

    def create_table(
        self, request: CreateTableRequest, request_data: bytes
    ) -> CreateTableResponse:
        response_dict = self._inner.create_table(request.model_dump(), request_data)
        return CreateTableResponse.from_dict(response_dict)

    def declare_table(self, request: DeclareTableRequest) -> DeclareTableResponse:
        response_dict = self._inner.declare_table(request.model_dump())
        return DeclareTableResponse.from_dict(response_dict)

    def rename_table(self, request: RenameTableRequest) -> RenameTableResponse:
        response_dict = self._inner.rename_table(request.model_dump())
        return RenameTableResponse.from_dict(response_dict)

    # Table version operations

    def list_table_versions(
        self, request: ListTableVersionsRequest
    ) -> ListTableVersionsResponse:
        response_dict = self._inner.list_table_versions(request.model_dump())
        return ListTableVersionsResponse.from_dict(response_dict)

    def create_table_version(self, request: dict) -> dict:
        """Create a table version (for external manifest store integration).

        Parameters
        ----------
        request : dict
            Request dictionary with keys:
            - id: List[str] - Table identifier
            - version: int - Version number to create
            - manifest_path: str - Path to staging manifest
            - manifest_size: int (optional) - Size in bytes
            - e_tag: str (optional) - ETag for optimistic concurrency

        Returns
        -------
        dict
            Response dictionary with optional transaction_id
        """
        return self._inner.create_table_version(request)

    def describe_table_version(self, request: dict) -> dict:
        """Describe a specific table version.

        Parameters
        ----------
        request : dict
            Request dictionary with keys:
            - id: List[str] - Table identifier
            - version: int (optional) - Version to describe (None = latest)

        Returns
        -------
        dict
            Response dictionary with version info:
            - version: dict with version, manifest_path, manifest_size, e_tag, timestamp
        """
        return self._inner.describe_table_version(request)

    def batch_delete_table_versions(self, request: dict) -> dict:
        """Delete multiple table versions in a single request.

        Parameters
        ----------
        request : dict
            Request dictionary with keys:
            - id: List[str] - Table identifier
            - versions: List[int] - List of version numbers to delete

        Returns
        -------
        dict
            Response dictionary with:
            - deleted_versions: List[int] - List of successfully deleted versions
        """
        return self._inner.batch_delete_table_versions(request)

    # Data manipulation operations

    def count_table_rows(
        self, request: CountTableRowsRequest
    ) -> CountTableRowsResponse:
        """Count the number of rows in a table, optionally filtered by a predicate.

        Parameters
        ----------
        request : CountTableRowsRequest
            Request with table id, optional version, and optional predicate filter

        Returns
        -------
        CountTableRowsResponse
            Response whose ``count`` is the number of rows matching the criteria
        """
        count = self._inner.count_table_rows(request.model_dump())
        return CountTableRowsResponse(count=count)

    def insert_into_table(
        self, request: InsertIntoTableRequest, request_data: bytes
    ) -> InsertIntoTableResponse:
        """Insert data into a table.

        Parameters
        ----------
        request : InsertIntoTableRequest
            Request with table id and write mode (append/overwrite)
        request_data : bytes
            Arrow IPC stream containing the data to insert

        Returns
        -------
        InsertIntoTableResponse
            Response with optional transaction_id
        """
        response_dict = self._inner.insert_into_table(
            request.model_dump(), request_data
        )
        return InsertIntoTableResponse.from_dict(response_dict)

    def merge_insert_into_table(
        self, request: MergeInsertIntoTableRequest, request_data: bytes
    ) -> MergeInsertIntoTableResponse:
        """Merge-insert data into a table (upsert operation).

        Parameters
        ----------
        request : MergeInsertIntoTableRequest
            Request with table id, merge keys, and merge behavior
        request_data : bytes
            Arrow IPC stream containing the data to merge

        Returns
        -------
        MergeInsertIntoTableResponse
            Response with optional transaction_id
        """
        response_dict = self._inner.merge_insert_into_table(
            request.model_dump(), request_data
        )
        return MergeInsertIntoTableResponse.from_dict(response_dict)

    def update_table(self, request: UpdateTableRequest) -> UpdateTableResponse:
        """Update rows in a table matching a filter.

        Parameters
        ----------
        request : UpdateTableRequest
            Request with table id, filter predicate, and column updates

        Returns
        -------
        UpdateTableResponse
            Response with optional transaction_id
        """
        response_dict = self._inner.update_table(request.model_dump())
        return UpdateTableResponse.from_dict(response_dict)

    def delete_from_table(
        self, request: DeleteFromTableRequest
    ) -> DeleteFromTableResponse:
        """Delete rows from a table matching a filter.

        Parameters
        ----------
        request : DeleteFromTableRequest
            Request with table id and filter predicate

        Returns
        -------
        DeleteFromTableResponse
            Response confirming the deletion
        """
        response_dict = self._inner.delete_from_table(request.model_dump())
        return DeleteFromTableResponse.from_dict(response_dict)

    def query_table(self, request) -> QueryTableResponse:
        """Query a table and return results as Arrow IPC.

        Parameters
        ----------
        request : QueryTableRequest or dict
            Query request with table id, optional filter, columns, vector search
            parameters, limit/offset, etc.

        Returns
        -------
        QueryTableResponse
            Response whose ``data`` is the query results in Arrow IPC file format
        """
        if hasattr(request, "model_dump"):
            request = request.model_dump()
        data = self._inner.query_table(request)
        return QueryTableResponse(data=data)

    # Index operations

    def create_table_index(
        self, request: CreateTableIndexRequest
    ) -> CreateTableIndexResponse:
        """Create an index on a table.

        Parameters
        ----------
        request : CreateTableIndexRequest
            Request with table id, index name, column, and index configuration

        Returns
        -------
        CreateTableIndexResponse
            Response with optional transaction_id
        """
        response_dict = self._inner.create_table_index(request.model_dump())
        return CreateTableIndexResponse.from_dict(response_dict)

    def list_table_indices(
        self, request: ListTableIndicesRequest
    ) -> ListTableIndicesResponse:
        """List all indices on a table.

        Parameters
        ----------
        request : ListTableIndicesRequest
            Request with table id

        Returns
        -------
        ListTableIndicesResponse
            Response with list of index metadata
        """
        response_dict = self._inner.list_table_indices(request.model_dump())
        return ListTableIndicesResponse.from_dict(response_dict)

    def describe_table_index_stats(
        self, request: DescribeTableIndexStatsRequest
    ) -> DescribeTableIndexStatsResponse:
        """Get statistics for a specific index.

        Parameters
        ----------
        request : DescribeTableIndexStatsRequest
            Request with table id and index name

        Returns
        -------
        DescribeTableIndexStatsResponse
            Response with index statistics
        """
        response_dict = self._inner.describe_table_index_stats(request.model_dump())
        return DescribeTableIndexStatsResponse.from_dict(response_dict)

    # Transaction operations

    def describe_transaction(
        self, request: DescribeTransactionRequest
    ) -> DescribeTransactionResponse:
        """Describe a transaction.

        Parameters
        ----------
        request : DescribeTransactionRequest
            Request with transaction id

        Returns
        -------
        DescribeTransactionResponse
            Response with transaction details
        """
        response_dict = self._inner.describe_transaction(request.model_dump())
        return DescribeTransactionResponse.from_dict(response_dict)

    def alter_transaction(
        self, request: AlterTransactionRequest
    ) -> AlterTransactionResponse:
        """Alter a transaction (commit or abort).

        Parameters
        ----------
        request : AlterTransactionRequest
            Request with transaction id and action

        Returns
        -------
        AlterTransactionResponse
            Response confirming the action
        """
        response_dict = self._inner.alter_transaction(request.model_dump())
        return AlterTransactionResponse.from_dict(response_dict)

    # Additional index operations

    def create_table_scalar_index(
        self, request: CreateTableIndexRequest
    ) -> CreateTableIndexResponse:
        """Create a scalar index on a table column."""
        response_dict = self._inner.create_table_scalar_index(request.model_dump())
        return CreateTableIndexResponse.from_dict(response_dict)

    def drop_table_index(
        self, request: DropTableIndexRequest
    ) -> DropTableIndexResponse:
        """Drop an index from a table."""
        response_dict = self._inner.drop_table_index(request.model_dump())
        return DropTableIndexResponse.from_dict(response_dict)

    # Additional table operations

    def list_all_tables(self, request: ListTablesRequest) -> ListTablesResponse:
        """List all tables recursively across all namespaces."""
        response_dict = self._inner.list_all_tables(request.model_dump())
        return ListTablesResponse.from_dict(response_dict)

    def restore_table(self, request: RestoreTableRequest) -> RestoreTableResponse:
        """Restore a previously dropped table."""
        response_dict = self._inner.restore_table(request.model_dump())
        return RestoreTableResponse.from_dict(response_dict)

    def update_table_schema_metadata(
        self, request: UpdateTableSchemaMetadataRequest
    ) -> UpdateTableSchemaMetadataResponse:
        """Update the schema metadata of a table."""
        response_dict = self._inner.update_table_schema_metadata(request.model_dump())
        return UpdateTableSchemaMetadataResponse.from_dict(response_dict)

    def get_table_stats(self, request: GetTableStatsRequest) -> GetTableStatsResponse:
        """Get statistics for a table."""
        response_dict = self._inner.get_table_stats(request.model_dump())
        return GetTableStatsResponse.from_dict(response_dict)

    # Query plan operations

    def explain_table_query_plan(self, request: ExplainTableQueryPlanRequest) -> str:
        """Explain the query plan for a table query."""
        return self._inner.explain_table_query_plan(request.model_dump())

    def analyze_table_query_plan(self, request: AnalyzeTableQueryPlanRequest) -> str:
        """Analyze the query plan for a table query with execution statistics."""
        return self._inner.analyze_table_query_plan(request.model_dump())

    # Column alteration operations

    def alter_table_add_columns(
        self, request: AlterTableAddColumnsRequest
    ) -> AlterTableAddColumnsResponse:
        """Add columns to a table."""
        response_dict = self._inner.alter_table_add_columns(request.model_dump())
        return AlterTableAddColumnsResponse.from_dict(response_dict)

    def alter_table_alter_columns(
        self, request: AlterTableAlterColumnsRequest
    ) -> AlterTableAlterColumnsResponse:
        """Alter existing columns in a table."""
        response_dict = self._inner.alter_table_alter_columns(request.model_dump())
        return AlterTableAlterColumnsResponse.from_dict(response_dict)

    def alter_table_drop_columns(
        self, request: AlterTableDropColumnsRequest
    ) -> AlterTableDropColumnsResponse:
        """Drop columns from a table."""
        response_dict = self._inner.alter_table_drop_columns(request.model_dump())
        return AlterTableDropColumnsResponse.from_dict(response_dict)

    def alter_table_backfill_columns(
        self, request: AlterTableBackfillColumnsRequest
    ) -> AlterTableBackfillColumnsResponse:
        """Trigger an async backfill job for a computed column."""
        response_dict = self._inner.alter_table_backfill_columns(request.model_dump())
        return AlterTableBackfillColumnsResponse.from_dict(response_dict)

    def refresh_materialized_view(
        self, request: RefreshMaterializedViewRequest
    ) -> RefreshMaterializedViewResponse:
        """Trigger an async materialized view refresh."""
        response_dict = self._inner.refresh_materialized_view(request.model_dump())
        return RefreshMaterializedViewResponse.from_dict(response_dict)

    def create_materialized_view(
        self, request: CreateMaterializedViewRequest
    ) -> CreateMaterializedViewResponse:
        """Create a materialized view backed by an optional UDTF/chunker."""
        response_dict = self._inner.create_materialized_view(request.model_dump())
        return CreateMaterializedViewResponse.from_dict(response_dict)

    # Table tag operations

    def list_table_tags(self, request: ListTableTagsRequest) -> ListTableTagsResponse:
        """List all tags on a table."""
        response_dict = self._inner.list_table_tags(request.model_dump())
        return ListTableTagsResponse.from_dict(response_dict)

    def get_table_tag_version(
        self, request: GetTableTagVersionRequest
    ) -> GetTableTagVersionResponse:
        """Get the version associated with a table tag."""
        response_dict = self._inner.get_table_tag_version(request.model_dump())
        return GetTableTagVersionResponse.from_dict(response_dict)

    def create_table_tag(
        self, request: CreateTableTagRequest
    ) -> CreateTableTagResponse:
        """Create a new tag pointing to a table version."""
        response_dict = self._inner.create_table_tag(request.model_dump())
        return CreateTableTagResponse.from_dict(response_dict)

    def delete_table_tag(
        self, request: DeleteTableTagRequest
    ) -> DeleteTableTagResponse:
        """Delete a tag from a table."""
        response_dict = self._inner.delete_table_tag(request.model_dump())
        return DeleteTableTagResponse.from_dict(response_dict)

    def update_table_tag(
        self, request: UpdateTableTagRequest
    ) -> UpdateTableTagResponse:
        """Update a tag to point to a different version."""
        response_dict = self._inner.update_table_tag(request.model_dump())
        return UpdateTableTagResponse.from_dict(response_dict)

    def create_table_branch(
        self, request: CreateTableBranchRequest
    ) -> CreateTableBranchResponse:
        """Create a new branch forked from a table version."""
        response_dict = self._inner.create_table_branch(request.model_dump())
        return CreateTableBranchResponse.from_dict(response_dict)

    def list_table_branches(
        self, request: ListTableBranchesRequest
    ) -> ListTableBranchesResponse:
        """List all branches of a table."""
        response_dict = self._inner.list_table_branches(request.model_dump())
        return ListTableBranchesResponse.from_dict(response_dict)

    def delete_table_branch(
        self, request: DeleteTableBranchRequest
    ) -> DeleteTableBranchResponse:
        """Delete a branch from a table."""
        response_dict = self._inner.delete_table_branch(request.model_dump())
        return DeleteTableBranchResponse.from_dict(response_dict)

    # Operation metrics methods

    def retrieve_ops_metrics(self) -> Dict[str, int]:
        """Retrieve operation metrics as a dictionary.

        Returns a dict where keys are operation names (e.g., "list_tables",
        "describe_table") and values are the number of times each operation
        was called.

        Returns an empty dict if `ops_metrics_enabled` was false when creating
        the namespace.

        Returns
        -------
        Dict[str, int]
            Operation name to call count mapping
        """
        return self._inner.retrieve_ops_metrics()

    def reset_ops_metrics(self) -> None:
        """Reset all operation metrics counters to zero.

        Does nothing if `ops_metrics_enabled` was false when creating the
        namespace.
        """
        self._inner.reset_ops_metrics()


class RestAdapter:
    """REST adapter server that creates a namespace backend and exposes it via REST.

    This adapter starts an HTTP server that exposes a Lance namespace backend
    via the Lance REST API. The backend namespace can be any implementation
    (DirectoryNamespace, etc.) created from the provided configuration.
    Useful for testing RestNamespace clients.

    Parameters
    ----------
    namespace_client_impl : str
        Namespace client implementation type ("dir", "rest", etc.)
    namespace_client_properties : dict, optional
        Configuration properties for the backend namespace client.
        For DirectoryNamespace ("dir"):

        - root (required): Root directory path or URI
        - manifest_enabled (optional): Enable manifest tracking (default: "true")
        - dir_listing_enabled (optional): Enable directory listing fallback
        - storage.* (optional): Storage options with "storage." prefix
    session : Session, optional
        Lance session for sharing object store connections with the backend namespace.
    host : str, optional
        Host address to bind to. Default "127.0.0.1".
    port : int, optional
        Port to listen on. Default 2333 per REST spec.
        Use 0 to let the OS assign an available ephemeral port.
        Use the `port` property after `start()` to get the actual port.

    Examples
    --------
    >>> import lance.namespace
    >>>
    >>> # Start REST adapter with DirectoryNamespace backend (auto port)
    >>> namespace_config = {"root": "memory://test"}
    >>> with lance.namespace.RestAdapter("dir", namespace_config) as adapter:
    ...     # Create REST client using the assigned port
    ...     client = lance.namespace.RestNamespace(uri=f"http://127.0.0.1:{adapter.port}")
    ...     # Use the client...
    """

    def __init__(
        self,
        namespace_client_impl: str,
        namespace_client_properties: Dict[str, str] = None,
        session=None,
        host: str = None,
        port: int = None,
    ):
        if PyRestAdapter is None:
            raise RuntimeError(
                "RestAdapter is not available. "
                "Lance was built without REST adapter support. "
                "Please rebuild with the 'rest-adapter' feature enabled."
            )

        # Convert to string properties
        if namespace_client_properties is None:
            namespace_client_properties = {}
        str_properties = {
            str(k): str(v) for k, v in namespace_client_properties.items()
        }

        # Create the underlying Rust adapter
        self._inner = PyRestAdapter(
            namespace_client_impl, str_properties, session, host, port
        )
        self.host = host
        self.namespace_client_impl = namespace_client_impl

    @property
    def port(self) -> int:
        """Get the actual port the server is listening on.

        Returns 0 if the server hasn't been started yet.
        """
        return self._inner.port

    def start(self):
        """Start the REST server in the background."""
        self._inner.start()

    def stop(self):
        """Stop the REST server."""
        self._inner.stop()

    def __enter__(self):
        """Start server when entering context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop server when exiting context."""
        self.stop()
        return False

    def __repr__(self) -> str:
        return f"RestAdapter(host='{self.host}', port={self.port})"
