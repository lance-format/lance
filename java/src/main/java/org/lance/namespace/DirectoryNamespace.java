/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.namespace;

import org.lance.JniLoader;
import org.lance.namespace.model.AlterTableAddColumnsRequest;
import org.lance.namespace.model.AlterTableAddColumnsResponse;
import org.lance.namespace.model.AlterTableAlterColumnsRequest;
import org.lance.namespace.model.AlterTableAlterColumnsResponse;
import org.lance.namespace.model.AlterTableDropColumnsRequest;
import org.lance.namespace.model.AlterTableDropColumnsResponse;
import org.lance.namespace.model.AlterTransactionRequest;
import org.lance.namespace.model.AlterTransactionResponse;
import org.lance.namespace.model.AnalyzeTableQueryPlanRequest;
import org.lance.namespace.model.BatchDeleteTableVersionsRequest;
import org.lance.namespace.model.BatchDeleteTableVersionsResponse;
import org.lance.namespace.model.CountTableRowsRequest;
import org.lance.namespace.model.CountTableRowsResponse;
import org.lance.namespace.model.CreateMaterializedViewRequest;
import org.lance.namespace.model.CreateMaterializedViewResponse;
import org.lance.namespace.model.CreateNamespaceRequest;
import org.lance.namespace.model.CreateNamespaceResponse;
import org.lance.namespace.model.CreateTableIndexRequest;
import org.lance.namespace.model.CreateTableIndexResponse;
import org.lance.namespace.model.CreateTableRequest;
import org.lance.namespace.model.CreateTableResponse;
import org.lance.namespace.model.CreateTableScalarIndexResponse;
import org.lance.namespace.model.CreateTableTagRequest;
import org.lance.namespace.model.CreateTableTagResponse;
import org.lance.namespace.model.CreateTableVersionRequest;
import org.lance.namespace.model.CreateTableVersionResponse;
import org.lance.namespace.model.DeclareTableRequest;
import org.lance.namespace.model.DeclareTableResponse;
import org.lance.namespace.model.DeleteFromTableRequest;
import org.lance.namespace.model.DeleteFromTableResponse;
import org.lance.namespace.model.DeleteTableTagRequest;
import org.lance.namespace.model.DeleteTableTagResponse;
import org.lance.namespace.model.DeregisterTableRequest;
import org.lance.namespace.model.DeregisterTableResponse;
import org.lance.namespace.model.DescribeNamespaceRequest;
import org.lance.namespace.model.DescribeNamespaceResponse;
import org.lance.namespace.model.DescribeTableIndexStatsRequest;
import org.lance.namespace.model.DescribeTableIndexStatsResponse;
import org.lance.namespace.model.DescribeTableRequest;
import org.lance.namespace.model.DescribeTableResponse;
import org.lance.namespace.model.DescribeTableVersionRequest;
import org.lance.namespace.model.DescribeTableVersionResponse;
import org.lance.namespace.model.DescribeTransactionRequest;
import org.lance.namespace.model.DescribeTransactionResponse;
import org.lance.namespace.model.DropNamespaceRequest;
import org.lance.namespace.model.DropNamespaceResponse;
import org.lance.namespace.model.DropTableIndexRequest;
import org.lance.namespace.model.DropTableIndexResponse;
import org.lance.namespace.model.DropTableRequest;
import org.lance.namespace.model.DropTableResponse;
import org.lance.namespace.model.ExplainTableQueryPlanRequest;
import org.lance.namespace.model.GetTableStatsRequest;
import org.lance.namespace.model.GetTableStatsResponse;
import org.lance.namespace.model.GetTableTagVersionRequest;
import org.lance.namespace.model.GetTableTagVersionResponse;
import org.lance.namespace.model.InsertIntoTableRequest;
import org.lance.namespace.model.InsertIntoTableResponse;
import org.lance.namespace.model.ListNamespacesRequest;
import org.lance.namespace.model.ListNamespacesResponse;
import org.lance.namespace.model.ListTableIndicesRequest;
import org.lance.namespace.model.ListTableIndicesResponse;
import org.lance.namespace.model.ListTableTagsRequest;
import org.lance.namespace.model.ListTableTagsResponse;
import org.lance.namespace.model.ListTableVersionsRequest;
import org.lance.namespace.model.ListTableVersionsResponse;
import org.lance.namespace.model.ListTablesRequest;
import org.lance.namespace.model.ListTablesResponse;
import org.lance.namespace.model.MergeInsertIntoTableRequest;
import org.lance.namespace.model.MergeInsertIntoTableResponse;
import org.lance.namespace.model.NamespaceExistsRequest;
import org.lance.namespace.model.NamespaceExistsResponse;
import org.lance.namespace.model.QueryTableRequest;
import org.lance.namespace.model.QueryTableResponse;
import org.lance.namespace.model.RegisterTableRequest;
import org.lance.namespace.model.RegisterTableResponse;
import org.lance.namespace.model.RenameTableRequest;
import org.lance.namespace.model.RenameTableResponse;
import org.lance.namespace.model.RestoreTableRequest;
import org.lance.namespace.model.RestoreTableResponse;
import org.lance.namespace.model.TableExistsRequest;
import org.lance.namespace.model.TableExistsResponse;
import org.lance.namespace.model.UpdateTableRequest;
import org.lance.namespace.model.UpdateTableResponse;
import org.lance.namespace.model.UpdateTableSchemaMetadataRequest;
import org.lance.namespace.model.UpdateTableSchemaMetadataResponse;
import org.lance.namespace.model.UpdateTableTagRequest;
import org.lance.namespace.model.UpdateTableTagResponse;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.apache.arrow.memory.BufferAllocator;

import java.io.Closeable;
import java.lang.reflect.Constructor;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * DirectoryNamespace implementation that provides Lance namespace functionality for directory-based
 * storage.
 *
 * <p>Supported storage backends:
 *
 * <ul>
 *   <li>Local filesystem
 *   <li>AWS S3 (s3://bucket/path)
 *   <li>Azure Blob Storage (az://container/path)
 *   <li>Google Cloud Storage (gs://bucket/path)
 * </ul>
 *
 * <p>This class wraps the native Rust implementation and provides a Java interface that implements
 * the LanceNamespace interface from lance-namespace-core.
 *
 * <p>Configuration properties:
 *
 * <ul>
 *   <li>root (required): Root directory path or URI (e.g., /path/to/dir, s3://bucket/path,
 *       az://container/path, gs://bucket/path)
 *   <li>manifest_enabled (optional): "true" or "false" (default: true)
 *   <li>dir_listing_enabled (optional): "true" or "false" (default: true)
 *   <li>inline_optimization_enabled (optional): "true" or "false" (default: true)
 *   <li>storage.* (optional): Storage options for cloud providers (e.g., storage.region=us-east-1
 *       for S3, storage.account_name=myaccount for Azure)
 * </ul>
 *
 * <p>Credential vending properties (requires credential-vendor-* features to be enabled):
 *
 * <p>When credential vendor properties are configured, describeTable() will return vended temporary
 * credentials. The vendor type is auto-selected based on the table location URI: s3:// for AWS,
 * gs:// for GCP, az:// for Azure.
 *
 * <ul>
 *   <li>Common properties:
 *       <ul>
 *         <li>credential_vendor.enabled (required): Set to "true" to enable credential vending
 *         <li>credential_vendor.permission (optional): read, write, or admin (default: read)
 *       </ul>
 *   <li>AWS-specific properties (for s3:// locations):
 *       <ul>
 *         <li>credential_vendor.aws_role_arn (required): IAM role ARN to assume
 *         <li>credential_vendor.aws_external_id (optional): External ID for assume role
 *         <li>credential_vendor.aws_region (optional): AWS region
 *         <li>credential_vendor.aws_role_session_name (optional): Role session name
 *         <li>credential_vendor.aws_duration_millis (optional): Duration in ms (default: 3600000,
 *             range: 15min-12hrs)
 *       </ul>
 *   <li>GCP-specific properties (for gs:// locations):
 *       <ul>
 *         <li>credential_vendor.gcp_service_account (optional): Service account to impersonate
 *         <li>Note: GCP uses Application Default Credentials (ADC). To use a service account key
 *             file, set the GOOGLE_APPLICATION_CREDENTIALS environment variable before starting.
 *         <li>Note: GCP token duration cannot be configured; it's determined by the STS endpoint
 *       </ul>
 *   <li>Azure-specific properties (for az:// locations):
 *       <ul>
 *         <li>credential_vendor.azure_account_name (required): Azure storage account name
 *         <li>credential_vendor.azure_tenant_id (optional): Azure tenant ID
 *         <li>credential_vendor.azure_duration_millis (optional): Duration in ms (default: 3600000,
 *             up to 7 days)
 *       </ul>
 *   <li>Testing properties:
 *       <ul>
 *         <li>ops_metrics_enabled (optional): "true" to enable operation metrics tracking. Use
 *             {@link #retrieveOpsMetrics()} to get call counts.
 *         <li>vend_input_storage_options (optional): "true" to return input storage options in
 *             describeTable() when no credential vendor is configured. Useful for testing.
 *         <li>vend_input_storage_options_refresh_interval_millis (optional): When set with
 *             vend_input_storage_options, adds expires_at_millis to storage options. Value is
 *             current_time_millis + this interval.
 *       </ul>
 * </ul>
 *
 * <p>Example usage (local filesystem):
 *
 * <pre>{@code
 * Map<String, String> properties = new HashMap<>();
 * properties.put("root", "/tmp/lance-data");
 * properties.put("manifest_enabled", "true");
 *
 * DirectoryNamespace namespace = new DirectoryNamespace();
 * namespace.initialize(properties, allocator);
 *
 * // Use namespace...
 * ListTablesResponse tables = namespace.listTables(request);
 *
 * // Clean up
 * namespace.close();
 * }</pre>
 *
 * <p>Example usage (AWS S3):
 *
 * <pre>{@code
 * Map<String, String> properties = new HashMap<>();
 * properties.put("root", "s3://my-bucket/lance-data");
 * properties.put("storage.region", "us-east-1");
 * // AWS credentials can be provided via environment variables or IAM roles
 *
 * DirectoryNamespace namespace = new DirectoryNamespace();
 * namespace.initialize(properties, allocator);
 * // Use namespace...
 * namespace.close();
 * }</pre>
 *
 * <p>Example usage (AWS S3 with credential vending):
 *
 * <pre>{@code
 * Map<String, String> properties = new HashMap<>();
 * properties.put("root", "s3://my-bucket/lance-data");
 * properties.put("credential_vendor.enabled", "true");
 * properties.put("credential_vendor.aws_role_arn", "arn:aws:iam::123456789012:role/MyRole");
 * properties.put("credential_vendor.aws_duration_millis", "3600000");  // 1 hour
 *
 * DirectoryNamespace namespace = new DirectoryNamespace();
 * namespace.initialize(properties, allocator);
 * // describeTable() will now return vended credentials (AWS vendor auto-selected from s3:// URI)
 * namespace.close();
 * }</pre>
 */
public class DirectoryNamespace implements LanceNamespace, Closeable {
  static {
    JniLoader.ensureLoaded();
  }

  private static final ObjectMapper OBJECT_MAPPER = createObjectMapper();

  private static ObjectMapper createObjectMapper() {
    ObjectMapper mapper = new ObjectMapper();
    mapper.registerModule(new JavaTimeModule());
    mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    return mapper;
  }

  private long nativeDirectoryNamespaceHandle;
  private BufferAllocator allocator;

  /** Creates a new DirectoryNamespace. Must call initialize() before use. */
  public DirectoryNamespace() {}

  @Override
  public void initialize(Map<String, String> configProperties, BufferAllocator allocator) {
    initialize(configProperties, allocator, null);
  }

  /**
   * Initialize with a dynamic context provider.
   *
   * <p>If contextProvider is null and the properties contain {@code dynamic_context_provider.impl},
   * the provider will be loaded from the class path. The class must implement {@link
   * DynamicContextProvider} and have a constructor accepting {@code Map<String, String>}.
   *
   * @param configProperties Configuration properties for the namespace
   * @param allocator Arrow buffer allocator
   * @param contextProvider Optional provider for per-request context (e.g., dynamic auth headers)
   */
  public void initialize(
      Map<String, String> configProperties,
      BufferAllocator allocator,
      DynamicContextProvider contextProvider) {
    if (this.nativeDirectoryNamespaceHandle != 0) {
      throw new IllegalStateException("DirectoryNamespace already initialized");
    }
    this.allocator = allocator;

    // If no explicit provider, try to create from properties
    DynamicContextProvider provider = contextProvider;
    if (provider == null) {
      provider = createProviderFromProperties(configProperties).orElse(null);
    }

    // Filter out provider properties before passing to native layer
    Map<String, String> filteredProperties = filterProviderProperties(configProperties);

    if (provider != null) {
      this.nativeDirectoryNamespaceHandle = createNativeWithProvider(filteredProperties, provider);
    } else {
      this.nativeDirectoryNamespaceHandle = createNative(filteredProperties);
    }
  }

  @Override
  public String namespaceId() {
    ensureInitialized();
    return namespaceIdNative(nativeDirectoryNamespaceHandle);
  }

  @Override
  public ListNamespacesResponse listNamespaces(ListNamespacesRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = listNamespacesNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, ListNamespacesResponse.class);
  }

  @Override
  public DescribeNamespaceResponse describeNamespace(DescribeNamespaceRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = describeNamespaceNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeNamespaceResponse.class);
  }

  @Override
  public CreateNamespaceResponse createNamespace(CreateNamespaceRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createNamespaceNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, CreateNamespaceResponse.class);
  }

  @Override
  public DropNamespaceResponse dropNamespace(DropNamespaceRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = dropNamespaceNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DropNamespaceResponse.class);
  }

  @Override
  public NamespaceExistsResponse namespaceExists(NamespaceExistsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    namespaceExistsNative(nativeDirectoryNamespaceHandle, requestJson);
    return new NamespaceExistsResponse();
  }

  @Override
  public ListTablesResponse listTables(ListTablesRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = listTablesNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, ListTablesResponse.class);
  }

  @Override
  public DescribeTableResponse describeTable(DescribeTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = describeTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeTableResponse.class);
  }

  @Override
  public RegisterTableResponse registerTable(RegisterTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = registerTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, RegisterTableResponse.class);
  }

  @Override
  public TableExistsResponse tableExists(TableExistsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    tableExistsNative(nativeDirectoryNamespaceHandle, requestJson);
    return new TableExistsResponse();
  }

  @Override
  public DropTableResponse dropTable(DropTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = dropTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DropTableResponse.class);
  }

  @Override
  public DeregisterTableResponse deregisterTable(DeregisterTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = deregisterTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DeregisterTableResponse.class);
  }

  @Override
  public CountTableRowsResponse countTableRows(CountTableRowsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    Long count = countTableRowsNative(nativeDirectoryNamespaceHandle, requestJson);
    return new CountTableRowsResponse().count(count);
  }

  @Override
  public CreateTableResponse createTable(CreateTableRequest request, byte[] requestData) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson =
        createTableNative(nativeDirectoryNamespaceHandle, requestJson, requestData);
    return fromJson(responseJson, CreateTableResponse.class);
  }

  @Override
  public DeclareTableResponse declareTable(DeclareTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = declareTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DeclareTableResponse.class);
  }

  @Override
  public RenameTableResponse renameTable(RenameTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = renameTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, RenameTableResponse.class);
  }

  @Override
  public InsertIntoTableResponse insertIntoTable(
      InsertIntoTableRequest request, byte[] requestData) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson =
        insertIntoTableNative(nativeDirectoryNamespaceHandle, requestJson, requestData);
    return fromJson(responseJson, InsertIntoTableResponse.class);
  }

  @Override
  public MergeInsertIntoTableResponse mergeInsertIntoTable(
      MergeInsertIntoTableRequest request, byte[] requestData) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson =
        mergeInsertIntoTableNative(nativeDirectoryNamespaceHandle, requestJson, requestData);
    return fromJson(responseJson, MergeInsertIntoTableResponse.class);
  }

  @Override
  public UpdateTableResponse updateTable(UpdateTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = updateTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, UpdateTableResponse.class);
  }

  @Override
  public DeleteFromTableResponse deleteFromTable(DeleteFromTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = deleteFromTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DeleteFromTableResponse.class);
  }

  @Override
  public QueryTableResponse queryTable(QueryTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    byte[] data = queryTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return new QueryTableResponse().data(data);
  }

  @Override
  public CreateTableIndexResponse createTableIndex(CreateTableIndexRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createTableIndexNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, CreateTableIndexResponse.class);
  }

  @Override
  public ListTableIndicesResponse listTableIndices(ListTableIndicesRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = listTableIndicesNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, ListTableIndicesResponse.class);
  }

  @Override
  public DescribeTableIndexStatsResponse describeTableIndexStats(
      DescribeTableIndexStatsRequest request, String indexName) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson =
        describeTableIndexStatsNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeTableIndexStatsResponse.class);
  }

  @Override
  public DescribeTransactionResponse describeTransaction(DescribeTransactionRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = describeTransactionNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeTransactionResponse.class);
  }

  @Override
  public AlterTransactionResponse alterTransaction(AlterTransactionRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = alterTransactionNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, AlterTransactionResponse.class);
  }

  // Table version operations

  @Override
  public ListTableVersionsResponse listTableVersions(ListTableVersionsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = listTableVersionsNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, ListTableVersionsResponse.class);
  }

  @Override
  public CreateTableVersionResponse createTableVersion(CreateTableVersionRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createTableVersionNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, CreateTableVersionResponse.class);
  }

  @Override
  public DescribeTableVersionResponse describeTableVersion(DescribeTableVersionRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = describeTableVersionNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DescribeTableVersionResponse.class);
  }

  @Override
  public BatchDeleteTableVersionsResponse batchDeleteTableVersions(
      BatchDeleteTableVersionsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson =
        batchDeleteTableVersionsNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, BatchDeleteTableVersionsResponse.class);
  }

  @Override
  public CreateTableScalarIndexResponse createTableScalarIndex(CreateTableIndexRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createTableScalarIndexNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, CreateTableScalarIndexResponse.class);
  }

  @Override
  public DropTableIndexResponse dropTableIndex(DropTableIndexRequest request, String indexName) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = dropTableIndexNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DropTableIndexResponse.class);
  }

  @Override
  public ListTablesResponse listAllTables(ListTablesRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = listAllTablesNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, ListTablesResponse.class);
  }

  @Override
  public RestoreTableResponse restoreTable(RestoreTableRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = restoreTableNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, RestoreTableResponse.class);
  }

  @Override
  public UpdateTableSchemaMetadataResponse updateTableSchemaMetadata(
      UpdateTableSchemaMetadataRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson =
        updateTableSchemaMetadataNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, UpdateTableSchemaMetadataResponse.class);
  }

  @Override
  public GetTableStatsResponse getTableStats(GetTableStatsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = getTableStatsNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, GetTableStatsResponse.class);
  }

  @Override
  public String explainTableQueryPlan(ExplainTableQueryPlanRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    return explainTableQueryPlanNative(nativeDirectoryNamespaceHandle, requestJson);
  }

  @Override
  public String analyzeTableQueryPlan(AnalyzeTableQueryPlanRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    return analyzeTableQueryPlanNative(nativeDirectoryNamespaceHandle, requestJson);
  }

  @Override
  public AlterTableAddColumnsResponse alterTableAddColumns(AlterTableAddColumnsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = alterTableAddColumnsNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, AlterTableAddColumnsResponse.class);
  }

  @Override
  public AlterTableAlterColumnsResponse alterTableAlterColumns(
      AlterTableAlterColumnsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = alterTableAlterColumnsNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, AlterTableAlterColumnsResponse.class);
  }

  @Override
  public AlterTableDropColumnsResponse alterTableDropColumns(AlterTableDropColumnsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = alterTableDropColumnsNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, AlterTableDropColumnsResponse.class);
  }

  @Override
  public ListTableTagsResponse listTableTags(ListTableTagsRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = listTableTagsNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, ListTableTagsResponse.class);
  }

  @Override
  public GetTableTagVersionResponse getTableTagVersion(GetTableTagVersionRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = getTableTagVersionNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, GetTableTagVersionResponse.class);
  }

  @Override
  public CreateTableTagResponse createTableTag(CreateTableTagRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createTableTagNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, CreateTableTagResponse.class);
  }

  @Override
  public DeleteTableTagResponse deleteTableTag(DeleteTableTagRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = deleteTableTagNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, DeleteTableTagResponse.class);
  }

  @Override
  public UpdateTableTagResponse updateTableTag(UpdateTableTagRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = updateTableTagNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, UpdateTableTagResponse.class);
  }

  @Override
  public CreateMaterializedViewResponse createMaterializedView(
      CreateMaterializedViewRequest request) {
    ensureInitialized();
    String requestJson = toJson(request);
    String responseJson = createMaterializedViewNative(nativeDirectoryNamespaceHandle, requestJson);
    return fromJson(responseJson, CreateMaterializedViewResponse.class);
  }

  @Override
  public void close() {
    if (nativeDirectoryNamespaceHandle != 0) {
      releaseNative(nativeDirectoryNamespaceHandle);
      nativeDirectoryNamespaceHandle = 0;
    }
  }

  /**
   * Returns the native handle for this namespace. Used internally for passing to Dataset.open() for
   * namespace commit handler support.
   */
  public long getNativeHandle() {
    ensureInitialized();
    return nativeDirectoryNamespaceHandle;
  }

  // Operation metrics methods

  /**
   * Retrieve operation metrics as a map.
   *
   * <p>Returns a map where keys are operation names (e.g., "list_tables", "describe_table") and
   * values are the number of times each operation was called.
   *
   * <p>Returns an empty map if {@code ops_metrics_enabled} was false when creating the namespace.
   *
   * @return operation name to call count mapping
   */
  public Map<String, Long> retrieveOpsMetrics() {
    ensureInitialized();
    return retrieveOpsMetricsNative(nativeDirectoryNamespaceHandle);
  }

  /**
   * Reset all operation metrics counters to zero.
   *
   * <p>Does nothing if {@code ops_metrics_enabled} was false when creating the namespace.
   */
  public void resetOpsMetrics() {
    ensureInitialized();
    resetOpsMetricsNative(nativeDirectoryNamespaceHandle);
  }

  private void ensureInitialized() {
    if (nativeDirectoryNamespaceHandle == 0) {
      throw new IllegalStateException(
          "DirectoryNamespace not initialized. Call initialize() first.");
    }
  }

  private static String toJson(Object obj) {
    try {
      return OBJECT_MAPPER.writeValueAsString(obj);
    } catch (JsonProcessingException e) {
      throw new RuntimeException("Failed to serialize request to JSON", e);
    }
  }

  private static <T> T fromJson(String json, Class<T> clazz) {
    try {
      return OBJECT_MAPPER.readValue(json, clazz);
    } catch (JsonProcessingException e) {
      throw new RuntimeException("Failed to deserialize response from JSON", e);
    }
  }

  // Native methods
  private native long createNative(Map<String, String> properties);

  private native long createNativeWithProvider(
      Map<String, String> properties, DynamicContextProvider contextProvider);

  private native void releaseNative(long handle);

  private native String namespaceIdNative(long handle);

  private native String listNamespacesNative(long handle, String requestJson);

  private native String describeNamespaceNative(long handle, String requestJson);

  private native String createNamespaceNative(long handle, String requestJson);

  private native String dropNamespaceNative(long handle, String requestJson);

  private native void namespaceExistsNative(long handle, String requestJson);

  private native String listTablesNative(long handle, String requestJson);

  private native String describeTableNative(long handle, String requestJson);

  private native String registerTableNative(long handle, String requestJson);

  private native void tableExistsNative(long handle, String requestJson);

  private native String dropTableNative(long handle, String requestJson);

  private native String deregisterTableNative(long handle, String requestJson);

  private native long countTableRowsNative(long handle, String requestJson);

  private native String createTableNative(long handle, String requestJson, byte[] requestData);

  private native String declareTableNative(long handle, String requestJson);

  private native String renameTableNative(long handle, String requestJson);

  private native String insertIntoTableNative(long handle, String requestJson, byte[] requestData);

  private native String mergeInsertIntoTableNative(
      long handle, String requestJson, byte[] requestData);

  private native String updateTableNative(long handle, String requestJson);

  private native String deleteFromTableNative(long handle, String requestJson);

  private native byte[] queryTableNative(long handle, String requestJson);

  private native String createTableIndexNative(long handle, String requestJson);

  private native String listTableIndicesNative(long handle, String requestJson);

  private native String describeTableIndexStatsNative(long handle, String requestJson);

  private native String describeTransactionNative(long handle, String requestJson);

  private native String alterTransactionNative(long handle, String requestJson);

  private native String listTableVersionsNative(long handle, String requestJson);

  private native String createTableVersionNative(long handle, String requestJson);

  private native String describeTableVersionNative(long handle, String requestJson);

  private native String batchDeleteTableVersionsNative(long handle, String requestJson);

  private native String createTableScalarIndexNative(long handle, String requestJson);

  private native String dropTableIndexNative(long handle, String requestJson);

  private native String listAllTablesNative(long handle, String requestJson);

  private native String restoreTableNative(long handle, String requestJson);

  private native String updateTableSchemaMetadataNative(long handle, String requestJson);

  private native String getTableStatsNative(long handle, String requestJson);

  private native String explainTableQueryPlanNative(long handle, String requestJson);

  private native String analyzeTableQueryPlanNative(long handle, String requestJson);

  private native String alterTableAddColumnsNative(long handle, String requestJson);

  private native String alterTableAlterColumnsNative(long handle, String requestJson);

  private native String alterTableDropColumnsNative(long handle, String requestJson);

  private native String listTableTagsNative(long handle, String requestJson);

  private native String getTableTagVersionNative(long handle, String requestJson);

  private native String createTableTagNative(long handle, String requestJson);

  private native String deleteTableTagNative(long handle, String requestJson);

  private native String updateTableTagNative(long handle, String requestJson);

  private native String createMaterializedViewNative(long handle, String requestJson);

  private native Map<String, Long> retrieveOpsMetricsNative(long handle);

  private native void resetOpsMetricsNative(long handle);

  // ==========================================================================
  // Provider loading helpers
  // ==========================================================================

  private static final String PROVIDER_PREFIX = "dynamic_context_provider.";
  private static final String IMPL_KEY = "dynamic_context_provider.impl";

  /**
   * Create a context provider from properties if configured.
   *
   * <p>Loads the class specified by {@code dynamic_context_provider.impl} from the class path and
   * instantiates it with the extracted provider properties.
   */
  private static Optional<DynamicContextProvider> createProviderFromProperties(
      Map<String, String> properties) {
    String className = properties.get(IMPL_KEY);
    if (className == null || className.isEmpty()) {
      return Optional.empty();
    }

    // Extract provider-specific properties (strip prefix, exclude impl key)
    Map<String, String> providerProps = new HashMap<>();
    for (Map.Entry<String, String> entry : properties.entrySet()) {
      String key = entry.getKey();
      if (key.startsWith(PROVIDER_PREFIX) && !key.equals(IMPL_KEY)) {
        String propName = key.substring(PROVIDER_PREFIX.length());
        providerProps.put(propName, entry.getValue());
      }
    }

    try {
      Class<?> providerClass = Class.forName(className);
      if (!DynamicContextProvider.class.isAssignableFrom(providerClass)) {
        throw new IllegalArgumentException(
            String.format(
                "Class '%s' does not implement DynamicContextProvider interface", className));
      }

      @SuppressWarnings("unchecked")
      Class<? extends DynamicContextProvider> typedClass =
          (Class<? extends DynamicContextProvider>) providerClass;

      Constructor<? extends DynamicContextProvider> constructor =
          typedClass.getConstructor(Map.class);
      return Optional.of(constructor.newInstance(providerProps));

    } catch (ClassNotFoundException e) {
      throw new IllegalArgumentException(
          String.format("Failed to load context provider class '%s': %s", className, e), e);
    } catch (NoSuchMethodException e) {
      throw new IllegalArgumentException(
          String.format(
              "Context provider class '%s' must have a public constructor "
                  + "that accepts Map<String, String>",
              className),
          e);
    } catch (ReflectiveOperationException e) {
      throw new IllegalArgumentException(
          String.format("Failed to instantiate context provider '%s': %s", className, e), e);
    }
  }

  /** Filter out dynamic_context_provider.* properties from the map. */
  private static Map<String, String> filterProviderProperties(Map<String, String> properties) {
    Map<String, String> filtered = new HashMap<>();
    for (Map.Entry<String, String> entry : properties.entrySet()) {
      if (!entry.getKey().startsWith(PROVIDER_PREFIX)) {
        filtered.put(entry.getKey(), entry.getValue());
      }
    }
    return filtered;
  }
}
