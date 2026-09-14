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

import org.lance.namespace.model.AlterTableAddColumnsRequest;
import org.lance.namespace.model.AlterTableAddColumnsResponse;
import org.lance.namespace.model.AlterTableAlterColumnsRequest;
import org.lance.namespace.model.AlterTableAlterColumnsResponse;
import org.lance.namespace.model.AlterTableDropColumnsRequest;
import org.lance.namespace.model.AlterTableDropColumnsResponse;
import org.lance.namespace.model.AlterTransactionRequest;
import org.lance.namespace.model.AlterTransactionResponse;
import org.lance.namespace.model.AnalyzeTableQueryPlanRequest;
import org.lance.namespace.model.BatchCommitTablesRequest;
import org.lance.namespace.model.BatchCommitTablesResponse;
import org.lance.namespace.model.BatchCreateTableVersionsRequest;
import org.lance.namespace.model.BatchCreateTableVersionsResponse;
import org.lance.namespace.model.BatchDeleteTableVersionsRequest;
import org.lance.namespace.model.BatchDeleteTableVersionsResponse;
import org.lance.namespace.model.CountTableRowsRequest;
import org.lance.namespace.model.CountTableRowsResponse;
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

import org.apache.arrow.memory.BufferAllocator;

import java.io.Closeable;
import java.util.Map;

/**
 * A custom namespace wrapper that delegates all operations to an inner DirectoryNamespace.
 *
 * <p>This class is used for testing to verify that the Java-Rust binding works correctly for custom
 * namespace implementations that wrap DirectoryNamespace.
 */
public class CustomNamespace implements LanceNamespace, Closeable {
  private final DirectoryNamespace inner;

  /**
   * Creates a CustomNamespace wrapping the given DirectoryNamespace.
   *
   * @param inner The DirectoryNamespace to wrap
   */
  public CustomNamespace(DirectoryNamespace inner) {
    this.inner = inner;
  }

  /**
   * Gets the wrapped DirectoryNamespace for metrics retrieval.
   *
   * @return The inner DirectoryNamespace
   */
  public DirectoryNamespace getInner() {
    return inner;
  }

  /**
   * Gets the native handle from the inner namespace. This is required for the JNI layer to work
   * correctly with wrapped namespaces.
   *
   * @return The native handle
   */
  public long getNativeHandle() {
    return inner.getNativeHandle();
  }

  @Override
  public void initialize(Map<String, String> configProperties, BufferAllocator allocator) {
    // Already initialized in constructor via inner namespace
  }

  @Override
  public String namespaceId() {
    return "CustomNamespace[" + inner.namespaceId() + "]";
  }

  // Namespace operations

  @Override
  public ListNamespacesResponse listNamespaces(ListNamespacesRequest request) {
    return inner.listNamespaces(request);
  }

  @Override
  public DescribeNamespaceResponse describeNamespace(DescribeNamespaceRequest request) {
    return inner.describeNamespace(request);
  }

  @Override
  public CreateNamespaceResponse createNamespace(CreateNamespaceRequest request) {
    return inner.createNamespace(request);
  }

  @Override
  public DropNamespaceResponse dropNamespace(DropNamespaceRequest request) {
    return inner.dropNamespace(request);
  }

  @Override
  public NamespaceExistsResponse namespaceExists(NamespaceExistsRequest request) {
    return inner.namespaceExists(request);
  }

  // Table operations

  @Override
  public ListTablesResponse listTables(ListTablesRequest request) {
    return inner.listTables(request);
  }

  @Override
  public DescribeTableResponse describeTable(DescribeTableRequest request) {
    return inner.describeTable(request);
  }

  @Override
  public RegisterTableResponse registerTable(RegisterTableRequest request) {
    return inner.registerTable(request);
  }

  @Override
  public TableExistsResponse tableExists(TableExistsRequest request) {
    return inner.tableExists(request);
  }

  @Override
  public DropTableResponse dropTable(DropTableRequest request) {
    return inner.dropTable(request);
  }

  @Override
  public DeregisterTableResponse deregisterTable(DeregisterTableRequest request) {
    return inner.deregisterTable(request);
  }

  @Override
  public CountTableRowsResponse countTableRows(CountTableRowsRequest request) {
    return inner.countTableRows(request);
  }

  // Data operations

  @Override
  public CreateTableResponse createTable(CreateTableRequest request, byte[] requestData) {
    return inner.createTable(request, requestData);
  }

  @Override
  public DeclareTableResponse declareTable(DeclareTableRequest request) {
    return inner.declareTable(request);
  }

  @Override
  public InsertIntoTableResponse insertIntoTable(
      InsertIntoTableRequest request, byte[] requestData) {
    return inner.insertIntoTable(request, requestData);
  }

  @Override
  public MergeInsertIntoTableResponse mergeInsertIntoTable(
      MergeInsertIntoTableRequest request, byte[] requestData) {
    return inner.mergeInsertIntoTable(request, requestData);
  }

  @Override
  public UpdateTableResponse updateTable(UpdateTableRequest request) {
    return inner.updateTable(request);
  }

  @Override
  public DeleteFromTableResponse deleteFromTable(DeleteFromTableRequest request) {
    return inner.deleteFromTable(request);
  }

  @Override
  public QueryTableResponse queryTable(QueryTableRequest request) {
    return inner.queryTable(request);
  }

  // Index operations

  @Override
  public CreateTableIndexResponse createTableIndex(CreateTableIndexRequest request) {
    return inner.createTableIndex(request);
  }

  @Override
  public CreateTableScalarIndexResponse createTableScalarIndex(CreateTableIndexRequest request) {
    return inner.createTableScalarIndex(request);
  }

  @Override
  public ListTableIndicesResponse listTableIndices(ListTableIndicesRequest request) {
    return inner.listTableIndices(request);
  }

  @Override
  public DescribeTableIndexStatsResponse describeTableIndexStats(
      DescribeTableIndexStatsRequest request, String indexName) {
    return inner.describeTableIndexStats(request, indexName);
  }

  @Override
  public DropTableIndexResponse dropTableIndex(DropTableIndexRequest request, String indexName) {
    return inner.dropTableIndex(request, indexName);
  }

  // Table version and schema operations

  @Override
  public ListTablesResponse listAllTables(ListTablesRequest request) {
    return inner.listAllTables(request);
  }

  @Override
  public RestoreTableResponse restoreTable(RestoreTableRequest request) {
    return inner.restoreTable(request);
  }

  @Override
  public RenameTableResponse renameTable(RenameTableRequest request) {
    return inner.renameTable(request);
  }

  @Override
  public ListTableVersionsResponse listTableVersions(ListTableVersionsRequest request) {
    return inner.listTableVersions(request);
  }

  @Override
  public CreateTableVersionResponse createTableVersion(CreateTableVersionRequest request) {
    return inner.createTableVersion(request);
  }

  @Override
  public DescribeTableVersionResponse describeTableVersion(DescribeTableVersionRequest request) {
    return inner.describeTableVersion(request);
  }

  @Override
  public BatchDeleteTableVersionsResponse batchDeleteTableVersions(
      BatchDeleteTableVersionsRequest request) {
    return inner.batchDeleteTableVersions(request);
  }

  @Override
  public BatchCreateTableVersionsResponse batchCreateTableVersions(
      BatchCreateTableVersionsRequest request) {
    return inner.batchCreateTableVersions(request);
  }

  @Override
  public BatchCommitTablesResponse batchCommitTables(BatchCommitTablesRequest request) {
    return inner.batchCommitTables(request);
  }

  @Override
  public UpdateTableSchemaMetadataResponse updateTableSchemaMetadata(
      UpdateTableSchemaMetadataRequest request) {
    return inner.updateTableSchemaMetadata(request);
  }

  @Override
  public GetTableStatsResponse getTableStats(GetTableStatsRequest request) {
    return inner.getTableStats(request);
  }

  // Query plan operations

  @Override
  public String explainTableQueryPlan(ExplainTableQueryPlanRequest request) {
    return inner.explainTableQueryPlan(request);
  }

  @Override
  public String analyzeTableQueryPlan(AnalyzeTableQueryPlanRequest request) {
    return inner.analyzeTableQueryPlan(request);
  }

  // Column operations

  @Override
  public AlterTableAddColumnsResponse alterTableAddColumns(AlterTableAddColumnsRequest request) {
    return inner.alterTableAddColumns(request);
  }

  @Override
  public AlterTableAlterColumnsResponse alterTableAlterColumns(
      AlterTableAlterColumnsRequest request) {
    return inner.alterTableAlterColumns(request);
  }

  @Override
  public AlterTableDropColumnsResponse alterTableDropColumns(AlterTableDropColumnsRequest request) {
    return inner.alterTableDropColumns(request);
  }

  // Tag operations

  @Override
  public ListTableTagsResponse listTableTags(ListTableTagsRequest request) {
    return inner.listTableTags(request);
  }

  @Override
  public GetTableTagVersionResponse getTableTagVersion(GetTableTagVersionRequest request) {
    return inner.getTableTagVersion(request);
  }

  @Override
  public CreateTableTagResponse createTableTag(CreateTableTagRequest request) {
    return inner.createTableTag(request);
  }

  @Override
  public DeleteTableTagResponse deleteTableTag(DeleteTableTagRequest request) {
    return inner.deleteTableTag(request);
  }

  @Override
  public UpdateTableTagResponse updateTableTag(UpdateTableTagRequest request) {
    return inner.updateTableTag(request);
  }

  // Transaction operations

  @Override
  public DescribeTransactionResponse describeTransaction(DescribeTransactionRequest request) {
    return inner.describeTransaction(request);
  }

  @Override
  public AlterTransactionResponse alterTransaction(AlterTransactionRequest request) {
    return inner.alterTransaction(request);
  }

  @Override
  public void close() {
    inner.close();
  }
}
