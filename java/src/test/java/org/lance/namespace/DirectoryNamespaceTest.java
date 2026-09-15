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

import org.lance.CommitBuilder;
import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.FragmentMetadata;
import org.lance.ReadOptions;
import org.lance.Transaction;
import org.lance.WriteParams;
import org.lance.namespace.errors.ErrorCode;
import org.lance.namespace.errors.LanceNamespaceException;
import org.lance.namespace.model.CountTableRowsRequest;
import org.lance.namespace.model.CreateNamespaceRequest;
import org.lance.namespace.model.CreateNamespaceResponse;
import org.lance.namespace.model.CreateTableIndexRequest;
import org.lance.namespace.model.CreateTableIndexResponse;
import org.lance.namespace.model.CreateTableRequest;
import org.lance.namespace.model.CreateTableResponse;
import org.lance.namespace.model.DeclareTableRequest;
import org.lance.namespace.model.DeclareTableResponse;
import org.lance.namespace.model.DeregisterTableRequest;
import org.lance.namespace.model.DeregisterTableResponse;
import org.lance.namespace.model.DescribeNamespaceRequest;
import org.lance.namespace.model.DescribeNamespaceResponse;
import org.lance.namespace.model.DescribeTableRequest;
import org.lance.namespace.model.DescribeTableResponse;
import org.lance.namespace.model.DescribeTableVersionRequest;
import org.lance.namespace.model.DescribeTableVersionResponse;
import org.lance.namespace.model.DropNamespaceRequest;
import org.lance.namespace.model.DropNamespaceResponse;
import org.lance.namespace.model.DropTableRequest;
import org.lance.namespace.model.DropTableResponse;
import org.lance.namespace.model.InsertIntoTableRequest;
import org.lance.namespace.model.InsertIntoTableResponse;
import org.lance.namespace.model.ListNamespacesRequest;
import org.lance.namespace.model.ListNamespacesResponse;
import org.lance.namespace.model.ListTableIndicesRequest;
import org.lance.namespace.model.ListTableIndicesResponse;
import org.lance.namespace.model.ListTableVersionsRequest;
import org.lance.namespace.model.ListTableVersionsResponse;
import org.lance.namespace.model.ListTablesRequest;
import org.lance.namespace.model.ListTablesResponse;
import org.lance.namespace.model.NamespaceExistsRequest;
import org.lance.namespace.model.QueryTableRequest;
import org.lance.namespace.model.QueryTableRequestVector;
import org.lance.namespace.model.RegisterTableRequest;
import org.lance.namespace.model.RegisterTableResponse;
import org.lance.namespace.model.TableExistsRequest;
import org.lance.operation.Append;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/** Tests for DirectoryNamespace implementation. */
public class DirectoryNamespaceTest {
  @TempDir Path tempDir;

  protected BufferAllocator allocator;
  protected LanceNamespace namespaceClient;
  protected DirectoryNamespace innerNamespaceClient;

  @BeforeEach
  void setUp() {
    allocator = new RootAllocator(Long.MAX_VALUE);
    innerNamespaceClient = new DirectoryNamespace();

    Map<String, String> config = new HashMap<>();
    config.put("root", tempDir.toString());
    innerNamespaceClient.initialize(config, allocator);
    namespaceClient = wrapNamespace(innerNamespaceClient);
  }

  /**
   * Factory method to wrap the DirectoryNamespace. Subclasses can override this to provide a custom
   * namespace implementation.
   *
   * @param inner The DirectoryNamespace to wrap
   * @return The namespace client to use in tests (may be the same as inner or a wrapper)
   */
  protected LanceNamespace wrapNamespace(DirectoryNamespace inner) {
    return inner;
  }

  @AfterEach
  void tearDown() {
    if (namespaceClient != null && namespaceClient instanceof java.io.Closeable) {
      try {
        ((java.io.Closeable) namespaceClient).close();
      } catch (Exception e) {
        // Ignore
      }
    }
    if (allocator != null) {
      allocator.close();
    }
  }

  private byte[] createTestTableData() throws Exception {
    Schema schema =
        new Schema(
            Arrays.asList(
                new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null),
                new Field("name", FieldType.nullable(new ArrowType.Utf8()), null),
                new Field("age", FieldType.nullable(new ArrowType.Int(32, true)), null)));

    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      IntVector idVector = (IntVector) root.getVector("id");
      VarCharVector nameVector = (VarCharVector) root.getVector("name");
      IntVector ageVector = (IntVector) root.getVector("age");

      // Allocate space for 3 rows
      idVector.allocateNew(3);
      nameVector.allocateNew(3);
      ageVector.allocateNew(3);

      idVector.set(0, 1);
      nameVector.set(0, "Alice".getBytes());
      ageVector.set(0, 30);

      idVector.set(1, 2);
      nameVector.set(1, "Bob".getBytes());
      ageVector.set(1, 25);

      idVector.set(2, 3);
      nameVector.set(2, "Charlie".getBytes());
      ageVector.set(2, 35);

      // Set value counts
      idVector.setValueCount(3);
      nameVector.setValueCount(3);
      ageVector.setValueCount(3);
      root.setRowCount(3);

      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
        writer.writeBatch();
      }
      return out.toByteArray();
    }
  }

  @Test
  void testNamespaceId() {
    String namespaceId = namespaceClient.namespaceId();
    assertNotNull(namespaceId);
    assertTrue(
        namespaceId.contains("DirectoryNamespace"),
        "namespaceId should contain 'DirectoryNamespace', got: " + namespaceId);
  }

  @Test
  void testOpenSpecificVersionDoesNotPassVersionToDescribeTable() throws Exception {
    VersionRejectingNamespace versionRejectingNamespace =
        new VersionRejectingNamespace(innerNamespaceClient);
    namespaceClient = versionRejectingNamespace;
    List<String> tableId = Arrays.asList("test_table");

    namespaceClient.createTable(new CreateTableRequest().id(tableId), createTestTableData());
    namespaceClient.insertIntoTable(
        new InsertIntoTableRequest().id(tableId).mode("append"), createTestTableData());

    try (Dataset versionOne =
        Dataset.open()
            .allocator(allocator)
            .namespaceClient(namespaceClient)
            .tableId(tableId)
            .readOptions(new ReadOptions.Builder().setVersion(1L).build())
            .build()) {
      assertEquals(1, versionOne.version());
      assertEquals(3, versionOne.countRows());
    }

    assertTrue(
        versionRejectingNamespace.getDescribeTableCallCount() > 0,
        "Expected describeTable to be called when opening through namespace");
  }

  @Test
  void testCreateAndListNamespaces() {
    // Create a namespace
    CreateNamespaceRequest createReq = new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    CreateNamespaceResponse createResp = namespaceClient.createNamespace(createReq);
    assertNotNull(createResp);

    // List namespaces
    ListNamespacesRequest listReq = new ListNamespacesRequest();
    ListNamespacesResponse listResp = namespaceClient.listNamespaces(listReq);
    assertNotNull(listResp);
    assertNotNull(listResp.getNamespaces());
    assertTrue(listResp.getNamespaces().contains("workspace"));
  }

  @Test
  void testDescribeNamespace() {
    // Create a namespace
    CreateNamespaceRequest createReq = new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createReq);

    // Describe namespace
    DescribeNamespaceRequest descReq =
        new DescribeNamespaceRequest().id(Arrays.asList("workspace"));
    DescribeNamespaceResponse descResp = namespaceClient.describeNamespace(descReq);
    assertNotNull(descResp);
    assertNotNull(descResp.getProperties());
  }

  @Test
  void testNamespaceExists() {
    // Create a namespace
    CreateNamespaceRequest createReq = new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createReq);

    // Check existence
    NamespaceExistsRequest existsReq = new NamespaceExistsRequest().id(Arrays.asList("workspace"));
    assertDoesNotThrow(() -> namespaceClient.namespaceExists(existsReq));

    // Check non-existent namespace
    NamespaceExistsRequest notExistsReq =
        new NamespaceExistsRequest().id(Arrays.asList("nonexistent"));
    LanceNamespaceException ex =
        assertThrows(
            LanceNamespaceException.class, () -> namespaceClient.namespaceExists(notExistsReq));
    assertEquals(ErrorCode.NAMESPACE_NOT_FOUND, ex.getErrorCode());
  }

  @Test
  void testDropNamespace() {
    // Create a namespace
    CreateNamespaceRequest createReq = new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createReq);

    // Drop namespace
    DropNamespaceRequest dropReq = new DropNamespaceRequest().id(Arrays.asList("workspace"));
    DropNamespaceResponse dropResp = namespaceClient.dropNamespace(dropReq);
    assertNotNull(dropResp);

    // Verify it's gone
    NamespaceExistsRequest existsReq = new NamespaceExistsRequest().id(Arrays.asList("workspace"));
    assertThrows(LanceNamespaceException.class, () -> namespaceClient.namespaceExists(existsReq));
  }

  @Test
  void testCreateTable() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Create table with data
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    CreateTableResponse createResp = namespaceClient.createTable(createReq, tableData);

    assertNotNull(createResp);
    assertNotNull(createResp.getLocation());
    assertTrue(createResp.getLocation().contains("test_table"));
    assertEquals(Long.valueOf(1), createResp.getVersion());
  }

  @Test
  void testListTables() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Create a table
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespaceClient.createTable(createReq, tableData);

    // List tables
    ListTablesRequest listReq = new ListTablesRequest().id(Arrays.asList("workspace"));
    ListTablesResponse listResp = namespaceClient.listTables(listReq);

    assertNotNull(listResp);
    assertNotNull(listResp.getTables());
    assertTrue(listResp.getTables().contains("test_table"));
  }

  @Test
  void testDescribeTable() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Create a table
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespaceClient.createTable(createReq, tableData);

    // Describe table
    DescribeTableRequest descReq =
        new DescribeTableRequest().id(Arrays.asList("workspace", "test_table"));
    DescribeTableResponse descResp = namespaceClient.describeTable(descReq);

    assertNotNull(descResp);
    assertNotNull(descResp.getLocation());
    assertTrue(descResp.getLocation().contains("test_table"));
  }

  @Test
  void testTableExists() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Create a table
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespaceClient.createTable(createReq, tableData);

    // Check existence
    TableExistsRequest existsReq =
        new TableExistsRequest().id(Arrays.asList("workspace", "test_table"));
    assertDoesNotThrow(() -> namespaceClient.tableExists(existsReq));

    // Check non-existent table
    TableExistsRequest notExistsReq =
        new TableExistsRequest().id(Arrays.asList("workspace", "nonexistent"));
    assertThrows(LanceNamespaceException.class, () -> namespaceClient.tableExists(notExistsReq));
  }

  @Test
  void testDropTable() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Create a table
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespaceClient.createTable(createReq, tableData);

    // Drop table
    DropTableRequest dropReq = new DropTableRequest().id(Arrays.asList("workspace", "test_table"));
    DropTableResponse dropResp = namespaceClient.dropTable(dropReq);
    assertNotNull(dropResp);

    // Verify it's gone
    TableExistsRequest existsReq =
        new TableExistsRequest().id(Arrays.asList("workspace", "test_table"));
    assertThrows(LanceNamespaceException.class, () -> namespaceClient.tableExists(existsReq));
  }

  @Test
  void testDescribeTableReturnsManagedVersioningWhenTrackingEnabled() throws Exception {
    // Create namespace with table_version_tracking_enabled and manifest_enabled
    DirectoryNamespace trackingNs = new DirectoryNamespace();
    Map<String, String> config = new HashMap<>();
    config.put("root", tempDir.toString());
    config.put("table_version_tracking_enabled", "true");
    config.put("manifest_enabled", "true");
    trackingNs.initialize(config, allocator);

    try {
      // Create parent namespace
      CreateNamespaceRequest createNsReq =
          new CreateNamespaceRequest().id(Arrays.asList("workspace"));
      trackingNs.createNamespace(createNsReq);

      // Create a table
      byte[] tableData = createTestTableData();
      CreateTableRequest createReq =
          new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
      trackingNs.createTable(createReq, tableData);

      // Describe table should return managedVersioning=true
      DescribeTableRequest descReq =
          new DescribeTableRequest().id(Arrays.asList("workspace", "test_table"));
      DescribeTableResponse descResp = trackingNs.describeTable(descReq);

      assertNotNull(descResp);
      assertNotNull(descResp.getLocation());
      assertTrue(
          Boolean.TRUE.equals(descResp.getManagedVersioning()),
          "Expected managedVersioning=true, got " + descResp.getManagedVersioning());
    } finally {
      trackingNs.close();
    }
  }

  @Test
  void testDescribeTableVersion() throws Exception {
    // Use multi-level table ID with manifest_enabled
    DirectoryNamespace trackingNs = new DirectoryNamespace();
    Map<String, String> config = new HashMap<>();
    config.put("root", tempDir.toString());
    config.put("manifest_enabled", "true");
    trackingNs.initialize(config, allocator);

    try {
      // Create parent namespace
      CreateNamespaceRequest createNsReq =
          new CreateNamespaceRequest().id(Arrays.asList("workspace"));
      trackingNs.createNamespace(createNsReq);

      // Create a table with multi-level ID
      byte[] tableData = createTestTableData();
      CreateTableRequest createReq =
          new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
      trackingNs.createTable(createReq, tableData);

      // Describe table version
      DescribeTableVersionRequest descReq =
          new DescribeTableVersionRequest()
              .id(Arrays.asList("workspace", "test_table"))
              .version(1L);
      DescribeTableVersionResponse descResp = trackingNs.describeTableVersion(descReq);

      assertNotNull(descResp);
      assertNotNull(descResp.getVersion());
      assertEquals(Long.valueOf(1), descResp.getVersion().getVersion());
      assertNotNull(descResp.getVersion().getManifestPath());
    } finally {
      trackingNs.close();
    }
  }

  /**
   * Creates a DirectoryNamespace configured for testing managed versioning with ops metrics.
   *
   * @param root The root path for the namespace
   * @return A DirectoryNamespace with table_version_tracking_enabled and ops_metrics_enabled
   */
  private DirectoryNamespace createManagedVersioningNamespace(Path root) {
    Map<String, String> dirProps = new HashMap<>();
    dirProps.put("root", root.toString());
    dirProps.put("table_version_tracking_enabled", "true");
    dirProps.put("manifest_enabled", "true");
    dirProps.put("ops_metrics_enabled", "true");

    DirectoryNamespace ns = new DirectoryNamespace();
    try (BufferAllocator allocator = new RootAllocator()) {
      ns.initialize(dirProps, allocator);
    }
    return ns;
  }

  private static int getCreateTableVersionCount(DirectoryNamespace ns) {
    Map<String, Long> metrics = ns.retrieveOpsMetrics();
    return metrics.getOrDefault("create_table_version", 0L).intValue();
  }

  private static int getDescribeTableVersionCount(DirectoryNamespace ns) {
    Map<String, Long> metrics = ns.retrieveOpsMetrics();
    return metrics.getOrDefault("describe_table_version", 0L).intValue();
  }

  private static int getListTableVersionsCount(DirectoryNamespace ns) {
    Map<String, Long> metrics = ns.retrieveOpsMetrics();
    return metrics.getOrDefault("list_table_versions", 0L).intValue();
  }

  @Test
  void testExternalManifestStoreInvokesNamespaceApis(@TempDir Path managedVersioningTempDir)
      throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Create namespace with table_version_tracking_enabled and ops_metrics_enabled
      DirectoryNamespace namespaceClient =
          createManagedVersioningNamespace(managedVersioningTempDir);
      String tableName = "test_table";
      java.util.List<String> tableId = Arrays.asList(tableName);

      // Create schema and data
      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("a", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("b", FieldType.nullable(new ArrowType.Int(32, true)), null)));

      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(2);
        bVector.allocateNew(2);

        aVector.set(0, 1);
        bVector.set(0, 2);
        aVector.set(1, 10);
        bVector.set(1, 20);

        aVector.setValueCount(2);
        bVector.setValueCount(2);
        root.setRowCount(2);

        ArrowReader testReader =
            new ArrowReader(allocator) {
              boolean firstRead = true;

              @Override
              public boolean loadNextBatch() {
                if (firstRead) {
                  firstRead = false;
                  return true;
                }
                return false;
              }

              @Override
              public long bytesRead() {
                return 0;
              }

              @Override
              protected void closeReadSource() {}

              @Override
              protected Schema readSchema() {
                return schema;
              }

              @Override
              public VectorSchemaRoot getVectorSchemaRoot() {
                return root;
              }
            };

        // Create dataset through namespace
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(testReader)
                .namespaceClient(namespaceClient)
                .tableId(tableId)
                .mode(WriteParams.WriteMode.CREATE)
                .execute()) {
          assertEquals(2, dataset.countRows());
          assertEquals(1, dataset.version());
        }
      }

      // Verify describe_table returns managed_versioning=true
      DescribeTableRequest descReq = new DescribeTableRequest();
      descReq.setId(tableId);
      DescribeTableResponse descResp = namespaceClient.describeTable(descReq);

      assertEquals(
          Boolean.TRUE,
          descResp.getManagedVersioning(),
          "Expected managedVersioning=true when table_version_tracking_enabled");

      // Open dataset through namespace - this should call list_table_versions for latest
      int initialListCount = getListTableVersionsCount(namespaceClient);
      try (Dataset dsFromNamespace =
          Dataset.open()
              .allocator(allocator)
              .namespaceClient(namespaceClient)
              .tableId(tableId)
              .build()) {

        assertEquals(2, dsFromNamespace.countRows());
        assertEquals(1, dsFromNamespace.version());
      }
      assertEquals(
          initialListCount + 1,
          getListTableVersionsCount(namespaceClient),
          "list_table_versions should have been called once when opening latest version");

      // Verify create_table_version was called once during CREATE
      assertEquals(
          1,
          getCreateTableVersionCount(namespaceClient),
          "create_table_version should have been called once during CREATE");

      try (VectorSchemaRoot appendRoot = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) appendRoot.getVector("a");
        IntVector bVector = (IntVector) appendRoot.getVector("b");

        aVector.allocateNew(2);
        bVector.allocateNew(2);

        aVector.set(0, 100);
        bVector.set(0, 200);
        aVector.set(1, 1000);
        bVector.set(1, 2000);

        aVector.setValueCount(2);
        bVector.setValueCount(2);
        appendRoot.setRowCount(2);

        ArrowReader appendReader =
            new ArrowReader(allocator) {
              boolean firstRead = true;

              @Override
              public boolean loadNextBatch() {
                if (firstRead) {
                  firstRead = false;
                  return true;
                }
                return false;
              }

              @Override
              public long bytesRead() {
                return 0;
              }

              @Override
              protected void closeReadSource() {}

              @Override
              protected Schema readSchema() {
                return schema;
              }

              @Override
              public VectorSchemaRoot getVectorSchemaRoot() {
                return appendRoot;
              }
            };

        // Append through namespace
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(appendReader)
                .namespaceClient(namespaceClient)
                .tableId(tableId)
                .mode(WriteParams.WriteMode.APPEND)
                .execute()) {
          assertEquals(4, dataset.countRows());
          assertEquals(2, dataset.version());
        }
      }

      assertEquals(
          2,
          getCreateTableVersionCount(namespaceClient),
          "create_table_version should have been called twice (once for CREATE, once for APPEND)");

      // Open latest version - should call list_table_versions
      int listCountBeforeLatest = getListTableVersionsCount(namespaceClient);
      try (Dataset latestDs =
          Dataset.open()
              .allocator(allocator)
              .namespaceClient(namespaceClient)
              .tableId(tableId)
              .build()) {

        assertEquals(4, latestDs.countRows());
        assertEquals(2, latestDs.version());
      }
      assertEquals(
          listCountBeforeLatest + 1,
          getListTableVersionsCount(namespaceClient),
          "list_table_versions should have been called once when opening latest version");

      // Open specific version (version 1) - should call describe_table_version
      int describeCountBeforeV1 = getDescribeTableVersionCount(namespaceClient);
      try (Dataset v1Ds =
          Dataset.open()
              .allocator(allocator)
              .namespaceClient(namespaceClient)
              .tableId(tableId)
              .readOptions(new ReadOptions.Builder().setVersion(1L).build())
              .build()) {

        assertEquals(2, v1Ds.countRows());
        assertEquals(1, v1Ds.version());
      }
      assertEquals(
          describeCountBeforeV1 + 1,
          getDescribeTableVersionCount(namespaceClient),
          "describe_table_version should have been called once when opening version 1");

      namespaceClient.close();
    }
  }

  @Test
  void testDatasetBasedCommitBuilderWithNamespace(@TempDir Path managedVersioningTempDir)
      throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      DirectoryNamespace namespaceClient =
          createManagedVersioningNamespace(managedVersioningTempDir);
      String tableName = "test_table";
      List<String> tableId = Arrays.asList(tableName);

      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("name", FieldType.nullable(new ArrowType.Utf8()), null)));

      // Create initial dataset through namespace using WriteDatasetBuilder
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector idVector = (IntVector) root.getVector("id");
        VarCharVector nameVector = (VarCharVector) root.getVector("name");

        idVector.allocateNew(2);
        nameVector.allocateNew(2);
        idVector.set(0, 1);
        idVector.set(1, 2);
        nameVector.set(0, "Alice".getBytes());
        nameVector.set(1, "Bob".getBytes());
        idVector.setValueCount(2);
        nameVector.setValueCount(2);
        root.setRowCount(2);

        ArrowReader reader =
            new ArrowReader(allocator) {
              boolean firstRead = true;

              @Override
              public boolean loadNextBatch() {
                if (firstRead) {
                  firstRead = false;
                  return true;
                }
                return false;
              }

              @Override
              public long bytesRead() {
                return 0;
              }

              @Override
              protected void closeReadSource() {}

              @Override
              protected Schema readSchema() {
                return schema;
              }

              @Override
              public VectorSchemaRoot getVectorSchemaRoot() {
                return root;
              }
            };

        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(reader)
                .namespaceClient(namespaceClient)
                .tableId(tableId)
                .mode(WriteParams.WriteMode.CREATE)
                .execute()) {
          assertEquals(2, dataset.countRows());
          assertEquals(1, dataset.version());
        }
      }

      // Verify initial create used createTableVersion once
      assertEquals(
          1,
          getCreateTableVersionCount(namespaceClient),
          "create_table_version should be called once during CREATE");

      // Open dataset through namespace (returns dataset with managed versioning)
      Dataset existingDataset =
          Dataset.open()
              .allocator(allocator)
              .namespaceClient(namespaceClient)
              .tableId(tableId)
              .build();

      // Get the dataset URI for Fragment.create()
      String datasetUri = existingDataset.uri();

      // Create a new fragment independently (simulating Spark worker behavior)
      List<FragmentMetadata> fragments;
      try (VectorSchemaRoot appendRoot = VectorSchemaRoot.create(schema, allocator)) {
        IntVector idVector = (IntVector) appendRoot.getVector("id");
        VarCharVector nameVector = (VarCharVector) appendRoot.getVector("name");

        idVector.allocateNew(2);
        nameVector.allocateNew(2);
        idVector.set(0, 3);
        idVector.set(1, 4);
        nameVector.set(0, "Charlie".getBytes());
        nameVector.set(1, "Diana".getBytes());
        idVector.setValueCount(2);
        nameVector.setValueCount(2);
        appendRoot.setRowCount(2);

        fragments =
            Fragment.create(datasetUri, allocator, appendRoot, new WriteParams.Builder().build());
      }

      // Commit using dataset-based CommitBuilder WITH namespace (the new path)
      int createCountBefore = getCreateTableVersionCount(namespaceClient);
      try (Transaction txn =
              new Transaction.Builder()
                  .readVersion(existingDataset.version())
                  .operation(Append.builder().fragments(fragments).build())
                  .build();
          Dataset committed =
              new CommitBuilder(existingDataset)
                  .namespaceClient(namespaceClient)
                  .tableId(tableId)
                  .execute(txn)) {
        assertEquals(2, committed.version());
        assertEquals(4, committed.countRows());
      }

      // Verify createTableVersion was called for the dataset-based commit
      assertEquals(
          createCountBefore + 1,
          getCreateTableVersionCount(namespaceClient),
          "create_table_version should be called for dataset-based CommitBuilder with namespace");

      // Verify the data is accessible through namespace
      try (Dataset latestDs =
          Dataset.open()
              .allocator(allocator)
              .namespaceClient(namespaceClient)
              .tableId(tableId)
              .build()) {
        assertEquals(4, latestDs.countRows());
        assertEquals(2, latestDs.version());
      }

      existingDataset.close();
      namespaceClient.close();
    }
  }

  @Test
  void testConcurrentCreateAndDropWithSingleInstance() throws Exception {
    // Initialize namespace first - create parent namespace to ensure __manifest table
    // is created before concurrent operations
    CreateNamespaceRequest createNsReq = new CreateNamespaceRequest().id(Arrays.asList("test_ns"));
    namespaceClient.createNamespace(createNsReq);

    int numTables = 10;
    ExecutorService executor = Executors.newFixedThreadPool(numTables);
    CountDownLatch startLatch = new CountDownLatch(1);
    CountDownLatch doneLatch = new CountDownLatch(numTables);
    AtomicInteger successCount = new AtomicInteger(0);
    AtomicInteger failCount = new AtomicInteger(0);

    for (int i = 0; i < numTables; i++) {
      final int tableIndex = i;
      executor.submit(
          () -> {
            try {
              startLatch.await();

              String tableName = "concurrent_table_" + tableIndex;
              byte[] tableData = createTestTableData();

              CreateTableRequest createReq =
                  new CreateTableRequest().id(Arrays.asList("test_ns", tableName));
              namespaceClient.createTable(createReq, tableData);

              DropTableRequest dropReq =
                  new DropTableRequest().id(Arrays.asList("test_ns", tableName));
              namespaceClient.dropTable(dropReq);

              successCount.incrementAndGet();
            } catch (Exception e) {
              failCount.incrementAndGet();
            } finally {
              doneLatch.countDown();
            }
          });
    }

    startLatch.countDown();
    assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "Timed out waiting for tasks to complete");

    executor.shutdown();
    assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));

    assertEquals(numTables, successCount.get(), "All tasks should succeed");
    assertEquals(0, failCount.get(), "No tasks should fail");

    ListTablesRequest listReq = new ListTablesRequest().id(Arrays.asList("test_ns"));
    ListTablesResponse listResp = namespaceClient.listTables(listReq);
    assertEquals(0, listResp.getTables().size(), "All tables should be dropped");
  }

  @Test
  void testConcurrentCreateAndDropWithMultipleInstances() throws Exception {
    // Initialize namespace first with a single instance to ensure __manifest
    // table is created and parent namespace exists before concurrent operations
    DirectoryNamespace initNs = new DirectoryNamespace();
    Map<String, String> initConfig = new HashMap<>();
    initConfig.put("root", tempDir.toString());
    initConfig.put("inline_optimization_enabled", "false");
    initNs.initialize(initConfig, allocator);

    CreateNamespaceRequest createNsReq = new CreateNamespaceRequest().id(Arrays.asList("test_ns"));
    initNs.createNamespace(createNsReq);
    initNs.close();

    int numTables = 10;
    ExecutorService executor = Executors.newFixedThreadPool(numTables);
    CountDownLatch startLatch = new CountDownLatch(1);
    CountDownLatch doneLatch = new CountDownLatch(numTables);
    AtomicInteger successCount = new AtomicInteger(0);
    AtomicInteger failCount = new AtomicInteger(0);
    List<DirectoryNamespace> namespaces = new ArrayList<>();

    for (int i = 0; i < numTables; i++) {
      final int tableIndex = i;
      executor.submit(
          () -> {
            DirectoryNamespace localNs = null;
            try {
              startLatch.await();

              localNs = new DirectoryNamespace();
              Map<String, String> config = new HashMap<>();
              config.put("root", tempDir.toString());
              config.put("inline_optimization_enabled", "false");
              localNs.initialize(config, allocator);

              synchronized (namespaces) {
                namespaces.add(localNs);
              }

              String tableName = "multi_ns_table_" + tableIndex;
              byte[] tableData = createTestTableData();

              CreateTableRequest createReq =
                  new CreateTableRequest().id(Arrays.asList("test_ns", tableName));
              localNs.createTable(createReq, tableData);

              DropTableRequest dropReq =
                  new DropTableRequest().id(Arrays.asList("test_ns", tableName));
              localNs.dropTable(dropReq);

              successCount.incrementAndGet();
            } catch (Exception e) {
              failCount.incrementAndGet();
            } finally {
              doneLatch.countDown();
            }
          });
    }

    startLatch.countDown();
    assertTrue(doneLatch.await(60, TimeUnit.SECONDS), "Timed out waiting for tasks to complete");

    executor.shutdown();
    assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));

    // Close all namespace instances
    for (DirectoryNamespace ns : namespaces) {
      try {
        ns.close();
      } catch (Exception e) {
        // Ignore
      }
    }

    assertEquals(numTables, successCount.get(), "All tasks should succeed");
    assertEquals(0, failCount.get(), "No tasks should fail");

    // Verify with a fresh namespace
    DirectoryNamespace verifyNs = new DirectoryNamespace();
    Map<String, String> config = new HashMap<>();
    config.put("root", tempDir.toString());
    verifyNs.initialize(config, allocator);

    ListTablesRequest listReq = new ListTablesRequest().id(Arrays.asList("test_ns"));
    ListTablesResponse listResp = verifyNs.listTables(listReq);
    assertEquals(0, listResp.getTables().size(), "All tables should be dropped");

    verifyNs.close();
  }

  @Test
  void testConcurrentCreateThenDropFromDifferentInstance() throws Exception {
    // Initialize namespace first with a single instance to ensure __manifest
    // table is created and parent namespace exists before concurrent operations
    DirectoryNamespace initNs = new DirectoryNamespace();
    Map<String, String> initConfig = new HashMap<>();
    initConfig.put("root", tempDir.toString());
    initConfig.put("inline_optimization_enabled", "false");
    initNs.initialize(initConfig, allocator);

    CreateNamespaceRequest createNsReq = new CreateNamespaceRequest().id(Arrays.asList("test_ns"));
    initNs.createNamespace(createNsReq);
    initNs.close();

    int numTables = 10;

    // First, create all tables using separate namespace instances
    ExecutorService createExecutor = Executors.newFixedThreadPool(numTables);
    CountDownLatch createStartLatch = new CountDownLatch(1);
    CountDownLatch createDoneLatch = new CountDownLatch(numTables);
    AtomicInteger createSuccessCount = new AtomicInteger(0);
    List<DirectoryNamespace> createNamespaces = new ArrayList<>();

    for (int i = 0; i < numTables; i++) {
      final int tableIndex = i;
      createExecutor.submit(
          () -> {
            DirectoryNamespace localNs = null;
            try {
              createStartLatch.await();

              localNs = new DirectoryNamespace();
              Map<String, String> config = new HashMap<>();
              config.put("root", tempDir.toString());
              config.put("inline_optimization_enabled", "false");
              localNs.initialize(config, allocator);

              synchronized (createNamespaces) {
                createNamespaces.add(localNs);
              }

              String tableName = "cross_instance_table_" + tableIndex;
              byte[] tableData = createTestTableData();

              CreateTableRequest createReq =
                  new CreateTableRequest().id(Arrays.asList("test_ns", tableName));
              localNs.createTable(createReq, tableData);

              createSuccessCount.incrementAndGet();
            } catch (Exception e) {
              // Ignore - test will fail on assertion
            } finally {
              createDoneLatch.countDown();
            }
          });
    }

    createStartLatch.countDown();
    assertTrue(createDoneLatch.await(60, TimeUnit.SECONDS), "Timed out waiting for creates");
    createExecutor.shutdown();

    assertEquals(numTables, createSuccessCount.get(), "All creates should succeed");

    // Close create namespaces
    for (DirectoryNamespace ns : createNamespaces) {
      try {
        ns.close();
      } catch (Exception e) {
        // Ignore
      }
    }

    // Now drop all tables using NEW namespace instances
    ExecutorService dropExecutor = Executors.newFixedThreadPool(numTables);
    CountDownLatch dropStartLatch = new CountDownLatch(1);
    CountDownLatch dropDoneLatch = new CountDownLatch(numTables);
    AtomicInteger dropSuccessCount = new AtomicInteger(0);
    AtomicInteger dropFailCount = new AtomicInteger(0);
    List<DirectoryNamespace> dropNamespaces = new ArrayList<>();

    for (int i = 0; i < numTables; i++) {
      final int tableIndex = i;
      dropExecutor.submit(
          () -> {
            DirectoryNamespace localNs = null;
            try {
              dropStartLatch.await();

              localNs = new DirectoryNamespace();
              Map<String, String> config = new HashMap<>();
              config.put("root", tempDir.toString());
              config.put("inline_optimization_enabled", "false");
              localNs.initialize(config, allocator);

              synchronized (dropNamespaces) {
                dropNamespaces.add(localNs);
              }

              String tableName = "cross_instance_table_" + tableIndex;

              DropTableRequest dropReq =
                  new DropTableRequest().id(Arrays.asList("test_ns", tableName));
              localNs.dropTable(dropReq);

              dropSuccessCount.incrementAndGet();
            } catch (Exception e) {
              dropFailCount.incrementAndGet();
            } finally {
              dropDoneLatch.countDown();
            }
          });
    }

    dropStartLatch.countDown();
    assertTrue(dropDoneLatch.await(60, TimeUnit.SECONDS), "Timed out waiting for drops");
    dropExecutor.shutdown();

    // Close drop namespaces
    for (DirectoryNamespace ns : dropNamespaces) {
      try {
        ns.close();
      } catch (Exception e) {
        // Ignore
      }
    }

    assertEquals(numTables, dropSuccessCount.get(), "All drops should succeed");
    assertEquals(0, dropFailCount.get(), "No drops should fail");
  }

  @Test
  void testCountTableRows() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Create a table with 3 rows
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespaceClient.createTable(createReq, tableData);

    // Count rows
    CountTableRowsRequest countReq =
        new CountTableRowsRequest().id(Arrays.asList("workspace", "test_table"));
    long count = namespaceClient.countTableRows(countReq).getCount();
    assertEquals(3, count);
  }

  @Test
  void testCountTableRowsWithFilter() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Create a table with 3 rows (ages: 30, 25, 35)
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespaceClient.createTable(createReq, tableData);

    // Count rows with predicate
    CountTableRowsRequest countReq =
        new CountTableRowsRequest()
            .id(Arrays.asList("workspace", "test_table"))
            .predicate("age > 28");
    long count = namespaceClient.countTableRows(countReq).getCount();
    assertEquals(2, count); // Alice (30) and Charlie (35)
  }

  @Test
  void testInsertIntoTable() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Create a table with 3 rows
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespaceClient.createTable(createReq, tableData);

    // Insert more data
    byte[] newData = createTestTableData(); // Another 3 rows
    InsertIntoTableRequest insertReq =
        new InsertIntoTableRequest().id(Arrays.asList("workspace", "test_table")).mode("append");
    InsertIntoTableResponse insertResp = namespaceClient.insertIntoTable(insertReq, newData);
    assertNotNull(insertResp);

    // Verify row count increased
    CountTableRowsRequest countReq =
        new CountTableRowsRequest().id(Arrays.asList("workspace", "test_table"));
    long count = namespaceClient.countTableRows(countReq).getCount();
    assertEquals(6, count);
  }

  @Test
  void testQueryTable() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Create a table with 3 rows
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespaceClient.createTable(createReq, tableData);

    // Query table - just verify we can execute a simple query
    QueryTableRequest queryReq =
        new QueryTableRequest()
            .id(Arrays.asList("workspace", "test_table"))
            .k(10)
            .vector(new QueryTableRequestVector());
    byte[] resultBytes = namespaceClient.queryTable(queryReq).getData();
    assertNotNull(resultBytes);
    assertTrue(resultBytes.length > 0);
  }

  @Test
  void testListTableVersions() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Create a table
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespaceClient.createTable(createReq, tableData);

    // List versions
    ListTableVersionsRequest listReq =
        new ListTableVersionsRequest().id(Arrays.asList("workspace", "test_table"));
    ListTableVersionsResponse listResp = namespaceClient.listTableVersions(listReq);
    assertNotNull(listResp);
    assertNotNull(listResp.getVersions());
    assertTrue(listResp.getVersions().size() >= 1);
  }

  @Test
  void testRegisterTable() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // First create a table directly to get a valid location
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "source_table"));
    CreateTableResponse createResp = namespaceClient.createTable(createReq, tableData);
    String location = createResp.getLocation();

    // Register the table under a new name (using relative path)
    String relativePath = "source_table.lance";
    RegisterTableRequest registerReq =
        new RegisterTableRequest()
            .id(Arrays.asList("workspace", "registered_table"))
            .location(relativePath);
    RegisterTableResponse registerResp = namespaceClient.registerTable(registerReq);
    assertNotNull(registerResp);

    // Verify the registered table exists
    TableExistsRequest existsReq =
        new TableExistsRequest().id(Arrays.asList("workspace", "registered_table"));
    assertDoesNotThrow(() -> namespaceClient.tableExists(existsReq));
  }

  @Test
  void testDeclareTable() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Declare a table
    DeclareTableRequest declareReq =
        new DeclareTableRequest().id(Arrays.asList("workspace", "declared_table"));
    DeclareTableResponse declareResp = namespaceClient.declareTable(declareReq);
    assertNotNull(declareResp);
    assertNotNull(declareResp.getLocation());

    DescribeTableRequest descReq =
        new DescribeTableRequest().id(Arrays.asList("workspace", "declared_table"));
    DescribeTableResponse descResp = namespaceClient.describeTable(descReq);
    assertNull(descResp.getIsOnlyDeclared());

    DescribeTableRequest checkReq =
        new DescribeTableRequest()
            .id(Arrays.asList("workspace", "declared_table"))
            .checkDeclared(true);
    DescribeTableResponse checkResp = namespaceClient.describeTable(checkReq);
    assertEquals(Boolean.TRUE, checkResp.getIsOnlyDeclared());
  }

  @Test
  void testDeregisterTable() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Create a table
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespaceClient.createTable(createReq, tableData);

    // Deregister the table
    DeregisterTableRequest deregisterReq =
        new DeregisterTableRequest().id(Arrays.asList("workspace", "test_table"));
    DeregisterTableResponse deregisterResp = namespaceClient.deregisterTable(deregisterReq);
    assertNotNull(deregisterResp);

    // Verify the table no longer exists
    TableExistsRequest existsReq =
        new TableExistsRequest().id(Arrays.asList("workspace", "test_table"));
    assertThrows(LanceNamespaceException.class, () -> namespaceClient.tableExists(existsReq));
  }

  @Test
  void testListTableIndices() throws Exception {
    // Create parent namespace
    CreateNamespaceRequest createNsReq =
        new CreateNamespaceRequest().id(Arrays.asList("workspace"));
    namespaceClient.createNamespace(createNsReq);

    // Create a table
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq =
        new CreateTableRequest().id(Arrays.asList("workspace", "test_table"));
    namespaceClient.createTable(createReq, tableData);

    // List indices (should be empty initially)
    ListTableIndicesRequest listReq =
        new ListTableIndicesRequest().id(Arrays.asList("workspace", "test_table"));
    ListTableIndicesResponse listResp = namespaceClient.listTableIndices(listReq);
    assertNotNull(listResp);
    assertNotNull(listResp.getIndexes());
    assertEquals(0, listResp.getIndexes().size());
  }

  @Test
  void testCreateScalarIndex() throws Exception {
    // Create table at root level
    byte[] tableData = createTestTableData();
    CreateTableRequest createReq = new CreateTableRequest().id(Arrays.asList("test_table"));
    namespaceClient.createTable(createReq, tableData);

    // Create scalar index on 'id' column
    CreateTableIndexRequest createIndexReq =
        new CreateTableIndexRequest()
            .id(Arrays.asList("test_table"))
            .column("id")
            .indexType("BTREE")
            .name("id_idx");
    CreateTableIndexResponse response = namespaceClient.createTableIndex(createIndexReq);
    assertNotNull(response);

    // List indices to verify
    ListTableIndicesRequest listReq = new ListTableIndicesRequest().id(Arrays.asList("test_table"));
    ListTableIndicesResponse listResp = namespaceClient.listTableIndices(listReq);
    assertNotNull(listResp);
    assertNotNull(listResp.getIndexes());
    assertEquals(1, listResp.getIndexes().size());
    assertEquals("id_idx", listResp.getIndexes().get(0).getIndexName());
    assertTrue(listResp.getIndexes().get(0).getColumns().contains("id"));
  }

  @Test
  void testCreateVectorIndex() throws Exception {
    // Create table with vector data (256 rows with 8-dim vectors)
    byte[] tableData = createVectorTableData(256, 8);
    CreateTableRequest createReq = new CreateTableRequest().id(Arrays.asList("vector_table"));
    namespaceClient.createTable(createReq, tableData);

    // Create vector index using IVF_FLAT
    CreateTableIndexRequest createIndexReq =
        new CreateTableIndexRequest()
            .id(Arrays.asList("vector_table"))
            .column("vector")
            .indexType("IVF_FLAT")
            .name("vector_idx")
            .distanceType("l2");
    CreateTableIndexResponse response = namespaceClient.createTableIndex(createIndexReq);
    assertNotNull(response);

    // List indices to verify
    ListTableIndicesRequest listReq =
        new ListTableIndicesRequest().id(Arrays.asList("vector_table"));
    ListTableIndicesResponse listResp = namespaceClient.listTableIndices(listReq);
    assertNotNull(listResp);
    assertNotNull(listResp.getIndexes());
    assertEquals(1, listResp.getIndexes().size());
    assertEquals("vector_idx", listResp.getIndexes().get(0).getIndexName());
    assertTrue(listResp.getIndexes().get(0).getColumns().contains("vector"));
  }

  private byte[] createVectorTableData(int numRows, int dim) throws Exception {
    Schema schema =
        new Schema(
            Arrays.asList(
                new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null),
                new Field(
                    "vector",
                    FieldType.nullable(new ArrowType.FixedSizeList(dim)),
                    Arrays.asList(
                        new Field(
                            "item",
                            FieldType.nullable(
                                new ArrowType.FloatingPoint(
                                    org.apache.arrow.vector.types.FloatingPointPrecision.SINGLE)),
                            null)))));

    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      IntVector idVector = (IntVector) root.getVector("id");
      org.apache.arrow.vector.complex.FixedSizeListVector vectorCol =
          (org.apache.arrow.vector.complex.FixedSizeListVector) root.getVector("vector");
      org.apache.arrow.vector.Float4Vector itemsVector =
          (org.apache.arrow.vector.Float4Vector) vectorCol.getDataVector();

      idVector.allocateNew(numRows);
      vectorCol.allocateNew();
      itemsVector.allocateNew(numRows * dim);

      for (int i = 0; i < numRows; i++) {
        idVector.set(i, i);
        for (int j = 0; j < dim; j++) {
          itemsVector.set(i * dim + j, (float) (i * dim + j) * 0.01f);
        }
        vectorCol.setNotNull(i);
      }

      idVector.setValueCount(numRows);
      itemsVector.setValueCount(numRows * dim);
      vectorCol.setValueCount(numRows);
      root.setRowCount(numRows);

      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
        writer.writeBatch();
      }
      return out.toByteArray();
    }
  }

  private static class VersionRejectingNamespace extends CustomNamespace {
    private final AtomicInteger describeTableCallCount = new AtomicInteger();

    VersionRejectingNamespace(DirectoryNamespace inner) {
      super(inner);
    }

    @Override
    public DescribeTableResponse describeTable(DescribeTableRequest request) {
      describeTableCallCount.incrementAndGet();
      assertNull(
          request.getVersion(),
          "Dataset version should be passed to dataset open, not describeTable");
      return super.describeTable(request);
    }

    int getDescribeTableCallCount() {
      return describeTableCallCount.get();
    }
  }
}
