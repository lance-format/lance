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
package org.lance;

import org.lance.namespace.DirectoryNamespace;
import org.lance.namespace.LanceNamespace;
import org.lance.namespace.errors.LanceNamespaceException;
import org.lance.namespace.model.CreateNamespaceRequest;
import org.lance.namespace.model.CreateTableRequest;
import org.lance.namespace.model.CreateTableResponse;
import org.lance.namespace.model.DeclareTableRequest;
import org.lance.namespace.model.DeclareTableResponse;
import org.lance.namespace.model.DropTableRequest;
import org.lance.namespace.model.DropTableResponse;
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
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.CreateBucketRequest;
import software.amazon.awssdk.services.s3.model.DeleteBucketRequest;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Request;
import software.amazon.awssdk.services.s3.model.S3Object;

import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Integration tests for Lance with S3 and credential refresh using StorageOptionsProvider.
 *
 * <p>This test uses DirectoryNamespace with native ops_metrics and vend_input_storage_options
 * features to track API calls and test credential refresh mechanisms.
 *
 * <p>These tests require LocalStack to be running. Run with: docker compose up -d
 *
 * <p>Set LANCE_INTEGRATION_TEST=1 environment variable to enable these tests.
 */
@EnabledIfEnvironmentVariable(named = "LANCE_INTEGRATION_TEST", matches = "1")
public class DirectoryNamespaceIntegrationTest {

  private static final String ENDPOINT_URL = "http://localhost:4566";
  private static final String REGION = "us-east-1";
  private static final String ACCESS_KEY = "ACCESS_KEY";
  private static final String SECRET_KEY = "SECRET_KEY";
  private static final String BUCKET_NAME = "lance-namespace-integtest-java";

  private static S3Client s3Client;
  private BufferAllocator testAllocator;
  private String testPrefix;

  @BeforeEach
  void setUpTest() {
    testAllocator = new RootAllocator(Long.MAX_VALUE);
    testPrefix = "test-" + UUID.randomUUID().toString().substring(0, 8);
  }

  @AfterEach
  void tearDownTest() {
    if (testAllocator != null) {
      testAllocator.close();
    }
  }

  @BeforeAll
  static void setup() {
    s3Client =
        S3Client.builder()
            .endpointOverride(URI.create(ENDPOINT_URL))
            .region(Region.of(REGION))
            .credentialsProvider(
                StaticCredentialsProvider.create(
                    AwsBasicCredentials.create(ACCESS_KEY, SECRET_KEY)))
            .forcePathStyle(true) // Required for LocalStack
            .build();

    // Delete bucket if it exists from previous run
    try {
      deleteBucket();
    } catch (Exception e) {
      // Ignore if bucket doesn't exist
    }

    // Create test bucket
    s3Client.createBucket(CreateBucketRequest.builder().bucket(BUCKET_NAME).build());
  }

  @AfterAll
  static void tearDown() {
    if (s3Client != null) {
      try {
        deleteBucket();
      } catch (Exception e) {
        // Ignore cleanup errors
      }
      s3Client.close();
    }
  }

  private static void deleteBucket() {
    // Delete all objects first
    List<S3Object> objects =
        s3Client
            .listObjectsV2(ListObjectsV2Request.builder().bucket(BUCKET_NAME).build())
            .contents();
    for (S3Object obj : objects) {
      s3Client.deleteObject(
          DeleteObjectRequest.builder().bucket(BUCKET_NAME).key(obj.key()).build());
    }
    s3Client.deleteBucket(DeleteBucketRequest.builder().bucket(BUCKET_NAME).build());
  }

  /**
   * Result holder for namespace creation that includes both the namespace to use for operations and
   * the inner DirectoryNamespace for metrics retrieval.
   */
  protected static class TrackingNamespaceResult {
    /** The namespace client to use for operations. May be wrapped. */
    public final LanceNamespace namespaceClient;

    /** The inner DirectoryNamespace for metrics retrieval. */
    public final DirectoryNamespace innerNamespaceClient;

    public TrackingNamespaceResult(
        LanceNamespace namespaceClient, DirectoryNamespace innerNamespaceClient) {
      this.namespaceClient = namespaceClient;
      this.innerNamespaceClient = innerNamespaceClient;
    }
  }

  /**
   * Creates a DirectoryNamespace configured for testing with ops metrics and credential vending.
   *
   * <p>Uses native DirectoryNamespace features:
   *
   * <ul>
   *   <li>ops_metrics_enabled=true: Tracks API call counts via retrieveOpsMetrics()
   *   <li>vend_input_storage_options=true: Returns input storage options in responses
   *   <li>vend_input_storage_options_refresh_interval_millis: Adds expires_at_millis
   * </ul>
   *
   * @param bucketName S3 bucket name or local path
   * @param storageOptions Storage options to pass through (credentials, endpoint, etc.)
   * @param credentialExpiresInSeconds Interval in seconds for credential expiration
   * @return TrackingNamespaceResult containing the namespace and inner DirectoryNamespace
   */
  protected TrackingNamespaceResult createTrackingNamespace(
      String bucketName, Map<String, String> storageOptions, int credentialExpiresInSeconds) {
    Map<String, String> dirProps = new HashMap<>();

    // Add refresh_offset_millis to storage options so that credentials are not
    // considered expired immediately. Set to 1 second (1000ms) so that refresh
    // checks work correctly with short-lived credentials in tests.
    Map<String, String> storageOptionsWithRefresh = new HashMap<>(storageOptions);
    storageOptionsWithRefresh.put("refresh_offset_millis", "1000");

    for (Map.Entry<String, String> entry : storageOptionsWithRefresh.entrySet()) {
      dirProps.put("storage." + entry.getKey(), entry.getValue());
    }

    // Set root based on bucket type
    if (bucketName.startsWith("/") || bucketName.startsWith("file://")) {
      dirProps.put("root", bucketName + "/namespace_root");
    } else {
      dirProps.put("root", "s3://" + bucketName + "/namespace_root");
    }

    // Enable ops metrics tracking
    dirProps.put("ops_metrics_enabled", "true");
    // Enable storage options vending
    dirProps.put("vend_input_storage_options", "true");
    // Set refresh interval in milliseconds
    dirProps.put(
        "vend_input_storage_options_refresh_interval_millis",
        String.valueOf(credentialExpiresInSeconds * 1000L));

    DirectoryNamespace innerNamespaceClient = new DirectoryNamespace();
    try (BufferAllocator allocator = new RootAllocator()) {
      innerNamespaceClient.initialize(dirProps, allocator);
    }
    LanceNamespace namespaceClient = wrapNamespace(innerNamespaceClient);
    return new TrackingNamespaceResult(namespaceClient, innerNamespaceClient);
  }

  /**
   * Factory method to wrap the DirectoryNamespace. Subclasses can override this to provide a custom
   * namespace implementation.
   *
   * @param inner The DirectoryNamespace to wrap
   * @return The namespace to use in tests (may be the same as inner or a wrapper)
   */
  protected LanceNamespace wrapNamespace(DirectoryNamespace inner) {
    return inner;
  }

  /**
   * Gets the number of describe_table calls made to the namespace.
   *
   * @param namespaceClient The DirectoryNamespace to check
   * @return Number of describe_table calls
   */
  protected static int getDescribeCallCount(DirectoryNamespace namespaceClient) {
    Map<String, Long> metrics = namespaceClient.retrieveOpsMetrics();
    return metrics.getOrDefault("describe_table", 0L).intValue();
  }

  /**
   * Gets the number of declare_table calls made to the namespace.
   *
   * @param namespaceClient The DirectoryNamespace to check
   * @return Number of declare_table calls
   */
  protected static int getDeclareCallCount(DirectoryNamespace namespaceClient) {
    Map<String, Long> metrics = namespaceClient.retrieveOpsMetrics();
    return metrics.getOrDefault("declare_table", 0L).intValue();
  }

  @Test
  void testOpenDatasetWithoutRefresh() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace with 60-second expiration (long enough to not expire during test)
      TrackingNamespaceResult nsResult = createTrackingNamespace(BUCKET_NAME, storageOptions, 60);
      LanceNamespace namespaceClient = nsResult.namespaceClient;
      DirectoryNamespace innerNamespaceClient = nsResult.innerNamespaceClient;
      String tableName = UUID.randomUUID().toString();

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

        // Create a test reader that returns our VectorSchemaRoot
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

        // Create dataset through namespace client
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(testReader)
                .namespaceClient(namespaceClient)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.CREATE)
                .storageOptions(storageOptions)
                .execute()) {
          assertEquals(2, dataset.countRows());
        }
      }

      // Verify declareTable was called
      assertEquals(
          1, getDeclareCallCount(innerNamespaceClient), "declareTable should be called once");

      // Open dataset through namespace client WITH refresh enabled
      ReadOptions readOptions = new ReadOptions.Builder().setStorageOptions(storageOptions).build();

      int callCountBeforeOpen = getDescribeCallCount(innerNamespaceClient);
      try (Dataset dsFromNamespace =
          Dataset.open()
              .allocator(allocator)
              .namespaceClient(namespaceClient)
              .tableId(Arrays.asList(tableName))
              .readOptions(readOptions)
              .build()) {
        // With the fix, describeTable should only be called once during open
        // to get the table location and initial storage options
        int callCountAfterOpen = getDescribeCallCount(innerNamespaceClient);
        assertEquals(
            1,
            callCountAfterOpen - callCountBeforeOpen,
            "describeTable should be called exactly once during open, got: "
                + (callCountAfterOpen - callCountBeforeOpen));

        // Verify we can read the data multiple times
        assertEquals(2, dsFromNamespace.countRows());
        assertEquals(2, dsFromNamespace.countRows());
        assertEquals(2, dsFromNamespace.countRows());

        // Perform operations that access S3
        List<Fragment> fragments = dsFromNamespace.getFragments();
        assertEquals(1, fragments.size());
        List<Version> versions = dsFromNamespace.listVersions();
        assertEquals(1, versions.size());

        // With the fix, credentials are cached so no additional calls are made
        int finalCallCount = getDescribeCallCount(innerNamespaceClient);
        int totalCalls = finalCallCount - callCountBeforeOpen;
        assertEquals(
            1,
            totalCalls,
            "describeTable should only be called once total (credentials are cached), got: "
                + totalCalls);
      }
    }
  }

  @Test
  void testStorageOptionsProviderWithRefresh() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace with 5-second expiration for faster testing
      TrackingNamespaceResult nsResult = createTrackingNamespace(BUCKET_NAME, storageOptions, 5);
      LanceNamespace namespaceClient = nsResult.namespaceClient;
      DirectoryNamespace innerNamespaceClient = nsResult.innerNamespaceClient;
      String tableName = UUID.randomUUID().toString();

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

        // Create a test reader that returns our VectorSchemaRoot
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

        // Create dataset through namespace client with refresh enabled
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(testReader)
                .namespaceClient(namespaceClient)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.CREATE)
                .storageOptions(storageOptions)
                .execute()) {
          assertEquals(2, dataset.countRows());
        }
      }

      // Verify declareTable was called
      assertEquals(
          1, getDeclareCallCount(innerNamespaceClient), "declareTable should be called once");

      // Open dataset through namespace client with refresh enabled
      ReadOptions readOptions = new ReadOptions.Builder().setStorageOptions(storageOptions).build();

      int callCountBeforeOpen = getDescribeCallCount(innerNamespaceClient);
      try (Dataset dsFromNamespace =
          Dataset.open()
              .allocator(allocator)
              .namespaceClient(namespaceClient)
              .tableId(Arrays.asList(tableName))
              .readOptions(readOptions)
              .build()) {
        // With the fix, describeTable should only be called once during open
        int callCountAfterOpen = getDescribeCallCount(innerNamespaceClient);
        assertEquals(
            1,
            callCountAfterOpen - callCountBeforeOpen,
            "describeTable should be called exactly once during open, got: "
                + (callCountAfterOpen - callCountBeforeOpen));

        // Verify we can read the data
        assertEquals(2, dsFromNamespace.countRows());

        // Record call count after initial reads
        int callCountAfterInitialReads = getDescribeCallCount(innerNamespaceClient);
        int callsAfterFirstRead = callCountAfterInitialReads - callCountBeforeOpen;
        assertEquals(
            1,
            callsAfterFirstRead,
            "describeTable should still be 1 (credentials are cached), got: "
                + callsAfterFirstRead);

        // Wait for credentials to be close to expiring (4 seconds - past the 3s refresh threshold)
        Thread.sleep(4000);

        // Perform read operations after expiration
        // Access fragments and versions which require S3 access and trigger credential refresh
        assertEquals(2, dsFromNamespace.countRows());
        List<Fragment> fragments = dsFromNamespace.getFragments();
        assertEquals(1, fragments.size());
        List<Version> versions = dsFromNamespace.listVersions();
        assertEquals(1, versions.size());

        int finalCallCount = getDescribeCallCount(innerNamespaceClient);
        int totalCallsAfterExpiration = finalCallCount - callCountBeforeOpen;
        assertEquals(
            2,
            totalCallsAfterExpiration,
            "Credentials should be refreshed once after expiration. "
                + "Expected 2 total calls (1 initial + 1 refresh), got: "
                + totalCallsAfterExpiration);
      }
    }
  }

  @Test
  void testWriteDatasetBuilderWithNamespaceCreate() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace
      TrackingNamespaceResult nsResult = createTrackingNamespace(BUCKET_NAME, storageOptions, 60);
      LanceNamespace namespaceClient = nsResult.namespaceClient;
      DirectoryNamespace innerNamespaceClient = nsResult.innerNamespaceClient;
      String tableName = UUID.randomUUID().toString();

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

        // Create a test reader that returns our VectorSchemaRoot
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

        int callCountBefore = getDeclareCallCount(innerNamespaceClient);

        // Use the write builder to create a dataset through namespace client
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(testReader)
                .namespaceClient(namespaceClient)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.CREATE)
                .storageOptions(storageOptions)
                .execute()) {

          // Verify declareTable was called
          int callCountAfter = getDeclareCallCount(innerNamespaceClient);
          assertEquals(1, callCountAfter - callCountBefore, "declareTable should be called once");

          // Verify dataset was created successfully
          assertEquals(2, dataset.countRows());
          assertEquals(schema, dataset.getSchema());
        }
      }
    }
  }

  @Test
  void testWriteDatasetBuilderWithNamespaceCreateCallCounts() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace with 60-second expiration (long enough that no refresh happens)
      // Credentials expire at T+60s. With a 1s refresh offset, refresh would happen at T+59s.
      // Since writes complete well under 59 seconds, NO credential refresh should occur.
      TrackingNamespaceResult nsResult = createTrackingNamespace(BUCKET_NAME, storageOptions, 60);
      LanceNamespace namespaceClient = nsResult.namespaceClient;
      DirectoryNamespace innerNamespaceClient = nsResult.innerNamespaceClient;
      String tableName = UUID.randomUUID().toString();

      // Verify initial call counts
      assertEquals(
          0, getDeclareCallCount(innerNamespaceClient), "declareTable should not be called yet");
      assertEquals(
          0, getDescribeCallCount(innerNamespaceClient), "describeTable should not be called yet");

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

        // Create a test reader that returns our VectorSchemaRoot
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

        // Use the write builder to create a dataset through namespace client
        // Write completes instantly, so NO describeTable call should happen for refresh.
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(testReader)
                .namespaceClient(namespaceClient)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.CREATE)
                .storageOptions(storageOptions)
                .execute()) {

          // Verify declareTable was called exactly ONCE
          assertEquals(
              1,
              getDeclareCallCount(innerNamespaceClient),
              "declareTable should be called exactly once");

          // Verify describeTable was NOT called during CREATE
          // Initial credentials come from declareTable response, and since credentials
          // don't expire during the fast write, NO refresh (describeTable) is needed
          assertEquals(
              0,
              getDescribeCallCount(innerNamespaceClient),
              "describeTable should NOT be called during CREATE - "
                  + "initial credentials come from declareTable response and don't expire");

          // Verify dataset was created successfully
          assertEquals(2, dataset.countRows());
          assertEquals(schema, dataset.getSchema());
        }
      }

      // Verify counts after dataset is closed
      assertEquals(
          1,
          getDeclareCallCount(innerNamespaceClient),
          "declareTable should still be 1 after close");
      assertEquals(
          0,
          getDescribeCallCount(innerNamespaceClient),
          "describeTable should still be 0 after close (no refresh needed)");

      // Now open the dataset through namespace client with long-lived credentials (60s expiration)
      ReadOptions readOptions = new ReadOptions.Builder().setStorageOptions(storageOptions).build();

      try (Dataset dsFromNamespace =
          Dataset.open()
              .allocator(allocator)
              .namespaceClient(namespaceClient)
              .tableId(Arrays.asList(tableName))
              .readOptions(readOptions)
              .build()) {

        // declareTable should NOT be called during open (only during CREATE)
        assertEquals(
            1,
            getDeclareCallCount(innerNamespaceClient),
            "declareTable should still be 1 (not called during open)");

        // describeTable is called exactly ONCE during open to get table location
        assertEquals(
            1,
            getDescribeCallCount(innerNamespaceClient),
            "describeTable should be called exactly once during open");

        // Verify we can read the data multiple times
        assertEquals(2, dsFromNamespace.countRows());
        assertEquals(2, dsFromNamespace.countRows());
        assertEquals(2, dsFromNamespace.countRows());

        // After multiple reads, no additional describeTable calls should be made
        // (credentials are cached and don't expire during this fast test)
        assertEquals(
            1,
            getDescribeCallCount(innerNamespaceClient),
            "describeTable should still be 1 after reads (credentials cached, no refresh needed)");
      }

      // Final verification
      assertEquals(1, getDeclareCallCount(innerNamespaceClient), "Final: declareTable = 1");
      assertEquals(1, getDescribeCallCount(innerNamespaceClient), "Final: describeTable = 1");
    }
  }

  @Test
  void testWriteDatasetBuilderWithNamespaceAppend() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace
      TrackingNamespaceResult nsResult = createTrackingNamespace(BUCKET_NAME, storageOptions, 60);
      LanceNamespace namespaceClient = nsResult.namespaceClient;
      DirectoryNamespace innerNamespaceClient = nsResult.innerNamespaceClient;
      String tableName = UUID.randomUUID().toString();

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

        // Create a test reader that returns our VectorSchemaRoot
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

        // Create initial dataset through namespace client
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(testReader)
                .namespaceClient(namespaceClient)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.CREATE)
                .storageOptions(storageOptions)
                .execute()) {
          assertEquals(2, dataset.countRows());
        }

        assertEquals(
            1, getDeclareCallCount(innerNamespaceClient), "declareTable should be called once");
        int initialDescribeCount = getDescribeCallCount(innerNamespaceClient);

        // Now append data using the write builder with namespace client
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
                return root;
              }
            };

        // Use the write builder to append to dataset through namespace client
        try (Dataset dataset =
            Dataset.write()
                .allocator(allocator)
                .reader(appendReader)
                .namespaceClient(namespaceClient)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.APPEND)
                .storageOptions(storageOptions)
                .execute()) {

          // Verify describeTable was called
          int callCountAfter = getDescribeCallCount(innerNamespaceClient);
          assertEquals(
              1,
              callCountAfter - initialDescribeCount,
              "describeTable should be called once for append");

          // Verify data was appended successfully
          assertEquals(4, dataset.countRows()); // Original 2 + appended 2
        }
      }
    }
  }

  @Test
  void testWriteDatasetBuilderWithNamespaceOverwrite() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace
      TrackingNamespaceResult nsResult = createTrackingNamespace(BUCKET_NAME, storageOptions, 60);
      LanceNamespace namespaceClient = nsResult.namespaceClient;
      DirectoryNamespace innerNamespaceClient = nsResult.innerNamespaceClient;
      String tableName = UUID.randomUUID().toString();

      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("a", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("b", FieldType.nullable(new ArrowType.Int(32, true)), null)));

      // Create initial dataset with 1 row
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(1);
        bVector.allocateNew(1);

        aVector.set(0, 1);
        bVector.set(0, 2);

        aVector.setValueCount(1);
        bVector.setValueCount(1);
        root.setRowCount(1);

        ArrowReader createReader =
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
                .reader(createReader)
                .namespaceClient(namespaceClient)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.CREATE)
                .storageOptions(storageOptions)
                .execute()) {
          assertEquals(1, dataset.countRows());
        }

        assertEquals(
            1, getDeclareCallCount(innerNamespaceClient), "declareTable should be called once");
        assertEquals(
            0,
            getDescribeCallCount(innerNamespaceClient),
            "describeTable should not be called yet");

        // Now overwrite with 2 rows
        aVector.allocateNew(2);
        bVector.allocateNew(2);

        aVector.set(0, 10);
        bVector.set(0, 20);
        aVector.set(1, 100);
        bVector.set(1, 200);

        aVector.setValueCount(2);
        bVector.setValueCount(2);
        root.setRowCount(2);

        ArrowReader overwriteReader =
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
                .reader(overwriteReader)
                .namespaceClient(namespaceClient)
                .tableId(Arrays.asList(tableName))
                .mode(WriteParams.WriteMode.OVERWRITE)
                .storageOptions(storageOptions)
                .execute()) {

          // Verify describeTable was called for overwrite
          assertEquals(
              1, getDeclareCallCount(innerNamespaceClient), "declareTable should still be 1");
          int describeCountAfterOverwrite = getDescribeCallCount(innerNamespaceClient);
          assertEquals(
              1, describeCountAfterOverwrite, "describeTable should be called once for overwrite");

          // Verify data was overwritten successfully
          assertEquals(2, dataset.countRows());
          assertEquals(
              2, dataset.listVersions().size()); // Version 1 (create) + Version 2 (overwrite)
        }

        // Verify we can open and read the dataset through namespace client
        try (Dataset ds =
            Dataset.open()
                .allocator(allocator)
                .namespaceClient(namespaceClient)
                .tableId(Arrays.asList(tableName))
                .readOptions(new ReadOptions.Builder().setStorageOptions(storageOptions).build())
                .build()) {
          assertEquals(2, ds.countRows(), "Should have 2 rows after overwrite");
          assertEquals(2, ds.listVersions().size(), "Should have 2 versions");
        }
      }
    }
  }

  @Test
  void testDistributedWriteWithNamespace() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace
      TrackingNamespaceResult nsResult = createTrackingNamespace(BUCKET_NAME, storageOptions, 60);
      LanceNamespace namespaceClient = nsResult.namespaceClient;
      DirectoryNamespace innerNamespaceClient = nsResult.innerNamespaceClient;
      String tableName = UUID.randomUUID().toString();

      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("a", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("b", FieldType.nullable(new ArrowType.Int(32, true)), null)));

      // Step 1: Declare table via namespace
      DeclareTableRequest request = new DeclareTableRequest();
      request.setId(Arrays.asList(tableName));
      DeclareTableResponse response = namespaceClient.declareTable(request);

      assertEquals(
          1, getDeclareCallCount(innerNamespaceClient), "declareTable should be called once");
      assertEquals(
          0, getDescribeCallCount(innerNamespaceClient), "describeTable should not be called yet");

      String tableUri = response.getLocation();
      Map<String, String> namespaceStorageOptions = response.getStorageOptions();

      // Merge storage options
      Map<String, String> mergedOptions = new HashMap<>(storageOptions);
      if (namespaceStorageOptions != null) {
        mergedOptions.putAll(namespaceStorageOptions);
      }

      WriteParams writeParams = new WriteParams.Builder().withStorageOptions(mergedOptions).build();
      List<String> tableId = Arrays.asList(tableName);

      // Step 2: Write multiple fragments in parallel (simulated)
      List<FragmentMetadata> allFragments = new ArrayList<>();

      // Fragment 1: 2 rows
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(2);
        bVector.allocateNew(2);
        aVector.set(0, 1);
        bVector.set(0, 2);
        aVector.set(1, 3);
        bVector.set(1, 4);
        aVector.setValueCount(2);
        bVector.setValueCount(2);
        root.setRowCount(2);

        List<FragmentMetadata> fragment1 =
            Fragment.create(tableUri, allocator, root, writeParams, namespaceClient, tableId);
        allFragments.addAll(fragment1);
      }

      // Fragment 2: 2 rows
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(2);
        bVector.allocateNew(2);
        aVector.set(0, 10);
        bVector.set(0, 20);
        aVector.set(1, 30);
        bVector.set(1, 40);
        aVector.setValueCount(2);
        bVector.setValueCount(2);
        root.setRowCount(2);

        List<FragmentMetadata> fragment2 =
            Fragment.create(tableUri, allocator, root, writeParams, namespaceClient, tableId);
        allFragments.addAll(fragment2);
      }

      // Fragment 3: 1 row
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector aVector = (IntVector) root.getVector("a");
        IntVector bVector = (IntVector) root.getVector("b");

        aVector.allocateNew(1);
        bVector.allocateNew(1);
        aVector.set(0, 100);
        bVector.set(0, 200);
        aVector.setValueCount(1);
        bVector.setValueCount(1);
        root.setRowCount(1);

        List<FragmentMetadata> fragment3 =
            Fragment.create(tableUri, allocator, root, writeParams, namespaceClient, tableId);
        allFragments.addAll(fragment3);
      }

      // Step 3: Commit all fragments as one operation
      FragmentOperation.Overwrite overwriteOp =
          new FragmentOperation.Overwrite(allFragments, schema);

      try (Dataset dataset =
          Dataset.commit(allocator, tableUri, overwriteOp, Optional.empty(), mergedOptions)) {
        assertEquals(5, dataset.countRows(), "Should have 5 total rows from all fragments");
        assertEquals(1, dataset.listVersions().size(), "Should have 1 version after commit");
      }

      // Step 4: Open dataset through namespace client and verify
      try (Dataset dsFromNamespace =
          Dataset.open()
              .allocator(allocator)
              .namespaceClient(namespaceClient)
              .tableId(Arrays.asList(tableName))
              .readOptions(new ReadOptions.Builder().setStorageOptions(storageOptions).build())
              .build()) {
        assertEquals(5, dsFromNamespace.countRows(), "Should read 5 rows through namespace client");
      }
    }
  }

  @Test
  void testFragmentCreateAndCommitWithNamespace() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace with 60-second expiration
      TrackingNamespaceResult nsResult = createTrackingNamespace(BUCKET_NAME, storageOptions, 60);
      LanceNamespace namespaceClient = nsResult.namespaceClient;
      DirectoryNamespace innerNamespaceClient = nsResult.innerNamespaceClient;
      String tableName = UUID.randomUUID().toString();

      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("value", FieldType.nullable(new ArrowType.Int(32, true)), null)));

      // Declare table via namespace
      DeclareTableRequest request = new DeclareTableRequest();
      request.setId(Arrays.asList(tableName));
      DeclareTableResponse response = namespaceClient.declareTable(request);

      assertEquals(
          1, getDeclareCallCount(innerNamespaceClient), "declareTable should be called once");

      String tableUri = response.getLocation();
      Map<String, String> namespaceStorageOptions = response.getStorageOptions();

      // Merge storage options
      Map<String, String> mergedOptions = new HashMap<>(storageOptions);
      if (namespaceStorageOptions != null) {
        mergedOptions.putAll(namespaceStorageOptions);
      }

      WriteParams writeParams = new WriteParams.Builder().withStorageOptions(mergedOptions).build();
      List<String> tableId = Arrays.asList(tableName);

      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector idVector = (IntVector) root.getVector("id");
        IntVector valueVector = (IntVector) root.getVector("value");

        // Write first fragment
        idVector.allocateNew(3);
        valueVector.allocateNew(3);

        idVector.set(0, 1);
        valueVector.set(0, 100);
        idVector.set(1, 2);
        valueVector.set(1, 200);
        idVector.set(2, 3);
        valueVector.set(2, 300);

        idVector.setValueCount(3);
        valueVector.setValueCount(3);
        root.setRowCount(3);

        // Create fragment with namespace client
        List<FragmentMetadata> fragments1 =
            Fragment.create(tableUri, allocator, root, writeParams, namespaceClient, tableId);

        assertEquals(1, fragments1.size());

        // Write second fragment with different data
        idVector.set(0, 4);
        valueVector.set(0, 400);
        idVector.set(1, 5);
        valueVector.set(1, 500);
        idVector.set(2, 6);
        valueVector.set(2, 600);
        root.setRowCount(3);

        // Create another fragment with the same namespace client
        List<FragmentMetadata> fragments2 =
            Fragment.create(tableUri, allocator, root, writeParams, namespaceClient, tableId);

        assertEquals(1, fragments2.size());

        // Commit first fragment to the dataset using Overwrite (for empty table)
        FragmentOperation.Overwrite overwriteOp =
            new FragmentOperation.Overwrite(fragments1, schema);
        try (Dataset updatedDataset =
            Dataset.commit(allocator, tableUri, overwriteOp, Optional.empty(), mergedOptions)) {
          assertEquals(1, updatedDataset.version());
          assertEquals(3, updatedDataset.countRows());

          // Append second fragment
          FragmentOperation.Append appendOp2 = new FragmentOperation.Append(fragments2);
          try (Dataset finalDataset =
              Dataset.commit(allocator, tableUri, appendOp2, Optional.of(1L), mergedOptions)) {
            assertEquals(2, finalDataset.version());
            assertEquals(6, finalDataset.countRows());
          }
        }
      }

      // Verify we can open and read the dataset through namespace client
      try (Dataset ds =
          Dataset.open()
              .allocator(allocator)
              .namespaceClient(namespaceClient)
              .tableId(Arrays.asList(tableName))
              .readOptions(new ReadOptions.Builder().setStorageOptions(storageOptions).build())
              .build()) {
        assertEquals(6, ds.countRows(), "Should have 6 rows total");
        assertEquals(2, ds.listVersions().size(), "Should have 2 versions");
      }
    }
  }

  @Test
  void testTransactionCommitWithNamespace() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      // Set up storage options
      Map<String, String> storageOptions = new HashMap<>();
      storageOptions.put("allow_http", "true");
      storageOptions.put("aws_access_key_id", ACCESS_KEY);
      storageOptions.put("aws_secret_access_key", SECRET_KEY);
      storageOptions.put("aws_endpoint", ENDPOINT_URL);
      storageOptions.put("aws_region", REGION);

      // Create tracking namespace
      TrackingNamespaceResult nsResult = createTrackingNamespace(BUCKET_NAME, storageOptions, 60);
      LanceNamespace namespaceClient = nsResult.namespaceClient;
      String tableName = UUID.randomUUID().toString();

      Schema schema =
          new Schema(
              Arrays.asList(
                  new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null),
                  new Field("name", FieldType.nullable(new ArrowType.Utf8()), null)));

      // Declare table via namespace
      DeclareTableRequest request = new DeclareTableRequest();
      request.setId(Arrays.asList(tableName));
      DeclareTableResponse response = namespaceClient.declareTable(request);

      String tableUri = response.getLocation();
      Map<String, String> namespaceStorageOptions = response.getStorageOptions();

      // Merge storage options
      Map<String, String> mergedOptions = new HashMap<>(storageOptions);
      if (namespaceStorageOptions != null) {
        mergedOptions.putAll(namespaceStorageOptions);
      }

      // First, write some initial data using Fragment.create and commit
      WriteParams writeParams = new WriteParams.Builder().withStorageOptions(mergedOptions).build();
      List<String> tableId = Arrays.asList(tableName);

      List<FragmentMetadata> initialFragments;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector idVector = (IntVector) root.getVector("id");
        org.apache.arrow.vector.VarCharVector nameVector =
            (org.apache.arrow.vector.VarCharVector) root.getVector("name");

        idVector.allocateNew(2);
        nameVector.allocateNew(2);

        idVector.set(0, 1);
        nameVector.setSafe(0, "Alice".getBytes());
        idVector.set(1, 2);
        nameVector.setSafe(1, "Bob".getBytes());

        idVector.setValueCount(2);
        nameVector.setValueCount(2);
        root.setRowCount(2);

        initialFragments =
            Fragment.create(tableUri, allocator, root, writeParams, namespaceClient, tableId);
      }

      // Commit initial fragments
      FragmentOperation.Overwrite overwriteOp =
          new FragmentOperation.Overwrite(initialFragments, schema);
      try (Dataset dataset =
          Dataset.commit(allocator, tableUri, overwriteOp, Optional.empty(), mergedOptions)) {
        assertEquals(1, dataset.version());
        assertEquals(2, dataset.countRows());
      }

      // Now test Transaction.commit with namespace client
      // Open dataset with namespace client using mergedOptions (which has expires_at_millis)
      ReadOptions readOptions = new ReadOptions.Builder().setStorageOptions(mergedOptions).build();

      try (Dataset datasetWithNamespaceClient =
          Dataset.open()
              .allocator(allocator)
              .namespaceClient(namespaceClient)
              .tableId(tableId)
              .readOptions(readOptions)
              .build()) {
        // Create more fragments to append
        List<FragmentMetadata> newFragments;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
          IntVector idVector = (IntVector) root.getVector("id");
          org.apache.arrow.vector.VarCharVector nameVector =
              (org.apache.arrow.vector.VarCharVector) root.getVector("name");

          idVector.allocateNew(2);
          nameVector.allocateNew(2);

          idVector.set(0, 3);
          nameVector.setSafe(0, "Charlie".getBytes());
          idVector.set(1, 4);
          nameVector.setSafe(1, "Diana".getBytes());

          idVector.setValueCount(2);
          nameVector.setValueCount(2);
          root.setRowCount(2);

          newFragments =
              Fragment.create(tableUri, allocator, root, writeParams, namespaceClient, tableId);
        }

        // Create and commit transaction
        Append appendOp = Append.builder().fragments(newFragments).build();
        try (Transaction transaction =
            new Transaction.Builder()
                .readVersion(datasetWithNamespaceClient.version())
                .operation(appendOp)
                .build()) {
          try (Dataset committedDataset =
              new CommitBuilder(datasetWithNamespaceClient).execute(transaction)) {
            assertEquals(2, committedDataset.version());
            assertEquals(4, committedDataset.countRows());
          }
        }
      }

      // Verify we can open and read the dataset through namespace client
      try (Dataset ds =
          Dataset.open()
              .allocator(allocator)
              .namespaceClient(namespaceClient)
              .tableId(Arrays.asList(tableName))
              .readOptions(new ReadOptions.Builder().setStorageOptions(storageOptions).build())
              .build()) {
        assertEquals(4, ds.countRows(), "Should have 4 rows total");
        assertEquals(2, ds.listVersions().size(), "Should have 2 versions");
      }
    }
  }

  private Map<String, String> createDirectoryNamespaceS3Config() {
    Map<String, String> config = new HashMap<>();
    config.put("root", "s3://" + BUCKET_NAME + "/" + testPrefix);
    config.put("storage.access_key_id", ACCESS_KEY);
    config.put("storage.secret_access_key", SECRET_KEY);
    config.put("storage.endpoint", ENDPOINT_URL);
    config.put("storage.region", REGION);
    config.put("storage.allow_http", "true");
    config.put("storage.virtual_hosted_style_request", "false");
    config.put("inline_optimization_enabled", "false");
    // Very high retry count to guarantee all concurrent operations succeed
    config.put("commit_retries", "2147483647");
    return config;
  }

  private byte[] createTestTableData() throws Exception {
    Schema schema =
        new Schema(
            Arrays.asList(
                new Field("id", FieldType.nullable(new ArrowType.Int(32, true)), null),
                new Field("name", FieldType.nullable(new ArrowType.Utf8()), null),
                new Field("age", FieldType.nullable(new ArrowType.Int(32, true)), null)));

    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, testAllocator)) {
      IntVector idVector = (IntVector) root.getVector("id");
      VarCharVector nameVector = (VarCharVector) root.getVector("name");
      IntVector ageVector = (IntVector) root.getVector("age");

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
  void testBasicCreateAndDropOnS3() throws Exception {
    DirectoryNamespace namespaceClient = new DirectoryNamespace();
    namespaceClient.initialize(createDirectoryNamespaceS3Config(), testAllocator);

    try {
      String tableName = "basic_test_table";
      List<String> tableId = Arrays.asList("test_ns", tableName);
      byte[] tableData = createTestTableData();

      // createTable requires the parent namespace to exist.
      CreateNamespaceRequest createNsReq =
          new CreateNamespaceRequest().id(Arrays.asList("test_ns"));
      namespaceClient.createNamespace(createNsReq);

      CreateTableRequest createReq = new CreateTableRequest().id(tableId);
      CreateTableResponse createResp = namespaceClient.createTable(createReq, tableData);
      assertNotNull(createResp);
      assertNotNull(createResp.getLocation());

      DropTableRequest dropReq = new DropTableRequest().id(tableId);
      DropTableResponse dropResp = namespaceClient.dropTable(dropReq);
      assertNotNull(dropResp);

      TableExistsRequest existsReq = new TableExistsRequest().id(tableId);
      assertThrows(LanceNamespaceException.class, () -> namespaceClient.tableExists(existsReq));
    } finally {
      namespaceClient.close();
    }
  }

  @Test
  void testConcurrentCreateAndDropWithSingleInstanceOnS3() throws Exception {
    DirectoryNamespace namespaceClient = new DirectoryNamespace();
    namespaceClient.initialize(createDirectoryNamespaceS3Config(), testAllocator);

    try {
      // Initialize namespace first - create parent namespace to ensure __manifest table
      // is created before concurrent operations
      CreateNamespaceRequest createNsReq =
          new CreateNamespaceRequest().id(Arrays.asList("test_ns"));
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

                String tableName = "s3_concurrent_table_" + tableIndex;
                List<String> tableId = Arrays.asList("test_ns", tableName);
                byte[] tableData = createTestTableData();

                CreateTableRequest createReq = new CreateTableRequest().id(tableId);
                namespaceClient.createTable(createReq, tableData);

                DropTableRequest dropReq = new DropTableRequest().id(tableId);
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
      assertTrue(doneLatch.await(120, TimeUnit.SECONDS), "Timed out waiting for tasks to complete");

      executor.shutdown();
      assertTrue(executor.awaitTermination(30, TimeUnit.SECONDS));

      assertEquals(numTables, successCount.get(), "All tasks should succeed");
      assertEquals(0, failCount.get(), "No tasks should fail");
    } finally {
      namespaceClient.close();
    }
  }

  @Test
  void testConcurrentCreateAndDropWithMultipleInstancesOnS3() throws Exception {
    Map<String, String> baseConfig = createDirectoryNamespaceS3Config();

    // Initialize namespace first with a single instance to ensure __manifest
    // table is created and parent namespace exists before concurrent operations
    DirectoryNamespace initNs = new DirectoryNamespace();
    initNs.initialize(new HashMap<>(baseConfig), testAllocator);
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
              localNs.initialize(new HashMap<>(baseConfig), testAllocator);

              synchronized (namespaces) {
                namespaces.add(localNs);
              }

              String tableName = "s3_multi_ns_table_" + tableIndex;
              List<String> tableId = Arrays.asList("test_ns", tableName);
              byte[] tableData = createTestTableData();

              CreateTableRequest createReq = new CreateTableRequest().id(tableId);
              localNs.createTable(createReq, tableData);

              DropTableRequest dropReq = new DropTableRequest().id(tableId);
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
    assertTrue(doneLatch.await(120, TimeUnit.SECONDS), "Timed out waiting for tasks to complete");

    executor.shutdown();
    assertTrue(executor.awaitTermination(30, TimeUnit.SECONDS));

    for (DirectoryNamespace ns : namespaces) {
      try {
        ns.close();
      } catch (Exception e) {
        // Ignore
      }
    }

    assertEquals(numTables, successCount.get(), "All tasks should succeed");
    assertEquals(0, failCount.get(), "No tasks should fail");
  }

  @Test
  void testConcurrentCreateThenDropFromDifferentInstanceOnS3() throws Exception {
    Map<String, String> baseConfig = createDirectoryNamespaceS3Config();

    // Initialize namespace first with a single instance to ensure __manifest
    // table is created and parent namespace exists before concurrent operations
    DirectoryNamespace initNs = new DirectoryNamespace();
    initNs.initialize(new HashMap<>(baseConfig), testAllocator);
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
              localNs.initialize(new HashMap<>(baseConfig), testAllocator);

              synchronized (createNamespaces) {
                createNamespaces.add(localNs);
              }

              String tableName = "s3_cross_instance_table_" + tableIndex;
              List<String> tableId = Arrays.asList("test_ns", tableName);
              byte[] tableData = createTestTableData();

              CreateTableRequest createReq = new CreateTableRequest().id(tableId);
              localNs.createTable(createReq, tableData);

              createSuccessCount.incrementAndGet();
            } catch (Exception e) {
              // Ignore
            } finally {
              createDoneLatch.countDown();
            }
          });
    }

    createStartLatch.countDown();
    assertTrue(createDoneLatch.await(120, TimeUnit.SECONDS), "Timed out waiting for creates");
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
              localNs.initialize(new HashMap<>(baseConfig), testAllocator);

              synchronized (dropNamespaces) {
                dropNamespaces.add(localNs);
              }

              String tableName = "s3_cross_instance_table_" + tableIndex;
              List<String> tableId = Arrays.asList("test_ns", tableName);

              DropTableRequest dropReq = new DropTableRequest().id(tableId);
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
    assertTrue(dropDoneLatch.await(120, TimeUnit.SECONDS), "Timed out waiting for drops");
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
}
