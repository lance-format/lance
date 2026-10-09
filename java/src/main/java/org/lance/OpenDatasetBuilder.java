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

import org.lance.namespace.LanceNamespace;
import org.lance.namespace.model.DescribeTableRequest;
import org.lance.namespace.model.DescribeTableResponse;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.util.Preconditions;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Builder for opening a Dataset.
 *
 * <p>This builder provides a fluent API for opening datasets either directly from a URI or from a
 * LanceNamespace. When using a namespace, the table location and storage options are automatically
 * fetched.
 *
 * <p>Example usage with URI:
 *
 * <pre>{@code
 * Dataset dataset = Dataset.open()
 *     .uri("s3://bucket/table.lance")
 *     .readOptions(options)
 *     .build();
 * }</pre>
 *
 * <p>Example usage with namespace client:
 *
 * <pre>{@code
 * Dataset dataset = Dataset.open()
 *     .namespaceClient(myNamespaceClient)
 *     .tableId(Arrays.asList("my_table"))
 *     .build();
 * }</pre>
 */
public class OpenDatasetBuilder {
  private BufferAllocator allocator;
  private boolean selfManagedAllocator = false;
  private String uri;
  private LanceNamespace namespaceClient;
  private List<String> tableId;
  private ReadOptions options = new ReadOptions.Builder().build();
  private Map<String, Map<String, String>> baseStoreParams = new HashMap<>();
  private Session session;

  /** Creates a new builder instance. Package-private, use Dataset.open() instead. */
  OpenDatasetBuilder() {}

  /**
   * Sets the buffer allocator.
   *
   * @param allocator Arrow buffer allocator
   * @return this builder instance
   */
  public OpenDatasetBuilder allocator(BufferAllocator allocator) {
    Preconditions.checkNotNull(allocator);
    this.allocator = allocator;
    this.selfManagedAllocator = false;
    return this;
  }

  /**
   * Sets the dataset URI.
   *
   * <p>Either uri() or namespaceClient()+tableId() must be specified, but not both.
   *
   * @param uri The dataset URI (e.g., "s3://bucket/table.lance" or "file:///path/to/table.lance")
   * @return this builder instance
   */
  public OpenDatasetBuilder uri(String uri) {
    this.uri = uri;
    return this;
  }

  /**
   * Sets the namespace client.
   *
   * <p>Must be used together with tableId(). Either uri() or namespaceClient()+tableId() must be
   * specified, but not both.
   *
   * @param namespaceClient The namespace implementation to fetch table info from
   * @return this builder instance
   */
  public OpenDatasetBuilder namespaceClient(LanceNamespace namespaceClient) {
    this.namespaceClient = namespaceClient;
    return this;
  }

  /**
   * Sets the table identifier.
   *
   * <p>Must be used together with namespaceClient(). Either uri() or namespaceClient()+tableId()
   * must be specified, but not both.
   *
   * @param tableId The table identifier (e.g., Arrays.asList("my_table"))
   * @return this builder instance
   */
  public OpenDatasetBuilder tableId(List<String> tableId) {
    this.tableId = tableId;
    return this;
  }

  /**
   * Sets the read options.
   *
   * @param options Read options
   * @return this builder instance
   */
  public OpenDatasetBuilder readOptions(ReadOptions options) {
    this.options = options;
    this.baseStoreParams = options.getBaseStoreParams();
    return this;
  }

  /**
   * Sets runtime-only object store parameters for registered base paths.
   *
   * <p>Entries are keyed by the exact {@link BasePath#getPath()} value persisted in the manifest.
   * Each value is used as-is for that base. These params are not persisted in the manifest. If a
   * base has no explicit entry, the read options storage options are used as a fallback.
   *
   * @param baseStoreParams object store parameters keyed by base path URI
   * @return this builder instance
   */
  public OpenDatasetBuilder baseStoreParams(Map<String, Map<String, String>> baseStoreParams) {
    this.baseStoreParams = new HashMap<>(baseStoreParams);
    return this;
  }

  /**
   * Sets the session to share caches between multiple datasets.
   *
   * <p>When a session is provided, the index and metadata caches from the session will be used
   * instead of creating new caches. This can improve cache hit rates when opening multiple related
   * datasets.
   *
   * <p>Note: When a session is provided, the indexCacheSizeBytes and metadataCacheSizeBytes
   * settings in ReadOptions are ignored because the session's caches are used instead.
   *
   * @param session The session to use
   * @return this builder instance
   */
  public OpenDatasetBuilder session(Session session) {
    this.session = session;
    return this;
  }

  /**
   * Opens the dataset with the configured parameters.
   *
   * <p>If a namespace client is configured, this automatically fetches the table location and
   * storage options from the namespace client via describeTable().
   *
   * @return Dataset
   * @throws IllegalArgumentException if required parameters are missing or invalid
   */
  public Dataset build() {
    // Validate that exactly one of uri or namespaceClient+tableId is provided
    boolean hasUri = uri != null;
    boolean hasNamespaceClient = namespaceClient != null && tableId != null;

    if (hasUri && hasNamespaceClient) {
      throw new IllegalArgumentException(
          "Cannot specify both uri and namespaceClient+tableId. Use one or the other.");
    }
    if (!hasUri && !hasNamespaceClient) {
      if (namespaceClient != null) {
        throw new IllegalArgumentException(
            "namespaceClient is set but tableId is missing. Both namespaceClient and tableId must"
                + " be provided together.");
      } else if (tableId != null) {
        throw new IllegalArgumentException(
            "tableId is set but namespaceClient is missing. Both namespaceClient and tableId must"
                + " be provided together.");
      } else {
        throw new IllegalArgumentException(
            "Either uri or namespaceClient+tableId must be provided.");
      }
    }

    Preconditions.checkNotNull(options, "options must be set");

    // Create allocator if not provided
    if (allocator == null) {
      allocator = new RootAllocator(Long.MAX_VALUE);
      selfManagedAllocator = true;
    }

    // Handle namespace client-based opening
    if (hasNamespaceClient) {
      return buildFromNamespaceClient();
    }

    // Handle URI-based opening
    return Dataset.open(allocator, selfManagedAllocator, uri, options, baseStoreParams, session);
  }

  private Dataset buildFromNamespaceClient() {
    // Call describe_table to get location and storage options
    DescribeTableRequest request = new DescribeTableRequest();
    request.setId(tableId);
    // Do not set the dataset version here. Some namespace implementations only support describing
    // the latest table metadata; the requested version is applied when opening the dataset below.

    DescribeTableResponse response = namespaceClient.describeTable(request);

    String location = response.getLocation();
    if (location == null || location.isEmpty()) {
      throw new IllegalArgumentException("Namespace client did not return a table location");
    }

    // Check if namespace client manages versioning (commits go through namespace client API)
    Boolean namespaceClientManagedVersioning = response.getManagedVersioning();

    Map<String, String> namespaceStorageOptions = response.getStorageOptions();

    ReadOptions.Builder optionsBuilder =
        new ReadOptions.Builder()
            .setIndexCacheSizeBytes(options.getIndexCacheSizeBytes())
            .setMetadataCacheSizeBytes(options.getMetadataCacheSizeBytes());

    options.getVersion().ifPresent(optionsBuilder::setVersion);
    options.getRef().ifPresent(optionsBuilder::setRef);
    options.getBlockSize().ifPresent(optionsBuilder::setBlockSize);
    options.getSerializedManifest().ifPresent(optionsBuilder::setSerializedManifest);
    optionsBuilder.setBaseStoreParams(baseStoreParams);

    Map<String, String> storageOptions = new HashMap<>(options.getStorageOptions());
    if (namespaceStorageOptions != null) {
      storageOptions.putAll(namespaceStorageOptions);
    }
    optionsBuilder.setStorageOptions(storageOptions);

    // Pass namespaceClient, tableId, and namespaceClientManagedVersioning to Rust
    // The Rust side will:
    // - Create a storage options provider when namespaceClient/tableId are non-null for credential
    // refresh
    // - Create an external manifest commit handler when namespaceClientManagedVersioning is true
    return Dataset.open(
        allocator,
        selfManagedAllocator,
        location,
        optionsBuilder.build(),
        baseStoreParams,
        session,
        namespaceClient,
        tableId,
        Boolean.TRUE.equals(namespaceClientManagedVersioning));
  }
}
