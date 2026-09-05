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

import org.lance.cleanup.CleanupExplanation;
import org.lance.cleanup.CleanupPolicy;
import org.lance.cleanup.RemovalStats;
import org.lance.compaction.CompactionOptions;
import org.lance.delta.DatasetDelta;
import org.lance.index.Index;
import org.lance.index.IndexBuildProgress;
import org.lance.index.IndexCriteria;
import org.lance.index.IndexDescription;
import org.lance.index.IndexOptions;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.OptimizeOptions;
import org.lance.index.scalar.ZoneStats;
import org.lance.ipc.DataStatistics;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;
import org.lance.memwal.InitializeMemWalParams;
import org.lance.memwal.MemWalIndexDetails;
import org.lance.memwal.ShardWriter;
import org.lance.memwal.ShardWriterConfig;
import org.lance.merge.MergeInsertParams;
import org.lance.merge.MergeInsertResult;
import org.lance.namespace.LanceNamespace;
import org.lance.operation.UpdateConfig;
import org.lance.operation.UpdateMap;
import org.lance.schema.ColumnAlteration;
import org.lance.schema.LanceSchema;
import org.lance.schema.SqlExpressions;
import org.lance.update.UpdateParams;
import org.lance.update.UpdateResult;
import org.lance.util.JsonUtils;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

import java.io.ByteArrayInputStream;
import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

/**
 * Class representing a Lance dataset, interfacing with the native lance library. This class
 * provides functionality to open and manage datasets with native code. The native library is loaded
 * statically and utilized through native methods. It implements the {@link java.io.Closeable}
 * interface to ensure proper resource management.
 */
public class Dataset implements Closeable {
  static {
    JniLoader.ensureLoaded();
  }

  private long nativeDatasetHandle;

  private BufferAllocator allocator;
  private boolean selfManagedAllocator = false;
  private Session session;
  private boolean ownsSession = false;

  /**
   * Serializes create/merge index builds on this Dataset handle. Always acquire the lifecycle read
   * lock first so a queued close cannot deadlock a reentrant read-lock owner.
   */
  private final ReentrantLock indexBuildLock = new ReentrantLock();

  private final LockManager lockManager = new LockManager(this);

  private Dataset() {}

  /**
   * Creates a builder for writing a dataset.
   *
   * <p>This builder supports writing datasets either directly to a URI or through a LanceNamespace.
   * Data can be provided via reader() or stream() methods.
   *
   * <p>Example usage with URI and reader:
   *
   * <pre>{@code
   * Dataset dataset = Dataset.write()
   *     .reader(myReader)
   *     .uri("s3://bucket/table.lance")
   *     .mode(WriteMode.CREATE)
   *     .execute();
   * }</pre>
   *
   * <p>Example usage with namespaceClient and empty table:
   *
   * <pre>{@code
   * Dataset dataset = Dataset.write()
   *     .schema(mySchema)
   *     .namespaceClient(myNamespaceClient)
   *     .tableId(Arrays.asList("my_table"))
   *     .mode(WriteMode.CREATE)
   *     .execute();
   * }</pre>
   *
   * @return A new WriteDatasetBuilder instance
   */
  public static WriteDatasetBuilder write() {
    return new WriteDatasetBuilder();
  }

  /**
   * Creates an empty dataset.
   *
   * @param allocator the buffer allocator
   * @param path dataset uri
   * @param schema dataset schema
   * @param params write params
   * @return Dataset
   * @deprecated Use {@link #write()} builder instead. For example: {@code
   *     Dataset.write().allocator(allocator).schema(schema).uri(path)
   *     .mode(WriteMode.CREATE).execute()}
   */
  @Deprecated
  public static Dataset create(
      BufferAllocator allocator, String path, Schema schema, WriteParams params) {
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(schema);
    Preconditions.checkNotNull(params);
    try (ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
      Data.exportSchema(allocator, schema, null, arrowSchema);
      Dataset dataset =
          createWithFfiSchema(
              arrowSchema.memoryAddress(),
              path,
              params.getMaxRowsPerFile(),
              params.getMaxRowsPerGroup(),
              params.getMaxBytesPerFile(),
              params.getMode(),
              params.getEnableStableRowIds(),
              params.getDataStorageVersion(),
              params.getEnableV2ManifestPaths(),
              params.getStorageOptions(),
              params.getBaseStoreParams(),
              params.getInitialBases(),
              params.getTargetBases(),
              params.getAllowExternalBlobOutsideBases(),
              params.getBlobPackFileSizeThreshold());
      dataset.allocator = allocator;
      return dataset;
    }
  }

  /**
   * Create a dataset with given stream.
   *
   * @param allocator buffer allocator
   * @param stream arrow stream
   * @param path dataset uri
   * @param params write parameters
   * @return Dataset
   * @deprecated Use {@link #write()} builder instead. For example: {@code
   *     Dataset.write().allocator(allocator).stream(stream).uri(path)
   *     .mode(WriteMode.CREATE).execute()}
   */
  /**
   * Create a dataset with given stream.
   *
   * @param allocator buffer allocator
   * @param stream arrow stream
   * @param path dataset uri
   * @param params write parameters
   * @return Dataset
   * @deprecated Use {@link #write()} builder instead.
   */
  @Deprecated
  public static Dataset create(
      BufferAllocator allocator, ArrowArrayStream stream, String path, WriteParams params) {
    return create(allocator, stream, path, params, null, null, false);
  }

  private static native Dataset createWithFfiSchema(
      long arrowSchemaMemoryAddress,
      String path,
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<String> mode,
      Optional<Boolean> enableStableRowIds,
      Optional<String> dataStorageVersion,
      Optional<Boolean> enableV2ManifestPaths,
      Map<String, String> storageOptions,
      Map<String, Map<String, String>> baseStoreParams,
      Optional<List<BasePath>> initialBases,
      Optional<List<String>> targetBases,
      Optional<Boolean> allowExternalBlobOutsideBases,
      Optional<Long> blobPackFileSizeThreshold);

  /**
   * Creates a dataset from an FFI arrow stream.
   *
   * @param arrowStreamMemoryAddress memory address of the arrow stream
   * @param path dataset uri
   * @param maxRowsPerFile max rows per file
   * @param maxRowsPerGroup max rows per group
   * @param maxBytesPerFile max bytes per file
   * @param mode write mode
   * @param enableStableRowIds whether to enable stable row ids
   * @param dataStorageVersion data storage version
   * @param enableV2ManifestPaths whether to enable v2 manifest paths
   * @param storageOptions storage options
   * @param baseStoreParams runtime-only object store parameters keyed by base path URI
   * @param initialBases initial bases
   * @param targetBases target bases
   * @param namespaceClient optional namespace client for managed versioning and credential refresh
   *     (can be null)
   * @param tableId optional table identifier within the namespace client (can be null)
   * @return Dataset
   */
  private static native Dataset createWithFfiStream(
      long arrowStreamMemoryAddress,
      String path,
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<String> mode,
      Optional<Boolean> enableStableRowIds,
      Optional<String> dataStorageVersion,
      Optional<Boolean> enableV2ManifestPaths,
      Map<String, String> storageOptions,
      Map<String, Map<String, String>> baseStoreParams,
      Optional<List<BasePath>> initialBases,
      Optional<List<String>> targetBases,
      Optional<Boolean> allowExternalBlobOutsideBases,
      Optional<Long> blobPackFileSizeThreshold,
      LanceNamespace namespaceClient,
      List<String> tableId,
      boolean namespaceClientManagedVersioning);

  /**
   * Creates a dataset with optional namespace client support for managed versioning.
   *
   * <p>When namespaceClient and tableId are provided, the Rust side will automatically create a
   * storage options provider for credential refresh. When namespaceClientManagedVersioning is true,
   * the commit handler will use the namespace client's create_table_version method for version
   * tracking.
   *
   * @param allocator buffer allocator
   * @param stream arrow stream
   * @param path dataset uri
   * @param params write parameters
   * @param namespaceClient optional namespace client for managed versioning and credential refresh
   *     (can be null)
   * @param tableId optional table identifier within the namespace client (can be null)
   * @param namespaceClientManagedVersioning whether namespace manages versioning (commits go
   *     through namespace API)
   * @return Dataset
   */
  static Dataset create(
      BufferAllocator allocator,
      ArrowArrayStream stream,
      String path,
      WriteParams params,
      LanceNamespace namespaceClient,
      List<String> tableId,
      boolean namespaceClientManagedVersioning) {
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(stream);
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(params);
    Dataset dataset =
        createWithFfiStream(
            stream.memoryAddress(),
            path,
            params.getMaxRowsPerFile(),
            params.getMaxRowsPerGroup(),
            params.getMaxBytesPerFile(),
            params.getMode(),
            params.getEnableStableRowIds(),
            params.getDataStorageVersion(),
            params.getEnableV2ManifestPaths(),
            params.getStorageOptions(),
            params.getBaseStoreParams(),
            params.getInitialBases(),
            params.getTargetBases(),
            params.getAllowExternalBlobOutsideBases(),
            params.getBlobPackFileSizeThreshold(),
            namespaceClient,
            tableId,
            namespaceClientManagedVersioning);
    dataset.allocator = allocator;
    return dataset;
  }

  /**
   * Open a dataset from the specified path.
   *
   * @param path file path
   * @return Dataset
   * @deprecated Use {@link #open()} builder instead: {@code Dataset.open().uri(path).build()}
   */
  @Deprecated
  public static Dataset open(String path) {
    return open(
        new RootAllocator(Long.MAX_VALUE),
        true,
        path,
        new ReadOptions.Builder().build(),
        new HashMap<>(),
        null);
  }

  /**
   * Open a dataset from the specified path.
   *
   * @param path file path
   * @param options the open options
   * @return Dataset
   * @deprecated Use {@link #open()} builder instead: {@code
   *     Dataset.open().uri(path).readOptions(options).build()}
   */
  @Deprecated
  public static Dataset open(String path, ReadOptions options) {
    return open(
        new RootAllocator(Long.MAX_VALUE), true, path, options, options.getBaseStoreParams(), null);
  }

  /**
   * Open a dataset from the specified path.
   *
   * @param path file path
   * @param allocator Arrow buffer allocator
   * @return Dataset
   * @deprecated Use {@link #open()} builder instead: {@code
   *     Dataset.open().allocator(allocator).uri(path).build()}
   */
  @Deprecated
  public static Dataset open(String path, BufferAllocator allocator) {
    return open(allocator, path, new ReadOptions.Builder().build());
  }

  /**
   * Open a dataset from the specified path with additional options.
   *
   * @param allocator Arrow buffer allocator
   * @param path file path
   * @param options the open options
   * @return Dataset
   * @deprecated Use {@link #open()} builder instead: {@code
   *     Dataset.open().allocator(allocator).uri(path).readOptions(options).build()}
   */
  @Deprecated
  public static Dataset open(BufferAllocator allocator, String path, ReadOptions options) {
    return open(allocator, false, path, options, options.getBaseStoreParams(), null);
  }

  /**
   * Open a dataset from the specified path with additional options.
   *
   * @param path file path
   * @param options the open options
   * @return Dataset
   */
  static Dataset open(
      BufferAllocator allocator,
      boolean selfManagedAllocator,
      String path,
      ReadOptions options,
      Session session) {
    return open(
        allocator, selfManagedAllocator, path, options, options.getBaseStoreParams(), session);
  }

  static Dataset open(
      BufferAllocator allocator,
      boolean selfManagedAllocator,
      String path,
      ReadOptions options,
      Map<String, Map<String, String>> baseStoreParams,
      Session session) {
    return open(
        allocator,
        selfManagedAllocator,
        path,
        options,
        baseStoreParams,
        session,
        null,
        null,
        false);
  }

  /**
   * Open a dataset from the specified path with additional options and namespace client.
   *
   * <p>When namespaceClient and tableId are provided, the Rust side will automatically create a
   * storage options provider for credential refresh.
   *
   * @param path file path
   * @param options the open options
   * @param namespaceClient the LanceNamespace to use for managed versioning and credential refresh
   *     (null if not using namespace client)
   * @param tableId table identifier (null if not using namespace client)
   * @param namespaceClientManagedVersioning whether namespace manages versioning (commits go
   *     through namespace API)
   * @return Dataset
   */
  static Dataset open(
      BufferAllocator allocator,
      boolean selfManagedAllocator,
      String path,
      ReadOptions options,
      Map<String, Map<String, String>> baseStoreParams,
      Session session,
      LanceNamespace namespaceClient,
      List<String> tableId,
      boolean namespaceClientManagedVersioning) {
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(options);

    Session effectiveSession = session;
    if (effectiveSession == null && options.getSession().isPresent()) {
      effectiveSession = options.getSession().get();
    }
    long sessionHandle = effectiveSession != null ? effectiveSession.getNativeHandle() : 0;

    Dataset dataset =
        openNative(
            path,
            options.getVersion(),
            options.getBlockSize(),
            options.getIndexCacheSizeBytes(),
            options.getMetadataCacheSizeBytes(),
            options.getStorageOptions(),
            baseStoreParams,
            options.getSerializedManifest(),
            sessionHandle,
            namespaceClient,
            tableId,
            namespaceClientManagedVersioning);
    dataset.allocator = allocator;
    dataset.selfManagedAllocator = selfManagedAllocator;
    if (effectiveSession != null) {
      dataset.session = effectiveSession;
    } else {
      dataset.session = Session.fromHandle(dataset.nativeGetSessionHandle());
      dataset.ownsSession = true;
    }
    return dataset;
  }

  private static native Dataset openNative(
      String path,
      Optional<Long> version,
      Optional<Integer> blockSize,
      long indexCacheSize,
      long metadataCacheSizeBytes,
      Map<String, String> storageOptions,
      Map<String, Map<String, String>> baseStoreParams,
      Optional<ByteBuffer> serializedManifest,
      long sessionHandle,
      LanceNamespace namespaceClient,
      List<String> tableId,
      boolean namespaceClientManagedVersioning);

  /**
   * List manifest locations without reading or deserializing the manifest contents.
   *
   * <p>The returned locations are not guaranteed to be ordered. This operation may list and
   * materialize the full manifest history.
   *
   * <p>This method is for datasets whose committed manifests can be listed authoritatively from the
   * object store. Namespace-managed tables, external version stores such as {@code s3+ddb}, and
   * tables using a custom commit handler are not supported.
   *
   * @param uri dataset URI
   * @return manifest locations
   */
  public static List<ManifestLocation> listManifestLocations(String uri) {
    return listManifestLocations(uri, new HashMap<>());
  }

  /**
   * List manifest locations without reading or deserializing the manifest contents.
   *
   * <p>The returned locations are not guaranteed to be ordered. This operation may list and
   * materialize the full manifest history.
   *
   * <p>This method is for datasets whose committed manifests can be listed authoritatively from the
   * object store. Namespace-managed tables, external version stores such as {@code s3+ddb}, and
   * tables using a custom commit handler are not supported.
   *
   * @param uri dataset URI
   * @param storageOptions object-store credentials and connection options
   * @return manifest locations
   */
  public static List<ManifestLocation> listManifestLocations(
      String uri, Map<String, String> storageOptions) {
    Preconditions.checkNotNull(uri, "uri must not be null");
    Preconditions.checkNotNull(storageOptions, "storageOptions must not be null");
    return listManifestLocationsNative(uri, storageOptions);
  }

  private static native List<ManifestLocation> listManifestLocationsNative(
      String uri, Map<String, String> storageOptions);

  /**
   * Creates a builder for opening a dataset.
   *
   * <p>This builder supports opening datasets either directly from a URI or from a LanceNamespace.
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
   * <p>Example usage with namespaceClient:
   *
   * <pre>{@code
   * Dataset dataset = Dataset.open()
   *     .namespaceClient(myNamespaceClient)
   *     .tableId(Arrays.asList("my_table"))
   *     .build();
   * }</pre>
   *
   * @return A new OpenDatasetBuilder instance
   */
  public static OpenDatasetBuilder open() {
    return new OpenDatasetBuilder();
  }

  /**
   * Create a new version of dataset. Use {@link Transaction} instead
   *
   * @param allocator the buffer allocator
   * @param path The file path of the dataset to open.
   * @param operation The operation to apply to the dataset.
   * @param readVersion The version of the dataset that was used as the base for the changes. This
   *     is not needed for overwrite or restore operations.
   * @return A new instance of {@link Dataset} linked to the opened dataset.
   */
  @Deprecated
  public static Dataset commit(
      BufferAllocator allocator,
      String path,
      FragmentOperation operation,
      Optional<Long> readVersion) {
    return commit(allocator, path, operation, readVersion, new HashMap<>());
  }

  @Deprecated
  public static Dataset commit(
      BufferAllocator allocator,
      String path,
      FragmentOperation operation,
      Optional<Long> readVersion,
      Map<String, String> storageOptions) {
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(path);
    Preconditions.checkNotNull(operation);
    Preconditions.checkNotNull(readVersion);
    Dataset dataset = operation.commit(allocator, path, readVersion, storageOptions);
    dataset.allocator = allocator;
    return dataset;
  }

  /** Use {@link Transaction} instead */
  @Deprecated
  public static native Dataset commitAppend(
      String path,
      Optional<Long> readVersion,
      List<FragmentMetadata> fragmentsMetadata,
      Map<String, String> storageOptions);

  /** Use {@link Transaction} instead */
  @Deprecated
  public static native Dataset commitOverwrite(
      String path,
      long arrowSchemaMemoryAddress,
      Optional<Long> readVersion,
      List<FragmentMetadata> fragmentsMetadata,
      Map<String, String> storageOptions);

  public BufferAllocator allocator() {
    return allocator;
  }

  /** Package-private setter for allocator, used by {@link CommitBuilder}. */
  void setAllocator(BufferAllocator allocator) {
    this.allocator = allocator;
  }

  /**
   * Create a new transaction builder at current version for the dataset. The dataset itself will
   * not refresh after the transaction committed.
   *
   * @return A new instance of {@link SourcedTransaction.Builder} linked to the opened dataset.
   */
  public SourcedTransaction.Builder newTransactionBuilder() {
    return new SourcedTransaction.Builder(this);
  }

  /**
   * Commit a single transaction and return a new Dataset with the new version. Original dataset
   * version will not be refreshed.
   *
   * @param transaction The transaction to commit
   * @return A new instance of {@link Dataset} linked to committed version.
   */
  public Dataset commitTransaction(Transaction transaction) {
    return commitTransaction(transaction, false, true);
  }

  /**
   * Commit a single transaction and return a new Dataset with the new version. Original dataset
   * version will not be refreshed.
   *
   * @param transaction The transaction to commit
   * @param detached If true, the commit will not be part of the main dataset lineage.
   * @param enableV2ManifestPaths If true, and this is a new dataset, uses the new V2 manifest
   *     paths. These paths provide more efficient opening of datasets with many versions on object
   *     stores. This parameter has no effect if the dataset already exists. To migrate an existing
   *     dataset, instead use the `migrateManifestPathsV2` method. Default is true. WARNING: turning
   *     this on will make the dataset unreadable for older versions of Lance (prior to 0.17.0).
   * @return A new instance of {@link Dataset} linked to committed version.
   */
  public Dataset commitTransaction(
      Transaction transaction, boolean detached, boolean enableV2ManifestPaths) {
    Preconditions.checkNotNull(transaction);
    Dataset dataset =
        new CommitBuilder(this)
            .detached(detached)
            .enableV2ManifestPaths(enableV2ManifestPaths)
            .execute(transaction);
    if (selfManagedAllocator) {
      dataset.allocator = new RootAllocator(Long.MAX_VALUE);
    } else {
      dataset.allocator = allocator;
    }
    return dataset;
  }

  /**
   * Drop a Dataset, deleting everything under {@code path} recursively.
   *
   * <p>To limit the damage a mistyped or misconfigured path can do, {@code path} must be a dataset
   * root, meaning it holds a manifest that can be read, or a namespace declare/deregister marker.
   * Anything else throws {@link IllegalArgumentException}, including a path that holds only data
   * files or only unreadable manifests: such leftovers need an explicit storage-level delete.
   *
   * <p>Note that a path which passes this check is deleted in full, including any unmanaged files
   * kept next to the dataset.
   *
   * @param path The file path of the dataset
   * @param storageOptions Storage options
   * @throws IllegalArgumentException if {@code path} is not a Lance dataset root
   */
  public static native void drop(String path, Map<String, String> storageOptions);

  /**
   * Migrate the manifest paths to the new format.
   *
   * <p>This will update the manifest to use the new v2 format for paths.
   *
   * <p>This function is idempotent, and can be run multiple times without changing the state of the
   * object store.
   *
   * <p>DANGER: this should not be run while other concurrent operations are happening. And it
   * should also run until completion before resuming other operations.
   */
  public void migrateManifestPathsV2() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeMigrateManifestPathsV2();
    }
  }

  private native void nativeMigrateManifestPathsV2();

  /**
   * Add columns to the dataset.
   *
   * @param sqlExpressions The SQL expressions to add columns
   * @param batchSize The number of rows to read at a time from the source dataset when applying the
   *     transform.
   */
  public void addColumns(SqlExpressions sqlExpressions, Optional<Long> batchSize) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeAddColumnsBySqlExpressions(sqlExpressions, batchSize);
    }
  }

  private native void nativeAddColumnsBySqlExpressions(
      SqlExpressions sqlExpressions, Optional<Long> batchSize);

  /**
   * Add columns to the dataset.
   *
   * @param stream The Arrow Array Stream generated by arrow reader to add columns.
   * @param batchSize The number of rows to read at a time from the source dataset when applying the
   *     transform.
   */
  public void addColumns(ArrowArrayStream stream, Optional<Long> batchSize) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeAddColumnsByReader(stream.memoryAddress(), batchSize);
    }
  }

  private native void nativeAddColumnsByReader(
      long arrowStreamMemoryAddress, Optional<Long> batchSize);

  /**
   * Add columns to the dataset.
   *
   * @param schema The Arrow schema definitions to add columns.
   */
  public void addColumns(Schema schema) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Preconditions.checkArgument(schema != null, "Schema is empty");
      try (ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
        Data.exportSchema(allocator, schema, null, arrowSchema);
        nativeAddColumnsBySchema(arrowSchema.memoryAddress());
      }
    }
  }

  /**
   * Add columns to the dataset.
   *
   * @param fields The Arrow field definitions to add columns.
   */
  public void addColumns(List<Field> fields) {
    Preconditions.checkArgument(fields != null && !fields.isEmpty(), "Fields are empty");
    addColumns(new Schema(fields));
  }

  private native void nativeAddColumnsBySchema(long schemaPtr);

  /**
   * Drop columns from the dataset.
   *
   * @param columns The columns to drop
   */
  public void dropColumns(List<String> columns) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeDropColumns(columns);
    }
  }

  private native void nativeDropColumns(List<String> columns);

  /**
   * Alter columns in the dataset.
   *
   * @param columnAlterations The list of columns need to be altered.
   */
  public void alterColumns(List<ColumnAlteration> columnAlterations) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      // Cast target types are carried across the FFI boundary through the Arrow C Data
      // Interface rather than ArrowType#toString(), which does not round-trip reliably on
      // the native side (parameterized types such as Int(64, true) fail to parse and the
      // cast would otherwise be silently dropped). One field is exported per alteration that
      // requests a type change, in the same order as {@code columnAlterations}.
      List<Field> castFields = new ArrayList<>();
      int castIndex = 0;
      for (ColumnAlteration alteration : columnAlterations) {
        if (alteration.getDataType().isPresent()) {
          castFields.add(new Field("f" + castIndex++, castFieldType(alteration), null));
        }
      }
      try (ArrowSchema castSchema = ArrowSchema.allocateNew(allocator)) {
        Data.exportSchema(allocator, new Schema(castFields), null, castSchema);
        nativeAlterColumns(columnAlterations, castSchema.memoryAddress());
      }
    }
  }

  private static FieldType castFieldType(ColumnAlteration alteration) {
    boolean nullable = alteration.getNullable().orElse(true);
    return new FieldType(nullable, alteration.getDataType().get(), null);
  }

  private native void nativeAlterColumns(List<ColumnAlteration> columnAlterations, long castAddr);

  /**
   * Create a new Dataset Scanner.
   *
   * @return a dataset scanner
   */
  public LanceScanner newScan() {
    return newScan(new ScanOptions.Builder().build());
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @param batchSize the scan options with batch size, columns filter, and substrait
   * @return a dataset scanner
   */
  public LanceScanner newScan(long batchSize) {
    return newScan(new ScanOptions.Builder().batchSize(batchSize).build());
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @param options the scan options
   * @return a dataset scanner
   */
  public LanceScanner newScan(ScanOptions options) {
    Preconditions.checkNotNull(options);
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return LanceScanner.create(this, options, allocator);
    }
  }

  /**
   * Select rows of data by index.
   *
   * @param indices the indices to take
   * @param columns the columns to take
   * @return an ArrowReader
   */
  public ArrowReader take(List<Long> indices, List<String> columns) throws IOException {
    Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      byte[] arrowData = nativeTake(indices, columns);
      ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(arrowData);
      ReadableByteChannel readChannel = Channels.newChannel(byteArrayInputStream);
      return new ArrowStreamReader(readChannel, allocator) {
        @Override
        public void close() throws IOException {
          super.close();
          readChannel.close();
          byteArrayInputStream.close();
        }
      };
    }
  }

  private native byte[] nativeTake(List<Long> indices, List<String> columns);

  /**
   * Select rows of data by their physical row IDs.
   *
   * @param rowIds the physical row IDs to retrieve (from the _rowid column)
   * @param columns the columns to include in the result
   * @return an ArrowReader containing the requested rows in input order
   * @throws IOException if a row ID does not exist or an I/O error occurs
   */
  public ArrowReader takeRows(List<Long> rowIds, List<String> columns) throws IOException {
    Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
    Preconditions.checkArgument(
        rowIds != null && !rowIds.isEmpty(), "rowIds cannot be null or empty");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      byte[] arrowData = nativeTakeRows(rowIds, columns);
      ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(arrowData);
      ReadableByteChannel readChannel = Channels.newChannel(byteArrayInputStream);
      return new ArrowStreamReader(readChannel, allocator) {
        @Override
        public void close() throws IOException {
          super.close();
          readChannel.close();
          byteArrayInputStream.close();
        }
      };
    }
  }

  private native byte[] nativeTakeRows(List<Long> rowIds, List<String> columns);

  /**
   * Randomly sample n rows from the dataset.
   *
   * <p>The returned rows are in row-id order (not random order), which allows the underlying take
   * operation to use an efficient sorted code path.
   *
   * @param n the number of rows to sample
   * @param columns the columns to include in the result
   * @return an ArrowReader containing the sampled rows
   * @throws IOException if an I/O error occurs
   */
  public ArrowReader sample(long n, List<String> columns) throws IOException {
    return sample(n, columns, Optional.empty());
  }

  /**
   * Randomly sample n rows from specific fragments of the dataset.
   *
   * <p>The returned rows are in row-id order (not random order), which allows the underlying take
   * operation to use an efficient sorted code path.
   *
   * @param n the number of rows to sample
   * @param columns the columns to include in the result
   * @param fragmentIds optional list of fragment IDs to restrict sampling to
   * @return an ArrowReader containing the sampled rows
   * @throws IOException if an I/O error occurs
   */
  public ArrowReader sample(long n, List<String> columns, Optional<List<Integer>> fragmentIds)
      throws IOException {
    Preconditions.checkArgument(n > 0, "n must be greater than 0");
    Preconditions.checkNotNull(columns, "columns cannot be null");
    Preconditions.checkArgument(!columns.isEmpty(), "columns cannot be empty");
    Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      byte[] arrowData = nativeSample(n, columns, fragmentIds);
      ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(arrowData);
      ReadableByteChannel readChannel = Channels.newChannel(byteArrayInputStream);
      return new ArrowStreamReader(readChannel, allocator) {
        @Override
        public void close() throws IOException {
          super.close();
          readChannel.close();
          byteArrayInputStream.close();
        }
      };
    }
  }

  private native byte[] nativeSample(
      long n, List<String> columns, Optional<List<Integer>> fragmentIds);

  /**
   * Delete rows of data by predicate.
   *
   * @param predicate the predicate to delete
   */
  public void delete(String predicate) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeDelete(predicate);
    }
  }

  private native void nativeDelete(String predicate);

  /**
   * Truncate the dataset by deleting all rows. The schema is preserved and a new version is
   * created.
   */
  public void truncateTable() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeTruncateTable();
    }
  }

  private native void nativeTruncateTable();

  /**
   * Gets the URI of the dataset.
   *
   * @return the URI of the dataset
   */
  public String uri() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeUri();
    }
  }

  private native String nativeUri();

  /**
   * Get the currently checked out version id of the dataset
   *
   * @return the version id of the dataset
   */
  public long version() {
    return getVersion().getId();
  }

  /**
   * Gets the currently checked out version of the dataset.
   *
   * @return the version of the dataset
   */
  public Version getVersion() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetVersion();
    }
  }

  private native Version nativeGetVersion();

  /**
   * Get the version history of the dataset.
   *
   * @return the version history of the dataset
   */
  public List<Version> listVersions() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeListVersions();
    }
  }

  private native List<Version> nativeListVersions();

  /**
   * Get the number of versions in the current version history.
   *
   * <p>Unlike {@link #listVersions()}, this method does not read or deserialize every manifest.
   * Detached versions are not included.
   *
   * @return the number of versions
   */
  public long getVersionCount() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetVersionCount();
    }
  }

  private native long nativeGetVersionCount();

  /**
   * @return the latest version of the dataset.
   */
  public long latestVersion() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetLatestVersionId();
    }
  }

  private native long nativeGetLatestVersionId();

  /**
   * Get the initial storage options used to open this dataset.
   *
   * <p>This returns the options that were provided when the dataset was opened, without any refresh
   * from the provider. Returns null if no storage options were provided.
   *
   * @return the initial storage options, or null if none were provided
   */
  public Map<String, String> getInitialStorageOptions() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetInitialStorageOptions();
    }
  }

  private native Map<String, String> nativeGetInitialStorageOptions();

  /**
   * Get the latest storage options, potentially refreshed from the provider.
   *
   * <p>If a storage options provider was configured and credentials are expiring, this will refresh
   * them.
   *
   * @return the latest storage options (static or refreshed from provider), or null if no storage
   *     options were configured for this dataset
   * @throws RuntimeException if an error occurs while fetching/refreshing options from the provider
   */
  public Map<String, String> getLatestStorageOptions() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetLatestStorageOptions();
    }
  }

  private native Map<String, String> nativeGetLatestStorageOptions();

  /** Checkout the dataset to the latest version. */
  public void checkoutLatest() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeCheckoutLatest();
    }
  }

  private native void nativeCheckoutLatest();

  /**
   * Checks out a specific version of the dataset. If the version is already checked out, it returns
   * a new Java Dataset object pointing to the same underlying Rust Dataset object
   *
   * @param version the version to check out
   * @return a new Dataset instance with the specified version checked out
   */
  public Dataset checkoutVersion(long version) {
    Preconditions.checkArgument(version > 0, "version number must be greater than 0");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return initializeCheckoutDataset(nativeCheckoutVersion(version));
    }
  }

  private native Dataset nativeCheckoutVersion(long version);

  /**
   * Checks out a specific tag of the dataset. If the underlying version is already checked out, it
   * returns a new Java Dataset object pointing to the same underlying Rust Dataset object
   *
   * @param tag the tag to check out
   * @return a new Dataset instance with the specified tag checked out
   */
  public Dataset checkoutTag(String tag) {
    Preconditions.checkArgument(tag != null, "Tag can not be null");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return initializeCheckoutDataset(nativeCheckoutTag(tag));
    }
  }

  private native Dataset nativeCheckoutTag(String tag);

  private Dataset initializeCheckoutDataset(Dataset checkedOutDataset) {
    if (selfManagedAllocator) {
      checkedOutDataset.allocator = new RootAllocator(Long.MAX_VALUE);
    } else {
      checkedOutDataset.allocator = allocator;
    }
    checkedOutDataset.session = Session.fromHandle(checkedOutDataset.nativeGetSessionHandle());
    checkedOutDataset.ownsSession = true;
    return checkedOutDataset;
  }

  /**
   * Restore the currently checked out version of the dataset as the latest version. This operation
   * produces a new version and doesn't influence any old versions and tags.
   */
  public void restore() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeRestore();
    }
  }

  private native void nativeRestore();

  /**
   * Creates a new index on the dataset
   *
   * @param columns the columns to index from
   * @param indexType the index type
   * @param name the name of the created index
   * @param params index params
   * @param replace whether to replace the existing index
   * @return the metadata of the created index
   * @deprecated please use {@link Dataset#createIndex(IndexOptions)} instead.
   */
  @Deprecated
  public Index createIndex(
      List<String> columns,
      IndexType indexType,
      Optional<String> name,
      IndexParams params,
      boolean replace) {
    return createIndex(
        IndexOptions.builder(columns, indexType, params)
            .replace(replace)
            .withIndexName(name.orElse(null))
            .build());
  }

  /**
   * Creates a new index on the dataset.
   *
   * <p>Concurrent {@link #createIndex} / {@link #mergeIndexMetadata} calls on the same Dataset
   * handle are serialized.
   *
   * @param options options for building index
   * @return the metadata of the created index
   */
  public Index createIndex(IndexOptions options) {
    Preconditions.checkNotNull(options, "options cannot be null");
    return createIndexInternal(options, null);
  }

  /**
   * Creates a new index on the dataset while reporting stage-level progress.
   *
   * <p>Stage names, work units, and whether a total is available depend on the index type. The
   * callback must be thread-safe because Lance may invoke it concurrently from native runtime
   * threads. Callbacks may re-enter read-only methods on this Dataset. Conflicting write re-entry
   * from a callback is rejected; unrelated concurrent callers keep their normal wait behavior.
   * Concurrent {@link #createIndex} / {@link #mergeIndexMetadata} calls on the same Dataset handle
   * are serialized.
   *
   * <pre>{@code
   * Index index = dataset.createIndex(options, new IndexBuildProgress() {
   *   public void stageStart(String stage, Optional<Long> total, String unit) { }
   *   public void stageProgress(String stage, long completed) { }
   *   public void stageComplete(String stage) { }
   * });
   * }</pre>
   *
   * @param options options for building index
   * @param progress thread-safe progress callback
   * @return the metadata of the created index
   */
  public Index createIndex(IndexOptions options, IndexBuildProgress progress) {
    Preconditions.checkNotNull(options, "options cannot be null");
    Preconditions.checkNotNull(progress, "progress cannot be null");
    return createIndexInternal(options, progress);
  }

  private Index createIndexInternal(IndexOptions options, IndexBuildProgress progress) {
    if (ContextIndexBuildProgress.isActive(this)) {
      throw new IllegalStateException("Dataset is busy in an index progress callback");
    }
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      acquireIndexBuildLock();
      try {
        if (progress == null) {
          return nativeCreateIndex(
              options.getColumns(),
              options.getIndexType().getValue(),
              options.getIndexName(),
              options.getIndexParams(),
              options.isReplace(),
              options.isTrain(),
              options.getFragmentIds(),
              options.getIndexUUID(),
              options.getPreprocessedData().map(ArrowArrayStream::memoryAddress));
        }
        return nativeCreateIndexWithProgress(
            options.getColumns(),
            options.getIndexType().getValue(),
            options.getIndexName(),
            options.getIndexParams(),
            options.isReplace(),
            options.isTrain(),
            options.getFragmentIds(),
            options.getIndexUUID(),
            options.getPreprocessedData().map(ArrowArrayStream::memoryAddress),
            new ContextIndexBuildProgress(this, progress));
      } finally {
        indexBuildLock.unlock();
      }
    }
  }

  private void acquireIndexBuildLock() {
    if (ContextIndexBuildProgress.isCallbackActive()) {
      // An outer index build waits for its callback to return, so waiting here could create a
      // cross-Dataset lock cycle between two concurrent builds.
      if (!indexBuildLock.tryLock()) {
        throw new IllegalStateException(
            "Dataset is busy with an index build and cannot start another build "
                + "from an index progress callback");
      }
    } else {
      indexBuildLock.lock();
    }
  }

  private native Index nativeCreateIndex(
      List<String> columns,
      int indexTypeCode,
      Optional<String> name,
      IndexParams params,
      boolean replace,
      boolean train,
      Optional<List<Integer>> fragments,
      Optional<String> indexUUID,
      Optional<Long> arrowStreamMemoryAddress);

  private native Index nativeCreateIndexWithProgress(
      List<String> columns,
      int indexTypeCode,
      Optional<String> name,
      IndexParams params,
      boolean replace,
      boolean train,
      Optional<List<Integer>> fragments,
      Optional<String> indexUUID,
      Optional<Long> arrowStreamMemoryAddress,
      IndexBuildProgress progress);

  /**
   * Drop an index by name.
   *
   * @param name the index name to drop
   */
  public void dropIndex(String name) {
    Preconditions.checkArgument(name != null && !name.isEmpty(), "name cannot be null or empty");
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeDropIndex(name);
    }
  }

  private native void nativeDropIndex(String name);

  public void mergeIndexMetadata(
      String indexUUID, IndexType indexType, Optional<Integer> batchReadHead) {
    mergeIndexMetadataInternal(indexUUID, indexType, batchReadHead, null);
  }

  private native void innerMergeIndexMetadata(
      String indexUUID, int indexType, Optional<Integer> batchReadHead);

  /**
   * Merge distributed index metadata while reporting stage-level progress.
   *
   * <p>Callback re-entry semantics match {@link #createIndex(IndexOptions, IndexBuildProgress)}:
   * read-only Dataset methods are allowed, conflicting write re-entry is rejected, and concurrent
   * create/merge builds on this handle are serialized.
   *
   * @param indexUUID shared UUID used by the distributed index parts
   * @param indexType type of index metadata to merge
   * @param batchReadHead optional limit for metadata read concurrency
   * @param progress thread-safe progress callback
   */
  public void mergeIndexMetadata(
      String indexUUID,
      IndexType indexType,
      Optional<Integer> batchReadHead,
      IndexBuildProgress progress) {
    Preconditions.checkNotNull(progress, "progress cannot be null");
    mergeIndexMetadataInternal(indexUUID, indexType, batchReadHead, progress);
  }

  private void mergeIndexMetadataInternal(
      String indexUUID,
      IndexType indexType,
      Optional<Integer> batchReadHead,
      IndexBuildProgress progress) {
    if (ContextIndexBuildProgress.isActive(this)) {
      throw new IllegalStateException("Dataset is busy in an index progress callback");
    }
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      acquireIndexBuildLock();
      try {
        if (progress == null) {
          innerMergeIndexMetadata(indexUUID, indexType.getValue(), batchReadHead);
        } else {
          innerMergeIndexMetadataWithProgress(
              indexUUID,
              indexType.getValue(),
              batchReadHead,
              new ContextIndexBuildProgress(this, progress));
        }
      } finally {
        indexBuildLock.unlock();
      }
    }
  }

  private native void innerMergeIndexMetadataWithProgress(
      String indexUUID,
      int indexType,
      Optional<Integer> batchReadHead,
      IndexBuildProgress progress);

  /** Merge one caller-defined group of existing uncommitted vector index segments. */
  public Index mergeExistingIndexSegments(List<Index> segments) {
    Preconditions.checkNotNull(segments, "segments cannot be null");
    Preconditions.checkArgument(!segments.isEmpty(), "segments cannot be empty");
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeMergeExistingIndexSegments(segments);
    }
  }

  private native Index nativeMergeExistingIndexSegments(List<Index> segments);

  /**
   * Publish one or more existing physical index segments as a logical index.
   *
   * @param indexName logical index name
   * @param column indexed column name
   * @param segments physical segment metadata to publish
   * @return committed manifest metadata
   */
  public List<Index> commitExistingIndexSegments(
      String indexName, String column, List<Index> segments) {
    Preconditions.checkArgument(
        indexName != null && !indexName.isEmpty(), "indexName cannot be null or empty");
    Preconditions.checkArgument(
        column != null && !column.isEmpty(), "column cannot be null or empty");
    Preconditions.checkNotNull(segments, "segments cannot be null");
    Preconditions.checkArgument(!segments.isEmpty(), "segments cannot be empty");
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeCommitExistingIndexSegments(indexName, column, segments);
    }
  }

  private native List<Index> nativeCommitExistingIndexSegments(
      String indexName, String column, List<Index> segments);

  /**
   * Count the number of rows in the dataset.
   *
   * @return num of rows
   */
  public long countRows() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeCountRows(Optional.empty());
    }
  }

  /**
   * Count the number of rows in the dataset.
   *
   * @param filter the filter expr to count row
   * @return num of rows
   */
  public long countRows(String filter) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Preconditions.checkArgument(
          null != filter && !filter.isEmpty(), "filter cannot be null or empty");
      return nativeCountRows(Optional.of(filter));
    }
  }

  private native long nativeCountRows(Optional<String> filter);

  /**
   * Returns the session associated with this dataset.
   *
   * <p>The session holds runtime state for the dataset, including index and metadata caches. If a
   * session was provided when opening the dataset, that session is returned. Otherwise, a new
   * session was created automatically.
   *
   * <p>The returned session can be used to open other datasets to share caches.
   *
   * @return the session associated with this dataset
   */
  public Session session() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return session;
    }
  }

  private native long nativeGetSessionHandle();

  /**
   * Count rows matching a filter using a specific scalar index. This directly queries the index and
   * counts matching row addresses, which is more efficient than scanning when the index covers the
   * filter column.
   *
   * @param indexName the name of the scalar index to use
   * @param filter the filter expression (e.g., "column = 5")
   * @param fragmentIds optional list of fragment IDs to restrict the count to
   * @return count of matching rows
   */
  public long countIndexedRows(
      String indexName, String filter, Optional<List<Integer>> fragmentIds) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Preconditions.checkArgument(
          indexName != null && !indexName.isEmpty(), "indexName cannot be null or empty");
      Preconditions.checkArgument(
          filter != null && !filter.isEmpty(), "filter cannot be null or empty");
      return nativeCountIndexedRows(indexName, filter, fragmentIds);
    }
  }

  private native long nativeCountIndexedRows(
      String indexName, String filter, Optional<List<Integer>> fragmentIds);

  /**
   * Calculate the size of the dataset.
   *
   * @return the size of the dataset
   */
  public long calculateDataSize() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetDataStatistics().getDataSize();
    }
  }

  /**
   * Calculate the statistics of the dataset.
   *
   * @return the statistics of the dataset
   */
  private native DataStatistics nativeGetDataStatistics();

  /**
   * Get all fragments in this dataset.
   *
   * @return A list of {@link Fragment}.
   */
  public List<Fragment> getFragments() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      // Set a pointer in Fragment to dataset, to make it is easier to issue IOs
      // later.
      //
      // We do not need to close Fragments.
      return this.getFragmentsNative().stream()
          .map(metadata -> new Fragment(this, metadata))
          .collect(Collectors.toList());
    }
  }

  private native List<FragmentMetadata> getFragmentsNative();

  /**
   * Returns the storage bases registered by the current manifest.
   *
   * <p>Files without a base id resolve against this dataset's own URI. Files with a base id resolve
   * against the matching entry in this list.
   */
  public List<BasePath> getBasePaths() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetBasePaths();
    }
  }

  private native List<BasePath> nativeGetBasePaths();

  /**
   * Get per-fragment statistics for all fragments in this dataset version.
   *
   * <p>Unlike {@link #getFragments()}, this is a metadata-only bulk operation: no per-fragment Java
   * objects are materialized, and native code fills the returned primitive arrays directly. This
   * makes it suitable for planning over datasets with a very large number of fragments. Row counts
   * match {@link FragmentMetadata#getNumRows()} (physical rows minus deleted rows).
   *
   * @return per-fragment statistics as parallel arrays, in manifest order
   */
  public FragmentStatistics getFragmentStatistics() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetFragmentStatistics();
    }
  }

  private native FragmentStatistics nativeGetFragmentStatistics();

  /**
   * Gets the arrow schema of the dataset.
   *
   * @return the arrow schema
   */
  public Schema getSchema() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      try (ArrowSchema ffiArrowSchema = ArrowSchema.allocateNew(allocator)) {
        importFfiSchema(ffiArrowSchema.memoryAddress());
        return Data.importSchema(allocator, ffiArrowSchema, null);
      }
    }
  }

  private native void importFfiSchema(long arrowSchemaMemoryAddress);

  /**
   * Get the {@link org.lance.schema.LanceSchema} of the dataset with field ids.
   *
   * @return the LanceSchema
   */
  public LanceSchema getLanceSchema() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetLanceSchema();
    }
  }

  private native LanceSchema nativeGetLanceSchema();

  /**
   * Get the {@link org.lance.Transaction} of the dataset at the current version.
   *
   * @return the Transaction
   */
  public Optional<Transaction> readTransaction() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return Optional.ofNullable(nativeReadTransaction());
    }
  }

  private native Transaction nativeReadTransaction();

  /**
   * Optimize index metadata and segments for this dataset.
   *
   * @param options options controlling index optimization behavior
   */
  public void optimizeIndices(OptimizeOptions options) {
    Preconditions.checkNotNull(options);
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeOptimizeIndices(options);
    }
  }

  private native void nativeOptimizeIndices(OptimizeOptions options);

  /**
   * @return all the created indexes names
   */
  public List<String> listIndexes() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeListIndexes();
    }
  }

  private native List<String> nativeListIndexes();

  /**
   * Get all indexes with full metadata.
   *
   * @return list of Index objects with complete metadata including index type and fragment coverage
   */
  public List<Index> getIndexes() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetIndexes();
    }
  }

  private native List<Index> nativeGetIndexes();

  /**
   * Get statistics for a specific index in JSON form.
   *
   * <p>The JSON structure matches the Rust/Python index_statistics API.
   *
   * @param indexName the name of the index
   * @return JSON string with index statistics
   */
  public Map<String, Object> getIndexStatistics(String indexName) {
    Preconditions.checkArgument(
        indexName != null && !indexName.isEmpty(), "indexName cannot be null or empty");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      String jsonDesc = nativeGetIndexStatistics(indexName);
      return JsonUtils.fromJson(jsonDesc);
    }
  }

  private native String nativeGetIndexStatistics(String indexName);

  /**
   * Describe indices on this dataset filtered by criteria.
   *
   * @param criteria filter options such as column, name or index capabilities
   * @return list of index descriptions
   */
  public List<IndexDescription> describeIndices(IndexCriteria criteria) {
    Preconditions.checkNotNull(criteria, "criteria cannot be null");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeDescribeIndices(Optional.of(criteria));
    }
  }

  /**
   * Describe all indices on this dataset.
   *
   * @return list of index descriptions
   */
  public List<IndexDescription> describeIndices() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeDescribeIndices(Optional.empty());
    }
  }

  private native List<IndexDescription> nativeDescribeIndices(Optional<IndexCriteria> criteria);

  /**
   * Read zonemap statistics for a column.
   *
   * <p>Returns per-zone min/max/null_count statistics for the given column, if a zonemap index
   * exists. Returns an empty list if no zonemap index exists for the column.
   *
   * <p>The zonemap index file is typically small (one row per zone), so this is a lightweight
   * metadata-only operation suitable for calling on the driver during scan planning.
   *
   * @param columnName the column name
   * @return list of per-zone statistics, ordered by (fragment_id, zone_start)
   */
  public List<ZoneStats> getZonemapStats(String columnName) {
    Preconditions.checkArgument(
        columnName != null && !columnName.isEmpty(), "columnName cannot be null or empty");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      byte[] ipcBytes = nativeGetZonemapStatsIpc(columnName);
      try {
        return ZonemapStatsCodec.decode(ipcBytes, allocator);
      } catch (IOException e) {
        throw new RuntimeException("Failed to decode zonemap stats IPC", e);
      }
    }
  }

  private native byte[] nativeGetZonemapStatsIpc(String columnName);

  /**
   * Get the table config of the dataset.
   *
   * @return the table config
   */
  public Map<String, String> getConfig() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetConfig();
    }
  }

  private native Map<String, String> nativeGetConfig();

  /**
   * Check whether the dataset uses stable row IDs.
   *
   * <p>Stable row IDs remain constant when rows are moved during compaction. This reads the
   * manifest feature flag directly rather than the user-facing config map.
   *
   * @return true if the dataset was created with stable row IDs enabled
   */
  public boolean hasStableRowIds() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeHasStableRowIds();
    }
  }

  private native boolean nativeHasStableRowIds();

  /**
   * Get the library version that wrote the current manifest.
   *
   * <p>Older manifests may not contain writer version metadata.
   *
   * @return the current manifest writer version, or empty if unavailable
   */
  public Optional<WriterVersion> getWriterVersion() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return Optional.ofNullable(nativeGetWriterVersion());
    }
  }

  private native WriterVersion nativeGetWriterVersion();

  /**
   * Get the Lance file format version of this dataset.
   *
   * <p>The returned string will be one of: "0.1" (legacy), "2.0", "2.1", or "2.2".
   *
   * @return the Lance file format version string
   */
  public String getLanceFileFormatVersion() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetLanceFileFormatVersion();
    }
  }

  private native String nativeGetLanceFileFormatVersion();

  /**
   * Compact the dataset to improve performance.
   *
   * <p>This operation performs several optimizations:
   *
   * <ul>
   *   <li>Removes deleted rows from fragments
   *   <li>Removes dropped columns from fragments
   *   <li>Merges fragments that are too small
   * </ul>
   *
   * @param options compaction options to control the behavior
   */
  public void compact(CompactionOptions options) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeCompact(options);
    }
  }

  /** Compact the dataset with default options. */
  public void compact() {
    compact(CompactionOptions.builder().build());
  }

  private native void nativeCompact(CompactionOptions options);

  /**
   * Update the config of the dataset. This operation will only overwrite and NOT delete the
   * existing config.
   *
   * @param tableConfig the config to update
   * @deprecated Use {@link #newTransactionBuilder()} with {@link UpdateConfig} operation instead
   */
  @Deprecated
  public void updateConfig(Map<String, String> tableConfig) {
    if (ContextIndexBuildProgress.isActive(this)) {
      throw new IllegalStateException("Dataset is busy in an index progress callback");
    }
    UpdateMap configUpdate = UpdateMap.builder().updates(tableConfig).replace(true).build();

    UpdateConfig operation = UpdateConfig.builder().configUpdates(configUpdate).build();

    Dataset newDataset = newTransactionBuilder().operation(operation).build().commit();
    updateToNewDataset(newDataset);
  }

  /**
   * Delete the config keys of the dataset.
   *
   * @param deleteKeys the config keys to delete
   * @deprecated Use {@link #newTransactionBuilder()} with {@link UpdateConfig} operation instead
   */
  @Deprecated
  public void deleteConfigKeys(Set<String> deleteKeys) {
    if (ContextIndexBuildProgress.isActive(this)) {
      throw new IllegalStateException("Dataset is busy in an index progress callback");
    }
    Map<String, String> deleteMap = new HashMap<>();
    deleteKeys.forEach(key -> deleteMap.put(key, null));
    UpdateMap configUpdate = UpdateMap.builder().updates(deleteMap).replace(false).build();

    UpdateConfig operation = UpdateConfig.builder().configUpdates(configUpdate).build();

    Dataset newDataset = newTransactionBuilder().operation(operation).build().commit();
    updateToNewDataset(newDataset);
  }

  /**
   * Updates the internal state of this dataset to match the provided new dataset. This is used by
   * deprecated void methods that need to update the current dataset instance.
   */
  private void updateToNewDataset(Dataset newDataset) {
    // Close the current handle to avoid resource leak
    close();

    // Replace all internal state with the new dataset
    this.nativeDatasetHandle = newDataset.nativeDatasetHandle;
    this.allocator = newDataset.allocator;
    this.selfManagedAllocator = newDataset.selfManagedAllocator;

    // Prevent the new dataset from closing the handle when it gets GC'd
    newDataset.nativeDatasetHandle = 0;
  }

  /**
   * Acquires a shared read lock that pins the native dataset handle, blocking a concurrent {@link
   * #close()} until the lock is released.
   *
   * <p>Any code that passes this {@link Dataset} into a native method must hold this lock for the
   * whole native call; otherwise {@code close()} can release the native dataset mid-call and crash
   * the JVM. The lock is reentrant and intended for try-with-resources use.
   *
   * @return the acquired read lock
   * @throws IllegalArgumentException if the dataset is already closed
   */
  public LockManager.ReadLock acquireReadLock() {
    LockManager.ReadLock readLock = lockManager.acquireReadLock();
    if (nativeDatasetHandle == 0) {
      readLock.close();
      throw new IllegalArgumentException("Dataset is closed");
    }
    return readLock;
  }

  /**
   * Closes this dataset and releases any system resources associated with it. If the dataset is
   * already closed, then invoking this method has no effect.
   */
  @Override
  public void close() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeDatasetHandle != 0) {
        releaseNativeDataset(nativeDatasetHandle);
        nativeDatasetHandle = 0;
      }
      if (selfManagedAllocator) {
        allocator.close();
      }
      if (ownsSession && session != null) {
        session.close();
        session = null;
        ownsSession = false;
      }
    }
  }

  /**
   * Native method to release the Lance dataset resources associated with the given handle.
   *
   * @param handle The native handle to the dataset resource.
   */
  private native void releaseNativeDataset(long handle);

  // ===== BlobFile / Blob dataset entry points (JNI) =====
  private native List<BlobFile> nativeTakeBlobs(List<Long> rowIds, String column);

  private native List<BlobFile> nativeTakeBlobsByIndices(List<Long> rowIndices, String column);

  /**
   * Open {@link BlobFile} handles for given row IDs on a blob column. Names and semantics align
   * with Rust/Python.
   *
   * <p>Pass logical row IDs read from {@code _rowid}, not physical row addresses from {@code
   * _rowaddr}.
   *
   * <pre>{@code
   * long rowId = 42L; // Example value from the _rowid column.
   * List<BlobFile> blobs = dataset.takeBlobs(List.of(rowId), "images");
   * for (BlobFile blob : blobs) {
   *   if (blob != null) {
   *     try (BlobFile file = blob) {
   *       byte[] data = file.read();
   *     }
   *   }
   * }
   * }</pre>
   *
   * @param rowIds logical row IDs from the {@code _rowid} column
   * @param column blob column name
   * @return one {@link BlobFile} per row ID; null blob values are represented by null elements
   */
  public List<BlobFile> takeBlobs(List<Long> rowIds, String column) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Preconditions.checkArgument(
          rowIds != null && !rowIds.isEmpty(), "rowIds cannot be null or empty");
      Preconditions.checkArgument(
          column != null && !column.isEmpty(), "column cannot be null or empty");
      return nativeTakeBlobs(rowIds, column);
    }
  }

  /**
   * Open {@link BlobFile} handles for given row indices on a blob column.
   *
   * <pre>{@code
   * List<BlobFile> blobs = dataset.takeBlobsByIndices(List.of(0L), "images");
   * for (BlobFile blob : blobs) {
   *   if (blob != null) {
   *     try (BlobFile file = blob) {
   *       byte[] data = file.read();
   *     }
   *   }
   * }
   * }</pre>
   *
   * @param rowIndices row offsets within dataset
   * @param column blob column name
   * @return one {@link BlobFile} per row index; null blob values are represented by null elements
   */
  public List<BlobFile> takeBlobsByIndices(List<Long> rowIndices, String column) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Preconditions.checkArgument(
          rowIndices != null && !rowIndices.isEmpty(), "rowIndices cannot be null or empty");
      Preconditions.checkArgument(
          column != null && !column.isEmpty(), "column cannot be null or empty");
      return nativeTakeBlobsByIndices(rowIndices, column);
    }
  }

  /**
   * Checks if the dataset is closed.
   *
   * @return true if the dataset is closed, false otherwise.
   */
  public boolean closed() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      return nativeDatasetHandle == 0;
    }
  }

  public Fragment getFragment(int fragmentId) {
    FragmentMetadata metadata = getFragmentNative(fragmentId);
    return new Fragment(this, metadata);
  }

  private native FragmentMetadata getFragmentNative(int fragmentId);

  /**
   * Returns a {@link Tags} instance for performing tag-related operations on the dataset.
   *
   * @return new {@code Tags} instance for dataset tag operations
   * @see Tags
   */
  public Tags tags() {
    return new Tags();
  }

  /** Branch operations aligned with Rust's Dataset branch APIs. */
  public Branches branches() {
    return new Branches();
  }

  /**
   * Create a branch at a specified version. The returned Dataset points to the created branch's
   * initial version.
   *
   * @param branch the branch name to create
   * @param ref the reference to create branch from
   * @return a new Dataset of the branch
   */
  public Dataset createBranch(String branch, Ref ref) {
    Preconditions.checkArgument(branch != null && ref != null, "branch and ref cannot be null");
    return innerCreateBranch(branch, ref, Optional.empty());
  }

  /**
   * Create a branch at a specified version. The returned Dataset points to the created branch's
   * initial version.
   *
   * @param branch the branch name to create
   * @param ref the reference to create branch from
   * @param storageOptions the storage options to create branch with
   * @return a new Dataset of the branch
   */
  public Dataset createBranch(String branch, Ref ref, Map<String, String> storageOptions) {
    Preconditions.checkArgument(branch != null && ref != null, "branch and ref cannot be null");
    Preconditions.checkArgument(
        storageOptions != null && !storageOptions.isEmpty(), "storageOptions cannot be null");
    return innerCreateBranch(branch, ref, Optional.of(storageOptions));
  }

  private Dataset innerCreateBranch(
      String branch, Ref ref, Optional<Map<String, String>> storageOptions) {
    Preconditions.checkArgument(branch != null, "Branch cannot be null");
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeCreateBranch(branch, ref, storageOptions);
    }
  }

  /**
   * Checkout using a unified {@link Ref} which can be a tag, the latest version on main/branch or a
   * specified (branch_name, version_number).
   *
   * @param ref the checkout reference
   * @return a new Dataset instance checked out to the specified reference
   */
  public Dataset checkout(Ref ref) {
    Preconditions.checkNotNull(ref);
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return initializeCheckoutDataset(nativeCheckout(ref));
    }
  }

  /**
   * Get the table metadata of the dataset.
   *
   * @return the table metadata as a map of key-value pairs
   */
  public Map<String, String> getTableMetadata() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeGetTableMetadata();
    }
  }

  private native Map<String, String> nativeGetTableMetadata();

  /** Tag operations of the dataset. */
  public class Tags {

    /**
     * Create a new tag on main branch. This is left for compatibility. We should use {@link
     * #create(String, Ref)} instead.
     *
     * @param tag the tag name
     * @param versionNumber the version number to tag
     */
    public void create(String tag, long versionNumber) {
      Preconditions.checkArgument(versionNumber > 0, "versionNumber must be greater than 0");
      create(tag, Ref.ofMain(versionNumber));
    }

    /**
     * Create a new tag on a specified branch.
     *
     * @param tag the tag name
     * @param ref the referenced version to tag
     */
    public void create(String tag, Ref ref) {
      Preconditions.checkArgument(tag != null, "Tag name cannot be null");
      Preconditions.checkArgument(ref != null, "ref cannot be null");
      try (LockManager.WriteLock readLock = lockManager.acquireWriteLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        nativeCreateTag(tag, ref);
      }
    }

    /**
     * Creates a new tag on the specified branch. This method will be removed in version 2.0.0. Use
     * {@link #create(String, Ref)} instead.
     *
     * @param tag the name of the tag to create
     * @param versionNumber the version number (or commit reference) to associate with the tag
     */
    @Deprecated
    public void create(String tag, long versionNumber, String targetBranch) {
      create(tag, Ref.ofBranch(targetBranch, versionNumber));
    }

    /**
     * Delete a tag from this dataset.
     *
     * @param tag the tag name
     */
    public void delete(String tag) {
      try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        nativeDeleteTag(tag);
      }
    }

    /**
     * Update a tag to a new version_number on main. This is left for compatibility. We should use
     * {@link #update(String, Ref)} instead.
     *
     * @param tag the tag name
     * @param versionNumber the versionNumber on main.
     */
    public void update(String tag, long versionNumber) {
      Preconditions.checkArgument(versionNumber > 0, "version_number must be greater than 0");
      nativeUpdateTag(tag, Ref.ofMain(versionNumber));
    }

    /**
     * Update a tag to a new reference.
     *
     * @param tag the tag name
     * @param ref the referenced version to tag
     */
    public void update(String tag, Ref ref) {
      Preconditions.checkArgument(tag != null, "tag cannot be null");
      Preconditions.checkArgument(ref != null, "ref cannot be null");
      try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        nativeUpdateTag(tag, ref);
      }
    }

    public void replaceMetadata(String tag, Map<String, String> metadata) {
      Preconditions.checkArgument(tag != null, "tag cannot be null");
      Preconditions.checkArgument(metadata != null, "metadata cannot be null");
      try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        nativeReplaceTagMetadata(tag, metadata);
      }
    }

    /**
     * List all tags of the dataset.
     *
     * @return a list of tags
     */
    public List<Tag> list() {
      try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        return nativeListTags();
      }
    }

    /**
     * Get the version of a tag in the dataset.
     *
     * @param tag the tag name
     * @return the version of the tag
     */
    public long getVersion(String tag) {
      try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        return nativeGetVersionByTag(tag);
      }
    }
  }

  /** Branch operations of the dataset. */
  public class Branches {

    /**
     * Delete a branch and its metadata.
     *
     * @param branchName the branch to delete
     */
    public void delete(String branchName) {
      try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        nativeDeleteBranch(branchName);
      }
    }

    /**
     * List all branches in this dataset.
     *
     * @return a list of Branch objects
     */
    public List<Branch> list() {
      try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        return nativeListBranches();
      }
    }

    public void replaceMetadata(String branchName, Map<String, String> metadata) {
      Preconditions.checkArgument(branchName != null, "branchName cannot be null");
      Preconditions.checkArgument(metadata != null, "metadata cannot be null");
      try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
        Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
        nativeReplaceBranchMetadata(branchName, metadata);
      }
    }
  }

  /**
   * Execute SQL query on the dataset. The underlying SQL engine is DataFusion. Please refer to the
   * DataFusion documentation for supported SQL syntax.
   *
   * @param sql SELECT statement to execute. The default FROM table name is `dataset`, for example:
   *     SELECT * FROM `dataset` LIMIT 10. If FROM table name is a custom value, the {@link
   *     SqlQuery#tableName(String)} should be invoked to set the custom table name.
   * @return a SqlQuery instance.
   */
  public SqlQuery sql(String sql) {
    return new SqlQuery(this, sql);
  }

  /**
   * Compute the delta between current version and this version.
   *
   * @param comparedAgainst the version to compare the current dataset against
   * @return a DatasetDelta view
   * @throws IllegalArgumentException if mutual exclusivity or completeness rules are violated
   */
  public DatasetDelta delta(long comparedAgainst) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeBuildDelta(Optional.of(comparedAgainst), Optional.empty(), Optional.empty());
    }
  }

  /**
   * Compute the delta between both {@code beginVersion} (exclusive) and {@code endVersion}
   * (inclusive).
   *
   * @param beginVersion the beginning version (exclusive) for explicit range
   * @param endVersion the ending version (inclusive) for explicit range
   * @return a DatasetDelta view
   * @throws IllegalArgumentException if mutual exclusivity or completeness rules are violated
   */
  public DatasetDelta delta(long beginVersion, long endVersion) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeBuildDelta(Optional.empty(), Optional.of(beginVersion), Optional.of(endVersion));
    }
  }

  private native DatasetDelta nativeBuildDelta(
      Optional<Long> comparedAgainst, Optional<Long> beginVersion, Optional<Long> endVersion);

  /**
   * Merge source data with the existing target data.
   *
   * <p>This will take in the source, merge it with the existing target data, and insert new rows,
   * update existing rows, and delete existing rows.
   *
   * <p>It is important that after merge insert, the current dataset is changed and should be
   * closed. The merged new dataset is contained in the MergeInsertResult.
   *
   * @param mergeInsert merge insert options
   * @param source ArrowArrayStream source data
   * @return MergeInsertResult containing the new merged Dataset.
   */
  public MergeInsertResult mergeInsert(MergeInsertParams mergeInsert, ArrowArrayStream source) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      MergeInsertResult result = nativeMergeInsert(mergeInsert, source.memoryAddress());

      Dataset newDataset = result.dataset();
      if (selfManagedAllocator) {
        newDataset.allocator = new RootAllocator(Long.MAX_VALUE);
      } else {
        newDataset.allocator = allocator;
      }

      return result;
    }
  }

  private native MergeInsertResult nativeMergeInsert(
      MergeInsertParams mergeInsert, long arrowStreamMemoryAddress);

  /**
   * Update column values for rows matching an optional predicate.
   *
   * <p>This is similar to SQL's {@code UPDATE} statement: the entries of {@link
   * UpdateParams#updates()} map target column names to SQL expressions evaluated for every row that
   * satisfies {@link UpdateParams#whereClause()}. If no predicate is provided, every row is
   * updated.
   *
   * <p>The predicate may reference dataset columns as well as the system columns {@code _rowid},
   * {@code _rowaddr}, and {@code _rowoffset}, allowing rows to be targeted by stable row id (e.g.
   * {@code "_rowid IN (1, 2, 3)"}).
   *
   * <p>This call does not mutate the current {@code Dataset} instance: it still references the
   * pre-update version. Callers should close this {@code Dataset} and switch to {@link
   * UpdateResult#getDataset()}, which holds the newly committed version.
   *
   * @param params update parameters
   * @return UpdateResult containing the new committed Dataset and the number of rows updated.
   */
  public UpdateResult update(UpdateParams params) {
    Preconditions.checkNotNull(params, "params must not be null");
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      UpdateResult result = nativeUpdate(params);

      Dataset newDataset = result.getDataset();
      if (selfManagedAllocator) {
        newDataset.allocator = new RootAllocator(Long.MAX_VALUE);
      } else {
        newDataset.allocator = allocator;
      }

      return result;
    }
  }

  private native UpdateResult nativeUpdate(UpdateParams params);

  /**
   * Initialize MemWAL on this dataset.
   *
   * <p>Must be called once before any call to {@link #memWalWriter}. Append-only tables may omit
   * primary-key metadata; primary keys are only required for primary-key lookup and last-write-wins
   * deduplication workflows.
   *
   * @param params MemWAL initialization parameters
   */
  public void initializeMemWal(InitializeMemWalParams params) {
    Preconditions.checkNotNull(params, "params must not be null");
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      nativeInitializeMemWal(params);
    }
  }

  private native void nativeInitializeMemWal(InitializeMemWalParams params);

  /**
   * Return the MemWAL index details for this dataset, or {@link Optional#empty()} if MemWAL has not
   * been initialized.
   *
   * @return the MemWAL index details, if any
   */
  public Optional<MemWalIndexDetails> memWalIndexDetails() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return Optional.ofNullable(nativeMemWalIndexDetails());
    }
  }

  private native MemWalIndexDetails nativeMemWalIndexDetails();

  /**
   * Get a {@link ShardWriter} for the specified shard.
   *
   * <p>{@link #initializeMemWal} must be called before using this method.
   *
   * @param shardId UUID string identifying the write shard
   * @param config writer configuration; pass {@code null} to use the Lance default configuration
   * @return a ShardWriter for the shard
   */
  public ShardWriter memWalWriter(String shardId, ShardWriterConfig config) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return ShardWriter.create(this, shardId, config);
    }
  }

  /** Get a {@link ShardWriter} for the specified shard using the Lance default configuration. */
  public ShardWriter memWalWriter(String shardId) {
    return memWalWriter(shardId, null);
  }

  private native void nativeCreateTag(String tag, Ref ref);

  private native void nativeDeleteTag(String tag);

  private native void nativeUpdateTag(String tag, Ref ref);

  private native void nativeReplaceTagMetadata(String tag, Map<String, String> metadata);

  private native List<Tag> nativeListTags();

  private native long nativeGetVersionByTag(String tag);

  // ===== Branch native methods =====
  private native Dataset nativeCheckout(Ref ref);

  private native Dataset nativeCreateBranch(
      String branch, Ref ref, Optional<Map<String, String>> storageOptions);

  private native void nativeDeleteBranch(String branch);

  private native List<Branch> nativeListBranches();

  private native void nativeReplaceBranchMetadata(String branch, Map<String, String> metadata);

  public Dataset shallowClone(String targetPath, Ref ref) {
    return shallowClone(targetPath, ref, null);
  }

  /**
   * Shallow clone the specified tag into a new dataset at the target path.
   *
   * <p>This creates a new dataset that references the data files from the source dataset without
   * copying them. Only metadata is written at the destination.
   *
   * @param targetPath the URI to clone the dataset into
   * @param ref the referred version of the current dataset
   * @param storageOptions Optional object store options for the destination dataset; empty uses
   *     default store parameters
   * @return a new Dataset instance at the target path
   */
  public Dataset shallowClone(String targetPath, Ref ref, Map<String, String> storageOptions) {
    Preconditions.checkArgument(targetPath != null, "Target path can not be null");
    Preconditions.checkArgument(ref != null, "globalVersion can not be null");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      Dataset newDataset = nativeShallowClone(targetPath, ref, Optional.ofNullable(storageOptions));
      if (selfManagedAllocator) {
        newDataset.allocator = new RootAllocator(Long.MAX_VALUE);
      } else {
        newDataset.allocator = allocator;
      }
      return newDataset;
    }
  }

  private native Dataset nativeShallowClone(
      String targetPath, Ref ref, Optional<Map<String, String>> storageOptions);

  /**
   * Create a cleanup operation for the specified policy.
   *
   * <p>Use {@link CleanupOperation#explain()} to inspect what cleanup would remove without deleting
   * files, or {@link CleanupOperation#execute()} to perform cleanup.
   *
   * @param policy cleanup policy
   * @return cleanup operation
   */
  public CleanupOperation cleanup(CleanupPolicy policy) {
    Preconditions.checkNotNull(policy, "policy cannot be null");
    return new CleanupOperation(this, policy);
  }

  /**
   * Cleanup dataset based on a specified policy.
   *
   * @param policy cleanup policy
   * @return removal stats
   */
  public RemovalStats cleanupWithPolicy(CleanupPolicy policy) {
    return cleanup(policy).execute();
  }

  RemovalStats executeCleanup(CleanupPolicy policy) {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeCleanupWithPolicy(policy);
    }
  }

  private native RemovalStats nativeCleanupWithPolicy(CleanupPolicy policy);

  CleanupExplanation explainCleanup(CleanupPolicy policy, Optional<Long> maxCandidateFiles) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDatasetHandle != 0, "Dataset is closed");
      return nativeExplainCleanupWithPolicy(policy, maxCandidateFiles);
    }
  }

  private native CleanupExplanation nativeExplainCleanupWithPolicy(
      CleanupPolicy policy, Optional<Long> maxCandidateFiles);
}
