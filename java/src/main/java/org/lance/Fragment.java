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

import org.lance.fragment.FragmentMergeResult;
import org.lance.fragment.FragmentUpdateResult;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;
import org.lance.namespace.LanceNamespace;
import org.lance.schema.LanceSchema;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.VectorSchemaRoot;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Fragment operations. */
public class Fragment {
  static {
    JniLoader.ensureLoaded();
  }

  /** Pointer to the {@link Dataset} instance in Java. */
  private final Dataset dataset;

  private final FragmentMetadata fragmentMetadata;

  public Fragment(Dataset dataset, int fragmentId) {
    Preconditions.checkNotNull(dataset);
    this.dataset = dataset;
    this.fragmentMetadata = dataset.getFragment(fragmentId).fragmentMetadata;
  }

  public Fragment(Dataset dataset, FragmentMetadata fragmentMetadata) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(fragmentMetadata);
    this.dataset = dataset;
    this.fragmentMetadata = fragmentMetadata;
  }

  public FragmentMetadata metadata() {
    return fragmentMetadata;
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @return a dataset scanner
   */
  public LanceScanner newScan() {
    return LanceScanner.create(
        dataset,
        new ScanOptions.Builder().fragmentIds(Arrays.asList(fragmentMetadata.getId())).build(),
        dataset.allocator());
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @param batchSize scan batch size
   * @return a dataset scanner
   */
  public LanceScanner newScan(long batchSize) {
    return LanceScanner.create(
        dataset,
        new ScanOptions.Builder()
            .fragmentIds(Arrays.asList(fragmentMetadata.getId()))
            .batchSize(batchSize)
            .build(),
        dataset.allocator());
  }

  /**
   * Create a new Dataset Scanner.
   *
   * @param options the scan options
   * @return a dataset scanner
   */
  public LanceScanner newScan(ScanOptions options) {
    Preconditions.checkNotNull(options);
    return LanceScanner.create(
        dataset,
        new ScanOptions.Builder(options)
            .fragmentIds(Arrays.asList(fragmentMetadata.getId()))
            .build(),
        dataset.allocator());
  }

  /**
   * Delete rows by row indexes.
   *
   * @param rowIndexes The row indexes to delete.
   * @return The fragment metadata after deletion. If all rows are deleted, return Null. Otherwise,
   *     returns a new fragment with the updated deletion vector.
   */
  public FragmentMetadata deleteRows(List<Integer> rowIndexes) {
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return nativeDeleteRows(dataset, fragmentMetadata.getId(), rowIndexes);
    }
  }

  private static native FragmentMetadata nativeDeleteRows(
      Dataset dataset, int fragmentId, List<Integer> rowIndexes);

  private native int countRowsNative(Dataset dataset, long fragmentId);

  public int getId() {
    return fragmentMetadata.getId();
  }

  /**
   * @return row counts in this Fragment
   */
  public int countRows() {
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return countRowsNative(dataset, fragmentMetadata.getId());
    }
  }

  /**
   * Merge the new columns into this Fragment, will return the new fragment with the same
   * FragmentId. This operation will perform a left-join with the right table (new data in stream)
   * on the column specified by leftOn and rightOn. For every row in current fragment, the new
   * column value is:
   *
   * <ol>
   *   <li>if no matched row on the right side, null value.
   *   <li>if there is exactly one corresponding row on the right side, column value of the matching
   *       row.
   *   <li>if there are multiple corresponding rows, column value of a random row.
   * </ol>
   *
   * The returned Result will be further committed.
   *
   * @param stream the input data stream
   * @param leftOn column name of current fragment to be joined on.
   * @param rightOn column name of new data to be joined on.
   * @return the fragment metadata and new schema.
   */
  public FragmentMergeResult mergeColumns(ArrowArrayStream stream, String leftOn, String rightOn) {
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return nativeMergeColumns(
          dataset, fragmentMetadata.getId(), stream.memoryAddress(), leftOn, rightOn);
    }
  }

  private native FragmentMergeResult nativeMergeColumns(
      Dataset dataset,
      long fragmentId,
      long arrowStreamMemoryAddress,
      String leftOn,
      String rightOn);

  /**
   * Update existed columns into this Fragment, will return the new fragment with the same
   * FragmentId and updated field ids. This operation will perform a left-outer-hash-join with the
   * right table (new data in stream) on the column specified by leftOn and rightOn. For every row
   * in current fragment, the updated column value is:
   *
   * <ol>
   *   <li>if no matched row on the right side, column value of the left side row.
   *   <li>if there is exactly one corresponding row on the right side, column value of the matching
   *       row.
   *   <li>if there are multiple corresponding rows, column value of a random row.
   * </ol>
   *
   * The returned Result will be further committed.
   *
   * @param stream the input data stream
   * @param leftOn column name of current fragment to be joined on.
   * @param rightOn column name of new data to be joined on.
   * @return the fragment metadata.
   */
  public FragmentUpdateResult updateColumns(
      ArrowArrayStream stream, String leftOn, String rightOn) {
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return nativeUpdateColumns(
          dataset, fragmentMetadata.getId(), stream.memoryAddress(), leftOn, rightOn);
    }
  }

  public FragmentUpdateResult updateColumns(ArrowArrayStream stream) {
    return updateColumns(stream, MetadataColumns.ROWID, MetadataColumns.ROWID);
  }

  private native FragmentUpdateResult nativeUpdateColumns(
      Dataset dataset,
      long fragmentId,
      long arrowStreamMemoryAddress,
      String leftOn,
      String rightOn);

  /**
   * Create a new fragment writer builder.
   *
   * <p>Example usage:
   *
   * <pre>{@code
   * List<FragmentMetadata> fragments = Fragment.write()
   *     .datasetUri("s3://bucket/dataset.lance")
   *     .allocator(allocator)
   *     .data(vectorSchemaRoot)
   *     .storageOptions(storageOptions)
   *     .execute();
   * }</pre>
   *
   * @return a new fragment writer builder
   */
  public static WriteFragmentBuilder write() {
    return new WriteFragmentBuilder();
  }

  /**
   * Create a fragment from the given data.
   *
   * @param datasetUri the dataset uri
   * @param allocator the buffer allocator
   * @param root the vector schema root
   * @param params the write params
   * @return the fragment metadata
   * @deprecated Use {@link #write()} builder instead. For example: {@code Fragment.write()
   *     .datasetUri(uri).allocator(allocator).data(root).writeParams(params).execute()}
   */
  @Deprecated
  public static List<FragmentMetadata> create(
      String datasetUri, BufferAllocator allocator, VectorSchemaRoot root, WriteParams params) {
    return create(datasetUri, allocator, root, params, null, null);
  }

  /**
   * Create a fragment from the given arrow stream.
   *
   * @param datasetUri the dataset uri
   * @param stream the arrow stream
   * @param params the write params
   * @return the fragment metadata
   * @deprecated Use {@link #write()} builder instead.
   */
  @Deprecated
  public static List<FragmentMetadata> create(
      String datasetUri, ArrowArrayStream stream, WriteParams params) {
    return create(datasetUri, stream, params, null, null);
  }

  /** Create a fragment from the given arrow array and schema. */
  static List<FragmentMetadata> create(
      String datasetUri,
      BufferAllocator allocator,
      VectorSchemaRoot root,
      WriteParams params,
      LanceNamespace namespaceClient,
      List<String> tableId) {
    return create(datasetUri, allocator, root, params, namespaceClient, tableId, null, null);
  }

  /** Create a fragment from the given arrow array and schema. */
  static List<FragmentMetadata> create(
      String datasetUri,
      BufferAllocator allocator,
      VectorSchemaRoot root,
      WriteParams params,
      LanceNamespace namespaceClient,
      List<String> tableId,
      LanceSchema schema,
      Session session) {
    Preconditions.checkNotNull(datasetUri);
    Preconditions.checkNotNull(allocator);
    Preconditions.checkNotNull(root);
    Preconditions.checkNotNull(params);
    long sessionHandle = getSessionHandle(session);
    try (ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator);
        ArrowArray arrowArray = ArrowArray.allocateNew(allocator)) {
      Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchema);
      if (schema != null) {
        try (ArrowSchema lanceSchema = ArrowSchema.allocateNew(allocator)) {
          Data.exportSchema(allocator, schema.asArrowSchemaWithFieldIds(), null, lanceSchema);
          return createWithFfiArray(
              datasetUri,
              arrowArray.memoryAddress(),
              arrowSchema.memoryAddress(),
              params.getMaxRowsPerFile(),
              params.getMaxRowsPerGroup(),
              params.getMaxBytesPerFile(),
              params.getMode(),
              params.getEnableStableRowIds(),
              params.getDataStorageVersion(),
              params.getStorageOptions(),
              params.getBaseStoreParams(),
              params.getInitialBases(),
              params.getTargetBases(),
              namespaceClient,
              tableId,
              params.getAllowExternalBlobOutsideBases(),
              params.getBlobPackFileSizeThreshold(),
              lanceSchema.memoryAddress(),
              sessionHandle);
        }
      }
      return createWithFfiArray(
          datasetUri,
          arrowArray.memoryAddress(),
          arrowSchema.memoryAddress(),
          params.getMaxRowsPerFile(),
          params.getMaxRowsPerGroup(),
          params.getMaxBytesPerFile(),
          params.getMode(),
          params.getEnableStableRowIds(),
          params.getDataStorageVersion(),
          params.getStorageOptions(),
          params.getBaseStoreParams(),
          params.getInitialBases(),
          params.getTargetBases(),
          namespaceClient,
          tableId,
          params.getAllowExternalBlobOutsideBases(),
          params.getBlobPackFileSizeThreshold(),
          0L,
          sessionHandle);
    }
  }

  /** Create a fragment from the given arrow stream. */
  static List<FragmentMetadata> create(
      String datasetUri,
      ArrowArrayStream stream,
      WriteParams params,
      LanceNamespace namespaceClient,
      List<String> tableId) {
    return create(datasetUri, null, stream, params, namespaceClient, tableId, null, null);
  }

  /** Create a fragment from the given arrow stream. */
  static List<FragmentMetadata> create(
      String datasetUri,
      BufferAllocator allocator,
      ArrowArrayStream stream,
      WriteParams params,
      LanceNamespace namespaceClient,
      List<String> tableId,
      LanceSchema schema,
      Session session) {
    Preconditions.checkNotNull(datasetUri);
    Preconditions.checkNotNull(stream);
    Preconditions.checkNotNull(params);
    long sessionHandle = getSessionHandle(session);
    if (schema != null) {
      Preconditions.checkNotNull(allocator, "allocator is required with schema");
      try (ArrowSchema lanceSchema = ArrowSchema.allocateNew(allocator)) {
        Data.exportSchema(allocator, schema.asArrowSchemaWithFieldIds(), null, lanceSchema);
        return createWithFfiStream(
            datasetUri,
            stream.memoryAddress(),
            params.getMaxRowsPerFile(),
            params.getMaxRowsPerGroup(),
            params.getMaxBytesPerFile(),
            params.getMode(),
            params.getEnableStableRowIds(),
            params.getDataStorageVersion(),
            params.getStorageOptions(),
            params.getBaseStoreParams(),
            params.getInitialBases(),
            params.getTargetBases(),
            namespaceClient,
            tableId,
            params.getAllowExternalBlobOutsideBases(),
            params.getBlobPackFileSizeThreshold(),
            lanceSchema.memoryAddress(),
            sessionHandle);
      }
    }
    return createWithFfiStream(
        datasetUri,
        stream.memoryAddress(),
        params.getMaxRowsPerFile(),
        params.getMaxRowsPerGroup(),
        params.getMaxBytesPerFile(),
        params.getMode(),
        params.getEnableStableRowIds(),
        params.getDataStorageVersion(),
        params.getStorageOptions(),
        params.getBaseStoreParams(),
        params.getInitialBases(),
        params.getTargetBases(),
        namespaceClient,
        tableId,
        params.getAllowExternalBlobOutsideBases(),
        params.getBlobPackFileSizeThreshold(),
        0L,
        sessionHandle);
  }

  /**
   * Resolves the native handle of an optional session. A closed session has a zero handle and is
   * treated as absent, matching how Dataset handles closed sessions.
   */
  private static long getSessionHandle(Session session) {
    return session == null ? 0L : session.getNativeHandle();
  }

  /** Create a fragment from the given arrow array and schema. */
  private static native List<FragmentMetadata> createWithFfiArray(
      String datasetUri,
      long arrowArrayMemoryAddress,
      long arrowSchemaMemoryAddress,
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<String> mode,
      Optional<Boolean> enableStableRowIds,
      Optional<String> dataStorageVersion,
      Map<String, String> storageOptions,
      Map<String, Map<String, String>> baseStoreParams,
      Optional<List<BasePath>> initialBases,
      Optional<List<String>> targetBases,
      LanceNamespace namespaceClient,
      List<String> tableId,
      Optional<Boolean> allowExternalBlobOutsideBases,
      Optional<Long> blobPackFileSizeThreshold,
      long schemaMemoryAddress,
      long sessionHandle);

  /** Create a fragment from the given arrow stream. */
  private static native List<FragmentMetadata> createWithFfiStream(
      String datasetUri,
      long arrowStreamMemoryAddress,
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<String> mode,
      Optional<Boolean> enableStableRowIds,
      Optional<String> dataStorageVersion,
      Map<String, String> storageOptions,
      Map<String, Map<String, String>> baseStoreParams,
      Optional<List<BasePath>> initialBases,
      Optional<List<String>> targetBases,
      LanceNamespace namespaceClient,
      List<String> tableId,
      Optional<Boolean> allowExternalBlobOutsideBases,
      Optional<Long> blobPackFileSizeThreshold,
      long schemaMemoryAddress,
      long sessionHandle);
}
