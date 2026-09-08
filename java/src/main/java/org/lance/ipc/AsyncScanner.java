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
package org.lance.ipc;

import org.lance.Dataset;
import org.lance.LanceException;
import org.lance.LockManager;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Async scanner that provides non-blocking scan operations via CompletableFuture.
 *
 * <p>This scanner spawns async I/O tasks in Rust and completes Java futures when data is ready,
 * preventing thread starvation in Java query engines like Presto/Trino.
 */
public class AsyncScanner implements AutoCloseable {
  private static final AtomicLong TASK_ID_GENERATOR = new AtomicLong(1);
  private final ConcurrentHashMap<Long, CompletableFuture<Long>> pendingTasks =
      new ConcurrentHashMap<>();

  private BufferAllocator allocator;
  private final LockManager lockManager = new LockManager();
  private long nativeAsyncScannerHandle;

  private AsyncScanner() {}

  /**
   * Create an AsyncScanner.
   *
   * @param dataset the dataset to scan
   * @param options scan options
   * @param allocator allocator
   * @return an AsyncScanner
   */
  public static AsyncScanner create(
      Dataset dataset, ScanOptions options, BufferAllocator allocator) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(options);
    Preconditions.checkNotNull(allocator);
    AsyncScanner scanner;
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      scanner =
          createAsyncScanner(
              dataset,
              options.getFragmentIds(),
              options.getColumns(),
              options.getSubstraitFilter(),
              options.getFilter(),
              options.getBatchSize(),
              options.getBatchSizeBytes(),
              options.getIoBufferSize(),
              options.getLimit(),
              options.getOffset(),
              options.getNearest(),
              options.getFullTextQuery(),
              options.isPrefilter(),
              options.isWithRowId(),
              options.isWithRowAddress(),
              options.getBatchReadahead(),
              options.getFragmentReadahead(),
              options.isScanInOrder(),
              options.getLateMaterialization(),
              options.getColumnOrderings(),
              options.isUseScalarIndex(),
              options.isFastSearch(),
              options.getSubstraitAggregate(),
              options.isIncludeDeletedRows(),
              options.isStrictBatchSize(),
              options.isDisableScoringAutoprojection());
    }
    scanner.allocator = allocator;
    return scanner;
  }

  static native AsyncScanner createAsyncScanner(
      Dataset dataset,
      Optional<List<Integer>> fragmentIds,
      Optional<List<String>> columns,
      Optional<ByteBuffer> substraitFilter,
      Optional<String> filter,
      Optional<Long> batchSize,
      Optional<Long> batchSizeBytes,
      Optional<Long> ioBufferSize,
      Optional<Long> limit,
      Optional<Long> offset,
      Optional<Query> query,
      Optional<FullTextQuery> fullTextQuery,
      boolean prefilter,
      boolean withRowId,
      boolean withRowAddress,
      int batchReadahead,
      Optional<Integer> fragmentReadahead,
      boolean scanInOrder,
      Optional<MaterializationStyle> lateMaterialization,
      Optional<List<ColumnOrdering>> columnOrderings,
      boolean useScalarIndex,
      boolean fastSearch,
      Optional<ByteBuffer> substraitAggregate,
      boolean includeDeletedRows,
      boolean strictBatchSize,
      boolean disableScoringAutoprojection);

  /**
   * Asynchronously scan batches and return a CompletableFuture.
   *
   * @return a CompletableFuture that will be completed with an ArrowReader when data is ready
   */
  public CompletableFuture<ArrowReader> scanBatchesAsync() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      if (nativeAsyncScannerHandle == 0) {
        CompletableFuture<ArrowReader> future = new CompletableFuture<>();
        future.completeExceptionally(new IllegalStateException("Scanner is closed"));
        return future;
      }

      long taskId = TASK_ID_GENERATOR.getAndIncrement();
      CompletableFuture<Long> streamPtrFuture = new CompletableFuture<>();
      pendingTasks.put(taskId, streamPtrFuture);

      // Start async scan in Rust
      nativeStartScan(taskId);

      // Transform stream pointer to ArrowReader
      return streamPtrFuture.handle(
          (streamPtr, error) -> {
            pendingTasks.remove(taskId);

            if (error != null) {
              throw new LanceException("Scan failed", error);
            }

            if (streamPtr < 0) {
              throw new LanceException("Native scan returned an invalid stream pointer");
            }

            try {
              ArrowArrayStream stream = ArrowArrayStream.wrap(streamPtr);
              return Data.importArrayStream(allocator, stream);
            } catch (Exception e) {
              throw new LanceException("Failed to import scan stream", e);
            }
          });
    }
  }

  /** Called by Rust dispatcher thread via JNI to complete a task successfully. */
  private void completeTask(long taskId, long resultPtr) {
    CompletableFuture<Long> future = pendingTasks.get(taskId);
    if (future != null) {
      future.complete(resultPtr);
    }
  }

  /** Called by Rust dispatcher thread via JNI to fail a task with an error. */
  private void failTask(long taskId, String errorMessage) {
    CompletableFuture<Long> future = pendingTasks.get(taskId);
    if (future != null) {
      future.completeExceptionally(new LanceException(errorMessage));
    }
  }

  private native void nativeStartScan(long taskId);

  /**
   * Get schema (synchronous operation).
   *
   * @return the schema
   */
  public Schema schema() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeAsyncScannerHandle != 0, "Scanner is closed");
      try (ArrowSchema ffiSchema = ArrowSchema.allocateNew(allocator)) {
        importFfiSchema(ffiSchema.memoryAddress());
        return Data.importSchema(allocator, ffiSchema, null);
      }
    }
  }

  private native void importFfiSchema(long arrowSchemaMemoryAddress);

  /**
   * Closes this scanner and releases any system resources associated with it. If the scanner is
   * already closed, then invoking this method has no effect.
   */
  @Override
  public void close() throws Exception {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeAsyncScannerHandle != 0) {
        // Cancel all pending tasks
        for (Long taskId : pendingTasks.keySet()) {
          nativeCancelTask(taskId);
        }
        pendingTasks.clear();

        releaseNativeScanner();
        nativeAsyncScannerHandle = 0;
      }
    }
  }

  private native void nativeCancelTask(long taskId);

  /** Native method to release the async scanner resources. */
  private native void releaseNativeScanner();
}
