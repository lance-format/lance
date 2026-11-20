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
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;
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
  private final ConcurrentHashMap<Long, CompletableFuture<ArrowReader>> pendingTasks =
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
              options.getFragmentSlices(),
              options.getIndexSegments(),
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
              options.getIgnoredScalarIndices(),
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
      Optional<List<FragmentSlice>> fragmentSlices,
      Optional<List<UUID>> indexSegments,
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
      Optional<List<String>> ignoredScalarIndices,
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
      CompletableFuture<ArrowReader> future = new CompletableFuture<>();
      pendingTasks.put(taskId, future);
      future.whenComplete(
          (reader, error) -> {
            if (future.isCancelled()) {
              try (LockManager.ReadLock ignored = lockManager.acquireReadLock()) {
                if (pendingTasks.remove(taskId, future) && nativeAsyncScannerHandle != 0) {
                  nativeCancelTask(taskId);
                }
              }
            }
          });
      try {
        nativeStartScan(taskId);
      } catch (Throwable error) {
        pendingTasks.remove(taskId);
        future.completeExceptionally(error);
      }
      return future;
    }
  }

  /** Called by Rust dispatcher thread via JNI to complete a task successfully. */
  private boolean completeTask(long taskId, long resultPtr) {
    CompletableFuture<ArrowReader> future = pendingTasks.get(taskId);
    if (future == null) {
      return false;
    }
    if (future.isDone()) {
      pendingTasks.remove(taskId, future);
      return false;
    }
    try (ArrowArrayStream stream = ArrowArrayStream.wrap(resultPtr)) {
      ArrowReader reader = Data.importArrayStream(allocator, stream);
      if (!future.complete(reader)) {
        reader.close();
      }
    } catch (Throwable error) {
      future.completeExceptionally(
          error instanceof Error
              ? error
              : new LanceException("Failed to import scan stream", error));
    } finally {
      pendingTasks.remove(taskId, future);
    }
    return true;
  }

  /** Called by Rust dispatcher thread via JNI to fail a task with an error. */
  private void failTask(long taskId, String errorMessage) {
    CompletableFuture<ArrowReader> future = pendingTasks.remove(taskId);
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
    List<CompletableFuture<ArrowReader>> cancelled = new ArrayList<>();
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeAsyncScannerHandle != 0) {
        // Cancel all pending tasks
        for (Long taskId : pendingTasks.keySet()) {
          nativeCancelTask(taskId);
        }
        releaseNativeScanner();
        nativeAsyncScannerHandle = 0;
      }
      cancelled.addAll(pendingTasks.values());
      pendingTasks.clear();
    }
    cancelled.forEach(future -> future.cancel(false));
  }

  private native void nativeCancelTask(long taskId);

  /** Native method to release the async scanner resources. */
  private native void releaseNativeScanner();
}
