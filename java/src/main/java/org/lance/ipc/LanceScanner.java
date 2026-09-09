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
import org.apache.arrow.dataset.scanner.ScanTask;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;

/** Scanner over a Fragment. */
public class LanceScanner implements org.apache.arrow.dataset.scanner.Scanner {
  Dataset dataset;

  ScanOptions options;

  BufferAllocator allocator;

  private long nativeScannerHandle;

  private final LockManager lockManager = new LockManager();

  private LanceScanner() {}

  /**
   * Create a Scanner.
   *
   * @param dataset the dataset to scan
   * @param options scan options
   * @param allocator allocator
   * @return a Scanner
   */
  public static LanceScanner create(
      Dataset dataset, ScanOptions options, BufferAllocator allocator) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(options);
    Preconditions.checkNotNull(allocator);
    LanceScanner scanner;
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      scanner =
          createScanner(
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
              options.isCollectStats(),
              options.isIncludeDeletedRows(),
              options.isStrictBatchSize(),
              options.isDisableScoringAutoprojection());
    }
    scanner.allocator = allocator;
    scanner.dataset = dataset;
    scanner.options = options;
    return scanner;
  }

  static native LanceScanner createScanner(
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
      boolean collectStats,
      boolean includeDeletedRows,
      boolean strictBatchSize,
      boolean disableScoringAutoprojection);

  /**
   * Closes this scanner and releases any system resources associated with it. If the scanner is
   * already closed, then invoking this method has no effect.
   */
  @Override
  public void close() throws Exception {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeScannerHandle != 0) {
        releaseNativeScanner(nativeScannerHandle);
        nativeScannerHandle = 0;
      }
    }
  }

  /**
   * Native method to release the Lance scanner resources associated with the given handle.
   *
   * @param handle The native handle to the scanner resource.
   */
  private native void releaseNativeScanner(long handle);

  @Override
  public ArrowReader scanBatches() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeScannerHandle != 0, "Scanner is closed");
      try (ArrowArrayStream s = ArrowArrayStream.allocateNew(allocator)) {
        openStream(s.memoryAddress());
        return Data.importArrayStream(allocator, s);
      } catch (IOException e) {
        throw new LanceException("Failed to open scan stream", e);
      }
    }
  }

  /**
   * Export this scan's results into a caller-owned Arrow C stream identified by its memory address,
   * using the Arrow C Data Interface release callback to transfer ownership.
   *
   * <p>This method intentionally takes a raw {@code streamAddress} (an {@code ArrowArrayStream}
   * memory address) rather than a Java {@link ArrowArrayStream} object. A typed parameter would be
   * an {@code org.apache.arrow.c.ArrowArrayStream} loaded by <em>Lance's</em> classloader / Arrow
   * version; a caller running a different Arrow version (or under a different classloader, e.g.
   * Spark + a native engine bundling its own Arrow) cannot construct that exact type and would hit
   * a {@code ClassCastException}/{@code NoSuchMethodError} at the very boundary this method exists
   * to cross. The C Data Interface ABI is stable across Arrow versions, so passing the C struct's
   * address keeps the two sides fully decoupled: the caller allocates the stream with <em>its
   * own</em> Arrow runtime and only the {@code long} address crosses into Lance. See gluten#12263
   * for the cross-Arrow-version integration that motivated this.
   *
   * <p>Unlike {@link #scanBatches()}, no Java Arrow {@link ArrowReader} is created on Lance's side:
   * Lance writes the C struct directly at {@code streamAddress} and the caller drives the read loop
   * with its own Arrow runtime.
   *
   * <p>The {@code streamAddress} must point to a freshly-allocated, empty {@code ArrowArrayStream}
   * (its {@code release} callback must be null). Exporting into a stream that already holds a
   * producer is rejected with an {@link IllegalArgumentException}, because overwriting the struct
   * would drop the existing {@code release} callback and leak the first producer. The caller owns
   * the stream and is responsible for closing it; the release callback installed by this call
   * routes back through Lance's native side.
   *
   * <p>The provided stream must not be shared across concurrent exports. An {@code
   * ArrowArrayStream} is a plain C struct in caller-owned memory with no internal synchronization,
   * so a single stream must be exported into, then drained, by one thread at a time. The
   * already-populated check above guards the sequential "export twice" mistake, but it cannot make
   * two concurrent exports into the same struct safe — that is a caller-side data race on
   * caller-owned memory, the same contract as Arrow's C Data Interface itself. Use a separate
   * stream per concurrent export.
   *
   * <p>Example (caller on its own Arrow version / allocator):
   *
   * <pre>{@code
   * try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(callerAllocator)) {
   *   scanner.exportArrowStream(stream.memoryAddress());
   *   try (ArrowReader reader = Data.importArrayStream(callerAllocator, stream)) {
   *     while (reader.loadNextBatch()) {
   *       VectorSchemaRoot batch = reader.getVectorSchemaRoot();
   *       // ...
   *     }
   *   }
   * }
   * }</pre>
   *
   * @param streamAddress the memory address of a freshly-allocated, empty {@code ArrowArrayStream}
   *     to populate
   * @throws IllegalArgumentException if the scanner is closed or the stream is already populated
   * @throws IOException if the native scan fails to start
   */
  public void exportArrowStream(long streamAddress) throws IOException {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeScannerHandle != 0, "Scanner is closed");
      openStream(streamAddress);
    }
  }

  private native void openStream(long streamAddress) throws IOException;

  @Override
  public Iterable<? extends ScanTask> scan() {
    // Marked as deprecated in Scanner
    throw new UnsupportedOperationException();
  }

  @Override
  public Schema schema() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeScannerHandle != 0, "Scanner is closed");
      try (ArrowSchema ffiArrowSchema = ArrowSchema.allocateNew(allocator)) {
        importFfiSchema(ffiArrowSchema.memoryAddress());
        return Data.importSchema(allocator, ffiArrowSchema, null);
      }
    }
  }

  private native void importFfiSchema(long arrowSchemaMemoryAddress);

  /**
   * Scan and return the number of matching rows (fulfill the given scan options).
   *
   * @return num of rows.
   */
  public long countRows() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeScannerHandle != 0, "Scanner is closed");
      return nativeCountRows();
    }
  }

  /**
   * Get scan execution statistics for the most recently executed scan.
   *
   * <p>Statistics are only collected when {@link ScanOptions.Builder#collectStats(boolean)} is set
   * to true. The values are populated when the scan stream is fully consumed and closed.
   *
   * <p>If statistics collection is disabled, or the scan has not finished yet, this method returns
   * {@link Optional#empty()}.
   */
  public Optional<ScanStats> getStats() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeScannerHandle != 0, "Scanner is closed");
      return Optional.ofNullable(nativeGetStats());
    }
  }

  private native ScanStats nativeGetStats();

  private native long nativeCountRows();
}
