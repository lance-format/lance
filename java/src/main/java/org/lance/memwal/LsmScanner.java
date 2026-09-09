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
package org.lance.memwal;

import org.lance.Dataset;
import org.lance.JniLoader;
import org.lance.LockManager;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.ipc.ArrowReader;

import java.io.Closeable;
import java.util.List;
import java.util.Optional;

/**
 * LSM-aware scanner covering all MemWAL data levels.
 *
 * <p>Results are deduplicated by primary key, always returning the newest version of each row
 * across the base table, SSTables, and (when created from a {@link ShardWriter}) the active
 * MemTable.
 *
 * <p>The builder methods ({@link #project}, {@link #filter}, {@link #limit}, {@link
 * #withRowAddress}, {@link #withMemtableGen}) mutate this scanner and return it for chaining.
 */
public class LsmScanner implements Closeable {
  static {
    JniLoader.ensureLoaded();
  }

  private long nativeLsmScannerHandle;
  BufferAllocator allocator;
  private final LockManager lockManager = new LockManager();

  private LsmScanner() {}

  /**
   * Create a scanner from a dataset and shard snapshots.
   *
   * <p>The scanner does not include any active MemTable; use {@link ShardWriter#lsmScanner} for
   * read-your-writes consistency.
   *
   * @param dataset the base dataset to scan
   * @param shardSnapshots shard snapshots specifying the SSTables to include
   * @return an LSM scanner
   */
  public static LsmScanner fromSnapshots(Dataset dataset, List<ShardSnapshot> shardSnapshots) {
    Preconditions.checkNotNull(dataset, "dataset must not be null");
    Preconditions.checkNotNull(shardSnapshots, "shardSnapshots must not be null");
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      LsmScanner scanner = createFromSnapshots(dataset, shardSnapshots);
      scanner.allocator = dataset.allocator();
      return scanner;
    }
  }

  static native LsmScanner createFromSnapshots(Dataset dataset, List<ShardSnapshot> shardSnapshots);

  /** Select specific columns to return. */
  public LsmScanner project(List<String> columns) {
    Preconditions.checkNotNull(columns, "columns must not be null");
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeLsmScannerHandle != 0, "LsmScanner is closed");
      nativeProject(columns);
      return this;
    }
  }

  private native void nativeProject(List<String> columns);

  /**
   * Set a SQL filter expression, for example {@code "value > 0.5"}.
   *
   * <p>If {@code expr} fails to parse, an {@link IllegalArgumentException} is thrown and this
   * scanner becomes unusable — build a new scanner rather than retrying on this instance.
   *
   * @param expr a SQL filter expression
   * @return this scanner
   */
  public LsmScanner filter(String expr) {
    Preconditions.checkNotNull(expr, "expr must not be null");
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeLsmScannerHandle != 0, "LsmScanner is closed");
      nativeFilter(expr);
      return this;
    }
  }

  private native void nativeFilter(String expr);

  /** Limit the number of rows returned, optionally skipping {@code offset} rows first. */
  public LsmScanner limit(Optional<Long> limit, Optional<Long> offset) {
    Preconditions.checkNotNull(limit, "limit must not be null");
    Preconditions.checkNotNull(offset, "offset must not be null");
    limit.ifPresent(
        l -> Preconditions.checkArgument(l >= 0, "limit must not be negative, got %s", l));
    offset.ifPresent(
        o -> Preconditions.checkArgument(o >= 0, "offset must not be negative, got %s", o));
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeLsmScannerHandle != 0, "LsmScanner is closed");
      nativeLimit(limit, offset);
      return this;
    }
  }

  /** Limit the number of rows returned, optionally skipping {@code offset} rows first. */
  public LsmScanner limit(long limit, Optional<Long> offset) {
    return limit(Optional.of(limit), offset);
  }

  /** Limit the number of rows returned. */
  public LsmScanner limit(long limit) {
    return limit(limit, Optional.empty());
  }

  private native void nativeLimit(Optional<Long> limit, Optional<Long> offset);

  /** Include the {@code _rowaddr} internal column in results. */
  public LsmScanner withRowAddress() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeLsmScannerHandle != 0, "LsmScanner is closed");
      nativeWithRowAddress();
      return this;
    }
  }

  private native void nativeWithRowAddress();

  /** Include the {@code _memtable_gen} internal column in results. */
  public LsmScanner withMemtableGen() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkArgument(nativeLsmScannerHandle != 0, "LsmScanner is closed");
      nativeWithMemtableGen();
      return this;
    }
  }

  private native void nativeWithMemtableGen();

  /** Execute the scan and return a streaming reader over the merged results. */
  public ArrowReader scanBatches() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeLsmScannerHandle != 0, "LsmScanner is closed");
      try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
        nativeOpenStream(stream.memoryAddress());
        return Data.importArrayStream(allocator, stream);
      }
    }
  }

  private native void nativeOpenStream(long streamAddress);

  /** Return the row count without materializing all column data. */
  public long countRows() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeLsmScannerHandle != 0, "LsmScanner is closed");
      return nativeCountRows();
    }
  }

  private native long nativeCountRows();

  /**
   * Close the scanner and release native resources. If the scanner is already closed, invoking this
   * method has no effect.
   */
  @Override
  public void close() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeLsmScannerHandle != 0) {
        releaseNativeLsmScanner(nativeLsmScannerHandle);
        nativeLsmScannerHandle = 0;
      }
    }
  }

  private native void releaseNativeLsmScanner(long handle);
}
