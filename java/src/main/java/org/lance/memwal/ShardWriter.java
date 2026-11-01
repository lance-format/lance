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
import java.util.Collections;
import java.util.List;

/**
 * Stateful writer for one MemWAL shard.
 *
 * <p>Obtain an instance via {@link Dataset#memWalWriter}. The writer must be closed when no longer
 * needed; using it inside a try-with-resources block is recommended:
 *
 * <pre>{@code
 * try (ShardWriter writer = dataset.memWalWriter(shardId)) {
 *   writer.put(reader);
 *   writer.delete(keys);
 * }
 * }</pre>
 *
 * <p>{@link #close()} flushes pending data and releases native resources. Statistics must be read
 * before the writer is closed.
 */
public class ShardWriter implements Closeable {
  static {
    JniLoader.ensureLoaded();
  }

  private long nativeShardWriterHandle;
  private BufferAllocator allocator;
  private final LockManager lockManager = new LockManager();

  private ShardWriter() {}

  /**
   * Create a writer for {@code shardId} on the given dataset.
   *
   * @param dataset the dataset MemWAL has been initialized on
   * @param shardId UUID string identifying the write shard
   * @param config writer configuration; pass {@code null} to use the Lance default configuration
   * @return a new ShardWriter
   */
  public static ShardWriter create(Dataset dataset, String shardId, ShardWriterConfig config) {
    Preconditions.checkNotNull(dataset, "dataset must not be null");
    Preconditions.checkNotNull(shardId, "shardId must not be null");
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      ShardWriter writer = createNative(dataset, shardId, config);
      writer.allocator = dataset.allocator();
      return writer;
    }
  }

  static native ShardWriter createNative(Dataset dataset, String shardId, ShardWriterConfig config);

  /** UUID string for this writer's shard. */
  public String shardId() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeShardWriterHandle != 0, "ShardWriter is closed");
      return nativeShardId();
    }
  }

  private native String nativeShardId();

  /**
   * Write data to the MemWAL.
   *
   * @param reader the source data; consumed fully by this call
   */
  public void put(ArrowReader reader) {
    Preconditions.checkNotNull(reader, "reader must not be null");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeShardWriterHandle != 0, "ShardWriter is closed");
      try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader, stream);
        nativePut(stream.memoryAddress());
      }
    }
  }

  private native void nativePut(long streamAddress);

  /**
   * Delete rows from the MemWAL by primary key.
   *
   * <p>Each batch in {@code reader} must carry this shard's primary key column(s); other columns
   * are ignored. Lance builds a tombstone row per key — the primary key plus {@code _tombstone =
   * true} and null in every other column — and appends it like an ordinary write. The tombstone is
   * the newest value for its key: it wins newest-per-PK resolution (suppressing the older real row)
   * and is then dropped from query results.
   *
   * <p>Only supported in memtable mode. Because a tombstone nulls every non-PK column, those
   * columns must be nullable in the base schema; deleting against a schema with a non-nullable
   * non-PK column errors. Deleting on a shard with no primary key columns also errors.
   *
   * @param reader the keys to delete; consumed fully by this call
   */
  public void delete(ArrowReader reader) {
    Preconditions.checkNotNull(reader, "reader must not be null");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeShardWriterHandle != 0, "ShardWriter is closed");
      try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader, stream);
        nativeDelete(stream.memoryAddress());
      }
    }
  }

  private native void nativeDelete(long streamAddress);

  /** Return a snapshot of cumulative write statistics. */
  public WriteStats stats() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeShardWriterHandle != 0, "ShardWriter is closed");
      return nativeStats();
    }
  }

  private native WriteStats nativeStats();

  /** Return current statistics of the active MemTable. */
  public MemTableStats memtableStats() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeShardWriterHandle != 0, "ShardWriter is closed");
      return nativeMemtableStats();
    }
  }

  private native MemTableStats nativeMemtableStats();

  /**
   * Create an LSM scanner that includes this writer's active MemTable.
   *
   * <p>The scanner covers the base table, the given SSTables, and the current active MemTable,
   * providing read-your-writes consistency. This writer's own shard is included automatically.
   *
   * @param shardSnapshots snapshots of other shards to include
   * @return an LSM scanner
   */
  public LsmScanner lsmScanner(List<ShardSnapshot> shardSnapshots) {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeShardWriterHandle != 0, "ShardWriter is closed");
      LsmScanner scanner =
          nativeLsmScanner(shardSnapshots == null ? Collections.emptyList() : shardSnapshots);
      scanner.allocator = allocator;
      return scanner;
    }
  }

  /** Create an LSM scanner that includes this writer's active MemTable. */
  public LsmScanner lsmScanner() {
    return lsmScanner(Collections.emptyList());
  }

  private native LsmScanner nativeLsmScanner(List<ShardSnapshot> shardSnapshots);

  /**
   * Flush pending data and close the writer, releasing native resources. If the writer is already
   * closed, invoking this method has no effect.
   */
  @Override
  public void close() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeShardWriterHandle != 0) {
        releaseNativeShardWriter(nativeShardWriterHandle);
        nativeShardWriterHandle = 0;
      }
    }
  }

  private native void releaseNativeShardWriter(long handle);
}
