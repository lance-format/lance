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
package org.lance.delta;

import org.lance.Dataset;
import org.lance.JniLoader;
import org.lance.LockManager;
import org.lance.Transaction;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.ipc.ArrowReader;

import java.io.Closeable;
import java.io.IOException;
import java.util.List;

/**
 * A view of differences between two versions of a dataset.
 *
 * <p>Created by {@link DatasetDeltaBuilder}. Provides methods to list transactions and stream
 * inserted/updated rows between two versions.
 */
public class DatasetDelta implements Closeable {
  static {
    JniLoader.ensureLoaded();
  }

  /** Native handle to the Rust DatasetDelta. */
  private long nativeDeltaHandle;

  /** Base dataset used to compute the delta. Also used for Transaction conversion. */
  private Dataset dataset;

  private final LockManager lockManager = new LockManager();

  private DatasetDelta() {}

  /**
   * List transactions between begin_version + 1 and end_version (inclusive).
   *
   * @return list of transactions
   */
  public List<Transaction> listTransactions() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDeltaHandle != 0, "DatasetDelta is closed");
      return nativeListTransactions();
    }
  }

  private native List<Transaction> nativeListTransactions();

  /** Return a streaming ArrowReader for inserted rows. */
  public ArrowReader getInsertedRows() throws IOException {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDeltaHandle != 0, "DatasetDelta is closed");
      BufferAllocator allocator = dataset.allocator();
      try (ArrowArrayStream s = ArrowArrayStream.allocateNew(allocator)) {
        nativeGetInsertedRows(s.memoryAddress());
        return Data.importArrayStream(allocator, s);
      }
    }
  }

  private native void nativeGetInsertedRows(long streamAddress) throws IOException;

  /** Return a streaming ArrowReader for updated rows. */
  public ArrowReader getUpdatedRows() throws IOException {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDeltaHandle != 0, "DatasetDelta is closed");
      BufferAllocator allocator = dataset.allocator();
      try (ArrowArrayStream s = ArrowArrayStream.allocateNew(allocator)) {
        nativeGetUpdatedRows(s.memoryAddress());
        return Data.importArrayStream(allocator, s);
      }
    }
  }

  private native void nativeGetUpdatedRows(long streamAddress) throws IOException;

  /**
   * Return a streaming ArrowReader of the row ids deleted in the range.
   *
   * <p>The batches carry a single {@code _rowid} column. Requires stable row ids.
   */
  public ArrowReader getDeletedRowIds() throws IOException {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(nativeDeltaHandle != 0, "DatasetDelta is closed");
      BufferAllocator allocator = dataset.allocator();
      try (ArrowArrayStream s = ArrowArrayStream.allocateNew(allocator)) {
        nativeGetDeletedRowIds(s.memoryAddress());
        return Data.importArrayStream(allocator, s);
      }
    }
  }

  private native void nativeGetDeletedRowIds(long streamAddress) throws IOException;

  @Override
  public void close() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeDeltaHandle != 0) {
        releaseNativeDelta(nativeDeltaHandle);
        nativeDeltaHandle = 0;
      }
    }
  }

  private native void releaseNativeDelta(long handle);
}
