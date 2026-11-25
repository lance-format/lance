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
package org.lance.fragment;

import org.lance.Fragment;
import org.lance.operation.DataOverlay;
import org.lance.operation.DataOverlay.DataOverlayGroup;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.dictionary.DictionaryProvider;

import java.io.IOException;
import java.util.Optional;

/**
 * Stages a data overlay for one fragment. Open it with {@link Fragment#writeOverlay}. The staged
 * file becomes visible only when its group is committed with {@link DataOverlay}.
 */
public final class OverlayWriter implements AutoCloseable {
  private long nativeOverlayWriterHandle;

  private OverlayWriter() {}

  /**
   * Stage a batch with no dictionary-encoded vectors. See {@link #writeBatch(VectorSchemaRoot,
   * DictionaryProvider)} for batch requirements and failure handling.
   *
   * @param batch row addresses and the replacement values
   * @throws IOException if staging the batch fails
   */
  public void writeBatch(VectorSchemaRoot batch) throws IOException {
    writeBatch(batch, null);
  }

  /**
   * Stage new values from one batch. Each batch must carry a non-null UInt64 {@code _rowaddr}
   * column with strictly increasing addresses belonging to this fragment, plus a subset of the
   * declared columns. Addresses must also strictly increase across batches for each value column. A
   * null value replaces the previous cell with null. Omit a row or column from the batch to leave
   * its cells untouched.
   *
   * <p>All batch vectors with an allocator must share a root allocator. The batch can use its own
   * root allocator, independent of the dataset.
   *
   * <p>If {@code writeBatch} throws, some values from the failed batch may already be staged. Call
   * {@link #abort()} or {@link #close()} to discard the staged overlay. Do not call {@link
   * #finish()} after a write failure.
   *
   * @param batch row addresses and the replacement values
   * @param dictionaryProvider dictionaries for encoded vectors, or null when none are used
   * @throws IOException if staging the batch fails
   */
  public synchronized void writeBatch(VectorSchemaRoot batch, DictionaryProvider dictionaryProvider)
      throws IOException {
    checkOpen();
    // Check before Arrow export can retain buffers from earlier vectors.
    BufferAllocator allocator = null;
    for (FieldVector vector : batch.getFieldVectors()) {
      BufferAllocator vectorAllocator = vector.getAllocator();
      if (vectorAllocator == null) {
        continue; // Null vectors have no buffers to retain.
      }
      BufferAllocator root = vectorAllocator.getRoot();
      if (allocator == null) {
        allocator = root;
      } else if (root != allocator) {
        throw new IllegalArgumentException(
            "Overlay batch vectors must share the same root allocator. Column '"
                + vector.getName()
                + "' uses a different root");
      }
    }
    if (allocator == null) {
      throw new IllegalArgumentException("Overlay batch must contain a '_rowaddr' column");
    }
    // Allocate temporary buffers at the root to avoid charging the export to a child allocator.
    try (ArrowArray array = ArrowArray.allocateNew(allocator);
        ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
      try {
        Data.exportVectorSchemaRoot(allocator, batch, dictionaryProvider, array, schema);
        nativeWriteBatch(array.memoryAddress(), schema.memoryAddress());
      } finally {
        // The native import moves the release callbacks out of these structs.
        if (array.snapshot().release != 0) {
          array.release();
        }
        if (schema.snapshot().release != 0) {
          schema.release();
        }
      }
    }
  }

  /**
   * Finish staging the overlay and close the writer. Call this only after all batches have been
   * staged successfully.
   *
   * @return the group to commit with the read version used to open this writer, or empty if no
   *     cells were staged
   * @throws IOException if the overlay file cannot be finalized
   */
  public synchronized Optional<DataOverlayGroup> finish() throws IOException {
    checkOpen();
    return Optional.ofNullable(nativeFinish());
  }

  /** Close the writer and discard any unfinished overlay on a best-effort basis. */
  public synchronized void abort() {
    if (nativeOverlayWriterHandle != 0) {
      nativeAbort();
    }
  }

  /** Discard an unfinished overlay. Closing an already closed writer has no effect. */
  @Override
  public void close() {
    abort();
  }

  private void checkOpen() {
    if (nativeOverlayWriterHandle == 0) {
      throw new IllegalStateException("Overlay writer is closed");
    }
  }

  private native void nativeWriteBatch(long batchAddress, long schemaAddress) throws IOException;

  private native DataOverlayGroup nativeFinish() throws IOException;

  private native void nativeAbort();
}
