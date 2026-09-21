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
package org.lance.file;

import org.lance.JniLoader;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.dictionary.DictionaryProvider;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.Optional;

public class LanceFileWriter implements AutoCloseable {

  static {
    JniLoader.ensureLoaded();
  }

  private long nativeFileWriterHandle;
  private BufferAllocator allocator;
  private DictionaryProvider dictionaryProvider;

  private static native LanceFileWriter openNative(
      String fileUri,
      Optional<String> dataStorageVersion,
      Map<String, String> storageOptions,
      FileWriteOptions writeOptions)
      throws IOException;

  private native void closeNative(long nativeLanceFileReaderHandle) throws IOException;

  private native void writeNative(long batchMemoryAddress, long schemaMemoryAddress)
      throws IOException;

  private LanceFileWriter() {}

  /**
   * Open a LanceFileWriter to write to a given file URI
   *
   * @param path the URI of the file to write to
   * @param allocator the BufferAllocator to use for the writer
   * @param dictionaryProvider the DictionaryProvider to use for the writer
   * @return a new LanceFileWriter
   */
  public static LanceFileWriter open(
      String path, BufferAllocator allocator, DictionaryProvider dictionaryProvider)
      throws IOException {
    return open(path, allocator, dictionaryProvider, Collections.emptyMap());
  }

  /**
   * Open a LanceFileWriter to write to a given file URI
   *
   * @param path the URI of the file to write to
   * @param allocator the BufferAllocator to use for the writer
   * @param dictionaryProvider the DictionaryProvider to use for the writer
   * @param storageOptions additional storage options for the writer
   * @return a new LanceFileWriter
   */
  public static LanceFileWriter open(
      String path,
      BufferAllocator allocator,
      DictionaryProvider dictionaryProvider,
      Map<String, String> storageOptions)
      throws IOException {
    return open(path, allocator, dictionaryProvider, Optional.empty(), storageOptions);
  }

  /**
   * Open a LanceFileWriter to write to a given file URI
   *
   * @param path the URI of the file to write to
   * @param allocator the BufferAllocator to use for the writer
   * @param dictionaryProvider the DictionaryProvider to use for the writer
   * @param dataStorageVersion the version of the data storage format to use (e.g., "legacy",
   *     "stable", "2.0")
   * @return a new LanceFileWriter
   */
  public static LanceFileWriter open(
      String path,
      BufferAllocator allocator,
      DictionaryProvider dictionaryProvider,
      Optional<String> dataStorageVersion,
      Map<String, String> storageOptions)
      throws IOException {
    return open(
        path,
        allocator,
        dictionaryProvider,
        dataStorageVersion,
        storageOptions,
        FileWriteOptions.builder().build());
  }

  /**
   * Open a LanceFileWriter with explicit file write options.
   *
   * @param path the URI of the file to write to
   * @param allocator the BufferAllocator to use for the writer
   * @param dictionaryProvider the DictionaryProvider to use for the writer
   * @param dataStorageVersion the version of the data storage format to use
   * @param storageOptions additional storage options for the writer
   * @param writeOptions options for configuring the current-format file writer
   * @return a new LanceFileWriter
   */
  public static LanceFileWriter open(
      String path,
      BufferAllocator allocator,
      DictionaryProvider dictionaryProvider,
      Optional<String> dataStorageVersion,
      Map<String, String> storageOptions,
      FileWriteOptions writeOptions)
      throws IOException {
    Preconditions.checkNotNull(writeOptions, "writeOptions must not be null");
    LanceFileWriter writer = openNative(path, dataStorageVersion, storageOptions, writeOptions);
    writer.allocator = allocator;
    writer.dictionaryProvider = dictionaryProvider;
    return writer;
  }

  /**
   * Write a batch of data
   *
   * @param batch the batch of data to write
   * @throws IOException if the batch cannot be written
   */
  public void write(VectorSchemaRoot batch) throws IOException {
    try (ArrowArray ffiArrowArray = ArrowArray.allocateNew(allocator);
        ArrowSchema ffiArrowSchema = ArrowSchema.allocateNew(allocator)) {
      Data.exportVectorSchemaRoot(
          allocator, batch, dictionaryProvider, ffiArrowArray, ffiArrowSchema);
      writeNative(ffiArrowArray.memoryAddress(), ffiArrowSchema.memoryAddress());
    }
  }

  /**
   * Add a schema metadata map to underlying file, the provided key-value pairs will override
   * existing ones with the same keys. User can retrieve those values from {@link
   * LanceFileReader#schema() reader schema}.
   *
   * <p>Note that this method does not write metadata to underlying file immediately. These metadata
   * will be maintained in an in-memory hashmap, and be flushed to file footer on close.
   *
   * @param metadata metadata
   * @throws IOException IOException
   */
  public void addSchemaMetadata(Map<String, String> metadata) throws IOException {
    nativeAddSchemaMetadata(metadata);
  }

  private native void nativeAddSchemaMetadata(Map<String, String> metadata) throws IOException;

  /**
   * Close the LanceFileWriter
   *
   * <p>This method must be called to release resources when the writer is no longer needed.
   *
   * <p>This method will also flush all remaining data and write the footer to the file.
   *
   * @throws Exception if the writer cannot be closed
   */
  @Override
  public void close() throws Exception {
    closeNative(nativeFileWriterHandle);
  }
}
