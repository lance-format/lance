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

import java.io.Closeable;
import java.io.IOException;

/**
 * BlobFile: a file-like abstraction for a single blob, aligned with Lance Rust/Python semantics.
 * Read-only: methods include read, readUpTo, readRange, seek, tell, size, close. All delegate to
 * JNI.
 *
 * <p>Usage example (try-with-resources for automatic resource management):
 *
 * <pre>{@code
 * try (BlobFile blob = dataset.takeBlob()) {
 *     long size = blob.size();
 *     byte[] data = blob.read(); // Read all content
 *     blob.seek(1024);
 *     byte[] partial = blob.readUpTo(512); // Read 512 bytes from position 1024
 * } catch (IOException e) {
 *     // Handle IO errors
 * }
 * }</pre>
 *
 * Auto-closes after try block, no manual close needed.
 */
public final class BlobFile implements Closeable {
  static {
    JniLoader.ensureLoaded();
  }

  /** Default sequential read-ahead size in bytes. */
  public static final int DEFAULT_READ_BUFFER_SIZE = 512 * 1024;

  /** Opaque native handle managed by lance-jni. */
  @SuppressWarnings("FieldCanBeLocal")
  private long nativeBlobHandle;

  /** Default no-arg constructor used by JNI to attach native handle. */
  public BlobFile() {}

  /** Read all remaining bytes from current cursor to end. */
  public byte[] read() throws IOException {
    return nativeRead();
  }

  /** Read up to len bytes from the current cursor. */
  public byte[] readUpTo(int len) throws IOException {
    if (len < 0) throw new IllegalArgumentException("len must be non-negative");
    return nativeReadUpTo(len);
  }

  /** Read a blob-local range without changing the current cursor. */
  public byte[] readRange(long offset, int len) throws IOException {
    if (offset < 0) throw new IllegalArgumentException("offset must be non-negative");
    if (len < 0) throw new IllegalArgumentException("len must be non-negative");
    return nativeReadRange(offset, len);
  }

  /** Seek to a new cursor position. */
  public void seek(long newCursor) throws IOException {
    if (newCursor < 0) throw new IllegalArgumentException("newCursor must be non-negative");
    nativeSeek(newCursor);
  }

  /**
   * Sequential read-ahead size in bytes. {@code 0} disables read-ahead. Range reads do not use this
   * buffer.
   */
  public void setReadBufferSize(long bufferSize) throws IOException {
    if (bufferSize < 0) {
      throw new IllegalArgumentException("bufferSize must be non-negative");
    }
    nativeSetReadBufferSize(bufferSize);
  }

  /** Return current cursor position. */
  public long tell() throws IOException {
    return nativeTell();
  }

  /** Return blob size in bytes. */
  public long size() {
    return nativeSize();
  }

  /** Close BlobFile and release associated resources. */
  @Override
  public void close() throws IOException {
    nativeClose();
  }

  // ===== JNI bindings =====
  private native byte[] nativeRead() throws IOException;

  private native byte[] nativeReadUpTo(int len) throws IOException;

  private native byte[] nativeReadRange(long offset, int len) throws IOException;

  private native void nativeSeek(long newCursor) throws IOException;

  private native void nativeSetReadBufferSize(long bufferSize) throws IOException;

  private native long nativeTell() throws IOException;

  private native long nativeSize();

  private native void nativeClose() throws IOException;
}
