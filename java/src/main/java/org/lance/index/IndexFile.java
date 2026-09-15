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
package org.lance.index;

import com.google.common.base.MoreObjects;

import java.util.Objects;

/** Metadata for a file within an index segment. */
public final class IndexFile {
  private final String path;
  private final long sizeBytes;
  private final Long fileMetadataSizeBytes;

  public IndexFile(String path, long sizeBytes) {
    this(path, sizeBytes, null);
  }

  /**
   * Creates index file metadata with an optional exact metadata suffix size.
   *
   * @param path path relative to the index directory
   * @param sizeBytes total file size
   * @param fileMetadataSizeBytes metadata suffix size; zero or null if unavailable
   */
  public IndexFile(String path, long sizeBytes, Long fileMetadataSizeBytes) {
    this.path = Objects.requireNonNull(path, "path must not be null");
    if (sizeBytes < 0) {
      throw new IllegalArgumentException("sizeBytes must be non-negative");
    }
    if (fileMetadataSizeBytes != null && fileMetadataSizeBytes < 0) {
      throw new IllegalArgumentException("fileMetadataSizeBytes must be non-negative");
    }
    this.sizeBytes = sizeBytes;
    this.fileMetadataSizeBytes =
        fileMetadataSizeBytes != null && fileMetadataSizeBytes == 0 ? null : fileMetadataSizeBytes;
  }

  public String getPath() {
    return path;
  }

  public long getSizeBytes() {
    return sizeBytes;
  }

  /** Returns the exact metadata suffix size, or null if it is unavailable. */
  public Long getFileMetadataSizeBytes() {
    return fileMetadataSizeBytes;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    IndexFile indexFile = (IndexFile) o;
    return sizeBytes == indexFile.sizeBytes
        && Objects.equals(path, indexFile.path)
        && Objects.equals(fileMetadataSizeBytes, indexFile.fileMetadataSizeBytes);
  }

  @Override
  public int hashCode() {
    return Objects.hash(path, sizeBytes, fileMetadataSizeBytes);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("path", path)
        .add("sizeBytes", sizeBytes)
        .add("fileMetadataSizeBytes", fileMetadataSizeBytes)
        .toString();
  }
}
