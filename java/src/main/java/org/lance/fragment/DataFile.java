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

import com.google.common.base.MoreObjects;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;

public class DataFile implements Serializable {
  private static final long serialVersionUID = -2827710928026343591L;
  private final String path;
  private final int[] fields;
  private final int[] columnIndices;
  private final int fileMajorVersion;
  private final int fileMinorVersion;
  private final Long fileSizeBytes;
  private final Integer baseId;
  private final Long fileMetadataSizeBytes;

  public DataFile(
      String path,
      int[] fields,
      int[] columnIndices,
      int fileMajorVersion,
      int fileMinorVersion,
      Long fileSizeBytes,
      Integer baseId) {
    this(
        path,
        fields,
        columnIndices,
        fileMajorVersion,
        fileMinorVersion,
        fileSizeBytes,
        baseId,
        null);
  }

  /**
   * Creates data file metadata with an optional exact metadata suffix size.
   *
   * @param path path to the data file
   * @param fields field IDs stored in the file
   * @param columnIndices physical column indices for the fields
   * @param fileMajorVersion major storage format version
   * @param fileMinorVersion minor storage format version
   * @param fileSizeBytes total file size, or null if unknown
   * @param baseId base path ID, or null for the dataset base
   * @param fileMetadataSizeBytes metadata suffix size; zero or null if unknown
   */
  public DataFile(
      String path,
      int[] fields,
      int[] columnIndices,
      int fileMajorVersion,
      int fileMinorVersion,
      Long fileSizeBytes,
      Integer baseId,
      Long fileMetadataSizeBytes) {
    if (fileMetadataSizeBytes != null && fileMetadataSizeBytes < 0) {
      throw new IllegalArgumentException("fileMetadataSizeBytes must be non-negative");
    }
    this.path = path;
    this.fields = fields;
    this.columnIndices = columnIndices;
    this.fileMajorVersion = fileMajorVersion;
    this.fileMinorVersion = fileMinorVersion;
    this.fileSizeBytes = fileSizeBytes;
    this.baseId = baseId;
    this.fileMetadataSizeBytes =
        fileMetadataSizeBytes != null && fileMetadataSizeBytes == 0 ? null : fileMetadataSizeBytes;
  }

  public String getPath() {
    return path;
  }

  public int[] getFields() {
    return fields;
  }

  public int[] getColumnIndices() {
    return columnIndices;
  }

  public int getFileMajorVersion() {
    return fileMajorVersion;
  }

  public int getFileMinorVersion() {
    return fileMinorVersion;
  }

  public Long getFileSizeBytes() {
    return fileSizeBytes;
  }

  public Optional<Integer> getBaseId() {
    return Optional.ofNullable(baseId);
  }

  /** Returns the exact metadata suffix size, or null if it is unavailable. */
  public Long getFileMetadataSizeBytes() {
    return fileMetadataSizeBytes;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    DataFile that = (DataFile) o;
    return fileMajorVersion == that.fileMajorVersion
        && fileMinorVersion == that.fileMinorVersion
        && Objects.equals(path, that.path)
        && Arrays.equals(fields, that.fields)
        && Arrays.equals(columnIndices, that.columnIndices)
        && Objects.equals(fileSizeBytes, that.fileSizeBytes)
        && Objects.equals(fileMetadataSizeBytes, that.fileMetadataSizeBytes);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("path", path)
        .add("fields", fields)
        .add("columnIndices", columnIndices)
        .add("fileMajorVersion", fileMajorVersion)
        .add("fileMinorVersion", fileMinorVersion)
        .add("fileSizeBytes", fileSizeBytes)
        .add("baseId", baseId)
        .add("fileMetadataSizeBytes", fileMetadataSizeBytes)
        .toString();
  }
}
