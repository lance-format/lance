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

import org.lance.fragment.DataFile;
import org.lance.fragment.DeletionFile;
import org.lance.fragment.RowIdMeta;
import org.lance.fragment.VersionMeta;

import com.google.common.base.MoreObjects;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Metadata of a Fragment in the dataset. Matching to lance Fragment. */
public class FragmentMetadata implements Serializable {
  private static final long serialVersionUID = -5886811251944130460L;
  private final int id;
  private final List<DataFile> files;
  private final List<DataFile> referencedLanceFiles;
  private final long physicalRows;
  private final DeletionFile deletionFile;
  private final RowIdMeta rowIdMeta;
  private final VersionMeta createdAtVersionMeta;
  private final VersionMeta lastUpdatedAtVersionMeta;
  private final Long clusteringId;
  private final Map<String, String> metadata;

  public FragmentMetadata(
      int id,
      List<DataFile> files,
      Long physicalRows,
      DeletionFile deletionFile,
      RowIdMeta rowIdMeta) {
    this(id, files, physicalRows, deletionFile, rowIdMeta, null, null, files);
  }

  public FragmentMetadata(
      int id,
      List<DataFile> files,
      Long physicalRows,
      DeletionFile deletionFile,
      RowIdMeta rowIdMeta,
      VersionMeta createdAtVersionMeta,
      VersionMeta lastUpdatedAtVersionMeta) {
    this(
        id,
        files,
        physicalRows,
        deletionFile,
        rowIdMeta,
        createdAtVersionMeta,
        lastUpdatedAtVersionMeta,
        files);
  }

  public FragmentMetadata(
      int id,
      List<DataFile> files,
      Long physicalRows,
      DeletionFile deletionFile,
      RowIdMeta rowIdMeta,
      VersionMeta createdAtVersionMeta,
      VersionMeta lastUpdatedAtVersionMeta,
      List<DataFile> referencedLanceFiles) {
    this(
        id,
        files,
        physicalRows,
        deletionFile,
        rowIdMeta,
        createdAtVersionMeta,
        lastUpdatedAtVersionMeta,
        referencedLanceFiles,
        null,
        Collections.emptyMap());
  }

  /**
   * Full constructor, including the clustering reference and generic metadata.
   *
   * @param clusteringId the clustering definition this fragment was organized under, or null; a
   *     non-negative value within the unsigned 32-bit range
   * @param metadata generic metadata attached to this fragment, preserved verbatim through
   *     operations on the fragment (including keys this library does not understand); null is
   *     treated as empty
   */
  public FragmentMetadata(
      int id,
      List<DataFile> files,
      Long physicalRows,
      DeletionFile deletionFile,
      RowIdMeta rowIdMeta,
      VersionMeta createdAtVersionMeta,
      VersionMeta lastUpdatedAtVersionMeta,
      List<DataFile> referencedLanceFiles,
      Long clusteringId,
      Map<String, String> metadata) {
    this.id = id;
    this.files = files;
    this.referencedLanceFiles = referencedLanceFiles;
    this.physicalRows = physicalRows;
    this.deletionFile = deletionFile;
    this.rowIdMeta = rowIdMeta;
    this.createdAtVersionMeta = createdAtVersionMeta;
    this.lastUpdatedAtVersionMeta = lastUpdatedAtVersionMeta;
    this.clusteringId = clusteringId;
    this.metadata =
        metadata == null
            ? Collections.emptyMap()
            : Collections.unmodifiableMap(new HashMap<>(metadata));
  }

  public int getId() {
    return id;
  }

  public List<DataFile> getFiles() {
    return files;
  }

  /**
   * Returns every Lance data file referenced by this fragment.
   *
   * <p>This includes both the base data files returned by {@link #getFiles()} and data overlay
   * files. Prefer this method when accounting for storage or copying a fragment.
   */
  public List<DataFile> getReferencedLanceFiles() {
    return referencedLanceFiles;
  }

  public long getPhysicalRows() {
    return physicalRows;
  }

  public DeletionFile getDeletionFile() {
    return deletionFile;
  }

  public long getNumDeletions() {
    if (deletionFile == null) {
      return 0;
    }
    Long deleted = deletionFile.getNumDeletedRows();
    if (deleted == null) {
      return 0;
    }
    return deleted;
  }

  public long getNumRows() {
    return getPhysicalRows() - getNumDeletions();
  }

  public RowIdMeta getRowIdMeta() {
    return rowIdMeta;
  }

  public VersionMeta getCreatedAtVersionMeta() {
    return createdAtVersionMeta;
  }

  public VersionMeta getLastUpdatedAtVersionMeta() {
    return lastUpdatedAtVersionMeta;
  }

  /**
   * The clustering definition this fragment was organized under when it was created, by its id in
   * the manifest, or null when the fragment has no clustering reference. Operations on the same
   * fragment keep it; a fragment written from this one does not inherit it.
   */
  public Long getClusteringId() {
    return clusteringId;
  }

  /**
   * Generic metadata attached to this fragment, like the dataset's table metadata. Operations on
   * the same fragment preserve every entry, including keys this library does not understand; new
   * fragments carry only what their writer sets. Never null; empty when there is none.
   */
  public Map<String, String> getMetadata() {
    return metadata;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    FragmentMetadata that = (FragmentMetadata) o;
    return id == that.id
        && physicalRows == that.physicalRows
        && Objects.equals(this.files, that.files)
        && Objects.equals(referencedLanceFiles, that.referencedLanceFiles)
        && Objects.equals(deletionFile, that.deletionFile)
        && Objects.equals(rowIdMeta, that.rowIdMeta)
        && Objects.equals(createdAtVersionMeta, that.createdAtVersionMeta)
        && Objects.equals(lastUpdatedAtVersionMeta, that.lastUpdatedAtVersionMeta)
        && Objects.equals(clusteringId, that.clusteringId)
        && Objects.equals(metadata, that.metadata);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        id,
        physicalRows,
        files,
        referencedLanceFiles,
        deletionFile,
        rowIdMeta,
        createdAtVersionMeta,
        lastUpdatedAtVersionMeta,
        clusteringId,
        metadata);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("id", id)
        .add("physicalRows", physicalRows)
        .add("files", files)
        .add("referencedLanceFiles", referencedLanceFiles)
        .add("deletionFile", deletionFile)
        .add("rowIdMeta", rowIdMeta)
        .add("createdAtVersionMeta", createdAtVersionMeta)
        .add("lastUpdatedAtVersionMeta", lastUpdatedAtVersionMeta)
        .add("clusteringId", clusteringId)
        .add("metadata", metadata)
        .toString();
  }
}
