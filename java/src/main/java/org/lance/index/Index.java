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

import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/**
 * Metadata for an index in the dataset. This class corresponds to the Rust Index struct in
 * lance/rust/lance-table/src/format/index.rs.
 */
public class Index {
  private final UUID uuid;
  private final List<Integer> fields;
  private final List<Integer> coveringFields;
  private final String name;
  private final long datasetVersion;
  private final List<Integer> fragments;
  private final byte[] indexDetails;
  private final int indexVersion;
  private final Instant createdAt;
  private final Integer baseId;
  private final Long sizeBytes;
  private final List<IndexFile> files;
  private final IndexType indexType;

  private Index(
      UUID uuid,
      List<Integer> fields,
      List<Integer> coveringFields,
      String name,
      long datasetVersion,
      List<Integer> fragments,
      byte[] indexDetails,
      int indexVersion,
      Instant createdAt,
      Integer baseId,
      Long sizeBytes,
      List<IndexFile> files,
      IndexType indexType) {
    this.uuid = uuid;
    // Both lists are defensively copied and unmodifiable so a caller cannot mutate them
    // after construction: equals/hashCode would silently drift from the index files that
    // actually carry the payload, and `coveringFields` is documented as the trailing slice
    // of `fields`, so mutating either alone breaks that relationship. Null-coalesced so an
    // explicit null from the builder reads as "none" rather than blowing up at first use.
    this.fields =
        fields == null
            ? Collections.emptyList()
            : Collections.unmodifiableList(new ArrayList<>(fields));
    this.coveringFields =
        coveringFields == null
            ? Collections.emptyList()
            : Collections.unmodifiableList(new ArrayList<>(coveringFields));
    this.name = name;
    this.datasetVersion = datasetVersion;
    this.fragments = fragments;
    this.indexDetails = indexDetails;
    this.indexVersion = indexVersion;
    this.createdAt = createdAt;
    this.baseId = baseId;
    boolean hasFiles = files != null && !files.isEmpty();
    if (hasFiles && sizeBytes == null) {
      long derivedSizeBytes = 0;
      try {
        for (IndexFile file : files) {
          derivedSizeBytes = Math.addExact(derivedSizeBytes, file.getSizeBytes());
        }
      } catch (ArithmeticException e) {
        throw new IllegalArgumentException(
            "index file sizes cannot be represented by Java long", e);
      }
      this.sizeBytes = derivedSizeBytes;
    } else {
      this.sizeBytes = !hasFiles && Long.valueOf(0).equals(sizeBytes) ? null : sizeBytes;
    }
    this.files = hasFiles ? files : null;
    this.indexType = indexType;
  }

  public UUID uuid() {
    return uuid;
  }

  /**
   * Get the field id list included in the index.
   *
   * @return the field IDs
   */
  public List<Integer> fields() {
    return fields;
  }

  /**
   * Fields whose values this index carries but is not keyed on. Always a suffix of {@link
   * #fields()}, and never all of it, so the first entry of {@code fields()} is always a column the
   * index is keyed on. Empty for an index that carries no extra columns.
   *
   * <p>These ids also appear in {@link #fields()} — that is deliberate, so that every consumer
   * reading {@code fields()} as the index's dependency set also covers them with no change.
   *
   * @return the covering field IDs, as an unmodifiable list
   */
  public List<Integer> coveringFields() {
    return coveringFields;
  }

  /**
   * Human readable index name
   *
   * @return the index name
   */
  public String name() {
    return name;
  }

  /**
   * The latest version of the dataset this index covers
   *
   * @return the dataset version
   */
  public long datasetVersion() {
    return datasetVersion;
  }

  public Optional<List<Integer>> fragments() {
    return Optional.ofNullable(fragments);
  }

  public Optional<byte[]> indexDetails() {
    return Optional.ofNullable(indexDetails);
  }

  public Optional<Integer> baseId() {
    return Optional.ofNullable(baseId);
  }

  /**
   * Get the total size of all files in this physical index segment.
   *
   * <p>The size is unavailable for indices created before index file sizes were tracked.
   *
   * @return the segment size in bytes, or empty if unavailable
   */
  public Optional<Long> getSizeBytes() {
    return Optional.ofNullable(sizeBytes);
  }

  /** Files and sizes for this index segment, or empty when older writers did not record them. */
  public Optional<List<IndexFile>> getFiles() {
    return Optional.ofNullable(files);
  }

  /**
   * Get the index version.
   *
   * @return the index version
   */
  public int indexVersion() {
    return indexVersion;
  }

  /**
   * Get the creation time of the index.
   *
   * @return the creation time
   */
  public Optional<Instant> createdAt() {
    return Optional.ofNullable(createdAt);
  }

  /**
   * Get the type of the index (e.g., BTREE, BITMAP, VECTOR).
   *
   * @return the index type, or null if unknown
   */
  public IndexType indexType() {
    return indexType;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Index index = (Index) o;
    return datasetVersion == index.datasetVersion
        && indexVersion == index.indexVersion
        && Objects.equals(uuid, index.uuid)
        && Objects.equals(fields, index.fields)
        && Objects.equals(coveringFields, index.coveringFields)
        && Objects.equals(name, index.name)
        && Objects.equals(fragments, index.fragments)
        && Arrays.equals(indexDetails, index.indexDetails)
        && Objects.equals(createdAt, index.createdAt)
        && Objects.equals(baseId, index.baseId)
        && Objects.equals(sizeBytes, index.sizeBytes)
        && Objects.equals(files, index.files)
        && indexType == index.indexType;
  }

  @Override
  public int hashCode() {
    int result =
        Objects.hash(
            uuid,
            fields,
            coveringFields,
            name,
            datasetVersion,
            indexVersion,
            createdAt,
            baseId,
            sizeBytes,
            files,
            fragments,
            indexType);
    result = 31 * result + Arrays.hashCode(indexDetails);
    return result;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("uuid", uuid)
        .add("fields", fields)
        .add("coveringFields", coveringFields)
        .add("name", name)
        .add("datasetVersion", datasetVersion)
        .add("indexVersion", indexVersion)
        .add("indexType", indexType)
        .add("createdAt", createdAt)
        .add("baseId", baseId)
        .add("sizeBytes", sizeBytes)
        .add("files", files)
        .toString();
  }

  /**
   * Create a new builder for Index.
   *
   * @return a new builder
   */
  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {

    private UUID uuid;
    private List<Integer> fields;
    private List<Integer> coveringFields = Collections.emptyList();
    private String name;
    private long datasetVersion;
    private List<Integer> fragments;
    private byte[] indexDetails;
    private int indexVersion;
    private Instant createdAt;
    private Integer baseId;
    private Long sizeBytes;
    private List<IndexFile> files;
    private IndexType indexType;

    private Builder() {}

    public Builder uuid(UUID uuid) {
      this.uuid = uuid;
      return this;
    }

    public Builder fields(List<Integer> fields) {
      this.fields = fields;
      return this;
    }

    public Builder coveringFields(List<Integer> coveringFields) {
      this.coveringFields = coveringFields;
      return this;
    }

    public Builder name(String name) {
      this.name = name;
      return this;
    }

    public Builder datasetVersion(long datasetVersion) {
      this.datasetVersion = datasetVersion;
      return this;
    }

    public Builder fragments(List<Integer> fragments) {
      this.fragments = fragments;
      return this;
    }

    public Builder indexDetails(byte[] indexDetails) {
      this.indexDetails = indexDetails;
      return this;
    }

    public Builder indexVersion(int indexVersion) {
      this.indexVersion = indexVersion;
      return this;
    }

    public Builder createdAt(Instant createdAt) {
      this.createdAt = createdAt;
      return this;
    }

    public Builder baseId(Integer baseId) {
      this.baseId = baseId;
      return this;
    }

    public Builder sizeBytes(Long sizeBytes) {
      this.sizeBytes = sizeBytes;
      return this;
    }

    public Builder files(List<IndexFile> files) {
      this.files = files;
      return this;
    }

    public Builder indexType(IndexType indexType) {
      this.indexType = indexType;
      return this;
    }

    public Index build() {
      return new Index(
          uuid,
          fields,
          coveringFields,
          name,
          datasetVersion,
          fragments,
          indexDetails,
          indexVersion,
          createdAt,
          baseId,
          sizeBytes,
          files,
          indexType);
    }
  }
}
