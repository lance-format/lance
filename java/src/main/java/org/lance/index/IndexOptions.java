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

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.util.Preconditions;

import java.util.List;
import java.util.Optional;

/** Options of building indexes. */
public class IndexOptions {
  private final boolean replace;
  private final boolean train;
  private final List<Integer> fragmentIds;
  private final String indexUUID;
  private final String indexName;
  private final List<String> columns;
  private final IndexType indexType;
  private final IndexParams indexParams;
  private final ArrowArrayStream preprocessedData;

  private IndexOptions(
      String indexName,
      List<String> columns,
      IndexType indexType,
      IndexParams indexParams,
      boolean replace,
      boolean train,
      List<Integer> fragmentIds,
      String indexUUID,
      ArrowArrayStream preprocessedData) {
    this.replace = replace;
    this.train = train;
    this.fragmentIds = fragmentIds;
    this.indexUUID = indexUUID;
    this.indexName = indexName;
    this.columns = columns;
    this.indexType = indexType;
    this.indexParams = indexParams;
    this.preprocessedData = preprocessedData;
  }

  public Optional<String> getIndexUUID() {
    return Optional.ofNullable(indexUUID);
  }

  public Optional<List<Integer>> getFragmentIds() {
    return Optional.ofNullable(fragmentIds);
  }

  public boolean isReplace() {
    return replace;
  }

  public boolean isTrain() {
    return train;
  }

  public Optional<String> getIndexName() {
    return Optional.ofNullable(indexName);
  }

  public IndexParams getIndexParams() {
    return indexParams;
  }

  public IndexType getIndexType() {
    return indexType;
  }

  public List<String> getColumns() {
    return columns;
  }

  public Optional<ArrowArrayStream> getPreprocessedData() {
    return Optional.ofNullable(preprocessedData);
  }

  public static Builder builder(
      List<String> columns, IndexType indexType, IndexParams indexParams) {
    return new Builder(columns, indexType, indexParams);
  }

  /** Builder class. */
  public static class Builder {
    private boolean replace = false;
    private boolean train = true;
    private List<Integer> fragmentIds = null;
    private String indexUUID = null;
    private String indexName = null;
    private ArrowArrayStream preprocessedData = null;
    private final List<String> columns;
    private final IndexType indexType;
    private final IndexParams indexParams;

    private Builder(List<String> columns, IndexType indexType, IndexParams indexParams) {
      this.columns = Preconditions.checkNotNull(columns);
      this.indexType = Preconditions.checkNotNull(indexType);
      this.indexParams = Preconditions.checkNotNull(indexParams);
    }

    /**
     * Replace the existing index if it exists.
     *
     * @param replace replace option
     */
    public Builder replace(boolean replace) {
      this.replace = replace;
      return this;
    }

    /**
     * If True, the index will be trained on the data to determine optimal structure. If False, an
     * empty index will be created that can be populated later.
     *
     * @param train train option
     */
    public Builder train(boolean train) {
      this.train = train;
      return this;
    }

    /**
     * If provided, the index will be created only on the specified fragments. This enables
     * distributed/fragment-level indexing. When provided, the method returns an IndexMetadata
     * object but does not commit the index to the dataset. The index can be committed later using
     * the commit API.
     *
     * @param fragmentIds fragmentIds option
     */
    public Builder withFragmentIds(List<Integer> fragmentIds) {
      this.fragmentIds = fragmentIds;
      return this;
    }

    /**
     * An output UUID, for index types that support specifying one. If omitted, a new UUID is
     * generated. Bitmap fragment builds reject this option: build independent segments and merge
     * the returned Index objects instead of sharing a UUID between workers.
     *
     * @param indexUUID indexUUID option
     */
    public Builder withIndexUUID(String indexUUID) {
      this.indexUUID = indexUUID;
      return this;
    }

    /**
     * Optional index name. If not provided, a name with format like 'column' + '_idx' will be
     * generated.
     *
     * @param indexName index name
     */
    public Builder withIndexName(String indexName) {
      this.indexName = indexName;
      return this;
    }

    /**
     * Optional preprocessed data. Some index types can consume it to avoid heavy computation e.g.
     * For ranged btree index, data can be ranged and sorted by distributed computing engines.
     *
     * @param preprocessedData preprocessed data.
     */
    public Builder withPreprocessedData(ArrowArrayStream preprocessedData) {
      this.preprocessedData = preprocessedData;
      return this;
    }

    public IndexOptions build() {
      return new IndexOptions(
          indexName,
          columns,
          indexType,
          indexParams,
          replace,
          train,
          fragmentIds,
          indexUUID,
          preprocessedData);
    }
  }
}
