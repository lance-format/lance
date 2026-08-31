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
package org.lance.ipc;

import org.lance.index.DistanceType;

import com.google.common.base.MoreObjects;
import org.apache.arrow.util.Preconditions;

import java.util.Optional;

public class Query {

  private final String column;
  private final float[] key;
  private final int k;
  private final int minimumNprobes;
  private final Optional<Integer> maximumNprobes;
  private final Optional<Integer> ef;
  private final Optional<Integer> centroidEf;
  private final Optional<Integer> refineFactor;
  private final Optional<DistanceType> distanceType;
  private final boolean useIndex;
  private final int queryParallelism;
  private final ApproxMode approxMode;

  private Query(Builder builder) {
    this.column = Preconditions.checkNotNull(builder.column, "Columns must be set");
    Preconditions.checkArgument(!builder.column.isEmpty(), "Column must not be empty");
    this.key = Preconditions.checkNotNull(builder.key, "Key must be set");
    Preconditions.checkArgument(builder.k > 0, "K must be greater than 0");
    Preconditions.checkArgument(
        builder.minimumNprobes > 0, "Minimum Nprobes must be greater than 0");
    Preconditions.checkArgument(
        !builder.maximumNprobes.isPresent()
            || builder.maximumNprobes.get() >= builder.minimumNprobes,
        "Maximum Nprobes must be greater than minimum Nprobes");
    this.k = builder.k;
    this.minimumNprobes = builder.minimumNprobes;
    this.maximumNprobes = builder.maximumNprobes;
    this.ef = builder.ef;
    Preconditions.checkArgument(
        !builder.centroidEf.isPresent()
            || !builder.maximumNprobes.isPresent()
            || builder.centroidEf.get() >= builder.maximumNprobes.get(),
        "Centroid ef must be greater than or equal to maximum Nprobes");
    this.centroidEf = builder.centroidEf;
    this.refineFactor = builder.refineFactor;
    this.distanceType = builder.distanceType;
    this.useIndex = builder.useIndex;
    this.queryParallelism = builder.queryParallelism;
    this.approxMode = builder.approxMode;
  }

  public String getColumn() {
    return column;
  }

  public float[] getKey() {
    return key;
  }

  public int getK() {
    return k;
  }

  public int getMinimumNprobes() {
    return minimumNprobes;
  }

  public Optional<Integer> getMaximumNprobes() {
    return maximumNprobes;
  }

  public Optional<Integer> getEf() {
    return ef;
  }

  public Optional<Integer> getCentroidEf() {
    return centroidEf;
  }

  public Optional<Integer> getRefineFactor() {
    return refineFactor;
  }

  public Optional<DistanceType> getDistanceType() {
    return distanceType;
  }

  public Optional<String> getDistanceTypeString() {
    return distanceType.map(DistanceType::toString);
  }

  public boolean isUseIndex() {
    return useIndex;
  }

  public int getQueryParallelism() {
    return queryParallelism;
  }

  public ApproxMode getApproxMode() {
    return approxMode;
  }

  public String getApproxModeString() {
    return approxMode.toRustString();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("column", column)
        .add("key", key)
        .add("k", k)
        .add("minimumNprobes", minimumNprobes)
        .add("maximumNprobes", maximumNprobes.orElse(null))
        .add("ef", ef.orElse(null))
        .add("centroidEf", centroidEf.orElse(null))
        .add("refineFactor", refineFactor.orElse(null))
        .add("distanceType", distanceType.orElse(null))
        .add("useIndex", useIndex)
        .add("queryParallelism", queryParallelism)
        .add("approxMode", approxMode)
        .toString();
  }

  public static class Builder {
    private String column;
    private float[] key;
    private int k = 10;
    private int minimumNprobes = 1;
    private Optional<Integer> maximumNprobes = Optional.empty();
    private Optional<Integer> ef = Optional.empty();
    private Optional<Integer> centroidEf = Optional.empty();
    private Optional<Integer> refineFactor = Optional.empty();
    private Optional<DistanceType> distanceType = Optional.empty();
    private boolean useIndex = true;
    private int queryParallelism = 0;
    private ApproxMode approxMode = ApproxMode.NORMAL;

    /**
     * Sets the column to be searched.
     *
     * @param column The name of the column to search in.
     * @return The Builder instance for method chaining.
     */
    public Builder setColumn(String column) {
      this.column = column;
      return this;
    }

    /**
     * Sets the vector to be searched.
     *
     * <p>This API accepts a single query vector. The array length must match the target vector
     * column dimension. Batch nearest-neighbor search with multiple query vectors requires a
     * list-shaped query input and is not available through this {@code float[]} entry point.
     *
     * @param key The search vector.
     * @return The Builder instance for method chaining.
     */
    public Builder setKey(float[] key) {
      this.key = key;
      return this;
    }

    /**
     * Sets the number of top results to return.
     *
     * @param k The number of top results to return.
     * @return The Builder instance for method chaining.
     */
    public Builder setK(int k) {
      this.k = k;
      return this;
    }

    /**
     * Sets the number of probes to load and search.
     *
     * <p>This is a convenience method that sets both the minimum and maximum number of probes to
     * the same value.
     *
     * @param nprobes The number of probes.
     * @return The Builder instance for method chaining.
     */
    public Builder setNprobes(int nprobes) {
      this.minimumNprobes = nprobes;
      this.maximumNprobes = Optional.of(nprobes);
      return this;
    }

    /**
     * Sets the minimum number of partitions to search.
     *
     * <p>This many partitions will always be loaded and searched on the query. Increasing this
     * number can improve recall at the cost of latency.
     *
     * @param minimumNprobes The minimum number of partitions to search.
     * @return The Builder instance for method chaining.
     */
    public Builder setMinimumNprobes(int minimumNprobes) {
      this.minimumNprobes = minimumNprobes;
      return this;
    }

    /**
     * Sets the maximum number of partitions to search.
     *
     * <p>These partitions will only be loaded and searched if we have not found the desired number
     * of results after searching the minimum number of partitions. Increasing this number can avoid
     * false negatives on queries with a highly selective prefilter. This setting does not affect
     * the recall of the query and will only affect the latency if the prefilter is highly
     * selective.
     *
     * @param maximumNprobes The maximum number of partitions to search.
     * @return The Builder instance for method chaining.
     */
    public Builder setMaximumNprobes(int maximumNprobes) {
      this.maximumNprobes = Optional.of(maximumNprobes);
      return this;
    }

    /**
     * Sets the number of candidates to reserve while searching. This is an optional parameter for
     * HNSW related index types.
     *
     * @param ef The number of candidates to reserve.
     * @return The Builder instance for method chaining.
     */
    public Builder setEf(int ef) {
      this.ef = Optional.of(ef);
      return this;
    }

    /**
     * Enables HNSW routing over IVF centroids and sets its candidate beam size.
     *
     * <p>The value must be at least maximum Nprobes. Exact centroid routing is used when this is
     * not set.
     *
     * @param centroidEf The number of centroid candidates to reserve.
     * @return The Builder instance for method chaining.
     */
    public Builder setCentroidEf(int centroidEf) {
      Preconditions.checkArgument(centroidEf > 0, "Centroid ef must be greater than 0");
      this.centroidEf = Optional.of(centroidEf);
      return this;
    }

    /**
     * Sets the refine factor for applying a refine step.
     *
     * @param refineFactor The refine factor.
     * @return The Builder instance for method chaining.
     */
    public Builder setRefineFactor(int refineFactor) {
      this.refineFactor = Optional.of(refineFactor);
      return this;
    }

    /**
     * Sets the distance metric type.
     *
     * <p>If not set, the query will use the index's metric type (if an index is available), or the
     * default metric for the data type (L2 for float vectors, Hamming for binary).
     *
     * @param distanceType The DistanceType to use for the query.
     * @return The Builder instance for method chaining.
     */
    public Builder setDistanceType(DistanceType distanceType) {
      this.distanceType = Optional.ofNullable(distanceType);
      return this;
    }

    /**
     * Sets whether to use an ANN index if available.
     *
     * @param useIndex True to use the index, false otherwise.
     * @return The Builder instance for method chaining.
     */
    public Builder setUseIndex(boolean useIndex) {
      this.useIndex = useIndex;
      return this;
    }

    /**
     * Sets vector partition search concurrency for each query.
     *
     * <p>The default is 0. Value 0 uses the automatic policy, which currently maps to the
     * single-worker sequential path. Value -1 uses the CPU pool size. Value 1 uses the
     * single-worker sequential path. Values greater than or equal to 2 use the partition-parallel
     * path and are clamped to the CPU pool size.
     *
     * @param queryParallelism The partition search concurrency policy.
     * @return The Builder instance for method chaining.
     */
    public Builder setQueryParallelism(int queryParallelism) {
      Preconditions.checkArgument(
          queryParallelism >= -1, "Query parallelism must be greater than or equal to -1");
      this.queryParallelism = queryParallelism;
      return this;
    }

    /**
     * Sets the speed / accuracy tradeoff for approximate vector search.
     *
     * <p>This setting currently only affects RQ-quantized vector indexes, such as IVF_RQ. Other
     * index types ignore this setting.
     *
     * @param approxMode The approximate search mode to use for the query.
     * @return The Builder instance for method chaining.
     */
    public Builder setApproxMode(ApproxMode approxMode) {
      this.approxMode = Preconditions.checkNotNull(approxMode, "ApproxMode must not be null");
      return this;
    }

    /**
     * Builds the Query object.
     *
     * @return A new immutable Query instance.
     * @throws IllegalStateException if any required fields are not set or have invalid values.
     */
    public Query build() {
      return new Query(this);
    }
  }
}
