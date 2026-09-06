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
package org.lance.index.vector;

import org.lance.index.DistanceType;

import com.google.common.base.MoreObjects;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/** Parameters for creating a vector index. */
public class VectorIndexParams {
  private final DistanceType distanceType;
  private final IvfBuildParams ivfParams;
  private final Optional<PQBuildParams> pqParams;
  private final Optional<HnswBuildParams> hnswParams;
  private final Optional<SQBuildParams> sqParams;
  private final Optional<RQBuildParams> rqParams;
  private final List<String> coveringColumns;

  private VectorIndexParams(Builder builder) {
    this.distanceType = builder.distanceType;
    this.ivfParams = builder.ivfParams;
    this.pqParams = builder.pqParams;
    this.hnswParams = builder.hnswParams;
    this.sqParams = builder.sqParams;
    this.rqParams = builder.rqParams;
    this.coveringColumns =
        builder.coveringColumns == null
            ? Collections.emptyList()
            : Collections.unmodifiableList(new ArrayList<>(builder.coveringColumns));
    validate();
  }

  private void validate() {
    if ((pqParams.isPresent() ? 1 : 0)
            + (sqParams.isPresent() ? 1 : 0)
            + (rqParams.isPresent() ? 1 : 0)
        > 1) {
      throw new IllegalArgumentException("Only one of PQ, SQ, or RQ can be specified at a time.");
    }
    if (hnswParams.isPresent() && !pqParams.isPresent() && !sqParams.isPresent()) {
      throw new IllegalArgumentException("HNSW must be combined with either PQ or SQ");
    }
  }

  /**
   * Create a new IVF index with flat quantizer.
   *
   * @param numPartitions the number of partitions of IVF (Inverted File Index)
   * @param distanceType the distance type for calculating the distance between vectors
   * @return the VectorIndexParams
   */
  public static VectorIndexParams ivfFlat(int numPartitions, DistanceType distanceType) {
    return new Builder(new IvfBuildParams.Builder().setNumPartitions(numPartitions).build())
        .setDistanceType(distanceType)
        .build();
  }

  /**
   * Create a new IVF index with PQ quantizer.
   *
   * @param numPartitions the number of partitions of IVF (Inverted File Index)
   * @param numBits maps the float vectors to integer vectors, each integer is of num_bits. Now only
   *     8 bits are supported
   * @param numSubVectors the number of sub-vectors for PQ (Product Quantization)
   * @param distanceType the distance type for calculating the distance between vectors
   * @param maxIterations K-means max iterations. This will run k-means clustering on each subvector
   *     to determine the centroids that will be used to quantize the subvectors.
   * @return the VectorIndexParams
   */
  public static VectorIndexParams ivfPq(
      int numPartitions,
      int numBits,
      int numSubVectors,
      DistanceType distanceType,
      int maxIterations) {
    IvfBuildParams ivfParams = new IvfBuildParams.Builder().setNumPartitions(numPartitions).build();
    PQBuildParams pqParams =
        new PQBuildParams.Builder()
            .setNumBits(numBits)
            .setNumSubVectors(numSubVectors)
            .setMaxIters(maxIterations)
            .build();

    return new Builder(ivfParams).setDistanceType(distanceType).setPqParams(pqParams).build();
  }

  /**
   * Create a new IVF index with PQ quantizer.
   *
   * @param distanceType the distance type for calculating the distance between vectors
   * @param ivf the IVF build parameters
   * @param pq the PQ build parameters
   * @return the VectorIndexParams
   */
  public static VectorIndexParams withIvfPqParams(
      DistanceType distanceType, IvfBuildParams ivf, PQBuildParams pq) {
    return new Builder(ivf).setDistanceType(distanceType).setPqParams(pq).build();
  }

  /**
   * Create a new IVF index with RQ quantizer.
   *
   * @param numPartitions the number of partitions of IVF (Inverted File Index)
   * @param numBits number of bits per dimension used by Rabit quantization
   * @param distanceType the distance type for calculating the distance between vectors
   * @return the VectorIndexParams
   */
  public static VectorIndexParams ivfRq(
      int numPartitions, byte numBits, DistanceType distanceType) {
    IvfBuildParams ivfParams = new IvfBuildParams.Builder().setNumPartitions(numPartitions).build();
    RQBuildParams rqParams = new RQBuildParams.Builder().setNumBits(numBits).build();

    return new Builder(ivfParams).setDistanceType(distanceType).setRqParams(rqParams).build();
  }

  /**
   * Create a new IVF index with RQ quantizer.
   *
   * @param distanceType the distance type for calculating the distance between vectors
   * @param ivf the IVF build parameters
   * @param rq the RQ build parameters
   * @return the VectorIndexParams
   */
  public static VectorIndexParams withIvfRqParams(
      DistanceType distanceType, IvfBuildParams ivf, RQBuildParams rq) {
    return new Builder(ivf).setDistanceType(distanceType).setRqParams(rq).build();
  }

  /**
   * Create a new IVF HNSW index with PQ quantizer. The dataset is partitioned into IVF partitions,
   * and each partition builds an HNSW graph.
   *
   * @param distanceType the distance type for calculating the distance between vectors
   * @param ivf the IVF build parameters
   * @param hnsw the HNSW build parameters
   * @param pq the PQ build parameters
   * @return the VectorIndexParams
   */
  public static VectorIndexParams withIvfHnswPqParams(
      DistanceType distanceType, IvfBuildParams ivf, HnswBuildParams hnsw, PQBuildParams pq) {
    return new Builder(ivf)
        .setDistanceType(distanceType)
        .setHnswParams(hnsw)
        .setPqParams(pq)
        .build();
  }

  /**
   * Create a new IVF HNSW index with SQ quantizer. The dataset is partitioned into IVF partitions,
   * and each partition builds an HNSW graph.
   *
   * @param distanceType the distance type for calculating the distance between vectors
   * @param ivf the IVF build parameters
   * @param hnsw the HNSW build parameters
   * @param sq the SQ build parameters
   * @return the VectorIndexParams
   */
  public static VectorIndexParams withIvfHnswSqParams(
      DistanceType distanceType, IvfBuildParams ivf, HnswBuildParams hnsw, SQBuildParams sq) {
    return new Builder(ivf)
        .setDistanceType(distanceType)
        .setHnswParams(hnsw)
        .setSqParams(sq)
        .build();
  }

  public static class Builder {
    private DistanceType distanceType = DistanceType.L2;
    private final IvfBuildParams ivfParams;
    private Optional<PQBuildParams> pqParams = Optional.empty();
    private Optional<HnswBuildParams> hnswParams = Optional.empty();
    private Optional<SQBuildParams> sqParams = Optional.empty();
    private Optional<RQBuildParams> rqParams = Optional.empty();
    private List<String> coveringColumns = Collections.emptyList();

    /**
     * Create a new builder to create a vector index.
     *
     * @param ivfParams the IVF build parameters
     */
    public Builder(IvfBuildParams ivfParams) {
      this.ivfParams = ivfParams;
    }

    /**
     * @param distanceType the distance type for calculating the distance between vectors
     * @return Builder
     */
    public Builder setDistanceType(DistanceType distanceType) {
      this.distanceType = distanceType;
      return this;
    }

    /**
     * @param pqParams the PQ quantizer build parameters
     * @return Builder
     */
    public Builder setPqParams(PQBuildParams pqParams) {
      this.pqParams = Optional.of(pqParams);
      return this;
    }

    /**
     * @param hnswParams the HNSW build parameters for building the HNSW graph for each IVF
     *     partition
     * @return Builder
     */
    public Builder setHnswParams(HnswBuildParams hnswParams) {
      this.hnswParams = Optional.of(hnswParams);
      return this;
    }

    /**
     * @param sqParams the sq build params
     * @return Builder
     */
    public Builder setSqParams(SQBuildParams sqParams) {
      this.sqParams = Optional.of(sqParams);
      return this;
    }

    /**
     * @param rqParams the RQ quantizer build parameters
     * @return Builder
     */
    public Builder setRqParams(RQBuildParams rqParams) {
      this.rqParams = Optional.of(rqParams);
      return this;
    }

    /**
     * Set the columns to cover ("include") in the index. Their values are stored inline in the
     * index so a query projecting only covered columns is answered from the index without a take
     * from the base table. Each must name a top-level, non-key column of the dataset; the columns
     * are validated by the core when the index is built. Empty by default (no covering).
     *
     * @param coveringColumns the covering column names
     * @return Builder
     */
    public Builder setCoveringColumns(List<String> coveringColumns) {
      this.coveringColumns = coveringColumns;
      return this;
    }

    public VectorIndexParams build() {
      return new VectorIndexParams(this);
    }
  }

  public DistanceType getDistanceType() {
    return distanceType;
  }

  public String getDistanceTypeString() {
    return distanceType.toString();
  }

  public IvfBuildParams getIvfParams() {
    return ivfParams;
  }

  public Optional<PQBuildParams> getPqParams() {
    return pqParams;
  }

  public Optional<HnswBuildParams> getHnswParams() {
    return hnswParams;
  }

  public Optional<SQBuildParams> getSqParams() {
    return sqParams;
  }

  public Optional<RQBuildParams> getRqParams() {
    return rqParams;
  }

  /**
   * Get the covering ("included") columns whose values are stored inline in the index. Empty when
   * the index has no covering columns.
   *
   * @return the covering column names, as an unmodifiable list
   */
  public List<String> getCoveringColumns() {
    return coveringColumns;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("distanceType", distanceType)
        .add("ivfParams", ivfParams)
        .add("pqParams", pqParams.orElse(null))
        .add("hnswParams", hnswParams.orElse(null))
        .add("sqParams", sqParams.orElse(null))
        .add("rqParams", rqParams.orElse(null))
        .add("coveringColumns", coveringColumns)
        .toString();
  }
}
