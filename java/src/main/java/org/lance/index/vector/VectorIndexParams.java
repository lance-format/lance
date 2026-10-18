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

import java.util.Optional;

/** Parameters for creating a vector index. */
public class VectorIndexParams {
  private final DistanceType distanceType;
  private final IvfBuildParams ivfParams;
  private final Optional<PQBuildParams> pqParams;
  private final Optional<HnswBuildParams> hnswParams;
  private final Optional<SQBuildParams> sqParams;
  private final Optional<RQBuildParams> rqParams;
  private final Optional<VamanaBuildParams> vamanaParams; // fork(vamana)

  private VectorIndexParams(Builder builder) {
    this.distanceType = builder.distanceType;
    this.ivfParams = builder.ivfParams;
    this.pqParams = builder.pqParams;
    this.hnswParams = builder.hnswParams;
    this.sqParams = builder.sqParams;
    this.rqParams = builder.rqParams;
    this.vamanaParams = builder.vamanaParams; // fork(vamana)
    validate();
  }

  private void validate() {
    if ((pqParams.isPresent() ? 1 : 0)
            + (sqParams.isPresent() ? 1 : 0)
            + (rqParams.isPresent() ? 1 : 0)
        > 1) {
      throw new IllegalArgumentException("Only one of PQ, SQ, or RQ can be specified at a time.");
    }
    // fork(vamana)
    if (hnswParams.isPresent() && vamanaParams.isPresent()) {
      throw new IllegalArgumentException("Only one of HNSW or Vamana can be specified at a time.");
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

  // fork(vamana)
  /**
   * Create a new IVF_VAMANA_FLAT index. The dataset is partitioned into IVF partitions, and each
   * partition builds a Vamana graph over its vectors as they are. Supports the l2 and cosine
   * distances only.
   *
   * @param distanceType the distance type for calculating the distance between vectors
   * @param ivf the IVF build parameters
   * @param vamana the Vamana build parameters
   * @return the VectorIndexParams
   */
  public static VectorIndexParams ivfVamana(
      DistanceType distanceType, IvfBuildParams ivf, VamanaBuildParams vamana) {
    return new Builder(ivf).setDistanceType(distanceType).setVamanaParams(vamana).build();
  }

  /**
   * Create a new IVF_VAMANA_SQ index. The dataset is partitioned into IVF partitions, and each
   * partition builds a Vamana graph over, and walks it with, 8-bit scalar-quantized codes. Supports
   * the l2 and cosine distances only.
   *
   * @param distanceType the distance type for calculating the distance between vectors
   * @param ivf the IVF build parameters
   * @param vamana the Vamana build parameters
   * @param sq the SQ build parameters
   * @return the VectorIndexParams
   */
  public static VectorIndexParams withIvfVamanaSqParams(
      DistanceType distanceType, IvfBuildParams ivf, VamanaBuildParams vamana, SQBuildParams sq) {
    return new Builder(ivf)
        .setDistanceType(distanceType)
        .setVamanaParams(vamana)
        .setSqParams(sq)
        .build();
  }

  /**
   * Create a new IVF_VAMANA_PQ index. The dataset is partitioned into IVF partitions, and each
   * partition builds a Vamana graph over, and walks it with, product-quantized codes. Supports the
   * l2 and cosine distances only.
   *
   * @param distanceType the distance type for calculating the distance between vectors
   * @param ivf the IVF build parameters
   * @param vamana the Vamana build parameters
   * @param pq the PQ build parameters
   * @return the VectorIndexParams
   */
  public static VectorIndexParams withIvfVamanaPqParams(
      DistanceType distanceType, IvfBuildParams ivf, VamanaBuildParams vamana, PQBuildParams pq) {
    return new Builder(ivf)
        .setDistanceType(distanceType)
        .setVamanaParams(vamana)
        .setPqParams(pq)
        .build();
  }

  public static class Builder {
    private DistanceType distanceType = DistanceType.L2;
    private final IvfBuildParams ivfParams;
    private Optional<PQBuildParams> pqParams = Optional.empty();
    private Optional<HnswBuildParams> hnswParams = Optional.empty();
    private Optional<SQBuildParams> sqParams = Optional.empty();
    private Optional<RQBuildParams> rqParams = Optional.empty();
    private Optional<VamanaBuildParams> vamanaParams = Optional.empty(); // fork(vamana)

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

    // fork(vamana)
    /**
     * @param vamanaParams the Vamana build parameters for building the Vamana graph for each IVF
     *     partition. Combines with no quantizer, with SQ or with PQ, and excludes HNSW
     * @return Builder
     */
    public Builder setVamanaParams(VamanaBuildParams vamanaParams) {
      this.vamanaParams = Optional.of(vamanaParams);
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

  public Optional<VamanaBuildParams> getVamanaParams() { // fork(vamana)
    return vamanaParams;
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
        .add("vamanaParams", vamanaParams.orElse(null)) // fork(vamana)
        .toString();
  }
}
