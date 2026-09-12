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

import com.google.common.base.MoreObjects;

/**
 * Parameters for building an IVF index. Train IVF centroids for the given vector column. This will
 * run k-means clustering on the given vector column to train the IVF centroids. This is the first
 * step in several vector indices. The centroids will be used to partition the vectors into
 * different clusters. IVF centroids are trained from a sample of the data (determined by the
 * sample_rate). While this sample is not huge it might still be quite large.
 */
public class IvfBuildParams {
  private final int numPartitions;
  private final int maxIters;
  private final int sampleRate;
  private final int shufflePartitionBatches;
  private final int shufflePartitionConcurrency;
  private final boolean useResidual;
  private final float[] centroids;
  private final HnswBuildParams centroidHnsw;

  private IvfBuildParams(Builder builder) {
    this.numPartitions = builder.numPartitions;
    this.maxIters = builder.maxIters;
    this.sampleRate = builder.sampleRate;
    this.shufflePartitionBatches = builder.shufflePartitionBatches;
    this.shufflePartitionConcurrency = builder.shufflePartitionConcurrency;
    this.useResidual = builder.useResidual;
    this.centroids = builder.centroids;
    this.centroidHnsw = builder.centroidHnsw;
  }

  public static class Builder {
    private int numPartitions = 32;
    private int maxIters = 50;
    private int sampleRate = 256;
    private int shufflePartitionBatches = 1024 * 10;
    private int shufflePartitionConcurrency = 2;
    private boolean useResidual = true;
    private float[] centroids = null;
    private HnswBuildParams centroidHnsw = null;

    /**
     * Persist an HNSW router over final Float32/L2 centroids. Null (default) disables construction.
     * Set centroidEf separately on queries to use the stored graph.
     *
     * @param params existing HNSW construction parameters
     * @return this builder
     */
    public Builder setCentroidHnsw(HnswBuildParams params) {
      this.centroidHnsw = params;
      return this;
    }

    /**
     * Parameters for building an IVF index. Train IVF centroids for the given vector column. This
     * will run k-means clustering on the given vector column to train the IVF centroids. This is
     * the first step in several vector indices. The centroids will be used to partition the vectors
     * into different clusters. IVF centroids are trained from a sample of the data (determined by
     * the sample_rate). While this sample is not huge it might still be quite large.
     */
    public Builder() {}

    /**
     * @param numPartitions set the number of partitions of IVF (Inverted File Index) Default to 32
     * @return Builder
     */
    public Builder setNumPartitions(int numPartitions) {
      this.numPartitions = numPartitions;
      return this;
    }

    /**
     * @param maxIters set the maximum number of iterations for k-means clustering. Default to 50.
     * @return Builder
     */
    public Builder setMaxIters(int maxIters) {
      this.maxIters = maxIters;
      return this;
    }

    /**
     * Set the sample rate for training IVF centroids IVF centroids are trained from a sample of the
     * data (determined by the sample_rate). While this sample is not huge it might still be quite
     * large. Default to 256.
     *
     * @param sampleRate set the sample rate for training IVF centroids
     * @return Builder
     */
    public Builder setSampleRate(int sampleRate) {
      this.sampleRate = sampleRate;
      return this;
    }

    /**
     * Sets the number of batches, using the row group size of the dataset, to include in each
     * shuffle partition. Default value is 10240. Assuming the row group size is 1024, each shuffle
     * partition will hold 10240 * 1024 = 10,485,760 rows. By making this value smaller, this
     * shuffle will consume less memory but will take longer to complete, and vice versa.
     *
     * @param shufflePartitionBatches the number of batches to include in shuffle
     * @return Builder
     */
    public Builder setShufflePartitionBatches(int shufflePartitionBatches) {
      this.shufflePartitionBatches = shufflePartitionBatches;
      return this;
    }

    /**
     * Set the number of shuffle partitions to process concurrently. Default value is 2. By making
     * this value smaller, this shuffle will consume less memory but will take longer to complete,
     * and vice versa.
     *
     * @param shufflePartitionConcurrency the number of shuffle partitions to process concurrently
     * @return Builder
     */
    public Builder setShufflePartitionConcurrency(int shufflePartitionConcurrency) {
      this.shufflePartitionConcurrency = shufflePartitionConcurrency;
      return this;
    }

    /**
     * Set whether to use residual for k-means clustering. Default value is true.
     *
     * @param useResidual whether to use residual for k-means clustering
     * @return Builder
     */
    public Builder setUseResidual(boolean useResidual) {
      this.useResidual = useResidual;
      return this;
    }

    /**
     * Set pre-trained IVF centroids.
     *
     * <p>The centroids are flattened as [numPartitions][dimension].
     *
     * @param centroids pre-trained IVF centroids
     * @return Builder
     */
    public Builder setCentroids(float[] centroids) {
      this.centroids = centroids;
      return this;
    }

    public IvfBuildParams build() {
      return new IvfBuildParams(this);
    }
  }

  public int getNumPartitions() {
    return numPartitions;
  }

  public int getMaxIters() {
    return maxIters;
  }

  public int getSampleRate() {
    return sampleRate;
  }

  public int getShufflePartitionBatches() {
    return shufflePartitionBatches;
  }

  public int getShufflePartitionConcurrency() {
    return shufflePartitionConcurrency;
  }

  public boolean useResidual() {
    return useResidual;
  }

  public float[] getCentroids() {
    return centroids;
  }

  /** Returns centroid graph construction parameters, or null when disabled. */
  public HnswBuildParams getCentroidHnsw() {
    return centroidHnsw;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("numPartitions", numPartitions)
        .add("maxIters", maxIters)
        .add("sampleRate", sampleRate)
        .add("shufflePartitionBatches", shufflePartitionBatches)
        .add("shufflePartitionConcurrency", shufflePartitionConcurrency)
        .add("useResidual", useResidual)
        .add("hasCentroids", centroids != null)
        .add("centroidHnsw", centroidHnsw)
        .toString();
  }
}
