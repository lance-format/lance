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

import org.lance.Dataset;
import org.lance.JniLoader;
import org.lance.LockManager;
import org.lance.index.DistanceType;

import org.apache.arrow.util.Preconditions;

/**
 * Training utilities for vector indexes.
 *
 * <p>These helpers expose the underlying Lance training routines so that callers can pre-train
 * models (IVF centroids, PQ codebooks, SQ params) and then pass the resulting artifacts into
 * distributed index build flows.
 */
public final class VectorTrainer {

  static {
    JniLoader.ensureLoaded();
  }

  private VectorTrainer() {}

  /**
   * Train IVF centroids for the given dataset column with {@link DistanceType#L2}.
   *
   * <p>Equivalent to {@link #trainIvfCentroids(Dataset, String, IvfBuildParams, DistanceType)} with
   * {@code distanceType = DistanceType.L2}.
   */
  public static float[] trainIvfCentroids(Dataset dataset, String column, IvfBuildParams params) {
    return trainIvfCentroids(dataset, column, params, DistanceType.L2);
  }

  /**
   * Train IVF centroids for the given dataset column.
   *
   * <p>The {@code distanceType} controls the geometry used to cluster centroids and must match the
   * distance type later passed to per-fragment index builds. If they disagree the centroids are
   * clustered on a different geometry than the encoded data and recall will silently degrade.
   *
   * @param dataset the dataset to sample training data from
   * @param column the vector column name
   * @param params IVF build parameters (numPartitions, sampleRate, etc.)
   * @param distanceType distance metric used during clustering
   * @return a flattened array of centroids laid out as [numPartitions][dimension]
   */
  public static float[] trainIvfCentroids(
      Dataset dataset, String column, IvfBuildParams params, DistanceType distanceType) {
    Preconditions.checkArgument(dataset != null, "dataset cannot be null");
    Preconditions.checkArgument(
        column != null && !column.isEmpty(), "column cannot be null or empty");
    Preconditions.checkArgument(params != null, "params cannot be null");
    Preconditions.checkArgument(distanceType != null, "distanceType cannot be null");
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return nativeTrainIvfCentroids(dataset, column, params, distanceType.toString());
    }
  }

  /**
   * Train a PQ codebook for the given dataset column with {@link DistanceType#L2}.
   *
   * <p>Equivalent to {@link #trainPqCodebook(Dataset, String, PQBuildParams, DistanceType)} with
   * {@code distanceType = DistanceType.L2}.
   */
  public static float[] trainPqCodebook(Dataset dataset, String column, PQBuildParams params) {
    return trainPqCodebook(dataset, column, params, DistanceType.L2);
  }

  /**
   * Train a PQ codebook for the given dataset column.
   *
   * <p>Training uses the column vectors, normalized for cosine. For IVF PQ builds that use
   * residuals, use {@link #trainPqCodebook(Dataset, String, PQBuildParams, DistanceType, float[])}
   * with the centroids used to build that index.
   *
   * @param dataset the dataset to sample training data from
   * @param column the vector column name
   * @param params PQ build parameters (numSubVectors, numBits, sampleRate, etc.)
   * @param distanceType distance metric used during codebook training
   * @return the flattened PQ codebook for {@link PQBuildParams.Builder#setCodebook(float[])}
   */
  public static float[] trainPqCodebook(
      Dataset dataset, String column, PQBuildParams params, DistanceType distanceType) {
    return trainPqCodebookInternal(dataset, column, params, distanceType, null);
  }

  /**
   * Train a PQ codebook for an IVF index using its pretrained centroids.
   *
   * <p>For L2 and cosine, training uses the residual between each vector and its assigned centroid.
   * Cosine normalizes vectors before computing residuals. Dot trains on the original vectors. Use
   * the same distance type and IVF centroids when building the index.
   *
   * <pre>{@code
   * DistanceType metric = DistanceType.L2;
   * float[] centroids = VectorTrainer.trainIvfCentroids(dataset, column, ivfParams, metric);
   * float[] codebook = VectorTrainer.trainPqCodebook(dataset, column, pqParams, metric, centroids);
   * }</pre>
   *
   * @param dataset the dataset to sample training data from
   * @param column the vector column name
   * @param params PQ build parameters (numSubVectors, numBits, sampleRate, etc.)
   * @param distanceType distance metric of the IVF index
   * @param centroids Float32 IVF centers laid out as [numPartitions][dimension]
   * @return the flattened PQ codebook for {@link PQBuildParams.Builder#setCodebook(float[])}
   * @throws IllegalArgumentException if the centroid array is null or has an invalid shape
   */
  public static float[] trainPqCodebook(
      Dataset dataset,
      String column,
      PQBuildParams params,
      DistanceType distanceType,
      float[] centroids) {
    Preconditions.checkArgument(centroids != null, "centroids cannot be null");
    return trainPqCodebookInternal(dataset, column, params, distanceType, centroids);
  }

  private static float[] trainPqCodebookInternal(
      Dataset dataset,
      String column,
      PQBuildParams params,
      DistanceType distanceType,
      float[] centroids) {
    Preconditions.checkArgument(dataset != null, "dataset cannot be null");
    Preconditions.checkArgument(
        column != null && !column.isEmpty(), "column cannot be null or empty");
    Preconditions.checkArgument(params != null, "params cannot be null");
    Preconditions.checkArgument(distanceType != null, "distanceType cannot be null");
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return nativeTrainPqCodebook(dataset, column, params, distanceType.toString(), centroids);
    }
  }

  private static native float[] nativeTrainIvfCentroids(
      Dataset dataset, String column, IvfBuildParams params, String distanceType);

  private static native float[] nativeTrainPqCodebook(
      Dataset dataset, String column, PQBuildParams params, String distanceType, float[] centroids);
}
