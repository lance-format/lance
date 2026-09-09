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

import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.TestVectorDataset;
import org.lance.index.vector.IvfBuildParams;
import org.lance.index.vector.PQBuildParams;
import org.lance.index.vector.RQBuildParams;
import org.lance.index.vector.SQBuildParams;
import org.lance.index.vector.VectorIndexParams;
import org.lance.index.vector.VectorTrainer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class VectorIndexTest {

  @ParameterizedTest
  @EnumSource(
      value = IndexType.class,
      names = {"IVF_FLAT", "IVF_PQ"})
  @SuppressWarnings("deprecation")
  public void testCreateIndexWithConcreteVectorType(IndexType indexType, @TempDir Path tempDir)
      throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve(indexType.name()))) {
      try (Dataset dataset = testVectorDataset.create()) {
        VectorIndexParams vectorIndexParams =
            indexType == IndexType.IVF_FLAT
                ? VectorIndexParams.ivfFlat(2, DistanceType.L2)
                : VectorIndexParams.ivfPq(2, 8, 2, DistanceType.L2, 2);
        IndexParams indexParams =
            IndexParams.builder().setVectorIndexParams(vectorIndexParams).build();

        Index index =
            dataset.createIndex(
                Collections.singletonList(TestVectorDataset.vectorColumnName),
                indexType,
                Optional.empty(),
                indexParams,
                false);

        assertNotNull(index);
        assertTrue(dataset.listIndexes().contains(index.name()));
      }
    }
  }

  @Test
  public void testCreateIvfFlatIndexPropagatesProgressFailure(@TempDir Path tempDir)
      throws Exception {
    String indexName = "ivf_progress_failure_idx";
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("ivf_progress_failure"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        long initialVersion = dataset.version();
        IndexParams indexParams =
            IndexParams.builder()
                .setVectorIndexParams(VectorIndexParams.ivfFlat(2, DistanceType.L2))
                .build();
        IndexOptions options =
            IndexOptions.builder(
                    Collections.singletonList(TestVectorDataset.vectorColumnName),
                    IndexType.IVF_FLAT,
                    indexParams)
                .withIndexName(indexName)
                .build();
        IndexBuildProgress progress =
            new IndexBuildProgress() {
              @Override
              public void stageStart(String stage, Optional<Long> total, String unit) {}

              @Override
              public void stageProgress(String stage, long completed) {
                if (stage.equals("train_ivf")) {
                  throw new IllegalStateException("vector progress callback failure");
                }
              }

              @Override
              public void stageComplete(String stage) {}
            };

        RuntimeException failure =
            assertThrows(RuntimeException.class, () -> dataset.createIndex(options, progress));

        assertFalse(
            failure instanceof IllegalArgumentException,
            "Progress callback failures should not be reported as invalid input: " + failure);
        assertTrue(
            causeChainContains(failure, "stageProgress")
                && causeChainContains(failure, "train_ivf")
                && causeChainContains(failure, "java.lang.IllegalStateException")
                && causeChainContains(failure, "vector progress callback failure"),
            "Expected callback context and original Java exception details, got: " + failure);
        assertEquals(initialVersion, dataset.version(), "Failed index build must not commit");
        assertFalse(
            dataset.listIndexes().contains(indexName), "Failed index build must not publish index");
      }
    }
  }

  @Test
  public void testCreateIvfFlatIndexDistributively(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("merge_ivfflat_index_metadata"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        List<Fragment> fragments = dataset.getFragments();
        assertTrue(
            fragments.size() >= 2,
            "Expected dataset to have at least two fragments for distributed indexing");

        int numPartitions = 2;

        IvfBuildParams ivfTrainParams =
            new IvfBuildParams.Builder().setNumPartitions(numPartitions).setMaxIters(1).build();

        float[] centroids =
            VectorTrainer.trainIvfCentroids(
                dataset, TestVectorDataset.vectorColumnName, ivfTrainParams);

        IvfBuildParams ivfParams =
            new IvfBuildParams.Builder()
                .setNumPartitions(numPartitions)
                .setMaxIters(1)
                .setCentroids(centroids)
                .build();

        VectorIndexParams vectorIndexParams =
            new VectorIndexParams.Builder(ivfParams).setDistanceType(DistanceType.L2).build();

        IndexParams indexParams =
            IndexParams.builder().setVectorIndexParams(vectorIndexParams).build();

        Index firstSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList(TestVectorDataset.vectorColumnName),
                        IndexType.IVF_FLAT,
                        indexParams)
                    .withIndexName(TestVectorDataset.indexName)
                    .withFragmentIds(Collections.singletonList(fragments.get(0).getId()))
                    .build());

        Index secondSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList(TestVectorDataset.vectorColumnName),
                        IndexType.IVF_FLAT,
                        indexParams)
                    .withIndexName(TestVectorDataset.indexName)
                    .withFragmentIds(Collections.singletonList(fragments.get(1).getId()))
                    .build());

        // The index should not be visible before metadata merge & commit
        assertFalse(
            dataset.listIndexes().contains(TestVectorDataset.indexName),
            "Partially created IVF_FLAT index should not present before commit");

        List<Index> committed =
            dataset.commitExistingIndexSegments(
                TestVectorDataset.indexName,
                TestVectorDataset.vectorColumnName,
                List.of(firstSegment, secondSegment));
        assertEquals(2, committed.size());
        assertTrue(dataset.listIndexes().contains(TestVectorDataset.indexName));
      }
    }
  }

  @Test
  public void testCreateIvfPqIndexDistributively(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("merge_ivfpq_index_metadata"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        List<Fragment> fragments = dataset.getFragments();
        assertTrue(
            fragments.size() >= 2,
            "Expected dataset to have at least two fragments for distributed indexing");

        int numPartitions = 2;
        int numSubVectors = 2;
        int numBits = 8;

        IvfBuildParams ivfTrainParams =
            new IvfBuildParams.Builder().setNumPartitions(numPartitions).setMaxIters(1).build();

        PQBuildParams pqTrainParams =
            new PQBuildParams.Builder()
                .setNumSubVectors(numSubVectors)
                .setNumBits(numBits)
                .setMaxIters(2)
                .setSampleRate(256)
                .build();

        float[] centroids =
            VectorTrainer.trainIvfCentroids(
                dataset, TestVectorDataset.vectorColumnName, ivfTrainParams);

        float[] codebook =
            VectorTrainer.trainPqCodebook(
                dataset, TestVectorDataset.vectorColumnName, pqTrainParams);

        IvfBuildParams ivfParams =
            new IvfBuildParams.Builder()
                .setNumPartitions(numPartitions)
                .setMaxIters(1)
                .setCentroids(centroids)
                .build();

        PQBuildParams pqParams =
            new PQBuildParams.Builder()
                .setNumSubVectors(numSubVectors)
                .setNumBits(numBits)
                .setMaxIters(2)
                .setSampleRate(256)
                .setCodebook(codebook)
                .build();

        VectorIndexParams vectorIndexParams =
            VectorIndexParams.withIvfPqParams(DistanceType.L2, ivfParams, pqParams);

        IndexParams indexParams =
            IndexParams.builder().setVectorIndexParams(vectorIndexParams).build();

        Index firstSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList(TestVectorDataset.vectorColumnName),
                        IndexType.IVF_PQ,
                        indexParams)
                    .withIndexName(TestVectorDataset.indexName)
                    .withFragmentIds(Collections.singletonList(fragments.get(0).getId()))
                    .build());

        Index secondSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList(TestVectorDataset.vectorColumnName),
                        IndexType.IVF_PQ,
                        indexParams)
                    .withIndexName(TestVectorDataset.indexName)
                    .withFragmentIds(Collections.singletonList(fragments.get(1).getId()))
                    .build());

        assertFalse(
            dataset.listIndexes().contains(TestVectorDataset.indexName),
            "Partially created IVF_PQ index should not present before commit");

        List<Index> committed =
            dataset.commitExistingIndexSegments(
                TestVectorDataset.indexName,
                TestVectorDataset.vectorColumnName,
                List.of(firstSegment, secondSegment));
        assertEquals(2, committed.size());
        assertTrue(dataset.listIndexes().contains(TestVectorDataset.indexName));
      }
    }
  }

  @Test
  public void testCreateIvfSqIndexDistributively(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("merge_ivfsq_index_metadata"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        List<Fragment> fragments = dataset.getFragments();
        assertTrue(
            fragments.size() >= 2,
            "Expected dataset to have at least two fragments for distributed indexing");

        int numPartitions = 2;
        short numBits = 8;

        IvfBuildParams ivfTrainParams =
            new IvfBuildParams.Builder().setNumPartitions(numPartitions).setMaxIters(1).build();

        SQBuildParams sqParams =
            new SQBuildParams.Builder().setNumBits(numBits).setSampleRate(256).build();

        float[] centroids =
            VectorTrainer.trainIvfCentroids(
                dataset, TestVectorDataset.vectorColumnName, ivfTrainParams);

        IvfBuildParams ivfParams =
            new IvfBuildParams.Builder()
                .setNumPartitions(numPartitions)
                .setMaxIters(1)
                .setCentroids(centroids)
                .build();

        VectorIndexParams vectorIndexParams =
            new VectorIndexParams.Builder(ivfParams)
                .setDistanceType(DistanceType.L2)
                .setSqParams(sqParams)
                .build();

        IndexParams indexParams =
            IndexParams.builder().setVectorIndexParams(vectorIndexParams).build();

        Index firstSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList(TestVectorDataset.vectorColumnName),
                        IndexType.IVF_SQ,
                        indexParams)
                    .withIndexName(TestVectorDataset.indexName)
                    .withFragmentIds(Collections.singletonList(fragments.get(0).getId()))
                    .build());

        Index secondSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList(TestVectorDataset.vectorColumnName),
                        IndexType.IVF_SQ,
                        indexParams)
                    .withIndexName(TestVectorDataset.indexName)
                    .withFragmentIds(Collections.singletonList(fragments.get(1).getId()))
                    .build());

        assertFalse(
            dataset.listIndexes().contains(TestVectorDataset.indexName),
            "Partially created IVF_SQ index should not present before commit");

        List<Index> committed =
            dataset.commitExistingIndexSegments(
                TestVectorDataset.indexName,
                TestVectorDataset.vectorColumnName,
                List.of(firstSegment, secondSegment));
        assertEquals(2, committed.size());
        assertTrue(dataset.listIndexes().contains(TestVectorDataset.indexName));
      }
    }
  }

  @Test
  public void testCreateIvfRqIndex(@TempDir Path tempDir) throws Exception {
    Path datasetPath = tempDir.resolve("ivf_rq_index");

    try (TestVectorDataset testVectorDataset = new TestVectorDataset(datasetPath)) {
      try (Dataset dataset = testVectorDataset.create()) {
        IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(2).build();
        RQBuildParams rq = new RQBuildParams.Builder().setNumBits((byte) 1).build();

        VectorIndexParams vectorIndexParams =
            VectorIndexParams.withIvfRqParams(DistanceType.L2, ivf, rq);
        IndexParams indexParams =
            IndexParams.builder().setVectorIndexParams(vectorIndexParams).build();

        dataset.createIndex(
            IndexOptions.builder(
                    Collections.singletonList(TestVectorDataset.vectorColumnName),
                    IndexType.IVF_RQ,
                    indexParams)
                .withIndexName(TestVectorDataset.indexName)
                .build());

        List<Index> indexes = dataset.getIndexes();
        Index rqIndex =
            indexes.stream()
                .filter(idx -> TestVectorDataset.indexName.equals(idx.name()))
                .findFirst()
                .orElse(null);

        assertNotNull(rqIndex, "Expected IVF_RQ index to be present");

        IndexType indexType = rqIndex.indexType();
        assertNotNull(indexType, "IndexType should be set for IVF_RQ index");

        // Today all vector indices share the same VectorIndexDetails type and map to VECTOR.
        // This assertion allows both VECTOR and IVF_RQ so it remains valid if the mapping
        // is refined in the future.
        assertTrue(
            indexType == IndexType.VECTOR || indexType == IndexType.IVF_RQ,
            "IndexType for IVF_RQ index should be VECTOR or IVF_RQ but was " + indexType);
      }
    }
  }

  /**
   * Regression test for the metric_type passthrough in the JNI VectorTrainer.
   *
   * <p>Before this fix, {@code trainIvfCentroids} hardcoded {@code MetricType::L2} on the Rust
   * side, so users requesting a non-L2 metric got centroids clustered on L2 geometry while
   * per-fragment encoders later quantized using the requested metric — silently degraded recall.
   *
   * <p>This test trains the same dataset twice with L2 and Cosine and asserts the centroid arrays
   * differ. With the bug present, the two arrays were identical because both paths fell through to
   * L2.
   */
  @Test
  public void testTrainIvfCentroidsHonorsDistanceType(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("ivf_centroids_metric"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        IvfBuildParams params =
            new IvfBuildParams.Builder().setNumPartitions(2).setMaxIters(10).build();

        float[] l2Centroids =
            VectorTrainer.trainIvfCentroids(
                dataset, TestVectorDataset.vectorColumnName, params, DistanceType.L2);
        float[] cosineCentroids =
            VectorTrainer.trainIvfCentroids(
                dataset, TestVectorDataset.vectorColumnName, params, DistanceType.Cosine);

        assertEquals(l2Centroids.length, cosineCentroids.length);
        assertFalse(
            arraysApproximatelyEqual(l2Centroids, cosineCentroids),
            "L2 and Cosine centroids should differ — Cosine normalizes input before clustering."
                + " If they are equal, the metric is not being threaded into the trainer.");
      }
    }
  }

  /**
   * Regression test for the metric_type passthrough in PQ codebook training.
   *
   * <p>Cosine training normalizes the input vectors before k-means; L2 does not. So the resulting
   * codebooks must differ when the metric is honored.
   */
  @Test
  public void testTrainPqCodebookHonorsDistanceType(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("pq_codebook_metric"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        PQBuildParams params =
            new PQBuildParams.Builder()
                .setNumSubVectors(2)
                .setNumBits(4)
                .setMaxIters(5)
                .setSampleRate(64)
                .build();

        float[] l2Codebook =
            VectorTrainer.trainPqCodebook(
                dataset, TestVectorDataset.vectorColumnName, params, DistanceType.L2);
        float[] cosineCodebook =
            VectorTrainer.trainPqCodebook(
                dataset, TestVectorDataset.vectorColumnName, params, DistanceType.Cosine);

        assertEquals(l2Codebook.length, cosineCodebook.length);
        assertFalse(
            arraysApproximatelyEqual(l2Codebook, cosineCodebook),
            "L2 and Cosine PQ codebooks should differ — Cosine normalizes the training data."
                + " If they are equal, the metric is not being threaded into the trainer.");
      }
    }
  }

  private static boolean arraysApproximatelyEqual(float[] a, float[] b) {
    if (a.length != b.length) {
      return false;
    }
    final float epsilon = 1e-6f;
    for (int i = 0; i < a.length; i++) {
      if (Math.abs(a[i] - b[i]) > epsilon) {
        return false;
      }
    }
    return true;
  }

  private static boolean causeChainContains(Throwable failure, String expected) {
    for (Throwable current = failure; current != null; current = current.getCause()) {
      if (current.getMessage() != null && current.getMessage().contains(expected)) {
        return true;
      }
    }
    return false;
  }
}
