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
import org.lance.ipc.Query;
import org.lance.ipc.ScanOptions;

import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;

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
   * {@code coveringColumns} must be a defensive, unmodifiable copy: mutating the list the caller
   * passed to the builder (or the list the getter returns) must not change the params, so the JNI
   * read at build time sees exactly what was requested.
   */
  @Test
  public void testCoveringColumnsIsDefensivelyCopied() {
    IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(2).build();
    List<String> cols = new ArrayList<>();
    cols.add("i");
    VectorIndexParams params = new VectorIndexParams.Builder(ivf).setCoveringColumns(cols).build();

    // Mutating the caller's list after build must not affect the params.
    cols.add("s");
    cols.clear();
    assertEquals(
        Collections.singletonList("i"),
        params.getCoveringColumns(),
        "include columns must be a defensive copy, unaffected by caller mutation");

    // The returned list must be unmodifiable.
    assertThrows(UnsupportedOperationException.class, () -> params.getCoveringColumns().add("x"));
  }

  /**
   * {@code Index.coveringFields} must be a defensive, unmodifiable copy, matching {@code
   * VectorIndexParams.coveringColumns}: mutating the caller's list (or the list the getter returns)
   * must not change the value object -- equals/hashCode would silently drift, and a later JNI
   * commit would read mutated covering metadata while the index files still carry the payload.
   */
  @Test
  public void testIndexCoveringFieldsIsDefensivelyCopied() {
    List<Integer> ids = new ArrayList<>();
    ids.add(3);
    Index index =
        Index.builder()
            .uuid(UUID.randomUUID())
            .fields(Collections.singletonList(0))
            .name("covered_idx")
            .datasetVersion(1L)
            .indexVersion(0)
            .coveringFields(ids)
            .build();

    // Mutating the caller's list after build must not affect the index.
    ids.add(7);
    ids.clear();
    assertEquals(
        Collections.singletonList(3),
        index.coveringFields(),
        "covering fields must be a defensive copy, unaffected by caller mutation");

    // The returned list must be unmodifiable.
    assertThrows(UnsupportedOperationException.class, () -> index.coveringFields().add(9));
  }

  /**
   * A covered ("included") column set on {@link VectorIndexParams} must be threaded through the JNI
   * create path into the built index's metadata, so the committed index reports the covered field
   * id. Without the wiring the index would build but silently carry no covering columns.
   *
   * <p>This guards the {@code covering_columns} read in {@code java/lance-jni/src/utils.rs}:
   * restoring it to an empty default leaves the index buildable but drops the covering declaration,
   * and the field-id assertions below fail.
   */
  @Test
  public void testCreateIvfFlatIndexWithCoveringColumns(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("ivf_flat_covering"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        // Several partitions, so the search below really probes an IVF partition instead of
        // degenerating into a scan of the one partition that holds everything.
        IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(4).build();

        // Cover the non-vector "i" column so a projection of it is answered from the index.
        VectorIndexParams vectorIndexParams =
            new VectorIndexParams.Builder(ivf)
                .setDistanceType(DistanceType.L2)
                .setCoveringColumns(Collections.singletonList("i"))
                .build();
        IndexParams indexParams =
            IndexParams.builder().setVectorIndexParams(vectorIndexParams).build();

        dataset.createIndex(
            IndexOptions.builder(
                    Collections.singletonList(TestVectorDataset.vectorColumnName),
                    IndexType.IVF_FLAT,
                    indexParams)
                .withIndexName(TestVectorDataset.indexName)
                .build());

        Index covered =
            dataset.getIndexes().stream()
                .filter(idx -> TestVectorDataset.indexName.equals(idx.name()))
                .findFirst()
                .orElse(null);
        assertNotNull(covered, "Expected covered IVF_FLAT index to be present");

        int coveredFieldId = fieldId(dataset, "i");
        int vectorFieldId = fieldId(dataset, TestVectorDataset.vectorColumnName);
        assertEquals(
            Collections.singletonList(coveredFieldId),
            covered.coveringFields(),
            "committed index must report the requested covering column's field id");
        // covering fields are always the trailing entries of fields, so the keyed vector field
        // comes first and the covering field is appended after it.
        assertEquals(
            Arrays.asList(vectorFieldId, coveredFieldId),
            covered.fields(),
            "covering field must be appended after the keyed vector field");

        float[] key = new float[32];
        for (int j = 0; j < key.length; j++) {
          key[j] = (float) (32 + j);
        }
        int k = 5;

        // The assertion that actually exercises the covering payload: project the covered
        // column together with an *uncovered* one and cross-check them row by row. Every row
        // of the fixture satisfies s == "s-" + i, and only "i" is carried by the index, so "s"
        // necessarily comes from a base-table take. A covering payload read positionally
        // rather than by name, or attached to the wrong row, makes the two disagree.
        Map<Integer, String> coveredRows = searchCoveredWithBaseColumn(dataset, key, k);
        assertEquals(k, coveredRows.size(), "covered search should return k distinct rows");
        coveredRows.forEach(
            (i, s) ->
                assertEquals(
                    "s-" + i,
                    s,
                    "covered 'i' must belong to the same row as the base-table 's'; a"
                        + " misaligned covering payload disagrees here"));

        // Recall against an exact scan. Worth keeping as a floor on search quality, but note
        // what it cannot show: it does NOT establish that the *index* served the query.
        // Covering is semantically transparent -- if the index were ignored and Lance fell
        // back to brute-force KNN, the fallback is exact and would score 1.0 here. Java has
        // no explain-plan binding, so there is no node chain to assert against as
        // test_create_index_covering_columns_serve_the_query_from_the_index does on the
        // Python side. That distinction is pinned instead by
        // testCoveredIndexServesTheSearchNotAFullScan below, which uses fast search over a
        // partially indexed dataset so the two outcomes genuinely differ.
        Set<Integer> exact = searchCoveredColumn(dataset, key, k, false);
        Set<Integer> ann = searchCoveredColumn(dataset, key, k, true);
        assertEquals(k, exact.size(), "exact KNN ground truth should return k distinct rows");
        Set<Integer> hits = new HashSet<>(ann);
        hits.retainAll(exact);
        double recall = (double) hits.size() / k;
        assertTrue(
            recall >= 0.5,
            "recall of the covered index against exact KNN must be at least 0.5 but was "
                + recall
                + " (ann="
                + ann
                + ", exact="
                + exact
                + ")");
      }
    }
  }

  /**
   * A covered search must actually be served by the index, not by a full scan that silently
   * produces the same answer.
   *
   * <p>Recall against an exact scan cannot show this: covering is semantically transparent, so if
   * the index became unusable and Lance fell back to brute-force KNN, the fallback is exact and
   * scores 1.0. Java has no explain-plan binding, so the Python trick of pinning the plan's node
   * chain is unavailable.
   *
   * <p>What is available is fast search, which restricts a query to indexed fragments. Index only 2
   * of the dataset's 5 fragments, and the two outcomes stop agreeing: {@code TestVectorDataset}
   * writes the *same* 80 vectors into every fragment, so the query key matches one row at distance
   * 0 in each of the 5 fragments ("i" = 1, 81, 161, 241, 321) with the next-nearest row ~181 away.
   * Served by the index under fast search, only the two indexed fragments can contribute, so every
   * returned "i" is below 160. Served by a full scan, the distance-0 rows from the three unindexed
   * fragments win and "i" values above 160 appear.
   */
  @Test
  public void testCoveredIndexServesTheSearchNotAFullScan(@TempDir Path tempDir) throws Exception {
    try (TestVectorDataset testVectorDataset =
        new TestVectorDataset(tempDir.resolve("ivf_flat_covering_fast_search"))) {
      try (Dataset dataset = testVectorDataset.create()) {
        List<Fragment> fragments = dataset.getFragments();
        assertTrue(fragments.size() >= 3, "fixture must have unindexed fragments left over");

        IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(4).build();
        VectorIndexParams vectorIndexParams =
            new VectorIndexParams.Builder(ivf)
                .setDistanceType(DistanceType.L2)
                .setCoveringColumns(Collections.singletonList("i"))
                .build();
        IndexParams indexParams =
            IndexParams.builder().setVectorIndexParams(vectorIndexParams).build();

        List<Index> segments = new ArrayList<>();
        for (int f = 0; f < 2; f++) {
          segments.add(
              dataset.createIndex(
                  IndexOptions.builder(
                          Collections.singletonList(TestVectorDataset.vectorColumnName),
                          IndexType.IVF_FLAT,
                          indexParams)
                      .withIndexName(TestVectorDataset.indexName)
                      .withFragmentIds(Collections.singletonList(fragments.get(f).getId()))
                      .build()));
        }
        dataset.commitExistingIndexSegments(
            TestVectorDataset.indexName, TestVectorDataset.vectorColumnName, segments);

        int coveredFieldId = fieldId(dataset, "i");
        for (Index segment : segments) {
          assertEquals(
              Collections.singletonList(coveredFieldId),
              segment.coveringFields(),
              "every distributed segment must carry the covering declaration");
        }

        float[] key = new float[32];
        for (int j = 0; j < key.length; j++) {
          key[j] = (float) (32 + j);
        }

        ScanOptions options =
            new ScanOptions.Builder()
                .columns(Collections.singletonList("i"))
                .fastSearch(true)
                .nearest(
                    new Query.Builder()
                        .setColumn(TestVectorDataset.vectorColumnName)
                        .setKey(key)
                        .setK(5)
                        .setUseIndex(true)
                        .build())
                .build();

        Set<Integer> values = new HashSet<>();
        try (Scanner scanner = dataset.newScan(options);
            ArrowReader reader = scanner.scanBatches()) {
          VectorSchemaRoot root = reader.getVectorSchemaRoot();
          while (reader.loadNextBatch()) {
            IntVector iVector = (IntVector) root.getVector("i");
            for (int row = 0; row < root.getRowCount(); row++) {
              values.add(iVector.get(row));
            }
          }
        }

        assertFalse(values.isEmpty(), "fast search over the covered index returned no rows");
        // Positive: the exact-match rows of both indexed fragments must be present.
        assertTrue(
            values.contains(1) && values.contains(81),
            "fast search must return the distance-0 row of each indexed fragment, got " + values);
        // Discriminating: any row from an unindexed fragment means a full scan served this.
        for (int value : values) {
          assertTrue(
              value < 160,
              "fast search must be confined to the 2 indexed fragments, but returned i="
                  + value
                  + " (a full scan, not the covered index, served this query); got "
                  + values);
        }
      }
    }
  }

  /** Resolve a top-level column's Lance field id, which is what index metadata records. */
  private static int fieldId(Dataset dataset, String name) {
    return dataset.getLanceSchema().fields().stream()
        .filter(field -> name.equals(field.getName()))
        .findFirst()
        .orElseThrow(() -> new AssertionError("No field named " + name + " in the dataset schema"))
        .getId();
  }

  /**
   * Run a nearest-neighbor search projecting the covered "i" column together with the uncovered "s"
   * column, returning i -&gt; s for the matched rows. Only "i" is carried by the index, so "s"
   * comes from a base-table take and the pair cross-checks the covering payload's row alignment.
   */
  private static Map<Integer, String> searchCoveredWithBaseColumn(
      Dataset dataset, float[] key, int k) throws Exception {
    ScanOptions options =
        new ScanOptions.Builder()
            .columns(Arrays.asList("i", "s"))
            .nearest(
                new Query.Builder()
                    .setColumn(TestVectorDataset.vectorColumnName)
                    .setKey(key)
                    .setK(k)
                    .setUseIndex(true)
                    .build())
            .build();

    Map<Integer, String> rows = new HashMap<>();
    try (Scanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      while (reader.loadNextBatch()) {
        IntVector iVector = (IntVector) root.getVector("i");
        VarCharVector sVector = (VarCharVector) root.getVector("s");
        for (int row = 0; row < root.getRowCount(); row++) {
          rows.put(iVector.get(row), new String(sVector.get(row), StandardCharsets.UTF_8));
        }
      }
    }
    return rows;
  }

  /**
   * Run a nearest-neighbor search projecting only the covered "i" column and return the matched
   * values, so a covered result can be compared against exact KNN ground truth.
   */
  private static Set<Integer> searchCoveredColumn(
      Dataset dataset, float[] key, int k, boolean useIndex) throws Exception {
    ScanOptions options =
        new ScanOptions.Builder()
            .columns(Collections.singletonList("i"))
            .nearest(
                new Query.Builder()
                    .setColumn(TestVectorDataset.vectorColumnName)
                    .setKey(key)
                    .setK(k)
                    .setUseIndex(useIndex)
                    .build())
            .build();

    Set<Integer> values = new HashSet<>();
    try (Scanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      while (reader.loadNextBatch()) {
        IntVector iVector = (IntVector) root.getVector("i");
        for (int row = 0; row < root.getRowCount(); row++) {
          values.add(iVector.get(row));
        }
      }
    }
    return values;
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
