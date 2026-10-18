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
import org.lance.WriteParams;
import org.lance.index.vector.HnswBuildParams;
import org.lance.index.vector.IvfBuildParams;
import org.lance.index.vector.PQBuildParams;
import org.lance.index.vector.RQBuildParams;
import org.lance.index.vector.SQBuildParams;
import org.lance.index.vector.VamanaBuildParams;
import org.lance.index.vector.VectorIndexParams;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.Query;
import org.lance.ipc.ScanOptions;
import org.lance.util.JsonUtils;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * IVF_VAMANA_* indices through the Java API. The graph itself is tested in Rust; these tests check
 * that Java reaches it: the index types and their parameters, the errors Lance raises for what it
 * refuses, search through the scanner, optimize, and segments built apart.
 */
public class VamanaIndexTest {
  private static final int ROWS = 600;
  private static final int DIM = 32;
  // A graph over product codes draws several times as slowly as one over scalar codes in a debug
  // build, so those cases index fewer dimensions, in four sub-vectors of two.
  private static final int PQ_DIM = 8;
  private static final int K = 10;
  private static final int EF = 60;
  private static final String COLUMN = "vector";
  private static final String INDEX = "vector_idx";
  // Each unlike its default, so that the test sees each one reach the build.
  private static final int MAX_DEGREE = 16;
  private static final int SEARCH_LIST_SIZE = 32;
  private static final float ALPHA = 1.1f;
  private static final VamanaBuildParams GRAPH = graph().build();

  @ParameterizedTest
  @CsvSource({
    "IVF_VAMANA_FLAT, L2",
    "IVF_VAMANA_FLAT, Cosine",
    "IVF_VAMANA_SQ, L2",
    "IVF_VAMANA_SQ, Cosine",
    "IVF_VAMANA_PQ, L2",
    "IVF_VAMANA_PQ, Cosine"
  })
  public void testAnIndexIsBuiltDescribedAndSearched(
      IndexType indexType, DistanceType metric, @TempDir Path tempDir) throws Exception {
    float[][] vectors = vectors(ROWS, dimensionOf(indexType));
    boolean[] nulls = new boolean[ROWS];
    for (int row = 0; row < ROWS; row += 25) {
      nulls[row] = true;
    }
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset =
            write(allocator, uri(tempDir), vectors, 0, nulls, WriteParams.WriteMode.CREATE)) {
      dataset.createIndex(indexOptions(indexType, metric, GRAPH));

      List<IndexDescription> descriptions = dataset.describeIndices();
      assertEquals(1, descriptions.size());
      assertEquals(indexType.name(), descriptions.get(0).getIndexType());
      assertGraph(
          MAX_DEGREE,
          SEARCH_LIST_SIZE,
          ALPHA,
          JsonUtils.fromJson(descriptions.get(0).getDetailsJson()).get("vamana"));
      Map<String, Object> stats = dataset.getIndexStatistics(INDEX);
      assertEquals(indexType.name(), stats.get("index_type"));
      for (Object segment : (List<?>) stats.get("indices")) {
        Map<?, ?> segmentStats = (Map<?, ?>) segment;
        assertEquals(metric.name().toLowerCase(Locale.ROOT), segmentStats.get("metric_type"));
        assertEquals(2, segmentStats.get("num_partitions"));
        Map<?, ?> subIndex = (Map<?, ?>) segmentStats.get("sub_index");
        assertEquals("VAMANA", subIndex.get("index_type"));
        assertGraph(MAX_DEGREE, SEARCH_LIST_SIZE, ALPHA, subIndex.get("params"));
      }

      assertTrue(recall(dataset, vectors, metric, nulls) >= 0.95);

      // As over HNSW, a search list shorter than the results is refused.
      IOException failure =
          assertThrows(IOException.class, () -> search(dataset, vectors[1], metric, K - 1));
      assertTrue(
          failure.getMessage().contains("ef must be greater than or equal to k"),
          failure.getMessage());
    }
  }

  static Stream<Arguments> refusals() {
    IvfBuildParams ivf =
        new IvfBuildParams.Builder()
            .setNumPartitions(2)
            .setCentroids(flatten(centroids(DistanceType.L2, DIM)))
            .build();
    SQBuildParams sq = new SQBuildParams.Builder().build();
    return Stream.of(
        Arguments.of(
            VectorIndexParams.withIvfVamanaSqParams(
                DistanceType.L2, ivf, graph().setMaxDegree(0).build(), sq),
            IllegalArgumentException.class,
            "Vamana max_degree must be between 1 and 1024, got 0"),
        Arguments.of(
            VectorIndexParams.withIvfVamanaSqParams(
                DistanceType.L2, ivf, graph().setMaxDegree(-1).build(), sq),
            IllegalArgumentException.class,
            "Vamana max_degree must not be negative, got -1"),
        Arguments.of(
            VectorIndexParams.withIvfVamanaSqParams(
                DistanceType.L2, ivf, graph().setSearchListSize(0).build(), sq),
            IllegalArgumentException.class,
            "Vamana search_list_size must be greater than zero"),
        Arguments.of(
            VectorIndexParams.withIvfVamanaSqParams(
                DistanceType.L2, ivf, graph().setSearchListSize(-1).build(), sq),
            IllegalArgumentException.class,
            "Vamana search_list_size must not be negative, got -1"),
        Arguments.of(
            VectorIndexParams.withIvfVamanaSqParams(
                DistanceType.L2, ivf, graph().setAlpha(0.9f).build(), sq),
            IllegalArgumentException.class,
            "Vamana alpha must be a finite value of at least 1.0"),
        Arguments.of(
            VectorIndexParams.withIvfVamanaSqParams(DistanceType.Dot, ivf, GRAPH, sq),
            UnsupportedOperationException.class,
            "IVF_VAMANA supports the l2 and cosine distances"),
        Arguments.of(
            VectorIndexParams.withIvfVamanaSqParams(
                DistanceType.L2,
                ivf,
                GRAPH,
                new SQBuildParams.Builder().setNumBits((short) 16).build()),
            UnsupportedOperationException.class,
            "IVF_VAMANA_SQ encodes 8-bit scalar codes only"),
        Arguments.of(
            new VectorIndexParams.Builder(ivf)
                .setVamanaParams(GRAPH)
                .setRqParams(new RQBuildParams.Builder().build())
                .build(),
            UnsupportedOperationException.class,
            "at most one PQ or SQ stage"));
  }

  @ParameterizedTest
  @MethodSource("refusals")
  public void testLanceRefusesWhatItCannotBuild(
      VectorIndexParams params,
      Class<? extends RuntimeException> error,
      String message,
      @TempDir Path tempDir)
      throws Exception {
    float[][] vectors = vectors(ROWS, DIM);
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset =
            write(allocator, uri(tempDir), vectors, 0, null, WriteParams.WriteMode.CREATE)) {
      IndexOptions options =
          IndexOptions.builder(
                  List.of(COLUMN),
                  IndexType.IVF_VAMANA_SQ,
                  IndexParams.builder().setVectorIndexParams(params).build())
              .withIndexName(INDEX)
              .build();
      RuntimeException failure = assertThrows(error, () -> dataset.createIndex(options));
      assertTrue(failure.getMessage().contains(message), failure.getMessage());
      assertTrue(dataset.describeIndices().isEmpty());
    }
  }

  @Test
  public void testHnswAndVamanaExcludeEachOther() {
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new VectorIndexParams.Builder(new IvfBuildParams.Builder().build())
                    .setHnswParams(new HnswBuildParams.Builder().build())
                    .setVamanaParams(GRAPH)
                    .setSqParams(new SQBuildParams.Builder().build())
                    .build());
    assertEquals("Only one of HNSW or Vamana can be specified at a time.", failure.getMessage());
  }

  @Test
  public void testUnsetParametersTakeLancesDefaults(@TempDir Path tempDir) throws Exception {
    // A small table: the defaults draw a far denser graph than the other cases need.
    float[][] vectors = vectors(100, DIM);
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset =
            write(allocator, uri(tempDir), vectors, 0, null, WriteParams.WriteMode.CREATE)) {
      dataset.createIndex(
          indexOptions(
              IndexType.IVF_VAMANA_FLAT, DistanceType.L2, new VamanaBuildParams.Builder().build()));

      for (Object segment : (List<?>) dataset.getIndexStatistics(INDEX).get("indices")) {
        Map<?, ?> subIndex = (Map<?, ?>) ((Map<?, ?>) segment).get("sub_index");
        assertGraph(64, 100, 1.2f, subIndex.get("params"));
      }
      assertTrue(recall(dataset, vectors, DistanceType.L2, null) >= 0.95);
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {0, 1})
  public void testAppendedRowsAreFoundBeforeAndAfterOptimize(
      int numIndicesToMerge, @TempDir Path tempDir) throws Exception {
    float[][] vectors = vectors(ROWS + 100, DIM);
    String uri = uri(tempDir);
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          write(
              allocator,
              uri,
              Arrays.copyOfRange(vectors, 0, ROWS),
              0,
              null,
              WriteParams.WriteMode.CREATE)) {
        dataset.createIndex(indexOptions(IndexType.IVF_VAMANA_SQ, DistanceType.L2, GRAPH));
      }
      try (Dataset dataset =
          write(
              allocator,
              uri,
              Arrays.copyOfRange(vectors, ROWS, ROWS + 100),
              ROWS,
              null,
              WriteParams.WriteMode.APPEND)) {
        assertEquals(100, dataset.getIndexStatistics(INDEX).get("num_unindexed_rows"));

        for (boolean optimized : new boolean[] {false, true}) {
          if (optimized) {
            dataset.optimizeIndices(
                OptimizeOptions.builder().numIndicesToMerge(numIndicesToMerge).build());
          }
          Map<Long, Float> result = search(dataset, vectors[ROWS + 17], DistanceType.L2, EF);
          assertEquals(ROWS + 17L, result.keySet().iterator().next());
          assertTrue(recall(dataset, vectors, DistanceType.L2, null) >= 0.95);
        }

        Map<String, Object> stats = dataset.getIndexStatistics(INDEX);
        assertEquals(0, stats.get("num_unindexed_rows"));
        assertEquals(ROWS + 100, stats.get("num_indexed_rows"));
        assertEquals(2 - numIndicesToMerge, ((List<?>) stats.get("indices")).size());
      }
    }
  }

  @Test
  public void testSegmentsBuiltApartAreMergedAndSearched(@TempDir Path tempDir) throws Exception {
    float[][] vectors = vectors(ROWS, DIM);
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset =
            write(allocator, uri(tempDir), vectors, 0, null, WriteParams.WriteMode.CREATE)) {
      List<Index> segments = new ArrayList<>();
      for (Fragment fragment : dataset.getFragments()) {
        IndexOptions options =
            indexOptionsBuilder(IndexType.IVF_VAMANA_SQ, DistanceType.L2, GRAPH)
                .withFragmentIds(List.of(fragment.getId()))
                .build();
        segments.add(dataset.createIndex(options));
      }
      Index merged = dataset.mergeExistingIndexSegments(segments);
      dataset.commitExistingIndexSegments(INDEX, COLUMN, List.of(merged));

      Map<String, Object> stats = dataset.getIndexStatistics(INDEX);
      assertEquals("IVF_VAMANA_SQ", stats.get("index_type"));
      assertEquals(1, stats.get("num_segments"));
      assertEquals(ROWS, stats.get("num_indexed_rows"));
      assertTrue(recall(dataset, vectors, DistanceType.L2, null) >= 0.95);
    }
  }

  private static VamanaBuildParams.Builder graph() {
    return new VamanaBuildParams.Builder()
        .setMaxDegree(MAX_DEGREE)
        .setSearchListSize(SEARCH_LIST_SIZE)
        .setAlpha(ALPHA);
  }

  private static int dimensionOf(IndexType indexType) {
    return indexType == IndexType.IVF_VAMANA_PQ ? PQ_DIM : DIM;
  }

  private static String uri(Path tempDir) {
    return tempDir.resolve("ds.lance").toString();
  }

  private static float[] randomVector(Random random, int dimension) {
    float[] vector = new float[dimension];
    for (int j = 0; j < dimension; j++) {
      vector[j] = random.nextFloat();
    }
    return vector;
  }

  private static float[][] vectors(int rows, int dimension) {
    Random random = new Random(3);
    float[][] vectors = new float[rows][];
    for (int row = 0; row < rows; row++) {
      vectors[row] = randomVector(random, dimension);
    }
    return vectors;
  }

  private static float[] flatten(float[][] vectors) {
    float[] values = new float[vectors.length * vectors[0].length];
    for (int row = 0; row < vectors.length; row++) {
      System.arraycopy(vectors[row], 0, values, row * vectors[row].length, vectors[row].length);
    }
    return values;
  }

  /**
   * `vectors` as rows with ids from `firstId` on, in two fragments; a row `nulls` marks holds no
   * vector.
   */
  private static Dataset write(
      BufferAllocator allocator,
      String uri,
      float[][] vectors,
      int firstId,
      boolean[] nulls,
      WriteParams.WriteMode mode) {
    int dimension = vectors[0].length;
    Field item =
        new Field(
            "item",
            FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)),
            null);
    Schema schema =
        new Schema(
            List.of(
                new Field("id", FieldType.notNullable(new ArrowType.Int(64, true)), null),
                new Field(
                    COLUMN,
                    FieldType.nullable(new ArrowType.FixedSizeList(dimension)),
                    List.of(item))));
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      BigIntVector ids = (BigIntVector) root.getVector("id");
      FixedSizeListVector column = (FixedSizeListVector) root.getVector(COLUMN);
      Float4Vector values = (Float4Vector) column.getDataVector();
      for (int row = 0; row < vectors.length; row++) {
        ids.setSafe(row, firstId + row);
        for (int j = 0; j < dimension; j++) {
          values.setSafe(row * dimension + j, vectors[row][j]);
        }
        if (nulls != null && nulls[row]) {
          column.setNull(row);
        } else {
          column.setNotNull(row);
        }
      }
      root.setRowCount(vectors.length);
      ArrowReader reader =
          new ArrowReader(allocator) {
            private boolean read = false;

            @Override
            public boolean loadNextBatch() {
              boolean first = !read;
              read = true;
              return first;
            }

            @Override
            public long bytesRead() {
              return 0;
            }

            @Override
            protected void closeReadSource() {}

            @Override
            protected Schema readSchema() {
              return schema;
            }

            @Override
            public VectorSchemaRoot getVectorSchemaRoot() {
              return root;
            }
          };
      Dataset dataset =
          Dataset.write()
              .allocator(allocator)
              .reader(reader)
              .uri(uri)
              .mode(mode)
              .maxRowsPerFile(vectors.length / 2)
              .execute();
      if (mode == WriteParams.WriteMode.CREATE) {
        assertEquals(2, dataset.getFragments().size());
      }
      return dataset;
    }
  }

  /**
   * Two IVF centroids that split the unit cube under `metric`: two points of its diagonal under l2;
   * under cosine, where the diagonal is one direction, unit vectors along the even axes and along
   * the odd ones.
   */
  private static float[][] centroids(DistanceType metric, int dimension) {
    float[][] centroids = new float[2][dimension];
    for (int j = 0; j < dimension; j++) {
      if (metric == DistanceType.Cosine) {
        centroids[j % 2][j] = (float) (1 / Math.sqrt(dimension / 2.0));
      } else {
        centroids[0][j] = 0.25f;
        centroids[1][j] = 0.75f;
      }
    }
    return centroids;
  }

  /**
   * Product codewords no build trains, the same on every run: the residuals of 256 seeded vectors
   * of the cube against the nearer centroid, as k-means would start from, normalized under cosine
   * as the codes measure them. Laid out sub-vector by sub-vector, as Lance reads a codebook.
   */
  private static float[] codebook(DistanceType metric, int dimension) {
    int subDimension = dimension / 4;
    float[][] centers = centroids(metric, dimension);
    Random random = new Random(5);
    float[] codebook = new float[256 * dimension];
    for (int code = 0; code < 256; code++) {
      float[] vector = randomVector(random, dimension);
      if (metric == DistanceType.Cosine) {
        double norm = Math.sqrt(dot(vector, vector));
        for (int j = 0; j < dimension; j++) {
          vector[j] = (float) (vector[j] / norm);
        }
      }
      float[] center =
          squaredL2(vector, centers[0]) <= squaredL2(vector, centers[1]) ? centers[0] : centers[1];
      for (int j = 0; j < dimension; j++) {
        codebook[((j / subDimension) * 256 + code) * subDimension + j % subDimension] =
            vector[j] - center[j];
      }
    }
    return codebook;
  }

  private static double dot(float[] left, float[] right) {
    double sum = 0;
    for (int j = 0; j < left.length; j++) {
      sum += (double) left[j] * right[j];
    }
    return sum;
  }

  private static double squaredL2(float[] left, float[] right) {
    double sum = 0;
    for (int j = 0; j < left.length; j++) {
      double difference = (double) left[j] - right[j];
      sum += difference * difference;
    }
    return sum;
  }

  /**
   * Centroids given, and for product codes a codebook, so that the codes are the same on every run.
   */
  private static IndexOptions.Builder indexOptionsBuilder(
      IndexType indexType, DistanceType metric, VamanaBuildParams graph) {
    int dimension = dimensionOf(indexType);
    IvfBuildParams ivf =
        new IvfBuildParams.Builder()
            .setNumPartitions(2)
            .setCentroids(flatten(centroids(metric, dimension)))
            .build();
    VectorIndexParams params;
    switch (indexType) {
      case IVF_VAMANA_FLAT:
        params = VectorIndexParams.ivfVamana(metric, ivf, graph);
        break;
      case IVF_VAMANA_SQ:
        params =
            VectorIndexParams.withIvfVamanaSqParams(
                metric, ivf, graph, new SQBuildParams.Builder().build());
        break;
      case IVF_VAMANA_PQ:
        PQBuildParams pq =
            new PQBuildParams.Builder()
                .setNumSubVectors(4)
                .setCodebook(codebook(metric, dimension))
                .build();
        params = VectorIndexParams.withIvfVamanaPqParams(metric, ivf, graph, pq);
        break;
      default:
        throw new IllegalArgumentException("Not an IVF_VAMANA_* type: " + indexType);
    }
    return IndexOptions.builder(
            List.of(COLUMN), indexType, IndexParams.builder().setVectorIndexParams(params).build())
        .withIndexName(INDEX);
  }

  private static IndexOptions indexOptions(
      IndexType indexType, DistanceType metric, VamanaBuildParams graph) {
    return indexOptionsBuilder(indexType, metric, graph).build();
  }

  /** The graph parameters, as Lance reports them in `params`. */
  private static void assertGraph(int maxDegree, int searchListSize, float alpha, Object params) {
    Map<?, ?> reported = (Map<?, ?>) params;
    assertEquals(Set.of("max_degree", "search_list_size", "alpha"), reported.keySet());
    assertEquals(maxDegree, reported.get("max_degree"));
    assertEquals(searchListSize, reported.get("search_list_size"));
    assertEquals(alpha, ((Number) reported.get("alpha")).floatValue());
  }

  /** The ids of the `K` rows nearest to `query`, nearest first, with their distances. */
  private static Map<Long, Float> search(
      Dataset dataset, float[] query, DistanceType metric, int ef) throws Exception {
    Query nearest =
        new Query.Builder()
            .setColumn(COLUMN)
            .setKey(query)
            .setK(K)
            .setNprobes(2)
            .setEf(ef)
            .setDistanceType(metric)
            .build();
    ScanOptions options =
        new ScanOptions.Builder().columns(List.of("id", "_distance")).nearest(nearest).build();
    Map<Long, Float> found = new LinkedHashMap<>();
    try (LanceScanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        BigIntVector ids = (BigIntVector) root.getVector("id");
        Float4Vector distances = (Float4Vector) root.getVector("_distance");
        for (int row = 0; row < root.getRowCount(); row++) {
          found.put(ids.get(row), distances.get(row));
        }
      }
    }
    return found;
  }

  private static double[] exactDistances(float[][] vectors, float[] query, DistanceType metric) {
    double[] distances = new double[vectors.length];
    for (int row = 0; row < vectors.length; row++) {
      distances[row] =
          metric == DistanceType.Cosine
              ? 1
                  - dot(vectors[row], query)
                      / Math.sqrt(dot(vectors[row], vectors[row]) * dot(query, query))
              : squaredL2(vectors[row], query);
    }
    return distances;
  }

  /**
   * recall@10 of ten seeded queries. Every distance returned must be the exact one: over scalar and
   * product codes the walk's shortlist is scored again from the dataset.
   */
  private static double recall(
      Dataset dataset, float[][] vectors, DistanceType metric, boolean[] nulls) throws Exception {
    Random random = new Random(7);
    int found = 0;
    for (int query = 0; query < 10; query++) {
      float[] key = randomVector(random, vectors[0].length);
      Map<Long, Float> result = search(dataset, key, metric, EF);
      double[] exact = exactDistances(vectors, key, metric);
      for (Map.Entry<Long, Float> entry : result.entrySet()) {
        int row = Math.toIntExact(entry.getKey());
        assertEquals(exact[row], entry.getValue(), 1e-5 + 1e-4 * Math.abs(exact[row]));
        assertFalse(nulls != null && nulls[row], "row " + row + " holds no vector");
      }
      Set<Long> nearest =
          IntStream.range(0, vectors.length)
              .filter(row -> nulls == null || !nulls[row])
              .boxed()
              .sorted(Comparator.comparingDouble(row -> exact[row]))
              .limit(K)
              .map(row -> (long) row)
              .collect(Collectors.toSet());
      found += (int) result.keySet().stream().filter(nearest::contains).count();
    }
    return found / (10.0 * K);
  }
}
