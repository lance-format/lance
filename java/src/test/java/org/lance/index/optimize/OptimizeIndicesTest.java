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
package org.lance.index.optimize;

import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.FragmentMetadata;
import org.lance.FragmentOperation;
import org.lance.WriteParams;
import org.lance.index.DistanceType;
import org.lance.index.Index;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.OptimizeOptions;
import org.lance.index.vector.VectorIndexParams;
import org.lance.ipc.Query;
import org.lance.ipc.ScanOptions;

import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.commons.lang3.SerializationUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.Serializable;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class OptimizeIndicesTest {
  private static final int DIM = 8;
  private static final String VECTOR_COLUMN = "vec";
  private static final String INDEX_NAME = "vec_idx";
  private static final Schema SCHEMA =
      new Schema(
          Arrays.asList(
              Field.notNullable("id", new ArrowType.Int(32, true)),
              new Field(
                  VECTOR_COLUMN,
                  FieldType.nullable(new ArrowType.FixedSizeList(DIM)),
                  Collections.singletonList(
                      Field.nullable(
                          "item", new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE))))));

  /**
   * The driver plans; workers receive each task and merge by Java serialization and run it on a
   * dataset they open themselves; the driver commits what comes back.
   */
  @Test
  void testDistributedOptimizeEndToEnd(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("dataset").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      Dataset.create(allocator, path, SCHEMA, new WriteParams.Builder().build()).close();
      appendFragments(allocator, path, 1, 0, 1, 100); // fragment 0: ids 0..99
      try (Dataset driver = Dataset.open(path)) {
        driver.createIndex(
            Collections.singletonList(VECTOR_COLUMN),
            IndexType.IVF_FLAT,
            Optional.of(INDEX_NAME),
            IndexParams.builder()
                .setVectorIndexParams(VectorIndexParams.ivfFlat(2, DistanceType.L2))
                .build(),
            true);
        appendFragments(allocator, path, driver.version(), 100, 4, 50); // fragments 1..4
        driver.checkoutLatest();

        // The 50-row fragments exceed the 40-row task target, so each is a task of its own.
        // The 100-row segment target still packs pairs of task results into two output segments.
        OptimizeOptions options =
            OptimizeOptions.builder().maxRowsPerTask(40).maxRowsPerSegment(100).build();
        OptimizeIndicesPlan plan = OptimizeIndices.plan(driver, options);
        assertEquals(
            Arrays.asList(
                Collections.singletonList(1),
                Collections.singletonList(2),
                Collections.singletonList(3),
                Collections.singletonList(4)),
            plan.getTasks().stream()
                .map(OptimizeIndexTask::getFragments)
                .collect(Collectors.toList()));
        assertEquals(2, plan.getMerges().size());
        assertEquals(
            Arrays.asList(0, 0, 1, 1),
            plan.getTasks().stream().map(OptimizeIndexTask::getGroup).collect(Collectors.toList()));

        List<IndexSegmentResult> built = new ArrayList<>();
        for (OptimizeIndexTask task : plan.getTasks()) {
          try (Dataset worker = Dataset.open(path)) {
            built.add(ship(ship(task).execute(worker).get()));
          }
        }
        List<IndexSegmentResult> merged = new ArrayList<>();
        for (OptimizeIndexMerge merge : plan.getMerges()) {
          List<IndexSegmentResult> inputs =
              built.stream()
                  .filter(result -> result.getGroup() == merge.getGroup())
                  .collect(Collectors.toList());
          try (Dataset worker = Dataset.open(path)) {
            merged.add(ship(ship(merge).execute(worker, inputs).get()));
          }
        }

        long version = driver.version();
        OptimizeIndices.commit(driver, merged, options);
        assertEquals(version + 1, driver.version());
        List<Index> segments =
            driver.getIndexes().stream()
                .filter(index -> index.name().equals(INDEX_NAME))
                .collect(Collectors.toList());
        assertEquals(3, segments.size());
        assertEquals(
            IntStream.rangeClosed(0, 4).boxed().collect(Collectors.toSet()),
            segments.stream()
                .flatMap(segment -> segment.fragments().get().stream())
                .collect(Collectors.toSet()));
        assertEquals(250, nearestId(driver, vector(250)));
      }
    }
  }

  @ParameterizedTest(name = "distributed={0}")
  @ValueSource(booleans = {false, true})
  void testBoundedVectorConsolidation(boolean distributed, @TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("dataset").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      Dataset.create(allocator, path, SCHEMA, new WriteParams.Builder().build()).close();
      appendFragments(allocator, path, 1, 0, 1, 100);
      try (Dataset dataset = Dataset.open(path)) {
        dataset.createIndex(
            List.of(VECTOR_COLUMN),
            IndexType.IVF_FLAT,
            Optional.of(INDEX_NAME),
            IndexParams.builder()
                .setVectorIndexParams(VectorIndexParams.ivfFlat(2, DistanceType.L2))
                .build(),
            true);
        OptimizeOptions append = OptimizeOptions.builder().numIndicesToMerge(0).build();
        for (int firstId : new int[] {100, 150}) {
          appendFragments(allocator, path, dataset.version(), firstId, 1, 50);
          dataset.checkoutLatest();
          dataset.optimizeIndices(append);
        }
        appendFragments(allocator, path, dataset.version(), 200, 1, 50);
        dataset.checkoutLatest();

        OptimizeOptions options = OptimizeOptions.builder().maxRowsToConsolidate(100).build();
        if (distributed) {
          OptimizeIndicesPlan plan = OptimizeIndices.plan(dataset, options);
          assertEquals(1, plan.getMerges().size());
          IndexSegmentResult merged = plan.getMerges().get(0).execute(dataset, List.of()).get();
          IllegalArgumentException error =
              assertThrows(
                  IllegalArgumentException.class,
                  () -> OptimizeIndices.commit(dataset, List.of(merged, merged), options));
          assertTrue(error.getMessage().contains("consolidation results replace a source twice"));
          OptimizeIndices.commit(dataset, List.of(merged), options);
        } else {
          dataset.optimizeIndices(options);
        }

        List<Index> segments = dataset.getIndexes();
        assertEquals(2, segments.size());
        assertEquals(
            Set.of(List.of(0), List.of(1, 2)),
            segments.stream()
                .map(segment -> segment.fragments().get())
                .collect(Collectors.toSet()));
        assertEquals(175, nearestId(dataset, vector(175)));
      }
    }
  }

  /** A worker gets the unit as the engine shipped it: through Java serialization. */
  private static <T extends Serializable> T ship(T unit) {
    return SerializationUtils.clone(unit);
  }

  private static void appendFragments(
      BufferAllocator allocator,
      String path,
      long readVersion,
      int firstId,
      int count,
      int rowsPerFragment) {
    List<FragmentMetadata> fragments = new ArrayList<>();
    for (int f = 0; f < count; f++) {
      try (VectorSchemaRoot root = VectorSchemaRoot.create(SCHEMA, allocator)) {
        root.allocateNew();
        IntVector ids = (IntVector) root.getVector("id");
        FixedSizeListVector vectors = (FixedSizeListVector) root.getVector(VECTOR_COLUMN);
        Float4Vector items = (Float4Vector) vectors.getDataVector();
        for (int i = 0; i < rowsPerFragment; i++) {
          int id = firstId + f * rowsPerFragment + i;
          ids.setSafe(i, id);
          vectors.setNotNull(i);
          float[] vector = vector(id);
          for (int j = 0; j < DIM; j++) {
            items.setSafe(i * DIM + j, vector[j]);
          }
        }
        root.setRowCount(rowsPerFragment);
        fragments.addAll(Fragment.create(path, allocator, root, new WriteParams.Builder().build()));
      }
    }
    Dataset.commit(
            allocator, path, new FragmentOperation.Append(fragments), Optional.of(readVersion))
        .close();
  }

  /** A vector only row {@code id} has; odd and even ids land in different IVF partitions. */
  private static float[] vector(int id) {
    float[] vector = new float[DIM];
    for (int j = 0; j < DIM; j++) {
      vector[j] = (id % 2 == 0 ? 1 : -1) * (id + j * 0.1f);
    }
    return vector;
  }

  /** The id of the row nearest to {@code key}, through the vector index. */
  private static int nearestId(Dataset dataset, float[] key) throws Exception {
    ScanOptions options =
        new ScanOptions.Builder()
            .nearest(
                new Query.Builder()
                    .setColumn(VECTOR_COLUMN)
                    .setKey(key)
                    .setK(1)
                    .setUseIndex(true)
                    .build())
            .build();
    try (Scanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      assertTrue(reader.loadNextBatch());
      return ((IntVector) reader.getVectorSchemaRoot().getVector("id")).get(0);
    }
  }
}
