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
import org.lance.FragmentMetadata;
import org.lance.FragmentOperation;
import org.lance.WriteParams;
import org.lance.index.scalar.ScalarIndexParams;
import org.lance.index.scalar.ZoneStats;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/** Tests for {@link Dataset#getZonemapStats(String)} and the {@link ZoneStats} data class. */
public class ZonemapStatsTest {

  private static Schema intSchema() {
    return new Schema(
        Arrays.asList(
            Field.nullable("id", new ArrowType.Int(32, true)),
            Field.nullable("value", new ArrowType.Int(32, true))),
        null);
  }

  /** Write a single fragment with sequential integer values. */
  private Dataset writeIntFragment(
      BufferAllocator allocator, String path, long version, int startValue, int rowCount) {
    Schema schema = intSchema();
    List<FragmentMetadata> metas;
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector idVec = (IntVector) root.getVector("id");
      IntVector valVec = (IntVector) root.getVector("value");
      for (int i = 0; i < rowCount; i++) {
        idVec.setSafe(i, startValue + i);
        valVec.setSafe(i, (startValue + i) * 10);
      }
      root.setRowCount(rowCount);
      metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
    }
    FragmentOperation.Append appendOp = new FragmentOperation.Append(metas);
    return Dataset.commit(allocator, path, appendOp, Optional.of(version));
  }

  // -------------------------------------------------------
  // ZoneStats data class tests
  // -------------------------------------------------------

  @Test
  public void testZoneStatsGetters() {
    ZoneStats stats = new ZoneStats(3, 100, 50, 10L, 99L, 5);
    assertEquals(3, stats.getFragmentId());
    assertEquals(100, stats.getZoneStart());
    assertEquals(50, stats.getZoneLength());
    assertEquals(10L, stats.getMin());
    assertEquals(99L, stats.getMax());
    assertEquals(5, stats.getNullCount());
  }

  @Test
  public void testZoneStatsNullMinMax() {
    ZoneStats stats = new ZoneStats(0, 0, 10, null, null, 10);
    assertNull(stats.getMin());
    assertNull(stats.getMax());
    assertEquals(10, stats.getNullCount());
  }

  @Test
  public void testZoneStatsToString() {
    ZoneStats stats = new ZoneStats(1, 0, 100, 0L, 99L, 0);
    String str = stats.toString();
    assertTrue(str.contains("fragmentId=1"));
    assertTrue(str.contains("min=0"));
    assertTrue(str.contains("max=99"));
  }

  // -------------------------------------------------------
  // getZonemapStats integration tests
  // -------------------------------------------------------

  @Test
  public void testGetZonemapStatsNoIndex(@TempDir Path tempDir) {
    String path = tempDir.resolve("no_index").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        List<ZoneStats> stats = dataset.getZonemapStats("id");
        assertNotNull(stats);
        assertTrue(stats.isEmpty());
      }
    }
  }

  @Test
  public void testGetZonemapStatsNonexistentColumn(@TempDir Path tempDir) {
    String path = tempDir.resolve("bad_col").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        assertThrows(IllegalArgumentException.class, () -> dataset.getZonemapStats("nonexistent"));
      }
    }
  }

  @Test
  public void testGetZonemapStatsWithData(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("with_data").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      Dataset ds2 = writeIntFragment(allocator, path, 1, 0, 100);
      ds2.close();

      try (Dataset dataset = Dataset.open(path, allocator)) {
        ScalarIndexParams params = ScalarIndexParams.create("zonemap", "{}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(params).build();
        dataset.createIndex(
            Collections.singletonList("value"),
            IndexType.ZONEMAP,
            Optional.of("value_zm"),
            indexParams,
            true);

        List<ZoneStats> stats = dataset.getZonemapStats("value");
        assertNotNull(stats);
        assertFalse(stats.isEmpty());

        for (ZoneStats z : stats) {
          assertEquals(0, z.getFragmentId());
          assertNotNull(z.getMin());
          assertNotNull(z.getMax());
          assertTrue(z.getZoneLength() > 0);
        }

        ZoneStats first = stats.get(0);
        assertTrue(
            ((Number) first.getMin()).longValue() <= 0,
            "First zone min should be <= 0, got: " + first.getMin());

        ZoneStats last = stats.get(stats.size() - 1);
        assertTrue(
            ((Number) last.getMax()).longValue() >= 990,
            "Last zone max should be >= 990, got: " + last.getMax());
      }
    }
  }

  @Test
  public void testGetZonemapStatsMultiFragment(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("multi_frag").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      Dataset ds2 = writeIntFragment(allocator, path, 1, 0, 50);
      ds2.close();
      Dataset ds3 = writeIntFragment(allocator, path, 2, 50, 50);
      ds3.close();

      try (Dataset dataset = Dataset.open(path, allocator)) {
        assertEquals(2, dataset.getFragments().size());

        ScalarIndexParams params = ScalarIndexParams.create("zonemap", "{}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(params).build();
        dataset.createIndex(
            Collections.singletonList("value"),
            IndexType.ZONEMAP,
            Optional.of("value_zm"),
            indexParams,
            true);

        List<ZoneStats> stats = dataset.getZonemapStats("value");
        assertNotNull(stats);
        assertFalse(stats.isEmpty());

        Set<Integer> fragmentIds = new HashSet<>();
        for (ZoneStats z : stats) {
          fragmentIds.add(z.getFragmentId());
        }
        assertEquals(2, fragmentIds.size(), "Expected zones from 2 fragments, got: " + fragmentIds);
      }
    }
  }

  @Test
  public void testGetZonemapStatsCommittedSegments(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("segmented_zonemap_stats").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      Dataset ds2 = writeIntFragment(allocator, path, 1, 0, 50);
      ds2.close();
      Dataset ds3 = writeIntFragment(allocator, path, 2, 50, 50);
      ds3.close();

      try (Dataset dataset = Dataset.open(path, allocator)) {
        List<Fragment> fragments = dataset.getFragments();
        assertEquals(2, fragments.size());

        ScalarIndexParams params = ScalarIndexParams.create("zonemap", "{}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(params).build();

        Index firstSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList("value"), IndexType.ZONEMAP, indexParams)
                    .withIndexName("value_zm")
                    .withFragmentIds(Collections.singletonList(fragments.get(0).getId()))
                    .build());
        Index secondSegment =
            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList("value"), IndexType.ZONEMAP, indexParams)
                    .withIndexName("value_zm")
                    .withFragmentIds(Collections.singletonList(fragments.get(1).getId()))
                    .build());

        dataset.commitExistingIndexSegments(
            "value_zm", "value", List.of(firstSegment, secondSegment));

        List<ZoneStats> stats = dataset.getZonemapStats("value");
        assertNotNull(stats);
        assertFalse(stats.isEmpty());

        Set<Integer> fragmentIds = new HashSet<>();
        for (ZoneStats z : stats) {
          fragmentIds.add(z.getFragmentId());
        }
        assertEquals(2, fragmentIds.size(), "Expected zones from 2 committed segments");
      }
    }
  }

  @Test
  public void testGetZonemapStatsWrongColumnReturnsEmpty(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("wrong_col").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        // empty
      }
      Dataset ds2 = writeIntFragment(allocator, path, 1, 0, 100);
      ds2.close();

      try (Dataset dataset = Dataset.open(path, allocator)) {
        ScalarIndexParams params = ScalarIndexParams.create("zonemap", "{}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(params).build();
        dataset.createIndex(
            Collections.singletonList("value"),
            IndexType.ZONEMAP,
            Optional.of("value_zm"),
            indexParams,
            true);

        List<ZoneStats> stats = dataset.getZonemapStats("id");
        assertNotNull(stats);
        assertTrue(stats.isEmpty(), "Expected empty for non-indexed column");
      }
    }
  }

  @Test
  public void testGetZonemapStatsReportsNanCount(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("nan_col").toString();
    Schema schema =
        new Schema(
            Collections.singletonList(
                Field.nullable(
                    "score", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE))),
            null);
    int rowCount = 100;
    int nanCount = 3;
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, schema, new WriteParams.Builder().build())) {
        // empty
      }
      List<FragmentMetadata> metas;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        root.allocateNew();
        Float8Vector scores = (Float8Vector) root.getVector("score");
        for (int i = 0; i < rowCount; i++) {
          scores.setSafe(i, i < nanCount ? Double.NaN : (double) i);
        }
        root.setRowCount(rowCount);
        metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
      }
      Dataset.commit(allocator, path, new FragmentOperation.Append(metas), Optional.of(1L)).close();

      try (Dataset dataset = Dataset.open(path, allocator)) {
        ScalarIndexParams params = ScalarIndexParams.create("zonemap", "{}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(params).build();
        dataset.createIndex(
            Collections.singletonList("score"),
            IndexType.ZONEMAP,
            Optional.of("score_zm"),
            indexParams,
            true);

        // 100 rows fit in one zone. The bounds exclude NaN; the count is the
        // only trace of it.
        List<ZoneStats> stats = dataset.getZonemapStats("score");
        assertEquals(1, stats.size());
        ZoneStats zone = stats.get(0);
        assertEquals(nanCount, zone.getNanCount());
        assertEquals((double) nanCount, zone.getMin());
        assertEquals((double) (rowCount - 1), zone.getMax());
      }
    }
  }

  @Test
  public void testGetZonemapStatsNullArgument(@TempDir Path tempDir) {
    String path = tempDir.resolve("null_arg").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        assertThrows(IllegalArgumentException.class, () -> dataset.getZonemapStats(null));
      }
    }
  }

  @Test
  public void testGetZonemapStatsEmptyArgument(@TempDir Path tempDir) {
    String path = tempDir.resolve("empty_arg").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset dataset =
          Dataset.create(allocator, path, intSchema(), new WriteParams.Builder().build())) {
        assertThrows(IllegalArgumentException.class, () -> dataset.getZonemapStats(""));
      }
    }
  }
}
