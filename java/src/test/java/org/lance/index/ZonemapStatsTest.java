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
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.math.BigDecimal;
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

  private static Schema uint64Schema() {
    return new Schema(
        Collections.singletonList(Field.nullable("u", new ArrowType.Int(64, false))), null);
  }

  /**
   * Write one fragment with a single UInt64 column. Values are raw 64-bit patterns, so a negative
   * Java long stands for an unsigned value at or above 2^63.
   */
  private static void writeUInt64ZonemapFixture(
      BufferAllocator allocator, String path, long[] values, int rowsPerZone) {
    Schema schema = uint64Schema();
    try (Dataset ds = Dataset.create(allocator, path, schema, new WriteParams.Builder().build())) {
      // empty
    }
    List<FragmentMetadata> metas;
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      UInt8Vector u = (UInt8Vector) root.getVector("u");
      for (int i = 0; i < values.length; i++) {
        u.setSafe(i, values[i]);
      }
      root.setRowCount(values.length);
      metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
    }
    Dataset.commit(allocator, path, new FragmentOperation.Append(metas), Optional.of(1L)).close();

    try (Dataset dataset = Dataset.open(path, allocator)) {
      ScalarIndexParams params =
          ScalarIndexParams.create("zonemap", "{\"rows_per_zone\": " + rowsPerZone + "}");
      IndexParams indexParams = IndexParams.builder().setScalarIndexParams(params).build();
      dataset.createIndex(
          Collections.singletonList("u"),
          IndexType.ZONEMAP,
          Optional.of("u_zm"),
          indexParams,
          true);
    }
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
  public void testGetZonemapStatsUnsupportedTypeReturnsEmpty(@TempDir Path tempDir)
      throws Exception {
    String path = tempDir.resolve("decimal_col").toString();
    Schema schema =
        new Schema(
            Collections.singletonList(Field.nullable("amount", new ArrowType.Decimal(10, 2, 128))),
            null);
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(allocator, path, schema, new WriteParams.Builder().build())) {
        // empty
      }
      List<FragmentMetadata> metas;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        root.allocateNew();
        DecimalVector amounts = (DecimalVector) root.getVector("amount");
        for (int i = 0; i < 100; i++) {
          amounts.setSafe(i, new BigDecimal(i).movePointLeft(2));
        }
        root.setRowCount(100);
        metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
      }
      Dataset.commit(allocator, path, new FragmentOperation.Append(metas), Optional.of(1L)).close();

      try (Dataset dataset = Dataset.open(path, allocator)) {
        ScalarIndexParams params = ScalarIndexParams.create("zonemap", "{}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(params).build();
        dataset.createIndex(
            Collections.singletonList("amount"),
            IndexType.ZONEMAP,
            Optional.of("amount_zm"),
            indexParams,
            true);

        // Decimal bounds have no Comparable representation: no stats, rather
        // than null bounds that would read as "all null".
        assertTrue(dataset.listIndexes().contains("amount_zm"));
        assertTrue(dataset.getZonemapStats("amount").isEmpty());
      }
    }
  }

  @Test
  public void testGetZonemapStatsUInt64LongMaxValue(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("uint64_long_max").toString();
    long[] values = new long[100];
    for (int i = 0; i < values.length; i++) {
      values[i] = i;
    }
    values[99] = Long.MAX_VALUE;

    try (BufferAllocator allocator = new RootAllocator()) {
      writeUInt64ZonemapFixture(allocator, path, values, 50);
      try (Dataset dataset = Dataset.open(path, allocator)) {
        assertTrue(dataset.listIndexes().contains("u_zm"));
        List<ZoneStats> stats = dataset.getZonemapStats("u");
        assertEquals(2, stats.size());

        assertEquals(Long.valueOf(0L), stats.get(0).getMin());
        assertEquals(Long.valueOf(49L), stats.get(0).getMax());
        assertEquals(Long.valueOf(50L), stats.get(1).getMin());
        assertEquals(
            Long.valueOf(Long.MAX_VALUE),
            stats.get(1).getMax(),
            "UInt64 max at Long.MAX_VALUE must be returned exactly");
      }
    }
  }

  @Test
  public void testGetZonemapStatsUInt64AboveLongMaxValue(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("uint64_above_long_max").toString();
    // 2^63 is the first unsigned value a Java long cannot hold; its raw 64-bit
    // pattern 0x8000000000000000 reads as Long.MIN_VALUE.
    long firstUnrepresentable = Long.MIN_VALUE;
    long[] values = new long[100];
    for (int i = 0; i < values.length; i++) {
      values[i] = i < 50 ? i : firstUnrepresentable + (i - 50);
    }

    try (BufferAllocator allocator = new RootAllocator()) {
      writeUInt64ZonemapFixture(allocator, path, values, 50);
      try (Dataset dataset = Dataset.open(path, allocator)) {
        assertTrue(dataset.listIndexes().contains("u_zm"));
        // The first zone is entirely representable, so a per-zone fallback
        // would return it instead of emptying the whole column.
        assertTrue(
            dataset.getZonemapStats("u").isEmpty(),
            "A UInt64 bound above Long.MAX_VALUE must empty the whole column");
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
