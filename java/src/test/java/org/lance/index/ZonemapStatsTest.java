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
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampMicroVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.Text;
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

  /** Single-column schemas keyed by an arbitrary value type. */
  private static Schema schemaWithValue(ArrowType valueType) {
    return new Schema(
        Arrays.asList(
            Field.nullable("id", new ArrowType.Int(32, true)), Field.nullable("value", valueType)),
        null);
  }

  private Dataset writeStringFragment(
      BufferAllocator allocator, String path, long version, int rowCount) {
    Schema schema = schemaWithValue(new ArrowType.Utf8());
    List<FragmentMetadata> metas;
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector idVec = (IntVector) root.getVector("id");
      VarCharVector valVec = (VarCharVector) root.getVector("value");
      for (int i = 0; i < rowCount; i++) {
        idVec.setSafe(i, i);
        valVec.setSafe(i, new Text(String.format("v-%04d", i)));
      }
      root.setRowCount(rowCount);
      metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
    }
    return Dataset.commit(
        allocator, path, new FragmentOperation.Append(metas), Optional.of(version));
  }

  private Dataset writeFloatFragment(
      BufferAllocator allocator, String path, long version, int rowCount) {
    Schema schema = schemaWithValue(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE));
    List<FragmentMetadata> metas;
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector idVec = (IntVector) root.getVector("id");
      Float8Vector valVec = (Float8Vector) root.getVector("value");
      for (int i = 0; i < rowCount; i++) {
        idVec.setSafe(i, i);
        valVec.setSafe(i, i * 0.5);
      }
      root.setRowCount(rowCount);
      metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
    }
    return Dataset.commit(
        allocator, path, new FragmentOperation.Append(metas), Optional.of(version));
  }

  private Dataset writeBooleanFragment(
      BufferAllocator allocator, String path, long version, int rowCount) {
    Schema schema = schemaWithValue(new ArrowType.Bool());
    List<FragmentMetadata> metas;
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector idVec = (IntVector) root.getVector("id");
      BitVector valVec = (BitVector) root.getVector("value");
      for (int i = 0; i < rowCount; i++) {
        idVec.setSafe(i, i);
        valVec.setSafe(i, (i & 1) == 0 ? 1 : 0);
      }
      root.setRowCount(rowCount);
      metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
    }
    return Dataset.commit(
        allocator, path, new FragmentOperation.Append(metas), Optional.of(version));
  }

  private Dataset writeDateFragment(
      BufferAllocator allocator, String path, long version, int rowCount) {
    Schema schema = schemaWithValue(new ArrowType.Date(DateUnit.DAY));
    List<FragmentMetadata> metas;
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector idVec = (IntVector) root.getVector("id");
      DateDayVector valVec = (DateDayVector) root.getVector("value");
      for (int i = 0; i < rowCount; i++) {
        idVec.setSafe(i, i);
        valVec.setSafe(i, 19000 + i);
      }
      root.setRowCount(rowCount);
      metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
    }
    return Dataset.commit(
        allocator, path, new FragmentOperation.Append(metas), Optional.of(version));
  }

  private Dataset writeTimestampMicroNonTzFragment(
      BufferAllocator allocator, String path, long version, int rowCount) {
    Schema schema = schemaWithValue(new ArrowType.Timestamp(TimeUnit.MICROSECOND, null));
    List<FragmentMetadata> metas;
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector idVec = (IntVector) root.getVector("id");
      TimeStampMicroVector valVec = (TimeStampMicroVector) root.getVector("value");
      for (int i = 0; i < rowCount; i++) {
        idVec.setSafe(i, i);
        valVec.setSafe(i, 1_700_000_000_000_000L + i * 1_000_000L);
      }
      root.setRowCount(rowCount);
      metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
    }
    return Dataset.commit(
        allocator, path, new FragmentOperation.Append(metas), Optional.of(version));
  }

  private Dataset writeAllNullValueFragment(
      BufferAllocator allocator, String path, long version, int rowCount) {
    Schema schema = schemaWithValue(new ArrowType.Int(32, true));
    List<FragmentMetadata> metas;
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector idVec = (IntVector) root.getVector("id");
      IntVector valVec = (IntVector) root.getVector("value");
      for (int i = 0; i < rowCount; i++) {
        idVec.setSafe(i, i);
        valVec.setNull(i);
      }
      root.setRowCount(rowCount);
      metas = Fragment.create(path, allocator, root, new WriteParams.Builder().build());
    }
    return Dataset.commit(
        allocator, path, new FragmentOperation.Append(metas), Optional.of(version));
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

  // -------------------------------------------------------
  // Type-coverage integration tests (review finding #2)
  // -------------------------------------------------------

  private List<ZoneStats> zonemapStatsOnValue(BufferAllocator allocator, String path) {
    try (Dataset dataset = Dataset.open(path, allocator)) {
      ScalarIndexParams params = ScalarIndexParams.create("zonemap", "{}");
      IndexParams indexParams = IndexParams.builder().setScalarIndexParams(params).build();
      dataset.createIndex(
          Collections.singletonList("value"),
          IndexType.ZONEMAP,
          Optional.of("value_zm"),
          indexParams,
          true);
      return dataset.getZonemapStats("value");
    }
  }

  @Test
  public void testGetZonemapStatsStringColumn(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("string_col").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(
              allocator,
              path,
              schemaWithValue(new ArrowType.Utf8()),
              new WriteParams.Builder().build())) {}
      writeStringFragment(allocator, path, 1, 100).close();

      List<ZoneStats> stats = zonemapStatsOnValue(allocator, path);
      assertFalse(stats.isEmpty());
      assertTrue(stats.get(0).getMin() instanceof String);
      assertTrue(stats.get(0).getMax() instanceof String);
      assertEquals("v-0000", stats.get(0).getMin());
      assertEquals("v-0099", stats.get(stats.size() - 1).getMax());
    }
  }

  @Test
  public void testGetZonemapStatsFloatColumn(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("float_col").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(
              allocator,
              path,
              schemaWithValue(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
              new WriteParams.Builder().build())) {}
      writeFloatFragment(allocator, path, 1, 100).close();

      List<ZoneStats> stats = zonemapStatsOnValue(allocator, path);
      assertFalse(stats.isEmpty());
      assertTrue(stats.get(0).getMin() instanceof Double);
      assertEquals(0.0, stats.get(0).getMin());
      assertEquals(49.5, stats.get(stats.size() - 1).getMax());
    }
  }

  @Test
  public void testGetZonemapStatsBooleanColumn(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("bool_col").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(
              allocator,
              path,
              schemaWithValue(new ArrowType.Bool()),
              new WriteParams.Builder().build())) {}
      writeBooleanFragment(allocator, path, 1, 100).close();

      List<ZoneStats> stats = zonemapStatsOnValue(allocator, path);
      assertFalse(stats.isEmpty());
      assertTrue(stats.get(0).getMin() instanceof Boolean);
      assertEquals(Boolean.FALSE, stats.get(0).getMin());
      assertEquals(Boolean.TRUE, stats.get(0).getMax());
    }
  }

  @Test
  public void testGetZonemapStatsDateColumn(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("date_col").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(
              allocator,
              path,
              schemaWithValue(new ArrowType.Date(DateUnit.DAY)),
              new WriteParams.Builder().build())) {}
      writeDateFragment(allocator, path, 1, 100).close();

      List<ZoneStats> stats = zonemapStatsOnValue(allocator, path);
      assertFalse(stats.isEmpty());
      assertTrue(stats.get(0).getMin() instanceof Long);
      assertEquals(19000L, stats.get(0).getMin());
      assertEquals(19099L, stats.get(stats.size() - 1).getMax());
    }
  }

  @Test
  public void testGetZonemapStatsTimestampNonTzColumn(@TempDir Path tempDir) throws Exception {
    // Regression: pre-fix this crashed in toComparable with
    // ClassCastException(LocalDateTime cannot be cast to Number).
    String path = tempDir.resolve("ts_nontz_col").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(
              allocator,
              path,
              schemaWithValue(new ArrowType.Timestamp(TimeUnit.MICROSECOND, null)),
              new WriteParams.Builder().build())) {}
      writeTimestampMicroNonTzFragment(allocator, path, 1, 100).close();

      List<ZoneStats> stats = zonemapStatsOnValue(allocator, path);
      assertFalse(stats.isEmpty());
      assertTrue(stats.get(0).getMin() instanceof Long);
      assertEquals(1_700_000_000_000_000L, stats.get(0).getMin());
      assertEquals(1_700_000_000_000_000L + 99 * 1_000_000L, stats.get(stats.size() - 1).getMax());
    }
  }

  @Test
  public void testGetZonemapStatsAllNullZone(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("all_null_zone").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      try (Dataset ds =
          Dataset.create(
              allocator,
              path,
              schemaWithValue(new ArrowType.Int(32, true)),
              new WriteParams.Builder().build())) {}
      writeAllNullValueFragment(allocator, path, 1, 64).close();

      List<ZoneStats> stats = zonemapStatsOnValue(allocator, path);
      assertFalse(stats.isEmpty());
      ZoneStats first = stats.get(0);
      assertNull(first.getMin(), "all-null zone must report null min");
      assertNull(first.getMax(), "all-null zone must report null max");
      assertEquals(
          first.getZoneLength(),
          first.getNullCount(),
          "all-null zone null_count must equal zone_length");
    }
  }
}
