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
package org.lance;

import org.lance.index.scalar.ZoneStats;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DateMilliVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampMicroVector;
import org.apache.arrow.vector.UInt4Vector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Codec-level unit tests for {@link ZonemapStatsCodec}. Synthesizes Arrow IPC byte streams directly
 * — no Lance dataset, no JNI — so we can exercise edge cases the Rust producer is contractually
 * forbidden from emitting (null in a non-nullable id column, intentionally chosen item types).
 */
public class ZonemapStatsCodecTest {

  @FunctionalInterface
  private interface BatchPopulator {
    void populate(VectorSchemaRoot root);
  }

  /**
   * Encode 1+ batches as one IPC stream. Each populator must call {@code root.setRowCount(...)}.
   */
  private static byte[] encodeIpc(
      Schema schema, BufferAllocator allocator, BatchPopulator... batches) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
        ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
      writer.start();
      for (BatchPopulator pop : batches) {
        root.allocateNew();
        pop.populate(root);
        writer.writeBatch();
        for (int j = 0; j < root.getFieldVectors().size(); j++) {
          root.getFieldVectors().get(j).clear();
        }
      }
      writer.end();
    }
    return out.toByteArray();
  }

  /** Canonical zonemap schema (mirrors Rust {@code zonemap_stats_as_batch}). */
  private static Schema zonemapSchema(ArrowType itemsType) {
    return new Schema(
        Arrays.asList(
            Field.nullable("min", itemsType),
            Field.nullable("max", itemsType),
            new Field("null_count", FieldType.notNullable(new ArrowType.Int(32, false)), null),
            new Field("nan_count", FieldType.notNullable(new ArrowType.Int(32, false)), null),
            new Field("fragment_id", FieldType.notNullable(new ArrowType.Int(64, false)), null),
            new Field("zone_start", FieldType.notNullable(new ArrowType.Int(64, false)), null),
            new Field("zone_length", FieldType.notNullable(new ArrowType.Int(64, false)), null)),
        null);
  }

  @Test
  public void decode_emptyBytes_returnsMutableEmptyList() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      List<ZoneStats> emptyBytes = ZonemapStatsCodec.decode(new byte[0], allocator);
      assertTrue(emptyBytes.isEmpty());
      emptyBytes.add(new ZoneStats(0, 0, 0, null, null, 0));
      assertEquals(1, emptyBytes.size());

      List<ZoneStats> nullBytes = ZonemapStatsCodec.decode(null, allocator);
      assertTrue(nullBytes.isEmpty());
      nullBytes.add(new ZoneStats(0, 0, 0, null, null, 0));
      assertEquals(1, nullBytes.size());
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"fragment_id", "zone_start", "zone_length", "null_count", "nan_count"})
  public void decode_schemaDeclaresNullableForFixedColumn_throwsWithFieldAndExpected(
      String nullableField) throws Exception {
    // Rust producer declares these five columns notNullable. If upstream ships them as nullable
    // (e.g. wrong schema), the codec must reject the stream at schema-validation time — before
    // ever touching row data — with a message that names the field and what was expected.
    List<Field> fields =
        new ArrayList<>(
            Arrays.asList(
                Field.nullable("min", new ArrowType.Int(32, true)),
                Field.nullable("max", new ArrowType.Int(32, true)),
                new Field("null_count", FieldType.notNullable(new ArrowType.Int(32, false)), null),
                new Field("nan_count", FieldType.notNullable(new ArrowType.Int(32, false)), null),
                new Field("fragment_id", FieldType.notNullable(new ArrowType.Int(64, false)), null),
                new Field("zone_start", FieldType.notNullable(new ArrowType.Int(64, false)), null),
                new Field(
                    "zone_length", FieldType.notNullable(new ArrowType.Int(64, false)), null)));
    for (int i = 0; i < fields.size(); i++) {
      if (fields.get(i).getName().equals(nullableField)) {
        Field f = fields.get(i);
        fields.set(i, Field.nullable(f.getName(), f.getType()));
      }
    }
    Schema schema = new Schema(fields, null);

    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc = encodeIpc(schema, allocator, root -> root.setRowCount(0));
      IllegalStateException ex =
          assertThrows(IllegalStateException.class, () -> ZonemapStatsCodec.decode(ipc, allocator));
      assertTrue(
          ex.getMessage().contains("'" + nullableField + "'"),
          "error must name the offending field, got: " + ex.getMessage());
      assertTrue(
          ex.getMessage().contains("nullability"),
          "error must mention nullability, got: " + ex.getMessage());
      assertTrue(
          ex.getMessage().contains("expected=false") && ex.getMessage().contains("actual=true"),
          "error must include expected/actual, got: " + ex.getMessage());
    }
  }

  @ParameterizedTest
  @ValueSource(
      strings = {
        "min",
        "max",
        "null_count",
        "nan_count",
        "fragment_id",
        "zone_start",
        "zone_length"
      })
  public void decode_missingRequiredField_throwsWithFieldName(String missingField)
      throws Exception {
    List<Field> fields =
        new ArrayList<>(
            Arrays.asList(
                Field.nullable("min", new ArrowType.Int(32, true)),
                Field.nullable("max", new ArrowType.Int(32, true)),
                new Field("null_count", FieldType.notNullable(new ArrowType.Int(32, false)), null),
                new Field("nan_count", FieldType.notNullable(new ArrowType.Int(32, false)), null),
                new Field("fragment_id", FieldType.notNullable(new ArrowType.Int(64, false)), null),
                new Field("zone_start", FieldType.notNullable(new ArrowType.Int(64, false)), null),
                new Field(
                    "zone_length", FieldType.notNullable(new ArrowType.Int(64, false)), null)));
    fields.removeIf(f -> f.getName().equals(missingField));
    Schema schema = new Schema(fields, null);

    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc = encodeIpc(schema, allocator, root -> root.setRowCount(0));
      IllegalStateException ex =
          assertThrows(IllegalStateException.class, () -> ZonemapStatsCodec.decode(ipc, allocator));
      assertTrue(
          ex.getMessage().contains("'" + missingField + "'"),
          "error must name the missing field, got: " + ex.getMessage());
      assertTrue(
          ex.getMessage().contains("missing"), "error must say 'missing', got: " + ex.getMessage());
    }
  }

  @ParameterizedTest
  @CsvSource({
    // field name,        wrong bitWidth, wrong signed
    "null_count,  64, false",
    "nan_count,   64, false",
    "fragment_id, 32, false",
    "zone_start,  32, false",
    "zone_length, 32, false",
    "null_count,  32, true",
    "fragment_id, 64, true",
  })
  public void decode_wrongFixedColumnType_throwsWithExpectedAndActual(
      String fieldName, int bitWidth, boolean signed) throws Exception {
    List<Field> fields =
        new ArrayList<>(
            Arrays.asList(
                Field.nullable("min", new ArrowType.Int(32, true)),
                Field.nullable("max", new ArrowType.Int(32, true)),
                new Field("null_count", FieldType.notNullable(new ArrowType.Int(32, false)), null),
                new Field("nan_count", FieldType.notNullable(new ArrowType.Int(32, false)), null),
                new Field("fragment_id", FieldType.notNullable(new ArrowType.Int(64, false)), null),
                new Field("zone_start", FieldType.notNullable(new ArrowType.Int(64, false)), null),
                new Field(
                    "zone_length", FieldType.notNullable(new ArrowType.Int(64, false)), null)));
    for (int i = 0; i < fields.size(); i++) {
      if (fields.get(i).getName().equals(fieldName)) {
        fields.set(
            i,
            new Field(fieldName, FieldType.notNullable(new ArrowType.Int(bitWidth, signed)), null));
      }
    }
    Schema schema = new Schema(fields, null);

    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc = encodeIpc(schema, allocator, root -> root.setRowCount(0));
      IllegalStateException ex =
          assertThrows(IllegalStateException.class, () -> ZonemapStatsCodec.decode(ipc, allocator));
      assertTrue(
          ex.getMessage().contains("'" + fieldName + "'"),
          "error must name the field, got: " + ex.getMessage());
      assertTrue(
          ex.getMessage().contains("type")
              && ex.getMessage().contains("expected=")
              && ex.getMessage().contains("actual="),
          "error must include expected/actual type, got: " + ex.getMessage());
    }
  }

  @Test
  public void decode_fragmentIdExceedsIntMax_throws() throws Exception {
    Schema schema = zonemapSchema(new ArrowType.Int(32, true));
    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc =
          encodeIpc(
              schema,
              allocator,
              root -> {
                ((IntVector) root.getVector("min")).setSafe(0, 1);
                ((IntVector) root.getVector("max")).setSafe(0, 2);
                ((UInt4Vector) root.getVector("null_count")).setSafe(0, 0);
                ((UInt4Vector) root.getVector("nan_count")).setSafe(0, 0);
                // Integer.MAX_VALUE + 1L written as an unsigned u64 — still under Long.MAX_VALUE
                // so the raw long is positive, but it overflows Java int.
                ((UInt8Vector) root.getVector("fragment_id"))
                    .setSafe(0, (long) Integer.MAX_VALUE + 1L);
                ((UInt8Vector) root.getVector("zone_start")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_length")).setSafe(0, 1024L);
                root.setRowCount(1);
              });
      IllegalStateException ex =
          assertThrows(IllegalStateException.class, () -> ZonemapStatsCodec.decode(ipc, allocator));
      assertTrue(
          ex.getMessage().contains("fragment_id"),
          "error must name fragment_id, got: " + ex.getMessage());
      assertTrue(
          ex.getMessage().contains("at row 0"), "error must include row, got: " + ex.getMessage());
      assertTrue(
          ex.getMessage().contains(Long.toUnsignedString((long) Integer.MAX_VALUE + 1L)),
          "error must include the offending value, got: " + ex.getMessage());
    }
  }

  @Test
  public void decode_fragmentIdHighU64_throws() throws Exception {
    // u64 value > Long.MAX_VALUE arrives as a negative signed long; codec must reject it.
    Schema schema = zonemapSchema(new ArrowType.Int(32, true));
    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc =
          encodeIpc(
              schema,
              allocator,
              root -> {
                ((IntVector) root.getVector("min")).setSafe(0, 1);
                ((IntVector) root.getVector("max")).setSafe(0, 2);
                ((UInt4Vector) root.getVector("null_count")).setSafe(0, 0);
                ((UInt4Vector) root.getVector("nan_count")).setSafe(0, 0);
                ((UInt8Vector) root.getVector("fragment_id")).setSafe(0, Long.MIN_VALUE);
                ((UInt8Vector) root.getVector("zone_start")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_length")).setSafe(0, 1024L);
                root.setRowCount(1);
              });
      IllegalStateException ex =
          assertThrows(IllegalStateException.class, () -> ZonemapStatsCodec.decode(ipc, allocator));
      assertTrue(
          ex.getMessage().contains("fragment_id"),
          "error must name fragment_id, got: " + ex.getMessage());
      assertTrue(
          ex.getMessage().contains("at row 0"), "error must include row, got: " + ex.getMessage());
    }
  }

  @Test
  public void decode_nullCountUInt32Boundary_preservesUnsignedValue() throws Exception {
    Schema schema = zonemapSchema(new ArrowType.Int(32, true));
    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc =
          encodeIpc(
              schema,
              allocator,
              root -> {
                ((IntVector) root.getVector("min")).setSafe(0, 1);
                ((IntVector) root.getVector("max")).setSafe(0, 2);
                // UInt32 max — must come out as 4_294_967_295L, not -1L or 0.
                ((UInt4Vector) root.getVector("null_count")).setSafe(0, 0xFFFFFFFF);
                ((UInt4Vector) root.getVector("nan_count")).setSafe(0, 0);
                ((UInt8Vector) root.getVector("fragment_id")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_start")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_length")).setSafe(0, 1024L);
                root.setRowCount(1);
              });
      List<ZoneStats> got = ZonemapStatsCodec.decode(ipc, allocator);
      assertEquals(1, got.size());
      assertEquals(4_294_967_295L, got.get(0).getNullCount());
    }
  }

  @Test
  public void decode_dateDay_returnsLongDays() throws Exception {
    Schema schema = zonemapSchema(new ArrowType.Date(DateUnit.DAY));
    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc =
          encodeIpc(
              schema,
              allocator,
              root -> {
                ((DateDayVector) root.getVector("min")).setSafe(0, 19737);
                ((DateDayVector) root.getVector("max")).setSafe(0, 19800);
                ((UInt4Vector) root.getVector("null_count")).setSafe(0, 0);
                ((UInt4Vector) root.getVector("nan_count")).setSafe(0, 0);
                ((UInt8Vector) root.getVector("fragment_id")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_start")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_length")).setSafe(0, 1024L);
                root.setRowCount(1);
              });
      List<ZoneStats> got = ZonemapStatsCodec.decode(ipc, allocator);
      assertEquals(1, got.size());
      assertEquals(19737L, got.get(0).getMin());
      assertEquals(19800L, got.get(0).getMax());
    }
  }

  @Test
  public void decode_dateMilli_returnsLongMillis() throws Exception {
    Schema schema = zonemapSchema(new ArrowType.Date(DateUnit.MILLISECOND));
    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc =
          encodeIpc(
              schema,
              allocator,
              root -> {
                ((DateMilliVector) root.getVector("min")).setSafe(0, 1_700_000_000_000L);
                ((DateMilliVector) root.getVector("max")).setSafe(0, 1_700_000_999_000L);
                ((UInt4Vector) root.getVector("null_count")).setSafe(0, 0);
                ((UInt4Vector) root.getVector("nan_count")).setSafe(0, 0);
                ((UInt8Vector) root.getVector("fragment_id")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_start")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_length")).setSafe(0, 1024L);
                root.setRowCount(1);
              });
      List<ZoneStats> got = ZonemapStatsCodec.decode(ipc, allocator);
      assertEquals(1, got.size());
      assertEquals(1_700_000_000_000L, got.get(0).getMin());
      assertEquals(1_700_000_999_000L, got.get(0).getMax());
    }
  }

  @Test
  public void decode_timestampMicro_returnsLongMicros() throws Exception {
    // The bug arm: Timestamp(MICROSECOND, null) — non-TZ. Arrow Java's getObject returns
    // LocalDateTime here; the existing codec's `(Number) raw` cast crashes on it.
    Schema schema = zonemapSchema(new ArrowType.Timestamp(TimeUnit.MICROSECOND, null));
    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc =
          encodeIpc(
              schema,
              allocator,
              root -> {
                ((TimeStampMicroVector) root.getVector("min")).setSafe(0, 1_700_000_000_000_000L);
                ((TimeStampMicroVector) root.getVector("max")).setSafe(0, 1_700_000_999_999_999L);
                ((UInt4Vector) root.getVector("null_count")).setSafe(0, 0);
                ((UInt4Vector) root.getVector("nan_count")).setSafe(0, 0);
                ((UInt8Vector) root.getVector("fragment_id")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_start")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_length")).setSafe(0, 1024L);
                root.setRowCount(1);
              });
      List<ZoneStats> got = ZonemapStatsCodec.decode(ipc, allocator);
      assertEquals(1, got.size());
      assertEquals(1_700_000_000_000_000L, got.get(0).getMin());
      assertEquals(1_700_000_999_999_999L, got.get(0).getMax());
    }
  }

  @Test
  public void decode_multipleBatches_preservesOrder() throws Exception {
    Schema schema = zonemapSchema(new ArrowType.Int(32, true));
    try (BufferAllocator allocator = new RootAllocator()) {
      byte[] ipc =
          encodeIpc(
              schema,
              allocator,
              // Batch A: fragment 0, two zones.
              root -> {
                IntVector min = (IntVector) root.getVector("min");
                IntVector max = (IntVector) root.getVector("max");
                UInt4Vector nullCount = (UInt4Vector) root.getVector("null_count");
                UInt4Vector nanCount = (UInt4Vector) root.getVector("nan_count");
                UInt8Vector fragId = (UInt8Vector) root.getVector("fragment_id");
                UInt8Vector start = (UInt8Vector) root.getVector("zone_start");
                UInt8Vector length = (UInt8Vector) root.getVector("zone_length");
                min.setSafe(0, 0);
                max.setSafe(0, 99);
                min.setSafe(1, 100);
                max.setSafe(1, 199);
                nullCount.setSafe(0, 0);
                nullCount.setSafe(1, 0);
                nanCount.setSafe(0, 0);
                nanCount.setSafe(1, 0);
                fragId.setSafe(0, 0L);
                fragId.setSafe(1, 0L);
                start.setSafe(0, 0L);
                start.setSafe(1, 100L);
                length.setSafe(0, 100L);
                length.setSafe(1, 100L);
                root.setRowCount(2);
              },
              // Batch B: fragment 1, one zone.
              root -> {
                ((IntVector) root.getVector("min")).setSafe(0, 200);
                ((IntVector) root.getVector("max")).setSafe(0, 299);
                ((UInt4Vector) root.getVector("null_count")).setSafe(0, 0);
                ((UInt4Vector) root.getVector("nan_count")).setSafe(0, 0);
                ((UInt8Vector) root.getVector("fragment_id")).setSafe(0, 1L);
                ((UInt8Vector) root.getVector("zone_start")).setSafe(0, 0L);
                ((UInt8Vector) root.getVector("zone_length")).setSafe(0, 100L);
                root.setRowCount(1);
              });

      List<ZoneStats> got = ZonemapStatsCodec.decode(ipc, allocator);
      assertEquals(3, got.size(), "two zones from batch A + one from batch B");
      assertEquals(0, got.get(0).getFragmentId());
      assertEquals(0L, got.get(0).getZoneStart());
      assertEquals(0, got.get(1).getFragmentId());
      assertEquals(100L, got.get(1).getZoneStart());
      assertEquals(1, got.get(2).getFragmentId());
      assertEquals(0L, got.get(2).getZoneStart());
      assertEquals(0L, got.get(0).getMin());
      assertEquals(99L, got.get(0).getMax());
      assertEquals(100L, got.get(1).getMin());
      assertEquals(299L, got.get(2).getMax());
    }
  }
}
