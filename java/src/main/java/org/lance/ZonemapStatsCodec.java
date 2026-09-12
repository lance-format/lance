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
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DateMilliVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.TimeStampMicroTZVector;
import org.apache.arrow.vector.TimeStampMicroVector;
import org.apache.arrow.vector.TimeStampMilliTZVector;
import org.apache.arrow.vector.TimeStampMilliVector;
import org.apache.arrow.vector.TimeStampNanoTZVector;
import org.apache.arrow.vector.TimeStampNanoVector;
import org.apache.arrow.vector.TimeStampSecTZVector;
import org.apache.arrow.vector.TimeStampSecVector;
import org.apache.arrow.vector.UInt4Vector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.types.Types;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Decodes the Arrow IPC byte stream returned by {@code nativeGetZonemapStatsIpc} into a {@code
 * List<ZoneStats>}.
 *
 * <p>This codec exists to keep {@code List<ZoneStats>} as the public surface of {@link
 * org.lance.Dataset#getZonemapStats(String)} while letting the JNI hot path return a single Arrow
 * IPC byte array instead of constructing N {@code ZoneStats} JNI objects per call. The type →
 * {@code Comparable} mapping mirrors the Rust {@code scalar_value_to_java} helper.
 *
 * <p>The Rust producer ({@code zonemap_stats_as_batch} in {@code
 * rust/lance-index/src/scalar/zonemap.rs}) is the only writer of this stream; its schema is fixed
 * and validated here field-by-field before any row is read. Schema drift surfaces as an {@link
 * IllegalStateException} naming the field plus the expected and actual value, instead of silently
 * coercing the wrong type through {@code Number.longValue()}.
 */
final class ZonemapStatsCodec {

  private ZonemapStatsCodec() {}

  /** Required column types/nullability per the Rust producer. {@code null} type means "any". */
  private static final ColumnContract[] CONTRACT =
      new ColumnContract[] {
        new ColumnContract("min", null, true),
        new ColumnContract("max", null, true),
        new ColumnContract("null_count", new ArrowType.Int(32, false), false),
        new ColumnContract("nan_count", new ArrowType.Int(32, false), false),
        new ColumnContract("fragment_id", new ArrowType.Int(64, false), false),
        new ColumnContract("zone_start", new ArrowType.Int(64, false), false),
        new ColumnContract("zone_length", new ArrowType.Int(64, false), false),
      };

  /**
   * Decode the IPC payload produced by the JNI call.
   *
   * @param ipcBytes Arrow IPC stream, possibly empty (length 0) when the column has no zonemap
   * @param allocator the {@link BufferAllocator} owning intermediate buffers; the caller retains
   *     ownership and is responsible for closing it
   * @return zone stats in the order the Rust side produced them
   */
  static List<ZoneStats> decode(byte[] ipcBytes, BufferAllocator allocator) throws IOException {
    if (ipcBytes == null || ipcBytes.length == 0) {
      return new ArrayList<>();
    }

    ArrayList<ZoneStats> out = new ArrayList<>();
    try (ByteArrayInputStream in = new ByteArrayInputStream(ipcBytes);
        ArrowStreamReader reader = new ArrowStreamReader(in, allocator)) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      validateSchema(root.getSchema());

      // Schema validation has cleared these casts: types and nullability are correct.
      UInt4Vector nullCountVec = (UInt4Vector) root.getVector("null_count");
      UInt8Vector fragmentIdVec = (UInt8Vector) root.getVector("fragment_id");
      UInt8Vector zoneStartVec = (UInt8Vector) root.getVector("zone_start");
      UInt8Vector zoneLengthVec = (UInt8Vector) root.getVector("zone_length");
      FieldVector minVec = root.getVector("min");
      FieldVector maxVec = root.getVector("max");

      while (reader.loadNextBatch()) {
        int n = root.getRowCount();
        out.ensureCapacity(out.size() + n);
        for (int i = 0; i < n; i++) {
          int fragmentId = toFragmentId(fragmentIdVec.get(i), i);
          long zoneStart = zoneStartVec.get(i);
          long zoneLength = zoneLengthVec.get(i);
          long nullCount = Integer.toUnsignedLong(nullCountVec.get(i));
          Comparable<?> min = toComparable(minVec, i);
          Comparable<?> max = toComparable(maxVec, i);
          out.add(new ZoneStats(fragmentId, zoneStart, zoneLength, min, max, nullCount));
        }
      }
    }
    return out;
  }

  /**
   * Reject the stream up-front if any required column is missing, the wrong type, or the wrong
   * nullability. Saves us from per-row defensive casts and gives the caller a single, descriptive
   * error instead of a {@link ClassCastException} or {@link NullPointerException} mid-batch.
   */
  private static void validateSchema(Schema schema) {
    for (ColumnContract c : CONTRACT) {
      Field f = null;
      for (Field cand : schema.getFields()) {
        if (cand.getName().equals(c.name)) {
          f = cand;
          break;
        }
      }
      if (f == null) {
        throw new IllegalStateException(
            "ZonemapStatsCodec: required field '" + c.name + "' is missing from IPC schema");
      }
      if (c.type != null && !c.type.equals(f.getType())) {
        throw new IllegalStateException(
            "ZonemapStatsCodec: field '"
                + c.name
                + "' has wrong type: expected="
                + c.type
                + " actual="
                + f.getType());
      }
      if (f.isNullable() != c.nullable) {
        throw new IllegalStateException(
            "ZonemapStatsCodec: field '"
                + c.name
                + "' has wrong nullability: expected="
                + c.nullable
                + " actual="
                + f.isNullable());
      }
    }
  }

  /**
   * Narrow a Rust u64 fragment id to the int that {@link ZoneStats#getFragmentId} exposes. Values
   * outside {@code [0, Integer.MAX_VALUE]} are rejected: a raw long {@code < 0} means the u64 was
   * above {@code Long.MAX_VALUE}, which by definition cannot fit in a signed int.
   */
  private static int toFragmentId(long raw, int row) {
    if (raw < 0 || raw > Integer.MAX_VALUE) {
      throw new IllegalStateException(
          "ZonemapStatsCodec: fragment_id "
              + Long.toUnsignedString(raw)
              + " at row "
              + row
              + " exceeds Integer.MAX_VALUE");
    }
    return (int) raw;
  }

  /**
   * Convert one row of an arbitrary Arrow vector to the {@code Comparable} the public {@code
   * ZoneStats} API exposes. Mirrors {@code scalar_value_to_java} on the Rust side: int family →
   * {@link Long}, float family → {@link Double}, utf8 → {@link String}, bool → {@link Boolean},
   * date / timestamp → {@link Long} (epoch units preserved as in Rust).
   *
   * <p>Date and timestamp arms must use the typed-vector {@code .get(i)} API rather than {@code
   * getObject(i)}: in Arrow Java, {@code DateDayVector#getObject} returns {@link
   * java.time.LocalDate}, {@code DateMilliVector#getObject} and the non-TZ {@code TimeStamp*Vector}
   * variants return {@link java.time.LocalDateTime}, neither of which is a {@link Number} — casting
   * through {@code Number} crashes with {@code ClassCastException}.
   */
  private static Comparable<?> toComparable(FieldVector v, int i) {
    if (v.isNull(i)) {
      return null;
    }
    Types.MinorType minor = v.getMinorType();
    switch (minor) {
      case TINYINT:
      case SMALLINT:
      case INT:
      case BIGINT:
      case UINT1:
      case UINT2:
      case UINT4:
      case UINT8:
        return ((Number) v.getObject(i)).longValue();

      case DATEDAY:
        return (long) ((DateDayVector) v).get(i);
      case DATEMILLI:
        return ((DateMilliVector) v).get(i);
      case TIMESTAMPSEC:
        return ((TimeStampSecVector) v).get(i);
      case TIMESTAMPMILLI:
        return ((TimeStampMilliVector) v).get(i);
      case TIMESTAMPMICRO:
        return ((TimeStampMicroVector) v).get(i);
      case TIMESTAMPNANO:
        return ((TimeStampNanoVector) v).get(i);
      case TIMESTAMPSECTZ:
        return ((TimeStampSecTZVector) v).get(i);
      case TIMESTAMPMILLITZ:
        return ((TimeStampMilliTZVector) v).get(i);
      case TIMESTAMPMICROTZ:
        return ((TimeStampMicroTZVector) v).get(i);
      case TIMESTAMPNANOTZ:
        return ((TimeStampNanoTZVector) v).get(i);

      case FLOAT2:
      case FLOAT4:
      case FLOAT8:
        return ((Number) v.getObject(i)).doubleValue();

      case BIT:
        return ((BitVector) v).get(i) != 0;

      case VARCHAR:
      case LARGEVARCHAR:
        {
          Object raw = v.getObject(i);
          if (raw instanceof byte[]) {
            return new String((byte[]) raw, StandardCharsets.UTF_8);
          }
          return raw.toString();
        }

      default:
        // Conservative parity with Rust's `_ => Ok(JObject::null())` arm.
        return null;
    }
  }

  private static final class ColumnContract {
    final String name;
    final ArrowType type;
    final boolean nullable;

    ColumnContract(String name, ArrowType type, boolean nullable) {
      this.name = name;
      this.type = type;
      this.nullable = nullable;
    }
  }
}
