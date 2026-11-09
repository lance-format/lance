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
package org.lance.index.scalar;

import java.io.Serializable;

/**
 * Per-zone statistics from a zonemap index.
 *
 * <p>Each zone covers a contiguous range of row offsets within a single fragment. The min/max
 * values are represented as {@link Comparable} objects (Long, Double, String, etc.) matching the
 * column's data type.
 *
 * <p>A zone never spans fragment boundaries: if a fragment has more rows than the zone size,
 * multiple zones are created within that fragment.
 *
 * <p>This class is populated from the Rust {@code ZoneMapStatistics} via JNI. See {@code
 * lance-index/src/scalar/zonemap.rs} for the on-disk format.
 */
public class ZoneStats implements Serializable {
  private static final long serialVersionUID = 1L;

  private final int fragmentId;
  private final long zoneStart;
  private final long zoneLength;
  private final Comparable<?> min;
  private final Comparable<?> max;
  private final long nullCount;
  private final long nanCount;

  /**
   * Constructs a new ZoneStats instance.
   *
   * @param fragmentId the fragment this zone belongs to
   * @param zoneStart the starting row offset within the fragment
   * @param zoneLength the span of row offsets (last_offset - first_offset + 1); may differ from
   *     physical row count if rows have been deleted
   * @param min the minimum value in the zone, or null if the zone has no comparable value
   * @param max the maximum value in the zone, or null if the zone has no comparable value
   * @param nullCount the number of null values in the zone
   * @param nanCount the number of NaN values in the zone; see {@link #getNanCount()}
   */
  public ZoneStats(
      int fragmentId,
      long zoneStart,
      long zoneLength,
      Comparable<?> min,
      Comparable<?> max,
      long nullCount,
      long nanCount) {
    this.fragmentId = fragmentId;
    this.zoneStart = zoneStart;
    this.zoneLength = zoneLength;
    this.min = min;
    this.max = max;
    this.nullCount = nullCount;
    this.nanCount = nanCount;
  }

  /** Constructs a zone with no NaN values; see {@link #getNanCount()}. */
  public ZoneStats(
      int fragmentId,
      long zoneStart,
      long zoneLength,
      Comparable<?> min,
      Comparable<?> max,
      long nullCount) {
    this(fragmentId, zoneStart, zoneLength, min, max, nullCount, 0);
  }

  /** Returns the fragment ID this zone belongs to. */
  public int getFragmentId() {
    return fragmentId;
  }

  /** Returns the starting row offset within the fragment. */
  public long getZoneStart() {
    return zoneStart;
  }

  /**
   * Returns the span of row offsets covered by this zone.
   *
   * <p>This is (last_row_offset - first_row_offset + 1), not the count of physical rows. Deletions
   * may create gaps within the span.
   */
  public long getZoneLength() {
    return zoneLength;
  }

  /**
   * Returns the minimum value in the zone.
   *
   * @return the min value, or null if the zone has no comparable value: every value is null, or
   *     every non-null value is NaN; {@link #getNullCount()} and {@link #getNanCount()} tell the
   *     two apart
   */
  public Comparable<?> getMin() {
    return min;
  }

  /**
   * Returns the maximum value in the zone.
   *
   * @return the max value, or null if the zone has no comparable value: every value is null, or
   *     every non-null value is NaN; {@link #getNullCount()} and {@link #getNanCount()} tell the
   *     two apart
   */
  public Comparable<?> getMax() {
    return max;
  }

  /** Returns the number of null values in the zone. */
  public long getNullCount() {
    return nullCount;
  }

  /**
   * Returns the number of NaN values in the zone. Always 0 for non-floating-point columns.
   *
   * <p>Indexes written at zone map version 1 or later keep NaN out of {@link #getMin()} and {@link
   * #getMax()}, so this count is the only trace of a NaN in the zone. A predicate that orders NaN
   * relative to other values (Spark, for example, treats NaN as greater than any other value) must
   * keep a zone with a positive NaN count regardless of its bounds. Older indexes also mark such a
   * zone with a NaN max.
   */
  public long getNanCount() {
    return nanCount;
  }

  @Override
  public String toString() {
    return String.format(
        "ZoneStats{fragmentId=%d, zoneStart=%d, zoneLength=%d,"
            + " min=%s, max=%s, nullCount=%d, nanCount=%d}",
        fragmentId, zoneStart, zoneLength, min, max, nullCount, nanCount);
  }
}
