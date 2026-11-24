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
package org.lance.ipc;

import com.google.common.base.MoreObjects;
import org.apache.arrow.util.Preconditions;

import java.io.Serializable;
import java.util.Objects;

/** A contiguous range of physical row offsets within a fragment. */
public final class FragmentSlice implements Serializable {
  private static final long serialVersionUID = 1L;

  private final int fragmentId;
  private final long rowOffset;
  private final long rowCount;

  /**
   * Creates a fragment slice covering {@code [rowOffset, rowOffset + rowCount)}.
   *
   * <p>Row offsets are physical positions in the fragment. Deletions do not compact these offsets,
   * and deleted rows in the range are omitted by a normal scan. The slice must be used with the
   * same dataset snapshot from which it was planned.
   *
   * @param fragmentId fragment ID in the dataset snapshot
   * @param rowOffset starting physical row offset in the fragment
   * @param rowCount number of physical row offsets covered; zero represents an empty slice
   */
  public FragmentSlice(int fragmentId, long rowOffset, long rowCount) {
    Preconditions.checkArgument(
        fragmentId >= 0, "fragmentId must be non-negative, got %s", fragmentId);
    Preconditions.checkArgument(
        rowOffset >= 0, "rowOffset must be non-negative, got %s", rowOffset);
    Preconditions.checkArgument(rowCount >= 0, "rowCount must be non-negative, got %s", rowCount);
    Preconditions.checkArgument(
        rowOffset <= Long.MAX_VALUE - rowCount,
        "rowOffset + rowCount overflows long: rowOffset=%s, rowCount=%s",
        rowOffset,
        rowCount);
    this.fragmentId = fragmentId;
    this.rowOffset = rowOffset;
    this.rowCount = rowCount;
  }

  public int getFragmentId() {
    return fragmentId;
  }

  public long getRowOffset() {
    return rowOffset;
  }

  public long getRowCount() {
    return rowCount;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof FragmentSlice)) {
      return false;
    }
    FragmentSlice that = (FragmentSlice) other;
    return fragmentId == that.fragmentId
        && rowOffset == that.rowOffset
        && rowCount == that.rowCount;
  }

  @Override
  public int hashCode() {
    return Objects.hash(fragmentId, rowOffset, rowCount);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fragmentId", fragmentId)
        .add("rowOffset", rowOffset)
        .add("rowCount", rowCount)
        .toString();
  }
}
