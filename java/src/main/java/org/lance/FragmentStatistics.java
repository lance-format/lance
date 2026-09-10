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

/**
 * Per-fragment statistics of a dataset version as parallel primitive arrays: {@code ids[i]}, {@code
 * rowCounts[i]} and {@code dataFileNums[i]} describe the same fragment. Returned by {@link
 * Dataset#getFragmentStatistics()} as a lightweight alternative to materializing full {@link
 * Fragment} objects.
 */
public final class FragmentStatistics {
  private final int[] ids;
  private final long[] rowCounts;
  private final int[] dataFileNums;

  FragmentStatistics(int[] ids, long[] rowCounts, int[] dataFileNums) {
    this.ids = ids;
    this.rowCounts = rowCounts;
    this.dataFileNums = dataFileNums;
  }

  /** Fragment IDs in manifest order. */
  public int[] getIds() {
    return ids;
  }

  /**
   * Row count per fragment, aligned with {@link #getIds()}. Matches {@link
   * FragmentMetadata#getNumRows()}: physical rows minus deleted rows.
   */
  public long[] getRowCounts() {
    return rowCounts;
  }

  /** Number of data files per fragment, aligned with {@link #getIds()}. */
  public int[] getDataFileNums() {
    return dataFileNums;
  }

  /** Number of fragments described. */
  public int size() {
    return ids.length;
  }
}
