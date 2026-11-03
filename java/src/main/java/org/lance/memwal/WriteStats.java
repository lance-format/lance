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
package org.lance.memwal;

import com.google.common.base.MoreObjects;

/** A snapshot of cumulative write statistics for a {@link ShardWriter}. */
public class WriteStats {
  private final long putCount;
  private final long putTimeMs;
  private final long walFlushCount;
  private final long walFlushBytes;
  private final long walFlushTimeMs;
  private final long memtableFlushCount;
  private final long memtableFlushRows;
  private final long memtableFlushTimeMs;
  private final long memtableFlushFailures;

  public WriteStats(
      long putCount,
      long putTimeMs,
      long walFlushCount,
      long walFlushBytes,
      long walFlushTimeMs,
      long memtableFlushCount,
      long memtableFlushRows,
      long memtableFlushTimeMs,
      long memtableFlushFailures) {
    this.putCount = putCount;
    this.putTimeMs = putTimeMs;
    this.walFlushCount = walFlushCount;
    this.walFlushBytes = walFlushBytes;
    this.walFlushTimeMs = walFlushTimeMs;
    this.memtableFlushCount = memtableFlushCount;
    this.memtableFlushRows = memtableFlushRows;
    this.memtableFlushTimeMs = memtableFlushTimeMs;
    this.memtableFlushFailures = memtableFlushFailures;
  }

  /** Number of {@link ShardWriter#put} calls. */
  public long putCount() {
    return putCount;
  }

  /** Total time spent in {@code put}, in milliseconds. */
  public long putTimeMs() {
    return putTimeMs;
  }

  /** Number of WAL flushes. */
  public long walFlushCount() {
    return walFlushCount;
  }

  /** Total bytes flushed to the WAL. */
  public long walFlushBytes() {
    return walFlushBytes;
  }

  /** Total time spent flushing the WAL, in milliseconds. */
  public long walFlushTimeMs() {
    return walFlushTimeMs;
  }

  /** Number of MemTable flushes. */
  public long memtableFlushCount() {
    return memtableFlushCount;
  }

  /** Total rows flushed from MemTables. */
  public long memtableFlushRows() {
    return memtableFlushRows;
  }

  /** Total time spent flushing MemTables, in milliseconds. */
  public long memtableFlushTimeMs() {
    return memtableFlushTimeMs;
  }

  /** Number of failed MemTable flush attempts, each retry counted separately. */
  public long memtableFlushFailures() {
    return memtableFlushFailures;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("putCount", putCount)
        .add("putTimeMs", putTimeMs)
        .add("walFlushCount", walFlushCount)
        .add("walFlushBytes", walFlushBytes)
        .add("walFlushTimeMs", walFlushTimeMs)
        .add("memtableFlushCount", memtableFlushCount)
        .add("memtableFlushRows", memtableFlushRows)
        .add("memtableFlushTimeMs", memtableFlushTimeMs)
        .add("memtableFlushFailures", memtableFlushFailures)
        .toString();
  }
}
