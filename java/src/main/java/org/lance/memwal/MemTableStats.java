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

import java.util.Optional;

/** Statistics of the active MemTable held by a {@link ShardWriter}. */
public class MemTableStats {
  private final long rowCount;
  private final long batchCount;
  private final long estimatedSizeBytes;
  private final long generation;
  private final Optional<Long> maxBufferedBatchPosition;
  private final long durableBatchCount;
  private final long globalOffset;
  private final Optional<Long> pendingWalStartBatchPosition;
  private final Optional<Long> pendingWalEndBatchPosition;
  private final long pendingWalBatchCount;
  private final long pendingWalRowCount;
  private final long pendingWalEstimatedBytes;
  private final long indexBytes;
  private final long graceBytes;
  private final long retainedBytes;

  public MemTableStats(
      long rowCount,
      long batchCount,
      long estimatedSizeBytes,
      long generation,
      Long maxBufferedBatchPosition,
      long durableBatchCount,
      long globalOffset,
      Long pendingWalStartBatchPosition,
      Long pendingWalEndBatchPosition,
      long pendingWalBatchCount,
      long pendingWalRowCount,
      long pendingWalEstimatedBytes,
      long indexBytes,
      long graceBytes,
      long retainedBytes) {
    this.rowCount = rowCount;
    this.batchCount = batchCount;
    this.estimatedSizeBytes = estimatedSizeBytes;
    this.generation = generation;
    this.maxBufferedBatchPosition = Optional.ofNullable(maxBufferedBatchPosition);
    this.durableBatchCount = durableBatchCount;
    this.globalOffset = globalOffset;
    this.pendingWalStartBatchPosition = Optional.ofNullable(pendingWalStartBatchPosition);
    this.pendingWalEndBatchPosition = Optional.ofNullable(pendingWalEndBatchPosition);
    this.pendingWalBatchCount = pendingWalBatchCount;
    this.pendingWalRowCount = pendingWalRowCount;
    this.pendingWalEstimatedBytes = pendingWalEstimatedBytes;
    this.indexBytes = indexBytes;
    this.graceBytes = graceBytes;
    this.retainedBytes = retainedBytes;
  }

  /** Number of rows currently buffered in the active MemTable. */
  public long rowCount() {
    return rowCount;
  }

  /** Number of record batches currently buffered in the active MemTable. */
  public long batchCount() {
    return batchCount;
  }

  /**
   * Row-data bytes of the active MemTable: the unit the flush trigger measures. Its in-memory
   * indexes are reported separately by {@link #indexBytes()} and are <em>not</em> included here.
   */
  public long estimatedSizeBytes() {
    return estimatedSizeBytes;
  }

  /** Generation number of the active MemTable. */
  public long generation() {
    return generation;
  }

  /** Highest WAL batch position buffered into the MemTable, if any. */
  public Optional<Long> maxBufferedBatchPosition() {
    return maxBufferedBatchPosition;
  }

  /**
   * Writer-global count of WAL-durable batches (exclusive; 0 means none). Compare against {@code
   * globalOffset() + batchCount()} to see what this MemTable still owes the WAL.
   */
  public long durableBatchCount() {
    return durableBatchCount;
  }

  /** Writer-global coordinate of this MemTable's batch 0. */
  public long globalOffset() {
    return globalOffset;
  }

  /** First WAL batch position pending flush, if any. */
  public Optional<Long> pendingWalStartBatchPosition() {
    return pendingWalStartBatchPosition;
  }

  /** Last WAL batch position pending flush, if any. */
  public Optional<Long> pendingWalEndBatchPosition() {
    return pendingWalEndBatchPosition;
  }

  /** Number of WAL batches pending flush. */
  public long pendingWalBatchCount() {
    return pendingWalBatchCount;
  }

  /** Number of rows in WAL batches pending flush. */
  public long pendingWalRowCount() {
    return pendingWalRowCount;
  }

  /** Estimated bytes of WAL batches pending flush. */
  public long pendingWalEstimatedBytes() {
    return pendingWalEstimatedBytes;
  }

  /**
   * Bytes held by the active MemTable's in-memory indexes, its primary-key bloom filter included.
   * Usually what explains a shard near its ceiling with few rows in it: an HNSW graph is
   * pre-allocated in full from the configured row capacity.
   */
  public long indexBytes() {
    return indexBytes;
  }

  /**
   * Bytes held by generations that have flushed but are lingering out the configured
   * frozen-MemTable grace. Resident, but no flush reclaims them — the sweeper does, on a timer.
   */
  public long graceBytes() {
    return graceBytes;
  }

  /**
   * Every resident byte this shard holds. The figure a process-wide budget meters, as opposed to
   * what a flush can still give back.
   */
  public long retainedBytes() {
    return retainedBytes;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("rowCount", rowCount)
        .add("batchCount", batchCount)
        .add("estimatedSizeBytes", estimatedSizeBytes)
        .add("generation", generation)
        .add("maxBufferedBatchPosition", maxBufferedBatchPosition.orElse(null))
        .add("durableBatchCount", durableBatchCount)
        .add("globalOffset", globalOffset)
        .add("pendingWalStartBatchPosition", pendingWalStartBatchPosition.orElse(null))
        .add("pendingWalEndBatchPosition", pendingWalEndBatchPosition.orElse(null))
        .add("pendingWalBatchCount", pendingWalBatchCount)
        .add("pendingWalRowCount", pendingWalRowCount)
        .add("pendingWalEstimatedBytes", pendingWalEstimatedBytes)
        .add("indexBytes", indexBytes)
        .add("graceBytes", graceBytes)
        .add("retainedBytes", retainedBytes)
        .toString();
  }
}
