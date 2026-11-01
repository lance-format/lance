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
import com.google.common.base.Preconditions;

/**
 * Points to an SSTable compacted into the base table.
 *
 * <p>Pass a list of these to {@link org.lance.merge.MergeInsertParams#markSstablesAsCompacted} so
 * Lance can record compaction progress.
 */
public class CompactedSsTable {
  private final String shardId;
  private final long generation;

  /**
   * @param shardId UUID string for the write shard
   * @param generation generation number from {@link ShardSnapshot#sstables()}
   */
  public CompactedSsTable(String shardId, long generation) {
    Preconditions.checkNotNull(shardId, "shardId must not be null");
    this.shardId = shardId;
    this.generation = generation;
  }

  /** UUID string for the write shard. */
  public String getShardId() {
    return shardId;
  }

  /** The compacted SSTable's generation number. */
  public long getGeneration() {
    return generation;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("shardId", shardId)
        .add("generation", generation)
        .toString();
  }
}
