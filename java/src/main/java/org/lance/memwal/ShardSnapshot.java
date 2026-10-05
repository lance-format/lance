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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Snapshot of a MemWAL shard's state, used when constructing scanners and planners.
 *
 * <p>The builder methods ({@link #withSpecId}, {@link #withCurrentGeneration}, {@link
 * #withSsTable}) mutate this instance and return it for chaining.
 */
public class ShardSnapshot {
  private final String shardId;
  private int specId = 0;
  private long currentGeneration = 0;
  private final List<SsTable> sstables = new ArrayList<>();

  /**
   * @param shardId UUID string for the write shard
   */
  public ShardSnapshot(String shardId) {
    Preconditions.checkNotNull(shardId, "shardId must not be null");
    this.shardId = shardId;
  }

  /** Set the {@link ShardingSpec} ID for this snapshot. */
  public ShardSnapshot withSpecId(int specId) {
    this.specId = specId;
    return this;
  }

  /** Set the current (active) generation number. */
  public ShardSnapshot withCurrentGeneration(long currentGeneration) {
    this.currentGeneration = currentGeneration;
    return this;
  }

  /** Add an SSTable with its storage path. */
  public ShardSnapshot withSsTable(long generation, String path) {
    Preconditions.checkNotNull(path, "path must not be null");
    this.sstables.add(new SsTable(generation, path));
    return this;
  }

  /** UUID string for this shard. */
  public String shardId() {
    return shardId;
  }

  /** The {@link ShardingSpec} ID for this snapshot. */
  public int specId() {
    return specId;
  }

  /** The current (active) generation number. */
  public long currentGeneration() {
    return currentGeneration;
  }

  /** The SSTables included in this snapshot. */
  public List<SsTable> sstables() {
    return Collections.unmodifiableList(sstables);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("shardId", shardId)
        .add("specId", specId)
        .add("currentGeneration", currentGeneration)
        .add("sstables", sstables)
        .toString();
  }
}
