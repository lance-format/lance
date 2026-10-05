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

import java.util.Collections;
import java.util.List;
import java.util.Map;

/** Details of the MemWAL index attached to a dataset. */
public class MemWalIndexDetails {
  private final long numShards;
  private final List<String> maintainedIndexes;
  private final boolean maintainAllIndexes;
  private final Map<String, String> writerConfigDefaults;
  private final List<ShardingSpec> shardingSpecs;

  public MemWalIndexDetails(
      long numShards,
      List<String> maintainedIndexes,
      boolean maintainAllIndexes,
      Map<String, String> writerConfigDefaults,
      List<ShardingSpec> shardingSpecs) {
    this.numShards = numShards;
    this.maintainedIndexes =
        maintainedIndexes == null ? Collections.emptyList() : maintainedIndexes;
    this.maintainAllIndexes = maintainAllIndexes;
    this.writerConfigDefaults =
        writerConfigDefaults == null ? Collections.emptyMap() : writerConfigDefaults;
    this.shardingSpecs = shardingSpecs == null ? Collections.emptyList() : shardingSpecs;
  }

  /** Number of shards tracked by the MemWAL index. */
  public long numShards() {
    return numShards;
  }

  /**
   * Names of the indexes maintained through the MemWAL. Empty when {@link #maintainAllIndexes()} is
   * set, which names no index because it covers every one.
   */
  public List<String> maintainedIndexes() {
    return maintainedIndexes;
  }

  /**
   * Whether every index the table has is maintained, including ones created later.
   *
   * <p>This is what separates maintaining everything from maintaining nothing: both leave {@link
   * #maintainedIndexes()} empty.
   */
  public boolean maintainAllIndexes() {
    return maintainAllIndexes;
  }

  /** Default {@link ShardWriterConfig} values persisted in the MemWAL index. */
  public Map<String, String> writerConfigDefaults() {
    return writerConfigDefaults;
  }

  /** The sharding specifications recorded in the MemWAL index. */
  public List<ShardingSpec> shardingSpecs() {
    return shardingSpecs;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("numShards", numShards)
        .add("maintainedIndexes", maintainedIndexes)
        .add("maintainAllIndexes", maintainAllIndexes)
        .add("writerConfigDefaults", writerConfigDefaults)
        .add("shardingSpecs", shardingSpecs)
        .toString();
  }
}
