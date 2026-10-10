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

import com.google.common.base.Preconditions;

import java.util.List;
import java.util.Optional;

/**
 * Parameters for {@link org.lance.Dataset#initializeMemWal}.
 *
 * <p>At most one sharding mode may be selected: bucket sharding ({@link #withBucketSharding}),
 * identity sharding ({@link #withIdentitySharding}), or {@link #withUnsharded()}. With none
 * selected, shards are managed manually by passing shard IDs to {@link
 * org.lance.Dataset#memWalWriter}.
 */
public class InitializeMemWalParams {
  private Optional<List<String>> maintainedIndexes = Optional.empty();
  private Optional<String> bucketColumn = Optional.empty();
  private Optional<Integer> numBuckets = Optional.empty();
  private Optional<String> identityColumn = Optional.empty();
  private boolean unsharded = false;
  private Optional<ShardWriterConfig> writerConfigDefaults = Optional.empty();

  /**
   * Names of the indexes to maintain through the MemWAL. Must already exist on the dataset, and an
   * empty list maintains none.
   *
   * <p>Left unset, the MemWAL maintains every index the table has, including ones created later.
   */
  public InitializeMemWalParams withMaintainedIndexes(List<String> maintainedIndexes) {
    Preconditions.checkNotNull(maintainedIndexes, "maintainedIndexes must not be null");
    this.maintainedIndexes = Optional.of(maintainedIndexes);
    return this;
  }

  /** Use bucket sharding, deriving shard IDs by hashing {@code column} into {@code numBuckets}. */
  public InitializeMemWalParams withBucketSharding(String column, int numBuckets) {
    Preconditions.checkNotNull(column, "column must not be null");
    this.bucketColumn = Optional.of(column);
    this.numBuckets = Optional.of(numBuckets);
    return this;
  }

  /** Use identity sharding, deriving shard IDs directly from {@code column}. */
  public InitializeMemWalParams withIdentitySharding(String column) {
    Preconditions.checkNotNull(column, "column must not be null");
    this.identityColumn = Optional.of(column);
    return this;
  }

  /** Use a single unsharded shard for all writes. */
  public InitializeMemWalParams withUnsharded() {
    this.unsharded = true;
    return this;
  }

  /** Default {@link ShardWriterConfig} recorded in the MemWAL index for all writers. */
  public InitializeMemWalParams withWriterConfigDefaults(ShardWriterConfig writerConfigDefaults) {
    Preconditions.checkNotNull(writerConfigDefaults, "writerConfigDefaults must not be null");
    this.writerConfigDefaults = Optional.of(writerConfigDefaults);
    return this;
  }

  /** Empty when every index the table has is maintained. */
  public Optional<List<String>> maintainedIndexes() {
    return maintainedIndexes;
  }

  public Optional<String> bucketColumn() {
    return bucketColumn;
  }

  public Optional<Integer> numBuckets() {
    return numBuckets;
  }

  public Optional<String> identityColumn() {
    return identityColumn;
  }

  public boolean unsharded() {
    return unsharded;
  }

  public Optional<ShardWriterConfig> writerConfigDefaults() {
    return writerConfigDefaults;
  }
}
