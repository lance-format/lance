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

import org.lance.Dataset;
import org.lance.JniLoader;
import org.lance.LockManager;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.FieldVector;

import java.util.List;
import java.util.Optional;

/**
 * Plans primary-key point lookups across all MemWAL LSM levels.
 *
 * <p>More efficient than {@link LsmScanner} for known-primary-key lookups thanks to bloom filter
 * optimizations and short-circuit evaluation.
 */
public class LsmPointLookupPlanner implements AutoCloseable {
  static {
    JniLoader.ensureLoaded();
  }

  private long nativeLookupPlannerHandle;
  private final BufferAllocator allocator;
  private final LockManager lockManager = new LockManager();

  /**
   * @param dataset the base dataset
   * @param shardSnapshots shard snapshots specifying the SSTables to include
   */
  public LsmPointLookupPlanner(Dataset dataset, List<ShardSnapshot> shardSnapshots) {
    this(dataset, shardSnapshots, null);
  }

  /**
   * @param dataset the base dataset
   * @param shardSnapshots shard snapshots specifying the SSTables to include
   * @param pkColumns primary key column names; inferred from schema metadata when {@code null}
   */
  public LsmPointLookupPlanner(
      Dataset dataset, List<ShardSnapshot> shardSnapshots, List<String> pkColumns) {
    Preconditions.checkNotNull(dataset, "dataset must not be null");
    Preconditions.checkNotNull(shardSnapshots, "shardSnapshots must not be null");
    this.allocator = dataset.allocator();
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      nativeCreate(dataset, shardSnapshots, Optional.ofNullable(pkColumns));
    }
  }

  private native void nativeCreate(
      Dataset dataset, List<ShardSnapshot> shardSnapshots, Optional<List<String>> pkColumns);

  /**
   * Plan a point lookup by primary key value.
   *
   * @param pkValue for a single-column primary key, a vector with exactly one element; for a
   *     composite primary key, a {@code StructVector} with one row and one child per key column
   * @param columns columns to project; pass {@code null} to return all columns
   * @return an executable plan
   */
  public ExecutionPlan planLookup(FieldVector pkValue, List<String> columns) {
    Preconditions.checkNotNull(pkValue, "pkValue must not be null");
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(
          nativeLookupPlannerHandle != 0, "LsmPointLookupPlanner is closed");
      try (ArrowArray array = ArrowArray.allocateNew(allocator);
          ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
        Data.exportVector(allocator, pkValue, null, array, schema);
        ExecutionPlan plan =
            nativePlanLookup(
                array.memoryAddress(), schema.memoryAddress(), Optional.ofNullable(columns));
        plan.allocator = allocator;
        return plan;
      }
    }
  }

  /** Plan a point lookup by primary key value, returning all columns. */
  public ExecutionPlan planLookup(FieldVector pkValue) {
    return planLookup(pkValue, null);
  }

  private native ExecutionPlan nativePlanLookup(
      long arrayAddress, long schemaAddress, Optional<List<String>> columns);

  /**
   * Close the planner and release native resources. If the planner is already closed, invoking this
   * method has no effect.
   */
  @Override
  public void close() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeLookupPlannerHandle != 0) {
        releaseNativeLookupPlanner(nativeLookupPlannerHandle);
        nativeLookupPlannerHandle = 0;
      }
    }
  }

  private native void releaseNativeLookupPlanner(long handle);
}
