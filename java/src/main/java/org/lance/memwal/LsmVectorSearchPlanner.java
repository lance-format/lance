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
import org.apache.arrow.vector.Float4Vector;

import java.util.List;
import java.util.Optional;

/**
 * Plans IVF-PQ vector KNN search across all MemWAL LSM levels.
 *
 * <p>Results include staleness filtering so only the latest version of each row is returned. The
 * output schema includes a {@code _distance} column.
 */
public class LsmVectorSearchPlanner implements AutoCloseable {
  static {
    JniLoader.ensureLoaded();
  }

  private long nativeVectorPlannerHandle;
  private final BufferAllocator allocator;
  private final LockManager lockManager = new LockManager();

  /**
   * @param dataset the base dataset
   * @param shardSnapshots shard snapshots specifying the SSTables to include
   * @param vectorColumn name of the {@code FixedSizeList<float32>} vector column
   */
  public LsmVectorSearchPlanner(
      Dataset dataset, List<ShardSnapshot> shardSnapshots, String vectorColumn) {
    this(dataset, shardSnapshots, vectorColumn, null, null, null);
  }

  /**
   * @param dataset the base dataset
   * @param shardSnapshots shard snapshots specifying the SSTables to include
   * @param vectorColumn name of the {@code FixedSizeList<float32>} vector column
   * @param pkColumns primary key column names; inferred from schema metadata when {@code null}
   * @param distanceType distance metric, one of {@code "l2"}, {@code "cosine"}, {@code "dot"},
   *     {@code "hamming"}; defaults to {@code "l2"} when {@code null}
   */
  public LsmVectorSearchPlanner(
      Dataset dataset,
      List<ShardSnapshot> shardSnapshots,
      String vectorColumn,
      List<String> pkColumns,
      String distanceType) {
    this(dataset, shardSnapshots, vectorColumn, pkColumns, distanceType, null);
  }

  /**
   * @param dataset the base dataset
   * @param shardSnapshots shard snapshots specifying the SSTables to include
   * @param vectorColumn name of the {@code FixedSizeList<float32>} vector column
   * @param pkColumns primary key column names; inferred from schema metadata when {@code null}
   * @param distanceType distance metric, one of {@code "l2"}, {@code "cosine"}, {@code "dot"},
   *     {@code "hamming"}; defaults to {@code "l2"} when {@code null}
   * @param filter SQL predicate applied as a prefilter to every LSM source before vector candidate
   *     selection; pass {@code null} for no predicate
   */
  public LsmVectorSearchPlanner(
      Dataset dataset,
      List<ShardSnapshot> shardSnapshots,
      String vectorColumn,
      List<String> pkColumns,
      String distanceType,
      String filter) {
    Preconditions.checkNotNull(dataset, "dataset must not be null");
    Preconditions.checkNotNull(shardSnapshots, "shardSnapshots must not be null");
    Preconditions.checkNotNull(vectorColumn, "vectorColumn must not be null");
    this.allocator = dataset.allocator();
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      nativeCreate(
          dataset,
          shardSnapshots,
          vectorColumn,
          Optional.ofNullable(pkColumns),
          Optional.ofNullable(distanceType),
          Optional.ofNullable(filter));
    }
  }

  private native void nativeCreate(
      Dataset dataset,
      List<ShardSnapshot> shardSnapshots,
      String vectorColumn,
      Optional<List<String>> pkColumns,
      Optional<String> distanceType,
      Optional<String> filter);

  /**
   * Plan a KNN vector search.
   *
   * @param query a flat float32 vector of length {@code vectorDim}
   * @param k number of nearest neighbours to return
   * @param nprobes number of IVF partitions to probe
   * @param columns columns to project; pass {@code null} to return all columns plus {@code
   *     _distance}
   * @param refineBaseTable when true, the base-table arm re-ranks candidates with exact distances
   *     (refine factor 1). Useful when the base table uses an approximate index (IVF-PQ).
   *     Auto-enabled whenever stale filtering is on (see {@code overfetchFactor}).
   * @param overfetchFactor over-fetch multiple for sources with rows superseded by newer
   *     generations. Must be at least {@code 1.0}:
   *     <ul>
   *       <li>{@code == 1.0}: no over-fetch — an affected source fetches exactly {@code k}
   *           candidates and may return fewer than {@code k} live rows after stale rows are
   *           dropped.
   *       <li>{@code > 1.0}: with over-fetch — such a source fetches {@code ceil(k *
   *           overfetchFactor)} candidates so dropping stale rows is less likely to under-fill the
   *           result.
   *     </ul>
   *
   * @return an executable plan
   */
  public ExecutionPlan planSearch(
      Float4Vector query,
      int k,
      int nprobes,
      List<String> columns,
      boolean refineBaseTable,
      double overfetchFactor) {
    Preconditions.checkNotNull(query, "query must not be null");
    Preconditions.checkArgument(k > 0, "k must be positive, got %s", k);
    Preconditions.checkArgument(nprobes > 0, "nprobes must be positive, got %s", nprobes);
    Preconditions.checkArgument(
        Double.isFinite(overfetchFactor) && overfetchFactor >= 1.0,
        "overfetchFactor must be finite and >= 1.0, got %s",
        overfetchFactor);
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      Preconditions.checkArgument(
          nativeVectorPlannerHandle != 0, "LsmVectorSearchPlanner is closed");
      try (ArrowArray array = ArrowArray.allocateNew(allocator);
          ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
        Data.exportVector(allocator, query, null, array, schema);
        ExecutionPlan plan =
            nativePlanSearch(
                array.memoryAddress(),
                schema.memoryAddress(),
                k,
                nprobes,
                Optional.ofNullable(columns),
                refineBaseTable,
                overfetchFactor);
        plan.allocator = allocator;
        return plan;
      }
    }
  }

  /**
   * Plan a KNN vector search with stale filtering on and no over-fetch ({@code overfetchFactor =
   * 1.0}).
   *
   * @param query a flat float32 vector of length {@code vectorDim}
   * @param k number of nearest neighbours to return
   * @param nprobes number of IVF partitions to probe
   * @param columns columns to project; pass {@code null} to return all columns plus {@code
   *     _distance}
   * @param refineBaseTable when true, the base-table arm re-ranks candidates with exact distances.
   * @return an executable plan
   */
  public ExecutionPlan planSearch(
      Float4Vector query, int k, int nprobes, List<String> columns, boolean refineBaseTable) {
    return planSearch(query, k, nprobes, columns, refineBaseTable, 1.0);
  }

  /**
   * Plan a KNN vector search with stale filtering on (no refine, no over-fetch).
   *
   * @param query a flat float32 vector of length {@code vectorDim}
   * @param k number of nearest neighbours to return
   * @param nprobes number of IVF partitions to probe
   * @param columns columns to project; pass {@code null} to return all columns plus {@code
   *     _distance}
   * @return an executable plan
   */
  public ExecutionPlan planSearch(Float4Vector query, int k, int nprobes, List<String> columns) {
    return planSearch(query, k, nprobes, columns, false, 1.0);
  }

  /** Plan a KNN vector search with default {@code nprobes} of 20. */
  public ExecutionPlan planSearch(Float4Vector query, int k) {
    return planSearch(query, k, 20, null, false, 1.0);
  }

  private native ExecutionPlan nativePlanSearch(
      long arrayAddress,
      long schemaAddress,
      int k,
      int nprobes,
      Optional<List<String>> columns,
      boolean refineBaseTable,
      double overfetchFactor);

  /**
   * Close the planner and release native resources. If the planner is already closed, invoking this
   * method has no effect.
   */
  @Override
  public void close() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeVectorPlannerHandle != 0) {
        releaseNativeVectorPlanner(nativeVectorPlannerHandle);
        nativeVectorPlannerHandle = 0;
      }
    }
  }

  private native void releaseNativeVectorPlanner(long handle);
}
