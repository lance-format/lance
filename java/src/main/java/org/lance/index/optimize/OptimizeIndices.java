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
package org.lance.index.optimize;

import org.lance.Dataset;
import org.lance.JniLoader;
import org.lance.LockManager;
import org.lance.index.OptimizeOptions;

import com.google.common.base.Preconditions;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Index optimization as plan, execute, merge and commit.
 *
 * <p>{@link Dataset#optimizeIndices(OptimizeOptions)} runs these four stages in one process. A
 * distributed engine runs the same stages across workers:
 *
 * <pre>{@code
 * // driver
 * OptimizeIndicesPlan plan = OptimizeIndices.plan(dataset, options);
 * // workers: one call per task, then one per merge; each worker opens the dataset itself
 * Optional<IndexSegmentResult> built = task.execute(workerDataset);
 * Optional<IndexSegmentResult> merged = merge.execute(workerDataset, builtResultsOfItsGroup);
 * // driver
 * OptimizeIndices.commit(dataset, mergedResults);
 * }</pre>
 *
 * <p>Without {@link OptimizeOptions#getMaxRowsPerTask()} and {@link
 * OptimizeOptions#getMaxRowsPerSegment()} every index is one task: the single-process optimize over
 * all of its segments and all of its unindexed fragments, with the caller's options, and a merge
 * that passes the result through. With a bound set, an index whose kind can merge segments (vector,
 * BTree, Bitmap, NGram, Inverted) is split: tasks build one uncommitted segment each over a group
 * of fragments, and merges combine the task results of their group with the committed segments they
 * absorb into one segment. Other indices stay one task, and an index a split build cannot handle is
 * skipped with a warning unless {@link OptimizeOptions#getIndexNames()} names it, which fails
 * instead.
 *
 * <p>The commit publishes every merged segment in one transaction anchored at the plan's read
 * version, so conflict detection covers everything committed since the plan was made. Results may
 * be partial: whatever they do not cover stays unindexed for the next optimize.
 *
 * <p>Tasks, merges and results are {@link java.io.Serializable} so an engine can ship them to its
 * workers; a worker only needs a {@link Dataset} opened on the same table, at any version.
 */
public class OptimizeIndices {
  static {
    JniLoader.ensureLoaded();
  }

  private OptimizeIndices() {}

  /**
   * Plan the optimization of the dataset's indices, as of the version the dataset has checked out.
   *
   * @param dataset the dataset
   * @param options what to optimize and how to split the work
   * @return the plan; empty when there is nothing to do
   */
  public static OptimizeIndicesPlan plan(Dataset dataset, OptimizeOptions options) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(options);
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return nativePlan(dataset, options);
    }
  }

  /**
   * Commit the merged results of one plan. The dataset is left at the new version.
   *
   * <p>The results must all come from the same plan. With no results nothing is committed, unless
   * the table requires a MemWAL catch-up commit.
   *
   * @param dataset the dataset
   * @param results the results of the plan's merges
   */
  public static void commit(Dataset dataset, List<IndexSegmentResult> results) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(results);
    List<String> resultJsons =
        results.stream().map(IndexSegmentResult::getJson).collect(Collectors.toList());
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      nativeCommit(dataset, resultJsons);
    }
  }

  private static native OptimizeIndicesPlan nativePlan(Dataset dataset, OptimizeOptions options);

  private static native void nativeCommit(Dataset dataset, List<String> results);
}
