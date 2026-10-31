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
package org.lance.index;

import java.util.List;
import java.util.Optional;

/**
 * Options for optimizing indices on a dataset.
 *
 * <p>This mirrors the behavior of {@code lance_index::optimize::OptimizeOptions} in Rust.
 *
 * <p>All fields are optional on the Java side except {@code retrain}. Defaults are delegated to the
 * Rust implementation.
 */
public class OptimizeOptions {

  private final Optional<Integer> numIndicesToMerge;
  private final Optional<List<String>> indexNames;
  private final boolean retrain;
  private final Optional<Long> maxRowsPerTask;
  private final Optional<Long> maxRowsPerSegment;

  private OptimizeOptions(
      Optional<Integer> numIndicesToMerge,
      Optional<List<String>> indexNames,
      boolean retrain,
      Optional<Long> maxRowsPerTask,
      Optional<Long> maxRowsPerSegment) {
    this.numIndicesToMerge = numIndicesToMerge;
    this.indexNames = indexNames;
    this.retrain = retrain;
    this.maxRowsPerTask = maxRowsPerTask;
    this.maxRowsPerSegment = maxRowsPerSegment;
  }

  /** Number of indices to merge per index name. */
  public Optional<Integer> getNumIndicesToMerge() {
    return numIndicesToMerge;
  }

  /**
   * Names of indices to optimize. If empty, all user indices will be considered (system indices are
   * always excluded).
   */
  public Optional<List<String>> getIndexNames() {
    return indexNames;
  }

  /** Whether to retrain the index instead of performing an incremental merge. */
  public boolean isRetrain() {
    return retrain;
  }

  /**
   * Upper bound on the rows one build task covers. Default: unbounded.
   *
   * <p>Unindexed fragments are packed into tasks of at most this many rows, and of at most {@link
   * #getMaxRowsPerSegment()}, since a task's output is never split (nor is a fragment); with
   * neither set each index is one task. Setting either bound splits the build of every index that
   * can merge segments, in {@link org.lance.Dataset#optimizeIndices(OptimizeOptions)} as in the
   * distributed plan; see {@link org.lance.index.optimize.OptimizeIndices} for how that differs.
   */
  public Optional<Long> getMaxRowsPerTask() {
    return maxRowsPerTask;
  }

  /**
   * Upper bound on the rows one committed segment holds. Default: unbounded.
   *
   * <p>Existing segments and new tasks are packed into output segments of at most this many rows;
   * unset merges everything into one segment. See {@link #getMaxRowsPerTask()} for what setting it
   * changes.
   */
  public Optional<Long> getMaxRowsPerSegment() {
    return maxRowsPerSegment;
  }

  /** Create a new builder for {@link OptimizeOptions}. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder for {@link OptimizeOptions}. */
  public static class Builder {
    private Optional<Integer> numIndicesToMerge = Optional.empty();
    private Optional<List<String>> indexNames = Optional.empty();
    private boolean retrain = false;
    private Optional<Long> maxRowsPerTask = Optional.empty();
    private Optional<Long> maxRowsPerSegment = Optional.empty();

    private Builder() {}

    /**
     * Set the number of indices to merge.
     *
     * @param numIndicesToMerge number of indices to merge per index name
     */
    public Builder numIndicesToMerge(int numIndicesToMerge) {
      this.numIndicesToMerge = Optional.of(numIndicesToMerge);
      return this;
    }

    /**
     * Restrict optimization to a subset of index names.
     *
     * @param indexNames index names to optimize
     */
    public Builder indexNames(List<String> indexNames) {
      this.indexNames = Optional.ofNullable(indexNames);
      return this;
    }

    /**
     * Whether to retrain the index.
     *
     * @param retrain if true, retrain instead of incremental merge
     */
    public Builder retrain(boolean retrain) {
      this.retrain = retrain;
      return this;
    }

    /**
     * Bound the rows one build task covers; see {@link OptimizeOptions#getMaxRowsPerTask()}.
     *
     * @param maxRowsPerTask upper bound on the rows one task indexes
     */
    public Builder maxRowsPerTask(long maxRowsPerTask) {
      this.maxRowsPerTask = Optional.of(maxRowsPerTask);
      return this;
    }

    /**
     * Bound the rows one committed segment holds; see {@link
     * OptimizeOptions#getMaxRowsPerSegment()}.
     *
     * @param maxRowsPerSegment upper bound on the rows one segment holds
     */
    public Builder maxRowsPerSegment(long maxRowsPerSegment) {
      this.maxRowsPerSegment = Optional.of(maxRowsPerSegment);
      return this;
    }

    public OptimizeOptions build() {
      return new OptimizeOptions(
          numIndicesToMerge, indexNames, retrain, maxRowsPerTask, maxRowsPerSegment);
    }
  }
}
