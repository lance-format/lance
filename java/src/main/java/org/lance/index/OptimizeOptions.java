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
  private final Optional<Long> maxRowsToConsolidate;

  private OptimizeOptions(
      Optional<Integer> numIndicesToMerge,
      Optional<List<String>> indexNames,
      boolean retrain,
      Optional<Long> maxRowsPerTask,
      Optional<Long> maxRowsPerSegment,
      Optional<Long> maxRowsToConsolidate) {
    this.numIndicesToMerge = numIndicesToMerge;
    this.indexNames = indexNames;
    this.retrain = retrain;
    this.maxRowsPerTask = maxRowsPerTask;
    this.maxRowsPerSegment = maxRowsPerSegment;
    this.maxRowsToConsolidate = maxRowsToConsolidate;
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
   * Packing target for the rows one build task covers. Default: unbounded.
   *
   * <p>Unindexed fragments are packed into tasks up to this target and up to {@link
   * #getMaxRowsPerSegment()}, since a task's output is never split. A fragment larger than the
   * target is indexed in a task of its own. With neither target set each index is one task. Setting
   * either target splits the build of every index that can merge segments, in {@link
   * org.lance.Dataset#optimizeIndices(OptimizeOptions)} as in the distributed plan; see {@link
   * org.lance.index.optimize.OptimizeIndices} for how that differs.
   */
  public Optional<Long> getMaxRowsPerTask() {
    return maxRowsPerTask;
  }

  /**
   * Packing target for the rows one committed segment holds. Default: unbounded.
   *
   * <p>Existing segments and new tasks are packed into output segments up to this target. An input
   * larger than the target becomes a segment of its own because segments and task outputs are not
   * split; unset merges everything into one segment. See {@link #getMaxRowsPerTask()} for what
   * setting it changes.
   */
  public Optional<Long> getMaxRowsPerSegment() {
    return maxRowsPerSegment;
  }

  /**
   * Physical row budget per vector index for consolidating its newest segments. At least two whole
   * segments must fit. Empty disables consolidation.
   *
   * <p>Each segment counts the greater of its physical index entries and its covered fragments'
   * physical rows, including deleted rows. Consolidation reuses the newest selected model.
   */
  public Optional<Long> getMaxRowsToConsolidate() {
    return maxRowsToConsolidate;
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
    private Optional<Long> maxRowsToConsolidate = Optional.empty();

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
     * Set the packing target for rows one build task covers; see {@link
     * OptimizeOptions#getMaxRowsPerTask()}.
     *
     * @param maxRowsPerTask packing target for the rows one task indexes; an oversize fragment is
     *     indexed alone
     */
    public Builder maxRowsPerTask(long maxRowsPerTask) {
      this.maxRowsPerTask = Optional.of(maxRowsPerTask);
      return this;
    }

    /**
     * Set the packing target for rows one committed segment holds; see {@link
     * OptimizeOptions#getMaxRowsPerSegment()}.
     *
     * @param maxRowsPerSegment packing target for the rows one segment holds; an oversize input is
     *     committed alone
     */
    public Builder maxRowsPerSegment(long maxRowsPerSegment) {
      this.maxRowsPerSegment = Optional.of(maxRowsPerSegment);
      return this;
    }

    /**
     * Enable {@linkplain OptimizeOptions#getMaxRowsToConsolidate() bounded vector consolidation}.
     * This cannot be combined with retraining, an explicit merge count, or {@link
     * #maxRowsPerSegment(long)}. An optional task limit must be positive and fit each fragment
     * selected for re-encoding.
     *
     * <pre>{@code
     * dataset.optimizeIndices(OptimizeOptions.builder().maxRowsToConsolidate(1_000_000).build());
     * }</pre>
     *
     * @param maxRowsToConsolidate positive physical row budget per logical vector index
     */
    public Builder maxRowsToConsolidate(long maxRowsToConsolidate) {
      this.maxRowsToConsolidate = Optional.of(maxRowsToConsolidate);
      return this;
    }

    public OptimizeOptions build() {
      return new OptimizeOptions(
          numIndicesToMerge,
          indexNames,
          retrain,
          maxRowsPerTask,
          maxRowsPerSegment,
          maxRowsToConsolidate);
    }
  }
}
