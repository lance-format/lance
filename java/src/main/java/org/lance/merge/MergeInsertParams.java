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
package org.lance.merge;

import org.lance.memwal.CompactedSsTable;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

public class MergeInsertParams {
  private final List<String> on;

  private WhenMatched whenMatched = WhenMatched.DoNothing;
  private Optional<String> whenMatchedUpdateExpr = Optional.empty();

  private WhenNotMatched whenNotMatched = WhenNotMatched.InsertAll;

  private WhenNotMatchedBySource whenNotMatchedBySource = WhenNotMatchedBySource.Keep;
  private Optional<String> whenNotMatchedBySourceDeleteExpr = Optional.empty();
  private Optional<ByteBuffer> whenNotMatchedBySourceDeleteSubstraitExpr = Optional.empty();

  private int conflictRetries = 10;
  private long retryTimeoutMs = 30 * 1000;
  private boolean skipAutoCleanup = false;
  private boolean useIndex = true;
  private List<CompactedSsTable> compactedSstables = Collections.emptyList();

  public MergeInsertParams(List<String> on) {
    this.on = on;
  }

  /**
   * Specify that when a row in the source table matches a row in the target table, the row is
   * deleted from the target table and the matched row based on the source table is inserted.
   *
   * <p>This can be used to achieve upsert behavior.
   *
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withMatchedUpdateAll() {
    this.whenMatched = WhenMatched.UpdateAll;
    return this;
  }

  /**
   * Specify that when a row in the source table matches a row in the target table, the row in the
   * target table is kept unchanged.
   *
   * <p>This can be used to achieve find-or-create behavior.
   *
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withMatchedDoNothing() {
    this.whenMatched = WhenMatched.DoNothing;
    return this;
  }

  /**
   * Specify that when a row in the source table matches a row in the target table, the row in the
   * target table is deleted.
   *
   * <p>This can be used to achieve "when matched delete" behavior.
   *
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withMatchedDelete() {
    this.whenMatched = WhenMatched.Delete;
    return this;
  }

  /**
   * Specify that when a row in the source table matches a row in the target table and the
   * expression evaluates to true, the row in the target table is updated by the matched row from
   * the source table.
   *
   * <p>This can be used to achieve upsert behavior.
   *
   * <p>The expression can reference source tables' columns with <code>source.</code> and target
   * tables' columns with <code>target.</code> This is an example: <code>
   * source.column1 = target.column1 AND source.column2 = target.column2</code>
   *
   * @param expr The expression to evaluate on the rows in the source table and target table.
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withMatchedUpdateIf(String expr) {
    Preconditions.checkNotNull(expr);
    this.whenMatched = WhenMatched.UpdateIf;
    this.whenMatchedUpdateExpr = Optional.of(expr);
    return this;
  }

  /**
   * Specify that when a row in the source table matches a row in the target table, the entire merge
   * operation will fail with an error.
   *
   * <p>This can be used to ensure that no existing rows are overwritten or modified after inserted.
   *
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withMatchedFail() {
    this.whenMatched = WhenMatched.Fail;
    return this;
  }

  /**
   * Specify what should happen when a source row has no match in the target.
   *
   * @param whenNotMatched The action to take when a source row has no match in the target.
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withNotMatched(WhenNotMatched whenNotMatched) {
    this.whenNotMatched = whenNotMatched;
    return this;
  }

  /**
   * Specify that when a target row has no match in the source, the row is kept in the target table.
   *
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withNotMatchedBySourceKeep() {
    this.whenNotMatchedBySource = WhenNotMatchedBySource.Keep;
    return this;
  }

  /**
   * Specify that when a target row has no match in the source, the row is deleted from the target
   * table.
   *
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withNotMatchedBySourceDelete() {
    this.whenNotMatchedBySource = WhenNotMatchedBySource.Delete;
    return this;
  }

  /**
   * Specify that when a target row has no match in the source and the expression evaluates to true,
   * the row is deleted from the target table.
   *
   * @param expr The sql expression to evaluate on the rows in the target table.
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withNotMatchedBySourceDeleteIf(String expr) {
    Preconditions.checkNotNull(expr);
    this.whenNotMatchedBySource = WhenNotMatchedBySource.DeleteIf;
    this.whenNotMatchedBySourceDeleteExpr = Optional.of(expr);
    this.whenNotMatchedBySourceDeleteSubstraitExpr = Optional.empty();
    return this;
  }

  /**
   * Specify that when a target row has no match in the source and the expression evaluates to true,
   * the row is deleted from the target table.
   *
   * @param expr The substrait expression to evaluate on the rows in the target table.
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withNotMatchedBySourceDeleteSubstraitIf(ByteBuffer expr) {
    Preconditions.checkNotNull(expr);
    this.whenNotMatchedBySource = WhenNotMatchedBySource.DeleteIf;
    this.whenNotMatchedBySourceDeleteExpr = Optional.empty();
    this.whenNotMatchedBySourceDeleteSubstraitExpr = Optional.of(expr);
    return this;
  }

  /**
   * Set number of times to retry the operation if there is contention.
   *
   * <p>If this is set greater than 0, then the operation will keep a copy of the input data either
   * in memory or on disk (depending on the size of the data) and will retry the operation if there
   * is contention.
   *
   * <p>Default is 10.
   *
   * @param retries Number of times to retry the operation if there is contention.
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withConflictRetries(int retries) {
    this.conflictRetries = retries;
    return this;
  }

  /**
   * Set the timeout in milliseconds used to limit retries.
   *
   * <p>This is the maximum time to spend on the operation before giving up. At least one attempt
   * will be made, regardless of how long it takes to complete. Subsequent attempts will be
   * cancelled once this timeout is reached. If the timeout has been reached during the first
   * attempt, the operation will be cancelled immediately.
   *
   * <p>Default is 30000.
   *
   * @param timeoutMs Timeout in milliseconds used to limit retries.
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withRetryTimeoutMs(long timeoutMs) {
    this.retryTimeoutMs = timeoutMs;
    return this;
  }

  /**
   * If true, skip auto cleanup during commits. This should be set to true for high frequency writes
   * to improve performance. This is also useful if the writer does not have delete permissions and
   * the clean up would just try and log a failure anyway.
   *
   * @param skipAutoCleanup Whether to skip auto cleanup during commits.
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withSkipAutoCleanup(boolean skipAutoCleanup) {
    this.skipAutoCleanup = skipAutoCleanup;
    return this;
  }

  /**
   * Controls whether to use indices for the merge operation.
   *
   * <p>When set to false, forces a full table scan even if an index exists on the join key. This
   * can be useful for benchmarking or when the optimizer chooses a suboptimal path.
   *
   * <p>Default is true (use index if available).
   *
   * @param useIndex Whether to use indices for the merge join
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams withUseIndex(boolean useIndex) {
    this.useIndex = useIndex;
    return this;
  }

  /**
   * Mark MemWAL SSTables as compacted into the base table.
   *
   * <p>Use this when merge insert compacts MemWAL SSTables. It updates MemWAL compaction progress
   * to prevent the same SSTables from being compacted again, in the same commit as the data.
   *
   * <p><b>For multi-pass compaction, call this only on the final successful data-changing pass.</b>
   * Intermediate passes must not carry compaction progress. Lance cannot tell whether a caller has
   * another pass planned, so it cannot enforce this: if a delete pass carried the marker and the
   * process then died before the matching upsert, the recorded progress would claim rows were
   * copied in that never were.
   *
   * @param sstables the SSTables being compacted
   * @return This MergeInsertParams instance
   */
  public MergeInsertParams markSstablesAsCompacted(List<CompactedSsTable> sstables) {
    Preconditions.checkNotNull(sstables, "sstables must not be null");
    this.compactedSstables = sstables;
    return this;
  }

  public List<String> on() {
    return on;
  }

  public List<CompactedSsTable> getCompactedSstables() {
    return compactedSstables;
  }

  public WhenMatched whenMatched() {
    return whenMatched;
  }

  public String whenMatchedValue() {
    return whenMatched.name();
  }

  public Optional<String> whenMatchedUpdateExpr() {
    return whenMatchedUpdateExpr;
  }

  public WhenNotMatched whenNotMatched() {
    return whenNotMatched;
  }

  public String whenNotMatchedValue() {
    return whenNotMatched.name();
  }

  public WhenNotMatchedBySource whenNotMatchedBySource() {
    return whenNotMatchedBySource;
  }

  public String whenNotMatchedBySourceValue() {
    return whenNotMatchedBySource.name();
  }

  public Optional<String> whenNotMatchedBySourceDeleteExpr() {
    return whenNotMatchedBySourceDeleteExpr;
  }

  public Optional<ByteBuffer> whenNotMatchedBySourceDeleteSubstraitExpr() {
    return whenNotMatchedBySourceDeleteSubstraitExpr;
  }

  public int conflictRetries() {
    return conflictRetries;
  }

  public long retryTimeoutMs() {
    return retryTimeoutMs;
  }

  public boolean skipAutoCleanup() {
    return skipAutoCleanup;
  }

  public boolean useIndex() {
    return useIndex;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("on", on)
        .add("whenMatched", whenMatched)
        .add("whenMatchedUpdateExpr", whenMatchedUpdateExpr.orElse(null))
        .add("whenNotMatched", whenNotMatched)
        .add("whenNotMatchedBySource", whenNotMatchedBySource)
        .add("whenNotMatchedBySourceDeleteExpr", whenNotMatchedBySourceDeleteExpr.orElse(null))
        .add(
            "whenNotMatchedBySourceDeleteSubstraitExpr",
            whenNotMatchedBySourceDeleteSubstraitExpr
                .map(buf -> "ByteBuffer[" + buf.remaining() + " bytes]")
                .orElse(null))
        .add("conflictRetries", conflictRetries)
        .add("retryTimeoutMs", retryTimeoutMs)
        .add("skipAutoCleanup", skipAutoCleanup)
        .add("useIndex", useIndex)
        .toString();
  }

  public enum WhenMatched {
    /**
     * The row is deleted from the target table and a new row is inserted based on the source table.
     * This can be used to achieve upsert behavior.
     */
    UpdateAll,

    /** The row is kept unchanged. This can be used to achieve find-or-create behavior. */
    DoNothing,

    /**
     * The row is updated (similar to UpdateAll) only for rows where the expression evaluates to
     * true.
     */
    UpdateIf,

    /**
     * The operation will fail if a match is found between the source and target table. This can be
     * used to ensure that no existing rows are overwritten or modified after inserted.
     */
    Fail,

    /**
     * The row is deleted from the target table when a row in the source table matches a row in the
     * target table.
     */
    Delete
  }

  public enum WhenNotMatched {
    /**
     * The new row is inserted into the target table. This is used in both find-or-create and upsert
     * operations
     */
    InsertAll,

    /** The new row is ignored. */
    DoNothing,
  }

  public enum WhenNotMatchedBySource {
    /**
     * Do not delete rows from the target table This can be used for a find-or-create or an upsert
     * operation
     */
    Keep,

    /** Delete all rows from target table that don't match a row in the source table */
    Delete,

    /**
     * Delete rows from the target table if there is no match AND the expression evaluates to true
     * This can be used to replace a region of data with new data
     */
    DeleteIf,
  }
}
