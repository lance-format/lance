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
package org.lance.compaction;

import org.lance.Dataset;
import org.lance.JniLoader;
import org.lance.LockManager;

import com.google.common.base.Preconditions;

import java.util.List;
import java.util.Optional;

/** The entrypoint of distributed compaction-related methods. */
public class Compaction {
  static {
    JniLoader.ensureLoaded();
  }

  public static CompactionPlan planCompaction(
      Dataset dataset, CompactionOptions compactionOptions) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(compactionOptions);

    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return nativePlanCompaction(
          dataset,
          compactionOptions.getTargetRowsPerFragment(),
          compactionOptions.getMaxRowsPerGroup(),
          compactionOptions.getMaxBytesPerFile(),
          compactionOptions.getMaterializeDeletions(),
          compactionOptions.getMaterializeDeletionsThreshold(),
          compactionOptions.getNumThreads(),
          compactionOptions.getBatchSize(),
          compactionOptions.getDeferIndexRemap(),
          compactionOptions.getCompactionMode(),
          compactionOptions.getBinaryCopyReadBatchBytes(),
          compactionOptions.getMaxSourceFragments(),
          compactionOptions.getMaxSourceRows(),
          compactionOptions.getMaxSourceBytes(),
          compactionOptions.getExcludedFragmentIds());
    }
  }

  public static CompactionMetrics commitCompaction(
      Dataset dataset, List<RewriteResult> rewriteResults, CompactionOptions compactionOptions) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(rewriteResults);
    Preconditions.checkNotNull(compactionOptions);
    return nativeCommitCompaction(
        dataset,
        rewriteResults,
        compactionOptions.getTargetRowsPerFragment(),
        compactionOptions.getMaxRowsPerGroup(),
        compactionOptions.getMaxBytesPerFile(),
        compactionOptions.getMaterializeDeletions(),
        compactionOptions.getMaterializeDeletionsThreshold(),
        compactionOptions.getNumThreads(),
        compactionOptions.getBatchSize(),
        compactionOptions.getDeferIndexRemap(),
        compactionOptions.getCompactionMode(),
        compactionOptions.getBinaryCopyReadBatchBytes(),
        compactionOptions.getMaxSourceFragments(),
        compactionOptions.getMaxSourceRows(),
        compactionOptions.getMaxSourceBytes(),
        compactionOptions.getExcludedFragmentIds());
  }

  /**
   * Java wrapper around the raw commit-compaction JNI call. It acquires the dataset read lock so
   * the native call cannot race with {@link Dataset#close()}; keep the raw native method private so
   * no caller can bypass this lock.
   */
  public static CompactionMetrics nativeCommitCompaction(
      Dataset dataset,
      List<RewriteResult> rewriteResults,
      Optional<Long> targetRowsPerFragment,
      Optional<Long> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<Boolean> materializeDeletions,
      Optional<Float> materializeDeletionsThreshold,
      Optional<Long> numThreads,
      Optional<Long> batchSize,
      Optional<Boolean> deferIndexRemap,
      Optional<String> compactionMode,
      Optional<Long> binaryCopyReadBatchBytes,
      Optional<Long> maxSourceFragments,
      Optional<Long> maxSourceRows,
      Optional<Long> maxSourceBytes,
      List<Long> excludedFragmentIds) {
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return commitCompactionNative(
          dataset,
          rewriteResults,
          targetRowsPerFragment,
          maxRowsPerGroup,
          maxBytesPerFile,
          materializeDeletions,
          materializeDeletionsThreshold,
          numThreads,
          batchSize,
          deferIndexRemap,
          compactionMode,
          binaryCopyReadBatchBytes,
          maxSourceFragments,
          maxSourceRows,
          maxSourceBytes,
          excludedFragmentIds);
    }
  }

  private static native CompactionMetrics commitCompactionNative(
      Dataset dataset,
      List<RewriteResult> rewriteResults,
      Optional<Long> targetRowsPerFragment,
      Optional<Long> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<Boolean> materializeDeletions,
      Optional<Float> materializeDeletionsThreshold,
      Optional<Long> numThreads,
      Optional<Long> batchSize,
      Optional<Boolean> deferIndexRemap,
      Optional<String> compactionMode,
      Optional<Long> binaryCopyReadBatchBytes,
      Optional<Long> maxSourceFragments,
      Optional<Long> maxSourceRows,
      Optional<Long> maxSourceBytes,
      List<Long> excludedFragmentIds);

  private static native CompactionPlan nativePlanCompaction(
      Dataset dataset,
      Optional<Long> targetRowsPerFragment,
      Optional<Long> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<Boolean> materializeDeletions,
      Optional<Float> materializeDeletionsThreshold,
      Optional<Long> numThreads,
      Optional<Long> batchSize,
      Optional<Boolean> deferIndexRemap,
      Optional<String> compactionMode,
      Optional<Long> binaryCopyReadBatchBytes,
      Optional<Long> maxSourceFragments,
      Optional<Long> maxSourceRows,
      Optional<Long> maxSourceBytes,
      List<Long> excludedFragmentIds);
}
