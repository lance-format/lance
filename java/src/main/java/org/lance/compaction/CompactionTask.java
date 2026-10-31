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
import org.lance.LockManager;

import com.google.common.base.MoreObjects;

import java.io.Serializable;
import java.util.List;
import java.util.Optional;

/** The compaction task which can be sent across network and executed individually. */
public class CompactionTask implements Serializable {
  private final TaskData taskData;
  private final long readVersion;
  private final CompactionOptions compactionOptions;

  public CompactionTask(TaskData taskData, long readVersion, CompactionOptions compactionOptions) {
    this.taskData = taskData;
    this.readVersion = readVersion;
    this.compactionOptions = compactionOptions;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("taskData", taskData.getFragments())
        .add("readVersion", readVersion)
        .add("compactionOptions", compactionOptions)
        .toString();
  }

  public RewriteResult execute(Dataset dataset) {
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return nativeExecute(
          dataset,
          taskData,
          readVersion,
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
          compactionOptions.getExcludedFragmentIds(),
          compactionOptions.getDataStorageVersion());
    }
  }

  private native RewriteResult nativeExecute(
      Dataset dataset,
      TaskData taskData,
      long readVersion,
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
      List<Long> excludedFragmentIds,
      Optional<String> dataStorageVersion);

  public CompactionOptions getCompactionOptions() {
    return compactionOptions;
  }

  public long getReadVersion() {
    return readVersion;
  }

  public TaskData getTaskData() {
    return taskData;
  }
}
