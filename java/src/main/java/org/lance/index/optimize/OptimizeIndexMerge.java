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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;

import java.io.Serializable;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Combine the task results of one group into the segment to commit; see {@link OptimizeIndices}.
 */
public class OptimizeIndexMerge implements Serializable {
  static {
    JniLoader.ensureLoaded();
  }

  /** The merge as the native side serializes it; the only part it reads back. */
  private final String json;

  private final int group;

  OptimizeIndexMerge(String json, int group) {
    this.json = json;
    this.group = group;
  }

  /**
   * Merge this group's inputs: the committed segments it absorbs plus the task results that carry
   * its group. Results of other groups are ignored; a result of another plan is an error. A lone
   * task result passes through unchanged.
   *
   * @param dataset a dataset opened on the planned table, at any version
   * @param taskResults results of the plan's tasks; at least those of this group
   * @return the segment to commit, or empty when the group has nothing to commit
   */
  public Optional<IndexSegmentResult> execute(
      Dataset dataset, List<IndexSegmentResult> taskResults) {
    Preconditions.checkNotNull(dataset);
    Preconditions.checkNotNull(taskResults);
    List<String> resultJsons =
        taskResults.stream().map(IndexSegmentResult::getJson).collect(Collectors.toList());
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return Optional.ofNullable(nativeExecute(dataset, json, resultJsons));
    }
  }

  private native IndexSegmentResult nativeExecute(
      Dataset dataset, String json, List<String> taskResults);

  /** The key task results carry to reach this merge. */
  public int getGroup() {
    return group;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("group", group).toString();
  }
}
