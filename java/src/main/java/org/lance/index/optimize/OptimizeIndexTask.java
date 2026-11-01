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

/**
 * Build one new, uncommitted index segment over a group of fragments; see {@link OptimizeIndices}.
 */
public class OptimizeIndexTask implements Serializable {
  static {
    JniLoader.ensureLoaded();
  }

  /** The task as the native side serializes it; the only part it reads back. */
  private final String json;

  private final String indexName;
  private final int group;
  private final List<Integer> fragments;

  OptimizeIndexTask(String json, String indexName, int group, List<Integer> fragments) {
    this.json = json;
    this.indexName = indexName;
    this.group = group;
    this.fragments = fragments;
  }

  /**
   * Build this task's segment. The dataset may be at any version; the task reads the version it was
   * planned from.
   *
   * @param dataset a dataset opened on the planned table
   * @return the segment, or empty when there was nothing to do after all
   */
  public Optional<IndexSegmentResult> execute(Dataset dataset) {
    Preconditions.checkNotNull(dataset);
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return Optional.ofNullable(nativeExecute(dataset, json));
    }
  }

  private native IndexSegmentResult nativeExecute(Dataset dataset, String json);

  /** The index the task builds a segment of. */
  public String getIndexName() {
    return indexName;
  }

  /** The merge this task's result feeds, by {@link OptimizeIndexMerge#getGroup()}. */
  public int getGroup() {
    return group;
  }

  /** The fragments to index, by id in the dataset at the plan's read version. */
  public List<Integer> getFragments() {
    return fragments;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("indexName", indexName)
        .add("group", group)
        .add("fragments", fragments)
        .toString();
  }
}
