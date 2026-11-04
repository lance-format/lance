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

import com.google.common.base.MoreObjects;

import java.io.Serializable;
import java.util.List;

/** A plan to optimize the indices of a dataset; see {@link OptimizeIndices}. */
public class OptimizeIndicesPlan implements Serializable {
  private final String planId;
  private final long readVersion;
  private final List<OptimizeIndexTask> tasks;
  private final List<OptimizeIndexMerge> merges;

  public OptimizeIndicesPlan(
      String planId,
      long readVersion,
      List<OptimizeIndexTask> tasks,
      List<OptimizeIndexMerge> merges) {
    this.planId = planId;
    this.readVersion = readVersion;
    this.tasks = tasks;
    this.merges = merges;
  }

  /** Identifies the plan. A merge takes only the results of its own plan. */
  public String getPlanId() {
    return planId;
  }

  /** The dataset version the plan was made from. */
  public long getReadVersion() {
    return readVersion;
  }

  /** The build units. Each writes one new segment. */
  public List<OptimizeIndexTask> getTasks() {
    return tasks;
  }

  /** The merge units. Each yields one segment the commit publishes. */
  public List<OptimizeIndexMerge> getMerges() {
    return merges;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("planId", planId)
        .add("readVersion", readVersion)
        .add("tasks", tasks.size())
        .add("merges", merges.size())
        .toString();
  }
}
