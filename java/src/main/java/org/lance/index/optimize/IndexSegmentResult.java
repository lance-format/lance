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

/** An index segment produced by a task or a merge; see {@link OptimizeIndices}. */
public class IndexSegmentResult implements Serializable {
  /** The result as the native side serializes it; the only part it reads back. */
  private final String json;

  private final String indexName;
  private final int group;

  IndexSegmentResult(String json, String indexName, int group) {
    this.json = json;
    this.indexName = indexName;
    this.group = group;
  }

  String getJson() {
    return json;
  }

  /** The index the segment belongs to. */
  public String getIndexName() {
    return indexName;
  }

  /** The merge group the segment belongs to. */
  public int getGroup() {
    return group;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("indexName", indexName)
        .add("group", group)
        .toString();
  }
}
