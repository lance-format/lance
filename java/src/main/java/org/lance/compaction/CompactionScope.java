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

/** Which kinds of task a compaction plans. */
public enum CompactionScope {
  /** Rewrite the fragments that need it and repack the columns of the others (default). */
  ALL("all"),
  /** Only rewrite fragments. */
  REWRITE_FRAGMENTS("rewrite_fragments"),
  /** Only repack columns. */
  REPACK_COLUMNS("repack_columns");

  private final String value;

  CompactionScope(String value) {
    this.value = value;
  }

  public String getValue() {
    return value;
  }
}
