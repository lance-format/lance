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

/** What a compaction task does. */
public enum CompactionTaskKind {
  /** Rewrite the task's fragments into new fragments. */
  REWRITE_FRAGMENTS("rewrite_fragments"),
  /** Rewrite some columns of the task's one fragment into new data files, keeping its rows. */
  REPACK_COLUMNS("repack_columns");

  private final String value;

  CompactionTaskKind(String value) {
    this.value = value;
  }

  public String getValue() {
    return value;
  }
}
