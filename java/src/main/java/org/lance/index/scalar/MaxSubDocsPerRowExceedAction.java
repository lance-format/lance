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
package org.lance.index.scalar;

/** Action taken when a JSON row exceeds the configured flattened sub-document limit. */
public enum MaxSubDocsPerRowExceedAction {
  /** Abort index ingestion with an error. */
  FAIL("fail"),
  /** Omit the source row from the index and continue ingestion. */
  SKIP_ROW("skip_row");

  private final String rustValue;

  MaxSubDocsPerRowExceedAction(String rustValue) {
    this.rustValue = rustValue;
  }

  String toRustString() {
    return rustValue;
  }
}
