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
package org.lance;

/** The unit treated as one full-text-search document. */
public enum DocumentGranularity {
  /** All text selected from one dataset row belongs to one document. */
  ROW("row"),

  /** Each element of the deepest list on the field path is one document. */
  LIST_ELEMENT("list_element");

  private final String rustString;

  DocumentGranularity(String rustString) {
    this.rustString = rustString;
  }

  /** Return the stable value understood by the Rust API and serialized index parameters. */
  public String toRustString() {
    return rustString;
  }
}
