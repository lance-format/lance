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
package org.lance.cleanup;

/** Kind of file identified by cleanup. */
public enum CleanupFileKind {
  MANIFEST,
  DATA,
  TRANSACTION,
  INDEX,
  DELETION,
  TEMPORARY_MANIFEST;

  public static CleanupFileKind fromRustString(String value) {
    switch (value) {
      case "manifest":
        return MANIFEST;
      case "data":
        return DATA;
      case "transaction":
        return TRANSACTION;
      case "index":
        return INDEX;
      case "deletion":
        return DELETION;
      case "temporary_manifest":
        return TEMPORARY_MANIFEST;
      default:
        throw new IllegalArgumentException("Unknown cleanup file kind: " + value);
    }
  }
}
