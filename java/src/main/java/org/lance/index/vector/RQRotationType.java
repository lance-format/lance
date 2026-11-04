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
package org.lance.index.vector;

/** Rotation used by a Rabit Quantizer. */
public enum RQRotationType {
  /** Serializable fast rotation, suitable for reuse across independent builds. */
  FAST("fast"),
  /** Dense matrix rotation, available for builds without a serialized model. */
  MATRIX("matrix");

  private final String rustName;

  RQRotationType(String rustName) {
    this.rustName = rustName;
  }

  /** Return the rotation name accepted by the Rust index builder. */
  public String toRustString() {
    return rustName;
  }
}
