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
package org.lance.ipc;

/**
 * Controls the speed / accuracy tradeoff for approximate vector search.
 *
 * <p>This setting currently affects RQ-quantized vector indexes (such as IVF_RQ) and HNSW sub-index
 * searches that exclude rows (for example through a prefilter or deletions), where {@code FAST}
 * enables the ACORN traversal. Other index types ignore this setting.
 */
public enum ApproxMode {
  /** Prefer lower query latency, which can reduce recall. */
  FAST("fast"),

  /** Use the index's default approximation behavior. */
  NORMAL("normal"),

  /** Prefer more accurate approximate scoring when supported by the RQ index. */
  ACCURATE("accurate");

  private final String value;

  ApproxMode(String value) {
    this.value = value;
  }

  /** Returns the lowercase value passed across the JNI boundary. */
  public String toRustString() {
    return value;
  }
}
