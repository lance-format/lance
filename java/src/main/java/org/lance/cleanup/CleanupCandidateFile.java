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

/** A file that cleanup identified as removable. */
public class CleanupCandidateFile {
  private final String path;
  private final CleanupFileKind kind;
  private final boolean unverified;
  private final long sizeBytes;

  public CleanupCandidateFile(String path, String kind, boolean unverified, long sizeBytes) {
    this(path, CleanupFileKind.fromRustString(kind), unverified, sizeBytes);
  }

  public CleanupCandidateFile(
      String path, CleanupFileKind kind, boolean unverified, long sizeBytes) {
    this.path = path;
    this.kind = kind;
    this.unverified = unverified;
    this.sizeBytes = sizeBytes;
  }

  public String getPath() {
    return path;
  }

  public CleanupFileKind getKind() {
    return kind;
  }

  public boolean isUnverified() {
    return unverified;
  }

  public long getSizeBytes() {
    return sizeBytes;
  }
}
