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

/** A branch that references the current branch lineage. */
public class CleanupReferencedBranch {
  private final String name;
  private final long referencedVersion;
  private final boolean cleanupCandidate;

  public CleanupReferencedBranch(String name, long referencedVersion, boolean cleanupCandidate) {
    this.name = name;
    this.referencedVersion = referencedVersion;
    this.cleanupCandidate = cleanupCandidate;
  }

  public String getName() {
    return name;
  }

  public long getReferencedVersion() {
    return referencedVersion;
  }

  public boolean isCleanupCandidate() {
    return cleanupCandidate;
  }
}
