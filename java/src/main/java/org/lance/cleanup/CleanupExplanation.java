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

import java.util.List;

/** Read-only explanation of what cleanup would remove. */
public class CleanupExplanation {
  private final long readVersion;
  private final RemovalStats stats;
  private final List<CleanupCandidateFile> candidateFiles;
  private final boolean candidateFilesTruncated;
  private final long candidateFileLimit;
  private final List<CleanupReferencedBranch> referencedBranches;
  private final List<String> warnings;

  public CleanupExplanation(
      long readVersion,
      RemovalStats stats,
      List<CleanupCandidateFile> candidateFiles,
      boolean candidateFilesTruncated,
      long candidateFileLimit,
      List<CleanupReferencedBranch> referencedBranches,
      List<String> warnings) {
    this.readVersion = readVersion;
    this.stats = stats;
    this.candidateFiles = candidateFiles;
    this.candidateFilesTruncated = candidateFilesTruncated;
    this.candidateFileLimit = candidateFileLimit;
    this.referencedBranches = referencedBranches;
    this.warnings = warnings;
  }

  public long getReadVersion() {
    return readVersion;
  }

  public RemovalStats getStats() {
    return stats;
  }

  public List<CleanupCandidateFile> getCandidateFiles() {
    return candidateFiles;
  }

  public boolean isCandidateFilesTruncated() {
    return candidateFilesTruncated;
  }

  public long getCandidateFileLimit() {
    return candidateFileLimit;
  }

  public List<CleanupReferencedBranch> getReferencedBranches() {
    return referencedBranches;
  }

  public List<String> getWarnings() {
    return warnings;
  }
}
