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

import org.lance.cleanup.CleanupExplanation;
import org.lance.cleanup.CleanupPolicy;
import org.lance.cleanup.RemovalStats;

import org.apache.arrow.util.Preconditions;

import java.util.Optional;

/**
 * A cleanup operation with separate read-only explain and destructive execute actions.
 *
 * <p>This is not a deletion plan. Calling {@link #execute()} re-evaluates the current dataset and
 * reference state before deleting files.
 */
public class CleanupOperation {
  private final Dataset dataset;
  private final CleanupPolicy policy;
  private Optional<Long> maxCandidateFiles = Optional.empty();

  CleanupOperation(Dataset dataset, CleanupPolicy policy) {
    this.dataset = Preconditions.checkNotNull(dataset, "dataset cannot be null");
    this.policy = Preconditions.checkNotNull(policy, "policy cannot be null");
  }

  /**
   * Set the maximum number of candidate files included in the {@link #explain()} result.
   *
   * <p>Defaults to 1000 if not set. The aggregate {@link RemovalStats} returned by {@link
   * #explain()} still account for all files that would be removed regardless of this limit; only
   * the per-file {@code candidateFiles} list is truncated.
   *
   * @param maxCandidateFiles maximum number of candidate files to include; must be positive
   * @return this operation for chaining
   */
  public CleanupOperation withMaxCandidateFiles(long maxCandidateFiles) {
    Preconditions.checkArgument(
        maxCandidateFiles > 0, "maxCandidateFiles must be positive, got %s", maxCandidateFiles);
    this.maxCandidateFiles = Optional.of(maxCandidateFiles);
    return this;
  }

  /**
   * Explain what cleanup would remove without deleting files.
   *
   * @return cleanup explanation
   */
  public CleanupExplanation explain() {
    return dataset.explainCleanup(policy, maxCandidateFiles);
  }

  /**
   * Execute cleanup, re-evaluating the current dataset and reference state before deleting files.
   *
   * @return removal stats
   */
  public RemovalStats execute() {
    return dataset.executeCleanup(policy);
  }
}
