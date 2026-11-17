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
package org.lance.expire;

/** Statistics returned by expiring dataset versions. */
public class ExpireVersionsStats {
  private final long versionsRemoved;
  private final long versionsRetained;
  private final long bytesRemoved;
  private final long failedDeletes;

  public ExpireVersionsStats(
      long versionsRemoved, long versionsRetained, long bytesRemoved, long failedDeletes) {
    this.versionsRemoved = versionsRemoved;
    this.versionsRetained = versionsRetained;
    this.bytesRemoved = bytesRemoved;
    this.failedDeletes = failedDeletes;
  }

  /** Manifests deleted. */
  public long getVersionsRemoved() {
    return versionsRemoved;
  }

  /** Manifests left in place, for any reason. */
  public long getVersionsRetained() {
    return versionsRetained;
  }

  /** Bytes of manifest removed, as reported by the listing. */
  public long getBytesRemoved() {
    return bytesRemoved;
  }

  /**
   * Manifests this run tried and failed to delete.
   *
   * <p>Expiring is best effort: a failure is counted here and the run continues, so a non-zero
   * value means the run completed without removing everything it selected. Those versions are still
   * expirable and a later run retries them. Callers that treat expiry as all-or-nothing should
   * check this rather than rely on the call throwing.
   */
  public long getFailedDeletes() {
    return failedDeletes;
  }
}
