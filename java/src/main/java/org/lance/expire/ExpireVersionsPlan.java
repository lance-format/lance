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

import java.util.Collections;
import java.util.List;

/**
 * What expiring versions would remove, without removing it.
 *
 * <p>Returned by a dry run. Nothing is deleted to produce this.
 */
public class ExpireVersionsPlan {
  private final List<Long> versions;
  private final ExpireVersionsStats stats;
  private final List<Long> taggedButKept;

  public ExpireVersionsPlan(
      List<Long> versions, ExpireVersionsStats stats, List<Long> taggedButKept) {
    this.versions = versions == null ? Collections.emptyList() : versions;
    this.stats = stats;
    this.taggedButKept = taggedButKept == null ? Collections.emptyList() : taggedButKept;
  }

  /** The versions that would be deleted, ascending. */
  public List<Long> getVersions() {
    return Collections.unmodifiableList(versions);
  }

  /** Stats as they would be after the run. */
  public ExpireVersionsStats getStats() {
    return stats;
  }

  /**
   * Tagged versions the policy would have expired but will keep.
   *
   * <p>Only populated when {@code errorIfTaggedOldVersions} is false; otherwise the run fails
   * instead of keeping them silently.
   */
  public List<Long> getTaggedButKept() {
    return Collections.unmodifiableList(taggedButKept);
  }
}
