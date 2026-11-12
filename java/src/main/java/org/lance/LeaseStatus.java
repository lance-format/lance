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

import com.google.common.base.MoreObjects;

import java.util.Objects;
import java.util.Optional;

/** Whether a {@link LeaseKeeper} still holds its lease. */
public final class LeaseStatus {
  private final String lostReason;

  LeaseStatus(String lostReason) {
    this.lostReason = lostReason;
  }

  /**
   * A held lease is listed, or a failed renewal can still be retried before it expires.
   *
   * @return whether the lease is held
   */
  public boolean isHeld() {
    return lostReason == null;
  }

  /**
   * A lost lease expired or was removed, so the keeper stopped renewing it. Publishing a result for
   * it still commits normally, but cleanup and compaction may have removed what the lease retained,
   * in which case the commit fails with a conflict.
   *
   * @return why the lease was lost, or empty while it is held
   */
  public Optional<String> getLostReason() {
    return Optional.ofNullable(lostReason);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    return Objects.equals(lostReason, ((LeaseStatus) o).lostReason);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(lostReason);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("held", isHeld())
        .add("lostReason", lostReason)
        .toString();
  }
}
