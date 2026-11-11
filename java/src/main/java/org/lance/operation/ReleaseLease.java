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
package org.lance.operation;

import com.google.common.base.MoreObjects;

import java.util.Objects;
import java.util.UUID;

/**
 * Release a retention lease, as recorded in a dataset's transaction history.
 *
 * <p>Read-only: release leases with {@link org.lance.Dataset#releaseRetentionLease}.
 */
public final class ReleaseLease implements Operation {
  private final UUID leaseId;

  private ReleaseLease(UUID leaseId) {
    this.leaseId = leaseId;
  }

  /** The ID of the released lease. */
  public UUID getLeaseId() {
    return leaseId;
  }

  @Override
  public String name() {
    return "ReleaseLease";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("leaseId", leaseId).toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    return leaseId.equals(((ReleaseLease) o).leaseId);
  }

  @Override
  public int hashCode() {
    return Objects.hash(leaseId);
  }
}
