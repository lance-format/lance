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

import java.time.Instant;
import java.util.Objects;
import java.util.UUID;

/**
 * Renew a retention lease, as recorded in a dataset's transaction history.
 *
 * <p>Read-only: renew leases with {@link org.lance.Dataset#renewRetentionLease}.
 */
public final class RenewLease implements Operation {
  private final UUID leaseId;
  private final Instant expiresAt;

  private RenewLease(UUID leaseId, Instant expiresAt) {
    this.leaseId = leaseId;
    this.expiresAt = expiresAt;
  }

  /** The ID of the renewed lease. */
  public UUID getLeaseId() {
    return leaseId;
  }

  /** The new absolute expiry time. */
  public Instant getExpiresAt() {
    return expiresAt;
  }

  @Override
  public String name() {
    return "RenewLease";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("leaseId", leaseId)
        .add("expiresAt", expiresAt)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    RenewLease that = (RenewLease) o;
    return leaseId.equals(that.leaseId) && expiresAt.equals(that.expiresAt);
  }

  @Override
  public int hashCode() {
    return Objects.hash(leaseId, expiresAt);
  }
}
