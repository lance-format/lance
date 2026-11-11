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
import java.util.Optional;
import java.util.UUID;

/**
 * Acquire a retention lease, as recorded in a dataset's transaction history.
 *
 * <p>Read-only: acquire leases with {@link org.lance.Dataset#acquireRetentionLease}.
 */
public final class AcquireLease implements Operation {
  private final UUID leaseId;
  private final Instant createdAt;
  private final Instant expiresAt;
  private final Optional<String> label;

  private AcquireLease(UUID leaseId, Instant createdAt, Instant expiresAt, String label) {
    this.leaseId = leaseId;
    this.createdAt = createdAt;
    this.expiresAt = expiresAt;
    this.label = Optional.ofNullable(label);
  }

  /** The UUID of the transaction the holder intends to publish. */
  public UUID getLeaseId() {
    return leaseId;
  }

  /** When the lease was acquired, by the acquirer's clock. */
  public Instant getCreatedAt() {
    return createdAt;
  }

  /** The absolute time the lease expires. */
  public Instant getExpiresAt() {
    return expiresAt;
  }

  /** Human-readable context stored with the lease. */
  public Optional<String> getLabel() {
    return label;
  }

  @Override
  public String name() {
    return "AcquireLease";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("leaseId", leaseId)
        .add("createdAt", createdAt)
        .add("expiresAt", expiresAt)
        .add("label", label)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    AcquireLease that = (AcquireLease) o;
    return leaseId.equals(that.leaseId)
        && createdAt.equals(that.createdAt)
        && expiresAt.equals(that.expiresAt)
        && label.equals(that.label);
  }

  @Override
  public int hashCode() {
    return Objects.hash(leaseId, createdAt, expiresAt, label);
  }
}
