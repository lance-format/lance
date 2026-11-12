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

import java.io.Serializable;
import java.time.Instant;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/**
 * A retention lease: it keeps {@link #getVersion()} and every later version of the dataset until
 * the holder commits a transaction for it (see {@link Transaction.Builder#forLease}), releases it
 * with {@link Dataset#releaseRetentionLease}, or lets it expire.
 *
 * <p>Leases are obtained from {@link Dataset#acquireRetentionLease} and are {@link Serializable},
 * so the holder can ship them to distributed workers. Lance validates the record whenever a lease
 * is passed back to it.
 */
public final class RetentionLease implements Serializable {
  private static final long serialVersionUID = 1L;

  private final UUID id;
  private final long version;
  private final Instant createdAt;
  private final Instant expiresAt;
  private final String label;

  // Constructed from native code only, so a lease always comes from Lance.
  private RetentionLease(
      UUID id, long version, Instant createdAt, Instant expiresAt, String label) {
    this.id = id;
    this.version = version;
    this.createdAt = createdAt;
    this.expiresAt = expiresAt;
    this.label = label;
  }

  /**
   * @return the UUID a transaction must carry to publish this lease's result
   */
  public UUID getId() {
    return id;
  }

  /**
   * @return the leased version: the version created by the acquisition commit
   */
  public long getVersion() {
    return version;
  }

  /**
   * @return when the lease was acquired, by the acquirer's clock
   */
  public Instant getCreatedAt() {
    return createdAt;
  }

  /**
   * An expired lease still retains history until a later commit removes it from the list.
   *
   * @return when the lease expires, by the clock of the acquirer or last renewer
   */
  public Instant getExpiresAt() {
    return expiresAt;
  }

  /**
   * @return human-readable context stored with the lease, if any
   */
  public Optional<String> getLabel() {
    return Optional.ofNullable(label);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    RetentionLease that = (RetentionLease) o;
    return version == that.version
        && id.equals(that.id)
        && createdAt.equals(that.createdAt)
        && expiresAt.equals(that.expiresAt)
        && Objects.equals(label, that.label);
  }

  @Override
  public int hashCode() {
    return Objects.hash(id, version, createdAt, expiresAt, label);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("id", id)
        .add("version", version)
        .add("createdAt", createdAt)
        .add("expiresAt", expiresAt)
        .add("label", label)
        .toString();
  }
}
