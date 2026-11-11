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

import java.time.Duration;
import java.util.Optional;

/**
 * Options for {@link Dataset#acquireRetentionLease(LeaseOptions)}.
 *
 * <p>Example:
 *
 * <pre>{@code
 * LeaseOptions options = LeaseOptions.builder().ttl(Duration.ofHours(1)).label("backfill").build();
 * }</pre>
 */
public class LeaseOptions {
  private final Optional<Duration> ttl;
  private final Optional<String> label;

  private LeaseOptions(Builder builder) {
    this.ttl = builder.ttl;
    this.label = builder.label;
  }

  public static Builder builder() {
    return new Builder();
  }

  /**
   * @return how long the lease lasts unless renewed, or empty for the default of 30 minutes
   */
  public Optional<Duration> getTtl() {
    return ttl;
  }

  /**
   * @return human-readable context stored with the lease, if any
   */
  public Optional<String> getLabel() {
    return label;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("ttl", ttl.orElse(null))
        .add("label", label.orElse(null))
        .toString();
  }

  /** Builder for {@link LeaseOptions}. */
  public static class Builder {
    private Optional<Duration> ttl = Optional.empty();
    private Optional<String> label = Optional.empty();

    /**
     * Set how long the lease lasts unless renewed, from the time of acquisition. Must be positive;
     * Lance rejects other values when the lease is acquired. Defaults to 30 minutes.
     *
     * @param ttl the lease duration
     * @return this builder
     */
    public Builder ttl(Duration ttl) {
      this.ttl = Optional.of(ttl);
      return this;
    }

    /**
     * Set human-readable context stored with the lease. It has no retention meaning; an empty
     * label is stored as no label.
     *
     * @param label the label
     * @return this builder
     */
    public Builder label(String label) {
      this.label = Optional.of(label);
      return this;
    }

    public LeaseOptions build() {
      return new LeaseOptions(this);
    }
  }
}
