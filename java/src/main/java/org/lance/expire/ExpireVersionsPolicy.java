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

import java.time.Duration;
import java.util.Optional;

/**
 * Policy for expiring dataset versions.
 *
 * <p>Expiring removes manifests and nothing else. Data files that only an expired version
 * referenced are left behind; the next cleanup reclaims them.
 *
 * <p>All fields are optional. Defaults deliberately live on the Rust side rather than here, so the
 * two cannot disagree. With neither a timestamp nor a version bound, nothing is expired.
 */
public class ExpireVersionsPolicy {
  private final Optional<Long> beforeTimestampMillis;
  private final Optional<Long> beforeVersion;
  private final Optional<Long> keepOnePerMicros;
  private final Optional<Boolean> errorIfTaggedOldVersions;
  private final Optional<Long> deleteRateLimit;

  private ExpireVersionsPolicy(
      Optional<Long> beforeTimestampMillis,
      Optional<Long> beforeVersion,
      Optional<Long> keepOnePerMicros,
      Optional<Boolean> errorIfTaggedOldVersions,
      Optional<Long> deleteRateLimit) {
    this.beforeTimestampMillis = beforeTimestampMillis;
    this.beforeVersion = beforeVersion;
    this.keepOnePerMicros = keepOnePerMicros;
    this.errorIfTaggedOldVersions = errorIfTaggedOldVersions;
    this.deleteRateLimit = deleteRateLimit;
  }

  public static Builder builder() {
    return new Builder();
  }

  public Optional<Long> getBeforeTimestampMillis() {
    return beforeTimestampMillis;
  }

  public Optional<Long> getBeforeVersion() {
    return beforeVersion;
  }

  public Optional<Long> getKeepOnePerMicros() {
    return keepOnePerMicros;
  }

  public Optional<Boolean> getErrorIfTaggedOldVersions() {
    return errorIfTaggedOldVersions;
  }

  public Optional<Long> getDeleteRateLimit() {
    return deleteRateLimit;
  }

  public static class Builder {
    private Optional<Long> beforeTimestampMillis = Optional.empty();
    private Optional<Long> beforeVersion = Optional.empty();
    private Optional<Long> keepOnePerMicros = Optional.empty();
    private Optional<Boolean> errorIfTaggedOldVersions = Optional.empty();
    private Optional<Long> deleteRateLimit = Optional.empty();

    /**
     * Expire versions whose manifest was written before this time.
     *
     * <p>This is the manifest object's write time, not the commit timestamp recorded inside it. Use
     * {@link #withBeforeVersion} when an exact boundary matters.
     */
    public Builder withBeforeTimestampMillis(long beforeTimestampMillis) {
      this.beforeTimestampMillis = Optional.of(beforeTimestampMillis);
      return this;
    }

    /** Expire versions numbered below this. Exact, and needs no timestamps. */
    public Builder withBeforeVersion(long beforeVersion) {
      this.beforeVersion = Optional.of(beforeVersion);
      return this;
    }

    /**
     * Instead of expiring every version past the cutoff, keep the newest one in each bucket of this
     * width. {@code Duration.ofHours(1)} keeps one version per hour.
     *
     * <p>Carried as microseconds, so the width reaches the bucketing unrounded rather than being
     * widened into deleting more history than was asked for.
     *
     * <p><b>Thinning carries a rollback risk on an actively committing table.</b> Resolving the
     * latest version starts at the version hint and probes upward, stopping at the first version
     * that is missing. Expiry refuses to remove anything at or above the hint, but that floor is
     * read once and hint writes are unconditional, so a commit that started earlier can publish a
     * lower hint afterwards. A gap above that lowered hint hides every version above it, and the
     * table reads as an older state while the newer manifests are still present. The one-hour
     * minimum below bounds the exposure — it keeps deletions away from the seconds-wide window
     * where that can happen — but does not remove it. Use thinning to repair a table that has
     * accumulated far more versions than it can carry, not as a default left switched on.
     *
     * @throws IllegalArgumentException if the width is null, negative, finer than a microsecond, or
     *     shorter than one hour
     */
    public Builder withKeepOnePer(Duration keepOnePer) {
      if (keepOnePer == null) {
        throw new IllegalArgumentException("keepOnePer cannot be null");
      }
      if (keepOnePer.isNegative()) {
        throw new IllegalArgumentException("keepOnePer cannot be negative: " + keepOnePer);
      }
      if (keepOnePer.compareTo(Duration.ofHours(1)) < 0) {
        throw new IllegalArgumentException(
            "keepOnePer must be at least one hour, got "
                + keepOnePer
                + "; thinning more finely puts deletions next to the window where a concurrent"
                + " commit can lower the version hint");
      }
      if (keepOnePer.getNano() % 1_000 != 0) {
        // Truncating would widen the bucket and delete more than asked, so refuse.
        throw new IllegalArgumentException(
            "keepOnePer must be a whole number of microseconds, got " + keepOnePer);
      }
      this.keepOnePerMicros =
          Optional.of(keepOnePer.getSeconds() * 1_000_000L + keepOnePer.getNano() / 1_000L);
      return this;
    }

    /** Fail instead of silently keeping a tagged version the policy would expire. */
    public Builder withErrorIfTaggedOldVersions(boolean errorIfTaggedOldVersions) {
      this.errorIfTaggedOldVersions = Optional.of(errorIfTaggedOldVersions);
      return this;
    }

    /** Maximum delete requests per second. One request is one manifest. */
    public Builder withDeleteRateLimit(long deleteRateLimit) {
      this.deleteRateLimit = Optional.of(deleteRateLimit);
      return this;
    }

    public ExpireVersionsPolicy build() {
      return new ExpireVersionsPolicy(
          beforeTimestampMillis,
          beforeVersion,
          keepOnePerMicros,
          errorIfTaggedOldVersions,
          deleteRateLimit);
    }
  }
}
