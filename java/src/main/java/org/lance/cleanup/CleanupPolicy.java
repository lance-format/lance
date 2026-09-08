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
package org.lance.cleanup;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Cleanup policy for dataset cleanup.
 *
 * <p>All fields are optional. We intentionally do not set default values here to avoid conflicting
 * with Rust-side defaults. Refer to Rust CleanupPolicy for defaults.
 */
public class CleanupPolicy {
  private final Optional<Long> beforeTimestampMillis;
  private final Optional<Long> beforeVersion;
  private final Optional<List<Long>> versions;
  private final Optional<Boolean> deleteUnverified;
  private final Optional<Boolean> errorIfTaggedOldVersions;
  private final Optional<Boolean> cleanReferencedBranches;
  private final Optional<Long> deleteRateLimit;

  private CleanupPolicy(
      Optional<Long> beforeTimestampMillis,
      Optional<Long> beforeVersion,
      Optional<List<Long>> versions,
      Optional<Boolean> deleteUnverified,
      Optional<Boolean> errorIfTaggedOldVersions,
      Optional<Boolean> cleanReferencedBranches,
      Optional<Long> deleteRateLimit) {
    this.beforeTimestampMillis = beforeTimestampMillis;
    this.beforeVersion = beforeVersion;
    this.versions = versions;
    this.deleteUnverified = deleteUnverified;
    this.errorIfTaggedOldVersions = errorIfTaggedOldVersions;
    this.cleanReferencedBranches = cleanReferencedBranches;
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

  public Optional<List<Long>> getVersions() {
    return versions;
  }

  public Optional<Boolean> getDeleteUnverified() {
    return deleteUnverified;
  }

  public Optional<Boolean> getErrorIfTaggedOldVersions() {
    return errorIfTaggedOldVersions;
  }

  public Optional<Boolean> getCleanReferencedBranches() {
    return cleanReferencedBranches;
  }

  public Optional<Long> getDeleteRateLimit() {
    return deleteRateLimit;
  }

  /** Builder for CleanupPolicy. */
  public static class Builder {
    private Optional<Long> beforeTimestampMillis = Optional.empty();
    private Optional<Long> beforeVersion = Optional.empty();
    private Optional<List<Long>> versions = Optional.empty();
    private Optional<Boolean> deleteUnverified = Optional.empty();
    private Optional<Boolean> errorIfTaggedOldVersions = Optional.empty();
    private Optional<Boolean> cleanReferencedBranches = Optional.empty();
    private Optional<Long> deleteRateLimit = Optional.empty();

    private Builder() {}

    /** Set a timestamp threshold in milliseconds since UNIX epoch (UTC). */
    public Builder withBeforeTimestampMillis(long beforeTimestampMillis) {
      this.beforeTimestampMillis = Optional.of(beforeTimestampMillis);
      return this;
    }

    /** Set a version threshold; versions older than this will be cleaned. */
    public Builder withBeforeVersion(long beforeVersion) {
      this.beforeVersion = Optional.of(beforeVersion);
      return this;
    }

    /** Set the exact dataset versions to clean. */
    public Builder withVersions(List<Long> versions) {
      this.versions = Optional.of(Collections.unmodifiableList(new ArrayList<>(versions)));
      return this;
    }

    /** If true, delete unverified data files even if they are recent. */
    public Builder withDeleteUnverified(boolean deleteUnverified) {
      this.deleteUnverified = Optional.of(deleteUnverified);
      return this;
    }

    /** If true, raise error when tagged versions are old and matched by policy. */
    public Builder withErrorIfTaggedOldVersions(boolean errorIfTaggedOldVersions) {
      this.errorIfTaggedOldVersions = Optional.of(errorIfTaggedOldVersions);
      return this;
    }

    /** If true, clean referenced branches before clean the current branch. */
    public Builder withCleanReferencedBranches(boolean cleanReferencedBranches) {
      this.cleanReferencedBranches = Optional.of(cleanReferencedBranches);
      return this;
    }

    /** Set the maximum number of delete operations per second. */
    public Builder withDeleteRateLimit(long deleteRateLimit) {
      this.deleteRateLimit = Optional.of(deleteRateLimit);
      return this;
    }

    public CleanupPolicy build() {
      return new CleanupPolicy(
          beforeTimestampMillis,
          beforeVersion,
          versions,
          deleteUnverified,
          errorIfTaggedOldVersions,
          cleanReferencedBranches,
          deleteRateLimit);
    }
  }
}
