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
package org.lance.delta;

import org.lance.Dataset;
import org.lance.JniLoader;
import org.lance.LockManager;

import java.util.Optional;

/**
 * Builder for creating a {@link DatasetDelta} to explore changes between versions.
 *
 * <ul>
 *   <li>Use comparedAgainstVersion to compare current dataset version.
 *   <li>Or specify an explicit range with beginVersion and endVersion.
 *   <li>These modes are mutually exclusive.
 * </ul>
 */
public class DatasetDeltaBuilder {
  static {
    JniLoader.ensureLoaded();
  }

  private final Dataset dataset;
  private Optional<Long> comparedAgainst = Optional.empty();
  private Optional<Long> beginVersion = Optional.empty();
  private Optional<Long> endVersion = Optional.empty();

  public DatasetDeltaBuilder(Dataset dataset) {
    this.dataset = dataset;
  }

  /**
   * Compare the current dataset version against the specified version. The delta will automatically
   * order the versions so that `begin_version` is less than `end_version`. Cannot be used together
   * with explicit `with_begin_version` and `with_end_version`.
   */
  public DatasetDeltaBuilder comparedAgainstVersion(long version) {
    this.comparedAgainst = Optional.of(version);
    return this;
  }

  /**
   * Set the beginning version for the delta (exclusive). Must be used together with
   * `with_end_version`.
   */
  public DatasetDeltaBuilder withBeginVersion(long version) {
    this.beginVersion = Optional.of(version);
    return this;
  }

  /**
   * Set the ending version for the delta (inclusive). Must be used together with
   * `with_begin_version`. Cannot be used together with `compared_against_version`.
   */
  public DatasetDeltaBuilder withEndVersion(long version) {
    this.endVersion = Optional.of(version);
    return this;
  }

  /** Build the DatasetDelta after validating builder state. */
  public DatasetDelta build() {
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return nativeBuild(dataset, comparedAgainst, beginVersion, endVersion);
    }
  }

  private static native DatasetDelta nativeBuild(
      Dataset dataset,
      Optional<Long> comparedAgainst,
      Optional<Long> beginVersion,
      Optional<Long> endVersion);
}
