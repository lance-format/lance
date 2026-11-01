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

import java.util.Optional;

/** Version metadata for the library that wrote a dataset manifest. */
public final class WriterVersion {
  private final String library;
  private final String version;
  private final String prerelease;
  private final String buildMetadata;

  WriterVersion(String library, String version, String prerelease, String buildMetadata) {
    this.library = library;
    this.version = version;
    this.prerelease = prerelease;
    this.buildMetadata = buildMetadata;
  }

  /** Name of the writer library, such as {@code lance}. */
  public String getLibrary() {
    return library;
  }

  /**
   * Version string reported by the writer library.
   *
   * <p>This value is opaque because writer libraries are not required to use semantic versioning.
   * When a writer does use semantic versioning, newer writers store the core version here and
   * expose prerelease and build metadata separately.
   */
  public String getVersion() {
    return version;
  }

  /** Optional semantic-version prerelease component, when supplied by the writer. */
  public Optional<String> getPrerelease() {
    return Optional.ofNullable(prerelease);
  }

  /** Optional semantic-version build metadata component, when supplied by the writer. */
  public Optional<String> getBuildMetadata() {
    return Optional.ofNullable(buildMetadata);
  }
}
