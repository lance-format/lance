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

/**
 * Output data file version. Exact selectors name a specific format version; release selectors are
 * resolved by the Rust engine. An operation cannot cross the dataset's V1/V2 boundary.
 *
 * <pre>{@code
 * CompactionOptions.builder().withDataStorageVersion(DataStorageVersion.V2_2).build();
 * }</pre>
 */
public enum DataStorageVersion {
  /** Legacy V1 format, usable only with legacy datasets. */
  LEGACY("0.1"),
  /** Exact V2.0 format. */
  V2_0("2.0"),
  /** Exact V2.1 format. */
  V2_1("2.1"),
  /** Exact V2.2 format. */
  V2_2("2.2"),
  /** Exact V2.3 format, currently unstable and intended for experimentation. */
  V2_3("2.3"),
  /** The stable format selected by the executing engine release. */
  STABLE("stable"),
  /** The next format selected by the executing engine release. */
  NEXT("next");

  private final String rustString;

  DataStorageVersion(String rustString) {
    this.rustString = rustString;
  }

  /** Returns the selector understood by the Rust API. */
  public String toRustString() {
    return rustString;
  }

  /**
   * Decodes a canonical selector returned by Rust or stored in serialized compaction options.
   *
   * @throws IllegalArgumentException if the selector is unknown
   */
  public static DataStorageVersion fromRustString(String value) {
    for (DataStorageVersion version : values()) {
      if (version.rustString.equals(value)) {
        return version;
      }
    }
    throw new IllegalArgumentException("Unknown data storage version: " + value);
  }
}
