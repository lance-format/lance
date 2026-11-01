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

/** Metadata describing the location of a dataset manifest. */
public final class ManifestLocation {
  private final long version;
  private final String path;
  private final long sizeBytes;
  private final ManifestNamingScheme namingScheme;
  private final String eTag;

  ManifestLocation(long version, String path, long sizeBytes, String namingScheme, String eTag) {
    this.version = version;
    this.path = path;
    this.sizeBytes = sizeBytes;
    this.namingScheme = ManifestNamingScheme.valueOf(namingScheme);
    this.eTag = eTag;
  }

  /** Dataset version represented by the manifest. */
  public long getVersion() {
    return version;
  }

  /** Manifest path relative to the object-store namespace or root. */
  public String getPath() {
    return path;
  }

  /** Manifest object size in bytes. */
  public long getSizeBytes() {
    return sizeBytes;
  }

  /** Naming scheme used by the manifest path. */
  public ManifestNamingScheme getNamingScheme() {
    return namingScheme;
  }

  /** Object-store entity tag, when available. */
  public Optional<String> getETag() {
    return Optional.ofNullable(eTag);
  }
}
