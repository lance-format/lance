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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Write Params for Write Operations of Lance. */
public class WriteParams {

  /** Write Mode. */
  public enum WriteMode {
    CREATE,
    APPEND,
    OVERWRITE
  }

  private final Optional<Integer> maxRowsPerFile;
  private final Optional<Integer> maxRowsPerGroup;
  private final Optional<Long> maxBytesPerFile;
  private final Optional<WriteMode> mode;
  private final Optional<Boolean> enableStableRowIds;
  private final Optional<String> dataStorageVersion;
  private final Optional<Boolean> enableV2ManifestPaths;
  private Map<String, String> storageOptions = new HashMap<>();
  private Map<String, Map<String, String>> baseStoreParams = new HashMap<>();
  private final Optional<List<BasePath>> initialBases;
  private final Optional<List<String>> targetBases;
  private final Optional<Boolean> allowExternalBlobOutsideBases;
  private final Optional<Long> blobPackFileSizeThreshold;

  private WriteParams(
      Optional<Integer> maxRowsPerFile,
      Optional<Integer> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<WriteMode> mode,
      Optional<Boolean> enableStableRowIds,
      Optional<String> dataStorageVersion,
      Optional<Boolean> enableV2ManifestPaths,
      Map<String, String> storageOptions,
      Map<String, Map<String, String>> baseStoreParams,
      Optional<List<BasePath>> initialBases,
      Optional<List<String>> targetBases,
      Optional<Boolean> allowExternalBlobOutsideBases,
      Optional<Long> blobPackFileSizeThreshold) {
    this.maxRowsPerFile = maxRowsPerFile;
    this.maxRowsPerGroup = maxRowsPerGroup;
    this.maxBytesPerFile = maxBytesPerFile;
    this.mode = mode;
    this.enableStableRowIds = enableStableRowIds;
    this.dataStorageVersion = dataStorageVersion;
    this.enableV2ManifestPaths = enableV2ManifestPaths;
    this.storageOptions = storageOptions;
    this.baseStoreParams = baseStoreParams;
    this.initialBases = initialBases;
    this.targetBases = targetBases;
    this.allowExternalBlobOutsideBases = allowExternalBlobOutsideBases;
    this.blobPackFileSizeThreshold = blobPackFileSizeThreshold;
  }

  public Optional<Integer> getMaxRowsPerFile() {
    return maxRowsPerFile;
  }

  public Optional<Integer> getMaxRowsPerGroup() {
    return maxRowsPerGroup;
  }

  public Optional<Long> getMaxBytesPerFile() {
    return maxBytesPerFile;
  }

  /**
   * Get Mode with name.
   *
   * @return mode
   */
  public Optional<String> getMode() {
    return mode.map(Enum::name);
  }

  public Optional<Boolean> getEnableStableRowIds() {
    return enableStableRowIds;
  }

  public Optional<String> getDataStorageVersion() {
    return dataStorageVersion;
  }

  public Optional<Boolean> getEnableV2ManifestPaths() {
    return enableV2ManifestPaths;
  }

  public Map<String, String> getStorageOptions() {
    return storageOptions;
  }

  public Map<String, Map<String, String>> getBaseStoreParams() {
    return baseStoreParams;
  }

  public Optional<List<BasePath>> getInitialBases() {
    return initialBases;
  }

  public Optional<List<String>> getTargetBases() {
    return targetBases;
  }

  /**
   * Get whether external blob URIs outside registered bases are allowed.
   *
   * <p>When true, blob v2 columns can reference external URIs that are not under any registered
   * base path. The URI is stored as an absolute external reference with base_id=0.
   *
   * @return Optional containing the setting, or empty if not set
   */
  public Optional<Boolean> getAllowExternalBlobOutsideBases() {
    return allowExternalBlobOutsideBases;
  }

  /**
   * Get the maximum size in bytes for blob v2 pack (.blob) sidecar files.
   *
   * <p>When a pack file reaches this size, a new one is started. If not set, defaults to 1 GiB.
   *
   * @return Optional containing the max pack file size in bytes, or empty if not set
   */
  public Optional<Long> getBlobPackFileSizeThreshold() {
    return blobPackFileSizeThreshold;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("maxRowsPerFile", maxRowsPerFile.orElse(null))
        .add("maxRowsPerGroup", maxRowsPerGroup.orElse(null))
        .add("maxBytesPerFile", maxBytesPerFile.orElse(null))
        .add("mode", mode.orElse(null))
        .add("dataStorageVersion", dataStorageVersion.orElse(null))
        .toString();
  }

  /** A builder of WriteParams. */
  public static class Builder {
    private Optional<Integer> maxRowsPerFile = Optional.empty();
    private Optional<Integer> maxRowsPerGroup = Optional.empty();
    private Optional<Long> maxBytesPerFile = Optional.empty();
    private Optional<WriteMode> mode = Optional.empty();
    private Optional<Boolean> enableStableRowIds = Optional.empty();
    private Optional<String> dataStorageVersion = Optional.empty();
    private Optional<Boolean> enableV2ManifestPaths;
    private Map<String, String> storageOptions = new HashMap<>();
    private Map<String, Map<String, String>> baseStoreParams = new HashMap<>();
    private Optional<List<BasePath>> initialBases = Optional.empty();
    private Optional<List<String>> targetBases = Optional.empty();
    private Optional<Boolean> allowExternalBlobOutsideBases = Optional.empty();
    private Optional<Long> blobPackFileSizeThreshold = Optional.empty();

    public Builder withMaxRowsPerFile(int maxRowsPerFile) {
      this.maxRowsPerFile = Optional.of(maxRowsPerFile);
      return this;
    }

    public Builder withMaxRowsPerGroup(int maxRowsPerGroup) {
      this.maxRowsPerGroup = Optional.of(maxRowsPerGroup);
      return this;
    }

    public Builder withMaxBytesPerFile(long maxBytesPerFile) {
      this.maxBytesPerFile = Optional.of(maxBytesPerFile);
      return this;
    }

    public Builder withMode(WriteMode mode) {
      this.mode = Optional.of(mode);
      return this;
    }

    public Builder withDataStorageVersion(String dataStorageVersion) {
      this.dataStorageVersion = Optional.of(dataStorageVersion);
      return this;
    }

    /**
     * Set storage options for the write.
     *
     * <p>For writes involving additional registered base paths, a key of the form {@code
     * base_<id>.<key>} applies {@code <key>} only to the base path with that id, overriding the
     * unscoped options that every base inherits. Exact per-base bindings set via {@link
     * #withBaseStoreParams(Map)} take precedence over base-scoped keys.
     *
     * @param storageOptions the storage options
     * @return this builder
     */
    public Builder withStorageOptions(Map<String, String> storageOptions) {
      this.storageOptions = storageOptions;
      return this;
    }

    /**
     * Set runtime-only object store parameters for registered base paths.
     *
     * <p>Entries are keyed by the exact {@link BasePath#getPath()} value persisted in the manifest.
     * Each value is the storage options map used as-is for that base. These params are not
     * persisted in the manifest. If a base has no explicit entry, {@link #withStorageOptions(Map)}
     * remains the fallback.
     *
     * @param baseStoreParams object store parameters keyed by base path URI
     * @return this builder
     */
    public Builder withBaseStoreParams(Map<String, Map<String, String>> baseStoreParams) {
      this.baseStoreParams = baseStoreParams;
      return this;
    }

    public Builder withEnableStableRowIds(boolean enableStableRowIds) {
      this.enableStableRowIds = Optional.of(enableStableRowIds);
      return this;
    }

    public Builder withEnableV2ManifestPaths(boolean enableV2ManifestPaths) {
      this.enableV2ManifestPaths = Optional.of(enableV2ManifestPaths);
      return this;
    }

    public Builder withInitialBases(List<BasePath> initialBases) {
      this.initialBases = Optional.of(initialBases);
      return this;
    }

    public Builder withTargetBases(List<String> targetBases) {
      this.targetBases = Optional.of(targetBases);
      return this;
    }

    /**
     * Allow external blob URIs outside registered bases.
     *
     * <p>When true, blob v2 columns can reference external URIs (e.g. pointing to blob files in
     * another Lance dataset) that are not under any registered base path. The URI is stored as an
     * absolute external reference with base_id=0.
     *
     * @param allow true to allow external blob URIs outside bases
     * @return this builder
     */
    public Builder withAllowExternalBlobOutsideBases(boolean allow) {
      this.allowExternalBlobOutsideBases = Optional.of(allow);
      return this;
    }

    /**
     * Set the maximum size in bytes for blob v2 pack (.blob) sidecar files.
     *
     * <p>When a pack file reaches this size, a new one is started. If not set, defaults to 1 GiB.
     *
     * @param maxBytes maximum pack file size in bytes
     * @return this builder
     */
    public Builder withBlobPackFileSizeThreshold(long maxBytes) {
      this.blobPackFileSizeThreshold = Optional.of(maxBytes);
      return this;
    }

    public WriteParams build() {
      return new WriteParams(
          maxRowsPerFile,
          maxRowsPerGroup,
          maxBytesPerFile,
          mode,
          enableStableRowIds,
          dataStorageVersion,
          enableV2ManifestPaths,
          storageOptions,
          baseStoreParams,
          initialBases,
          targetBases,
          allowExternalBlobOutsideBases,
          blobPackFileSizeThreshold);
    }
  }
}
