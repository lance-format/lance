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

import org.apache.arrow.util.Preconditions;

import java.io.Closeable;
import java.util.Collections;
import java.util.Map;

/**
 * A user session that holds runtime state for Lance datasets.
 *
 * <p>A session can be shared between multiple datasets to share caches (index cache and metadata
 * cache), increasing cache hit rates and reducing memory usage.
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * // Create a shared session with default cache sizes
 * Session session = Session.builder().build();
 *
 * // Create a session with custom cache sizes
 * Session customSession = Session.builder()
 *     .indexCacheSizeBytes(2L * 1024 * 1024 * 1024)  // 2 GiB
 *     .metadataCacheSizeBytes(512L * 1024 * 1024)    // 512 MiB
 *     .build();
 *
 * // Select registered cache backends using URIs or structured configuration
 * Session backendSession = Session.builder()
 *     .indexCacheBackend("moka://?capacity=1073741824")
 *     .metadataCacheBackend(CacheBackendConfig.builder("moka")
 *         .option("capacity", "268435456")
 *         .build())
 *     .build();
 *
 * // Open multiple datasets with shared session
 * Dataset ds1 = Dataset.open()
 *     .uri("s3://bucket/table1.lance")
 *     .session(session)
 *     .build();
 *
 * Dataset ds2 = Dataset.open()
 *     .uri("s3://bucket/table2.lance")
 *     .session(session)
 *     .build();
 *
 * // Verify session sharing
 * assert ds1.session().isSameAs(ds2.session());
 *
 * // Clean up - session must be closed separately
 * ds1.close();
 * ds2.close();
 * session.close();
 * }</pre>
 */
public class Session implements Closeable {
  static {
    JniLoader.ensureLoaded();
  }

  /** Default index cache size: 6 GiB */
  public static final long DEFAULT_INDEX_CACHE_SIZE_BYTES = 6L * 1024 * 1024 * 1024;

  /** Default metadata cache size: 1 GiB */
  public static final long DEFAULT_METADATA_CACHE_SIZE_BYTES = 1L * 1024 * 1024 * 1024;

  private long nativeSessionHandle;

  private Session(long handle) {
    this.nativeSessionHandle = handle;
  }

  /**
   * Creates a new builder for configuring a Session.
   *
   * @return a new Builder instance
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Creates a new session with default cache sizes.
   *
   * @return a new Session instance
   * @deprecated Use {@link #builder()} instead
   */
  @Deprecated
  public static Session create() {
    return builder().build();
  }

  /**
   * Creates a new session with custom cache sizes.
   *
   * @param indexCacheSizeBytes the size of the index cache in bytes
   * @param metadataCacheSizeBytes the size of the metadata cache in bytes
   * @return a new Session instance
   * @deprecated Use {@link #builder()} instead
   */
  @Deprecated
  public static Session create(long indexCacheSizeBytes, long metadataCacheSizeBytes) {
    return builder()
        .indexCacheSizeBytes(indexCacheSizeBytes)
        .metadataCacheSizeBytes(metadataCacheSizeBytes)
        .build();
  }

  /** Builder for creating Session instances with custom configuration. */
  public static class Builder {
    private Long indexCacheSizeBytes;
    private Long metadataCacheSizeBytes;
    private String indexCacheBackendUri;
    private CacheBackendConfig indexCacheBackendConfig;
    private String metadataCacheBackendUri;
    private CacheBackendConfig metadataCacheBackendConfig;

    private Builder() {}

    /**
     * Sets the size of the index cache in bytes.
     *
     * @param indexCacheSizeBytes the size of the index cache in bytes (must be non-negative)
     * @return this builder instance
     */
    public Builder indexCacheSizeBytes(long indexCacheSizeBytes) {
      Preconditions.checkArgument(indexCacheSizeBytes >= 0, "indexCacheSizeBytes must be >= 0");
      this.indexCacheSizeBytes = indexCacheSizeBytes;
      return this;
    }

    /**
     * Selects a registered index cache backend using a backend URI.
     *
     * <p>For example, {@code moka://?capacity=1048576}. This option is mutually exclusive with
     * {@link #indexCacheSizeBytes(long)}.
     *
     * @param backendUri backend URI whose scheme identifies the registered backend
     * @return this builder instance
     */
    public Builder indexCacheBackend(String backendUri) {
      Preconditions.checkNotNull(backendUri, "backendUri must not be null");
      this.indexCacheBackendUri = backendUri;
      this.indexCacheBackendConfig = null;
      return this;
    }

    /**
     * Selects a registered index cache backend using structured configuration.
     *
     * <p>This option is mutually exclusive with {@link #indexCacheSizeBytes(long)}.
     *
     * @param backendConfig backend kind and backend-specific options
     * @return this builder instance
     */
    public Builder indexCacheBackend(CacheBackendConfig backendConfig) {
      Preconditions.checkNotNull(backendConfig, "backendConfig must not be null");
      this.indexCacheBackendConfig = backendConfig;
      this.indexCacheBackendUri = null;
      return this;
    }

    /**
     * Sets the size of the metadata cache in bytes.
     *
     * @param metadataCacheSizeBytes the size of the metadata cache in bytes (must be non-negative)
     * @return this builder instance
     */
    public Builder metadataCacheSizeBytes(long metadataCacheSizeBytes) {
      Preconditions.checkArgument(
          metadataCacheSizeBytes >= 0, "metadataCacheSizeBytes must be >= 0");
      this.metadataCacheSizeBytes = metadataCacheSizeBytes;
      return this;
    }

    /**
     * Selects a registered metadata cache backend using a backend URI.
     *
     * <p>For example, {@code moka://?capacity=1048576}. This option is mutually exclusive with
     * {@link #metadataCacheSizeBytes(long)}.
     *
     * @param backendUri backend URI whose scheme identifies the registered backend
     * @return this builder instance
     */
    public Builder metadataCacheBackend(String backendUri) {
      Preconditions.checkNotNull(backendUri, "backendUri must not be null");
      this.metadataCacheBackendUri = backendUri;
      this.metadataCacheBackendConfig = null;
      return this;
    }

    /**
     * Selects a registered metadata cache backend using structured configuration.
     *
     * <p>This option is mutually exclusive with {@link #metadataCacheSizeBytes(long)}.
     *
     * @param backendConfig backend kind and backend-specific options
     * @return this builder instance
     */
    public Builder metadataCacheBackend(CacheBackendConfig backendConfig) {
      Preconditions.checkNotNull(backendConfig, "backendConfig must not be null");
      this.metadataCacheBackendConfig = backendConfig;
      this.metadataCacheBackendUri = null;
      return this;
    }

    /**
     * Builds the Session with the configured settings.
     *
     * @return a new Session instance
     */
    public Session build() {
      validateCacheConfiguration();
      long handle =
          createNative(
              indexCacheSizeBytes == null ? -1 : indexCacheSizeBytes,
              metadataCacheSizeBytes == null ? -1 : metadataCacheSizeBytes,
              indexCacheBackendUri,
              backendKind(indexCacheBackendConfig),
              backendOptions(indexCacheBackendConfig),
              metadataCacheBackendUri,
              backendKind(metadataCacheBackendConfig),
              backendOptions(metadataCacheBackendConfig));
      return new Session(handle);
    }

    private void validateCacheConfiguration() {
      Preconditions.checkArgument(
          indexCacheSizeBytes == null
              || (indexCacheBackendUri == null && indexCacheBackendConfig == null),
          "indexCacheSizeBytes and indexCacheBackend are mutually exclusive; set one or the other");
      Preconditions.checkArgument(
          metadataCacheSizeBytes == null
              || (metadataCacheBackendUri == null && metadataCacheBackendConfig == null),
          "metadataCacheSizeBytes and metadataCacheBackend are mutually exclusive; "
              + "set one or the other");
    }

    private static String backendKind(CacheBackendConfig config) {
      return config == null ? null : config.getKind();
    }

    private static Map<String, String> backendOptions(CacheBackendConfig config) {
      return config == null ? Collections.emptyMap() : config.getOptions();
    }
  }

  /**
   * Creates a Session from an existing native handle. This is used internally when retrieving the
   * session from a dataset.
   *
   * @param handle the native session handle
   * @return a new Session instance wrapping the handle
   */
  static Session fromHandle(long handle) {
    Preconditions.checkArgument(handle != 0, "Invalid session handle");
    return new Session(handle);
  }

  /**
   * Returns the current size of the session in bytes.
   *
   * <p>This includes the size of both index and metadata caches. Note that computing this is not
   * trivial as it walks the caches.
   *
   * @return the size of the session in bytes
   */
  public long sizeBytes() {
    Preconditions.checkArgument(nativeSessionHandle != 0, "Session is closed");
    return sizeBytesNative();
  }

  /**
   * Returns statistics for the metadata cache of this session.
   *
   * <p>The returned statistics are a snapshot; call this method periodically to observe how hits
   * and misses evolve over time.
   *
   * @return the metadata cache statistics
   */
  public CacheStats metadataCacheStats() {
    Preconditions.checkArgument(nativeSessionHandle != 0, "Session is closed");
    return metadataCacheStatsNative();
  }

  /**
   * Returns whether the other session is the same as this one.
   *
   * <p>Two sessions are considered the same if they share the same underlying native session. This
   * comparison uses the underlying Arc pointer equality, so sessions obtained from different
   * sources (e.g., directly created vs obtained from a dataset) will be correctly identified as the
   * same if they share the same underlying session.
   *
   * @param other the other session to compare
   * @return true if both sessions share the same underlying session
   */
  public boolean isSameAs(Session other) {
    if (other == null) {
      return false;
    }
    if (this.nativeSessionHandle == 0 || other.nativeSessionHandle == 0) {
      return false;
    }
    return isSameAsNative(this.nativeSessionHandle, other.nativeSessionHandle);
  }

  /**
   * Returns the native session handle. Used internally for passing to JNI methods.
   *
   * @return the native session handle
   */
  long getNativeHandle() {
    return nativeSessionHandle;
  }

  /**
   * Checks if this session is closed.
   *
   * @return true if the session is closed, false otherwise
   */
  public boolean isClosed() {
    return nativeSessionHandle == 0;
  }

  /**
   * Closes this session and releases any resources associated with it.
   *
   * <p>After calling this method, the session should not be used. Datasets that were opened with
   * this session will continue to work until they are closed, as they hold their own reference to
   * the underlying native session.
   */
  @Override
  public void close() {
    if (nativeSessionHandle != 0) {
      releaseNative(nativeSessionHandle);
      nativeSessionHandle = 0;
    }
  }

  @Override
  public String toString() {
    if (nativeSessionHandle == 0) {
      return "Session(closed)";
    }
    return String.format("Session(sizeBytes=%d)", sizeBytes());
  }

  private static native long createNative(
      long indexCacheSizeBytes,
      long metadataCacheSizeBytes,
      String indexCacheBackendUri,
      String indexCacheBackendKind,
      Map<String, String> indexCacheBackendOptions,
      String metadataCacheBackendUri,
      String metadataCacheBackendKind,
      Map<String, String> metadataCacheBackendOptions);

  private native long sizeBytesNative();

  private native CacheStats metadataCacheStatsNative();

  private static native void releaseNative(long handle);

  private static native boolean isSameAsNative(long handle1, long handle2);
}
