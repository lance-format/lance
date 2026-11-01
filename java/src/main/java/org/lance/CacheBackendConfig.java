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

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Structured configuration for a cache backend registered with the native Lance runtime.
 *
 * <p>The {@code kind} selects a registered backend constructor, while {@code options} contains
 * backend-specific string settings. For example:
 *
 * <pre>{@code
 * CacheBackendConfig config = CacheBackendConfig.builder("moka")
 *     .option("capacity", "1048576")
 *     .build();
 * }</pre>
 *
 * <p>Third-party native backend crates register their constructors with Lance at application
 * startup. This class selects and configures one of those registered constructors; it does not
 * register a Java implementation as a native cache backend.
 */
public final class CacheBackendConfig {
  private final String kind;
  private final Map<String, String> options;

  private CacheBackendConfig(Builder builder) {
    this.kind = builder.kind;
    this.options = Collections.unmodifiableMap(new HashMap<>(builder.options));
  }

  /**
   * Creates a builder for a registered backend kind.
   *
   * @param kind registered backend identifier, such as {@code moka}
   * @return a new builder
   */
  public static Builder builder(String kind) {
    return new Builder(kind);
  }

  /** Returns the registered backend identifier. */
  public String getKind() {
    return kind;
  }

  /** Returns an immutable map of backend-specific options. */
  public Map<String, String> getOptions() {
    return options;
  }

  /** Builder for {@link CacheBackendConfig}. */
  public static final class Builder {
    private final String kind;
    private final Map<String, String> options = new HashMap<>();

    private Builder(String kind) {
      Preconditions.checkNotNull(kind, "kind must not be null");
      Preconditions.checkArgument(!kind.isEmpty(), "kind must not be empty");
      this.kind = kind;
    }

    /**
     * Adds a backend-specific option.
     *
     * @param key option name
     * @param value option value
     * @return this builder
     */
    public Builder option(String key, String value) {
      Preconditions.checkNotNull(key, "cache backend option key must not be null");
      Preconditions.checkNotNull(value, "cache backend option value must not be null");
      Preconditions.checkArgument(!key.isEmpty(), "cache backend option key must not be empty");
      options.put(key, value);
      return this;
    }

    /**
     * Replaces the current backend options.
     *
     * @param options backend-specific options
     * @return this builder
     */
    public Builder options(Map<String, String> options) {
      Preconditions.checkNotNull(options, "options must not be null");
      this.options.clear();
      for (Map.Entry<String, String> option : options.entrySet()) {
        option(option.getKey(), option.getValue());
      }
      return this;
    }

    /** Builds the immutable backend configuration. */
    public CacheBackendConfig build() {
      return new CacheBackendConfig(this);
    }
  }
}
