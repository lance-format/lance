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
 * Options for collecting session cache diagnostics.
 *
 * <pre>{@code
 * CacheDiagnosticsOptions options =
 *     CacheDiagnosticsOptions.builder().refresh(true).byType(true).build();
 * CacheDiagnostics diagnostics = session.getCacheDiagnostics(options);
 * }</pre>
 */
public final class CacheDiagnosticsOptions {
  private final boolean refresh;
  private final boolean byType;

  private CacheDiagnosticsOptions(Builder builder) {
    this.refresh = builder.refresh;
    this.byType = builder.byType;
  }

  /** Returns whether backend maintenance is requested before sampling. */
  public boolean getRefresh() {
    return refresh;
  }

  /** Returns whether bounded per-type activity and scanned occupancy are requested. */
  public boolean getByType() {
    return byType;
  }

  /** Creates a builder whose options are disabled by default. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder for {@link CacheDiagnosticsOptions}. */
  public static final class Builder {
    private boolean refresh;
    private boolean byType;

    private Builder() {}

    /** Requests backend maintenance before sampling. */
    public Builder refresh(boolean refresh) {
      this.refresh = refresh;
      return this;
    }

    /** Requests bounded per-type activity and a resident-entry occupancy scan. */
    public Builder byType(boolean byType) {
      this.byType = byType;
      return this;
    }

    /** Builds the immutable options. */
    public CacheDiagnosticsOptions build() {
      return new CacheDiagnosticsOptions(this);
    }
  }
}
