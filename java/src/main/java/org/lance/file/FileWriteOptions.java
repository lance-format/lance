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
package org.lance.file;

import java.util.Optional;

/**
 * Options for configuring a current-format Lance file writer.
 *
 * <p>These options are ignored for legacy V1 files.
 */
public class FileWriteOptions {
  private final Optional<Long> dataCacheBytes;
  private final Optional<Long> maxPageBytes;

  private FileWriteOptions(Builder builder) {
    this.dataCacheBytes = builder.dataCacheBytes;
    this.maxPageBytes = builder.maxPageBytes;
  }

  /**
   * Returns the total column-data buffering budget.
   *
   * <p>The budget is divided evenly across top-level columns. An empty value uses 8 MiB per column.
   */
  public Optional<Long> getDataCacheBytes() {
    return dataCacheBytes;
  }

  /** Returns the best-effort maximum page size, or an empty value to use the writer default. */
  public Optional<Long> getMaxPageBytes() {
    return maxPageBytes;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private Optional<Long> dataCacheBytes = Optional.empty();
    private Optional<Long> maxPageBytes = Optional.empty();

    private Builder() {}

    /**
     * Sets the total column-data buffering budget.
     *
     * @param dataCacheBytes buffer budget in bytes
     */
    public Builder dataCacheBytes(long dataCacheBytes) {
      this.dataCacheBytes = Optional.of(dataCacheBytes);
      return this;
    }

    /**
     * Sets the best-effort maximum page size.
     *
     * @param maxPageBytes positive page size in bytes
     */
    public Builder maxPageBytes(long maxPageBytes) {
      this.maxPageBytes = Optional.of(maxPageBytes);
      return this;
    }

    public FileWriteOptions build() {
      return new FileWriteOptions(this);
    }
  }
}
