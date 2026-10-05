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
package org.lance.ipc;

import org.apache.arrow.util.Preconditions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Controls whether projected columns are fetched before or after filtering.
 *
 * <p>For example, a selective filter can fetch a large payload column only for matching rows:
 *
 * <pre>{@code
 * ScanOptions options = new ScanOptions.Builder()
 *     .filter("id = 42")
 *     .lateMaterialization(MaterializationStyle.allEarlyExcept(
 *         Collections.singletonList("payload")))
 *     .build();
 * }</pre>
 *
 * <p>This policy only affects projected columns that are not used by a filter in a regular scan.
 * Vector search and full-text search always use late materialization.
 */
public final class MaterializationStyle {
  /** Available materialization policies. */
  public enum Mode {
    /** Let Lance choose based on the storage system and column size. */
    HEURISTIC("heuristic"),
    /** Fetch all eligible projected columns after filtering. */
    ALL_LATE("all_late"),
    /** Fetch all projected columns before filtering. */
    ALL_EARLY("all_early"),
    /** Fetch only the specified projected columns after filtering. */
    ALL_EARLY_EXCEPT("all_early_except");

    private final String rustValue;

    Mode(String rustValue) {
      this.rustValue = rustValue;
    }

    /** Returns the explicit value expected by the Rust binding. */
    public String toRustString() {
      return rustValue;
    }
  }

  private final Mode mode;
  private final List<String> columns;

  private MaterializationStyle(Mode mode, List<String> columns) {
    this.mode = mode;
    this.columns = Collections.unmodifiableList(new ArrayList<>(columns));
  }

  /** Use Lance's storage-aware heuristic. This is the default when no style is specified. */
  public static MaterializationStyle heuristic() {
    return new MaterializationStyle(Mode.HEURISTIC, Collections.emptyList());
  }

  /** Fetch all eligible projected columns after filtering. */
  public static MaterializationStyle allLate() {
    return new MaterializationStyle(Mode.ALL_LATE, Collections.emptyList());
  }

  /** Fetch all projected columns before filtering. */
  public static MaterializationStyle allEarly() {
    return new MaterializationStyle(Mode.ALL_EARLY, Collections.emptyList());
  }

  /**
   * Fetch only the specified projected columns after filtering; all other columns are fetched
   * before filtering. An empty list is equivalent to {@link #allEarly()}.
   */
  public static MaterializationStyle allEarlyExcept(List<String> lateColumns) {
    Preconditions.checkNotNull(lateColumns, "lateColumns must not be null");
    for (String column : lateColumns) {
      Preconditions.checkArgument(
          column != null && !column.isEmpty(), "lateColumns must not contain null or empty names");
    }
    return new MaterializationStyle(Mode.ALL_EARLY_EXCEPT, lateColumns);
  }

  /** Returns the selected materialization mode. */
  public Mode getMode() {
    return mode;
  }

  /** Returns the explicit value expected by the Rust binding. */
  public String toRustString() {
    return mode.toRustString();
  }

  /** Columns to fetch after filtering when using {@link Mode#ALL_EARLY_EXCEPT}. */
  public List<String> getColumns() {
    return columns;
  }

  @Override
  public String toString() {
    return "MaterializationStyle{" + "mode=" + mode + ", columns=" + columns + '}';
  }
}
