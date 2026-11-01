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
package org.lance.operation;

import org.lance.BasePath;
import org.lance.FragmentMetadata;

import com.google.common.base.MoreObjects;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Overwrite the dataset with new fragments. This operation will overwrite the existing dataset.
 * Note: 1. The operation won't delete table config keys which do not exist in configUpsertValues.
 * 2. If we want to create a new Dataset, use {@link org.lance.Dataset}.create instead.
 */
public class Overwrite extends SchemaOperation {
  private final List<FragmentMetadata> fragments;
  private final Optional<Map<String, String>> configUpsertValues;
  private final Optional<List<BasePath>> initialBases;

  protected Overwrite(
      List<FragmentMetadata> fragments, Schema schema, Map<String, String> configUpsertValues) {
    this(fragments, schema, configUpsertValues, null);
  }

  protected Overwrite(
      List<FragmentMetadata> fragments,
      Schema schema,
      Map<String, String> configUpsertValues,
      List<BasePath> initialBases) {
    super(schema);
    this.fragments = fragments;
    this.configUpsertValues =
        configUpsertValues == null || configUpsertValues.isEmpty()
            ? Optional.empty()
            : Optional.of(configUpsertValues);
    this.initialBases = Optional.ofNullable(initialBases);
  }

  public List<FragmentMetadata> fragments() {
    return fragments;
  }

  public Optional<Map<String, String>> configUpsertValues() {
    return configUpsertValues;
  }

  /**
   * Base paths used when creating a dataset; absent for a regular overwrite. An explicit empty list
   * is retained because Rust distinguishes it from an absent value.
   */
  public Optional<List<BasePath>> initialBases() {
    return initialBases;
  }

  public Optional<List<BasePath>> getInitialBases() {
    return initialBases;
  }

  public static Builder builder() {
    return new Builder();
  }

  @Override
  public String name() {
    return "Overwrite";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fragments", fragments)
        .add("schema", schema())
        .add("configUpsertValues", configUpsertValues)
        .add("initialBases", initialBases)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    if (!super.equals(o)) return false;
    Overwrite that = (Overwrite) o;
    return Objects.equals(fragments, that.fragments)
        && Objects.equals(configUpsertValues, that.configUpsertValues)
        && initialBasesEqual(initialBases, that.initialBases);
  }

  private static boolean initialBasesEqual(
      Optional<List<BasePath>> left, Optional<List<BasePath>> right) {
    if (left.isEmpty() || right.isEmpty()) return left.isEmpty() && right.isEmpty();
    List<BasePath> leftBases = left.orElseThrow();
    List<BasePath> rightBases = right.orElseThrow();
    if (leftBases.size() != rightBases.size()) return false;
    for (int i = 0; i < leftBases.size(); i++) {
      BasePath a = leftBases.get(i);
      BasePath b = rightBases.get(i);
      if (a.getId() != b.getId()
          || a.isDatasetRoot() != b.isDatasetRoot()
          || !Objects.equals(a.getName(), b.getName())
          || !Objects.equals(a.getPath(), b.getPath())) {
        return false;
      }
    }
    return true;
  }

  public static class Builder {
    private List<FragmentMetadata> fragments;
    private Schema schema;
    private Map<String, String> configUpsertValues;
    private List<BasePath> initialBases;

    private Builder() {}

    public Builder fragments(List<FragmentMetadata> fragments) {
      this.fragments = fragments;
      return this;
    }

    public Builder schema(Schema schema) {
      this.schema = schema;
      return this;
    }

    public Builder configUpsertValues(Map<String, String> configUpsertValues) {
      this.configUpsertValues = configUpsertValues;
      return this;
    }

    public Builder initialBases(List<BasePath> initialBases) {
      this.initialBases = initialBases;
      return this;
    }

    public Overwrite build() {
      return new Overwrite(fragments, schema, configUpsertValues, initialBases);
    }
  }
}
