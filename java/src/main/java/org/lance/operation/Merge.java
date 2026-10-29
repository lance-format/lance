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

import org.lance.FragmentMetadata;

import com.google.common.base.MoreObjects;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.List;
import java.util.Objects;

/**
 * Merge operation that combines new fragments with a specific schema. This operation is used for
 * schema evolution and column modifications.
 */
public class Merge extends SchemaOperation {
  private final List<FragmentMetadata> fragments;
  // True when this merge makes no nullability-affecting schema change: it
  // introduces no field that data staged against an earlier schema could not
  // safely omit. Without the assertion the merge conservatively conflicts with
  // concurrent appends, whose fragments would omit new columns and read as null.
  private final boolean preservesNullability;

  protected Merge(List<FragmentMetadata> fragments, Schema schema, boolean preservesNullability) {
    super(schema);
    this.fragments = fragments;
    this.preservesNullability = preservesNullability;
  }

  public List<FragmentMetadata> fragments() {
    return fragments;
  }

  public boolean preservesNullability() {
    return preservesNullability;
  }

  @Override
  public String name() {
    return "Merge";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fragments", fragments)
        .add("schema", schema())
        .add("preservesNullability", preservesNullability)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    if (!super.equals(o)) return false;
    Merge that = (Merge) o;
    return Objects.equals(fragments, that.fragments)
        && preservesNullability == that.preservesNullability;
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), fragments, preservesNullability);
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private List<FragmentMetadata> fragments;
    private Schema schema;
    // No assertion by default, which conservatively conflicts.
    private boolean preservesNullability = false;

    private Builder() {}

    public Builder fragments(List<FragmentMetadata> fragments) {
      this.fragments = fragments;
      return this;
    }

    public Builder schema(Schema schema) {
      this.schema = schema;
      return this;
    }

    public Builder preservesNullability(boolean preservesNullability) {
      this.preservesNullability = preservesNullability;
      return this;
    }

    public Merge build() {
      return new Merge(fragments, schema, preservesNullability);
    }
  }
}
