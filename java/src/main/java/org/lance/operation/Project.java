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

import com.google.common.base.MoreObjects;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.Objects;

/**
 * Project to a new schema. This Operation only changes the schema, not the data. Note: 1. For
 * removing columns. The data will be removed after compaction. 2. Project will modify column
 * positions, not ids(a.k.a. field id)
 */
public class Project extends SchemaOperation {
  // True when this projection makes no nullability-affecting schema change,
  // as a rename or a drop does not. Without the assertion the projection
  // conservatively conflicts with concurrent value writes.
  private final boolean preservesNullability;

  private Project(Schema schema, boolean preservesNullability) {
    super(schema);
    this.preservesNullability = preservesNullability;
  }

  public boolean preservesNullability() {
    return preservesNullability;
  }

  @Override
  public String name() {
    return "Project";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("schema", schema())
        .add("preservesNullability", preservesNullability)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (!super.equals(o)) {
      return false;
    }
    return preservesNullability == ((Project) o).preservesNullability;
  }

  @Override
  public int hashCode() {
    return Objects.hash(schema(), preservesNullability);
  }

  public static Builder builder() {
    return new Builder();
  }

  public static class Builder {
    private Schema schema;
    private boolean preservesNullability;

    public Builder() {}

    public Builder schema(Schema schema) {
      this.schema = schema;
      return this;
    }

    public Builder preservesNullability(boolean preservesNullability) {
      this.preservesNullability = preservesNullability;
      return this;
    }

    public Project build() {
      return new Project(schema, preservesNullability);
    }
  }
}
