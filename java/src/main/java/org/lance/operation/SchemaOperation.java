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

import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.Objects;

/**
 * Schema related base operation.
 *
 * <p>For legacy datasets, each field is assigned a field id in the following order:
 *
 * <ol>
 *   <li>Parse from field metadata with key {@code lance:field_id}.
 *   <li>Otherwise, set field id from txn read version dataset's schema field (with the same name).
 *   <li>Otherwise, allocate based on the max field id of the dataset.
 * </ol>
 *
 * <p>Datasets use stable field IDs only after explicit migration. On those datasets, compatible
 * existing fields keep their identities, while new IDs are assigned by the dataset allocator and
 * field mappings in fragments committed by the same operation are remapped to the canonical IDs.
 */
public abstract class SchemaOperation implements Operation {
  private final Schema schema;
  private ArrowSchema cSchema;

  protected SchemaOperation(Schema schema) {
    this.schema = schema;
  }

  public Schema schema() {
    return schema;
  }

  /**
   * Export the schema to rust jni.
   *
   * @param allocator the buffer allocator
   * @return the schema address
   */
  public long exportSchema(BufferAllocator allocator) {
    if (cSchema == null) {
      this.cSchema = ArrowSchema.allocateNew(allocator);
      Data.exportSchema(allocator, schema, null, cSchema);
    }
    return cSchema.memoryAddress();
  }

  public void release() {
    if (cSchema != null) {
      cSchema.close();
    }
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    SchemaOperation that = (SchemaOperation) o;
    return Objects.equals(schema, that.schema);
  }
}
