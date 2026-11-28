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

import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.types.pojo.Schema;

import java.util.List;
import java.util.Map;
import java.util.Optional;

@Deprecated
/** Fragment related operations. Use {@link Transaction} instead */
public abstract class FragmentOperation {
  protected static void validateFragments(List<FragmentMetadata> fragments) {
    if (fragments == null || fragments.isEmpty()) {
      throw new IllegalArgumentException("fragments cannot be null or empty");
    }
  }

  public abstract Dataset commit(
      BufferAllocator allocator,
      String path,
      Optional<Long> readVersion,
      Map<String, String> storageOptions);

  /** Fragment append operation. */
  public static class Append extends FragmentOperation {
    private final List<FragmentMetadata> fragments;

    public Append(List<FragmentMetadata> fragments) {
      validateFragments(fragments);
      this.fragments = fragments;
    }

    @Override
    public Dataset commit(
        BufferAllocator allocator,
        String path,
        Optional<Long> readVersion,
        Map<String, String> storageOptions) {
      Preconditions.checkNotNull(allocator);
      Preconditions.checkNotNull(path);
      Preconditions.checkNotNull(readVersion);
      return Dataset.commitAppend(allocator, path, readVersion, fragments, storageOptions);
    }
  }

  /** Fragment overwrite operation. */
  public static class Overwrite extends FragmentOperation {
    private final List<FragmentMetadata> fragments;
    private final Schema schema;

    public Overwrite(List<FragmentMetadata> fragments, Schema schema) {
      validateFragments(fragments);
      this.fragments = fragments;
      this.schema = schema;
    }

    @Override
    public Dataset commit(
        BufferAllocator allocator,
        String path,
        Optional<Long> readVersion,
        Map<String, String> storageOptions) {
      Preconditions.checkNotNull(allocator);
      Preconditions.checkNotNull(path);
      Preconditions.checkNotNull(readVersion);
      try (ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
        Data.exportSchema(allocator, schema, null, arrowSchema);
        return Dataset.commitOverwrite(
            allocator, path, arrowSchema.memoryAddress(), readVersion, fragments, storageOptions);
      }
    }
  }
}
