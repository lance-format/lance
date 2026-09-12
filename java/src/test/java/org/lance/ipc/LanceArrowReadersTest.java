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

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LanceArrowReadersTest {

  @Test
  public void loadNextBatchAddsContextToArrowChildMismatch() throws Exception {
    try (BufferAllocator allocator = new RootAllocator();
        ArrowReader reader =
            LanceArrowReaders.withDiagnostics(
                allocator, childMismatchReader(allocator), "unit-test stream")) {
      IllegalArgumentException error =
          assertThrows(IllegalArgumentException.class, reader::loadNextBatch);

      assertTrue(error.getMessage().contains("unit-test stream"));
      assertTrue(error.getMessage().contains("reader-owned VectorSchemaRoot"));
      assertTrue(error.getMessage().contains("found 0 expected 6"));
    }
  }

  @Test
  public void loadNextBatchAddsContextToRuntimeException() throws Exception {
    try (BufferAllocator allocator = new RootAllocator();
        ArrowReader reader =
            LanceArrowReaders.withDiagnostics(
                allocator, runtimeFailureReader(allocator), "runtime stream")) {
      RuntimeException error = assertThrows(RuntimeException.class, reader::loadNextBatch);

      assertTrue(error.getMessage().contains("runtime stream"));
      assertTrue(error.getMessage().contains("delegate failed"));
    }
  }

  @Test
  public void getVectorSchemaRootInitializesDelegateOnlyOnce() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      CountingReader delegate = new CountingReader(allocator);
      try (ArrowReader reader = LanceArrowReaders.withDiagnostics(allocator, delegate, "source")) {
        VectorSchemaRoot root = reader.getVectorSchemaRoot();

        assertSame(root, reader.getVectorSchemaRoot());
        assertEquals(1, delegate.readSchemaCount);
      }
    }
  }

  @Test
  public void closeForwardsToDelegateOnce() throws Exception {
    try (BufferAllocator allocator = new RootAllocator()) {
      CloseCountingReader delegate = new CloseCountingReader(allocator);
      ArrowReader reader = LanceArrowReaders.withDiagnostics(allocator, delegate, "source");

      reader.close();

      assertEquals(1, delegate.closeCount);
      assertTrue(delegate.closeReadSource);
    }
  }

  private ArrowReader childMismatchReader(BufferAllocator allocator) {
    return new ArrowReader(allocator) {
      @Override
      public boolean loadNextBatch() {
        throw new IllegalArgumentException(
            "should have as many children as in the schema: found 0 expected 6");
      }

      @Override
      public long bytesRead() {
        return 0;
      }

      @Override
      protected void closeReadSource() {}

      @Override
      protected Schema readSchema() {
        return new Schema(Collections.emptyList());
      }
    };
  }

  private ArrowReader runtimeFailureReader(BufferAllocator allocator) {
    return new ArrowReader(allocator) {
      @Override
      public boolean loadNextBatch() {
        throw new IllegalStateException("delegate failed");
      }

      @Override
      public long bytesRead() {
        return 0;
      }

      @Override
      protected void closeReadSource() {}

      @Override
      protected Schema readSchema() {
        return new Schema(Collections.emptyList());
      }
    };
  }

  private static class CountingReader extends ArrowReader {
    private int readSchemaCount = 0;

    private CountingReader(BufferAllocator allocator) {
      super(allocator);
    }

    @Override
    public boolean loadNextBatch() {
      return false;
    }

    @Override
    public long bytesRead() {
      return 0;
    }

    @Override
    protected void closeReadSource() {}

    @Override
    protected Schema readSchema() {
      readSchemaCount++;
      return new Schema(Collections.emptyList());
    }
  }

  private static class CloseCountingReader extends ArrowReader {
    private int closeCount = 0;
    private boolean closeReadSource = false;

    private CloseCountingReader(BufferAllocator allocator) {
      super(allocator);
    }

    @Override
    public boolean loadNextBatch() {
      return false;
    }

    @Override
    public long bytesRead() {
      return 0;
    }

    @Override
    public void close(boolean closeReadSource) {
      closeCount++;
      this.closeReadSource = closeReadSource;
    }

    @Override
    protected void closeReadSource() {}

    @Override
    protected Schema readSchema() {
      return new Schema(Collections.emptyList());
    }
  }
}
