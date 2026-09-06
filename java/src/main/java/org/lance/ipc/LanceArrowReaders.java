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

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.dictionary.Dictionary;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;

import java.io.IOException;
import java.util.Map;
import java.util.Set;

/** Utilities for importing Arrow readers from Lance-owned Arrow C Data streams. */
public final class LanceArrowReaders {
  private static final String CHILD_MISMATCH_ERROR =
      "should have as many children as in the schema";

  private LanceArrowReaders() {}

  /** Import a C Data stream and attach Lance-specific diagnostics to batch-load failures. */
  public static ArrowReader importArrayStream(
      BufferAllocator allocator, ArrowArrayStream stream, String source) {
    return withDiagnostics(allocator, Data.importArrayStream(allocator, stream), source);
  }

  static ArrowReader withDiagnostics(
      BufferAllocator allocator, ArrowReader delegate, String source) {
    return new DiagnosticArrowReader(allocator, delegate, source);
  }

  private static final class DiagnosticArrowReader extends ArrowReader {
    private final ArrowReader delegate;
    private final String source;
    private Schema schema;

    private DiagnosticArrowReader(BufferAllocator allocator, ArrowReader delegate, String source) {
      super(allocator);
      this.delegate = delegate;
      this.source = source;
    }

    @Override
    public VectorSchemaRoot getVectorSchemaRoot() throws IOException {
      VectorSchemaRoot root = delegate.getVectorSchemaRoot();
      schema = root.getSchema();
      return root;
    }

    @Override
    public Map<Long, Dictionary> getDictionaryVectors() throws IOException {
      return delegate.getDictionaryVectors();
    }

    @Override
    public Dictionary lookup(long id) {
      return delegate.lookup(id);
    }

    @Override
    public Set<Long> getDictionaryIds() {
      return delegate.getDictionaryIds();
    }

    @Override
    public boolean loadNextBatch() throws IOException {
      try {
        boolean loaded = delegate.loadNextBatch();
        if (loaded) {
          schema = delegate.getVectorSchemaRoot().getSchema();
        }
        return loaded;
      } catch (IllegalArgumentException e) {
        throw new IllegalArgumentException(loadFailureMessage(source, e), e);
      } catch (IOException e) {
        throw new IOException(loadFailureMessage(source, e), e);
      } catch (RuntimeException e) {
        throw new RuntimeException(loadFailureMessage(source, e), e);
      }
    }

    @Override
    public long bytesRead() {
      return delegate.bytesRead();
    }

    @Override
    public void close() throws IOException {
      // This reader is a pass-through diagnostics wrapper. It never initializes
      // ArrowReader's private root, so closing the delegate is the only owner cleanup needed.
      delegate.close();
    }

    @Override
    public void close(boolean closeReadSource) throws IOException {
      delegate.close(closeReadSource);
    }

    @Override
    protected void closeReadSource() throws IOException {
      delegate.close(false);
    }

    @Override
    protected Schema readSchema() throws IOException {
      if (schema == null) {
        throw new IOException("Diagnostic Arrow reader does not own a schema");
      }
      return schema;
    }
  }

  private static String loadFailureMessage(String source, Exception error) {
    String arrowMessage = error.getMessage();
    if (arrowMessage != null && arrowMessage.contains(CHILD_MISMATCH_ERROR)) {
      return String.format(
          "Failed to load next Arrow batch from %s: Arrow vector children did not match "
              + "the schema. This usually means the Arrow C Data stream emitted a malformed "
              + "nested batch, or the reader-owned VectorSchemaRoot was closed or mutated before "
              + "loading the next batch. Arrow error: %s",
          source, arrowMessage);
    }
    return String.format("Failed to load next Arrow batch from %s: %s", source, arrowMessage);
  }
}
