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

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.AutoCloseables;
import org.apache.arrow.util.Preconditions;
import org.apache.arrow.vector.VectorSchemaRoot;

import javax.annotation.Nullable;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Portable Blob V2 capture, transport, and planned reads.
 *
 * <p>Locations and contexts are opaque protobuf messages validated by Rust. Workers must open the
 * exact source snapshot with independently resolved credentials. Check {@link
 * #getProtocolVersion()} and {@link #getSupportedCapabilities()} before scheduling workers. Capture
 * does not retain a snapshot; acquire {@link SnapshotRetention} before issuing references.
 *
 * <p>All indices, versions, offsets, lengths, and budgets use unsigned 64-bit values represented by
 * Java long bit patterns. Use {@link Long#parseUnsignedLong(String)} when needed. Native validation
 * rejects overflow and out-of-bounds ranges without truncating to signed values.
 *
 * <pre>{@code
 * SourceContext source = SourceContext.fromDataset(dataset, "catalog/table", "worker/storage");
 * SnapshotRetention retention =
 *     SnapshotRetention.acquire(dataset, source.getSnapshotIdentity(), uniqueQueryId);
 * try (Capture capture = Capture.create(dataset, source, fieldId);
 *      Reader reader = Reader.create(List.of(source), List.of(workerDataset), 8 << 20, 32 << 20)) {
 *   Batch refs = capture.capture(source, scannedBatch, "blob", allocator);
 *   try (ReadStream stream = reader.read(refs, List.of(Request.range(0, 42, 0, 4)))) {
 *     try (VectorSchemaRoot result = stream.next(allocator)) {
 *       // request_index: uint64; data: nullable large_binary
 *     }
 *   }
 * } finally {
 *   // Only after every worker and retry has stopped.
 *   retention.release(dataset);
 * }
 * }</pre>
 */
public final class BlobReferences {
  static {
    JniLoader.ensureLoaded();
  }

  private BlobReferences() {}

  /** Supported execution-wire version, available without opening a dataset. */
  public static native int getProtocolVersion();

  /** Supported source capabilities: object ranges (1), immutable external objects (2). */
  public static native int[] getSupportedCapabilities();

  /** SDK operations, including snapshot_retention_v1; distinct from wire capabilities. */
  public static native String[] getSupportedOperations();

  /** Exact snapshot metadata and authenticated worker context key; contains no credentials. */
  public static final class SourceContext {
    private final byte[] encoded;
    private final String[] description;

    private SourceContext(byte[] encoded) {
      this.encoded = encoded.clone();
      this.description = describeContextNative(this.encoded);
    }

    /**
     * Capture exact manifest metadata, without acquiring retention or reading payloads.
     * tableIdentity must distinguish table incarnations and catalog/storage authorities.
     * accessContextKey must resolve authenticated storage and per-base routing on every worker.
     * Neither value may be empty.
     */
    public static SourceContext fromDataset(
        Dataset dataset, String tableIdentity, String accessContextKey) throws IOException {
      return fromDataset(dataset, tableIdentity, accessContextKey, false);
    }

    /**
     * Capture metadata. Setting assumeImmutableExternal accepts the application's external object
     * immutability/lifetime guarantee; it does not enforce object versions or conditional reads.
     */
    public static SourceContext fromDataset(
        Dataset dataset,
        String tableIdentity,
        String accessContextKey,
        boolean assumeImmutableExternal)
        throws IOException {
      Preconditions.checkNotNull(dataset);
      try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
        return new SourceContext(
            captureContextNative(
                dataset, tableIdentity, accessContextKey, assumeImmutableExternal));
      }
    }

    /** Validate a source message from another process before binding it. */
    public static SourceContext decode(byte[] encoded) {
      return new SourceContext(encoded);
    }

    /** Serialize the original context intact. */
    public byte[] encode() {
      return encoded.clone();
    }

    public String getTableIdentity() {
      return description[0];
    }

    public String getTableUri() {
      return description[1];
    }

    @Nullable
    public String getBranch() {
      return description[2];
    }

    public long getVersion() {
      return Long.parseUnsignedLong(description[3]);
    }

    public String getAccessContextKey() {
      return description[4];
    }

    /** Snapshot identity shared with ordinary scans, independent of worker routing. */
    public SnapshotIdentity getSnapshotIdentity() {
      return SnapshotIdentity.decode(snapshotIdentityNative(encoded));
    }
  }

  /** Nullable reference Arrow array and its source table, serialized for exchange/spill. */
  public static final class Batch {
    private final byte[] encoded;

    private Batch(byte[] encoded) {
      this.encoded = encoded;
    }

    /** Decode and validate a message, including every locator and context fingerprint. */
    public static Batch decode(byte[] encoded) {
      return new Batch(validateBatchNative(encoded));
    }

    /** Encode for another process. */
    public byte[] encode() {
      return encoded.clone();
    }

    /** Contexts in the order required by the worker reader's binding table. */
    public SourceContext[] getContexts() {
      byte[][] contexts = batchContextsNative(encoded);
      SourceContext[] result = new SourceContext[contexts.length];
      for (int i = 0; i < contexts.length; i++) result[i] = SourceContext.decode(contexts[i]);
      return result;
    }

    /** Export one nullable lance.blob.ref extension column. The caller must close the root. */
    public VectorSchemaRoot toArrow(BufferAllocator allocator) {
      try (ArrowArray array = ArrowArray.allocateNew(allocator);
          ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
        exportBatchNative(encoded, array.memoryAddress(), schema.memoryAddress());
        return Data.importVectorSchemaRoot(allocator, array, schema, null);
      }
    }

    /**
     * Rebuild after Arrow filtering/reordering, preserving the extension field and context table.
     */
    public static Batch fromArrow(
        List<SourceContext> contexts, VectorSchemaRoot root, BufferAllocator allocator) {
      try (ArrowArray array = ArrowArray.allocateNew(allocator);
          ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
        Data.exportVectorSchemaRoot(allocator, root, null, array, schema);
        return new Batch(
            importBatchNative(
                encodeContexts(contexts), array.memoryAddress(), schema.memoryAddress()));
      }
    }

    /** Remap source slots before combining arrays produced by independent captures. */
    public Batch remapContexts(List<SourceContext> contexts) {
      return new Batch(remapBatchNative(encoded, encodeContexts(contexts)));
    }
  }

  private abstract static class NativeResource implements AutoCloseable {
    @SuppressWarnings("FieldCanBeLocal")
    private long nativeHandle;

    protected synchronized boolean isClosed() {
      return nativeHandle == 0;
    }

    /** Release native resources. Streams must be closed separately from their reader factory. */
    @Override
    public synchronized void close() throws IOException {
      if (nativeHandle != 0) closeNative(this);
    }
  }

  private static final class DatasetBinding extends NativeResource {
    private DatasetBinding() {}
  }

  /**
   * Flat references plus the selected nested container path. Uint64 leaves in the index Arrow batch
   * address {@link #getReferences()}; null leaves represent SQL nulls. Struct parent nulls,
   * List/LargeList offsets, empty lists, and element order are preserved.
   */
  public static final class NestedBatch {
    private final byte[] encoded;

    private NestedBatch(byte[] encoded) {
      this.encoded = encoded;
    }

    /** Decode both the reference table and shape mapping from another process. */
    public static NestedBatch decode(byte[] encoded) {
      return new NestedBatch(validateNestedBatchNative(encoded));
    }

    /** Rebuild after Arrow shape operations or remapping the flat reference contexts. */
    public static NestedBatch fromArrow(
        Batch references, VectorSchemaRoot indices, BufferAllocator allocator) {
      try (ArrowArray array = ArrowArray.allocateNew(allocator);
          ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
        Data.exportVectorSchemaRoot(allocator, indices, null, array, schema);
        return new NestedBatch(
            importNestedBatchNative(
                references.encoded, array.memoryAddress(), schema.memoryAddress()));
      }
    }

    /** Encode for exchange/spill. This does not acquire snapshot retention. */
    public byte[] encode() {
      return encoded.clone();
    }

    /** Flat nullable references accepted by the whole/range reader. */
    public Batch getReferences() {
      return new Batch(nestedReferencesNative(encoded));
    }

    /** Export the one-column shape mapping. The caller must close the returned root. */
    public VectorSchemaRoot getIndices(BufferAllocator allocator) {
      try (ArrowArray array = ArrowArray.allocateNew(allocator);
          ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
        exportNestedIndicesNative(encoded, array.memoryAddress(), schema.memoryAddress());
        return Data.importVectorSchemaRoot(allocator, array, schema, null);
      }
    }
  }

  /** Capture session for one exact snapshot and original Blob V2 leaf field ID. */
  public static final class Capture extends NativeResource {
    private Capture() {}

    public static Capture create(Dataset dataset, SourceContext context, int fieldId) {
      Preconditions.checkNotNull(dataset);
      try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
        return createCaptureNative(dataset, context.encoded, fieldId);
      }
    }

    /**
     * Consume the original descriptor field and physical uint64 _rowaddr from a scan batch. No
     * payloads, HEAD requests, or descriptor re-takes are performed. Preserve scanSource through
     * operators; column aliases cannot establish provenance. Use {@link #captureNested} for nested
     * leaves.
     */
    public Batch capture(
        SourceContext scanSource,
        VectorSchemaRoot scanBatch,
        String column,
        BufferAllocator allocator)
        throws IOException {
      try (ArrowArray array = ArrowArray.allocateNew(allocator);
          ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
        Data.exportVectorSchemaRoot(allocator, scanBatch, null, array, schema);
        return new Batch(
            captureNative(
                this, scanSource.encoded, column, array.memoryAddress(), schema.memoryAddress()));
      }
    }

    /**
     * Capture the selected leaf below Struct, List, and LargeList containers. column identifies the
     * original root scan column; the session's field ID identifies the leaf. No sibling payloads
     * are read. Map, union, and fixed-size list containers are unsupported.
     *
     * <pre>{@code
     * try (Capture capture = Capture.create(dataset, source, leafFieldId)) {
     *   NestedBatch nested = capture.captureNested(source, scanBatch, "media", allocator);
     *   Batch references = nested.getReferences();
     *   try (VectorSchemaRoot indices = nested.getIndices(allocator)) {
     *     // Preserve these offsets and indices through UNNEST/reconstruction.
     *   }
     * }
     * }</pre>
     */
    public NestedBatch captureNested(
        SourceContext scanSource,
        VectorSchemaRoot scanBatch,
        String column,
        BufferAllocator allocator)
        throws IOException {
      try (ArrowArray array = ArrowArray.allocateNew(allocator);
          ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
        Data.exportVectorSchemaRoot(allocator, scanBatch, null, array, schema);
        return new NestedBatch(
            captureNestedNative(
                this, scanSource.encoded, column, array.memoryAddress(), schema.memoryAddress()));
      }
    }
  }

  /** One request indexed by the engine; indices are unique within each read operation. */
  public static final class Request {
    private final long referenceIndex;
    private final long requestIndex;
    private final boolean hasRange;
    private final long offset;
    private final long length;

    private Request(
        long referenceIndex, long requestIndex, boolean hasRange, long offset, long length) {
      this.referenceIndex = referenceIndex;
      this.requestIndex = requestIndex;
      this.hasRange = hasRange;
      this.offset = offset;
      this.length = length;
    }

    /** Read a whole nullable value, resolving unknown external length in the native reader. */
    public static Request whole(long referenceIndex, long requestIndex) {
      return new Request(referenceIndex, requestIndex, false, 0, 0);
    }

    /** Read (offset, length) relative to the logical blob. An empty range at the end is valid. */
    public static Request range(long referenceIndex, long requestIndex, long offset, long length) {
      return new Request(referenceIndex, requestIndex, true, offset, length);
    }

    public long getReferenceIndex() {
      return referenceIndex;
    }

    public long getRequestIndex() {
      return requestIndex;
    }
  }

  /**
   * Planned reader factory sharing finite native I/O and materialization budgets across streams.
   * One oversized demand may run alone when the budget is idle. Budget limits are not hard
   * per-value caps; use chunked ranges if needed. Output Arrow buffers retain native reservations.
   * Extra copies made by callers require separate accounting.
   */
  public static final class Reader extends NativeResource {
    private Reader() {}

    /**
     * Bind contexts to independently authenticated exact-version worker datasets, in context-table
     * order. Per-base configuration and credential refresh come from those datasets. Each dataset
     * is cloned independently; closing it after its binding is captured does not invalidate the
     * reader.
     */
    public static Reader create(
        List<SourceContext> contexts,
        List<Dataset> datasets,
        long ioBufferSizeBytes,
        long materializationReadaheadBytes) {
      Preconditions.checkNotNull(datasets);
      Dataset[] sourceDatasets = datasets.toArray(new Dataset[0]);
      List<DatasetBinding> bindings = new ArrayList<>(sourceDatasets.length);
      try {
        for (Dataset dataset : sourceDatasets) {
          Preconditions.checkNotNull(dataset);
          // Never hold several lifecycle locks: queued close() calls can block new readers.
          try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
            bindings.add(bindDatasetNative(dataset));
          }
        }
        return createReaderNative(
            encodeContexts(contexts),
            bindings.toArray(new DatasetBinding[0]),
            ioBufferSizeBytes,
            materializationReadaheadBytes);
      } finally {
        AutoCloseables.closeNoChecked(AutoCloseables.all(bindings));
      }
    }

    /** Submit a finite request set. Validation happens before payload I/O. */
    public ReadStream read(Batch references, List<Request> requests) {
      int count = requests.size();
      long[] referenceIndices = new long[count],
          requestIndices = new long[count],
          offsets = new long[count],
          lengths = new long[count];
      boolean[] hasRanges = new boolean[count];
      for (int i = 0; i < count; i++) {
        Request request = Objects.requireNonNull(requests.get(i), "request");
        referenceIndices[i] = request.referenceIndex;
        requestIndices[i] = request.requestIndex;
        hasRanges[i] = request.hasRange;
        offsets[i] = request.offset;
        lengths[i] = request.length;
      }
      return readNative(
          this, references.encoded, referenceIndices, requestIndices, hasRanges, offsets, lengths);
    }

    public long getReservedBytes() {
      return reservedBytesNative(this, false);
    }

    public long getIoReservedBytes() {
      return reservedBytesNative(this, true);
    }

    /**
     * Cumulative counters with the same snake_case names as Rust/Python. Payload counters measure
     * scheduler submissions after coalescing/splitting, excluding backend retries. Logical bytes
     * count duplicates. Reservation peaks cover the shared native context and retained outputs.
     * Descriptor scans before capture are excluded; descriptor_requests is always zero.
     *
     * <pre>{@code
     * long physicalBytes = reader.getMetrics().get("payload_bytes");
     * long logicalBytes = reader.getMetrics().get("returned_logical_bytes");
     * }</pre>
     */
    public Map<String, Long> getMetrics() {
      return metricsNative(this);
    }
  }

  /**
   * Cancellable native result stream. Close it on success, failure, and cancellation. Closing the
   * reader factory does not close its streams. Errors terminate the stream and never become nulls.
   */
  public static final class ReadStream extends NativeResource {
    private ReadStream() {}

    /**
     * Await one completed request, or return null at EOF. The returned root has one row:
     * request_index (uint64) and data (nullable large_binary). Order is unspecified. Close each
     * result promptly to release its reservation and admit later work. Previously returned roots
     * remain valid after cancel/close. Only one thread may call next at a time.
     */
    @Nullable
    public VectorSchemaRoot next(BufferAllocator allocator) throws IOException {
      if (isClosed()) return null;
      try (ArrowArray array = ArrowArray.allocateNew(allocator);
          ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
        if (!nextNative(this, array.memoryAddress(), schema.memoryAddress())) return null;
        return Data.importVectorSchemaRoot(allocator, array, schema, null);
      }
    }

    /** Cancel concurrently with a blocked next call or budget admission. */
    public void cancel() {
      if (!isClosed()) cancelNative(this);
    }
  }

  private static byte[][] encodeContexts(List<SourceContext> contexts) {
    byte[][] encoded = new byte[contexts.size()][];
    for (int i = 0; i < contexts.size(); i++) encoded[i] = contexts.get(i).encoded;
    return encoded;
  }

  private static native byte[] captureContextNative(
      Dataset dataset,
      String tableIdentity,
      String accessContextKey,
      boolean assumeImmutableExternal)
      throws IOException;

  private static native String[] describeContextNative(byte[] context);

  private static native byte[] validateBatchNative(byte[] batch);

  private static native byte[][] batchContextsNative(byte[] batch);

  private static native void exportBatchNative(byte[] batch, long array, long schema);

  private static native byte[] importBatchNative(byte[][] contexts, long array, long schema);

  private static native byte[] remapBatchNative(byte[] batch, byte[][] contexts);

  private static native Capture createCaptureNative(Dataset dataset, byte[] context, int fieldId);

  private static native byte[] captureNative(
      Capture capture, byte[] scanSource, String column, long array, long schema)
      throws IOException;

  private static native byte[] captureNestedNative(
      Capture capture, byte[] scanSource, String column, long array, long schema)
      throws IOException;

  private static native byte[] validateNestedBatchNative(byte[] encoded);

  private static native byte[] importNestedBatchNative(byte[] references, long array, long schema);

  private static native byte[] nestedReferencesNative(byte[] encoded);

  private static native void exportNestedIndicesNative(byte[] encoded, long array, long schema);

  private static native Map<String, Long> metricsNative(Reader reader);

  private static native DatasetBinding bindDatasetNative(Dataset dataset);

  private static native Reader createReaderNative(
      byte[][] contexts,
      DatasetBinding[] datasets,
      long ioBufferSizeBytes,
      long materializationReadaheadBytes);

  private static native ReadStream readNative(
      Reader reader,
      byte[] batch,
      long[] referenceIndices,
      long[] requestIndices,
      boolean[] hasRanges,
      long[] offsets,
      long[] lengths);

  private static native boolean nextNative(ReadStream stream, long array, long schema)
      throws IOException;

  private static native void cancelNative(ReadStream stream);

  private static native void closeNative(NativeResource resource) throws IOException;

  private static native long reservedBytesNative(Reader reader, boolean io);

  private static native byte[] snapshotIdentityNative(byte[] context);
}
