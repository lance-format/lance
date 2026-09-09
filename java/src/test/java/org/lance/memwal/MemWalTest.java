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
package org.lance.memwal;

import org.lance.Dataset;
import org.lance.index.DistanceType;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.vector.VectorIndexParams;
import org.lance.merge.MergeInsertParams;
import org.lance.merge.MergeInsertResult;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.FixedSizeListVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.channels.Channels;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Integration tests for the MemWAL Java bindings, mirroring python/python/tests/test_mem_wal.py.
 */
public class MemWalTest {
  private static final Map<String, String> PK_META =
      Collections.singletonMap("lance-schema:unenforced-primary-key", "true");

  private static final Schema LOOKUP_SCHEMA =
      new Schema(
          Arrays.asList(
              new Field(
                  "id", new FieldType(false, new ArrowType.Int(64, true), null, PK_META), null),
              Field.nullable("name", new ArrowType.Utf8())));
  private static final Schema APPEND_ONLY_SCHEMA =
      new Schema(
          Arrays.asList(
              new Field(
                  "id",
                  new FieldType(false, new ArrowType.Int(64, true), null, Collections.emptyMap()),
                  null),
              Field.nullable("name", new ArrowType.Utf8())));

  /** Build a single-batch root where {@code name = "{prefix}_{id}"}. */
  private static VectorSchemaRoot lookupRoot(BufferAllocator allocator, long[] ids, String prefix) {
    VectorSchemaRoot root = VectorSchemaRoot.create(LOOKUP_SCHEMA, allocator);
    BigIntVector idVector = (BigIntVector) root.getVector("id");
    VarCharVector nameVector = (VarCharVector) root.getVector("name");
    idVector.allocateNew(ids.length);
    nameVector.allocateNew();
    for (int i = 0; i < ids.length; i++) {
      idVector.set(i, ids[i]);
      nameVector.setSafe(i, (prefix + "_" + ids[i]).getBytes(StandardCharsets.UTF_8));
    }
    root.setRowCount(ids.length);
    return root;
  }

  /** Build a single-batch root carrying only the {@code id} primary key, for deletes. */
  private static VectorSchemaRoot keysRoot(BufferAllocator allocator, long[] ids) {
    VectorSchemaRoot root =
        VectorSchemaRoot.create(
            new Schema(
                Collections.singletonList(Field.nullable("id", new ArrowType.Int(64, true)))),
            allocator);
    BigIntVector idVector = (BigIntVector) root.getVector("id");
    idVector.allocateNew(ids.length);
    for (int i = 0; i < ids.length; i++) {
      idVector.set(i, ids[i]);
    }
    root.setRowCount(ids.length);
    return root;
  }

  /** Build a single-batch append-only root without primary-key metadata. */
  private static VectorSchemaRoot appendOnlyRoot(
      BufferAllocator allocator, long[] ids, String prefix) {
    VectorSchemaRoot root = VectorSchemaRoot.create(APPEND_ONLY_SCHEMA, allocator);
    BigIntVector idVector = (BigIntVector) root.getVector("id");
    VarCharVector nameVector = (VarCharVector) root.getVector("name");
    idVector.allocateNew(ids.length);
    nameVector.allocateNew();
    for (int i = 0; i < ids.length; i++) {
      idVector.set(i, ids[i]);
      nameVector.setSafe(i, (prefix + "_" + ids[i]).getBytes(StandardCharsets.UTF_8));
    }
    root.setRowCount(ids.length);
    return root;
  }

  /** Wrap an in-memory root into an {@link ArrowReader} via the Arrow IPC stream format. */
  private static ArrowReader toReader(BufferAllocator allocator, VectorSchemaRoot root)
      throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, Channels.newChannel(out))) {
      writer.start();
      writer.writeBatch();
      writer.end();
    }
    return new ArrowStreamReader(new ByteArrayInputStream(out.toByteArray()), allocator);
  }

  /** Write a base dataset of `(id, name)` rows at {@code path}. */
  private static Dataset writeLookupDataset(
      BufferAllocator allocator, String path, long[] ids, String prefix) throws Exception {
    try (VectorSchemaRoot root = lookupRoot(allocator, ids, prefix);
        ArrowReader reader = toReader(allocator, root)) {
      return Dataset.write().allocator(allocator).reader(reader).uri(path).execute();
    }
  }

  /** Write an append-only base dataset of `(id, name)` rows at {@code path}. */
  private static Dataset writeAppendOnlyDataset(
      BufferAllocator allocator, String path, long[] ids, String prefix) throws Exception {
    try (VectorSchemaRoot root = appendOnlyRoot(allocator, ids, prefix);
        ArrowReader reader = toReader(allocator, root)) {
      return Dataset.write().allocator(allocator).reader(reader).uri(path).execute();
    }
  }

  /**
   * Stage a <em>faithful</em> SSTable at {@code genPath}: the Lance dataset plus its primary-key
   * dedup sidecar ({@code _pk_index/}), mirroring what production flush emits. The LSM scanner's
   * cross-generation block-list opens the sidecar, so a dataset alone (no sidecar) is not a state
   * production produces. Mirrors the Python {@code _write_sstable} test helper.
   */
  private static void writeSsTable(
      BufferAllocator allocator, String genPath, long[] ids, String prefix) throws Exception {
    writeLookupDataset(allocator, genPath, ids, prefix).close();
    try (VectorSchemaRoot root = lookupRoot(allocator, ids, prefix);
        ArrowReader reader = toReader(allocator, root);
        ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
      Data.exportArrayStream(allocator, reader, stream);
      nativeWritePkSidecar(genPath, stream.memoryAddress(), Collections.singletonList("id"));
    }
  }

  /**
   * Test-support native: write the primary-key dedup sidecar for an SSTable dataset already staged
   * at {@code genPath}. See {@link #writeSsTable}.
   */
  private static native void nativeWritePkSidecar(
      String genPath, long streamAddress, List<String> pkColumns);

  /** Read an LSM scanner fully into an {@code id -> name} map. */
  private static Map<Long, String> readByName(ArrowReader reader) throws Exception {
    Map<Long, String> byId = new HashMap<>();
    while (reader.loadNextBatch()) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      BigIntVector idVector = (BigIntVector) root.getVector("id");
      VarCharVector nameVector = (VarCharVector) root.getVector("name");
      for (int i = 0; i < root.getRowCount(); i++) {
        byId.put(idVector.get(i), new String(nameVector.get(i), StandardCharsets.UTF_8));
      }
    }
    return byId;
  }

  @Test
  void testMemWalIndexDetailsNoneBeforeInit(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("base").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, path, new long[] {1, 2, 3}, "base")) {
      assertFalse(dataset.memWalIndexDetails().isPresent());
    }
  }

  @Test
  void testInitializeMemWalUnsharded(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("base").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, path, new long[] {1, 2, 3}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams().withUnsharded());

      Optional<MemWalIndexDetails> details = dataset.memWalIndexDetails();
      assertTrue(details.isPresent());
      assertEquals(Collections.emptyList(), details.get().maintainedIndexes());
    }
  }

  @Test
  void testInitializeMemWalBucketShardingWithoutPrimaryKey(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("append_only").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeAppendOnlyDataset(allocator, path, new long[] {1, 2, 3}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams().withBucketSharding("id", 4));

      Optional<MemWalIndexDetails> details = dataset.memWalIndexDetails();
      assertTrue(details.isPresent());
      assertEquals(4L, details.get().numShards());
      ShardingField field = details.get().shardingSpecs().get(0).fields().get(0);
      assertEquals("bucket", field.transform().get());
      assertEquals("4", field.parameters().get("num_buckets"));
    }
  }

  @Test
  void testInitializeMemWalBucketShardingUsesConfiguredColumn(@TempDir Path tempDir)
      throws Exception {
    String path = tempDir.resolve("base").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, path, new long[] {1, 2, 3}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams().withBucketSharding("name", 4));

      MemWalIndexDetails details = dataset.memWalIndexDetails().get();
      ShardingField field = details.shardingSpecs().get(0).fields().get(0);
      int nameFieldId =
          dataset.getLanceSchema().fields().stream()
              .filter(f -> f.getName().equals("name"))
              .findFirst()
              .get()
              .getId();
      assertEquals("bucket", field.transform().get());
      assertEquals(nameFieldId, field.sourceIds().get(0));
    }
  }

  @Test
  void testShardingEvaluatorBucketAndIdentity(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("append_only").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeAppendOnlyDataset(allocator, path, new long[] {1}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams().withBucketSharding("id", 4));
      ShardingSpec bucketSpec = dataset.memWalIndexDetails().get().shardingSpecs().get(0);
      ShardingField bucketField = bucketSpec.fields().get(0);

      try (VectorSchemaRoot root = appendOnlyRoot(allocator, new long[] {1, 2, 3}, "eval");
          ArrowReader reader =
              ShardingEvaluator.evaluate(allocator, root, bucketSpec, dataset.getLanceSchema())) {
        assertTrue(reader.loadNextBatch());
        VectorSchemaRoot result = reader.getVectorSchemaRoot();
        IntVector buckets = (IntVector) result.getVector(bucketField.fieldId());
        assertEquals(3, result.getRowCount());
        assertEquals(0, buckets.get(0));
        assertEquals(0, buckets.get(1));
        assertEquals(3, buckets.get(2));
        assertFalse(reader.loadNextBatch());
      }

      int nameFieldId =
          dataset.getLanceSchema().fields().stream()
              .filter(f -> f.getName().equals("name"))
              .findFirst()
              .get()
              .getId();
      ShardingSpec identitySpec =
          new ShardingSpec(
              7,
              Collections.singletonList(
                  new ShardingField(
                      "name_identity",
                      Collections.singletonList(nameFieldId),
                      "identity",
                      null,
                      "utf8",
                      Collections.emptyMap())));
      try (VectorSchemaRoot root = appendOnlyRoot(allocator, new long[] {1}, "eval");
          ArrowReader reader =
              ShardingEvaluator.evaluate(allocator, root, identitySpec, dataset.getLanceSchema())) {
        assertTrue(reader.loadNextBatch());
        VarCharVector names =
            (VarCharVector) reader.getVectorSchemaRoot().getVector("name_identity");
        assertEquals("eval_1", new String(names.get(0), StandardCharsets.UTF_8));
        assertFalse(reader.loadNextBatch());
      }

      Map<String, String> stringBucketParameters = new HashMap<>();
      stringBucketParameters.put("column", "key");
      stringBucketParameters.put("num_buckets", "8");
      ShardingSpec stringBucketSpec =
          new ShardingSpec(
              8,
              Collections.singletonList(
                  new ShardingField(
                      "key_bucket",
                      Collections.emptyList(),
                      "bucket",
                      null,
                      "int32",
                      stringBucketParameters)));
      Schema stringSchema =
          new Schema(Collections.singletonList(Field.nullable("key", new ArrowType.Utf8())));
      try (VectorSchemaRoot root = VectorSchemaRoot.create(stringSchema, allocator)) {
        VarCharVector keyVector = (VarCharVector) root.getVector("key");
        keyVector.allocateNew();
        keyVector.setSafe(0, "a".getBytes(StandardCharsets.UTF_8));
        keyVector.setSafe(1, "b".getBytes(StandardCharsets.UTF_8));
        keyVector.setNull(2);
        root.setRowCount(3);
        try (ArrowReader reader = ShardingEvaluator.evaluate(allocator, root, stringBucketSpec)) {
          assertTrue(reader.loadNextBatch());
          IntVector buckets = (IntVector) reader.getVectorSchemaRoot().getVector("key_bucket");
          assertEquals(1, buckets.get(0));
          assertEquals(5, buckets.get(1));
          assertEquals(0, buckets.get(2));
          assertFalse(reader.loadNextBatch());
        }
      }
    }
  }

  @Test
  void testInitializeMemWalRejectsConflictingSharding(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("base").toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, path, new long[] {1}, "base")) {
      InitializeMemWalParams params =
          new InitializeMemWalParams().withUnsharded().withBucketSharding("id", 4);
      assertThrows(IllegalArgumentException.class, () -> dataset.initializeMemWal(params));
    }
  }

  @Test
  void testShardWriterPutAndLsmScanner(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("base").toString();
    String shardId = UUID.randomUUID().toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, path, new long[] {0}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams());

      ShardWriterConfig config =
          new ShardWriterConfig()
              .withDurableWrite(true)
              .withMaxWalBufferSize(1)
              .withMaxWalFlushIntervalMs(10)
              .withMaxMemtableBatches(1);

      try (ShardWriter writer = dataset.memWalWriter(shardId, config)) {
        assertEquals(shardId, writer.shardId());

        try (VectorSchemaRoot root = lookupRoot(allocator, new long[] {1, 2}, "writer");
            ArrowReader reader = toReader(allocator, root)) {
          writer.put(reader);
        }

        Map<Long, String> byId = Collections.emptyMap();
        long deadline = System.currentTimeMillis() + 10_000;
        while (System.currentTimeMillis() < deadline) {
          try (LsmScanner scanner = writer.lsmScanner();
              ArrowReader reader = scanner.scanBatches()) {
            byId = readByName(reader);
          }
          if ("writer_1".equals(byId.get(1L)) && "writer_2".equals(byId.get(2L))) {
            break;
          }
          Thread.sleep(50);
        }
        assertEquals("writer_1", byId.get(1L), "writer.lsmScanner() must see written rows");
        assertEquals("writer_2", byId.get(2L), "writer.lsmScanner() must see written rows");

        WriteStats stats = writer.stats();
        assertEquals(1, stats.putCount());

        MemTableStats memtableStats = writer.memtableStats();
        assertTrue(memtableStats.generation() >= 0);
      }
    }
  }

  @Test
  void testShardWriterDeleteMasksBaseRow(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("base").toString();
    String shardId = UUID.randomUUID().toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, path, new long[] {1, 2, 3}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams());

      ShardWriterConfig config =
          new ShardWriterConfig()
              .withDurableWrite(true)
              .withMaxWalBufferSize(1)
              .withMaxWalFlushIntervalMs(10);

      try (ShardWriter writer = dataset.memWalWriter(shardId, config)) {
        try (VectorSchemaRoot root = lookupRoot(allocator, new long[] {4}, "writer");
            ArrowReader reader = toReader(allocator, root)) {
          writer.put(reader);
        }
        try (VectorSchemaRoot keys = keysRoot(allocator, new long[] {2});
            ArrowReader reader = toReader(allocator, keys)) {
          writer.delete(reader);
        }

        Map<Long, String> byId = Collections.emptyMap();
        long deadline = System.currentTimeMillis() + 10_000;
        while (System.currentTimeMillis() < deadline) {
          try (LsmScanner scanner = writer.lsmScanner();
              ArrowReader reader = scanner.scanBatches()) {
            byId = readByName(reader);
          }
          if (!byId.containsKey(2L) && "writer_4".equals(byId.get(4L))) {
            break;
          }
          Thread.sleep(50);
        }

        assertEquals("base_1", byId.get(1L));
        assertFalse(byId.containsKey(2L), "deleted base row should be masked by the tombstone");
        assertEquals("base_3", byId.get(3L));
        assertEquals("writer_4", byId.get(4L));
      }
    }
  }

  @Test
  void testLsmScannerFromSnapshots(@TempDir Path tempDir) throws Exception {
    String basePath = tempDir.resolve("base").toString();
    String shardId = UUID.randomUUID().toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, basePath, new long[] {1, 2, 3}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams());

      // SSTable overwrites id=2.
      String genPath = basePath + "/_mem_wal/" + shardId + "/gen_1";
      writeSsTable(allocator, genPath, new long[] {2}, "gen1");

      ShardSnapshot snapshot =
          new ShardSnapshot(shardId).withSsTable(1, "gen_1").withCurrentGeneration(2);

      try (LsmScanner scanner =
              LsmScanner.fromSnapshots(dataset, Collections.singletonList(snapshot));
          ArrowReader reader = scanner.scanBatches()) {
        Map<Long, String> byId = readByName(reader);
        assertEquals(3, byId.size(), "Expected 3 deduplicated rows");
        assertEquals("base_1", byId.get(1L));
        assertEquals("gen1_2", byId.get(2L), "SSTable must win over base");
        assertEquals("base_3", byId.get(3L));
      }

      try (LsmScanner scanner =
              LsmScanner.fromSnapshots(dataset, Collections.singletonList(snapshot))
                  .limit(Optional.empty(), Optional.of(1L));
          ArrowReader reader = scanner.scanBatches()) {
        Map<Long, String> byId = readByName(reader);
        assertEquals(2, byId.size(), "Offset-only LSM scan should not require a limit");
      }
    }
  }

  @Test
  void testPointLookup(@TempDir Path tempDir) throws Exception {
    String basePath = tempDir.resolve("base").toString();
    String shardId = UUID.randomUUID().toString();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = writeLookupDataset(allocator, basePath, new long[] {1, 2, 3}, "base")) {
      dataset.initializeMemWal(new InitializeMemWalParams());

      String genPath = basePath + "/_mem_wal/" + shardId + "/gen_1";
      writeSsTable(allocator, genPath, new long[] {2}, "gen1");

      ShardSnapshot snapshot =
          new ShardSnapshot(shardId).withSsTable(1, "gen_1").withCurrentGeneration(2);

      try (LsmPointLookupPlanner planner =
          new LsmPointLookupPlanner(dataset, Collections.singletonList(snapshot))) {
        // id=2 must resolve to the SSTable value.
        assertEquals("gen1_2", lookup(planner, allocator, 2L));
        // id=1 only exists in the base table.
        assertEquals("base_1", lookup(planner, allocator, 1L));
        // id=99 does not exist.
        assertEquals(null, lookup(planner, allocator, 99L));
      }
    }
  }

  /** Run a single-key point lookup and return the resolved name, or {@code null} if absent. */
  private static String lookup(LsmPointLookupPlanner planner, BufferAllocator allocator, long id)
      throws Exception {
    try (BigIntVector pkVector = new BigIntVector("id", allocator)) {
      pkVector.allocateNew(1);
      pkVector.set(0, id);
      pkVector.setValueCount(1);
      try (ExecutionPlan plan = planner.planLookup(pkVector);
          ArrowReader reader = plan.toReader()) {
        Map<Long, String> byId = readByName(reader);
        return byId.get(id);
      }
    }
  }

  @Test
  void testMergeInsertMarkSstablesAsCompacted(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("base").toString();
    String shardId = UUID.randomUUID().toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      Dataset dataset = writeLookupDataset(allocator, path, new long[] {1, 2, 3}, "base");
      try {
        dataset.initializeMemWal(new InitializeMemWalParams());

        MergeInsertParams params =
            new MergeInsertParams(Collections.singletonList("id"))
                .withMatchedUpdateAll()
                .withNotMatched(MergeInsertParams.WhenNotMatched.InsertAll)
                .markSstablesAsCompacted(
                    Collections.singletonList(new CompactedSsTable(shardId, 1)));

        try (VectorSchemaRoot root = lookupRoot(allocator, new long[] {2, 4}, "merged");
            ArrowReader reader = toReader(allocator, root);
            ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
          Data.exportArrayStream(allocator, reader, stream);
          MergeInsertResult result = dataset.mergeInsert(params, stream);
          Dataset merged = result.dataset();
          try {
            assertEquals(4, merged.countRows(), "merge insert should yield 4 rows");
          } finally {
            merged.close();
          }
        }
      } finally {
        dataset.close();
      }
    }
  }

  private static final int VDIM = 32;

  private static Schema vectorSchema() {
    Field idField =
        new Field("id", new FieldType(false, new ArrowType.Int(32, true), null, PK_META), null);
    Field vecField =
        new Field(
            "vec",
            FieldType.nullable(new ArrowType.FixedSizeList(VDIM)),
            Collections.singletonList(
                new Field(
                    "item",
                    FieldType.nullable(new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)),
                    null)));
    return new Schema(Arrays.asList(idField, vecField));
  }

  @Test
  void testVectorSearch(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("vec").toString();
    int rows = 400;
    try (BufferAllocator allocator = new RootAllocator()) {
      Schema schema = vectorSchema();
      Dataset dataset;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        IntVector idVector = (IntVector) root.getVector("id");
        FixedSizeListVector vecVector = (FixedSizeListVector) root.getVector("vec");
        Float4Vector items = (Float4Vector) vecVector.getDataVector();
        idVector.allocateNew(rows);
        vecVector.allocateNew();
        for (int i = 0; i < rows; i++) {
          idVector.set(i, i);
          vecVector.setNotNull(i);
          for (int d = 0; d < VDIM; d++) {
            items.setSafe(i * VDIM + d, (float) (i * VDIM + d));
          }
        }
        root.setRowCount(rows);
        try (ArrowReader reader = toReader(allocator, root)) {
          dataset = Dataset.write().allocator(allocator).reader(reader).uri(path).execute();
        }
      }

      try {
        IndexParams params =
            IndexParams.builder()
                .setVectorIndexParams(VectorIndexParams.ivfPq(2, 8, 2, DistanceType.L2, 2))
                .build();
        dataset.createIndex(
            Collections.singletonList("vec"),
            IndexType.VECTOR,
            Optional.of("vec_idx"),
            params,
            true);
        dataset.initializeMemWal(
            new InitializeMemWalParams()
                .withMaintainedIndexes(Collections.singletonList("vec_idx")));

        try (LsmVectorSearchPlanner planner =
            new LsmVectorSearchPlanner(dataset, Collections.emptyList(), "vec")) {
          int[] queryIds = {10, 50, 100, 200, 300, 350};
          int found = 0;
          for (int queryId : queryIds) {
            try (Float4Vector query = new Float4Vector("q", allocator)) {
              query.allocateNew(VDIM);
              for (int d = 0; d < VDIM; d++) {
                query.set(d, (float) (queryId * VDIM + d));
              }
              query.setValueCount(VDIM);
              try (ExecutionPlan plan = planner.planSearch(query, 10);
                  ArrowReader reader = plan.toReader()) {
                while (reader.loadNextBatch()) {
                  VectorSchemaRoot result = reader.getVectorSchemaRoot();
                  IntVector ids = (IntVector) result.getVector("id");
                  for (int i = 0; i < result.getRowCount(); i++) {
                    if (ids.get(i) == queryId) {
                      found++;
                    }
                  }
                }
              }
            }
          }
          double recall = (double) found / queryIds.length;
          assertTrue(recall >= 0.5, "vector search recall too low: " + recall);
        }

        try (LsmVectorSearchPlanner planner =
                new LsmVectorSearchPlanner(
                    dataset, Collections.emptyList(), "vec", null, null, "id >= 200");
            Float4Vector query = new Float4Vector("q", allocator)) {
          int filteredQueryId = 250;
          query.allocateNew(VDIM);
          for (int d = 0; d < VDIM; d++) {
            query.set(d, (float) (filteredQueryId * VDIM + d));
          }
          query.setValueCount(VDIM);

          int filteredRows = 0;
          boolean foundFilteredNearest = false;
          try (ExecutionPlan plan =
                  planner.planSearch(query, 20, 2, Collections.singletonList("id"), false);
              ArrowReader reader = plan.toReader()) {
            while (reader.loadNextBatch()) {
              VectorSchemaRoot result = reader.getVectorSchemaRoot();
              IntVector ids = (IntVector) result.getVector("id");
              for (int i = 0; i < result.getRowCount(); i++) {
                int id = ids.get(i);
                assertTrue(id >= 200, "filtered vector search returned id " + id);
                if (id == filteredQueryId) {
                  foundFilteredNearest = true;
                }
                filteredRows++;
              }
            }
          }
          assertTrue(filteredRows > 0, "filtered vector search should return rows");
          assertTrue(
              foundFilteredNearest, "filtered vector search recall missed id " + filteredQueryId);
        }
      } finally {
        dataset.close();
      }
    }
  }
}
