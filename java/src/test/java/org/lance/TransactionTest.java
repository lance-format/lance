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

import org.lance.index.Index;
import org.lance.index.IndexFile;
import org.lance.index.IndexOptions;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.scalar.ScalarIndexParams;
import org.lance.memwal.CompactedSsTable;
import org.lance.memwal.InitializeMemWalParams;
import org.lance.operation.Append;
import org.lance.operation.Clone;
import org.lance.operation.CreateIndex;
import org.lance.operation.DataOverlay;
import org.lance.operation.DataReplacement;
import org.lance.operation.Delete;
import org.lance.operation.KeyExistenceFilter;
import org.lance.operation.Operation;
import org.lance.operation.Overwrite;
import org.lance.operation.ReserveFragments;
import org.lance.operation.Restore;
import org.lance.operation.Rewrite;
import org.lance.operation.RewrittenIndex;
import org.lance.operation.Update;
import org.lance.operation.UpdateBases;
import org.lance.operation.UpdateMemWalState;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.beans.Introspector;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TransactionTest {

  private static final byte[] OFFSETS_1_3_5 =
      new byte[] {0x3A, 0x30, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0x10, 0, 0, 0, 1, 0, 3, 0, 5, 0};

  @Test
  public void testReadTransactionCreateIndex(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("read_transaction_create_index").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
      }

      try (Dataset dataset = testDataset.write(1, 10)) {
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();

        dataset.createIndex(
            IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, indexParams)
                .withIndexName("btree_id_index")
                .build());

        assertTrue(
            dataset.listIndexes().contains("btree_id_index"),
            "Expected 'btree_id_index' to be created");

        Transaction readTx = dataset.readTransaction().orElse(null);
        assertNotNull(readTx, "readTransaction() should return a transaction for CreateIndex");
        assertEquals("CreateIndex", readTx.operation().name());

        assertInstanceOf(CreateIndex.class, readTx.operation());
        CreateIndex op = (CreateIndex) readTx.operation();
        assertFalse(op.getNewIndices().isEmpty(), "newIndices should not be empty for CreateIndex");
        assertTrue(
            op.getRemovedIndices().isEmpty(), "removedIndices should be empty for CreateIndex");
        assertEquals("btree_id_index", (op.getNewIndices().get(0).name()));
        Index readIndex = op.getNewIndices().get(0);
        assertTrue(readIndex.getFiles().isPresent());
        assertEquals(
            readIndex.getSizeBytes().orElseThrow().longValue(),
            readIndex.getFiles().orElseThrow().stream().mapToLong(IndexFile::getSizeBytes).sum());
      }
    }
  }

  @Test
  public void testCommitToUri(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testCommitToUri").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      Schema schema = testDataset.getSchema();

      // Create fragments at the dataset path
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(20);

      // Build a transaction targeting a URI (no existing dataset)
      try (Transaction txn =
          new Transaction.Builder()
              .operation(
                  Overwrite.builder()
                      .fragments(Collections.singletonList(fragmentMeta))
                      .schema(schema)
                      .build())
              .build()) {
        try (Dataset committedDataset = new CommitBuilder(datasetPath, allocator).execute(txn)) {
          assertEquals(1, committedDataset.version());
          assertEquals(20, committedDataset.countRows());
          try (Transaction read = committedDataset.readTransaction().orElseThrow()) {
            Overwrite operation = assertInstanceOf(Overwrite.class, read.operation());
            assertTrue(operation.getInitialBases().isEmpty());
            assertTrue(operation.configUpsertValues().isEmpty());
          }
        }
      }
    }
  }

  @Test
  public void testTagRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testTagRoundTrip").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);

        try (Transaction txn =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .tag("v1.0")
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .build()) {
          assertEquals("v1.0", txn.tag().orElse(null));

          try (Dataset committed = new CommitBuilder(dataset).execute(txn)) {
            Transaction readTx = committed.readTransaction().orElse(null);
            assertNotNull(readTx);
            assertEquals("v1.0", readTx.tag().orElse(null));
          }
        }
      }
    }
  }

  @Test
  public void testEmptyTransactionMetadataCanonicalization(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("empty_transaction_metadata").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      FragmentMetadata fragment = testDataset.createNewFragment(1);
      try (Dataset dataset = testDataset.createEmptyDataset();
          Transaction transaction =
              new Transaction.Builder()
                  .readVersion(dataset.version())
                  .tag("")
                  .transactionProperties(Collections.emptyMap())
                  .operation(
                      Append.builder().fragments(Collections.singletonList(fragment)).build())
                  .build()) {
        assertTrue(transaction.tag().isEmpty());
        assertTrue(transaction.transactionProperties().isEmpty());
        try (Dataset committed = new CommitBuilder(dataset).execute(transaction);
            Transaction read = committed.readTransaction().orElseThrow()) {
          assertTrue(read.tag().isEmpty());
          assertTrue(read.transactionProperties().isEmpty());
        }
      }
    }
  }

  @Test
  public void testTransactionPropertiesRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testTransactionPropertiesRoundTrip").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);

        Map<String, String> properties = new HashMap<>();
        properties.put("source", "ingestion-pipeline");
        properties.put("batchId", "42");

        try (Transaction txn =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .transactionProperties(properties)
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .build()) {
          try (Dataset committed = new CommitBuilder(dataset).execute(txn)) {
            Transaction readTx = committed.readTransaction().orElse(null);
            assertNotNull(readTx);
            Map<String, String> readProps = readTx.transactionProperties().orElse(null);
            assertNotNull(readProps);
            assertEquals("ingestion-pipeline", readProps.get("source"));
            assertEquals("42", readProps.get("batchId"));
          }
        }
      }
    }
  }

  @Test
  public void testCustomUuid(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testCustomUuid").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);

        String customUuid = "custom-uuid-12345";
        try (Transaction txn =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .uuid(customUuid)
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .build()) {
          assertEquals(customUuid, txn.uuid());

          try (Dataset committed = new CommitBuilder(dataset).execute(txn)) {
            Transaction readTx = committed.readTransaction().orElse(null);
            assertNotNull(readTx);
            assertEquals(customUuid, readTx.uuid());
          }
        }
      }
    }
  }

  @Test
  public void testCloneRoundTrip(@TempDir Path tempDir) {
    String sourcePath = tempDir.resolve("clone_source").toString();
    String targetPath = tempDir.resolve("clone_target").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset sourceDataset =
          new TestUtils.SimpleTestDataset(allocator, sourcePath);
      sourceDataset.createEmptyDataset().close();
      try (Dataset source = sourceDataset.write(1, 10);
          Transaction transaction =
              new Transaction.Builder()
                  .readVersion(source.version())
                  .operation(
                      Clone.builder()
                          .shallow(false)
                          .refVersion(source.version())
                          .refPath(sourcePath)
                          .build())
                  .build();
          Dataset cloned = new CommitBuilder(targetPath, allocator).execute(transaction);
          Transaction read = cloned.readTransaction().orElseThrow()) {
        assertInstanceOf(Clone.class, read.operation());
        Clone clone = (Clone) read.operation();
        assertEquals(
            Clone.builder().shallow(false).refVersion(source.version()).refPath(sourcePath).build(),
            clone);
        assertFalse(clone.isShallow());
        assertEquals(sourcePath, clone.getRefPath());
        assertEquals(source.version(), clone.getRefVersion());
        assertTrue(clone.getRefName().isEmpty());
        assertTrue(clone.getBranchName().isEmpty());
      }
    }
  }

  @Test
  public void testUpdateBasesRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("update_bases").toString();
    String basePath = tempDir.resolve("external_base").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset();
          Transaction transaction =
              new Transaction.Builder()
                  .readVersion(dataset.version())
                  .operation(
                      UpdateBases.builder()
                          .newBases(
                              Collections.singletonList(
                                  new BasePath(0, Optional.of("external"), basePath, false)))
                          .build())
                  .build();
          Dataset committed = new CommitBuilder(dataset).execute(transaction);
          Transaction read = committed.readTransaction().orElseThrow()) {
        UpdateBases operation = assertInstanceOf(UpdateBases.class, read.operation());
        assertEquals(
            UpdateBases.builder()
                .newBases(
                    Collections.singletonList(
                        new BasePath(0, Optional.of("external"), basePath, false)))
                .build(),
            operation);
        assertEquals(1, operation.getNewBases().size());
        assertEquals("external", operation.getNewBases().get(0).getName().orElseThrow());
        assertEquals(basePath, operation.getNewBases().get(0).getPath());
      }
    }
  }

  @Test
  public void testUpdateMemWalStateRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("update_memwal").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        dataset.initializeMemWal(new InitializeMemWalParams().withUnsharded());
        CompactedSsTable sstable = new CompactedSsTable(UUID.randomUUID().toString(), 7);
        try (Transaction transaction =
                new Transaction.Builder()
                    .readVersion(dataset.version())
                    .operation(
                        UpdateMemWalState.builder()
                            .compactedSstables(Collections.singletonList(sstable))
                            .build())
                    .build();
            Dataset committed = new CommitBuilder(dataset).execute(transaction);
            Transaction read = committed.readTransaction().orElseThrow()) {
          UpdateMemWalState operation = assertInstanceOf(UpdateMemWalState.class, read.operation());
          assertEquals(
              UpdateMemWalState.builder()
                  .compactedSstables(Collections.singletonList(sstable))
                  .build(),
              operation);
          assertEquals(1, operation.getCompactedSstables().size());
          assertEquals(sstable.getShardId(), operation.getCompactedSstables().get(0).getShardId());
          assertEquals(7, operation.getCompactedSstables().get(0).getGeneration());
        }
      }
    }
  }

  @Test
  public void testDataOverlayRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("data_overlay").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 10)) {
        FragmentMetadata fragment = dataset.getFragments().get(0).metadata();
        DataOverlay.DataOverlayFile overlay =
            DataOverlay.DataOverlayFile.forCommit(
                fragment.getFiles().get(0), DataOverlay.OverlayCoverage.shared(OFFSETS_1_3_5));
        DataOverlay.DataOverlayGroup group =
            new DataOverlay.DataOverlayGroup(fragment.getId(), Collections.singletonList(overlay));
        try (Transaction transaction =
                new Transaction.Builder()
                    .readVersion(dataset.version())
                    .operation(
                        DataOverlay.builder().groups(Collections.singletonList(group)).build())
                    .build();
            Dataset committed = new CommitBuilder(dataset).execute(transaction);
            Transaction read = committed.readTransaction().orElseThrow()) {
          DataOverlay operation = assertInstanceOf(DataOverlay.class, read.operation());
          assertEquals(
              DataOverlay.builder().groups(Collections.singletonList(group)).build(), operation);
          DataOverlay.DataOverlayFile readOverlay =
              operation.getGroups().get(0).getOverlays().get(0);
          assertEquals(fragment.getId(), operation.getGroups().get(0).getFragmentId());
          assertTrue(readOverlay.getCoverage().isShared());
          assertArrayEquals(OFFSETS_1_3_5, readOverlay.getCoverage().getBitmaps().get(0));
          assertEquals(0, readOverlay.getCommittedVersion());
        }
      }
    }
  }

  @Test
  public void testDataOverlayPerFieldCoverageValidation(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("data_overlay_per_field").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 10)) {
        FragmentMetadata fragment = dataset.getFragments().get(0).metadata();
        org.lance.fragment.DataFile dataFile = fragment.getFiles().get(0);
        int fieldCount = dataFile.getFields().length;
        assertTrue(fieldCount > 0);

        assertThrows(
            IllegalArgumentException.class,
            () ->
                new DataOverlay.DataOverlayFile(
                    dataFile, DataOverlay.OverlayCoverage.perField(Collections.emptyList()), 0));
        assertThrows(
            IllegalArgumentException.class,
            () ->
                new DataOverlay.DataOverlayFile(
                    dataFile,
                    DataOverlay.OverlayCoverage.perField(
                        Collections.nCopies(fieldCount + 1, OFFSETS_1_3_5)),
                    0));

        DataOverlay.DataOverlayFile overlay =
            new DataOverlay.DataOverlayFile(
                dataFile,
                DataOverlay.OverlayCoverage.perField(
                    Collections.nCopies(fieldCount, OFFSETS_1_3_5)),
                0);
        DataOverlay.DataOverlayGroup group =
            new DataOverlay.DataOverlayGroup(fragment.getId(), Collections.singletonList(overlay));
        try (Transaction transaction =
                new Transaction.Builder()
                    .readVersion(dataset.version())
                    .operation(
                        DataOverlay.builder().groups(Collections.singletonList(group)).build())
                    .build();
            Dataset committed = new CommitBuilder(dataset).execute(transaction);
            Transaction read = committed.readTransaction().orElseThrow()) {
          DataOverlay readOperation = assertInstanceOf(DataOverlay.class, read.operation());
          assertEquals(
              fieldCount,
              readOperation
                  .getGroups()
                  .get(0)
                  .getOverlays()
                  .get(0)
                  .getCoverage()
                  .getBitmaps()
                  .size());
        }
      }
    }
  }

  @Test
  public void testOverwriteInitialBasesAndUpdateFilterRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("overwrite_initial_bases").toString();
    String basePath = tempDir.resolve("initial_base").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Transaction overwrite =
              new Transaction.Builder()
                  .operation(
                      Overwrite.builder()
                          .fragments(Collections.emptyList())
                          .schema(testDataset.getSchema())
                          .initialBases(
                              Collections.singletonList(
                                  new BasePath(0, Optional.of("initial"), basePath, false)))
                          .build())
                  .build();
          Dataset dataset = new CommitBuilder(datasetPath, allocator).execute(overwrite);
          Transaction readOverwrite = dataset.readTransaction().orElseThrow()) {
        Overwrite operation = assertInstanceOf(Overwrite.class, readOverwrite.operation());
        BasePath expectedBasePath = new BasePath(0, Optional.of("initial"), basePath, false);
        BasePath readBasePath = operation.getInitialBases().orElseThrow().get(0);
        assertEquals(expectedBasePath.getId(), readBasePath.getId());
        assertEquals(expectedBasePath.getName(), readBasePath.getName());
        assertEquals(expectedBasePath.getPath(), readBasePath.getPath());
        assertEquals(expectedBasePath.isDatasetRoot(), readBasePath.isDatasetRoot());

        dataset.initializeMemWal(new InitializeMemWalParams().withUnsharded());
        KeyExistenceFilter filter = KeyExistenceFilter.exact(new int[] {0}, new long[] {42});
        CompactedSsTable compacted = new CompactedSsTable(UUID.randomUUID().toString(), 8);
        try (Transaction update =
                new Transaction.Builder()
                    .readVersion(dataset.version())
                    .operation(
                        Update.builder()
                            .compactedSstables(Collections.singletonList(compacted))
                            .insertedRowsFilter(filter)
                            .updateMode(Optional.of(Update.UpdateMode.RewriteRows))
                            .build())
                    .build();
            Dataset committed = new CommitBuilder(dataset).execute(update);
            Transaction readUpdate = committed.readTransaction().orElseThrow()) {
          Update updateOperation = assertInstanceOf(Update.class, readUpdate.operation());
          KeyExistenceFilter readFilter = updateOperation.getInsertedRowsFilter().orElseThrow();
          assertArrayEquals(new int[] {0}, readFilter.getFieldIds());
          assertArrayEquals(new long[] {42}, readFilter.getExactKeyHashes());
          assertEquals(Update.UpdateMode.RewriteRows, updateOperation.updateMode().orElseThrow());
          CompactedSsTable readCompacted = updateOperation.getCompactedSstables().get(0);
          assertEquals(compacted.getShardId(), readCompacted.getShardId());
          assertEquals(compacted.getGeneration(), readCompacted.getGeneration());
        }
      }
    }
  }

  @Test
  public void testBloomFilterRoundTripAndValidation(@TempDir Path tempDir) {
    byte[] bitmap = new byte[32];
    bitmap[0] = 1;
    KeyExistenceFilter filter =
        KeyExistenceFilter.bloom(new int[] {0}, bitmap, bitmap.length * Byte.SIZE, 8192, 0.00057);

    assertThrows(
        IllegalArgumentException.class,
        () -> KeyExistenceFilter.bloom(new int[] {0}, bitmap, 0, 8192, 0.00057));
    assertThrows(
        IllegalArgumentException.class,
        () -> KeyExistenceFilter.bloom(new int[] {0}, bitmap, 8, 8192, 0.00057));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            KeyExistenceFilter.bloom(new int[] {0}, bitmap, bitmap.length * Byte.SIZE, 0, 0.00057));
    for (double probability : Arrays.asList(0.0, 1.0, Double.NaN, Double.POSITIVE_INFINITY)) {
      assertThrows(
          IllegalArgumentException.class,
          () ->
              KeyExistenceFilter.bloom(
                  new int[] {0}, bitmap, bitmap.length * Byte.SIZE, 8192, probability));
    }

    String datasetPath = tempDir.resolve("bloom_filter").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset();
          Transaction update =
              new Transaction.Builder()
                  .readVersion(dataset.version())
                  .operation(
                      Update.builder()
                          .insertedRowsFilter(filter)
                          .updateMode(Optional.of(Update.UpdateMode.RewriteRows))
                          .build())
                  .build();
          Dataset committed = new CommitBuilder(dataset).execute(update);
          Transaction read = committed.readTransaction().orElseThrow()) {
        assertEquals(
            filter,
            assertInstanceOf(Update.class, read.operation()).getInsertedRowsFilter().orElseThrow());
      }
    }
  }

  @Test
  public void testCompatibilityAccessorsAndConstructor() throws Exception {
    Overwrite.class.getDeclaredConstructor(List.class, Schema.class, Map.class);
    assertTrue(
        Arrays.stream(Introspector.getBeanInfo(Overwrite.class).getPropertyDescriptors())
            .anyMatch(property -> property.getName().equals("initialBases")));
    assertTrue(
        Arrays.stream(Introspector.getBeanInfo(Update.class).getPropertyDescriptors())
            .anyMatch(property -> property.getName().equals("compactedSstables")));
    assertTrue(
        Arrays.stream(Introspector.getBeanInfo(Update.class).getPropertyDescriptors())
            .anyMatch(property -> property.getName().equals("insertedRowsFilter")));
    assertTrue(Update.builder().build().updateMode().isEmpty());
    assertThrows(
        NullPointerException.class, () -> Update.builder().compactedSstables(null).build());
    Overwrite emptyOverwrite =
        Overwrite.builder()
            .fragments(Collections.emptyList())
            .schema(new Schema(Collections.emptyList()))
            .configUpsertValues(Collections.emptyMap())
            .initialBases(Collections.emptyList())
            .build();
    assertTrue(emptyOverwrite.configUpsertValues().isEmpty());
    assertTrue(emptyOverwrite.getInitialBases().isPresent());
    assertTrue(emptyOverwrite.getInitialBases().orElseThrow().isEmpty());
  }

  @Test
  public void testUnrepresentableEmptyUpdateModeAndExplicitEmptyInitialBasesAreRejected(
      @TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("unrepresentable_optionals").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertInvalidOperation(
            dataset, Update.builder().build(), "update.updateMode must be specified");

        Overwrite overwrite =
            Overwrite.builder()
                .fragments(Collections.emptyList())
                .schema(testDataset.getSchema())
                .initialBases(Collections.emptyList())
                .build();
        assertTrue(overwrite.getInitialBases().isPresent());
        try (Transaction transaction =
            new Transaction.Builder().readVersion(dataset.version()).operation(overwrite).build()) {
          IllegalArgumentException error =
              assertThrows(
                  IllegalArgumentException.class,
                  () -> new CommitBuilder(dataset).execute(transaction));
          assertTrue(error.getMessage().contains("register new bases"));
        }
      }
    }
  }

  @Test
  public void testIndexSizeMustBeRepresentable(@TempDir Path tempDir) {
    Index empty = indexMetadata("empty", 0L, Collections.emptyList());
    assertTrue(empty.getSizeBytes().isEmpty());
    assertTrue(empty.getFiles().isEmpty());
    Index consistent =
        indexMetadata(
            "consistent",
            30L,
            Arrays.asList(new IndexFile("a.idx", 10), new IndexFile("b.idx", 20)));
    assertEquals(30L, consistent.getSizeBytes().orElseThrow());
    assertEquals(2, consistent.getFiles().orElseThrow().size());
    Index derived =
        indexMetadata(
            "derived",
            null,
            Arrays.asList(new IndexFile("derived-a.idx", 10), new IndexFile("derived-b.idx", 20)));
    assertEquals(30L, derived.getSizeBytes().orElseThrow());

    String datasetPath = tempDir.resolve("index_files").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset();
          Transaction missingFiles =
              new Transaction.Builder()
                  .readVersion(dataset.version())
                  .operation(
                      CreateIndex.builder()
                          .withNewIndices(
                              Collections.singletonList(indexMetadata("missing", 10L, null)))
                          .build())
                  .build()) {
        IllegalArgumentException error =
            assertThrows(
                IllegalArgumentException.class,
                () -> new CommitBuilder(dataset).execute(missingFiles));
        assertTrue(error.getMessage().contains("without files"));
      }

      try (Dataset dataset = Dataset.open(datasetPath, allocator);
          Transaction mismatch =
              new Transaction.Builder()
                  .readVersion(dataset.version())
                  .operation(
                      CreateIndex.builder()
                          .withNewIndices(
                              Collections.singletonList(
                                  indexMetadata(
                                      "mismatch",
                                      11L,
                                      Collections.singletonList(new IndexFile("c.idx", 10)))))
                          .build())
                  .build()) {
        IllegalArgumentException error =
            assertThrows(
                IllegalArgumentException.class, () -> new CommitBuilder(dataset).execute(mismatch));
        assertTrue(error.getMessage().contains("does not match"));
      }

      try (Dataset dataset = Dataset.open(datasetPath, allocator)) {
        for (List<IndexFile> files :
            Arrays.asList(
                Arrays.asList(
                    new IndexFile("large-a.idx", Long.MAX_VALUE), new IndexFile("large-b.idx", 1)),
                Arrays.asList(
                    new IndexFile("overflow-a.idx", Long.MAX_VALUE),
                    new IndexFile("overflow-b.idx", Long.MAX_VALUE),
                    new IndexFile("overflow-c.idx", Long.MAX_VALUE)))) {
          IllegalArgumentException error =
              assertThrows(
                  IllegalArgumentException.class, () -> indexMetadata("too-large", null, files));
          assertTrue(error.getMessage().contains("cannot be represented"));
        }
      }
    }
  }

  private static Index indexMetadata(String name, Long sizeBytes, List<IndexFile> files) {
    return Index.builder()
        .uuid(UUID.randomUUID())
        .fields(Collections.singletonList(0))
        .coveringFields(Collections.emptyList())
        .name(name)
        .datasetVersion(1)
        .indexVersion(0)
        .sizeBytes(sizeBytes)
        .files(files)
        .indexType(IndexType.BTREE)
        .build();
  }

  @Test
  public void testInvalidUnsignedTransactionFieldsAreRejected(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("invalid_unsigned_fields").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        try (Transaction transaction =
            new Transaction.Builder().readVersion(-1).operation(Update.builder().build()).build()) {
          IllegalArgumentException error =
              assertThrows(
                  IllegalArgumentException.class,
                  () -> new CommitBuilder(dataset).execute(transaction));
          assertTrue(error.getMessage().contains("transaction.readVersion"));
        }

        for (long invalid : new long[] {-1, 1L << 32}) {
          for (boolean preserving : new boolean[] {false, true}) {
            Update.Builder update = Update.builder();
            if (preserving) {
              update.fieldsForPreservingFragBitmap(new long[] {invalid});
            } else {
              update.fieldsModified(new long[] {invalid});
            }
            try (Transaction transaction =
                new Transaction.Builder()
                    .readVersion(dataset.version())
                    .operation(update.build())
                    .build()) {
              IllegalArgumentException error =
                  assertThrows(
                      IllegalArgumentException.class,
                      () -> new CommitBuilder(dataset).execute(transaction));
              assertTrue(error.getMessage().contains("[0]"));
            }
          }
        }

        try (Transaction transaction =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .operation(
                    UpdateBases.builder()
                        .newBases(
                            Collections.singletonList(
                                new BasePath(-1, Optional.empty(), "invalid", false)))
                        .build())
                .build()) {
          IllegalArgumentException error =
              assertThrows(
                  IllegalArgumentException.class,
                  () -> new CommitBuilder(dataset).execute(transaction));
          assertTrue(error.getMessage().contains("non-negative"));
        }

        org.lance.fragment.DataFile dataFile =
            new org.lance.fragment.DataFile("overlay", new int[0], new int[0], 2, 1, 0L, null);
        DataOverlay.DataOverlayGroup invalidGroup =
            new DataOverlay.DataOverlayGroup(
                -1,
                Collections.singletonList(
                    new DataOverlay.DataOverlayFile(
                        dataFile, DataOverlay.OverlayCoverage.shared(OFFSETS_1_3_5), 0)));
        try (Transaction transaction =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .operation(
                    DataOverlay.builder().groups(Collections.singletonList(invalidGroup)).build())
                .build()) {
          IllegalArgumentException error =
              assertThrows(
                  IllegalArgumentException.class,
                  () -> new CommitBuilder(dataset).execute(transaction));
          assertTrue(error.getMessage().contains("dataOverlay.fragmentId"));
        }

        assertInvalidOperation(
            dataset,
            Delete.builder().deletedFragmentIds(Collections.singletonList(-1L)).build(),
            "delete.deletedFragmentIds[0]");
        assertInvalidOperation(
            dataset,
            Update.builder().removedFragmentIds(Collections.singletonList(-1L)).build(),
            "update.removedFragmentIds[0]");
        assertInvalidOperation(
            dataset,
            Update.builder()
                .updatedFragmentOffsets(Collections.singletonMap(-1L, OFFSETS_1_3_5))
                .updateMode(Optional.of(Update.UpdateMode.RewriteColumns))
                .build(),
            "updatedFragmentOffsets.fragmentId");
        assertInvalidOperation(dataset, Restore.builder().version(-1).build(), "restore.version");
        assertInvalidOperation(
            dataset,
            ReserveFragments.builder().numFragments(-1).build(),
            "reserveFragments.numFragments");
        assertInvalidOperation(
            dataset,
            DataReplacement.builder()
                .replacements(
                    Collections.singletonList(
                        new DataReplacement.DataReplacementGroup(-1, dataFile)))
                .build(),
            "dataReplacement.fragmentId");
        assertInvalidOperation(
            dataset,
            CreateIndex.builder()
                .withNewIndices(
                    Collections.singletonList(
                        Index.builder()
                            .uuid(UUID.randomUUID())
                            .fields(Collections.singletonList(0))
                            .coveringFields(Collections.emptyList())
                            .name("negative-version")
                            .datasetVersion(-1)
                            .indexVersion(0)
                            .indexType(IndexType.BTREE)
                            .build()))
                .build(),
            "index.datasetVersion");
        assertInvalidOperation(
            dataset,
            CreateIndex.builder()
                .withNewIndices(
                    Collections.singletonList(
                        Index.builder()
                            .uuid(UUID.randomUUID())
                            .fields(Collections.singletonList(0))
                            .coveringFields(Collections.emptyList())
                            .name("negative-fragment")
                            .datasetVersion(1)
                            .fragments(Collections.singletonList(-1))
                            .indexVersion(0)
                            .indexType(IndexType.BTREE)
                            .build()))
                .build(),
            "index.fragments[0]");
      }
    }
  }

  private static void assertInvalidOperation(
      Dataset dataset, Operation operation, String expectedMessage) {
    try (Transaction transaction =
        new Transaction.Builder().readVersion(dataset.version()).operation(operation).build()) {
      IllegalArgumentException error =
          assertThrows(
              IllegalArgumentException.class,
              () -> new CommitBuilder(dataset).execute(transaction));
      assertTrue(error.getMessage().contains(expectedMessage), error.getMessage());
    }
  }

  @Test
  public void testUnknownOperationReturnsDiagnosticError(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("unknown_operation").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset();
          Transaction transaction =
              new Transaction.Builder()
                  .readVersion(dataset.version())
                  .operation(() -> "FutureOperation")
                  .build()) {
        IllegalArgumentException error =
            assertThrows(
                IllegalArgumentException.class,
                () -> new CommitBuilder(dataset).execute(transaction));
        assertTrue(error.getMessage().contains("FutureOperation"));
      }
    }
  }

  @Test
  public void testRewrittenIndexFilesModelPreservesAbsentAndEmpty() {
    UUID oldId = UUID.randomUUID();
    UUID newId = UUID.randomUUID();
    RewrittenIndex absent =
        RewrittenIndex.builder()
            .oldId(oldId)
            .newId(newId)
            .newIndexDetailsTypeUrl("type.googleapis.com/lance.index.BTreeIndexDetails")
            .newIndexDetailsValue(new byte[0])
            .newIndexVersion(1)
            .build();
    RewrittenIndex empty =
        RewrittenIndex.builder()
            .oldId(oldId)
            .newId(newId)
            .newIndexDetailsTypeUrl("type.googleapis.com/lance.index.BTreeIndexDetails")
            .newIndexDetailsValue(new byte[0])
            .newIndexVersion(1)
            .newIndexFiles(Collections.emptyList())
            .build();
    RewrittenIndex populated =
        RewrittenIndex.builder()
            .oldId(oldId)
            .newId(newId)
            .newIndexDetailsTypeUrl("type.googleapis.com/lance.index.BTreeIndexDetails")
            .newIndexDetailsValue(new byte[0])
            .newIndexVersion(1)
            .newIndexFiles(Collections.singletonList(new IndexFile("index.idx", 123)))
            .build();

    assertTrue(absent.getNewIndexFiles().isEmpty());
    assertTrue(empty.getNewIndexFiles().isEmpty());
    assertEquals(123, populated.getNewIndexFiles().orElseThrow().get(0).getSizeBytes());
  }

  @Test
  public void testRewrittenIndexFilesRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("rewritten_index_files").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 10)) {
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        dataset.createIndex(
            IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, indexParams)
                .withIndexName("btree_id")
                .build());
        Index oldIndex = dataset.getIndexes().get(0);
        IndexFile indexFile = new IndexFile("index.idx", 123);
        RewrittenIndex rewrittenIndex =
            RewrittenIndex.builder()
                .oldId(oldIndex.uuid())
                .newId(UUID.randomUUID())
                .newIndexDetailsTypeUrl("type.googleapis.com/lance.index.BTreeIndexDetails")
                .newIndexDetailsValue(new byte[0])
                .newIndexVersion(oldIndex.indexVersion())
                .newIndexFiles(Collections.singletonList(indexFile))
                .build();
        try (Transaction transaction =
                new Transaction.Builder()
                    .readVersion(dataset.version())
                    .operation(
                        Rewrite.builder()
                            .rewrittenIndices(Collections.singletonList(rewrittenIndex))
                            .build())
                    .build();
            Dataset committed = new CommitBuilder(dataset).execute(transaction);
            Transaction read = committed.readTransaction().orElseThrow()) {
          Rewrite operation = assertInstanceOf(Rewrite.class, read.operation());
          IndexFile readFile =
              operation.rewrittenIndices().get(0).getNewIndexFiles().orElseThrow().get(0);
          assertEquals("index.idx", readFile.getPath());
          assertEquals(123, readFile.getSizeBytes());
        }
      }
    }
  }
}
