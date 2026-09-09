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

import org.lance.fragment.DeletionFile;
import org.lance.fragment.DeletionFileType;
import org.lance.fragment.FragmentMergeResult;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;
import org.lance.operation.Merge;
import org.lance.operation.Project;
import org.lance.operation.Update;
import org.lance.schema.LanceField;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FragmentTest {
  @Test
  void testDeletionFileRelativePath() {
    DeletionFile bitmap = new DeletionFile(7, 11, 2L, DeletionFileType.BITMAP, null);
    DeletionFile array = new DeletionFile(7, 11, 2L, DeletionFileType.ARRAY, null);
    DeletionFile highBitId = new DeletionFile(-1L, 11, 2L, DeletionFileType.BITMAP, null);

    assertEquals("_deletions/5-11-7.bin", bitmap.getRelativePath(5));
    assertEquals("_deletions/5-11-7.arrow", array.getRelativePath(5));
    assertEquals("_deletions/5-11-18446744073709551615.bin", highBitId.getRelativePath(5));
    assertThrows(IllegalArgumentException.class, () -> bitmap.getRelativePath(-1));
  }

  @Test
  void testFragmentCreateFfiArray(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("new_fragment_array").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.createNewFragment(20);
    }
  }

  @Test
  void testFragmentCreate(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("new_fragment").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int rowCount = 21;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      assertEquals(fragmentMeta.getFiles(), fragmentMeta.getReferencedLanceFiles());

      // Commit fragment
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(fragmentMeta));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        Fragment fragment = dataset.getFragments().get(0);

        try (LanceScanner scanner = fragment.newScan()) {
          Schema schemaRes = scanner.schema();
          assertEquals(testDataset.getSchema(), schemaRes);
        }
      }
    }
  }

  @Test
  void testWriteFragmentWithSchemaOverride(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("fragment_schema_override").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        List<org.apache.arrow.vector.types.pojo.Field> fieldList =
            new ArrayList<>(testDataset.getSchema().getFields());
        Collections.reverse(fieldList);

        try (Transaction projectTxn =
                new Transaction.Builder()
                    .readVersion(dataset.version())
                    .operation(Project.builder().schema(new Schema(fieldList)).build())
                    .build();
            Dataset evolvedDataset = new CommitBuilder(dataset).execute(projectTxn);
            VectorSchemaRoot root =
                VectorSchemaRoot.create(evolvedDataset.getSchema(), allocator)) {
          root.allocateNew();
          VarCharVector nameVector = (VarCharVector) root.getVector("name");
          IntVector idVector = (IntVector) root.getVector("id");
          nameVector.setSafe(0, "Person 1".getBytes(StandardCharsets.UTF_8));
          idVector.setSafe(0, 1);
          root.setRowCount(1);

          List<FragmentMetadata> fragments =
              Fragment.write()
                  .datasetUri(datasetPath)
                  .allocator(allocator)
                  .data(root)
                  .schema(evolvedDataset.getLanceSchema())
                  .mode(WriteParams.WriteMode.APPEND)
                  .execute();

          assertEquals(1, fragments.size());
          assertEquals(1, fragments.get(0).getPhysicalRows());
          assertArrayEquals(
              evolvedDataset.getLanceSchema().fields().stream()
                  .mapToInt(LanceField::getId)
                  .toArray(),
              fragments.get(0).getFiles().get(0).getFields());

          FragmentOperation.Append appendOp = new FragmentOperation.Append(fragments);
          try (Dataset appendedDataset =
                  Dataset.commit(
                      allocator, datasetPath, appendOp, Optional.of(evolvedDataset.version()));
              ArrowReader reader = appendedDataset.newScan().scanBatches()) {
            assertEquals(3, appendedDataset.version());
            assertEquals(1, appendedDataset.countRows());
            assertTrue(reader.loadNextBatch());
            VectorSchemaRoot batch = reader.getVectorSchemaRoot();
            assertEquals(1, batch.getRowCount());
            assertEquals(
                "Person 1",
                new String(
                    ((VarCharVector) batch.getVector("name")).get(0), StandardCharsets.UTF_8));
            assertEquals(1, ((IntVector) batch.getVector("id")).get(0));
          }
        }
      }
    }
  }

  @Test
  void testWriteFragmentWithSession(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("fragment_with_session").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        Session session = Session.builder().build()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      long sizeBefore = session.sizeBytes();
      try (VectorSchemaRoot root = VectorSchemaRoot.create(testDataset.getSchema(), allocator)) {
        root.allocateNew();
        VarCharVector nameVector = (VarCharVector) root.getVector("name");
        IntVector idVector = (IntVector) root.getVector("id");
        nameVector.setSafe(0, "Person 1".getBytes(StandardCharsets.UTF_8));
        idVector.setSafe(0, 1);
        root.setRowCount(1);

        // First append in APPEND mode without an explicit schema: the
        // manifest load for schema inference populates the shared session's
        // metadata cache.
        List<FragmentMetadata> firstFragments =
            appendWithSession(datasetPath, allocator, root, session);
        assertEquals(1, firstFragments.size());
        assertEquals(1, firstFragments.get(0).getPhysicalRows());
        assertTrue(session.sizeBytes() > sizeBefore);
        long hitsAfterFirst = session.metadataCacheStats().getHits();

        // Second append through the same session: schema inference reads the
        // manifest cached by the first write, so cache hits must increase.
        List<FragmentMetadata> secondFragments =
            appendWithSession(datasetPath, allocator, root, session);
        assertEquals(1, secondFragments.size());
        assertEquals(1, secondFragments.get(0).getPhysicalRows());
        assertTrue(session.metadataCacheStats().getHits() > hitsAfterFirst);

        // A closed session has a zero native handle and degrades to "no
        // session", matching Dataset's behavior for closed sessions.
        Session closedSession = Session.builder().build();
        closedSession.close();
        List<FragmentMetadata> fragmentsWithClosedSession =
            appendWithSession(datasetPath, allocator, root, closedSession);
        assertEquals(1, fragmentsWithClosedSession.size());
      }
    }
  }

  private static List<FragmentMetadata> appendWithSession(
      String datasetPath, RootAllocator allocator, VectorSchemaRoot root, Session session) {
    return Fragment.write()
        .datasetUri(datasetPath)
        .allocator(allocator)
        .data(root)
        .mode(WriteParams.WriteMode.APPEND)
        .session(session)
        .execute();
  }

  @Test
  void commitWithoutVersion(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("commit_without_version").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata meta = testDataset.createNewFragment(20);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(meta));
      assertThrows(
          IllegalArgumentException.class,
          () -> {
            Dataset.commit(allocator, datasetPath, appendOp, Optional.empty());
          });
    }
  }

  @Test
  void appendWithoutFragment(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("append_without_fragment").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      assertThrows(
          IllegalArgumentException.class,
          () -> {
            new FragmentOperation.Append(new ArrayList<>());
          });
    }
  }

  @Test
  void testOverwriteCommit(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testOverwriteCommit").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      // Commit fragment
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      FragmentOperation.Overwrite overwrite =
          new FragmentOperation.Overwrite(
              Collections.singletonList(fragmentMeta), testDataset.getSchema());
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, overwrite, Optional.of(1L))) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        Fragment fragment = dataset.getFragments().get(0);

        try (LanceScanner scanner = fragment.newScan()) {
          Schema schemaRes = scanner.schema();
          assertEquals(testDataset.getSchema(), schemaRes);
        }
      }

      // Commit fragment again
      rowCount = 40;
      fragmentMeta = testDataset.createNewFragment(rowCount);
      overwrite =
          new FragmentOperation.Overwrite(
              Collections.singletonList(fragmentMeta), testDataset.getSchema());
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, overwrite, Optional.of(2L))) {
        assertEquals(3, dataset.version());
        assertEquals(3, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        Fragment fragment = dataset.getFragments().get(0);

        try (LanceScanner scanner = fragment.newScan()) {
          Schema schemaRes = scanner.schema();
          assertEquals(testDataset.getSchema(), schemaRes);
        }
      }
    }
  }

  @Test
  void testEmptyFragments(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testEmptyFragments").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      List<FragmentMetadata> fragments = testDataset.createNewFragment(0, 10);
      assertEquals(0, fragments.size());
    }
  }

  @Test
  void testMultiFragments(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testMultiFragments").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      List<FragmentMetadata> fragments = testDataset.createNewFragment(20, 10);
      assertEquals(2, fragments.size());
    }
  }

  @Test
  void testDeleteRows(@TempDir Path tempDir) throws IOException {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset();

      int totalRows = 100;
      try (Dataset dataset2 = testDataset.write(1, totalRows)) {
        assertEquals(totalRows, dataset2.countRows());

        Fragment fragment = dataset2.getFragments().get(0);
        List<Integer> rowIndexes = readAllRows(fragment);

        // Case 1. Test delete some rows

        Collections.shuffle(rowIndexes);
        int deleteCount = rowIndexes.size() / 2;
        FragmentMetadata updateFragment = fragment.deleteRows(rowIndexes.subList(0, deleteCount));

        assertNotNull(updateFragment);
        assertNotNull(updateFragment.getDeletionFile());

        Update update =
            Update.builder()
                .updatedFragments(Collections.singletonList(updateFragment))
                .updateMode(Optional.of(Update.UpdateMode.RewriteRows))
                .build();
        Dataset dataset3;
        try (Transaction txn =
            new Transaction.Builder().readVersion(dataset2.version()).operation(update).build()) {
          dataset3 = new CommitBuilder(dataset2).execute(txn);
        }

        assertEquals(totalRows - deleteCount, dataset3.countRows());

        // Case 2. Test more some rows
        fragment = dataset3.getFragments().get(0);
        rowIndexes = readAllRows(fragment);

        int deleteCount2 = rowIndexes.size() / 2;
        updateFragment = fragment.deleteRows(rowIndexes.subList(0, deleteCount2));

        assertNotNull(updateFragment);
        assertNotNull(updateFragment.getDeletionFile());

        update =
            Update.builder()
                .updatedFragments(Collections.singletonList(updateFragment))
                .updateMode(Optional.of(Update.UpdateMode.RewriteRows))
                .build();
        Dataset dataset4;
        try (Transaction txn =
            new Transaction.Builder().readVersion(dataset3.version()).operation(update).build()) {
          dataset4 = new CommitBuilder(dataset3).execute(txn);
        }
        assertEquals(totalRows - deleteCount - deleteCount2, dataset4.countRows());

        // Case 3. Test delete all rows

        fragment = dataset4.getFragments().get(0);
        rowIndexes = readAllRows(fragment);

        updateFragment = fragment.deleteRows(rowIndexes);

        assertNull(updateFragment);

        update =
            Update.builder()
                .removedFragmentIds(Collections.singletonList(Long.valueOf(fragment.getId())))
                .updateMode(Optional.of(Update.UpdateMode.RewriteRows))
                .build();
        Dataset dataset5;
        try (Transaction txn =
            new Transaction.Builder().readVersion(dataset4.version()).operation(update).build()) {
          dataset5 = new CommitBuilder(dataset4).execute(txn);
        }

        assertEquals(0, dataset5.countRows());
      }
    }
  }

  private List<Integer> readAllRows(Fragment fragment) throws IOException {
    List<Long> rowAddrs = new ArrayList<>();

    LanceScanner scanner = fragment.newScan(new ScanOptions.Builder().withRowAddress(true).build());
    try (ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        UInt8Vector rowAddressVector = (UInt8Vector) root.getVector("_rowaddr");
        for (int i = 0; i < rowAddressVector.getValueCount(); i++) {
          rowAddrs.add(rowAddressVector.get(i));
        }
      }
    }

    return rowAddrs.stream().map(RowAddress::rowIndex).collect(Collectors.toList());
  }

  @Test
  void testMergeColumns(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testMergeColumns").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.MergeColumnTestDataset testDataset =
          new TestUtils.MergeColumnTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      int rowCount = 21;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);

      // Commit fragment
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(fragmentMeta));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        assertEquals(2, dataset.version());
        assertEquals(2, dataset.latestVersion());
        assertEquals(rowCount, dataset.countRows());
        Fragment fragment = dataset.getFragments().get(0);

        try (LanceScanner scanner = fragment.newScan()) {
          Schema schemaRes = scanner.schema();
          assertEquals(testDataset.getSchema(), schemaRes);
        }

        FragmentMergeResult mergeResult = testDataset.mergeColumn(fragment, 10);

        try (Transaction transaction =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .operation(
                    Merge.builder()
                        .fragments(Collections.singletonList(mergeResult.getFragmentMetadata()))
                        .schema(mergeResult.getSchema().asArrowSchema())
                        .build())
                .build()) {
          try (Dataset newDs = new CommitBuilder(dataset).execute(transaction)) {
            assertEquals(3, newDs.version());
            assertEquals(3, newDs.latestVersion());
            Fragment newFrag = newDs.getFragments().get(0);
            try (LanceScanner scanner = newFrag.newScan()) {
              Schema schemaRes = scanner.schema();
              assertTrue(
                  schemaRes.getFields().stream()
                      .anyMatch(field -> field.getName().equals("new_col1")));
              assertTrue(
                  schemaRes.getFields().stream()
                      .anyMatch(field -> field.getName().equals("new_col2")));

              try (ArrowReader reader = scanner.scanBatches()) {
                assertTrue(reader.loadNextBatch());
                VectorSchemaRoot root = reader.getVectorSchemaRoot();
                VarCharVector newCol1Vec = (VarCharVector) root.getVector("new_col1");
                VarCharVector newCol2Vec = (VarCharVector) root.getVector("new_col2");
                assertEquals(21, newCol2Vec.getValueCount());

                // The first 10 rows are not null
                assertNotNull(newCol1Vec.get(9));
                // Remaining rows are null
                assertNull(newCol1Vec.get(10));
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testFragmentStatistics(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("fragment_statistics").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      // Two fragments with different row counts
      FragmentMetadata frag1 = testDataset.createNewFragment(21);
      FragmentMetadata frag2 = testDataset.createNewFragment(9);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(frag1, frag2));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        List<Fragment> fragments = dataset.getFragments();
        FragmentStatistics stats = dataset.getFragmentStatistics();
        assertEquals(fragments.size(), stats.size());

        // Parity with getFragments across all three arrays
        assertArrayEquals(fragments.stream().mapToInt(Fragment::getId).toArray(), stats.getIds());
        assertArrayEquals(
            fragments.stream().mapToLong(f -> f.metadata().getNumRows()).toArray(),
            stats.getRowCounts());
        assertArrayEquals(
            fragments.stream().mapToInt(f -> f.metadata().getFiles().size()).toArray(),
            stats.getDataFileNums());

        assertEquals(30, Arrays.stream(stats.getRowCounts()).sum());

        dataset.delete("id < 5");
        assertArrayEquals(new long[] {16, 4}, dataset.getFragmentStatistics().getRowCounts());
      }
    }
  }

  @Test
  void testFragmentStatisticsOnEmptyDataset(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("fragment_statistics_empty").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(0, dataset.getFragmentStatistics().size());
      }
    }
  }

  @Test
  void testFragmentStatisticsPreservesLegacyMissingRowCount() {
    String historicalPath =
        Path.of("..", "test_data", "v0.7.5", "with_deletions")
            .toAbsolutePath()
            .normalize()
            .toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        Dataset dataset = Dataset.open(historicalPath, allocator)) {
      FragmentStatistics stats = dataset.getFragmentStatistics();
      assertArrayEquals(new int[] {0}, stats.getIds());
      assertArrayEquals(new long[] {0}, stats.getRowCounts());
      assertArrayEquals(new int[] {1}, stats.getDataFileNums());
    }
  }

  @Test
  void testCountRowsConcurrentWithClose(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("count_rows_close_race").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(100);
      FragmentOperation.Append appendOp = new FragmentOperation.Append(Arrays.asList(fragmentMeta));
      Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L));
      Fragment fragment = dataset.getFragments().get(0);

      int threadCount = 8;
      ExecutorService executor = Executors.newFixedThreadPool(threadCount);
      try {
        CountDownLatch start = new CountDownLatch(1);
        List<Future<?>> futures = new ArrayList<>();
        for (int i = 0; i < threadCount; i++) {
          futures.add(
              executor.submit(
                  () -> {
                    start.await();
                    // Hammer countRows until close() wins the race. The only acceptable
                    // failure is the "Dataset is closed" rejection; anything else (a native
                    // crash or "Null pointer in rust value from Java") means the native
                    // handle was released while still in use.
                    while (true) {
                      try {
                        fragment.countRows();
                      } catch (IllegalArgumentException e) {
                        assertEquals("Dataset is closed", e.getMessage());
                        return null;
                      }
                    }
                  }));
        }
        start.countDown();
        Thread.sleep(50);
        dataset.close();
        for (Future<?> future : futures) {
          future.get(30, TimeUnit.SECONDS);
        }
      } finally {
        executor.shutdownNow();
      }
    }
  }
}
