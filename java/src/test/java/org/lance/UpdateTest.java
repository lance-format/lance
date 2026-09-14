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

import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;
import org.lance.operation.Append;
import org.lance.update.UpdateParams;
import org.lance.update.UpdateResult;

import com.google.common.collect.ImmutableMap;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

public class UpdateTest {
  private static final int ROW_COUNT = 5;
  private static final String ROW_ID_COLUMN = "_rowid";

  @TempDir private Path tempDir;
  private RootAllocator allocator;
  private TestUtils.SimpleTestDataset testDataset;
  private Dataset dataset;

  @BeforeEach
  public void setup() {
    String datasetPath = tempDir.resolve(UUID.randomUUID().toString()).toString();
    allocator = new RootAllocator(Long.MAX_VALUE);
    testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);

    // Enable stable row ids so that `_rowid` is a stable u64 identifier and can be
    // used as an update predicate (the core feature exercised by the rowid tests).
    Dataset empty =
        testDataset.createDatasetWithWriteParams(
            new WriteParams.Builder().withEnableStableRowIds(true).build());

    FragmentMetadata fragment = testDataset.createNewFragment(ROW_COUNT);
    SourcedTransaction transaction =
        new SourcedTransaction.Builder(empty)
            .operation(Append.builder().fragments(Arrays.asList(fragment)).build())
            .readVersion(empty.version())
            .build();
    dataset = transaction.commit();
    empty.close();
  }

  @AfterEach
  public void tearDown() {
    if (dataset != null) {
      dataset.close();
    }
    allocator.close();
  }

  @ParameterizedTest
  @EnumSource(DataStorageVersion.class)
  public void testUpdateAllRows(DataStorageVersion version) {
    if (version == DataStorageVersion.LEGACY) {
      IllegalArgumentException error =
          Assertions.assertThrows(
              IllegalArgumentException.class,
              () ->
                  dataset.update(
                      new UpdateParams(ImmutableMap.of("name", "'updated'"))
                          .withDataStorageVersion(version)));
      Assertions.assertTrue(error.getMessage().contains("V1 and V2"));
      return;
    }
    String exactVersion =
        version == DataStorageVersion.STABLE
            ? "2.2"
            : version == DataStorageVersion.NEXT ? "2.3" : version.toRustString();
    UpdateResult result =
        dataset.update(
            new UpdateParams(ImmutableMap.of("name", "'updated'")).withDataStorageVersion(version));

    Assertions.assertEquals(ROW_COUNT, result.getNumRowsUpdated());
    try (Dataset newDataset = result.getDataset()) {
      Assertions.assertTrue(
          newDataset.getFragments().stream()
              .flatMap(fragment -> fragment.metadata().getFiles().stream())
              .allMatch(
                  file ->
                      exactVersion.equals(
                          file.getFileMajorVersion() + "." + file.getFileMinorVersion())));
      List<String> names = readNames(newDataset);
      Assertions.assertEquals(ROW_COUNT, names.size());
      for (String name : names) {
        Assertions.assertEquals("updated", name);
      }
    }
  }

  @Test
  public void testUpdateWithWhere() {
    UpdateResult result =
        dataset.update(new UpdateParams(ImmutableMap.of("name", "'updated'")).withWhere("id = 2"));

    Assertions.assertEquals(1, result.getNumRowsUpdated());
    try (Dataset newDataset = result.getDataset()) {
      assertNamesById(
          newDataset, id -> id == 2 ? "updated" : "Person " + id, /* expectedRows= */ ROW_COUNT);
    }
  }

  @Test
  public void testUpdateByRowId() throws Exception {
    List<long[]> sample = readRowIdsAndIds(dataset);
    long targetRowId = sample.get(0)[2];
    int targetId = (int) sample.get(1)[2];

    UpdateResult result =
        dataset.update(
            new UpdateParams(ImmutableMap.of("name", "'updated'"))
                .withWhere(ROW_ID_COLUMN + " = " + targetRowId));

    Assertions.assertEquals(1, result.getNumRowsUpdated());
    try (Dataset newDataset = result.getDataset()) {
      assertNamesById(
          newDataset,
          id -> id == targetId ? "updated" : "Person " + id,
          /* expectedRows= */ ROW_COUNT);
    }
  }

  @Test
  public void testUpdateByRowIdInList() throws Exception {
    List<long[]> sample = readRowIdsAndIds(dataset);
    long[] rowIds = sample.get(0);
    long[] ids = sample.get(1);
    int[] indices = new int[] {0, 2, 4};

    StringBuilder inList = new StringBuilder();
    List<Integer> targetIds = new ArrayList<>();
    for (int i = 0; i < indices.length; i++) {
      if (i > 0) {
        inList.append(", ");
      }
      inList.append(rowIds[indices[i]]);
      targetIds.add((int) ids[indices[i]]);
    }

    UpdateResult result =
        dataset.update(
            new UpdateParams(ImmutableMap.of("name", "'updated'"))
                .withWhere(ROW_ID_COLUMN + " IN (" + inList + ")"));

    Assertions.assertEquals(targetIds.size(), result.getNumRowsUpdated());
    try (Dataset newDataset = result.getDataset()) {
      assertNamesById(
          newDataset,
          id -> targetIds.contains(id) ? "updated" : "Person " + id,
          /* expectedRows= */ ROW_COUNT);
    }
  }

  @Test
  public void testUpdateWithRetryParameters() {
    // Ensure builder-style retry knobs round-trip through the JNI layer.
    UpdateResult result =
        dataset.update(
            new UpdateParams(ImmutableMap.of("name", "'retried'"))
                .withConflictRetries(3)
                .withRetryTimeoutMs(60_000));

    Assertions.assertEquals(ROW_COUNT, result.getNumRowsUpdated());
    try (Dataset newDataset = result.getDataset()) {
      for (String name : readNames(newDataset)) {
        Assertions.assertEquals("retried", name);
      }
    }
  }

  @Test
  public void testUpdateRejectsEmptyUpdates() {
    Assertions.assertThrows(
        IllegalArgumentException.class, () -> new UpdateParams(Collections.emptyMap()));
  }

  @Test
  public void testUpdateRejectsNegativeRetryParameters() {
    UpdateParams params = new UpdateParams(ImmutableMap.of("name", "'x'"));
    Assertions.assertThrows(IllegalArgumentException.class, () -> params.withConflictRetries(-1));
    Assertions.assertThrows(IllegalArgumentException.class, () -> params.withRetryTimeoutMs(-1));
  }

  @Test
  public void testUpdatePropagatesInvalidWhereClause() {
    // Unknown column should surface as an exception from the Rust core, not a silent no-op.
    Assertions.assertThrows(
        Exception.class,
        () ->
            dataset.update(
                new UpdateParams(ImmutableMap.of("name", "'x'"))
                    .withWhere("nonexistent_column = 1")));
  }

  @Test
  public void testUpdatePropagatesInvalidUpdateExpression() {
    // Malformed SQL expression should surface as an exception, not a silent no-op.
    Assertions.assertThrows(
        Exception.class,
        () -> dataset.update(new UpdateParams(ImmutableMap.of("name", "this is not sql"))));
  }

  /**
   * Returns a 2-element list: index 0 contains the {@code _rowid} values, index 1 contains the
   * {@code id} values. Both arrays are sized to {@link #ROW_COUNT}.
   */
  private List<long[]> readRowIdsAndIds(Dataset ds) throws Exception {
    long[] rowIds = new long[ROW_COUNT];
    long[] ids = new long[ROW_COUNT];
    try (LanceScanner scanner =
        ds.newScan(
            new ScanOptions.Builder().columns(Arrays.asList("id")).withRowId(true).build())) {
      try (ArrowReader reader = scanner.scanBatches()) {
        int row = 0;
        while (reader.loadNextBatch()) {
          VectorSchemaRoot batch = reader.getVectorSchemaRoot();
          // Lance stable `_rowid` is an Arrow uint64; Arrow Java represents uint64 as
          // `UInt8Vector` (the "8" refers to byte width, not bit width).
          UInt8Vector rowIdVector = (UInt8Vector) batch.getVector(ROW_ID_COLUMN);
          IntVector idVector = (IntVector) batch.getVector("id");
          for (int i = 0; i < batch.getRowCount(); i++) {
            rowIds[row] = rowIdVector.get(i);
            ids[row] = idVector.get(i);
            row++;
          }
        }
        Assertions.assertEquals(ROW_COUNT, row);
      }
    }
    return Arrays.asList(rowIds, ids);
  }

  private interface IntToString {
    String apply(int id);
  }

  private void assertNamesById(Dataset ds, IntToString expected, int expectedRows) {
    try (LanceScanner scanner =
        ds.newScan(new ScanOptions.Builder().columns(Arrays.asList("id", "name")).build())) {
      try (ArrowReader reader = scanner.scanBatches()) {
        int seen = 0;
        while (reader.loadNextBatch()) {
          VectorSchemaRoot batch = reader.getVectorSchemaRoot();
          IntVector idVector = (IntVector) batch.getVector("id");
          VarCharVector nameVector = (VarCharVector) batch.getVector("name");
          for (int i = 0; i < batch.getRowCount(); i++) {
            int id = idVector.get(i);
            String name = new String(nameVector.get(i));
            Assertions.assertEquals(expected.apply(id), name);
            seen++;
          }
        }
        Assertions.assertEquals(expectedRows, seen);
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private List<String> readNames(Dataset ds) {
    List<String> names = new ArrayList<>();
    try (LanceScanner scanner =
        ds.newScan(new ScanOptions.Builder().columns(Arrays.asList("name")).build())) {
      try (ArrowReader reader = scanner.scanBatches()) {
        while (reader.loadNextBatch()) {
          VectorSchemaRoot batch = reader.getVectorSchemaRoot();
          VarCharVector nameVector = (VarCharVector) batch.getVector("name");
          for (int i = 0; i < batch.getRowCount(); i++) {
            names.add(new String(nameVector.get(i)));
          }
        }
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return names;
  }
}
