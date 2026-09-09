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

import org.lance.index.IndexOptions;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.scalar.ScalarIndexParams;
import org.lance.ipc.ColumnOrdering;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.MaterializationStyle;
import org.lance.ipc.ScanOptions;
import org.lance.ipc.ScanStats;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ScannerTest {
  private static Dataset dataset;

  @BeforeAll
  static void setup() {}

  @AfterAll
  static void tearDown() {
    // Cleanup resources used by the tests
    if (dataset != null) {
      dataset.close();
    }
  }

  @Test
  void testScannerPerformanceOptions() {
    MaterializationStyle materialization =
        MaterializationStyle.allEarlyExcept(Collections.singletonList("name"));
    ScanOptions options =
        new ScanOptions.Builder()
            .batchSize(1024)
            .batchSizeBytes(64 * 1024)
            .ioBufferSize(8 * 1024 * 1024)
            .batchReadahead(4)
            .fragmentReadahead(2)
            .scanInOrder(false)
            .lateMaterialization(materialization)
            .build();

    assertEquals(1024L, options.getBatchSize().orElseThrow());
    assertEquals(64 * 1024L, options.getBatchSizeBytes().orElseThrow());
    assertEquals(8 * 1024 * 1024L, options.getIoBufferSize().orElseThrow());
    assertEquals(4, options.getBatchReadahead());
    assertEquals(2, options.getFragmentReadahead().orElseThrow());
    assertFalse(options.isScanInOrder());
    assertEquals(materialization, options.getLateMaterialization().orElseThrow());
    assertEquals("heuristic", MaterializationStyle.heuristic().toRustString());
    assertEquals("all_late", MaterializationStyle.allLate().toRustString());
    assertEquals("all_early", MaterializationStyle.allEarly().toRustString());
    assertEquals("all_early_except", materialization.toRustString());
    assertEquals(Collections.singletonList("name"), materialization.getColumns());

    ScanOptions defaults = new ScanOptions.Builder().build();
    assertTrue(defaults.getBatchSizeBytes().isEmpty());
    assertTrue(defaults.getIoBufferSize().isEmpty());
    assertTrue(defaults.getFragmentReadahead().isEmpty());
    assertTrue(defaults.isScanInOrder());
    assertTrue(defaults.getLateMaterialization().isEmpty());

    assertThrows(
        IllegalArgumentException.class, () -> new ScanOptions.Builder().batchSizeBytes(0).build());
    assertThrows(
        IllegalArgumentException.class, () -> new ScanOptions.Builder().ioBufferSize(0).build());
    assertThrows(
        IllegalArgumentException.class,
        () -> new ScanOptions.Builder().fragmentReadahead(0).build());
    assertThrows(
        IllegalArgumentException.class,
        () -> new ScanOptions.Builder().batchSizeBytes(1024).strictBatchSize(true).build());
    assertTrue(MaterializationStyle.allEarlyExcept(Collections.emptyList()).getColumns().isEmpty());
  }

  static Stream<MaterializationStyle> materializationStyles() {
    return Stream.of(
        MaterializationStyle.heuristic(),
        MaterializationStyle.allLate(),
        MaterializationStyle.allEarly(),
        MaterializationStyle.allEarlyExcept(Collections.emptyList()));
  }

  @ParameterizedTest
  @MethodSource("materializationStyles")
  void testMaterializationStylesAcrossJni(
      MaterializationStyle materializationStyle, @TempDir Path tempDir) throws Exception {
    String datasetPath =
        tempDir.resolve("materialization_" + materializationStyle.getMode().name()).toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      try (Dataset dataset = testDataset.write(1, 40);
          LanceScanner scanner =
              dataset.newScan(
                  new ScanOptions.Builder()
                      .filter("id < 20")
                      .lateMaterialization(materializationStyle)
                      .build());
          ArrowReader reader = scanner.scanBatches()) {
        int rowCount = 0;
        while (reader.loadNextBatch()) {
          rowCount += reader.getVectorSchemaRoot().getRowCount();
        }
        assertEquals(20, rowCount);
      }
    }
  }

  @Test
  void testDatasetScanner(@TempDir Path tempDir) throws IOException {
    String datasetPath = tempDir.resolve("dataset_scanner").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      int batchRows = 20;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        Scanner scanner = dataset.newScan(batchRows);
        testDataset.validateScanResults(dataset, scanner, totalRows, batchRows);
      }
    }
  }

  @Test
  void testDatasetScannerFilter(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_filter").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // write id with value from 0 to 39
      try (Dataset dataset = testDataset.write(1, 40)) {
        try (Scanner scanner =
            dataset.newScan(new ScanOptions.Builder().filter("id < 20").build())) {
          testDataset.validateScanResults(dataset, scanner, 20, 20);
        }
      }
    }
  }

  @Test
  void testDatasetScannerColumns(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_columns").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      int batchRows = 20;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(batchRows)
                    .columns(Arrays.asList("id"))
                    .build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            int index = 0;
            while (reader.loadNextBatch()) {
              List<FieldVector> fieldVectors = root.getFieldVectors();
              assertEquals(1, fieldVectors.size());
              FieldVector fieldVector = fieldVectors.get(0);
              assertEquals(ArrowType.ArrowTypeID.Int, fieldVector.getField().getType().getTypeID());
              assertEquals(batchRows, fieldVector.getValueCount());
              IntVector vector = (IntVector) fieldVector;
              for (int i = 0; i < batchRows; i++) {
                assertEquals(index, vector.get(i));
                index++;
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testDatasetScannerSchema(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_schema").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(totalRows)
                    .columns(Arrays.asList("id"))
                    .build())) {
          Schema expectedSchema =
              new Schema(Arrays.asList(Field.nullable("id", new ArrowType.Int(32, true))));
          assertEquals(expectedSchema, scanner.schema());
        }
      }
    }
  }

  /**
   * Imports a caller-owned C stream populated by {@link LanceScanner#exportArrowStream(long)} and
   * returns the {@code id} values in the order the stream produced them.
   *
   * <p>The projected schema is asserted to be exactly a single {@code id: int32} field, and the
   * assertion is made on the imported reader <em>before</em> the first {@code loadNextBatch()} call
   * so that it still runs for an empty (zero-batch) result — a regression that exported the wrong
   * schema for an empty scan would otherwise slip through. See {@code
   * org.apache.arrow.vector.ipc.ArrowReader#getVectorSchemaRoot()}, which exposes the schema as
   * soon as the stream is imported.
   *
   * <p>This helper intentionally makes <em>no</em> assertion about per-batch row counts. The
   * scanner's {@code batchSize} is only a hint unless {@code strictBatchSize(true)} is set, so the
   * number of batches and the rows per batch are not part of the contract being tested here; that
   * dimension is covered separately by {@link #testExportArrowStreamStrictBatchSize}. Row ordering
   * and exact values are asserted by the callers against the returned list.
   */
  private static List<Integer> drainIdStream(BufferAllocator allocator, ArrowArrayStream stream)
      throws IOException {
    List<Integer> ids = new ArrayList<>();
    try (ArrowReader reader = Data.importArrayStream(allocator, stream)) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      List<Field> fields = root.getSchema().getFields();
      assertEquals(1, fields.size());
      Field idField = fields.get(0);
      assertEquals("id", idField.getName());
      // Pin the exact type, not just the ArrowTypeID family: the projected column is a nullable
      // signed int32. ArrowTypeID.Int alone also matches int8/16/64 and unsigned, and the
      // (IntVector) cast below only guards the width on non-empty results — an empty scan that
      // exported e.g. int64 or a non-nullable id would otherwise slip through this helper.
      assertTrue(idField.isNullable());
      ArrowType.Int idType = (ArrowType.Int) idField.getType();
      assertEquals(32, idType.getBitWidth());
      assertTrue(idType.getIsSigned());
      while (reader.loadNextBatch()) {
        IntVector vector = (IntVector) root.getVector("id");
        int rowsInBatch = vector.getValueCount();
        for (int i = 0; i < rowsInBatch; i++) {
          ids.add(vector.get(i));
        }
      }
    }
    return ids;
  }

  /**
   * Happy path: a single-fragment ordered scan exported through a caller-owned C stream returns
   * every row exactly once, in scan order. The caller allocates the {@link ArrowArrayStream} from
   * its own allocator and passes only the memory address; the scanner fills the C struct in place.
   * This is the cross-Arrow-version / cross-classloader boundary the API exists to serve.
   */
  @Test
  void testExportArrowStream(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("export_stream_basic").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      int batchRows = 20;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(batchRows)
                    .columns(Arrays.asList("id"))
                    .build())) {
          try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
            scanner.exportArrowStream(stream.memoryAddress());
            // SimpleTestDataset writes id = 0..totalRows-1; an ordered scan must return them in
            // exactly that sequence, so assert the exact ordering (no sort).
            List<Integer> ids = drainIdStream(allocator, stream);
            assertEquals(totalRows, ids.size());
            for (int i = 0; i < totalRows; i++) {
              assertEquals(i, ids.get(i));
            }
          }
        }
      }
    }
  }

  /**
   * A scan that spans multiple fragments is exported as a single C stream that concatenates the
   * fragments in fragment order. {@code createNewFragment(40, 10)} produces 4 fragments of 10 rows
   * (ids 0-9, 10-19, 20-29, 30-39), and an ordered scan must return 0..39 in exactly that order.
   *
   * <p>The expected ids are asserted in stream order without sorting: sorting would mask a
   * regression that returned fragments out of order, which is exactly the kind of bug this test
   * exists to catch. A non-divisor batch size (7) is used so batch boundaries do not line up with
   * fragment boundaries, exercising the stream's batch stitching across fragments.
   */
  @Test
  void testExportArrowStreamMultipleFragments(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("export_stream_multi_fragment").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      // maxRowsPerFile < totalRows forces multiple fragments (4 fragments of 10 rows).
      List<FragmentMetadata> fragments = testDataset.createNewFragment(totalRows, 10);
      assertEquals(4, fragments.size());
      FragmentOperation.Append appendOp = new FragmentOperation.Append(fragments);
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        int batchRows = 7; // deliberately not a divisor of any fragment size
        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(batchRows)
                    .columns(Arrays.asList("id"))
                    .build())) {
          try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
            scanner.exportArrowStream(stream.memoryAddress());
            List<Integer> ids = drainIdStream(allocator, stream);
            assertEquals(totalRows, ids.size());
            // Assert exact scan order (no sort) so out-of-order fragments would fail.
            for (int i = 0; i < totalRows; i++) {
              assertEquals(i, ids.get(i), "row " + i + " out of expected scan order");
            }
          }
        }
      }
    }
  }

  /**
   * A pushed-down filter is honored by the exported stream: only matching rows cross the C-data
   * boundary. {@code id < 20} over ids 0..39 must yield exactly 0..19 in order. Asserted in scan
   * order without sorting so a filter/ordering regression cannot hide behind a sort.
   */
  @Test
  void testExportArrowStreamWithFilter(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("export_stream_filter").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 40)) {
        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(50)
                    .columns(Arrays.asList("id"))
                    .filter("id < 20")
                    .build())) {
          try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
            scanner.exportArrowStream(stream.memoryAddress());
            List<Integer> ids = drainIdStream(allocator, stream);
            assertEquals(20, ids.size());
            for (int i = 0; i < 20; i++) {
              assertEquals(i, ids.get(i));
            }
          }
        }
      }
    }
  }

  /**
   * Pushed-down limit and offset are honored by the exported stream. Over ids 0..39, {@code
   * offset(10).limit(5)} must yield exactly [10, 11, 12, 13, 14] in order — asserted as an exact
   * ordered list so both the window bounds and the ordering are checked.
   */
  @Test
  void testExportArrowStreamWithLimitOffset(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("export_stream_limit_offset").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 40)) {
        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(50)
                    .columns(Arrays.asList("id"))
                    .limit(5)
                    .offset(10)
                    .build())) {
          try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
            scanner.exportArrowStream(stream.memoryAddress());
            List<Integer> ids = drainIdStream(allocator, stream);
            assertEquals(Arrays.asList(10, 11, 12, 13, 14), ids);
          }
        }
      }
    }
  }

  /**
   * Column projection is reflected in the exported stream's schema. {@code SimpleTestDataset} has
   * columns {@code (id, name)}; projecting only {@code name} must produce a stream whose schema is
   * exactly that one column. The schema is checked on the imported reader before draining, and the
   * full row count is verified after.
   */
  @Test
  void testExportArrowStreamProjectsRequestedColumnsOnly(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("export_stream_projection").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 10)) {
        // Project only "name"; the exported stream's schema must contain exactly that column.
        try (LanceScanner scanner =
            dataset.newScan(new ScanOptions.Builder().columns(Arrays.asList("name")).build())) {
          try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
            scanner.exportArrowStream(stream.memoryAddress());
            try (ArrowReader reader = Data.importArrayStream(allocator, stream)) {
              VectorSchemaRoot root = reader.getVectorSchemaRoot();
              assertEquals(1, root.getSchema().getFields().size());
              assertEquals("name", root.getSchema().getFields().get(0).getName());
              int rows = 0;
              while (reader.loadNextBatch()) {
                rows += root.getRowCount();
              }
              assertEquals(10, rows);
            }
          }
        }
      }
    }
  }

  /**
   * A scan that matches no rows ({@code id < 0}) still exports a valid, well-formed stream that
   * yields zero rows. {@link #drainIdStream} asserts the projected schema ({@code id: int32}) on
   * the imported reader before any {@code loadNextBatch()}, so this case also guards the empty-scan
   * schema — a regression that exported a wrong or absent schema for zero-row results would fail
   * here even though no batch is ever produced.
   */
  @Test
  void testExportArrowStreamEmptyResult(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("export_stream_empty").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 40)) {
        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder().columns(Arrays.asList("id")).filter("id < 0").build())) {
          try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
            scanner.exportArrowStream(stream.memoryAddress());
            List<Integer> ids = drainIdStream(allocator, stream);
            assertTrue(ids.isEmpty());
          }
        }
      }
    }
  }

  /**
   * Guards against the sequential "export twice into the same stream" mistake. After the first
   * export installs a producer (non-null {@code release} callback), a second export into the same
   * stream must be rejected with {@link IllegalArgumentException} rather than overwriting the C
   * struct in place — overwriting would drop the first producer's release callback and leak it.
   *
   * <p>The test also verifies the rejection is non-destructive: the first producer is still intact
   * and fully drainable (all 40 rows) after the rejected second call. This is the single-threaded
   * misuse case; concurrent exports into one caller-owned stream are the caller's responsibility,
   * as documented on {@link LanceScanner#exportArrowStream(long)}.
   */
  @Test
  void testExportArrowStreamRejectsPopulatedStream(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("export_stream_reject_populated").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 40)) {
        try (LanceScanner scanner =
            dataset.newScan(new ScanOptions.Builder().columns(Arrays.asList("id")).build())) {
          try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
            // First export populates the stream and installs a release callback.
            scanner.exportArrowStream(stream.memoryAddress());
            // Exporting again into the same (already-populated) stream must be rejected rather
            // than silently overwriting and leaking the first producer's release callback.
            IllegalArgumentException ex =
                assertThrows(
                    IllegalArgumentException.class,
                    () -> scanner.exportArrowStream(stream.memoryAddress()));
            assertTrue(ex.getMessage().toLowerCase().contains("already populated"));
            // The first producer is still intact and drainable.
            try (ArrowReader reader = Data.importArrayStream(allocator, stream)) {
              int rows = 0;
              VectorSchemaRoot root = reader.getVectorSchemaRoot();
              while (reader.loadNextBatch()) {
                rows += root.getRowCount();
              }
              assertEquals(40, rows);
            }
          }
        }
      }
    }
  }

  /**
   * A null (0) stream address is rejected with {@link IllegalArgumentException} before any native
   * dereference, so a caller mistake cannot turn into a native null-pointer write.
   */
  @Test
  void testExportArrowStreamRejectsNullAddress(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("export_stream_reject_null").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 10)) {
        try (LanceScanner scanner =
            dataset.newScan(new ScanOptions.Builder().columns(Arrays.asList("id")).build())) {
          assertThrows(IllegalArgumentException.class, () -> scanner.exportArrowStream(0L));
        }
      }
    }
  }

  /**
   * Exporting from a closed scanner is rejected with {@link IllegalArgumentException} (the native
   * scanner handle is zero after {@code close()}), rather than dereferencing a freed handle. The
   * scanner is closed explicitly here, so it is intentionally not in a try-with-resources.
   */
  @Test
  void testExportArrowStreamRejectsClosedScanner(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("export_stream_reject_closed").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 10)) {
        LanceScanner scanner =
            dataset.newScan(new ScanOptions.Builder().columns(Arrays.asList("id")).build());
        scanner.close();
        try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
          assertThrows(
              IllegalArgumentException.class,
              () -> scanner.exportArrowStream(stream.memoryAddress()));
        }
      }
    }
  }

  /**
   * Null values survive the C-data export round-trip. {@code writeSortByDataset} writes 10 rows
   * (insertion order) in which {@code id} is null at rows 2 and 5 and {@code name} is null at rows
   * 0 and 6. An unordered scan returns rows in insertion order, so the exported stream must
   * reproduce both the non-null values and the null positions exactly — null/validity bitmaps are a
   * common casualty of an incorrect C-data export, so this guards them explicitly.
   */
  @Test
  void testExportArrowStreamPreservesNulls(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("export_stream_nulls").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.writeSortByDataset(1)) {
        // Insertion order, row -> (id, name):
        //   0 -> (0,    null)   3 -> (2,  "P2")   6 -> (3,  null)   9 -> (5, "P5")
        //   1 -> (1,  "P0")     4 -> (2,  "P3")   7 -> (4,  "P4")
        //   2 -> (null,"P1")    5 -> (null,"P3")  8 -> (4,  "P5")
        Integer[] expectedIds = {0, 1, null, 2, 2, null, 3, 4, 4, 5};
        String[] expectedNames = {null, "P0", "P1", "P2", "P3", "P3", null, "P4", "P5", "P5"};
        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder().columns(Arrays.asList("id", "name")).build())) {
          try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
            scanner.exportArrowStream(stream.memoryAddress());
            try (ArrowReader reader = Data.importArrayStream(allocator, stream)) {
              VectorSchemaRoot root = reader.getVectorSchemaRoot();
              assertEquals(2, root.getSchema().getFields().size());
              int row = 0;
              while (reader.loadNextBatch()) {
                IntVector idVector = (IntVector) root.getVector("id");
                VarCharVector nameVector = (VarCharVector) root.getVector("name");
                for (int i = 0; i < root.getRowCount(); i++, row++) {
                  if (expectedIds[row] == null) {
                    assertTrue(idVector.isNull(i), "id should be null at row " + row);
                  } else {
                    assertEquals(
                        expectedIds[row].intValue(), idVector.get(i), "id mismatch at row " + row);
                  }
                  if (expectedNames[row] == null) {
                    assertTrue(nameVector.isNull(i), "name should be null at row " + row);
                  } else {
                    assertEquals(
                        expectedNames[row],
                        new String(nameVector.get(i), StandardCharsets.UTF_8),
                        "name mismatch at row " + row);
                  }
                }
              }
              assertEquals(expectedIds.length, row);
            }
          }
        }
      }
    }
  }

  /**
   * With {@code strictBatchSize(true)}, the exported stream must split into batches no larger than
   * the requested batch size, and still reproduce every row in order. This is the one place the
   * per-batch size is part of the contract; the other export tests deliberately leave batch sizing
   * unasserted because it is only a hint by default. Mirrors {@link #testStrictBatchSize} but over
   * the C-data export path. A batch size of 10 over 25 rows yields batches of at most 10.
   */
  @Test
  void testExportArrowStreamStrictBatchSize(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("export_stream_strict_batch").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 25;
      int batchSize = 10;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(batchSize)
                    .strictBatchSize(true)
                    .columns(Arrays.asList("id"))
                    .build())) {
          try (ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
            scanner.exportArrowStream(stream.memoryAddress());
            try (ArrowReader reader = Data.importArrayStream(allocator, stream)) {
              VectorSchemaRoot root = reader.getVectorSchemaRoot();
              List<Integer> ids = new ArrayList<>();
              while (reader.loadNextBatch()) {
                int rowsInBatch = root.getRowCount();
                assertTrue(
                    rowsInBatch <= batchSize,
                    "strict: batch of " + rowsInBatch + " should be <= " + batchSize);
                IntVector idVector = (IntVector) root.getVector("id");
                for (int i = 0; i < rowsInBatch; i++) {
                  ids.add(idVector.get(i));
                }
              }
              assertEquals(totalRows, ids.size());
              for (int i = 0; i < totalRows; i++) {
                assertEquals(i, ids.get(i));
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testDatasetScannerCountRows(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_count").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // write id with value from 0 to 39
      try (Dataset dataset = testDataset.write(1, 40)) {
        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .columns(Arrays.asList())
                    .withRowId(true)
                    .filter("id < 20")
                    .build())) {
          assertEquals(20, scanner.countRows());
        }
      }
    }
  }

  @Test
  void testDatasetScannerStats(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_stats").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 40)) {
        try (LanceScanner scanner =
            dataset.newScan(new ScanOptions.Builder().batchSize(20).build())) {
          assertTrue(scanner.getStats().isEmpty());
          try (ArrowReader reader = scanner.scanBatches()) {
            while (reader.loadNextBatch()) {
              // Consume all batches.
            }
          }
          assertTrue(scanner.getStats().isEmpty());
        }

        try (LanceScanner scanner =
            dataset.newScan(new ScanOptions.Builder().batchSize(20).collectStats(true).build())) {
          assertTrue(scanner.getStats().isEmpty());
          try (ArrowReader reader = scanner.scanBatches()) {
            while (reader.loadNextBatch()) {
              // Consume all batches.
            }
          }

          Optional<ScanStats> statsOpt = scanner.getStats();
          assertTrue(statsOpt.isPresent());
          ScanStats stats = statsOpt.get();
          assertTrue(stats.getBytesRead() > 0 || !stats.getAllCounts().isEmpty());
          // Even without an index on this dataset, the two new counters must
          // still marshal through JNI and default to zero rather than throwing.
          assertTrue(stats.getIndexCacheHits() >= 0);
          assertTrue(stats.getIndexCacheMisses() >= 0);
        }
      }
    }
  }

  @Test
  void testFragmentScanner(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("fragment_scanner").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      int batchRows = 20;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        Fragment fragment = dataset.getFragments().get(0);
        try (Scanner scanner = fragment.newScan(batchRows)) {
          testDataset.validateScanResults(dataset, scanner, totalRows, batchRows);
        }
      }
    }
  }

  @Test
  void testFragmentScannerFilter(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("fragment_scanner_filter").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // write id with value from 0 to 39
      try (Dataset dataset = testDataset.write(1, 40)) {
        Fragment fragment = dataset.getFragments().get(0);
        try (Scanner scanner =
            fragment.newScan(new ScanOptions.Builder().filter("id < 20").build())) {
          testDataset.validateScanResults(dataset, scanner, 20, 20);
        }
      }
    }
  }

  @Test
  void testFragmentScannerColumns(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("fragment_scanner_columns").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;
      int batchRows = 20;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        Fragment fragment = dataset.getFragments().get(0);
        try (Scanner scanner =
            fragment.newScan(
                new ScanOptions.Builder()
                    .batchSize(batchRows)
                    .columns(Arrays.asList("id"))
                    .build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            int index = 0;
            while (reader.loadNextBatch()) {
              List<FieldVector> fieldVectors = root.getFieldVectors();
              assertEquals(1, fieldVectors.size());
              FieldVector fieldVector = fieldVectors.get(0);
              assertEquals(ArrowType.ArrowTypeID.Int, fieldVector.getField().getType().getTypeID());
              assertEquals(batchRows, fieldVector.getValueCount());
              IntVector vector = (IntVector) fieldVector;
              for (int i = 0; i < batchRows; i++) {
                assertEquals(index, vector.get(i));
                index++;
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testScanFragment(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("fragment_scanner_single_fragment").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata metadata0 = testDataset.createNewFragment(3);
      FragmentMetadata metadata1 = testDataset.createNewFragment(5);
      FragmentMetadata metadata2 = testDataset.createNewFragment(7);
      FragmentOperation.Append appendOp =
          new FragmentOperation.Append(Arrays.asList(metadata0, metadata1, metadata2));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        List<Fragment> frags = dataset.getFragments();
        assertEquals(3, frags.size());
        validScanResult(dataset, frags.get(0).getId(), 3);
        validScanResult(dataset, frags.get(1).getId(), 5);
        validScanResult(dataset, frags.get(2).getId(), 7);
      }
    }
  }

  @Test
  void testScanFragments(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("fragments_scanner").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      FragmentMetadata metadata0 = testDataset.createNewFragment(3);
      FragmentMetadata metadata1 = testDataset.createNewFragment(5);
      FragmentMetadata metadata2 = testDataset.createNewFragment(7);
      FragmentOperation.Append appendOp =
          new FragmentOperation.Append(Arrays.asList(metadata0, metadata1, metadata2));
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, appendOp, Optional.of(1L))) {
        List<Fragment> frags = dataset.getFragments();
        assertEquals(3, frags.size());
        try (Scanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(1024)
                    .fragmentIds(Arrays.asList(frags.get(1).getId(), frags.get(2).getId()))
                    .build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            assertEquals(
                dataset.getSchema().getFields(),
                reader.getVectorSchemaRoot().getSchema().getFields());
            int rowcount = 0;
            reader.loadNextBatch();
            int currentRowCount = reader.getVectorSchemaRoot().getRowCount();
            assertEquals(5, currentRowCount);
            rowcount += currentRowCount;
            reader.loadNextBatch();
            currentRowCount = reader.getVectorSchemaRoot().getRowCount();
            assertEquals(7, currentRowCount);
            rowcount += currentRowCount;
            assertEquals(12, rowcount);
          }
        }
      }
    }
  }

  @Test
  void testDatasetScannerLimit(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_limit").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 100;
      int limit = 50;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner = dataset.newScan(new ScanOptions.Builder().limit(limit).build())) {
          testDataset.validateScanResults(dataset, scanner, limit, limit);
        }
      }
    }
  }

  @Test
  void testDatasetScannerOffset(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_offset").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 100;
      int offset = 50;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner = dataset.newScan(new ScanOptions.Builder().offset(offset).build())) {
          testDataset.validateScanResults(
              dataset, scanner, totalRows - offset, totalRows - offset, offset);
        }
      }
    }
  }

  @Test
  void testDatasetScannerWithRowId(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_with_row_id").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 50;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner = dataset.newScan(new ScanOptions.Builder().withRowId(true).build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            assertTrue(
                root.getSchema().getFields().stream()
                    .anyMatch(field -> field.getName().equals("_rowid")));
            while (reader.loadNextBatch()) {
              List<FieldVector> fieldVectors = root.getFieldVectors();
              assertTrue(
                  fieldVectors.stream().anyMatch(vector -> vector.getName().equals("_rowid")));
            }
          }
        }
      }
    }
  }

  @Test
  void testDatasetScannerBatchReadahead(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_batch_readahead").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      int totalRows = 2000;
      int maxRowsPerFile = 100; // ~20 fragments
      List<FragmentMetadata> fragments = testDataset.createNewFragment(totalRows, maxRowsPerFile);
      assertTrue(fragments.size() > 1, "expected multiple fragments, got " + fragments.size());

      FragmentOperation.Append append = new FragmentOperation.Append(fragments);
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, append, Optional.of(1L))) {
        int batchReadahead = 2; // far below the default (num compute CPUs)
        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder().batchSize(50).batchReadahead(batchReadahead).build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            int rowCount = 0;
            long idSum = 0;
            while (reader.loadNextBatch()) {
              VectorSchemaRoot root = reader.getVectorSchemaRoot();
              IntVector ids = (IntVector) root.getVector("id");
              for (int i = 0; i < root.getRowCount(); i++) {
                idSum += ids.get(i);
              }
              rowCount += root.getRowCount();
            }
            assertEquals(totalRows, rowCount);
            // ids are the contiguous range [0, totalRows)
            assertEquals((long) totalRows * (totalRows - 1) / 2, idSum);
          }
        }
      }
    }
  }

  @Test
  void testDatasetScannerPerformanceOptions(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_performance_options").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      int totalRows = 2000;
      int matchingRows = 1000;
      WriteParams writeParams =
          new WriteParams.Builder()
              .withMaxRowsPerFile(100)
              .withDataStorageVersion(LanceConstants.FILE_FORMAT_VERSION_STABLE)
              .build();
      List<FragmentMetadata> fragments = testDataset.createNewFragment(totalRows, writeParams);
      FragmentOperation.Append append = new FragmentOperation.Append(fragments);
      try (Dataset dataset = Dataset.commit(allocator, datasetPath, append, Optional.of(1L))) {
        try (LanceScanner scanner =
                dataset.newScan(
                    new ScanOptions.Builder()
                        .filter("id < " + matchingRows)
                        .batchSize(200)
                        .batchSizeBytes(256)
                        .ioBufferSize(1024 * 1024)
                        .batchReadahead(2)
                        .fragmentReadahead(2)
                        .scanInOrder(false)
                        .lateMaterialization(MaterializationStyle.allEarly())
                        .build());
            ArrowReader reader = scanner.scanBatches()) {
          int rowCount = 0;
          long idSum = 0;
          while (reader.loadNextBatch()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            assertTrue(
                root.getRowCount() < 200,
                "batchSizeBytes should split batches before the row limit");
            IntVector ids = (IntVector) root.getVector("id");
            for (int i = 0; i < root.getRowCount(); i++) {
              idSum += ids.get(i);
            }
            rowCount += root.getRowCount();
          }
          assertEquals(matchingRows, rowCount);
          assertEquals((long) matchingRows * (matchingRows - 1) / 2, idSum);
        }

        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .filter("id < " + matchingRows)
                    .lateMaterialization(
                        MaterializationStyle.allEarlyExcept(Collections.singletonList("name")))
                    .build())) {
          assertEquals(matchingRows, scanner.countRows());
        }
      }
    }
  }

  @Test
  void testDatasetScannerSortBy(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testDatasetScannerSortBy").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.writeSortByDataset(1)) {
        ColumnOrdering.Builder nameBuilder = new ColumnOrdering.Builder();
        nameBuilder.setColumnName("name");
        nameBuilder.setAscending(true);
        nameBuilder.setNullFirst(false);

        ColumnOrdering.Builder idBuilder = new ColumnOrdering.Builder();
        idBuilder.setColumnName("id");
        idBuilder.setAscending(false);
        idBuilder.setNullFirst(true);

        List<ColumnOrdering> columnOrderings =
            Arrays.asList(nameBuilder.build(), idBuilder.build());
        ScanOptions.Builder scanOptionBuilder = new ScanOptions.Builder();
        scanOptionBuilder
            .columns(Arrays.asList("name", "id"))
            .limit(10)
            .setColumnOrderings(columnOrderings);
        ScanOptions scanOptions = scanOptionBuilder.build();
        try (Scanner scanner = dataset.newScan(scanOptions)) {
          try (ArrowReader reader = scanner.scanBatches()) {
            while (reader.loadNextBatch()) {
              List<FieldVector> fieldVectors = reader.getVectorSchemaRoot().getFieldVectors();
              VarCharVector nameVector = (VarCharVector) fieldVectors.get(0);
              /* dataset context
               * i: |  id   | name | :i
               * 1: |  1    |  P0  | :0
               * 2: | null  |  P1  | :1
               * 3: |  2    |  P2  | :2
               * 5: | null  |  P3  | :3
               * 4: |  2    |  P3  | :4
               * 7: |  4    |  P4  | :5
               * 9: |  5    |  P5  | :6
               * 8: |  4    |  P5  | :7
               * 6: |  3    | null | :8
               * 0: |  0    | null | :9
               */
              assertEquals("P0", new String(nameVector.get(0)));
              assertEquals("P1", new String(nameVector.get(1)));
              assertEquals("P2", new String(nameVector.get(2)));
              assertEquals("P3", new String(nameVector.get(3)));
              assertEquals("P3", new String(nameVector.get(4)));
              assertEquals("P4", new String(nameVector.get(5)));
              assertEquals("P5", new String(nameVector.get(6)));
              assertEquals("P5", new String(nameVector.get(7)));
              assertTrue(nameVector.isNull(8));
              assertTrue(nameVector.isNull(9));

              IntVector idVector = (IntVector) fieldVectors.get(1);
              assertEquals(1, idVector.get(0));
              assertTrue(idVector.isNull(1));
              assertEquals(2, idVector.get(2));
              assertTrue(idVector.isNull(3));
              assertEquals(2, idVector.get(4));
              assertEquals(4, idVector.get(5));
              assertEquals(5, idVector.get(6));
              assertEquals(4, idVector.get(7));
              assertEquals(3, idVector.get(8));
              assertEquals(0, idVector.get(9));
            }
          }
        }
      }
    }
  }

  @Test
  void testDatasetScannerCombinedParams(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_combined_params").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 600;
      int limit = 200;
      int offset = 300;
      int batchSize = 50;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        try (Scanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .limit(limit)
                    .offset(offset)
                    .withRowId(true)
                    .batchSize(batchSize)
                    .batchReadahead(3)
                    .build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            List<String> fieldNames =
                root.getSchema().getFields().stream()
                    .map(Field::getName)
                    .collect(Collectors.toList());
            assertTrue(fieldNames.contains("_rowid"), "Schema should contain _rowid column");
            assertTrue(fieldNames.contains("id"), "Schema should contain id column");

            int rowCount = 0;
            int expectedIdStart = offset;
            while (reader.loadNextBatch()) {
              List<FieldVector> fieldVectors = root.getFieldVectors();
              assertTrue(
                  fieldVectors.stream().anyMatch(vector -> vector.getName().equals("_rowid")));
              IntVector idVector = (IntVector) root.getVector("id");
              int batchRowCount = root.getRowCount();
              rowCount += batchRowCount;
              assertTrue(batchRowCount <= batchSize, "Batch size should not exceed " + batchSize);

              for (int i = 0; i < batchRowCount; i++) {
                int expectedId = expectedIdStart + i;
                assertEquals(
                    expectedId,
                    idVector.get(i),
                    "Mismatch at row "
                        + (rowCount - batchRowCount + i)
                        + ". Expected: "
                        + expectedId
                        + ", Actual: "
                        + idVector.get(i));
              }
              expectedIdStart += batchRowCount;
            }
            assertEquals(limit, rowCount, "Total rows should match the limit");
          }
        }
      }
    }
  }

  @Test
  void testUseScalarIndex(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_use_scalar_index").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 100;
      try (Dataset dataset = testDataset.write(1, totalRows)) {
        // Create a scalar index on the 'id' column
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        IndexOptions options =
            IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, indexParams)
                .withIndexName("id_btree_index")
                .replace(true)
                .build();
        dataset.createIndex(options);

        // Verify index was created
        assertTrue(
            dataset.listIndexes().contains("id_btree_index"),
            "Expected 'id_btree_index' to be in the list of indexes: " + dataset.listIndexes());

        // Test with useScalarIndex = true (default)
        List<Integer> resultsWithIndex = new ArrayList<>();
        try (Scanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .filter("id < 50")
                    .useScalarIndex(true)
                    .columns(Collections.singletonList("id"))
                    .build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            while (reader.loadNextBatch()) {
              VectorSchemaRoot root = reader.getVectorSchemaRoot();
              IntVector idVector = (IntVector) root.getVector("id");
              for (int i = 0; i < root.getRowCount(); i++) {
                resultsWithIndex.add(idVector.get(i));
              }
            }
          }
        }

        // Test with useScalarIndex = false
        List<Integer> resultsWithoutIndex = new ArrayList<>();
        try (Scanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .filter("id < 50")
                    .useScalarIndex(false)
                    .columns(Collections.singletonList("id"))
                    .build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            while (reader.loadNextBatch()) {
              VectorSchemaRoot root = reader.getVectorSchemaRoot();
              IntVector idVector = (IntVector) root.getVector("id");
              for (int i = 0; i < root.getRowCount(); i++) {
                resultsWithoutIndex.add(idVector.get(i));
              }
            }
          }
        }

        // Results should be the same regardless of whether scalar index is used
        assertEquals(
            resultsWithIndex.size(),
            resultsWithoutIndex.size(),
            "Result count should be the same with or without scalar index");
        assertEquals(50, resultsWithIndex.size(), "Should return 50 rows (id < 50)");
        assertEquals(
            resultsWithIndex,
            resultsWithoutIndex,
            "Results should be identical with or without scalar index");
      }
    }
  }

  @Test
  void testFastSearchSkipsUnindexedFragments(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("dataset_scanner_fast_search_scalar_index").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 100)) {
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        IndexOptions options =
            IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, indexParams)
                .withIndexName("id_btree_index")
                .replace(true)
                .build();
        dataset.createIndex(options);

        FragmentMetadata metadata = testDataset.createNewFragment(10);
        FragmentOperation.Append appendOp =
            new FragmentOperation.Append(Collections.singletonList(metadata));
        try (Dataset appended =
            Dataset.commit(allocator, datasetPath, appendOp, Optional.of(dataset.version()))) {
          try (LanceScanner scanner =
              appended.newScan(new ScanOptions.Builder().filter("id < 5").build())) {
            assertEquals(10, scanner.countRows());
          }

          try (LanceScanner scanner =
              appended.newScan(
                  new ScanOptions.Builder().filter("id < 5").fastSearch(true).build())) {
            assertEquals(5, scanner.countRows());
          }
        }
      }
    }
  }

  @Test
  void testIncludeDeletedRows(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("include_deleted_rows").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 10)) {
        assertEquals(10, dataset.countRows());

        // Delete rows where id >= 5
        dataset.delete("id >= 5");
        assertEquals(5, dataset.countRows());

        // Default scan should exclude deleted rows
        try (LanceScanner scanner =
            dataset.newScan(new ScanOptions.Builder().batchSize(20).build())) {
          assertEquals(5, scanner.countRows(), "default scan: should exclude deleted rows");
        }

        // includeDeletedRows=true should surface deleted rows
        // NOTE: includeDeletedRows requires withRowId=true
        try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .batchSize(20)
                    .withRowId(true)
                    .includeDeletedRows(true)
                    .build())) {
          assertEquals(10, scanner.countRows(), "includeDeletedRows: should include deleted rows");
        }
      }
    }
  }

  @Test
  void testStrictBatchSize(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("strict_batch_size").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 25)) {
        int batchSize = 10;

        // With strictBatchSize=true, no batch should exceed batchSize
        try (Scanner scanner =
            dataset.newScan(
                new ScanOptions.Builder().batchSize(batchSize).strictBatchSize(true).build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            int totalRows = 0;
            while (reader.loadNextBatch()) {
              int rows = reader.getVectorSchemaRoot().getRowCount();
              assertTrue(rows <= batchSize, "strict: batch " + rows + " should be <= " + batchSize);
              totalRows += rows;
            }
            assertEquals(25, totalRows);
          }
        }

        // strictBatchSize=false (default) — batch size may vary
        try (Scanner scanner =
            dataset.newScan(new ScanOptions.Builder().batchSize(batchSize).build())) {
          try (ArrowReader reader = scanner.scanBatches()) {
            int totalRows = 0;
            while (reader.loadNextBatch()) {
              totalRows += reader.getVectorSchemaRoot().getRowCount();
            }
            assertEquals(25, totalRows);
          }
        }
      }
    }
  }

  @Test
  void testDisableScoringAutoprojection(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("disable_scoring_autoprojection").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 10)) {
        // Smoke test: verify the option is accepted and scan still works
        ScanOptions options =
            new ScanOptions.Builder().batchSize(20).disableScoringAutoprojection(true).build();

        try (LanceScanner scanner = dataset.newScan(options)) {
          assertEquals(
              10,
              scanner.countRows(),
              "scan with disableScoringAutoprojection should return all rows");
        }

        // Also verify it doesn't break when combined with other options
        ScanOptions combinedOptions =
            new ScanOptions.Builder()
                .batchSize(20)
                .filter("id < 5")
                .disableScoringAutoprojection(true)
                .includeDeletedRows(false)
                .strictBatchSize(false)
                .build();

        try (LanceScanner scanner = dataset.newScan(combinedOptions)) {
          assertEquals(
              5,
              scanner.countRows(),
              "scan with disableScoringAutoprojection + filter should work");
        }
      }
    }
  }

  private void validScanResult(Dataset dataset, int fragmentId, int rowCount) throws Exception {
    try (Scanner scanner =
        dataset.newScan(
            new ScanOptions.Builder()
                .batchSize(1024)
                .fragmentIds(Arrays.asList(fragmentId))
                .build())) {
      try (ArrowReader reader = scanner.scanBatches()) {
        assertEquals(
            dataset.getSchema().getFields(), reader.getVectorSchemaRoot().getSchema().getFields());
        reader.loadNextBatch();
        assertEquals(rowCount, reader.getVectorSchemaRoot().getRowCount());
        assertFalse(reader.loadNextBatch());
      }
    }
  }
}
