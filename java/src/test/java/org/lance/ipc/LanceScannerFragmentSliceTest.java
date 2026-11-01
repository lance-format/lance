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

import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.FragmentMetadata;
import org.lance.FragmentOperation;
import org.lance.LanceConstants;
import org.lance.TestUtils;
import org.lance.WriteParams;
import org.lance.update.UpdateParams;

import com.google.common.collect.ImmutableMap;
import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LanceScannerFragmentSliceTest {

  @Test
  void testFragmentSliceValueSemanticsAndValidation() {
    FragmentSlice slice = new FragmentSlice(7, 11, 13);

    assertEquals(7, slice.getFragmentId());
    assertEquals(11, slice.getRowOffset());
    assertEquals(13, slice.getRowCount());
    assertEquals(slice, new FragmentSlice(7, 11, 13));
    assertEquals(slice.hashCode(), new FragmentSlice(7, 11, 13).hashCode());
    assertTrue(slice.toString().contains("fragmentId=7"));

    assertThrows(IllegalArgumentException.class, () -> new FragmentSlice(-1, 0, 1));
    assertThrows(IllegalArgumentException.class, () -> new FragmentSlice(0, -1, 1));
    assertThrows(IllegalArgumentException.class, () -> new FragmentSlice(0, 0, -1));
    assertThrows(IllegalArgumentException.class, () -> new FragmentSlice(0, Long.MAX_VALUE, 1));
  }

  @Test
  void testScanOptionsPreserveFragmentSlices() {
    FragmentSlice slice = new FragmentSlice(7, 11, 13);
    ScanOptions options =
        new ScanOptions.Builder().fragmentSlices(Collections.singletonList(slice)).build();

    assertEquals(Collections.singletonList(slice), options.getFragmentSlices().orElseThrow());
    assertEquals(
        options.getFragmentSlices(), new ScanOptions.Builder(options).build().getFragmentSlices());
    assertTrue(options.toString().contains("fragmentSlices"));
  }

  @Test
  void testOverlappingSlicesAreMergedAcrossFragments(@TempDir Path tempDir) throws Exception {
    WriteParams writeParams = stableStorageWriteParams(false);
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = createDataset(allocator, tempDir, writeParams, 5, 6)) {
      List<Fragment> fragments = dataset.getFragments();
      int firstFragmentId = fragments.get(0).getId();
      int secondFragmentId = fragments.get(1).getId();
      ScanOptions options =
          new ScanOptions.Builder()
              .fragmentSlices(
                  Arrays.asList(
                      new FragmentSlice(firstFragmentId, 1, 3),
                      new FragmentSlice(firstFragmentId, 2, 2),
                      new FragmentSlice(secondFragmentId, 4, 2)))
              .columns(Collections.singletonList("id"))
              .build();

      try (LanceScanner scanner = dataset.newScan(options)) {
        assertEquals(Arrays.asList(1, 2, 3, 4, 5), readIds(scanner));
      }
    }
  }

  @Test
  void testFilterAndLimitApplyAfterSlices(@TempDir Path tempDir) throws Exception {
    WriteParams writeParams = stableStorageWriteParams(false);
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = createDataset(allocator, tempDir, writeParams, 8)) {
      int fragmentId = dataset.getFragments().get(0).getId();
      ScanOptions options =
          new ScanOptions.Builder()
              .fragmentSlices(Collections.singletonList(new FragmentSlice(fragmentId, 1, 5)))
              .columns(Collections.singletonList("id"))
              .filter("id >= 2")
              .limit(3)
              .build();

      try (LanceScanner scanner = dataset.newScan(options)) {
        assertEquals(Arrays.asList(2, 3, 4), readIds(scanner));
      }
    }
  }

  @Test
  void testFragmentScopesIntersectSlices(@TempDir Path tempDir) throws Exception {
    WriteParams writeParams = stableStorageWriteParams(false);
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = createDataset(allocator, tempDir, writeParams, 5, 6)) {
      List<Fragment> fragments = dataset.getFragments();
      int firstFragmentId = fragments.get(0).getId();
      int secondFragmentId = fragments.get(1).getId();
      ScanOptions options =
          new ScanOptions.Builder()
              .fragmentSlices(
                  Arrays.asList(
                      new FragmentSlice(firstFragmentId, 1, 3),
                      new FragmentSlice(secondFragmentId, 4, 2)))
              .fragmentIds(Collections.singletonList(secondFragmentId))
              .columns(Collections.singletonList("id"))
              .build();

      try (LanceScanner scanner = dataset.newScan(options)) {
        assertEquals(Arrays.asList(4, 5), readIds(scanner));
      }

      ScanOptions allSliceFragments =
          new ScanOptions.Builder(options)
              .fragmentIds(Arrays.asList(firstFragmentId, secondFragmentId))
              .build();
      try (LanceScanner scanner = fragments.get(0).newScan(allSliceFragments)) {
        assertEquals(Arrays.asList(1, 2, 3), readIds(scanner));
      }
    }
  }

  @Test
  void testDeletedRowsDoNotShiftPhysicalOffsets(@TempDir Path tempDir) throws Exception {
    WriteParams writeParams = stableStorageWriteParams(true);
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = createDataset(allocator, tempDir, writeParams, 8)) {
      assertTrue(dataset.hasStableRowIds());
      int fragmentId = dataset.getFragments().get(0).getId();
      dataset.delete("id = 2");
      ScanOptions options =
          new ScanOptions.Builder()
              .fragmentSlices(Collections.singletonList(new FragmentSlice(fragmentId, 1, 5)))
              .columns(Collections.singletonList("id"))
              .build();

      try (LanceScanner scanner = dataset.newScan(options)) {
        assertEquals(Arrays.asList(1, 3, 4, 5), readIds(scanner));
      }
    }
  }

  @Test
  void testCountScanRespectsSlices(@TempDir Path tempDir) throws Exception {
    WriteParams writeParams = stableStorageWriteParams(true);
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = createDataset(allocator, tempDir, writeParams, 8)) {
      int fragmentId = dataset.getFragments().get(0).getId();
      dataset.delete("id = 2");
      ScanOptions options =
          new ScanOptions.Builder()
              .fragmentSlices(Collections.singletonList(new FragmentSlice(fragmentId, 1, 5)))
              .columns(Collections.emptyList())
              .withRowId(true)
              .build();

      try (LanceScanner scanner = dataset.newScan(options);
          ArrowReader reader = scanner.scanBatches()) {
        long rowCount = 0;
        while (reader.loadNextBatch()) {
          rowCount += reader.getVectorSchemaRoot().getRowCount();
        }
        assertEquals(4, rowCount);
      }
    }
  }

  @Test
  void testIncludeDeletedRowsRespectsSlices(@TempDir Path tempDir) throws Exception {
    WriteParams writeParams = stableStorageWriteParams(true);
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = createDataset(allocator, tempDir, writeParams, 8)) {
      int fragmentId = dataset.getFragments().get(0).getId();
      dataset.delete("id = 2");
      ScanOptions options =
          new ScanOptions.Builder()
              .fragmentSlices(Collections.singletonList(new FragmentSlice(fragmentId, 1, 5)))
              .columns(Collections.singletonList("id"))
              .withRowId(true)
              .includeDeletedRows(true)
              .build();

      try (LanceScanner scanner = dataset.newScan(options);
          ArrowReader reader = scanner.scanBatches()) {
        assertTrue(reader.loadNextBatch());
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        IntVector ids = (IntVector) root.getVector("id");
        UInt8Vector rowIds = (UInt8Vector) root.getVector("_rowid");
        List<Integer> actualIds = new ArrayList<>();
        for (int index = 0; index < root.getRowCount(); index++) {
          actualIds.add(ids.get(index));
          assertEquals(index == 1, rowIds.isNull(index));
        }
        assertEquals(Arrays.asList(1, 2, 3, 4, 5), actualIds);
        assertFalse(reader.loadNextBatch());
      }
    }
  }

  @Test
  void testUpdatedStableIdsDoNotEscapePhysicalSlices(@TempDir Path tempDir) throws Exception {
    WriteParams writeParams = stableStorageWriteParams(true);
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = createDataset(allocator, tempDir, writeParams, 8)) {
      int originalFragmentId = dataset.getFragments().get(0).getId();
      try (Dataset updated =
          dataset
              .update(
                  new UpdateParams(ImmutableMap.of("name", "'updated'"))
                      .withWhere("id = 2 OR id = 3"))
              .getDataset()) {
        int replacementFragmentId =
            updated.getFragments().stream()
                .map(Fragment::getId)
                .filter(fragmentId -> fragmentId != originalFragmentId)
                .findFirst()
                .orElseThrow();
        ScanOptions options =
            new ScanOptions.Builder()
                .fragmentSlices(
                    Arrays.asList(
                        new FragmentSlice(originalFragmentId, 2, 1),
                        new FragmentSlice(replacementFragmentId, 1, 1)))
                .columns(Collections.singletonList("id"))
                .build();

        try (LanceScanner scanner = updated.newScan(options)) {
          assertEquals(Collections.singletonList(3), readIds(scanner));
        }
      }
    }
  }

  @Test
  void testEmptySlicesReturnNoRows(@TempDir Path tempDir) throws Exception {
    WriteParams writeParams = stableStorageWriteParams(false);
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = createDataset(allocator, tempDir, writeParams, 5)) {
      int fragmentId = dataset.getFragments().get(0).getId();
      ScanOptions zeroLengthSlice =
          new ScanOptions.Builder()
              .fragmentSlices(Collections.singletonList(new FragmentSlice(fragmentId, 5, 0)))
              .columns(Collections.singletonList("id"))
              .build();
      try (LanceScanner scanner = dataset.newScan(zeroLengthSlice)) {
        assertEquals(Collections.emptyList(), readIds(scanner));
      }

      ScanOptions emptySliceList =
          new ScanOptions.Builder()
              .fragmentSlices(Collections.emptyList())
              .columns(Collections.singletonList("id"))
              .build();
      try (LanceScanner scanner = dataset.newScan(emptySliceList)) {
        assertEquals(Collections.emptyList(), readIds(scanner));
      }
    }
  }

  @Test
  void testInvalidNativeSlicesAreRejected(@TempDir Path tempDir) throws Exception {
    WriteParams writeParams = stableStorageWriteParams(false);
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = createDataset(allocator, tempDir, writeParams, 5)) {
      int fragmentId = dataset.getFragments().get(0).getId();

      assertThrows(
          IllegalArgumentException.class,
          () ->
              dataset.newScan(
                  new ScanOptions.Builder()
                      .fragmentSlices(
                          Collections.singletonList(new FragmentSlice(fragmentId, 4, 2)))
                      .build()));
      assertThrows(
          IllegalArgumentException.class,
          () ->
              dataset.newScan(
                  new ScanOptions.Builder()
                      .fragmentSlices(Collections.singletonList(new FragmentSlice(999, 0, 1)))
                      .build()));
    }
  }

  @Test
  void testLegacyStorageRejectsSlices(@TempDir Path tempDir) throws Exception {
    WriteParams writeParams =
        new WriteParams.Builder()
            .withDataStorageVersion(LanceConstants.FILE_FORMAT_VERSION_LEGACY)
            .build();
    try (BufferAllocator allocator = new RootAllocator();
        Dataset dataset = createDataset(allocator, tempDir, writeParams, 5);
        LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .fragmentSlices(
                        Collections.singletonList(
                            new FragmentSlice(dataset.getFragments().get(0).getId(), 0, 2)))
                    .build())) {
      UnsupportedOperationException error =
          assertThrows(UnsupportedOperationException.class, scanner::scanBatches);
      assertTrue(error.getMessage().contains("legacy-storage"));
    }
  }

  private static WriteParams stableStorageWriteParams(boolean enableStableRowIds) {
    return new WriteParams.Builder()
        .withDataStorageVersion(LanceConstants.FILE_FORMAT_VERSION_STABLE)
        .withEnableStableRowIds(enableStableRowIds)
        .build();
  }

  private static Dataset createDataset(
      BufferAllocator allocator,
      Path datasetPath,
      WriteParams writeParams,
      int... fragmentRowCounts)
      throws Exception {
    String path = datasetPath.toString();
    TestUtils.SimpleTestDataset testDataset = new TestUtils.SimpleTestDataset(allocator, path);
    testDataset.createDatasetWithWriteParams(writeParams).close();
    List<FragmentMetadata> metadata = new ArrayList<>();
    for (int rowCount : fragmentRowCounts) {
      metadata.addAll(testDataset.createNewFragment(rowCount, writeParams));
    }
    return Dataset.commit(allocator, path, new FragmentOperation.Append(metadata), Optional.of(1L));
  }

  private static List<Integer> readIds(Scanner scanner) throws IOException {
    List<Integer> ids = new ArrayList<>();
    try (ArrowReader reader = scanner.scanBatches()) {
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      while (reader.loadNextBatch()) {
        IntVector vector = (IntVector) root.getVector("id");
        for (int index = 0; index < root.getRowCount(); index++) {
          ids.add(vector.get(index));
        }
      }
    }
    return ids;
  }
}
