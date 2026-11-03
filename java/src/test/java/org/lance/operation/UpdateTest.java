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
package org.lance.operation;

import org.lance.CommitBuilder;
import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.FragmentMetadata;
import org.lance.TestUtils;
import org.lance.Transaction;
import org.lance.fragment.FragmentUpdateResult;
import org.lance.ipc.LanceScanner;
import org.lance.operation.Update.UpdateMode;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.TimeStampSecTZVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class UpdateTest extends OperationTestBase {

  @Test
  void testUpdate(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testUpdate").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Commit fragment
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build()) {
        try (Dataset dataset = new CommitBuilder(this.dataset).execute(appendTxn)) {
          assertEquals(2, dataset.version());
          assertEquals(2, dataset.latestVersion());
          assertEquals(rowCount, dataset.countRows());
          assertThrows(
              IllegalArgumentException.class,
              () -> {
                try (Transaction txn =
                    new Transaction.Builder()
                        .readVersion(dataset.version())
                        .operation(Append.builder().fragments(new ArrayList<>()).build())
                        .build()) {
                  new CommitBuilder(dataset).execute(txn).close();
                }
              });
        }
      }

      dataset = Dataset.open(datasetPath, allocator);
      // Update fragments
      rowCount = 40;
      FragmentMetadata newFragment = testDataset.createNewFragment(rowCount);
      try (Transaction updateTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Update.builder()
                      .removedFragmentIds(
                          Collections.singletonList(
                              Long.valueOf(dataset.getFragments().get(0).getId())))
                      .newFragments(Collections.singletonList(newFragment))
                      .updateMode(Optional.of(UpdateMode.RewriteRows))
                      .build())
              .build()) {
        try (Dataset dataset = new CommitBuilder(this.dataset).execute(updateTxn)) {
          assertEquals(3, dataset.version());
          assertEquals(3, dataset.latestVersion());
          assertEquals(rowCount, dataset.countRows());
        }
      }
    }
  }

  @Test
  void testUpdatedFragmentOffsetsRoundTrip(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testUpdatedFragmentOffsetsRoundTrip").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Append an initial fragment so we have a real fragment id.
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build()) {
        new CommitBuilder(dataset).execute(appendTxn).close();
      }

      dataset = Dataset.open(datasetPath, allocator);
      Fragment existingFragment = dataset.getFragments().get(0);
      long fragmentId = existingFragment.getId();
      // Use the committed fragment's own metadata as the updatedFragment so that
      // the updatedFragmentOffsets key is valid (must match a fragment in updatedFragments).
      FragmentMetadata existingFragmentMeta = existingFragment.metadata();

      // Build Update with non-empty updatedFragmentOffsets. Values are portable RoaringBitmap
      // bytes encoding {1, 3, 5}: cookie(4) + containerCount(4) + key(2) + card-1(2) +
      // offset(4) + elems(6). Offset = 16 (start of container data from beginning of stream).
      Map<Long, byte[]> offsets = new HashMap<>();
      offsets.put(
          fragmentId,
          new byte[] {
            (byte) 0x3A,
            (byte) 0x30,
            (byte) 0x00,
            (byte) 0x00, // cookie = 12346
            (byte) 0x01,
            (byte) 0x00,
            (byte) 0x00,
            (byte) 0x00, // 1 container
            (byte) 0x00,
            (byte) 0x00, // container key 0
            (byte) 0x02,
            (byte) 0x00, // cardinality - 1 = 2
            (byte) 0x10,
            (byte) 0x00,
            (byte) 0x00,
            (byte) 0x00, // offset = 16
            (byte) 0x01,
            (byte) 0x00, // element 1
            (byte) 0x03,
            (byte) 0x00, // element 3
            (byte) 0x05,
            (byte) 0x00 // element 5
          });

      try (Transaction updateTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Update.builder()
                      .updatedFragments(Collections.singletonList(existingFragmentMeta))
                      .updateMode(Optional.of(UpdateMode.RewriteColumns))
                      .updatedFragmentOffsets(offsets)
                      .build())
              .build()) {
        try (Dataset committed = new CommitBuilder(dataset).execute(updateTxn)) {
          // Read the committed transaction back (exercises the IntoJava JNI path).
          try (Transaction readTx = committed.readTransaction().orElseThrow()) {
            assertInstanceOf(Update.class, readTx.operation());
            Update readOp = (Update) readTx.operation();

            Map<Long, byte[]> readOffsets = readOp.updatedFragmentOffsets();
            assertEquals(1, readOffsets.size());
            assertArrayEquals(offsets.get(fragmentId), readOffsets.get(fragmentId));
          }
        }
      }
    }
  }

  @Test
  void testUpdateColumns(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testUpdateColumns").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.UpdateColumnTestDataset testDataset =
          new TestUtils.UpdateColumnTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      /* dataset content
       * _rowid |   id   |     name     | timeStamp |
       *   0:   |    0   |  "Person 0"  |     0     |
       *   1:   |    1   |  "Person 1"  |    null   |
       *   2:   |  null  |     null     |     2     |
       *   3:   |  null  |     null     |    null   |
       *   4:   |    4   |  "Person 4"  |     4     |
       *   5:   |  null  |     null     |    null   |
       */
      int rowCount = 6;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
              .build()) {
        try (Dataset dataset = new CommitBuilder(this.dataset).execute(appendTxn)) {
          assertEquals(2, dataset.version());
          assertEquals(2, dataset.latestVersion());
          assertEquals(rowCount, dataset.countRows());
        }
      }

      dataset = Dataset.open(datasetPath, allocator);
      Fragment targetFragment = dataset.getFragments().get(0);
      int updateRowCount = 4;
      /* source fragment content
       * _rowid |   id   |     name     |
       *   0:   |   100  |  "Update 0"  |
       *   1:   |  null  |     null     |
       *   2:   |    2   |  "Update 2"  |
       *   3:   |  null  |     null     |
       */
      FragmentUpdateResult updateResult = testDataset.updateColumn(targetFragment, updateRowCount);
      try (Transaction updateTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Update.builder()
                      .updatedFragments(
                          Collections.singletonList(updateResult.getUpdatedFragment()))
                      .fieldsModified(updateResult.getFieldsModified())
                      .updateMode(Optional.of(UpdateMode.RewriteColumns))
                      .build())
              .build()) {
        try (Dataset dataset = new CommitBuilder(this.dataset).execute(updateTxn)) {
          assertEquals(3, dataset.version());
          assertEquals(3, dataset.latestVersion());
          Fragment fragment = dataset.getFragments().get(0);
          try (LanceScanner scanner = fragment.newScan(rowCount)) {
            List<Integer> actualIds = new ArrayList<>(rowCount);
            List<String> actualNames = new ArrayList<>(rowCount);
            List<Long> actualTimeStamps = new ArrayList<>(rowCount);
            try (ArrowReader reader = scanner.scanBatches()) {
              while (reader.loadNextBatch()) {
                VectorSchemaRoot root = reader.getVectorSchemaRoot();
                IntVector idVector = (IntVector) root.getVector("id");
                for (int i = 0; i < idVector.getValueCount(); i++) {
                  actualIds.add(idVector.isNull(i) ? null : idVector.getObject(i));
                }
                VarCharVector nameVector = (VarCharVector) root.getVector("name");
                for (int i = 0; i < nameVector.getValueCount(); i++) {
                  actualNames.add(nameVector.isNull(i) ? null : nameVector.getObject(i).toString());
                }
                TimeStampSecTZVector timeStampVector =
                    (TimeStampSecTZVector) root.getVector("timeStamp");
                for (int i = 0; i < timeStampVector.getValueCount(); i++) {
                  actualTimeStamps.add(
                      timeStampVector.isNull(i) ? null : timeStampVector.getObject(i));
                }
              }
            }
            /* result dataset content
             * _rowid |   id   |     name     | timeStamp |
             *   0:   |   100  |  "Update 0"  |     0     |
             *   1:   |  null  |     null     |    null   |
             *   2:   |    2   |  "Update 2"  |     2     |
             *   3:   |  null  |     null     |    null   |
             *   4:   |    4   |  "Person 4"  |     4     |
             *   5:   |  null  |     null     |    null   |
             */
            List<Integer> expectIds = Arrays.asList(100, null, 2, null, 4, null);
            List<String> expectNames =
                Arrays.asList("Update 0", null, "Update 2", null, "Person 4", null);
            List<Long> expectTimeStamps = Arrays.asList(0L, null, 2L, null, 4L, null);
            assertEquals(expectIds, actualIds);
            assertEquals(expectNames, actualNames);
            assertEquals(expectTimeStamps, actualTimeStamps);
          }
        }
      }
    }
  }
}
