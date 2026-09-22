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
import org.lance.WriteParams;
import org.lance.ipc.LanceScanner;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class OverwriteTest extends OperationTestBase {

  @Test
  void testOverwrite(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testOverwrite").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Commit fragment
      int rowCount = 20;
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(rowCount);
      try (Transaction txn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Overwrite.builder()
                      .fragments(Collections.singletonList(fragmentMeta))
                      .schema(testDataset.getSchema())
                      .build())
              .build()) {
        try (Dataset dataset = new CommitBuilder(this.dataset).execute(txn)) {
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

      // Try to commit from stale version (v1) - should fail with retryable error
      rowCount = 40;
      fragmentMeta = testDataset.createNewFragment(rowCount);
      try (Transaction staleTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Overwrite.builder()
                      .fragments(Collections.singletonList(fragmentMeta))
                      .schema(testDataset.getSchema())
                      .configUpsertValues(Collections.singletonMap("config_key", "config_value"))
                      .build())
              .transactionProperties(Collections.singletonMap("key", "value"))
              .build()) {
        assertEquals("value", staleTxn.transactionProperties().map(m -> m.get("key")).orElse(null));

        RuntimeException ex =
            assertThrows(
                RuntimeException.class, () -> new CommitBuilder(dataset).execute(staleTxn).close());
        assertTrue(
            ex.getMessage().contains("Retryable commit conflict"),
            "Expected retryable commit conflict error, got: " + ex.getMessage());
      }

      // Checkout latest and retry - should succeed
      dataset.checkoutLatest();
      try (Transaction retryTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Overwrite.builder()
                      .fragments(Collections.singletonList(fragmentMeta))
                      .schema(testDataset.getSchema())
                      .configUpsertValues(Collections.singletonMap("config_key", "config_value"))
                      .build())
              .transactionProperties(Collections.singletonMap("key", "value"))
              .build()) {
        try (Dataset dataset = new CommitBuilder(this.dataset).execute(retryTxn)) {
          assertEquals(3, dataset.version());
          assertEquals(3, dataset.latestVersion());
          assertEquals(rowCount, dataset.countRows());
          assertEquals("config_value", dataset.getConfig().get("config_key"));
          Fragment fragment = dataset.getFragments().get(0);

          try (LanceScanner scanner = fragment.newScan()) {
            Schema schemaRes = scanner.schema();
            assertEquals(testDataset.getSchema(), schemaRes);
          }
          try (Transaction committedTxn = dataset.readTransaction().orElseThrow()) {
            assertEquals(retryTxn.readVersion(), committedTxn.readVersion());
            assertEquals(retryTxn.uuid(), committedTxn.uuid());
            assertEquals(retryTxn.transactionProperties(), committedTxn.transactionProperties());
            Overwrite committedOverwrite = (Overwrite) committedTxn.operation();
            assertEquals(testDataset.getSchema(), committedOverwrite.schema());
            assertEquals(
                Collections.singletonMap("config_key", "config_value"),
                committedOverwrite.configUpsertValues().orElseThrow());
            assertArrayEquals(
                new int[] {0, 1},
                committedOverwrite.fragments().get(0).getFiles().get(0).getFields());
          }
        }
      }
    }
  }

  @Test
  void testOverwriteWithDifferentFieldTypes(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testOverwriteFieldTypes").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      // Create initial dataset with schema: id (int32), name (utf8)
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      FragmentMetadata fragmentMeta = testDataset.createNewFragment(10);
      try (Transaction txn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(
                  Overwrite.builder()
                      .fragments(Collections.singletonList(fragmentMeta))
                      .schema(testDataset.getSchema())
                      .build())
              .build()) {
        dataset = new CommitBuilder(this.dataset).execute(txn);
      }
      assertEquals(2, dataset.version());
      assertEquals(10, dataset.countRows());

      // Overwrite with a new schema where "id" changes from int32 to int64
      // and "name" changes from utf8 to int64
      Schema newSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(64, true)),
                  Field.nullable("name", new ArrowType.Int(64, true))));

      int newRowCount = 5;
      List<FragmentMetadata> newFragments;
      try (VectorSchemaRoot root = VectorSchemaRoot.create(newSchema, allocator)) {
        root.allocateNew();
        BigIntVector idVector = (BigIntVector) root.getVector("id");
        BigIntVector nameVector = (BigIntVector) root.getVector("name");
        for (int i = 0; i < newRowCount; i++) {
          idVector.setSafe(i, (long) i * 100);
          nameVector.setSafe(i, (long) i * 200);
        }
        root.setRowCount(newRowCount);
        newFragments =
            Fragment.create(datasetPath, allocator, root, new WriteParams.Builder().build());
      }

      try (Transaction txn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(Overwrite.builder().fragments(newFragments).schema(newSchema).build())
              .build()) {
        try (Dataset overwritten = new CommitBuilder(this.dataset).execute(txn)) {
          assertEquals(3, overwritten.version());
          assertEquals(newRowCount, overwritten.countRows());

          // Verify the schema has the new types
          Schema resultSchema = overwritten.getSchema();
          assertEquals(newSchema, resultSchema);
        }
      }
    }
  }
}
