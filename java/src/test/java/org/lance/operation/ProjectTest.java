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
import org.lance.TestUtils;
import org.lance.Transaction;
import org.lance.ipc.LanceScanner;
import org.lance.schema.LanceField;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ProjectTest extends OperationTestBase {

  @Test
  void testProjection(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testProjection").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      assertEquals(testDataset.getSchema(), dataset.getSchema());
      List<Field> fieldList = new ArrayList<>(testDataset.getSchema().getFields());
      Collections.reverse(fieldList);
      try (Transaction txn1 =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(Project.builder().schema(new Schema(fieldList)).build())
              .build()) {
        try (Dataset committedDataset = new CommitBuilder(dataset).execute(txn1)) {
          assertEquals(1, dataset.version());
          assertEquals(2, committedDataset.version());
          assertEquals(new Schema(fieldList), committedDataset.getSchema());
          fieldList.remove(1);
          try (Transaction txn2 =
              new Transaction.Builder()
                  .readVersion(committedDataset.version())
                  .operation(Project.builder().schema(new Schema(fieldList)).build())
                  .build()) {
            try (Dataset committedDataset2 = new CommitBuilder(committedDataset).execute(txn2)) {
              assertEquals(2, committedDataset.version());
              assertEquals(3, committedDataset2.version());
              assertEquals(new Schema(fieldList), committedDataset2.getSchema());
            }
          }
        }
      }
    }
  }

  @Test
  void testPreservesNullabilityEqualityAndRoundTrip(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testAssertsNonNull").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      Schema schema = testDataset.getSchema();

      // The assertion is part of the operation's identity.
      assertNotEquals(
          Project.builder().schema(schema).preservesNullability(true).build(),
          Project.builder().schema(schema).preservesNullability(false).build());
      assertEquals(
          Project.builder().schema(schema).preservesNullability(true).build(),
          Project.builder().schema(schema).preservesNullability(true).build());

      // The explicit non-default assertion must survive the JNI round trip.
      try (Transaction txn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(Project.builder().schema(schema).preservesNullability(true).build())
              .build()) {
        try (Dataset committed = new CommitBuilder(dataset).execute(txn)) {
          Transaction readBack = committed.readTransaction().orElseThrow();
          Project project = (Project) readBack.operation();
          assertTrue(project.preservesNullability());
        }
      }
    }
  }

  @Test
  void testProjectPreservesExplicitRenameIdentity(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testProjectPreservesExplicitRenameIdentity").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      Field existing = dataset.getSchema().getFields().get(0);
      int existingId =
          dataset.getLanceSchema().fields().stream()
              .filter(field -> field.getName().equals(existing.getName()))
              .findFirst()
              .map(LanceField::getId)
              .orElseThrow();
      Map<String, String> metadata = new HashMap<>(existing.getMetadata());
      metadata.put("lance:field_id", String.valueOf(existingId));
      Field renamed =
          new Field(
              "renamed",
              new FieldType(
                  existing.isNullable(), existing.getType(), existing.getDictionary(), metadata),
              existing.getChildren());

      try (Transaction transaction =
              new Transaction.Builder()
                  .readVersion(dataset.version())
                  .operation(
                      Project.builder()
                          .schema(new Schema(Collections.singletonList(renamed)))
                          .build())
                  .build();
          Dataset committed = new CommitBuilder(dataset).execute(transaction)) {
        LanceField committedField = committed.getLanceSchema().fields().get(0);
        assertEquals("renamed", committedField.getName());
        assertEquals(existingId, committedField.getId());
      }
    }
  }

  @Test
  void testLegacyProjectPreservesNonContiguousFieldIds(@TempDir Path tempDir) throws Exception {
    Path source =
        Path.of("..", "test_data", "v0.10.5", "corrupt_schema").toAbsolutePath().normalize();
    Path datasetPath = tempDir.resolve("legacy");
    copyDirectory(source, datasetPath);

    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE);
        Dataset legacy = Dataset.open(datasetPath.toString(), allocator)) {
      legacy.dropColumns(Collections.singletonList("y"));
      List<Field> projectedFields = new ArrayList<>(legacy.getSchema().getFields());
      Collections.reverse(projectedFields);

      try (Transaction transaction =
              new Transaction.Builder()
                  .readVersion(legacy.version())
                  .operation(Project.builder().schema(new Schema(projectedFields)).build())
                  .build();
          Dataset projected = new CommitBuilder(legacy).execute(transaction)) {
        List<LanceField> fields = projected.getLanceSchema().fields();
        assertEquals(5, findField(fields, "c").getId());
        assertEquals(4, findField(fields, "b").getId());
        assertEquals(0, findField(fields, "x").getId());

        try (LanceScanner scanner = projected.newScan();
            ArrowReader reader = scanner.scanBatches()) {
          assertTrue(reader.loadNextBatch());
          VectorSchemaRoot root = reader.getVectorSchemaRoot();
          assertEquals("c", root.getSchema().getFields().get(0).getName());
          assertEquals("b", root.getSchema().getFields().get(1).getName());
          assertEquals("x", root.getSchema().getFields().get(2).getName());
          assertEquals(0L, root.getVector("c").getObject(0));
          assertEquals(0L, root.getVector("b").getObject(0));
          assertEquals(0L, root.getVector("x").getObject(0));
          assertEquals(5L, root.getVector("c").getObject(1));
          assertEquals(4L, root.getVector("b").getObject(1));
          assertEquals(1L, root.getVector("x").getObject(1));
        }
      }
    }
  }

  private LanceField findField(List<LanceField> fields, String fieldName) {
    return fields.stream()
        .filter(field -> field.getName().equals(fieldName))
        .findFirst()
        .orElseThrow(
            () -> new IllegalStateException(String.format("field '%s' not found", fieldName)));
  }
}
