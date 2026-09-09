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

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
}
