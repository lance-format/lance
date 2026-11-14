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

import org.lance.operation.Append;
import org.lance.operation.OperationTestBase;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CommitBuilderExpectedHeadTest extends OperationTestBase {

  @Test
  void testExpectedHeadCommitsWhenUnchanged(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("expectedHead").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      long expected = dataset.version();
      FragmentMetadata fragment = testDataset.createNewFragment(10);
      try (Transaction txn =
          new Transaction.Builder()
              .readVersion(expected)
              .operation(Append.builder().fragments(Collections.singletonList(fragment)).build())
              .build()) {
        try (Dataset committed =
            new CommitBuilder(dataset)
                .expectedLatestVersion(expected)
                .execute(txn)) {
          assertEquals(expected + 1, committed.version());
        }
      }
    }
  }

  @Test
  void testExpectedHeadRejectsChangedHead(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("expectedHeadStale").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      long expected = dataset.version();

      FragmentMetadata fragment = testDataset.createNewFragment(10);
      try (Transaction txn =
          new Transaction.Builder()
              .readVersion(expected)
              .operation(Append.builder().fragments(Collections.singletonList(fragment)).build())
              .build()) {
        try (Dataset advanced = new CommitBuilder(dataset).execute(txn)) {
          assertEquals(expected + 1, advanced.version());
        }
      }

      FragmentMetadata staleFragment = testDataset.createNewFragment(10);
      try (Transaction staleTxn =
          new Transaction.Builder()
              .readVersion(expected)
              .operation(
                  Append.builder()
                      .fragments(Collections.singletonList(staleFragment))
                      .build())
              .build()) {
        RuntimeException error =
            assertThrows(
                RuntimeException.class,
                () ->
                    new CommitBuilder(dataset)
                        .expectedLatestVersion(expected)
                        .execute(staleTxn)
                        .close());
        assertTrue(
            error.getMessage().contains("Retryable commit conflict"),
            "Expected retryable commit conflict error, got: " + error.getMessage());
      }
    }
  }
}
