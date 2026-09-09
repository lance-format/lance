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
import org.lance.operation.Delete;
import org.lance.operation.OperationTestBase;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CommitBuilderStorageFormatTest extends OperationTestBase {

  /**
   * Append to a freshly created (2.2) dataset with the given storage format and return the
   * committed dataset's format version.
   */
  private String commitWithStorageFormat(String datasetPath, String storageFormat)
      throws Exception {
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      FragmentMetadata fragment = testDataset.createNewFragment(10);
      try (Transaction txn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(Append.builder().fragments(Collections.singletonList(fragment)).build())
              .build()) {
        try (Dataset committed =
            new CommitBuilder(dataset).storageFormat(storageFormat).execute(txn)) {
          assertEquals(2, committed.version());
          return committed.getLanceFileFormatVersion();
        }
      }
    }
  }

  /**
   * The numeric versions are what {@link Dataset#getLanceFileFormatVersion()} returns and what
   * {@link WriteParams.Builder#withDataStorageVersion(String)} accepts, so they must work here too
   * — a caller that encodes fragments as "2.2" has to be able to commit them as "2.2".
   */
  @Test
  void testCanonicalVersionAccepted(@TempDir Path tempDir) throws Exception {
    assertEquals(
        LanceConstants.FILE_FORMAT_VERSION_2_2,
        commitWithStorageFormat(
            tempDir.resolve("canonical").toString(), LanceConstants.FILE_FORMAT_VERSION_2_2));
  }

  /** The "v"-prefixed spelling shipped in this method's Javadoc and stays accepted. */
  @Test
  void testDeprecatedAliasAccepted(@TempDir Path tempDir) throws Exception {
    assertEquals(
        LanceConstants.FILE_FORMAT_VERSION_2_2,
        commitWithStorageFormat(tempDir.resolve("alias").toString(), "v2_2"));
  }

  /**
   * A delete adds no data files, so nothing about it depends on the storage format — but {@link
   * CommitBuilder#storageFormat(String)} is still validated against the existing dataset for any
   * operation other than overwrite. A caller that forwards a configured format on every commit hits
   * this on row-level operations against a table written in a different version, so the failure is
   * a mismatch error rather than anything to do with parsing.
   */
  @Test
  void testMismatchedFormatRejectedOnRowLevelOperation(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("mismatch").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      FragmentMetadata fragment = testDataset.createNewFragment(10);
      try (Transaction appendTxn =
          new Transaction.Builder()
              .readVersion(dataset.version())
              .operation(Append.builder().fragments(Collections.singletonList(fragment)).build())
              .build()) {
        dataset = new CommitBuilder(dataset).execute(appendTxn);
      }
      assertEquals(LanceConstants.FILE_FORMAT_VERSION_2_2, dataset.getLanceFileFormatVersion());

      List<Long> fragmentIds =
          dataset.getFragments().stream()
              .map(f -> Long.valueOf(f.getId()))
              .collect(Collectors.toList());

      // "2.1" parses fine, so a failure here is the mismatch guard and not the parser.
      try (Transaction deleteTxn = deleteAll(fragmentIds)) {
        IllegalArgumentException error =
            assertThrows(
                IllegalArgumentException.class,
                () ->
                    new CommitBuilder(dataset)
                        .storageFormat(LanceConstants.FILE_FORMAT_VERSION_2_1)
                        .execute(deleteTxn));
        assertTrue(error.getMessage().contains("Storage format mismatch"), error.getMessage());
      }

      // The same delete succeeds when the format agrees with the dataset.
      try (Transaction deleteTxn = deleteAll(fragmentIds)) {
        try (Dataset deleted =
            new CommitBuilder(dataset)
                .storageFormat(LanceConstants.FILE_FORMAT_VERSION_2_2)
                .execute(deleteTxn)) {
          assertEquals(0, deleted.countRows());
        }
      }
    }
  }

  private Transaction deleteAll(List<Long> fragmentIds) {
    return new Transaction.Builder()
        .readVersion(dataset.version())
        .operation(Delete.builder().deletedFragmentIds(fragmentIds).predicate("1=1").build())
        .build();
  }

  @Test
  void testUnknownFormatRejected(@TempDir Path tempDir) {
    assertThrows(
        IllegalArgumentException.class,
        () -> commitWithStorageFormat(tempDir.resolve("bogus").toString(), "bogus"));
    // The alias set is frozen at what shipped, so it does not extend to newer versions.
    assertThrows(
        IllegalArgumentException.class,
        () -> commitWithStorageFormat(tempDir.resolve("v23").toString(), "v2_3"));
  }
}
