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

import org.lance.cleanup.CleanupExplanation;
import org.lance.cleanup.CleanupPolicy;
import org.lance.cleanup.RemovalStats;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.time.Duration;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CleanupTest {
  @Test
  public void testCleanupBeforeVersion(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("test_dataset_for_cleanup").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();

      testDataset.write(1, 10).close();
      testDataset.write(2, 10).close();

      try (Dataset dataset = testDataset.write(3, 10)) {
        RemovalStats stats =
            dataset.cleanupWithPolicy(CleanupPolicy.builder().withBeforeVersion(3L).build());
        assertEquals(2L, stats.getOldVersions());
        assertEquals(0L, stats.getDataFilesRemoved());
        assertEquals(2L, stats.getTransactionFilesRemoved());
        assertEquals(0L, stats.getIndexFilesRemoved());
        assertEquals(0L, stats.getDeletionFilesRemoved());
      }
    }
  }

  @Test
  public void testCleanupSpecificVersions(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("test_dataset_for_cleanup").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();

      testDataset.write(1, 10).close();
      testDataset.write(2, 10).close();

      try (Dataset dataset = testDataset.write(3, 10)) {
        assertEquals(4, dataset.listVersions().size());

        RemovalStats stats =
            dataset.cleanupWithPolicy(CleanupPolicy.builder().withVersions(List.of(2L)).build());

        assertEquals(1L, stats.getOldVersions());
        assertEquals(3, dataset.listVersions().size());
        assertTrue(dataset.listVersions().stream().noneMatch(version -> version.getId() == 2L));
      }
    }
  }

  @Test
  public void testExplainCleanupBeforeVersion(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("test_dataset_for_cleanup").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();

      testDataset.write(1, 10).close();
      testDataset.write(2, 10).close();

      try (Dataset dataset = testDataset.write(3, 10)) {
        CleanupPolicy policy = CleanupPolicy.builder().withBeforeVersion(3L).build();
        CleanupOperation cleanup = dataset.cleanup(policy);
        CleanupExplanation explanation = cleanup.explain();

        assertEquals(2L, explanation.getStats().getOldVersions());
        assertEquals(2L, explanation.getStats().getTransactionFilesRemoved());
        assertTrue(explanation.getStats().getBytesRemoved() > 0);
        assertTrue(explanation.getReadVersion() > 0);
        assertTrue(explanation.getCandidateFiles().size() > 0);
        assertTrue(explanation.getReferencedBranches().isEmpty());

        List<Version> versions = dataset.listVersions();
        assertEquals(4, versions.size());

        RemovalStats stats = cleanup.execute();
        assertEquals(explanation.getStats().getOldVersions(), stats.getOldVersions());
      }
    }
  }

  @Test
  public void testCleanupBeforeTimestamp(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("test_dataset_for_cleanup").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();

      testDataset.write(1, 10).close();

      Thread.sleep(100L);
      long beforeTs = System.currentTimeMillis();

      testDataset.write(2, 10).close();

      try (Dataset dataset = testDataset.write(3, 10)) {
        RemovalStats stats =
            dataset.cleanupWithPolicy(
                CleanupPolicy.builder().withBeforeTimestampMillis(beforeTs).build());
        assertEquals(2L, stats.getOldVersions());
      }
    }
  }

  @Test
  public void testCleanupTaggedVersion(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("test_dataset_for_cleanup").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();

      Dataset ds = testDataset.write(1, 10);
      ds.tags().create("tag-2", 2L);

      testDataset.write(2, 10).close();

      try (Dataset dataset = testDataset.write(3, 10)) {
        // cleanup with tag-2 should throw exception
        Assertions.assertThrows(
            RuntimeException.class,
            () ->
                dataset.cleanupWithPolicy(
                    CleanupPolicy.builder()
                        .withErrorIfTaggedOldVersions(true)
                        .withBeforeVersion(3L)
                        .build()));

        // cleanup with tag-2 should not throw exception when set errorIfTaggedOldVersions to false
        RemovalStats stats =
            dataset.cleanupWithPolicy(
                CleanupPolicy.builder()
                    .withErrorIfTaggedOldVersions(false)
                    .withBeforeVersion(3L)
                    .build());
        assertEquals(1L, stats.getOldVersions());

        // The version with tag-2 should not be cleaned up
        Assertions.assertEquals("tag-2", dataset.tags().list().get(0).getName());
      }
    }
  }

  @Test
  public void testExplainCleanupWithMaxCandidateFiles(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("test_dataset_for_cleanup").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();

      testDataset.write(1, 10).close();
      testDataset.write(2, 10).close();

      try (Dataset dataset = testDataset.write(3, 10)) {
        CleanupPolicy policy = CleanupPolicy.builder().withBeforeVersion(3L).build();
        CleanupExplanation full = dataset.cleanup(policy).explain();
        assertTrue(full.getCandidateFiles().size() > 1);
        assertEquals(1000L, full.getCandidateFileLimit());

        CleanupExplanation truncated = dataset.cleanup(policy).withMaxCandidateFiles(1L).explain();
        assertEquals(1L, truncated.getCandidateFileLimit());
        assertEquals(1, truncated.getCandidateFiles().size());
        assertTrue(truncated.isCandidateFilesTruncated());
        assertTrue(!truncated.getWarnings().isEmpty());
        // Aggregate stats stay accurate even when the per-file list is truncated.
        assertEquals(full.getStats().getOldVersions(), truncated.getStats().getOldVersions());

        Assertions.assertThrows(
            IllegalArgumentException.class,
            () -> dataset.cleanup(policy).withMaxCandidateFiles(0L));
      }
    }
  }

  @Test
  public void testCleanupWithRateLimit(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("test_dataset_for_cleanup").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();
      testDataset.write(1, 100).close();
      testDataset.write(2, 100).close();
      try (Dataset dataset = testDataset.write(3, 100)) {
        List<Version> versions = dataset.listVersions();
        assertEquals(4, versions.size());
        long beforeTimestampMillis =
            versions.get(versions.size() - 1).getDataTime().toInstant().toEpochMilli() + 1;
        long start = System.nanoTime();
        RemovalStats stats =
            dataset.cleanupWithPolicy(
                CleanupPolicy.builder()
                    .withBeforeTimestampMillis(beforeTimestampMillis)
                    .withDeleteRateLimit(1L)
                    .build());
        long elapsed = System.nanoTime() - start;

        assertEquals(3L, stats.getOldVersions());
        assertTrue(stats.getBytesRemoved() > 0);
        assertTrue(elapsed >= Duration.ofSeconds(2).toNanos());
      }
    }
  }

  @Test
  public void testFailedDeletesReachesJavaAcrossJni(@TempDir Path tempDir) throws Exception {
    // Best-effort deletion makes failedDeletes the only programmatic signal that cleanup
    // returned normally without removing everything it identified. Asserting it is zero on
    // a successful run would prove nothing: the six-argument constructor defaults it to
    // zero, so a JNI projection that dropped the field would still pass. The value has to
    // be non-zero, which means a delete has to genuinely fail.
    //
    // Removing a file needs write permission on its parent, so making _transactions
    // read-only fails exactly those deletes while the manifest deletes still succeed.
    String datasetPath = tempDir.resolve("test_dataset_for_failed_deletes").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();
      testDataset.write(2, 10).close();

      Path transactions = Path.of(datasetPath, "_transactions");
      assertTrue(Files.isDirectory(transactions), "expected a _transactions directory");
      Set<PosixFilePermission> original = Files.getPosixFilePermissions(transactions);

      // Every write lands a transaction file, so the directory stays writable until the
      // last version is committed and only then becomes read-only.
      try (Dataset dataset = testDataset.write(3, 10)) {
        Files.setPosixFilePermissions(
            transactions,
            EnumSet.of(PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_EXECUTE));
        try {
          // Root ignores these bits, so confirm deletion is really blocked before relying
          // on it; otherwise this asserts nothing and should skip rather than fail.
          Path probe = transactions.resolve("permission-probe");
          boolean blocked;
          try {
            Files.createFile(probe);
            Files.deleteIfExists(probe);
            blocked = false;
          } catch (IOException expected) {
            blocked = true;
          }
          Assumptions.assumeTrue(blocked, "filesystem permissions do not block deletion here");

          RemovalStats stats =
              dataset.cleanupWithPolicy(CleanupPolicy.builder().withBeforeVersion(3L).build());

          // The call returns normally rather than throwing, and reports what it could not
          // remove. Both halves matter: the old behaviour discarded the whole sweep.
          assertTrue(
              stats.getFailedDeletes() > 0,
              "expected the blocked transaction deletes to be counted, got "
                  + stats.getFailedDeletes());
          assertTrue(stats.getOldVersions() > 0, "the sweep must continue past a failed delete");
        } finally {
          Files.setPosixFilePermissions(transactions, original);
        }
      }
    }
  }

  @Test
  public void testRemovalStatsCarriesFailedDeletes() {
    RemovalStats stats = new RemovalStats(1L, 2L, 3L, 4L, 5L, 6L, 7L);
    assertEquals(7L, stats.getFailedDeletes());

    // The pre-existing six-argument constructor stays source compatible and defaults to
    // zero, so callers that predate best-effort deletion keep compiling.
    assertEquals(0L, new RemovalStats(1L, 2L, 3L, 4L, 5L, 6L).getFailedDeletes());
  }
}
