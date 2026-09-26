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

import org.lance.expire.ExpireVersionsPlan;
import org.lance.expire.ExpireVersionsPolicy;
import org.lance.expire.ExpireVersionsStats;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.time.Duration;
import java.util.EnumSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ExpireVersionsTest {

  @Test
  public void testExpireVersionsRemovesHistoryButNotData(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("expire_basic").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();
      testDataset.write(2, 10).close();

      try (Dataset dataset = testDataset.write(3, 10)) {
        // Counted after the last write: a write adds a data file, and the point of this
        // assertion is that *expiring* does not remove one.
        long dataFilesBefore = countFiles(Path.of(datasetPath, "data"));
        long latest = dataset.version();
        ExpireVersionsStats stats =
            dataset.expireVersions(
                ExpireVersionsPolicy.builder().withBeforeVersion(latest).build());

        assertEquals(latest - 1, stats.getVersionsRemoved());
        assertEquals(0L, stats.getFailedDeletes());
        assertTrue(stats.getBytesRemoved() > 0);

        // Expiring removes history, never data: reclaiming those files is cleanup's job.
        assertEquals(
            dataFilesBefore,
            countFiles(Path.of(datasetPath, "data")),
            "expireVersions must not delete data files");
      }
    }
  }

  @Test
  public void testExpireVersionsWithNoConditionRemovesNothing(@TempDir Path tempDir) {
    // A default policy must be inert rather than a request to drop all history.
    String datasetPath = tempDir.resolve("expire_noop").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();

      try (Dataset dataset = testDataset.write(2, 10)) {
        ExpireVersionsStats stats = dataset.expireVersions(ExpireVersionsPolicy.builder().build());
        assertEquals(0L, stats.getVersionsRemoved());
      }
    }
  }

  @Test
  public void testFailedDeletesReachesJavaAcrossJni(@TempDir Path tempDir) throws Exception {
    // failedDeletes is the only signal that a run returned normally without removing
    // everything it selected. Asserting it is zero on a clean run would prove nothing,
    // so force a real failure: removing a file needs write permission on its parent.
    String datasetPath = tempDir.resolve("expire_failed").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();
      testDataset.write(2, 10).close();

      Path versions = Path.of(datasetPath, "_versions");
      assertTrue(Files.isDirectory(versions), "expected a _versions directory");
      Set<PosixFilePermission> original = Files.getPosixFilePermissions(versions);

      try (Dataset dataset = testDataset.write(3, 10)) {
        Files.setPosixFilePermissions(
            versions,
            EnumSet.of(PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_EXECUTE));
        try {
          Path probe = versions.resolve("permission-probe");
          boolean blocked;
          try {
            Files.createFile(probe);
            Files.deleteIfExists(probe);
            blocked = false;
          } catch (IOException expected) {
            blocked = true;
          }
          Assumptions.assumeTrue(blocked, "filesystem permissions do not block deletion here");

          ExpireVersionsStats stats =
              dataset.expireVersions(
                  ExpireVersionsPolicy.builder().withBeforeVersion(dataset.version()).build());

          assertTrue(
              stats.getFailedDeletes() > 0,
              "expected the blocked manifest deletes to be counted, got "
                  + stats.getFailedDeletes());
          assertEquals(
              0L,
              stats.getVersionsRemoved(),
              "a manifest that could not be deleted is not removed");
        } finally {
          Files.setPosixFilePermissions(versions, original);
        }
      }
    }
  }

  @Test
  public void testExplainExpireVersionsDeletesNothing(@TempDir Path tempDir) {
    // A dry run has to be genuinely dry: the value of it is deciding whether to run the
    // destructive call, so it must not be the destructive call.
    String datasetPath = tempDir.resolve("expire_explain").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();
      testDataset.write(2, 10).close();

      try (Dataset dataset = testDataset.write(3, 10)) {
        long latest = dataset.version();
        long manifestsBefore = countFiles(Path.of(datasetPath, "_versions"));

        ExpireVersionsPlan plan =
            dataset.explainExpireVersions(
                ExpireVersionsPolicy.builder().withBeforeVersion(latest).build());

        assertEquals(latest - 1, plan.getVersions().size());
        assertEquals(latest - 1, plan.getStats().getVersionsRemoved());
        assertTrue(plan.getTaggedButKept().isEmpty());

        // The versions it named are still on disk.
        assertEquals(
            manifestsBefore,
            countFiles(Path.of(datasetPath, "_versions")),
            "explainExpireVersions must not delete anything");
        assertEquals(latest, dataset.version());

        // And running for real afterwards removes exactly what the dry run predicted.
        ExpireVersionsStats stats =
            dataset.expireVersions(
                ExpireVersionsPolicy.builder().withBeforeVersion(latest).build());
        assertEquals(plan.getStats().getVersionsRemoved(), stats.getVersionsRemoved());
      }
    }
  }

  @Test
  public void testKeepOnePerRejectsWidthsBelowOneHour() {
    // Thinning competes with the version hint, whose protective floor is read once and can
    // be lowered afterwards by a slow committer. Keeping survivors at least an hour apart
    // keeps deletions away from that window, so finer widths are refused outright.
    for (Duration tooFine :
        new Duration[] {Duration.ZERO, Duration.ofMinutes(59), Duration.ofSeconds(1)}) {
      IllegalArgumentException e =
          assertThrows(
              IllegalArgumentException.class,
              () -> ExpireVersionsPolicy.builder().withKeepOnePer(tooFine));
      assertTrue(e.getMessage().contains("at least one hour"), e.getMessage());
    }

    assertThrows(
        IllegalArgumentException.class,
        () -> ExpireVersionsPolicy.builder().withKeepOnePer(Duration.ofSeconds(-1)));
    assertThrows(
        IllegalArgumentException.class, () -> ExpireVersionsPolicy.builder().withKeepOnePer(null));

    // Exactly one hour is accepted, and a finer-than-microsecond component is still
    // refused rather than silently truncated.
    ExpireVersionsPolicy.builder().withKeepOnePer(Duration.ofHours(1));
    assertThrows(
        IllegalArgumentException.class,
        () -> ExpireVersionsPolicy.builder().withKeepOnePer(Duration.ofHours(1).plusNanos(1)));
  }

  @Test
  public void testExpireVersionsRejectsNullPolicy(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("expire_null").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 10)) {
        assertThrows(NullPointerException.class, () -> dataset.expireVersions(null));
      }
    }
  }

  private static long countFiles(Path dir) {
    try {
      if (!Files.isDirectory(dir)) {
        return 0;
      }
      try (var stream = Files.list(dir)) {
        return stream.count();
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
