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

import org.lance.BlobReferences.Batch;
import org.lance.BlobReferences.Capture;
import org.lance.BlobReferences.NestedBatch;
import org.lance.BlobReferences.ReadStream;
import org.lance.BlobReferences.Reader;
import org.lance.BlobReferences.Request;
import org.lance.BlobReferences.SourceContext;
import org.lance.cleanup.CleanupPolicy;
import org.lance.compaction.CompactionOptions;
import org.lance.file.FileWriteOptions;
import org.lance.index.Index;
import org.lance.index.IndexCriteria;
import org.lance.index.IndexDescription;
import org.lance.index.IndexOptions;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.OptimizeOptions;
import org.lance.index.scalar.BTreeIndexParams;
import org.lance.index.scalar.NGramIndexParams;
import org.lance.index.scalar.ScalarIndexParams;
import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;
import org.lance.operation.Append;
import org.lance.operation.Overwrite;
import org.lance.operation.UpdateConfig;
import org.lance.operation.UpdateMap;
import org.lance.schema.ColumnAlteration;
import org.lance.schema.LanceField;
import org.lance.schema.SqlExpressions;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.LargeVarBinaryVector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadInfo;
import java.lang.management.ThreadMXBean;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.ClosedChannelException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Clock;
import java.time.Instant;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class DatasetTest {
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
  void testWriteStreamAndOpenPath(@TempDir Path tempDir) throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("write_stream").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.RandomAccessDataset testDataset =
          new TestUtils.RandomAccessDataset(allocator, datasetPath);
      testDataset.createDatasetAndValidate();
      testDataset.openDatasetAndValidate();
    }
  }

  @Test
  void testCreateEmptyDataset(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("new_empty_dataset").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
    }
  }

  @Test
  void testWriteRejectsNegativeDataCacheBytes(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("negative_data_cache_bytes").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      WriteParams params =
          new WriteParams.Builder()
              .withFileWriteOptions(FileWriteOptions.builder().dataCacheBytes(-1).build())
              .build();

      assertThrows(
          IllegalArgumentException.class, () -> testDataset.createDatasetWithWriteParams(params));
    }
  }

  @Test
  void testWriteDatasetBuilderRejectsNullFileWriteOptions() {
    NullPointerException error =
        assertThrows(
            NullPointerException.class, () -> new WriteDatasetBuilder().fileWriteOptions(null));

    assertEquals("fileWriteOptions must not be null", error.getMessage());
  }

  @Test
  void testGetLanceFileFormatVersion(@TempDir Path tempDir) {
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      // Test default version (V2_2)
      String defaultPath = tempDir.resolve("default_version").toString();
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, defaultPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(LanceConstants.FILE_FORMAT_VERSION_2_2, dataset.getLanceFileFormatVersion());
        WriterVersion writerVersion = dataset.getWriterVersion().orElseThrow(AssertionError::new);
        assertEquals("lance", writerVersion.getLibrary());
        assertFalse(writerVersion.getVersion().isEmpty());
      }

      // Test LEGACY version
      String legacyPath = tempDir.resolve("legacy_version").toString();
      try (Dataset legacyDataset =
          Dataset.write()
              .allocator(allocator)
              .uri(legacyPath)
              .schema(testDataset.getSchema())
              .mode(WriteParams.WriteMode.CREATE)
              .dataStorageVersion(LanceConstants.FILE_FORMAT_VERSION_LEGACY)
              .execute()) {
        assertEquals(
            LanceConstants.FILE_FORMAT_VERSION_0_1, legacyDataset.getLanceFileFormatVersion());
      }

      // This dataset was written before writer_version was added to the manifest.
      String historicalPath =
          Path.of("..", "test_data", "v0.7.5", "with_deletions")
              .toAbsolutePath()
              .normalize()
              .toString();
      try (Dataset historicalDataset = Dataset.open(historicalPath, allocator)) {
        assertTrue(historicalDataset.getWriterVersion().isEmpty());
      }

      // This fixture was written by lance 2.0.0-beta.1. Reading it through Dataset verifies
      // that the manifest's prerelease qualifier survives the Rust-to-Java JNI mapping.
      String prereleasePath =
          Path.of("..", "test_data", "pre_file_sizes", "index_without_file_sizes")
              .toAbsolutePath()
              .normalize()
              .toString();
      try (Dataset prereleaseDataset = Dataset.open(prereleasePath, allocator)) {
        WriterVersion writerVersion =
            prereleaseDataset.getWriterVersion().orElseThrow(AssertionError::new);
        assertEquals("lance", writerVersion.getLibrary());
        assertEquals("2.0.0", writerVersion.getVersion());
        assertEquals("beta.1", writerVersion.getPrerelease().orElseThrow(AssertionError::new));
        assertTrue(writerVersion.getBuildMetadata().isEmpty());
      }
    }
  }

  @Test
  void testWriterVersionPreservesOpaqueAndOptionalFields() {
    WriterVersion writerVersion =
        new WriterVersion("custom-writer", "release-2026", "preview.1", "build.42");

    assertEquals("custom-writer", writerVersion.getLibrary());
    assertEquals("release-2026", writerVersion.getVersion());
    assertEquals("preview.1", writerVersion.getPrerelease().orElseThrow(AssertionError::new));
    assertEquals("build.42", writerVersion.getBuildMetadata().orElseThrow(AssertionError::new));
  }

  @Test
  void testListManifestLocations(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("manifest_locations").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 1).close();

      List<ManifestLocation> manifests = Dataset.listManifestLocations(datasetPath);
      assertEquals(2, manifests.size());
      assertEquals(
          Set.of(1L, 2L),
          manifests.stream().map(ManifestLocation::getVersion).collect(Collectors.toSet()));
      for (ManifestLocation manifest : manifests) {
        assertTrue(manifest.getPath().contains("manifest_locations/_versions/"));
        assertFalse(manifest.getPath().startsWith("_versions/"));
        assertTrue(manifest.getPath().endsWith(".manifest"));
        assertTrue(manifest.getSizeBytes() > 0);
        assertNotNull(manifest.getNamingScheme());
      }
    }
  }

  @Test
  void testCreateDirNotExist(@TempDir Path tempDir) throws IOException, URISyntaxException {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
    }
  }

  @Test
  void testOpenInvalidPath(@TempDir Path tempDir) {
    String validPath = tempDir.resolve("Invalid_dataset").toString();
    assertThrows(
        RuntimeException.class,
        () -> {
          dataset = Dataset.open(validPath, new RootAllocator(Long.MAX_VALUE));
        });
  }

  @Test
  void testDatasetVersion(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("dataset_version").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      ZonedDateTime before = ZonedDateTime.now();
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        ZonedDateTime time1 = dataset.getVersion().getDataTime();
        assertEquals(1, dataset.version());
        assertEquals(dataset.getVersion().getId(), dataset.version());
        assertTrue(time1.isEqual(before) || time1.isAfter(before));
        assertTrue(time1.isEqual(ZonedDateTime.now()) || time1.isBefore(ZonedDateTime.now()));
        assertEquals(time1.getZone(), Clock.systemUTC().getZone());
        assertEquals(1, dataset.latestVersion());

        // Write first batch of data
        try (Dataset dataset2 = testDataset.write(1, 5)) {
          ZonedDateTime time2 = dataset2.getVersion().getDataTime();
          assertEquals(1, dataset.version());
          assertEquals(dataset.getVersion().getId(), dataset.version());
          assertEquals(2, dataset.latestVersion());
          assertEquals(2, dataset2.version());
          assertEquals(dataset2.getVersion().getId(), dataset2.version());
          assertEquals(2, dataset2.latestVersion());
          assertTrue(time2.isEqual(before) || time2.isAfter(before));
          assertTrue(time2.isEqual(time1) || time2.isAfter(time1));
          assertTrue(time1.isEqual(dataset.getVersion().getDataTime()));

          // Open dataset with version 1
          ReadOptions options1 = new ReadOptions.Builder().setVersion(1).build();
          try (Dataset datasetV1 = Dataset.open(allocator, datasetPath, options1)) {
            assertEquals(1, datasetV1.version());
            assertEquals(datasetV1.getVersion().getId(), datasetV1.version());
            assertTrue(time1.isEqual(dataset.getVersion().getDataTime()));
            assertEquals(2, datasetV1.latestVersion());
          }

          // Write second batch of data
          try (Dataset dataset3 = testDataset.write(2, 3)) {
            ZonedDateTime time3 = dataset3.getVersion().getDataTime();
            assertEquals(1, dataset.version());
            assertEquals(dataset.getVersion().getId(), dataset.version());
            assertTrue(time1.isEqual(dataset.getVersion().getDataTime()));
            assertEquals(3, dataset.latestVersion());
            assertEquals(2, dataset2.version());
            assertEquals(dataset2.getVersion().getId(), dataset2.version());
            assertTrue(time2.isEqual(dataset2.getVersion().getDataTime()));
            assertEquals(3, dataset2.latestVersion());
            assertTrue(time3.isEqual(before) || time3.isAfter(before));
            assertEquals(3, dataset3.version());
            assertEquals(dataset3.getVersion().getId(), dataset3.version());
            assertEquals(3, dataset3.latestVersion());

            // Open dataset with version 2
            ReadOptions options2 = new ReadOptions.Builder().setVersion(2).build();
            try (Dataset datasetV2 = Dataset.open(allocator, datasetPath, options2)) {
              assertEquals(2, datasetV2.version());
              assertEquals(datasetV2.getVersion().getId(), datasetV2.version());
              assertTrue(time2.isEqual(datasetV2.getVersion().getDataTime()));
              assertEquals(3, datasetV2.latestVersion());
            }

            // Open dataset with latest version (3)
            try (Dataset datasetLatest = Dataset.open(datasetPath, allocator)) {
              assertEquals(3, datasetLatest.version());
              assertEquals(datasetLatest.getVersion().getId(), datasetLatest.version());
              assertTrue(time3.isEqual(datasetLatest.getVersion().getDataTime()));
              assertEquals(3, datasetLatest.latestVersion());
            }

            List<Version> versions = dataset.listVersions();
            assertEquals(3, versions.size());
            assertEquals(3, dataset.getVersionCount());
            assertEquals(3, dataset2.getVersionCount());
            assertEquals(3, dataset3.getVersionCount());
            assertEquals(1, versions.get(0).getId());
            assertEquals(2, versions.get(1).getId());
            assertEquals(3, versions.get(2).getId());
            assertTrue(time1.isEqual(versions.get(0).getDataTime()));
            assertTrue(time2.isEqual(versions.get(1).getDataTime()));
            assertTrue(time3.isEqual(versions.get(2).getDataTime()));
            assertArrayEquals(versions.toArray(), dataset2.listVersions().toArray());
            assertArrayEquals(versions.toArray(), dataset3.listVersions().toArray());
            dataset.checkoutLatest();
            assertEquals(3, dataset.version());
            assertEquals(dataset.getVersion().getId(), dataset.version());
            assertTrue(time3.isEqual(dataset.getVersion().getDataTime()));
            assertEquals(3, dataset.latestVersion());

            List<ManifestSummary> summaries =
                versions.stream().map(Version::getManifestSummary).collect(Collectors.toList());
            assertEquals(0, summaries.get(0).getTotalFragments());
            assertEquals(0, summaries.get(0).getTotalDataFiles());
            assertEquals(0, summaries.get(0).getTotalDataFileRows());
            assertEquals(0, summaries.get(0).getTotalRows());
            assertEquals(1, summaries.get(1).getTotalFragments());
            assertEquals(1, summaries.get(1).getTotalDataFiles());
            assertEquals(5, summaries.get(1).getTotalDataFileRows());
            assertEquals(5, summaries.get(1).getTotalRows());
            assertEquals(2, summaries.get(2).getTotalFragments());
            assertEquals(2, summaries.get(2).getTotalDataFiles());
            assertEquals(8, summaries.get(2).getTotalDataFileRows());
            assertEquals(8, summaries.get(2).getTotalRows());
          }
        }
      }
    }
  }

  @Test
  void testDatasetCheckoutVersion(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("dataset_checkout_version").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      // version 1, empty dataset
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
        assertEquals(1, dataset.latestVersion());
        assertEquals(0, dataset.countRows());
      }

      // write first batch of data, version 2
      try (Dataset dataset2 = testDataset.write(1, 5)) {
        assertEquals(2, dataset2.version());
        assertEquals(2, dataset2.latestVersion());
        assertEquals(5, dataset2.countRows());

        // checkout the dataset at version 1
        try (Dataset checkoutV1 = dataset2.checkoutVersion(1)) {
          assertNotNull(checkoutV1.getSchema());
          assertEquals(1, checkoutV1.version());
          assertEquals(2, checkoutV1.latestVersion());
          assertEquals(0, checkoutV1.countRows());
        }
      }
    }
  }

  @Test
  void testTags(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("dataset_tags").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      // version 1, empty dataset
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
        dataset.tags().create("tag1", Ref.ofMain());
        dataset.tags().replaceMetadata("tag1", Map.of("description", "primary tag"));
        List<Tag> tags = dataset.tags().list();
        Tag tag1 = tags.get(0);
        assertEquals(1, tags.size());
        assertEquals(1, tag1.getVersion());
        assertEquals(Map.of("description", "primary tag"), tag1.getMetadata());
        assertTrue(tag1.getCreatedAt().isPresent());
        assertTrue(tag1.getUpdatedAt().isPresent());
        assertEquals(tag1.getCreatedAt(), tag1.getUpdatedAt());
        assertEquals(1, dataset.tags().getVersion("tag1"));
      }

      // write first batch of data, version 2
      try (Dataset dataset2 = testDataset.write(1, 5)) {
        assertEquals(2, dataset2.version());
        assertEquals(1, dataset2.tags().list().size());
        assertEquals(1, dataset2.tags().list().get(0).getVersion());
        assertEquals(1, dataset2.tags().getVersion("tag1"));
        dataset2.tags().create("tag2", Ref.ofMain(2));
        assertEquals(2, dataset2.tags().list().size());
        assertEquals(1, dataset2.tags().getVersion("tag1"));
        assertEquals(2, dataset2.tags().getVersion("tag2"));
        dataset2.tags().replaceMetadata("tag2", Map.of("description", "rollback tag"));
        Instant tag2CreatedAt =
            dataset2.tags().list().stream()
                .filter(tag -> tag.getName().equals("tag2"))
                .findFirst()
                .orElseThrow()
                .getCreatedAt()
                .orElseThrow();
        Instant tag2UpdatedAt =
            dataset2.tags().list().stream()
                .filter(tag -> tag.getName().equals("tag2"))
                .findFirst()
                .orElseThrow()
                .getUpdatedAt()
                .orElseThrow();
        assertEquals(tag2CreatedAt, tag2UpdatedAt);
        assertEquals(2, dataset2.tags().list().size());
        assertEquals(1, dataset2.tags().list().get(0).getVersion());
        assertEquals(2, dataset2.tags().list().get(1).getVersion());
        assertEquals(1, dataset2.tags().getVersion("tag1"));
        assertEquals(2, dataset2.tags().getVersion("tag2"));
        assertEquals(
            Map.of("description", "rollback tag"),
            dataset2.tags().list().stream()
                .filter(tag -> tag.getName().equals("tag2"))
                .findFirst()
                .orElseThrow()
                .getMetadata());
        dataset2.tags().update("tag2", Ref.ofMain(1));
        Instant updatedTag2CreatedAt =
            dataset2.tags().list().stream()
                .filter(tag -> tag.getName().equals("tag2"))
                .findFirst()
                .orElseThrow()
                .getCreatedAt()
                .orElseThrow();
        Instant updatedTag2UpdatedAt =
            dataset2.tags().list().stream()
                .filter(tag -> tag.getName().equals("tag2"))
                .findFirst()
                .orElseThrow()
                .getUpdatedAt()
                .orElseThrow();
        assertEquals(1, dataset2.tags().getVersion("tag2"));
        assertEquals(updatedTag2CreatedAt, tag2CreatedAt);
        assertFalse(updatedTag2UpdatedAt.isBefore(tag2UpdatedAt));
        assertEquals(
            Map.of("description", "rollback tag"),
            dataset2.tags().list().stream()
                .filter(tag -> tag.getName().equals("tag2"))
                .findFirst()
                .orElseThrow()
                .getMetadata());
        dataset2.tags().replaceMetadata("tag2", Collections.emptyMap());
        assertEquals(
            Collections.emptyMap(),
            dataset2.tags().list().stream()
                .filter(tag -> tag.getName().equals("tag2"))
                .findFirst()
                .orElseThrow()
                .getMetadata());
        dataset2.tags().delete("tag2");
        assertEquals(1, dataset2.tags().list().size());
        assertEquals(1, dataset2.tags().list().get(0).getVersion());
        assertEquals(1, dataset2.tags().getVersion("tag1"));
        assertThrows(RuntimeException.class, () -> dataset2.tags().getVersion("tag2"));

        // checkout the dataset at version 1
        try (Dataset checkoutV1 = dataset2.checkoutTag("tag1")) {
          assertNotNull(checkoutV1.getSchema());
          assertEquals(1, checkoutV1.version());
          assertEquals(2, checkoutV1.latestVersion());
          assertEquals(0, checkoutV1.countRows());
          assertEquals(1, checkoutV1.tags().list().size());
          assertEquals(1, checkoutV1.tags().list().get(0).getVersion());
          assertEquals(1, checkoutV1.tags().getVersion("tag1"));
        }

        try (Dataset branch = dataset2.createBranch("branch", Ref.ofMain(2))) {
          branch.tags().create("tag_on_branch", Ref.ofBranch("branch"));
          branch.tags().replaceMetadata("tag_on_branch", Map.of("description", "branch tag"));
          assertEquals(2, dataset2.tags().getVersion("tag_on_branch"));
          List<Tag> tags = dataset2.tags().list();
          Optional<Tag> tagOptional =
              dataset2.tags().list().stream()
                  .filter(t -> t.getName().equals("tag_on_branch"))
                  .findFirst();
          assertEquals(2, tags.size());
          assertTrue(tagOptional.isPresent());
          assertEquals(2, tagOptional.get().getVersion());
          assertEquals(Optional.of("branch"), tagOptional.get().getBranch());
          assertEquals(Map.of("description", "branch tag"), tagOptional.get().getMetadata());

          dataset2.tags().update("tag1", Ref.ofBranch("branch"));
          tags = dataset2.tags().list();
          tagOptional =
              dataset2.tags().list().stream()
                  .filter(t -> t.getName().equals("tag_on_branch"))
                  .findFirst();
          assertEquals(2, tags.size());
          assertTrue(tagOptional.isPresent());
          assertEquals(2, tagOptional.get().getVersion());
          assertEquals(Optional.of("branch"), tagOptional.get().getBranch());
        }

        assertEquals(2, dataset2.tags().list().size());
        dataset2.tags().delete("tag_on_branch");
        assertEquals(1, dataset2.tags().list().size());
      }
    }
  }

  @Test
  void testDatasetRestore(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("dataset_restore").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      // version 1, empty dataset
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
        assertEquals(1, dataset.latestVersion());
        assertEquals(0, dataset.countRows());
      }
      // write first batch of data, version 2
      try (Dataset dataset2 = testDataset.write(1, 5)) {
        assertEquals(2, dataset2.version());
        assertEquals(2, dataset2.latestVersion());
        assertEquals(5, dataset2.countRows());
        dataset2.tags().create("tag1", 2);
        try (Dataset dataset3 = dataset2.checkoutVersion(1)) {
          assertEquals(1, dataset3.version());
          assertEquals(2, dataset3.latestVersion());
          assertEquals(0, dataset3.countRows());
          dataset3.restore();
          assertEquals(3, dataset3.version());
          assertEquals(3, dataset3.latestVersion());
          assertEquals(0, dataset3.countRows());

          try (Dataset dataset4 = dataset3.checkoutTag("tag1")) {
            assertEquals(2, dataset4.version());
            assertEquals(3, dataset4.latestVersion());
            assertEquals(5, dataset4.countRows());
            dataset4.restore();
            assertEquals(4, dataset4.version());
            assertEquals(4, dataset4.latestVersion());
            assertEquals(5, dataset4.countRows());
          }
        }
      }
    }
  }

  @Test
  void testDatasetUri(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("dataset_uri").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(datasetPath, dataset.uri());
      }
    }
  }

  @Test
  void testOpenNonExist(@TempDir Path tempDir) throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("non_exist").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      assertThrows(
          IllegalArgumentException.class,
          () -> {
            Dataset.open(datasetPath, allocator);
          });
    }
  }

  @Test
  void testOpenSerializedManifest(@TempDir Path tempDir) throws IOException {
    Path datasetPath = tempDir.resolve("serialized_manifest");
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath.toString());

      try (Dataset dataset1 = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset1.version());
        Path manifestPath = datasetPath.resolve("_versions");
        try (Stream<Path> fileStream = Files.list(manifestPath)) {
          // Ignore the version hint file, which is not a manifest.
          assertEquals(
              1,
              fileStream
                  .filter(p -> !p.getFileName().toString().startsWith("latest_version_hint"))
                  .count());
          ByteBuffer manifestBuffer = readManifest(manifestPath.resolve("1.manifest"));
          try (Dataset dataset2 = testDataset.write(1, 5)) {
            assertEquals(2, dataset2.version());
            assertEquals(2, dataset2.latestVersion());
            // When reading from the serialized manifest, it shouldn't know about the second dataset
            ReadOptions readOptions =
                new ReadOptions.Builder().setSerializedManifest(manifestBuffer).build();
            Dataset dataset1Manifest = Dataset.open(allocator, datasetPath.toString(), readOptions);
            assertEquals(1, dataset1Manifest.version());
          }
        }
      }
    }
  }

  @Test
  void testCreateExist(@TempDir Path tempDir) throws IOException, URISyntaxException {
    String datasetPath = tempDir.resolve("create_exist").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      assertThrows(
          IllegalArgumentException.class,
          () -> {
            testDataset.createEmptyDataset().close();
          });
    }
  }

  @Test
  void testGetSchemaWithClosedDataset(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      Dataset dataset = testDataset.createEmptyDataset();
      dataset.close();
      assertThrows(RuntimeException.class, dataset::getSchema);
    }
  }

  @Test
  void testDropColumns(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      assertEquals(testDataset.getSchema(), dataset.getSchema());
      dataset.dropColumns(Collections.singletonList("name"));

      Schema changedSchema =
          new Schema(
              Collections.singletonList(Field.nullable("id", new ArrowType.Int(32, true))), null);

      assertEquals(changedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          changedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));
    }
  }

  @Test
  void testAlterColumns(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      assertEquals(testDataset.getSchema(), dataset.getSchema());

      ColumnAlteration nameColumnAlteration =
          new ColumnAlteration.Builder("name")
              .rename("new_name")
              .nullable(true)
              .castTo(new ArrowType.Utf8())
              .build();

      dataset.alterColumns(Collections.singletonList(nameColumnAlteration));

      Schema changedSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.notNullable("new_name", new ArrowType.Utf8())),
              null);

      assertEquals(changedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          changedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));

      nameColumnAlteration =
          new ColumnAlteration.Builder("new_name")
              .rename("new_name_2")
              .castTo(new ArrowType.LargeUtf8())
              .build();

      dataset.alterColumns(Collections.singletonList(nameColumnAlteration));
      changedSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.notNullable("new_name_2", new ArrowType.LargeUtf8())),
              null);

      assertEquals(changedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          changedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));

      nameColumnAlteration = new ColumnAlteration.Builder("new_name_2").build();
      dataset.alterColumns(Collections.singletonList(nameColumnAlteration));
      assertNotNull(dataset.getSchema().findField("new_name_2"));
    }
  }

  @Test
  void testAlterColumnsCastType(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Widen "id" from Int32 to Int64. The cast target type is a parameterized ArrowType, which
      // must survive the trip to the native side; regression test for a dropped cast that left the
      // stored type unchanged.
      ColumnAlteration widenId =
          new ColumnAlteration.Builder("id").castTo(new ArrowType.Int(64, true)).build();
      dataset.alterColumns(Collections.singletonList(widenId));

      assertEquals(new ArrowType.Int(64, true), dataset.getSchema().findField("id").getType());

      // A cast combined with rename must apply both.
      ColumnAlteration renameAndWiden =
          new ColumnAlteration.Builder("id")
              .rename("id_long")
              .castTo(new ArrowType.Int(64, true))
              .build();
      dataset.alterColumns(Collections.singletonList(renameAndWiden));

      List<String> fieldNames =
          dataset.getSchema().getFields().stream().map(Field::getName).collect(Collectors.toList());
      assertFalse(fieldNames.contains("id"));
      assertTrue(fieldNames.contains("id_long"));
      assertEquals(new ArrowType.Int(64, true), dataset.getSchema().findField("id_long").getType());
    }
  }

  @Test
  void testAddColumnBySqlExpressions(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      SqlExpressions sqlExpressions =
          new SqlExpressions.Builder().withExpression("double_id", "id * 2").build();
      dataset.addColumns(sqlExpressions, Optional.empty());

      Schema changedSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.nullable("name", new ArrowType.Utf8()),
                  Field.nullable("double_id", new ArrowType.Int(32, true))),
              null);

      assertEquals(changedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          changedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));

      sqlExpressions = new SqlExpressions.Builder().withExpression("triple_id", "id * 3").build();
      dataset.addColumns(sqlExpressions, Optional.empty());
      changedSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.nullable("name", new ArrowType.Utf8()),
                  Field.nullable("double_id", new ArrowType.Int(32, true)),
                  Field.nullable("triple_id", new ArrowType.Int(32, true))),
              null);
      assertEquals(changedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          changedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));
    }
  }

  @Test
  void testAddColumnsByStream(@TempDir Path tempDir) throws IOException {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      try (Dataset initialDataset = testDataset.createEmptyDataset()) {
        try (Dataset datasetV1 = testDataset.write(1, 3)) {
          assertEquals(3, datasetV1.countRows());
        }
      }

      dataset = Dataset.open(datasetPath, allocator);

      Schema newColumnSchema =
          new Schema(
              Collections.singletonList(Field.nullable("age", new ArrowType.Int(32, true))), null);

      try (VectorSchemaRoot vector = VectorSchemaRoot.create(newColumnSchema, allocator);
          ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {

        IntVector ageVector = (IntVector) vector.getVector("age");
        ageVector.allocateNew(3);
        ageVector.set(0, 25);
        ageVector.set(1, 30);
        ageVector.set(2, 35);
        vector.setRowCount(3);

        class SimpleVectorReader extends ArrowReader {
          private boolean batchLoaded = false;

          protected SimpleVectorReader(BufferAllocator allocator) {
            super(allocator);
          }

          @Override
          public boolean loadNextBatch() {
            if (!batchLoaded) {
              batchLoaded = true;
              return true;
            }
            return false;
          }

          @Override
          public VectorSchemaRoot getVectorSchemaRoot() {
            return vector;
          }

          @Override
          public long bytesRead() {
            return vector.getFieldVectors().stream().mapToLong(FieldVector::getBufferSize).sum();
          }

          @Override
          protected void closeReadSource() {}

          @Override
          protected Schema readSchema() {
            return newColumnSchema;
          }
        }

        try (ArrowReader reader = new SimpleVectorReader(allocator)) {
          Data.exportArrayStream(allocator, reader, stream);

          dataset.addColumns(stream, Optional.of(3L));

          Schema expectedSchema =
              new Schema(
                  Arrays.asList(
                      Field.nullable("id", new ArrowType.Int(32, true)),
                      Field.nullable("name", new ArrowType.Utf8()),
                      Field.nullable("age", new ArrowType.Int(32, true))),
                  null);
          Schema actualSchema = dataset.getSchema();
          assertEquals(expectedSchema.getFields(), actualSchema.getFields());

          try (LanceScanner scanner = dataset.newScan()) {
            try (ArrowReader resultReader = scanner.scanBatches()) {
              assertTrue(resultReader.loadNextBatch());
              VectorSchemaRoot root = resultReader.getVectorSchemaRoot();
              assertEquals(3, root.getRowCount());

              IntVector idVector = (IntVector) root.getVector("id");
              IntVector ageVectorResult = (IntVector) root.getVector("age");
              for (int i = 0; i < 3; i++) {
                assertEquals(i, idVector.get(i));
                assertEquals(25 + i * 5, ageVectorResult.get(i));
              }
            }
          }
        }
      }
    } catch (Exception e) {
      fail("Exception occurred during test: " + e.getMessage(), e);
    }
  }

  @Test
  void testAddColumnByFieldsOrSchema(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      Field newColumnField = Field.nullable("age", new ArrowType.Int(32, true));
      dataset.addColumns(Collections.singletonList(newColumnField));
      Schema expectedSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.nullable("name", new ArrowType.Utf8()),
                  Field.nullable("age", new ArrowType.Int(32, true))));
      assertEquals(expectedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          expectedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));

      Field complexField =
          new Field(
              "extra",
              FieldType.nullable(new ArrowType.Struct()),
              Arrays.asList(
                  Field.nullable("tag1", new ArrowType.Int(64, true)),
                  Field.nullable("tag2", new ArrowType.Utf8())));

      Schema addedColumns =
          new Schema(
              Arrays.asList(
                  Field.nullable("height", new ArrowType.Int(64, true)),
                  Field.nullable("desc", new ArrowType.Utf8()),
                  complexField));
      dataset.addColumns(addedColumns);

      expectedSchema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.nullable("name", new ArrowType.Utf8()),
                  Field.nullable("age", new ArrowType.Int(32, true)),
                  Field.nullable("height", new ArrowType.Int(64, true)),
                  Field.nullable("desc", new ArrowType.Utf8()),
                  complexField),
              null);

      assertEquals(expectedSchema.getFields().size(), dataset.getSchema().getFields().size());
      assertEquals(
          expectedSchema.getFields().stream().map(Field::getName).collect(Collectors.toList()),
          dataset.getSchema().getFields().stream()
              .map(Field::getName)
              .collect(Collectors.toList()));
    }
  }

  @Test
  void testDropPath(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      Dataset.drop(datasetPath, new HashMap<>());
    }
  }

  @Test
  void testDropRejectsNonDatasetPath(@TempDir Path tempDir) {
    Path warehouse = tempDir.resolve("warehouse");
    Path tablePath = warehouse.resolve("table.lance");
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, tablePath.toString());
      dataset = testDataset.createEmptyDataset();

      // Pointing at the parent of a dataset must not wipe out the whole warehouse.
      assertThrows(
          IllegalArgumentException.class,
          () -> Dataset.drop(warehouse.toString(), new HashMap<>()));
      assertTrue(Files.exists(tablePath));
    }
  }

  @Test
  void testTake(@TempDir Path tempDir) throws IOException, ClosedChannelException {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      try (Dataset dataset2 = testDataset.write(1, 5)) {
        List<Long> indices = Arrays.asList(1L, 4L);
        List<String> columns = Arrays.asList("id", "name");
        try (ArrowReader reader = dataset2.take(indices, columns)) {
          while (reader.loadNextBatch()) {
            VectorSchemaRoot result = reader.getVectorSchemaRoot();
            assertNotNull(result);
            assertEquals(indices.size(), result.getRowCount());

            for (int i = 0; i < indices.size(); i++) {
              assertEquals(indices.get(i).intValue(), result.getVector("id").getObject(i));
              assertNotNull(result.getVector("name").getObject(i));
            }
          }
        }
      }
    }
  }

  @Test
  void testTakeRows(@TempDir Path tempDir) throws IOException, ClosedChannelException {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      try (Dataset dataset2 = testDataset.write(1, 5)) {
        // For a single-fragment dataset, physical row IDs match row offsets
        List<Long> rowIds = Arrays.asList(1L, 4L);
        List<String> columns = Arrays.asList("id", "name");
        try (ArrowReader reader = dataset2.takeRows(rowIds, columns)) {
          while (reader.loadNextBatch()) {
            VectorSchemaRoot result = reader.getVectorSchemaRoot();
            assertNotNull(result);
            assertEquals(rowIds.size(), result.getRowCount());

            for (int i = 0; i < rowIds.size(); i++) {
              assertEquals(rowIds.get(i).intValue(), result.getVector("id").getObject(i));
              assertNotNull(result.getVector("name").getObject(i));
            }
          }
        }

        // Verify input order is preserved: reversed input yields reversed output
        List<Long> reversed = Arrays.asList(4L, 1L);
        try (ArrowReader reader = dataset2.takeRows(reversed, columns)) {
          assertTrue(reader.loadNextBatch());
          VectorSchemaRoot result = reader.getVectorSchemaRoot();
          assertEquals(4, result.getVector("id").getObject(0));
          assertEquals(1, result.getVector("id").getObject(1));
        }

        // Empty row IDs should be rejected
        assertThrows(
            IllegalArgumentException.class,
            () -> {
              dataset2.takeRows(Collections.emptyList(), columns);
            });
        assertThrows(
            IllegalArgumentException.class,
            () -> {
              dataset2.takeRows(null, columns);
            });
      }
    }
  }

  @Test
  void testSample(@TempDir Path tempDir) throws IOException {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      try (Dataset dataset2 = testDataset.write(1, 20)) {
        // Sample without fragment filter
        List<String> columns = Arrays.asList("id", "name");
        try (ArrowReader reader = dataset2.sample(5, columns)) {
          assertTrue(reader.loadNextBatch());
          VectorSchemaRoot result = reader.getVectorSchemaRoot();
          assertNotNull(result);
          assertEquals(5, result.getRowCount());
          assertEquals(2, result.getSchema().getFields().size());
        }

        // Sample with fragment filter
        List<Fragment> fragments = dataset2.getFragments();
        assertFalse(fragments.isEmpty());
        List<Integer> fragmentIds =
            fragments.stream().map(f -> f.getId()).collect(Collectors.toList());
        try (ArrowReader reader = dataset2.sample(3, columns, Optional.of(fragmentIds))) {
          assertTrue(reader.loadNextBatch());
          VectorSchemaRoot result = reader.getVectorSchemaRoot();
          assertNotNull(result);
          assertEquals(3, result.getRowCount());
        }

        // Sample more than available rows returns all rows
        try (ArrowReader reader = dataset2.sample(100, columns)) {
          assertTrue(reader.loadNextBatch());
          VectorSchemaRoot result = reader.getVectorSchemaRoot();
          assertNotNull(result);
          assertEquals(20, result.getRowCount());
        }
      }
    }
  }

  @Test
  void testCountRows(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      try (Dataset dataset2 = testDataset.write(1, 5)) {
        assertEquals(5, dataset2.countRows());
        // get id = 3 and 4
        assertEquals(2, dataset2.countRows("id > 2"));

        assertThrows(IllegalArgumentException.class, () -> dataset2.countRows(null));
        assertThrows(IllegalArgumentException.class, () -> dataset2.countRows(""));
      }
    }
  }

  @Test
  void testCalculateDataSize(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      try (Dataset dataset2 = testDataset.write(1, 5)) {
        assertEquals(112, dataset2.calculateDataSize());
      }
    }
  }

  @Test
  void testDeleteRows(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      try (Dataset dataset2 = testDataset.write(1, 5)) {
        // Initially there are 5 rows
        assertEquals(5, dataset2.countRows());

        // Delete rows where id > 2 (should delete id=3, id=4)
        dataset2.delete("id > 2");

        // Now verify we have 3 rows left (id=0, id=1, id=2)
        assertEquals(3, dataset2.countRows());

        // Verify the rows that remain
        assertEquals(0, dataset2.countRows("id > 2"));
        assertEquals(3, dataset2.countRows("id <= 2"));

        // Delete another row
        dataset2.delete("id = 1");

        // Now verify we have 2 rows left (id=0, id=2)
        assertEquals(2, dataset2.countRows());
        assertEquals(1, dataset2.countRows("id = 0"));
        assertEquals(1, dataset2.countRows("id = 2"));
        assertEquals(0, dataset2.countRows("id = 1"));
      }
    }
  }

  @Test
  void testUpdateConfig(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      assertEquals(1, dataset.version());
      Map<String, String> originalConfig = dataset.getConfig();
      Map<String, String> updateConfig = new HashMap<>();
      updateConfig.put("key1", "value1");
      updateConfig.put("key2", "value2");

      UpdateMap configUpdate = UpdateMap.builder().updates(updateConfig).replace(false).build();

      dataset =
          dataset
              .newTransactionBuilder()
              .operation(UpdateConfig.builder().configUpdates(configUpdate).build())
              .build()
              .commit();
      originalConfig.putAll(updateConfig);
      assertEquals(2, dataset.version());
      Map<String, String> currentConfig = dataset.getConfig();
      for (String configKey : currentConfig.keySet()) {
        assertEquals(currentConfig.get(configKey), originalConfig.get(configKey));
      }
      assertEquals(originalConfig.size(), currentConfig.size());

      Map<String, String> updateConfig2 = new HashMap<>();
      updateConfig2.put("key1", "value3");

      UpdateMap configUpdate2 = UpdateMap.builder().updates(updateConfig2).replace(false).build();

      dataset =
          dataset
              .newTransactionBuilder()
              .operation(UpdateConfig.builder().configUpdates(configUpdate2).build())
              .build()
              .commit();
      currentConfig = dataset.getConfig();
      originalConfig.putAll(updateConfig2);
      assertEquals(3, dataset.version());
      for (String configKey : currentConfig.keySet()) {
        assertEquals(currentConfig.get(configKey), originalConfig.get(configKey));
      }
      assertEquals(originalConfig.size(), currentConfig.size());
    }
  }

  @Test
  void testDeleteConfigKeys(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      assertEquals(1, dataset.version());
      Map<String, String> originalConfig = dataset.getConfig();
      Map<String, String> config = new HashMap<>();
      config.put("key1", "value1");
      config.put("key2", "value2");

      UpdateMap configUpdate = UpdateMap.builder().updates(config).replace(false).build();

      dataset =
          dataset
              .newTransactionBuilder()
              .operation(UpdateConfig.builder().configUpdates(configUpdate).build())
              .build()
              .commit();
      assertEquals(2, dataset.version());
      Map<String, String> currentConfig = dataset.getConfig();
      assertTrue(currentConfig.keySet().containsAll(config.keySet()));
      assertEquals(originalConfig.size() + 2, currentConfig.size());

      Set<String> deleteKeys = new HashSet<>();
      deleteKeys.add("key1");
      dataset.deleteConfigKeys(deleteKeys);
      assertEquals(3, dataset.version());
      originalConfig = currentConfig;
      currentConfig = dataset.getConfig();
      assertEquals(originalConfig.size() - 1, currentConfig.size());
      assertTrue(currentConfig.containsKey("key2"));
      assertFalse(currentConfig.containsKey("key1"));
      deleteKeys.add("key2");
      dataset.deleteConfigKeys(deleteKeys);
      assertEquals(4, dataset.version());
      currentConfig = dataset.getConfig();
      assertEquals(originalConfig.size() - 2, currentConfig.size());
      assertFalse(currentConfig.containsKey("key2"));
      assertFalse(currentConfig.containsKey("key1"));
    }
  }

  @Test
  void testGetLanceSchema(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.ComplexTestDataset testDataset =
          new TestUtils.ComplexTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      assertEquals(testDataset.getSchema(), dataset.getLanceSchema().asArrowSchema());
    }
  }

  @Test
  void testReplaceSchemaMetadata(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      assertEquals(1, dataset.version());
      Map<String, String> replaceMetadata = new HashMap<>();
      replaceMetadata.put("key1", "value1");
      replaceMetadata.put("key2", "value2");
      UpdateMap schemaMetadataReplace =
          UpdateMap.builder().updates(replaceMetadata).replace(true).build();
      dataset =
          dataset
              .newTransactionBuilder()
              .operation(
                  UpdateConfig.builder().schemaMetadataUpdates(schemaMetadataReplace).build())
              .build()
              .commit();
      assertEquals(2, dataset.version());
      Map<String, String> currentMetadata = dataset.getSchema().getCustomMetadata();
      for (String configKey : currentMetadata.keySet()) {
        assertEquals(currentMetadata.get(configKey), replaceMetadata.get(configKey));
      }
      assertEquals(replaceMetadata.size(), currentMetadata.size());

      Map<String, String> replaceConfig2 = new HashMap<>();
      replaceConfig2.put("key1", "value3");
      Map<String, String> schemaUpdates = new HashMap<>();
      schemaUpdates.put("key1", "value3");
      UpdateMap schemaMetadataUpdate =
          UpdateMap.builder().updates(schemaUpdates).replace(true).build();
      dataset =
          dataset
              .newTransactionBuilder()
              .operation(UpdateConfig.builder().schemaMetadataUpdates(schemaMetadataUpdate).build())
              .build()
              .commit();
      currentMetadata = dataset.getSchema().getCustomMetadata();
      assertEquals(3, dataset.version());
      assertEquals(1, currentMetadata.size());
      assertEquals("value3", currentMetadata.get("key1"));
    }
  }

  @Test
  void testReplaceFieldConfig(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();
      assertEquals(1, dataset.version());
      LanceField field = dataset.getLanceSchema().fields().get(0);
      Map<String, String> replaceMetadata = new HashMap<>();
      replaceMetadata.put("key1", "value1");
      replaceMetadata.put("key2", "value2");
      Map<Integer, UpdateMap> fieldMetadataUpdates = new HashMap<>();
      UpdateMap fieldUpdateMap = UpdateMap.builder().updates(replaceMetadata).replace(true).build();
      fieldMetadataUpdates.put(field.getId(), fieldUpdateMap);
      dataset =
          dataset
              .newTransactionBuilder()
              .operation(UpdateConfig.builder().fieldMetadataUpdates(fieldMetadataUpdates).build())
              .build()
              .commit();
      assertEquals(2, dataset.version());
      Map<String, String> currentMetadata = dataset.getSchema().getFields().get(0).getMetadata();
      for (String configKey : currentMetadata.keySet()) {
        assertEquals(currentMetadata.get(configKey), replaceMetadata.get(configKey));
      }
      assertEquals(replaceMetadata.size(), currentMetadata.size());

      Map<String, String> replaceConfig2 = new HashMap<>();
      replaceConfig2.put("key1", "value3");
      Map<Integer, UpdateMap> fieldMetadataUpdates2 = new HashMap<>();
      UpdateMap fieldUpdateMap2 = UpdateMap.builder().updates(replaceConfig2).replace(true).build();
      fieldMetadataUpdates2.put(field.getId(), fieldUpdateMap2);
      dataset =
          dataset
              .newTransactionBuilder()
              .operation(UpdateConfig.builder().fieldMetadataUpdates(fieldMetadataUpdates2).build())
              .build()
              .commit();
      currentMetadata = dataset.getSchema().getFields().get(0).getMetadata();
      assertEquals(3, dataset.version());
      assertEquals(1, currentMetadata.size());
      assertEquals("value3", currentMetadata.get("key1"));

      assertThrows(
          IllegalArgumentException.class,
          () -> {
            Map<Integer, UpdateMap> badUpdates = new HashMap<>();
            UpdateMap badUpdateMap =
                UpdateMap.builder().updates(replaceConfig2).replace(true).build();
            badUpdates.put(Integer.MAX_VALUE, badUpdateMap);
            dataset
                .newTransactionBuilder()
                .operation(UpdateConfig.builder().fieldMetadataUpdates(badUpdates).build())
                .build()
                .commit();
          });
      assertThrows(
          IllegalArgumentException.class,
          () -> {
            Map<Integer, UpdateMap> badUpdates2 = new HashMap<>();
            UpdateMap badUpdateMap2 =
                UpdateMap.builder().updates(replaceConfig2).replace(true).build();
            badUpdates2.put(-1, badUpdateMap2);
            dataset
                .newTransactionBuilder()
                .operation(UpdateConfig.builder().fieldMetadataUpdates(badUpdates2).build())
                .build()
                .commit();
          });
    }
  }

  @Test
  void testReadTransaction(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("read_transaction").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      // version 1, empty dataset
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        assertEquals(1, dataset.version());
        assertEquals(1, dataset.latestVersion());
        assertEquals(0, dataset.countRows());
        Transaction readTransaction =
            dataset
                .readTransaction()
                .orElseThrow(() -> new IllegalStateException("transaction is empty"));
        assertEquals(0, readTransaction.readVersion());
        assertNotNull(readTransaction.uuid());
        assertInstanceOf(Overwrite.class, readTransaction.operation());
        try (Dataset dataset2 = testDataset.write(1, 5)) {
          assertEquals(2, dataset2.version());
          assertEquals(2, dataset2.latestVersion());
          assertEquals(5, dataset2.countRows());
          readTransaction =
              dataset2
                  .readTransaction()
                  .orElseThrow(() -> new IllegalStateException("transaction is empty"));
          assertEquals(1, readTransaction.readVersion());
          assertNotNull(readTransaction.uuid());
          assertInstanceOf(Append.class, readTransaction.operation());
        }
      }
    }
  }

  @Test
  void testCommitTransactionDetachedTrue(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testCommitTransactionDetachedTrue").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset suite = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset base = suite.createEmptyDataset(true)) {
        assertEquals(1, base.version());
        assertEquals(1, base.latestVersion());
        assertEquals(0, base.countRows());
        long baseVersion = base.version();
        long baseLatestVersion = base.latestVersion();
        long baseRowCount = base.countRows();
        FragmentMetadata fragment = suite.createNewFragment(5);
        Append append = Append.builder().fragments(Collections.singletonList(fragment)).build();
        SourcedTransaction transaction = base.newTransactionBuilder().operation(append).build();
        try (Dataset committed = base.commitTransaction(transaction.transaction(), true, false)) {
          // Original dataset is not refreshed to the new version.
          assertEquals(baseVersion, base.version());
          assertEquals(baseRowCount, base.countRows());

          // Latest version should not change.
          assertEquals(base.latestVersion(), baseLatestVersion);

          // Committed dataset has a detached version.
          assertNotEquals(baseVersion + 1, committed.version());
          assertNotEquals(committed.version(), committed.latestVersion());
          assertEquals(baseRowCount + 5, committed.countRows());
        }
      }
    }
  }

  @Test
  void testCommitTransactionDetachedTrueOnV1ManifestThrowsUnsupported(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("commitTransactionDetachedTrueOnV1").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset suite = new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset = suite.createEmptyDataset()) {
        List<Version> versionsBefore = dataset.listVersions();
        long versionIdBefore = versionsBefore.get(0).getId();

        FragmentMetadata fragment = suite.createNewFragment(3);
        Append append = Append.builder().fragments(Collections.singletonList(fragment)).build();
        SourcedTransaction transaction = dataset.newTransactionBuilder().operation(append).build();
        UnsupportedOperationException ex =
            assertThrows(
                UnsupportedOperationException.class,
                () -> dataset.commitTransaction(transaction.transaction(), true, false));

        // Error should indicate detached commits are not supported on v1 manifests.
        assertNotNull(ex.getMessage());
        assertTrue(ex.getMessage().toLowerCase().contains("detached"));

        // Dataset state should remain unchanged after the failed detached commit.
        assertEquals(1, dataset.version());
        assertEquals(1, dataset.latestVersion());
        assertEquals(0, dataset.countRows());
        List<Version> versionsAfter = dataset.listVersions();
        assertEquals(1, versionsAfter.size());
        assertEquals(versionIdBefore, versionsAfter.get(0).getId());
      }
    }
  }

  @Test
  void testEnableStableRowIds(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("enable_stable_row_ids").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset dataset =
          testDataset.createDatasetWithWriteParams(
              new WriteParams.Builder().withEnableStableRowIds(true).build())) {
        // Step1: write two fragments
        FragmentMetadata frag1 = testDataset.createNewFragment(10);
        FragmentMetadata frag2 = testDataset.createNewFragment(10);

        SourcedTransaction.Builder builder = new SourcedTransaction.Builder(dataset);
        Append append = Append.builder().fragments(Arrays.asList(frag1, frag2)).build();
        SourcedTransaction transaction =
            builder.operation(append).readVersion(dataset.version()).build();

        // Step2: if move-stable-rowid is enabled, the rowids of new fragments should be
        // consecutive.
        try (Dataset newDataset = transaction.commit()) {
          assertEquals(2, newDataset.version());

          LanceScanner scanner =
              newDataset.newScan(
                  new ScanOptions.Builder()
                      .withRowId(true)
                      // load data in one batch
                      .batchSize(20)
                      .fragmentIds(Arrays.asList(0, 1))
                      .build());

          try (ArrowReader reader = scanner.scanBatches()) {
            List<Long> rowIds = new ArrayList<>();
            while (reader.loadNextBatch()) {
              VectorSchemaRoot root = reader.getVectorSchemaRoot();
              UInt8Vector rowidVec = (UInt8Vector) (root.getVector("_rowid"));
              for (int i = 0; i < rowidVec.getValueCount(); i++) {
                rowIds.add(rowidVec.get(i));
              }
            }

            assertEquals(20, rowIds.size());

            // rowids should be consecutive even across fragments
            Collections.sort(rowIds);
            for (int i = 0; i < rowIds.size() - 1; i++) {
              assertEquals(rowIds.get(i) + 1, (long) rowIds.get(i + 1));
            }
          }
        }
      }
    }
  }

  @Test
  void testHasStableRowIds(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("uses_stable_row_ids").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      // Dataset created without stable row IDs
      try (Dataset ds = testDataset.createEmptyDataset()) {
        assertFalse(ds.hasStableRowIds());
      }
    }

    String datasetPathWithRowIds = tempDir.resolve("uses_stable_row_ids_enabled").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPathWithRowIds);

      // Dataset created with stable row IDs
      try (Dataset ds =
          testDataset.createDatasetWithWriteParams(
              new WriteParams.Builder().withEnableStableRowIds(true).build())) {
        assertTrue(ds.hasStableRowIds());
      }
    }
  }

  @Test
  void testCompact(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Write data to create multiple small fragments for compaction
      try (Dataset dataset2 = testDataset.write(1, 10)) {
        assertEquals(10, dataset2.countRows());
        long initialVersion = dataset2.version();

        // Test compact with default options
        dataset2.compact();

        // Verify data integrity - row count should remain the same
        assertEquals(10, dataset2.countRows());

        // Version may or may not increase depending on whether compaction was needed
        assertTrue(dataset2.version() >= initialVersion);

        // Test compact with custom options
        CompactionOptions customOptions =
            CompactionOptions.builder()
                .withTargetRowsPerFragment(20)
                .withMaxRowsPerGroup(1024)
                .withMaterializeDeletions(true)
                .withMaterializeDeletionsThreshold(0.1f)
                .build();

        long preCustomCompactVersion = dataset2.version();
        dataset2.compact(customOptions);

        // Verify data integrity after custom compaction
        assertEquals(10, dataset2.countRows());

        // Test that CompactionOptions getters work correctly
        assertEquals(20, customOptions.getTargetRowsPerFragment().get());
        assertEquals(1024, customOptions.getMaxRowsPerGroup().get());
        assertEquals(0.1f, customOptions.getMaterializeDeletionsThreshold().get(), 0.001f);
        assertFalse(customOptions.getMaxBytesPerFile().isPresent());
      }
    }
  }

  @Test
  void testCompactWithDeletions(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Write data and then delete some rows to test deletion materialization
      try (Dataset dataset2 = testDataset.write(1, 20)) {
        assertEquals(20, dataset2.countRows());

        // Delete some rows to create deletions for materialization
        dataset2.delete("id < 5");
        assertEquals(15, dataset2.countRows());

        long versionBeforeCompact = dataset2.version();

        // Compact with deletion materialization
        CompactionOptions options =
            CompactionOptions.builder()
                .withMaterializeDeletions(true)
                .withMaterializeDeletionsThreshold(0.2f) // 20% threshold
                .build();

        dataset2.compact(options);

        // Verify that compaction happened
        assertTrue(dataset2.version() > versionBeforeCompact);

        // Verify data integrity - should still have 15 rows after compaction
        assertEquals(15, dataset2.countRows());

        // Verify deleted rows are still gone
        assertEquals(0, dataset2.countRows("id < 5"));
        assertEquals(15, dataset2.countRows("id >= 5"));
      }
    }
  }

  @Test
  void testCompactWithMaxBytesAndBatchSize(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Write larger dataset to test maxBytesPerFile and batchSize
      try (Dataset dataset2 = testDataset.write(1, 50)) {
        assertEquals(50, dataset2.countRows());
        long initialVersion = dataset2.version();

        // Test compact with maxBytesPerFile and batchSize options
        CompactionOptions options =
            CompactionOptions.builder()
                .withTargetRowsPerFragment(100)
                .withMaxBytesPerFile(1024 * 1024) // 1MB limit
                .withBatchSize(10) // Process 10 rows at a time
                .withNumThreads(2) // Use 2 threads
                .withMaterializeDeletions(false)
                .withDeferIndexRemap(true)
                .build();

        dataset2.compact(options);

        // Verify data integrity (compaction may or may not create new version)
        assertTrue(dataset2.version() >= initialVersion);

        // Verify data integrity
        assertEquals(50, dataset2.countRows());

        // Test that all options are set correctly
        assertEquals(100, options.getTargetRowsPerFragment().get());
        assertTrue(options.getMaxBytesPerFile().isPresent());
        assertEquals(1024 * 1024, options.getMaxBytesPerFile().get().intValue());
        assertTrue(options.getBatchSize().isPresent());
        assertEquals(10, options.getBatchSize().get().intValue());
        assertTrue(options.getNumThreads().isPresent());
        assertEquals(2, options.getNumThreads().get().intValue());
        assertFalse(options.getMaterializeDeletions().get());
        assertTrue(options.getDeferIndexRemap().get());
      }
    }
  }

  @Test
  void testMultipleCompactions(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Write initial data
      try (Dataset dataset2 = testDataset.write(1, 30)) {
        assertEquals(30, dataset2.countRows());
        long version1 = dataset2.version();

        // First compaction with default options
        dataset2.compact();
        long version2 = dataset2.version();
        assertTrue(version2 >= version1);
        assertEquals(30, dataset2.countRows());

        // Delete some rows
        dataset2.delete("id < 10");
        assertEquals(20, dataset2.countRows());
        long version3 = dataset2.version();
        assertTrue(version3 > version2);

        // Second compaction with deletion materialization
        CompactionOptions deletionOptions =
            CompactionOptions.builder()
                .withMaterializeDeletions(true)
                .withMaterializeDeletionsThreshold(0.3f)
                .build();

        dataset2.compact(deletionOptions);
        long version4 = dataset2.version();
        assertTrue(version4 > version3);
        assertEquals(20, dataset2.countRows());

        // Verify deleted rows are still gone
        assertEquals(0, dataset2.countRows("id < 10"));

        // Third compaction with different target fragment size
        CompactionOptions fragmentOptions =
            CompactionOptions.builder()
                .withTargetRowsPerFragment(5)
                .withMaxRowsPerGroup(512)
                .build();

        dataset2.compact(fragmentOptions);
        long version5 = dataset2.version();
        assertTrue(version5 >= version4);
        assertEquals(20, dataset2.countRows());

        // Verify multiple compactions preserve data integrity
        assertEquals(0, dataset2.countRows("id < 10"));
        assertEquals(20, dataset2.countRows("id >= 10"));

        // Verify we can still query the data correctly
        assertEquals(10, dataset2.countRows("id >= 10 AND id < 20"));
        assertEquals(10, dataset2.countRows("id >= 20"));
      }
    }
  }

  @Test
  void testCompactWithAllOptions(@TempDir Path tempDir) {
    String testMethodName = new Object() {}.getClass().getEnclosingMethod().getName();
    String datasetPath = tempDir.resolve(testMethodName).toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      dataset = testDataset.createEmptyDataset();

      // Write data and create some deletions
      try (Dataset dataset2 = testDataset.write(1, 25)) {
        assertEquals(25, dataset2.countRows());

        // Delete some rows to test deletion materialization
        dataset2.delete("id % 5 = 0"); // Delete every 5th row
        assertEquals(20, dataset2.countRows()); // Should have 20 rows left

        long versionBeforeCompact = dataset2.version();

        // Test compaction with all options set
        CompactionOptions allOptions =
            CompactionOptions.builder()
                .withTargetRowsPerFragment(15)
                .withMaxRowsPerGroup(256)
                .withMaxBytesPerFile(512 * 1024) // 512KB
                .withMaterializeDeletions(true)
                .withMaterializeDeletionsThreshold(0.15f) // 15% threshold
                .withNumThreads(1)
                .withBatchSize(5)
                .withDeferIndexRemap(false)
                .build();

        dataset2.compact(allOptions);

        // Verify compaction occurred
        assertTrue(dataset2.version() > versionBeforeCompact);
        assertEquals(20, dataset2.countRows());

        // Verify deleted rows remain deleted
        assertEquals(0, dataset2.countRows("id % 5 = 0"));
        assertEquals(20, dataset2.countRows("id % 5 != 0"));

        // Verify all CompactionOptions settings are preserved
        assertEquals(15, allOptions.getTargetRowsPerFragment().get());
        assertEquals(256, allOptions.getMaxRowsPerGroup().get());
        assertTrue(allOptions.getMaxBytesPerFile().isPresent());
        assertEquals(512 * 1024, allOptions.getMaxBytesPerFile().get().intValue());
        assertTrue(allOptions.getMaterializeDeletions().get());
        assertEquals(0.15f, allOptions.getMaterializeDeletionsThreshold().get(), 0.001f);
        assertTrue(allOptions.getNumThreads().isPresent());
        assertEquals(1, allOptions.getNumThreads().get().intValue());
        assertTrue(allOptions.getBatchSize().isPresent());
        assertEquals(5, allOptions.getBatchSize().get().intValue());
        assertFalse(allOptions.getDeferIndexRemap().get());
      }
    }
  }

  /**
   * This method must be aligned with the implementation in <a
   * href="https://github.com/lancedb/lance/blob/main/rust/lance-table/src/io/manifest.rs#L95-L96">...</a>
   */
  public ByteBuffer readManifest(Path filePath) throws IOException {
    byte[] fileBytes = Files.readAllBytes(filePath);
    int fileSize = fileBytes.length;

    // Basic file size validation
    if (fileSize < 16) {
      throw new IllegalArgumentException("File too small");
    }

    // Read the last 16 bytes of the file to get metadata
    // Structure: [manifest_pos (8 bytes)][magic (8 bytes)]
    ByteBuffer tailBuffer = ByteBuffer.wrap(fileBytes, fileSize - 16, 16);
    tailBuffer.order(ByteOrder.LITTLE_ENDIAN);
    long manifestPos = tailBuffer.getLong(); // Read manifest start position
    // Magic number bytes are read but not validated, we simply skip over them
    tailBuffer.getLong(); // This reads and skips the 8-byte magic number

    // Remove strict validation since file_size can be larger than manifest_size
    // due to index and transaction metadata at the beginning of the file
    // Only ensure manifestPos is not negative and doesn't cause overflow
    if (manifestPos < 0 || manifestPos >= Integer.MAX_VALUE) {
      throw new IllegalArgumentException("Invalid manifest position: " + manifestPos);
    }

    int manifestStart = (int) manifestPos;

    // Verify we have enough data for the length field
    if (manifestStart + 4 > fileSize) {
      throw new IllegalArgumentException("Manifest position beyond file bounds");
    }

    // Calculate the actual length of the protobuf data
    // The structure is: [4-byte length][protobuf data][8-byte manifest_pos][8-byte magic]
    byte[] trimmedManifest =
        Arrays.copyOfRange(fileBytes, manifestStart + 4, fileBytes.length - 16);
    ByteBuffer manifestBuffer = ByteBuffer.allocateDirect(trimmedManifest.length);
    manifestBuffer.put(trimmedManifest);
    manifestBuffer.flip();
    return manifestBuffer;
  }

  @Test
  void testShallowClone(@TempDir Path tempDir) {
    String srcPath = tempDir.resolve("shallow_clone_version_src").toString();
    String dstPathByVersion = tempDir.resolve("shallow_clone_version_dst").toString();
    String dstPathByTag = tempDir.resolve("shallow_clone_tag_dst").toString();

    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      // Prepare a simple source dataset with some rows
      TestUtils.SimpleTestDataset suite = new TestUtils.SimpleTestDataset(allocator, srcPath);
      try (Dataset empty = suite.createEmptyDataset()) {
        assertEquals(1, empty.version());
      }

      try (Dataset src = suite.write(1, 5)) { // write 5 rows -> version 2
        assertEquals(2, src.version());
        long srcRowCount = src.countRows();
        Schema srcSchema = src.getSchema();

        // shallow clone by version
        try (Dataset clone = src.shallowClone(dstPathByVersion, Ref.ofMain(src.version()))) {
          // Validate the version cloned dataset
          assertNotNull(clone);
          assertEquals(dstPathByVersion, clone.uri());
          assertEquals(srcSchema.getFields(), clone.getSchema().getFields());
          assertEquals(srcRowCount, clone.countRows());
        }

        // Ensure the dataset at targetPath can be opened successfully
        try (Dataset opened =
            Dataset.open(allocator, dstPathByVersion, new ReadOptions.Builder().build())) {
          assertNotNull(opened);
          assertEquals(srcSchema.getFields(), opened.getSchema().getFields());
          assertEquals(srcRowCount, opened.countRows());
        }

        // shallow clone by tag
        src.tags().create("tag", src.version());
        try (Dataset clone = src.shallowClone(dstPathByTag, Ref.ofTag("tag"))) {
          // Validate the tag cloned dataset
          assertNotNull(clone);
          assertEquals(dstPathByTag, clone.uri());
          assertEquals(srcSchema.getFields(), clone.getSchema().getFields());
          assertEquals(srcRowCount, clone.countRows());
        }

        // Ensure the dataset at targetPath can be opened successfully
        try (Dataset opened =
            Dataset.open(allocator, dstPathByTag, new ReadOptions.Builder().build())) {
          assertNotNull(opened);
          assertEquals(srcSchema.getFields(), opened.getSchema().getFields());
          assertEquals(srcRowCount, opened.countRows());
        }
      }
    }
  }

  @Test
  void testBranches(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("testBranches").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset suite = new TestUtils.SimpleTestDataset(allocator, datasetPath);

      try (Dataset mainV1 = suite.createEmptyDataset()) {
        assertEquals(1, mainV1.version());

        // Step 1. write to main dataset, 5 rows -> main:2
        try (Dataset mainV2 = suite.write(1, 5)) {
          assertEquals(2, mainV2.version());
          assertEquals(5, mainV2.countRows());

          // Step2. create branch2 based on main:2
          try (Dataset branch1V2 = mainV2.createBranch("branch1", Ref.ofMain(2))) {
            assertEquals(2, branch1V2.version());

            // Write batch B on branch1: 3 rows -> global@3
            FragmentMetadata fragB = suite.createNewFragment(3);
            Append appendB = Append.builder().fragments(Collections.singletonList(fragB)).build();
            try (Dataset branch1V3 =
                branch1V2.newTransactionBuilder().operation(appendB).build().commit()) {
              assertEquals(3, branch1V3.version());
              assertEquals(8, branch1V3.countRows()); // A(5) + B(3)

              // Step 3. Create branch2 based on branch1's latest version (simulate tag 't1')
              mainV1.tags().create("tag", Ref.ofBranch("branch1", 3));

              try (Dataset branch2V3 = branch1V2.createBranch("branch2", Ref.ofTag("tag"))) {
                assertEquals(3, branch2V3.version());
                assertEquals(8, branch2V3.countRows()); // A(5) + B(3)

                // Step 4. Write batch C on branch2: 2 rows -> branch2:4
                FragmentMetadata fragC = suite.createNewFragment(2);
                Append appendC =
                    Append.builder().fragments(Collections.singletonList(fragC)).build();
                try (Dataset branch2V4 =
                    branch2V3.newTransactionBuilder().operation(appendC).build().commit()) {
                  assertEquals(4, branch2V4.version());
                  assertEquals(10, branch2V4.countRows()); // A(5) + B(3) + C(2)

                  // Step 5. Validate branch listing metadata;
                  // delete branch1;
                  // validate listing again
                  List<Branch> branches = branch2V4.branches().list();
                  Optional<Branch> b1 =
                      branches.stream().filter(b -> b.getName().equals("branch1")).findFirst();
                  Optional<Branch> b2 =
                      branches.stream().filter(b -> b.getName().equals("branch2")).findFirst();
                  assertTrue(b1.isPresent(), "branch1 should be listed");
                  assertTrue(b2.isPresent(), "branch2 should be listed");
                  Branch branch1Meta = b1.get();
                  Branch branch2Meta = b2.get();

                  // Metadata fields and consistency checks
                  assertEquals("branch1", branch1Meta.getName());
                  assertEquals(2, branch1Meta.getParentVersion());
                  assertFalse(branch1Meta.getParentBranch().isPresent());
                  assertEquals(1, branch1Meta.getBranchIdentifier().size());
                  assertEquals(2, branch1Meta.getBranchIdentifier().get(0).getVersion());
                  assertFalse(branch1Meta.getBranchIdentifier().get(0).getUuid().isEmpty());
                  assertTrue(branch1Meta.getCreateAt() > 0);
                  assertTrue(branch1Meta.getManifestSize() > 0);
                  assertEquals(Collections.emptyMap(), branch1Meta.getMetadata());
                  mainV2
                      .branches()
                      .replaceMetadata("branch1", Map.of("description", "long-lived branch"));
                  branches = branch2V4.branches().list();
                  b1 = branches.stream().filter(b -> b.getName().equals("branch1")).findFirst();
                  assertTrue(b1.isPresent());
                  assertEquals(Map.of("description", "long-lived branch"), b1.get().getMetadata());
                  mainV2.branches().replaceMetadata("branch1", Collections.emptyMap());
                  branches = branch2V4.branches().list();
                  b1 = branches.stream().filter(b -> b.getName().equals("branch1")).findFirst();
                  assertTrue(b1.isPresent());
                  assertEquals(Collections.emptyMap(), b1.get().getMetadata());

                  assertEquals("branch2", branch2Meta.getName());
                  assertTrue(branch2Meta.getParentBranch().isPresent());
                  assertEquals("branch1", branch2Meta.getParentBranch().get());
                  assertEquals(2, branch2Meta.getBranchIdentifier().size());
                  assertEquals(
                      branch1Meta.getBranchIdentifier().get(0),
                      branch2Meta.getBranchIdentifier().get(0));
                  assertEquals(3, branch2Meta.getBranchIdentifier().get(1).getVersion());
                  assertFalse(branch2Meta.getBranchIdentifier().get(1).getUuid().isEmpty());
                  assertEquals(3, branch2Meta.getParentVersion());
                  assertTrue(branch2Meta.getCreateAt() > 0);
                  assertTrue(branch2Meta.getManifestSize() > 0);

                  // Delete branch1 and verify listing
                  mainV2.branches().delete("branch2");
                  assertEquals(1, mainV2.branches().list().size());

                  // Step 6. use checkout_branch to checkout branch1
                  try (Dataset branch2V4New = mainV2.checkout(Ref.ofBranch("branch1"))) {
                    assertEquals(3, branch2V4New.version());
                    assertEquals(8, branch2V4New.countRows()); // A(5) + B(3)
                  }

                  // Step 7. use checkout reference to checkout branch2
                  try (Dataset branch2V4New = mainV2.checkout(Ref.ofBranch("branch1", 2))) {
                    assertEquals(2, branch2V4New.version());
                    assertEquals(5, branch2V4New.countRows()); // A(5)
                  }

                  // Step 8. open the dataset directly at a reference
                  mainV2.tags().create("main_tag", Ref.ofMain(2));
                  try (Dataset opened = openAt(allocator, datasetPath, Ref.ofBranch("branch1"))) {
                    assertEquals(3, opened.version());
                    assertEquals(8, opened.countRows()); // A(5) + B(3)
                    assertTrue(opened.uri().contains("tree/branch1"), opened.uri());
                  }
                  try (Dataset opened =
                      openAt(allocator, datasetPath, Ref.ofBranch("branch1", 2))) {
                    assertEquals(2, opened.version());
                    assertEquals(5, opened.countRows()); // A(5)
                    assertTrue(opened.uri().contains("tree/branch1"), opened.uri());
                  }
                  // main has no version 3, only branch1 does.
                  try (Dataset opened =
                      openAt(allocator, datasetPath, Ref.ofBranch("branch1", 3))) {
                    assertEquals(3, opened.version());
                    assertEquals(8, opened.countRows()); // A(5) + B(3)
                    assertTrue(opened.uri().contains("tree/branch1"), opened.uri());
                  }
                  try (Dataset opened = openAt(allocator, datasetPath, Ref.ofMain(1))) {
                    assertEquals(1, opened.version());
                    assertEquals(0, opened.countRows());
                    assertFalse(opened.uri().contains("tree/"), opened.uri());
                  }
                  try (Dataset opened = openAt(allocator, datasetPath, Ref.ofMain())) {
                    assertEquals(2, opened.version());
                    assertEquals(5, opened.countRows()); // A(5)
                    assertFalse(opened.uri().contains("tree/"), opened.uri());
                  }
                  try (Dataset opened = openAt(allocator, datasetPath, Ref.ofBranch("main", 1))) {
                    assertEquals(1, opened.version());
                    assertFalse(opened.uri().contains("tree/"), opened.uri());
                  }
                  try (Dataset opened = openAt(allocator, datasetPath, Ref.ofTag("main_tag"))) {
                    assertEquals(2, opened.version());
                    assertEquals(5, opened.countRows()); // A(5)
                    assertFalse(opened.uri().contains("tree/"), opened.uri());
                  }
                  // "tag" points at branch1:3, so it opens branch1, not version 3 of main.
                  try (Dataset opened = openAt(allocator, datasetPath, Ref.ofTag("tag"))) {
                    assertEquals(3, opened.version());
                    assertEquals(8, opened.countRows()); // A(5) + B(3)
                    assertTrue(opened.uri().contains("tree/branch1"), opened.uri());
                  }
                  IOException missing =
                      assertThrows(
                          IOException.class,
                          () -> openAt(allocator, datasetPath, Ref.ofBranch("no_such_branch")));
                  assertTrue(missing.getMessage().contains("no_such_branch"), missing.getMessage());
                  RuntimeException missingTag =
                      assertThrows(
                          RuntimeException.class,
                          () -> openAt(allocator, datasetPath, Ref.ofTag("no_such_tag")));
                  assertTrue(
                      missingTag.getMessage().contains("no_such_tag"), missingTag.getMessage());
                  IllegalArgumentException missingVersion =
                      assertThrows(
                          IllegalArgumentException.class,
                          () -> openAt(allocator, datasetPath, Ref.ofBranch("branch1", 99)));
                  assertTrue(
                      missingVersion.getMessage().contains("branch1"), missingVersion.getMessage());
                }
              }
            }
          }
        }
      }
    }
  }

  private static Dataset openAt(BufferAllocator allocator, String path, Ref ref) {
    return Dataset.open(allocator, path, new ReadOptions.Builder().setRef(ref).build());
  }

  @Test
  void testReadOptionsRejectsConflictingRef() {
    ReadOptions.Builder withVersion =
        new ReadOptions.Builder().setVersion(1).setRef(Ref.ofBranch("branch1"));
    IllegalArgumentException e = assertThrows(IllegalArgumentException.class, withVersion::build);
    assertTrue(e.getMessage().contains("both version (1) and ref"), e.getMessage());

    ReadOptions.Builder withManifest =
        new ReadOptions.Builder()
            .setSerializedManifest(ByteBuffer.allocateDirect(1))
            .setRef(Ref.ofBranch("branch1"));
    e = assertThrows(IllegalArgumentException.class, withManifest::build);
    assertTrue(e.getMessage().contains("serialized manifest and ref"), e.getMessage());
  }

  @Test
  void testOptimizingIndices(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("optimize_scalar").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      // version 1, empty dataset
      try (Dataset ignored = testDataset.createEmptyDataset()) {
        // write first fragment at version 1 -> dataset version 2
        try (Dataset dsWithData = testDataset.write(1, 10)) {
          ScalarIndexParams scalarParams =
              ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");
          IndexParams indexParams =
              IndexParams.builder().setScalarIndexParams(scalarParams).build();

          dsWithData.createIndex(
              Collections.singletonList("id"),
              IndexType.BTREE,
              Optional.of("id_idx"),
              indexParams,
              true);

          List<Index> beforeIndexes = dsWithData.getIndexes();
          Index idIndexBefore =
              beforeIndexes.stream()
                  .filter(idx -> "id_idx".equals(idx.name()))
                  .findFirst()
                  .orElse(null);
          assertNotNull(idIndexBefore);
          List<Integer> beforeFragments = idIndexBefore.fragments().orElse(Collections.emptyList());
          assertTrue(beforeFragments.contains(0));
          assertEquals(1, beforeFragments.size());
        }

        // append new fragment using readVersion 2 -> dataset version 3
        try (Dataset dsAppended = testDataset.write(2, 10)) {
          OptimizeOptions options = OptimizeOptions.builder().numIndicesToMerge(0).build();
          dsAppended.optimizeIndices(options);

          List<Index> idIndexes =
              dsAppended.getIndexes().stream()
                  .filter(idx -> "id_idx".equals(idx.name()))
                  .collect(Collectors.toList());
          assertEquals(
              2,
              idIndexes.size(),
              "append-only optimize must add a delta segment instead of merging");

          Set<Integer> coveredFragments =
              idIndexes.stream()
                  .flatMap(idx -> idx.fragments().orElse(Collections.emptyList()).stream())
                  .collect(Collectors.toSet());
          assertEquals(new HashSet<>(Arrays.asList(0, 1)), coveredFragments);
        }
      }
    }
  }

  // ===== Blob API tests =====

  @ParameterizedTest
  @ValueSource(strings = {"acquire", "release", "cleanup"})
  void testSnapshotRetentionConflictException(String operation, @TempDir Path tempDir)
      throws Exception {
    Path path = tempDir.resolve("retention_conflict");
    try (RootAllocator allocator = new RootAllocator();
        Dataset dataset =
            new TestUtils.SimpleTestDataset(allocator, path.toString()).createEmptyDataset()) {
      SnapshotIdentity snapshot = SnapshotIdentity.fromDataset(dataset, "catalog/conflict");
      SnapshotRetention token = SnapshotRetention.acquire(dataset, snapshot, "query");
      Path owner =
          path.resolve("_refs/retention-v1/maintenance").resolve(UUID.randomUUID().toString());
      Files.createDirectories(owner.getParent());
      Files.writeString(
          owner, "{\"ticket\":1,\"expires_at\":\"2100-01-01T00:00:00Z\",\"mode\":\"exclusive\"}");
      try {
        LanceRefConflictException error =
            assertThrows(
                LanceRefConflictException.class,
                () -> {
                  switch (operation) {
                    case "acquire":
                      SnapshotRetention.acquire(dataset, snapshot, "other-query");
                      break;
                    case "release":
                      token.release(dataset);
                      break;
                    case "cleanup":
                      dataset.cleanup(CleanupPolicy.builder().build()).execute();
                      break;
                    default:
                      throw new AssertionError(operation);
                  }
                });
        assertTrue(error.getMessage().contains("maintenance lock"));
      } finally {
        Files.delete(owner);
        token.release(dataset);
      }
      assertTrue(dataset.tags().list().isEmpty());
    }
  }

  @Test
  void testSnapshotAndBlobEntryPointsRejectNullAndClosedDatasets(@TempDir Path tempDir)
      throws Exception {
    try (RootAllocator allocator = new RootAllocator();
        Dataset original =
            createBlobV2SelectorDataset(allocator, tempDir.resolve("closed").toString())) {
      SourceContext source = SourceContext.fromDataset(original, "catalog/closed", "workers");
      SnapshotIdentity snapshot = source.getSnapshotIdentity();
      SnapshotRetention token = SnapshotRetention.acquire(original, snapshot, "query");
      assertThrows(NullPointerException.class, () -> SnapshotIdentity.fromDataset(null, "table"));
      assertThrows(
          NullPointerException.class, () -> SourceContext.fromDataset(null, "table", "workers"));
      assertThrows(NullPointerException.class, () -> Capture.create(null, source, 0));
      assertThrows(
          NullPointerException.class,
          () -> Reader.create(List.of(source), Arrays.asList((Dataset) null), 1, 1));
      assertThrows(
          NullPointerException.class, () -> SnapshotRetention.acquire(null, snapshot, "other"));
      assertThrows(NullPointerException.class, () -> token.release(null));
      token.release(original);
      original.close();
      assertThrows(
          IllegalArgumentException.class, () -> SnapshotIdentity.fromDataset(original, "table"));
      assertThrows(
          IllegalArgumentException.class,
          () -> SourceContext.fromDataset(original, "table", "workers"));
      assertThrows(IllegalArgumentException.class, () -> Capture.create(original, source, 0));
      assertThrows(
          IllegalArgumentException.class,
          () -> Reader.create(List.of(source), List.of(original), 1, 1));
      assertThrows(
          IllegalArgumentException.class,
          () -> SnapshotRetention.acquire(original, snapshot, "other"));
      assertThrows(IllegalArgumentException.class, () -> token.release(original));
      try (Dataset worker =
          Dataset.open().allocator(allocator).uri(tempDir.resolve("closed").toString()).build()) {
        assertThrows(
            NullPointerException.class,
            () -> Reader.create(List.of(source, source), Arrays.asList(worker, null), 1, 1));
        assertThrows(
            IllegalArgumentException.class,
            () -> Reader.create(List.of(source, source), List.of(worker, original), 1, 1));
        assertThrows(
            IllegalArgumentException.class,
            () -> Reader.create(List.of(source), List.of(worker, worker), 1, 1));
        assertEquals(source.getVersion(), worker.version());
        try (Reader reader = Reader.create(List.of(source), List.of(worker), 1, 1)) {
          assertEquals(0, reader.getReservedBytes());
        }
      }
    }
  }

  @Test
  void testBlobReferenceHandleCloneAndConcurrentClose(@TempDir Path tempDir) throws Exception {
    ExecutorService executor = Executors.newSingleThreadExecutor();
    try (RootAllocator allocator = new RootAllocator();
        Dataset dataset =
            createBlobV2SelectorDataset(allocator, tempDir.resolve("concurrent").toString())) {
      SourceContext source = SourceContext.fromDataset(dataset, "catalog/concurrent", "workers");
      Batch refs = capturePortableBlobBatches(dataset, source, allocator).get(0);
      try (Reader reader = Reader.create(List.of(source), List.of(dataset), 1, 1)) {
        Future<Long> reserved;
        CountDownLatch started = new CountDownLatch(1);
        synchronized (reader) {
          reserved =
              executor.submit(
                  () -> {
                    started.countDown();
                    return reader.getReservedBytes();
                  });
          assertTrue(started.await(1, TimeUnit.SECONDS));
          assertThrows(TimeoutException.class, () -> reserved.get(50, TimeUnit.MILLISECONDS));
        }
        assertEquals(0L, reserved.get(1, TimeUnit.SECONDS).longValue());
        try (ReadStream stream =
                reader.read(refs, List.of(Request.whole(0, 0), Request.whole(0, 1)));
            VectorSchemaRoot held = stream.next(allocator)) {
          assertNotNull(held);
          CountDownLatch reading = new CountDownLatch(1);
          Future<?> pending =
              executor.submit(
                  () -> {
                    reading.countDown();
                    try (VectorSchemaRoot next = stream.next(allocator)) {
                      assertNull(next);
                    } catch (IOException | IllegalArgumentException error) {
                      assertTrue(
                          error.getMessage().contains("cancel")
                              || error.getMessage().contains("closed"));
                    }
                  });
          assertTrue(reading.await(1, TimeUnit.SECONDS));
          stream.close();
          stream.cancel();
          pending.get(1, TimeUnit.SECONDS);
          assertTrue(reader.getReservedBytes() >= 2);
        }
        assertEquals(0, reader.getReservedBytes());
      }
    } finally {
      executor.shutdownNow();
    }
  }

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void testBlobReaderBindsDatasetsIndependently(boolean reverseOrder, @TempDir Path tempDir)
      throws Exception {
    ExecutorService executor = Executors.newSingleThreadExecutor();
    try (RootAllocator allocator = new RootAllocator();
        Dataset left = createBlobV2SelectorDataset(allocator, tempDir.resolve("left").toString());
        Dataset right =
            createBlobV2SelectorDataset(allocator, tempDir.resolve("right").toString())) {
      Dataset first = reverseOrder ? right : left;
      Dataset second = reverseOrder ? left : right;
      SourceContext firstSource = SourceContext.fromDataset(first, "catalog/first", "workers");
      SourceContext secondSource = SourceContext.fromDataset(second, "catalog/second", "workers");
      List<SourceContext> contexts = List.of(firstSource, secondSource);
      List<Batch> references =
          List.of(
              capturePortableBlobBatches(first, firstSource, allocator)
                  .get(0)
                  .remapContexts(contexts),
              capturePortableBlobBatches(second, secondSource, allocator)
                  .get(0)
                  .remapContexts(contexts));
      FutureTask<Reader> creating =
          new FutureTask<>(() -> Reader.create(contexts, List.of(first, second), 1, 1));
      Thread binding = new Thread(creating, "blob-reader-binding");
      binding.setDaemon(true);
      ThreadMXBean threads = ManagementFactory.getThreadMXBean();
      try {
        synchronized (second) {
          binding.start();
          long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(1);
          ThreadInfo blocked = null;
          while (System.nanoTime() < deadline) {
            blocked = threads.getThreadInfo(binding.getId());
            if (blocked != null
                && blocked.getThreadState() == Thread.State.BLOCKED
                && blocked.getLockInfo() != null
                && blocked.getLockInfo().getIdentityHashCode() == System.identityHashCode(second)) {
              break;
            }
            Thread.yield();
          }
          assertNotNull(blocked);
          assertEquals(Thread.State.BLOCKED, blocked.getThreadState());
          assertNotNull(blocked.getLockInfo());
          assertEquals(
              System.identityHashCode(second), blocked.getLockInfo().getIdentityHashCode());
          // Binding the second dataset must not pin the first dataset's lifecycle lock.
          executor.submit(first::close).get(1, TimeUnit.SECONDS);
        }
        try (Reader reader = creating.get(1, TimeUnit.SECONDS)) {
          second.close();
          for (Batch batch : references) {
            try (ReadStream stream = reader.read(batch, List.of(Request.whole(0, 0)));
                VectorSchemaRoot result = stream.next(allocator)) {
              assertNotNull(result);
              assertArrayEquals(
                  new byte[] {0, 10}, ((LargeVarBinaryVector) result.getVector("data")).get(0));
              assertNull(stream.next(allocator));
            }
          }
        }
      } finally {
        // Release the blocked monitor before cleanup, including when this regression fails.
        creating.get(1, TimeUnit.SECONDS).close();
      }
    } finally {
      executor.shutdownNow();
    }
  }

  @Test
  void testSnapshotRetentionForOrdinaryData(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("retained_data").toString();
    try (RootAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset fixture = new TestUtils.SimpleTestDataset(allocator, path);
      try (Dataset empty = fixture.createEmptyDataset();
          Dataset first = fixture.write(empty.version(), 2);
          Dataset original = fixture.write(first.version(), 2)) {
        UpdateConfig config =
            UpdateConfig.builder()
                .configUpdates(
                    UpdateMap.builder()
                        .updates(Map.of("lance.snapshot_retention.lock_expiration_hours", "6"))
                        .replace(false)
                        .build())
                .build();
        try (Dataset configured =
            original.newTransactionBuilder().operation(config).build().commit()) {
          assertEquals(2, configured.getFragments().size());
          SnapshotIdentity snapshot = SnapshotIdentity.fromDataset(configured, "catalog/ordinary");
          snapshot = SnapshotIdentity.decode(snapshot.encode());
          SnapshotRetention retention =
              SnapshotRetention.acquire(configured, snapshot, "ordinary-query");
          retention = SnapshotRetention.decode(retention.encode());
          assertArrayEquals(snapshot.encode(), retention.getSnapshot().encode());
          try (Dataset latest = Dataset.open().allocator(allocator).uri(path).build()) {
            latest.delete("true");
            CleanupPolicy policy =
                CleanupPolicy.builder()
                    .withBeforeTimestampMillis(Instant.now().plusSeconds(3600).toEpochMilli())
                    .withErrorIfTaggedOldVersions(true)
                    .build();
            try {
              latest.cleanup(policy).execute();
              int rows = 0;
              int sum = 0;
              try (LanceScanner scanner = configured.newScan();
                  ArrowReader reader = scanner.scanBatches()) {
                while (reader.loadNextBatch()) {
                  VectorSchemaRoot batch = reader.getVectorSchemaRoot();
                  IntVector ids = (IntVector) batch.getVector("id");
                  rows += batch.getRowCount();
                  for (int i = 0; i < batch.getRowCount(); i++) {
                    sum += ids.get(i);
                  }
                }
              }
              assertEquals(4, rows);
              assertEquals(2, sum);
            } finally {
              retention.release(latest);
            }
            retention.release(latest);
            assertTrue(latest.cleanup(policy).execute().getDataFilesRemoved() >= 2);
          }
        }
      }
    }
  }

  @Test
  void testPortableBlobReferences(@TempDir Path tempDir) throws Exception {
    String path = tempDir.resolve("portable_blobs").toString();
    try (RootAllocator allocator = new RootAllocator();
        Dataset original = createBlobV2SelectorDataset(allocator, path)) {
      assertEquals(1, BlobReferences.getProtocolVersion());
      assertArrayEquals(new int[] {1, 2}, BlobReferences.getSupportedCapabilities());
      assertTrue(
          Arrays.asList(BlobReferences.getSupportedOperations()).contains("snapshot_retention_v1"));
      SourceContext source = SourceContext.fromDataset(original, "catalog/portable", "workers");
      source = SourceContext.decode(source.encode());
      assertEquals(original.version(), source.getVersion());
      assertEquals("workers", source.getAccessContextKey());
      assertNull(source.getBranch());
      SnapshotRetention retention =
          SnapshotRetention.acquire(original, source.getSnapshotIdentity(), "portable-query");
      retention = SnapshotRetention.decode(retention.encode());
      assertArrayEquals(source.getSnapshotIdentity().encode(), retention.getSnapshot().encode());
      assertArrayEquals(
          SnapshotIdentity.fromDataset(original, "catalog/portable").encode(),
          source.getSnapshotIdentity().encode());
      try {
        List<Batch> batches = capturePortableBlobBatches(original, source, allocator);
        try (Dataset latest = Dataset.open().allocator(allocator).uri(path).build()) {
          latest.delete("true");
          CleanupPolicy policy =
              CleanupPolicy.builder()
                  .withBeforeTimestampMillis(Instant.now().plusSeconds(3600).toEpochMilli())
                  .withErrorIfTaggedOldVersions(true)
                  .build();
          latest.cleanup(policy).execute();
          try (Dataset worker =
                  Dataset.open()
                      .allocator(allocator)
                      .uri(source.getTableUri())
                      .readOptions(
                          new ReadOptions.Builder().setVersion(source.getVersion()).build())
                      .build();
              Reader reader = Reader.create(List.of(source), List.of(worker), 1024, 1024)) {
            int fragment = 0;
            for (Batch captured : batches) {
              Batch batch = Batch.decode(captured.encode()).remapContexts(List.of(source));
              try (VectorSchemaRoot root = batch.toArrow(allocator)) {
                batch = Batch.fromArrow(Arrays.asList(batch.getContexts()), root, allocator);
              }
              Map<Long, byte[]> results = new HashMap<>();
              try (ReadStream stream =
                  reader.read(
                      batch,
                      List.of(
                          Request.whole(0, -1L),
                          Request.whole(1, 1),
                          Request.whole(2, 2),
                          Request.range(0, 3, 0, 1),
                          Request.range(0, 4, 1, 1)))) {
                VectorSchemaRoot next;
                while ((next = stream.next(allocator)) != null) {
                  try (VectorSchemaRoot result = next) {
                    results.put(
                        ((UInt8Vector) result.getVector("request_index")).get(0),
                        ((LargeVarBinaryVector) result.getVector("data")).get(0));
                  }
                }
              }
              assertEquals(5, results.size());
              assertArrayEquals(new byte[] {(byte) (3 * fragment), 10}, results.get(-1L));
              assertTrue(results.containsKey(1L));
              assertNull(results.get(1L));
              assertArrayEquals(new byte[0], results.get(2L));
              assertArrayEquals(new byte[] {(byte) (3 * fragment)}, results.get(3L));
              assertArrayEquals(new byte[] {10}, results.get(4L));
              assertEquals(0, reader.getReservedBytes());
              assertEquals(0, reader.getIoReservedBytes());
              fragment++;
            }
            assertEquals(2, fragment);
          }
          retention.release(latest);
          retention.release(latest);
          assertTrue(latest.cleanup(policy).execute().getDataFilesRemoved() >= 2);
        }
      } finally {
        retention.release(original);
      }
    }
  }

  @Test
  void testPortableBlobReferenceValidationAndCancellation(@TempDir Path tempDir) throws Exception {
    try (RootAllocator allocator = new RootAllocator();
        Dataset dataset =
            createBlobV2SelectorDataset(allocator, tempDir.resolve("refs").toString())) {
      SourceContext source = SourceContext.fromDataset(dataset, "catalog/refs", "workers");
      Batch batch = capturePortableBlobBatches(dataset, source, allocator).get(0);
      try (Reader reader = Reader.create(List.of(source), List.of(dataset), 1, 1)) {
        IllegalArgumentException error =
            assertThrows(
                IllegalArgumentException.class,
                () -> reader.read(batch, List.of(Request.range(0, 1, 0, -1L))));
        assertTrue(error.getMessage().contains("range"));
        assertThrows(
            IllegalArgumentException.class,
            () -> reader.read(batch, List.of(Request.range(0, 1, -1L, 1))));
        assertThrows(
            IllegalArgumentException.class,
            () -> reader.read(batch, List.of(Request.whole(0, 1), Request.whole(0, 1))));
        try (ReadStream stream =
                reader.read(batch, List.of(Request.whole(0, 0), Request.whole(0, 1)));
            VectorSchemaRoot first = stream.next(allocator)) {
          assertNotNull(first);
          assertTrue(reader.getReservedBytes() >= 2);
          stream.cancel();
          IOException cancelled = assertThrows(IOException.class, () -> stream.next(allocator));
          assertTrue(cancelled.getMessage().contains("cancel"));
          stream.close();
          assertNull(stream.next(allocator));
          stream.cancel();
          assertArrayEquals(
              new byte[] {0, 10}, ((LargeVarBinaryVector) first.getVector("data")).get(0));
          assertTrue(reader.getReservedBytes() >= 2);
        }
        assertEquals(0, reader.getReservedBytes());
      }
      byte[] unsupported = source.encode();
      unsupported[1] = 99;
      assertThrows(IllegalArgumentException.class, () -> SourceContext.decode(unsupported));
    }
  }

  @Test
  void testPortableNestedBlobReferences(@TempDir Path tempDir) throws Exception {
    try (RootAllocator allocator = new RootAllocator();
        Dataset dataset =
            createNestedBlobDataset(allocator, tempDir.resolve("nested").toString())) {
      SourceContext source = SourceContext.fromDataset(dataset, "catalog/nested", "workers");
      int fieldId =
          dataset
              .getLanceSchema()
              .fields()
              .get(0)
              .getChildren()
              .get(0)
              .getChildren()
              .get(0)
              .getId();
      try (Capture capture = Capture.create(dataset, source, fieldId);
          LanceScanner scanner =
              dataset.newScan(new ScanOptions.Builder().withRowAddress(true).build());
          ArrowReader scanned = scanner.scanBatches();
          Reader reader = Reader.create(List.of(source), List.of(dataset), 1024, 1024)) {
        int batchIndex = 0;
        while (scanned.loadNextBatch()) {
          NestedBatch nested =
              capture.captureNested(source, scanned.getVectorSchemaRoot(), "media", allocator);
          nested = NestedBatch.decode(nested.encode());
          try (VectorSchemaRoot indices = nested.getIndices(allocator)) {
            NestedBatch rebuilt =
                NestedBatch.fromArrow(
                    nested.getReferences().remapContexts(List.of(source)), indices, allocator);
            assertArrayEquals(nested.getReferences().encode(), rebuilt.getReferences().encode());
            StructVector media = (StructVector) indices.getVector("media");
            ListVector clips = (ListVector) media.getChild("clips");
            if (batchIndex == 0) {
              assertEquals(0, clips.getElementStartIndex(0));
              assertEquals(3, clips.getElementEndIndex(0));
              assertEquals(3, clips.getElementStartIndex(1));
              assertEquals(3, clips.getElementEndIndex(1));
              Map<Long, byte[]> values = new HashMap<>();
              try (ReadStream stream =
                  reader.read(
                      nested.getReferences(),
                      List.of(
                          Request.whole(0, 0),
                          Request.whole(0, 1),
                          Request.whole(1, 2),
                          Request.whole(2, 3)))) {
                VectorSchemaRoot next;
                while ((next = stream.next(allocator)) != null) {
                  try (VectorSchemaRoot result = next) {
                    values.put(
                        ((UInt8Vector) result.getVector("request_index")).get(0),
                        ((LargeVarBinaryVector) result.getVector("data")).get(0));
                  }
                }
              }
              assertArrayEquals("first".getBytes(StandardCharsets.UTF_8), values.get(0L));
              assertArrayEquals(values.get(0L), values.get(1L));
              assertNull(values.get(2L));
              assertArrayEquals(new byte[0], values.get(3L));
            } else {
              assertTrue(media.isNull(0));
            }
          }
          batchIndex++;
        }
        assertEquals(2, batchIndex);
        assertEquals(1L, reader.getMetrics().get("payload_requests"));
        assertEquals(5L, reader.getMetrics().get("payload_bytes"));
        assertEquals(10L, reader.getMetrics().get("returned_logical_bytes"));
        assertEquals(1L, reader.getMetrics().get("duplicate_demands"));
        assertEquals(0L, reader.getMetrics().get("reserved_bytes"));
      }
    }
  }

  private static Dataset createNestedBlobDataset(BufferAllocator allocator, String path) {
    Field blob =
        new Field(
            "item",
            new FieldType(
                true,
                ArrowType.Struct.INSTANCE,
                null,
                Map.of(
                    "ARROW:extension:name",
                    "lance.blob.v2",
                    "lance-encoding:blob-inline-size-threshold",
                    "0",
                    "lance-encoding:blob-dedicated-size-threshold",
                    "1")),
            List.of(
                Field.nullable("data", ArrowType.LargeBinary.INSTANCE),
                Field.nullable("uri", ArrowType.Utf8.INSTANCE)));
    Schema schema =
        new Schema(
            List.of(
                new Field(
                    "media",
                    FieldType.nullable(ArrowType.Struct.INSTANCE),
                    List.of(
                        new Field(
                            "clips",
                            FieldType.nullable(ArrowType.List.INSTANCE),
                            List.of(blob))))));
    try (Dataset empty =
            Dataset.write()
                .allocator(allocator)
                .uri(path)
                .schema(schema)
                .dataStorageVersion(LanceConstants.FILE_FORMAT_VERSION_2_2)
                .execute();
        VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      StructVector media = (StructVector) root.getVector("media");
      ListVector clips = (ListVector) media.getChild("clips");
      StructVector blobs = (StructVector) clips.getDataVector();
      LargeVarBinaryVector data = (LargeVarBinaryVector) blobs.getChild("data");
      for (int row = 0; row < 4; row++) media.setIndexDefined(row);
      clips.startNewValue(0);
      clips.endValue(0, 3);
      clips.startNewValue(1);
      clips.endValue(1, 0);
      clips.startNewValue(2);
      clips.endValue(2, 0);
      clips.setNull(2);
      media.setNull(2);
      clips.startNewValue(3);
      clips.endValue(3, 1);
      blobs.setIndexDefined(0);
      data.setSafe(0, "first".getBytes(StandardCharsets.UTF_8));
      blobs.setNull(1);
      blobs.setIndexDefined(2);
      data.setSafe(2, new byte[0]);
      blobs.setIndexDefined(3);
      data.setSafe(3, "second".getBytes(StandardCharsets.UTF_8));
      root.setRowCount(4);
      List<FragmentMetadata> fragments =
          Fragment.create(
              path,
              allocator,
              root,
              new WriteParams.Builder()
                  .withMode(WriteParams.WriteMode.APPEND)
                  .withDataStorageVersion(LanceConstants.FILE_FORMAT_VERSION_2_2)
                  .withMaxRowsPerFile(2)
                  .build());
      try (Transaction transaction =
          new Transaction.Builder()
              .readVersion(empty.version())
              .operation(Append.builder().fragments(fragments).build())
              .build()) {
        return new CommitBuilder(empty).execute(transaction);
      }
    }
  }

  @Test
  void testSharedBlobReferenceConformance(@TempDir Path tempDir) throws Exception {
    JsonNode fixture =
        new ObjectMapper()
            .readTree(Files.readString(Path.of("../test_data/blob_reference_v1/conformance.json")));
    assertEquals(fixture.get("protocol_version").asInt(), BlobReferences.getProtocolVersion());
    try (RootAllocator allocator = new RootAllocator();
        Dataset dataset =
            createConformanceBlobDataset(
                allocator, tempDir.resolve("conformance").toString(), fixture)) {
      SourceContext source = SourceContext.fromDataset(dataset, "catalog/conformance", "workers");
      int fieldId = dataset.getLanceSchema().fields().get(0).getId();
      try (Capture capture = Capture.create(dataset, source, fieldId);
          LanceScanner scanner =
              dataset.newScan(new ScanOptions.Builder().withRowAddress(true).build());
          ArrowReader scanned = scanner.scanBatches();
          Reader reader = Reader.create(List.of(source), List.of(dataset), 1024, 1024)) {
        assertTrue(scanned.loadNextBatch());
        Batch refs = capture.capture(source, scanned.getVectorSchemaRoot(), "blob", allocator);
        refs = Batch.decode(refs.encode());
        List<Request> requests = new ArrayList<>();
        for (JsonNode request : fixture.get("requests")) {
          long index = Long.parseUnsignedLong(request.get("request_index").asText());
          JsonNode range = request.get("range");
          requests.add(
              range.isNull()
                  ? Request.whole(request.get("reference_index").asLong(), index)
                  : Request.range(
                      request.get("reference_index").asLong(),
                      index,
                      range.get(0).asLong(),
                      range.get(1).asLong()));
        }
        Map<Long, byte[]> results = new HashMap<>();
        try (ReadStream stream = reader.read(refs, requests)) {
          VectorSchemaRoot next;
          while ((next = stream.next(allocator)) != null) {
            try (VectorSchemaRoot result = next) {
              results.put(
                  ((UInt8Vector) result.getVector("request_index")).get(0),
                  ((LargeVarBinaryVector) result.getVector("data")).get(0));
            }
          }
        }
        for (JsonNode request : fixture.get("requests")) {
          long index = Long.parseUnsignedLong(request.get("request_index").asText());
          JsonNode expected = request.get("expected");
          assertTrue(results.containsKey(index));
          if (expected.isNull()) assertNull(results.get(index));
          else
            assertArrayEquals(
                expected.asText().getBytes(StandardCharsets.UTF_8), results.get(index));
        }
        Batch validated = refs;
        for (JsonNode range : fixture.get("invalid_ranges")) {
          long offset = Long.parseUnsignedLong(range.get("offset").asText());
          long length = Long.parseUnsignedLong(range.get("length").asText());
          IllegalArgumentException error =
              assertThrows(
                  IllegalArgumentException.class,
                  () -> reader.read(validated, List.of(Request.range(2, 0, offset, length))));
          assertTrue(error.getMessage().contains("range"));
        }
        assertEquals(1L, reader.getMetrics().get("duplicate_demands"));
        assertEquals(0L, reader.getMetrics().get("reserved_bytes"));
        assertFalse(scanned.loadNextBatch());
      }
    }
  }

  private static Dataset createConformanceBlobDataset(
      BufferAllocator allocator, String path, JsonNode fixture) {
    Field blob =
        new Field(
            "blob",
            new FieldType(
                true,
                ArrowType.Struct.INSTANCE,
                null,
                Map.of(
                    "ARROW:extension:name",
                    "lance.blob.v2",
                    "lance-encoding:blob-inline-size-threshold",
                    "4",
                    "lance-encoding:blob-dedicated-size-threshold",
                    "8")),
            List.of(
                Field.nullable("data", ArrowType.LargeBinary.INSTANCE),
                Field.nullable("uri", ArrowType.Utf8.INSTANCE)));
    Schema schema = new Schema(List.of(blob));
    try (Dataset empty =
            Dataset.write()
                .allocator(allocator)
                .uri(path)
                .schema(schema)
                .dataStorageVersion(LanceConstants.FILE_FORMAT_VERSION_2_2)
                .execute();
        VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      StructVector blobs = (StructVector) root.getVector("blob");
      LargeVarBinaryVector data = (LargeVarBinaryVector) blobs.getChild("data");
      JsonNode payloads = fixture.get("payloads");
      for (int index = 0; index < payloads.size(); index++) {
        if (payloads.get(index).isNull()) blobs.setNull(index);
        else {
          blobs.setIndexDefined(index);
          data.setSafe(index, payloads.get(index).asText().getBytes(StandardCharsets.UTF_8));
        }
      }
      root.setRowCount(payloads.size());
      List<FragmentMetadata> fragments =
          Fragment.create(
              path,
              allocator,
              root,
              new WriteParams.Builder()
                  .withMode(WriteParams.WriteMode.APPEND)
                  .withDataStorageVersion(LanceConstants.FILE_FORMAT_VERSION_2_2)
                  .build());
      try (Transaction transaction =
          new Transaction.Builder()
              .readVersion(empty.version())
              .operation(Append.builder().fragments(fragments).build())
              .build()) {
        return new CommitBuilder(empty).execute(transaction);
      }
    }
  }

  private static List<Batch> capturePortableBlobBatches(
      Dataset dataset, SourceContext source, BufferAllocator allocator) throws Exception {
    List<Batch> batches = new ArrayList<>();
    try (Capture capture = Capture.create(dataset, source, 1);
        LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder().columns(List.of("blobs")).withRowAddress(true).build());
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        batches.add(capture.capture(source, reader.getVectorSchemaRoot(), "blobs", allocator));
      }
    }
    return batches;
  }

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void testTakeBlobsSelectorsAcrossCompaction(boolean buffered, @TempDir Path tempDir)
      throws Exception {
    String path = tempDir.resolve("blob_selectors").toString();
    try (RootAllocator allocator = new RootAllocator();
        Dataset original = createBlobV2SelectorDataset(allocator, path);
        Dataset current = Dataset.open().allocator(allocator).uri(path).build()) {
      assertTrue(original.hasStableRowIds());
      assertEquals(2, original.getFragments().size());
      List<Long> originalAddresses = assertBlobSelectors(original, false, buffered);
      assertNotEquals(3L, originalAddresses.get(0).longValue());

      IllegalArgumentException missingId =
          assertThrows(
              IllegalArgumentException.class,
              () -> current.takeBlobs(Arrays.asList(3L, 6L, 1L), "blobs"));
      assertTrue(missingId.getMessage().contains("first missing row ID 6"));
      IllegalArgumentException addressAsId =
          assertThrows(
              IllegalArgumentException.class, () -> current.takeBlobs(originalAddresses, "blobs"));
      assertTrue(
          addressAsId.getMessage().contains("first missing row ID " + originalAddresses.get(0)));
      IllegalArgumentException missingIndex =
          assertThrows(
              IllegalArgumentException.class,
              () -> current.takeBlobsByIndices(Collections.singletonList(6L), "blobs"));
      assertTrue(missingIndex.getMessage().contains("non-existent fragment"));
      IllegalArgumentException missingAddress =
          assertThrows(
              IllegalArgumentException.class,
              () -> current.takeBlobsByAddresses(Collections.singletonList(-1L), "blobs"));
      assertTrue(missingAddress.getMessage().contains("non-existent fragment: 4294967295"));

      current.delete("id = 0");
      IllegalArgumentException deletedId =
          assertThrows(
              IllegalArgumentException.class,
              () -> current.takeBlobs(Arrays.asList(3L, 0L, 1L), "blobs"));
      assertTrue(deletedId.getMessage().contains("first missing row ID 0"));
      UnsupportedOperationException deletedAddress =
          assertThrows(
              UnsupportedOperationException.class,
              () -> current.takeBlobsByAddresses(Collections.singletonList(0L), "blobs"));
      assertTrue(deletedAddress.getMessage().contains("must not target deleted rows"));
      assertBlobSelectors(current, true, buffered);

      current.compact(CompactionOptions.builder().withTargetRowsPerFragment(6L).build());
      assertTrue(current.version() > original.version());
      assertEquals(1, current.getFragments().size());
      List<Long> compactedAddresses = assertBlobSelectors(current, true, buffered);
      assertNotEquals(originalAddresses, compactedAddresses);
      IllegalArgumentException staleAddress =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  current.takeBlobsByAddresses(
                      Collections.singletonList(originalAddresses.get(0)), "blobs"));
      assertTrue(staleAddress.getMessage().contains("non-existent fragment"));

      for (List<Long> addresses :
          List.of(
              List.of(0L, 1L << 32),
              List.of(1L << 32, 0L),
              originalAddresses,
              List.of(compactedAddresses.get(0), 0L))) {
        IllegalArgumentException staleAddresses =
            assertThrows(
                IllegalArgumentException.class,
                () -> {
                  if (buffered) {
                    current.takeBlobsByAddresses(addresses, "blobs", 16L);
                  } else {
                    current.takeBlobsByAddresses(addresses, "blobs");
                  }
                });
        assertTrue(staleAddresses.getMessage().contains("fragment"));
      }

      assertEquals(originalAddresses, assertBlobSelectors(original, false, buffered));
      try (Dataset reopened = current.checkoutVersion(original.version())) {
        assertEquals(originalAddresses, assertBlobSelectors(reopened, false, buffered));
        try (BlobFile blob = reopened.takeBlobsByAddresses(List.of(0L), "blobs").get(0)) {
          assertArrayEquals(new byte[] {0, 10}, blob.read());
        }
      }
    }
  }

  private static Dataset createBlobV2SelectorDataset(BufferAllocator allocator, String path) {
    Field blobField =
        new Field(
            "blobs",
            new FieldType(
                true,
                ArrowType.Struct.INSTANCE,
                null,
                Collections.singletonMap("ARROW:extension:name", "lance.blob.v2")),
            Arrays.asList(
                Field.nullable("data", ArrowType.LargeBinary.INSTANCE),
                Field.nullable("uri", ArrowType.Utf8.INSTANCE)));
    Schema schema =
        new Schema(Arrays.asList(Field.notNullable("id", new ArrowType.Int(32, true)), blobField));
    try (Dataset empty =
            Dataset.write()
                .allocator(allocator)
                .uri(path)
                .schema(schema)
                .enableStableRowIds(true)
                .dataStorageVersion(LanceConstants.FILE_FORMAT_VERSION_2_2)
                .execute();
        VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
      root.allocateNew();
      IntVector ids = (IntVector) root.getVector("id");
      StructVector blobs = (StructVector) root.getVector("blobs");
      LargeVarBinaryVector data = (LargeVarBinaryVector) blobs.getChild("data");
      for (int i = 0; i < 6; i++) {
        ids.setSafe(i, i);
        if (i % 3 == 1) {
          blobs.setNull(i);
        } else {
          blobs.setIndexDefined(i);
          data.setSafe(i, i % 3 == 2 ? new byte[0] : new byte[] {(byte) i, 10});
        }
      }
      root.setRowCount(6);
      List<FragmentMetadata> fragments =
          Fragment.create(
              path,
              allocator,
              root,
              new WriteParams.Builder()
                  .withMode(WriteParams.WriteMode.APPEND)
                  .withEnableStableRowIds(true)
                  .withDataStorageVersion(LanceConstants.FILE_FORMAT_VERSION_2_2)
                  .withMaxRowsPerFile(3)
                  .build());
      try (Transaction transaction =
          new Transaction.Builder()
              .readVersion(empty.version())
              .operation(Append.builder().fragments(fragments).build())
              .build()) {
        return new CommitBuilder(empty).execute(transaction);
      }
    }
  }

  private static List<Long> assertBlobSelectors(Dataset dataset, boolean deleted, boolean buffered)
      throws Exception {
    Map<Long, Long> addressesById = new HashMap<>();
    try (LanceScanner scanner =
            dataset.newScan(
                new ScanOptions.Builder()
                    .columns(List.of("id"))
                    .withRowId(true)
                    .withRowAddress(true)
                    .build());
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        IntVector ids = (IntVector) root.getVector("id");
        UInt8Vector rowIds = (UInt8Vector) root.getVector("_rowid");
        UInt8Vector rowAddresses = (UInt8Vector) root.getVector("_rowaddr");
        for (int i = 0; i < root.getRowCount(); i++) {
          assertEquals(ids.get(i), rowIds.get(i));
          addressesById.put(rowIds.get(i), rowAddresses.get(i));
        }
      }
    }
    List<Long> rowIds = Arrays.asList(3L, 1L, 5L, 3L, 4L, 2L, 1L, 5L);
    List<Long> rowAddresses = rowIds.stream().map(addressesById::get).collect(Collectors.toList());
    List<Long> rowIndices =
        rowIds.stream().map(id -> deleted ? id - 1 : id).collect(Collectors.toList());
    assertBlobSelection(dataset.takeBlobs(rowIds, "blobs"));
    assertBlobSelection(dataset.takeBlobsByIndices(rowIndices, "blobs"));
    assertBlobSelection(
        buffered
            ? dataset.takeBlobsByAddresses(rowAddresses, "blobs", 16L)
            : dataset.takeBlobsByAddresses(rowAddresses, "blobs"));
    return rowAddresses;
  }

  private static void assertBlobSelection(List<BlobFile> blobs) throws IOException {
    byte[][] expected = {
      new byte[] {3, 10},
      null,
      new byte[0],
      new byte[] {3, 10},
      null,
      new byte[0],
      null,
      new byte[0]
    };
    try {
      assertEquals(expected.length, blobs.size());
      for (int i = 0; i < expected.length; i++) {
        if (expected[i] == null) {
          assertNull(blobs.get(i));
        } else {
          assertNotNull(blobs.get(i));
          assertEquals(expected[i].length, blobs.get(i).size());
          assertArrayEquals(expected[i], blobs.get(i).read());
        }
      }
    } finally {
      for (BlobFile blob : blobs) {
        if (blob != null) {
          blob.close();
        }
      }
    }
  }

  @Test
  void testTakeBlobsByAddressesValidatesInputs(@TempDir Path tempDir) throws Exception {
    try (RootAllocator allocator = new RootAllocator();
        Dataset dataset =
            createBlobV2SelectorDataset(allocator, tempDir.resolve("blobs").toString())) {
      IllegalArgumentException nullList =
          assertThrows(
              IllegalArgumentException.class, () -> dataset.takeBlobsByAddresses(null, "blobs"));
      assertTrue(nullList.getMessage().contains("rowAddresses cannot be null or empty"));
      IllegalArgumentException emptyList =
          assertThrows(
              IllegalArgumentException.class,
              () -> dataset.takeBlobsByAddresses(Collections.emptyList(), "blobs"));
      assertTrue(emptyList.getMessage().contains("rowAddresses cannot be null or empty"));
      IllegalArgumentException nullAddress =
          assertThrows(
              IllegalArgumentException.class,
              () -> dataset.takeBlobsByAddresses(Arrays.asList(0L, null), "blobs"));
      assertTrue(nullAddress.getMessage().contains("rowAddresses[1] cannot be null"));
      for (String column : Arrays.asList(null, "")) {
        IllegalArgumentException invalidColumn =
            assertThrows(
                IllegalArgumentException.class,
                () -> dataset.takeBlobsByAddresses(List.of(0L), column));
        assertTrue(invalidColumn.getMessage().contains("column cannot be null or empty"));
      }
      IllegalArgumentException negativeBuffer =
          assertThrows(
              IllegalArgumentException.class,
              () -> dataset.takeBlobsByAddresses(Collections.emptyList(), "blobs", -1L));
      assertTrue(negativeBuffer.getMessage().contains("bufferSize must be non-negative"));
      try (BlobFile blob = dataset.takeBlobsByAddresses(List.of(0L), "blobs", 0L).get(0)) {
        assertArrayEquals(new byte[] {0, 10}, blob.read());
      }
      dataset.close();
      IllegalArgumentException closed =
          assertThrows(
              IllegalArgumentException.class,
              () -> dataset.takeBlobsByAddresses(List.of(0L), "blobs"));
      assertTrue(closed.getMessage().contains("Dataset is closed"));
    }
  }

  @Test
  void testReadZeroLengthBlob(@TempDir Path tempDir) throws Exception {
    String base = tempDir.resolve("testReadZeroLengthBlob").toString();
    try (Dataset ds = TestUtils.createBlobDataset(base, 128, 8)) {
      List<BlobFile> blobs = ds.takeBlobsByIndices(Collections.singletonList(0L), "blobs");
      assertEquals(1, blobs.size());
      BlobFile blobFile = blobs.get(0);
      assertEquals(0L, blobFile.size());
      assertArrayEquals(new byte[0], blobFile.read());
      blobFile.close();
    }
  }

  @Test
  void testTakeNullBlobPreservesSelection(@TempDir Path tempDir) throws Exception {
    String base = tempDir.resolve("testTakeNullBlobPreservesSelection").toString();
    try (Dataset ds = TestUtils.createBlobDataset(base, 128, 8)) {
      List<BlobFile> blobs = ds.takeBlobsByIndices(Collections.singletonList(15L), "blobs");
      assertEquals(1, blobs.size());
      assertNull(blobs.get(0));
    }
  }

  @Test
  void testReadLargeBlobAndRanges(@TempDir Path tempDir) throws Exception {
    String base = tempDir.resolve("testReadLargeBlobAndRanges").toString();
    try (Dataset ds = TestUtils.createBlobDataset(base, 128, 8)) {
      List<BlobFile> blobs = ds.takeBlobsByIndices(Collections.singletonList(1L), "blobs");
      BlobFile blobFile = blobs.get(0);
      long size = blobFile.size();
      assertTrue(size >= 1_000_000, "expected large blob size");
      byte[] data1 = blobFile.readUpTo(256);
      assertEquals(256, data1.length);
      assertEquals(256L, blobFile.tell());
      blobFile.seek(512);
      byte[] range = blobFile.readUpTo(256);
      assertEquals(256, range.length);
      assertEquals(768L, blobFile.tell());
      blobFile.seek(0);
      byte[] all = blobFile.read();
      assertEquals(size, all.length);
      assertArrayEquals(Arrays.copyOfRange(all, 0, 256), data1);
      assertArrayEquals(Arrays.copyOfRange(all, 512, 768), range);
      blobFile.close();
    }
  }

  @Test
  void testReadSmallBlobSequentialIntegrity(@TempDir Path tempDir) throws Exception {
    String base = tempDir.resolve("testReadSmallBlobSequentialIntegrity").toString();
    try (Dataset ds = TestUtils.createBlobDataset(base, 64, 4)) {
      List<BlobFile> blobs = ds.takeBlobsByIndices(Collections.singletonList(2L), "blobs");
      BlobFile blobFile = blobs.get(0);
      long size = blobFile.size();
      assertTrue(size >= 128, "expected small blob size");

      blobFile.seek(0);
      byte[] data1 = blobFile.readUpTo(64);
      byte[] data2 = blobFile.readUpTo(64);
      byte[] restData = blobFile.read();
      byte[] combined = new byte[data1.length + data2.length + restData.length];
      System.arraycopy(data1, 0, combined, 0, data1.length);
      System.arraycopy(data2, 0, combined, data2.length, data2.length);
      System.arraycopy(restData, 0, combined, data1.length + data2.length, restData.length);

      blobFile.seek(0);
      byte[] allData = blobFile.read();
      assertArrayEquals(allData, combined);
      blobFile.close();
    }
  }

  @Test
  void testReadUpToFillsAcrossReadAheadBoundary(@TempDir Path tempDir) throws Exception {
    String base = tempDir.resolve("testReadUpToFillsAcrossReadAheadBoundary").toString();
    try (Dataset ds = TestUtils.createBlobDataset(base, 64, 4)) {
      List<BlobFile> blobs = ds.takeBlobsByIndices(Collections.singletonList(2L), "blobs", 16L);
      BlobFile blobFile = blobs.get(0);
      byte[] first = blobFile.readUpTo(4);
      assertEquals(4, first.length);
      byte[] second = blobFile.readUpTo(20);
      assertEquals(20, second.length);
      assertEquals(24L, blobFile.tell());
      blobFile.seek(0);
      byte[] all = blobFile.read();
      assertArrayEquals(Arrays.copyOfRange(all, 0, 4), first);
      assertArrayEquals(Arrays.copyOfRange(all, 4, 24), second);
      blobFile.close();
    }
  }

  @Test
  void testSetReadBufferSizeZeroStillReadsPayload(@TempDir Path tempDir) throws Exception {
    String base = tempDir.resolve("testSetReadBufferSizeZeroStillReadsPayload").toString();
    try (Dataset ds = TestUtils.createBlobDataset(base, 64, 4)) {
      List<BlobFile> blobs = ds.takeBlobsByIndices(Collections.singletonList(2L), "blobs", 0L);
      BlobFile blobFile = blobs.get(0);
      byte[] first = blobFile.readUpTo(64);
      byte[] rest = blobFile.read();
      blobFile.seek(0);
      byte[] all = blobFile.read();
      byte[] combined = new byte[first.length + rest.length];
      System.arraycopy(first, 0, combined, 0, first.length);
      System.arraycopy(rest, 0, combined, first.length, rest.length);
      assertArrayEquals(all, combined);
      blobFile.close();
    }
  }

  @Test
  void testTakeBlobsRejectsNegativeBufferSizeForEmptySelection(@TempDir Path tempDir)
      throws Exception {
    String base =
        tempDir.resolve("testTakeBlobsRejectsNegativeBufferSizeForEmptySelection").toString();
    try (Dataset ds = TestUtils.createBlobDataset(base, 64, 4)) {
      assertThrows(
          IllegalArgumentException.class,
          () -> ds.takeBlobsByIndices(Collections.emptyList(), "blobs", -1L));
      assertThrows(
          IllegalArgumentException.class,
          () -> ds.takeBlobs(Collections.emptyList(), "blobs", -1L));
    }
  }

  @Test
  void testSetReadBufferSizeFailureClosesOpenedHandles(@TempDir Path tempDir) throws Exception {
    String base = tempDir.resolve("testSetReadBufferSizeFailureClosesOpenedHandles").toString();
    try (Dataset ds = TestUtils.createBlobDataset(base, 64, 4)) {
      List<BlobFile> blobs = ds.takeBlobsByIndices(Arrays.asList(0L, 1L), "blobs");
      blobs.get(1).close();
      assertThrows(RuntimeException.class, () -> Dataset.setBlobReadBufferSize(blobs, 1024L));
      assertThrows(RuntimeException.class, () -> blobs.get(0).readUpTo(1));
    }
  }

  @Test
  public void testIndexStatistics(@TempDir Path tempDir) throws Exception {
    Path datasetPath = tempDir.resolve("testIndexStatistics");

    try (TestVectorDataset vectorDataset = new TestVectorDataset(datasetPath)) {
      try (Dataset dataset = vectorDataset.create()) {
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        dataset.createIndex(
            Collections.singletonList("i"),
            IndexType.BTREE,
            Optional.of(TestVectorDataset.indexName),
            indexParams,
            true);

        Map<String, Object> stats = dataset.getIndexStatistics(TestVectorDataset.indexName);
        assertNotNull(stats, "Index statistics JSON should not be null");
        assertFalse(stats.isEmpty(), "Index statistics JSON should not be empty");

        assertEquals(
            TestVectorDataset.indexName,
            stats.get("name"),
            "Index statistics should contain the index name");
        assertEquals(
            "BTree",
            stats.get("index_type"),
            "Index statistics should contain index_type information");
      }
    }
  }

  @Test
  public void testDescribeIndicesByName(@TempDir Path tempDir) throws Exception {
    Path datasetPath = tempDir.resolve("testDescribeIndicesByName");

    try (TestVectorDataset vectorDataset = new TestVectorDataset(datasetPath)) {
      try (Dataset dataset = vectorDataset.create()) {
        dataset.createIndex(
            Collections.singletonList("i"),
            IndexType.BTREE,
            Optional.of("index1"),
            IndexParams.builder().setScalarIndexParams(BTreeIndexParams.builder().build()).build(),
            true);

        dataset.createIndex(
            Collections.singletonList("s"),
            IndexType.NGRAM,
            Optional.of("index2"),
            IndexParams.builder().setScalarIndexParams(NGramIndexParams.builder().build()).build(),
            true);

        IndexCriteria criteria = new IndexCriteria.Builder().hasName("index1").build();

        List<IndexDescription> descriptions = dataset.describeIndices(criteria);
        assertEquals(1, descriptions.size(), "Expected exactly one matching index");

        IndexDescription desc = descriptions.get(0);
        assertEquals("index1", desc.getName());
        assertTrue(desc.getRowsIndexed() > 0, "rowsIndexed should be positive");
        assertNotNull(desc.getMetadata(), "Metadata list should not be null");
        assertFalse(desc.getMetadata().isEmpty(), "Metadata list should not be empty");
        assertEquals(
            desc.getMetadata(), desc.getSegments(), "segments alias should match metadata");
        assertNotNull(desc.getDetailsJson(), "Details JSON should not be null");

        assertEquals(1, desc.getSegments().size(), "Expected exactly one physical segment");
        assertEquals("index1", desc.getSegments().get(0).name());
        assertEquals(
            Collections.emptyList(),
            desc.getSegments().get(0).coveringFields(),
            "no covering columns are declared yet");
        assertTrue(
            desc.getSegments().get(0).getSizeBytes().orElse(0L) > 0,
            "segment size should be positive");
        assertEquals(
            desc.getSegments().get(0).getSizeBytes(),
            desc.getTotalSizeBytes(),
            "single-segment size should equal the logical index size");

        descriptions = dataset.describeIndices();
        assertEquals(2, descriptions.size(), "Expected exactly one matching index");
        for (IndexDescription indexDesc : descriptions) {
          assertTrue(indexDesc.getRowsIndexed() > 0, "rowsIndexed should be positive");
          assertNotNull(indexDesc.getMetadata(), "Metadata list should not be null");
          assertFalse(indexDesc.getMetadata().isEmpty(), "Metadata list should not be empty");
          assertEquals(
              indexDesc.getMetadata(),
              indexDesc.getSegments(),
              "segments alias should match metadata");
          assertNotNull(indexDesc.getDetailsJson(), "Details JSON should not be null");
          assertTrue(
              indexDesc.getTotalSizeBytes().orElse(0L) > 0, "total index size should be positive");
        }
      }
    }
  }

  @Test
  void testDropIndex(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("drop_index").toString();

    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      // 1. write two fragments
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        List<Fragment> fragments = dataset.getFragments();
        assertEquals(2, dataset.getFragments().size());

        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{\"zone_size\": 2048}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();

        // 2. partially create index
        List<Index> segments = new ArrayList<>();
        for (Fragment fragment : fragments) {
          segments.add(
              dataset.createIndex(
                  IndexOptions.builder(
                          Collections.singletonList("name"), IndexType.BTREE, indexParams)
                      .withIndexName("test_index")
                      .withFragmentIds(Collections.singletonList(fragment.getId()))
                      .build()));
        }

        // then no index should have been created
        assertFalse(
            dataset.listIndexes().contains("test_index"),
            "Partially created index should not present");

        // 3. commit the index
        List<Index> committed = dataset.commitExistingIndexSegments("test_index", "name", segments);
        assertEquals(2, committed.size());
        assertTrue(dataset.listIndexes().contains("test_index"));

        List<Index> indexes = dataset.getIndexes();
        assertTrue(indexes.stream().anyMatch(idx -> idx.name().equals("test_index")));

        dataset.dropIndex("test_index");

        List<String> indexNamesAfterDrop = dataset.listIndexes();
        assertFalse(indexNamesAfterDrop.contains("test_index"));
      }
    }
  }

  @Test
  void testDropIndexNonExistent(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("drop_index_nonexistent").toString();

    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", new ArrowType.Utf8())),
            null);

    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      try (Dataset dataset =
          Dataset.create(allocator, datasetPath, schema, new WriteParams.Builder().build())) {
        assertEquals(1, dataset.version());

        assertThrows(RuntimeException.class, () -> dataset.dropIndex("nonexistent"));

        assertEquals(1, dataset.version());
        assertTrue(dataset.listIndexes().isEmpty());
        assertTrue(dataset.getIndexes().isEmpty());
      }
    }
  }
}
