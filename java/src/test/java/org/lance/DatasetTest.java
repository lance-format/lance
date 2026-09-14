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

import org.lance.compaction.CompactionOptions;
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

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.UInt8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.ClosedChannelException;
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
                }
              }
            }
          }
        }
      }
    }
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
