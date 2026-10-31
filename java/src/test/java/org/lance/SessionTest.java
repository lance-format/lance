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

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SessionTest {

  @Test
  void testCreateSessionWithDefaults() {
    try (Session session = Session.builder().build()) {
      assertNotNull(session);
      assertFalse(session.isClosed());
      assertTrue(session.sizeBytes() >= 0);
    }
  }

  @Test
  void testCreateSessionWithCustomCacheSizes() {
    long indexCacheSize = 512L * 1024 * 1024; // 512 MiB
    long metadataCacheSize = 128L * 1024 * 1024; // 128 MiB

    try (Session session =
        Session.builder()
            .indexCacheSizeBytes(indexCacheSize)
            .metadataCacheSizeBytes(metadataCacheSize)
            .build()) {
      assertNotNull(session);
      assertFalse(session.isClosed());
      assertTrue(session.sizeBytes() >= 0);
    }
  }

  @Test
  void testCreateSessionWithPartialCustomCacheSizes() {
    // Only set index cache size, metadata should use default
    try (Session session = Session.builder().indexCacheSizeBytes(512L * 1024 * 1024).build()) {
      assertNotNull(session);
      assertFalse(session.isClosed());
    }

    // Only set metadata cache size, index should use default
    try (Session session = Session.builder().metadataCacheSizeBytes(128L * 1024 * 1024).build()) {
      assertNotNull(session);
      assertFalse(session.isClosed());
    }
  }

  @Test
  void testCreateSessionWithCacheBackendUris() {
    try (Session session =
        Session.builder()
            .indexCacheBackend("moka://?capacity=1048576")
            .metadataCacheBackend("moka://?capacity=524288")
            .build()) {
      assertNotNull(session);
      assertFalse(session.isClosed());
      assertEquals(0, session.metadataCacheStats().getNumEntries());
    }
  }

  @Test
  void testCreateSessionWithStructuredCacheBackendConfigs() {
    CacheBackendConfig indexBackend =
        CacheBackendConfig.builder("moka").option("capacity", "1048576").build();
    CacheBackendConfig metadataBackend =
        CacheBackendConfig.builder("moka")
            .options(Collections.singletonMap("capacity", "524288"))
            .build();

    assertEquals("moka", indexBackend.getKind());
    assertEquals(Collections.singletonMap("capacity", "1048576"), indexBackend.getOptions());
    assertThrows(
        UnsupportedOperationException.class, () -> indexBackend.getOptions().put("capacity", "1"));

    try (Session session =
        Session.builder()
            .indexCacheBackend(indexBackend)
            .metadataCacheBackend(metadataBackend)
            .build()) {
      assertNotNull(session);
      assertFalse(session.isClosed());
    }
  }

  @Test
  void testCacheBackendReplacesPreviousDescriptorForSameTier() {
    CacheBackendConfig structuredBackend =
        CacheBackendConfig.builder("moka").option("capacity", "1048576").build();

    try (Session session =
        Session.builder()
            .indexCacheBackend("missing://")
            .indexCacheBackend(structuredBackend)
            .build()) {
      assertNotNull(session);
    }

    try (Session session =
        Session.builder()
            .metadataCacheBackend(structuredBackend)
            .metadataCacheBackend("moka://?capacity=1048576")
            .build()) {
      assertNotNull(session);
    }
  }

  @Test
  void testCacheBackendRejectsSizeAndBackend() {
    IllegalArgumentException indexError =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Session.builder()
                    .indexCacheSizeBytes(1024)
                    .indexCacheBackend("moka://?capacity=1048576")
                    .build());
    assertTrue(
        indexError
            .getMessage()
            .contains("indexCacheSizeBytes and indexCacheBackend are mutually exclusive"));

    IllegalArgumentException metadataError =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Session.builder()
                    .metadataCacheBackend(
                        CacheBackendConfig.builder("moka").option("capacity", "1048576").build())
                    .metadataCacheSizeBytes(1024)
                    .build());
    assertTrue(
        metadataError
            .getMessage()
            .contains("metadataCacheSizeBytes and metadataCacheBackend are mutually exclusive"));
  }

  @Test
  void testCacheBackendRejectsUnknownKind() {
    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () -> Session.builder().indexCacheBackend("missing://").build());
    assertTrue(error.getMessage().contains("unknown cache backend kind"));
  }

  @Test
  void testCacheBackendRejectsInvalidMokaConfig() {
    IllegalArgumentException missingCapacity =
        assertThrows(
            IllegalArgumentException.class,
            () -> Session.builder().indexCacheBackend("moka://").build());
    assertTrue(missingCapacity.getMessage().contains("capacity is required"));

    IllegalArgumentException unknownOption =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Session.builder()
                    .metadataCacheBackend(
                        CacheBackendConfig.builder("moka").option("unknown", "value").build())
                    .build());
    assertTrue(unknownOption.getMessage().contains("unknown option"));
  }

  @Test
  void testCacheBackendConfigValidatesInputs() {
    assertThrows(NullPointerException.class, () -> CacheBackendConfig.builder(null));
    assertThrows(IllegalArgumentException.class, () -> CacheBackendConfig.builder(""));
    assertThrows(
        NullPointerException.class,
        () -> CacheBackendConfig.builder("moka").options((Map<String, String>) null));
    assertThrows(
        NullPointerException.class,
        () -> CacheBackendConfig.builder("moka").option("capacity", null));
  }

  @Test
  void testSessionClose() {
    Session session = Session.builder().build();
    assertFalse(session.isClosed());

    session.close();
    assertTrue(session.isClosed());

    // Calling close again should be safe
    session.close();
    assertTrue(session.isClosed());
  }

  @Test
  void testSessionSizeBytesAfterClose() {
    Session session = Session.builder().build();
    session.close();

    assertThrows(IllegalArgumentException.class, session::sizeBytes);
  }

  @Test
  void testSessionIsSameAs() {
    try (Session session1 = Session.builder().build();
        Session session2 = Session.builder().build()) {
      // Same session should be equal to itself
      assertTrue(session1.isSameAs(session1));
      assertTrue(session2.isSameAs(session2));

      // Different sessions should not be equal
      assertFalse(session1.isSameAs(session2));
      assertFalse(session2.isSameAs(session1));

      // Null comparison
      assertFalse(session1.isSameAs(null));
    }
  }

  @Test
  void testDatasetSharesSession(@TempDir Path tempDir) {
    String datasetPath1 = tempDir.resolve("dataset1").toString();
    String datasetPath2 = tempDir.resolve("dataset2").toString();

    try (BufferAllocator allocator = new RootAllocator();
        Session session = Session.builder().build()) {
      // Create first dataset with session
      TestUtils.SimpleTestDataset testDataset1 =
          new TestUtils.SimpleTestDataset(allocator, datasetPath1);
      try (Dataset ds1 = testDataset1.createEmptyDataset()) {
        // Now reopen with shared session
        try (Dataset ds1WithSession =
            Dataset.open().allocator(allocator).uri(datasetPath1).session(session).build()) {

          // Create second dataset
          TestUtils.SimpleTestDataset testDataset2 =
              new TestUtils.SimpleTestDataset(allocator, datasetPath2);
          try (Dataset ds2 = testDataset2.createEmptyDataset()) {
            // Reopen with shared session
            try (Dataset ds2WithSession =
                Dataset.open().allocator(allocator).uri(datasetPath2).session(session).build()) {

              // Both datasets should share the same session
              Session session1 = ds1WithSession.session();
              Session session2 = ds2WithSession.session();

              assertNotNull(session1);
              assertNotNull(session2);
              assertTrue(session1.isSameAs(session2));
              assertTrue(session1.isSameAs(session));
            }
          }
        }
      }
    }
  }

  @Test
  void testDatasetSessionFromReadOptions(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("dataset_session_options").toString();

    try (BufferAllocator allocator = new RootAllocator();
        Session session = Session.builder().build()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      try (Dataset ds = testDataset.createEmptyDataset()) {
        // Reopen with session in ReadOptions
        ReadOptions options = new ReadOptions.Builder().setSession(session).build();

        try (Dataset dsWithSession =
            Dataset.open().allocator(allocator).uri(datasetPath).readOptions(options).build()) {

          Session datasetSession = dsWithSession.session();
          assertNotNull(datasetSession);
          assertTrue(datasetSession.isSameAs(session));
        }
      }
    }
  }

  @Test
  void testSessionPersistsAfterDatasetClose(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("dataset_session_persist").toString();

    try (BufferAllocator allocator = new RootAllocator();
        Session session = Session.builder().build()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      // Open and close dataset with session
      Dataset ds = Dataset.open().allocator(allocator).uri(datasetPath).session(session).build();
      ds.close();

      // Session should still be open and usable
      assertFalse(session.isClosed());
      assertTrue(session.sizeBytes() >= 0);

      // Can open another dataset with the same session
      try (Dataset ds2 =
          Dataset.open().allocator(allocator).uri(datasetPath).session(session).build()) {
        assertNotNull(ds2.session());
        assertTrue(ds2.session().isSameAs(session));
      }
    }
  }

  @Test
  void testInternalSessionClosedWithDataset(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("dataset_internal_session").toString();

    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      // Open dataset WITHOUT providing a session - internal session will be created
      Dataset ds = Dataset.open().allocator(allocator).uri(datasetPath).build();

      // Get the internal session
      Session internalSession = ds.session();
      assertNotNull(internalSession);
      assertFalse(internalSession.isClosed());

      // Close the dataset - internal session should be closed too
      ds.close();

      // The internal session should now be closed
      assertTrue(internalSession.isClosed());
    }
  }

  @Test
  void testUserProvidedSessionNotClosedWithDataset(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("dataset_user_session").toString();

    try (BufferAllocator allocator = new RootAllocator();
        Session userSession = Session.builder().build()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      // Open dataset WITH user-provided session
      Dataset ds =
          Dataset.open().allocator(allocator).uri(datasetPath).session(userSession).build();

      // Get the session from dataset - should be the same as user-provided
      Session datasetSession = ds.session();
      assertTrue(datasetSession.isSameAs(userSession));

      // Close the dataset
      ds.close();

      // User-provided session should NOT be closed
      assertFalse(userSession.isClosed());
      assertTrue(userSession.sizeBytes() >= 0);
    }
  }

  @Test
  void testCheckedOutDatasetsShareInternalSession(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("dataset_checkout_session").toString();

    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      try (Dataset source = Dataset.open().allocator(allocator).uri(datasetPath).build()) {
        source.tags().create("version-one", Ref.ofMain(1));
        Session sourceSession = source.session();

        try (Dataset byVersion = source.checkoutVersion(1);
            Dataset byTag = source.checkoutTag("version-one");
            Dataset byRef = source.checkout(Ref.ofMain(1))) {
          assertNotNull(byVersion.session());
          assertNotNull(byTag.session());
          assertNotNull(byRef.session());
          assertTrue(byVersion.session().isSameAs(sourceSession));
          assertTrue(byTag.session().isSameAs(sourceSession));
          assertTrue(byRef.session().isSameAs(sourceSession));

          source.close();
          assertTrue(sourceSession.isClosed());
          assertFalse(byVersion.session().isClosed());
          assertFalse(byTag.session().isClosed());
          assertFalse(byRef.session().isClosed());
        }
      }
    }
  }

  @Test
  void testSessionToString() {
    try (Session session = Session.builder().build()) {
      String str = session.toString();
      assertNotNull(str);
      assertTrue(str.startsWith("Session("));
    }

    Session closedSession = Session.builder().build();
    closedSession.close();
    assertEquals("Session(closed)", closedSession.toString());
  }

  @Test
  void testMetadataCacheStats(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve("dataset_metadata_cache_stats").toString();

    try (BufferAllocator allocator = new RootAllocator();
        Session session = Session.builder().build()) {
      CacheStats initialStats = session.metadataCacheStats();
      assertEquals(0, initialStats.getHits());
      assertEquals(0, initialStats.getMisses());
      assertEquals(0, initialStats.getNumEntries());
      assertEquals(0, initialStats.getSizeBytes());
      assertEquals(0.0, initialStats.getHitRatio());

      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      // First open populates the metadata cache, producing misses
      try (Dataset ds =
          Dataset.open().allocator(allocator).uri(datasetPath).session(session).build()) {
        ds.countRows();
      }
      CacheStats statsAfterFirstOpen = session.metadataCacheStats();
      assertTrue(statsAfterFirstOpen.getMisses() > 0);
      assertTrue(statsAfterFirstOpen.getNumEntries() > 0);
      assertTrue(statsAfterFirstOpen.getSizeBytes() > 0);

      // Reopening the same dataset should hit the shared metadata cache
      try (Dataset ds =
          Dataset.open().allocator(allocator).uri(datasetPath).session(session).build()) {
        ds.countRows();
      }
      CacheStats statsAfterSecondOpen = session.metadataCacheStats();
      assertTrue(statsAfterSecondOpen.getHits() > statsAfterFirstOpen.getHits());
      assertTrue(statsAfterSecondOpen.getHitRatio() > 0.0);

      // Stats are not accessible once the session is closed (close is idempotent,
      // so the implicit close from try-with-resources remains safe)
      session.close();
      assertThrows(IllegalArgumentException.class, session::metadataCacheStats);
    }
  }

  @Test
  void testInvalidCacheSizes() {
    assertThrows(
        IllegalArgumentException.class, () -> Session.builder().indexCacheSizeBytes(-1).build());
    assertThrows(
        IllegalArgumentException.class, () -> Session.builder().metadataCacheSizeBytes(-1).build());
    assertThrows(
        IllegalArgumentException.class,
        () -> Session.builder().indexCacheSizeBytes(-1).metadataCacheSizeBytes(-1).build());
  }
}
