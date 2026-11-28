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
package org.lance.namespace;

import org.lance.Dataset;
import org.lance.LockManager;
import org.lance.namespace.model.CreateTableRequest;
import org.lance.namespace.model.CreateTableVersionRequest;
import org.lance.namespace.model.CreateTableVersionResponse;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for CustomNamespace implementation that wraps DirectoryNamespace.
 *
 * <p>This test class extends DirectoryNamespaceTest to verify that all tests pass when using a
 * CustomNamespace wrapper around DirectoryNamespace. This validates that the Java-Rust binding
 * works correctly for custom namespace implementations.
 */
public class CustomNamespaceTest extends DirectoryNamespaceTest {

  @Override
  protected LanceNamespace wrapNamespace(DirectoryNamespace inner) {
    return new CustomNamespace(inner);
  }

  @Test
  void testCustomNamespaceId() {
    String namespaceId = namespaceClient.namespaceId();
    assertNotNull(namespaceId);
    assertTrue(
        namespaceId.startsWith("CustomNamespace["),
        "namespaceId should start with 'CustomNamespace[', got: " + namespaceId);
    assertTrue(
        namespaceId.contains("DirectoryNamespace"),
        "namespaceId should contain 'DirectoryNamespace', got: " + namespaceId);
  }

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void testConfigCommitFinishingAfterLifecycleChange(boolean close, @TempDir Path root)
      throws Exception {
    CountDownLatch published = new CountDownLatch(1);
    CountDownLatch resume = new CountDownLatch(1);
    try (DirectoryNamespace inner = createManagedVersioningNamespace(root)) {
      List<String> id = Collections.singletonList("config");
      inner.createTable(new CreateTableRequest().id(id), createTestTableData());
      CustomNamespace delayed =
          new CustomNamespace(inner) {
            @Override
            public CreateTableVersionResponse createTableVersion(
                CreateTableVersionRequest request) {
              CreateTableVersionResponse result = super.createTableVersion(request);
              published.countDown();
              try {
                if (!resume.await(5, TimeUnit.SECONDS)) {
                  throw new IllegalStateException("Timed out returning the committed version");
                }
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(e);
              }
              return result;
            }
          };
      try (Dataset ds = Dataset.open().namespaceClient(delayed).tableId(id).build()) {
        FutureTask<Void> update =
            new FutureTask<>(() -> ds.updateConfig(Map.of("snapshot", "first")), null);
        FutureTask<Void> lifecycle = new FutureTask<>(close ? ds::close : ds::checkoutLatest, null);
        Thread updater = new Thread(update);
        Thread changer = new Thread(lifecycle);
        // Hold a shared lease until both writes queue, so their order is independent of scheduling.
        try (LockManager.ReadLock pin = ds.acquireReadLock()) {
          updater.start();
          try {
            assertTrue(published.await(5, TimeUnit.SECONDS));
            if (!close) {
              try (Dataset other = Dataset.open().namespaceClient(inner).tableId(id).build()) {
                other.updateConfig(Map.of("snapshot", "latest"));
              }
            }
            changer.start();
            awaitLifecycleWrite(changer);
          } finally {
            resume.countDown();
          }
          awaitLifecycleWrite(updater);
        }
        lifecycle.get(5, TimeUnit.SECONDS);
        update.get(5, TimeUnit.SECONDS);
        if (close) {
          assertTrue(ds.closed());
        } else {
          assertEquals("latest", ds.getConfig().get("snapshot"));
        }
      }
    }
  }

  private static void awaitLifecycleWrite(Thread thread) throws InterruptedException {
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
    while (System.nanoTime() < deadline) {
      if (thread.getState() == Thread.State.WAITING
          && Arrays.stream(thread.getStackTrace())
              .anyMatch(
                  frame -> frame.getClassName().equals(LockManager.WriteLock.class.getName()))) {
        return;
      }
      Thread.sleep(1);
    }
    fail("Lifecycle write did not queue: " + thread.getState());
  }

  @Test
  void testCustomNamespaceInnerAccess() {
    assertTrue(
        namespaceClient instanceof CustomNamespace, "namespaceClient should be a CustomNamespace");
    CustomNamespace customNs = (CustomNamespace) namespaceClient;
    assertNotNull(customNs.getInner(), "inner namespace should not be null");
    assertSame(
        innerNamespaceClient, customNs.getInner(), "inner namespace should be the same instance");
  }
}
