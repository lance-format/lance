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

import org.lance.index.IndexOptions;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.scalar.ScalarIndexParams;
import org.lance.ipc.AsyncScanner;
import org.lance.ipc.MaterializationStyle;
import org.lance.ipc.ScanOptions;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Example tests demonstrating AsyncScanner usage with CompletableFuture-based API.
 *
 * <p>AsyncScanner provides non-blocking scan operations that prevent thread starvation in Java
 * query engines like Presto/Trino.
 */
public class AsyncScannerTest {
  private static Dataset dataset;

  @BeforeAll
  static void setup() {}

  @AfterAll
  static void tearDown() {
    if (dataset != null) {
      dataset.close();
    }
  }

  @Test
  @SuppressWarnings("unchecked")
  void testNativeScanFailureUsesLanceException() throws Exception {
    Constructor<AsyncScanner> constructor = AsyncScanner.class.getDeclaredConstructor();
    constructor.setAccessible(true);
    AsyncScanner scanner = constructor.newInstance();

    Field pendingTasksField = AsyncScanner.class.getDeclaredField("pendingTasks");
    pendingTasksField.setAccessible(true);
    ConcurrentHashMap<Long, CompletableFuture<Long>> pendingTasks =
        (ConcurrentHashMap<Long, CompletableFuture<Long>>) pendingTasksField.get(scanner);
    CompletableFuture<Long> pendingTask = new CompletableFuture<>();
    pendingTasks.put(1L, pendingTask);

    Method failTask = AsyncScanner.class.getDeclaredMethod("failTask", long.class, String.class);
    failTask.setAccessible(true);
    failTask.invoke(scanner, 1L, "scan I/O failed");

    CompletionException failure = assertThrows(CompletionException.class, pendingTask::join);
    assertEquals(LanceException.class, failure.getCause().getClass());
    assertEquals("scan I/O failed", failure.getCause().getMessage());
  }

  /**
   * Example 1: Basic async scan with CompletableFuture.
   *
   * <p>This shows the simplest usage - create an async scanner and wait for results.
   */
  @Test
  void testBasicAsyncScan(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_basic").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;

      try (Dataset dataset = testDataset.write(1, totalRows)) {
        // Create AsyncScanner with same options as LanceScanner
        ScanOptions options = new ScanOptions.Builder().batchSize(20L).build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
          // Start async scan - returns CompletableFuture<ArrowReader>
          CompletableFuture<ArrowReader> future = scanner.scanBatchesAsync();

          // Wait for result (blocks current thread, but doesn't block Rust I/O threads)
          ArrowReader reader = future.get(10, TimeUnit.SECONDS);
          assertNotNull(reader);

          // Read all batches
          int rowCount = 0;
          while (reader.loadNextBatch()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            rowCount += root.getRowCount();
          }

          assertEquals(totalRows, rowCount, "Should read all rows");
          reader.close();
        }
      }
    }
  }

  /**
   * Example 2: Async scan with filter.
   *
   * <p>Shows how to use async scanner with SQL-like filters.
   */
  @Test
  void testAsyncScanWithFilter(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_filter").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      try (Dataset dataset = testDataset.write(1, 40)) {
        // Scan with filter - only rows where id < 20
        ScanOptions options = new ScanOptions.Builder().filter("id < 20").build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
          CompletableFuture<ArrowReader> future = scanner.scanBatchesAsync();

          ArrowReader reader = future.get(10, TimeUnit.SECONDS);
          int rowCount = 0;
          while (reader.loadNextBatch()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            rowCount += root.getRowCount();
          }

          assertEquals(20, rowCount, "Should read only filtered rows");
          reader.close();
        }
      }
    }
  }

  @Test
  void testAsyncScanWithPerformanceOptions(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_performance_options").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      try (Dataset dataset = testDataset.write(1, 40)) {
        ScanOptions options =
            new ScanOptions.Builder()
                .filter("id < 20")
                .batchSize(10)
                .batchSizeBytes(1024)
                .ioBufferSize(1024 * 1024)
                .batchReadahead(2)
                .fragmentReadahead(2)
                .scanInOrder(false)
                .lateMaterialization(MaterializationStyle.allEarly())
                .build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator);
            ArrowReader reader = scanner.scanBatchesAsync().get(10, TimeUnit.SECONDS)) {
          assertEquals(20, countRows(reader));
        }
      }
    }
  }

  @Test
  void testFastSearchSkipsUnindexedFragments(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_fast_search_scalar_index").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      try (Dataset dataset = testDataset.write(1, 100)) {
        ScalarIndexParams scalarParams = ScalarIndexParams.create("btree", "{}");
        IndexParams indexParams = IndexParams.builder().setScalarIndexParams(scalarParams).build();
        IndexOptions indexOptions =
            IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, indexParams)
                .withIndexName("id_btree_index")
                .replace(true)
                .build();
        dataset.createIndex(indexOptions);

        FragmentMetadata metadata = testDataset.createNewFragment(10);
        FragmentOperation.Append appendOp =
            new FragmentOperation.Append(Collections.singletonList(metadata));
        try (Dataset appended =
            Dataset.commit(allocator, datasetPath, appendOp, Optional.of(dataset.version()))) {
          ScanOptions normalOptions = new ScanOptions.Builder().filter("id < 5").build();
          try (AsyncScanner scanner = AsyncScanner.create(appended, normalOptions, allocator)) {
            ArrowReader reader = scanner.scanBatchesAsync().get(10, TimeUnit.SECONDS);
            assertEquals(10, countRows(reader));
            reader.close();
          }

          ScanOptions fastOptions =
              new ScanOptions.Builder().filter("id < 5").fastSearch(true).build();
          try (AsyncScanner scanner = AsyncScanner.create(appended, fastOptions, allocator)) {
            ArrowReader reader = scanner.scanBatchesAsync().get(10, TimeUnit.SECONDS);
            assertEquals(5, countRows(reader));
            reader.close();
          }
        }
      }
    }
  }

  private static int countRows(ArrowReader reader) throws Exception {
    int rowCount = 0;
    while (reader.loadNextBatch()) {
      rowCount += reader.getVectorSchemaRoot().getRowCount();
    }
    return rowCount;
  }

  @Test
  void testIncludeDeletedRowsAsync(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_include_deleted_rows").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 10)) {
        assertEquals(10, dataset.countRows());

        // Delete half the rows
        dataset.delete("id >= 5");
        assertEquals(5, dataset.countRows());

        // Async scan without includeDeletedRows — should only see live rows
        ScanOptions defaultOptions = new ScanOptions.Builder().batchSize(20L).build();
        try (AsyncScanner scanner = AsyncScanner.create(dataset, defaultOptions, allocator)) {
          ArrowReader reader = scanner.scanBatchesAsync().get(10, TimeUnit.SECONDS);
          assertEquals(5, countRows(reader), "default async scan: should exclude deleted rows");
          reader.close();
        }

        // Async scan with includeDeletedRows=true — should see all rows
        ScanOptions includeDeletedOptions =
            new ScanOptions.Builder()
                .batchSize(20L)
                .withRowId(true) // required by includeDeletedRows
                .includeDeletedRows(true)
                .build();
        try (AsyncScanner scanner =
            AsyncScanner.create(dataset, includeDeletedOptions, allocator)) {
          ArrowReader reader = scanner.scanBatchesAsync().get(10, TimeUnit.SECONDS);
          assertEquals(
              10, countRows(reader), "includeDeletedRows async: should include deleted rows");
          reader.close();
        }
      }
    }
  }

  @Test
  void testStrictBatchSizeAsync(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_strict_batch_size").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      try (Dataset dataset = testDataset.write(1, 25)) {
        int batchSize = 10;

        ScanOptions strictOptions =
            new ScanOptions.Builder().batchSize(batchSize).strictBatchSize(true).build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, strictOptions, allocator)) {
          ArrowReader reader = scanner.scanBatchesAsync().get(10, TimeUnit.SECONDS);
          int totalRows = 0;
          while (reader.loadNextBatch()) {
            int rows = reader.getVectorSchemaRoot().getRowCount();
            assertTrue(
                rows <= batchSize, "strict async: batch " + rows + " should be <= " + batchSize);
            totalRows += rows;
          }
          assertEquals(25, totalRows, "strictBatchSize async: should read all rows");
          reader.close();
        }
      }
    }
  }

  /**
   * Example 3: Multiple concurrent async scans.
   *
   * <p>Shows how to run multiple scans in parallel without blocking threads. This is the key
   * benefit for query engines like Presto/Trino.
   */
  @Test
  void testConcurrentAsyncScans(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_concurrent").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 100;

      try (Dataset dataset = testDataset.write(1, totalRows)) {
        // Create 5 concurrent scans with different filters
        List<CompletableFuture<Integer>> futures = new ArrayList<>();

        for (int i = 0; i < 5; i++) {
          final int rangeStart = i * 20;
          final int rangeEnd = rangeStart + 20;
          String filter = String.format("id >= %d AND id < %d", rangeStart, rangeEnd);

          ScanOptions options = new ScanOptions.Builder().filter(filter).build();

          AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator);

          // Chain async operations: scan -> read -> count rows -> cleanup
          CompletableFuture<Integer> future =
              scanner
                  .scanBatchesAsync()
                  .thenApply(
                      reader -> {
                        try {
                          int count = 0;
                          while (reader.loadNextBatch()) {
                            count += reader.getVectorSchemaRoot().getRowCount();
                          }
                          reader.close();
                          scanner.close();
                          return count;
                        } catch (Exception e) {
                          throw new RuntimeException(e);
                        }
                      });

          futures.add(future);
        }

        // Wait for all scans to complete
        CompletableFuture<Void> allDone =
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]));
        allDone.get(30, TimeUnit.SECONDS);

        // Verify each scan read the expected number of rows
        for (CompletableFuture<Integer> future : futures) {
          assertEquals(20, future.get(), "Each range should have 20 rows");
        }
      }
    }
  }

  /**
   * Example 4: Async scan with error handling.
   *
   * <p>Shows how to handle errors in async operations.
   */
  @Test
  void testAsyncScanErrorHandling(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_error").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      try (Dataset dataset = testDataset.write(1, 40)) {
        ScanOptions options = new ScanOptions.Builder().build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
          CompletableFuture<ArrowReader> future =
              scanner
                  .scanBatchesAsync()
                  .whenComplete(
                      (reader, error) -> {
                        if (error != null) {
                          // Handle error
                          System.err.println("Scan failed: " + error.getMessage());
                        } else {
                          // Process successful result
                          assertNotNull(reader);
                        }
                      });

          ArrowReader reader = future.get(10, TimeUnit.SECONDS);
          assertNotNull(reader);
          reader.close();
        }
      }
    }
  }

  /**
   * Example 5: Async scan with projection (column selection).
   *
   * <p>Shows how to select specific columns for better performance.
   */
  @Test
  void testAsyncScanWithProjection(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_projection").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      try (Dataset dataset = testDataset.write(1, 40)) {
        // Select only "id" column
        ScanOptions options = new ScanOptions.Builder().columns(List.of("id")).build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
          CompletableFuture<ArrowReader> future = scanner.scanBatchesAsync();

          ArrowReader reader = future.get(10, TimeUnit.SECONDS);

          // Verify schema has only one column
          assertEquals(1, reader.getVectorSchemaRoot().getFieldVectors().size());
          assertEquals("id", reader.getVectorSchemaRoot().getVector(0).getName());

          reader.close();
        }
      }
    }
  }

  /**
   * Regression test for classloader isolation.
   *
   * <p>Verifies that AsyncScanner works correctly when the calling thread has an isolated
   * (non-system) context classloader, simulating environments like Spark executors, web containers,
   * or shaded JARs where application classes are loaded by a custom classloader.
   *
   * <p>The fix under test moved class resolution from the JNI dispatcher thread (which only has the
   * system classloader after attach_current_thread_permanently()) to JNI_OnLoad (which has the
   * correct application classloader), passing a GlobalRef to the dispatcher.
   */
  @Test
  void testAsyncScanWithIsolatedClassloader(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_classloader").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      int totalRows = 40;

      try (Dataset dataset = testDataset.write(1, totalRows)) {
        ScanOptions options = new ScanOptions.Builder().batchSize(20L).build();

        // Use a classloader that delegates only to the bootstrap classloader,
        // making it unable to find any org.lance.* classes on its own.
        ClassLoader originalCl = Thread.currentThread().getContextClassLoader();
        ClassLoader restrictedCl = new ClassLoader(null) {
              // Intentionally empty: parent is null (bootstrap only)
            };

        // --- Part 1: Swap context classloader on the current thread ---
        try {
          Thread.currentThread().setContextClassLoader(restrictedCl);

          try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
            CompletableFuture<ArrowReader> future = scanner.scanBatchesAsync();
            ArrowReader reader = future.get(10, TimeUnit.SECONDS);
            assertNotNull(reader);

            int rowCount = 0;
            while (reader.loadNextBatch()) {
              VectorSchemaRoot root = reader.getVectorSchemaRoot();
              rowCount += root.getRowCount();
            }

            assertEquals(
                totalRows, rowCount, "Async scan should succeed despite restrictive classloader");
            reader.close();
          }
        } finally {
          Thread.currentThread().setContextClassLoader(originalCl);
        }

        // --- Part 2: Run from a thread with an isolated classloader ---
        // Simulates Spark executor threads that have a non-system classloader.
        ExecutorService isolatedExecutor =
            Executors.newSingleThreadExecutor(
                r -> {
                  Thread t = new Thread(r, "isolated-classloader-thread");
                  t.setContextClassLoader(restrictedCl);
                  return t;
                });
        try {
          CompletableFuture<Integer> result =
              CompletableFuture.supplyAsync(
                  () -> {
                    try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
                      CompletableFuture<ArrowReader> future = scanner.scanBatchesAsync();
                      ArrowReader reader = future.get(10, TimeUnit.SECONDS);
                      int rowCount = 0;
                      while (reader.loadNextBatch()) {
                        rowCount += reader.getVectorSchemaRoot().getRowCount();
                      }
                      reader.close();
                      return rowCount;
                    } catch (Exception e) {
                      throw new RuntimeException(e);
                    }
                  },
                  isolatedExecutor);

          assertEquals(
              totalRows,
              result.get(15, TimeUnit.SECONDS),
              "Async scan from isolated classloader thread should succeed");
        } finally {
          isolatedExecutor.shutdown();
        }
      }
    }
  }

  /**
   * Example 6: Using thenCompose for sequential async operations.
   *
   * <p>Shows how to chain multiple async operations together.
   */
  @Test
  void testAsyncChaining(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("async_scanner_chaining").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      try (Dataset dataset = testDataset.write(1, 40)) {
        ScanOptions options = new ScanOptions.Builder().build();

        try (AsyncScanner scanner = AsyncScanner.create(dataset, options, allocator)) {
          // Chain operations: scan -> read first batch -> extract values
          CompletableFuture<List<Integer>> future =
              scanner
                  .scanBatchesAsync()
                  .thenApply(
                      reader -> {
                        try {
                          List<Integer> values = new ArrayList<>();
                          if (reader.loadNextBatch()) {
                            VectorSchemaRoot root = reader.getVectorSchemaRoot();
                            IntVector idVector = (IntVector) root.getVector("id");
                            for (int i = 0; i < root.getRowCount(); i++) {
                              values.add(idVector.get(i));
                            }
                          }
                          reader.close();
                          return values;
                        } catch (Exception e) {
                          throw new RuntimeException(e);
                        }
                      });

          List<Integer> values = future.get(10, TimeUnit.SECONDS);
          assertTrue(values.size() > 0, "Should read some values");
        }
      }
    }
  }

  /**
   * Regression test: reproduce the JNI classloader bug via a forked JVM.
   *
   * <p>Launches a child JVM where only {@code target/test-classes/} is on the system classpath. All
   * lance classes are loaded by an isolated {@link URLClassLoader} with a null parent (bootstrap
   * classloader only). This means JNI's {@code FindClass} on a native thread attached via {@code
   * attach_current_thread_permanently()} will use the system classloader, which <b>cannot</b> find
   * {@code org.lance.ipc.AsyncScanner}.
   *
   * <p>With the fix, {@code JNI_OnLoad} pre-resolves the class on the loading thread (which has the
   * correct classloader) and passes a {@code GlobalRef} to the dispatcher — so the dispatcher never
   * calls {@code FindClass}.
   */
  @Test
  void testClassloaderIsolationWithForkedJvm(@TempDir Path tempDir) throws Exception {
    // --- Collect full classpath from the current classloader hierarchy ---
    List<String> classpathEntries = new ArrayList<>();
    ClassLoader cl = Thread.currentThread().getContextClassLoader();
    if (cl == null) {
      cl = AsyncScannerTest.class.getClassLoader();
    }
    while (cl != null) {
      if (cl instanceof URLClassLoader) {
        for (URL url : ((URLClassLoader) cl).getURLs()) {
          classpathEntries.add(url.getPath());
        }
      }
      cl = cl.getParent();
    }

    // Fallback: if no URLClassLoader found (Java 9+ AppClassLoader), use java.class.path
    if (classpathEntries.isEmpty()) {
      String cp = System.getProperty("java.class.path");
      if (cp != null) {
        for (String entry : cp.split(java.io.File.pathSeparator)) {
          classpathEntries.add(entry);
        }
      }
    }

    // Find the test-classes directory
    String testClassesDir = null;
    for (String entry : classpathEntries) {
      if (entry.contains("test-classes")) {
        testClassesDir = entry;
        break;
      }
    }
    assertNotNull(testClassesDir, "Could not find test-classes directory in classpath");

    // Full classpath for the isolated URLClassLoader inside the forked JVM
    String fullClasspath = String.join(java.io.File.pathSeparator, classpathEntries);

    // --- Build the forked JVM command ---
    String javaHome = System.getProperty("java.home");
    String javaBin = javaHome + java.io.File.separator + "bin" + java.io.File.separator + "java";

    List<String> command = new ArrayList<>();
    command.add(javaBin);

    // Add --add-opens flags matching surefire config (required by Arrow/Netty)
    command.add("--add-opens=java.base/java.lang=ALL-UNNAMED");
    command.add("--add-opens=java.base/java.lang.invoke=ALL-UNNAMED");
    command.add("--add-opens=java.base/java.lang.reflect=ALL-UNNAMED");
    command.add("--add-opens=java.base/java.io=ALL-UNNAMED");
    command.add("--add-opens=java.base/java.net=ALL-UNNAMED");
    command.add("--add-opens=java.base/java.nio=ALL-UNNAMED");
    command.add("--add-opens=java.base/java.util=ALL-UNNAMED");
    command.add("--add-opens=java.base/java.util.concurrent=ALL-UNNAMED");
    command.add("--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED");
    command.add("--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED");
    command.add("--add-opens=java.base/sun.nio.ch=ALL-UNNAMED");
    command.add("--add-opens=java.base/sun.nio.cs=ALL-UNNAMED");
    command.add("--add-opens=java.base/sun.security.action=ALL-UNNAMED");
    command.add("--add-opens=java.base/sun.util.calendar=ALL-UNNAMED");
    command.add("--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED");
    command.add("-XX:+IgnoreUnrecognizedVMOptions");
    command.add("-Dio.netty.tryReflectionSetAccessible=true");

    // System classpath: ONLY test-classes (no lance main classes)
    command.add("-cp");
    command.add(testClassesDir);

    command.add("org.lance.ClassloaderBugBootstrap");
    command.add(fullClasspath);
    command.add(tempDir.toString());

    ProcessBuilder pb = new ProcessBuilder(command);
    pb.redirectErrorStream(false);

    Process process = pb.start();

    // Read stdout and stderr concurrently to avoid deadlock when buffers fill
    CompletableFuture<String> stderrFuture =
        CompletableFuture.supplyAsync(
            () -> {
              try (BufferedReader errReader =
                  new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
                return errReader.lines().collect(Collectors.joining("\n"));
              } catch (Exception e) {
                return "Failed to read stderr: " + e.getMessage();
              }
            });

    String stdout;
    try (BufferedReader outReader =
        new BufferedReader(new InputStreamReader(process.getInputStream()))) {
      stdout = outReader.lines().collect(Collectors.joining("\n"));
    }

    String stderr = stderrFuture.get(60, TimeUnit.SECONDS);

    boolean exited = process.waitFor(60, TimeUnit.SECONDS);
    assertTrue(exited, "Forked JVM did not exit within 60 seconds");

    int exitCode = process.exitValue();
    assertEquals(
        0,
        exitCode,
        "Forked JVM exited with code "
            + exitCode
            + "\n--- stdout ---\n"
            + stdout
            + "\n--- stderr ---\n"
            + stderr);
    assertTrue(
        stdout.contains("SUCCESS"),
        "Expected SUCCESS in stdout but got:\n--- stdout ---\n"
            + stdout
            + "\n--- stderr ---\n"
            + stderr);
  }
}
