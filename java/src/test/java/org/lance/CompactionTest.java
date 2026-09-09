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

import org.lance.compaction.Compaction;
import org.lance.compaction.CompactionMetrics;
import org.lance.compaction.CompactionMode;
import org.lance.compaction.CompactionOptions;
import org.lance.compaction.CompactionPlan;
import org.lance.compaction.CompactionTask;
import org.lance.compaction.RewriteResult;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Add test for distributed compaction. */
public class CompactionTest {
  @Test
  public void testBasicCompaction(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("test_dataset_for_compaction").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);

      testDataset.createEmptyDataset().close();

      // Step-1: write two fragments
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        CompactionOptions compactionOptions =
            CompactionOptions.builder()
                .withTargetRowsPerFragment(100)
                .withNumThreads(1)
                .withMaxSourceRows(1000)
                .withMaxSourceBytes(10L * 1024 * 1024)
                .build();
        CompactionPlan compactionPlan = Compaction.planCompaction(dataset, compactionOptions);

        // The source budgets are loose, so the plan is unaffected and the
        // options must survive the JNI round trip.
        assertEquals(Optional.of(1000L), compactionPlan.getCompactionOptions().getMaxSourceRows());
        assertEquals(
            Optional.of(10L * 1024 * 1024),
            compactionPlan.getCompactionOptions().getMaxSourceBytes());

        // will plan to compact two fragments into one.
        assertEquals(1, compactionPlan.getCompactionTasks().size());
        CompactionTask task = compactionPlan.getCompactionTasks().get(0);
        assertEquals(2, task.getTaskData().getFragments().size());

        // Step-2: individually execute single task

        // mock network transferring
        task = serializeAndDeserialize(task);
        RewriteResult result = task.execute(dataset);
        CompactionMetrics metrics = result.getMetrics();
        // remove previous fragments and add new single fragment
        assertEquals(2, metrics.getFragmentsRemoved());
        assertEquals(1, metrics.getFragmentsAdded());

        // Step-3: commit the RewriteResults

        // mock network transferring
        result = serializeAndDeserialize(result);
        CompactionMetrics ignored =
            Compaction.commitCompaction(
                dataset, Collections.singletonList(result), compactionPlan.getCompactionOptions());

        // checkout to the latest snapshot and verify row num and fragment num.
        dataset.checkoutLatest();
        assertEquals(1, dataset.getFragments().size());
        assertEquals(20, dataset.getFragments().get(0).countRows());
      }
    }
  }

  @Test
  public void testDeletionCompaction(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("test_dataset_for_compaction").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      // Step-1: write two fragments
      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        dataset.delete("_rowid <= 8");

        dataset.checkoutLatest();
        // still 2 fragments
        assertEquals(2, dataset.getFragments().size());

        CompactionOptions compactionOptions =
            CompactionOptions.builder()
                .withMaterializeDeletions(true)
                .withMaterializeDeletionsThreshold(0.5f)
                .withNumThreads(1)
                .build();
        CompactionPlan compactionPlan = Compaction.planCompaction(dataset, compactionOptions);

        assertEquals(1, compactionPlan.getCompactionTasks().size());

        CompactionTask task = compactionPlan.getCompactionTasks().get(0);

        task = serializeAndDeserialize(task);
        RewriteResult result = task.execute(dataset);
        assertEquals(2, result.getMetrics().getFragmentsRemoved());
        assertEquals(1, result.getMetrics().getFragmentsAdded());

        result = serializeAndDeserialize(result);
        CompactionMetrics ignored =
            Compaction.commitCompaction(
                dataset, Collections.singletonList(result), compactionPlan.getCompactionOptions());

        // checkout to the latest snapshot and verify row num and fragment num.
        dataset.checkoutLatest();
        assertEquals(1, dataset.getFragments().size());
        assertEquals(11, dataset.getFragments().get(0).countRows());
      }
    }
  }

  @Test
  public void testExcludedFragmentIds(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("test_excluded_fragment_ids").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();
      testDataset.write(1, 10).close();
      testDataset.write(2, 10).close();
      testDataset.write(3, 10).close();
      try (Dataset dataset = testDataset.write(4, 10)) {
        CompactionOptions options =
            CompactionOptions.builder()
                .withTargetRowsPerFragment(100)
                .withExcludedFragmentIds(Arrays.asList(1L, 1L, 999L))
                .build();

        CompactionPlan plan = Compaction.planCompaction(dataset, options);

        assertEquals(
            Arrays.asList(1L, 1L, 999L), plan.getCompactionOptions().getExcludedFragmentIds());
        assertEquals(1, plan.getCompactionTasks().size());
        assertEquals(2, plan.getCompactionTasks().get(0).getTaskData().getFragments().size());
        assertEquals(
            2, plan.getCompactionTasks().get(0).getTaskData().getFragments().get(0).getId());
        assertEquals(
            3, plan.getCompactionTasks().get(0).getTaskData().getFragments().get(1).getId());

        CompactionTask task = serializeAndDeserialize(plan.getCompactionTasks().get(0));
        assertEquals(
            Arrays.asList(1L, 1L, 999L), task.getCompactionOptions().getExcludedFragmentIds());
      }
    }
  }

  @ParameterizedTest
  @EnumSource(CompactionMode.class)
  public void testCompactionModeRoundTrip(CompactionMode mode, @TempDir Path tempDir)
      throws Exception {
    String datasetPath = tempDir.resolve("test_dataset_for_compaction").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.SimpleTestDataset testDataset =
          new TestUtils.SimpleTestDataset(allocator, datasetPath);
      testDataset.createEmptyDataset().close();

      testDataset.write(1, 10).close();
      try (Dataset dataset = testDataset.write(2, 10)) {
        CompactionOptions compactionOptions =
            CompactionOptions.builder()
                .withTargetRowsPerFragment(100)
                .withNumThreads(1)
                .withCompactionMode(mode)
                .build();
        CompactionPlan compactionPlan = Compaction.planCompaction(dataset, compactionOptions);

        // The plan's options are rebuilt by the native layer; the mode must come
        // back as a CompactionMode enum, not a raw String.
        assertEquals(
            Optional.of(mode.getValue()),
            compactionPlan.getCompactionOptions().getCompactionMode());

        CompactionTask task = serializeAndDeserialize(compactionPlan.getCompactionTasks().get(0));
        RewriteResult result = task.execute(dataset);
        assertEquals(2, result.getMetrics().getFragmentsRemoved());
        assertEquals(1, result.getMetrics().getFragmentsAdded());
      }
    }
  }

  /**
   * A serialized CompactionOptions produced by the class as it existed before maxSourceRows and
   * maxSourceBytes were added (no declared serialVersionUID, stream ends after maxSourceFragments),
   * built with targetRowsPerFragment=1024, materializeDeletions=true,
   * compactionMode=TRY_BINARY_COPY, maxSourceFragments=4.
   */
  private static final String PRE_SOURCE_BUDGET_OPTIONS_BASE64 =
      "rO0ABXNyACZvcmcubGFuY2UuY29tcGFjdGlvbi5Db21wYWN0aW9uT3B0aW9ucys6bRwua1fWAwALTAAJYmF0Y2hTaXpl"
          + "dAAUTGphdmEvdXRpbC9PcHRpb25hbDtMABhiaW5hcnlDb3B5UmVhZEJhdGNoQnl0ZXNxAH4AAUwADmNvbXBhY3Rpb25N"
          + "b2RlcQB+AAFMAA9kZWZlckluZGV4UmVtYXBxAH4AAUwAFG1hdGVyaWFsaXplRGVsZXRpb25zcQB+AAFMAB1tYXRlcmlh"
          + "bGl6ZURlbGV0aW9uc1RocmVzaG9sZHEAfgABTAAPbWF4Qnl0ZXNQZXJGaWxlcQB+AAFMAA9tYXhSb3dzUGVyR3JvdXBx"
          + "AH4AAUwAEm1heFNvdXJjZUZyYWdtZW50c3EAfgABTAAKbnVtVGhyZWFkc3EAfgABTAAVdGFyZ2V0Um93c1BlckZyYWdt"
          + "ZW50cQB+AAF4cHNyAA5qYXZhLmxhbmcuTG9uZzuL5JDMjyPfAgABSgAFdmFsdWV4cgAQamF2YS5sYW5nLk51bWJlcoas"
          + "lR0LlOCLAgAAeHAAAAAAAAAEAHBwc3IAEWphdmEubGFuZy5Cb29sZWFuzSBygNWc+u4CAAFaAAV2YWx1ZXhwAXBwcHB0"
          + "AA90cnlfYmluYXJ5X2NvcHlwc3EAfgADAAAAAAAAAAR4";

  @Test
  public void testDeserializeOptionsFromOlderVersion() throws Exception {
    byte[] serialized = Base64.getDecoder().decode(PRE_SOURCE_BUDGET_OPTIONS_BASE64);
    CompactionOptions options;
    try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(serialized))) {
      options = (CompactionOptions) in.readObject();
    }
    assertEquals(Optional.of(1024L), options.getTargetRowsPerFragment());
    assertEquals(Optional.of(true), options.getMaterializeDeletions());
    assertEquals(
        Optional.of(CompactionMode.TRY_BINARY_COPY.getValue()), options.getCompactionMode());
    assertEquals(Optional.of(4L), options.getMaxSourceFragments());
    // Fields absent from the old stream deserialize as unset.
    assertEquals(Optional.empty(), options.getMaxSourceRows());
    assertEquals(Optional.empty(), options.getMaxSourceBytes());
    assertEquals(Collections.emptyList(), options.getExcludedFragmentIds());
  }

  private static <T> T serializeAndDeserialize(T object)
      throws IOException, ClassNotFoundException {
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    try (ObjectOutputStream out = new ObjectOutputStream(outputStream)) {
      out.writeObject(object);
    }
    byte[] serialized = outputStream.toByteArray();
    ByteArrayInputStream inputStream = new ByteArrayInputStream(serialized);
    try (ObjectInputStream in = new ObjectInputStream(inputStream)) {
      @SuppressWarnings("unchecked")
      T deserialized = (T) in.readObject();
      return deserialized;
    }
  }
}
