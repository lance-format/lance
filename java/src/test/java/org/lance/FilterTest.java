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

import org.lance.ipc.LanceScanner;
import org.lance.ipc.ScanOptions;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.ByteArrayReadableSeekableByteChannel;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class FilterTest {
  private static BufferAllocator allocator;
  private static Dataset dataset;

  @BeforeAll
  static void setup(@TempDir Path tempDir) throws IOException {
    String datasetPath = tempDir.resolve("filter_test_dataset").toString();
    allocator = new RootAllocator();
    TestUtils.SimpleTestDataset testDataset =
        new TestUtils.SimpleTestDataset(allocator, datasetPath);
    testDataset.createEmptyDataset().close();
    // write id with value from 0 to 39
    dataset = testDataset.write(1, 40);
  }

  @AfterAll
  static void tearDown() {
    // Cleanup resources used by the tests
    if (dataset != null) {
      dataset.close();
    }
    if (allocator != null) {
      allocator.close();
    }
  }

  @Test
  void testFilters() throws Exception {
    testFilter("id == 10", 1);
    testFilter("id == 10", 1);
    testFilter("id != 10", 39);
    testFilter("id > 10", 29);
    testFilter("id >= 10", 30);
    testFilter("id < 10", 10);
    testFilter("id <= 10", 11);
    testFilter("id >= 10 and id < 20", 10);
    testFilter("id < 10 or id > 30", 19);
    testFilter("id != 10 and id < 20", 19);
    testFilter("id < 5 or id > 35", 9);
    testFilter("(id >= 5 and id <= 15) or (id >= 25 and id <= 35)", 22);
    testFilter("id == 5 or (id >= 30 and id < 35)", 6);
    testFilter("id IS NOT NULL", 40);
    testFilter("id IS NULL", 0);
    testFilter("id IN (5, 15, 25, 35)", 4);

    testFilter("name LIKE 'Person%'", 40);
    testFilter("name LIKE 'Person 1%'", 11);
    testFilter("name LIKE '%0'", 4);
    testFilter("name LIKE '%son 1'", 1);
    testFilter("name LIKE '%son 1%'", 11);
    testFilter("name LIKE '%son%'", 40);
    testFilter("name == 'Person 1'", 1);
    testFilter("name IS NULL", 0);
    testFilter("name IS NOT NULL", 40);

    testFilter("name LIKE 'Person%' AND name LIKE '%0'", 4);
    testFilter("name LIKE 'Person%' AND name LIKE '%1%'", 13);
    testFilter("name LIKE '%son%' AND name LIKE '%0'", 4);
    testFilter("name LIKE '%son%' AND name LIKE '%1'", 4);
    testFilter("name LIKE 'Person 1%' AND name LIKE '%1'", 2);

    testFilter("(name IS NOT NULL) AND (name == 'Person 1')", 1);
    testFilter("(name IS NOT NULL) AND (name == 'Person')", 0);
    // Not supported, bug?, LanceError(IO): Schema error: No field named person. Valid fields are
    // id, name.
    // testFilter("(name IS NOT NULL) AND (name == Person)", 0);

    // Not supported
    // testFilter("\"id\" == 10", 1);
    // testFilter("'id' == 10", 1);
  }

  /** Java hands strings to JNI as modified UTF-8, which encodes these characters differently. */
  @Test
  void testFilterAndColumnsWithCharactersOutsideTheBmp(@TempDir Path tempDir) throws Exception {
    String column = "tag👍";
    String[] tags = {"plain", "👍", "𠀀"};
    String datasetPath = tempDir.resolve("supplementary_characters").toString();
    try (BufferAllocator allocator = new RootAllocator()) {
      Schema schema =
          new Schema(
              Arrays.asList(
                  Field.nullable("id", new ArrowType.Int(32, true)),
                  Field.nullable(column, new ArrowType.Utf8())));
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        root.allocateNew();
        IntVector ids = (IntVector) root.getVector("id");
        VarCharVector values = (VarCharVector) root.getVector(column);
        for (int i = 0; i < tags.length; i++) {
          ids.setSafe(i, i);
          values.setSafe(i, tags[i].getBytes(StandardCharsets.UTF_8));
        }
        root.setRowCount(tags.length);
        try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
          writer.writeBatch();
        }
      }
      try (ArrowStreamReader reader =
              new ArrowStreamReader(
                  new ByteArrayReadableSeekableByteChannel(out.toByteArray()), allocator);
          Dataset written =
              Dataset.write().allocator(allocator).reader(reader).uri(datasetPath).execute()) {
        assertEquals(1, written.countRows("`" + column + "` = '" + tags[1] + "'"));
        ScanOptions options =
            new ScanOptions.Builder()
                .columns(Collections.singletonList(column))
                .filter("`" + column + "` = '" + tags[2] + "'")
                .build();
        List<String> matched = new ArrayList<>();
        try (LanceScanner scanner = written.newScan(options);
            ArrowReader batches = scanner.scanBatches()) {
          while (batches.loadNextBatch()) {
            VarCharVector vector = (VarCharVector) batches.getVectorSchemaRoot().getVector(column);
            for (int i = 0; i < vector.getValueCount(); i++) {
              matched.add(new String(vector.get(i), StandardCharsets.UTF_8));
            }
          }
        }
        assertEquals(Collections.singletonList(tags[2]), matched);
      }
    }
  }

  private void testFilter(String filter, int expectedCount) throws Exception {
    try (LanceScanner scanner =
        dataset.newScan(
            new ScanOptions.Builder()
                .columns(Arrays.asList())
                .withRowId(true)
                .filter(filter)
                .build())) {
      assertEquals(expectedCount, scanner.countRows());
    }
  }
}
