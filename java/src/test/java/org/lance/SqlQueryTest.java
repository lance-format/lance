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

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class SqlQueryTest {
  private static final String NAME = "sqlquery_test_dataset";
  private static BufferAllocator allocator;
  private static Dataset dataset;

  @BeforeAll
  static void setup(@TempDir Path tempDir) {
    String datasetPath = tempDir.resolve(NAME).toString();
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
  public void testToRecordBatches() throws IOException {
    // Test normal query
    ArrowReader reader = dataset.sql("select * from " + NAME).tableName(NAME).intoBatchRecords();
    Assertions.assertEquals(
        "Schema<id: Int(32, true), name: Utf8>",
        reader.getVectorSchemaRoot().getSchema().toString());
    int rowCount = 0;
    int totalSum = 0;
    while (reader.loadNextBatch()) {
      rowCount += reader.getVectorSchemaRoot().getRowCount();
      for (int index = 0; index < reader.getVectorSchemaRoot().getRowCount(); index++) {
        int id = (Integer) reader.getVectorSchemaRoot().getVector(0).getObject(index);
        totalSum += id;
      }
    }
    Assertions.assertEquals(40, rowCount);
    Assertions.assertEquals(780, totalSum);
    reader.close();

    // Test agg query
    reader = dataset.sql("select sum(id) from " + NAME).tableName(NAME).intoBatchRecords();
    Assertions.assertEquals(
        "Schema<sum(sqlquery_test_dataset.id): Int(64, true)>",
        reader.getVectorSchemaRoot().getSchema().toString());
    Assertions.assertTrue(reader.loadNextBatch());
    long sum = (Long) reader.getVectorSchemaRoot().getVector(0).getObject(0);
    Assertions.assertEquals(780, sum);
    reader.close();

    // Test empty result
    reader =
        dataset.sql("select * from " + NAME + " where id < 0").tableName(NAME).intoBatchRecords();
    Assertions.assertEquals(
        "Schema<id: Int(32, true), name: Utf8>",
        reader.getVectorSchemaRoot().getSchema().toString());
    rowCount = 0;
    while (reader.loadNextBatch()) {
      rowCount += reader.getVectorSchemaRoot().getRowCount();
    }
    Assertions.assertEquals(0, rowCount);
    reader.close();

    // Test withRowId and rowAddr
    reader =
        dataset
            .sql("select id, name, _rowid, _rowaddr from " + NAME)
            .tableName(NAME)
            .withRowId(true)
            .withRowAddr(true)
            .intoBatchRecords();
    Assertions.assertEquals(
        "Schema<id: Int(32, true), name: Utf8, _rowid: Int(64, false), _rowaddr: Int(64, false)>",
        reader.getVectorSchemaRoot().getSchema().toString());
    reader.close();
  }

  @Test
  public void testRegisterArrow() throws Exception {
    // An in-memory Arrow relation (column `id`) to semi-join against the dataset; 100 is absent.
    Schema idSchema =
        new Schema(Collections.singletonList(Field.nullable("id", new ArrowType.Int(32, true))));
    try (VectorSchemaRoot ids = VectorSchemaRoot.create(idSchema, allocator)) {
      ids.allocateNew();
      IntVector idVector = (IntVector) ids.getVector("id");
      int[] wanted = {1, 5, 39, 100};
      for (int i = 0; i < wanted.length; i++) {
        idVector.setSafe(i, wanted[i]);
      }
      ids.setRowCount(wanted.length);

      // The native side consumes the stream; we allocate and close it (try-with-resources).
      try (ArrowArrayStream stream = toStream(ids)) {
        ArrowReader reader =
            dataset
                .sql(
                    "select id from "
                        + NAME
                        + " where id in (select id from filter_ids) order by id")
                .tableName(NAME)
                .registerArrow("filter_ids", stream)
                .intoBatchRecords();
        List<Integer> got = new ArrayList<>();
        while (reader.loadNextBatch()) {
          VectorSchemaRoot root = reader.getVectorSchemaRoot();
          for (int i = 0; i < root.getRowCount(); i++) {
            got.add((Integer) root.getVector(0).getObject(i));
          }
        }
        reader.close();
        // 100 is not in the dataset; the rest are the intersection, in order.
        Assertions.assertEquals(Arrays.asList(1, 5, 39), got);
      }
    }
  }

  @Test
  public void testRegisterArrowMultiple() throws Exception {
    // Register two relations in one query and join both; only ids present in all three survive.
    try (VectorSchemaRoot a = idTable(1, 2, 3, 10);
        VectorSchemaRoot b = idTable(2, 3, 4, 10);
        ArrowArrayStream sa = toStream(a);
        ArrowArrayStream sb = toStream(b)) {
      ArrowReader reader =
          dataset
              .sql(
                  "select id from "
                      + NAME
                      + " where id in (select id from a) and id in (select id from b) order by id")
              .tableName(NAME)
              .registerArrow("a", sa)
              .registerArrow("b", sb)
              .intoBatchRecords();
      List<Integer> got = new ArrayList<>();
      while (reader.loadNextBatch()) {
        VectorSchemaRoot root = reader.getVectorSchemaRoot();
        for (int i = 0; i < root.getRowCount(); i++) {
          got.add((Integer) root.getVector(0).getObject(i));
        }
      }
      reader.close();
      // a = {1,2,3,10}, b = {2,3,4,10}, dataset ids 0..39, so the intersection is {2,3,10}.
      Assertions.assertEquals(Arrays.asList(2, 3, 10), got);
    }
  }

  @Test
  public void testRegisterArrowValidationAndReuse() throws Exception {
    // registerArrow rejects invalid inputs at the boundary.
    try (VectorSchemaRoot ids = idTable(1, 2);
        ArrowArrayStream stream = toStream(ids)) {
      SqlQuery q = dataset.sql("select id from " + NAME).tableName(NAME);
      Assertions.assertThrows(IllegalArgumentException.class, () -> q.registerArrow("", stream));
      Assertions.assertThrows(IllegalArgumentException.class, () -> q.registerArrow(null, stream));
      Assertions.assertThrows(IllegalArgumentException.class, () -> q.registerArrow("ids", null));
    }

    // A query with registered relations is single-use: the stream is consumed on the first call,
    // so a second intoBatchRecords() throws rather than handing JNI a dead stream.
    try (VectorSchemaRoot ids = idTable(1, 2);
        ArrowArrayStream stream = toStream(ids)) {
      SqlQuery q =
          dataset
              .sql("select id from " + NAME + " where id in (select id from ids)")
              .tableName(NAME)
              .registerArrow("ids", stream);
      q.intoBatchRecords().close();
      Assertions.assertThrows(IllegalStateException.class, q::intoBatchRecords);

      // Registering another relation after the query is consumed is also rejected.
      try (VectorSchemaRoot more = idTable(3);
          ArrowArrayStream moreStream = toStream(more)) {
        Assertions.assertThrows(
            IllegalStateException.class, () -> q.registerArrow("more", moreStream));
      }
    }

    // A relation whose name resolves to the query's own table name would hide the dataset, so the
    // native build rejects it (the Rust InvalidInput surfaces as IllegalArgumentException). Assert
    // on the message too: DataFusion's own "table already exists" error maps to the same exception.
    try (VectorSchemaRoot ids = idTable(1, 2);
        ArrowArrayStream stream = toStream(ids)) {
      SqlQuery q =
          dataset.sql("select id from " + NAME).tableName(NAME).registerArrow(NAME, stream);
      IllegalArgumentException error =
          Assertions.assertThrows(IllegalArgumentException.class, q::intoBatchRecords);
      Assertions.assertTrue(
          error.getMessage().contains("resolves to")
              && error.getMessage().contains("the query's table name '" + NAME + "'"),
          "unexpected message: " + error.getMessage());
    }
  }

  /** Build a single-column (`id`: Int32) in-memory relation. */
  private VectorSchemaRoot idTable(int... ids) {
    Schema schema =
        new Schema(Collections.singletonList(Field.nullable("id", new ArrowType.Int(32, true))));
    VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
    root.allocateNew();
    IntVector idVector = (IntVector) root.getVector("id");
    for (int i = 0; i < ids.length; i++) {
      idVector.setSafe(i, ids[i]);
    }
    root.setRowCount(ids.length);
    return root;
  }

  /**
   * Serialize a single-batch root to a self-contained Arrow C-Data stream (mirrors
   * MergeInsertTest).
   */
  private ArrowArrayStream toStream(VectorSchemaRoot root) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
      writer.start();
      writer.writeBatch();
      writer.end();
    }
    ArrowStreamReader reader =
        new ArrowStreamReader(new ByteArrayInputStream(out.toByteArray()), allocator);
    ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator);
    Data.exportArrayStream(allocator, reader, stream);
    return stream;
  }
}
