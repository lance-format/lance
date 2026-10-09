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

import org.lance.delta.DatasetDelta;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.arrow.vector.util.ByteArrayReadableSeekableByteChannel;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/** Tests for Dataset.delta() Java interface bridging Rust semantics. */
public class DeltaTest {

  @Test
  public void testInsertedRowsComparedAgainst(@TempDir Path tempDir) throws IOException {
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      String uri = tempDir.resolve("delta_demo").toString();
      // Build initial batch (2 rows)
      Schema schema =
          new Schema(
              Arrays.asList(
                  Field.notNullable(
                      "id", new org.apache.arrow.vector.types.pojo.ArrowType.Int(32, true)),
                  Field.nullable(
                      "val", org.apache.arrow.vector.types.pojo.ArrowType.Utf8.INSTANCE)));

      VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
      root.allocateNew();
      IntVector idVec = (IntVector) root.getVector("id");
      VarCharVector valVec = (VarCharVector) root.getVector("val");
      idVec.setSafe(0, 1);
      idVec.setSafe(1, 2);
      valVec.setSafe(0, "a".getBytes());
      valVec.setSafe(1, "b".getBytes());
      root.setRowCount(2);
      byte[] batch1;
      // Create an output stream explicitly and pass it to ArrowStreamWriter
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
        writer.start();
        writer.writeBatch();
        writer.end();
      }
      batch1 = out.toByteArray();
      root.close();

      try (ArrowStreamReader reader1 =
              new ArrowStreamReader(new ByteArrayReadableSeekableByteChannel(batch1), allocator);
          org.apache.arrow.c.ArrowArrayStream stream1 =
              org.apache.arrow.c.ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader1, stream1);
        Dataset ds =
            Dataset.write().stream(stream1)
                .uri(uri)
                .mode(WriteParams.WriteMode.CREATE)
                .enableStableRowIds(true)
                .execute();

        // Append one row (v2)
        VectorSchemaRoot root2 = VectorSchemaRoot.create(schema, allocator);
        root2.allocateNew();
        IntVector idVec2 = (IntVector) root2.getVector("id");
        VarCharVector valVec2 = (VarCharVector) root2.getVector("val");
        idVec2.setSafe(0, 3);
        valVec2.setSafe(0, "c".getBytes());
        root2.setRowCount(1);
        byte[] batch2;
        ByteArrayOutputStream out2 = new ByteArrayOutputStream();
        try (ArrowStreamWriter writer2 = new ArrowStreamWriter(root2, null, out2)) {
          writer2.start();
          writer2.writeBatch();
          writer2.end();
        }
        batch2 = out2.toByteArray();
        root2.close();

        try (ArrowStreamReader reader2 =
                new ArrowStreamReader(new ByteArrayReadableSeekableByteChannel(batch2), allocator);
            ArrowArrayStream stream2 = ArrowArrayStream.allocateNew(allocator)) {
          Data.exportArrayStream(allocator, reader2, stream2);
          Dataset ds2 =
              Dataset.write().stream(stream2).uri(uri).mode(WriteParams.WriteMode.APPEND).execute();

          DatasetDelta delta = ds2.delta(1L);
          try (ArrowReader inserted = delta.getInsertedRows()) {
            int total = 0;
            boolean foundRow = false;

            while (inserted.loadNextBatch()) {
              VectorSchemaRoot outRoot = inserted.getVectorSchemaRoot();
              Schema outSchema = outRoot.getSchema();
              List<String> names =
                  outSchema.getFields().stream().map(Field::getName).collect(Collectors.toList());
              Assertions.assertTrue(names.contains("_row_created_at_version"));
              Assertions.assertTrue(names.contains("_row_last_updated_at_version"));

              IntVector outId = (IntVector) outRoot.getVector("id");
              VarCharVector outVal = (VarCharVector) outRoot.getVector("val");

              for (int i = 0; i < outRoot.getRowCount(); i++) {
                int id = outId.get(i);
                byte[] bytes = outVal.get(i);
                String val = new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
                if (id == 3 && "c".equals(val)) {
                  foundRow = true;
                }
              }

              total += outRoot.getRowCount();
            }

            Assertions.assertEquals(1, total);
            Assertions.assertTrue(foundRow, "Inserted row (id=3, val=c) not found in delta");
          }
        }
      }
    }
  }

  @Test
  public void testListTransactionsExplicitRange(@TempDir Path tempDir) throws IOException {
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      String uri = tempDir.resolve("delta_demo_tx").toString();
      Schema schema =
          new Schema(
              Arrays.asList(
                  Field.notNullable(
                      "id", new org.apache.arrow.vector.types.pojo.ArrowType.Int(32, true)),
                  Field.nullable(
                      "val", org.apache.arrow.vector.types.pojo.ArrowType.Utf8.INSTANCE)));

      // v1: create with two rows.
      byte[] batch1 = writeBatch(allocator, schema, new int[] {1, 2}, new String[] {"a", "b"});
      try (ArrowStreamReader reader1 =
              new ArrowStreamReader(new ByteArrayReadableSeekableByteChannel(batch1), allocator);
          ArrowArrayStream stream1 = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader1, stream1);
        Dataset.write().stream(stream1)
            .uri(uri)
            .mode(WriteParams.WriteMode.CREATE)
            .execute()
            .close();
      }

      // v2: append one row.
      byte[] batch2 = writeBatch(allocator, schema, new int[] {3}, new String[] {"c"});
      try (ArrowStreamReader reader2 =
              new ArrowStreamReader(new ByteArrayReadableSeekableByteChannel(batch2), allocator);
          ArrowArrayStream stream2 = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader2, stream2);
        try (Dataset ds2 =
            Dataset.write().stream(stream2).uri(uri).mode(WriteParams.WriteMode.APPEND).execute()) {
          DatasetDelta delta = ds2.delta(1L, 2L);
          List<Transaction> txs = delta.listTransactions();
          Assertions.assertEquals(1, txs.size(), "delta v1..v2 should contain exactly one txn");
        }
      }
    }
  }

  @Test
  public void testGetDeletedRowIds(@TempDir Path tempDir) throws IOException {
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      String uri = tempDir.resolve("delta_demo_delete").toString();
      Schema schema =
          new Schema(
              Arrays.asList(
                  Field.notNullable(
                      "id", new org.apache.arrow.vector.types.pojo.ArrowType.Int(32, true)),
                  Field.nullable(
                      "val", org.apache.arrow.vector.types.pojo.ArrowType.Utf8.INSTANCE)));

      // v1: create with three rows, keeping stable row ids so deletions are reportable.
      byte[] batch1 =
          writeBatch(allocator, schema, new int[] {1, 2, 3}, new String[] {"a", "b", "c"});
      try (ArrowStreamReader reader1 =
              new ArrowStreamReader(new ByteArrayReadableSeekableByteChannel(batch1), allocator);
          ArrowArrayStream stream1 = ArrowArrayStream.allocateNew(allocator)) {
        Data.exportArrayStream(allocator, reader1, stream1);
        Dataset.write().stream(stream1)
            .uri(uri)
            .mode(WriteParams.WriteMode.CREATE)
            .enableStableRowIds(true)
            .execute()
            .close();
      }

      // v2: delete one row.
      try (Dataset ds = Dataset.open(uri, allocator)) {
        ds.delete("id = 2");
      }

      try (Dataset ds2 = Dataset.open(uri, allocator)) {
        DatasetDelta delta = ds2.delta(1L);
        try (ArrowReader deleted = delta.getDeletedRowIds()) {
          int total = 0;
          while (deleted.loadNextBatch()) {
            VectorSchemaRoot outRoot = deleted.getVectorSchemaRoot();
            List<String> names =
                outRoot.getSchema().getFields().stream()
                    .map(Field::getName)
                    .collect(Collectors.toList());
            Assertions.assertEquals(Arrays.asList("_rowid"), names);
            total += outRoot.getRowCount();
          }
          Assertions.assertEquals(1, total, "exactly one row was deleted");
        }
      }
    }
  }

  /** Helper: serialize a single Arrow batch with the given schema and (id, val) pairs. */
  private static byte[] writeBatch(RootAllocator allocator, Schema schema, int[] ids, String[] vals)
      throws IOException {
    Assertions.assertEquals(ids.length, vals.length, "ids and vals must align");
    VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
    try {
      root.allocateNew();
      IntVector idVec = (IntVector) root.getVector("id");
      VarCharVector valVec = (VarCharVector) root.getVector("val");
      for (int i = 0; i < ids.length; i++) {
        idVec.setSafe(i, ids[i]);
        valVec.setSafe(i, vals[i].getBytes(java.nio.charset.StandardCharsets.UTF_8));
      }
      root.setRowCount(ids.length);
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
        writer.start();
        writer.writeBatch();
        writer.end();
      }
      return out.toByteArray();
    } finally {
      root.close();
    }
  }
}
