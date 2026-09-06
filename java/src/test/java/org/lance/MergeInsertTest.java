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
import org.lance.merge.MergeInsertParams;
import org.lance.merge.MergeInsertResult;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import java.util.UUID;

public class MergeInsertTest {
  private static final Schema WIDE_SCHEMA =
      new Schema(
          Arrays.asList(
              Field.nullable("id", new ArrowType.Int(32, true)),
              Field.nullable("name", new ArrowType.Utf8()),
              Field.nullable("score", new ArrowType.Int(32, true))));

  /** {@link #WIDE_SCHEMA} minus "score", so a source over it is a partial-schema update. */
  private static final Schema PARTIAL_SOURCE_SCHEMA =
      new Schema(
          Arrays.asList(
              Field.nullable("id", new ArrowType.Int(32, true)),
              Field.nullable("name", new ArrowType.Utf8())));

  /** {@link #createWideDataset()} after the rows with id 1 and 4 are renamed. */
  private static final String WIDE_AFTER_UPDATE =
      "{0=Person 0/0, 1=Updated 1/10, 2=Person 2/20, 3=Person 3/30, 4=Updated 4/40, 5=Person 5/50}";

  @TempDir private Path tempDir;
  private RootAllocator allocator;
  private TestUtils.SimpleTestDataset testDataset;
  private Dataset dataset;

  @BeforeEach
  public void setup() {
    String datasetPath = tempDir.resolve(UUID.randomUUID().toString()).toString();
    allocator = new RootAllocator(Long.MAX_VALUE);
    testDataset = new TestUtils.SimpleTestDataset(allocator, datasetPath);
    testDataset.createEmptyDataset().close();
    dataset = testDataset.write(1, 5);
  }

  @AfterEach
  public void tearDown() {
    dataset.close();
    allocator.close();
  }

  @Test
  public void testWhenNotMatchedInsertAll() throws Exception {
    // Test insert all unmatched source rows

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id")), sourceStream);

        Assertions.assertEquals(
            "{0=Person 0, 1=Person 1, 2=Person 2, 3=Person 3, 4=Person 4, 7=Source 7, 8=Source 8, 9=Source 9}",
            readAll(result.dataset()).toString());
      }
    }
  }

  @Test
  public void testWhenNotMatchedDoNothing() throws Exception {
    // Test ignore unmatched source rows

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedUpdateAll()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing),
                sourceStream);

        Assertions.assertEquals(
            "{0=Source 0, 1=Source 1, 2=Source 2, 3=Person 3, 4=Person 4}",
            readAll(result.dataset()).toString());
      }
    }
  }

  @Test
  public void testWhenMatchedUpdateIf() throws Exception {
    // Test update matched rows if expression is true

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedUpdateIf("target.name = 'Person 0' or target.name = 'Person 1'")
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing),
                sourceStream);

        Assertions.assertEquals(
            "{0=Source 0, 1=Source 1, 2=Person 2, 3=Person 3, 4=Person 4}",
            readAll(result.dataset()).toString());
      }
    }
  }

  @Test
  public void testWhenNotMatchedBySourceDelete() throws Exception {
    // Test delete target rows which are not matched with source.

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withNotMatchedBySourceDelete()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing),
                sourceStream);

        Assertions.assertEquals(
            "{0=Person 0, 1=Person 1, 2=Person 2}", readAll(result.dataset()).toString());
      }
    }
  }

  @Test
  public void testWhenNotMatchedBySourceDeleteIf() throws Exception {
    // Test delete target rows which are not matched with source if expression is true

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withNotMatchedBySourceDeleteIf("name = 'Person 3'")
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing),
                sourceStream);

        Assertions.assertEquals(
            "{0=Person 0, 1=Person 1, 2=Person 2, 4=Person 4}",
            readAll(result.dataset()).toString());
      }
    }
  }

  @Test
  public void testWhenMatchedFailWithMatches() throws Exception {
    // Test fail when there are matched rows

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        String originalDataset = readAll(dataset).toString();

        Assertions.assertThrows(
            Exception.class,
            () ->
                dataset.mergeInsert(
                    new MergeInsertParams(Collections.singletonList("id")).withMatchedFail(),
                    sourceStream));

        // Verify dataset remains unchanged
        Assertions.assertEquals(
            originalDataset,
            readAll(dataset).toString(),
            "Dataset should remain unchanged after failed mergeInsert");
      }
    }
  }

  @Test
  public void testWhenMatchedFailWithoutMatches() throws Exception {
    // Test success when there are no matched rows

    try (VectorSchemaRoot root = VectorSchemaRoot.create(testDataset.getSchema(), allocator)) {
      root.allocateNew();

      IntVector idVector = (IntVector) root.getVector("id");
      VarCharVector nameVector = (VarCharVector) root.getVector("name");

      List<Integer> sourceIds = Arrays.asList(100, 101, 102);
      for (int i = 0; i < sourceIds.size(); i++) {
        idVector.setSafe(i, sourceIds.get(i));
        String name = "New Data " + sourceIds.get(i);
        nameVector.setSafe(i, name.getBytes(StandardCharsets.UTF_8));
      }

      root.setRowCount(sourceIds.size());

      try (ArrowArrayStream sourceStream = convertToStream(root, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id")).withMatchedFail(),
                sourceStream);

        // Verify new data is inserted
        Map<Integer, String> resultMap = readAll(result.dataset());
        for (int id : sourceIds) {
          Assertions.assertTrue(resultMap.containsKey(id));
          Assertions.assertEquals("New Data " + id, resultMap.get(id));
        }
      }
    }
  }

  @Test
  public void testWhenMatchedDelete() throws Exception {
    // Test delete matched target rows if expression is true

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedDelete()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing),
                sourceStream);

        Assertions.assertEquals("{3=Person 3, 4=Person 4}", readAll(result.dataset()).toString());
      }
    }
  }

  private VectorSchemaRoot buildSource(Schema schema, RootAllocator allocator) {
    List<Integer> sourceIds = Arrays.asList(0, 1, 2, 7, 8, 9);

    VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
    root.allocateNew();

    IntVector idVector = (IntVector) root.getVector("id");
    VarCharVector nameVector = (VarCharVector) root.getVector("name");

    for (int i = 0; i < sourceIds.size(); i++) {
      idVector.setSafe(i, sourceIds.get(i));
      String name = "Source " + sourceIds.get(i);
      nameVector.setSafe(i, name.getBytes(StandardCharsets.UTF_8));
    }

    root.setRowCount(sourceIds.size());

    return root;
  }

  private ArrowArrayStream convertToStream(VectorSchemaRoot root, RootAllocator allocator)
      throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
      writer.start();
      writer.writeBatch();
      writer.end();
    }

    ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());
    ArrowStreamReader reader = new ArrowStreamReader(in, allocator);

    ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator);
    Data.exportArrayStream(allocator, reader, stream);

    return stream;
  }

  @Test
  public void testMergeInsertWithoutIndex() throws Exception {
    // Verify that merge insert with useIndex=false still completes and
    // produces results consistent with the default (useIndex=true).

    try (VectorSchemaRoot source = buildSource(testDataset.getSchema(), allocator)) {
      try (ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            dataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedUpdateAll()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.InsertAll)
                    .withUseIndex(false),
                sourceStream);

        Assertions.assertEquals(
            "{0=Source 0, 1=Source 1, 2=Source 2, 3=Person 3, 4=Person 4, 7=Source 7, 8=Source 8, 9=Source 9}",
            readAll(result.dataset()).toString(),
            "merge insert with useIndex=false should produce correct upsert results");
      }
    }
  }

  @ParameterizedTest
  @CsvSource({"Auto, 2", "RewriteRows, 3"})
  public void testWriteModeOnIndexedPartialUpdate(
      MergeInsertParams.MergeWriteMode writeMode, int expectedFragments) throws Exception {
    try (Dataset wideDataset = createWideDataset()) {
      createIdIndex(wideDataset);

      try (VectorSchemaRoot source = buildPartialSource(1, 4);
          ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            wideDataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedUpdateAll()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing)
                    .withWriteMode(writeMode),
                sourceStream);

        Assertions.assertEquals(2, result.stats().numUpdatedRows());
        try (Dataset merged = result.dataset()) {
          Assertions.assertEquals(expectedFragments, merged.getFragments().size());
          Assertions.assertEquals(WIDE_AFTER_UPDATE, readWide(merged).toString());
        }
      }
    }
  }

  @Test
  public void testWriteModeRewriteColumns() throws Exception {
    try (Dataset wideDataset = createWideDataset()) {
      try (VectorSchemaRoot source = buildPartialSource(1, 4);
          ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        MergeInsertResult result =
            wideDataset.mergeInsert(
                new MergeInsertParams(Collections.singletonList("id"))
                    .withMatchedUpdateAll()
                    .withNotMatched(MergeInsertParams.WhenNotMatched.DoNothing)
                    .withWriteMode(MergeInsertParams.MergeWriteMode.RewriteColumns),
                sourceStream);

        Assertions.assertEquals(2, result.stats().numUpdatedRows());
        try (Dataset merged = result.dataset()) {
          Assertions.assertEquals(2, merged.getFragments().size());
          // "score", which the source omits, is carried through untouched.
          Assertions.assertEquals(WIDE_AFTER_UPDATE, readWide(merged).toString());
        }
      }
    }
  }

  @Test
  public void testWriteModeRewriteColumnsRejectsInserts() throws Exception {

    try (Dataset wideDataset = createWideDataset()) {
      try (VectorSchemaRoot source = buildPartialSource(100);
          ArrowArrayStream sourceStream = convertToStream(source, allocator)) {
        Exception e =
            Assertions.assertThrows(
                Exception.class,
                () ->
                    wideDataset.mergeInsert(
                        // WhenNotMatched defaults to InsertAll.
                        new MergeInsertParams(Collections.singletonList("id"))
                            .withMatchedUpdateAll()
                            .withWriteMode(MergeInsertParams.MergeWriteMode.RewriteColumns)
                            .withConflictRetries(0),
                        sourceStream));
        Assertions.assertTrue(
            e.getMessage().contains("RewriteColumns cannot express"),
            "expected the RewriteColumns rejection, got: " + e.getMessage());
      }
    }
  }

  /** A dataset of 6 rows over 2 fragments, whose "score" column no partial source carries. */
  private Dataset createWideDataset() {
    String path = tempDir.resolve(UUID.randomUUID().toString()).toString();
    Dataset.create(allocator, path, WIDE_SCHEMA, new WriteParams.Builder().build()).close();

    List<FragmentMetadata> fragments;
    try (VectorSchemaRoot root = VectorSchemaRoot.create(WIDE_SCHEMA, allocator)) {
      root.allocateNew();
      IntVector idVector = (IntVector) root.getVector("id");
      VarCharVector nameVector = (VarCharVector) root.getVector("name");
      IntVector scoreVector = (IntVector) root.getVector("score");
      for (int i = 0; i < 6; i++) {
        idVector.setSafe(i, i);
        nameVector.setSafe(i, ("Person " + i).getBytes(StandardCharsets.UTF_8));
        scoreVector.setSafe(i, i * 10);
      }
      root.setRowCount(6);
      fragments =
          Fragment.create(
              path, allocator, root, new WriteParams.Builder().withMaxRowsPerFile(3).build());
    }

    Dataset wideDataset =
        Dataset.commit(allocator, path, new FragmentOperation.Append(fragments), Optional.of(1L));
    Assertions.assertEquals(2, wideDataset.getFragments().size());
    return wideDataset;
  }

  private void createIdIndex(Dataset dataset) {
    IndexParams indexParams =
        IndexParams.builder().setScalarIndexParams(ScalarIndexParams.create("btree", "{}")).build();
    dataset.createIndex(
        IndexOptions.builder(Collections.singletonList("id"), IndexType.BTREE, indexParams)
            .withIndexName("id_btree")
            .replace(true)
            .build());
  }

  /** A source over {@link #WIDE_SCHEMA} minus "score", renaming the rows it names. */
  private VectorSchemaRoot buildPartialSource(int... ids) {
    VectorSchemaRoot source = VectorSchemaRoot.create(PARTIAL_SOURCE_SCHEMA, allocator);
    source.allocateNew();
    IntVector idVector = (IntVector) source.getVector("id");
    VarCharVector nameVector = (VarCharVector) source.getVector("name");
    for (int i = 0; i < ids.length; i++) {
      idVector.setSafe(i, ids[i]);
      nameVector.setSafe(i, ("Updated " + ids[i]).getBytes(StandardCharsets.UTF_8));
    }
    source.setRowCount(ids.length);
    return source;
  }

  /** Reads a {@link #WIDE_SCHEMA} dataset as id -&gt; "name/score". */
  private TreeMap<Integer, String> readWide(Dataset dataset) throws Exception {
    TreeMap<Integer, String> rows = new TreeMap<>();
    try (ArrowReader reader = dataset.newScan().scanBatches()) {
      while (reader.loadNextBatch()) {
        VectorSchemaRoot batch = reader.getVectorSchemaRoot();
        IntVector ids = (IntVector) batch.getVector("id");
        VarCharVector names = (VarCharVector) batch.getVector("name");
        IntVector scores = (IntVector) batch.getVector("score");
        for (int i = 0; i < batch.getRowCount(); i++) {
          rows.put(ids.get(i), new String(names.get(i)) + "/" + scores.get(i));
        }
      }
    }
    return rows;
  }

  private TreeMap<Integer, String> readAll(Dataset dataset) throws Exception {
    try (ArrowReader reader = dataset.newScan().scanBatches()) {
      TreeMap<Integer, String> map = new TreeMap<>();

      while (reader.loadNextBatch()) {
        VectorSchemaRoot batch = reader.getVectorSchemaRoot();
        for (int i = 0; i < batch.getRowCount(); i++) {
          IntVector idVector = (IntVector) batch.getVector("id");
          VarCharVector nameVector = (VarCharVector) batch.getVector("name");
          map.put(idVector.get(i), new String(nameVector.get(i)));
        }
      }

      return map;
    }
  }
}
