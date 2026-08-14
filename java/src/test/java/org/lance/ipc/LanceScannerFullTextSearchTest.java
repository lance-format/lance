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
package org.lance.ipc;

import org.lance.Dataset;
import org.lance.DocumentGranularity;
import org.lance.WriteParams;
import org.lance.index.IndexOptions;
import org.lance.index.IndexParams;
import org.lance.index.IndexType;
import org.lance.index.scalar.InvertedIndexParams;
import org.lance.index.scalar.ScalarIndexParams;

import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LanceScannerFullTextSearchTest {

  private static final List<String> DEFAULT_DOCUMENTS =
      Arrays.asList("hello world", "hello lance", "other text");

  private static ScalarIndexParams defaultIndexParams() {
    return ScalarIndexParams.create(
        "inverted",
        "{\"base_tokenizer\":\"simple\",\"language\":\"English\",\"with_position\":true}");
  }

  @Test
  void testMatchQuery() throws Exception {
    runFtsQuery(
        "memory://fts_java_match",
        FullTextQuery.match("hello", "doc", DocumentGranularity.ROW),
        2L);
  }

  @Test
  void testJiebaTokenizer() throws Exception {
    ScalarIndexParams indexParams =
        InvertedIndexParams.builder()
            .baseTokenizer("jieba/default")
            .stem(false)
            .removeStopWords(false)
            .build();

    runFtsQuery(
        "memory://fts_java_jieba",
        FullTextQuery.match("我们", "doc"),
        1L,
        Arrays.asList("我们都有光明的前途", "光明的前途"),
        indexParams);
  }

  @Test
  void testCodeAnalyzerBuilderCreatesIndex() throws Exception {
    ScalarIndexParams indexParams = InvertedIndexParams.builder().analyzer("code").build();

    runFtsQuery(
        "memory://fts_java_code_analyzer",
        FullTextQuery.match("helloWorld", "doc"),
        1L,
        Arrays.asList("helloWorld", "other_value"),
        indexParams);
  }

  @Test
  void testExplicitListElementGranularityReachesRustRouting() {
    RuntimeException error =
        assertThrows(
            RuntimeException.class,
            () ->
                runFtsQuery(
                    "memory://fts_java_list_element_validation",
                    FullTextQuery.match("hello", "doc", DocumentGranularity.LIST_ELEMENT),
                    0L));
    assertTrue(error.getMessage().contains("requested ListElement"), error.getMessage());
    assertTrue(error.getMessage().contains("'doc_idx' (Row)"), error.getMessage());
  }

  @Test
  void testPhraseQuery() throws Exception {
    runFtsQuery("memory://fts_java_phrase", FullTextQuery.phrase("hello world", "doc", 0), 1L);
  }

  @Test
  void testBoostQuery() throws Exception {
    FullTextQuery positive = FullTextQuery.match("hello", "doc");
    FullTextQuery negative = FullTextQuery.match("world", "doc");
    FullTextQuery boosted = FullTextQuery.boost(positive, negative, 0.3f);

    runFtsQuery("memory://fts_java_boost", boosted, 2L);
  }

  @Test
  void testMultiMatch() throws Exception {
    FullTextQuery multiMatch = FullTextQuery.multiMatch("hello", Arrays.asList("doc", "title"));
    runFtsQuery("memory://fts_java_multimatch", multiMatch, 3);
  }

  @Test
  void testCombinedFields() throws Exception {
    // "hello" appears in doc or title of every row, so all 3 match.
    FullTextQuery combined = FullTextQuery.combinedFields("hello", Arrays.asList("doc", "title"));
    runFtsQuery("memory://fts_java_combined", combined, 3);
  }

  @Test
  void testCombinedFieldsWithBoosts() throws Exception {
    FullTextQuery combined =
        FullTextQuery.combinedFields(
            "hello",
            Arrays.asList("doc", "title"),
            Arrays.asList(2.0f, 1.0f),
            FullTextQuery.Operator.OR);
    runFtsQuery("memory://fts_java_combined_boosts", combined, 3);
  }

  @Test
  void testCombinedFieldsInvalidBoostPropagates() throws Exception {
    // Per-column weights must be >= 1. The check lives in the Rust core and must
    // surface across the JNI boundary when the query runs.
    FullTextQuery combined =
        FullTextQuery.combinedFields(
            "hello",
            Arrays.asList("doc", "title"),
            Arrays.asList(0.5f, 1.0f),
            FullTextQuery.Operator.OR);
    withIndexedDataset(
        "memory://fts_java_combined_bad_boost",
        dataset -> {
          ScanOptions scanOptions = new ScanOptions.Builder().fullTextQuery(combined).build();
          IllegalArgumentException ex =
              assertThrows(
                  IllegalArgumentException.class,
                  () -> {
                    try (LanceScanner scanner = dataset.newScan(scanOptions);
                        ArrowReader arrowReader = scanner.scanBatches()) {
                      while (arrowReader.loadNextBatch()) {
                        // Drain batches to force query execution.
                      }
                    }
                  });
          assertTrue(
              ex.getMessage().contains("combined_fields boost for column 'doc'"),
              "expected invalid-boost validation error, got: " + ex.getMessage());
        });
  }

  @Test
  void testBooleanQuery() throws Exception {
    FullTextQuery.MatchQuery shouldMatch =
        (FullTextQuery.MatchQuery) FullTextQuery.match("hello", "doc");
    FullTextQuery.MatchQuery mustNotMatch =
        (FullTextQuery.MatchQuery) FullTextQuery.match("lance", "doc");

    FullTextQuery.BooleanClause shouldClause =
        new FullTextQuery.BooleanClause(FullTextQuery.Occur.SHOULD, shouldMatch);
    FullTextQuery.BooleanClause mustNotClause =
        new FullTextQuery.BooleanClause(FullTextQuery.Occur.MUST_NOT, mustNotMatch);

    FullTextQuery booleanQuery =
        FullTextQuery.booleanQuery(Arrays.asList(shouldClause, mustNotClause));

    runFtsQuery("memory://fts_java_boolean", booleanQuery, 1L);
  }

  private void runFtsQuery(String uri, FullTextQuery query, long expectedTotal) throws Exception {
    runFtsQuery(uri, query, expectedTotal, DEFAULT_DOCUMENTS, defaultIndexParams());
  }

  private void runFtsQuery(
      String uri,
      FullTextQuery query,
      long expectedTotal,
      List<String> documents,
      ScalarIndexParams scalarParams)
      throws Exception {
    withIndexedDataset(
        uri,
        documents,
        scalarParams,
        dataset -> {
          ScanOptions scanOptions = new ScanOptions.Builder().fullTextQuery(query).build();

          try (LanceScanner scanner = dataset.newScan(scanOptions)) {
            long total = 0L;
            try (ArrowReader arrowReader = scanner.scanBatches()) {
              while (arrowReader.loadNextBatch()) {
                total += arrowReader.getVectorSchemaRoot().getRowCount();
              }
            }
            assertEquals(expectedTotal, total);
          }
        });
  }

  /** Same dataset as the three-argument {@link #runFtsQuery}, without running a query on it. */
  private void withIndexedDataset(String uri, IndexedDatasetConsumer consumer) throws Exception {
    withIndexedDataset(uri, DEFAULT_DOCUMENTS, defaultIndexParams(), consumer);
  }

  /**
   * Create an in-memory dataset with two text columns ({@code doc}, {@code title}), each backed by
   * an inverted index, and hand it to {@code consumer}. Resources are released when the consumer
   * returns.
   */
  private void withIndexedDataset(
      String uri,
      List<String> documents,
      ScalarIndexParams scalarParams,
      IndexedDatasetConsumer consumer)
      throws Exception {
    Schema schema =
        new Schema(
            Arrays.asList(
                Field.nullable("doc", ArrowType.Utf8.INSTANCE),
                Field.nullable("title", ArrowType.Utf8.INSTANCE)),
            null);

    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
        VarCharVector docVector = (VarCharVector) root.getVector("doc");
        VarCharVector titleVector = (VarCharVector) root.getVector("title");

        docVector.allocateNew();
        titleVector.allocateNew();
        List<String> titles = Arrays.asList("bye world", "bye lance", "say hello");
        for (int i = 0; i < documents.size(); i++) {
          docVector.setSafe(i, documents.get(i).getBytes(StandardCharsets.UTF_8));
          titleVector.setSafe(i, titles.get(i).getBytes(StandardCharsets.UTF_8));
        }
        root.setRowCount(documents.size());

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (ArrowStreamWriter writer = new ArrowStreamWriter(root, null, out)) {
          writer.start();
          writer.writeBatch();
          writer.end();
        }

        byte[] arrowData = out.toByteArray();
        ByteArrayInputStream in = new ByteArrayInputStream(arrowData);
        try (ArrowStreamReader reader = new ArrowStreamReader(in, allocator);
            ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator)) {
          Data.exportArrayStream(allocator, reader, stream);

          WriteParams writeParams =
              new WriteParams.Builder().withMode(WriteParams.WriteMode.CREATE).build();

          try (Dataset dataset = Dataset.create(allocator, stream, uri, writeParams)) {
            IndexParams indexParams =
                IndexParams.builder().setScalarIndexParams(scalarParams).build();

            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList("doc"), IndexType.INVERTED, indexParams)
                    .withIndexName("doc_idx")
                    .build());

            dataset.createIndex(
                IndexOptions.builder(
                        Collections.singletonList("title"), IndexType.INVERTED, indexParams)
                    .withIndexName("title_idx")
                    .build());

            consumer.accept(dataset);
          }
        }
      }
    }
  }

  @FunctionalInterface
  private interface IndexedDatasetConsumer {
    void accept(Dataset dataset) throws Exception;
  }
}
