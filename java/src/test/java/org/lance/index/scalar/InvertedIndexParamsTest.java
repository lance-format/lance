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
package org.lance.index.scalar;

import org.lance.DocumentGranularity;
import org.lance.util.JsonUtils;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InvertedIndexParamsTest {

  @Test
  void testIcuSplitTokenizerVariant() {
    ScalarIndexParams params = InvertedIndexParams.builder().baseTokenizer("icu/split").build();

    assertEquals("inverted", params.getIndexType());
    String jsonParams = params.getJsonParams().orElseThrow(AssertionError::new);
    assertTrue(jsonParams.contains("\"base_tokenizer\":\"icu/split\""));
  }

  @Test
  void defaultBlockSizeIsSerialized() {
    ScalarIndexParams params = InvertedIndexParams.builder().build();

    Map<String, Object> json = JsonUtils.fromJson(params.getJsonParams().orElseThrow());
    assertEquals(128, ((Number) json.get("block_size")).intValue());
    assertEquals("row", json.get("document_granularity"));
    assertEquals(40, ((Number) json.get("max_token_length")).intValue());
  }

  @Test
  void documentGranularityIsSerialized() {
    ScalarIndexParams params =
        InvertedIndexParams.builder().documentGranularity(DocumentGranularity.LIST_ELEMENT).build();

    Map<String, Object> json = JsonUtils.fromJson(params.getJsonParams().orElseThrow());
    assertEquals("list_element", json.get("document_granularity"));
  }

  @Test
  void blockSizeIsSerialized() {
    ScalarIndexParams params = InvertedIndexParams.builder().blockSize(128).build();

    assertEquals("inverted", params.getIndexType());
    Map<String, Object> json = JsonUtils.fromJson(params.getJsonParams().orElseThrow());
    assertEquals(128, ((Number) json.get("block_size")).intValue());
  }

  @Test
  void invalidBlockSizeIsRejected() {
    assertThrows(
        IllegalArgumentException.class, () -> InvertedIndexParams.builder().blockSize(129));
    assertThrows(
        IllegalArgumentException.class, () -> InvertedIndexParams.builder().blockSize(512));
  }

  @Test
  void formatVersionThreeSupportsBothBlockSizes() {
    for (int blockSize : new int[] {128, 256}) {
      ScalarIndexParams params =
          InvertedIndexParams.builder().blockSize(blockSize).formatVersion(3).build();
      Map<String, Object> json = JsonUtils.fromJson(params.getJsonParams().orElseThrow());
      assertEquals(blockSize, ((Number) json.get("block_size")).intValue());
      assertEquals(3, ((Number) json.get("format_version")).intValue());
    }

    assertThrows(
        IllegalArgumentException.class,
        () -> InvertedIndexParams.builder().blockSize(256).formatVersion(2).build());
  }

  @Test
  void formatVersionFourIsRejected() {
    assertThrows(
        IllegalArgumentException.class, () -> InvertedIndexParams.builder().formatVersion(4));
  }

  @Test
  void codeAnalyzerRequiresFormatVersionThreeWhenExplicit() {
    ScalarIndexParams params =
        InvertedIndexParams.builder().baseTokenizer("code").formatVersion(3).build();
    Map<String, Object> json = JsonUtils.fromJson(params.getJsonParams().orElseThrow());
    assertEquals(3, ((Number) json.get("format_version")).intValue());

    assertThrows(
        IllegalArgumentException.class,
        () -> InvertedIndexParams.builder().baseTokenizer("code").formatVersion(2).build());
    assertThrows(
        IllegalArgumentException.class,
        () -> InvertedIndexParams.builder().analyzer("code").formatVersion(1).build());
    assertThrows(
        IllegalArgumentException.class,
        () -> InvertedIndexParams.builder().analyzer("code").formatVersion(2).build());
  }

  @Test
  void additionalBuildParametersAreSerialized() {
    ScalarIndexParams params =
        InvertedIndexParams.builder()
            .analyzer("code")
            .lanceTokenizer("text")
            .splitIdentifiers(true)
            .splitOnNumerics(false)
            .preserveOriginal(true)
            .indexOperators(true)
            .memoryLimit(4096)
            .numWorkers(4)
            .formatVersion(3)
            .build();

    Map<String, Object> json = JsonUtils.fromJson(params.getJsonParams().orElseThrow());
    assertEquals("code", json.get("analyzer"));
    assertEquals("text", json.get("lance_tokenizer"));
    assertEquals(true, json.get("split_identifiers"));
    assertEquals(false, json.get("split_on_numerics"));
    assertEquals(true, json.get("preserve_original"));
    assertEquals(true, json.get("index_operators"));
    assertEquals(4096L, ((Number) json.get("memory_limit")).longValue());
    assertEquals(4, ((Number) json.get("num_workers")).intValue());
  }

  @Test
  void nullMaxTokenLengthIsSerializedAsNull() {
    ScalarIndexParams params = InvertedIndexParams.builder().maxTokenLength(null).build();

    Map<String, Object> json = JsonUtils.fromJson(params.getJsonParams().orElseThrow());
    assertTrue(json.containsKey("max_token_length"));
    assertNull(json.get("max_token_length"));
  }

  @Test
  void zeroResourceParametersAreSerialized() {
    ScalarIndexParams params = InvertedIndexParams.builder().memoryLimit(0).numWorkers(0).build();

    Map<String, Object> json = JsonUtils.fromJson(params.getJsonParams().orElseThrow());
    assertEquals(0L, ((Number) json.get("memory_limit")).longValue());
    assertEquals(0, ((Number) json.get("num_workers")).intValue());
  }

  @Test
  void negativeResourceParametersAreRejected() {
    assertThrows(
        IllegalArgumentException.class, () -> InvertedIndexParams.builder().memoryLimit(-1));
    assertThrows(
        IllegalArgumentException.class, () -> InvertedIndexParams.builder().numWorkers(-1));
  }
}
