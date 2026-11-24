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

import org.lance.util.JsonUtils;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Builder-style configuration for MinHash LSH scalar index parameters.
 *
 * <p>A MinHash LSH index finds near-duplicate texts in a string column by their estimated Jaccard
 * similarity. Every segment of one index must be built with the same parameters: segments built
 * with different parameters are rejected when they are merged or committed together.
 *
 * <p>The setters only reject non-positive values. The remaining constraints (ranges, {@code
 * numHashes} being a multiple of {@code numBands}, supported tokenizer settings) are checked when
 * the index is created.
 *
 * <p>Example, building one segment per worker and committing them as one index:
 *
 * <pre>{@code
 * IndexParams params =
 *     IndexParams.builder()
 *         .setScalarIndexParams(
 *             MinHashLshIndexParams.builder()
 *                 .numHashes(128)
 *                 .numBands(16)
 *                 .shingleSize(3)
 *                 .tokenizer(InvertedIndexParams.builder().baseTokenizer("simple"))
 *                 .build())
 *         .build();
 * // On each worker:
 * Index segment =
 *     dataset.createIndex(
 *         IndexOptions.builder(List.of("text"), IndexType.MINHASH_LSH, params)
 *             .withIndexName("text_idx")
 *             .withFragmentIds(fragmentIds)
 *             .build());
 * // With every worker's segment, optionally after mergeExistingIndexSegments:
 * dataset.commitExistingIndexSegments("text_idx", "text", segments);
 * }</pre>
 */
public final class MinHashLshIndexParams {

  private static final String INDEX_TYPE = "minhashlsh";

  private MinHashLshIndexParams() {}

  /**
   * Create a new builder for MinHash LSH index parameters.
   *
   * @return a new {@link Builder}
   */
  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private Integer numHashes;
    private Integer numBands;
    private Integer shingleSize;
    private Map<String, Object> tokenizer;

    /**
     * Configure the number of MinHash values per signature.
     *
     * <p>The default is {@code 128}. Must be in {@code [1, 4096]} and a multiple of {@code
     * numBands}. More hashes make the similarity estimate more precise (error about {@code 1 /
     * sqrt(numHashes)}) and the signatures larger.
     *
     * @param numHashes number of hashes per signature, must be positive
     * @return this builder
     * @throws IllegalArgumentException if {@code numHashes} is not positive
     */
    public Builder numHashes(int numHashes) {
      if (numHashes <= 0) {
        throw new IllegalArgumentException("numHashes must be positive");
      }
      this.numHashes = numHashes;
      return this;
    }

    /**
     * Configure the number of LSH bands.
     *
     * <p>The default is {@code 16}. Must be in {@code [1, 256]}. Two rows become candidates when
     * all {@code numHashes / numBands} values of at least one band agree, so together with {@code
     * numHashes} it sets the approximate similarity threshold {@code (1 / numBands) ^ (numBands /
     * numHashes)}.
     *
     * @param numBands number of bands, must be positive
     * @return this builder
     * @throws IllegalArgumentException if {@code numBands} is not positive
     */
    public Builder numBands(int numBands) {
      if (numBands <= 0) {
        throw new IllegalArgumentException("numBands must be positive");
      }
      this.numBands = numBands;
      return this;
    }

    /**
     * Configure the number of consecutive tokens joined into one shingle.
     *
     * <p>The default is {@code 3}. Shorter shingles tolerate more edits but let unrelated texts
     * that share common phrases look alike. Texts shorter than this produce a single shingle of all
     * their tokens.
     *
     * @param shingleSize tokens per shingle, must be positive
     * @return this builder
     * @throws IllegalArgumentException if {@code shingleSize} is not positive
     */
    public Builder shingleSize(int shingleSize) {
      if (shingleSize <= 0) {
        throw new IllegalArgumentException("shingleSize must be positive");
      }
      this.shingleSize = shingleSize;
      return this;
    }

    /**
     * Configure how text is split into tokens, using the full-text search tokenizer settings.
     *
     * <p>Only the tokenizer settings of {@code tokenizer} are read, when this method is called:
     * analyzer, base tokenizer, language, maximum token length, lower casing, stemming, stop words,
     * ASCII folding, N-gram and code analyzer settings. Full-text search build settings such as
     * {@code withPosition}, {@code blockSize}, {@code memoryLimit}, {@code numWorkers}, {@code
     * formatVersion} and {@code documentGranularity} are ignored.
     *
     * <p>Stemming and stop word removal default to off, unlike full-text search, because merging
     * different words inflates similarity. Custom stop words, {@code lanceTokenizer}, and
     * tokenizers that load language models (Jieba, Lindera) are rejected when the index is built,
     * because the index cannot record them.
     *
     * @param tokenizer full-text search parameters whose tokenizer settings to use
     * @return this builder
     */
    public Builder tokenizer(InvertedIndexParams.Builder tokenizer) {
      Objects.requireNonNull(tokenizer, "tokenizer must not be null");
      this.tokenizer = tokenizer.tokenizerParams();
      return this;
    }

    /** Build a {@link ScalarIndexParams} instance for a MinHash LSH index. */
    public ScalarIndexParams build() {
      Map<String, Object> params = new HashMap<>();
      if (numHashes != null) {
        params.put("num_hashes", numHashes);
      }
      if (numBands != null) {
        params.put("num_bands", numBands);
      }
      if (shingleSize != null) {
        params.put("shingle_size", shingleSize);
      }
      if (tokenizer != null) {
        params.put("tokenizer", tokenizer);
      }

      String json = JsonUtils.toJson(params);
      return ScalarIndexParams.create(INDEX_TYPE, json);
    }
  }
}
