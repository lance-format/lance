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

import com.google.common.base.Preconditions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Builder-style configuration for inverted (full-text) scalar index parameters. */
public final class InvertedIndexParams {

  private static final String INDEX_TYPE = "inverted";

  private InvertedIndexParams() {}

  /**
   * Create a new builder for inverted index parameters.
   *
   * @return a new {@link Builder}
   */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder for inverted scalar index parameters. */
  public static final class Builder {
    private String analyzer;
    private String lanceTokenizer;
    private String baseTokenizer;
    private String language;
    private Boolean withPosition;
    private Integer maxTokenLength = 40;
    private Boolean lowerCase;
    private Boolean stem;
    private Boolean removeStopWords;
    private List<String> customStopWords;
    private Boolean asciiFolding;
    private Integer minNgramLength;
    private Integer maxNgramLength;
    private Boolean prefixOnly;
    private Integer blockSize = 128;
    private Boolean splitIdentifiers;
    private Boolean splitOnNumerics;
    private Boolean preserveOriginal;
    private Boolean indexOperators;
    private Long memoryLimit;
    private Integer numWorkers;
    private Integer formatVersion;
    private DocumentGranularity documentGranularity = DocumentGranularity.ROW;

    /**
     * Configure the analyzer preset.
     *
     * <p>Supported values are {@code "text"} and {@code "code"}. The code analyzer selects the code
     * tokenizer defaults and requires FTS format v3. If unset, the analyzer is inferred from {@link
     * #baseTokenizer(String)}.
     *
     * @param analyzer analyzer preset
     * @return this builder
     */
    public Builder analyzer(String analyzer) {
      Objects.requireNonNull(analyzer, "analyzer must not be null");
      if (analyzer.isEmpty()) {
        throw new IllegalArgumentException("analyzer must not be empty");
      }
      this.analyzer = analyzer;
      return this;
    }

    /**
     * Configure the document-level tokenizer used before lexical tokenization.
     *
     * <p>Supported values are {@code "text"} for plain strings and {@code "json"} for JSON strings.
     * If unset, Lance infers the document tokenizer from the Arrow field type.
     *
     * @param lanceTokenizer document-level tokenizer
     * @return this builder
     */
    public Builder lanceTokenizer(String lanceTokenizer) {
      Objects.requireNonNull(lanceTokenizer, "lanceTokenizer must not be null");
      if (lanceTokenizer.isEmpty()) {
        throw new IllegalArgumentException("lanceTokenizer must not be empty");
      }
      this.lanceTokenizer = lanceTokenizer;
      return this;
    }

    /**
     * Configure the base tokenizer.
     *
     * <p>Supported values include:
     *
     * <ul>
     *   <li>{@code "simple"} (default): splits tokens on whitespace and punctuation
     *   <li>{@code "whitespace"}: splits tokens on whitespace
     *   <li>{@code "raw"}: no tokenization
     *   <li>{@code "ngram"}: N-Gram tokenizer
     *   <li>{@code "code"}: code-aware tokenizer
     *   <li>{@code "icu"}: ICU dictionary-based Unicode word segmentation
     *   <li>{@code "icu/split"}: ICU segmentation with simple-style delimiter splitting
     *   <li>{@code "lindera/*"}: Lindera tokenizer
     *   <li>{@code "jieba/*"}: Jieba tokenizer
     * </ul>
     *
     * <p>Lindera and Jieba tokenizers load their language models from the directory configured by
     * {@code LANCE_LANGUAGE_MODEL_HOME}, or from Lance's platform-specific default language model
     * directory. The tokenizer suffix selects a model directory, for example {@code jieba/default}.
     * The {@code code} tokenizer requires FTS format v3.
     *
     * @param baseTokenizer tokenizer identifier string
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder baseTokenizer(String baseTokenizer) {
      Objects.requireNonNull(baseTokenizer, "baseTokenizer must not be null");
      if (baseTokenizer.isEmpty()) {
        throw new IllegalArgumentException("baseTokenizer must not be empty");
      }
      this.baseTokenizer = baseTokenizer;
      return this;
    }

    /**
     * Configure the language used for stemming and stop words.
     *
     * @param language language name understood by Tantivy, for example {@code "English"}
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder language(String language) {
      Objects.requireNonNull(language, "language must not be null");
      if (language.isEmpty()) {
        throw new IllegalArgumentException("language must not be empty");
      }
      this.language = language;
      return this;
    }

    /**
     * Configure whether to store token positions in the index.
     *
     * @param withPosition whether to store term positions
     * @return this builder
     */
    public Builder withPosition(boolean withPosition) {
      this.withPosition = withPosition;
      return this;
    }

    /**
     * Configure the maximum token length.
     *
     * <p>The default is {@code 40}. Set this to {@code null} to disable the maximum token length
     * filter.
     *
     * @param maxTokenLength maximum token length, or {@code null} for no limit; non-null values
     *     must be positive
     * @return this builder
     * @throws IllegalArgumentException if {@code maxTokenLength} is not null and is not positive
     */
    public Builder maxTokenLength(Integer maxTokenLength) {
      if (maxTokenLength != null && maxTokenLength <= 0) {
        throw new IllegalArgumentException("maxTokenLength must be positive when specified");
      }
      this.maxTokenLength = maxTokenLength;
      return this;
    }

    /**
     * Configure whether to lower case tokens.
     *
     * @param lowerCase whether to lower case tokens
     * @return this builder
     */
    public Builder lowerCase(boolean lowerCase) {
      this.lowerCase = lowerCase;
      return this;
    }

    /**
     * Configure whether to apply stemming.
     *
     * @param stem whether to apply stemming
     * @return this builder
     */
    public Builder stem(boolean stem) {
      this.stem = stem;
      return this;
    }

    /**
     * Configure whether to remove stop words.
     *
     * @param removeStopWords whether to remove stop words
     * @return this builder
     */
    public Builder removeStopWords(boolean removeStopWords) {
      this.removeStopWords = removeStopWords;
      return this;
    }

    /**
     * Configure custom stop words. When set, these override the built-in stop word list for the
     * configured language.
     *
     * @param customStopWords list of stop words
     * @return this builder
     */
    public Builder customStopWords(List<String> customStopWords) {
      Objects.requireNonNull(customStopWords, "customStopWords must not be null");
      this.customStopWords = new ArrayList<>(customStopWords);
      return this;
    }

    /**
     * Configure whether to apply ASCII folding
     *
     * @param asciiFolding whether to enable ASCII folding
     * @return this builder
     */
    public Builder asciiFolding(boolean asciiFolding) {
      this.asciiFolding = asciiFolding;
      return this;
    }

    /**
     * Configure the minimum N-gram length (only used when {@code baseTokenizer = "ngram"}).
     *
     * @param minNgramLength minimum N-gram length, must be &gt; 0 and &lt;= {@code maxNgramLength}
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder minNgramLength(int minNgramLength) {
      if (minNgramLength <= 0) {
        throw new IllegalArgumentException("minNgramLength must be positive");
      }
      this.minNgramLength = minNgramLength;
      return this;
    }

    /**
     * Configure the maximum N-gram length (only used when {@code baseTokenizer = "ngram"}).
     *
     * @param maxNgramLength maximum N-gram length, must be &gt; 0 and &gt;= {@code minNgramLength}
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder maxNgramLength(int maxNgramLength) {
      if (maxNgramLength <= 0) {
        throw new IllegalArgumentException("maxNgramLength must be positive");
      }
      this.maxNgramLength = maxNgramLength;
      return this;
    }

    /**
     * Configure whether only prefix N-grams are generated (only used when {@code baseTokenizer =
     * "ngram"}).
     *
     * @param prefixOnly whether to generate only prefix N-grams
     * @return this builder
     */
    public Builder prefixOnly(boolean prefixOnly) {
      this.prefixOnly = prefixOnly;
      return this;
    }

    /**
     * Configure the number of documents in each compressed posting block.
     *
     * <p>Supported values are {@code 128} and {@code 256}. New indexes default to {@code 128} when
     * this is not set.
     *
     * <p>{@code blockSize = 256} requires FTS format v3. Format v3 also supports the default {@code
     * blockSize = 128}.
     *
     * @param blockSize posting block size
     * @return this builder
     * @throws IllegalArgumentException if {@code blockSize} is unsupported
     */
    public Builder blockSize(int blockSize) {
      if (blockSize != 128 && blockSize != 256) {
        throw new IllegalArgumentException("blockSize must be one of 128 or 256");
      }
      this.blockSize = blockSize;
      return this;
    }

    /**
     * Configure whether code identifiers are split into subwords.
     *
     * <p>This option is valid only with the {@code code} analyzer.
     *
     * @param splitIdentifiers whether to split identifiers
     * @return this builder
     */
    public Builder splitIdentifiers(boolean splitIdentifiers) {
      this.splitIdentifiers = splitIdentifiers;
      return this;
    }

    /**
     * Configure whether code identifier subwords are split at letter-number boundaries.
     *
     * <p>This option is valid only with the {@code code} analyzer.
     *
     * @param splitOnNumerics whether to split at numeric boundaries
     * @return this builder
     */
    public Builder splitOnNumerics(boolean splitOnNumerics) {
      this.splitOnNumerics = splitOnNumerics;
      return this;
    }

    /**
     * Configure whether complete code identifiers are indexed alongside their subwords.
     *
     * <p>This option is valid only with the {@code code} analyzer.
     *
     * @param preserveOriginal whether to preserve complete identifiers
     * @return this builder
     */
    public Builder preserveOriginal(boolean preserveOriginal) {
      this.preserveOriginal = preserveOriginal;
      return this;
    }

    /**
     * Configure whether code operators such as {@code ::}, {@code ->}, and {@code !=} are indexed.
     *
     * <p>This option is valid only with the {@code code} analyzer.
     *
     * @param indexOperators whether to index operators
     * @return this builder
     */
    public Builder indexOperators(boolean indexOperators) {
      this.indexOperators = indexOperators;
      return this;
    }

    /**
     * Configure the total memory limit in MiB for the build stage.
     *
     * <p>The limit is split evenly across FTS workers and is not persisted with the index. If
     * unset, each worker uses a 2 GiB build-time limit.
     *
     * <p>A value of {@code 0} is passed through to Rust.
     *
     * @param memoryLimit total memory limit in MiB, must be non-negative
     * @return this builder
     * @throws IllegalArgumentException if {@code memoryLimit} is negative
     */
    public Builder memoryLimit(long memoryLimit) {
      if (memoryLimit < 0) {
        throw new IllegalArgumentException("memoryLimit must be non-negative");
      }
      this.memoryLimit = memoryLimit;
      return this;
    }

    /**
     * Configure the number of workers used for the build stage.
     *
     * <p>The effective value is capped at the available compute-intensive CPU count and is not
     * persisted with the index. Rust clamps a value of {@code 0} to one worker.
     *
     * @param numWorkers requested worker count, must be non-negative
     * @return this builder
     * @throws IllegalArgumentException if {@code numWorkers} is negative
     */
    public Builder numWorkers(int numWorkers) {
      if (numWorkers < 0) {
        throw new IllegalArgumentException("numWorkers must be non-negative");
      }
      this.numWorkers = numWorkers;
      return this;
    }

    /**
     * This option has no effect because the Rust inverted-index builder does not support skipping
     * the partition merge stage.
     *
     * @param skipMerge whether to skip partition merge
     * @return this builder
     * @deprecated this option has no effect and will be removed in a future release
     */
    @Deprecated
    public Builder skipMerge(boolean skipMerge) {
      return this;
    }

    /**
     * Configure the on-disk FTS format version to write when creating a new index.
     *
     * <p>If unset, Lance uses {@code LANCE_FTS_FORMAT_VERSION} when present and otherwise selects
     * v3 for the code analyzer, {@code baseTokenizer = "code"}, or {@code blockSize = 256}, and v2
     * for other indexes. Format v3 supports both posting block sizes. Formats v1 and v2 support
     * only {@code blockSize = 128} and cannot be used with the code analyzer or code base
     * tokenizer.
     *
     * @param formatVersion FTS format version, must be 1, 2, or 3
     * @return this builder
     * @throws IllegalArgumentException
     */
    public Builder formatVersion(int formatVersion) {
      if (formatVersion != 1 && formatVersion != 2 && formatVersion != 3) {
        throw new IllegalArgumentException("formatVersion must be 1, 2, or 3");
      }
      this.formatVersion = formatVersion;
      return this;
    }

    /**
     * Configure the unit treated as one FTS document.
     *
     * <p>{@link DocumentGranularity#LIST_ELEMENT} uses each element of the deepest list on the
     * indexed field path as one document. The default is {@link DocumentGranularity#ROW}.
     *
     * @param documentGranularity document boundary semantics
     * @return this builder
     */
    public Builder documentGranularity(DocumentGranularity documentGranularity) {
      this.documentGranularity =
          Objects.requireNonNull(documentGranularity, "documentGranularity must not be null");
      return this;
    }

    /** Build a {@link ScalarIndexParams} instance for an inverted index. */
    public ScalarIndexParams build() {
      if (formatVersion != null) {
        Preconditions.checkArgument(
            formatVersion == 3 || blockSize == 128, "formatVersion 1 and 2 require blockSize 128");
        Preconditions.checkArgument(
            (!"code".equals(analyzer) && !"code".equals(baseTokenizer)) || formatVersion == 3,
            "code analyzer and baseTokenizer 'code' require formatVersion 3");
      }
      Map<String, Object> params = new HashMap<>();
      if (analyzer != null) {
        params.put("analyzer", analyzer);
      }
      if (lanceTokenizer != null) {
        params.put("lance_tokenizer", lanceTokenizer);
      }
      if (baseTokenizer != null) {
        params.put("base_tokenizer", baseTokenizer);
      }
      if (language != null) {
        params.put("language", language);
      }
      if (withPosition != null) {
        params.put("with_position", withPosition);
      }
      params.put("max_token_length", maxTokenLength);
      if (lowerCase != null) {
        params.put("lower_case", lowerCase);
      }
      if (stem != null) {
        params.put("stem", stem);
      }
      if (removeStopWords != null) {
        params.put("remove_stop_words", removeStopWords);
      }
      if (customStopWords != null) {
        params.put("custom_stop_words", new ArrayList<>(customStopWords));
      }
      if (asciiFolding != null) {
        params.put("ascii_folding", asciiFolding);
      }
      if (minNgramLength != null) {
        params.put("min_ngram_length", minNgramLength);
      }
      if (maxNgramLength != null) {
        Preconditions.checkArgument(
            minNgramLength == null || maxNgramLength >= minNgramLength,
            "maxNgramLength {} shouldn't less than minNgramLength {}",
            maxNgramLength,
            minNgramLength);
        params.put("max_ngram_length", maxNgramLength);
      }
      if (prefixOnly != null) {
        params.put("prefix_only", prefixOnly);
      }
      if (blockSize != null) {
        params.put("block_size", blockSize);
      }
      if (splitIdentifiers != null) {
        params.put("split_identifiers", splitIdentifiers);
      }
      if (splitOnNumerics != null) {
        params.put("split_on_numerics", splitOnNumerics);
      }
      if (preserveOriginal != null) {
        params.put("preserve_original", preserveOriginal);
      }
      if (indexOperators != null) {
        params.put("index_operators", indexOperators);
      }
      if (memoryLimit != null) {
        params.put("memory_limit", memoryLimit);
      }
      if (numWorkers != null) {
        params.put("num_workers", numWorkers);
      }
      if (formatVersion != null) {
        params.put("format_version", formatVersion);
      }
      params.put("document_granularity", documentGranularity.toRustString());

      String json = JsonUtils.toJson(params);
      return ScalarIndexParams.create(INDEX_TYPE, json);
    }
  }
}
