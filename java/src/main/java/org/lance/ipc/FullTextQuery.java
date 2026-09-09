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

import org.lance.DocumentGranularity;

import com.google.common.base.MoreObjects;
import org.apache.arrow.util.Preconditions;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Base type for full text search queries used by Lance scanner.
 *
 * <p>Match and phrase overloads without a {@link DocumentGranularity} infer the unique indexed
 * granularity for the field. They use row documents when no index exists and fail as ambiguous when
 * row and list-element indexes coexist.
 */
public abstract class FullTextQuery {
  public enum Type {
    MATCH,
    MATCH_PHRASE,
    BOOST,
    MULTI_MATCH,
    BOOLEAN
  }

  public enum Operator {
    AND,
    OR
  }

  public enum Occur {
    /** The clause may match and contributes its score when it does. */
    SHOULD,
    /** The clause must match and contributes its score. */
    MUST,
    /** The clause must not match and never contributes to the score. */
    MUST_NOT
  }

  public static final class BooleanClause {
    private final Occur occur;
    private final FullTextQuery query;

    public BooleanClause(Occur occur, FullTextQuery query) {
      this.occur = Objects.requireNonNull(occur, "occur must not be null");
      this.query = Objects.requireNonNull(query, "query must not be null");
    }

    public Occur getOccur() {
      return occur;
    }

    public FullTextQuery getQuery() {
      return query;
    }
  }

  public abstract Type getType();

  public static FullTextQuery match(String queryText, String column) {
    return new MatchQuery(
        queryText, column, 1.0f, Optional.empty(), 50, Operator.OR, 0, Optional.empty());
  }

  public static FullTextQuery match(
      String queryText, String column, DocumentGranularity documentGranularity) {
    return match(
        queryText, column, 1.0f, Optional.empty(), 50, Operator.OR, 0, documentGranularity);
  }

  public static FullTextQuery match(
      String queryText,
      String column,
      float boost,
      Optional<Integer> fuzziness,
      int maxExpansions,
      Operator operator,
      int prefixLength) {
    return new MatchQuery(
        queryText,
        column,
        boost,
        fuzziness,
        maxExpansions,
        operator,
        prefixLength,
        Optional.empty());
  }

  public static FullTextQuery match(
      String queryText,
      String column,
      float boost,
      Optional<Integer> fuzziness,
      int maxExpansions,
      Operator operator,
      int prefixLength,
      DocumentGranularity documentGranularity) {
    return new MatchQuery(
        queryText,
        column,
        boost,
        fuzziness,
        maxExpansions,
        operator,
        prefixLength,
        Optional.of(
            Objects.requireNonNull(documentGranularity, "documentGranularity must not be null")));
  }

  public static FullTextQuery phrase(String queryText, String column) {
    return new PhraseQuery(queryText, column, 0, Optional.empty());
  }

  public static FullTextQuery phrase(
      String queryText, String column, DocumentGranularity documentGranularity) {
    return phrase(queryText, column, 0, documentGranularity);
  }

  public static FullTextQuery phrase(String queryText, String column, int slop) {
    return new PhraseQuery(queryText, column, slop, Optional.empty());
  }

  public static FullTextQuery phrase(
      String queryText, String column, int slop, DocumentGranularity documentGranularity) {
    return new PhraseQuery(
        queryText,
        column,
        slop,
        Optional.of(
            Objects.requireNonNull(documentGranularity, "documentGranularity must not be null")));
  }

  public static FullTextQuery multiMatch(String queryText, List<String> columns) {
    return multiMatch(queryText, columns, null, Operator.OR);
  }

  public static FullTextQuery multiMatch(
      String queryText, List<String> columns, List<Float> boosts, Operator operator) {
    return new MultiMatchQuery(queryText, columns, boosts, operator);
  }

  public static FullTextQuery boost(FullTextQuery positive, FullTextQuery negative) {
    return boost(positive, negative, 0.5f);
  }

  public static FullTextQuery boost(
      FullTextQuery positive, FullTextQuery negative, float negativeBoost) {
    return new BoostQuery(positive, negative, negativeBoost);
  }

  public static FullTextQuery booleanQuery(List<BooleanClause> clauses) {
    return new BooleanQuery(clauses);
  }

  /** Match query on a single column. */
  public static final class MatchQuery extends FullTextQuery {
    private final String queryText;
    private final String column;
    private final float boost;
    private final Optional<Integer> fuzziness;
    private final int maxExpansions;
    private final Operator operator;
    private final int prefixLength;
    private final Optional<DocumentGranularity> documentGranularity;

    MatchQuery(
        String queryText,
        String column,
        float boost,
        Optional<Integer> fuzziness,
        int maxExpansions,
        Operator operator,
        int prefixLength,
        Optional<DocumentGranularity> documentGranularity) {
      Preconditions.checkArgument(
          queryText != null && !queryText.isEmpty(), "queryText must not be null or empty");
      Preconditions.checkArgument(
          column != null && !column.isEmpty(), "column must not be null or empty");
      Preconditions.checkArgument(maxExpansions >= 1, "maxExpansions must be >= 1");
      Preconditions.checkArgument(prefixLength >= 0, "prefixLength must be >= 0");

      this.queryText = queryText;
      this.column = column;
      this.boost = boost;
      this.fuzziness = fuzziness;
      this.maxExpansions = maxExpansions;
      this.operator = operator == null ? Operator.OR : operator;
      this.prefixLength = prefixLength;
      this.documentGranularity = Objects.requireNonNull(documentGranularity);
    }

    @Override
    public Type getType() {
      return Type.MATCH;
    }

    public String getQueryText() {
      return queryText;
    }

    public String getColumn() {
      return column;
    }

    public float getBoost() {
      return boost;
    }

    public Optional<Integer> getFuzziness() {
      return fuzziness;
    }

    public int getMaxExpansions() {
      return maxExpansions;
    }

    public Operator getOperator() {
      return operator;
    }

    public int getPrefixLength() {
      return prefixLength;
    }

    /** Returns the explicit granularity, or empty when query planning should infer it. */
    public Optional<DocumentGranularity> getDocumentGranularity() {
      return documentGranularity;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof MatchQuery)) return false;
      MatchQuery other = (MatchQuery) o;
      return Float.compare(boost, other.boost) == 0
          && maxExpansions == other.maxExpansions
          && prefixLength == other.prefixLength
          && operator == other.operator
          && Objects.equals(documentGranularity, other.documentGranularity)
          && Objects.equals(queryText, other.queryText)
          && Objects.equals(column, other.column)
          && Objects.equals(fuzziness, other.fuzziness);
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          queryText,
          column,
          boost,
          fuzziness,
          maxExpansions,
          operator,
          prefixLength,
          documentGranularity);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("type", getType())
          .add("queryText", queryText)
          .add("column", column)
          .add("boost", boost)
          .add("fuzziness", fuzziness)
          .add("maxExpansions", maxExpansions)
          .add("operator", operator)
          .add("prefixLength", prefixLength)
          .add("documentGranularity", documentGranularity)
          .toString();
    }
  }

  /** Phrase query on a single column. */
  public static final class PhraseQuery extends FullTextQuery {
    private final String queryText;
    private final String column;
    private final int slop;
    private final Optional<DocumentGranularity> documentGranularity;

    PhraseQuery(
        String queryText,
        String column,
        int slop,
        Optional<DocumentGranularity> documentGranularity) {
      Preconditions.checkArgument(
          queryText != null && !queryText.isEmpty(), "queryText must not be null or empty");
      Preconditions.checkArgument(
          column != null && !column.isEmpty(), "column must not be null or empty");
      Preconditions.checkArgument(slop >= 0, "slop must be >= 0");

      this.queryText = queryText;
      this.column = column;
      this.slop = slop;
      this.documentGranularity = Objects.requireNonNull(documentGranularity);
    }

    @Override
    public Type getType() {
      return Type.MATCH_PHRASE;
    }

    public String getQueryText() {
      return queryText;
    }

    public String getColumn() {
      return column;
    }

    public int getSlop() {
      return slop;
    }

    /** Returns the explicit granularity, or empty when query planning should infer it. */
    public Optional<DocumentGranularity> getDocumentGranularity() {
      return documentGranularity;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof PhraseQuery)) return false;
      PhraseQuery other = (PhraseQuery) o;
      return slop == other.slop
          && Objects.equals(documentGranularity, other.documentGranularity)
          && Objects.equals(queryText, other.queryText)
          && Objects.equals(column, other.column);
    }

    @Override
    public int hashCode() {
      return Objects.hash(queryText, column, slop, documentGranularity);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("type", getType())
          .add("queryText", queryText)
          .add("column", column)
          .add("slop", slop)
          .add("documentGranularity", documentGranularity)
          .toString();
    }
  }

  /** Multi-match query across multiple columns. */
  public static final class MultiMatchQuery extends FullTextQuery {
    private final String queryText;
    private final List<String> columns;
    private final Optional<List<Float>> boosts;
    private final Operator operator;

    MultiMatchQuery(String queryText, List<String> columns, List<Float> boosts, Operator operator) {
      Preconditions.checkArgument(
          queryText != null && !queryText.isEmpty(), "queryText must not be null or empty");
      Preconditions.checkArgument(
          columns != null && !columns.isEmpty(), "columns must not be null or empty");

      this.queryText = queryText;
      this.columns =
          Collections.unmodifiableList(new java.util.ArrayList<>(Objects.requireNonNull(columns)));
      this.boosts = boosts == null ? Optional.empty() : Optional.of(boosts);
      this.operator = operator == null ? Operator.OR : operator;
    }

    @Override
    public Type getType() {
      return Type.MULTI_MATCH;
    }

    public String getQueryText() {
      return queryText;
    }

    public List<String> getColumns() {
      return columns;
    }

    public Optional<List<Float>> getBoosts() {
      return boosts;
    }

    public Operator getOperator() {
      return operator;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof MultiMatchQuery)) return false;
      MultiMatchQuery other = (MultiMatchQuery) o;
      return operator == other.operator
          && Objects.equals(queryText, other.queryText)
          && Objects.equals(columns, other.columns)
          && Objects.equals(boosts, other.boosts);
    }

    @Override
    public int hashCode() {
      return Objects.hash(queryText, columns, operator, boosts);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("type", getType())
          .add("queryText", queryText)
          .add("columns", columns)
          .add("boosts", boosts)
          .add("operator", operator)
          .toString();
    }
  }

  /** Boost query combining positive and negative queries. */
  public static final class BoostQuery extends FullTextQuery {
    private final FullTextQuery positive;
    private final FullTextQuery negative;
    private final Float negativeBoost;

    BoostQuery(FullTextQuery positive, FullTextQuery negative, float negativeBoost) {
      this.positive = Objects.requireNonNull(positive, "positive must not be null");
      this.negative = Objects.requireNonNull(negative, "negative must not be null");
      this.negativeBoost = negativeBoost;
    }

    @Override
    public Type getType() {
      return Type.BOOST;
    }

    public FullTextQuery getPositive() {
      return positive;
    }

    public FullTextQuery getNegative() {
      return negative;
    }

    public float getNegativeBoost() {
      return negativeBoost;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("type", getType())
          .add("positive", positive)
          .add("negative", negative)
          .add("negativeBoost", negativeBoost)
          .toString();
    }
  }

  /** Boolean query composed of multiple clauses. */
  public static final class BooleanQuery extends FullTextQuery {
    private final List<BooleanClause> clauses;

    BooleanQuery(List<BooleanClause> clauses) {
      Preconditions.checkArgument(
          clauses != null && !clauses.isEmpty(), "clauses must not be null or empty");
      this.clauses =
          Collections.unmodifiableList(new java.util.ArrayList<>(Objects.requireNonNull(clauses)));
    }

    @Override
    public Type getType() {
      return Type.BOOLEAN;
    }

    public List<BooleanClause> getClauses() {
      return clauses;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("type", getType())
          .add("clauses", clauses)
          .toString();
    }
  }
}
