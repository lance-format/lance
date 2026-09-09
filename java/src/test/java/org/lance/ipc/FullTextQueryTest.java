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

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FullTextQueryTest {

  @Test
  void testMatchQueryDefaults() {
    FullTextQuery.MatchQuery q =
        (FullTextQuery.MatchQuery) FullTextQuery.match("hello world", "body");

    assertEquals(FullTextQuery.Type.MATCH, q.getType());
    assertEquals("hello world", q.getQueryText());
    assertEquals("body", q.getColumn());
    assertEquals(1.0f, q.getBoost());
    assertFalse(q.getFuzziness().isPresent());
    assertEquals(50, q.getMaxExpansions());
    assertEquals(FullTextQuery.Operator.OR, q.getOperator());
    assertEquals(0, q.getPrefixLength());
    assertEquals(Optional.empty(), q.getDocumentGranularity());
  }

  @Test
  void testMatchQueryCustomParameters() {
    FullTextQuery.MatchQuery q =
        (FullTextQuery.MatchQuery)
            FullTextQuery.match(
                "hello",
                "title",
                2.0f,
                Optional.of(1),
                10,
                FullTextQuery.Operator.AND,
                3,
                DocumentGranularity.LIST_ELEMENT);

    assertEquals(FullTextQuery.Type.MATCH, q.getType());
    assertEquals("hello", q.getQueryText());
    assertEquals("title", q.getColumn());
    assertEquals(2.0f, q.getBoost());
    assertEquals(Optional.of(1), q.getFuzziness());
    assertEquals(10, q.getMaxExpansions());
    assertEquals(FullTextQuery.Operator.AND, q.getOperator());
    assertEquals(3, q.getPrefixLength());
    assertEquals(Optional.of(DocumentGranularity.LIST_ELEMENT), q.getDocumentGranularity());
  }

  @Test
  void testPhraseQueryDefaults() {
    FullTextQuery.PhraseQuery q =
        (FullTextQuery.PhraseQuery) FullTextQuery.phrase("exact match", "content");

    assertEquals(FullTextQuery.Type.MATCH_PHRASE, q.getType());
    assertEquals("exact match", q.getQueryText());
    assertEquals("content", q.getColumn());
    assertEquals(0, q.getSlop());
    assertEquals(Optional.empty(), q.getDocumentGranularity());
  }

  @Test
  void testPhraseQueryCustomSlop() {
    FullTextQuery.PhraseQuery q =
        (FullTextQuery.PhraseQuery)
            FullTextQuery.phrase("ordered terms", "content", 2, DocumentGranularity.LIST_ELEMENT);

    assertEquals(FullTextQuery.Type.MATCH_PHRASE, q.getType());
    assertEquals("ordered terms", q.getQueryText());
    assertEquals("content", q.getColumn());
    assertEquals(2, q.getSlop());
    assertEquals(Optional.of(DocumentGranularity.LIST_ELEMENT), q.getDocumentGranularity());
  }

  @Test
  void testMultiMatchWithoutBoosts() {
    FullTextQuery.MultiMatchQuery q =
        (FullTextQuery.MultiMatchQuery)
            FullTextQuery.multiMatch("hello", Arrays.asList("title", "body"));

    assertEquals(FullTextQuery.Type.MULTI_MATCH, q.getType());
    assertEquals("hello", q.getQueryText());
    assertEquals(Arrays.asList("title", "body"), q.getColumns());
    assertFalse(q.getBoosts().isPresent());
    assertEquals(FullTextQuery.Operator.OR, q.getOperator());
  }

  @Test
  void testMultiMatchWithBoosts() {
    FullTextQuery.MultiMatchQuery q =
        (FullTextQuery.MultiMatchQuery)
            FullTextQuery.multiMatch(
                "hello",
                Arrays.asList("title", "body"),
                Arrays.asList(2.0f, 0.5f),
                FullTextQuery.Operator.AND);

    assertEquals(FullTextQuery.Type.MULTI_MATCH, q.getType());
    assertTrue(q.getBoosts().isPresent());
    assertEquals(2, q.getBoosts().get().size());
    assertEquals(2.0f, q.getBoosts().get().get(0));
    assertEquals(0.5f, q.getBoosts().get().get(1));
    assertEquals(FullTextQuery.Operator.AND, q.getOperator());
    assertNotNull(q.toString());
  }

  @Test
  void testBoostQuery() {
    FullTextQuery.MatchQuery positive =
        (FullTextQuery.MatchQuery) FullTextQuery.match("good", "body");
    FullTextQuery.MatchQuery negative =
        (FullTextQuery.MatchQuery) FullTextQuery.match("bad", "body");

    FullTextQuery.BoostQuery q =
        (FullTextQuery.BoostQuery) FullTextQuery.boost(positive, negative, 0.3f);

    assertEquals(FullTextQuery.Type.BOOST, q.getType());
    assertEquals(positive, q.getPositive());
    assertEquals(negative, q.getNegative());
    assertEquals(Float.valueOf(0.3f), q.getNegativeBoost());
  }

  @Test
  void testBooleanQuery() {
    FullTextQuery.MatchQuery match =
        (FullTextQuery.MatchQuery) FullTextQuery.match("hello", "body");
    FullTextQuery.MatchQuery mustNot =
        (FullTextQuery.MatchQuery) FullTextQuery.match("spam", "body");

    FullTextQuery.BooleanClause shouldClause =
        new FullTextQuery.BooleanClause(FullTextQuery.Occur.SHOULD, match);
    FullTextQuery.BooleanClause mustNotClause =
        new FullTextQuery.BooleanClause(FullTextQuery.Occur.MUST_NOT, mustNot);

    FullTextQuery.BooleanQuery q =
        (FullTextQuery.BooleanQuery)
            FullTextQuery.booleanQuery(Arrays.asList(shouldClause, mustNotClause));

    assertEquals(FullTextQuery.Type.BOOLEAN, q.getType());
    assertNotNull(q.getClauses());
    assertEquals(2, q.getClauses().size());
    assertEquals(FullTextQuery.Occur.SHOULD, q.getClauses().get(0).getOccur());
    assertEquals(FullTextQuery.Type.MATCH, q.getClauses().get(0).getQuery().getType());
    assertEquals(FullTextQuery.Occur.MUST_NOT, q.getClauses().get(1).getOccur());
  }

  @Test
  void testBooleanQuerySingleClause() {
    FullTextQuery.MatchQuery match =
        (FullTextQuery.MatchQuery) FullTextQuery.match("hello", "body");
    FullTextQuery.BooleanClause shouldClause =
        new FullTextQuery.BooleanClause(FullTextQuery.Occur.SHOULD, match);

    FullTextQuery.BooleanQuery q =
        (FullTextQuery.BooleanQuery)
            FullTextQuery.booleanQuery(Collections.singletonList(shouldClause));

    assertEquals(FullTextQuery.Type.BOOLEAN, q.getType());
    assertEquals(1, q.getClauses().size());
    assertEquals(FullTextQuery.Occur.SHOULD, q.getClauses().get(0).getOccur());
  }

  @Test
  void testMatchQueryEquality() {
    FullTextQuery a = FullTextQuery.match("hello", "body");
    FullTextQuery b = FullTextQuery.match("hello", "body");
    assertEquals(a, b, "Two MatchQuery instances with the same fields must be equal");
    assertEquals(
        a.hashCode(), b.hashCode(), "Equal MatchQuery instances must have the same hashCode");
  }

  @Test
  void testMatchQueryInequalityDifferentText() {
    FullTextQuery a = FullTextQuery.match("hello", "body");
    FullTextQuery b = FullTextQuery.match("world", "body");
    assertFalse(a.equals(b), "MatchQuery instances with different queryText must not be equal");
  }

  @Test
  void testMatchQueryInequalityDifferentColumn() {
    FullTextQuery a = FullTextQuery.match("hello", "body");
    FullTextQuery b = FullTextQuery.match("hello", "title");
    assertFalse(a.equals(b), "MatchQuery instances with different column must not be equal");
  }

  @Test
  void testMatchQueryEqualityWithOptions() {
    FullTextQuery a =
        FullTextQuery.match(
            "hello", "body", 2.0f, Optional.of(1), 10, FullTextQuery.Operator.AND, 2);
    FullTextQuery b =
        FullTextQuery.match(
            "hello", "body", 2.0f, Optional.of(1), 10, FullTextQuery.Operator.AND, 2);
    assertEquals(a, b, "MatchQuery instances with all custom fields equal must be equal");
    assertEquals(a.hashCode(), b.hashCode());
  }

  @Test
  void testMatchQueryInequalityDifferentFuzziness() {
    FullTextQuery a =
        FullTextQuery.match(
            "hello", "body", 1.0f, Optional.of(1), 50, FullTextQuery.Operator.OR, 0);
    FullTextQuery b =
        FullTextQuery.match(
            "hello", "body", 1.0f, Optional.of(2), 50, FullTextQuery.Operator.OR, 0);
    assertFalse(a.equals(b), "MatchQuery instances with different fuzziness must not be equal");
  }

  @Test
  void testPhraseQueryEquality() {
    FullTextQuery a = FullTextQuery.phrase("hello world", "body");
    FullTextQuery b = FullTextQuery.phrase("hello world", "body");
    assertEquals(a, b, "Two PhraseQuery instances with the same fields must be equal");
    assertEquals(a.hashCode(), b.hashCode());
  }

  @Test
  void testPhraseQueryEqualityWithSlop() {
    FullTextQuery a = FullTextQuery.phrase("hello world", "body", 2);
    FullTextQuery b = FullTextQuery.phrase("hello world", "body", 2);
    assertEquals(a, b, "PhraseQuery instances with same slop must be equal");
    assertEquals(a.hashCode(), b.hashCode());
  }

  @Test
  void testPhraseQueryInequalityDifferentSlop() {
    FullTextQuery a = FullTextQuery.phrase("hello world", "body", 1);
    FullTextQuery b = FullTextQuery.phrase("hello world", "body", 2);
    assertFalse(a.equals(b), "PhraseQuery instances with different slop must not be equal");
  }

  @Test
  void testPhraseQueryInequalityDifferentColumn() {
    FullTextQuery a = FullTextQuery.phrase("hello world", "body");
    FullTextQuery b = FullTextQuery.phrase("hello world", "title");
    assertFalse(a.equals(b));
  }

  @Test
  void testMultiMatchQueryEquality() {
    FullTextQuery a = FullTextQuery.multiMatch("hello", Arrays.asList("title", "body"));
    FullTextQuery b = FullTextQuery.multiMatch("hello", Arrays.asList("title", "body"));
    assertEquals(a, b, "Two MultiMatchQuery instances with the same fields must be equal");
    assertEquals(a.hashCode(), b.hashCode());
  }

  @Test
  void testMultiMatchQueryEqualityWithOperator() {
    FullTextQuery a =
        FullTextQuery.multiMatch(
            "hello", Arrays.asList("title", "body"), null, FullTextQuery.Operator.AND);
    FullTextQuery b =
        FullTextQuery.multiMatch(
            "hello", Arrays.asList("title", "body"), null, FullTextQuery.Operator.AND);
    assertEquals(a, b);
    assertEquals(a.hashCode(), b.hashCode());
  }

  @Test
  void testMultiMatchQueryInequalityDifferentOperator() {
    FullTextQuery a =
        FullTextQuery.multiMatch(
            "hello", Arrays.asList("title", "body"), null, FullTextQuery.Operator.AND);
    FullTextQuery b =
        FullTextQuery.multiMatch(
            "hello", Arrays.asList("title", "body"), null, FullTextQuery.Operator.OR);
    assertFalse(a.equals(b), "MultiMatchQuery instances with different operator must not be equal");
  }

  @Test
  void testMultiMatchQueryInequalityDifferentColumns() {
    FullTextQuery a = FullTextQuery.multiMatch("hello", Arrays.asList("title", "body"));
    FullTextQuery b = FullTextQuery.multiMatch("hello", Arrays.asList("title", "content"));
    assertFalse(a.equals(b), "MultiMatchQuery instances with different columns must not be equal");
  }

  @Test
  void testMultiMatchQueryEqualityWithBoosts() {
    FullTextQuery a =
        FullTextQuery.multiMatch(
            "hello",
            Arrays.asList("title", "body"),
            Arrays.asList(2.0f, 0.5f),
            FullTextQuery.Operator.OR);
    FullTextQuery b =
        FullTextQuery.multiMatch(
            "hello",
            Arrays.asList("title", "body"),
            Arrays.asList(2.0f, 0.5f),
            FullTextQuery.Operator.OR);
    assertEquals(a, b, "MultiMatchQuery instances with same boosts must be equal");
    assertEquals(a.hashCode(), b.hashCode());
  }

  @Test
  void testMultiMatchQueryInequalityDifferentBoosts() {
    FullTextQuery a =
        FullTextQuery.multiMatch(
            "hello",
            Arrays.asList("title", "body"),
            Arrays.asList(2.0f, 0.5f),
            FullTextQuery.Operator.OR);
    FullTextQuery b =
        FullTextQuery.multiMatch(
            "hello",
            Arrays.asList("title", "body"),
            Arrays.asList(1.0f, 0.5f),
            FullTextQuery.Operator.OR);
    assertFalse(a.equals(b), "MultiMatchQuery instances with different boosts must not be equal");
  }

  @Test
  void testDifferentSubtypesNotEqual() {
    FullTextQuery match = FullTextQuery.match("hello", "body");
    FullTextQuery phrase = FullTextQuery.phrase("hello", "body");
    assertFalse(
        match.equals(phrase),
        "MatchQuery and PhraseQuery must not be equal even with same text/column");
  }
}
