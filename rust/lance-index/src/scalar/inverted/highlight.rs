// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Logical text matching and original-value ranges. No index or storage access.
use super::query::fuzzy::FuzzyAutomaton;
use super::query::{FtsQuery, FtsSearchParams, Operator, Tokens};
use super::tokenizer::document_tokenizer::LanceTokenizer;
use arrow::buffer::OffsetBuffer;
use arrow_array::{ArrayRef, ListArray, StringArray, StructArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Fields};
use fst::Automaton;
use icu_segmenter::GraphemeClusterSegmenter;
use lance_core::{Error, Result};
use lance_tokenizer::Token;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

/// Result column containing ranges in complete original string values.
pub const HIGHLIGHTS_COL: &str = "_highlights";

/// A half-open UTF-8 byte range in a complete original string value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HighlightMatch {
    /// Inclusive start byte, expanded to a grapheme boundary.
    pub start: u64,
    /// Exclusive end byte, expanded to a grapheme boundary.
    pub end: u64,
}

/// Matching ranges in one logical source value. No source text is copied into the result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HighlightedValue {
    /// Canonical logical field path.
    pub field: String,
    /// Nested array ordinals, empty for a scalar string.
    pub value_index: Vec<u32>,
    /// All eligible ranges, sorted and merged where they overlap.
    pub matches: Vec<HighlightMatch>,
}

/// One original string value with its logical array coordinates.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HighlightValue {
    pub value_index: Vec<u32>,
    pub text: String,
}

/// Source values of one logical document field.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HighlightDocument {
    pub field: String,
    pub values: Vec<HighlightValue>,
}

#[derive(Debug, Clone)]
enum TermMatcher {
    Exact(String),
    Fuzzy(Arc<FuzzyAutomaton>),
}
impl TermMatcher {
    fn matches(&self, text: &str) -> bool {
        match self {
            Self::Exact(term) => term == text,
            Self::Fuzzy(automaton) => {
                let state = text.bytes().fold(automaton.start(), |state, byte| {
                    automaton.accept(&state, byte)
                });
                automaton.is_match(&state)
            }
        }
    }
}

/// A resolved lexical leaf, independent of candidate selection and scoring.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct HighlightLeaf {
    pub fields: Vec<String>,
    pub tokenizer: Arc<dyn LanceTokenizer>,
    tokens: Tokens,
    matchers: Vec<TermMatcher>,
    operator: Operator,
    phrase_slop: Option<u32>,
}
impl HighlightLeaf {
    pub fn try_new(
        fields: Vec<String>,
        tokenizer: Arc<dyn LanceTokenizer>,
        tokens: Tokens,
        operator: Operator,
        phrase_slop: Option<u32>,
        params: &FtsSearchParams,
    ) -> Result<Self> {
        let matchers = (0..tokens.len())
            .map(|i| {
                let term = tokens.get_token(i);
                if phrase_slop.is_some() || params.fuzziness == Some(0) {
                    Ok(TermMatcher::Exact(term.to_owned()))
                } else {
                    Ok(TermMatcher::Fuzzy(Arc::new(FuzzyAutomaton::new(
                        term,
                        tokens.token_type(),
                        params,
                    )?)))
                }
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            fields,
            tokenizer,
            tokens,
            matchers,
            operator,
            phrase_slop,
        })
    }
}

#[derive(Debug, Clone)]
struct Evidence {
    field: String,
    value: usize,
    range: Range<usize>,
}

/// Query-local logical matching. It does not reproduce an index's bounded vocabulary.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct PreparedHighlightQuery {
    query: FtsQuery,
    leaves: Vec<HighlightLeaf>,
}
impl PreparedHighlightQuery {
    pub fn new(query: FtsQuery, leaves: Vec<HighlightLeaf>) -> Self {
        Self { query, leaves }
    }

    pub fn highlight(&self, documents: &[HighlightDocument]) -> Result<Vec<HighlightedValue>> {
        let mut analyzed = Vec::new();
        for document in documents {
            for leaf in self
                .leaves
                .iter()
                .filter(|leaf| leaf.fields.contains(&document.field))
            {
                if !analyzed.iter().any(|previous: &AnalyzedDocument| {
                    previous.field == document.field
                        && Arc::ptr_eq(&previous.tokenizer, &leaf.tokenizer)
                }) {
                    analyzed.push(analyze_document(
                        document,
                        &self.leaves,
                        leaf.tokenizer.clone(),
                    )?);
                }
            }
        }
        let results = self
            .leaves
            .iter()
            .map(|leaf| match_leaf(leaf, &analyzed))
            .collect::<Result<Vec<_>>>()?;
        let mut cursor = 0;
        let (_, evidence) = evaluate(&self.query, &results, &mut cursor)?;
        if cursor != results.len() {
            return Err(Error::internal(
                "highlight prepared leaf count does not match query",
            ));
        }
        let mut ranges_by_value = BTreeMap::<(&str, usize), Vec<Range<usize>>>::new();
        for hit in &evidence {
            ranges_by_value
                .entry((&hit.field, hit.value))
                .or_default()
                .push(hit.range.clone());
        }
        let mut values = Vec::new();
        for doc in documents {
            for (index, value) in doc.values.iter().enumerate() {
                if let Some(ranges) = ranges_by_value.remove(&(doc.field.as_str(), index)) {
                    values.push(HighlightedValue {
                        field: doc.field.clone(),
                        value_index: value.value_index.clone(),
                        matches: display_ranges(&value.text, ranges),
                    });
                }
            }
        }
        values.sort_by(|a, b| (&a.field, &a.value_index).cmp(&(&b.field, &b.value_index)));
        Ok(values)
    }
}

fn display_ranges(text: &str, ranges: Vec<Range<usize>>) -> Vec<HighlightMatch> {
    let boundaries = GraphemeClusterSegmenter::new()
        .segment_str(text)
        .collect::<Vec<_>>();
    let mut ranges = ranges
        .into_iter()
        .map(|range| {
            let start = boundaries
                .partition_point(|offset| *offset <= range.start)
                .saturating_sub(1);
            let end = boundaries.partition_point(|offset| *offset < range.end);
            boundaries[start]..boundaries[end]
        })
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut merged = Vec::<HighlightMatch>::new();
    for range in ranges {
        if let Some(last) = merged.last_mut()
            && (range.start as u64) < last.end
        {
            last.end = last.end.max(range.end as u64);
        } else if range.start < range.end {
            merged.push(HighlightMatch {
                start: range.start as u64,
                end: range.end as u64,
            });
        }
    }
    merged
}

type MatchResult = (bool, Vec<Evidence>);

fn evaluate(query: &FtsQuery, results: &[MatchResult], cursor: &mut usize) -> Result<MatchResult> {
    match query {
        FtsQuery::Match(_) | FtsQuery::Phrase(_) | FtsQuery::CombinedFields(_) => {
            let result = results
                .get(*cursor)
                .ok_or_else(|| Error::internal("highlight prepared leaf is missing"))?
                .clone();
            *cursor += 1;
            Ok(result)
        }
        FtsQuery::Boost(query) => evaluate(&query.positive, results, cursor),
        FtsQuery::MultiMatch(query) => {
            let mut matched = false;
            let mut evidence = Vec::new();
            for query in &query.match_queries {
                let (hit, spans) = evaluate(&FtsQuery::Match(query.clone()), results, cursor)?;
                matched |= hit;
                if hit {
                    evidence.extend(spans);
                }
            }
            Ok((matched, evidence))
        }
        FtsQuery::Boolean(query) => {
            let mut must_match = true;
            let mut should_match = false;
            let mut excluded = false;
            let mut evidence = Vec::new();
            for child in &query.must {
                let (matched, spans) = evaluate(child, results, cursor)?;
                must_match &= matched;
                if matched {
                    evidence.extend(spans);
                }
            }
            for child in &query.should {
                let (matched, spans) = evaluate(child, results, cursor)?;
                should_match |= matched;
                if matched {
                    evidence.extend(spans);
                }
            }
            for child in &query.must_not {
                excluded |= evaluate(child, results, cursor)?.0;
            }
            let matched = must_match && !excluded && (!query.must.is_empty() || should_match);
            Ok((matched, if matched { evidence } else { Vec::new() }))
        }
    }
}

#[derive(Debug, Clone)]
struct Occurrence {
    position: u32,
    range: Range<usize>,
}

struct AnalyzedDocument {
    tokenizer: Arc<dyn LanceTokenizer>,
    field: String,
    offsets: Vec<Range<usize>>,
    tokens: Vec<Token>,
}

fn analyze_document(
    document: &HighlightDocument,
    leaves: &[HighlightLeaf],
    tokenizer: Arc<dyn LanceTokenizer>,
) -> Result<AnalyzedDocument> {
    let relevant = leaves
        .iter()
        .filter(|leaf| {
            leaf.fields.contains(&document.field) && Arc::ptr_eq(&leaf.tokenizer, &tokenizer)
        })
        .collect::<Vec<_>>();
    let mut logical = String::new();
    let mut offsets = Vec::with_capacity(document.values.len());
    for value in &document.values {
        if !logical.is_empty() {
            logical.push(' ');
        }
        let start = logical.len();
        logical.push_str(&value.text);
        offsets.push(start..logical.len());
    }
    if document.values.is_empty() {
        return Ok(AnalyzedDocument {
            tokenizer,
            field: document.field.clone(),
            offsets,
            tokens: Vec::new(),
        });
    }
    let mut analyzer = tokenizer.box_clone();
    let mut stream = analyzer.token_stream_for_doc(&logical);
    let mut tokens = Vec::new();
    while let Some(token) = stream.try_next().map_err(Error::invalid_input)? {
        if token.offset_from > token.offset_to
            || !logical.is_char_boundary(token.offset_from)
            || !logical.is_char_boundary(token.offset_to)
        {
            return Err(Error::invalid_input(format!(
                "highlight analyzer returned invalid offsets {}..{} for field '{}'",
                token.offset_from, token.offset_to, document.field
            )));
        }
        if relevant.iter().any(|leaf| {
            leaf.matchers
                .iter()
                .any(|matcher| matcher.matches(&token.text))
        }) {
            tokens.push(token.clone());
        }
    }
    Ok(AnalyzedDocument {
        tokenizer,
        field: document.field.clone(),
        offsets,
        tokens,
    })
}

fn match_leaf(leaf: &HighlightLeaf, documents: &[AnalyzedDocument]) -> Result<MatchResult> {
    let mut found = BTreeSet::new();
    let mut required = BTreeSet::new();
    let mut matched_phrase = false;
    let mut evidence = Vec::new();
    for field in &leaf.fields {
        let doc = documents
            .iter()
            .find(|doc| &doc.field == field && Arc::ptr_eq(&doc.tokenizer, &leaf.tokenizer))
            .ok_or_else(|| {
                Error::internal(format!("highlight source field '{field}' is missing"))
            })?;
        let tokens = &leaf.tokens;
        for i in 0..tokens.len() {
            required.insert(tokens.position(i));
        }
        let mut occurrences = vec![Vec::new(); tokens.len()];
        for token in &doc.tokens {
            for (i, hits) in occurrences.iter_mut().enumerate() {
                if leaf.matchers[i].matches(&token.text) {
                    found.insert(tokens.position(i));
                    hits.push(Occurrence {
                        position: u32::try_from(token.position).map_err(|_| {
                            Error::invalid_input("highlight token position exceeds u32")
                        })?,
                        range: token.offset_from..token.offset_to,
                    });
                }
            }
        }
        for hits in &mut occurrences {
            hits.sort_by_key(|hit| hit.position);
        }
        let ranges = if let Some(slop) = leaf.phrase_slop {
            phrase_ranges(tokens, &occurrences, slop)
        } else {
            occurrences
                .iter()
                .flatten()
                .map(|hit| hit.range.clone())
                .collect()
        };
        matched_phrase |= leaf.phrase_slop.is_some() && !ranges.is_empty();
        for range in ranges {
            let first = doc
                .offsets
                .partition_point(|source| source.end <= range.start);
            for (value, source) in doc
                .offsets
                .iter()
                .enumerate()
                .skip(first)
                .take_while(|(_, source)| source.start < range.end)
            {
                let start = range.start.max(source.start);
                let end = range.end.min(source.end);
                if start < end {
                    evidence.push(Evidence {
                        field: field.clone(),
                        value,
                        range: start - source.start..end - source.start,
                    });
                }
            }
        }
    }
    let matched = if leaf.phrase_slop.is_some() {
        matched_phrase
    } else {
        !found.is_empty() && (leaf.operator == Operator::Or || found == required)
    };
    Ok((matched, if matched { evidence } else { Vec::new() }))
}

/// Retain every occurrence with both a complete prefix and a complete suffix.
/// Adjacent position lists are joined with prefix counts, avoiding enumeration of phrase combinations.
fn phrase_ranges(tokens: &Tokens, positions: &[Vec<Occurrence>], slop: u32) -> Vec<Range<usize>> {
    if positions.is_empty() {
        return Vec::new();
    }
    let mut forward = positions
        .iter()
        .map(|hits| vec![false; hits.len()])
        .collect::<Vec<_>>();
    forward[0].fill(true);
    for i in 1..positions.len() {
        let Some(delta) = tokens.position(i).checked_sub(tokens.position(i - 1)) else {
            return Vec::new();
        };
        let prefix = prefix_counts(&forward[i - 1]);
        for (j, hit) in positions[i].iter().enumerate() {
            let high = u64::from(hit.position).checked_sub(u64::from(delta));
            if let Some(high) = high {
                let low = high.saturating_sub(u64::from(slop));
                forward[i][j] = any_in_window(&positions[i - 1], &prefix, low, high);
            }
        }
    }
    let mut suffix = vec![true; positions.last().map_or(0, Vec::len)];
    let mut ranges = Vec::new();
    for i in (0..positions.len()).rev() {
        if i + 1 < positions.len() {
            let Some(delta) = tokens.position(i + 1).checked_sub(tokens.position(i)) else {
                return Vec::new();
            };
            let prefix = prefix_counts(&suffix);
            suffix = positions[i]
                .iter()
                .map(|hit| {
                    let low = u64::from(hit.position) + u64::from(delta);
                    any_in_window(&positions[i + 1], &prefix, low, low + u64::from(slop))
                })
                .collect();
        }
        for (j, hit) in positions[i].iter().enumerate() {
            if forward[i][j] && suffix[j] {
                ranges.push(hit.range.clone());
            }
        }
    }
    ranges
}
fn prefix_counts(bits: &[bool]) -> Vec<usize> {
    let mut result = Vec::with_capacity(bits.len() + 1);
    result.push(0);
    for (i, bit) in bits.iter().enumerate() {
        result.push(result[i] + usize::from(*bit));
    }
    result
}
fn any_in_window(hits: &[Occurrence], prefix: &[usize], low: u64, high: u64) -> bool {
    let start = hits.partition_point(|hit| u64::from(hit.position) < low);
    let end = hits.partition_point(|hit| u64::from(hit.position) <= high);
    prefix[end] > prefix[start]
}

fn match_fields() -> Fields {
    vec![
        Field::new("start", DataType::UInt64, false),
        Field::new("end", DataType::UInt64, false),
    ]
    .into()
}

fn value_fields() -> Fields {
    vec![
        Field::new("field", DataType::Utf8, false),
        Field::new(
            "value_index",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
            false,
        ),
        Field::new(
            "matches",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(match_fields()),
                false,
            ))),
            false,
        ),
    ]
    .into()
}

/// Arrow field appended to an opted-in FTS result.
pub fn highlights_field() -> Field {
    Field::new(
        HIGHLIGHTS_COL,
        DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(value_fields()),
            false,
        ))),
        false,
    )
}

/// Encode source ranges without passing offsets through JSON.
#[doc(hidden)]
pub fn highlights_array(rows: &[Vec<HighlightedValue>]) -> Result<ArrayRef> {
    let highlighted = rows.iter().flatten().collect::<Vec<_>>();
    let matches = highlighted
        .iter()
        .flat_map(|value| &value.matches)
        .collect::<Vec<_>>();
    let matched = StructArray::try_new(
        match_fields(),
        vec![
            Arc::new(UInt64Array::from_iter_values(
                matches.iter().map(|hit| hit.start),
            )),
            Arc::new(UInt64Array::from_iter_values(
                matches.iter().map(|hit| hit.end),
            )),
        ],
        None,
    )?;
    let list = |lengths: Vec<usize>, values: ArrayRef| -> Result<ArrayRef> {
        let mut offsets = vec![0i32];
        let mut total = 0i32;
        for length in lengths {
            total = total
                .checked_add(
                    i32::try_from(length)
                        .map_err(|_| Error::invalid_input("highlight array exceeds i32 offsets"))?,
                )
                .ok_or_else(|| Error::invalid_input("highlight array exceeds i32 offsets"))?;
            offsets.push(total);
        }
        Ok(Arc::new(ListArray::try_new(
            Arc::new(Field::new("item", values.data_type().clone(), false)),
            OffsetBuffer::new(offsets.into()),
            values,
            None,
        )?))
    };
    let values = StructArray::try_new(
        value_fields(),
        vec![
            Arc::new(StringArray::from_iter_values(
                highlighted.iter().map(|value| &value.field),
            )),
            list(
                highlighted
                    .iter()
                    .map(|value| value.value_index.len())
                    .collect(),
                Arc::new(UInt32Array::from_iter_values(
                    highlighted
                        .iter()
                        .flat_map(|value| value.value_index.iter().copied()),
                )),
            )?,
            list(
                highlighted
                    .iter()
                    .map(|value| value.matches.len())
                    .collect(),
                Arc::new(matched),
            )?,
        ],
        None,
    )?;
    list(rows.iter().map(Vec::len).collect(), Arc::new(values))
}

#[cfg(test)]
mod tests {
    use super::super::query::{
        BooleanQuery, MatchQuery, Occur, PhraseQuery, try_collect_query_tokens,
    };
    use super::super::tokenizer::document_tokenizer::TextTokenizer;
    use super::*;
    use arrow_array::{Array, cast::AsArray};
    use lance_tokenizer::{
        AsciiFoldingFilter, Language, LowerCaser, SimpleTokenizer, Stemmer, StopWordFilter,
        TextAnalyzer,
    };
    use rstest::rstest;

    fn analyzer() -> Box<dyn LanceTokenizer> {
        Box::new(TextTokenizer::new(
            TextAnalyzer::builder(SimpleTokenizer::default())
                .filter(LowerCaser)
                .filter(AsciiFoldingFilter)
                .filter(Stemmer::new(Language::English))
                .filter(StopWordFilter::remove(vec!["the".into()]))
                .build(),
        ))
    }
    fn leaf(text: &str, slop: Option<u32>, fuzziness: u32, budget: usize) -> HighlightLeaf {
        let mut tokenizer = analyzer();
        let tokens = try_collect_query_tokens(text, &mut tokenizer).unwrap();
        HighlightLeaf::try_new(
            vec!["body".into()],
            tokenizer.into(),
            tokens,
            Operator::Or,
            slop,
            &FtsSearchParams::new()
                .with_fuzziness(Some(fuzziness))
                .with_max_expansions(budget),
        )
        .unwrap()
    }
    fn term(text: &str) -> FtsQuery {
        MatchQuery::new(text.into())
            .with_column(Some("body".into()))
            .with_fuzziness(Some(0))
            .into()
    }
    fn documents(text: &[&str]) -> Vec<HighlightDocument> {
        let document_count = text.len();
        vec![HighlightDocument {
            field: "body".into(),
            values: text
                .iter()
                .enumerate()
                .map(|(i, text)| HighlightValue {
                    value_index: if document_count == 1 {
                        vec![]
                    } else {
                        vec![i as u32]
                    },
                    text: (*text).into(),
                })
                .collect(),
        }]
    }
    fn marked<'a>(values: &[HighlightedValue], source: &'a [&str]) -> Vec<&'a str> {
        values
            .iter()
            .flat_map(|value| {
                let text = source[value.value_index.first().copied().unwrap_or(0) as usize];
                value
                    .matches
                    .iter()
                    .map(move |hit| &text[hit.start as usize..hit.end as usize])
            })
            .collect()
    }
    #[rstest]
    #[case::stem("run", "She was running quickly.", "running")]
    #[case::fold("cafe", "A café résumé.", "café")]
    #[case::combining("cafe", "Cafe\u{301}", "Cafe\u{301}")]
    #[case::unicode("rust", "中文🙂 Rust", "Rust")]
    #[case::markup("rust", "<script>alert(1)</script> Rust & storage", "Rust")]
    fn source_offsets(#[case] query: &str, #[case] source: &str, #[case] expected: &str) {
        let matcher = PreparedHighlightQuery::new(term(query), vec![leaf(query, None, 0, 1)]);
        let values = matcher.highlight(&documents(&[source])).unwrap();
        assert_eq!(marked(&values, &[source]), vec![expected]);
    }
    #[rstest]
    #[case::one(1)]
    #[case::many(50)]
    fn fuzzy_ranges_do_not_use_candidate_budget(#[case] budget: usize) {
        let matcher = PreparedHighlightQuery::new(term("cat"), vec![leaf("cat", None, 1, budget)]);
        let values = matcher.highlight(&documents(&["bat cat dog"])).unwrap();
        assert_eq!(marked(&values, &["bat cat dog"]), vec!["bat", "cat"]);
    }
    #[test]
    fn excluded_and_unsatisfied_branches_emit_no_ranges() {
        let query = BooleanQuery::new([
            (
                Occur::Should,
                BooleanQuery::new([(Occur::Must, term("alpha")), (Occur::Must, term("beta"))])
                    .into(),
            ),
            (Occur::Should, term("gamma")),
            (Occur::MustNot, term("blocked")),
        ]);
        let matcher = PreparedHighlightQuery::new(
            query.into(),
            ["alpha", "beta", "gamma", "blocked"]
                .map(|t| leaf(t, None, 0, 1))
                .to_vec(),
        );
        assert_eq!(
            marked(
                &matcher.highlight(&documents(&["alpha gamma"])).unwrap(),
                &["alpha gamma"]
            ),
            vec!["gamma"]
        );
        assert!(
            matcher
                .highlight(&documents(&["alpha gamma blocked"]))
                .unwrap()
                .is_empty()
        );
        assert!(
            matcher
                .highlight(&documents(&["alpha"]))
                .unwrap()
                .is_empty()
        );
    }
    #[rstest]
    #[case::gap("new the york", "new fast york. new york.", 0, vec!["new","york"])]
    #[case::slop("new york", "new fast york. new very fast york.", 1, vec!["new","york"])]
    #[case::alternatives("alpha beta", "alpha beta beta", 1, vec!["alpha","beta","beta"])]
    fn phrase_ranges_include_all_participating_tokens(
        #[case] text: &str,
        #[case] source: &str,
        #[case] slop: u32,
        #[case] expected: Vec<&str>,
    ) {
        let query = PhraseQuery::new(text.into()).with_column(Some("body".into()));
        let matcher = PreparedHighlightQuery::new(query.into(), vec![leaf(text, Some(slop), 0, 1)]);
        assert_eq!(
            marked(
                &matcher.highlight(&documents(&[source])).unwrap(),
                &[source]
            ),
            expected
        );
    }
    #[test]
    fn phrase_preserves_cross_value_coordinates() {
        let query = PhraseQuery::new("new york".into()).with_column(Some("body".into()));
        let matcher =
            PreparedHighlightQuery::new(query.into(), vec![leaf("new york", Some(0), 0, 1)]);
        let values = matcher.highlight(&documents(&["new", "york"])).unwrap();
        assert_eq!(marked(&values, &["new", "york"]), vec!["new", "york"]);
        assert_eq!(
            values
                .iter()
                .map(|v| v.value_index.clone())
                .collect::<Vec<_>>(),
            vec![vec![0], vec![1]]
        );
    }
    #[test]
    fn all_dense_occurrences_are_returned() {
        let source = "alpha ".repeat(10_000);
        let query = PhraseQuery::new("alpha alpha".into()).with_column(Some("body".into()));
        let matcher =
            PreparedHighlightQuery::new(query.into(), vec![leaf("alpha alpha", Some(0), 0, 1)]);
        let values = matcher.highlight(&documents(&[&source])).unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].matches.len(), 10_000);
        assert_eq!(values[0].matches.last().unwrap().end, 59_999);
    }
    #[test]
    fn empty_values_and_arrow_schema() {
        let matcher = PreparedHighlightQuery::new(term("rust"), vec![leaf("rust", None, 0, 1)]);
        for source in [documents(&[]), documents(&[""]), documents(&["nothing"])] {
            assert!(matcher.highlight(&source).unwrap().is_empty());
        }
        let result =
            highlights_array(&[vec![], matcher.highlight(&documents(&["rust"])).unwrap()]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result.data_type(), highlights_field().data_type());
        let value = result.as_list::<i32>().value(1);
        assert_eq!(value.as_struct().num_columns(), 3);
        assert!(value.as_struct().column_by_name("text").is_none());
    }
    #[rstest]
    #[case::prefix("cat", "bat cat cut", 1, vec!["cat", "cut"])]
    #[case::unicode("東京", "東京 東亰", 0, vec!["東京", "東亰"])]
    fn fuzzy_prefix_and_unicode(
        #[case] text: &str,
        #[case] source: &str,
        #[case] prefix: u32,
        #[case] expected: Vec<&str>,
    ) {
        let mut tokenizer = analyzer();
        let tokens = try_collect_query_tokens(text, &mut tokenizer).unwrap();
        let leaf = HighlightLeaf::try_new(
            vec!["body".into()],
            tokenizer.into(),
            tokens,
            Operator::Or,
            None,
            &FtsSearchParams::new()
                .with_fuzziness(Some(1))
                .with_prefix_length(prefix),
        )
        .unwrap();
        let matcher = PreparedHighlightQuery::new(term(text), vec![leaf]);
        assert_eq!(
            marked(
                &matcher.highlight(&documents(&[source])).unwrap(),
                &[source]
            ),
            expected
        );
    }
    #[test]
    fn phrase_ranges_match_exhaustive_occurrence_paths() {
        let mut tokenizer = analyzer();
        let tokens = try_collect_query_tokens("alpha beta gamma", &mut tokenizer).unwrap();
        for mask in 0..512 {
            let positions = (0..3)
                .map(|term| {
                    (0..3)
                        .filter(|position| mask & (1 << (term * 3 + position)) != 0)
                        .map(|position| Occurrence {
                            position,
                            range: position as usize * 2..position as usize * 2 + 1,
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            for slop in 0..=1 {
                let mut expected = BTreeSet::new();
                for a in &positions[0] {
                    for b in &positions[1] {
                        for c in &positions[2] {
                            if b.position > a.position
                                && b.position <= a.position + 1 + slop
                                && c.position > b.position
                                && c.position <= b.position + 1 + slop
                            {
                                for hit in [a, b, c] {
                                    expected.insert((hit.range.start, hit.range.end));
                                }
                            }
                        }
                    }
                }
                let actual = phrase_ranges(&tokens, &positions, slop)
                    .into_iter()
                    .map(|range| (range.start, range.end))
                    .collect::<BTreeSet<_>>();
                assert_eq!(actual, expected, "mask={mask}, slop={slop}");
            }
        }
    }

    #[test]
    fn conjunction_requires_every_logical_query_position() {
        let mut leaf = leaf("cat elephant", None, 1, 1);
        leaf.operator = Operator::And;
        let matcher = PreparedHighlightQuery::new(term("cat elephant"), vec![leaf]);
        assert!(
            matcher
                .highlight(&documents(&["bat cat"]))
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            marked(
                &matcher
                    .highlight(&documents(&["bat cat elephant"]))
                    .unwrap(),
                &["bat cat elephant"]
            ),
            vec!["bat", "cat", "elephant"]
        );
    }
    #[test]
    fn all_array_values_keep_their_coordinates() {
        let matcher = PreparedHighlightQuery::new(term("rust"), vec![leaf("rust", None, 0, 1)]);
        let sources = vec!["rust"; 1000];
        let values = matcher.highlight(&documents(&sources)).unwrap();
        assert_eq!(values.len(), sources.len());
        for (i, value) in values.iter().enumerate() {
            assert_eq!(value.value_index, vec![i as u32]);
            assert_eq!(value.matches, vec![HighlightMatch { start: 0, end: 4 }]);
        }
    }
}
