// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Logical fuzzy term predicates, shared by dictionary traversal and source matching.
use super::{FtsSearchParams, fuzzy_term_options};
use crate::scalar::inverted::document_tokenizer::DocType;
use fst::Automaton;
use lance_core::{Error, Result};
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashMap, VecDeque};

const UNICODE_LEVENSHTEIN_STATE_LIMIT: usize = 10_000;

type ByteTransitions = Box<[Option<usize>; 256]>;

struct UnicodeDfaState {
    transitions: ByteTransitions,
    is_match: bool,
}

/// A Unicode-scalar Levenshtein automaton for FST dictionaries.
///
/// `fst` 0.4's built-in automaton can lose exact transitions when multiple
/// non-ASCII query scalars share a UTF-8 lead byte. This implementation fixes
/// that overlap while retaining the same execution model: query construction
/// interns bounded Levenshtein rows and compiles a byte DFA, then FST traversal
/// performs one table lookup per byte without allocating.
pub(in crate::scalar::inverted) struct UnicodeLevenshtein {
    states: Vec<UnicodeDfaState>,
    start_state: usize,
    #[cfg(test)]
    exact_override_counts: Vec<usize>,
}

#[derive(Debug)]
struct UnicodeLevenshteinError {
    state_limit: usize,
}

impl std::fmt::Display for UnicodeLevenshteinError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Unicode Levenshtein automaton exceeds state limit of {}",
            self.state_limit
        )
    }
}

#[derive(Default)]
struct ExactUtf8Trie {
    target: Option<usize>,
    children: BTreeMap<u8, Self>,
}

impl ExactUtf8Trie {
    fn insert(&mut self, bytes: &[u8], target: usize) {
        let mut node = self;
        for byte in bytes {
            node = node.children.entry(*byte).or_default();
        }
        node.target = Some(target);
    }
}

struct UnicodeLevenshteinBuilder {
    query: Vec<char>,
    exact_prefix: Vec<u8>,
    max_distance: usize,
    state_limit: usize,
    states: Vec<UnicodeDfaState>,
    rows: HashMap<Vec<usize>, usize>,
    pending_rows: VecDeque<Vec<usize>>,
    default_utf8_transitions: HashMap<usize, ByteTransitions>,
    #[cfg(test)]
    exact_override_counts: Vec<usize>,
}

impl UnicodeLevenshtein {
    fn new(
        fuzzy_suffix: &str,
        exact_prefix: &str,
        max_distance: u32,
    ) -> std::result::Result<Self, UnicodeLevenshteinError> {
        Self::new_with_limit(
            fuzzy_suffix,
            exact_prefix,
            max_distance,
            UNICODE_LEVENSHTEIN_STATE_LIMIT,
        )
    }

    fn new_with_limit(
        fuzzy_suffix: &str,
        exact_prefix: &str,
        max_distance: u32,
        state_limit: usize,
    ) -> std::result::Result<Self, UnicodeLevenshteinError> {
        UnicodeLevenshteinBuilder {
            query: fuzzy_suffix.chars().collect(),
            exact_prefix: exact_prefix.as_bytes().to_vec(),
            max_distance: max_distance as usize,
            state_limit,
            states: Vec::new(),
            rows: HashMap::new(),
            pending_rows: VecDeque::new(),
            default_utf8_transitions: HashMap::new(),
            #[cfg(test)]
            exact_override_counts: Vec::new(),
        }
        .build()
    }
}

impl Automaton for UnicodeLevenshtein {
    type State = Option<usize>;

    #[inline]
    fn start(&self) -> Self::State {
        Some(self.start_state)
    }

    #[inline]
    fn is_match(&self, state: &Self::State) -> bool {
        state.is_some_and(|state| self.states[state].is_match)
    }

    #[inline]
    fn can_match(&self, state: &Self::State) -> bool {
        state.is_some()
    }

    #[inline]
    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        state.and_then(|state| self.states[state].transitions[byte as usize])
    }
}

pub(in crate::scalar::inverted) struct AsciiFuzzyAutomaton {
    levenshtein: fst::automaton::Levenshtein,
    exact_prefix: Vec<u8>,
}

#[derive(Clone, Copy)]
pub(in crate::scalar::inverted) struct AsciiFuzzyState {
    levenshtein: Option<usize>,
    exact_prefix_bytes_matched: Option<usize>,
}

impl Automaton for AsciiFuzzyAutomaton {
    type State = AsciiFuzzyState;

    #[inline]
    fn start(&self) -> Self::State {
        AsciiFuzzyState {
            levenshtein: self.levenshtein.start(),
            exact_prefix_bytes_matched: Some(0),
        }
    }

    #[inline]
    fn is_match(&self, state: &Self::State) -> bool {
        self.levenshtein.is_match(&state.levenshtein)
            && state.exact_prefix_bytes_matched == Some(self.exact_prefix.len())
    }

    #[inline]
    fn can_match(&self, state: &Self::State) -> bool {
        self.levenshtein.can_match(&state.levenshtein) && state.exact_prefix_bytes_matched.is_some()
    }

    #[inline]
    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        let exact_prefix_bytes_matched = state.exact_prefix_bytes_matched.and_then(|position| {
            if position == self.exact_prefix.len() {
                Some(position)
            } else if self.exact_prefix[position] == byte {
                Some(position + 1)
            } else {
                None
            }
        });
        AsciiFuzzyState {
            levenshtein: self.levenshtein.accept(&state.levenshtein, byte),
            exact_prefix_bytes_matched,
        }
    }
}

pub(in crate::scalar::inverted) enum FuzzyAutomaton {
    Ascii(AsciiFuzzyAutomaton),
    Unicode(UnicodeLevenshtein),
}

pub(in crate::scalar::inverted) enum FuzzyAutomatonState {
    Ascii(AsciiFuzzyState),
    Unicode(Option<usize>),
}

impl FuzzyAutomaton {
    pub(in crate::scalar::inverted) fn new(
        token: &str,
        token_type: &DocType,
        params: &FtsSearchParams,
    ) -> Result<Self> {
        let fuzzy = fuzzy_term_options(token, token_type, params.fuzziness, params.prefix_length);
        if token.is_ascii() {
            let levenshtein = fst::automaton::Levenshtein::new(token, fuzzy.edit_distance)
                .map_err(|error| {
                    Error::index(format!("failed to construct the fuzzy query: {error}"))
                })?;
            Ok(Self::Ascii(AsciiFuzzyAutomaton {
                levenshtein,
                exact_prefix: fuzzy.exact_prefix.as_bytes().to_vec(),
            }))
        } else {
            let levenshtein = UnicodeLevenshtein::new(
                fuzzy.fuzzy_suffix,
                fuzzy.exact_prefix,
                fuzzy.edit_distance,
            )
            .map_err(|error| {
                Error::index(format!("failed to construct the fuzzy query: {error}"))
            })?;
            Ok(Self::Unicode(levenshtein))
        }
    }
}

impl Automaton for FuzzyAutomaton {
    type State = FuzzyAutomatonState;

    #[inline]
    fn start(&self) -> Self::State {
        match self {
            Self::Ascii(automaton) => FuzzyAutomatonState::Ascii(automaton.start()),
            Self::Unicode(automaton) => FuzzyAutomatonState::Unicode(automaton.start()),
        }
    }

    #[inline]
    fn is_match(&self, state: &Self::State) -> bool {
        match (self, state) {
            (Self::Ascii(automaton), FuzzyAutomatonState::Ascii(state)) => {
                automaton.is_match(state)
            }
            (Self::Unicode(automaton), FuzzyAutomatonState::Unicode(state)) => {
                automaton.is_match(state)
            }
            _ => false,
        }
    }

    #[inline]
    fn can_match(&self, state: &Self::State) -> bool {
        match (self, state) {
            (Self::Ascii(automaton), FuzzyAutomatonState::Ascii(state)) => {
                automaton.can_match(state)
            }
            (Self::Unicode(automaton), FuzzyAutomatonState::Unicode(state)) => {
                automaton.can_match(state)
            }
            _ => false,
        }
    }

    #[inline]
    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        match (self, state) {
            (Self::Ascii(automaton), FuzzyAutomatonState::Ascii(state)) => {
                FuzzyAutomatonState::Ascii(automaton.accept(state, byte))
            }
            (Self::Unicode(automaton), FuzzyAutomatonState::Unicode(state)) => {
                FuzzyAutomatonState::Unicode(automaton.accept(state, byte))
            }
            (Self::Ascii(_), _) => FuzzyAutomatonState::Ascii(AsciiFuzzyState {
                levenshtein: None,
                exact_prefix_bytes_matched: None,
            }),
            (Self::Unicode(_), _) => FuzzyAutomatonState::Unicode(None),
        }
    }
}

impl UnicodeLevenshteinBuilder {
    fn empty_transitions() -> ByteTransitions {
        Box::new([None; 256])
    }

    fn add_state(
        &mut self,
        transitions: ByteTransitions,
        is_match: bool,
    ) -> std::result::Result<usize, UnicodeLevenshteinError> {
        if self.states.len() >= self.state_limit {
            return Err(UnicodeLevenshteinError {
                state_limit: self.state_limit,
            });
        }
        let state = self.states.len();
        self.states.push(UnicodeDfaState {
            transitions,
            is_match,
        });
        Ok(state)
    }

    fn advance_row(&self, distances: &[usize], candidate: Option<char>) -> Vec<usize> {
        let cutoff = self.max_distance.saturating_add(1);
        let mut next = Vec::with_capacity(self.query.len() + 1);
        next.push(distances[0].saturating_add(1).min(cutoff));
        for (query_index, query) in self.query.iter().enumerate() {
            let substitution_cost = usize::from(Some(*query) != candidate);
            let insertion = distances[query_index + 1].saturating_add(1);
            let deletion = next[query_index].saturating_add(1);
            let substitution = distances[query_index].saturating_add(substitution_cost);
            next.push(insertion.min(deletion).min(substitution).min(cutoff));
        }
        next
    }

    fn intern_row(
        &mut self,
        distances: Vec<usize>,
    ) -> std::result::Result<Option<usize>, UnicodeLevenshteinError> {
        if distances
            .iter()
            .all(|distance| *distance > self.max_distance)
        {
            return Ok(None);
        }
        if let Some(state) = self.rows.get(&distances) {
            return Ok(Some(*state));
        }
        let is_match = distances
            .last()
            .is_some_and(|distance| *distance <= self.max_distance);
        let state = self.add_state(Self::empty_transitions(), is_match)?;
        self.rows.insert(distances.clone(), state);
        self.pending_rows.push_back(distances);
        Ok(Some(state))
    }

    fn state_with_range(
        &mut self,
        start: u8,
        end: u8,
        target: usize,
    ) -> std::result::Result<usize, UnicodeLevenshteinError> {
        let mut transitions = Self::empty_transitions();
        transitions[start as usize..=end as usize].fill(Some(target));
        self.add_state(transitions, false)
    }

    fn build_default_utf8_transitions(
        &mut self,
        target: usize,
    ) -> std::result::Result<ByteTransitions, UnicodeLevenshteinError> {
        if let Some(transitions) = self.default_utf8_transitions.get(&target) {
            return Ok(transitions.clone());
        }

        let one_continuation = self.state_with_range(0x80, 0xbf, target)?;
        let two_continuations = self.state_with_range(0x80, 0xbf, one_continuation)?;
        let three_continuations = self.state_with_range(0x80, 0xbf, two_continuations)?;
        let e0_second = self.state_with_range(0xa0, 0xbf, one_continuation)?;
        let ed_second = self.state_with_range(0x80, 0x9f, one_continuation)?;
        let f0_second = self.state_with_range(0x90, 0xbf, two_continuations)?;
        let f4_second = self.state_with_range(0x80, 0x8f, two_continuations)?;

        let mut transitions = Self::empty_transitions();
        transitions[0x00..=0x7f].fill(Some(target));
        transitions[0xc2..=0xdf].fill(Some(one_continuation));
        transitions[0xe0] = Some(e0_second);
        transitions[0xe1..=0xec].fill(Some(two_continuations));
        transitions[0xed] = Some(ed_second);
        transitions[0xee..=0xef].fill(Some(two_continuations));
        transitions[0xf0] = Some(f0_second);
        transitions[0xf1..=0xf3].fill(Some(three_continuations));
        transitions[0xf4] = Some(f4_second);
        self.default_utf8_transitions
            .insert(target, transitions.clone());
        Ok(transitions)
    }

    fn overlay_exact_trie(
        &mut self,
        transitions: &mut ByteTransitions,
        trie: &ExactUtf8Trie,
    ) -> std::result::Result<(), UnicodeLevenshteinError> {
        for (byte, child) in &trie.children {
            if let Some(target) = child.target {
                debug_assert!(child.children.is_empty());
                transitions[*byte as usize] = Some(target);
                continue;
            }

            let mut child_transitions = transitions[*byte as usize]
                .map(|state| self.states[state].transitions.clone())
                .unwrap_or_else(Self::empty_transitions);
            self.overlay_exact_trie(&mut child_transitions, child)?;
            transitions[*byte as usize] = Some(self.add_state(child_transitions, false)?);
        }
        Ok(())
    }

    fn build_boundary_state(
        &mut self,
        distances: &[usize],
    ) -> std::result::Result<(), UnicodeLevenshteinError> {
        let boundary_state = self.rows[distances];
        let mismatch = self.advance_row(distances, None);
        let mismatch_state = self.intern_row(mismatch)?;
        let mut transitions = match mismatch_state {
            Some(target) => self.build_default_utf8_transitions(target)?,
            None => Self::empty_transitions(),
        };

        let mut exact_trie = ExactUtf8Trie::default();
        let mut exact_scalars = SmallVec::<[char; 8]>::new();
        for (query_index, query_scalar) in self.query.iter().copied().enumerate() {
            if distances[query_index] <= self.max_distance && !exact_scalars.contains(&query_scalar)
            {
                exact_scalars.push(query_scalar);
            }
        }
        #[cfg(test)]
        self.exact_override_counts.push(exact_scalars.len());
        for query_scalar in exact_scalars {
            let exact = self.advance_row(distances, Some(query_scalar));
            if let Some(target) = self.intern_row(exact)? {
                let mut encoded = [0; 4];
                exact_trie.insert(query_scalar.encode_utf8(&mut encoded).as_bytes(), target);
            }
        }
        self.overlay_exact_trie(&mut transitions, &exact_trie)?;
        self.states[boundary_state].transitions = transitions;
        Ok(())
    }

    fn build(mut self) -> std::result::Result<UnicodeLevenshtein, UnicodeLevenshteinError> {
        let cutoff = self.max_distance.saturating_add(1);
        let initial_distances = (0..=self.query.len())
            .map(|distance| distance.min(cutoff))
            .collect::<Vec<_>>();
        let initial_is_match = initial_distances
            .last()
            .is_some_and(|distance| *distance <= self.max_distance);
        let suffix_start = self.add_state(Self::empty_transitions(), initial_is_match)?;
        self.rows.insert(initial_distances.clone(), suffix_start);
        self.pending_rows.push_back(initial_distances);
        while let Some(distances) = self.pending_rows.pop_front() {
            self.build_boundary_state(&distances)?;
        }

        let mut start_state = suffix_start;
        for byte in std::mem::take(&mut self.exact_prefix).into_iter().rev() {
            let mut transitions = Self::empty_transitions();
            transitions[byte as usize] = Some(start_state);
            start_state = self.add_state(transitions, false)?;
        }

        Ok(UnicodeLevenshtein {
            states: self.states,
            start_state,
            #[cfg(test)]
            exact_override_counts: self.exact_override_counts,
        })
    }
}

impl std::fmt::Debug for FuzzyAutomaton {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("FuzzyAutomaton")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fst::{IntoStreamer, Streamer};
    use rstest::rstest;
    fn automaton_accepts<A: Automaton>(automaton: &A, candidate: &[u8]) -> bool {
        let mut state = automaton.start();
        for byte in candidate {
            state = automaton.accept(&state, *byte);
            if !automaton.can_match(&state) {
                break;
            }
        }
        automaton.is_match(&state)
    }

    #[rstest]
    #[case::empty("", "", 0, "", true)]
    #[case::empty_with_one_insertion("", "", 1, "é", true)]
    #[case::unicode_deletion("بسرع", "", 1, "بسر", true)]
    #[case::unicode_distance_two("猫咪", "", 2, "小猫咪呀", true)]
    #[case::unicode_over_distance("猫咪", "", 1, "小猫咪呀", false)]
    #[case::exact_unicode_prefix("clair", "é", 1, "éclait", true)]
    #[case::wrong_unicode_prefix("clair", "é", 1, "àclait", false)]
    fn unicode_levenshtein_table_matches_scalar_distance(
        #[case] fuzzy_suffix: &str,
        #[case] exact_prefix: &str,
        #[case] max_distance: u32,
        #[case] candidate: &str,
        #[case] expected: bool,
    ) {
        let automaton = UnicodeLevenshtein::new(fuzzy_suffix, exact_prefix, max_distance).unwrap();

        assert_eq!(
            automaton_accepts(&automaton, candidate.as_bytes()),
            expected
        );
    }

    #[test]
    fn unicode_levenshtein_rejects_invalid_utf8_and_dead_states() {
        let automaton = UnicodeLevenshtein::new("بسرع", "", 1).unwrap();
        let start = automaton.start();
        assert!(automaton.can_match(&start));

        let partial = automaton.accept(&start, 0xd8);
        assert!(automaton.can_match(&partial));
        assert!(!automaton.is_match(&partial));

        let invalid = automaton.accept(&partial, b'a');
        assert!(!automaton.can_match(&invalid));
        assert!(!automaton.is_match(&invalid));

        let empty = UnicodeLevenshtein::new("", "", 0).unwrap();
        let dead = empty.accept(&empty.start(), b'a');
        assert!(!empty.can_match(&dead));
    }

    #[test]
    fn unicode_levenshtein_enforces_construction_state_limit() {
        let Err(error) = UnicodeLevenshtein::new_with_limit("بسرع", "", 1, 1) else {
            panic!("a one-state limit must reject the Unicode fuzzy DFA");
        };

        assert_eq!(error.state_limit, 1);
    }

    #[test]
    fn unicode_levenshtein_only_overrides_row_active_scalars() {
        let query = "ابتثجحخدذرزسشصضطظعغفقكلمن";
        let scalar_count = query.chars().count();
        let automaton = UnicodeLevenshtein::new(query, "", 1).unwrap();
        let row_count = automaton.exact_override_counts.len();
        let override_count = automaton.exact_override_counts.iter().sum::<usize>();

        assert!(row_count > 0);
        assert!(
            automaton
                .exact_override_counts
                .iter()
                .all(|count| *count <= 3),
            "distance=1 has at most a three-position active DP band"
        );
        assert!(
            override_count <= row_count * 3 && override_count < row_count * scalar_count,
            "row-active overrides must stay well below all-scalars-per-row construction"
        );

        let repeated = UnicodeLevenshtein::new("بببببببببببببببب", "", 1).unwrap();
        assert!(
            repeated
                .exact_override_counts
                .iter()
                .all(|count| *count <= 1),
            "the same active scalar must install only one exact override per row"
        );
    }

    #[test]
    fn ascii_fuzzy_automaton_keeps_exact_prefix_semantics() {
        let params = FtsSearchParams::new()
            .with_fuzziness(Some(1))
            .with_prefix_length(1);
        let automaton = FuzzyAutomaton::new("cafe", &DocType::Text, &params).unwrap();

        assert!(automaton_accepts(&automaton, "café".as_bytes()));
        assert!(!automaton_accepts(&automaton, "dafé".as_bytes()));
    }

    #[test]
    fn unicode_levenshtein_handles_shared_utf8_lead_bytes() {
        let mut builder = fst::MapBuilder::memory();
        for (token_id, token) in ["assemblees", "café", "بسرعة"].into_iter().enumerate() {
            builder.insert(token, token_id as u64).unwrap();
        }
        let map = builder.into_map();
        let mut matches = Vec::new();

        let mut stream = map
            .search(UnicodeLevenshtein::new("بسرع", "", 1).unwrap())
            .into_stream();
        while let Some((key, _)) = stream.next() {
            matches.push(std::str::from_utf8(key).unwrap().to_owned());
        }

        assert_eq!(matches, vec!["بسرعة"]);
    }
}
