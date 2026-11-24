// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-License-Identifier: MIT
// Adapted from Tantivy v0.24.2 stemmer filter.
// Copyright (c) 2017-present Tantivy contributors.

use std::borrow::Cow;
use std::collections::HashMap;

use frostem::Algorithm;
use serde::{Deserialize, Serialize};

use crate::{Token, TokenFilter, TokenStream, Tokenizer};

#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Copy, Clone)]
pub enum Language {
    Arabic,
    Danish,
    Dutch,
    English,
    Finnish,
    French,
    German,
    Greek,
    Hungarian,
    Italian,
    Norwegian,
    Portuguese,
    Romanian,
    Russian,
    Spanish,
    Swedish,
    Tamil,
    Turkish,
}

impl Language {
    fn algorithm(self) -> Algorithm {
        match self {
            Self::Arabic => Algorithm::Arabic,
            Self::Danish => Algorithm::Danish,
            Self::Dutch => Algorithm::Dutch,
            Self::English => Algorithm::English,
            Self::Finnish => Algorithm::Finnish,
            Self::French => Algorithm::French,
            Self::German => Algorithm::German,
            Self::Greek => Algorithm::Greek,
            Self::Hungarian => Algorithm::Hungarian,
            Self::Italian => Algorithm::Italian,
            Self::Norwegian => Algorithm::Norwegian,
            Self::Portuguese => Algorithm::Portuguese,
            Self::Romanian => Algorithm::Romanian,
            Self::Russian => Algorithm::Russian,
            Self::Spanish => Algorithm::Spanish,
            Self::Swedish => Algorithm::Swedish,
            Self::Tamil => Algorithm::Tamil,
            Self::Turkish => Algorithm::Turkish,
        }
    }
}

#[derive(Clone)]
pub struct Stemmer {
    stemmer_algorithm: Algorithm,
}

impl Stemmer {
    pub fn new(language: Language) -> Self {
        Self {
            stemmer_algorithm: language.algorithm(),
        }
    }
}

impl Default for Stemmer {
    fn default() -> Self {
        Self::new(Language::English)
    }
}

impl TokenFilter for Stemmer {
    type Tokenizer<T: Tokenizer> = StemmerFilter<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        StemmerFilter {
            stemmer_algorithm: self.stemmer_algorithm,
            inner: tokenizer,
            cache: (self.stemmer_algorithm == Algorithm::English).then(StemCache::default),
        }
    }
}

pub struct StemmerFilter<T> {
    stemmer_algorithm: Algorithm,
    inner: T,
    cache: Option<StemCache>,
}

// Keep the cache local to each analyzer worker instead of copying a populated
// dictionary when flat search or index construction clones its tokenizer.
impl<T: Clone> Clone for StemmerFilter<T> {
    fn clone(&self) -> Self {
        Self {
            stemmer_algorithm: self.stemmer_algorithm,
            inner: self.inner.clone(),
            cache: (self.stemmer_algorithm == Algorithm::English).then(StemCache::default),
        }
    }
}

impl<T: Tokenizer> Tokenizer for StemmerFilter<T> {
    type TokenStream<'a> = StemmerTokenStream<'a, T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        StemmerTokenStream {
            tail: self.inner.token_stream(text),
            stemmer: frostem::Stemmer::new(self.stemmer_algorithm),
            cache: &mut self.cache,
        }
    }
}

pub struct StemmerTokenStream<'a, T> {
    tail: T,
    stemmer: frostem::Stemmer,
    cache: &'a mut Option<StemCache>,
}

// Bound both dictionary cardinality and string payloads. Long tokens retain
// the ordinary stemming path, including when no RemoveLongFilter is present.
const MAX_CACHED_STEMS: usize = 1024;
const MAX_CACHED_TOKEN_BYTES: usize = 64;
// Retire the dictionary when fewer than 45% of cacheable inputs repeat in a
// window. A low hit rate can cost more than calling Snowball directly.
const CACHE_SAMPLE_TOKENS: usize = 1024;

#[derive(Default)]
struct StemCache {
    // None records an unchanged token, so a hit can retain the input buffer.
    stems: HashMap<String, Option<String>>,
    lookups: usize,
    hits: usize,
}

impl<T: TokenStream> TokenStream for StemmerTokenStream<'_, T> {
    fn advance(&mut self) -> bool {
        if !self.tail.advance() {
            return false;
        }
        let token = self.tail.token_mut();
        // English Snowball returns before its prelude for fewer than three
        // characters. Fewer than three UTF-8 bytes is a conservative subset.
        if token.text.len() < 3 && self.stemmer.algorithm() == Algorithm::English {
            return true;
        }
        // Snowball handles very short tokens cheaply; hashing and copying
        // them into the dictionary can cost more than stemming them directly.
        // Cache the profiled English ASCII path. Other languages and Unicode
        // retain direct Snowball processing, whose cost can differ considerably.
        if self.cache.is_none()
            || !(3..=MAX_CACHED_TOKEN_BYTES).contains(&token.text.len())
            || !token.text.is_ascii()
        {
            if let Cow::Owned(stemmed) = self.stemmer.stem(&token.text) {
                token.text = stemmed;
            }
            return true;
        }
        // High-cardinality text may cost more to cache than to stem. Recheck
        // each window so a repeated prefix cannot keep a low-value cache alive.
        if self.cache.as_ref().is_some_and(|cache| {
            cache.lookups == CACHE_SAMPLE_TOKENS && cache.hits < CACHE_SAMPLE_TOKENS * 9 / 20
        }) {
            *self.cache = None;
        }
        if let Some(cache) = self.cache.as_mut() {
            if cache.lookups == CACHE_SAMPLE_TOKENS {
                cache.lookups = 0;
                cache.hits = 0;
            }
            cache.lookups += 1;
            if let Some(stemmed) = cache.stems.get(token.text.as_str()) {
                cache.hits += 1;
                if let Some(stemmed) = stemmed {
                    token.text.clone_from(stemmed);
                }
                return true;
            }
        }
        let stemmed = self.stemmer.stem(&token.text);
        if let Some(cache) = self.cache.as_mut()
            && stemmed.len() <= MAX_CACHED_TOKEN_BYTES
        {
            if cache.stems.len() == MAX_CACHED_STEMS {
                cache.stems.clear();
            }
            let cached = match &stemmed {
                Cow::Owned(stemmed) => Some(stemmed.clone()),
                Cow::Borrowed(_) => None,
            };
            cache.stems.insert(token.text.clone(), cached);
        }
        if let Cow::Owned(stemmed) = stemmed {
            token.text = stemmed;
        }
        true
    }

    fn token(&self) -> &Token {
        self.tail.token()
    }

    fn token_mut(&mut self) -> &mut Token {
        self.tail.token_mut()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Language, RawTokenizer, SimpleTokenizer, Stemmer, TextAnalyzer, Token, TokenStream,
    };

    #[test]
    fn test_short_english_tokens_are_unchanged() {
        let reference = frostem::Stemmer::new(Language::English.algorithm());
        let mut analyzer = TextAnalyzer::builder(RawTokenizer::default())
            .filter(Stemmer::default())
            .build();
        // Exhaust all valid UTF-8 strings shorter than three bytes, including
        // punctuation, upper case, control characters and multibyte letters.
        let texts = std::iter::once(String::new())
            .chain((0..=0x7f).map(|value| char::from(value).to_string()))
            .chain((0..=0x7f).flat_map(|first| {
                (0..=0x7f).map(move |second| format!("{}{}", char::from(first), char::from(second)))
            }))
            .chain(
                (0x80..=0x7ff)
                    .filter_map(char::from_u32)
                    .map(|ch| ch.to_string()),
            );
        for text in texts {
            assert_eq!(reference.stem(&text), text);
            let mut stream = analyzer.token_stream(&text);
            assert_eq!(stream.next().unwrap().text, text);
            assert!(!stream.advance());
        }
    }

    #[test]
    fn test_repeated_stems_keep_each_occurrence_position() {
        let mut analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
            .filter(Stemmer::default())
            .build();
        let mut stream = analyzer.token_stream("running runs runner running");
        for (position, (start, end, text)) in [
            (0, 7, "run"),
            (8, 12, "run"),
            (13, 19, "runner"),
            (20, 27, "run"),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                stream.next().unwrap(),
                &Token {
                    offset_from: start,
                    offset_to: end,
                    position,
                    text: text.into(),
                    position_length: 1,
                }
            );
        }
        assert!(!stream.advance());
    }

    #[rstest::rstest]
    #[case::arabic(Language::Arabic, "والكتابات")]
    #[case::danish(Language::Danish, "hestens")]
    #[case::dutch(Language::Dutch, "paarden")]
    #[case::english(Language::English, "running")]
    #[case::english_unchanged(Language::English, "pear")]
    #[case::english_unicode(Language::English, "café")]
    #[case::finnish(Language::Finnish, "taloissa")]
    #[case::french(Language::French, "chevaux")]
    #[case::german(Language::German, "häusern")]
    #[case::greek(Language::Greek, "αντιθετε")]
    #[case::hungarian(Language::Hungarian, "házakban")]
    #[case::italian(Language::Italian, "cavalli")]
    #[case::norwegian(Language::Norwegian, "hestens")]
    #[case::portuguese(Language::Portuguese, "corações")]
    #[case::romanian(Language::Romanian, "caselor")]
    #[case::russian(Language::Russian, "машины")]
    #[case::spanish(Language::Spanish, "caballos")]
    #[case::swedish(Language::Swedish, "hästarna")]
    #[case::tamil(Language::Tamil, "மரங்கள்")]
    #[case::turkish(Language::Turkish, "evlerimizden")]
    fn test_repeated_stems_preserve_tokens(#[case] language: Language, #[case] word: &str) {
        let reference = frostem::Stemmer::new(language.algorithm());
        let mut analyzer = TextAnalyzer::builder(RawTokenizer::default())
            .filter(Stemmer::new(language))
            .build();
        // Reuse across documents, including empty/oversized tokens and a
        // mutation of the previous output by a downstream filter/consumer.
        let long = word.repeat(100);
        for text in [word, "", word, &long, word, "RUNNING", word] {
            let expected = Token {
                offset_from: 0,
                offset_to: text.len(),
                position: 0,
                text: reference.stem(text).into_owned(),
                position_length: 1,
            };
            let mut stream = analyzer.token_stream(text);
            assert!(stream.advance());
            assert_eq!(stream.token(), &expected);
            stream.token_mut().text.push_str("changed by consumer");
            assert!(!stream.advance());
        }
        let mut cloned = analyzer.clone();
        let mut stream = cloned.token_stream(word);
        assert_eq!(stream.next().unwrap().text, reference.stem(word));
    }

    #[rstest::rstest]
    fn test_stemming_after_dictionary_turnover(
        #[values(false, true)] repeated_prefix: bool,
        #[values(false, true)] repeat_each_word: bool,
    ) {
        let reference = frostem::Stemmer::new(Language::English.algorithm());
        let mut analyzer = TextAnalyzer::builder(RawTokenizer::default())
            .filter(Stemmer::default())
            .build();
        if repeated_prefix {
            for _ in 0..super::CACHE_SAMPLE_TOKENS {
                let mut stream = analyzer.token_stream("running");
                assert_eq!(stream.next().unwrap().text, "run");
            }
        }
        for pass in 0..2 {
            for index in 0..(super::MAX_CACHED_STEMS * 2) {
                let text = format!("running{index}");
                for _ in 0..if repeat_each_word { 2 } else { 1 } {
                    let mut stream = analyzer.token_stream(&text);
                    let token = stream.next().unwrap();
                    assert_eq!(token.text, reference.stem(&text), "pass {pass}");
                    assert_eq!(token.position, 0);
                    assert!(!stream.advance());
                }
            }
        }
        for _ in 0..3 {
            let mut stream = analyzer.token_stream("running");
            assert_eq!(stream.next().unwrap().text, "run");
        }
    }

    #[test]
    fn test_greek_stemmer_handles_multibyte_suffixes() {
        let mut analyzer = TextAnalyzer::builder(RawTokenizer::default())
            .filter(Stemmer::new(Language::Greek))
            .build();
        let mut stream = analyzer.token_stream("αντιθετε");

        assert!(stream.advance());
        assert_eq!(stream.token().text, "ανετ");
    }
}
