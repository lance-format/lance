// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use icu_segmenter::iterators::WordBreakIterator;
use icu_segmenter::scaffold::Utf8;
use icu_segmenter::{WordSegmenter, WordSegmenterBorrowed, options::WordBreakInvariantOptions};

use crate::{TextAnalyzer, TextAnalyzerBuilder, Token, TokenStream, Tokenizer};

#[derive(Clone)]
pub struct IcuTokenizer {
    segmenter: WordSegmenterBorrowed<'static>,
    split_on_non_alphanumeric: bool,
}

impl Default for IcuTokenizer {
    fn default() -> Self {
        Self {
            segmenter: WordSegmenter::new_dictionary(WordBreakInvariantOptions::default()),
            split_on_non_alphanumeric: false,
        }
    }
}

impl IcuTokenizer {
    /// Split ICU word segments again on simple-tokenizer delimiters.
    pub fn with_simple_split(mut self) -> Self {
        self.split_on_non_alphanumeric = true;
        self
    }

    /// Convert to a tokenizer that borrows its input and emits tokens on demand.
    ///
    /// The streaming tokenizer reuses one token buffer instead of allocating a
    /// token list for the complete document. Its streams borrow their input;
    /// the original tokenizer continues to produce owned streams.
    ///
    /// ```
    /// use lance_tokenizer::{IcuTokenizer, TokenStream, Tokenizer};
    /// let mut tokenizer = IcuTokenizer::default().into_streaming();
    /// let mut stream = tokenizer.token_stream("Hello, 世界!");
    /// assert!(stream.advance());
    /// assert_eq!(stream.token().text, "Hello");
    /// ```
    pub fn into_streaming(self) -> IcuStreamingTokenizer {
        IcuStreamingTokenizer { inner: self }
    }

    pub fn analyzer(self) -> TextAnalyzer {
        TextAnalyzer::builder(self).build()
    }

    pub fn analyzer_builder(self) -> TextAnalyzerBuilder {
        TextAnalyzer::builder(self).dynamic()
    }
}

pub struct IcuTokenStream {
    tokens: Vec<Token>,
    index: usize,
}

fn push_token(tokens: &mut Vec<Token>, text: &str, offset_from: usize, offset_to: usize) {
    if offset_from == offset_to {
        return;
    }

    let token_text = &text[offset_from..offset_to];
    if token_text.chars().any(char::is_alphanumeric) {
        tokens.push(Token {
            offset_from,
            offset_to,
            position: tokens.len(),
            text: token_text.to_owned(),
            position_length: 1,
        });
    }
}

fn push_tokens_split_on_non_alphanumeric(
    tokens: &mut Vec<Token>,
    text: &str,
    offset_from: usize,
    offset_to: usize,
) {
    let mut part_start = offset_from;
    for (relative_offset, c) in text[offset_from..offset_to].char_indices() {
        if !c.is_alphanumeric() {
            let delimiter_offset = offset_from + relative_offset;
            push_token(tokens, text, part_start, delimiter_offset);
            part_start = delimiter_offset + c.len_utf8();
        }
    }
    push_token(tokens, text, part_start, offset_to);
}

impl TokenStream for IcuTokenStream {
    fn advance(&mut self) -> bool {
        if self.index < self.tokens.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.tokens[self.index - 1]
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.index - 1]
    }
}

impl Tokenizer for IcuTokenizer {
    type TokenStream<'a> = IcuTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        let mut boundaries = self.segmenter.segment_str(text);
        let mut tokens = Vec::new();
        let Some(mut offset_from) = boundaries.next() else {
            return IcuTokenStream { tokens, index: 0 };
        };

        for offset_to in boundaries {
            if self.split_on_non_alphanumeric {
                push_tokens_split_on_non_alphanumeric(&mut tokens, text, offset_from, offset_to);
            } else {
                push_token(&mut tokens, text, offset_from, offset_to);
            }
            offset_from = offset_to;
        }

        IcuTokenStream { tokens, index: 0 }
    }
}

/// ICU segmentation with a borrowed input and one reusable token buffer.
///
/// Construct this tokenizer with [`IcuTokenizer::into_streaming`].
#[derive(Clone)]
pub struct IcuStreamingTokenizer {
    inner: IcuTokenizer,
}

/// Borrowed token stream produced by [`IcuStreamingTokenizer`].
pub struct IcuStreamingTokenStream<'a> {
    text: &'a str,
    boundaries: WordBreakIterator<'static, 'a, Utf8>,
    offset: usize,
    segment_end: usize,
    split_on_non_alphanumeric: bool,
    next_position: usize,
    token: Token,
}

impl TokenStream for IcuStreamingTokenStream<'_> {
    fn advance(&mut self) -> bool {
        loop {
            if self.offset == self.segment_end {
                let Some(end) = self.boundaries.next() else {
                    return false;
                };
                self.segment_end = end;
                if self.offset == end {
                    continue;
                }
            }
            let start = self.offset;
            let mut end = self.segment_end;
            self.offset = self.segment_end;
            if self.split_on_non_alphanumeric {
                for (relative, character) in self.text[start..end].char_indices() {
                    if !character.is_alphanumeric() {
                        end = start + relative;
                        self.offset = end + character.len_utf8();
                        break;
                    }
                }
            }
            if start == end || !self.text[start..end].chars().any(char::is_alphanumeric) {
                continue;
            }
            self.token.offset_from = start;
            self.token.offset_to = end;
            self.token.position = self.next_position;
            self.token.position_length = 1;
            self.token.text.clear();
            self.token.text.push_str(&self.text[start..end]);
            self.next_position += 1;
            return true;
        }
    }

    fn token(&self) -> &Token {
        &self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.token
    }
}

impl Tokenizer for IcuStreamingTokenizer {
    type TokenStream<'a> = IcuStreamingTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        IcuStreamingTokenStream {
            text,
            boundaries: self.inner.segmenter.segment_str(text),
            offset: 0,
            segment_end: 0,
            split_on_non_alphanumeric: self.inner.split_on_non_alphanumeric,
            next_position: 0,
            token: Token::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{IcuTokenizer, Token, TokenStream, Tokenizer};

    #[test]
    fn test_icu_owned_stream_outlives_input() {
        let mut stream = {
            let text = String::from("hello 世界");
            let mut tokenizer = IcuTokenizer::default();
            tokenizer.token_stream(&text)
        };
        assert!(stream.advance());
        assert_eq!(stream.token().text, "hello");
        assert!(stream.advance());
        assert_eq!(stream.token().text, "世界");
        assert!(!stream.advance());
    }

    #[rstest::rstest]
    #[case::mixed("Hello, こんにちは世界!", false)]
    #[case::apostrophe("Mark'd ye his words?", false)]
    #[case::split("foo_bar__baz-alpha.beta", true)]
    #[case::chinese("你好，世界！测试 hello_world", true)]
    #[case::greek("ΟΣ ΟΣΑ Σίσυφος", false)]
    #[case::empty("", false)]
    #[case::punctuation(" !🙂__---\n", true)]
    fn test_streaming_icu_matches_owned(#[case] text: &str, #[case] split: bool) {
        let expected = collect_tokens_with_split(text, split);
        let tokenizer = if split {
            IcuTokenizer::default().with_simple_split()
        } else {
            IcuTokenizer::default()
        };
        let mut tokenizer = tokenizer.into_streaming();
        let mut stream = tokenizer.token_stream(text);
        let mut actual = Vec::new();
        stream.process(&mut |token| actual.push(token.clone()));
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_streaming_icu_resets_mutated_token_fields() {
        let mut tokenizer = IcuTokenizer::default().with_simple_split().into_streaming();
        let mut stream = tokenizer.token_stream("_hello__世界_");
        assert!(stream.advance());
        assert_eq!(stream.token().text, "hello");
        let token = stream.token_mut();
        token.text.push_str("changed");
        token.position = 99;
        token.position_length = 7;
        token.offset_to = 999;
        assert!(stream.advance());
        assert_eq!(
            stream.token(),
            &Token {
                text: "世界".to_owned(),
                offset_from: 8,
                offset_to: 14,
                position: 1,
                position_length: 1
            }
        );
        assert!(!stream.advance());
        assert!(!stream.advance());
    }

    fn collect_tokens_with_split(text: &str, split_on_non_alphanumeric: bool) -> Vec<Token> {
        let mut tokenizer = IcuTokenizer::default();
        if split_on_non_alphanumeric {
            tokenizer = tokenizer.with_simple_split();
        }
        let mut stream = tokenizer.token_stream(text);
        let mut tokens = Vec::new();
        stream.process(&mut |token| tokens.push(token.clone()));
        tokens
    }

    fn collect_tokens(text: &str) -> Vec<Token> {
        collect_tokens_with_split(text, false)
    }

    #[test]
    fn test_icu_tokenizer_segments_mixed_text() {
        let tokens = collect_tokens("Hello, こんにちは世界!");

        assert_eq!(
            tokens
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            vec!["Hello", "こんにちは", "世界"]
        );
        assert_eq!(
            tokens
                .iter()
                .map(|token| (token.offset_from, token.offset_to, token.position))
                .collect::<Vec<_>>(),
            vec![(0, 5, 0), (7, 22, 1), (22, 28, 2)]
        );
    }

    #[test]
    fn test_icu_tokenizer_skips_non_word_segments() {
        let tokens = collect_tokens("Mark'd ye his words?");

        assert_eq!(
            tokens
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            vec!["Mark'd", "ye", "his", "words"]
        );
    }

    #[test]
    fn test_icu_tokenizer_splits_on_non_alphanumeric_when_enabled() {
        let tokens = collect_tokens_with_split("foo_bar__baz-alpha.beta", true);

        assert_eq!(
            tokens
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            vec!["foo", "bar", "baz", "alpha", "beta"]
        );
        assert_eq!(
            tokens
                .iter()
                .map(|token| (token.offset_from, token.offset_to, token.position))
                .collect::<Vec<_>>(),
            vec![(0, 3, 0), (4, 7, 1), (9, 12, 2), (13, 18, 3), (19, 23, 4)]
        );
    }

    #[test]
    fn test_icu_tokenizer_split_control_keeps_icu_segmentation() {
        let tokens = collect_tokens_with_split("hello_world こんにちは世界", true);

        assert_eq!(
            tokens
                .iter()
                .map(|token| token.text.as_str())
                .collect::<Vec<_>>(),
            vec!["hello", "world", "こんにちは", "世界"]
        );
    }
}
