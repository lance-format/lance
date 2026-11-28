// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::VecDeque;
use std::mem;

use crate::{Token, TokenFilter, TokenStream, Tokenizer};

/// Splits code identifiers into subword tokens.
///
/// The original identifier can be preserved at the first subword position. If
/// a token is not a compound identifier then it is passed through unchanged.
///
/// # Examples
///
/// ```
/// use lance_tokenizer::{CodeLexTokenizer, TextAnalyzer, TokenStream, WordDelimiterFilter};
///
/// let mut analyzer = TextAnalyzer::builder(CodeLexTokenizer::new(false))
///     .filter(WordDelimiterFilter::new(true, true))
///     .build();
/// let mut stream = analyzer.token_stream("parseHTML2JSON");
///
/// let mut tokens = Vec::new();
/// stream.process(&mut |token| tokens.push(token.text.clone()));
/// assert_eq!(tokens, vec!["parseHTML2JSON", "parse", "HTML", "2", "JSON"]);
/// ```
#[derive(Clone)]
pub struct WordDelimiterFilter {
    preserve_original: bool,
    split_on_numerics: bool,
}

impl WordDelimiterFilter {
    pub fn new(preserve_original: bool, split_on_numerics: bool) -> Self {
        Self {
            preserve_original,
            split_on_numerics,
        }
    }
}

impl TokenFilter for WordDelimiterFilter {
    type Tokenizer<T: Tokenizer> = WordDelimiterFilterWrapper<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        WordDelimiterFilterWrapper {
            tokenizer,
            preserve_original: self.preserve_original,
            split_on_numerics: self.split_on_numerics,
        }
    }
}

#[derive(Clone)]
/// Tokenizer wrapper produced by [`WordDelimiterFilter`].
pub struct WordDelimiterFilterWrapper<T> {
    tokenizer: T,
    preserve_original: bool,
    split_on_numerics: bool,
}

impl<T: Tokenizer> Tokenizer for WordDelimiterFilterWrapper<T> {
    type TokenStream<'a> = WordDelimiterTokenStream<T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        WordDelimiterTokenStream {
            tail: self.tokenizer.token_stream(text),
            current: Token::default(),
            pending: VecDeque::new(),
            preserve_original: self.preserve_original,
            split_on_numerics: self.split_on_numerics,
            position_offset: 0,
            parts: Vec::new(),
            spare_texts: Vec::new(),
        }
    }
}

/// Token stream produced by [`WordDelimiterFilterWrapper`].
pub struct WordDelimiterTokenStream<T> {
    tail: T,
    current: Token,
    pending: VecDeque<Token>,
    preserve_original: bool,
    split_on_numerics: bool,
    position_offset: usize,
    // Reused across tokens to avoid allocating per identifier.
    parts: Vec<(usize, usize)>,
    // Text buffers of emitted parts, reused for the parts of later tokens.
    spare_texts: Vec<String>,
}

fn is_identifier_text(text: &str) -> bool {
    text.chars().any(|ch| ch == '_' || ch.is_alphanumeric())
}

fn split_identifier(text: &str, split_on_numerics: bool, pieces: &mut Vec<(usize, usize)>) {
    pieces.clear();
    let mut segment_start = None;
    for (offset, ch) in text.char_indices() {
        if ch == '_' {
            if let Some(start) = segment_start.take() {
                split_segment(text, start, offset, split_on_numerics, pieces);
            }
        } else if segment_start.is_none() {
            segment_start = Some(offset);
        }
    }
    if let Some(start) = segment_start {
        split_segment(text, start, text.len(), split_on_numerics, pieces);
    }
}

fn split_segment(
    text: &str,
    start: usize,
    end: usize,
    split_on_numerics: bool,
    pieces: &mut Vec<(usize, usize)>,
) {
    let mut chars = text[start..end].char_indices().peekable();
    let Some((_, mut prev)) = chars.next() else {
        return;
    };
    let mut piece_start = start;
    while let Some((offset, cur)) = chars.next() {
        let next = chars.peek().map(|&(_, next)| next);
        if is_boundary(prev, cur, next, split_on_numerics) {
            pieces.push((piece_start, start + offset));
            piece_start = start + offset;
        }
        prev = cur;
    }
    pieces.push((piece_start, end));
}

fn is_boundary(prev: char, cur: char, next: Option<char>, split_on_numerics: bool) -> bool {
    if split_on_numerics && prev.is_ascii_digit() != cur.is_ascii_digit() {
        return true;
    }
    if prev.is_lowercase() && cur.is_uppercase() {
        return true;
    }
    if prev.is_uppercase() && cur.is_uppercase() && next.is_some_and(char::is_lowercase) {
        return true;
    }
    false
}

fn load_token(target: &mut Token, source: &Token, position: usize, position_length: usize) {
    target.offset_from = source.offset_from;
    target.offset_to = source.offset_to;
    target.position = position;
    target.text.clone_from(&source.text);
    target.position_length = position_length;
}

impl<T: TokenStream> WordDelimiterTokenStream<T> {
    // Writes tokens that are not split straight into `current`, reusing its
    // text buffer, and queues the parts of split identifiers.
    fn load_current_tail_token(&mut self) {
        let token = self.tail.token();
        let adjusted_position = token.position.saturating_add(self.position_offset);
        if !is_identifier_text(&token.text) {
            load_token(
                &mut self.current,
                token,
                adjusted_position,
                token.position_length,
            );
            return;
        }

        split_identifier(&token.text, self.split_on_numerics, &mut self.parts);
        let parts = &self.parts;
        if parts.len() <= 1 {
            load_token(
                &mut self.current,
                token,
                adjusted_position,
                token.position_length,
            );
            return;
        }

        if self.preserve_original {
            load_token(&mut self.current, token, adjusted_position, parts.len());
        }

        for (part_idx, (start, end)) in parts.iter().copied().enumerate() {
            let mut text = self.spare_texts.pop().unwrap_or_default();
            text.clear();
            text.push_str(&token.text[start..end]);
            self.pending.push_back(Token {
                offset_from: token.offset_from + start,
                offset_to: token.offset_from + end,
                position: adjusted_position + part_idx,
                text,
                position_length: 1,
            });
        }
        self.position_offset += parts.len() - 1;
        if !self.preserve_original
            && let Some(first) = self.pending.pop_front()
        {
            self.set_current(first);
        }
    }

    fn set_current(&mut self, token: Token) {
        let previous = mem::replace(&mut self.current, token);
        self.spare_texts.push(previous.text);
    }
}

impl<T: TokenStream> TokenStream for WordDelimiterTokenStream<T> {
    fn advance(&mut self) -> bool {
        if let Some(token) = self.pending.pop_front() {
            self.set_current(token);
            return true;
        }
        if !self.tail.advance() {
            return false;
        }
        self.load_current_tail_token();
        true
    }

    fn token(&self) -> &Token {
        &self.current
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.current
    }
}

#[cfg(test)]
mod tests {
    use crate::{CodeLexTokenizer, TextAnalyzer, Token, WordDelimiterFilter};

    fn collect_tokens(text: &str) -> Vec<Token> {
        let mut analyzer = TextAnalyzer::builder(CodeLexTokenizer::new(false))
            .filter(WordDelimiterFilter::new(true, true))
            .build();
        let mut stream = analyzer.token_stream(text);
        let mut tokens = Vec::new();
        stream.process(&mut |token| tokens.push(token.clone()));
        tokens
    }

    #[test]
    fn test_word_delimiter_code_identifiers() {
        let tokens = collect_tokens("getUserName XMLHttpRequest parseHTML2JSON utf8_reader");
        let texts = tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            texts,
            vec![
                "getUserName",
                "get",
                "User",
                "Name",
                "XMLHttpRequest",
                "XML",
                "Http",
                "Request",
                "parseHTML2JSON",
                "parse",
                "HTML",
                "2",
                "JSON",
                "utf8_reader",
                "utf",
                "8",
                "reader",
            ]
        );
    }

    #[test]
    fn test_word_delimiter_positions_span_compounds() {
        let tokens = collect_tokens("getUserName next");
        let positions = tokens
            .iter()
            .map(|token| (token.text.as_str(), token.position, token.position_length))
            .collect::<Vec<_>>();
        assert_eq!(
            positions,
            vec![
                ("getUserName", 0, 3),
                ("get", 0, 1),
                ("User", 1, 1),
                ("Name", 2, 1),
                ("next", 3, 1),
            ]
        );
    }
}
