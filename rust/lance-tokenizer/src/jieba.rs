// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use crate::{TextAnalyzer, TextAnalyzerBuilder, Token, TokenStream, Tokenizer};

/// Shares one dictionary between clones: the index hands every query its own tokenizer
/// clone, and copying the whole dictionary (about 350k words) cost each query and each
/// opened segment hundreds of milliseconds.
#[derive(Clone)]
pub struct JiebaTokenizer {
    jieba: Arc<jieba_rs::Jieba>,
}

impl JiebaTokenizer {
    pub fn new(jieba: jieba_rs::Jieba) -> Self {
        Self {
            jieba: Arc::new(jieba),
        }
    }

    pub fn analyzer(self) -> TextAnalyzer {
        TextAnalyzer::builder(self).build()
    }

    pub fn analyzer_builder(self) -> TextAnalyzerBuilder {
        TextAnalyzer::builder(self).dynamic()
    }
}

pub struct JiebaTokenStream {
    tokens: Vec<Token>,
    index: usize,
}

impl TokenStream for JiebaTokenStream {
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

impl Tokenizer for JiebaTokenizer {
    type TokenStream<'a> = JiebaTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        let mut indices = text.char_indices().collect::<Vec<_>>();
        indices.push((text.len(), '\0'));
        let orig_tokens = self
            .jieba
            .tokenize(text, jieba_rs::TokenizeMode::Search, true);
        let tokens = orig_tokens
            .into_iter()
            .map(|token| Token {
                offset_from: indices[token.start].0,
                offset_to: indices[token.end].0,
                position: token.start,
                text: text[indices[token.start].0..indices[token.end].0].to_owned(),
                position_length: token.end - token.start,
            })
            .collect();
        JiebaTokenStream { tokens, index: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clones_share_the_dictionary() {
        let mut jieba = jieba_rs::Jieba::empty();
        jieba.add_word("蓝天", Some(10), None);
        jieba.add_word("白云", Some(10), None);
        let tokenizer = JiebaTokenizer::new(jieba);
        let clone = tokenizer.clone();
        assert!(Arc::ptr_eq(&tokenizer.jieba, &clone.jieba));

        let mut clone = clone;
        let mut stream = clone.token_stream("蓝天白云");
        let mut texts = Vec::new();
        while stream.advance() {
            texts.push(stream.token().text.clone());
        }
        assert_eq!(texts, ["蓝天", "白云"]);
    }
}
