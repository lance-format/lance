// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-License-Identifier: MIT
// Adapted from lindera-tantivy v2.0.0.
// Copyright (c) lindera-tantivy contributors.

use std::path::Path;
use std::sync::Arc;

use lindera::token::Token as LinderaToken;
use lindera::tokenizer::{Tokenizer as LinderaCoreTokenizer, TokenizerBuilder};

use crate::{TextAnalyzer, TextAnalyzerBuilder, Token, TokenStream, Tokenizer};

#[derive(Clone)]
pub struct LinderaTokenizer {
    // The core tokenizer owns the dictionary, which can be hundreds of MB
    // (e.g. IPADIC). Query paths clone the tokenizer for every search, so it is
    // shared behind an `Arc` to make cloning cheap. `tokenize` only needs `&self`.
    tokenizer: Arc<LinderaCoreTokenizer>,
    token: Token,
}

impl LinderaTokenizer {
    pub fn new() -> std::io::Result<Self> {
        let builder = TokenizerBuilder::new().map_err(invalid_data)?;
        let tokenizer = builder.build().map_err(invalid_data)?;
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            token: Token::default(),
        })
    }

    pub fn from_file(file_path: &Path) -> std::io::Result<Self> {
        let builder = TokenizerBuilder::from_file(file_path).map_err(invalid_data)?;
        let tokenizer = builder.build().map_err(invalid_data)?;
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            token: Token::default(),
        })
    }

    pub fn from_segmenter(segmenter: lindera::segmenter::Segmenter) -> Self {
        Self {
            tokenizer: Arc::new(LinderaCoreTokenizer::new(segmenter)),
            token: Token::default(),
        }
    }

    pub fn analyzer(self) -> TextAnalyzer {
        TextAnalyzer::builder(self).build()
    }

    pub fn analyzer_builder(self) -> TextAnalyzerBuilder {
        TextAnalyzer::builder(self).dynamic()
    }
}

pub struct LinderaTokenStream<'a> {
    tokens: Vec<LinderaToken<'a>>,
    token: &'a mut Token,
}

impl<'a> TokenStream for LinderaTokenStream<'a> {
    fn advance(&mut self) -> bool {
        if self.tokens.is_empty() {
            return false;
        }
        let token = self.tokens.remove(0);
        self.token.text = token.surface.to_string();
        self.token.offset_from = token.byte_start;
        self.token.offset_to = token.byte_end;
        self.token.position = token.position;
        self.token.position_length = token.position_length;
        true
    }

    fn token(&self) -> &Token {
        self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        self.token
    }
}

impl Tokenizer for LinderaTokenizer {
    type TokenStream<'a> = LinderaTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        self.token.reset();
        LinderaTokenStream {
            tokens: self.tokenizer.tokenize(text).unwrap_or_default(),
            token: &mut self.token,
        }
    }
}

fn invalid_data(err: impl std::fmt::Debug) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{err:?}"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lindera::dictionary::{DictionaryBuilder, Metadata, load_fs_dictionary};
    use lindera::mode::Mode;
    use lindera::segmenter::Segmenter;

    use super::*;

    /// Build a tiny MeCab-format dictionary so the test does not depend on a
    /// downloaded one.
    fn tiny_tokenizer() -> LinderaTokenizer {
        let root = std::env::temp_dir().join(format!(
            "lance-lindera-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let (src, dst) = (root.join("src"), root.join("dst"));
        std::fs::create_dir_all(&src).unwrap();
        let files = [
            ("char.def", "DEFAULT 0 1 0\nSPACE 0 1 0\n0x0020 SPACE\n"),
            (
                "unk.def",
                "DEFAULT,0,0,0,名詞,一般,*,*,*,*,*\nSPACE,0,0,0,記号,空白,*,*,*,*,*\n",
            ),
            ("matrix.def", "1 1\n0 0 0\n"),
            (
                "words.csv",
                "東京,0,0,0,名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トーキョー\n",
            ),
        ];
        for (name, content) in files {
            std::fs::write(src.join(name), content).unwrap();
        }
        DictionaryBuilder::new(Metadata::default())
            .build_dictionary(&src, &dst)
            .unwrap();
        // The dictionary is read fully into memory, so the files can go now.
        let dictionary = load_fs_dictionary(&dst).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
        LinderaTokenizer::from_segmenter(Segmenter::new(Mode::Normal, dictionary, None))
    }

    #[test]
    fn test_clone_shares_dictionary() {
        let tokenizer = tiny_tokenizer();
        let mut cloned = tokenizer.clone();
        assert!(Arc::ptr_eq(&tokenizer.tokenizer, &cloned.tokenizer));
        assert_eq!(Arc::strong_count(&tokenizer.tokenizer), 2);

        // The clone keeps its own token state and still tokenizes.
        let mut stream = cloned.token_stream("東京");
        assert!(stream.advance());
        assert_eq!(stream.token().text, "東京");
        assert!(!stream.advance());

        drop(cloned);
        assert_eq!(Arc::strong_count(&tokenizer.tokenizer), 1);
    }
}
