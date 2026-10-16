// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-License-Identifier: MIT
// Adapted from Tantivy v0.24.2 stop-word filter.
// Copyright (c) 2017-present Tantivy contributors.

#[path = "stop_word_filter/stopwords.rs"]
mod stopwords;

use std::collections::HashSet;
use std::sync::Arc;

use crate::{Language, Token, TokenFilter, TokenStream, Tokenizer};

// Chinese function words for the all-language union, picked from the NLTK
// Chinese list that `stop_words::get("zh")` returns (841 entries). That list
// also holds content words (所有, 结果, 使用, 取得, 以上, 必要, ...). Because
// the union is applied to every text, and kanji are shared with Japanese, those
// entries removed searchable words from Japanese text too (#9890). This keeps
// only particles, pronouns, demonstratives, conjunctions, prepositions,
// question words and interjections, and leaves out any entry that is a content
// word in Chinese or Japanese.
#[rustfmt::skip]
const CHINESE: &[&str] = &[
    // particles, aspect markers and modal particles
    "的", "了", "着", "过", "得", "之", "乎", "也", "矣", "焉", "哉", "兮", "么", "吗", "吧", "呢",
    "啊", "呀", "嘛", "啦", "罢了", "而已", "的话",
    // pronouns and demonstratives
    "我", "你", "您", "他", "她", "它", "俺", "咱", "们", "我们", "你们", "他们", "她们", "它们",
    "咱们", "俺们", "我的", "你的", "他的", "她的", "它的", "它们的", "彼", "其", "此", "某", "各",
    "每", "这", "那", "哪", "这个", "那个", "哪个", "这些", "那些", "哪些", "这里", "那里", "哪里",
    "这儿", "那儿", "哪儿", "这样", "那样", "这么", "那么", "这种", "其中", "其他", "及其",
    // question words
    "谁", "何", "啥", "咋", "怎", "几", "什么", "怎么", "怎样", "为什么",
    // conjunctions
    "和", "与", "及", "且", "而", "或", "但", "则", "即", "又", "并", "既", "虽", "若", "如", "因",
    "倘", "以及", "而且", "并且", "或者", "还是", "但是", "可是", "然而", "因为", "所以", "因此",
    "于是", "然后", "如果", "虽然", "即使", "尽管", "不但", "不仅", "而是", "只是",
    // prepositions and coverbs
    "在", "于", "从", "对", "把", "被", "让", "给", "跟", "向", "到", "至", "由", "为", "以", "按",
    "据", "凭", "较", "连", "关于", "对于", "由于", "为了",
    // auxiliaries and common verbs used as function words
    "是", "有", "可", "将", "就", "乃", "所", "该", "等", "另", "别", "不是", "就是", "是否",
    "是的", "没有", "已经", "可以", "还有",
    // interjections
    "哎", "唉", "哦", "嗯", "喂", "哈", "哇", "哟", "呃", "呵", "嘿", "嘻", "咦", "呗", "呜", "哼",
    "嗳", "嗬", "喏", "哩", "啐", "呸", "吱", "咚", "嘎", "嘘", "哗", "咳", "呕",
];

fn all_stop_words() -> impl Iterator<Item = &'static str> {
    [
        stop_words::get("ar"),
        stopwords::DANISH,
        stopwords::DUTCH,
        // Use the fuller `stop-words` crate English list (~198 words) rather
        // than the local Tantivy-style list (~33 words), which omits extremely
        // common pronouns/function words (you, my, your, we, she, what, ...).
        // Those omissions let the highest-frequency English tokens through the
        // ICU stop-word path and build pathologically large posting lists.
        stop_words::get("en"),
        stopwords::FINNISH,
        stopwords::FRENCH,
        stopwords::GERMAN,
        stop_words::get("el"),
        stopwords::HUNGARIAN,
        stopwords::ITALIAN,
        stopwords::NORWEGIAN,
        stopwords::PORTUGUESE,
        stop_words::get("ro"),
        stopwords::RUSSIAN,
        stopwords::SPANISH,
        stopwords::SWEDISH,
        stop_words::get("ta"),
        stop_words::get("tr"),
        CHINESE,
        stop_words::get("ja"),
        stop_words::get("ko"),
    ]
    .into_iter()
    .flat_map(|words| words.iter().copied())
}

#[derive(Clone)]
pub struct StopWordFilter {
    words: Arc<HashSet<String>>,
}

impl StopWordFilter {
    pub fn new(language: Language) -> Option<Self> {
        let words = match language {
            Language::Arabic => stop_words::get("ar"),
            Language::Danish => stopwords::DANISH,
            Language::Dutch => stopwords::DUTCH,
            // Use the fuller `stop-words` crate English list (~198 words); the
            // local Tantivy-style list (~33 words) omits common pronouns/function
            // words (you, my, your, we, ...) that would otherwise leak through
            // stop-word removal and build pathologically large posting lists.
            Language::English => stop_words::get("en"),
            Language::Finnish => stopwords::FINNISH,
            Language::French => stopwords::FRENCH,
            Language::German => stopwords::GERMAN,
            Language::Greek => stop_words::get("el"),
            Language::Hungarian => stopwords::HUNGARIAN,
            Language::Italian => stopwords::ITALIAN,
            Language::Norwegian => stopwords::NORWEGIAN,
            Language::Portuguese => stopwords::PORTUGUESE,
            Language::Romanian => stop_words::get("ro"),
            Language::Russian => stopwords::RUSSIAN,
            Language::Spanish => stopwords::SPANISH,
            Language::Swedish => stopwords::SWEDISH,
            Language::Tamil => stop_words::get("ta"),
            Language::Turkish => stop_words::get("tr"),
        };
        Some(Self::remove(words.iter().map(|word| (*word).to_owned())))
    }

    pub fn all() -> Self {
        Self::remove(all_stop_words().map(str::to_owned))
    }

    pub fn remove<W: IntoIterator<Item = String>>(words: W) -> Self {
        Self {
            words: Arc::new(words.into_iter().collect()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::all_stop_words;
    use crate::StopWordFilter;
    use std::collections::HashSet;

    #[test]
    fn test_external_stop_word_lists_are_available() {
        let words = all_stop_words().collect::<HashSet<_>>();
        for word in ["إلى", "και", "acesta", "அவர்", "ama", "的", "ある", "그리고"]
        {
            assert!(
                words.contains(word),
                "built-in stop words should contain {word}"
            );
        }
    }

    #[test]
    fn test_cjk_content_words_are_not_stop_words() {
        let words = all_stop_words().collect::<HashSet<_>>();
        // Japanese content words that were on the NLTK Chinese list, then
        // Chinese ones (#9890).
        for word in [
            "取得", "表示", "使用", "必要", "以上", "以下", "全部", "可能", "最近", "部分", "所有",
            "结果", "用", "本", "来", "要",
        ] {
            assert!(!words.contains(word), "{word} should not be a stop word");
        }
    }

    #[test]
    fn test_language_stop_word_lists_are_available() {
        for (language, word) in [
            (crate::Language::Arabic, "إلى"),
            (crate::Language::Greek, "και"),
            (crate::Language::Romanian, "acesta"),
            (crate::Language::Tamil, "அவர்"),
            (crate::Language::Turkish, "ama"),
        ] {
            let filter = StopWordFilter::new(language).unwrap();
            assert!(
                filter.words.contains(word),
                "{language:?} should contain {word}"
            );
        }
    }
}

impl TokenFilter for StopWordFilter {
    type Tokenizer<T: Tokenizer> = StopWordFilterWrapper<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        StopWordFilterWrapper {
            words: self.words,
            inner: tokenizer,
        }
    }
}

#[derive(Clone)]
pub struct StopWordFilterWrapper<T> {
    words: Arc<HashSet<String>>,
    inner: T,
}

impl<T: Tokenizer> Tokenizer for StopWordFilterWrapper<T> {
    type TokenStream<'a> = StopWordFilterStream<T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        StopWordFilterStream {
            words: self.words.clone(),
            tail: self.inner.token_stream(text),
        }
    }
}

pub struct StopWordFilterStream<T> {
    words: Arc<HashSet<String>>,
    tail: T,
}

impl<T> StopWordFilterStream<T> {
    fn predicate(&self, token: &Token) -> bool {
        !self.words.contains(&token.text)
    }
}

impl<T: TokenStream> TokenStream for StopWordFilterStream<T> {
    fn advance(&mut self) -> bool {
        while self.tail.advance() {
            if self.predicate(self.tail.token()) {
                return true;
            }
        }
        false
    }

    fn token(&self) -> &Token {
        self.tail.token()
    }

    fn token_mut(&mut self) -> &mut Token {
        self.tail.token_mut()
    }
}
