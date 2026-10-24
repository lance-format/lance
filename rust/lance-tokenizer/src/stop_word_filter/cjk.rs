// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-License-Identifier: MIT
// Curated from the stop-words 0.10.0 NLTK/Stopwords ISO vocabulary.
// Upstream: https://github.com/cmccomb/rust-stop-words
/*
MIT License

Copyright (c) 2023 Chris McComb

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Function-word allowlist adapted from https://github.com/lance-format/lance/pull/10069.
// Keep core function words; preserve interjections, content senses and negation.
// ICU shares this list across languages, so Han homographs require conservative curation.
// Core Chinese stop words (的, 我, 有, 是) retain Japanese content-sense collisions.
pub(super) const CHINESE: &[&str] = &[
    // particles, aspect markers and modal particles
    "的",
    "了",
    "过",
    "之",
    "乎",
    "也",
    "矣",
    "焉",
    "哉",
    "兮",
    "么",
    "吗",
    "吧",
    "呢",
    "啊",
    "呀",
    "嘛",
    "啦",
    "罢了",
    "而已",
    "的话",
    // pronouns and demonstratives
    "我",
    "你",
    "您",
    "他",
    "她",
    "它",
    "俺",
    "咱",
    "们",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "咱们",
    "俺们",
    "我的",
    "你的",
    "他的",
    "她的",
    "它的",
    "它们的",
    "其",
    "此",
    "某",
    "各",
    "每",
    "这",
    "那",
    "哪",
    "这个",
    "那个",
    "哪个",
    "这些",
    "那些",
    "哪些",
    "这里",
    "那里",
    "哪里",
    "这儿",
    "那儿",
    "哪儿",
    "这样",
    "那样",
    "这么",
    "那么",
    "这种",
    "其中",
    "其他",
    "及其",
    // question words
    "谁",
    "何",
    "啥",
    "咋",
    "怎",
    "几",
    "什么",
    "怎么",
    "怎样",
    "为什么",
    // conjunctions
    "和",
    "与",
    "及",
    "且",
    "而",
    "或",
    "但",
    "则",
    "即",
    "又",
    "并",
    "既",
    "虽",
    "若",
    "如",
    "倘",
    "以及",
    "而且",
    "并且",
    "或者",
    "还是",
    "但是",
    "可是",
    "然而",
    "因为",
    "所以",
    "因此",
    "于是",
    "然后",
    "如果",
    "虽然",
    "即使",
    "尽管",
    "不但",
    "不仅",
    "而是",
    "只是",
    // prepositions and coverbs
    "在",
    "于",
    "从",
    "对",
    "把",
    "被",
    "让",
    "给",
    "跟",
    "向",
    "到",
    "至",
    "为",
    "以",
    "按",
    "据",
    "凭",
    "较",
    "连",
    "关于",
    "对于",
    "由于",
    "为了",
    // auxiliaries and common verbs used as function words
    "是",
    "有",
    "就",
    "乃",
    "该",
    "另",
    "就是",
    "是否",
    "是的",
    "已经",
    "可以",
    "还有",
    // Traditional Chinese forms retained from the existing curation.
    "嗎",
    "我們",
    "你們",
];

// Preserve reviewed content words, modifiers and negative markers.
// Keep single-kana grammar/stems; common noun spellings use kanji (絵, 木, 詩).
pub(super) const JAPANESE: &[&str] = &[
    "あそこ",
    "あっ",
    "あの",
    "あのかた",
    "あの人",
    "あります",
    "ある",
    "あれ",
    "い",
    "いう",
    "います",
    "いる",
    "う",
    "え",
    "お",
    "および",
    "おります",
    "か",
    "から",
    "が",
    "き",
    "ここ",
    "こちら",
    "こと",
    "この",
    "これ",
    "これら",
    "さ",
    "さらに",
    "し",
    "しかし",
    "する",
    "せ",
    "せる",
    "そこ",
    "そして",
    "その",
    "その他",
    "それ",
    "それぞれ",
    "それで",
    "た",
    "ただし",
    "たち",
    "ため",
    "たり",
    "だ",
    "だっ",
    "だれ",
    "つ",
    "て",
    "で",
    "です",
    "では",
    "でも",
    "と",
    "という",
    "といった",
    "として",
    "とともに",
    "とも",
    "と共に",
    "どこ",
    "どの",
    "なお",
    "ながら",
    "なっ",
    "など",
    "なに",
    "なら",
    "なり",
    "なる",
    "なん",
    "に",
    "において",
    "における",
    "について",
    "にて",
    "によって",
    "により",
    "による",
    "に対して",
    "に対する",
    "に関する",
    "の",
    "ので",
    "は",
    "ば",
    "へ",
    "ほか",
    "ほど",
    "ます",
    "また",
    "または",
    "まで",
    "も",
    "や",
    "よう",
    "より",
    "ら",
    "られ",
    "られる",
    "れ",
    "れる",
    "を",
    "何",
    "及び",
    "我々",
    "私",
    "私達",
    "貴方",
    "貴方方",
];

// These valid monolingual stop words have content meanings in other languages.
pub(super) const UNION_EXCLUSIONS: &[&str] = &["和", "私"];

#[cfg(test)]
mod tests {
    use crate::{
        IcuTokenizer, LowerCaser, RawTokenizer, Stemmer, StopWordFilter, TextAnalyzer, TokenStream,
    };

    #[test]
    fn preserves_cjk_search_terms() {
        let text = "取得 表示 使用 必要 全部 以上 以下 可能 一般 今後 最近 部分 以前 以後 相当 当然 普通 出来 本 朝 用 地 前面 十分 大量 左右 最後 多数 一致 自己 非常 所有 结果 和 私 ない できる 不是 没有 不能 哎 唉 哦 嗯 喂 哈 哇 哟 呃 呵 嘿 嘻 咦 呗 呜 哼 嗳 嗬 喏 哩 啐 呸 吱 咚 嘎 嘘 哗 咳 呕 着 得 由 可 将 所 等 别 うち とき ところ 特に な なかっ ん のみ もの 彼女 因 あり おり ものの 彼";
        for split in [false, true] {
            for stem in [false, true] {
                let base = if split {
                    IcuTokenizer::default().with_simple_split()
                } else {
                    IcuTokenizer::default()
                };
                let mut builder = TextAnalyzer::builder(base.clone())
                    .dynamic()
                    .filter_dynamic(LowerCaser);
                if stem {
                    builder = builder.filter_dynamic(Stemmer::default());
                }
                let mut unfiltered_builder = TextAnalyzer::builder(base)
                    .dynamic()
                    .filter_dynamic(LowerCaser);
                if stem {
                    unfiltered_builder = unfiltered_builder.filter_dynamic(Stemmer::default());
                }
                let mut unfiltered = unfiltered_builder.build();
                let mut filtered = builder.filter_dynamic(StopWordFilter::all()).build();
                let mut expected = Vec::new();
                unfiltered
                    .token_stream(text)
                    .process(&mut |token| expected.push(token.text.clone()));
                let mut actual = Vec::new();
                filtered
                    .token_stream(text)
                    .process(&mut |token| actual.push(token.text.clone()));
                assert!(!expected.is_empty());
                assert_eq!(actual, expected, "split={split}, stem={stem}");
            }
        }
    }

    #[test]
    fn retains_cjk_function_word_filtering() {
        for word in [
            "的", "了", "是", "我", "你", "他", "啊", "呀", "嘛", "啦", "吗", "吧", "呢", "の",
            "を", "は", "え", "き", "し", "嗎", "我們", "你們",
        ] {
            let mut analyzer = TextAnalyzer::builder(RawTokenizer::default())
                .filter(StopWordFilter::all())
                .build();
            assert!(
                !analyzer.token_stream(word).advance(),
                "function word {word} was not filtered"
            );
        }
    }
}
