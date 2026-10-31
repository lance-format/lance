// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(test)]
mod tests {
    use crate::{
        IcuTokenizer, Language, LowerCaser, RawTokenizer, Stemmer, StopWordFilter, TextAnalyzer,
        TokenStream,
    };

    #[test]
    fn preserves_russian_search_terms() {
        let text = "не нет ни ничего никогда нельзя человек жизнь сказал один два три";
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
    fn retains_russian_function_word_filtering() {
        for word in ["и", "в"] {
            let mut analyzer = TextAnalyzer::builder(RawTokenizer::default())
                .filter(StopWordFilter::all())
                .build();
            assert!(
                !analyzer.token_stream(word).advance(),
                "function word {word} was not filtered"
            );
        }
    }

    #[test]
    fn scopes_russian_language_specific_changes() {
        for (language, word, removed) in [
            (Language::Russian, "её", true),
            (Language::Russian, "неё", true),
            (Language::Russian, "не", false),
        ] {
            let mut analyzer = TextAnalyzer::builder(RawTokenizer::default())
                .filter(StopWordFilter::new(language).unwrap())
                .build();
            assert_eq!(
                !analyzer.token_stream(word).advance(),
                removed,
                "{language:?}: {word}"
            );
        }
    }
}
