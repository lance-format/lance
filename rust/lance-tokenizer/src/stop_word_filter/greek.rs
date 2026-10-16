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

// Preserve grammatical filtering while retaining content, numbers and polarity.
pub(super) const GREEK: &[&str] = &[
    "αλλα",
    "αν",
    "αντι",
    "απο",
    "αυτα",
    "αυτεσ",
    "αυτη",
    "αυτο",
    "αυτοι",
    "αυτοσ",
    "αυτουσ",
    "αυτων",
    "αἱ",
    "αἳ",
    "αἵ",
    "αὐτόσ",
    "αὐτὸς",
    "αὖ",
    "γάρ",
    "γα",
    "γε",
    "για",
    "γοῦν",
    "γὰρ",
    "δέ",
    "δή",
    "δαί",
    "δαὶ",
    "δε",
    "διά",
    "διὰ",
    "δὲ",
    "δὴ",
    "εαν",
    "ειμαι",
    "ειμαστε",
    "ειναι",
    "εισαι",
    "ειστε",
    "εκεινα",
    "εκεινεσ",
    "εκεινη",
    "εκεινο",
    "εκεινοι",
    "εκεινοσ",
    "εκεινουσ",
    "εκεινων",
    "ενω",
    "επ",
    "επι",
    "εἰ",
    "εἰμί",
    "εἰμὶ",
    "εἰς",
    "εἰσ",
    "εἴ",
    "εἴμι",
    "εἴτε",
    "η",
    "θα",
    "ισωσ",
    "κ",
    "καί",
    "καίτοι",
    "καθ",
    "και",
    "κατ",
    "κατά",
    "κατα",
    "κατὰ",
    "καὶ",
    "κι",
    "κἀν",
    "κἂν",
    "μέν",
    "μα",
    "με",
    "μεθ",
    "μετ",
    "μετά",
    "μετα",
    "μετὰ",
    "μἐν",
    "μὲν",
    "μὴν",
    "να",
    "ο",
    "οι",
    "ομωσ",
    "οπωσ",
    "οσο",
    "οτι",
    "οἱ",
    "οἳ",
    "οἷς",
    "οὓς",
    "οὕτω",
    "οὕτως",
    "οὕτωσ",
    "οὖν",
    "οὗ",
    "οὗτος",
    "οὗτοσ",
    "παρ",
    "παρά",
    "παρα",
    "παρὰ",
    "περί",
    "περὶ",
    "ποια",
    "ποιεσ",
    "ποιο",
    "ποιοι",
    "ποιοσ",
    "ποιουσ",
    "ποιων",
    "ποτε",
    "που",
    "ποῦ",
    "προ",
    "προσ",
    "πρόσ",
    "πρὸ",
    "πρὸς",
    "πως",
    "πωσ",
    "σε",
    "στη",
    "στην",
    "στο",
    "στον",
    "σόσ",
    "σύ",
    "σύν",
    "σὸς",
    "σὺ",
    "σὺν",
    "τά",
    "τήν",
    "τί",
    "τίς",
    "τίσ",
    "τα",
    "ταῖς",
    "τε",
    "την",
    "τησ",
    "τι",
    "τινα",
    "τις",
    "τισ",
    "το",
    "τοί",
    "τοι",
    "τοιοῦτος",
    "τοιοῦτοσ",
    "τον",
    "τοτε",
    "του",
    "τούσ",
    "τοὺς",
    "τοῖς",
    "τοῦ",
    "των",
    "τό",
    "τόν",
    "τότε",
    "τὰ",
    "τὰς",
    "τὴν",
    "τὸ",
    "τὸν",
    "τῆς",
    "τῆσ",
    "τῇ",
    "τῶν",
    "τῷ",
    "ωσ",
    "ἀλλά",
    "ἀλλὰ",
    "ἀπ",
    "ἀπό",
    "ἀπὸ",
    "ἀφ",
    "ἂν",
    "ἃ",
    "ἄλλος",
    "ἄλλοσ",
    "ἄν",
    "ἄρα",
    "ἅμα",
    "ἐάν",
    "ἐγώ",
    "ἐγὼ",
    "ἐκ",
    "ἐμόσ",
    "ἐμὸς",
    "ἐν",
    "ἐξ",
    "ἐπί",
    "ἐπεὶ",
    "ἐπὶ",
    "ἐστι",
    "ἐφ",
    "ἐὰν",
    "ἑαυτοῦ",
    "ἔτι",
    "ἡ",
    "ἢ",
    "ἣ",
    "ἤ",
    "ἥ",
    "ἧς",
    "ἵνα",
    "ὁ",
    "ὃ",
    "ὃν",
    "ὃς",
    "ὅ",
    "ὅδε",
    "ὅθεν",
    "ὅπερ",
    "ὅς",
    "ὅσ",
    "ὅστις",
    "ὅστισ",
    "ὅτε",
    "ὅτι",
    "ὑμόσ",
    "ὑπ",
    "ὑπέρ",
    "ὑπό",
    "ὑπὲρ",
    "ὑπὸ",
    "ὡς",
    "ὡσ",
    "ὥς",
    "ὥστε",
    "ὦ",
    "ᾧ",
    "από",
    "είμαι",
    "είναι",
];

// These valid monolingual stop words have content meanings in other languages.
// Language-specific additions remain opt-in instead of widening the ICU union.
pub(super) const UNION_EXCLUSIONS: &[&str] = &["από", "είμαι", "είναι"];

#[cfg(test)]
mod tests {
    use crate::{
        IcuTokenizer, Language, LowerCaser, RawTokenizer, Stemmer, StopWordFilter, TextAnalyzer,
        TokenStream,
    };

    #[test]
    fn preserves_greek_search_terms() {
        let text = "δεν μη μην οὐκ οὔτε δαίσ δαὶς";
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
    fn retains_greek_function_word_filtering() {
        for word in ["και", "το"] {
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
    fn scopes_greek_language_specific_changes() {
        for (language, word, removed) in [
            (Language::Greek, "από", true),
            (Language::Greek, "είμαι", true),
            (Language::Greek, "είναι", true),
            (Language::Greek, "δεν", false),
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
