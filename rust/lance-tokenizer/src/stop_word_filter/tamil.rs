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
pub(super) const TAMIL: &[&str] = &[
    "அங்கு",
    "அங்கே",
    "அடுத்த",
    "அதனால்",
    "அதன்",
    "அதற்கு",
    "அதிக",
    "அதில்",
    "அது",
    "அதே",
    "அதை",
    "அந்த",
    "அந்தக்",
    "அந்தப்",
    "அல்லது",
    "அவன்",
    "அவரது",
    "அவர்",
    "அவர்கள்",
    "அவள்",
    "அவை",
    "ஆகிய",
    "ஆகியோர்",
    "ஆகும்",
    "இங்கு",
    "இங்கே",
    "இதனால்",
    "இதனை",
    "இதன்",
    "இதற்கு",
    "இதில்",
    "இது",
    "இதை",
    "இந்த",
    "இந்தக்",
    "இந்தத்",
    "இந்தப்",
    "இன்னும்",
    "இப்போது",
    "இருக்கும்",
    "இருந்த",
    "இருந்தது",
    "இருந்து",
    "இவர்",
    "இவை",
    "உன்",
    "உள்ள",
    "உள்ளது",
    "உள்ளன",
    "எந்த",
    "என",
    "எனக்",
    "எனக்கு",
    "எனப்படும்",
    "எனவும்",
    "எனவே",
    "எனினும்",
    "எனும்",
    "என்",
    "என்ன",
    "என்னும்",
    "என்பது",
    "என்பதை",
    "என்ற",
    "என்று",
    "என்றும்",
    "எல்லாம்",
    "ஏன்",
    "கொண்ட",
    "கொண்டு",
    "கொள்ள",
    "சற்று",
    "சிறு",
    "சில",
    "சேர்ந்த",
    "தனது",
    "தன்",
    "தவிர",
    "தான்",
    "நான்",
    "நாம்",
    "நீ",
    "பற்றி",
    "பற்றிய",
    "பல",
    "பலரும்",
    "பல்வேறு",
    "பின்",
    "பின்னர்",
    "பிற",
    "பிறகு",
    "பெரும்",
    "போது",
    "போன்ற",
    "போல",
    "போல்",
    "மட்டுமே",
    "மட்டும்",
    "மற்ற",
    "மற்றும்",
    "மிக",
    "மிகவும்",
    "மீது",
    "மேலும்",
    "மேல்",
    "யார்",
    "வந்த",
    "வந்து",
    "வரும்",
    "வரை",
    "வரையில்",
    "விட",
    "விட்டு",
    "வேண்டும்",
    "வேறு",
];

#[cfg(test)]
mod tests {
    use crate::{
        IcuTokenizer, Language, LowerCaser, RawTokenizer, Stemmer, StopWordFilter, TextAnalyzer,
        TokenStream,
    };

    #[test]
    fn preserves_tamil_search_terms() {
        let text = "அன்று இடத்தில் இடம் இரு முறை முதல் பேர் ஒரு ஓர் ஒரே";
        let split = false;
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

    #[test]
    fn retains_tamil_function_word_filtering() {
        for word in ["அவர்", "அவன்"] {
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
    fn scopes_tamil_language_specific_changes() {
        for (language, word, removed) in [
            (Language::Tamil, "இடம்", false),
            (Language::Tamil, "முறை", false),
            (Language::Tamil, "அவர்", true),
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
