// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_index::scalar::InvertedIndexParams;
use lance_index::scalar::inverted::query::collect_query_tokens;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn extract_kwarg<'py, T>(kwargs: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>>
where
    T: FromPyObjectOwned<'py>,
{
    kwargs
        .get_item(key)?
        .map(|value| value.extract::<T>().map_err(Into::into))
        .transpose()
}

pub(crate) struct FtsTokenizerOptions {
    analyzer: Option<String>,
    base_tokenizer: Option<String>,
    language: Option<String>,
    // The outer option tracks omission; the inner None explicitly disables the limit.
    max_token_length: Option<Option<usize>>,
    lower_case: Option<bool>,
    stem: Option<bool>,
    remove_stop_words: Option<bool>,
    // Preserve omitted versus explicit None to match create-index keyword semantics.
    custom_stop_words: Option<Option<Vec<String>>>,
    ascii_folding: Option<bool>,
    min_ngram_length: Option<u32>,
    max_ngram_length: Option<u32>,
    prefix_only: Option<bool>,
    split_identifiers: Option<bool>,
    split_on_numerics: Option<bool>,
    preserve_original: Option<bool>,
    index_operators: Option<bool>,
}

impl FtsTokenizerOptions {
    pub(crate) fn from_kwargs(kwargs: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            analyzer: extract_kwarg(kwargs, "analyzer")?,
            base_tokenizer: extract_kwarg(kwargs, "base_tokenizer")?,
            language: extract_kwarg(kwargs, "language")?,
            max_token_length: extract_kwarg(kwargs, "max_token_length")?,
            lower_case: extract_kwarg(kwargs, "lower_case")?,
            stem: extract_kwarg(kwargs, "stem")?,
            remove_stop_words: extract_kwarg(kwargs, "remove_stop_words")?,
            custom_stop_words: extract_kwarg(kwargs, "custom_stop_words")?,
            ascii_folding: extract_kwarg(kwargs, "ascii_folding")?,
            min_ngram_length: extract_kwarg(kwargs, "min_ngram_length")?,
            max_ngram_length: extract_kwarg(kwargs, "max_ngram_length")?,
            prefix_only: extract_kwarg(kwargs, "prefix_only")?,
            split_identifiers: extract_kwarg(kwargs, "split_identifiers")?,
            split_on_numerics: extract_kwarg(kwargs, "split_on_numerics")?,
            preserve_original: extract_kwarg(kwargs, "preserve_original")?,
            index_operators: extract_kwarg(kwargs, "index_operators")?,
        })
    }

    pub(crate) fn apply(self, mut params: InvertedIndexParams) -> PyResult<InvertedIndexParams> {
        match (self.analyzer.as_deref(), self.base_tokenizer.as_deref()) {
            (Some("text"), Some("code")) => {
                return Err(PyValueError::new_err(
                    "base_tokenizer='code' requires analyzer='code'",
                ));
            }
            (Some("code"), Some(base_tokenizer)) if base_tokenizer != "code" => {
                return Err(PyValueError::new_err(format!(
                    "analyzer='code' requires base_tokenizer='code', got '{base_tokenizer}'"
                )));
            }
            _ => {}
        }

        let uses_code_analyzer = match self.analyzer.as_deref() {
            Some("code") => true,
            Some("text") | None => self.base_tokenizer.as_deref() == Some("code"),
            Some(_) => true,
        };
        if !uses_code_analyzer
            && [
                self.split_identifiers,
                self.split_on_numerics,
                self.preserve_original,
                self.index_operators,
            ]
            .into_iter()
            .flatten()
            .any(|value| value)
        {
            return Err(PyValueError::new_err(
                "code analyzer flags require analyzer='code'",
            ));
        }

        if let Some(analyzer) = self.analyzer {
            params = params
                .analyzer(&analyzer)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
        }
        if let Some(base_tokenizer) = self.base_tokenizer {
            params = params.base_tokenizer(base_tokenizer);
        }
        if let Some(language) = self.language {
            params = params.language(&language).map_err(|err| {
                PyValueError::new_err(format!("can't set tokenizer language to {language}: {err}"))
            })?;
        }
        if let Some(max_token_length) = self.max_token_length {
            params = params.max_token_length(max_token_length);
        }
        if let Some(lower_case) = self.lower_case {
            params = params.lower_case(lower_case);
        }
        if let Some(stem) = self.stem {
            params = params.stem(stem);
        }
        if let Some(remove_stop_words) = self.remove_stop_words {
            params = params.remove_stop_words(remove_stop_words);
        }
        if let Some(custom_stop_words) = self.custom_stop_words {
            params = params.custom_stop_words(custom_stop_words);
        }
        if let Some(ascii_folding) = self.ascii_folding {
            params = params.ascii_folding(ascii_folding);
        }
        if let Some(min_ngram_length) = self.min_ngram_length {
            params = params.ngram_min_length(min_ngram_length);
        }
        if let Some(max_ngram_length) = self.max_ngram_length {
            params = params.ngram_max_length(max_ngram_length);
        }
        if let Some(prefix_only) = self.prefix_only {
            params = params.ngram_prefix_only(prefix_only);
        }
        if let Some(split_identifiers) = self.split_identifiers {
            params = params.split_identifiers(split_identifiers);
        }
        if let Some(split_on_numerics) = self.split_on_numerics {
            params = params.split_on_numerics(split_on_numerics);
        }
        if let Some(preserve_original) = self.preserve_original {
            params = params.preserve_original(preserve_original);
        }
        if let Some(index_operators) = self.index_operators {
            params = params.index_operators(index_operators);
        }
        Ok(params)
    }
}

/// A token produced by the full-text search query tokenizer.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct FtsToken {
    /// The token text after all configured filters have been applied.
    pub text: String,
    /// The position used by full-text query matching.
    pub position: u32,
}

#[pymethods]
impl FtsToken {
    fn __repr__(&self) -> String {
        format!("FtsToken(text={:?}, position={})", self.text, self.position)
    }
}

/// Tokenize a full-text search query without creating a dataset or index.
///
/// The tokenizer options are the same as the tokenizer-related options accepted by
/// ``LanceDataset.create_scalar_index(..., index_type="INVERTED")``. ``None`` uses
/// the selected analyzer profile's default, except ``max_token_length=None``, which
/// disables the length limit; omitting it keeps the default limit of 40. Returned
/// positions are normalized to the first retained token while preserving gaps left
/// by token filters.
///
/// Examples
/// --------
/// >>> import lance
/// >>> tokens = lance.tokenize("the Cats and Dogs")
/// >>> [(token.text, token.position) for token in tokens]
/// [('cat', 0), ('dog', 2)]
#[pyfunction]
#[pyo3(signature = (
    query,
    *,
    analyzer = None,
    base_tokenizer = None,
    language = None,
    max_token_length = Some(40),
    lower_case = None,
    stem = None,
    remove_stop_words = None,
    custom_stop_words = None,
    ascii_folding = None,
    min_ngram_length = None,
    max_ngram_length = None,
    prefix_only = None,
    split_identifiers = None,
    split_on_numerics = None,
    preserve_original = None,
    index_operators = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn tokenize(
    query: &str,
    analyzer: Option<String>,
    base_tokenizer: Option<String>,
    language: Option<String>,
    max_token_length: Option<usize>,
    lower_case: Option<bool>,
    stem: Option<bool>,
    remove_stop_words: Option<bool>,
    custom_stop_words: Option<Vec<String>>,
    ascii_folding: Option<bool>,
    min_ngram_length: Option<u32>,
    max_ngram_length: Option<u32>,
    prefix_only: Option<bool>,
    split_identifiers: Option<bool>,
    split_on_numerics: Option<bool>,
    preserve_original: Option<bool>,
    index_operators: Option<bool>,
) -> PyResult<Vec<FtsToken>> {
    let params = FtsTokenizerOptions {
        analyzer,
        base_tokenizer,
        language,
        max_token_length: Some(max_token_length),
        lower_case,
        stem,
        remove_stop_words,
        custom_stop_words: custom_stop_words.map(Some),
        ascii_folding,
        min_ngram_length,
        max_ngram_length,
        prefix_only,
        split_identifiers,
        split_on_numerics,
        preserve_original,
        index_operators,
    }
    .apply(InvertedIndexParams::default())?;

    let mut tokenizer = params
        .build()
        .map_err(|err| PyValueError::new_err(format!("Failed to build tokenizer: {err}")))?;
    let tokens = collect_query_tokens(query, &mut tokenizer);
    Ok((0..tokens.len())
        .map(|index| FtsToken {
            text: tokens.get_token(index).to_string(),
            position: tokens.position(index),
        })
        .collect())
}
