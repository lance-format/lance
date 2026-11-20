// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Marks Hugging Face rate-limit errors that OpenDAL already handled.
//!
//! OpenDAL's retry layer owns the `client_max_retries` budget for a Hugging Face
//! store. Once it returns a 429 error, Lance's reader and list retry loops must
//! not repeat the request: the server-directed delay has already elapsed and
//! another request would arrive before the next allowed time.

use std::error::Error as StdError;
use std::fmt::{Display, Formatter};

/// Wraps the source of a Hugging Face rate-limit error so the retry loops can
/// recognize it without parsing the error message.
#[derive(Debug)]
pub struct HuggingFaceRateLimited {
    source: Box<dyn StdError + Send + Sync>,
}

impl Display for HuggingFaceRateLimited {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.source, formatter)
    }
}

impl StdError for HuggingFaceRateLimited {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.source.as_ref())
    }
}

/// Add the marker to an `object_store` error without changing its display.
pub fn mark_huggingface_rate_limited(error: object_store::Error) -> object_store::Error {
    match error {
        object_store::Error::Generic { store, source }
            if !source.is::<HuggingFaceRateLimited>() =>
        {
            object_store::Error::Generic {
                store,
                source: Box::new(HuggingFaceRateLimited { source }),
            }
        }
        other => other,
    }
}

/// Whether `error` came from a Hugging Face 429 that OpenDAL already handled.
pub fn is_huggingface_rate_limited(error: &object_store::Error) -> bool {
    matches!(
        error,
        object_store::Error::Generic { source, .. }
            if source.is::<HuggingFaceRateLimited>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_is_typed_and_preserves_the_error_display() {
        let original = object_store::Error::Generic {
            store: "RateLimited",
            source: "RateLimited (temporary) at read".into(),
        };
        let display = original.to_string();

        let marked = mark_huggingface_rate_limited(original);

        assert!(is_huggingface_rate_limited(&marked));
        assert_eq!(marked.to_string(), display);
    }
}
