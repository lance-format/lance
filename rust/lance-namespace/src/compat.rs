// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Deserialization shims that keep older SDK builds working against current native code.

use crate::error::NamespaceError;
use crate::models::MergeInsertIntoTableRequest;

/// Deserialize a [`MergeInsertIntoTableRequest`] whose `on` field may be a bare string.
///
/// Java and Python SDK requests reach the Rust implementations as JSON across JNI and
/// pyo3, so a jar or wheel built against lance-namespace 0.11 or earlier sends
/// `"on": "id"` where the current model expects `"on": ["id"]`. A scalar is promoted to
/// a one-element list so those callers keep working.
///
/// This is inbound only. A Java namespace implementation called *from* Rust still needs
/// a jar matching the current model.
pub fn merge_insert_request_from_json(
    mut value: serde_json::Value,
) -> crate::Result<MergeInsertIntoTableRequest> {
    if let Some(on) = value.get_mut("on")
        && let Some(column) = on.as_str()
    {
        *on = serde_json::json!([column]);
    }

    serde_json::from_value(value).map_err(|e| {
        NamespaceError::InvalidInput {
            message: format!("Failed to parse merge_insert_into_table request: {}", e),
        }
        .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> MergeInsertIntoTableRequest {
        merge_insert_request_from_json(serde_json::from_str(json).unwrap()).unwrap()
    }

    #[test]
    fn scalar_on_is_promoted_to_a_single_column_key() {
        let request = parse(r#"{"id": ["t"], "on": "id"}"#);
        assert_eq!(request.on, Some(vec!["id".to_string()]));
        assert_eq!(request.id, Some(vec!["t".to_string()]));
    }

    #[test]
    fn list_on_is_preserved() {
        let request = parse(r#"{"id": ["t"], "on": ["a", "b"], "use_index": true}"#);
        assert_eq!(request.on, Some(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(request.use_index, Some(true));
    }

    #[test]
    fn absent_and_null_on_stay_absent() {
        assert_eq!(parse(r#"{"id": ["t"]}"#).on, None);
        assert_eq!(parse(r#"{"id": ["t"], "on": null}"#).on, None);
    }

    #[test]
    fn a_non_string_non_list_on_is_still_rejected() {
        let error =
            merge_insert_request_from_json(serde_json::json!({"id": ["t"], "on": 7})).unwrap_err();
        assert!(
            error.to_string().contains("invalid type"),
            "unexpected error: {error}"
        );
    }
}
