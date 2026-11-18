// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shard writer close-stage composition.

use super::super::*;

#[test]
fn test_merge_close_stage_preserves_first_error() {
    let result = ShardWriter::merge_close_stage(
        Err(Error::io("primary close error")),
        "secondary close stage",
        Err(Error::io("secondary close error")),
    );

    let error = result.expect_err("close must preserve the first error");
    assert!(matches!(&error, Error::IO { .. }));
    assert!(
        error.to_string().contains("primary close error"),
        "unexpected error: {error}"
    );
    assert!(
        !error.to_string().contains("secondary close error"),
        "secondary error replaced the primary error: {error}"
    );
}
