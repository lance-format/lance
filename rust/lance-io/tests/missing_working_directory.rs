// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Opening a relative location once the working directory is gone.
//!
//! This test removes the working directory and does not put it back, which is
//! process-wide state that any sibling test resolving a relative path would
//! read, so keep this file to the one test. Windows refuses to remove the
//! working directory of a running process, so there is nothing to test there.
#![cfg(unix)]

use lance_core::utils::tempfile::TempStdDir;
use lance_io::object_store::ObjectStore;

#[tokio::test]
async fn relative_uri_fails_cleanly_when_the_working_directory_is_gone() {
    let tmp = TempStdDir::default();
    let workdir = tmp.join("workdir");
    std::fs::create_dir(&workdir).unwrap();
    std::env::set_current_dir(&workdir).unwrap();
    std::fs::remove_dir(&workdir).unwrap();

    // Resolving the relative location has to read the working directory.
    let err = ObjectStore::from_uri("./dataset.lance")
        .await
        .expect_err("a relative location cannot resolve without a working directory");

    assert!(
        matches!(err, lance_core::Error::InvalidInput { .. }),
        "got: {err:?}"
    );
    assert!(err.to_string().contains("dataset.lance"), "got: {err}");
}
