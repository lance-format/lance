// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

fn build_ivf_rq_num_bits(num_bits: Option<i32>) -> Result<u8> {
    let mut request = CreateTableIndexRequest::new("vector".to_string(), "IVF_RQ".to_string());
    request.num_bits = num_bits;

    let DirectoryIndexParams::Vector {
        index_type: IndexType::IvfRq,
        params,
    } = DirectoryNamespace::build_index_params(&request)?
    else {
        panic!("expected IVF_RQ vector index params");
    };
    match params.stages.as_slice() {
        [StageParams::Ivf(_), StageParams::RQ(rq)] => Ok(rq.num_bits),
        stages => panic!("expected IVF and RQ stages, got {stages:?}"),
    }
}

#[rstest]
#[case::omitted(None, 5)]
#[case::explicit_one(Some(1), 1)]
#[case::explicit_max(Some(9), 9)]
fn test_build_index_params_ivf_rq_num_bits(#[case] requested: Option<i32>, #[case] expected: u8) {
    assert_eq!(build_ivf_rq_num_bits(requested).unwrap(), expected);
}

#[rstest]
#[case::negative(-1)]
#[case::zero(0)]
#[case::above_max(10)]
#[case::conversion_overflow(i32::MAX)]
fn test_build_index_params_rejects_invalid_ivf_rq_num_bits(#[case] requested: i32) {
    let error =
        build_ivf_rq_num_bits(Some(requested)).expect_err("invalid IVF_RQ num_bits should fail");
    let message = error.to_string();

    assert_eq!(mutation_error_code(error), ErrorCode::InvalidInput);
    assert!(
        message.contains(&format!(
            "IVF_RQ num_bits must be in 1..=9, got {requested}"
        )),
        "unexpected error message: {message}"
    );
}

#[test]
fn test_build_engine_fts_query_match() {
    let mut ns_match = lance_namespace::models::MatchQuery::new("hello world".to_string());
    ns_match.column = Some("body".to_string());
    ns_match.operator = Some("AND".to_string());
    ns_match.fuzziness = Some(1);
    ns_match.max_expansions = Some(30);
    ns_match.boost = Some(2.0);
    ns_match.prefix_length = Some(2);

    let mut ns_query = lance_namespace::models::FtsQuery::new();
    ns_query.r#match = Some(Box::new(ns_match));

    match build_engine_fts_query(&ns_query).unwrap() {
        FtsQuery::Match(m) => {
            assert_eq!(m.terms, "hello world");
            assert_eq!(m.column, Some("body".to_string()));
            assert_eq!(m.operator, Operator::And);
            assert_eq!(m.fuzziness, Some(1));
            assert_eq!(m.max_expansions, 30);
            assert_eq!(m.boost, 2.0);
            assert_eq!(m.prefix_length, 2);
        }
        other => panic!("expected Match, got {:?}", other),
    }
}

/// Wraps a namespace `MatchQuery` (with a column) as an `FtsQuery` for use as a clause in
/// compound queries (boolean / boost).
fn ns_match_query(terms: &str, column: &str) -> lance_namespace::models::FtsQuery {
    let mut m = lance_namespace::models::MatchQuery::new(terms.to_string());
    m.column = Some(column.to_string());
    let mut q = lance_namespace::models::FtsQuery::new();
    q.r#match = Some(Box::new(m));
    q
}

#[test]
fn test_build_engine_fts_query_phrase() {
    let mut ns_phrase = lance_namespace::models::PhraseQuery::new("hello world".to_string());
    ns_phrase.column = Some("body".to_string());
    ns_phrase.slop = Some(2);

    let mut ns_query = lance_namespace::models::FtsQuery::new();
    ns_query.phrase = Some(Box::new(ns_phrase));

    match build_engine_fts_query(&ns_query).unwrap() {
        FtsQuery::Phrase(p) => {
            assert_eq!(p.terms, "hello world");
            assert_eq!(p.column, Some("body".to_string()));
            assert_eq!(p.slop, 2);
        }
        other => panic!("expected Phrase, got {:?}", other),
    }
}

#[test]
fn test_build_engine_fts_query_multi_match() {
    let mut m1 = lance_namespace::models::MatchQuery::new("hello".to_string());
    m1.column = Some("title".to_string());
    let mut m2 = lance_namespace::models::MatchQuery::new("hello".to_string());
    m2.column = Some("body".to_string());
    m2.boost = Some(2.0);

    let ns_multi = lance_namespace::models::MultiMatchQuery::new(vec![m1, m2]);
    let mut ns_query = lance_namespace::models::FtsQuery::new();
    ns_query.multi_match = Some(Box::new(ns_multi));

    match build_engine_fts_query(&ns_query).unwrap() {
        FtsQuery::MultiMatch(mm) => {
            assert_eq!(mm.match_queries.len(), 2);
            assert_eq!(mm.match_queries[0].terms, "hello");
            assert_eq!(mm.match_queries[0].column, Some("title".to_string()));
            assert_eq!(mm.match_queries[1].column, Some("body".to_string()));
            assert_eq!(mm.match_queries[1].boost, 2.0);
        }
        other => panic!("expected MultiMatch, got {:?}", other),
    }
}

#[test]
fn test_build_engine_fts_query_boolean() {
    // BooleanQuery::new(must, must_not, should)
    let ns_boolean = lance_namespace::models::BooleanQuery::new(
        vec![ns_match_query("must-term", "body")],
        vec![ns_match_query("must-not-term", "body")],
        vec![ns_match_query("should-term", "body")],
    );
    let mut ns_query = lance_namespace::models::FtsQuery::new();
    ns_query.boolean = Some(Box::new(ns_boolean));

    match build_engine_fts_query(&ns_query).unwrap() {
        FtsQuery::Boolean(b) => {
            assert!(matches!(&b.must[..], [FtsQuery::Match(m)] if m.terms == "must-term"));
            assert!(matches!(&b.must_not[..], [FtsQuery::Match(m)] if m.terms == "must-not-term"));
            assert!(matches!(&b.should[..], [FtsQuery::Match(m)] if m.terms == "should-term"));
        }
        other => panic!("expected Boolean, got {:?}", other),
    }
}

#[test]
fn test_build_engine_fts_query_boost() {
    let mut ns_boost = lance_namespace::models::BoostQuery::new(
        ns_match_query("positive-term", "body"),
        ns_match_query("negative-term", "body"),
    );
    ns_boost.negative_boost = Some(0.25);

    let mut ns_query = lance_namespace::models::FtsQuery::new();
    ns_query.boost = Some(Box::new(ns_boost));

    match build_engine_fts_query(&ns_query).unwrap() {
        FtsQuery::Boost(b) => {
            assert!(
                matches!(b.positive.as_ref(), FtsQuery::Match(m) if m.terms == "positive-term")
            );
            assert!(
                matches!(b.negative.as_ref(), FtsQuery::Match(m) if m.terms == "negative-term")
            );
            assert_eq!(b.negative_boost, 0.25);
        }
        other => panic!("expected Boost, got {:?}", other),
    }
}

#[test]
fn test_build_engine_fts_query_requires_a_variant() {
    // An FtsQuery with no variant set is rejected rather than silently ignored.
    let empty = lance_namespace::models::FtsQuery::new();
    assert!(build_engine_fts_query(&empty).is_err());
}

/// `map_mutation_error` must classify commit-conflict variants the same way as
/// `convert_lance_commit_error` in `manifest.rs`: `CommitConflict` is a retries-exhausted
/// version collision that is safe to retry (`Throttling`), while the semantic-conflict variants
/// map to `ConcurrentModification`.
#[test]
fn test_map_mutation_error_commit_conflict_alignment() {
    let boxed = || -> Box<dyn std::error::Error + Send + Sync + 'static> {
        Box::<dyn std::error::Error + Send + Sync>::from("inner conflict")
    };

    let throttling_cases = vec![lance_core::Error::commit_conflict_source(1, boxed())];
    for err in throttling_cases {
        let code = mutation_error_code(DirectoryNamespace::map_mutation_error(
            err,
            "update",
            "memory://t",
        ));
        assert_eq!(code, ErrorCode::Throttling);
    }

    let concurrent_cases = vec![
        lance_core::Error::too_much_write_contention("contention"),
        lance_core::Error::retryable_commit_conflict_source(1, boxed()),
        lance_core::Error::incompatible_transaction_source(boxed()),
        lance_core::Error::version_conflict("conflict", 0, 3),
    ];
    for err in concurrent_cases {
        let code = mutation_error_code(DirectoryNamespace::map_mutation_error(
            err,
            "update",
            "memory://t",
        ));
        assert_eq!(code, ErrorCode::ConcurrentModification);
    }
}

#[test]
fn test_classify_storage_error_maps_variants_and_preserves_evidence() {
    let throttle: Error = ObjectStoreError::Generic {
        store: "test",
        source: "list request failed, after 3 retries, max_retries: 3 - 503 ServerBusy".into(),
    }
    .into();
    assert!(matches!(throttle, Error::IO { .. }));
    let classified = DirectoryNamespace::classify_storage_error(throttle);
    let msg = classified.to_string();
    assert_eq!(mutation_error_code(classified), ErrorCode::Throttling);
    assert!(
        msg.to_ascii_lowercase().contains("serverbusy"),
        "throttle evidence lost: {msg}"
    );

    let service: Error = ObjectStoreError::Generic {
        store: "test",
        source: "504 Gateway Timeout".into(),
    }
    .into();
    assert_eq!(
        mutation_error_code(DirectoryNamespace::classify_storage_error(service)),
        ErrorCode::ServiceUnavailable
    );

    let internal: Error = ObjectStoreError::Generic {
        store: "test",
        source: "disk caught fire".into(),
    }
    .into();
    assert_eq!(
        mutation_error_code(DirectoryNamespace::classify_storage_error(internal)),
        ErrorCode::Internal
    );

    // A pre-existing namespace error keeps its own code rather than being reclassified.
    let preexisting: Error = NamespaceError::TableAlreadyExists {
        message: "t".to_string(),
    }
    .into();
    assert_eq!(
        mutation_error_code(DirectoryNamespace::classify_storage_error(preexisting)),
        ErrorCode::TableAlreadyExists
    );
}

#[test]
fn test_is_manifest_table_absent_error() {
    let table_not_found: Error = NamespaceError::TableNotFound {
        message: "t".to_string(),
    }
    .into();
    assert!(DirectoryNamespace::is_manifest_table_absent_error(
        &table_not_found
    ));
    let raw_not_found: Error = ObjectStoreError::NotFound {
        path: "t".to_string(),
        source: "x".into(),
    }
    .into();
    assert!(DirectoryNamespace::is_manifest_table_absent_error(
        &raw_not_found
    ));

    let throttle: Error = ObjectStoreError::Generic {
        store: "test",
        source: "after 3 retries, max_retries: 3 ServerBusy".into(),
    }
    .into();
    assert!(!DirectoryNamespace::is_manifest_table_absent_error(
        &throttle
    ));
    let internal: Error = NamespaceError::Internal {
        message: "boom".to_string(),
    }
    .into();
    assert!(!DirectoryNamespace::is_manifest_table_absent_error(
        &internal
    ));
}

#[test]
fn test_map_open_error() {
    let not_found = || NamespaceError::TableNotFound {
        message: "table at 'x' not found: ...".to_string(),
    };

    let throttle: Error = ObjectStoreError::Generic {
        store: "test",
        source: "after 3 retries, max_retries: 3 - 503 ServerBusy".into(),
    }
    .into();
    assert_eq!(
        mutation_error_code(DirectoryNamespace::map_open_error(throttle, not_found())),
        ErrorCode::Throttling
    );

    let generic_io: Error = ObjectStoreError::Generic {
        store: "test",
        source: "connection reset".into(),
    }
    .into();
    assert_eq!(
        mutation_error_code(DirectoryNamespace::map_open_error(generic_io, not_found())),
        ErrorCode::Internal
    );

    let io_not_found: Error = ObjectStoreError::NotFound {
        path: "x".to_string(),
        source: "missing".into(),
    }
    .into();
    assert_eq!(
        mutation_error_code(DirectoryNamespace::map_open_error(
            io_not_found,
            not_found()
        )),
        ErrorCode::TableNotFound
    );

    let dataset_not_found = Error::dataset_not_found("x".to_string(), "missing".into());
    assert_eq!(
        mutation_error_code(DirectoryNamespace::map_open_error(
            dataset_not_found,
            not_found()
        )),
        ErrorCode::TableNotFound
    );

    // RefNotFound is not an IO error, so it is not reclassified as a storage error.
    let ref_not_found = Error::RefNotFound {
        message: "branch 'b' does not exist".to_string(),
    };
    assert_eq!(
        mutation_error_code(DirectoryNamespace::map_open_error(
            ref_not_found,
            not_found()
        )),
        ErrorCode::TableNotFound
    );

    // The caller's not-found variant is honored, but a throttle still propagates.
    let version_miss = Error::RefNotFound {
        message: "version 5 does not exist".to_string(),
    };
    assert_eq!(
        mutation_error_code(DirectoryNamespace::map_open_error(
            version_miss,
            NamespaceError::TableVersionNotFound {
                message: "version 5 not found".to_string(),
            },
        )),
        ErrorCode::TableVersionNotFound
    );
    let version_throttle: Error = ObjectStoreError::Generic {
        store: "test",
        source: "after 3 retries, max_retries: 3 - 503 ServerBusy".into(),
    }
    .into();
    assert_eq!(
        mutation_error_code(DirectoryNamespace::map_open_error(
            version_throttle,
            NamespaceError::TableVersionNotFound {
                message: "version 5 not found".to_string(),
            },
        )),
        ErrorCode::Throttling
    );
}

/// The shared parser must decode both naming schemes; this is the cheap
/// V1 no-regression guard (creating a real V1 table is not exposed here).
#[test]
fn test_manifest_version_from_filename() {
    // V1: the plain version number.
    assert_eq!(
        DirectoryNamespace::manifest_version_from_filename("5.manifest"),
        Some(5)
    );
    assert_eq!(
        DirectoryNamespace::manifest_version_from_filename("0.manifest"),
        Some(0)
    );
    // V2: version stored as u64::MAX - version, zero-padded to 20 digits.
    let v2_five = format!("{:020}.manifest", u64::MAX - 5);
    assert_eq!(
        DirectoryNamespace::manifest_version_from_filename(&v2_five),
        Some(5)
    );
    let v2_zero = format!("{:020}.manifest", u64::MAX);
    assert_eq!(
        DirectoryNamespace::manifest_version_from_filename(&v2_zero),
        Some(0)
    );
    // Non-manifest and detached (`d`-prefixed) entries are ignored.
    assert_eq!(
        DirectoryNamespace::manifest_version_from_filename("data.lance"),
        None
    );
    assert_eq!(
        DirectoryNamespace::manifest_version_from_filename("d5.manifest"),
        None
    );
}
