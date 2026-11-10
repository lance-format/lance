// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Deriving the retention lease list a commit writes from its predecessor.
//!
//! See the [Retention Leases specification](https://lance.org/format/table/retention_leases/#lifecycle).

use chrono::{DateTime, Utc};
use lance_core::{Error, Result};
use uuid::Uuid;

use crate::format::{RetentionLease, validate_retention_lease_ids};
use crate::transaction::{Operation, Transaction};

/// The outcome of [`derive_retention_leases`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseChange {
    /// The commit writes `leases` as the new list, which may equal the
    /// predecessor's. Removing expired leases alone is not a reason to write
    /// a version; that is the caller's decision.
    Updated {
        leases: Vec<RetentionLease>,
        /// The lease an `AcquireLease` added or a `RenewLease` extended.
        lease: Option<RetentionLease>,
    },
    /// A lease operation that requests nothing new, so no version is written:
    /// re-acquiring a listed ID, renewing to a time not later than the current
    /// expiry, or releasing an absent ID.
    NoOp {
        /// The listed record for the target ID; `None` for an absent release.
        stored: Option<RetentionLease>,
    },
}

/// Derive the lease list a commit attempt writes.
///
/// `previous` must be the list of the manifest this attempt builds on, never
/// a cached one, and `now` the time read once for this attempt. Expired leases
/// are removed, then the transaction's own change is applied:
///
/// * `AcquireLease` adds a lease on `new_version`, or returns the listed one.
/// * `RenewLease` extends a listed, unexpired lease.
/// * `ReleaseLease` removes a lease if listed.
/// * `Restore`, `Overwrite`, and `Clone` start from an empty list.
/// * Any other operation releases the lease whose ID equals the transaction
///   UUID, if listed, publishing the holder's result.
///
/// Detached commits neither carry nor change leases; callers must not derive
/// a list for them.
pub fn derive_retention_leases(
    previous: &[RetentionLease],
    transaction: &Transaction,
    new_version: u64,
    now: DateTime<Utc>,
) -> Result<LeaseChange> {
    validate_retention_lease_ids(previous)?;
    let mut leases = previous
        .iter()
        .filter(|lease| !lease.is_expired(now))
        .cloned()
        .collect::<Vec<_>>();
    let position =
        |leases: &[RetentionLease], id: &Uuid| leases.iter().position(|lease| lease.id() == *id);

    match &transaction.operation {
        Operation::AcquireLease {
            lease_id,
            created_at,
            expires_at,
            label,
        } => {
            if Uuid::parse_str(&transaction.uuid).ok() == Some(*lease_id) {
                return Err(Error::invalid_input(format!(
                    "AcquireLease lease_id {lease_id} must differ from the acquiring \
                     transaction's own UUID, or its commit would release the lease"
                )));
            }
            if expires_at <= created_at {
                return Err(Error::invalid_input(format!(
                    "AcquireLease {lease_id} expires_at {expires_at} must be later than \
                     created_at {created_at}"
                )));
            }
            if let Some(index) = position(&leases, lease_id) {
                return Ok(LeaseChange::NoOp {
                    stored: Some(leases[index].clone()),
                });
            }
            if *expires_at <= now {
                return Err(Error::invalid_input(format!(
                    "AcquireLease {lease_id} expires_at {expires_at} is not in the future \
                     (now {now})"
                )));
            }
            let lease = RetentionLease::try_new(
                *lease_id,
                new_version,
                *created_at,
                *expires_at,
                label.clone(),
            )?;
            leases.push(lease.clone());
            Ok(LeaseChange::Updated {
                leases,
                lease: Some(lease),
            })
        }
        Operation::RenewLease {
            lease_id,
            expires_at,
        } => {
            let Some(index) = position(&leases, lease_id) else {
                let reason = match previous.iter().find(|lease| lease.id() == *lease_id) {
                    Some(expired) => format!("it expired at {} (now {now})", expired.expires_at()),
                    None => "it is not listed".to_string(),
                };
                return Err(Error::invalid_input(format!(
                    "Cannot renew retention lease {lease_id}: {reason}"
                )));
            };
            if *expires_at <= leases[index].expires_at() {
                return Ok(LeaseChange::NoOp {
                    stored: Some(leases[index].clone()),
                });
            }
            leases[index] = leases[index].with_expires_at(*expires_at);
            let lease = Some(leases[index].clone());
            Ok(LeaseChange::Updated { leases, lease })
        }
        Operation::ReleaseLease { lease_id } => match position(&leases, lease_id) {
            Some(index) => {
                leases.remove(index);
                Ok(LeaseChange::Updated {
                    leases,
                    lease: None,
                })
            }
            None => Ok(LeaseChange::NoOp { stored: None }),
        },
        Operation::Restore { .. } | Operation::Overwrite { .. } | Operation::Clone { .. } => {
            Ok(LeaseChange::Updated {
                leases: Vec::new(),
                lease: None,
            })
        }
        Operation::Unknown { .. } => Err(Error::not_supported(format!(
            "Transaction {} has an operation this version of Lance does not recognize, \
             so it cannot derive its retention leases",
            transaction.uuid
        ))),
        Operation::Append { .. }
        | Operation::Delete { .. }
        | Operation::CreateIndex { .. }
        | Operation::Rewrite { .. }
        | Operation::Merge { .. }
        | Operation::ReserveFragments { .. }
        | Operation::Update { .. }
        | Operation::Project { .. }
        | Operation::UpdateConfig { .. }
        | Operation::DataReplacement { .. }
        | Operation::DataOverlay { .. }
        | Operation::UpdateMemWalState { .. }
        | Operation::UpdateBases { .. } => {
            // Not every transaction UUID is a UUID; such a transaction cannot
            // publish a lease.
            if let Ok(publication) = Uuid::parse_str(&transaction.uuid) {
                leases.retain(|lease| lease.id() != publication);
            }
            Ok(LeaseChange::Updated {
                leases,
                lease: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::retention_lease::tests::{at, lease};
    use crate::transaction::TransactionBuilder;
    use lance_core::datatypes::Schema;

    const NEW_VERSION: u64 = 9;
    const NOW: i64 = 100;

    fn transaction(uuid: u128, operation: Operation) -> Transaction {
        Transaction {
            uuid: Uuid::from_u128(uuid).hyphenated().to_string(),
            ..Transaction::new_from_version(5, operation)
        }
    }

    fn acquire(id: u128, created_at: i64, expires_at: i64) -> Operation {
        Operation::AcquireLease {
            lease_id: Uuid::from_u128(id),
            created_at: at(created_at),
            expires_at: at(expires_at),
            label: Some("build".to_string()),
        }
    }

    fn renew(id: u128, expires_at: i64) -> Operation {
        Operation::RenewLease {
            lease_id: Uuid::from_u128(id),
            expires_at: at(expires_at),
        }
    }

    fn release(id: u128) -> Operation {
        Operation::ReleaseLease {
            lease_id: Uuid::from_u128(id),
        }
    }

    fn append() -> Operation {
        Operation::Append { fragments: vec![] }
    }

    fn derive(previous: &[RetentionLease], operation: Operation) -> Result<LeaseChange> {
        derive_retention_leases(previous, &transaction(99, operation), NEW_VERSION, at(NOW))
    }

    fn updated(leases: Vec<RetentionLease>) -> LeaseChange {
        LeaseChange::Updated {
            leases,
            lease: None,
        }
    }

    #[test]
    fn test_acquire_adds_lease_on_new_version() {
        let previous = vec![lease(1, 2, 200)];
        let expected = RetentionLease::try_new(
            Uuid::from_u128(5),
            NEW_VERSION,
            at(90),
            at(300),
            Some("build".to_string()),
        )
        .unwrap();
        assert_eq!(
            derive(&previous, acquire(5, 90, 300)).unwrap(),
            LeaseChange::Updated {
                leases: vec![lease(1, 2, 200), expected.clone()],
                lease: Some(expected),
            }
        );
    }

    #[test]
    fn test_acquire_of_listed_id_returns_stored_record() {
        let stored = lease(5, 2, 200);
        assert_eq!(
            derive(std::slice::from_ref(&stored), acquire(5, 90, 300)).unwrap(),
            LeaseChange::NoOp {
                stored: Some(stored)
            }
        );
    }

    #[test]
    fn test_acquire_of_expired_id_adds_fresh_lease() {
        let change = derive(&[lease(5, 2, NOW)], acquire(5, 90, 300)).unwrap();
        let LeaseChange::Updated { leases, lease } = change else {
            panic!("expected Updated, got {change:?}");
        };
        assert_eq!(leases.len(), 1);
        assert_eq!(lease.unwrap().version(), NEW_VERSION);
    }

    #[rstest::rstest]
    #[case::not_after_created(acquire(5, 150, 150), "must be later than created_at")]
    #[case::past(acquire(5, 10, 50), "is not in the future")]
    #[case::at_now(acquire(5, 10, NOW), "is not in the future")]
    #[case::own_uuid(acquire(99, 10, 300), "must differ from the acquiring transaction")]
    #[case::renew_absent(
        renew(5, 300),
        "Cannot renew retention lease 00000000-0000-0000-0000-000000000005: it is not listed"
    )]
    #[case::renew_expired(
        renew(1, 300),
        "it expired at 1970-01-01 00:01:40 UTC (now 1970-01-01 00:01:40 UTC)"
    )]
    fn test_invalid_requests_rejected(#[case] operation: Operation, #[case] message: &str) {
        let error = derive(&[lease(1, 2, NOW)], operation).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "got: {error}");
        assert!(error.to_string().contains(message), "got: {error}");
    }

    #[test]
    fn test_listed_id_still_rejects_malformed_acquire() {
        let error = derive(&[lease(5, 2, 200)], acquire(5, 150, 150)).unwrap_err();
        assert!(error.to_string().contains("must be later than created_at"));
    }

    #[test]
    fn test_renew_extends_only_expires_at() {
        let renewed = lease(1, 2, 200).with_expires_at(at(500));
        assert_eq!(
            derive(&[lease(1, 2, 200), lease(2, 3, 200)], renew(1, 500)).unwrap(),
            LeaseChange::Updated {
                leases: vec![renewed.clone(), lease(2, 3, 200)],
                lease: Some(renewed),
            }
        );
    }

    #[rstest::rstest]
    #[case::earlier(150)]
    #[case::equal(200)]
    fn test_renew_to_not_later_time_is_noop(#[case] expires_at: i64) {
        let stored = lease(1, 2, 200);
        assert_eq!(
            derive(std::slice::from_ref(&stored), renew(1, expires_at)).unwrap(),
            LeaseChange::NoOp {
                stored: Some(stored)
            }
        );
    }

    #[test]
    fn test_release_removes_listed_lease() {
        assert_eq!(
            derive(&[lease(1, 2, 200), lease(2, 3, 200)], release(1)).unwrap(),
            updated(vec![lease(2, 3, 200)])
        );
    }

    #[rstest::rstest]
    #[case::absent(5)]
    #[case::expired(1)]
    fn test_release_of_unlisted_lease_is_noop(#[case] id: u128) {
        assert_eq!(
            derive(&[lease(1, 2, NOW)], release(id)).unwrap(),
            LeaseChange::NoOp { stored: None }
        );
    }

    #[rstest::rstest]
    #[case::other_uuid(99, vec![lease(1, 2, 200), lease(2, 3, 200)])]
    #[case::publication(1, vec![lease(2, 3, 200)])]
    fn test_ordinary_commit_expires_and_publishes(
        #[case] uuid: u128,
        #[case] expected: Vec<RetentionLease>,
    ) {
        let previous = [lease(1, 2, 200), lease(3, 2, NOW), lease(2, 3, 200)];
        let change = derive_retention_leases(
            &previous,
            &transaction(uuid, append()),
            NEW_VERSION,
            at(NOW),
        )
        .unwrap();
        assert_eq!(change, updated(expected));
    }

    #[test]
    fn test_for_lease_transaction_publishes_lease() {
        let leased = lease(1, 2, 200);
        let publication = TransactionBuilder::for_lease(&leased, append()).build();
        assert_eq!(publication.read_version, 2);
        let change =
            derive_retention_leases(&[leased], &publication, NEW_VERSION, at(NOW)).unwrap();
        assert_eq!(change, updated(vec![]));
    }

    #[test]
    fn test_non_uuid_transaction_publishes_nothing() {
        let mut transaction = transaction(1, append());
        transaction.uuid = "not-a-uuid".to_string();
        let previous = [lease(1, 2, 200)];
        let change =
            derive_retention_leases(&previous, &transaction, NEW_VERSION, at(NOW)).unwrap();
        assert_eq!(change, updated(previous.to_vec()));
    }

    #[rstest::rstest]
    #[case::acquire(acquire(7, 10, 300))]
    #[case::renew(renew(2, 300))]
    #[case::release(release(2))]
    fn test_lease_operations_never_publish(#[case] operation: Operation) {
        let previous = [lease(1, 2, 200), lease(2, 3, 200)];
        let change =
            derive_retention_leases(&previous, &transaction(1, operation), NEW_VERSION, at(NOW))
                .unwrap();
        let LeaseChange::Updated { leases, .. } = change else {
            panic!("expected Updated, got {change:?}");
        };
        assert!(leases.iter().any(|lease| lease.id() == Uuid::from_u128(1)));
    }

    #[rstest::rstest]
    #[case::restore(Operation::Restore { version: 1 })]
    #[case::overwrite(Operation::Overwrite {
        fragments: vec![],
        schema: Schema::default(),
        config_upsert_values: None,
        initial_bases: None,
    })]
    #[case::clone(Operation::Clone {
        is_shallow: true,
        ref_name: None,
        ref_version: 2,
        ref_path: "source".to_string(),
        branch_name: None,
    })]
    fn test_operations_that_start_without_leases(#[case] operation: Operation) {
        let previous = [lease(1, 2, 200)];
        let change =
            derive_retention_leases(&previous, &transaction(1, operation), NEW_VERSION, at(NOW))
                .unwrap();
        assert_eq!(change, updated(vec![]));
    }

    #[test]
    fn test_duplicate_previous_ids_rejected() {
        let error = derive(&[lease(1, 2, 200), lease(1, 3, 300)], append()).unwrap_err();
        assert!(
            error.to_string().contains("duplicate lease id"),
            "got: {error}"
        );
    }

    #[test]
    fn test_unknown_operation_rejected() {
        let error = derive(&[], Operation::Unknown {}).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "got: {error}");
    }
}
