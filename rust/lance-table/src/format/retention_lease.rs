// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Retention lease records stored in the manifest.
//!
//! See the [Retention Leases specification](https://lance.org/format/table/retention_leases/).

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::format::pb;

/// A listed retention lease: it keeps [`Self::version`] and every later
/// version until the holder publishes a transaction whose UUID is
/// [`Self::id`], releases it, or lets it expire.
///
/// A lease can only be obtained by acquiring one or by deserializing one
/// (serde or protobuf), which lets the holder ship it to distributed workers.
/// Deserialization validates the record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, DeepSizeOf)]
#[serde(try_from = "RetentionLeaseFields")]
pub struct RetentionLease {
    id: Uuid,
    version: u64,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    label: Option<String>,
}

/// Unvalidated serde form of [`RetentionLease`].
#[derive(Deserialize)]
struct RetentionLeaseFields {
    id: Uuid,
    version: u64,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    label: Option<String>,
}

impl TryFrom<RetentionLeaseFields> for RetentionLease {
    type Error = Error;

    fn try_from(fields: RetentionLeaseFields) -> Result<Self> {
        Self::try_new(
            fields.id,
            fields.version,
            fields.created_at,
            fields.expires_at,
            fields.label,
        )
    }
}

impl RetentionLease {
    pub(crate) fn try_new(
        id: Uuid,
        version: u64,
        created_at: DateTime<Utc>,
        expires_at: DateTime<Utc>,
        label: Option<String>,
    ) -> Result<Self> {
        if version == 0 {
            return Err(Error::invalid_input(format!(
                "retention lease {id} has version 0; the leased version must be positive"
            )));
        }
        if expires_at <= created_at {
            return Err(Error::invalid_input(format!(
                "retention lease {id} expires_at {expires_at} must be later than created_at {created_at}"
            )));
        }
        Ok(Self {
            id,
            version,
            created_at,
            expires_at,
            label: label.filter(|label| !label.is_empty()),
        })
    }

    /// The UUID of the transaction the holder intends to publish.
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// The leased version: the version created by the acquisition commit.
    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    pub fn expires_at(&self) -> DateTime<Utc> {
        self.expires_at
    }

    /// Human-readable context with no retention meaning.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub(crate) fn is_expired(&self, now: DateTime<Utc>) -> bool {
        self.expires_at <= now
    }

    pub(crate) fn with_expires_at(&self, expires_at: DateTime<Utc>) -> Self {
        Self {
            expires_at,
            ..self.clone()
        }
    }
}

/// Reject a lease list with duplicate IDs, which writers must not carry forward.
pub fn validate_retention_lease_ids(leases: &[RetentionLease]) -> Result<()> {
    let mut seen = HashSet::with_capacity(leases.len());
    for lease in leases {
        if !seen.insert(lease.id) {
            return Err(Error::corrupt_file_named(
                "manifest",
                format!(
                    "retention lease list contains duplicate lease id {}",
                    lease.id
                ),
            ));
        }
    }
    Ok(())
}

pub fn timestamp_from_pb(
    timestamp: Option<&prost_types::Timestamp>,
    name: &str,
) -> Result<DateTime<Utc>> {
    let timestamp = timestamp.ok_or_else(|| Error::invalid_input(format!("{name} is missing")))?;
    // The protobuf Timestamp contract limits nanos to [0, 1e9); chrono would
    // otherwise read larger values as a leap second.
    u32::try_from(timestamp.nanos)
        .ok()
        .filter(|nanos| *nanos < 1_000_000_000)
        .and_then(|nanos| DateTime::from_timestamp(timestamp.seconds, nanos))
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "{name} is out of range: seconds={}, nanos={}",
                timestamp.seconds, timestamp.nanos
            ))
        })
}

pub fn timestamp_to_pb(timestamp: &DateTime<Utc>) -> prost_types::Timestamp {
    prost_types::Timestamp {
        seconds: timestamp.timestamp(),
        nanos: timestamp.timestamp_subsec_nanos() as i32,
    }
}

pub fn uuid_from_pb(uuid: Option<&pb::Uuid>, name: &str) -> Result<Uuid> {
    let uuid = uuid.ok_or_else(|| Error::invalid_input(format!("{name} is missing")))?;
    Uuid::try_from(uuid).map_err(|_| {
        Error::invalid_input(format!(
            "{name} must be 16 bytes, got {} bytes",
            uuid.uuid.len()
        ))
    })
}

impl TryFrom<&pb::RetentionLease> for RetentionLease {
    type Error = Error;

    fn try_from(message: &pb::RetentionLease) -> Result<Self> {
        let id = uuid_from_pb(message.id.as_ref(), "retention lease id")?;
        let version = message.version.ok_or_else(|| {
            Error::invalid_input(format!("retention lease {id} is missing its version"))
        })?;
        let created_at = timestamp_from_pb(
            message.created_at.as_ref(),
            &format!("retention lease {id} created_at"),
        )?;
        let expires_at = timestamp_from_pb(
            message.expires_at.as_ref(),
            &format!("retention lease {id} expires_at"),
        )?;
        Self::try_new(id, version, created_at, expires_at, message.label.clone())
    }
}

impl From<&RetentionLease> for pb::RetentionLease {
    fn from(lease: &RetentionLease) -> Self {
        Self {
            id: Some(pb::Uuid::from(&lease.id)),
            version: Some(lease.version),
            created_at: Some(timestamp_to_pb(&lease.created_at)),
            expires_at: Some(timestamp_to_pb(&lease.expires_at)),
            label: lease.label.clone(),
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn at(seconds: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(seconds, 0).unwrap()
    }

    pub fn lease(id: u128, version: u64, expires_at: i64) -> RetentionLease {
        RetentionLease::try_new(
            Uuid::from_u128(id),
            version,
            at(0),
            at(expires_at),
            Some(format!("lease {id}")),
        )
        .unwrap()
    }

    fn valid_pb() -> pb::RetentionLease {
        pb::RetentionLease {
            id: Some(pb::Uuid { uuid: vec![7; 16] }),
            version: Some(3),
            created_at: Some(prost_types::Timestamp {
                seconds: 100,
                nanos: 5,
            }),
            expires_at: Some(prost_types::Timestamp {
                seconds: 200,
                nanos: 0,
            }),
            label: Some("build index".to_string()),
        }
    }

    #[test]
    fn test_pb_roundtrip() {
        let lease = RetentionLease::try_from(&valid_pb()).unwrap();
        assert_eq!(lease.id(), Uuid::from_bytes([7; 16]));
        assert_eq!(lease.version(), 3);
        assert_eq!(
            lease.created_at(),
            DateTime::from_timestamp(100, 5).unwrap()
        );
        assert_eq!(lease.expires_at(), at(200));
        assert_eq!(lease.label(), Some("build index"));
        assert_eq!(pb::RetentionLease::from(&lease), valid_pb());
    }

    #[test]
    fn test_empty_label_means_no_label() {
        let mut message = valid_pb();
        message.label = Some(String::new());
        let lease = RetentionLease::try_from(&message).unwrap();
        assert_eq!(lease.label(), None);
        assert_eq!(pb::RetentionLease::from(&lease).label, None);
    }

    #[rstest::rstest]
    #[case::missing_id(|m: &mut pb::RetentionLease| m.id = None, "id is missing")]
    #[case::short_id(|m: &mut pb::RetentionLease| m.id = Some(pb::Uuid { uuid: vec![1; 15] }), "must be 16 bytes")]
    #[case::missing_version(|m: &mut pb::RetentionLease| m.version = None, "missing its version")]
    #[case::zero_version(|m: &mut pb::RetentionLease| m.version = Some(0), "version 0")]
    #[case::missing_created_at(|m: &mut pb::RetentionLease| m.created_at = None, "created_at is missing")]
    #[case::missing_expires_at(|m: &mut pb::RetentionLease| m.expires_at = None, "expires_at is missing")]
    #[case::negative_nanos(|m: &mut pb::RetentionLease| m.created_at.as_mut().unwrap().nanos = -1, "out of range")]
    #[case::leap_nanos(|m: &mut pb::RetentionLease| m.created_at.as_mut().unwrap().nanos = 1_000_000_000, "out of range")]
    #[case::expires_at_created(|m: &mut pb::RetentionLease| m.expires_at = m.created_at, "must be later than created_at")]
    #[case::expires_before_created(|m: &mut pb::RetentionLease| m.expires_at.as_mut().unwrap().seconds = 50, "must be later than created_at")]
    fn test_pb_rejects_invalid_record(
        #[case] corrupt: fn(&mut pb::RetentionLease),
        #[case] message: &str,
    ) {
        let mut record = valid_pb();
        corrupt(&mut record);
        let error = RetentionLease::try_from(&record).unwrap_err();
        assert!(error.to_string().contains(message), "got: {error}");
    }

    #[test]
    fn test_serde_roundtrip_validates() {
        let lease = lease(1, 4, 60);
        let json = serde_json::to_string(&lease).unwrap();
        assert_eq!(
            serde_json::from_str::<RetentionLease>(&json).unwrap(),
            lease
        );

        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["version"] = 0.into();
        let error = serde_json::from_value::<RetentionLease>(value).unwrap_err();
        assert!(error.to_string().contains("version 0"), "got: {error}");
    }

    #[test]
    fn test_duplicate_ids_rejected() {
        validate_retention_lease_ids(&[lease(1, 2, 10), lease(2, 2, 10)]).unwrap();
        let error = validate_retention_lease_ids(&[lease(1, 2, 10), lease(1, 3, 20)]).unwrap_err();
        assert!(
            error.to_string().contains("duplicate lease id"),
            "got: {error}"
        );
    }
}
