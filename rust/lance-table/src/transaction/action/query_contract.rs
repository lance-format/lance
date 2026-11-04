// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What the segments of one logical index must agree on to be queried together.
//!
//! The query path unions a logical index's segments, so a segment that answers
//! a different query -- distances under another metric, say -- would mix
//! incomparable results into one answer. Model state trained per segment, such
//! as centroids or codebooks, is not part of the contract: independently
//! trained segments are fine as long as they answer the same query.
//!
//! The contract is keyed by the type of the index details. A type with no
//! contract here constrains nothing beyond its type url.

use prost::Message;
use prost_types::Any;

const VECTOR_DETAILS_TYPE: &str = "VectorIndexDetails";

/// The query a segment answers, as far as its index type defines one.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum QueryContract {
    Vector(VectorContract),
    /// Details of a type that has a contract, but which could not be decoded
    /// or predate the fields the contract reads. Without a way to tell which
    /// differences matter, every byte does.
    Opaque(Vec<u8>),
}

/// What `validate_vector_query_compatibility` in `lance` checks, read from
/// the manifest's details rather than the opened index. The dimension is left
/// out: it follows from the field, which the segments already share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VectorContract {
    metric: i32,
    has_hnsw: bool,
    quantization: Quantization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Quantization {
    Flat,
    Product,
    Scalar,
    Rabit,
}

/// One facet two contracts disagree on, with each side's value.
#[derive(Debug, PartialEq)]
pub(super) struct Mismatch {
    pub facet: &'static str,
    pub ours: String,
    pub theirs: String,
}

impl QueryContract {
    /// The contract of a segment with these details, or `None` when its index
    /// type defines none.
    pub(super) fn of(details: Option<&Any>) -> Option<Self> {
        let details = details?;
        if !details.type_url.ends_with(VECTOR_DETAILS_TYPE) {
            return None;
        }
        let contract = VectorDetails::decode(details.value.as_slice())
            .ok()
            .and_then(|decoded| decoded.contract());
        Some(contract.map_or_else(|| Self::Opaque(details.value.clone()), Self::Vector))
    }

    /// The first facet on which `self` differs from `other`, if any.
    pub(super) fn mismatch(&self, other: &Self) -> Option<Mismatch> {
        match (self, other) {
            (Self::Vector(ours), Self::Vector(theirs)) => ours.mismatch(theirs),
            _ if self == other => None,
            _ => Some(Mismatch {
                facet: "index details",
                ours: self.summary(),
                theirs: other.summary(),
            }),
        }
    }

    fn summary(&self) -> String {
        match self {
            Self::Vector(contract) => format!("{contract:?}"),
            Self::Opaque(bytes) => format!("{} undecodable bytes", bytes.len()),
        }
    }
}

impl VectorContract {
    fn mismatch(&self, other: &Self) -> Option<Mismatch> {
        if self.metric != other.metric {
            return Some(Mismatch {
                facet: "metric",
                ours: metric_name(self.metric),
                theirs: metric_name(other.metric),
            });
        }
        if self.has_hnsw != other.has_hnsw {
            let sub_index = |has_hnsw: bool| if has_hnsw { "HNSW" } else { "none" }.to_string();
            return Some(Mismatch {
                facet: "sub-index",
                ours: sub_index(self.has_hnsw),
                theirs: sub_index(other.has_hnsw),
            });
        }
        if self.quantization != other.quantization {
            return Some(Mismatch {
                facet: "quantization",
                ours: format!("{:?}", self.quantization),
                theirs: format!("{:?}", other.quantization),
            });
        }
        None
    }
}

/// Names follow `VectorMetricType` in `protos/index.proto`.
fn metric_name(metric: i32) -> String {
    match metric {
        0 => "L2".into(),
        1 => "Cosine".into(),
        2 => "Dot".into(),
        3 => "Hamming".into(),
        other => format!("unknown metric {other}"),
    }
}

/// The fields of `lance.index.pb.VectorIndexDetails` the contract reads. The
/// generated type lives in `lance-index`, which depends on this crate, so the
/// tags here mirror `protos/index.proto` and every other field is skipped.
#[derive(Clone, PartialEq, Message)]
struct VectorDetails {
    #[prost(int32, tag = "1")]
    metric_type: i32,
    #[prost(message, optional, tag = "3")]
    hnsw_index_config: Option<Skipped>,
    #[prost(oneof = "Compression", tags = "4, 5, 6, 8")]
    compression: Option<Compression>,
}

/// A message whose presence matters but whose contents do not.
#[derive(Clone, PartialEq, Message)]
struct Skipped {}

#[derive(Clone, PartialEq, prost::Oneof)]
enum Compression {
    #[prost(message, tag = "4")]
    Pq(Skipped),
    #[prost(message, tag = "5")]
    Sq(Skipped),
    #[prost(message, tag = "6")]
    Rq(Skipped),
    #[prost(message, tag = "8")]
    Flat(Skipped),
}

impl VectorDetails {
    /// `None` for details written before the compression was recorded, which
    /// do not say enough to compare.
    fn contract(&self) -> Option<VectorContract> {
        let quantization = match self.compression.as_ref()? {
            Compression::Pq(_) => Quantization::Product,
            Compression::Sq(_) => Quantization::Scalar,
            Compression::Rq(_) => Quantization::Rabit,
            Compression::Flat(_) => Quantization::Flat,
        };
        Some(VectorContract {
            metric: self.metric_type,
            has_hnsw: self.hnsw_index_config.is_some(),
            quantization,
        })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use rstest::rstest;

    pub(in crate::transaction::action) const L2: i32 = 0;
    pub(in crate::transaction::action) const DOT: i32 = 2;

    /// Encoded `VectorIndexDetails` with the given facets, plus a training
    /// knob (`target_partition_size`, tag 2) that differs per segment without
    /// affecting the contract.
    pub(in crate::transaction::action) fn vector_details(
        metric: i32,
        hnsw: bool,
        compression: Option<&str>,
        target_partition_size: u64,
    ) -> Any {
        #[derive(Clone, PartialEq, Message)]
        struct WithPartitionSize {
            #[prost(int32, tag = "1")]
            metric_type: i32,
            #[prost(uint64, tag = "2")]
            target_partition_size: u64,
            #[prost(message, optional, tag = "3")]
            hnsw_index_config: Option<Skipped>,
            #[prost(oneof = "Compression", tags = "4, 5, 6, 8")]
            compression: Option<Compression>,
        }
        let compression = compression.map(|kind| match kind {
            "pq" => Compression::Pq(Skipped {}),
            "sq" => Compression::Sq(Skipped {}),
            "rq" => Compression::Rq(Skipped {}),
            "flat" => Compression::Flat(Skipped {}),
            other => panic!("unknown compression {other}"),
        });
        Any {
            type_url: "type.googleapis.com/lance.index.pb.VectorIndexDetails".into(),
            value: WithPartitionSize {
                metric_type: metric,
                target_partition_size,
                hnsw_index_config: hnsw.then_some(Skipped {}),
                compression,
            }
            .encode_to_vec(),
        }
    }

    fn contract(details: &Any) -> QueryContract {
        QueryContract::of(Some(details)).unwrap()
    }

    #[rstest]
    #[case::metric(vector_details(DOT, false, Some("flat"), 0), "metric", "Dot")]
    #[case::sub_index(vector_details(L2, true, Some("flat"), 0), "sub-index", "HNSW")]
    #[case::quantization(vector_details(L2, false, Some("pq"), 0), "quantization", "Product")]
    fn test_vector_contracts_differ_by_facet(
        #[case] other: Any,
        #[case] facet: &str,
        #[case] value: &str,
    ) {
        let ours = contract(&vector_details(L2, false, Some("flat"), 0));
        let mismatch = contract(&other).mismatch(&ours).unwrap();
        assert_eq!(mismatch.facet, facet);
        assert_eq!(mismatch.ours, value);
    }

    #[test]
    fn test_independently_trained_vector_segments_share_a_contract() {
        let ours = vector_details(L2, true, Some("sq"), 1024);
        let theirs = vector_details(L2, true, Some("sq"), 4096);
        assert_ne!(ours.value, theirs.value);
        assert_eq!(contract(&ours).mismatch(&contract(&theirs)), None);
    }

    /// Details that do not say how vectors are compressed predate the field,
    /// and cannot be compared facet by facet; only identical bytes agree.
    #[rstest]
    #[case::no_compression(vector_details(L2, false, None, 1024))]
    #[case::not_a_message(Any {
        type_url: "type.googleapis.com/lance.index.pb.VectorIndexDetails".into(),
        value: vec![0xff, 0xff],
    })]
    fn test_undecodable_vector_details_must_match_byte_for_byte(#[case] details: Any) {
        let opaque = contract(&details);
        assert!(matches!(opaque, QueryContract::Opaque(_)), "{opaque:?}");
        assert_eq!(opaque.mismatch(&opaque.clone()), None);

        let decodable = contract(&vector_details(L2, false, Some("flat"), 1024));
        assert_eq!(opaque.mismatch(&decodable).unwrap().facet, "index details");
        assert_eq!(decodable.mismatch(&opaque).unwrap().facet, "index details");
    }

    #[test]
    fn test_other_index_types_have_no_contract() {
        let details = Any {
            type_url: "type.googleapis.com/lance.index.pb.BTreeIndexDetails".into(),
            value: vec![1, 2, 3],
        };
        assert_eq!(QueryContract::of(Some(&details)), None);
        assert_eq!(QueryContract::of(None), None);
    }
}
