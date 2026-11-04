// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, BTreeSet};

use proptest::prelude::*;
use roaring::RoaringTreemap;

use super::*;
use crate::format::pb::fragment_reuse_index_details::transition;
use crate::system_index::frag_reuse::ledger::Mapping;

pub fn digest(id: u64, physical_rows: u64, num_deleted_rows: u64) -> pb::FragmentDigest {
    pb::FragmentDigest {
        id,
        physical_rows,
        num_deleted_rows,
    }
}

pub fn address(fragment: u64, offset: u64) -> u64 {
    (fragment << 32) | offset
}

/// A fragment as a rewrite sees it: its physical row count and which
/// offsets are deleted.
#[derive(Debug, Clone)]
pub struct Source {
    pub id: u64,
    pub physical_rows: u64,
    pub deleted: BTreeSet<u64>,
}

impl Source {
    fn digest(&self) -> pb::FragmentDigest {
        digest(self.id, self.physical_rows, self.deleted.len() as u64)
    }

    fn survivors(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.physical_rows)
            .filter(|offset| !self.deleted.contains(offset))
            .map(|offset| address(self.id, offset))
    }
}

pub fn changed_row_addrs(sources: &[Source]) -> Vec<u8> {
    let survivors: RoaringTreemap = sources.iter().flat_map(Source::survivors).collect();
    let mut bytes = Vec::new();
    survivors.serialize_into(&mut bytes).unwrap();
    bytes
}

pub fn ordered(sources: &[Source], destinations: &[(u64, u64)]) -> pb::Transition {
    pb::Transition {
        sources: sources.iter().map(Source::digest).collect(),
        destinations: destinations
            .iter()
            .map(|&(id, rows)| digest(id, rows, 0))
            .collect(),
        mapping: Some(transition::Mapping::OrderedCompaction(
            pb::OrderedCompaction {
                changed_row_addrs: changed_row_addrs(sources),
            },
        )),
    }
}

pub fn legacy(
    dataset_version: u64,
    sources: &[Source],
    destinations: &[(u64, u64)],
) -> pb::Version {
    let transition = ordered(sources, destinations);
    let Some(transition::Mapping::OrderedCompaction(mapping)) = transition.mapping else {
        unreachable!()
    };
    pb::Version {
        dataset_version,
        groups: vec![pb::Group {
            changed_row_addrs: mapping.changed_row_addrs,
            old_fragments: transition.sources,
            new_fragments: transition.destinations,
        }],
    }
}

pub fn whole(id: u64, physical_rows: u64) -> Source {
    Source {
        id,
        physical_rows,
        deleted: BTreeSet::new(),
    }
}

pub fn partition(source: u64, destination: u64, map_id: &str) -> pb::Transition {
    pb::Transition {
        sources: vec![digest(source, 4, 0)],
        destinations: vec![digest(destination, 4, 0)],
        mapping: Some(transition::Mapping::StablePartition(pb::StablePartition {
            map_id: map_id.to_string(),
            map_size_bytes: 64,
            base_id: None,
        })),
    }
}

pub const MAP_A: &str = "00000000-0000-0000-0000-00000000000a";
pub const MAP_B: &str = "00000000-0000-0000-0000-00000000000b";

/// An envelope field no released writer emits.
pub fn unknown_record() -> Vec<u8> {
    length_delimited(9, b"future")
}

/// A transition carrying a field this implementation does not know.
pub fn transition_with_unknown_field() -> Vec<u8> {
    let mut payload = ordered(&[whole(7, 2)], &[(8, 2)]).encode_to_vec();
    payload.extend_from_slice(&length_delimited(15, b"future"));
    length_delimited(TRANSITION_FIELD, &payload)
}

#[test]
fn content_round_trips_byte_for_byte() {
    let mut content = length_delimited(
        LEGACY_VERSION_FIELD,
        &legacy(3, &[whole(0, 2)], &[(1, 2)]).encode_to_vec(),
    );
    content.extend(unknown_record());
    content.extend(transition_with_unknown_field());
    content.extend(length_delimited(
        TRANSITION_FIELD,
        &ordered(&[whole(1, 2)], &[(2, 2)]).encode_to_vec(),
    ));
    let content = Bytes::from(content);
    let history = RawHistory::new(HistoryFormat::Tagged, content.clone())
        .validate()
        .unwrap();
    assert_eq!(history.content(), content);
    assert!(!history.is_fully_interpretable());
    assert_eq!(
        history
            .records()
            .map(|record| match record {
                Record::LegacyVersion(_) => "legacy",
                Record::Transition(_) => "transition",
                Record::Uninterpretable { .. } => "uninterpretable",
            })
            .collect::<Vec<_>>(),
        ["legacy", "uninterpretable", "uninterpretable", "transition"]
    );
}

#[test]
fn records_added_since_is_append_only() {
    let base = FragReuseHistory::empty(HistoryFormat::Tagged)
        .append(RewriteRecord::new(
            1,
            vec![ordered(&[whole(0, 1)], &[(1, 1)])],
        ))
        .unwrap();
    let added = ordered(&[whole(1, 1)], &[(2, 1)]);
    let proposed = base
        .append(RewriteRecord::new(1, vec![added.clone()]))
        .unwrap();
    assert_eq!(
        proposed.records_added_since(&base).unwrap(),
        vec![Record::Transition(added)]
    );

    let dropped = FragReuseHistory::empty(HistoryFormat::Tagged)
        .append(RewriteRecord::new(
            1,
            vec![ordered(&[whole(1, 1)], &[(2, 1)])],
        ))
        .unwrap();
    let error = dropped.records_added_since(&base).unwrap_err();
    assert!(error.to_string().contains("drops"), "{error}");

    let survivor = |deleted: u64| Source {
        id: 0,
        physical_rows: 2,
        deleted: BTreeSet::from([deleted]),
    };
    let recorded = FragReuseHistory::empty(HistoryFormat::Tagged)
        .append(RewriteRecord::new(
            1,
            vec![ordered(&[survivor(1)], &[(1, 1)])],
        ))
        .unwrap();
    let changed = FragReuseHistory::empty(HistoryFormat::Tagged)
        .append(RewriteRecord::new(
            1,
            vec![ordered(&[survivor(0)], &[(1, 1)])],
        ))
        .unwrap();
    let error = changed.records_added_since(&recorded).unwrap_err();
    assert!(error.to_string().contains("changes"), "{error}");
}

/// One rewrite the model applies: which live fragments it consumes, which of
/// their rows are deleted first, and how the survivors split into
/// destinations.
#[derive(Debug, Clone)]
struct Rewrite {
    sources: Vec<usize>,
    deletes: Vec<(usize, u64)>,
    splits: Vec<u64>,
}

fn rewrite_strategy() -> impl Strategy<Value = Rewrite> {
    (
        prop::collection::vec(0usize..8, 1..4),
        prop::collection::vec((0usize..8, 0u64..6), 0..4),
        prop::collection::vec(1u64..5, 1..4),
    )
        .prop_map(|(sources, deletes, splits)| Rewrite {
            sources,
            deletes,
            splits,
        })
}

/// Apply `rewrites` to a table of `initial` fragments, recording the first
/// `legacy_rewrites` as version 0 compaction versions and the rest as tagged
/// transitions, and return the history with every original address's final
/// location (`None` when a rewrite dropped it).
fn simulate(
    initial: &[u64],
    rewrites: &[Rewrite],
    legacy_rewrites: usize,
) -> (FragReuseHistory, BTreeMap<u64, Option<u64>>) {
    let mut live: Vec<Source> = initial
        .iter()
        .enumerate()
        .map(|(id, &rows)| whole(id as u64, rows))
        .collect();
    let mut next_id = initial.len() as u64;
    let mut location: BTreeMap<u64, Option<u64>> = live
        .iter()
        .flat_map(|source| (0..source.physical_rows).map(move |o| address(source.id, o)))
        .map(|addr| (addr, Some(addr)))
        .collect();
    let mut history = FragReuseHistory::empty(HistoryFormat::Legacy);
    for (step, rewrite) in rewrites.iter().enumerate() {
        if live.is_empty() {
            break;
        }
        let mut picked: Vec<usize> = rewrite
            .sources
            .iter()
            .map(|choice| choice % live.len())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        for &(choice, offset) in &rewrite.deletes {
            let source = &mut live[picked[choice % picked.len()]];
            source.deleted.insert(offset % source.physical_rows);
        }
        let sources: Vec<Source> = picked.iter().map(|&i| live[i].clone()).collect();
        let survivors: Vec<u64> = sources.iter().flat_map(Source::survivors).collect();

        let mut destinations = Vec::new();
        let mut remaining = survivors.len() as u64;
        for &size in &rewrite.splits {
            if remaining == 0 {
                break;
            }
            let rows = size.min(remaining);
            destinations.push((next_id, rows));
            next_id += 1;
            remaining -= rows;
        }
        if remaining > 0 {
            destinations.push((next_id, remaining));
            next_id += 1;
        }

        let mut new_addresses = destinations
            .iter()
            .flat_map(|&(id, rows)| (0..rows).map(move |offset| address(id, offset)));
        let moved: BTreeMap<u64, u64> = survivors
            .iter()
            .map(|&old| (old, new_addresses.next().unwrap()))
            .collect();
        for current in location.values_mut() {
            if let Some(addr) = *current
                && sources.iter().any(|source| source.id == addr >> 32)
            {
                *current = moved.get(&addr).copied();
            }
        }

        history = if step < legacy_rewrites {
            history
                .append(RewriteRecord::new(
                    step as u64 + 1,
                    vec![ordered(&sources, &destinations)],
                ))
                .unwrap()
        } else {
            history
                .lift()
                .unwrap()
                .append(RewriteRecord::new(
                    1,
                    vec![ordered(&sources, &destinations)],
                ))
                .unwrap()
        };

        picked.sort_unstable_by(|a, b| b.cmp(a));
        for i in picked {
            live.remove(i);
        }
        live.extend(destinations.iter().map(|&(id, rows)| whole(id, rows)));
    }
    (history, location)
}

/// Follow `addr` through every transition that consumes its fragment.
pub fn translate(ledger: &FragReuseLedger, mut addr: u64) -> Option<u64> {
    while let Some(position) = ledger.consumer((addr >> 32) as u32) {
        let Mapping::OrderedCompaction(remap) = ledger.transitions()[position].mapping() else {
            panic!("the model only records ordered compactions");
        };
        addr = remap.get(addr)??;
    }
    Some(addr)
}

proptest! {
    #[test]
    fn translation_matches_a_per_address_model(
        initial in prop::collection::vec(1u64..6, 1..5),
        rewrites in prop::collection::vec(rewrite_strategy(), 1..5),
        legacy_rewrites in 0usize..3,
    ) {
        let (history, location) = simulate(&initial, &rewrites, legacy_rewrites);
        for (&original, &expected) in &location {
            prop_assert_eq!(translate(history.ledger(), original), expected, "address {:#x}", original);
        }

        let decoded = RawHistory::new(history.format(), history.content()).validate().unwrap();
        prop_assert_eq!(&decoded, &history);
        prop_assert_eq!(decoded.fragment_bitmap(), history.fragment_bitmap());
    }

}
