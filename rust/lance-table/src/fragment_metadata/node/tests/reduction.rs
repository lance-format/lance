// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::format::pb::fragment_action::Action;
use crate::format::{DataFile, DeletionFile, DeletionFileType, Fragment, pb};
use crate::fragment_metadata::support::{make_backfill_data_file, make_fragment};
use crate::fragment_metadata::{action, node};
use proptest::prelude::*;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rstest::rstest;
use std::collections::BTreeMap;
use std::num::NonZeroU64;

fn tagged(actions: Vec<pb::FragmentAction>) -> Vec<pb::FragmentTreeMutation> {
    actions
        .into_iter()
        .enumerate()
        .map(|(i, action)| pb::FragmentTreeMutation {
            action_sequence: i as u64 + 1,
            action: Some(action),
            fragment_count_delta: 0,
            total_rows_delta: 0,
            visible_rows_delta: 0,
        })
        .collect()
}

fn equivalent(base: Fragment, actions: Vec<pb::FragmentAction>) {
    let actions = tagged(actions);
    let mut expected = BTreeMap::from([(base.id, base.clone())]);
    node::apply_actions(&mut expected, actions.clone()).unwrap();
    let normalized = node::squash_buffer(actions, 0);
    let mut actual = BTreeMap::from([(base.id, base)]);
    node::apply_actions(&mut actual, normalized).unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
#[case::descriptor_fields(false)]
#[case::path_collision(true)]
fn add_replace_preserves_first_match_and_descriptor(#[case] collision: bool) {
    let mut base = make_fragment(7);
    let mut added = make_backfill_data_file(7, 0);
    if collision {
        added.path = base.files[0].path.clone();
    }
    // Replacement must retain field/column mappings from the matched file,
    // even when the replacement descriptor supplies different mappings.
    base.physical_rows = Some(100);
    let mut replacement = make_backfill_data_file(7, 9);
    replacement.fields = added.fields.clone();
    replacement.column_indices = vec![9].into();
    equivalent(
        base,
        vec![
            action::add_data_file(7, &added),
            action::replace_data_file(7, &added.path, &replacement),
        ],
    );
}

#[test]
fn replacement_chain_can_change_which_slot_matches() {
    let mut base = make_fragment(7);
    let second = make_backfill_data_file(7, 0);
    let first_path = base.files[0].path.clone();
    base.files.push(second.clone());
    let final_file = make_backfill_data_file(7, 9);
    equivalent(
        base,
        vec![
            action::replace_data_file(7, &second.path, &make_fragment(7).files[0]),
            action::replace_data_file(7, &first_path, &final_file),
        ],
    );
}

// Deliberately does not call node::apply_actions or any reducer function.
// This is the ordered-vector storage specification, not transaction validation.
fn reference_apply(state: &mut BTreeMap<u64, Fragment>, action: &pb::FragmentAction) {
    match action.action.clone().unwrap() {
        Action::UpsertFragment(f) => {
            state.insert(f.id, Fragment::try_from(f).unwrap());
        }
        Action::RemoveFragment(id) => {
            state.remove(&id);
        }
        Action::AddDataFile(a) => state
            .get_mut(&a.frag_id)
            .unwrap()
            .files
            .push(DataFile::try_from(a.file.unwrap()).unwrap()),
        Action::RemoveDataFile(a) => state
            .get_mut(&a.frag_id)
            .unwrap()
            .files
            .retain(|f| f.path != a.path),
        Action::ReplaceDataFile(a) => {
            let file = state
                .get_mut(&a.frag_id)
                .unwrap()
                .files
                .iter_mut()
                .find(|f| f.path == a.expected_path)
                .unwrap();
            file.path = a.path.clone();
            file.file_size_bytes = lance_io::utils::CachedFileSize::new(a.file_size_bytes);
            file.base_id = a.base_id;
        }
        Action::AddDeletionFile(a) => {
            let encoded = a.deletion_file.unwrap();
            let count = encoded.num_deleted_rows as usize;
            let mut file = DeletionFile::try_from(encoded).unwrap();
            file.num_deleted_rows = Some(count);
            state.get_mut(&a.frag_id).unwrap().deletion_file = Some(file)
        }
        Action::ClearDeletionFile(a) => state.get_mut(&a.frag_id).unwrap().deletion_file = None,
    }
}

fn random_action(
    state: &BTreeMap<u64, Fragment>,
    rng: &mut SmallRng,
    step: u32,
) -> pb::FragmentAction {
    let id = [0, 7, 123_456_789, u32::MAX as u64][rng.random_range(0..4)];
    let mut new_fragment = make_fragment(id);
    new_fragment.physical_rows = Some(100);
    if !state.contains_key(&id) {
        return action::upsert_fragment(&new_fragment);
    }
    let fragment = &state[&id];
    match rng.random_range(0..10) {
        0 => action::upsert_fragment(&new_fragment),
        1 => action::remove_fragment(id),
        2 | 3 => {
            let mut file = make_backfill_data_file(id, step);
            // Aliases and non-default descriptor fields are intentional.
            file.path = format!("alias-{}.lance", rng.random_range(0..4));
            file.column_indices = vec![rng.random_range(0..16)].into();
            action::add_data_file(id, &file)
        }
        4 | 5 if !fragment.files.is_empty() => {
            let path = &fragment.files[rng.random_range(0..fragment.files.len())].path;
            let mut file = make_backfill_data_file(id, step);
            file.path = format!("alias-{}.lance", rng.random_range(0..4));
            file.file_size_bytes = NonZeroU64::new(rng.random_range(1..5000)).into();
            file.base_id = Some(rng.random_range(1..8));
            action::replace_data_file(id, path, &file)
        }
        6 => action::remove_data_file(id, format!("alias-{}.lance", rng.random_range(0..4))),
        7 | 8 => action::add_deletion_file(
            id,
            &DeletionFile {
                read_version: step as u64 + 1,
                id: step as u64,
                file_type: DeletionFileType::Bitmap,
                num_deleted_rows: Some(rng.random_range(0..100)),
                base_id: None,
            },
        ),
        _ => action::clear_deletion_file(id),
    }
}

fn check_random_sequence(seed: u64, steps: usize) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let base = BTreeMap::from([(7, make_fragment(7).with_physical_rows(100))]);
    let mut expected = base.clone();
    let mut pending = Vec::new();
    let mut originals = Vec::new();
    for step in 0..steps {
        let action = random_action(&expected, &mut rng, step as u32);
        reference_apply(&mut expected, &action);
        let mut next = tagged(vec![action]).remove(0);
        next.action_sequence = step as u64 + 1;
        originals.push(next.clone());
        pending.push(next);
        // Each eight-action batch models one validated commit. Compare every
        // snapshot, including the prefix after incremental normalization.
        if step % 8 == 7 || step + 1 == steps {
            pending = node::squash_buffer(pending, 0);
            let mut actual = base.clone();
            node::apply_actions(&mut actual, pending.clone()).unwrap();
            assert_eq!(actual, expected, "seed={seed}, step={step}");
            let mut one_shot = base.clone();
            node::apply_actions(&mut one_shot, node::squash_buffer(originals.clone(), 0)).unwrap();
            assert_eq!(one_shot, expected, "one-shot seed={seed}, step={step}");
            assert_eq!(
                node::squash_buffer(pending.clone(), 0),
                pending,
                "normal form idempotence"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]
    #[test]
    fn random_normalization(seed in any::<u64>(), steps in 1usize..257) {
        check_random_sequence(seed, steps);
    }
}

#[rstest]
#[case::deletion_register(0)]
#[case::whole_reset(1)]
#[case::append_remove(2)]
#[case::duplicate_remove(3)]
#[case::same_path_replacement(4)]
fn legal_reductions_preserve_aggregates(#[case] case: u8) {
    let base = make_fragment(7);
    let file = make_backfill_data_file(7, 0);
    let deletion = DeletionFile {
        read_version: 1,
        id: 2,
        file_type: DeletionFileType::Bitmap,
        num_deleted_rows: Some(0),
        base_id: None,
    };
    let actions = match case {
        0 => vec![
            action::add_deletion_file(7, &deletion),
            action::add_data_file(7, &file),
            action::clear_deletion_file(7),
        ],
        1 => vec![
            action::add_data_file(7, &file),
            action::upsert_fragment(&base),
            action::add_data_file(7, &file),
        ],
        2 => vec![
            action::add_data_file(7, &file),
            action::remove_data_file(7, &file.path),
        ],
        3 => vec![
            action::remove_data_file(7, &file.path),
            action::remove_data_file(7, &file.path),
        ],
        _ => vec![
            action::replace_data_file(7, &base.files[0].path, &base.files[0]),
            action::replace_data_file(7, &base.files[0].path, &file),
        ],
    };
    equivalent(base, actions.clone());
    let mut input = tagged(actions);
    for (i, action) in input.iter_mut().enumerate() {
        action.fragment_count_delta = i as i64 - 1;
        action.total_rows_delta = i as i64 * 10 - 5;
        action.visible_rows_delta = i as i64 * 5 - 3;
    }
    let output = node::squash_buffer(input.clone(), 0);
    assert!(output.len() < input.len());
    assert_eq!(
        input.iter().map(|t| t.fragment_count_delta).sum::<i64>(),
        output.iter().map(|t| t.fragment_count_delta).sum::<i64>()
    );
    assert_eq!(
        input.iter().map(|t| t.total_rows_delta).sum::<i64>(),
        output.iter().map(|t| t.total_rows_delta).sum::<i64>()
    );
    assert_eq!(
        input.iter().map(|t| t.visible_rows_delta).sum::<i64>(),
        output.iter().map(|t| t.visible_rows_delta).sum::<i64>()
    );
    assert_eq!(
        input.last().unwrap().action_sequence,
        output.last().unwrap().action_sequence
    );
}

/// A buffer whose keys are all distinct has nothing to combine, so it comes
/// back ordered by action sequence and otherwise unchanged.
#[test]
fn distinct_keys_come_back_in_action_sequence_order() {
    let mut input = tagged(vec![
        action::clear_deletion_file(9),
        action::remove_fragment(2),
        action::upsert_fragment(&make_fragment(5)),
    ]);
    input[0].action_sequence = 7;
    input[1].action_sequence = 3;
    input[2].action_sequence = 5;
    let mut expected = input.clone();
    expected.sort_by_key(|tagged| tagged.action_sequence);
    assert_eq!(node::squash_buffer(input, 0), expected);
}

#[test]
fn unrepresentable_aggregate_keeps_original_run() {
    let mut input = tagged(vec![
        action::clear_deletion_file(7),
        action::clear_deletion_file(7),
    ]);
    input[0].visible_rows_delta = i64::MAX;
    input[1].visible_rows_delta = 1;
    assert_eq!(node::squash_buffer(input.clone(), 0), input);
}

#[test]
fn history_below_contiguous_from_is_never_combined() {
    let input = tagged(vec![
        action::upsert_fragment(&make_fragment(7)),
        action::clear_deletion_file(7),
        action::clear_deletion_file(7),
        action::clear_deletion_file(7),
    ]);
    assert_eq!(node::squash_buffer(input.clone(), 0).len(), 1);
    let output = node::squash_buffer(input.clone(), 3);
    assert_eq!(output[..2], input[..2]);
    assert_eq!(output.len(), 3);
    assert_eq!(output[2].action_sequence, 4);
}
