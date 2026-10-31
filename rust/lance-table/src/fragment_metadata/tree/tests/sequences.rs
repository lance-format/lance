// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The model edits its records by hand and does not call the tree's reducer.

use super::*;
use crate::format::{DeletionFile, DeletionFileType};
use proptest::prelude::*;
use rand::{Rng, SeedableRng, rngs::SmallRng};

const CHECKPOINT_POLICY: SnapshotPolicy = SnapshotPolicy {
    inline_root_bytes: 0,
    max_suffix_bytes: 0,
};

const SUFFIX_POLICY: SnapshotPolicy = SnapshotPolicy {
    inline_root_bytes: 0,
    max_suffix_bytes: 1024,
};

/// The tree a sequence starts from and how its commits publish.
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// A routing level of small nodes under the root. The semantic buffer
    /// matches the node size, so most commits drain.
    Interiors,
    /// The root has room for every leaf a sequence adds. Opening squashes the
    /// root buffer and the suffix together, while validation replays them one
    /// by one.
    RootOverLeaves,
}

impl Shape {
    fn bootstrap_fragments(self) -> u64 {
        match self {
            Self::Interiors => 64,
            Self::RootOverLeaves => 16,
        }
    }

    fn bootstrap_height(self) -> u32 {
        match self {
            Self::Interiors => 2,
            Self::RootOverLeaves => 1,
        }
    }

    fn config(self) -> FragmentTreeConfig {
        let max_node_bytes = match self {
            Self::Interiors => 512,
            Self::RootOverLeaves => 4096,
        };
        FragmentTreeConfig::new(max_node_bytes, u32::MAX).with_max_leaf_bytes(6 * 1024)
    }

    fn policy(self, rng: &mut SmallRng) -> SnapshotPolicy {
        match self {
            Self::Interiors => random_policy(rng),
            // A checkpoint keeps undrained actions in the root buffer, where
            // a later suffix can act on the same fragments again.
            Self::RootOverLeaves if rng.random_bool(0.5) => CHECKPOINT_POLICY,
            Self::RootOverLeaves => SUFFIX_POLICY,
        }
    }

    /// A bulk commit leaves the root buffer empty, so a root over leaves
    /// commits in bulk less often.
    fn mode(self, rng: &mut SmallRng) -> CommitMode {
        let bulk_share = match self {
            Self::Interiors => 0.5,
            Self::RootOverLeaves => 0.25,
        };
        commit_mode(rng.random_bool(bulk_share))
    }
}

struct Model {
    fragments: BTreeMap<u64, Fragment>,
    next_fragment_id: u64,
}

impl Model {
    fn upsert(&mut self, id: u64, rng: &mut SmallRng) -> pb::FragmentAction {
        let mut fragment = make_fragment(id);
        fragment.physical_rows = Some(rng.random_range(1..=8));
        let action = action::upsert_fragment(&fragment).unwrap();
        self.fragments.insert(id, fragment);
        action
    }

    fn append(&mut self, rng: &mut SmallRng) -> pb::FragmentAction {
        let id = self.next_fragment_id;
        self.next_fragment_id += 1;
        self.upsert(id, rng)
    }

    fn remove(&mut self, id: u64) -> pb::FragmentAction {
        self.fragments.remove(&id);
        action::remove_fragment(id)
    }

    /// Thin out a run of live ids so leaves and interiors shrink and merge.
    fn contract(&mut self, rng: &mut SmallRng) -> Vec<pb::FragmentAction> {
        let ids: Vec<u64> = self.fragments.keys().copied().collect();
        let start = rng.random_range(0..ids.len());
        ids[start..]
            .iter()
            .take(16)
            .enumerate()
            .filter(|(position, _)| position % 4 != 3)
            .map(|(_, id)| self.remove(*id))
            .collect()
    }

    fn backfill(&mut self, id: u64, salt: u32) -> pb::FragmentAction {
        let file = make_backfill_data_file(id, salt);
        let action = action::add_data_file(id, &file);
        self.fragments.get_mut(&id).unwrap().files.push(file);
        action
    }

    /// Backfills append, so the base file stays first.
    fn replace_base_file(&mut self, id: u64, salt: u32) -> pb::FragmentAction {
        let replacement = make_replacement_data_file(id, salt);
        let base = &mut self.fragments.get_mut(&id).unwrap().files[0];
        let action = action::replace_data_file(id, &base.path, &replacement);
        base.path = replacement.path;
        base.file_size_bytes = replacement.file_size_bytes;
        base.base_id = replacement.base_id;
        action
    }

    fn delete_rows(&mut self, id: u64, rng: &mut SmallRng, salt: u32) -> pb::FragmentAction {
        let fragment = self.fragments.get_mut(&id).unwrap();
        let deletion_file = DeletionFile {
            read_version: u64::from(salt),
            id: u64::from(salt),
            file_type: if rng.random_bool(0.5) {
                DeletionFileType::Array
            } else {
                DeletionFileType::Bitmap
            },
            num_deleted_rows: Some(rng.random_range(1..=fragment.physical_rows.unwrap())),
            base_id: None,
        };
        let action = action::add_deletion_file(id, &deletion_file).unwrap();
        fragment.deletion_file = Some(deletion_file);
        action
    }

    fn clear_deletions(&mut self, id: u64) -> pb::FragmentAction {
        self.fragments.get_mut(&id).unwrap().deletion_file = None;
        action::clear_deletion_file(id)
    }

    fn physical_rows(&self) -> u64 {
        self.fragments
            .values()
            .map(|fragment| fragment.physical_rows.unwrap() as u64)
            .sum()
    }

    fn visible_rows(&self) -> u64 {
        self.fragments.values().map(visible_rows).sum()
    }

    fn fragment_at_row_offset(&self, mut offset: u64) -> OffsetResolution {
        for fragment in self.fragments.values() {
            let visible = visible_rows(fragment);
            if offset < visible {
                return OffsetResolution::Found(Box::new(fragment.clone()), offset);
            }
            offset -= visible;
        }
        OffsetResolution::BeyondEnd
    }
}

fn visible_rows(fragment: &Fragment) -> u64 {
    let deleted = fragment
        .deletion_file
        .as_ref()
        .map_or(0, |file| file.num_deleted_rows.unwrap());
    (fragment.physical_rows.unwrap() - deleted) as u64
}

/// Each action sees the model after the commit's earlier actions.
fn random_commit(model: &mut Model, rng: &mut SmallRng, salt: &mut u32) -> Vec<pb::FragmentAction> {
    let mut actions = Vec::new();
    for _ in 0..rng.random_range(1..=6) {
        *salt += 1;
        let ids: Vec<u64> = model.fragments.keys().copied().collect();
        if ids.is_empty() || rng.random_range(0..5) == 0 {
            actions.push(model.append(rng));
            continue;
        }
        let id = ids[rng.random_range(0..ids.len())];
        actions.push(match rng.random_range(0..6) {
            0 => model.remove(id),
            1 => model.backfill(id, *salt),
            2 => model.replace_base_file(id, *salt),
            3 => model.delete_rows(id, rng, *salt),
            4 => model.clear_deletions(id),
            _ => model.upsert(id, rng),
        });
    }
    actions
}

fn random_policy(rng: &mut SmallRng) -> SnapshotPolicy {
    match rng.random_range(0..3) {
        0 => SnapshotPolicy::default(),
        1 => CHECKPOINT_POLICY,
        _ => SUFFIX_POLICY,
    }
}

fn random_context(fixture: &Fixture, leaves: &LanceCache, rng: &mut SmallRng) -> TreeContext {
    if rng.random_bool(0.5) {
        fixture.context().with_leaf_cache(leaves.clone())
    } else {
        fixture.context()
    }
}

async fn check_against_model(
    tree: &FragmentTree,
    model: &Model,
    leaf_reads: &SingleFlightLeafCache,
    rng: &mut SmallRng,
    label: &str,
) {
    if let Err(error) = tree.validate().await {
        panic!("{label}: {error}");
    }
    let expected: Vec<_> = model.fragments.values().cloned().collect();
    assert_eq!(tree.materialize().await.unwrap(), expected, "{label}");
    check_lookups(tree, model, rng, label).await;
    check_streams(tree, model, leaf_reads, rng, label).await;
    assert_eq!(
        (
            tree.count_fragments(),
            tree.count_rows(),
            tree.count_visible_rows(),
            tree.next_fragment_id(),
        ),
        (
            model.fragments.len() as u64,
            model.physical_rows(),
            model.visible_rows(),
            model.next_fragment_id,
        ),
        "{label}"
    );
}

async fn check_lookups(tree: &FragmentTree, model: &Model, rng: &mut SmallRng, label: &str) {
    let upper = model.next_fragment_id + 2;
    let ids: RoaringBitmap = (0..upper)
        .filter(|_| rng.random_bool(0.25))
        .map(|id| u32::try_from(id).unwrap())
        .collect();
    let selected: Vec<_> = model
        .fragments
        .values()
        .filter(|fragment| ids.contains(fragment.id as u32))
        .cloned()
        .collect();
    assert_eq!(
        tree.resolve_fragments(&ids).await.unwrap(),
        selected,
        "{label}"
    );

    let id = rng.random_range(0..upper);
    assert_eq!(
        tree.resolve_fragment(id).await.unwrap(),
        model.fragments.get(&id).cloned(),
        "{label} point read of {id}"
    );

    let offset = rng.random_range(0..=model.visible_rows());
    assert_eq!(
        tree.fragment_at_row_offset(offset).await.unwrap(),
        model.fragment_at_row_offset(offset),
        "{label} row offset {offset}"
    );
}

async fn check_streams(
    tree: &FragmentTree,
    model: &Model,
    leaf_reads: &SingleFlightLeafCache,
    rng: &mut SmallRng,
    label: &str,
) {
    let bound = rng.random_range(0..model.next_fragment_id + 2);
    let scanned: Vec<_> = Arc::new(tree.clone())
        .scan(bound)
        .try_collect()
        .await
        .unwrap();
    let from_bound: Vec<_> = model
        .fragments
        .range(bound..)
        .map(|(_, fragment)| fragment.clone())
        .collect();
    assert_eq!(scanned, from_bound, "{label} scan from {bound}");

    let leaves: Vec<LeafFragments> = Arc::new(tree.clone())
        .leaf_stream(leaf_reads, |_| true)
        .try_collect()
        .await
        .unwrap();
    let flattened: Vec<_> = leaves
        .into_iter()
        .flat_map(|leaf| match leaf {
            LeafFragments::Stored { fragments, .. } => fragments.to_vec(),
            LeafFragments::Replayed(fragments) => fragments,
        })
        .collect();
    let expected: Vec<_> = model.fragments.values().cloned().collect();
    assert_eq!(flattened, expected, "{label} leaf stream");
}

struct Published {
    tree: FragmentTree,
    version: u64,
}

struct Sequence {
    fixture: Fixture,
    shape: Shape,
    leaves: LanceCache,
    leaf_reads: SingleFlightLeafCache,
    /// One pass over every published version, as a version listing makes.
    summaries: LeafSummaries,
    published: Published,
    model: Model,
    salt: u32,
    rng: SmallRng,
}

impl Sequence {
    async fn bootstrap(seed: u64, shape: Shape) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let bootstrapped = shape.bootstrap_fragments();
        let fixture = Fixture::new(bootstrapped, shape.config(), shape.policy(&mut rng)).await;
        assert_eq!(
            fixture.tree.height(),
            shape.bootstrap_height(),
            "{shape:?} seed {seed}"
        );
        let published = Published {
            tree: fixture.tree.clone(),
            version: 1,
        };
        Self {
            fixture,
            shape,
            leaves: LanceCache::with_capacity(16 * 1024 * 1024),
            leaf_reads: single_flight_cache(),
            summaries: LeafSummaries::default(),
            published,
            model: Model {
                fragments: (0..bootstrapped)
                    .map(|id| (id, make_fragment(id)))
                    .collect(),
                next_fragment_id: bootstrapped,
            },
            salt: 0,
            rng,
        }
    }

    async fn step(&mut self) -> &'static str {
        match self.rng.random_range(0..8) {
            0 => {
                self.published.tree = FragmentTree::open_snapshot(
                    random_context(&self.fixture, &self.leaves, &mut self.rng),
                    &self.published.tree.descriptor,
                    self.published.version,
                    self.fixture.config.clone(),
                    self.model.next_fragment_id,
                )
                .await
                .unwrap();
                "reopen"
            }
            1 => {
                self.published.tree = self.published.tree.clone().hydrate().await.unwrap();
                "hydrate"
            }
            2 if self.model.fragments.len() >= 8 => {
                let actions = self.model.contract(&mut self.rng);
                self.commit(actions).await;
                "contraction"
            }
            _ => {
                let actions = random_commit(&mut self.model, &mut self.rng, &mut self.salt);
                self.commit(actions).await;
                "commit"
            }
        }
    }

    async fn commit(&mut self, actions: Vec<pb::FragmentAction>) {
        let ids: Vec<_> = actions.iter().filter_map(action::target_frag_id).collect();
        let retention = if self.rng.random_bool(0.5) {
            InventoryRetention::Retain
        } else {
            InventoryRetention::Discard
        };
        let mut attempt = self
            .published
            .tree
            .begin_commit(self.shape.mode(&mut self.rng))
            .with_inventory_retention(retention);
        attempt.resolve_touched(&ids).await.unwrap();
        let policy = self.shape.policy(&mut self.rng);
        let (prepared, _) = attempt
            .prepare(ValidatedCommit::fragment_actions(actions), policy)
            .await
            .unwrap();
        let published = &mut self.published;
        published.version += 1;
        let context = random_context(&self.fixture, &self.leaves, &mut self.rng);
        published.tree = prepared
            .into_published(published.version, self.fixture.config.clone(), context)
            .unwrap();
    }

    async fn check(&mut self, label: &str) {
        if matches!(self.shape, Shape::RootOverLeaves) {
            assert_eq!(self.published.tree.height(), 1, "{label}");
        }
        check_against_model(
            &self.published.tree,
            &self.model,
            &self.leaf_reads,
            &mut self.rng,
            label,
        )
        .await;
        let expected: Vec<_> = self.model.fragments.values().cloned().collect();
        assert_eq!(
            self.published
                .tree
                .summary(&mut self.summaries)
                .await
                .unwrap(),
            ManifestSummary::of(&expected),
            "{label}"
        );
    }
}

async fn check_operation_sequence(seed: u64, steps: usize, shape: Shape) {
    let mut sequence = Sequence::bootstrap(seed, shape).await;
    sequence
        .check(&format!("{shape:?} seed {seed} bootstrap"))
        .await;
    for step in 0..steps {
        let operation = sequence.step().await;
        let label = format!("{shape:?} seed {seed} step {step} after {operation}");
        sequence.check(&label).await;
    }
}

proptest! {
    // Every case runs both shapes and reads each tree in full after every
    // step, so few cases keep the test near half a second when optimized.
    #![proptest_config(ProptestConfig::with_cases(6))]
    #[test]
    fn random_operation_sequences_match_a_hand_applied_model(seed in any::<u64>(), steps in 1usize..=12) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        for shape in [Shape::Interiors, Shape::RootOverLeaves] {
            runtime.block_on(check_operation_sequence(seed, steps, shape));
        }
    }
}
