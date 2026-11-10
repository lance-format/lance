// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::index::tests::support::*;

use std::sync::Arc;

use arrow::datatypes::Int32Type;
use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{array, gen_batch};
use lance_index::IndexType;
use lance_index::frag_reuse::{FragDigest, FragReuseGroup, FragReuseIndexDetails};
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
use lance_table::format::{DataStorageFormat, DeletionFileType};
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::{WriteMode, WriteParams};
use crate::index::DatasetIndexExt;
use crate::index::frag_reuse_with_stable_row_ids::history::tests::{
    assert_answers_through, catch_up_i_idx, coverage_of, create_zone_map, history,
};
use crate::index::lifecycle::build_index_metadata_from_segments;
use crate::index::loading::load_all_indices;
use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount, compact_into_two};

/// A compaction's commit is the version after its reuse record, and only if
/// that commit is the rewrite that produced the fragments.
#[tokio::test]
async fn test_a_compaction_is_dated_by_its_own_commit_record() {
    let test_dir = TempStrDir::default();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = |values: std::ops::Range<i32>| {
        RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(values))],
        )
        .unwrap()
    };
    let params = WriteParams {
        max_rows_per_file: 2,
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch(0..4))], schema.clone());
    let mut dataset = Dataset::write(reader, &test_dir, Some(params))
        .await
        .unwrap();
    // An index, so the deferred compaction writes a reuse record.
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let sources = all_fragment_ids(&dataset);
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1_000,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let committed = dataset.manifest.version;
    let recorded = dataset
        .open_frag_reuse_index(&NoOpMetricsCollector)
        .await
        .unwrap()
        .expect("a deferred compaction records itself")
        .details
        .versions[0]
        .dataset_version;
    let produced = all_fragment_ids(&dataset);
    let reader = RecordBatchIterator::new(vec![Ok(batch(4..6))], schema);
    dataset.append(reader, None).await.unwrap();

    let history = History::new(&dataset).await.unwrap();
    assert_eq!(
        history
            .rewrite_commit(recorded, produced.iter().copied())
            .await
            .unwrap(),
        Some(committed)
    );
    assert_eq!(
        history
            .rewrite_commit(recorded, sources.iter().copied())
            .await
            .unwrap(),
        None,
        "the commit is a rewrite, but not the one that produced these fragments"
    );
    assert_eq!(
        history
            .rewrite_commit(committed, produced.iter().copied())
            .await
            .unwrap(),
        None,
        "the commit after this version is an append, not a rewrite"
    );
}

/// A segment's carried columns can go stale independently of its keyed column,
/// so the staleness check has to walk every entry of `fields`, not just the
/// keyed subtree. Here the carried column did not exist when the segment was
/// built, so the segment cannot be carrying its values and its coverage of that
/// fragment must be pruned. Walking only the keyed subtree sees no change and
/// leaves the fragment covered, which is the bug.
#[tokio::test]
async fn test_prune_stale_coverage_notices_a_changed_carried_column() {
    use crate::dataset::NewColumnTransform;
    use lance_datagen::{BatchCount, RowCount, array};

    let test_dir = tempfile::tempdir().unwrap();
    let reader = lance_datagen::gen_batch()
        .col("id", array::step::<arrow_array::types::Int32Type>())
        .col(
            "vector",
            array::rand_vec::<arrow_array::types::Float32Type>(8.into()),
        )
        .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    let mut dataset = Dataset::write(reader, test_dir.path().to_str().unwrap(), None)
        .await
        .unwrap();

    let vector_field_id = dataset.schema().field("vector").unwrap().id;
    // Built against the schema as it stands now, before `payload` exists.
    let metadata = write_vector_segment_metadata(
        &dataset,
        "vector_idx",
        vector_field_id,
        Uuid::new_v4(),
        [0_u32],
        b"segment",
    )
    .await;

    // `payload` lands in a new data file on the same fragment, so the fragment's
    // file layout changes for `payload` while `vector`'s file is untouched.
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("payload".into(), "id * 2".into())]),
            None,
            None,
        )
        .await
        .unwrap();
    let payload_field_id = dataset.schema().field("payload").unwrap().id;

    let mut covered = metadata.clone();
    covered.fields = vec![vector_field_id, payload_field_id];
    covered.covering_fields = vec![payload_field_id];

    let new_indices = build_index_metadata_from_segments(
        &dataset,
        "vector_idx",
        vector_field_id,
        vec![segment_from_metadata(&covered)],
    )
    .await
    .unwrap();

    assert_eq!(new_indices.len(), 1);
    assert!(
        new_indices[0].fragment_bitmap.as_ref().unwrap().is_empty(),
        "coverage of a fragment whose carried column changed must be pruned, got {:?}",
        new_indices[0].fragment_bitmap
    );

    // Control: with nothing carried, the same segment over the same fragment
    // keeps its coverage -- so the assertion above is about the carried column,
    // not about `add_columns` invalidating everything.
    let plain_indices = build_index_metadata_from_segments(
        &dataset,
        "vector_idx",
        vector_field_id,
        vec![segment_from_metadata(&metadata)],
    )
    .await
    .unwrap();
    assert!(
        !plain_indices[0]
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .is_empty(),
        "a non-covering segment must keep its coverage across the same add_columns"
    );
}

/// An index old enough to predate the fragment bitmap has its coverage
/// reconstructed from the version it was written against, so it arrives in
/// that version's fragment-id space. `load_all_indices` moves a *stored*
/// bitmap into the current space through the fragment-reuse index and leaves
/// a `None` one alone, so the reconstruction has to make that move itself.
///
/// Without it, coverage still names the fragments the rows moved out of:
/// deferred planning loses the boundary between indexed and unindexed rows,
/// and the unreadable-index commit guard cannot detect a covered rewrite.
#[tokio::test]
async fn test_reconstructed_coverage_follows_a_deferred_compaction() {
    use crate::dataset::index::DatasetIndexRemapperOptions;
    use crate::dataset::optimize::{CompactionPlan, TaskData, commit_compaction};

    let test_dir = tempfile::tempdir().unwrap();
    let test_uri = test_dir.path().to_str().unwrap();

    let write_params = WriteParams {
        enable_stable_row_ids: false,
        max_rows_per_file: 5,
        ..Default::default()
    };
    let mut dataset = Dataset::write(two_column_reader(), test_uri, Some(write_params.clone()))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);

    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    dataset
        .create_index_builder(&["id"], IndexType::BTree, &btree_params)
        .name("id_idx".to_string())
        .train(false)
        .await
        .unwrap();
    hide_index_from_this_build(&mut dataset, "id_idx").await;

    // Drop the bitmap, which is what an index written before it existed
    // looks like: coverage has to be reconstructed from `dataset_version`.
    let hidden = manifest_index(&dataset, "id_idx").await;
    let legacy = IndexMetadata {
        fragment_bitmap: None,
        ..hidden.clone()
    };
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![legacy],
            removed_indices: vec![hidden],
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();

    // Deferred remap is the one compaction a build that cannot read the
    // index may run: it hands the repair on through a fragment-reuse index.
    // Fragments 0 and 1 become fragment 2, and the index still covers those
    // rows - now under a different id.
    compact_files(
        &mut dataset,
        CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let after_defer = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect::<Vec<_>>();
    assert_eq!(after_defer, vec![2]);

    // The commit-boundary half, taken first: the planner half below rewrites
    // fragment 2 when the remap is missing, which would leave this nothing
    // to ask about and hide whether the guard is sensitive on its own.
    let plan = CompactionPlan {
        tasks: vec![TaskData {
            fragments: dataset.fragments().as_ref().clone(),
            output_clustering_id: None,
            output_metadata: Default::default(),
        }],
        read_version: dataset.version().version,
        options: CompactionOptions::default(),
    };
    let mut rewrites = Vec::new();
    for task in plan.compaction_tasks() {
        rewrites.push(task.execute(&dataset).await.unwrap());
    }
    let err = commit_compaction(
        &mut dataset,
        rewrites,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &plan.options,
    )
    .await
    .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("id_idx") && message.contains("[2]"),
        "the refusal has to name the index and the fragment in current ids: {message}"
    );

    // Two fragments the index never covered, so the planner half has
    // something to compact and cannot pass by finding nothing to do.
    let mut dataset = Dataset::write(
        two_column_reader(),
        test_uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..write_params
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 3);

    // Eager planning refuses pending FRI moves before considering coverage.
    let version = dataset.version().version;
    let err = compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotSupported { .. }), "{err}");
    assert!(err.to_string().contains("pending remaps"), "{err}");
    dataset.checkout_latest().await.unwrap();
    assert_eq!(dataset.version().version, version);
    assert_eq!(dataset.get_fragments().len(), 3);

    // Deferred planning still honors the reconstructed boundary: the
    // appended pair coalesces and the indexed fragment 2 is left alone.
    let metrics = compact_files(
        &mut dataset,
        CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(metrics.fragments_removed, 2);
    assert_eq!(metrics.fragments_added, 1);
    assert!(
        dataset
            .get_fragments()
            .iter()
            .any(|fragment| fragment.id() == 2),
        "the fragment the reconstructed coverage maps to was rewritten"
    );
}

/// `fragments` fragments of 10 rows of `i`, with a BTree index on it so
/// that deferred compactions record their groups. Returns the table and
/// the fragments the index was built over.
async fn recording_btree_table(uri: &str, fragments: u32) -> (Dataset, RoaringBitmap) {
    let mut dataset = gen_batch()
        .col("i", array::step::<Int32Type>())
        .into_dataset_with_params(
            uri,
            FragmentCount::from(fragments),
            FragmentRowCount::from(10),
            Some(WriteParams {
                max_rows_per_file: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("i_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let built = dataset
        .manifest
        .fragments
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    (dataset, built)
}

/// Twelve compactions each rewrite both fragments into two, so both
/// outputs come from both inputs and a current fragment reaches every
/// earlier one along many paths. Traced back to the index's build, each
/// fragment is followed once: the work stays within the fragments the
/// history names, and the build's two fragments come back once each.
#[tokio::test]
async fn test_sources_at_build_follows_each_shared_ancestor_once() {
    let dir = TempStrDir::default();
    let (mut dataset, built) = recording_btree_table(dir.as_str(), 2).await;
    let version = dataset.manifest.version;
    for _ in 0..12 {
        let current = dataset
            .manifest
            .fragments
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>();
        compact_into_two(&mut dataset, &current).await;
    }
    let history = dataset.frag_reuse_index().await.unwrap().unwrap();
    let versions = &history.details.versions;
    let named = versions
        .iter()
        .flat_map(|reuse| reuse.old_frag_ids().into_iter().chain(reuse.new_frag_ids()))
        .map(|id| id as u32)
        .collect::<RoaringBitmap>();
    assert_eq!((versions.len(), named.len()), (12, 26));

    for fragment in dataset.manifest.fragments.iter() {
        let visits = std::cell::Cell::new(0);
        let mut sources = sources_at_build(versions, version, fragment.id as u32, |id| {
            visits.set(visits.get() + 1);
            built.contains(id)
        })
        .unwrap();
        assert!(
            visits.get() <= named.len(),
            "tracing fragment {} took {} visits and returned {} sources",
            fragment.id,
            visits.get(),
            sources.len()
        );
        sources.sort_unstable();
        assert_eq!(sources, [0, 1]);
    }
}

/// Compactions rewrite fragments 0 and 1 into 4 and 5, 2 and 3 into 6 and
/// 7, then 5 and 6 into 8 and 9. The index is caught up past the first
/// one, so a cleanup trims its record: 4 and 5 no longer trace to the
/// build, and neither do 8 and 9, though their other source, 6, does, as 7
/// still does.
#[tokio::test]
async fn test_sources_at_build_fails_on_a_missing_link_beside_a_traced_one() {
    let dir = TempStrDir::default();
    let (mut dataset, built) = recording_btree_table(dir.as_str(), 4).await;
    let version = dataset.manifest.version;
    let first = compact_into_two(&mut dataset, &[0, 1]).await;
    crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["i"],
        Some("i_idx".into()),
    )
    .await
    .unwrap();
    let second = compact_into_two(&mut dataset, &[2, 3]).await;
    let third = compact_into_two(&mut dataset, &[first[1], second[0]]).await;
    let trace = |versions: &[FragReuseVersion], fragment: u64| {
        sources_at_build(versions, version, fragment as u32, |id| built.contains(id)).map(
            |mut sources| {
                sources.sort_unstable();
                sources
            },
        )
    };
    let recorded = dataset.frag_reuse_index().await.unwrap().unwrap();
    assert_eq!(
        trace(&recorded.details.versions, third[0]),
        Some(vec![0, 1, 2, 3])
    );

    crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut dataset)
        .await
        .unwrap();
    let kept = dataset.frag_reuse_index().await.unwrap().unwrap();
    assert_eq!(kept.details.versions.len(), 2);
    for fragment in [first[0], third[0], third[1]] {
        assert_eq!(
            trace(&kept.details.versions, fragment),
            None,
            "fragment {fragment}"
        );
    }
    assert_eq!(trace(&kept.details.versions, second[1]), Some(vec![2, 3]));
}

/// Twelve recorded compactions each rewrite both fragments into two, so
/// both outputs come from both inputs. Traced back to the zone map's
/// build, each fragment is followed once and the build's two fragments
/// come back once each; caught up past every compaction, the zone map
/// covers both current fragments and answers through them.
#[tokio::test]
async fn test_tracing_a_zone_map_follows_each_shared_ancestor_once() {
    let dir = TempStrDir::default();
    let mut dataset = gen_batch()
        .col("i", array::step::<Int32Type>())
        .into_dataset_with_params(
            dir.as_str(),
            FragmentCount::from(2),
            FragmentRowCount::from(50),
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 50,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    create_zone_map(&mut dataset).await;
    let version = dataset.manifest.version;
    let built = RoaringBitmap::from_iter([0, 1]);
    for _ in 0..12 {
        let current = dataset
            .manifest
            .fragments
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>();
        compact_into_two(&mut dataset, &current).await;
    }
    let versions = history(&dataset).await;
    let named = versions
        .iter()
        .flat_map(|reuse| reuse.old_frag_ids().into_iter().chain(reuse.new_frag_ids()))
        .map(|id| id as u32)
        .collect::<RoaringBitmap>();
    assert_eq!((versions.len(), named.len()), (12, 26));

    let current = dataset
        .manifest
        .fragments
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect::<RoaringBitmap>();
    for fragment in current.iter() {
        let visits = std::cell::Cell::new(0);
        let mut sources = sources_at_build(&versions, version, fragment, |id| {
            visits.set(visits.get() + 1);
            built.contains(id)
        })
        .unwrap();
        assert!(
            visits.get() <= named.len(),
            "tracing fragment {fragment} took {} visits and returned {} sources",
            visits.get(),
            sources.len()
        );
        sources.sort_unstable();
        assert_eq!(sources, [0, 1]);
    }
    catch_up_i_idx(&mut dataset).await;
    assert_eq!(
        coverage_of(&load_all_indices(&dataset).await.unwrap(), "i_idx"),
        current
    );
    assert_answers_through(&dataset, "i_idx", "i < 50").await;
}

/// A table's history held in memory: a manifest per retained version, the
/// fragments each rewrite commit wrote, and each deletion file's rows.
#[derive(Default)]
struct MemoryHistory {
    manifests: BTreeMap<u64, Manifest>,
    rewrites: HashMap<u64, HashSet<u32>>,
    deletions: HashMap<(u64, u64), RoaringBitmap>,
}

#[derive(Clone)]
struct MemoryTable {
    history: Arc<MemoryHistory>,
    version: u64,
}

impl TableVersion for MemoryTable {
    fn manifest(&self) -> &Manifest {
        &self.history.manifests[&self.version]
    }

    async fn retained_versions(&self) -> Result<Vec<u64>> {
        Ok(self.history.manifests.keys().copied().collect())
    }

    async fn checkout(&self, version: u64) -> Result<Self> {
        Ok(Self {
            history: self.history.clone(),
            version,
        })
    }

    async fn rewrite_outputs(&self, version: u64) -> Result<Option<HashSet<u32>>> {
        Ok(self.history.rewrites.get(&version).cloned())
    }

    async fn deleted_rows(
        &self,
        fragment: &Fragment,
        deletion_file: &DeletionFile,
    ) -> Result<Arc<DeletionVector>> {
        let rows = self.history.deletions[&(fragment.id, deletion_file.id)].clone();
        Ok(Arc::new(DeletionVector::Bitmap(rows)))
    }
}

/// A row as the model tracks it: `token` follows the row wherever compaction
/// moves it.
#[derive(Clone, Copy, Debug)]
struct ModelRow {
    token: u32,
    val: u32,
}

#[derive(Clone, Debug)]
struct ModelOverlay {
    committed_version: u64,
    path: String,
    offset: u32,
    val: u32,
}

#[derive(Clone, Debug)]
struct ModelFragment {
    id: u64,
    rows: Vec<ModelRow>,
    val_file: String,
    spare_file: String,
    deleted: RoaringBitmap,
    /// The deletion file's (id, read version).
    deletion_file: Option<(u64, u64)>,
    overlays: Vec<ModelOverlay>,
}

impl ModelFragment {
    /// Live rows with their values as a reader sees them, through overlays.
    fn live_rows(&self) -> impl Iterator<Item = ModelRow> + '_ {
        self.rows
            .iter()
            .enumerate()
            .filter(|(offset, _)| !self.deleted.contains(*offset as u32))
            .map(|(offset, row)| {
                let val = self
                    .overlays
                    .iter()
                    .rfind(|overlay| overlay.offset == offset as u32)
                    .map_or(row.val, |overlay| overlay.val);
                ModelRow { val, ..*row }
            })
    }

    fn digest(&self) -> FragDigest {
        FragDigest {
            id: self.id,
            physical_rows: self.rows.len(),
            num_deleted_rows: self.deleted.len() as usize,
        }
    }

    fn to_fragment(&self, schema: &LanceSchema) -> Fragment {
        let val = schema.field("val").unwrap().id;
        let spare = schema.field("spare").unwrap().id;
        let mut fragment = Fragment::new(self.id).with_physical_rows(self.rows.len());
        fragment.files = vec![
            DataFile::new_legacy_from_fields(&self.val_file, vec![val], None),
            DataFile::new_legacy_from_fields(&self.spare_file, vec![spare], None),
        ];
        fragment.deletion_file = self.deletion_file.map(|(id, read_version)| DeletionFile {
            read_version,
            id,
            file_type: DeletionFileType::Bitmap,
            num_deleted_rows: Some(self.deleted.len() as usize),
            base_id: None,
            inline: None,
        });
        fragment.overlays = self
            .overlays
            .iter()
            .map(|overlay| DataOverlayFile {
                data_file: DataFile::new_legacy_from_fields(&overlay.path, vec![val], None),
                coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([overlay.offset])),
                committed_version: overlay.committed_version,
            })
            .collect();
        fragment
    }
}

#[derive(Clone, Default)]
struct ModelVersion {
    fragments: Vec<ModelFragment>,
    reuse: Vec<FragReuseVersion>,
}

#[derive(Clone, Copy, Debug)]
enum ModelOp {
    Append,
    Delete,
    UpdateVal,
    UpdateSpare,
    Overlay,
    /// A deferred compaction, recorded in the fragment reuse index.
    Compact,
    /// The remap job dropping the oldest reuse record.
    Trim,
    Restore,
    CleanUp,
    Config,
}

/// A table with `val` and `spare` columns, its history kept as both the
/// manifests the coverage check reads and the rows each version holds.
struct ModelTable {
    rng: u64,
    schema: LanceSchema,
    versions: BTreeMap<u64, ModelVersion>,
    history: MemoryHistory,
    version: u64,
    next_fragment: u64,
    next_token: u32,
    next_val: u32,
    next_file: u64,
    ops: Vec<ModelOp>,
}

impl ModelTable {
    fn new(seed: u64, fragments: usize) -> Self {
        let schema = LanceSchema::try_from(&Schema::new(vec![
            Field::new("val", DataType::Int32, true),
            Field::new("spare", DataType::Int32, true),
        ]))
        .unwrap();
        let mut table = Self {
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            schema,
            versions: BTreeMap::new(),
            history: MemoryHistory::default(),
            version: 0,
            next_fragment: 0,
            next_token: 0,
            next_val: 0,
            next_file: 0,
            ops: Vec::new(),
        };
        let fragments = (0..fragments).map(|_| table.new_fragment(3)).collect();
        table.commit(
            ModelVersion {
                fragments,
                reuse: Vec::new(),
            },
            None,
        );
        table
    }

    fn next(&mut self, bound: usize) -> usize {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng % bound as u64) as usize
    }

    fn file(&mut self) -> String {
        self.next_file += 1;
        format!("{}.lance", self.next_file)
    }

    fn new_val(&mut self) -> u32 {
        self.next_val += 1;
        self.next_val
    }

    fn new_fragment(&mut self, rows: usize) -> ModelFragment {
        let rows = (0..rows)
            .map(|_| {
                self.next_token += 1;
                ModelRow {
                    token: self.next_token,
                    val: self.new_val(),
                }
            })
            .collect();
        self.fragment_of(rows)
    }

    fn fragment_of(&mut self, rows: Vec<ModelRow>) -> ModelFragment {
        self.next_fragment += 1;
        ModelFragment {
            id: self.next_fragment - 1,
            rows,
            val_file: self.file(),
            spare_file: self.file(),
            deleted: RoaringBitmap::new(),
            deletion_file: None,
            overlays: Vec::new(),
        }
    }

    fn current(&self) -> &ModelVersion {
        &self.versions[&self.version]
    }

    fn commit(&mut self, state: ModelVersion, rewrite: Option<HashSet<u32>>) {
        self.version += 1;
        let fragments = state
            .fragments
            .iter()
            .map(|fragment| fragment.to_fragment(&self.schema))
            .collect();
        let mut manifest = Manifest::new(
            self.schema.clone(),
            Arc::new(fragments),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        manifest.version = self.version;
        self.history.manifests.insert(self.version, manifest);
        if let Some(rewrite) = rewrite {
            self.history.rewrites.insert(self.version, rewrite);
        }
        self.versions.insert(self.version, state);
    }

    fn run(&mut self, len: usize) {
        use ModelOp::*;
        const KINDS: [ModelOp; 12] = [
            Append,
            Delete,
            UpdateVal,
            UpdateSpare,
            Overlay,
            Compact,
            Compact,
            Compact,
            Trim,
            Restore,
            CleanUp,
            Config,
        ];
        for _ in 0..len {
            let op = KINDS[self.next(KINDS.len())];
            self.ops.push(op);
            self.apply(op);
        }
    }

    fn apply(&mut self, op: ModelOp) {
        let mut state = self.current().clone();
        let picked = self.next(state.fragments.len().max(1));
        let mut rewrite = None;
        match op {
            ModelOp::Append => {
                let fragment = self.new_fragment(3);
                state.fragments.push(fragment);
            }
            ModelOp::Delete => {
                let Some(fragment) = state.fragments.get_mut(picked) else {
                    return;
                };
                let offset = self.next(fragment.rows.len()) as u32;
                fragment.deleted.insert(offset);
                self.next_file += 1;
                fragment.deletion_file = Some((self.next_file, self.version));
                self.history
                    .deletions
                    .insert((fragment.id, self.next_file), fragment.deleted.clone());
            }
            ModelOp::UpdateVal => {
                let Some(fragment) = state.fragments.get(picked) else {
                    return;
                };
                // Rewriting the column in place folds the overlays on it.
                let mut rows = fragment.rows.clone();
                let mut live = fragment.live_rows().collect::<Vec<_>>().into_iter();
                for (offset, row) in rows.iter_mut().enumerate() {
                    if !fragment.deleted.contains(offset as u32) {
                        *row = live.next().unwrap();
                    }
                }
                let offset = self.next(rows.len());
                rows[offset].val = self.new_val();
                let val_file = self.file();
                let fragment = &mut state.fragments[picked];
                fragment.rows = rows;
                fragment.val_file = val_file;
                fragment.overlays.clear();
            }
            ModelOp::UpdateSpare => {
                let spare_file = self.file();
                let Some(fragment) = state.fragments.get_mut(picked) else {
                    return;
                };
                fragment.spare_file = spare_file;
            }
            ModelOp::Overlay => {
                let path = self.file();
                let val = self.new_val();
                let committed_version = self.version + 1;
                let Some(fragment) = state.fragments.get_mut(picked) else {
                    return;
                };
                let offset = self.next(fragment.rows.len()) as u32;
                fragment.overlays.push(ModelOverlay {
                    committed_version,
                    path,
                    offset,
                    val,
                });
            }
            ModelOp::Compact => {
                let len = 2 + self.next(2);
                if state.fragments.len() < len {
                    return;
                }
                let start = self.next(state.fragments.len() - len + 1);
                let old = state
                    .fragments
                    .drain(start..start + len)
                    .collect::<Vec<_>>();
                let rows = old
                    .iter()
                    .flat_map(ModelFragment::live_rows)
                    .collect::<Vec<_>>();
                let outputs = 1 + self.next(2);
                let new = rows
                    .chunks(rows.len().div_ceil(outputs).max(1))
                    .map(|rows| self.fragment_of(rows.to_vec()))
                    .collect::<Vec<_>>();
                state.reuse.push(FragReuseVersion {
                    dataset_version: self.version,
                    groups: vec![FragReuseGroup {
                        changed_row_addrs: Vec::new(),
                        old_frags: old.iter().map(ModelFragment::digest).collect(),
                        new_frags: new.iter().map(ModelFragment::digest).collect(),
                    }],
                });
                rewrite = Some(new.iter().map(|fragment| fragment.id as u32).collect());
                state.fragments.splice(start..start, new);
            }
            ModelOp::Trim => {
                if state.reuse.is_empty() {
                    return;
                }
                state.reuse.remove(0);
            }
            ModelOp::Restore => {
                let retained = self.versions.keys().copied().collect::<Vec<_>>();
                let version = retained[self.next(retained.len())];
                state = self.versions[&version].clone();
            }
            ModelOp::CleanUp => {
                let retained = self.versions.keys().copied().collect::<Vec<_>>();
                let version = retained[self.next(retained.len())];
                if version != self.version {
                    self.versions.remove(&version);
                    self.history.manifests.remove(&version);
                    self.history.rewrites.remove(&version);
                }
                return;
            }
            ModelOp::Config => {}
        }
        self.commit(state, rewrite);
    }

    /// A `val` segment over `fragments` built at the current version, and the
    /// values it indexed for each row.
    fn stage(&self, fragments: &RoaringBitmap) -> (IndexMetadata, HashMap<u32, u32>) {
        let indexed = self
            .current()
            .fragments
            .iter()
            .filter(|fragment| fragments.contains(fragment.id as u32))
            .flat_map(ModelFragment::live_rows)
            .map(|row| (row.token, row.val))
            .collect();
        let segment = IndexMetadata {
            uuid: Uuid::new_v4(),
            name: "val_idx".to_string(),
            fields: vec![self.schema.field("val").unwrap().id],
            covering_fields: vec![],
            dataset_version: self.version,
            fragment_bitmap: Some(fragments.clone()),
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: Default::default(),
        };
        (segment, indexed)
    }
}

/// Stages `val` segments at two versions while a seeded workload runs, then
/// carries their coverage to the latest version. Every row of a fragment that
/// stays covered must be indexed, with the value it holds now. Returns how many
/// fragments stayed covered.
async fn carry_under_workload(seed: u64) -> u64 {
    let mut table = ModelTable::new(seed, 6);
    let len = table.next(3);
    table.run(len);
    let live = table
        .current()
        .fragments
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect::<Vec<_>>();
    let first = live
        .into_iter()
        .filter(|_| table.next(3) != 0)
        .collect::<RoaringBitmap>();
    let (first, mut indexed) = table.stage(&first);
    let len = table.next(4);
    table.run(len);
    // Include outputs reached from the first segment through compaction. Keep
    // the indexed row sets disjoint so the oracle preserves each build's values.
    let second = table
        .current()
        .fragments
        .iter()
        .filter(|fragment| {
            fragment
                .live_rows()
                .all(|row| !indexed.contains_key(&row.token))
        })
        .map(|fragment| fragment.id as u32)
        .collect::<RoaringBitmap>();
    let (second, more) = table.stage(&second);
    indexed.extend(more);
    let len = table.next(6);
    table.run(len);

    let segments = [first, second];
    let staged_coverage = segments
        .iter()
        .filter_map(|segment| segment.fragment_bitmap.clone())
        .fold(RoaringBitmap::new(), |coverage, bitmap| coverage | bitmap);
    let frag_reuse_index = CompactFragReuseIndex::from_row_id_maps(
        Uuid::nil(),
        vec![],
        FragReuseIndexDetails {
            versions: table.current().reuse.clone(),
        },
    );
    let version = table.version;
    let fragments = table.current().fragments.clone();
    let ops = std::mem::take(&mut table.ops);
    let latest = MemoryTable {
        history: Arc::new(table.history),
        version,
    };
    let carried = carry_coverage(
        &latest,
        "val_idx",
        &frag_reuse_index,
        &segments,
        &staged_coverage,
    )
    .await
    .unwrap();

    for fragment in fragments
        .iter()
        .filter(|fragment| carried.contains(fragment.id as u32))
    {
        for row in fragment.live_rows() {
            assert_eq!(
                indexed.get(&row.token),
                Some(&row.val),
                "seed {seed} {ops:?}: fragment {} stayed covered, but its row {} changed",
                fragment.id,
                row.token,
            );
        }
    }
    carried.len()
}

/// Many short workloads of writes, overlays, restores, cleanups and recorded
/// compactions: coverage carried to the latest version only stays on
/// fragments whose rows the segments indexed with the values they hold now.
#[tokio::test]
async fn test_carried_coverage_holds_only_unchanged_rows() {
    let mut covered = 0;
    for seed in 0..1000 {
        covered += carry_under_workload(seed).await;
    }
    assert!(covered >= 1000, "only {covered} fragments stayed covered");
}
