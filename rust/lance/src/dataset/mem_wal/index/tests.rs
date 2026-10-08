// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use super::plugin::{FlushContext, FlushOutcome, MemIndex, MemIndexBuildContext, MemIndexPlugin};
use super::query::{MemMatches, MemQuery, SearchContext};
use lance_index::IndexType;
use lance_index::pbold;
use lance_index::scalar::InvertedIndexParams;
use lance_index::scalar::SargableQuery;
use lance_index::scalar::expression::{SargableQueryParser, ScalarQueryParser};
use lance_index::scalar::inverted::DocumentGranularity;
use lance_index::scalar::registry::{TrainingCriteria, TrainingOrdering};
use lance_table::format::IndexMetadata;
use prost::Message as _;
use std::collections::BTreeMap;
use std::sync::RwLock as StdRwLock;

use super::test_plugin::{Deviation, wrapped};
use super::*;
use crate::dataset::mem_wal::memtable::batch_store::StoredBatch;
use arrow_array::{Int32Array, StringArray};
use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema};
use datafusion::common::ScalarValue;
use lance_index::scalar::inverted::InvertedListFormatVersion;
use rstest::rstest;
use std::sync::Arc;
use uuid::Uuid;

impl IndexStore {
    /// The index named `name`, if it is a `T`.
    pub(crate) fn typed<T: 'static>(&self, name: &str) -> Option<&T> {
        (self.indexes.get(name)?.as_ref() as &dyn std::any::Any).downcast_ref::<T>()
    }

    pub(crate) fn get_btree(&self, name: &str) -> Option<&BTreeMemIndex> {
        self.typed(name)
    }

    pub(crate) fn get_hnsw(&self, name: &str) -> Option<&HnswMemIndex> {
        self.typed(name)
    }

    pub(crate) fn get_fts(&self, name: &str) -> Option<&FtsMemIndex> {
        self.typed(name)
    }
}

/// A value-to-positions index, for exercising the plugin paths with a kind
/// Lance does not build in.
#[derive(Debug, Default)]
struct StubMemIndex {
    columns: Vec<String>,
    postings: StdRwLock<BTreeMap<String, Vec<RowPosition>>>,
}

impl StubMemIndex {
    fn new(column: String) -> Self {
        Self {
            columns: vec![column],
            postings: StdRwLock::new(BTreeMap::new()),
        }
    }

    fn key(array: &dyn arrow_array::Array, row: usize) -> Option<String> {
        (!array.is_null(row))
            .then(|| ScalarValue::try_from_array(array, row).ok())
            .flatten()
            .map(|value| value.to_string())
    }
}

#[async_trait::async_trait]
impl MemIndex for StubMemIndex {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn can_answer(&self, query: &dyn MemQuery) -> bool {
        matches!(
            query.as_any().downcast_ref::<SargableQuery>(),
            Some(SargableQuery::Equals(_) | SargableQuery::IsIn(_))
        )
    }

    fn insert(&self, batch: &RecordBatch, row_offset: RowPosition) -> Result<()> {
        let array = &batch[self.columns[0].as_str()];
        let mut postings = self.postings.write().unwrap();
        for row in 0..array.len() {
            if let Some(key) = Self::key(array.as_ref(), row) {
                postings
                    .entry(key)
                    .or_default()
                    .push(row_offset + row as u64);
            }
        }
        Ok(())
    }

    fn resident_bytes(&self) -> usize {
        let postings = self.postings.read().unwrap();
        postings
            .iter()
            .map(|(key, positions)| {
                key.len() + positions.len() * std::mem::size_of::<RowPosition>()
            })
            .sum()
    }

    fn search(&self, query: &dyn MemQuery, ctx: &SearchContext) -> Result<Option<MemMatches>> {
        let Some(query) = query.as_any().downcast_ref::<SargableQuery>() else {
            return Ok(None);
        };
        let postings = self.postings.read().unwrap();
        let hits: Vec<RowPosition> = match query {
            SargableQuery::Equals(value) => postings
                .get(&value.to_string())
                .cloned()
                .unwrap_or_default(),
            SargableQuery::IsIn(values) => values
                .iter()
                .filter_map(|value| postings.get(&value.to_string()))
                .flatten()
                .copied()
                .collect(),
            _ => return Ok(None),
        };
        Ok(Some(MemMatches::exact(
            hits.into_iter().filter(|p| *p <= ctx.max_visible),
        )))
    }

    async fn flush(&self, _ctx: &FlushContext<'_>) -> Result<FlushOutcome> {
        Ok(FlushOutcome::BuildFromGeneration)
    }
}

/// Claims base-table indexes whose details message is `.0`.
#[derive(Debug)]
struct StubPlugin(&'static str);

#[async_trait::async_trait]
impl MemIndexPlugin for StubPlugin {
    fn name(&self) -> &str {
        "Stub"
    }
    fn details_message(&self) -> &str {
        self.0
    }
    fn flush_index_type(&self) -> IndexType {
        IndexType::ZoneMap
    }
    fn training_criteria(&self) -> TrainingCriteria {
        TrainingCriteria::new(TrainingOrdering::Values).with_row_id()
    }
    fn query_parser(
        &self,
        index_name: String,
        _index_details: Option<&prost_types::Any>,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        Some(Box::new(SargableQueryParser::new(
            index_name,
            "Stub".to_string(),
            false,
        )))
    }
    fn validate(&self, ctx: &MemIndexBuildContext<'_>) -> Result<()> {
        ctx.single_column().map(|_| ())
    }
    fn create(&self, ctx: &MemIndexBuildContext<'_>) -> Result<Arc<dyn MemIndex>> {
        let (column, _) = ctx.single_column()?;
        Ok(Arc::new(StubMemIndex::new(column.to_string())))
    }
}

fn add_stub(store: &mut IndexStore, name: &str, column: &str) {
    let spec = MemIndexSpec::for_plugin(name, 0, column, Arc::new(StubPlugin("StubIndexDetails")));
    store.add_index(
        name.to_string(),
        spec.build(&LanceSchema::default(), 1_000, 16).unwrap(),
    );
}

/// A type url resolves by its message name, whatever package or host
/// prefix it carries.
#[rstest]
#[case::btree("/lance.table.BTreeIndexDetails", Some("BTree"))]
#[case::fts("/lance.table.InvertedIndexDetails", Some("Inverted"))]
#[case::fts_legacy("/lance.index.pb.InvertedIndexDetails", Some("Inverted"))]
#[case::vector("/lance.index.pb.VectorIndexDetails", Some("Hnsw"))]
// Written by older MemWAL flushes.
#[case::vector_legacy_flush("type.googleapis.com/lance.index.VectorIndexDetails", Some("Hnsw"))]
// Kinds with no built-in memtable plugin.
#[case::bitmap("/lance.table.BitmapIndexDetails", None)]
#[case::label_list("/lance.table.LabelListIndexDetails", None)]
#[case::ngram("/lance.table.NGramIndexDetails", None)]
#[case::zone_map("/lance.table.ZoneMapIndexDetails", None)]
#[case::bloom_filter("/lance.index.pb.BloomFilterIndexDetails", None)]
#[case::json("/lance.index.pb.JsonIndexDetails", None)]
#[case::fm("/lance.index.pb.FMIndexDetails", None)]
#[case::absent("", None)]
fn type_urls_resolve_to_the_plugin_that_maintains_them(
    #[case] type_url: &str,
    #[case] expected: Option<&str>,
) {
    let registry = MemIndexRegistry::default();
    let found = registry
        .plugin_for_details_url(type_url)
        .map(|plugin| plugin.name());
    assert_eq!(found, expected);
}

/// A type url reaches the plugin whose message is exactly its own, and a
/// plugin must claim a bare message name.
#[test]
fn a_type_url_reaches_the_plugin_for_exactly_its_message() {
    let registry = MemIndexRegistry::default()
        .with_plugin(Arc::new(StubPlugin("MyBTreeIndexDetails")))
        .unwrap();
    let found = |url: &str| registry.plugin_for_details_url(url).map(|p| p.name());
    assert_eq!(found("/acme.MyBTreeIndexDetails"), Some("Stub"));
    assert_eq!(found("/lance.table.BTreeIndexDetails"), Some("BTree"));
    assert_eq!(found("/lance.table.IndexDetails"), None);

    for bad in ["", "acme.MyIndexDetails", "acme/MyIndexDetails"] {
        assert!(
            MemIndexRegistry::empty()
                .with_plugin(Arc::new(StubPlugin(bad)))
                .is_err(),
            "{bad:?} is not a bare message name"
        );
    }
}

/// Adding a second plugin for a kind is refused.
#[test]
fn a_registry_refuses_two_plugins_for_one_kind() {
    let mut registry = MemIndexRegistry::default();
    let error = registry
        .add_plugin(Arc::new(BTreeMemIndexPlugin))
        .expect_err("BTreeIndexDetails is already claimed");
    assert!(
        error.to_string().contains("BTreeIndexDetails"),
        "the error must name the contested kind: {error}"
    );
}

/// Replacing a built-in swaps it rather than adding a second claimant, and the
/// replacement must claim a bare message name.
#[test]
fn a_registry_replaces_a_builtin_on_request() {
    let mut registry = MemIndexRegistry::default();
    registry
        .replace_plugin(Arc::new(StubPlugin("BTreeIndexDetails")))
        .unwrap();
    assert_eq!(
        registry
            .plugin_for_details_url("/lance.table.BTreeIndexDetails")
            .map(|plugin| plugin.name()),
        Some("Stub")
    );
    assert_eq!(
        registry.plugins().len(),
        MemIndexRegistry::default().plugins().len()
    );
    assert!(
        registry
            .replace_plugin(Arc::new(StubPlugin("lance.BTreeIndexDetails")))
            .is_err()
    );
}

/// The shard schema these tests build their memtables against.
fn test_lance_schema() -> LanceSchema {
    LanceSchema::try_from(create_test_schema().as_ref()).unwrap()
}

/// A query finds the index on its column that answers it.
#[test]
fn an_index_is_found_by_the_question_not_by_its_type() {
    let arrow = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("description", DataType::Utf8, true),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        ),
    ]));
    let lance = LanceSchema::try_from(arrow.as_ref()).unwrap();
    let specs = vec![
        MemIndexSpec::btree("id_idx", 0, "id"),
        MemIndexSpec::fts("text_idx", 1, "description"),
        MemIndexSpec::hnsw("vec_idx", 2, "vector", DistanceType::L2),
    ];
    let store = IndexStore::from_specs(&specs, &lance, 1_000, 16).unwrap();

    let equality = SargableQuery::Equals(ScalarValue::Int32(Some(1)));
    assert!(
        store.index_answering("id", &equality).is_some(),
        "the B-tree answers equality on its column"
    );
    assert!(
        store.index_answering("description", &equality).is_none(),
        "a full-text index does not answer equality"
    );

    let knn = |distance_type| VectorMemQuery {
        vector: arrow_array::FixedSizeListArray::new_null(
            Arc::new(Field::new("item", DataType::Float32, true)),
            1,
            1,
        ),
        k: 10,
        ef: None,
        distance_type: Some(distance_type),
    };
    assert!(
        store
            .index_answering("vector", &knn(DistanceType::L2))
            .is_some()
    );
    assert!(
        store
            .index_answering("vector", &knn(DistanceType::Cosine))
            .is_none(),
        "a graph built for L2 cannot answer in another metric"
    );

    let fts = FtsMemQuery::probe(DocumentGranularity::Row);
    assert!(store.index_answering("description", &fts).is_some());
    assert!(
        store
            .index_answering(
                "description",
                &FtsMemQuery::probe(DocumentGranularity::ListElement)
            )
            .is_none(),
        "a whole-row index cannot say which list element matched"
    );
    assert!(
        store.index_answering("id", &fts).is_none(),
        "a B-tree does not answer a text query"
    );
}

/// A B-tree hands the flush its sorted rows; a full-text index, which writes
/// its own file, has none to give.
#[tokio::test]
async fn an_index_that_writes_its_own_file_offers_no_training_data() {
    let arrow = create_test_schema();
    let lance = LanceSchema::try_from(arrow.as_ref()).unwrap();
    let specs = vec![
        MemIndexSpec::btree("id_idx", 0, "id"),
        MemIndexSpec::fts("text_idx", 2, "description"),
    ];
    let store = IndexStore::from_specs(&specs, &lance, 1_000, 16).unwrap();

    let batch = create_test_batch(&arrow, 1);
    store.insert(&batch, 0).unwrap();

    let btree = store.get_index("id_idx").unwrap();
    let outcome = btree
        .flush(&FlushContext::training_only(4096))
        .await
        .unwrap();
    assert!(matches!(outcome, FlushOutcome::TrainingData(_)));

    let fts = store.get_index("text_idx").unwrap();
    let Err(error) = fts.flush(&FlushContext::training_only(4096)).await else {
        panic!("a full-text index has no training data to give");
    };
    assert!(error.to_string().contains("no generation"), "{error}");
}

fn create_test_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("description", DataType::Utf8, true),
    ]))
}

fn create_test_batch(schema: &ArrowSchema, start_id: i32) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(vec![start_id, start_id + 1, start_id + 2])),
            Arc::new(StringArray::from(vec!["alice", "bob", "charlie"])),
            Arc::new(StringArray::from(vec![
                "hello world",
                "goodbye world",
                "hello again",
            ])),
        ],
    )
    .unwrap()
}

fn create_sized_batch(schema: &ArrowSchema, start_id: i32, num_rows: usize) -> RecordBatch {
    let ids: Vec<i32> = (0..num_rows as i32).map(|i| start_id + i).collect();
    let names: Vec<String> = ids.iter().map(|id| format!("name-{id}")).collect();
    let descriptions: Vec<String> = ids.iter().map(|id| format!("hello world {id}")).collect();
    RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(StringArray::from(descriptions)),
        ],
    )
    .unwrap()
}

fn fts_index_metadata(index_version: i32) -> IndexMetadata {
    fts_index_metadata_with_details(index_version, None)
}

fn fts_index_metadata_with_details(
    index_version: i32,
    details: Option<pbold::InvertedIndexDetails>,
) -> IndexMetadata {
    let index_details = details.map(|details| {
        let mut value = Vec::new();
        details.encode(&mut value).unwrap();
        Arc::new(prost_types::Any {
            type_url: "type.googleapis.com/lance.index.InvertedIndexDetails".to_string(),
            value,
        })
    });

    IndexMetadata {
        uuid: Uuid::new_v4(),
        fields: vec![2],
        covering_fields: vec![],
        name: "desc_idx".to_string(),
        dataset_version: 1,
        fragment_bitmap: None,
        index_details,
        index_version,
        created_at: None,
        base_id: None,
        files: None,
    }
}

/// Single-column `id` batch for primary-key lookup tests.
fn id_batch(ids: &[i32]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )])),
        vec![Arc::new(Int32Array::from(ids.to_vec()))],
    )
    .unwrap()
}

fn id_vector_batch(ids: &[i32]) -> RecordBatch {
    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            false,
        ),
    ]));
    let mut vectors = FixedSizeListBuilder::new(Float32Builder::new(), 2);
    for id in ids {
        vectors.values().append_value(*id as f32);
        vectors.values().append_value(*id as f32 + 0.5);
        vectors.append(true);
    }
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids.to_vec())),
            Arc::new(vectors.finish()),
        ],
    )
    .unwrap()
}

fn id_name_vector_batch(rows: &[(i32, &str)]) -> RecordBatch {
    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            false,
        ),
    ]));
    let mut ids = Vec::with_capacity(rows.len());
    let mut names = Vec::with_capacity(rows.len());
    let mut vectors = FixedSizeListBuilder::new(Float32Builder::new(), 2);
    for (id, name) in rows {
        ids.push(*id);
        names.push(*name);
        vectors.values().append_value(*id as f32);
        vectors.values().append_value(name.len() as f32);
        vectors.append(true);
    }
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
            Arc::new(vectors.finish()),
        ],
    )
    .unwrap()
}

#[test]
fn pk_newest_visible_single_column() {
    let mut store = IndexStore::new();
    store.enable_pk_index(&[("id".to_string(), 0)]);
    // id=1 at positions 0 and 2 (an update), id=2 at position 1.
    store.insert(&id_batch(&[1, 2]), 0).unwrap();
    store.insert(&id_batch(&[1]), 2).unwrap();

    let one = [ScalarValue::Int32(Some(1))];
    // Watermark above the update sees the newest position; below it, the older.
    assert_eq!(store.pk_newest_visible(&one, 5), Some(2));
    assert_eq!(store.pk_newest_visible(&one, 1), Some(0));
    assert!(store.pk_is_newest(&one, 2, 5));
    assert!(!store.pk_is_newest(&one, 0, 5));
    // Absent key (probed by the typed value, as the block-list does).
    assert!(!store.pk_contains_key(&ScalarValue::Int32(Some(9)), 5));
}

fn pk_vector_store() -> IndexStore {
    let mut store = IndexStore::new();
    store.add_hnsw(
        "vector_hnsw".to_string(),
        1,
        "vector".to_string(),
        DistanceType::L2,
        64,
        8,
    );
    store.enable_pk_index(&[("id".to_string(), 0)]);
    store
}

/// A key written again, in the same insert or a later one, is a rewrite.
#[rstest]
#[case::within_one_insert(&[3, 3])]
#[case::across_inserts(&[1])]
fn pk_has_overrides_tracks_single_column_rewrites(#[case] second: &[i32]) {
    let store = pk_vector_store();
    store.insert(&id_vector_batch(&[1, 2]), 0).unwrap();
    assert!(!store.pk_has_overrides(), "new keys are not rewrites");
    store.insert(&id_vector_batch(second), 2).unwrap();
    assert!(store.pk_has_overrides());
}

#[test]
#[should_panic(expected = "a primary-key index must be enabled before any row is inserted")]
fn enable_pk_index_after_search_rows_panics() {
    let mut store = IndexStore::new();
    store.add_hnsw(
        "vector_hnsw".to_string(),
        1,
        "vector".to_string(),
        DistanceType::L2,
        64,
        8,
    );
    store.insert(&id_vector_batch(&[1, 2]), 0).unwrap();

    store.enable_pk_index(&[("id".to_string(), 0)]);
}

/// Rewrites are recorded with no search index present.
#[test]
fn pk_has_overrides_tracks_rewrites_without_a_search_index() {
    let mut store = IndexStore::new();
    store.enable_pk_index(&[("id".to_string(), 0)]);

    store.insert(&id_batch(&[1, 2]), 0).unwrap();
    assert!(!store.pk_has_overrides());
    store.insert(&id_batch(&[1]), 2).unwrap();
    assert!(store.pk_has_overrides());
}

#[test]
fn pk_has_overrides_tracks_fts_rewrites() {
    let mut store = IndexStore::new();
    store.enable_pk_index(&[("id".to_string(), 0)]);
    store.add_fts("text_fts".to_string(), 1, "text".to_string());

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 1])),
            Arc::new(StringArray::from(vec!["alpha", "beta"])),
        ],
    )
    .unwrap();
    store.insert(&batch, 0).unwrap();
    assert!(
        store.pk_has_overrides(),
        "FTS PK rewrites must disable index-level FTS limit/WAND pushdown"
    );
}

#[test]
fn pk_newest_visible_composite_seeks_encoded_tuple() {
    let mut store = IndexStore::new();
    store.enable_pk_index(&[("id".to_string(), 0), ("name".to_string(), 1)]);
    // Rows: (1,"a")@0, (1,"b")@1, (1,"a")@2 — an update of (1,"a").
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1])),
            Arc::new(StringArray::from(vec!["a", "b", "a"])),
        ],
    )
    .unwrap();
    store.insert(&batch, 0).unwrap();

    let tuple_1a = [ScalarValue::Int32(Some(1)), ScalarValue::from("a")];
    let tuple_1b = [ScalarValue::Int32(Some(1)), ScalarValue::from("b")];
    // (1,"a")'s newest visible row is its re-write at position 2.
    assert_eq!(store.pk_newest_visible(&tuple_1a, 5), Some(2));
    assert!(store.pk_is_newest(&tuple_1a, 2, 5));
    assert!(!store.pk_is_newest(&tuple_1a, 0, 5));
    // (1,"b") only exists at position 1.
    assert_eq!(store.pk_newest_visible(&tuple_1b, 5), Some(1));
    // Watermark below the re-write: the older (1,"a")@0 is the newest visible.
    assert_eq!(store.pk_newest_visible(&tuple_1a, 1), Some(0));
    // An absent tuple (probed by its Binary-encoded key, as the block-list
    // does).
    let tuple_2a = [ScalarValue::Int32(Some(2)), ScalarValue::from("a")];
    let key_2a = ScalarValue::Binary(Some(encode_pk_tuple(&tuple_2a).unwrap()));
    assert!(!store.pk_contains_key(&key_2a, 5));
}

#[test]
fn pk_has_overrides_tracks_composite_rewrites() {
    let mut store = IndexStore::new();
    store.add_hnsw(
        "vector_hnsw".to_string(),
        2,
        "vector".to_string(),
        lance_linalg::distance::DistanceType::L2,
        64,
        8,
    );
    store.enable_pk_index(&[("id".to_string(), 0), ("name".to_string(), 1)]);
    let first = id_name_vector_batch(&[(1, "a"), (1, "b")]);
    store.insert(&first, 0).unwrap();
    assert!(!store.pk_has_overrides());

    let rewrite = id_name_vector_batch(&[(1, "a")]);
    store.insert(&rewrite, 2).unwrap();
    assert!(
        store.pk_has_overrides(),
        "repeated composite PK must disable HNSW"
    );
}

/// Each spec becomes an index under its own name, of its own kind.
#[test]
fn from_specs_builds_each_index_under_its_name() {
    let specs = vec![
        MemIndexSpec::btree("id_idx", 0, "id"),
        MemIndexSpec::fts("desc_idx", 2, "description"),
    ];
    let store = IndexStore::from_specs(&specs, &test_lance_schema(), 100, 10).unwrap();
    assert_eq!(store.len(), 2);
    store
        .insert(&create_test_batch(&create_test_schema(), 0), 0)
        .unwrap();

    assert_eq!(store.get_btree("id_idx").unwrap().len(), 3);
    assert_eq!(store.get_fts("desc_idx").unwrap().doc_count(), 3);
    assert!(store.get_btree("desc_idx").is_none());
    assert!(store.get_fts("missing").is_none());
    assert_eq!(store.index_names(), vec!["desc_idx", "id_idx"]);
    assert!(!store.is_empty() && IndexStore::new().is_empty());
}

/// Two indexes under one name would leave one of them unmaintained.
#[test]
fn from_specs_refuses_a_name_used_twice() {
    let specs = vec![
        MemIndexSpec::btree("idx", 0, "id"),
        MemIndexSpec::fts("idx", 2, "description"),
    ];
    let error = IndexStore::from_specs(&specs, &test_lance_schema(), 100, 10).unwrap_err();
    assert!(error.to_string().contains("'idx'"), "{error}");
}

#[test]
fn fts_registry_routes_row_and_element_targets_independently() {
    let mut store = IndexStore::new();
    store
        .add_fts_with_params(
            "tags_idx".to_string(),
            1,
            "tags".to_string(),
            InvertedIndexParams::default(),
        )
        .unwrap();
    store
        .add_fts_with_params(
            "tags_element_idx".to_string(),
            1,
            "tags".to_string(),
            InvertedIndexParams::default().document_granularity(
                lance_index::scalar::inverted::DocumentGranularity::ListElement,
            ),
        )
        .unwrap();

    assert_eq!(
        store.fts_granularities_on("tags"),
        vec![DocumentGranularity::Row, DocumentGranularity::ListElement]
    );
    for granularity in [DocumentGranularity::Row, DocumentGranularity::ListElement] {
        let index = store
            .index_answering("tags", &FtsMemQuery::probe(granularity))
            .unwrap();
        let index = (index.as_ref() as &dyn std::any::Any)
            .downcast_ref::<FtsMemIndex>()
            .unwrap();
        assert_eq!(index.document_granularity(), granularity);
    }
}

fn fts_params(resolved: &ResolvedIndex) -> &FtsParams {
    (resolved.params.as_ref() as &dyn std::any::Any)
        .downcast_ref()
        .unwrap()
}

#[test]
fn fts_from_metadata_preserves_format_version() {
    let arrow_schema = create_test_schema();
    let schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

    for (index_version, expected_format_version) in [
        (0, InvertedListFormatVersion::V1),
        (1, InvertedListFormatVersion::V1),
        (2, InvertedListFormatVersion::V2),
        (3, InvertedListFormatVersion::V3),
    ] {
        let config = FtsMemIndexPlugin::resolve_from_metadata(
            "fts_idx",
            &schema,
            &fts_index_metadata(index_version),
        )
        .unwrap();

        let config = fts_params(&config);
        assert_eq!(
            config.params.resolved_format_version(),
            expected_format_version
        );
    }
}

#[test]
fn fts_from_metadata_rejects_unsupported_format_version() {
    let arrow_schema = create_test_schema();
    let schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();

    let err = FtsMemIndexPlugin::resolve_from_metadata("fts_idx", &schema, &fts_index_metadata(4))
        .unwrap_err();
    assert!(
        err.to_string().contains("unsupported index_version 4"),
        "{err}"
    );
}

#[test]
fn fts_from_metadata_accepts_element_document_v3_capability() {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
        Field::new(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
    ]));
    let schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();
    let tags = schema.field("tags").unwrap();
    for (block_size, expected_format_version) in [
        (128, InvertedListFormatVersion::V2),
        (256, InvertedListFormatVersion::V3),
    ] {
        let params = InvertedIndexParams::default()
            .block_size(block_size)
            .unwrap()
            .document_granularity(lance_index::scalar::inverted::DocumentGranularity::ListElement);
        let details = pbold::InvertedIndexDetails::try_from(&params).unwrap();
        let mut metadata = fts_index_metadata_with_details(3, Some(details));
        metadata.fields = vec![tags.id];
        let config =
            FtsMemIndexPlugin::resolve_from_metadata("fts_idx", &schema, &metadata).unwrap();

        assert_eq!(config.columns, vec!["tags".to_string()]);
        let config = fts_params(&config);
        assert_eq!(
            config.params.get_document_granularity(),
            lance_index::scalar::inverted::DocumentGranularity::ListElement
        );
        assert_eq!(
            config.params.resolved_format_version(),
            expected_format_version
        );
    }
}

#[test]
fn fts_from_metadata_accepts_v3_with_legacy_block_size() {
    let arrow_schema = create_test_schema();
    let schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();
    let mut legacy_details =
        pbold::InvertedIndexDetails::try_from(&InvertedIndexParams::default()).unwrap();
    legacy_details.posting_format_version = None;

    for metadata in [
        fts_index_metadata(3),
        fts_index_metadata_with_details(3, Some(legacy_details)),
    ] {
        let config =
            FtsMemIndexPlugin::resolve_from_metadata("fts_idx", &schema, &metadata).unwrap();
        let config = fts_params(&config);
        assert_eq!(
            config.params.resolved_format_version(),
            InvertedListFormatVersion::V3
        );
        assert_eq!(config.params.posting_block_size(), 128);
    }
}

#[test]
fn fts_from_metadata_accepts_v3_with_256_block_size() {
    let arrow_schema = create_test_schema();
    let schema = LanceSchema::try_from(arrow_schema.as_ref()).unwrap();
    let params = InvertedIndexParams::default().block_size(256).unwrap();
    let details = pbold::InvertedIndexDetails::try_from(&params).unwrap();

    let config = FtsMemIndexPlugin::resolve_from_metadata(
        "fts_idx",
        &schema,
        &fts_index_metadata_with_details(3, Some(details)),
    )
    .unwrap();

    let config = fts_params(&config);
    assert_eq!(
        config.params.resolved_format_version(),
        InvertedListFormatVersion::V3
    );
    assert_eq!(config.params.posting_block_size(), 256);
}

/// HNSW reports its graph before the first insert allocates it.
#[test]
fn test_resident_bytes_charges_hnsw_before_first_insert() {
    let max_rows = 100_000;
    let btree_only = IndexStore::from_specs(
        &[MemIndexSpec::btree("pk_idx", 0, "id")],
        &test_lance_schema(),
        max_rows,
        1_000,
    )
    .unwrap();
    assert_eq!(
        btree_only.resident_bytes(),
        0,
        "a BTree index allocates per row, so an untouched one holds nothing"
    );

    let vector_schema = LanceSchema::try_from(id_vector_batch(&[]).schema().as_ref()).unwrap();
    let with_hnsw = IndexStore::from_specs(
        &[MemIndexSpec::hnsw("vec_idx", 1, "vector", DistanceType::L2)],
        &vector_schema,
        max_rows,
        1_000,
    )
    .unwrap();
    assert!(
        with_hnsw.resident_bytes() > max_rows * 128,
        "HNSW must report its preallocated graph, got {}",
        with_hnsw.resident_bytes()
    );
}

/// A plugin index's growth is counted in the store's total.
#[test]
fn test_resident_bytes_includes_plugin_growth() {
    let mut store = IndexStore::new();
    add_stub(&mut store, "id_stub", "id");
    assert_eq!(
        store.resident_bytes(),
        0,
        "an untouched plugin index holds nothing"
    );

    let schema = create_test_schema();
    let batch = create_sized_batch(&schema, 0, 512);
    store.insert(&batch, 0).unwrap();

    assert!(
        store.resident_bytes() > 0,
        "indexed rows must be charged to the store total"
    );
    let mut text = IndexStore::new();
    text.add_fts("desc_idx".to_string(), 2, "description".to_string());
    text.insert(&batch, 0).unwrap();
    assert!(
        text.resident_bytes() > 1,
        "a full-text index charges its rows"
    );
    assert_eq!(
        store.resident_bytes(),
        store.get_index("id_stub").unwrap().resident_bytes(),
        "the store total must account for plugin indexes"
    );
}

fn vector_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("description", DataType::Utf8, true),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        ),
        Field::new(
            "f64_vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 4),
            true,
        ),
    ]))
}

/// A spec that would fail every insert is rejected when the shard opens.
#[rstest]
#[case::btree_ok(MemIndexSpec::btree("idx", 0, "id"), None)]
#[case::btree_missing_column(
    MemIndexSpec::btree("idx", 9, "nope"),
    Some("not in the shard schema")
)]
// Column exists, but its field_id names a *different* column ("id" is 0, not 1).
#[case::btree_field_id_column_mismatch(MemIndexSpec::btree("idx", 1, "id"), Some("has field_id 0"))]
#[case::fts_ok(MemIndexSpec::fts("idx", 1, "description"), None)]
#[case::fts_non_utf8(
    MemIndexSpec::fts("idx", 0, "id"),
    Some("must resolve to Utf8, LargeUtf8, Utf8View, or JSON")
)]
#[case::fts_missing_column(
    MemIndexSpec::fts("idx", 9, "nope"),
    Some("does not exist in the dataset schema")
)]
#[case::fts_format_without_code_support(
    MemIndexSpec::fts_with_params(
        "idx",
        1,
        "description",
        InvertedIndexParams::default()
            .base_tokenizer("code".to_string())
            .format_version(InvertedListFormatVersion::V2),
    ),
    Some("requires FTS format_version=3")
)]
#[case::hnsw_ok(MemIndexSpec::hnsw("idx", 2, "vector", DistanceType::L2), None)]
#[case::hnsw_not_a_vector(
    MemIndexSpec::hnsw("idx", 0, "id", DistanceType::L2),
    Some("requires a FixedSizeList<Float32> column")
)]
#[case::hnsw_wrong_item_type(
    MemIndexSpec::hnsw("idx", 3, "f64_vector", DistanceType::L2),
    Some("column 'f64_vector' is FixedSizeList")
)]
#[case::hnsw_missing_column(
    MemIndexSpec::hnsw("idx", 9, "nope", DistanceType::L2),
    Some("not in the shard schema")
)]
#[case::flush_params_rejected(
    wrapped(MemIndexSpec::btree("idx", 0, "id"), Deviation::RejectsFlushParams),
    Some("flush parameters rejected")
)]
fn test_validate_index_specs(#[case] spec: MemIndexSpec, #[case] expected_error: Option<&str>) {
    let schema = vector_schema();
    let lance_schema = LanceSchema::try_from(schema.as_ref()).unwrap();
    let result = validate_index_specs(&[spec], &schema, &lance_schema, &[]);
    match expected_error {
        None => result.expect("a valid spec passes"),
        Some(fragment) => {
            let message = result.expect_err("an invalid spec is refused").to_string();
            assert!(
                message.contains(fragment),
                "error must explain the mismatch; wanted {fragment:?}, got {message:?}"
            );
        }
    }
}

#[test]
fn test_validate_nested_fts_index_config() {
    let content_fields = Fields::from(vec![Field::new("content", DataType::Utf8, true)]);
    let doc_item = Arc::new(Field::new("item", DataType::Struct(content_fields), true));
    let group_fields = Fields::from(vec![Field::new("docs", DataType::List(doc_item), true)]);
    let group_item = Arc::new(Field::new("item", DataType::Struct(group_fields), true));
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "groups",
        DataType::List(group_item),
        true,
    )]));
    let lance_schema = LanceSchema::try_from(schema.as_ref()).unwrap();
    let resolved = crate::index::scalar::inverted::resolve_fts_field(
        &lance_schema,
        "groups.docs.content",
        lance_index::scalar::inverted::DocumentGranularity::ListElement,
    )
    .unwrap();

    let params = InvertedIndexParams::default()
        .document_granularity(lance_index::scalar::inverted::DocumentGranularity::ListElement);
    let config = MemIndexSpec::fts_with_params(
        "idx",
        resolved.final_field_id,
        "groups.docs.content",
        params.clone(),
    );
    validate_index_specs(&[config], &schema, &lance_schema, &[]).unwrap();

    let wrong_field_id = MemIndexSpec::fts_with_params(
        "idx",
        resolved.final_field_id + 1,
        "groups.docs.content",
        params,
    );
    let error = validate_index_specs(&[wrong_field_id], &schema, &lance_schema, &[]).unwrap_err();
    assert!(error.to_string().contains("final field_id"), "{error}");
}

#[test]
fn test_validate_index_specs_rejects_diverged_lance_schema() {
    let arrow_schema = ArrowSchema::new(vec![Field::new("id", DataType::Int32, false)]);
    let lance_schema = LanceSchema::try_from(&ArrowSchema::new(vec![Field::new(
        "other",
        DataType::Int32,
        false,
    )]))
    .expect("test Lance schema must be valid");
    let config = MemIndexSpec::btree("idx", 0, "id");

    let error = validate_index_specs(&[config], &arrow_schema, &lance_schema, &[])
        .expect_err("diverged Arrow and Lance schemas must be rejected");
    assert!(
        matches!(error, Error::InvalidInput { .. }),
        "expected InvalidInput, got {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains("index 'idx'"),
        "error must name the index: {message}"
    );
    assert!(
        message.contains("column 'id'"),
        "error must name the column: {message}"
    );
    assert!(
        message.contains("available columns: [other]"),
        "error must show what the schema does hold, which is what makes the \
         divergence visible: {message}"
    );
}

/// A composite PK builds an order-preserving encoded key, so its columns must
/// be encodable. A single-column PK aliases a BTree entry, which accepts any
/// type — so it must *not* be rejected here.
#[test]
fn test_validate_composite_pk_column_types() {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new(
            "coords",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            true,
        ),
    ]));
    let lance_schema = LanceSchema::try_from(schema.as_ref()).unwrap();

    validate_index_specs(&[], &schema, &lance_schema, &["id".into(), "name".into()])
        .expect("Int32 + Utf8 composite PK must be encodable");

    let err = validate_index_specs(&[], &schema, &lance_schema, &["id".into(), "coords".into()])
        .expect_err("a FixedSizeList PK column has no order-preserving encoding");
    assert!(
        err.to_string().contains("order-preserving key encoding"),
        "error must name the reason, got {err}"
    );

    // A single-column PK of the same type is fine: it aliases a BTree.
    validate_index_specs(&[], &schema, &lance_schema, &["coords".into()])
        .expect("single-column PK aliases a BTree and accepts any type");

    // But every PK column must exist. A single-column PK naming an absent
    // column is rejected here, not left to fail deterministically on every
    // later index build and WAL replay.
    let err = validate_index_specs(&[], &schema, &lance_schema, &["missing".into()])
        .expect_err("a single-column PK on an absent column must be rejected");
    assert!(
        err.to_string().contains("not in the shard schema"),
        "error must name the missing column, got {err}"
    );
}

#[test]
fn test_index_store_indexed_count() {
    let schema = create_test_schema();
    let mut store = IndexStore::new();

    // field_id 0 for "id" column, field_id 2 for "description" column
    store.add_btree("id_idx".to_string(), 0, "id".to_string());
    store.add_fts("desc_idx".to_string(), 2, "description".to_string());

    // Initial watermark should be 0 (no data indexed yet)
    assert_eq!(store.indexed_count(), 0);

    // Insert with batch position tracking
    let batch = create_test_batch(&schema, 0);
    store
        .insert_with_batch_position(&batch, 0, Some(5))
        .unwrap();

    // Indexing batch position 5 means the prefix [0, 6) is indexed.
    assert_eq!(store.indexed_count(), 6);

    // Insert with higher batch position
    store
        .insert_with_batch_position(&batch, 3, Some(10))
        .unwrap();

    // Advances to cover batch position 10.
    assert_eq!(store.indexed_count(), 11);

    // Insert without batch position shouldn't change the cursor
    store.insert(&batch, 6).unwrap();
    assert_eq!(store.indexed_count(), 11);
}

/// `insert_batches` picks the inline or the threaded path by row count, so
/// exercise both and assert they leave the same index state: every row indexed
/// exactly once, in every index, with a timing reported for each.
#[rstest]
#[case::inline(8)]
#[case::threaded(PARALLEL_INDEX_MIN_ROWS + 64)]
fn test_insert_batches_indexes_every_row_once(#[case] num_rows: usize) {
    let schema = create_test_schema();
    let mut store = IndexStore::new();
    store.add_btree("id_idx".to_string(), 0, "id".to_string());
    store.add_fts("desc_idx".to_string(), 2, "description".to_string());
    add_stub(&mut store, "id_stub", "id");

    let batch = create_sized_batch(&schema, 0, num_rows);
    store
        .insert_batches(&[StoredBatch::new(batch, 0, 2)])
        .unwrap();

    let btree = store.get_btree("id_idx").unwrap();
    for id in 0..num_rows as i32 {
        let positions = btree.get(&ScalarValue::Int32(Some(id)));
        assert_eq!(
            positions.len(),
            1,
            "id={id} should be indexed exactly once, got {positions:?}"
        );
    }
    assert_eq!(store.get_fts("desc_idx").unwrap().doc_count(), num_rows);

    let stub = store.get_index("id_stub").unwrap();
    for id in 0..num_rows as i32 {
        let query = SargableQuery::Equals(ScalarValue::Int32(Some(id)));
        let found = stub
            .search(&query, &SearchContext::new(u64::MAX))
            .unwrap()
            .expect("the stub answers equality");
        let positions = found.as_filter().expect("a filter answer").at_most.len();
        assert_eq!(
            positions, 1,
            "id={id} should be indexed exactly once, got {positions}"
        );
    }
    assert_eq!(store.indexed_count(), 3);
}

fn stand_in(spec: MemIndexSpec) -> MemIndexSpec {
    wrapped(spec, Deviation::None)
}

/// Rewrites are tracked whatever type maintains the search index.
#[test]
fn a_stand_in_search_index_still_tracks_primary_key_rewrites() {
    let schema = LanceSchema::try_from(id_vector_batch(&[]).schema().as_ref()).unwrap();
    let spec = stand_in(MemIndexSpec::hnsw(
        "vector_hnsw",
        1,
        "vector",
        lance_linalg::distance::DistanceType::L2,
    ));
    let mut store = IndexStore::from_specs(&[spec], &schema, 64, 8).unwrap();
    store.enable_pk_index(&[("id".to_string(), 0)]);

    store.insert(&id_vector_batch(&[1, 2]), 0).unwrap();
    assert!(!store.pk_has_overrides());
    store.insert(&id_vector_batch(&[2]), 2).unwrap();
    assert!(
        store.pk_has_overrides(),
        "rewriting key 2 must be seen whatever type maintains the search index"
    );
}

/// A full-text index of another type reports its granularity.
#[test]
fn a_stand_in_full_text_index_reports_its_granularity() {
    let arrow = ArrowSchema::new(vec![Field::new(
        "tags",
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        true,
    )]);
    let schema = LanceSchema::try_from(&arrow).unwrap();
    let spec = stand_in(MemIndexSpec::fts_with_params(
        "tags_fts",
        0,
        "tags",
        InvertedIndexParams::default().document_granularity(DocumentGranularity::ListElement),
    ));
    let store = IndexStore::from_specs(&[spec], &schema, 64, 8).unwrap();
    assert_eq!(
        store.fts_granularities_on("tags"),
        vec![DocumentGranularity::ListElement]
    );
}

/// Settings that differ only in a cached schema lookup are the same index; a
/// plugin of another type with the same name, or another column or field, is
/// not.
#[test]
fn same_index_compares_what_an_index_is_built_from() {
    let arrow = ArrowSchema::new(vec![Field::new("text", DataType::Utf8, true)]);
    let schema = LanceSchema::try_from(&arrow).unwrap();
    let unresolved = MemIndexSpec::fts("text_fts", 0, "text");
    let mut resolved = unresolved.clone();
    resolved.params = Arc::new(fts::FtsParams {
        params: Default::default(),
        resolved_field: Some(
            crate::index::scalar::inverted::resolve_fts_field(
                &schema,
                "text",
                DocumentGranularity::Row,
            )
            .unwrap(),
        ),
    });
    assert!(unresolved.same_index(&resolved));

    assert!(!unresolved.same_index(&stand_in(unresolved.clone())));
    assert!(!unresolved.same_index(&MemIndexSpec::fts("text_fts", 0, "other")));
    assert!(!unresolved.same_index(&MemIndexSpec::fts("text_fts", 1, "text")));
    assert!(!unresolved.same_index(&MemIndexSpec::fts_with_params(
        "text_fts",
        0,
        "text",
        InvertedIndexParams::default().with_position(true),
    )));
    let mut rebuilt = unresolved.clone();
    rebuilt.details = Some(Arc::new(prost_types::Any {
        type_url: "/lance.table.InvertedIndexDetails".to_string(),
        value: vec![1],
    }));
    assert!(!unresolved.same_index(&rebuilt));
}

/// One index listed twice is one index; two different ones under one name are
/// refused before any is built.
#[test]
fn validate_index_specs_refuses_a_name_used_twice() {
    let schema = create_test_schema();
    let same = [
        MemIndexSpec::btree("idx", 0, "id"),
        MemIndexSpec::btree("idx", 0, "id"),
    ];
    validate_index_specs(&same, &schema, &test_lance_schema(), &[]).unwrap();
    let different = [
        MemIndexSpec::btree("idx", 0, "id"),
        MemIndexSpec::btree("idx", 1, "name"),
    ];
    let error = validate_index_specs(&different, &schema, &test_lance_schema(), &[]).unwrap_err();
    assert!(error.to_string().contains("configured twice"), "{error}");
}

/// Columns and field ids that do not pair up are refused, whatever the plugin
/// checks itself.
#[test]
fn a_spec_whose_columns_and_field_ids_differ_in_count_is_refused() {
    let mut spec = MemIndexSpec::for_plugin("idx", 0, "id", Arc::new(StubPlugin("StubDetails")));
    spec.field_ids.push(1);
    let schema = test_lance_schema();
    for error in [
        spec.validate(&schema).unwrap_err(),
        spec.build(&schema, 10, 1).unwrap_err(),
    ] {
        assert!(
            error
                .to_string()
                .contains("names 1 columns but 2 field ids"),
            "{error}"
        );
    }
}

/// A composite key needs every key column in each batch.
#[test]
fn a_composite_key_column_missing_from_a_batch_fails_the_insert() {
    let mut store = IndexStore::new();
    store.enable_pk_index(&[("id".to_string(), 0), ("name".to_string(), 1)]);
    let error = store.insert(&id_batch(&[1]), 0).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("primary-key column 'name' is not in the batch"),
        "{error}"
    );
}

#[test]
fn the_default_registry_holds_the_built_in_kinds() {
    let registry = MemIndexRegistry::default();
    let names: Vec<&str> = registry
        .plugins()
        .iter()
        .map(|plugin| plugin.name())
        .collect();
    assert_eq!(names, ["BTree", "Hnsw", "Inverted"]);
    assert!(MemIndexRegistry::empty().plugins().is_empty());
}

/// A shared key index found by its entry still reports rewrites when its plugin
/// hands out a new key capability each time it is asked.
#[test]
fn a_shared_key_index_tracks_rewrites_through_a_fresh_capability() {
    let mut store = IndexStore::from_specs(
        &[wrapped(
            MemIndexSpec::btree("id_idx", 0, "id"),
            Deviation::FreshKeyCapability,
        )],
        &test_lance_schema(),
        64,
        8,
    )
    .unwrap();
    store.enable_pk_index(&[("id".to_string(), 0)]);
    store.insert(&id_batch(&[1, 2]), 0).unwrap();
    store.insert(&id_batch(&[2]), 2).unwrap();
    assert_eq!(
        store.pk_newest_visible(&[ScalarValue::Int32(Some(2))], 2),
        Some(2)
    );
    assert!(store.pk_has_overrides());
}

/// A user index on exactly the key column is shared and counted once; without
/// one, or beside one that cannot serve, the store keeps its own B-tree.
#[test]
fn the_key_index_is_shared_or_owned() {
    let equality = SargableQuery::Equals(ScalarValue::Int32(Some(2)));

    let mut shared = IndexStore::new();
    shared.add_btree("id_idx".to_string(), 0, "id".to_string());
    shared.add_btree("name_idx".to_string(), 1, "name".to_string());
    shared.enable_pk_index(&[("id".to_string(), 0)]);
    shared
        .insert(&id_name_vector_batch(&[(1, "a"), (2, "b"), (3, "c")]), 0)
        .unwrap();
    assert_eq!(
        shared.resident_bytes(),
        shared.get_index("id_idx").unwrap().resident_bytes()
            + shared.get_index("name_idx").unwrap().resident_bytes()
    );
    // Rows reach every index while the key index reports rewrites.
    assert_eq!(shared.get_btree("name_idx").unwrap().len(), 3);

    let mut owned = IndexStore::new();
    owned.enable_pk_index(&[("id".to_string(), 0)]);
    owned.insert(&id_batch(&[1, 2, 3]), 0).unwrap();
    assert!(owned.is_empty());
    assert!(owned.resident_bytes() > 0, "the owned key index is counted");
    let answer = owned
        .index_answering("id", &equality)
        .expect("the owned key index answers")
        .search(&equality, &SearchContext::new(u64::MAX))
        .unwrap()
        .unwrap();
    assert_eq!(
        Vec::from(answer.as_filter().unwrap().at_most.clone()),
        vec![1]
    );

    // A user index on the key column that cannot serve as the key index gets an
    // owned one beside it, and still answers filters itself.
    let mut beside = IndexStore::from_specs(
        &[wrapped(
            MemIndexSpec::btree("id_idx", 0, "id"),
            Deviation::NotAKeyIndex,
        )],
        &test_lance_schema(),
        64,
        8,
    )
    .unwrap();
    beside.enable_pk_index(&[("id".to_string(), 0)]);
    beside.insert(&id_batch(&[1, 2, 2]), 0).unwrap();
    assert!(beside.pk_has_overrides());
    assert!(beside.resident_bytes() > beside.get_index("id_idx").unwrap().resident_bytes());
    let found = beside.index_answering("id", &equality).unwrap();
    assert!(Arc::ptr_eq(&found, beside.get_index("id_idx").unwrap()));
}

/// A key lookup with the wrong number of values finds nothing.
#[rstest]
#[case::single_column(&[("id", 0)], 2)]
#[case::composite(&[("id", 0), ("name", 1)], 1)]
fn a_key_lookup_of_the_wrong_arity_finds_nothing(
    #[case] key: &[(&str, i32)],
    #[case] arity: usize,
) {
    let mut store = IndexStore::new();
    let key: Vec<(String, i32)> = key.iter().map(|(c, id)| (c.to_string(), *id)).collect();
    store.enable_pk_index(&key);
    store.insert(&id_name_vector_batch(&[(1, "a")]), 0).unwrap();
    let values = vec![ScalarValue::Int32(Some(1)); arity];
    assert_eq!(store.pk_newest_visible(&values, u64::MAX), None);
}

/// Counts the insert calls it gets, and refuses to be fed one batch at a time.
#[derive(Debug, Default)]
struct ApplyCounter {
    columns: Vec<String>,
    calls: AtomicUsize,
    batches: AtomicUsize,
}

#[async_trait::async_trait]
impl MemIndex for ApplyCounter {
    fn columns(&self) -> &[String] {
        &self.columns
    }
    fn can_answer(&self, _query: &dyn MemQuery) -> bool {
        false
    }
    fn insert(&self, _batch: &RecordBatch, _row_offset: RowPosition) -> Result<()> {
        Err(Error::internal("indexed one batch at a time"))
    }
    fn insert_batches(&self, batches: &[StoredBatch]) -> Result<()> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.batches.fetch_add(batches.len(), Ordering::Relaxed);
        Ok(())
    }
    fn resident_bytes(&self) -> usize {
        0
    }
    fn search(&self, _query: &dyn MemQuery, _ctx: &SearchContext) -> Result<Option<MemMatches>> {
        Ok(None)
    }
    async fn flush(&self, _ctx: &FlushContext<'_>) -> Result<FlushOutcome> {
        Ok(FlushOutcome::Skip)
    }
}

fn three_batches(rows_per_batch: usize) -> Vec<StoredBatch> {
    let schema = create_test_schema();
    (0..3)
        .map(|n| {
            let start = n * rows_per_batch;
            StoredBatch::new(
                create_sized_batch(&schema, start as i32, rows_per_batch),
                start as u64,
                n,
            )
        })
        .collect()
}

/// An index receives every batch of one write in one call.
#[rstest]
#[case::inline(8)]
#[case::threaded(PARALLEL_INDEX_MIN_ROWS + 64)]
fn an_index_gets_every_batch_of_a_write_at_once(#[case] rows_per_batch: usize) {
    let counter = Arc::new(ApplyCounter {
        columns: vec!["id".to_string()],
        ..Default::default()
    });
    let mut store = IndexStore::new();
    store.add_btree("id_idx".to_string(), 0, "id".to_string());
    store.add_index("counter".to_string(), counter.clone());

    store
        .insert_batches(&three_batches(rows_per_batch))
        .unwrap();
    assert_eq!(counter.calls.load(Ordering::Relaxed), 1);
    assert_eq!(counter.batches.load(Ordering::Relaxed), 3);
    assert_eq!(store.indexed_count(), 3);
}

/// A failed index stops the write: the error comes back and the batches are
/// not counted indexed.
#[rstest]
#[case::inline(8)]
#[case::threaded(PARALLEL_INDEX_MIN_ROWS + 64)]
fn a_failed_index_leaves_the_batches_unindexed(#[case] rows_per_batch: usize) {
    let mut store = IndexStore::new();
    store.add_btree("id_idx".to_string(), 0, "id".to_string());
    store.add_index(
        "missing".to_string(),
        Arc::new(BTreeMemIndex::new(9, "missing".to_string())),
    );
    let error = store
        .insert_batches(&three_batches(rows_per_batch))
        .unwrap_err();
    assert!(error.to_string().contains("missing"), "{error}");
    assert_eq!(store.indexed_count(), 0);
}

/// An index that panics on its own thread fails the write, naming it.
#[test]
fn a_panicking_index_fails_the_write() {
    let store = IndexStore::from_specs(
        &[
            MemIndexSpec::btree("id_idx", 0, "id"),
            wrapped(
                MemIndexSpec::btree("name_idx", 1, "name"),
                Deviation::PanicsOnInsert,
            ),
        ],
        &test_lance_schema(),
        PARALLEL_INDEX_MIN_ROWS * 4,
        8,
    )
    .unwrap();
    let error = store
        .insert_batches(&three_batches(PARALLEL_INDEX_MIN_ROWS + 64))
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("'name_idx' panicked: insert panicked"),
        "{error}"
    );
}

/// The store's own key index, single-column or composite, takes every row and
/// sees rewrites on either insert path.
#[rstest]
#[case::owned_inline(&[("id", 0)], 8)]
#[case::owned_threaded(&[("id", 0)], PARALLEL_INDEX_MIN_ROWS + 64)]
#[case::composite_inline(&[("id", 0), ("name", 1)], 8)]
#[case::composite_threaded(&[("id", 0), ("name", 1)], PARALLEL_INDEX_MIN_ROWS + 64)]
fn an_owned_key_index_takes_every_row(#[case] key: &[(&str, i32)], #[case] rows: usize) {
    let mut store = IndexStore::new();
    store.add_btree("name_idx".to_string(), 1, "name".to_string());
    let key: Vec<(String, i32)> = key.iter().map(|(c, id)| (c.to_string(), *id)).collect();
    store.enable_pk_index(&key);

    let batches = three_batches(rows);
    store.insert_batches(&batches).unwrap();
    assert!(!store.pk_has_overrides());
    let last = &batches[2].data;
    let id = last
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .value(0);
    let name = last
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .value(0);
    let values: Vec<ScalarValue> = match key.len() {
        1 => vec![ScalarValue::Int32(Some(id))],
        _ => vec![
            ScalarValue::Int32(Some(id)),
            ScalarValue::Utf8(Some(name.to_string())),
        ],
    };
    assert_eq!(
        store.pk_newest_visible(&values, u64::MAX),
        Some(batches[2].row_offset)
    );

    store
        .insert_batches(&[StoredBatch::new(last.clone(), 3 * rows as u64, 3)])
        .unwrap();
    assert!(store.pk_has_overrides());
}

/// HNSW graph settings that cannot build a graph fail when the shard opens,
/// not on the first durable insert.
#[test]
fn hnsw_settings_that_cannot_build_a_graph_are_refused() {
    let schema = LanceSchema::try_from(id_vector_batch(&[]).schema().as_ref()).unwrap();
    let spec = MemIndexSpec::hnsw_with_params(
        "vector_hnsw",
        1,
        "vector",
        DistanceType::L2,
        lance_index::vector::hnsw::builder::HnswBuildParams::default().num_edges(0),
    );
    assert!(spec.validate(&schema).is_err());
}
