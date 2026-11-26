// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::query::MaskFilter;
use super::*;

/// Build a partition whose documents all score the same: term "alpha" once, doc
/// length 1. `row_ids` is indexed by DocId, so an unsorted slice models a
/// post-compaction remap.
async fn build_tied_partition(store: &Arc<LanceIndexStore>, partition_id: u64, row_ids: &[u64]) {
    let mut builder = InnerBuilder::new(partition_id, false, TokenSetFormat::default());
    builder.tokens.get_or_add("alpha");
    builder.posting_lists.push(PostingListBuilder::new(false));
    for (doc_id, &row_id) in row_ids.iter().enumerate() {
        builder.posting_lists[0].add(doc_id as u32, PositionRecorder::Count(1));
        builder.docs.append(row_id, 1);
    }
    builder.write(store.as_ref()).await.unwrap();
}

/// Load a tied corpus against an explicit cache. An uncached index keeps no
/// partition's row addresses resident between queries, so its walks start without
/// them.
async fn load_tied_index_with(
    partitions: &[(u64, Vec<u64>)],
    cache: &LanceCache,
) -> (TempObjDir, Arc<InvertedIndex>) {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    for (partition_id, row_ids) in partitions {
        build_tied_partition(&store, *partition_id, row_ids).await;
    }
    write_test_metadata(
        &store,
        partitions.iter().map(|(id, _)| *id).collect(),
        InvertedIndexParams::default(),
    )
    .await;
    let index = InvertedIndex::load(store, None, cache).await.unwrap();
    (tmpdir, index)
}

async fn search_alpha_with(
    index: &InvertedIndex,
    limit: usize,
    prefilter: Arc<dyn PreFilter>,
) -> Vec<u64> {
    let (row_ids, scores) = index
        .bm25_search(
            Arc::new(Tokens::new(vec!["alpha".to_owned()], DocType::Text)),
            Arc::new(FtsSearchParams::new().with_limit(Some(limit))),
            Operator::Or,
            prefilter,
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap();
    for window in scores.windows(2) {
        assert!(
            (window[0] - window[1]).abs() < 1e-6,
            "premise: scores must be tied, got {scores:?}"
        );
    }
    row_ids
}

/// Search for "alpha" and return `(row_id, doc_index)` per hit.
async fn search_alpha_documents(index: &InvertedIndex, limit: usize) -> Vec<(u64, Vec<u32>)> {
    index
        .bm25_search_documents(
            Arc::new(Tokens::new(vec!["alpha".to_owned()], DocType::Text)),
            Arc::new(FtsSearchParams::new().with_limit(Some(limit))),
            Operator::Or,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap()
        .into_iter()
        .map(|document| (document.row_id, document.doc_index))
        .collect()
}

/// Row addresses descending with DocId, from `base + num_docs - 1` down to
/// `base`, so the lowest addresses are the last documents the walk reaches.
fn descending_row_ids(base: u64, num_docs: u64) -> Vec<u64> {
    (0..num_docs).rev().map(|doc_id| base + doc_id).collect()
}

fn one_partition(row_ids: Vec<u64>) -> Vec<(u64, Vec<u64>)> {
    vec![(0, row_ids)]
}

/// A single partition whose row addresses no longer ascend with DocId, as after
/// a compaction remap. Postings are still walked in DocId order, so a collector
/// that kept the first documents it saw would answer 105, 100, 107.
fn remapped_row_ids() -> Vec<(u64, Vec<u64>)> {
    one_partition(vec![105, 100, 107, 102, 104, 101, 106, 103])
}

/// Two partitions with interleaved row addresses, so a tie spans partitions and
/// the shared WAND floor and the merge have to rank it.
fn interleaved_row_ids() -> Vec<(u64, Vec<u64>)> {
    vec![(0, vec![100, 102, 104, 106]), (1, vec![101, 103, 105, 107])]
}

/// Twelve partitions holding row addresses striped across them, so the six
/// winners come from six different partitions and the answer turns on the order
/// `buffer_unordered` completes them in.
fn striped_row_ids() -> Vec<(u64, Vec<u64>)> {
    const PARTITIONS: u64 = 12;
    (0..PARTITIONS)
        .map(|partition_id| {
            let row_ids = (0..3)
                .map(|offset| partition_id + offset * PARTITIONS)
                .collect::<Vec<_>>();
            (partition_id, row_ids)
        })
        .collect()
}

/// `count` partitions of `per_partition` documents each, addresses descending
/// inside each. Each band fits its own collector while the bands together pass
/// the merge's bound, the only buffer on the merge path that grows with the
/// partition count.
///
/// The walk defers ties because no address projection is resident: the index
/// loads through a cache that has not read this partition's documents.
fn descending_partitions(per_partition: u64, count: u64) -> Vec<(u64, Vec<u64>)> {
    (0..count)
        .map(|partition_id| {
            (
                partition_id,
                descending_row_ids(partition_id * 1_000, per_partition),
            )
        })
        .collect()
}

/// How a tied corpus compares to `limit + SCORE_FLOOR_BUFFER`, the bound that
/// picks whether the merge or the address-ordered retry ranks the tie.
#[derive(Debug, Clone, Copy)]
enum Band {
    /// Every partition holds its whole band, and the merge stays inside its own.
    FitsCollector,
    /// Every partition holds its band, but the bands together overrun the merge's
    /// bound, so the merge compacts them.
    MergeOverruns,
    /// Every partition's band is wider than its collector, forcing the
    /// address-ordered retry.
    CollectorOverruns,
}

/// The filter a search runs with. An allow-all prefilter gives the walk the
/// partition's addresses; an unfiltered walk has them only if a cache kept them.
#[derive(Debug, Clone, Copy)]
enum FilterPath {
    Unfiltered,
    AllowAll,
}

/// The cache an index loads with, and what a search leaves resident behind it.
#[derive(Debug, Clone, Copy)]
enum Residency {
    /// The 64 MiB cache an index is loaded with by default.
    Roomy,
    /// Nothing stays resident, so every walk starts without addresses.
    Nothing,
}

/// Search a corpus tied at a single score at each limit: every limit must return
/// the lowest row addresses, so top-k at a small `k` is an ordered prefix of
/// top-k at a larger `k`. `rounds` repeats the searches against one loaded index,
/// so a later search starts from what an earlier one left in the cache.
async fn assert_tied_top_k_is_an_ordered_prefix(
    partitions: &[(u64, Vec<u64>)],
    limits: &[usize],
    band: Band,
    filter_path: FilterPath,
    residency: Residency,
    rounds: usize,
) {
    let docs_per_partition = partitions
        .iter()
        .map(|(_, row_ids)| row_ids.len())
        .collect::<Vec<_>>();
    let max_limit = *limits.iter().max().unwrap();
    let min_limit = *limits.iter().min().unwrap();
    match band {
        Band::FitsCollector | Band::MergeOverruns => assert!(
            docs_per_partition
                .iter()
                .all(|docs| *docs <= max_limit + SCORE_FLOOR_BUFFER),
            "premise: every partition must hold its whole tie band, got {docs_per_partition:?} documents for limit {max_limit}"
        ),
        Band::CollectorOverruns => assert!(
            docs_per_partition
                .iter()
                .all(|docs| *docs > min_limit + SCORE_FLOOR_BUFFER),
            "premise: every partition's tie band must overflow its collector, got {docs_per_partition:?} documents for limit {min_limit}"
        ),
    }
    if matches!(band, Band::MergeOverruns) {
        let total = docs_per_partition.iter().sum::<usize>();
        assert!(
            total > max_limit + SCORE_FLOOR_BUFFER,
            "premise: the partitions together must overrun the merge's band bound, got {total} documents for limit {max_limit}"
        );
    }

    let expected = ascending_row_ids(partitions);
    assert!(
        expected.len() >= max_limit,
        "premise: the corpus must hold at least {max_limit} tied documents"
    );

    let cache = match residency {
        Residency::Roomy => LanceCache::with_capacity(64 * 1024 * 1024),
        Residency::Nothing => LanceCache::no_cache(),
    };
    let (_tmpdir, index) = load_tied_index_with(partitions, &cache).await;
    if matches!(residency, Residency::Nothing) {
        assert!(
            !index.has_resident_document_projections(),
            "premise: the first search must start without resident addresses"
        );
    }

    let prefilter: Arc<dyn PreFilter> = match filter_path {
        FilterPath::Unfiltered => Arc::new(NoFilter),
        FilterPath::AllowAll => Arc::new(MaskFilter(RowAddrMask::from_allowed(
            RowAddrTreeMap::from_iter(expected.iter().copied()),
        ))),
    };

    for round in 0..rounds {
        for &limit in limits {
            let found = search_alpha_with(&index, limit, prefilter.clone()).await;
            assert_eq!(
                found,
                expected[..limit],
                "round {round}: top-{limit} must be the {limit} lowest row addresses"
            );
        }
    }
}

/// The corpus's row addresses sorted ascending, the order a tie returns under
/// the `row_id ASC` tiebreak.
fn ascending_row_ids(partitions: &[(u64, Vec<u64>)]) -> Vec<u64> {
    let mut row_ids = partitions
        .iter()
        .flat_map(|(_, ids)| ids.iter().copied())
        .collect::<Vec<_>>();
    row_ids.sort_unstable();
    row_ids
}

#[rstest::rstest]
#[case::single_partition_row_id_remap(remapped_row_ids(), &[3, 5], Band::FitsCollector, FilterPath::Unfiltered, Residency::Roomy, 1)]
#[case::two_partitions_interleaved_row_ids(interleaved_row_ids(), &[3, 6], Band::FitsCollector, FilterPath::Unfiltered, Residency::Roomy, 1)]
#[case::twelve_partitions_striped_row_ids(striped_row_ids(), &[3, 6], Band::FitsCollector, FilterPath::Unfiltered, Residency::Roomy, 8)]
#[case::allow_all_prefilter(remapped_row_ids(), &[3, 5], Band::FitsCollector, FilterPath::AllowAll, Residency::Roomy, 1)]
#[case::six_partitions_overrun_the_merge_band(descending_partitions(100, 6), &[4], Band::MergeOverruns, FilterPath::Unfiltered, Residency::Roomy, 1)]
#[case::ascending_row_ids_uncached(one_partition((0..1_024).collect()), &[10, 3], Band::CollectorOverruns, FilterPath::Unfiltered, Residency::Nothing, 1)]
#[case::ascending_row_ids_uncached_allow_all(one_partition((0..1_024).collect()), &[10, 3], Band::CollectorOverruns, FilterPath::AllowAll, Residency::Nothing, 1)]
#[case::descending_row_ids_overflow_one_partition(one_partition(descending_row_ids(10_001, 512)), &[3, 5], Band::CollectorOverruns, FilterPath::Unfiltered, Residency::Roomy, 1)]
#[tokio::test]
async fn test_fts_topk_tied_scores_are_an_ordered_prefix(
    #[case] partitions: Vec<(u64, Vec<u64>)>,
    #[case] limits: &[usize],
    #[case] band: Band,
    #[case] filter_path: FilterPath,
    #[case] residency: Residency,
    #[case] rounds: usize,
) {
    assert_tied_top_k_is_an_ordered_prefix(
        &partitions,
        limits,
        band,
        filter_path,
        residency,
        rounds,
    )
    .await;
}

/// Build an element-granularity partition of coordinate rank `RANK`. Entries are
/// appended in the given order, so DocId order can disagree with the element
/// coordinates.
async fn build_tied_element_partition_of_rank<const RANK: usize>(
    store: &Arc<LanceIndexStore>,
    partition_id: u64,
    elements: &[(u64, [u32; RANK])],
) {
    let mut builder = InnerBuilder::new(partition_id, false, TokenSetFormat::default());
    builder.docs = DocSet::with_coordinate_rank(RANK);
    builder.tokens.get_or_add("alpha");
    builder.posting_lists.push(PostingListBuilder::new(false));
    for (doc_id, (row_id, coordinates)) in elements.iter().enumerate() {
        builder.posting_lists[0].add(doc_id as u32, PositionRecorder::Count(1));
        builder
            .docs
            .append_with_doc_index(*row_id, 1, coordinates)
            .unwrap();
    }
    builder.write(store.as_ref()).await.unwrap();
}

/// `build_tied_element_partition_of_rank` at coordinate rank 1.
async fn build_tied_element_partition(
    store: &Arc<LanceIndexStore>,
    partition_id: u64,
    elements: &[(u64, u32)],
) {
    let elements = elements
        .iter()
        .map(|(row_id, doc_index)| (*row_id, [*doc_index]))
        .collect::<Vec<_>>();
    build_tied_element_partition_of_rank(store, partition_id, &elements).await;
}

/// Build an element-granularity partition of coordinate rank 2, as a
/// `List<List<String>>` column stores its elements. Each `(row_id, [doc_index_0,
/// doc_index_1])` entry is appended in the given order.
async fn build_tied_nested_element_partition(
    store: &Arc<LanceIndexStore>,
    partition_id: u64,
    elements: &[(u64, [u32; 2])],
) {
    build_tied_element_partition_of_rank(store, partition_id, elements).await;
}

#[tokio::test]
async fn test_fts_topk_tied_elements_of_one_row_resolve_by_doc_index() {
    // The winning element of a row is visited after the heap fills: a collector
    // that ranked the row address alone, or the posting DocId, would drop it.
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    build_tied_element_partition(&store, 0, &[(100, 1), (100, 2), (100, 0), (50, 0)]).await;
    write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
    let cache = LanceCache::with_capacity(64 * 1024 * 1024);
    let index = InvertedIndex::load(store, None, &cache).await.unwrap();

    let expected = vec![(50, vec![0]), (100, vec![0])];
    // No addresses are resident yet, so the walk defers the tie and the whole
    // k-th-score band survives.
    assert_eq!(search_alpha_documents(&index, 2).await, expected);

    // With the address projection resident, the walk ranks by row address and
    // element coordinate.
    index.partitions[0]
        .docs
        .modern()
        .unwrap()
        .prewarm()
        .await
        .unwrap();
    assert!(index.has_resident_document_projections());
    assert_eq!(search_alpha_documents(&index, 2).await, expected);

    let top4 = search_alpha_documents(&index, 4).await;
    assert_eq!(
        top4,
        vec![
            (50, vec![0]),
            (100, vec![0]),
            (100, vec![1]),
            (100, vec![2])
        ]
    );
    assert_eq!(top4[..2], expected);
}

#[tokio::test]
async fn test_fts_topk_tied_elements_past_the_band_cap_keep_doc_index_order() {
    // More elements than `limit + SCORE_FLOOR_BUFFER`, appended in coordinate
    // order, so DocId order and `doc_index` order name the same elements. The
    // descending direction lives in
    // `test_fts_topk_tied_elements_with_descending_doc_index_beat_the_band_cap`.
    const TRUNCATED_ROW: u64 = 200;
    const TRUNCATED_ROW_ELEMENTS: u32 = 135;
    const OTHER_ROW: u64 = 100;
    const OTHER_ROW_ELEMENTS: u32 = 3;
    // The smaller limit runs second, so its answer can be compared against the
    // larger one as a prefix.
    const MAX_LIMIT: usize = 5;
    const MIN_LIMIT: usize = 3;

    assert!(
        TRUNCATED_ROW_ELEMENTS as usize > MAX_LIMIT + SCORE_FLOOR_BUFFER,
        "premise: row {TRUNCATED_ROW} must hold more elements than the band cap, got {TRUNCATED_ROW_ELEMENTS} for limit {MAX_LIMIT}"
    );

    // Elements of the wide row come first, so the heap fills from that row and the
    // second row's addresses outrank them only once the walk reaches them.
    let fixture = (0..TRUNCATED_ROW_ELEMENTS)
        .map(|doc_index| (TRUNCATED_ROW, doc_index))
        .chain((0..OTHER_ROW_ELEMENTS).map(|doc_index| (OTHER_ROW, doc_index)))
        .collect::<Vec<_>>();
    let element_counts = [
        (TRUNCATED_ROW, TRUNCATED_ROW_ELEMENTS),
        (OTHER_ROW, OTHER_ROW_ELEMENTS),
    ];

    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    build_tied_element_partition(&store, 0, &fixture).await;
    write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
    let cache = LanceCache::with_capacity(64 * 1024 * 1024);
    let index = InvertedIndex::load(store, None, &cache).await.unwrap();
    index.partitions[0]
        .docs
        .modern()
        .unwrap()
        .prewarm()
        .await
        .unwrap();
    assert!(
        index.has_resident_document_projections(),
        "premise: resident addresses let the walk rank element ties by coordinate"
    );

    let top5 = search_alpha_documents(&index, MAX_LIMIT).await;
    assert_eq!(
        top5,
        vec![
            (OTHER_ROW, vec![0]),
            (OTHER_ROW, vec![1]),
            (OTHER_ROW, vec![2]),
            (TRUNCATED_ROW, vec![0]),
            (TRUNCATED_ROW, vec![1]),
        ],
        "the DocId proxy must hand back the lowest `doc_index` elements of row {TRUNCATED_ROW}"
    );
    assert_eq!(
        search_alpha_documents(&index, MIN_LIMIT).await,
        top5[..MIN_LIMIT],
        "top-{MIN_LIMIT} must be an ordered prefix of top-{MAX_LIMIT}"
    );

    // A coordinate only counts as correct together with the address it belongs to:
    // an element past the row's element count came from another row.
    for (row_id, coordinates) in &top5 {
        let Some(&elements) = element_counts
            .iter()
            .find(|(id, _)| *id == *row_id)
            .map(|(_, count)| count)
        else {
            panic!("row {row_id} is not part of the fixture");
        };
        assert_eq!(coordinates.len(), 1, "coordinate rank 1 gives one element");
        assert!(
            coordinates[0] < elements,
            "element {coordinates:?} does not belong to row {row_id}, which holds {elements}"
        );
    }
}

/// One row holds more elements than the band cap, with element coordinates
/// opposite to DocId order.
#[tokio::test]
async fn test_fts_topk_tied_elements_with_descending_doc_index_beat_the_band_cap() {
    const ROW: u64 = 100;
    const ELEMENTS: u32 = 135;
    const LIMITS: &[usize] = &[5, 2, 1];

    assert!(
        ELEMENTS as usize > LIMITS[0] + SCORE_FLOOR_BUFFER,
        "premise: row {ROW} must hold more elements than the band cap at limit {}, got {ELEMENTS}",
        LIMITS[0]
    );

    let fixture = (0..ELEMENTS)
        .rev()
        .map(|doc_index| (ROW, doc_index))
        .collect::<Vec<_>>();
    let expected = (0..ELEMENTS)
        .map(|doc_index| (ROW, vec![doc_index]))
        .collect::<Vec<_>>();

    for residency in [Residency::Nothing, Residency::Roomy] {
        let cache = match residency {
            Residency::Roomy => LanceCache::with_capacity(64 * 1024 * 1024),
            Residency::Nothing => LanceCache::no_cache(),
        };
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        build_tied_element_partition(&store, 0, &fixture).await;
        write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
        let index = InvertedIndex::load(store, None, &cache).await.unwrap();
        if matches!(residency, Residency::Roomy) {
            index.partitions[0]
                .docs
                .modern()
                .unwrap()
                .prewarm()
                .await
                .unwrap();
            assert!(
                index.has_resident_document_projections(),
                "premise: resident addresses let the walk rank element ties by coordinate"
            );
        } else {
            assert!(
                !index.has_resident_document_projections(),
                "premise: the walk starts without resident addresses, so it defers the tie and retries"
            );
        }

        let widest = search_alpha_documents(&index, LIMITS[0]).await;
        assert_eq!(
            widest,
            expected[..LIMITS[0]],
            "{residency:?}: top-{} must be the {} lowest `doc_index` elements of row {ROW}",
            LIMITS[0],
            LIMITS[0]
        );
        for &limit in &LIMITS[1..] {
            let found = search_alpha_documents(&index, limit).await;
            assert_eq!(
                found,
                expected[..limit],
                "{residency:?}: top-{limit} must be the {limit} lowest `doc_index` elements of row {ROW}"
            );
            assert_eq!(
                found,
                widest[..limit],
                "{residency:?}: top-{limit} must be an ordered prefix of top-{}",
                LIMITS[0]
            );
        }
    }
}

/// A `List<List<String>>` row: its elements share rank 0, so rank 1 decides, and
/// the key packs both ranks into one value.
#[tokio::test]
async fn test_fts_topk_tied_nested_list_elements_resolve_by_the_second_doc_index() {
    const ROW: u64 = 100;
    const ELEMENTS: u32 = 135;
    const LIMITS: &[usize] = &[5, 2, 1];

    assert!(
        ELEMENTS as usize > LIMITS[0] + SCORE_FLOOR_BUFFER,
        "premise: row {ROW} must hold more elements than the band cap at limit {}, got {ELEMENTS}",
        LIMITS[0]
    );

    // A key holding rank 0 alone ranks all 135 elements equal.
    let fixture = (0..ELEMENTS)
        .rev()
        .map(|doc_index| (ROW, [0, doc_index]))
        .collect::<Vec<_>>();
    let expected = (0..ELEMENTS)
        .map(|doc_index| (ROW, vec![0, doc_index]))
        .collect::<Vec<_>>();

    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    build_tied_nested_element_partition(&store, 0, &fixture).await;
    write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
    let cache = LanceCache::with_capacity(64 * 1024 * 1024);
    let index = InvertedIndex::load(store, None, &cache).await.unwrap();
    index.partitions[0]
        .docs
        .modern()
        .unwrap()
        .prewarm()
        .await
        .unwrap();
    assert!(
        index.has_resident_document_projections(),
        "premise: resident addresses let the walk rank element ties by coordinate"
    );

    let widest = search_alpha_documents(&index, LIMITS[0]).await;
    assert_eq!(
        widest,
        expected[..LIMITS[0]],
        "top-{} must be the {} lowest (`doc_index_0`, `doc_index_1`) elements of row {ROW}",
        LIMITS[0],
        LIMITS[0]
    );
    for &limit in &LIMITS[1..] {
        let found = search_alpha_documents(&index, limit).await;
        assert_eq!(
            found,
            expected[..limit],
            "top-{limit} must be the {limit} lowest (`doc_index_0`, `doc_index_1`) elements of row {ROW}"
        );
        assert_eq!(
            found,
            widest[..limit],
            "top-{limit} must be an ordered prefix of top-{}",
            LIMITS[0]
        );
    }
}

/// One row of a rank-`RANK` partition holds more elements than the band cap, and
/// its last coordinate runs opposite to DocId order. The earlier ranks are
/// constant, so one packed coordinate key ranks every element equal.
async fn assert_tied_deep_element_top_k_is_an_ordered_prefix<const RANK: usize>(limits: &[usize]) {
    const ROW: u64 = 100;
    const ELEMENTS: u32 = 135;

    assert!(
        ELEMENTS as usize > limits[0] + SCORE_FLOOR_BUFFER,
        "premise: row {ROW} must hold more elements than the band cap at limit {}, got {ELEMENTS}",
        limits[0]
    );

    let coordinates = |doc_index: u32| {
        let mut path = [0; RANK];
        path[RANK - 1] = doc_index;
        path
    };
    let fixture = (0..ELEMENTS)
        .rev()
        .map(|doc_index| (ROW, coordinates(doc_index)))
        .collect::<Vec<_>>();
    let expected = (0..ELEMENTS)
        .map(|doc_index| (ROW, coordinates(doc_index).to_vec()))
        .collect::<Vec<_>>();

    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    build_tied_element_partition_of_rank(&store, 0, &fixture).await;
    write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
    let cache = LanceCache::with_capacity(64 * 1024 * 1024);
    let index = InvertedIndex::load(store, None, &cache).await.unwrap();
    index.partitions[0]
        .docs
        .modern()
        .unwrap()
        .prewarm()
        .await
        .unwrap();
    assert!(
        index.has_resident_document_projections(),
        "premise: resident addresses let the walk rank element ties by coordinate"
    );

    let widest = search_alpha_documents(&index, limits[0]).await;
    assert_eq!(
        widest,
        expected[..limits[0]],
        "top-{} must be the {} lowest element coordinates of row {ROW}",
        limits[0],
        limits[0]
    );
    for &limit in &limits[1..] {
        let found = search_alpha_documents(&index, limit).await;
        assert_eq!(
            found,
            expected[..limit],
            "top-{limit} must be the {limit} lowest element coordinates of row {ROW}"
        );
        assert_eq!(
            found,
            widest[..limit],
            "top-{limit} must be an ordered prefix of top-{}",
            limits[0]
        );
    }
}

#[tokio::test]
async fn test_fts_topk_tied_rank_three_list_elements_resolve_by_the_third_doc_index() {
    // A `List<List<List<String>>>` row: its elements share ranks 0-1, so the first
    // packed key ranks all 135 equal and ranking needs rank 2.
    assert_tied_deep_element_top_k_is_an_ordered_prefix::<3>(&[5, 2, 1]).await;
}

#[tokio::test]
async fn test_fts_topk_tied_rank_four_list_elements_resolve_by_the_fourth_doc_index() {
    // A `List<List<List<List<String>>>>` row: its elements share ranks 0-2, so only
    // rank 3, the second rank of the second packed key, separates them.
    assert_tied_deep_element_top_k_is_an_ordered_prefix::<4>(&[5, 2, 1]).await;
}

/// Coordinate rank 5 is deeper than the two packed keys cover, so the walk gets
/// no key and retains the row's k-th-score band.
#[tokio::test]
async fn test_fts_topk_tied_rank_five_list_elements_stay_exact_inside_the_band_cap() {
    const ROW: u64 = 100;
    const ELEMENTS: u32 = 20;
    const LIMITS: &[usize] = &[5, 2, 1];

    // Past `limit + SCORE_FLOOR_BUFFER` the DocId proxy decides which elements
    // survive, so this fixture has to fit every limit.
    let min_limit = *LIMITS.iter().min().unwrap();
    assert!(
        ELEMENTS as usize <= min_limit + SCORE_FLOOR_BUFFER,
        "premise: every limit must hold the row's whole tie band, got {ELEMENTS} elements for limit {min_limit}"
    );

    // Ranks 0-3 are constant and rank 4 runs opposite to DocId order.
    let coordinates = |doc_index: u32| {
        let mut path = [0; 5];
        path[4] = doc_index;
        path
    };
    let fixture = (0..ELEMENTS)
        .rev()
        .map(|doc_index| (ROW, coordinates(doc_index)))
        .collect::<Vec<_>>();
    let expected = (0..ELEMENTS)
        .map(|doc_index| (ROW, coordinates(doc_index).to_vec()))
        .collect::<Vec<_>>();

    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    build_tied_element_partition_of_rank(&store, 0, &fixture).await;
    write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
    let cache = LanceCache::with_capacity(64 * 1024 * 1024);
    let index = InvertedIndex::load(store, None, &cache).await.unwrap();
    index.partitions[0]
        .docs
        .modern()
        .unwrap()
        .prewarm()
        .await
        .unwrap();
    assert!(
        index.has_resident_document_projections(),
        "premise: resident addresses make the walk retain element ties"
    );

    let widest = search_alpha_documents(&index, LIMITS[0]).await;
    assert_eq!(
        widest,
        expected[..LIMITS[0]],
        "top-{} must be the {} lowest element coordinates of row {ROW}",
        LIMITS[0],
        LIMITS[0]
    );
    for &limit in &LIMITS[1..] {
        let found = search_alpha_documents(&index, limit).await;
        assert_eq!(
            found,
            expected[..limit],
            "top-{limit} must be the {limit} lowest element coordinates of row {ROW}"
        );
        assert_eq!(
            found,
            widest[..limit],
            "top-{limit} must be an ordered prefix of top-{}",
            LIMITS[0]
        );
    }
}

#[tokio::test]
async fn test_fts_topk_element_ties_are_not_ranked_by_doc_index_without_row_addresses() {
    // A walk with no resident addresses cannot rank a tie between two rows, so it
    // retains the band and the merge ranks it by row address. An element coordinate
    // on its own would rank that tie by element position across rows, and drop the
    // winner of the result order from the band.
    let fixture = [(200, 0), (100, 5), (300, 1)];
    let expected = [(100, vec![5]), (200, vec![0]), (300, vec![1])];

    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    build_tied_element_partition(&store, 0, &fixture).await;
    write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
    let cache = LanceCache::no_cache();
    let index = InvertedIndex::load(store, None, &cache).await.unwrap();
    assert!(
        !index.has_resident_document_projections(),
        "premise: the walk starts without resident addresses, so it defers every tie"
    );

    for limit in (1..=fixture.len()).rev() {
        let found = search_alpha_documents(&index, limit).await;
        assert_eq!(
            found,
            expected[..limit],
            "top-{limit} must be the {limit} lowest row addresses of the fixture"
        );
    }
}

#[test]
fn test_push_top_k_document_breaks_legacy_ties_by_row_id() {
    // The legacy merge compares row addresses, so a lower row_id evicts the
    // worst tied entry and a higher one loses, in any arrival order.
    let mut candidates = BinaryHeap::new();
    for (row_id, score) in [(20u64, 1.0f32), (10, 1.0), (5, 1.0), (30, 1.0), (15, 0.5)] {
        push_top_k_document(&mut candidates, 2, ScoredDoc::new(row_id, score));
    }
    let ranked = candidates
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse(doc)| (doc.row_id, doc.score.0))
        .collect::<Vec<_>>();
    assert_eq!(ranked, vec![(5, 1.0), (10, 1.0)]);
}

#[test]
fn test_push_top_k_document_rejects_nan_scores() {
    // `OrderedFloat` compares with `total_cmp`, which sorts NaN above every real
    // score. The legacy merge drops a NaN, in an under-filled heap too.
    let mut candidates = BinaryHeap::new();
    push_top_k_document(&mut candidates, 4, ScoredDoc::new(7, f32::NAN));
    assert!(candidates.is_empty(), "NaN must not become a candidate");

    push_top_k_document(&mut candidates, 4, ScoredDoc::new(1, 1.0));
    push_top_k_document(&mut candidates, 4, ScoredDoc::new(2, f32::NAN));
    let ranked = candidates
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse(doc)| doc.row_id)
        .collect::<Vec<_>>();
    assert_eq!(ranked, vec![1]);
}

/// Counts reads of the element coordinate columns across every partition.
#[derive(Default, Debug)]
struct CoordinateReadCounter {
    reads: AtomicUsize,
}

struct CountingCoordinateReader {
    inner: Arc<dyn IndexReader>,
    counter: Arc<CoordinateReadCounter>,
}

#[cfg_attr(coverage, coverage(off))]
#[async_trait]
impl IndexReader for CountingCoordinateReader {
    async fn read_record_batch(&self, n: u64, batch_size: u64) -> Result<RecordBatch> {
        self.inner.read_record_batch(n, batch_size).await
    }
    async fn read_global_buffer(&self, index: u32) -> Result<bytes::Bytes> {
        self.inner.read_global_buffer(index).await
    }
    async fn read_range(
        &self,
        range: std::ops::Range<usize>,
        projection: Option<&[&str]>,
    ) -> Result<RecordBatch> {
        if projection.is_some_and(|columns| columns.contains(&doc_index_storage_column(0).as_str()))
        {
            self.counter.reads.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.read_range(range, projection).await
    }
    async fn num_batches(&self, batch_size: u64) -> u32 {
        self.inner.num_batches(batch_size).await
    }
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }
    fn schema(&self) -> &lance_core::datatypes::Schema {
        self.inner.schema()
    }
}

#[derive(Debug)]
struct CoordinateReads {
    counter: Arc<CoordinateReadCounter>,
}

impl ReaderWrapper for CoordinateReads {
    fn wrap(&self, _name: &str, reader: Arc<dyn IndexReader>) -> Arc<dyn IndexReader> {
        Arc::new(CountingCoordinateReader {
            inner: reader,
            counter: self.counter.clone(),
        })
    }
}

#[tokio::test]
async fn test_element_fts_resolves_coordinates_once_per_partition() {
    // Past the bound the merge compacts after every partition, and each
    // compaction re-resolves every merged candidate. Reading the coordinate
    // columns from the docs file each time is quadratic in the partition count,
    // so they are cached.
    const PARTITIONS: u64 = 8;
    const PER_PARTITION: u64 = 100;
    let tmpdir = TempObjDir::default();
    let inner = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    for partition_id in 0..PARTITIONS {
        let elements = (0..PER_PARTITION)
            .map(|element| (partition_id * 1_000 + element, element as u32))
            .collect::<Vec<_>>();
        build_tied_element_partition(&inner, partition_id, &elements).await;
    }
    write_test_metadata(
        &inner,
        (0..PARTITIONS).collect(),
        InvertedIndexParams::default(),
    )
    .await;

    let counter = Arc::new(CoordinateReadCounter::default());
    let store: Arc<dyn IndexStore> = Arc::new(WrappingReaderStore {
        inner: inner.clone(),
        wrapper: Arc::new(CoordinateReads {
            counter: counter.clone(),
        }),
    });
    let cache = LanceCache::with_capacity(64 * 1024 * 1024);
    let index = InvertedIndex::load(store, None, &cache).await.unwrap();

    let documents = search_alpha_documents(&index, 4).await;
    assert_eq!(
        documents,
        vec![(0, vec![0]), (1, vec![1]), (2, vec![2]), (3, vec![3])],
        "the exact lowest row_ids, ordered by row_id then doc_index"
    );
    let reads = counter.reads.load(Ordering::Relaxed);
    assert!(
        reads <= PARTITIONS as usize,
        "coordinates must be read once per partition, got {reads} reads for {PARTITIONS} partitions"
    );
}

#[tokio::test]
async fn test_fts_topk_tied_scores_retry_every_overflowed_partition() {
    // Five partitions of 200 documents with addresses striped across them: each
    // winner sits in a different partition, and every partition overflows its
    // band at once. A broad single-term query over quantized doc lengths does
    // that, and all five retries then run concurrently over resolved addresses.
    const PARTITIONS: u64 = 5;
    const PER_PARTITION: u64 = 200;
    const BASE: u64 = 10_000;
    let partitions = (0..PARTITIONS)
        .map(|partition_id| {
            // Row addresses descend with DocId, so the winners sit at the end of
            // each walk, past the point where the band overflows.
            let row_ids = (0..PER_PARTITION)
                .map(|doc_id| BASE + partition_id + (PER_PARTITION - doc_id) * PARTITIONS)
                .collect::<Vec<_>>();
            (partition_id, row_ids)
        })
        .collect::<Vec<_>>();

    // Each partition's lowest address belongs to its last DocId.
    let expected = (0..PARTITIONS)
        .map(|partition_id| BASE + partition_id + PARTITIONS)
        .collect::<Vec<_>>();
    assert_eq!(
        expected,
        ascending_row_ids(&partitions)[..PARTITIONS as usize]
    );

    assert_tied_top_k_is_an_ordered_prefix(
        &partitions,
        &[5, 3],
        Band::CollectorOverruns,
        FilterPath::Unfiltered,
        Residency::Roomy,
        1,
    )
    .await;
}
