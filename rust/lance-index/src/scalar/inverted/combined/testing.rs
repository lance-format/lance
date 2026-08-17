// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared test fixtures for the `combined_fields` submodules: term cursors,
//! document views, index and flat-scan corpora, and the exact reference scans
//! the tests compare against.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_array::{ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_core::Result;
use lance_core::cache::{LanceCache, WeakLanceCache};
use lance_core::error::DataFusionResult;
use lance_core::utils::tempfile::TempObjDir;
use lance_io::object_store::ObjectStore;
use lance_select::RowAddrTreeMap;
use roaring::RoaringTreemap;

use super::super::builder::{BLOCK_SIZE, InnerBuilder, PositionRecorder};
use super::super::documents::{AddressKeyedDocuments, PartitionDocuments};
use super::super::encoding::{
    MAX_POSTING_BLOCK_SIZE, compress_posting_list_with_tail_codec_and_block_size,
};
use super::super::index::{
    CompressedPostingList, FTS_FORMAT_VERSION_KEY, InvertedIndex, InvertedListFormatVersion,
    METADATA_FILE, NUM_TOKEN_COL, POSTING_BLOCK_SIZE_KEY, POSTING_TAIL_CODEC_KEY,
    PostingListBuilder, PostingTailCodec, TOKEN_SET_FORMAT_KEY, TokenSetFormat,
};
use super::super::query::{FtsSearchParams, Operator, Tokens};
use super::super::tokenizer::document_tokenizer::DocType;
use super::super::tokenizer::{InvertedIndexParams, LEGACY_BLOCK_SIZE};
use super::{
    CombinedCorpusStats, CombinedFieldColumn, build_combined_bm25_scorer, combined_fields_search,
};
use crate::metrics::NoOpMetricsCollector;
use crate::prefilter::NoFilter;
use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::{IndexStore, RowIdRemapper};

pub(super) fn compressed_list(postings: &[(u32, u32)]) -> CompressedPostingList {
    let doc_ids: Vec<u32> = postings.iter().map(|(d, _)| *d).collect();
    let freqs: Vec<u32> = postings.iter().map(|(_, f)| *f).collect();
    let blocks = compress_posting_list_with_tail_codec_and_block_size(
        doc_ids.len(),
        doc_ids.iter(),
        freqs.iter(),
        std::iter::repeat(0.0f32),
        PostingTailCodec::VarintDelta,
        BLOCK_SIZE,
    )
    .unwrap();
    CompressedPostingList::new(
        blocks,
        1.0,
        doc_ids.len() as u32,
        PostingTailCodec::VarintDelta,
        BLOCK_SIZE,
        None,
        None,
    )
}

#[derive(Debug)]
struct DeletedRows(Vec<u64>);

impl RowIdRemapper for DeletedRows {
    fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        (!self.0.contains(&row_id)).then_some(row_id)
    }

    fn remap_row_addrs_tree_map(&self, _: &RowAddrTreeMap) -> RowAddrTreeMap {
        unreachable!("document fixtures only remap single row ids")
    }

    fn remap_row_ids_roaring_tree_map(&self, _: &RoaringTreemap) -> RoaringTreemap {
        unreachable!("document fixtures only remap single row ids")
    }

    fn remap_row_ids_record_batch(&self, _: RecordBatch, _: usize) -> Result<RecordBatch> {
        unreachable!("document fixtures only remap single row ids")
    }
}

/// Modern identity documents, written to a real store and loaded through
/// [`PartitionDocuments::address_keyed`] so the fixture is the representation
/// production scores against rather than a stand-in.
///
/// `dead_rows` are addresses a fragment-reuse remapper has deleted: the slot
/// survives (posting lists key on its DocId positionally) but
/// `row_address` answers [`RowAddress::TOMBSTONE_ROW`] for it. Every part of
/// the returned view is resident, so the temporary store is dropped here.
pub(super) async fn modern_identity_docs(
    num_tokens: &[u32],
    dead_rows: &[u64],
) -> AddressKeyedDocuments {
    const DOCS_PATH: &str = "combined_fixture_docs.lance";
    let tmpdir = TempObjDir::default();
    let cache = Arc::new(LanceCache::no_cache());
    let store: Arc<dyn IndexStore> = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        cache.clone(),
    ));
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new(lance_core::ROW_ID, arrow_schema::DataType::UInt64, false),
        arrow_schema::Field::new(NUM_TOKEN_COL, arrow_schema::DataType::UInt32, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(
                (0..num_tokens.len() as u64).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt32Array::from(num_tokens.to_vec())) as ArrayRef,
        ],
    )
    .unwrap();
    let mut writer = store.new_index_file(DOCS_PATH, schema).await.unwrap();
    writer.write_record_batch(batch).await.unwrap();
    writer.finish().await.unwrap();

    let reader = store.open_index_file(DOCS_PATH).await.unwrap();
    let remapper: Option<Arc<dyn RowIdRemapper>> = (!dead_rows.is_empty())
        .then(|| Arc::new(DeletedRows(dead_rows.to_vec())) as Arc<dyn RowIdRemapper>);
    PartitionDocuments::try_new(
        store.clone(),
        DOCS_PATH.to_owned(),
        0,
        WeakLanceCache::from(cache.as_ref()),
        reader.as_ref(),
        remapper,
        false,
    )
    .unwrap()
    .address_keyed()
    .await
    .unwrap()
}

/// `_rowid` plus one `Utf8` document column per entry of `docs`
/// (`docs[column][row]`), delivered as `num_batches` equal batches.
pub(super) fn flat_input(
    row_ids: &[u64],
    docs: &[Vec<Option<&str>>],
    num_batches: usize,
) -> SendableRecordBatchStream {
    use arrow_array::StringArray;
    use arrow_schema::{DataType, Field, Schema};

    let mut fields = vec![lance_core::ROW_ID_FIELD.clone()];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(UInt64Array::from(row_ids.to_vec()))];
    for (slot, column) in docs.iter().enumerate() {
        fields.push(Field::new(format!("doc{slot}"), DataType::Utf8, true));
        arrays.push(Arc::new(StringArray::from(column.clone())));
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
    // No rows means no batches at all, the way an empty scan arrives.
    let rows_per_batch = row_ids.len().div_ceil(num_batches.max(1));
    let batches = (0..row_ids.len().div_ceil(rows_per_batch.max(1)))
        .map(|i| {
            let start = i * rows_per_batch;
            let len = (row_ids.len() - start).min(rows_per_batch);
            DataFusionResult::Ok(batch.slice(start, len))
        })
        .collect::<Vec<_>>();
    Box::pin(RecordBatchStreamAdapter::new(schema, stream::iter(batches)))
}

pub(super) fn flat_columns(weights: &[f32]) -> Vec<CombinedFieldColumn> {
    weights
        .iter()
        .enumerate()
        .map(|(slot, weight)| CombinedFieldColumn {
            column: format!("doc{slot}"),
            weight: *weight,
            indices: Vec::new(),
        })
        .collect()
}

/// Every `(row_id, score)` the flat scan emits, in emission order, with the
/// score as raw bits so the comparison is exact.
pub(super) async fn flat_scores(
    stream: SendableRecordBatchStream,
) -> (Vec<usize>, Vec<(u64, u32)>) {
    use futures::TryStreamExt;

    let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
    let sizes = batches.iter().map(|batch| batch.num_rows()).collect();
    let scored = batches
        .iter()
        .flat_map(|batch| {
            let row_ids = batch.column(0).as_primitive::<UInt64Type>();
            let scores = batch.column(1).as_primitive::<Float32Type>();
            (0..batch.num_rows())
                .map(|i| (row_ids.value(i), scores.value(i).to_bits()))
                .collect::<Vec<_>>()
        })
        .collect();
    (sizes, scored)
}

// Legacy element-per-document corpus statistics.
//
// Released V1/V2 indexes indexed every `List<String>` element as its own document,
// so one row there owns a run of documents. `combined_fields` scores at row level,
// so `docCount'` / `docFreq'` must be row level too or `idf'` and `avgdl'` describe
// a different corpus than the frequencies they divide. The current builder writes
// one document per row, so these fixtures write the legacy partition files
// directly, and assert that they really are element-per-document so the coverage
// cannot lapse.

/// A fixture corpus: `rows[row][element]` is one indexed element's tokens.
pub(super) type ElementRows<'a> = Vec<Vec<Vec<&'a str>>>;

/// Build a one-partition FTS index whose documents are the individual
/// elements of `rows`, each carrying its row's id (row `r` has row id `r`).
/// `vocab` fixes the token order; tokens absent from `rows` are left out of
/// the token set, as the builder leaves them out. Elements that tokenize to
/// nothing are skipped, also as the builder does.
pub(super) async fn element_document_index(
    format_version: InvertedListFormatVersion,
    vocab: &[&str],
    rows: &ElementRows<'_>,
) -> (Arc<InvertedIndex>, TempObjDir) {
    let vocab: Vec<&str> = vocab
        .iter()
        .copied()
        .filter(|token| rows.iter().flatten().any(|element| element.contains(token)))
        .collect();
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    let block_size = match format_version {
        InvertedListFormatVersion::V3 => MAX_POSTING_BLOCK_SIZE,
        InvertedListFormatVersion::V1 | InvertedListFormatVersion::V2 => LEGACY_BLOCK_SIZE,
    };
    let posting_tail_codec = format_version.posting_tail_codec();
    let mut builder = InnerBuilder::new_with_format_version_and_block_size(
        0,
        false,
        TokenSetFormat::default(),
        format_version,
        block_size,
    );
    let mut postings: Vec<PostingListBuilder> = vocab
        .iter()
        .map(|token| {
            builder.tokens.add((*token).to_owned());
            PostingListBuilder::new_with_posting_tail_codec_and_block_size(
                false,
                posting_tail_codec,
                block_size,
            )
        })
        .collect();
    let mut doc_id = 0u32;
    for (row, elements) in rows.iter().enumerate() {
        for element in elements {
            if element.is_empty() {
                continue;
            }
            for (token_id, token) in vocab.iter().enumerate() {
                let freq = element.iter().filter(|t| *t == token).count() as u32;
                if freq > 0 {
                    postings[token_id].add(doc_id, PositionRecorder::Count(freq));
                }
            }
            builder.docs.append(row as u64, element.len() as u32);
            doc_id += 1;
        }
    }
    builder.set_posting_lists(postings);
    builder.write(store.as_ref()).await.unwrap();

    let params = InvertedIndexParams::default()
        .block_size(block_size)
        .unwrap();
    let metadata = HashMap::from([
        (
            "partitions".to_owned(),
            serde_json::to_string(&vec![0u64]).unwrap(),
        ),
        ("params".to_owned(), serde_json::to_string(&params).unwrap()),
        (
            TOKEN_SET_FORMAT_KEY.to_owned(),
            TokenSetFormat::default().to_string(),
        ),
        (
            POSTING_TAIL_CODEC_KEY.to_owned(),
            posting_tail_codec.as_str().to_owned(),
        ),
        (
            FTS_FORMAT_VERSION_KEY.to_owned(),
            format_version.index_version().to_string(),
        ),
        (POSTING_BLOCK_SIZE_KEY.to_owned(), block_size.to_string()),
    ]);
    let mut writer = store
        .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
        .await
        .unwrap();
    writer.finish_with_metadata(metadata).await.unwrap();

    let index = InvertedIndex::load(store, None, &LanceCache::no_cache())
        .await
        .unwrap();
    (index, tmpdir)
}

/// The same corpus as one document per row: a row's elements joined into a
/// single document, which is what the current builder writes for a
/// `List<String>` column.
pub(super) fn as_row_documents<'a>(rows: &ElementRows<'a>) -> ElementRows<'a> {
    rows.iter()
        .map(|elements| {
            let joined: Vec<&str> = elements.iter().flatten().copied().collect();
            if joined.is_empty() {
                Vec::new()
            } else {
                vec![joined]
            }
        })
        .collect()
}

pub(super) fn combined_columns(indices: Vec<Arc<InvertedIndex>>) -> Vec<CombinedFieldColumn> {
    indices
        .into_iter()
        .enumerate()
        .map(|(slot, index)| CombinedFieldColumn {
            column: format!("col{slot}"),
            weight: 1.0,
            indices: vec![index],
        })
        .collect()
}

/// Run a `combined_fields` top-k over `columns`, returning `(row_id, score)`.
pub(super) async fn combined_top_k(
    columns: &[CombinedFieldColumn],
    terms: &[&str],
    limit: usize,
) -> Vec<(u64, f32)> {
    let tokens = Tokens::new(
        terms.iter().map(|t| (*t).to_owned()).collect(),
        DocType::Text,
    );
    let scorer = build_combined_bm25_scorer(columns, &tokens, CombinedCorpusStats::IndexOnly, None)
        .await
        .unwrap();
    let params = FtsSearchParams::new().with_limit(Some(limit));
    let (row_ids, scores) = combined_fields_search(
        columns,
        &tokens,
        &params,
        Operator::Or,
        &scorer,
        Arc::new(NoFilter),
        &NoOpMetricsCollector,
    )
    .await
    .unwrap();
    row_ids.into_iter().zip(scores).collect()
}
