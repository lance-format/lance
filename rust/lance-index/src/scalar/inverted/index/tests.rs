// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::scalar::inverted::document_tokenizer::DocType;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::stream;
use lance_core::cache::{LanceCache, QuickCacheBackend};
use lance_core::utils::tempfile::TempObjDir;
use lance_io::object_store::ObjectStore;

use crate::metrics::{LocalMetricsCollector, NoOpMetricsCollector};
use crate::prefilter::NoFilter;
use crate::scalar::ScalarIndex;
use crate::scalar::inverted::builder::{
    BLOCK_SIZE, InnerBuilder, InvertedIndexBuilder, PositionRecorder, doc_file_path,
    inverted_list_schema, inverted_list_schema_for_version_with_block_size,
    inverted_list_schema_for_version_with_block_size_and_impacts, posting_file_path,
    token_file_path,
};
use crate::scalar::inverted::encoding::{
    compress_positions, compress_posting_list_with_tail_codec,
    compress_posting_list_with_tail_codec_and_block_size, decompress_posting_list_with_tail_codec,
    encode_position_stream_block_into,
};
use crate::scalar::inverted::query::{FtsSearchParams, Operator};
use crate::scalar::lance_format::LanceIndexStore;
use arrow::array::{
    AsArray, GenericListBuilder, GenericStringBuilder, Int32Builder, LargeBinaryBuilder,
    ListBuilder, UInt32Builder,
};
use arrow::datatypes::{Float32Type, UInt32Type};
use arrow_array::{ArrayRef, Float32Array, RecordBatch, StringArray, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::scalar::inverted::tokenizer::document_tokenizer::TextTokenizer;
use lance_tokenizer::{Language, SimpleTokenizer, StopWordFilter, TextAnalyzer};

use super::*;

const TEST_GROUP_TOKENS: usize = 128;

async fn write_single_partition_index(
    store: Arc<LanceIndexStore>,
    params: InvertedIndexParams,
    token_set_format: TokenSetFormat,
    token: &str,
    row_id: u64,
) -> Result<Arc<InvertedIndex>> {
    let block_size = params.posting_block_size();
    let format_version = params.resolved_format_version();
    let mut partition = InnerBuilder::new_with_format_version_and_block_size(
        0,
        false,
        token_set_format,
        format_version,
        block_size,
    );
    partition.tokens.get_or_add(token);
    let mut posting_list = PostingListBuilder::new_with_posting_tail_codec_and_block_size(
        false,
        format_version.posting_tail_codec(),
        block_size,
    );
    posting_list.add(0, PositionRecorder::Count(1));
    partition.posting_lists.push(posting_list);
    partition.docs.append(row_id, 1);
    partition.write(store.as_ref()).await?;

    let metadata = HashMap::from([
        (
            "partitions".to_owned(),
            serde_json::to_string(&vec![0_u64]).unwrap(),
        ),
        ("params".to_owned(), serde_json::to_string(&params).unwrap()),
        (
            TOKEN_SET_FORMAT_KEY.to_owned(),
            token_set_format.to_string(),
        ),
        (
            POSTING_TAIL_CODEC_KEY.to_owned(),
            format_version.posting_tail_codec().as_str().to_owned(),
        ),
        (
            FTS_FORMAT_VERSION_KEY.to_owned(),
            format_version.index_version().to_string(),
        ),
        (POSTING_BLOCK_SIZE_KEY.to_owned(), block_size.to_string()),
    ]);
    let mut writer = store
        .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
        .await?;
    writer.finish_with_metadata(metadata).await?;

    InvertedIndex::load(store, None, &LanceCache::no_cache()).await
}

fn empty_doc_stream() -> SendableRecordBatchStream {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc", DataType::Utf8, true),
        Field::new(ROW_ID, DataType::UInt64, false),
    ]));
    Box::pin(RecordBatchStreamAdapter::new(
        schema,
        stream::iter(Vec::<datafusion_common::Result<RecordBatch>>::new()),
    ))
}

async fn write_test_metadata(
    store: &Arc<LanceIndexStore>,
    partition_ids: Vec<u64>,
    params: InvertedIndexParams,
) {
    let format_version = params.resolved_format_version();
    let metadata = HashMap::from([
        (
            "partitions".to_owned(),
            serde_json::to_string(&partition_ids).unwrap(),
        ),
        ("params".to_owned(), serde_json::to_string(&params).unwrap()),
        (
            TOKEN_SET_FORMAT_KEY.to_owned(),
            TokenSetFormat::default().to_string(),
        ),
        (
            POSTING_TAIL_CODEC_KEY.to_owned(),
            format_version.posting_tail_codec().as_str().to_owned(),
        ),
        (
            FTS_FORMAT_VERSION_KEY.to_owned(),
            format_version.index_version().to_string(),
        ),
        (
            POSTING_BLOCK_SIZE_KEY.to_owned(),
            params.posting_block_size().to_string(),
        ),
    ]);
    let mut writer = store
        .new_index_file(METADATA_FILE, Arc::new(arrow_schema::Schema::empty()))
        .await
        .unwrap();
    writer.finish_with_metadata(metadata).await.unwrap();
}

/// Rewrites the reader a [`WrappingReaderStore`] opens for each index file.
pub(super) trait ReaderWrapper: std::fmt::Debug + Send + Sync + 'static {
    fn wrap(&self, name: &str, reader: Arc<dyn IndexReader>) -> Arc<dyn IndexReader>;
}

/// Index store that hands every opened file to a [`ReaderWrapper`].
#[derive(Debug)]
pub(super) struct WrappingReaderStore {
    pub(super) inner: Arc<dyn IndexStore>,
    pub(super) wrapper: Arc<dyn ReaderWrapper>,
}

#[cfg_attr(coverage, coverage(off))]
impl DeepSizeOf for WrappingReaderStore {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.inner.deep_size_of_children(context)
    }
}

#[cfg_attr(coverage, coverage(off))]
#[async_trait]
impl IndexStore for WrappingReaderStore {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_arc(&self) -> Arc<dyn IndexStore> {
        Arc::new(Self {
            inner: self.inner.clone(),
            wrapper: self.wrapper.clone(),
        })
    }
    fn io_parallelism(&self) -> usize {
        self.inner.io_parallelism()
    }
    fn with_io_priority(&self, io_priority: u64) -> Arc<dyn IndexStore> {
        Arc::new(Self {
            inner: self.inner.with_io_priority(io_priority),
            wrapper: self.wrapper.clone(),
        })
    }
    fn with_io_buffer_size(&self, bytes: u64) -> Arc<dyn IndexStore> {
        Arc::new(Self {
            inner: self.inner.with_io_buffer_size(bytes),
            wrapper: self.wrapper.clone(),
        })
    }
    async fn new_index_file(
        &self,
        name: &str,
        schema: Arc<arrow_schema::Schema>,
    ) -> Result<Box<dyn crate::scalar::IndexWriter>> {
        self.inner.new_index_file(name, schema).await
    }
    async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>> {
        let reader = self.inner.open_index_file(name).await?;
        Ok(self.wrapper.wrap(name, reader))
    }
    async fn copy_index_file(
        &self,
        name: &str,
        dest_store: &dyn IndexStore,
    ) -> Result<crate::scalar::IndexFile> {
        self.inner.copy_index_file(name, dest_store).await
    }
    async fn copy_index_file_to(
        &self,
        name: &str,
        new_name: &str,
        dest_store: &dyn IndexStore,
    ) -> Result<crate::scalar::IndexFile> {
        self.inner
            .copy_index_file_to(name, new_name, dest_store)
            .await
    }
    async fn rename_index_file(
        &self,
        name: &str,
        new_name: &str,
    ) -> Result<crate::scalar::IndexFile> {
        self.inner.rename_index_file(name, new_name).await
    }
    async fn delete_index_file(&self, name: &str) -> Result<()> {
        self.inner.delete_index_file(name).await
    }
    async fn list_files_with_sizes(&self) -> Result<Vec<crate::scalar::IndexFile>> {
        self.inner.list_files_with_sizes().await
    }
}

mod flat_search;
mod format_and_builder;
mod grouping;
mod lifecycle;
mod position_cache;
mod prewarm_cache;
mod query;
mod scoring;
mod stats;
mod topk_tiebreak;
