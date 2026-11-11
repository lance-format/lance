// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_index_core::remapping::RowAddrTranslator;
use std::{
    ops::Bound,
    sync::{Arc, Mutex},
};

use arrow_array::{Array, ArrayRef, LargeBinaryArray, RecordBatch, StructArray, UInt8Array};
use arrow_schema::{DataType, Field, Schema, SortOptions};
use async_trait::async_trait;
use datafusion_common::{DataFusionError, ScalarValue, config::ConfigOptions};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::{Expr, Operator, ScalarUDF};
use datafusion_physical_expr::{
    PhysicalExpr, PhysicalSortExpr, ScalarFunctionExpr,
    expressions::{Column, Literal},
};
use datafusion_physical_plan::{
    ExecutionPlan, projection::ProjectionExec, sorts::sort::SortExec,
    stream::RecordBatchStreamAdapter,
};
use futures::{StreamExt, stream::BoxStream};
use lance_arrow::json::{JsonEncoding, JsonValues, json_field};
use lance_core::deepsize::DeepSizeOf;
use lance_datafusion_exec::exec::{LanceExecutionOptions, OneShotExec, execute_plan};
use lance_datafusion_exec::udf::json::JsonbType;
use prost::Message;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use lance_core::{
    Error, Result,
    cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache},
    error::LanceOptionExt,
};

use crate::{
    Index, IndexType,
    metrics::MetricsCollector,
    registry::IndexPluginRegistry,
    scalar::{
        AnyQuery, CreatedIndex, IndexStore, RowIdRemapper, ScalarIndex, SearchOptions,
        SearchResult, UpdateCriteria,
        expression::{IndexedExpression, ScalarIndexExpr, ScalarIndexSearch, ScalarQueryParser},
        registry::{
            BasicTrainer, ScalarIndexLoad, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering,
            TrainingRequest, VALUE_COLUMN_NAME,
        },
    },
};

/// A JSON segment whose target index stores row ids (or, for details written
/// before `JsonIndexDetails::target_index_version` existed, whose target's domain
/// is whatever a build from that time wrote; see `legacy_target_index_version`).
pub const JSON_ROW_ID_DOMAIN_VERSION: u32 = 0;

/// A JSON segment whose target index stores physical row addresses.
///
/// The wrapper's own version carries its target's domain so that
/// `IndexMetadata::results_are_row_addrs` (which only sees the wrapper's details
/// and version) can classify the segment, and so that a build predating this
/// version, which would load the target as row-id domain, ignores the segment
/// instead of misreading it.
pub const JSON_ROW_ADDR_DOMAIN_VERSION: u32 = 1;

const JSON_INDEX_VERSION: u32 = JSON_ROW_ADDR_DOMAIN_VERSION;

/// The version a build predating `JsonIndexDetails::target_index_version` used
/// when loading the target, for the targets whose format has changed since.
///
/// Those builds loaded the target at the target plugin's then-current version,
/// which for BTree, Bitmap, and LabelList was the row-id-domain format. `None`
/// for every other target, whose format has not changed since: its plugin's
/// current version is still correct.
fn legacy_target_index_version(target_details: &prost_types::Any) -> Option<u32> {
    if target_details.type_url.ends_with("BTreeIndexDetails") {
        Some(crate::scalar::btree::BTREE_ROW_ID_DOMAIN_VERSION)
    } else if target_details.type_url.ends_with("BitmapIndexDetails") {
        Some(crate::scalar::bitmap::BITMAP_ROW_ID_DOMAIN_VERSION)
    } else if target_details.type_url.ends_with("LabelListIndexDetails") {
        Some(crate::scalar::label_list::LABEL_LIST_ROW_ID_DOMAIN_VERSION)
    } else {
        None
    }
}

/// Whether `index_details` belong to a JSON index whose freshly trained segments
/// are [`JSON_ROW_ADDR_DOMAIN_VERSION`], so that a segment of it written before
/// that version is classified as row-id domain while a fresh one is not.
///
/// That covers targets that moved from row ids to row addresses (BTree, Bitmap,
/// LabelList), whose legacy segments really store row ids, and targets that
/// always stored row addresses (ZoneMap, BloomFilter, FM), whose legacy
/// segments store addresses but carry the version-0 wrapper that
/// `IndexMetadata::results_are_row_addrs` reads as row ids. Either way such a
/// segment cannot share a named index with a fresh one and must be rebuilt.
pub fn wraps_row_addr_domain_target(index_details: &prost_types::Any) -> bool {
    index_details.type_url.ends_with("JsonIndexDetails")
        && crate::pb::JsonIndexDetails::decode(index_details.value.as_slice())
            .ok()
            .and_then(|details| details.target_details)
            .is_some_and(|target_details| {
                legacy_target_index_version(&target_details).is_some()
                    || lance_table::format::index_details_are_row_addrs(&target_details, 0)
            })
}

/// The JSON segment version that matches the domain of a target persisted with
/// `target_details` at `target_index_version`.
fn json_index_version_for_target(
    target_details: &prost_types::Any,
    target_index_version: u32,
) -> u32 {
    if lance_table::format::index_details_are_row_addrs(target_details, target_index_version as i32)
    {
        JSON_ROW_ADDR_DOMAIN_VERSION
    } else {
        JSON_ROW_ID_DOMAIN_VERSION
    }
}

/// Relabel a freshly trained JSON segment in the pre-migration row-id-domain
/// format.
///
/// Only valid on a dataset without stable row ids, where a row address and a row
/// id are the same value, so the target's trained data is identical in either
/// domain. Mirrors the plain BTree/Bitmap/LabelList relabeling in
/// `build_scalar_index`, and keeps such segments readable by builds that
/// predate [`JSON_ROW_ADDR_DOMAIN_VERSION`].
pub fn relabel_as_row_id_domain(created: &mut CreatedIndex) -> Result<()> {
    let mut details = crate::pb::JsonIndexDetails::decode(created.index_details.value.as_slice())?;
    let target_details = details.target_details.as_ref().expect_ok()?;
    if let Some(row_id_version) = legacy_target_index_version(target_details) {
        details.target_index_version = Some(row_id_version);
    }
    created.index_details = prost_types::Any::from_msg(&details)?;
    created.index_version = JSON_ROW_ID_DOMAIN_VERSION;
    Ok(())
}

/// A JSON index that indexes a field in a JSON column
///
/// The underlying index can be any other type of scalar index
#[derive(Debug)]
pub struct JsonIndex {
    target_index: Arc<dyn ScalarIndex>,
    path: String,
    /// The `JsonTargetDataType` this index's details recorded, held as the raw
    /// enum value so a variant this build does not recognize survives a `remap`
    /// or `update` performed by this build instead of being erased.
    ///
    /// `Unspecified` for an index built before the field existed.
    recorded_target_data_type: i32,
    target_plugin_name: Option<String>,
}

impl JsonIndex {
    pub fn new(target_index: Arc<dyn ScalarIndex>, path: String) -> Self {
        Self {
            target_index,
            path,
            recorded_target_data_type: crate::pb::JsonTargetDataType::Unspecified as i32,
            target_plugin_name: None,
        }
    }

    /// Record the `JsonTargetDataType` this index's details carried.
    ///
    /// Takes the raw value straight from the decoded details rather than a
    /// parsed enum, so a value written by a newer build is carried forward
    /// untouched.
    pub fn with_target_data_type(mut self, target_data_type: i32) -> Self {
        self.recorded_target_data_type = target_data_type;
        self
    }

    /// The raw `JsonTargetDataType` to write into details this index produces.
    ///
    /// Prefers what the details already recorded, so a variant this build does
    /// not recognize survives. Falls back to asking the target index, which
    /// upgrades an index built before the field existed: total on the `update`
    /// path, which already requires the target to report its training type, and
    /// BTree-only on the `remap` path, where an index that cannot answer is left
    /// no worse off than before.
    ///
    /// Deliberately infallible: a type with no representation here is recorded
    /// as `Unspecified`, leaving details exactly as lossy as they already were,
    /// rather than failing the compaction that produced them.
    fn target_data_type_to_record(&self) -> i32 {
        if self.recorded_target_data_type != crate::pb::JsonTargetDataType::Unspecified as i32 {
            return self.recorded_target_data_type;
        }
        self.target_index
            .training_data_type()
            .as_ref()
            .and_then(|data_type| JsonIndexTargetType::try_from(data_type).ok())
            .map(|target_type| target_type.to_pb() as i32)
            .unwrap_or(crate::pb::JsonTargetDataType::Unspecified as i32)
    }

    /// The type the target index was trained on, or `None` when it cannot be
    /// recovered and a rebuild has to infer it from the data again.
    ///
    /// Prefers the value the details recorded. Indices built before that field
    /// existed record `Unspecified`, and so does a value this build does not
    /// recognize, so both fall back to asking the target index -- which only
    /// BTree answers.
    ///
    /// Unlike [`Self::target_data_type_to_record`], this errors rather than
    /// degrading when the target reports an Arrow type with no representation
    /// here, because a rebuild driven by the result must not silently use the
    /// wrong type.
    fn resolve_target_data_type(&self) -> Result<Option<JsonIndexTargetType>> {
        if let Some(recorded) = JsonIndexTargetType::from_pb(self.recorded_target_data_type) {
            return Ok(Some(recorded));
        }
        self.target_index
            .training_data_type()
            .as_ref()
            .map(JsonIndexTargetType::try_from)
            .transpose()
    }
}

impl DeepSizeOf for JsonIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.target_index.deep_size_of_children(context)
            + self.path.deep_size_of_children(context)
            + self.target_plugin_name.deep_size_of_children(context)
    }
}

/// Only the wrapper metadata is retained here. The target plugin owns its cache
/// representation and reconstructs it against each request's store/remapper.
#[derive(Debug, DeepSizeOf)]
struct JsonIndexState {
    path: String,
    target_data_type: i32,
    target_plugin_name: String,
}

impl JsonIndexState {
    fn from_index(index: &dyn ScalarIndex) -> Result<Self> {
        let index = index
            .as_any()
            .downcast_ref::<JsonIndex>()
            .ok_or_else(|| Error::internal("JSON cache received a non-JSON index"))?;
        Ok(Self {
            path: index.path.clone(),
            target_data_type: index.recorded_target_data_type,
            target_plugin_name: index
                .target_plugin_name
                .clone()
                .ok_or_else(|| Error::internal("JSON cache index has no target plugin"))?,
        })
    }

    fn wrap(&self, target_index: Arc<dyn ScalarIndex>) -> Arc<dyn ScalarIndex> {
        Arc::new(JsonIndex {
            target_index,
            path: self.path.clone(),
            recorded_target_data_type: self.target_data_type,
            target_plugin_name: Some(self.target_plugin_name.clone()),
        })
    }
}

struct JsonIndexStateKey;

impl CacheKey for JsonIndexStateKey {
    type ValueType = JsonIndexState;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        "state".into()
    }

    fn type_name() -> &'static str {
        "JsonIndexState"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.json-index-state-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_variant(0);
    }
}

#[async_trait]
impl Index for JsonIndex {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn index_type(&self) -> IndexType {
        // TODO: This causes the index to appear as btree in list_indices call.  Need better logic
        // in list_indices to use details instead of index_type.
        IndexType::Scalar
    }

    async fn prewarm(&self) -> Result<()> {
        self.target_index.prewarm().await
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        self.target_index.statistics()
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        self.target_index.calculate_included_frags().await
    }
}

impl JsonIndex {
    /// The JSON index over a rewritten target index.
    fn wrap_target(&self, target_created: CreatedIndex) -> Result<CreatedIndex> {
        let index_version = json_index_version_for_target(
            &target_created.index_details,
            target_created.index_version,
        );
        let json_details = crate::pb::JsonIndexDetails {
            path: self.path.clone(),
            target_details: Some(target_created.index_details),
            target_data_type: self.target_data_type_to_record(),
            target_index_version: Some(target_created.index_version),
        };
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&json_details)?,
            index_version,
            files: target_created.files,
        })
    }
}

#[async_trait]
impl ScalarIndex for JsonIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        self.search_with_options(query, SearchOptions::default(), metrics)
            .await
    }

    async fn search_with_options(
        &self,
        query: &dyn AnyQuery,
        options: SearchOptions,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<JsonQuery>().unwrap();
        self.target_index
            .search_with_options(query.target_query.as_ref(), options, metrics)
            .await
    }

    fn lookup<'a>(
        &'a self,
        keys: ArrayRef,
        metrics: &'a dyn MetricsCollector,
    ) -> Result<BoxStream<'a, Result<RecordBatch>>> {
        self.target_index.lookup(keys, metrics)
    }

    fn results_are_row_addresses(&self) -> bool {
        self.target_index.results_are_row_addresses()
    }

    #[allow(deprecated)]
    fn can_remap(&self) -> bool {
        self.target_index.can_remap()
    }

    async fn remap(
        &self,
        mapping: &RowAddrRemap,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let target_created = self.target_index.remap(mapping, dest_store).await?;
        self.wrap_target(target_created)
    }

    async fn remap_streaming(
        &self,
        translator: &RowAddrTranslator,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let target_created = self
            .target_index
            .remap_streaming(translator, dest_store)
            .await?;
        self.wrap_target(target_created)
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        old_data_filter: Option<super::OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        let target_criteria = self.target_index.update_criteria().data_criteria;
        let target_type = self.target_index.training_data_type().ok_or_else(|| {
            Error::not_supported(format!(
                "JSON index updates for path '{}' require target index {} to report its training data type",
                self.path,
                self.target_index.index_type()
            ))
        })?;
        let new_data = JsonIndexPlugin::extract_json(new_data, self.path.clone())?;
        let new_data =
            JsonIndexPlugin::convert_stream_by_type(new_data, target_type, self.path.clone())?;
        let new_data = if target_criteria.ordering == TrainingOrdering::Values {
            JsonIndexPlugin::sort_stream_by_value(new_data).await?
        } else {
            new_data
        };
        let target_created = self
            .target_index
            .update(new_data, dest_store, old_data_filter)
            .await?;
        self.wrap_target(target_created)
    }

    fn update_criteria(&self) -> UpdateCriteria {
        let target_criteria = self.target_index.update_criteria();
        UpdateCriteria {
            requires_old_data: target_criteria.requires_old_data,
            data_criteria: json_scan_criteria(&target_criteria.data_criteria),
        }
    }

    fn derive_index_params(&self) -> Result<super::ScalarIndexParams> {
        let target_params = self.target_index.derive_index_params()?;
        let target_data_type = self.resolve_target_data_type()?;
        let params = JsonIndexParameters {
            target_index_type: target_params.index_type,
            target_index_parameters: target_params.params,
            target_data_type,
            path: self.path.clone(),
        };
        Ok(super::ScalarIndexParams::new("json".to_string()).with_params(&params))
    }

    fn training_data_type(&self) -> Option<DataType> {
        self.target_index.training_data_type()
    }
}

/// Parameters for a [`JsonIndex`]
#[derive(Debug, Serialize, Deserialize)]
pub struct JsonIndexParameters {
    target_index_type: String,
    target_index_parameters: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_data_type: Option<JsonIndexTargetType>,
    path: String,
}

/// The target index type named by a JSON index's own serialized parameters
/// (the `params` field of the `ScalarIndexParams` that `JsonIndex::derive_index_params`
/// returns -- not the target's own params). `None` if `params_json` doesn't parse as
/// [`JsonIndexParameters`].
#[deprecated(note = "no longer used by Lance; parse `JsonIndexParameters` directly")]
pub fn json_wrapped_target_type(params_json: &str) -> Option<String> {
    serde_json::from_str::<JsonIndexParameters>(params_json)
        .ok()
        .map(|parameters| parameters.target_index_type)
}

/// The Arrow type a JSON index decodes the value at its path into, and trains
/// its target index on.
///
/// `LargeBinary` is retained only to read parameters derived from legacy JSON
/// indices whose arrays or objects were indexed as binary values. Current JSON
/// index creation accepts scalar paths only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum JsonIndexTargetType {
    Boolean,
    Int64,
    Float64,
    Utf8,
    // Retained for parameters derived from legacy array/object indices.
    LargeBinary,
}

impl JsonIndexTargetType {
    /// The matching protobuf variant, for writing into `JsonIndexDetails`.
    fn to_pb(self) -> crate::pb::JsonTargetDataType {
        match self {
            Self::Boolean => crate::pb::JsonTargetDataType::Boolean,
            Self::Int64 => crate::pb::JsonTargetDataType::Int64,
            Self::Float64 => crate::pb::JsonTargetDataType::Float64,
            Self::Utf8 => crate::pb::JsonTargetDataType::Utf8,
            Self::LargeBinary => crate::pb::JsonTargetDataType::LargeBinary,
        }
    }

    /// The type recorded by a raw `JsonTargetDataType` value, or `None` if it
    /// records no type.
    ///
    /// `None` covers both `Unspecified`, written by builds that predate the
    /// field, and any variant added after this build, which it cannot act on
    /// but must not mistake for a type it knows.
    fn from_pb(value: i32) -> Option<Self> {
        match crate::pb::JsonTargetDataType::try_from(value).ok()? {
            crate::pb::JsonTargetDataType::Unspecified => None,
            crate::pb::JsonTargetDataType::Boolean => Some(Self::Boolean),
            crate::pb::JsonTargetDataType::Int64 => Some(Self::Int64),
            crate::pb::JsonTargetDataType::Float64 => Some(Self::Float64),
            crate::pb::JsonTargetDataType::Utf8 => Some(Self::Utf8),
            crate::pb::JsonTargetDataType::LargeBinary => Some(Self::LargeBinary),
        }
    }

    /// The name used for this type in `details_as_json`, and so in the details
    /// a `list_indices` response reports.
    fn as_json_name(self) -> &'static str {
        match self {
            Self::Boolean => "boolean",
            Self::Int64 => "int64",
            Self::Float64 => "float64",
            Self::Utf8 => "utf8",
            Self::LargeBinary => "large_binary",
        }
    }
}

impl TryFrom<&DataType> for JsonIndexTargetType {
    type Error = Error;

    fn try_from(data_type: &DataType) -> Result<Self> {
        match data_type {
            DataType::Boolean => Ok(Self::Boolean),
            DataType::Int64 => Ok(Self::Int64),
            DataType::Float64 => Ok(Self::Float64),
            DataType::Utf8 => Ok(Self::Utf8),
            DataType::LargeBinary => Ok(Self::LargeBinary),
            _ => Err(Error::not_supported(format!(
                "JSON index target data type {data_type:?} cannot be preserved for rebuilds"
            ))),
        }
    }
}

impl From<JsonIndexTargetType> for DataType {
    fn from(data_type: JsonIndexTargetType) -> Self {
        match data_type {
            JsonIndexTargetType::Boolean => Self::Boolean,
            JsonIndexTargetType::Int64 => Self::Int64,
            JsonIndexTargetType::Float64 => Self::Float64,
            JsonIndexTargetType::Utf8 => Self::Utf8,
            JsonIndexTargetType::LargeBinary => Self::LargeBinary,
        }
    }
}

// TODO: Do we really need to wrap the query or could we just return the target query directly?
//
// I think the only thing we really gain is a different format impl (e.g. it shows up as a json query
// in the explain plan) but I don't know if that helps the user much.
/// A predicate evaluated by an index on a value extracted from a JSON column.
///
/// The path and target query are exposed so distributed planners can preserve
/// the extraction boundary when encoding and reconstructing the query.
///
/// ```
/// use std::sync::Arc;
/// use lance_index::scalar::{SargableQuery, json::JsonQuery};
///
/// let query = JsonQuery::new(Arc::new(SargableQuery::IsNull()), "address.city".into());
/// let rebuilt = JsonQuery::new(query.target_query().clone(), query.path().to_owned());
/// assert_eq!(query, rebuilt);
/// ```
#[derive(Debug, Clone)]
pub struct JsonQuery {
    target_query: Arc<dyn AnyQuery>,
    path: String,
}

impl JsonQuery {
    pub fn new(target_query: Arc<dyn AnyQuery>, path: String) -> Self {
        Self { target_query, path }
    }

    /// The JSON path used to extract the values searched by the target index.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The typed predicate applied to the extracted JSON values.
    pub fn target_query(&self) -> &Arc<dyn AnyQuery> {
        &self.target_query
    }
}

impl PartialEq for JsonQuery {
    fn eq(&self, other: &Self) -> bool {
        self.target_query.dyn_eq(other.target_query.as_ref()) && self.path == other.path
    }
}

impl AnyQuery for JsonQuery {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn format(&self, col: &str) -> String {
        format!("Json({}->{})", self.target_query.format(col), self.path)
    }

    fn to_expr(&self, _col: String) -> Expr {
        todo!()
    }

    fn dyn_eq(&self, other: &dyn AnyQuery) -> bool {
        match other.as_any().downcast_ref::<Self>() {
            Some(o) => self == o,
            None => false,
        }
    }

    fn reports_null_rows(&self) -> bool {
        self.target_query.reports_null_rows()
    }
}

#[derive(Debug)]
pub struct JsonQueryParser {
    path: String,
    target_parser: Box<dyn ScalarQueryParser>,
}

impl JsonQueryParser {
    pub fn new(path: String, target_parser: Box<dyn ScalarQueryParser>) -> Self {
        Self {
            path,
            target_parser,
        }
    }

    fn wrap_search(&self, target_expr: IndexedExpression) -> IndexedExpression {
        if let Some(scalar_query) = target_expr.scalar_query {
            let scalar_query = match scalar_query {
                ScalarIndexExpr::Query(ScalarIndexSearch {
                    column,
                    index_name,
                    index_type,
                    query,
                    needs_recheck,
                    fragment_bitmap,
                }) => ScalarIndexExpr::Query(ScalarIndexSearch {
                    column,
                    index_name,
                    index_type,
                    query: Arc::new(JsonQuery::new(query, self.path.clone())),
                    needs_recheck,
                    fragment_bitmap,
                }),
                // This code path should only be hit on leaf expr
                _ => unreachable!(),
            };
            IndexedExpression {
                scalar_query: Some(scalar_query),
                refine_expr: target_expr.refine_expr,
            }
        } else {
            target_expr
        }
    }

    fn path_matches_typed_accessor(path: &str) -> bool {
        // A bare number is an object field in JSONPath but a root-array index in
        // the typed accessors, so its meaning also depends on the input value.
        if path.parse::<usize>().is_ok() {
            return false;
        }
        let Ok(json_path) = lance_jsonb::jsonpath::parse_json_path(path.as_bytes()) else {
            return false;
        };

        matches!(
            json_path.paths.as_slice(),
            [lance_jsonb::jsonpath::Path::DotField(field)] if field.as_ref() == path
        )
    }
}

impl ScalarQueryParser for JsonQueryParser {
    fn visit_between(
        &self,
        column: &str,
        low: &Bound<ScalarValue>,
        high: &Bound<ScalarValue>,
    ) -> Option<IndexedExpression> {
        self.target_parser
            .visit_between(column, low, high)
            .map(|target_expr| self.wrap_search(target_expr))
    }
    fn visit_in_list(&self, column: &str, in_list: &[ScalarValue]) -> Option<IndexedExpression> {
        self.target_parser
            .visit_in_list(column, in_list)
            .map(|target_expr| self.wrap_search(target_expr))
    }
    fn visit_is_bool(&self, column: &str, value: bool) -> Option<IndexedExpression> {
        self.target_parser
            .visit_is_bool(column, value)
            .map(|target_expr| self.wrap_search(target_expr))
    }
    fn visit_is_null(&self, column: &str) -> Option<IndexedExpression> {
        self.target_parser
            .visit_is_null(column)
            .map(|target_expr| self.wrap_search(target_expr))
    }
    fn visit_comparison(
        &self,
        column: &str,
        value: &ScalarValue,
        op: &Operator,
    ) -> Option<IndexedExpression> {
        self.target_parser
            .visit_comparison(column, value, op)
            .map(|target_expr| self.wrap_search(target_expr))
    }
    fn visit_scalar_function(
        &self,
        column: &str,
        data_type: &DataType,
        func: &ScalarUDF,
        args: &[Expr],
    ) -> Option<IndexedExpression> {
        self.target_parser
            .visit_scalar_function(column, data_type, func, args)
            .map(|target_expr| self.wrap_search(target_expr))
    }

    // TODO: maybe we should address it by https://github.com/lance-format/lance/issues/4624
    fn is_valid_reference(&self, func: &Expr, _data_type: &DataType) -> Option<DataType> {
        match func {
            Expr::ScalarFunction(udf) => {
                // Only the typed accessors are routable. `json_extract` evaluates to
                // serialized JSON text (`"click"`, `2`) while index keys hold decoded
                // native values (`click`, `2`), so an indexed `json_extract` predicate
                // would answer a different question than the unindexed one. Quoting is
                // also not order-preserving (`ab` < `ab!` but `"ab"` > `"ab!"`), so even
                // a Utf8 index cannot serve `json_extract` ranges. Let these fall back
                // to a full scan until literals are transcoded into the index's
                // representation. See https://github.com/lance-format/lance/issues/8806.
                let value_type = match udf.name() {
                    "json_get_int" => DataType::Int64,
                    "json_get_float" => DataType::Float64,
                    "json_get_bool" => DataType::Boolean,
                    "json_get_string" => DataType::Utf8,
                    _ => return None,
                };
                if udf.args.len() != 2 {
                    return None;
                }
                // Index training evaluates `self.path` as JSONPath, while typed accessors
                // perform one literal object-key lookup (or a root-array index lookup).
                // These operations are equivalent only when the path parses as one bare
                // object key with no normalization. In particular, `$.user.age` and
                // `user.age` must fall back to a scan instead of querying nested index keys.
                if !Self::path_matches_typed_accessor(&self.path) {
                    return None;
                }
                // We already know index 0 is a column reference to the column so we just need to
                // ensure that index 1 matches our path
                match &udf.args[1] {
                    Expr::Literal(ScalarValue::Utf8(Some(path)), _) if path == &self.path => {
                        Some(value_type)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

pub struct JsonTrainingRequest {
    parameters: JsonIndexParameters,
    target_request: Box<dyn TrainingRequest>,
    criteria: TrainingCriteria,
}

fn json_scan_criteria(target_criteria: &TrainingCriteria) -> TrainingCriteria {
    let mut criteria = target_criteria.clone();
    if criteria.ordering == TrainingOrdering::Values {
        // The scanner can only order the raw JSON column. The JSON wrapper sorts
        // the extracted path value before passing it to the target index.
        criteria.ordering = TrainingOrdering::None;
    }
    criteria
}

impl JsonTrainingRequest {
    pub fn new(parameters: JsonIndexParameters, target_request: Box<dyn TrainingRequest>) -> Self {
        let criteria = json_scan_criteria(target_request.criteria());
        Self {
            parameters,
            target_request,
            criteria,
        }
    }
}

impl TrainingRequest for JsonTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

/// Plugin implementation for a [`JsonIndex`]
#[derive(Default)]
pub struct JsonIndexPlugin {
    registry: Mutex<Option<Arc<IndexPluginRegistry>>>,
}

impl std::fmt::Debug for JsonIndexPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JsonIndexPlugin")
    }
}

impl JsonIndexPlugin {
    fn registry(&self) -> Result<Arc<IndexPluginRegistry>> {
        Ok(self.registry.lock().unwrap().as_ref().expect_ok()?.clone())
    }

    /// Present the indexed JSON column as JSONB, the encoding the JSON path
    /// functions read, whichever encoding it arrives in.
    fn jsonb_values(data: SendableRecordBatchStream) -> Result<SendableRecordBatchStream> {
        let schema = data.schema();
        let (value_idx, value_field) = schema.column_with_name(VALUE_COLUMN_NAME).expect_ok()?;
        if JsonEncoding::of_field(value_field) != Some(JsonEncoding::Text) {
            return Ok(data);
        }
        let value_field = value_field.clone();
        let mut fields = schema.fields().to_vec();
        fields[value_idx] = Arc::new(json_field(VALUE_COLUMN_NAME, value_field.is_nullable()));
        let jsonb_schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
        let output_schema = jsonb_schema.clone();
        let stream = data.map(move |batch| {
            let batch = batch?;
            let values = JsonValues::try_new(&value_field, batch.column(value_idx))
                .expect("the value field holds JSON text")
                .to_jsonb()?;
            let mut columns = batch.columns().to_vec();
            columns[value_idx] = Arc::new(values);
            Ok(RecordBatch::try_new(jsonb_schema.clone(), columns)?)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            output_schema,
            stream,
        )))
    }

    /// Extract a JSON path and its type tag while preserving all row-location columns.
    fn extract_json(
        data: SendableRecordBatchStream,
        path: String,
    ) -> Result<SendableRecordBatchStream> {
        let input = Arc::new(OneShotExec::new(Self::jsonb_values(data)?));
        let input_schema = input.schema();
        let value_column_idx = input_schema
            .column_with_name(VALUE_COLUMN_NAME)
            .expect_ok()?
            .0;

        // Call json_extract_with_type UDF
        let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
            Vec::with_capacity(input_schema.fields().len());
        exprs.push((
            Arc::new(ScalarFunctionExpr::try_new(
                Arc::new(lance_datafusion_exec::udf::json::json_extract_with_type_udf()),
                vec![
                    Arc::new(Column::new(VALUE_COLUMN_NAME, value_column_idx)),
                    Arc::new(Literal::new(ScalarValue::Utf8(Some(path)))),
                ],
                &input_schema,
                Arc::new(ConfigOptions::default()),
            )?) as Arc<dyn PhysicalExpr>,
            "json_result".to_string(),
        ));
        for (column_idx, field) in input_schema.fields().iter().enumerate() {
            if field.name() != VALUE_COLUMN_NAME {
                exprs.push((
                    Arc::new(Column::new(field.name(), column_idx)) as Arc<dyn PhysicalExpr>,
                    field.name().clone(),
                ));
            }
        }

        let project = ProjectionExec::try_new(exprs, input)?;
        project
            .execute(0, Arc::new(TaskContext::default()))
            .map_err(Into::into)
    }

    fn infer_type_from_batch(batch: &RecordBatch, path: &str) -> Result<Option<DataType>> {
        let json_result_column = batch
            .column_by_name("json_result")
            .ok_or_else(|| Error::invalid_input_source("Missing json_result column".into()))?;
        let struct_array = json_result_column
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| Error::invalid_input_source("json_result is not a struct".into()))?;
        let type_array = struct_array
            .column_by_name("type_tag")
            .ok_or_else(|| Error::invalid_input_source("Missing type_tag column in struct".into()))?
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| Error::invalid_input_source("type_tag is not UInt8".into()))?;

        for type_tag in type_array.iter().flatten() {
            let jsonb_type = JsonbType::from_u8(type_tag).ok_or_else(|| {
                Error::invalid_input_source(
                    format!("JSON path '{path}' produced invalid type tag {type_tag}").into(),
                )
            })?;
            let data_type = match jsonb_type {
                JsonbType::Null => continue,
                JsonbType::Boolean => DataType::Boolean,
                JsonbType::Int64 => DataType::Int64,
                JsonbType::Float64 => DataType::Float64,
                JsonbType::String => DataType::Utf8,
                JsonbType::Array | JsonbType::Object => {
                    return Err(Error::invalid_input(format!(
                        "Cannot create a JSON index for JSON path '{path}' with JSON type {jsonb_type:?}; only scalar values are supported"
                    )));
                }
            };
            return Ok(Some(data_type));
        }
        Ok(None)
    }

    /// Extract JSON and infer the target type from the first non-null value.
    ///
    /// Only the prefix needed for inference is buffered. Once a type is found,
    /// the remainder of the input stays streaming.
    async fn extract_json_with_type_info(
        data: SendableRecordBatchStream,
        path: String,
    ) -> Result<(SendableRecordBatchStream, DataType)> {
        let mut stream = Self::extract_json(data, path.clone())?;
        let schema = stream.schema();
        let mut buffered_batches = Vec::new();
        let mut inferred_type = None;

        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            inferred_type = Self::infer_type_from_batch(&batch, &path)?;
            buffered_batches.push(batch);
            if inferred_type.is_some() {
                break;
            }
        }

        let inferred_type = inferred_type.unwrap_or(DataType::Utf8);
        let buffered = futures::stream::iter(buffered_batches.into_iter().map(Ok));
        let recreated_stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            buffered.chain(stream),
        )) as SendableRecordBatchStream;

        Ok((recreated_stream, inferred_type))
    }

    fn validate_json_types(
        binary_array: &LargeBinaryArray,
        type_array: &UInt8Array,
        target_type: &DataType,
        path: &str,
    ) -> Result<()> {
        for index in 0..binary_array.len() {
            if binary_array.is_null(index) {
                continue;
            }
            if type_array.is_null(index) {
                return Err(Error::invalid_input_source(
                    format!(
                        "JSON path '{path}' has a value at batch row {index} without a type tag"
                    )
                    .into(),
                ));
            }
            let type_tag = type_array.value(index);
            let actual_type = JsonbType::from_u8(type_tag).ok_or_else(|| {
                Error::invalid_input_source(
                    format!(
                        "JSON path '{path}' produced invalid type tag {type_tag} at batch row {index}"
                    )
                    .into(),
                )
            })?;
            if actual_type == JsonbType::Null {
                continue;
            }
            let is_compatible = match target_type {
                DataType::Boolean => actual_type == JsonbType::Boolean,
                DataType::Int64 => actual_type == JsonbType::Int64,
                DataType::Float64 => {
                    matches!(actual_type, JsonbType::Int64 | JsonbType::Float64)
                }
                DataType::Utf8 => actual_type == JsonbType::String,
                DataType::LargeBinary => {
                    matches!(actual_type, JsonbType::Array | JsonbType::Object)
                }
                _ => false,
            };
            if !is_compatible {
                return Err(Error::invalid_input_source(
                    format!(
                        "JSON path '{path}' expected {target_type:?}, but batch row {index} has JSON type {actual_type:?}"
                    )
                    .into(),
                ));
            }
        }
        Ok(())
    }

    fn convert_batch_by_type(
        batch: RecordBatch,
        output_schema: Arc<Schema>,
        passthrough_indices: &[usize],
        target_type: &DataType,
        path: &str,
    ) -> Result<RecordBatch> {
        let json_result_column = batch
            .column_by_name("json_result")
            .ok_or_else(|| Error::invalid_input_source("Missing json_result column".into()))?;

        let struct_array = json_result_column
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| Error::invalid_input_source("json_result is not a struct".into()))?;

        let value_array = struct_array
            .column_by_name("value")
            .ok_or_else(|| Error::invalid_input_source("Missing value column in struct".into()))?;

        let binary_array = value_array
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| Error::invalid_input_source("value is not LargeBinary".into()))?;
        let type_array = struct_array
            .column_by_name("type_tag")
            .ok_or_else(|| Error::invalid_input_source("Missing type_tag column in struct".into()))?
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| Error::invalid_input_source("type_tag is not UInt8".into()))?;

        Self::validate_json_types(binary_array, type_array, target_type, path)?;
        let is_null = |index| {
            binary_array.is_null(index)
                || (!type_array.is_null(index)
                    && type_array.value(index) == JsonbType::Null.as_u8())
        };

        let converted_array: Arc<dyn Array> = match target_type {
            DataType::Boolean => {
                let mut builder =
                    arrow_array::builder::BooleanBuilder::with_capacity(binary_array.len());
                for i in 0..binary_array.len() {
                    if is_null(i) {
                        builder.append_null();
                    } else {
                        let raw_jsonb = lance_jsonb::RawJsonb::new(binary_array.value(i));
                        let value = lance_jsonb::from_raw_jsonb::<bool>(&raw_jsonb).map_err(|error| {
                            Error::invalid_input_source(
                                format!(
                                    "Failed to convert JSON path '{path}' at batch row {i} to Boolean: {error}"
                                )
                                .into(),
                            )
                        })?;
                        builder.append_value(value);
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::Int64 => {
                let mut builder =
                    arrow_array::builder::Int64Builder::with_capacity(binary_array.len());
                for i in 0..binary_array.len() {
                    if is_null(i) {
                        builder.append_null();
                    } else {
                        let raw_jsonb = lance_jsonb::RawJsonb::new(binary_array.value(i));
                        let value = lance_jsonb::from_raw_jsonb::<i64>(&raw_jsonb).map_err(|error| {
                            Error::invalid_input_source(
                                format!(
                                    "Failed to convert JSON path '{path}' at batch row {i} to Int64: {error}"
                                )
                                .into(),
                            )
                        })?;
                        builder.append_value(value);
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::Float64 => {
                let mut builder =
                    arrow_array::builder::Float64Builder::with_capacity(binary_array.len());
                for i in 0..binary_array.len() {
                    if is_null(i) {
                        builder.append_null();
                    } else {
                        let raw_jsonb = lance_jsonb::RawJsonb::new(binary_array.value(i));
                        let value = lance_jsonb::from_raw_jsonb::<f64>(&raw_jsonb).map_err(|error| {
                            Error::invalid_input_source(
                                format!(
                                    "Failed to convert JSON path '{path}' at batch row {i} to Float64: {error}"
                                )
                                .into(),
                            )
                        })?;
                        builder.append_value(value);
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::Utf8 => {
                let mut builder =
                    arrow_array::builder::StringBuilder::with_capacity(binary_array.len(), 1024);
                for i in 0..binary_array.len() {
                    if is_null(i) {
                        builder.append_null();
                    } else {
                        let raw_jsonb = lance_jsonb::RawJsonb::new(binary_array.value(i));
                        let value =
                            lance_jsonb::from_raw_jsonb::<String>(&raw_jsonb).map_err(|error| {
                                Error::invalid_input_source(
                                    format!(
                                        "Failed to convert JSON path '{path}' at batch row {i} to Utf8: {error}"
                                    )
                                    .into(),
                                )
                            })?;
                        builder.append_value(value);
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::LargeBinary => Arc::new(LargeBinaryArray::from_iter(
                (0..binary_array.len()).map(|i| (!is_null(i)).then(|| binary_array.value(i))),
            )),
            _ => {
                return Err(Error::invalid_input_source(
                    format!("Unsupported JSON index target type: {target_type:?}").into(),
                ));
            }
        };

        let mut columns = Vec::with_capacity(output_schema.fields().len());
        columns.push(converted_array);
        columns.extend(
            passthrough_indices
                .iter()
                .map(|column_idx| batch.column(*column_idx).clone()),
        );
        Ok(RecordBatch::try_new(output_schema, columns)?)
    }

    /// Convert extracted JSON values one batch at a time to the target's trained type.
    fn convert_stream_by_type(
        data: SendableRecordBatchStream,
        target_type: DataType,
        path: String,
    ) -> Result<SendableRecordBatchStream> {
        let input_schema = data.schema();
        let passthrough_indices = input_schema
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(column_idx, field)| (field.name() != "json_result").then_some(column_idx))
            .collect::<Vec<_>>();
        let output_schema = Arc::new(Schema::new(
            std::iter::once(Field::new(VALUE_COLUMN_NAME, target_type.clone(), true))
                .chain(
                    passthrough_indices
                        .iter()
                        .map(|column_idx| input_schema.field(*column_idx).clone()),
                )
                .collect::<Vec<_>>(),
        ));
        let stream_schema = output_schema.clone();
        let converted = data.map(move |batch_result| {
            let batch = batch_result?;
            Self::convert_batch_by_type(
                batch,
                output_schema.clone(),
                &passthrough_indices,
                &target_type,
                &path,
            )
            .map_err(DataFusionError::from)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            stream_schema,
            converted,
        )))
    }

    /// Sort a value stream and its row-location columns ascending by value.
    ///
    /// Target index types that require `TrainingOrdering::Values` (e.g. btree, whose
    /// per-page min/max stats are taken from the first/last row of each page) need this
    /// as the only sort in the JSON training path: `JsonTrainingRequest` requests
    /// unordered input from the scanner, since the scanner can only sort on the raw
    /// JSON column, not on the value at `path`.
    async fn sort_stream_by_value(
        data: SendableRecordBatchStream,
    ) -> Result<SendableRecordBatchStream> {
        let input = Arc::new(OneShotExec::new(data));
        let value_idx = input.schema().index_of(VALUE_COLUMN_NAME)?;
        let sort_expr = PhysicalSortExpr {
            expr: Arc::new(Column::new(VALUE_COLUMN_NAME, value_idx)),
            options: SortOptions {
                descending: false,
                nulls_first: true,
            },
        };
        let plan = Arc::new(SortExec::new([sort_expr].into(), input));
        execute_plan(
            plan,
            LanceExecutionOptions {
                use_spilling: true,
                ..Default::default()
            },
        )
    }
}

#[async_trait]
impl BasicTrainer for JsonIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        if JsonEncoding::of_field(field).is_none()
            && !matches!(field.data_type(), DataType::Binary | DataType::LargeBinary)
        {
            return Err(Error::invalid_input_source(
                "A JSON index can only be created on a JSON, Binary or LargeBinary field.".into(),
            ));
        }

        let params = serde_json::from_str::<JsonIndexParameters>(params)?;
        if params.target_data_type == Some(JsonIndexTargetType::LargeBinary) {
            return Err(Error::invalid_input(format!(
                "Cannot create a JSON index for JSON path '{}' with target data type LargeBinary (arrays or objects); only scalar values are supported",
                params.path
            )));
        }
        // Initial builds infer the type from the data. Derived rebuild parameters
        // carry the learned type so every new segment uses the same target schema.
        let target_type = params
            .target_data_type
            .map(DataType::from)
            .unwrap_or(DataType::Utf8);
        let registry = self.registry()?;
        let target_plugin = registry.get_plugin_by_name(&params.target_index_type)?;
        if target_plugin.name().eq_ignore_ascii_case("inverted") {
            return Err(Error::not_supported(format!(
                "JSON-path indexes do not support target index type '{}'; create an INVERTED index directly on the JSON column instead",
                params.target_index_type
            )));
        }
        let target_trainer = target_plugin.basic_trainer().ok_or_else(|| {
            Error::invalid_input_source(
                format!("The '{}' index type does not support basic training, please refer to the index's documentation for more details on how to create this index.", params.target_index_type).into(),
            )
        })?;
        let target_request = target_trainer.new_training_request(
            params.target_index_parameters.as_deref().unwrap_or("{}"),
            &Field::new("", target_type, true),
        )?;

        Ok(Box::new(JsonTrainingRequest::new(params, target_request)))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        fragment_ids: Option<Vec<u32>>,
        progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let request = (request as Box<dyn std::any::Any>)
            .downcast::<JsonTrainingRequest>()
            .unwrap();
        let path = request.parameters.path.clone();

        let (data_stream, target_type) =
            if let Some(target_data_type) = request.parameters.target_data_type {
                (
                    Self::extract_json(data, path.clone())?,
                    DataType::from(target_data_type),
                )
            } else {
                Self::extract_json_with_type_info(data, path.clone()).await?
            };

        // Captured before `target_type` is consumed below. Recorded even when it
        // was inferred from the data, so a later rebuild reproduces this type
        // instead of inferring its own.
        let recorded_target_data_type = JsonIndexTargetType::try_from(&target_type)?.to_pb() as i32;

        // Initial builds use the inferred type; rebuilds use the learned target
        // type carried in the derived parameters.
        let converted_stream =
            Self::convert_stream_by_type(data_stream, target_type.clone(), path.clone())?;

        // `JsonTrainingRequest::criteria()` asked the scanner for unordered input (see
        // its constructor), since the scanner can only sort on the raw JSON column, not
        // on the value at `path`. If the target index needs value-ordered input, this is
        // the one place that sort happens: on the extracted value, after extraction.
        //
        // Deliberately `request.target_request.criteria()` here, not `request.criteria()`:
        // the latter is `JsonTrainingRequest`'s own criteria, which is always `None` (that's
        // what asked the scanner for unordered input above) and would never take this branch.
        let converted_stream =
            if request.target_request.criteria().ordering == TrainingOrdering::Values {
                Self::sort_stream_by_value(converted_stream).await?
            } else {
                converted_stream
            };

        let registry = self.registry()?;
        let target_plugin = registry.get_plugin_by_name(&request.parameters.target_index_type)?;

        let target_trainer = target_plugin.basic_trainer().ok_or_else(|| {
            Error::invalid_input_source(
                format!("The '{}' index type does not support basic training, please refer to the index's documentation for more details on how to create this index.", request.parameters.target_index_type).into(),
            )
        })?;
        let target_request = target_trainer.new_training_request(
            request
                .parameters
                .target_index_parameters
                .as_deref()
                .unwrap_or("{}"),
            &Field::new("", target_type, true),
        )?;

        let target_index = target_trainer
            .train_index(
                converted_stream,
                index_store,
                target_request,
                fragment_ids,
                progress,
            )
            .await?;

        // BTree, Bitmap, and LabelList's `BasicTrainer::train_index` always reports the
        // current (address-domain) version: that generic path has no dataset awareness
        // to tell whether a row id and a row address coincide. Unlike the plain-index
        // path, which `build_scalar_index` in the `lance` crate downgrades itself when
        // the dataset lacks stable row ids, a JSON-wrapped target is relabeled the same
        // way by `relabel_as_row_id_domain`, called from that same spot. So this reports
        // the target's natural (address-domain) version here, and lets that caller
        // decide whether to downgrade it.
        let index_version =
            json_index_version_for_target(&target_index.index_details, target_index.index_version);

        let index_details = crate::pb::JsonIndexDetails {
            path,
            target_details: Some(target_index.index_details),
            target_data_type: recorded_target_data_type,
            target_index_version: Some(target_index.index_version),
        };
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&index_details)?,
            index_version,
            files: target_index.files,
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for JsonIndexPlugin {
    fn can_remap(&self, details: &prost_types::Any) -> Result<bool> {
        let details = match crate::pb::JsonIndexDetails::decode(details.value.as_slice()) {
            Ok(details) => details,
            Err(error) => {
                log::warn!(
                    "Cannot determine whether a JSON index supports remapping because its \
                     details cannot be decoded: {error}"
                );
                return Ok(false);
            }
        };
        let Some(target_details) = details.target_details.as_ref() else {
            log::warn!(
                "Cannot determine whether a JSON index supports remapping because its details \
                 are missing target_details"
            );
            return Ok(false);
        };
        let registry = self.registry()?;
        let target_plugin = match registry.get_plugin_by_details(target_details) {
            Ok(plugin) => plugin,
            Err(error) => {
                log::warn!(
                    "Cannot determine whether a JSON index supports remapping because its \
                     target plugin cannot be found: {error}"
                );
                return Ok(false);
            }
        };
        match target_plugin.can_remap(target_details) {
            Ok(can_remap) => Ok(can_remap),
            Err(error) => {
                log::warn!(
                    "Cannot determine whether a JSON index supports remapping because its \
                     target plugin cannot answer: {error}"
                );
                Ok(false)
            }
        }
    }

    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn name(&self) -> &str {
        "Json"
    }

    fn provides_exact_answer(&self) -> bool {
        // TODO: Need to lookup target plugin via details to figure this out correctly
        true
    }

    fn attach_registry(&self, registry: Arc<IndexPluginRegistry>) {
        let mut reg_ref = self.registry.lock().unwrap();
        *reg_ref = Some(registry);
    }

    fn version(&self) -> u32 {
        JSON_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        // TODO: Allow return Result here
        let registry = self.registry().unwrap();
        let json_details =
            crate::pb::JsonIndexDetails::decode(index_details.value.as_slice()).unwrap();
        let target_details = json_details.target_details.as_ref().expect_ok().unwrap();
        let target_plugin = registry.get_plugin_by_details(target_details).unwrap();
        // Older JSON-wrapped inverted indexes were never routable. Current training rejects this
        // combination, so keep legacy metadata inert instead of enabling only part of its
        // unsupported query path.
        if target_plugin.name().eq_ignore_ascii_case("inverted") {
            return None;
        }
        // TODO: Use something like ${index_name}_${path} for the index name?  Don't have access to path here tho
        let target_parser = target_plugin.new_query_parser(index_name, target_details)?;
        Some(Box::new(JsonQueryParser::new(
            json_details.path.clone(),
            target_parser,
        )) as Box<dyn ScalarQueryParser>)
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        _index_version: u32,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let registry = self.registry().unwrap();
        let json_details = crate::pb::JsonIndexDetails::decode(index_details.value.as_slice())?;
        let target_details = json_details.target_details.as_ref().expect_ok()?;
        let target_plugin = registry.get_plugin_by_details(target_details).unwrap();

        // `_index_version` is this *wrapper's* version (`JSON_INDEX_VERSION`), not the
        // target's. The target's own persisted version lives in `target_index_version`;
        //
        // If `target_index_version` is missing then we have to guess.  Our best guess
        // is that this is an old index written before `target_index_version` was added.
        // These indexes were the old versions that stored row ids and not row addresses.
        let target_index_version = json_details
            .target_index_version
            .or_else(|| legacy_target_index_version(target_details))
            .unwrap_or(0);

        let target_index = target_plugin
            .load_index(
                index_store,
                target_details,
                target_index_version,
                frag_reuse_index,
                &cache.with_key_prefix("json_target"),
            )
            .await?;
        let mut index = JsonIndex::new(target_index, json_details.path)
            .with_target_data_type(json_details.target_data_type);
        index.target_plugin_name = Some(target_plugin.name().to_string());
        Ok(Arc::new(index))
    }

    async fn get_from_cache(
        &self,
        index_store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Option<Arc<dyn ScalarIndex>>> {
        let Some(state) = cache.get_with_key(&JsonIndexStateKey).await else {
            return Ok(None);
        };
        let registry = self.registry()?;
        let plugin = registry.get_plugin_by_name(&state.target_plugin_name)?;
        Ok(plugin
            .get_from_cache(
                index_store,
                frag_reuse_index,
                &cache.with_key_prefix("json_target"),
            )
            .await?
            .map(|target| state.wrap(target)))
    }

    async fn put_in_cache(
        &self,
        index_store: Arc<dyn IndexStore>,
        cache: &LanceCache,
        index: Arc<dyn ScalarIndex>,
    ) -> Result<()> {
        let state = JsonIndexState::from_index(index.as_ref())?;
        let json = index.as_any().downcast_ref::<JsonIndex>().expect_ok()?;
        let registry = self.registry()?;
        let plugin = registry.get_plugin_by_name(&state.target_plugin_name)?;
        plugin
            .put_in_cache(
                index_store,
                &cache.with_key_prefix("json_target"),
                json.target_index.clone(),
            )
            .await?;
        cache
            .insert_with_key(&JsonIndexStateKey, Arc::new(state))
            .await;
        Ok(())
    }

    async fn get_or_insert_in_cache(
        &self,
        index_store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
        load: ScalarIndexLoad<'_>,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let registry = self.registry()?;
        let target_cache = cache.with_key_prefix("json_target");
        let mut pending_load = Some(load);
        let mut loaded = None;
        let state = cache
            .get_or_insert_with_key(JsonIndexStateKey, || async {
                let index = pending_load.take().expect_ok()?.await?;
                let state = JsonIndexState::from_index(index.as_ref())?;
                let json = index.as_any().downcast_ref::<JsonIndex>().expect_ok()?;
                // Populate the target before publishing wrapper metadata so concurrent
                // cold callers do not start a second target load in the gap.
                registry
                    .get_plugin_by_name(&state.target_plugin_name)?
                    .put_in_cache(
                        index_store.clone(),
                        &target_cache,
                        json.target_index.clone(),
                    )
                    .await?;
                loaded = Some(index);
                Ok(state)
            })
            .await?;
        let plugin = registry.get_plugin_by_name(&state.target_plugin_name)?;
        let target_load: ScalarIndexLoad = Box::pin(async move {
            let index = match loaded {
                Some(index) => index,
                None => pending_load.take().expect_ok()?.await?,
            };
            Ok(index
                .as_any()
                .downcast_ref::<JsonIndex>()
                .expect_ok()?
                .target_index
                .clone())
        });
        let target = plugin
            .get_or_insert_in_cache(index_store, frag_reuse_index, &target_cache, target_load)
            .await?;
        Ok(state.wrap(target))
    }

    fn details_as_json(&self, details: &prost_types::Any) -> Result<serde_json::Value> {
        let registry = self.registry().unwrap();
        let json_details = crate::pb::JsonIndexDetails::decode(details.value.as_slice())?;
        let target_details = json_details.target_details.as_ref().expect_ok()?;
        let target_plugin = registry.get_plugin_by_details(target_details).unwrap();
        let target_details_json = target_plugin.details_as_json(target_details)?;
        let mut details_json = serde_json::json!({
            "path": json_details.path,
            "target_details": target_details_json,
        });
        // Omitted rather than reported as null when the index records no type,
        // so the details of an index built before the field existed read the
        // same as they always have. An unrecognized value keeps its raw number
        // in the string, so the field's JSON type does not depend on the writer.
        if json_details.target_data_type != crate::pb::JsonTargetDataType::Unspecified as i32 {
            let reported = JsonIndexTargetType::from_pb(json_details.target_data_type)
                .map(|target_type| target_type.as_json_name().to_string())
                .unwrap_or_else(|| format!("unknown({})", json_details.target_data_type));
            details_json["target_data_type"] = serde_json::json!(reported);
        }
        Ok(details_json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::bitmap::{BitmapIndex, BitmapIndexPlugin};
    use crate::scalar::lance_format::LanceIndexStore;
    use crate::scalar::lookup_test_util::lookup_pairs;
    use crate::scalar::{SargableQuery, TextQuery};
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, record_batch};
    use arrow_schema::{DataType, Field, Schema};
    use lance_core::cache::LanceCache;
    use lance_core::utils::tempfile::TempObjDir;
    use lance_core::{ROW_ADDR, ROW_ID, utils::address::RowAddress};
    use lance_io::object_store::ObjectStore;
    use rstest::rstest;
    use std::ops::Bound;
    use std::sync::Arc;

    #[derive(Debug, Clone)]
    struct RequestStore {
        inner: Arc<dyn IndexStore>,
        opens: Arc<std::sync::atomic::AtomicUsize>,
        reads: Arc<std::sync::atomic::AtomicUsize>,
        denied: Arc<std::sync::atomic::AtomicBool>,
    }

    impl RequestStore {
        fn new(inner: Arc<dyn IndexStore>) -> Arc<Self> {
            Arc::new(Self {
                inner,
                opens: Arc::default(),
                reads: Arc::default(),
                denied: Arc::default(),
            })
        }
    }

    impl DeepSizeOf for RequestStore {
        fn deep_size_of_children(&self, _: &mut lance_core::deepsize::Context) -> usize {
            0
        }
    }

    struct RequestReader {
        inner: Arc<dyn crate::scalar::IndexReader>,
        reads: Arc<std::sync::atomic::AtomicUsize>,
        denied: Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait]
    impl crate::scalar::IndexReader for RequestReader {
        async fn read_record_batch(&self, n: u64, batch_size: u64) -> Result<RecordBatch> {
            if self.denied.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(Error::not_supported("request binding is unavailable"));
            }
            self.reads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.read_record_batch(n, batch_size).await
        }

        async fn read_range(
            &self,
            range: std::ops::Range<usize>,
            projection: Option<&[&str]>,
        ) -> Result<RecordBatch> {
            if self.denied.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(Error::not_supported("request binding is unavailable"));
            }
            self.reads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

    #[async_trait]
    impl IndexStore for RequestStore {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn clone_arc(&self) -> Arc<dyn IndexStore> {
            Arc::new(self.clone())
        }

        fn is_same_storage_binding(&self, other: &dyn IndexStore) -> bool {
            other
                .as_any()
                .downcast_ref::<Self>()
                .is_some_and(|other| Arc::ptr_eq(&self.opens, &other.opens))
        }

        fn io_parallelism(&self) -> usize {
            self.inner.io_parallelism()
        }

        async fn new_index_file(
            &self,
            name: &str,
            schema: Arc<Schema>,
        ) -> Result<Box<dyn crate::scalar::IndexWriter>> {
            self.inner.new_index_file(name, schema).await
        }

        async fn open_index_file(&self, name: &str) -> Result<Arc<dyn crate::scalar::IndexReader>> {
            if self.denied.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(Error::not_supported("request binding is unavailable"));
            }
            self.opens
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(Arc::new(RequestReader {
                inner: self.inner.open_index_file(name).await?,
                reads: self.reads.clone(),
                denied: self.denied.clone(),
            }))
        }

        fn with_io_priority(&self, _: u64) -> Arc<dyn IndexStore> {
            self.clone_arc()
        }

        fn with_io_buffer_size(&self, _: u64) -> Arc<dyn IndexStore> {
            self.clone_arc()
        }

        async fn copy_index_file(
            &self,
            name: &str,
            dest: &dyn IndexStore,
        ) -> Result<crate::scalar::IndexFile> {
            self.inner.copy_index_file(name, dest).await
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

    #[rstest]
    #[case::btree("btree")]
    #[case::ngram("ngram")]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_cache_request_rotation(#[case] target: &str) {
        use crate::metrics::NoOpMetricsCollector;
        use lance_select::RowAddrTreeMap;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (store, _dir) = local_json_index_store();
        let created = train_json_index(
            store.clone(),
            target,
            TrainingOrdering::None,
            "v",
            &[r#"{"v":"bravo"}"#, r#"{"v":null}"#, r#"{"v":"alpha"}"#],
        )
        .await;
        let a = RequestStore::new(store.clone());
        let b = RequestStore::new(store);
        let weak_a = Arc::downgrade(&a);
        let weak_b = Arc::downgrade(&b);
        let cache = LanceCache::with_capacity(1024 * 1024);
        let loads = AtomicUsize::new(0);
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let query: Arc<dyn AnyQuery> = if target == "btree" {
            Arc::new(SargableQuery::Equals(ScalarValue::Utf8(Some(
                "bravo".into(),
            ))))
        } else {
            Arc::new(TextQuery::StringContains("bravo".into()))
        };

        for (iteration, request) in [&a, &a, &b, &a].into_iter().enumerate() {
            let index = plugin
                .get_or_insert_in_cache(
                    request.clone(),
                    None,
                    &cache,
                    Box::pin(async {
                        loads.fetch_add(1, Ordering::Relaxed);
                        plugin
                            .load_index(
                                request.clone(),
                                &created.index_details,
                                created.index_version,
                                None,
                                &cache,
                            )
                            .await
                    }),
                )
                .await
                .unwrap();
            let reads_after_open =
                a.reads.load(Ordering::Relaxed) + b.reads.load(Ordering::Relaxed);
            assert_eq!(
                reads_after_open,
                usize::from(iteration == 0),
                "target metadata must only be read once for {target}, iteration {iteration}"
            );
            let result = index
                .search(
                    &JsonQuery::new(query.clone(), "v".into()),
                    &NoOpMetricsCollector,
                )
                .await
                .unwrap();
            let selected = RowAddrTreeMap::from_iter([0]);
            assert_eq!(
                result,
                if target == "btree" {
                    SearchResult::exact(selected).with_nulls(RowAddrTreeMap::from_iter([1]))
                } else {
                    SearchResult::at_most(selected)
                }
            );
            // Search pages/postings add reads; isolate target metadata loads.
            a.reads.store(0, Ordering::Relaxed);
            b.reads.store(0, Ordering::Relaxed);
        }
        assert_eq!(
            loads.load(Ordering::Relaxed),
            1,
            "A-A-B-A target loads for {target}"
        );
        drop(a);
        drop(b);
        assert!(weak_a.upgrade().is_none(), "cache retains request A");
        assert!(weak_b.upgrade().is_none(), "cache retains request B");
    }

    #[rstest]
    #[case::btree("btree")]
    #[case::ngram("ngram")]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_cache_concurrent_cold_open(#[case] target: &str) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (store, _dir) = local_json_index_store();
        let created = train_json_index(
            store.clone(),
            target,
            TrainingOrdering::None,
            "v",
            &[r#"{"v":"alpha"}"#],
        )
        .await;
        let requests = [RequestStore::new(store.clone()), RequestStore::new(store)];
        let cache = LanceCache::with_capacity(1024 * 1024);
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let loads = AtomicUsize::new(0);
        let indices = futures::future::join_all(requests.iter().map(|request| {
            plugin.get_or_insert_in_cache(
                request.clone(),
                None,
                &cache,
                Box::pin(async {
                    loads.fetch_add(1, Ordering::Relaxed);
                    tokio::task::yield_now().await;
                    plugin
                        .load_index(
                            request.clone(),
                            &created.index_details,
                            created.index_version,
                            None,
                            &cache,
                        )
                        .await
                }),
            )
        }))
        .await;
        for index in indices {
            index.unwrap();
        }
        assert_eq!(loads.load(Ordering::Relaxed), 1);

        cache.clear().await;
        assert!(
            plugin
                .get_from_cache(requests[0].clone(), None, &cache)
                .await
                .unwrap()
                .is_none()
        );
        let no_cache = LanceCache::no_cache();
        for _ in 0..2 {
            plugin
                .get_or_insert_in_cache(
                    requests[0].clone(),
                    None,
                    &no_cache,
                    Box::pin(async {
                        loads.fetch_add(1, Ordering::Relaxed);
                        plugin
                            .load_index(
                                requests[0].clone(),
                                &created.index_details,
                                created.index_version,
                                None,
                                &no_cache,
                            )
                            .await
                    }),
                )
                .await
                .unwrap();
        }
        assert_eq!(loads.load(Ordering::Relaxed), 3);
        let tiny_cache = LanceCache::with_capacity(1);
        for _ in 0..2 {
            plugin
                .get_or_insert_in_cache(
                    requests[0].clone(),
                    None,
                    &tiny_cache,
                    Box::pin(async {
                        loads.fetch_add(1, Ordering::Relaxed);
                        plugin
                            .load_index(
                                requests[0].clone(),
                                &created.index_details,
                                created.index_version,
                                None,
                                &tiny_cache,
                            )
                            .await
                    }),
                )
                .await
                .unwrap();
            // Moka applies weighted eviction asynchronously. The public size
            // query runs its pending maintenance before reporting residency.
            assert_eq!(tiny_cache.size_bytes().await, 0);
        }
        assert_eq!(loads.load(Ordering::Relaxed), 5);
    }

    #[rstest]
    #[case::btree("btree")]
    #[case::ngram("ngram")]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_cached_state_uses_current_readers(#[case] target: &str) {
        use crate::metrics::NoOpMetricsCollector;
        use std::sync::atomic::Ordering;

        let (store, _dir) = local_json_index_store();
        let created = train_json_index(
            store.clone(),
            target,
            TrainingOrdering::None,
            "v",
            &[r#"{"v":"alpha"}"#],
        )
        .await;
        let a = RequestStore::new(store.clone());
        let b = RequestStore::new(store);
        let cache = LanceCache::with_capacity(1024 * 1024);
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let index = plugin
            .load_index(
                a.clone(),
                &created.index_details,
                created.index_version,
                None,
                &cache,
            )
            .await
            .unwrap();
        plugin
            .put_in_cache(a.clone(), &cache, index.clone())
            .await
            .unwrap();
        a.denied.store(true, Ordering::Relaxed);
        a.reads.store(0, Ordering::Relaxed);
        let remapper = Arc::new(crate::scalar::lookup_test_util::CompactedToFragmentOne {
            deleted: Default::default(),
        });
        let weak_remapper = Arc::downgrade(&remapper);
        let rebound = plugin
            .get_from_cache(b.clone(), Some(remapper.clone()), &cache)
            .await
            .unwrap()
            .unwrap();
        let query: Arc<dyn AnyQuery> = if target == "btree" {
            Arc::new(SargableQuery::Equals(ScalarValue::Utf8(Some(
                "alpha".into(),
            ))))
        } else {
            Arc::new(TextQuery::StringContains("alpha".into()))
        };
        let result = rebound
            .search(
                &JsonQuery::new(query.clone(), "v".into()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        assert_eq!(
            result.row_addrs().selected_rows(),
            &lance_select::RowAddrTreeMap::from_iter([1u64 << 32])
        );
        assert!(b.opens.load(Ordering::Relaxed) > 0);
        assert!(b.reads.load(Ordering::Relaxed) > 0);
        assert_eq!(a.reads.load(Ordering::Relaxed), 0);
        drop(rebound);
        drop(remapper);
        assert!(
            weak_remapper.upgrade().is_none(),
            "cache retains request remapper"
        );
        cache.clear().await;
        plugin
            .put_in_cache(a.clone(), &cache, index.clone())
            .await
            .unwrap();
        b.denied.store(true, Ordering::Relaxed);
        let error = match plugin.get_from_cache(b.clone(), None, &cache).await {
            Err(error) => error,
            Ok(Some(index)) => index
                .search(&JsonQuery::new(query, "v".into()), &NoOpMetricsCollector)
                .await
                .unwrap_err(),
            Ok(None) => panic!("target state was just cached"),
        };
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("request binding is unavailable"));
        drop(index);
        let weak_a = Arc::downgrade(&a);
        drop(a);
        assert!(weak_a.upgrade().is_none());
    }

    #[rstest]
    #[case::bare_key("name", true)]
    #[case::unicode_key("名字", true)]
    #[case::root_prefixed("$.name", false)]
    #[case::nested("name.first", false)]
    #[case::jsonpath_array_index("$[0]", false)]
    #[case::accessor_array_index("0", false)]
    fn test_path_matches_typed_accessor(#[case] path: &str, #[case] expected: bool) {
        assert_eq!(JsonQueryParser::path_matches_typed_accessor(path), expected);
    }

    /// Lookup keys are values of the indexed path, in the target index's
    /// type, and are answered by the target index, in its row domain.
    #[rstest]
    #[case::row_addresses(true)]
    #[case::row_ids(false)]
    #[tokio::test]
    async fn test_json_lookup_forwards_to_target(#[case] results_are_row_addresses: bool) {
        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        ));
        let batch =
            record_batch!(("value", Int64, [1, 1, 2]), (ROW_ADDR, UInt64, [0, 2, 1])).unwrap();
        let schema = batch.schema();
        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(async move { Ok(batch) }),
        ));
        BitmapIndexPlugin::train_bitmap_index(stream, store.as_ref())
            .await
            .unwrap();
        let target = BitmapIndex::load(
            store,
            None,
            &LanceCache::no_cache(),
            results_are_row_addresses,
        )
        .await
        .unwrap();
        let index = JsonIndex::new(target, "x".to_string());
        assert_eq!(index.results_are_row_addresses(), results_are_row_addresses);

        let keys: ArrayRef = Arc::new(Int64Array::from(vec![1, 3]));
        assert_eq!(lookup_pairs(&index, keys).await, vec![(0, 0), (0, 2)]);
    }

    #[test]
    fn test_json_index_rejects_inverted_target() {
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let trainer = plugin.basic_trainer().unwrap();
        let error = trainer
            .new_training_request(
                r#"{"target_index_type":"inverted","path":"$.title"}"#,
                &Field::new(VALUE_COLUMN_NAME, DataType::LargeBinary, true),
            )
            .err()
            .expect("an inverted target should be rejected");

        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(
            error
                .to_string()
                .contains("JSON-path indexes do not support target index type 'inverted'"),
            "{error}"
        );
    }

    #[test]
    fn test_json_index_rejects_large_binary_target() {
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let error = plugin
            .basic_trainer()
            .unwrap()
            .new_training_request(
                r#"{"target_index_type":"btree","target_data_type":"LargeBinary","path":"$.v"}"#,
                &Field::new(VALUE_COLUMN_NAME, DataType::LargeBinary, true),
            )
            .err()
            .expect("a LargeBinary target should be rejected");

        assert!(matches!(error, Error::InvalidInput { .. }));
        let message = error.to_string();
        assert!(message.contains("JSON path '$.v'"), "{message}");
        assert!(message.contains("LargeBinary"), "{message}");
        assert!(message.contains("only scalar values"), "{message}");
    }

    #[rstest]
    #[case::array(r#"{"v": [1, 2]}"#, JsonbType::Array)]
    #[case::empty_array(r#"{"v": []}"#, JsonbType::Array)]
    #[case::null_array_items(r#"{"v": [null, null]}"#, JsonbType::Array)]
    #[case::object(r#"{"v": {"a": 1}}"#, JsonbType::Object)]
    #[case::empty_object(r#"{"v": {}}"#, JsonbType::Object)]
    #[case::null_object_value(r#"{"v": {"a": null}}"#, JsonbType::Object)]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_index_rejects_non_scalar_path(
        #[case] json_doc: &str,
        #[case] json_type: JsonbType,
    ) {
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let trainer = plugin.basic_trainer().unwrap();
        let request = trainer
            .new_training_request(
                r#"{"target_index_type":"btree","path":"$.v"}"#,
                &Field::new(VALUE_COLUMN_NAME, DataType::LargeBinary, true),
            )
            .unwrap();

        // Inference must skip null and missing values, including an all-null batch.
        let null_batch = json_update_batch(&[r#"{"v": null}"#, r#"{}"#], vec![0, 1]);
        let value_batch = json_update_batch(&[json_doc], vec![2]);
        let data = Box::pin(RecordBatchStreamAdapter::new(
            null_batch.schema(),
            futures::stream::iter([Ok(null_batch), Ok(value_batch)]),
        )) as SendableRecordBatchStream;
        let (store, _tmpdir) = local_json_index_store();
        let error = trainer
            .train_index(
                data,
                store.as_ref(),
                request,
                None,
                crate::progress::noop_progress(),
            )
            .await
            .err()
            .expect("an array or object path should be rejected");

        assert!(matches!(error, Error::InvalidInput { .. }));
        let message = error.to_string();
        assert!(message.contains("JSON path '$.v'"), "{message}");
        assert!(
            message.contains(&format!("JSON type {json_type:?}")),
            "{message}"
        );
        assert!(message.contains("only scalar values"), "{message}");
    }

    #[test]
    #[allow(deprecated)]
    fn test_json_wrapped_target_type() {
        assert_eq!(
            json_wrapped_target_type(r#"{"target_index_type":"btree","path":"x"}"#),
            Some("btree".to_string())
        );
        // Case is preserved here; callers that care compare case-insensitively.
        assert_eq!(
            json_wrapped_target_type(r#"{"target_index_type":"BTREE","path":"x"}"#),
            Some("BTREE".to_string())
        );
        assert_eq!(json_wrapped_target_type("not json"), None);
        assert_eq!(json_wrapped_target_type(r#"{"path":"x"}"#), None);
    }

    #[rstest]
    #[case::btree(
        prost_types::Any::from_msg(&crate::pbold::BTreeIndexDetails::default()).unwrap(),
        true
    )]
    #[case::zonemap(
        prost_types::Any::from_msg(&crate::pbold::ZoneMapIndexDetails::default()).unwrap(),
        false
    )]
    fn test_json_can_remap_delegates_to_target(
        #[case] target_details: prost_types::Any,
        #[case] expected: bool,
    ) {
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let details = prost_types::Any::from_msg(&crate::pb::JsonIndexDetails {
            path: "$.value".to_string(),
            target_details: Some(target_details),
            ..Default::default()
        })
        .unwrap();

        assert_eq!(plugin.can_remap(&details).unwrap(), expected);
    }

    #[rstest]
    #[case::missing_target(None)]
    #[case::unknown_target(
        Some(prost_types::Any {
            type_url: "type.googleapis.com/example.UnknownIndexDetails".to_string(),
            value: Vec::new(),
        })
    )]
    fn test_json_can_remap_defaults_to_false_for_invalid_target(
        #[case] target_details: Option<prost_types::Any>,
    ) {
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let details = prost_types::Any::from_msg(&crate::pb::JsonIndexDetails {
            path: "$.value".to_string(),
            target_details,
            ..Default::default()
        })
        .unwrap();

        assert!(!plugin.can_remap(&details).unwrap());
    }

    #[test]
    fn test_json_can_remap_defaults_to_false_for_malformed_details() {
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let details = prost_types::Any {
            type_url: String::new(),
            value: b"not a json index".to_vec(),
        };

        assert!(!plugin.can_remap(&details).unwrap());
    }

    #[test]
    fn test_json_query_parser_forwards_target_details() {
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let target_details = prost_types::Any::from_msg(&crate::pbold::ZoneMapIndexDetails {
            rows_per_zone: Some(128),
            use_seeds: Some(false),
            has_null_bitmap: Some(true),
        })
        .unwrap();
        let index_details = prost_types::Any::from_msg(&crate::pb::JsonIndexDetails {
            path: "$.value".to_string(),
            target_details: Some(target_details),
            target_data_type: crate::pb::JsonTargetDataType::Unspecified as i32,
            target_index_version: None,
        })
        .unwrap();

        let parser = plugin
            .new_query_parser("json_value".to_string(), &index_details)
            .expect("zone map target should provide a query parser");
        let expression = parser
            .visit_is_null("doc")
            .expect("zone map target should parse IS NULL");
        let Some(ScalarIndexExpr::Query(search)) = expression.scalar_query else {
            panic!("JSON parser should wrap the target index query");
        };
        assert_eq!(search.index_type, "ZoneMap");
        let query = search.query.as_any().downcast_ref::<JsonQuery>().unwrap();
        assert_eq!(query.path(), "$.value");
        assert!(matches!(
            query
                .target_query()
                .as_any()
                .downcast_ref::<SargableQuery>(),
            Some(SargableQuery::IsNull())
        ));
        let rebuilt = JsonQuery::new(query.target_query().clone(), query.path().to_owned());
        assert!(search.query.dyn_eq(&rebuilt));
        assert!(
            !search.needs_recheck,
            "the target's exact null-tracking detail should reach its query parser"
        );
    }

    #[test]
    fn test_json_query_parser_keeps_inverted_target_inert() {
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let target_details = prost_types::Any::from_msg(&crate::pbold::InvertedIndexDetails {
            base_tokenizer: Some("simple".to_string()),
            language: serde_json::to_string(&crate::scalar::inverted::Language::English).unwrap(),
            ..Default::default()
        })
        .unwrap();
        let index_details = prost_types::Any::from_msg(&crate::pb::JsonIndexDetails {
            path: "$.value".to_string(),
            target_details: Some(target_details),
            target_data_type: crate::pb::JsonTargetDataType::Unspecified as i32,
            target_index_version: None,
        })
        .unwrap();

        assert!(
            plugin
                .new_query_parser("json_value".to_string(), &index_details)
                .is_none(),
            "legacy JSON-wrapped inverted indexes must remain unroutable"
        );
    }

    // Note: The old test_detect_json_value_type test has been removed as we now use
    // JSONB's inherent type information instead of string-based type detection

    #[tokio::test]
    async fn test_json_extract_with_type_info() {
        use arrow_array::{LargeBinaryArray, UInt64Array};
        use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
        use futures::stream;

        // Create test JSONB data
        let json_data = vec![
            r#"{"name": "Alice", "age": 30, "active": true}"#,
            r#"{"name": "Bob", "age": 25, "active": false}"#,
            r#"{"name": "Charlie", "age": 35, "active": true}"#,
        ];

        // Convert JSON strings to JSONB binary format
        let mut jsonb_values = Vec::new();
        for json_str in &json_data {
            let owned_jsonb: lance_jsonb::OwnedJsonb = json_str.parse().unwrap();
            jsonb_values.push(Some(owned_jsonb.to_vec()));
        }

        // Create test batch with JSONB data
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::LargeBinary, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));

        let jsonb_array = LargeBinaryArray::from(
            jsonb_values
                .iter()
                .map(|v| v.as_deref())
                .collect::<Vec<_>>(),
        );
        let row_ids = UInt64Array::from(vec![1, 2, 3]);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(jsonb_array) as ArrayRef,
                Arc::new(row_ids) as ArrayRef,
            ],
        )
        .unwrap();

        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![Ok(batch)]),
        )) as SendableRecordBatchStream;

        // Test type inference for integer field
        let (_result_stream, inferred_type) =
            JsonIndexPlugin::extract_json_with_type_info(stream, "$.age".to_string())
                .await
                .unwrap();

        assert_eq!(inferred_type, DataType::Int64);

        // Create new test stream for boolean field
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(LargeBinaryArray::from(vec![
                    json_data[0]
                        .parse::<lance_jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                    json_data[1]
                        .parse::<lance_jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                    json_data[2]
                        .parse::<lance_jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                ])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![1, 2, 3])) as ArrayRef,
            ],
        )
        .unwrap();

        let stream2 = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::iter(vec![Ok(batch2)]),
        )) as SendableRecordBatchStream;

        // Test type inference for boolean field
        let (_, inferred_type) =
            JsonIndexPlugin::extract_json_with_type_info(stream2, "$.active".to_string())
                .await
                .unwrap();

        assert_eq!(inferred_type, DataType::Boolean);

        // Create test stream for string field
        let batch3 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(LargeBinaryArray::from(vec![
                    json_data[0]
                        .parse::<lance_jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                    json_data[1]
                        .parse::<lance_jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                    json_data[2]
                        .parse::<lance_jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                ])) as ArrayRef,
                Arc::new(UInt64Array::from(vec![1, 2, 3])) as ArrayRef,
            ],
        )
        .unwrap();

        let stream3 = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![Ok(batch3)]),
        )) as SendableRecordBatchStream;

        // Test type inference for string field
        let (_, inferred_type) =
            JsonIndexPlugin::extract_json_with_type_info(stream3, "$.name".to_string())
                .await
                .unwrap();

        assert_eq!(inferred_type, DataType::Utf8);
    }

    /// Trains a JSON-path index of `target_index_type` over `json_docs` (fed to the
    /// trainer in exactly the given order, with row ids `0..json_docs.len()`) and
    /// returns the loaded index. `store` is a caller-owned `LanceIndexStore` so the
    /// caller controls how long the backing `TempObjDir` stays alive.
    async fn train_and_load_json_index(
        store: Arc<dyn IndexStore>,
        target_index_type: &str,
        expected_ordering: TrainingOrdering,
        path: &str,
        json_docs: &[&str],
    ) -> Arc<dyn ScalarIndex> {
        let created = train_json_index(
            store.clone(),
            target_index_type,
            expected_ordering,
            path,
            json_docs,
        )
        .await;
        load_json_index(store, &created.index_details).await
    }

    /// Train a JSON index without loading it.
    async fn train_json_index(
        store: Arc<dyn IndexStore>,
        target_index_type: &str,
        expected_ordering: TrainingOrdering,
        path: &str,
        json_docs: &[&str],
    ) -> CreatedIndex {
        use crate::progress::noop_progress;
        use arrow_array::{LargeBinaryArray, UInt64Array};
        use futures::stream;

        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let trainer = plugin.basic_trainer().unwrap();
        let params = format!(r#"{{"target_index_type":"{target_index_type}","path":"{path}"}}"#);
        let request = trainer
            .new_training_request(
                &params,
                &Field::new(VALUE_COLUMN_NAME, DataType::LargeBinary, true),
            )
            .unwrap();

        assert_eq!(request.criteria().ordering, expected_ordering);

        let jsonb: Vec<Vec<u8>> = json_docs
            .iter()
            .map(|s| s.parse::<lance_jsonb::OwnedJsonb>().unwrap().to_vec())
            .collect();

        let mut fields = Vec::with_capacity(3);
        fields.push(Field::new(VALUE_COLUMN_NAME, DataType::LargeBinary, true));
        let mut columns = Vec::with_capacity(3);
        columns.push(Arc::new(LargeBinaryArray::from(
            jsonb.iter().map(|v| Some(v.as_slice())).collect::<Vec<_>>(),
        )) as ArrayRef);
        if request.criteria().needs_row_ids {
            fields.push(Field::new(ROW_ID, DataType::UInt64, false));
            columns.push(Arc::new(UInt64Array::from(
                (0..json_docs.len() as u64).collect::<Vec<_>>(),
            )) as ArrayRef);
        }
        if request.criteria().needs_row_addrs {
            fields.push(Field::new(ROW_ADDR, DataType::UInt64, false));
            columns.push(Arc::new(UInt64Array::from(
                (0..json_docs.len())
                    .map(|row_idx| {
                        RowAddress::new_from_parts((row_idx / 2) as u32, (row_idx % 2) as u32)
                            .into()
                    })
                    .collect::<Vec<u64>>(),
            )) as ArrayRef);
        }
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(vec![Ok(batch)]),
        )) as SendableRecordBatchStream;

        trainer
            .train_index(data, store.as_ref(), request, None, noop_progress())
            .await
            .unwrap()
    }

    fn local_json_index_store() -> (Arc<dyn IndexStore>, lance_core::utils::tempfile::TempObjDir) {
        use crate::scalar::lance_format::LanceIndexStore;
        use lance_core::utils::tempfile::TempObjDir;
        use lance_io::object_store::ObjectStore;

        let tmpdir = TempObjDir::default();
        let store = Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            tmpdir.clone(),
            Arc::new(LanceCache::no_cache()),
        )) as Arc<dyn IndexStore>;
        (store, tmpdir)
    }

    fn json_update_batch(json_docs: &[&str], row_addrs: Vec<u64>) -> RecordBatch {
        use arrow_array::{LargeBinaryArray, UInt64Array};

        let jsonb = json_docs
            .iter()
            .map(|json| json.parse::<lance_jsonb::OwnedJsonb>().unwrap().to_vec())
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::LargeBinary, true),
            Field::new(ROW_ADDR, DataType::UInt64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(LargeBinaryArray::from_iter_values(
                    jsonb.iter().map(Vec::as_slice),
                )),
                Arc::new(UInt64Array::from(row_addrs)),
            ],
        )
        .unwrap()
    }

    fn json_update_stream(json_docs: &[&str], row_addrs: Vec<u64>) -> SendableRecordBatchStream {
        use futures::stream;

        let batch = json_update_batch(json_docs, row_addrs);
        let schema = batch.schema();
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter([Ok(batch)]),
        ))
    }

    #[rstest]
    #[case::all_null_delta(
        &[r#"{"v": 1}"#],
        &[r#"{"v": null}"#, r#"{"other": 2}"#],
        SargableQuery::IsNull(),
        vec![1, 2]
    )]
    #[case::integer_delta_for_float_index(
        &[r#"{"v": 1.5}"#],
        &[r#"{"v": 2}"#],
        SargableQuery::Equals(ScalarValue::Float64(Some(2.0))),
        vec![1]
    )]
    // Spill-enabled index builds share the cached DataFusion memory pool within the
    // test process, so keep them in one resource group.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_btree_update_uses_trained_target_type(
        #[case] initial_docs: &[&str],
        #[case] update_docs: &[&str],
        #[case] query: SargableQuery,
        #[case] expected_row_ids: Vec<u64>,
    ) {
        use crate::metrics::NoOpMetricsCollector;
        use lance_select::RowAddrTreeMap;

        let (source_store, _source_dir) = local_json_index_store();
        let index = train_and_load_json_index(
            source_store,
            "btree",
            TrainingOrdering::None,
            "v",
            initial_docs,
        )
        .await;
        let expected_type = if matches!(&query, SargableQuery::IsNull()) {
            DataType::Int64
        } else {
            DataType::Float64
        };
        assert_eq!(index.training_data_type(), Some(expected_type));

        let (dest_store, _dest_dir) = local_json_index_store();
        let row_ids =
            (initial_docs.len() as u64..(initial_docs.len() + update_docs.len()) as u64).collect();
        let created = index
            .update(
                json_update_stream(update_docs, row_ids),
                dest_store.as_ref(),
                None,
            )
            .await
            .unwrap();

        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        let updated = plugin
            .load_index(
                dest_store,
                &created.index_details,
                0,
                None,
                &LanceCache::no_cache(),
            )
            .await
            .unwrap();
        let result = updated
            .search(
                &JsonQuery::new(Arc::new(query), "v".to_string()),
                &NoOpMetricsCollector,
            )
            .await
            .unwrap();
        assert_eq!(
            result,
            SearchResult::exact(RowAddrTreeMap::from_iter(expected_row_ids))
        );
    }

    /// JSON text input is indexed from the same values as stored JSONB.
    #[tokio::test]
    async fn test_json_text_input_extracts_like_jsonb() {
        use arrow_array::{Int64Array, StringArray};
        use futures::{TryStreamExt, stream};
        use lance_arrow::ARROW_EXT_NAME_KEY;
        use lance_arrow::json::ARROW_JSON_EXT_NAME;

        let docs = [r#"{"v": 1}"#, r#"{"v": 2}"#];
        let jsonb = json_update_batch(&docs, vec![0, 1]);
        let text_field = Field::new(VALUE_COLUMN_NAME, DataType::Utf8, true).with_metadata(
            std::collections::HashMap::from([(
                ARROW_EXT_NAME_KEY.to_string(),
                ARROW_JSON_EXT_NAME.to_string(),
            )]),
        );
        let text_schema = Arc::new(Schema::new(vec![
            text_field,
            jsonb.schema().field(1).clone(),
        ]));
        let text = RecordBatch::try_new(
            text_schema.clone(),
            vec![
                Arc::new(StringArray::from(docs.to_vec())),
                jsonb.column(1).clone(),
            ],
        )
        .unwrap();

        let mut extracted = Vec::new();
        for batch in [jsonb, text] {
            let schema = batch.schema();
            let raw_stream = Box::pin(RecordBatchStreamAdapter::new(
                schema,
                stream::iter([Ok(batch)]),
            )) as SendableRecordBatchStream;
            let converted = JsonIndexPlugin::convert_stream_by_type(
                JsonIndexPlugin::extract_json(raw_stream, "v".to_string()).unwrap(),
                DataType::Int64,
                "v".to_string(),
            )
            .unwrap();
            let batches = converted.try_collect::<Vec<_>>().await.unwrap();
            extracted.push(
                batches[0][VALUE_COLUMN_NAME]
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .clone(),
            );
        }
        assert_eq!(extracted[0], Int64Array::from(vec![1, 2]));
        assert_eq!(extracted[1], extracted[0]);
    }

    #[tokio::test]
    async fn test_json_conversion_is_streaming() {
        use arrow_array::Int64Array;
        use futures::stream;

        let first_batch = json_update_batch(&[r#"{"v": 1}"#], vec![0]);
        let schema = first_batch.schema();
        let raw_stream = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter([
                Ok(first_batch),
                Err(DataFusionError::Execution(
                    "second batch must not be polled for the first output".to_string(),
                )),
            ]),
        )) as SendableRecordBatchStream;

        let extracted = JsonIndexPlugin::extract_json(raw_stream, "v".to_string()).unwrap();
        let mut converted =
            JsonIndexPlugin::convert_stream_by_type(extracted, DataType::Int64, "v".to_string())
                .unwrap();

        let first = converted.next().await.unwrap().unwrap();
        let values = first[VALUE_COLUMN_NAME]
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(values.value(0), 1);
        let error = converted.next().await.unwrap().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("second batch must not be polled")
        );
    }

    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_btree_update_reports_type_drift() {
        let (source_store, _source_dir) = local_json_index_store();
        let index = train_and_load_json_index(
            source_store,
            "btree",
            TrainingOrdering::None,
            "v",
            &[r#"{"v": 1}"#],
        )
        .await;
        let (dest_store, _dest_dir) = local_json_index_store();
        let error = index
            .update(
                json_update_stream(&[r#"{"v": true}"#], vec![1]),
                dest_store.as_ref(),
                None,
            )
            .await
            .err()
            .expect("type drift should fail the update");
        let message = error.to_string();
        assert!(message.contains("JSON path 'v'"), "{message}");
        assert!(message.contains("expected Int64"), "{message}");
        assert!(message.contains("JSON type Boolean"), "{message}");
    }

    /// `bitmap` covers the target types that do not implement
    /// `training_data_type`, where the derived type can only come from the
    /// details.
    #[rstest]
    #[case::btree("btree", true)]
    #[case::bitmap("bitmap", false)]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_derived_params_preserve_wrapper(
        #[case] target_index_type: &str,
        // Only some targets round-trip their own parameters; bitmap has none.
        #[case] expect_target_parameters: bool,
    ) {
        let (store, _tmpdir) = local_json_index_store();
        let index = train_and_load_json_index(
            store,
            target_index_type,
            TrainingOrdering::None,
            "v",
            &[r#"{"v": 1}"#],
        )
        .await;

        let derived = index.derive_index_params().unwrap();
        assert_eq!(derived.index_type, "json");
        let raw_params = derived.params.as_deref().unwrap();
        assert!(
            raw_params.contains(r#""target_data_type":"Int64""#),
            "unexpected parameter spelling: {raw_params}"
        );
        let parameters: JsonIndexParameters = serde_json::from_str(raw_params).unwrap();
        assert_eq!(parameters.path, "v");
        assert_eq!(parameters.target_index_type, target_index_type);
        assert_eq!(
            parameters.target_index_parameters.is_some(),
            expect_target_parameters
        );
        assert_eq!(
            parameters.target_data_type,
            Some(JsonIndexTargetType::Int64)
        );
    }

    fn decode_details(details: &prost_types::Any) -> crate::pb::JsonIndexDetails {
        crate::pb::JsonIndexDetails::decode(details.value.as_slice()).unwrap()
    }

    fn decode_target_data_type(details: &prost_types::Any) -> i32 {
        decode_details(details).target_data_type
    }

    /// Rewrite `target_data_type` in `details`.
    ///
    /// With `Unspecified` this produces the details a build that predates the
    /// field would have written; with an out-of-range value, the details a build
    /// that added a variant this one does not know would have written.
    fn details_with_target_data_type(details: &prost_types::Any, value: i32) -> prost_types::Any {
        let mut decoded = decode_details(details);
        decoded.target_data_type = value;
        prost_types::Any::from_msg(&decoded).unwrap()
    }

    async fn load_json_index(
        store: Arc<dyn IndexStore>,
        details: &prost_types::Any,
    ) -> Arc<dyn ScalarIndex> {
        let registry = IndexPluginRegistry::with_default_plugins();
        registry
            .get_plugin_by_name("json")
            .unwrap()
            .load_index(store, details, 0, None, &LanceCache::no_cache())
            .await
            .unwrap()
    }

    /// A fresh build records the type it trained on, including when it inferred
    /// that type from the data, and including for targets that cannot report it
    /// back afterwards. `expected_json_name` is the spelling that reaches
    /// `list_indices`, which is user-visible and so pinned per variant.
    #[rstest]
    #[case::btree_int64("btree", r#"{"v": 1}"#, crate::pb::JsonTargetDataType::Int64, "int64")]
    #[case::btree_float64(
        "btree",
        r#"{"v": 1.5}"#,
        crate::pb::JsonTargetDataType::Float64,
        "float64"
    )]
    #[case::btree_utf8("btree", r#"{"v": "a"}"#, crate::pb::JsonTargetDataType::Utf8, "utf8")]
    #[case::btree_boolean(
        "btree",
        r#"{"v": true}"#,
        crate::pb::JsonTargetDataType::Boolean,
        "boolean"
    )]
    // Nothing to read a type tag from, so inference falls back to Utf8 and the
    // build records the fallback rather than leaving the type unspecified.
    #[case::btree_absent_path(
        "btree",
        r#"{"other": 1}"#,
        crate::pb::JsonTargetDataType::Utf8,
        "utf8"
    )]
    #[case::btree_explicit_null(
        "btree",
        r#"{"v": null}"#,
        crate::pb::JsonTargetDataType::Utf8,
        "utf8"
    )]
    #[case::bitmap_utf8("bitmap", r#"{"v": "a"}"#, crate::pb::JsonTargetDataType::Utf8, "utf8")]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_details_record_target_data_type(
        #[case] target_index_type: &str,
        #[case] json_doc: &str,
        #[case] expected: crate::pb::JsonTargetDataType,
        #[case] expected_json_name: &str,
    ) {
        let (store, _tmpdir) = local_json_index_store();
        let details = train_json_index(
            store,
            target_index_type,
            TrainingOrdering::None,
            "v",
            &[json_doc],
        )
        .await
        .index_details;

        let decoded = decode_details(&details);
        assert_eq!(decoded.target_data_type, expected as i32);
        assert!(
            decoded.target_index_version.is_some(),
            "a fresh build must record the target index version"
        );

        let registry = IndexPluginRegistry::with_default_plugins();
        let details_json = registry
            .get_plugin_by_name("json")
            .unwrap()
            .details_as_json(&details)
            .unwrap();
        assert_eq!(details_json["target_data_type"], expected_json_name);
    }

    /// Compaction must not drop the recorded type. `bitmap` is the case that
    /// re-deriving from the target index cannot serve.
    #[rstest]
    #[case::btree("btree")]
    #[case::bitmap("bitmap")]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_details_target_data_type_survives_remap(#[case] target_index_type: &str) {
        let (store, _tmpdir) = local_json_index_store();
        let details = train_json_index(
            store.clone(),
            target_index_type,
            TrainingOrdering::None,
            "v",
            &[r#"{"v": "a"}"#, r#"{"v": "b"}"#],
        )
        .await
        .index_details;
        let index = load_json_index(store, &details).await;
        assert_eq!(
            decode_target_data_type(&details),
            crate::pb::JsonTargetDataType::Utf8 as i32
        );
        let original_target_version = decode_details(&details).target_index_version;
        assert!(original_target_version.is_some());

        let (dest_store, _dest_dir) = local_json_index_store();
        let remapped = index
            .remap(&RowAddrRemap::empty(), dest_store.as_ref())
            .await
            .unwrap();

        assert_eq!(
            decode_target_data_type(&remapped.index_details),
            crate::pb::JsonTargetDataType::Utf8 as i32
        );
        assert_eq!(
            decode_details(&remapped.index_details).target_index_version,
            original_target_version
        );
    }

    /// Details written before this field existed must keep working: the index
    /// loads, answers queries, and can still be compacted.
    #[rstest]
    #[case::btree("btree")]
    #[case::bitmap("bitmap")]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_legacy_details_without_target_data_type(#[case] target_index_type: &str) {
        use crate::metrics::NoOpMetricsCollector;
        use lance_select::RowAddrTreeMap;

        let (store, _tmpdir) = local_json_index_store();
        let details = train_json_index(
            store.clone(),
            target_index_type,
            TrainingOrdering::None,
            "v",
            &[r#"{"v": 1}"#, r#"{"v": 2}"#],
        )
        .await
        .index_details;
        let legacy_details = details_with_target_data_type(
            &details,
            crate::pb::JsonTargetDataType::Unspecified as i32,
        );

        let index = load_json_index(store, &legacy_details).await;

        let query = JsonQuery::new(
            Arc::new(SargableQuery::Equals(ScalarValue::Int64(Some(2)))),
            "v".to_string(),
        );
        let result = index.search(&query, &NoOpMetricsCollector).await.unwrap();
        assert_eq!(
            result,
            SearchResult::exact(RowAddrTreeMap::from_iter([1u64])),
            "a legacy index must still answer queries"
        );

        let (dest_store, _dest_dir) = local_json_index_store();
        let remapped = index
            .remap(&RowAddrRemap::empty(), dest_store.as_ref())
            .await
            .unwrap();

        // BTree can report its training type, so compacting upgrades the details
        // in passing. Targets that cannot report it stay as lossy as they were.
        let expected = if target_index_type == "btree" {
            crate::pb::JsonTargetDataType::Int64 as i32
        } else {
            crate::pb::JsonTargetDataType::Unspecified as i32
        };
        assert_eq!(decode_target_data_type(&remapped.index_details), expected);

        // Same split for a rebuild: BTree recovers the type, and a target that
        // cannot report one must derive no type so the rebuild re-infers it
        // rather than being handed a wrong one.
        let derived = index.derive_index_params().unwrap();
        let parameters: JsonIndexParameters =
            serde_json::from_str(derived.params.as_deref().unwrap()).unwrap();
        let expected_derived = (target_index_type == "btree").then_some(JsonIndexTargetType::Int64);
        assert_eq!(parameters.target_data_type, expected_derived);
    }

    /// A variant added after this build must survive a compaction this build
    /// performs, rather than being erased on the way through.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_unknown_target_data_type_survives_remap() {
        const FUTURE_VARIANT: i32 = 99;

        let (store, _tmpdir) = local_json_index_store();
        let details = train_json_index(
            store.clone(),
            "btree",
            TrainingOrdering::None,
            "v",
            &[r#"{"v": 1}"#],
        )
        .await
        .index_details;
        let future_details = details_with_target_data_type(&details, FUTURE_VARIANT);

        let index = load_json_index(store, &future_details).await;
        let (dest_store, _dest_dir) = local_json_index_store();
        let remapped = index
            .remap(&RowAddrRemap::empty(), dest_store.as_ref())
            .await
            .unwrap();

        assert_eq!(
            decode_target_data_type(&remapped.index_details),
            FUTURE_VARIANT
        );
    }

    /// `update` rewrites the details without retraining the target, so it must
    /// copy the recorded type through. An unrecognized variant is the case that
    /// distinguishes copying from re-deriving: BTree would answer `Int64` here.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_unknown_target_data_type_survives_update() {
        const FUTURE_VARIANT: i32 = 99;

        let (store, _tmpdir) = local_json_index_store();
        let details = train_json_index(
            store.clone(),
            "btree",
            TrainingOrdering::None,
            "v",
            &[r#"{"v": 1}"#],
        )
        .await
        .index_details;
        let future_details = details_with_target_data_type(&details, FUTURE_VARIANT);

        let index = load_json_index(store, &future_details).await;
        let (dest_store, _dest_dir) = local_json_index_store();
        let updated = index
            .update(
                json_update_stream(&[r#"{"v": 2}"#], vec![1]),
                dest_store.as_ref(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            decode_target_data_type(&updated.index_details),
            FUTURE_VARIANT
        );
        assert_eq!(
            decode_details(&updated.index_details).target_index_version,
            decode_details(&details).target_index_version
        );
    }

    /// An index built before the field existed is upgraded by an `update`,
    /// which -- unlike `remap` -- already requires the target to report the type
    /// it was trained on.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_update_records_target_data_type_for_legacy_details() {
        let (store, _tmpdir) = local_json_index_store();
        let details = train_json_index(
            store.clone(),
            "btree",
            TrainingOrdering::None,
            "v",
            &[r#"{"v": 1}"#],
        )
        .await
        .index_details;
        let legacy_details = details_with_target_data_type(
            &details,
            crate::pb::JsonTargetDataType::Unspecified as i32,
        );

        let index = load_json_index(store, &legacy_details).await;
        let (dest_store, _dest_dir) = local_json_index_store();
        let updated = index
            .update(
                json_update_stream(&[r#"{"v": 2}"#], vec![1]),
                dest_store.as_ref(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            decode_target_data_type(&updated.index_details),
            crate::pb::JsonTargetDataType::Int64 as i32
        );
    }

    /// `details_as_json` feeds `list_indices`, so the recorded type has to be
    /// visible there, and a legacy index's details have to read as they always
    /// have.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_details_as_json_reports_target_data_type() {
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name("json").unwrap();

        let (store, _tmpdir) = local_json_index_store();
        let details = train_json_index(
            store,
            "btree",
            TrainingOrdering::None,
            "v",
            &[r#"{"v": 1}"#],
        )
        .await
        .index_details;

        assert_eq!(
            plugin.details_as_json(&details).unwrap(),
            serde_json::json!({
                "path": "v",
                "target_details": {},
                "target_data_type": "int64",
            })
        );

        let legacy_details = details_with_target_data_type(
            &details,
            crate::pb::JsonTargetDataType::Unspecified as i32,
        );
        assert_eq!(
            plugin.details_as_json(&legacy_details).unwrap(),
            serde_json::json!({
                "path": "v",
                "target_details": {},
            })
        );

        // A variant this build does not know is still reported, as a string so
        // that the type of the field does not depend on who wrote the index.
        let future_details = details_with_target_data_type(&details, 99);
        assert_eq!(
            plugin.details_as_json(&future_details).unwrap(),
            serde_json::json!({
                "path": "v",
                "target_details": {},
                "target_data_type": "unknown(99)",
            })
        );
    }

    /// Before `target_index_version` was recorded, `JsonIndexPlugin::load_index`
    /// passed the target plugin's current max version unconditionally, so a
    /// freshly trained JSON-wrapped Bitmap/BTree target -- whose postings are
    /// always address-valued (see `BitmapTrainingRequest`'s and the BTree
    /// trainer's own criteria) -- would load as address-domain even though
    /// those builds only ever stored row ids through this wrapper. Details
    /// missing `target_index_version` (the legacy case `load_index`'s fallback
    /// covers) must still load the target at its row-id-domain floor version
    /// rather than the plugin's current one.
    ///
    /// LabelList shares the same floor-version handling in `load_index` but
    /// has no case here: JSON path extraction rejects array/object values
    /// outright (see `test_json_index_rejects_non_scalar_path`), so a
    /// LabelList target -- which requires a List/LargeList column -- can
    /// never actually be trained through this wrapper to exercise it.
    #[rstest]
    #[case::btree("btree")]
    #[case::bitmap("bitmap")]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_legacy_details_load_target_as_row_ids(#[case] target_index_type: &str) {
        let (store, _tmpdir) = local_json_index_store();
        let created = train_json_index(
            store.clone(),
            target_index_type,
            TrainingOrdering::None,
            "v",
            &[r#"{"v": 1}"#, r#"{"v": 2}"#],
        )
        .await;
        let mut details = decode_details(&created.index_details);
        details.target_index_version = None;
        let legacy_details = prost_types::Any::from_msg(&details).unwrap();

        let index = load_json_index(store, &legacy_details).await;
        let json_index = index.as_any().downcast_ref::<JsonIndex>().unwrap();
        assert!(
            !json_index.target_index.results_are_row_addresses(),
            "a JSON-wrapped {target_index_type} target with no recorded target_index_version \
             must load at its row-id-domain floor version, not its plugin's current \
             (address-domain) max version"
        );
    }

    /// A freshly trained JSON segment records its target's own persisted
    /// version, and the wrapper's version carries that same domain so that
    /// `IndexMetadata::results_are_row_addrs` can classify it without decoding
    /// the nested target details.
    #[rstest]
    #[case::btree("btree", TrainingOrdering::None, &[r#"{"v": 1}"#, r#"{"v": 2}"#], true)]
    #[case::bitmap("bitmap", TrainingOrdering::None, &[r#"{"v": 1}"#, r#"{"v": 2}"#], true)]
    #[case::zonemap("zonemap", TrainingOrdering::Addresses, &[r#"{"v": 1}"#, r#"{"v": 2}"#], true)]
    #[case::ngram("ngram", TrainingOrdering::None, &[r#"{"v": "alpha"}"#, r#"{"v": "bravo"}"#], false)]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_index_version_tracks_target_domain(
        #[case] target_index_type: &str,
        #[case] expected_ordering: TrainingOrdering,
        #[case] json_docs: &[&str],
        #[case] address_domain: bool,
    ) {
        let (store, _tmpdir) = local_json_index_store();
        let created = train_json_index(
            store.clone(),
            target_index_type,
            expected_ordering,
            "v",
            json_docs,
        )
        .await;

        let expected_version = if address_domain {
            JSON_ROW_ADDR_DOMAIN_VERSION
        } else {
            JSON_ROW_ID_DOMAIN_VERSION
        };
        assert_eq!(created.index_version, expected_version);
        assert!(
            decode_details(&created.index_details)
                .target_index_version
                .is_some()
        );

        let index = load_json_index(store, &created.index_details).await;
        let json_index = index.as_any().downcast_ref::<JsonIndex>().unwrap();
        assert_eq!(
            json_index.target_index.results_are_row_addresses(),
            address_domain
        );
    }

    /// Relabeling (for a dataset without stable row ids) moves both the wrapper
    /// and its target to their row-id-domain versions, and `update` keeps the
    /// relabeled segment there rather than bumping it back.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_relabel_as_row_id_domain() {
        let (store, _tmpdir) = local_json_index_store();
        let mut created = train_json_index(
            store.clone(),
            "btree",
            TrainingOrdering::None,
            "v",
            &[r#"{"v": 1}"#, r#"{"v": 2}"#],
        )
        .await;
        relabel_as_row_id_domain(&mut created).unwrap();
        assert_eq!(created.index_version, JSON_ROW_ID_DOMAIN_VERSION);
        assert_eq!(
            decode_details(&created.index_details).target_index_version,
            Some(crate::scalar::btree::BTREE_ROW_ID_DOMAIN_VERSION)
        );

        let index = load_json_index(store, &created.index_details).await;
        let json_index = index.as_any().downcast_ref::<JsonIndex>().unwrap();
        assert!(!json_index.target_index.results_are_row_addresses());

        let (dest_store, _dest_dir) = local_json_index_store();
        let updated = index
            .update(
                json_update_stream(&[r#"{"v": 3}"#], vec![2]),
                dest_store.as_ref(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(updated.index_version, JSON_ROW_ID_DOMAIN_VERSION);
        assert_eq!(
            decode_details(&updated.index_details).target_index_version,
            Some(crate::scalar::btree::BTREE_ROW_ID_DOMAIN_VERSION)
        );
    }

    #[test]
    fn test_wraps_row_addr_domain_target() {
        let json_over = |target: prost_types::Any| {
            prost_types::Any::from_msg(&crate::pb::JsonIndexDetails {
                path: "v".to_string(),
                target_details: Some(target),
                target_data_type: crate::pb::JsonTargetDataType::Unspecified as i32,
                target_index_version: None,
            })
            .unwrap()
        };
        let btree =
            prost_types::Any::from_msg(&crate::pbold::BTreeIndexDetails::default()).unwrap();
        let bitmap =
            prost_types::Any::from_msg(&crate::pbold::BitmapIndexDetails::default()).unwrap();
        let ngram =
            prost_types::Any::from_msg(&crate::pbold::NGramIndexDetails::default()).unwrap();
        let zonemap =
            prost_types::Any::from_msg(&crate::pbold::ZoneMapIndexDetails::default()).unwrap();

        assert!(wraps_row_addr_domain_target(&json_over(btree.clone())));
        assert!(wraps_row_addr_domain_target(&json_over(bitmap)));
        assert!(wraps_row_addr_domain_target(&json_over(zonemap)));
        assert!(!wraps_row_addr_domain_target(&json_over(ngram)));
        assert!(!wraps_row_addr_domain_target(&btree));
    }

    /// Regression test for https://github.com/lance-format/lance/issues/7859.
    #[rstest]
    #[case::zonemap("zonemap", TrainingOrdering::Addresses)]
    #[case::fm("fm", TrainingOrdering::None)]
    #[tokio::test]
    async fn test_json_index_preserves_row_addresses(
        #[case] target_index_type: &str,
        #[case] expected_ordering: TrainingOrdering,
    ) {
        let (store, _tmpdir) = local_json_index_store();
        let index = train_and_load_json_index(
            store,
            target_index_type,
            expected_ordering,
            "value",
            &[
                r#"{"value": "alpha"}"#,
                r#"{"value": "bravo"}"#,
                r#"{"value": "charlie"}"#,
            ],
        )
        .await;

        assert_eq!(
            index.calculate_included_frags().await.unwrap(),
            RoaringBitmap::from_iter([0, 1])
        );
    }

    /// Regression test for https://github.com/lance-format/lance/issues/7485.
    ///
    /// A JSON-path btree index over float values returned wrong results because the
    /// btree trainer assumes its input arrives sorted by value (page min/max come from
    /// the first/last row of each page), but the value at `path` is extracted by this
    /// plugin *after* the scanner has already produced its rows, so a scan sorted on
    /// the raw JSON column does not sort the extracted value. This exercises the fix
    /// end to end: `JsonTrainingRequest::criteria()` must ask for unordered input (so
    /// the scanner does not waste time sorting on the wrong key), and `train_index` must
    /// sort the extracted value stream itself before training the target btree.
    ///
    /// Rows are fed in raw storage order (not sorted by value) to simulate what an
    /// unordered scan would produce.
    ///
    #[rstest]
    #[case::range_gt_zero(
        SargableQuery::Range(Bound::Excluded(ScalarValue::Float64(Some(0.0))), Bound::Unbounded),
        vec![0, 1]
    )]
    #[case::range_gte_page_min(
        SargableQuery::Range(Bound::Included(ScalarValue::Float64(Some(10.5))), Bound::Unbounded),
        vec![0, 1]
    )]
    #[case::equals_non_exact_float(
        SargableQuery::Equals(ScalarValue::Float64(Some(40.1))),
        vec![1]
    )]
    #[case::equals_exact_float(SargableQuery::Equals(ScalarValue::Float64(Some(10.5))), vec![0])]
    #[case::range_covers_all(
        SargableQuery::Range(Bound::Unbounded, Bound::Excluded(ScalarValue::Float64(Some(100.0)))),
        // Row 2's address is fragment 1, offset 0 (`RowAddress::new_from_parts(1, 0)`),
        // not the row index 2: `needs_row_addrs` test data is address-domain.
        vec![0, 1, 1u64 << 32]
    )]
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_float_btree_index_unsorted_input(
        #[case] query: SargableQuery,
        #[case] expected: Vec<u64>,
    ) {
        use crate::metrics::NoOpMetricsCollector;
        use lance_select::RowAddrTreeMap;

        // row0=10.5, row1=40.1, row2=-3.2: storage order does not match ascending value
        // order (-3.2, 10.5, 40.1), so a btree trained on this order without an explicit
        // value sort would record a corrupted page max of -3.2.
        let (store, _tmpdir) = local_json_index_store();
        let index = train_and_load_json_index(
            store,
            "btree",
            TrainingOrdering::None,
            "latitude",
            &[
                r#"{"latitude": 10.5}"#,
                r#"{"latitude": 40.1}"#,
                r#"{"latitude": -3.2}"#,
            ],
        )
        .await;

        let json_query = JsonQuery::new(Arc::new(query.clone()), "latitude".to_string());
        let result = index
            .search(&json_query, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            result,
            SearchResult::exact(RowAddrTreeMap::from_iter(expected.iter().copied())),
            "query {query:?}"
        );
    }

    /// Regression test for a null value at `path` surviving `sort_stream_by_value`.
    ///
    /// `sort_stream_by_value` sorts the extracted `(value, row_id)` stream with
    /// `nulls_first: true`. This checks that a null row's row_id stays paired with its
    /// (null) value through that sort -- if the sort ever reordered values without their
    /// row_ids, a null-valued row could be attributed to the wrong id -- and that the
    /// resulting btree still answers `IsNull` and non-null range/equality queries
    /// correctly with nulls mixed in and fed out of value order.
    ///
    /// Row 1's `path` contains an explicit JSON null. The extracted binary value still
    /// contains JSONB bytes, so conversion must use the accompanying type tag to turn it
    /// into an Arrow null before sorting and training the target index.
    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_btree_index_null_at_path() {
        use crate::metrics::NoOpMetricsCollector;
        use lance_select::RowAddrTreeMap;

        let (store, _tmpdir) = local_json_index_store();
        let index = train_and_load_json_index(
            store,
            "btree",
            TrainingOrdering::None,
            "v",
            &[
                r#"{"v": 40.1}"#, // row 0
                r#"{"v": null}"#, // row 1
                r#"{"v": -3.2}"#, // row 2
                r#"{"v": 10.5}"#, // row 3
            ],
        )
        .await;

        let search = |query: SargableQuery| {
            let index = index.clone();
            let json_query = JsonQuery::new(Arc::new(query), "v".to_string());
            async move {
                index
                    .search(&json_query, &NoOpMetricsCollector)
                    .await
                    .unwrap()
            }
        };

        // Range/equality queries carry row 1 in `nulls` (three-valued logic: `NULL > 0`
        // is unknown, not false -- see `NullableRowAddrSet`), which is pre-existing
        // btree/framework behavior. Asserting it exactly here is exactly the property
        // this test targets: row 1's row_id must stay paired with its null value
        // through `sort_stream_by_value`, not just be excluded from `selected`.
        assert_eq!(
            search(SargableQuery::IsNull()).await,
            SearchResult::exact(RowAddrTreeMap::from_iter([1u64])),
            "IsNull"
        );
        // Row 3's address is fragment 1, offset 1 (`RowAddress::new_from_parts(1, 1)`),
        // not the row index 3: `needs_row_addrs` test data is address-domain.
        let row3_addr = (1u64 << 32) | 1;
        assert_eq!(
            search(SargableQuery::Range(
                Bound::Excluded(ScalarValue::Float64(Some(0.0))),
                Bound::Unbounded,
            ))
            .await,
            SearchResult::exact(RowAddrTreeMap::from_iter([0u64, row3_addr]))
                .with_nulls(RowAddrTreeMap::from_iter([1u64])),
            "> 0"
        );
        assert_eq!(
            search(SargableQuery::Equals(ScalarValue::Float64(Some(40.1)))).await,
            SearchResult::exact(RowAddrTreeMap::from_iter([0u64]))
                .with_nulls(RowAddrTreeMap::from_iter([1u64])),
            "= 40.1"
        );
        let row2_addr = 1u64 << 32;
        assert_eq!(
            search(SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::Float64(Some(100.0))),
            ))
            .await,
            SearchResult::exact(RowAddrTreeMap::from_iter([0u64, row2_addr, row3_addr]))
                .with_nulls(RowAddrTreeMap::from_iter([1u64])),
            "< 100 (null is neither < 100 nor >= 100)"
        );
    }

    /// Regression coverage for the non-`Values`-ordering branch in `train_index`: a
    /// JSON-path index over a target that does not need value-ordered input (ngram
    /// requires `TrainingOrdering::None`) must skip `sort_stream_by_value` entirely and
    /// still produce correct results from rows fed out of value order.
    #[tokio::test]
    async fn test_json_ngram_index_skips_value_sort() {
        use crate::metrics::NoOpMetricsCollector;
        use lance_select::RowAddrTreeMap;

        let (store, _tmpdir) = local_json_index_store();
        let index = train_and_load_json_index(
            store,
            "ngram",
            TrainingOrdering::None,
            "tag",
            &[
                r#"{"tag": "unique-charlie"}"#,
                r#"{"tag": "unique-alpha"}"#,
                r#"{"tag": "unique-bravo"}"#,
            ],
        )
        .await;

        let json_query = JsonQuery::new(
            Arc::new(TextQuery::StringContains("unique-bravo".to_string())),
            "tag".to_string(),
        );
        let result = index
            .search(&json_query, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            result,
            SearchResult::at_most(RowAddrTreeMap::from_iter([2u64])),
        );
    }
}
