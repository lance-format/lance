// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::{
    ops::Bound,
    sync::{Arc, Mutex},
};

use arrow_array::{Array, LargeBinaryArray, RecordBatch, StructArray, UInt8Array};
use arrow_schema::{DataType, Field, Schema, SortOptions};
use async_trait::async_trait;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::{
    execution::SendableRecordBatchStream,
    physical_plan::{ExecutionPlan, projection::ProjectionExec, sorts::sort::SortExec},
};
use datafusion_common::{DataFusionError, ScalarValue, config::ConfigOptions};
use datafusion_expr::{Expr, Operator, ScalarUDF};
use datafusion_physical_expr::{
    PhysicalExpr, PhysicalSortExpr, ScalarFunctionExpr,
    expressions::{Column, Literal},
};
use futures::StreamExt;
use lance_core::deepsize::DeepSizeOf;
use lance_datafusion::exec::{
    LanceExecutionOptions, OneShotExec, execute_plan, get_session_context,
};
use lance_datafusion::udf::json::{
    JSON_EXTRACT_BOOLEAN_UDF_NAME, JSON_EXTRACT_FLOAT64_UDF_NAME, JSON_EXTRACT_INT64_UDF_NAME,
    JSON_EXTRACT_LARGE_BINARY_UDF_NAME, JSON_EXTRACT_UTF8_UDF_NAME, JsonbType,
    json_extract_typed_udf, json_extract_with_type_udf, normalize_json_path,
};
use prost::Message;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use lance_core::{Error, Result, cache::LanceCache, error::LanceOptionExt};

use crate::{
    Index, IndexType,
    metrics::MetricsCollector,
    registry::IndexPluginRegistry,
    scalar::{
        AnyQuery, CreatedIndex, IndexStore, RowIdRemapper, ScalarIndex, SearchOptions,
        SearchResult, UpdateCriteria,
        expression::{IndexedExpression, ScalarIndexExpr, ScalarIndexSearch, ScalarQueryParser},
        registry::{
            BasicTrainer, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
            VALUE_COLUMN_NAME,
        },
    },
};

const JSON_INDEX_VERSION: u32 = 1;
const JSON_INDEX_CONVERSION: &str = "jsonpath_typed_v1";
/// Conversion identity for version-0 indices whose native key type has been
/// recovered and persisted without rewriting their keys.
pub const JSON_INDEX_LEGACY_NATIVE_CONVERSION: &str = "legacy_native_v0";

fn index_version_for_target_type(target_type: &DataType) -> u32 {
    // Version-1 changes Utf8 keys from decoded strings to serialized JSON text.
    // Native typed keys retain their version-0 encoding so released readers can
    // continue to use current-created numeric, boolean, and binary indices.
    if target_type == &DataType::Utf8 {
        JSON_INDEX_VERSION
    } else {
        0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonIndexConversion {
    LegacyV0,
    LegacyNativeV0,
    TypedV1,
}

/// A JSON index that indexes a field in a JSON column
///
/// The underlying index can be any other type of scalar index
#[derive(Debug)]
pub struct JsonIndex {
    target_index: Arc<dyn ScalarIndex>,
    path: String,
    target_type: Option<DataType>,
    conversion: JsonIndexConversion,
}

impl JsonIndex {
    pub fn new(target_index: Arc<dyn ScalarIndex>, path: String, target_type: DataType) -> Self {
        Self {
            target_index,
            path,
            target_type: Some(target_type),
            conversion: JsonIndexConversion::TypedV1,
        }
    }

    fn new_legacy(target_index: Arc<dyn ScalarIndex>, path: String) -> Self {
        let target_type = target_index.training_data_type();
        Self {
            target_index,
            path,
            target_type,
            conversion: JsonIndexConversion::LegacyV0,
        }
    }

    fn new_upgraded_legacy(
        target_index: Arc<dyn ScalarIndex>,
        path: String,
        target_type: DataType,
    ) -> Self {
        Self {
            target_index,
            path,
            target_type: Some(target_type),
            conversion: JsonIndexConversion::LegacyNativeV0,
        }
    }

    fn is_legacy(&self) -> bool {
        matches!(
            self.conversion,
            JsonIndexConversion::LegacyV0 | JsonIndexConversion::LegacyNativeV0
        )
    }

    fn wrap_target_created_index(&self, target_created: CreatedIndex) -> Result<CreatedIndex> {
        let (target_data_type, conversion, index_version) = match self.conversion {
            JsonIndexConversion::LegacyV0 => (None, None, 0),
            JsonIndexConversion::LegacyNativeV0 => {
                let target_type = self.target_type.as_ref().ok_or_else(|| {
                    Error::internal("Upgraded version-0 JSON index is missing its target type")
                })?;
                (
                    Some(json_target_type_name(target_type)?.to_string()),
                    Some(JSON_INDEX_LEGACY_NATIVE_CONVERSION.to_string()),
                    0,
                )
            }
            JsonIndexConversion::TypedV1 => {
                let target_type = self.target_type.as_ref().ok_or_else(|| {
                    Error::internal("Version-1 JSON index is missing its target type")
                })?;
                (
                    Some(json_target_type_name(target_type)?.to_string()),
                    Some(JSON_INDEX_CONVERSION.to_string()),
                    index_version_for_target_type(target_type),
                )
            }
        };
        let json_details = crate::pb::JsonIndexDetails {
            path: self.path.clone(),
            target_details: Some(target_created.index_details),
            target_data_type,
            conversion,
        };
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&json_details)?,
            index_version,
            files: target_created.files,
        })
    }
}

impl DeepSizeOf for JsonIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.target_index.deep_size_of_children(context) + self.path.deep_size_of_children(context)
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

    fn can_remap(&self) -> bool {
        self.target_index.can_remap()
    }

    async fn remap(
        &self,
        mapping: &RowAddrRemap,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let target_created = self.target_index.remap(mapping, dest_store).await?;
        self.wrap_target_created_index(target_created)
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        old_data_filter: Option<super::OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        if self.is_legacy() {
            return Err(Error::not_supported(format!(
                "Legacy version-0 JSON index at path '{}' must be fully rebuilt before adding new data",
                self.path
            )));
        }
        let target_type = self
            .target_type
            .clone()
            .ok_or_else(|| Error::internal("Version-1 JSON index is missing its target type"))?;
        let target_criteria = self.target_index.update_criteria().data_criteria;
        let new_data =
            JsonIndexPlugin::extract_json_typed(new_data, self.path.clone(), target_type)?;
        let new_data = if target_criteria.ordering == TrainingOrdering::Values {
            JsonIndexPlugin::sort_stream_by_value(new_data).await?
        } else {
            new_data
        };
        let target_created = self
            .target_index
            .update(new_data, dest_store, old_data_filter)
            .await?;
        self.wrap_target_created_index(target_created)
    }

    fn update_criteria(&self) -> UpdateCriteria {
        let target_criteria = self.target_index.update_criteria();
        UpdateCriteria {
            requires_old_data: self.is_legacy() || target_criteria.requires_old_data,
            data_criteria: json_scan_criteria(&target_criteria.data_criteria),
        }
    }

    fn requires_full_rebuild(&self) -> bool {
        self.is_legacy()
    }

    fn derive_index_params(&self) -> Result<super::ScalarIndexParams> {
        let target_params = self.target_index.derive_index_params()?;
        let target_data_type = self
            .target_type
            .as_ref()
            .map(JsonIndexTargetType::try_from)
            .transpose()?;
        let params = JsonIndexParameters {
            target_index_type: target_params.index_type,
            target_index_parameters: target_params.params,
            target_data_type,
            path: self.path.clone(),
        };
        Ok(super::ScalarIndexParams::new("json".to_string()).with_params(&params))
    }

    fn training_data_type(&self) -> Option<DataType> {
        self.target_type.clone()
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum JsonIndexTargetType {
    Boolean,
    Int64,
    Float64,
    Utf8,
    LargeBinary,
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

fn json_target_type_name(data_type: &DataType) -> Result<&'static str> {
    match JsonIndexTargetType::try_from(data_type)? {
        JsonIndexTargetType::Boolean => Ok("Boolean"),
        JsonIndexTargetType::Int64 => Ok("Int64"),
        JsonIndexTargetType::Float64 => Ok("Float64"),
        JsonIndexTargetType::Utf8 => Ok("Utf8"),
        JsonIndexTargetType::LargeBinary => Ok("LargeBinary"),
    }
}

fn parse_json_target_type(data_type: &str) -> Option<DataType> {
    match data_type {
        "Boolean" => Some(DataType::Boolean),
        "Int64" => Some(DataType::Int64),
        "Float64" => Some(DataType::Float64),
        "Utf8" => Some(DataType::Utf8),
        "LargeBinary" => Some(DataType::LargeBinary),
        _ => None,
    }
}

// TODO: Do we really need to wrap the query or could we just return the target query directly?
//
// I think the only thing we really gain is a different format impl (e.g. it shows up as a json query
// in the explain plan) but I don't know if that helps the user much.
#[derive(Debug, Clone)]
pub struct JsonQuery {
    target_query: Arc<dyn AnyQuery>,
    path: String,
}

impl JsonQuery {
    pub fn new(target_query: Arc<dyn AnyQuery>, path: String) -> Self {
        Self { target_query, path }
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
}

#[derive(Debug)]
pub struct JsonQueryParser {
    path: String,
    target_type: DataType,
    conversion: JsonIndexConversion,
    target_parser: Box<dyn ScalarQueryParser>,
}

impl JsonQueryParser {
    pub fn new(
        path: String,
        target_type: DataType,
        target_parser: Box<dyn ScalarQueryParser>,
    ) -> Self {
        Self {
            path,
            target_type,
            conversion: JsonIndexConversion::TypedV1,
            target_parser,
        }
    }

    fn new_legacy(
        path: String,
        target_type: DataType,
        target_parser: Box<dyn ScalarQueryParser>,
    ) -> Self {
        Self {
            path,
            target_type,
            conversion: JsonIndexConversion::LegacyV0,
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

    fn is_valid_reference(&self, func: &Expr, _data_type: &DataType) -> Option<DataType> {
        let Expr::ScalarFunction(udf) = func else {
            return None;
        };
        if udf.args.len() != 2 {
            return None;
        }
        let Expr::Literal(ScalarValue::Utf8(Some(path)), _) = &udf.args[1] else {
            return None;
        };

        match self.conversion {
            JsonIndexConversion::LegacyV0 | JsonIndexConversion::LegacyNativeV0 => {
                // Version-0 indices stored decoded values and preserved their
                // unnormalized parameter path. Native typed JSONPath extraction
                // and direct typed getters are compatible only when their exact
                // output type matches the physical B-tree key type. Decoded Utf8
                // keys are compatible only with json_get_string, never serialized
                // typed Utf8 extraction.
                let reference_type = match udf.name() {
                    JSON_EXTRACT_INT64_UDF_NAME | "json_get_int" => DataType::Int64,
                    JSON_EXTRACT_FLOAT64_UDF_NAME | "json_get_float" => DataType::Float64,
                    JSON_EXTRACT_BOOLEAN_UDF_NAME | "json_get_bool" => DataType::Boolean,
                    "json_get_string" => DataType::Utf8,
                    _ => return None,
                };
                if self.target_type != reference_type {
                    return None;
                }

                let reference_path = if udf.name().starts_with("json_get") {
                    direct_json_key_path(path)?
                } else {
                    normalize_json_path(path).ok()?
                };
                let indexed_path = normalize_json_path(&self.path).ok()?;
                (reference_path == indexed_path).then_some(reference_type)
            }
            JsonIndexConversion::TypedV1 => {
                let reference_type = match udf.name() {
                    JSON_EXTRACT_INT64_UDF_NAME | "json_get_int" => DataType::Int64,
                    JSON_EXTRACT_FLOAT64_UDF_NAME | "json_get_float" => DataType::Float64,
                    JSON_EXTRACT_BOOLEAN_UDF_NAME | "json_get_bool" => DataType::Boolean,
                    JSON_EXTRACT_UTF8_UDF_NAME => DataType::Utf8,
                    JSON_EXTRACT_LARGE_BINARY_UDF_NAME => DataType::LargeBinary,
                    // json_get_string returns decoded string contents while typed Utf8
                    // JSONPath extraction preserves serialized JSON text.
                    _ => return None,
                };
                if self.target_type != reference_type {
                    return None;
                }

                let normalized_path = if udf.name().starts_with("json_get") {
                    direct_json_key_path(path)?
                } else {
                    normalize_json_path(path).ok()?
                };
                let indexed_path = normalize_json_path(&self.path).ok()?;
                (normalized_path == indexed_path).then_some(reference_type)
            }
        }
    }
}

fn direct_json_key_path(key: &str) -> Option<String> {
    let mut chars = key.chars();
    let first = chars.next()?;
    if !(first == '_' || first.is_ascii_alphabetic())
        || !chars.all(|character| character == '_' || character.is_ascii_alphanumeric())
    {
        return None;
    }
    Some(format!("$.{key}"))
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

    fn query_parser(
        &self,
        index_name: String,
        index_details: &prost_types::Any,
        legacy_target_type: Option<&DataType>,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        let registry = self.registry().ok()?;
        let json_details =
            crate::pb::JsonIndexDetails::decode(index_details.value.as_slice()).ok()?;
        let target_details = json_details.target_details.as_ref()?;
        let target_plugin = registry.get_plugin_by_details(target_details).ok()?;
        let target_parser = target_plugin.new_query_parser(index_name, target_details)?;
        let parser = match (
            json_details.target_data_type.as_deref(),
            json_details.conversion.as_deref(),
        ) {
            (None, None) => JsonQueryParser::new_legacy(
                json_details.path.clone(),
                legacy_target_type?.clone(),
                target_parser,
            ),
            (Some(data_type), Some(conversion))
                if conversion == JSON_INDEX_LEGACY_NATIVE_CONVERSION =>
            {
                JsonQueryParser::new_legacy(
                    json_details.path.clone(),
                    parse_json_target_type(data_type)?,
                    target_parser,
                )
            }
            (Some(data_type), Some(conversion)) if conversion == JSON_INDEX_CONVERSION => {
                JsonQueryParser::new(
                    json_details.path.clone(),
                    parse_json_target_type(data_type)?,
                    target_parser,
                )
            }
            _ => return None,
        };
        Some(Box::new(parser) as Box<dyn ScalarQueryParser>)
    }

    /// Evaluate a typed JSONPath while preserving all row-location columns.
    ///
    /// The same internal UDF is embedded in scan predicates by the expression
    /// planner, keeping indexed and unindexed conversion semantics identical.
    fn extract_json_typed(
        data: SendableRecordBatchStream,
        path: String,
        target_type: DataType,
    ) -> Result<SendableRecordBatchStream> {
        let path = normalize_json_path(&path).map_err(Error::from)?;
        let input = Arc::new(OneShotExec::new(data));
        let input_schema = input.schema();
        let value_column_idx = input_schema
            .column_with_name(VALUE_COLUMN_NAME)
            .expect_ok()?
            .0;

        let typed_udf = json_extract_typed_udf(&target_type).ok_or_else(|| {
            Error::not_supported(format!(
                "JSON index target data type {target_type:?} is not supported"
            ))
        })?;
        let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
            Vec::with_capacity(input_schema.fields().len());
        exprs.push((
            Arc::new(ScalarFunctionExpr::try_new(
                Arc::new(typed_udf),
                vec![
                    Arc::new(Column::new(VALUE_COLUMN_NAME, value_column_idx)),
                    Arc::new(Literal::new(ScalarValue::Utf8(Some(path)))),
                ],
                &input_schema,
                Arc::new(ConfigOptions::default()),
            )?) as Arc<dyn PhysicalExpr>,
            VALUE_COLUMN_NAME.to_string(),
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
        let ctx = get_session_context(&LanceExecutionOptions::default());
        project.execute(0, ctx.task_ctx()).map_err(Into::into)
    }

    /// Extract a JSONPath value and type tag while preserving row-location columns.
    fn extract_json_with_type(
        data: SendableRecordBatchStream,
        path: String,
    ) -> Result<SendableRecordBatchStream> {
        let path = normalize_json_path(&path).map_err(Error::from)?;
        let input = Arc::new(OneShotExec::new(data));
        let input_schema = input.schema();
        let value_column_idx = input_schema
            .column_with_name(VALUE_COLUMN_NAME)
            .expect_ok()?
            .0;
        let mut exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
            Vec::with_capacity(input_schema.fields().len());
        exprs.push((
            Arc::new(ScalarFunctionExpr::try_new(
                Arc::new(json_extract_with_type_udf()),
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
        let ctx = get_session_context(&LanceExecutionOptions::default());
        project.execute(0, ctx.task_ctx()).map_err(Into::into)
    }

    fn infer_type_from_batch(batch: &RecordBatch, path: &str) -> Result<Option<DataType>> {
        let json_result = batch
            .column_by_name("json_result")
            .ok_or_else(|| Error::invalid_input_source("Missing json_result column".into()))?
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| Error::invalid_input_source("json_result is not a struct".into()))?;
        let type_array = json_result
            .column_by_name("type_tag")
            .ok_or_else(|| Error::invalid_input_source("Missing type_tag column".into()))?
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| Error::invalid_input_source("type_tag is not UInt8".into()))?;

        for type_tag in type_array.iter().flatten() {
            let jsonb_type = JsonbType::from_u8(type_tag).ok_or_else(|| {
                Error::invalid_input_source(
                    format!("JSONPath '{path}' produced invalid type tag {type_tag}").into(),
                )
            })?;
            let data_type = match jsonb_type {
                JsonbType::Null => continue,
                JsonbType::Boolean => DataType::Boolean,
                JsonbType::Int64 => DataType::Int64,
                JsonbType::Float64 => DataType::Float64,
                JsonbType::String => DataType::Utf8,
                JsonbType::Array | JsonbType::Object => DataType::LargeBinary,
            };
            return Ok(Some(data_type));
        }
        Ok(None)
    }

    /// Infer the target type from compact extracted values.
    ///
    /// Only extracted values and row locations are buffered until the first
    /// non-null value is found. Large unrelated JSON fields are released as each
    /// input batch is evaluated, and the JSONPath is evaluated only once.
    async fn infer_json_type(
        data: SendableRecordBatchStream,
        path: String,
    ) -> Result<(SendableRecordBatchStream, DataType)> {
        let path = normalize_json_path(&path).map_err(Error::from)?;
        let mut extracted = Self::extract_json_with_type(data, path.clone())?;
        let schema = extracted.schema();
        let mut buffered_batches = Vec::new();
        let mut inferred_type = None;

        while let Some(batch_result) = extracted.next().await {
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
            buffered.chain(extracted),
        )) as SendableRecordBatchStream;

        Ok((recreated_stream, inferred_type))
    }

    fn validate_json_types(
        values: &LargeBinaryArray,
        type_tags: &UInt8Array,
        target_type: &DataType,
        path: &str,
    ) -> Result<()> {
        for index in 0..values.len() {
            if values.is_null(index) {
                continue;
            }
            if type_tags.is_null(index) {
                return Err(Error::invalid_input_source(
                    format!(
                        "JSONPath '{path}' has a value at batch row {index} without a type tag"
                    )
                    .into(),
                ));
            }
            let type_tag = type_tags.value(index);
            let actual_type = JsonbType::from_u8(type_tag).ok_or_else(|| {
                Error::invalid_input_source(
                    format!(
                        "JSONPath '{path}' produced invalid type tag {type_tag} at batch row {index}"
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
                        "JSONPath '{path}' expected {target_type:?}, but batch row {index} has JSON type {actual_type:?}"
                    )
                    .into(),
                ));
            }
        }
        Ok(())
    }

    fn convert_extracted_batch(
        batch: RecordBatch,
        output_schema: Arc<Schema>,
        passthrough_indices: &[usize],
        target_type: &DataType,
        path: &str,
    ) -> Result<RecordBatch> {
        let json_result = batch
            .column_by_name("json_result")
            .ok_or_else(|| Error::invalid_input_source("Missing json_result column".into()))?
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| Error::invalid_input_source("json_result is not a struct".into()))?;
        let values = json_result
            .column_by_name("value")
            .ok_or_else(|| Error::invalid_input_source("Missing value column".into()))?
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .ok_or_else(|| Error::invalid_input_source("value is not LargeBinary".into()))?;
        let type_tags = json_result
            .column_by_name("type_tag")
            .ok_or_else(|| Error::invalid_input_source("Missing type_tag column".into()))?
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| Error::invalid_input_source("type_tag is not UInt8".into()))?;

        Self::validate_json_types(values, type_tags, target_type, path)?;
        let is_utf8 = target_type == &DataType::Utf8;
        let is_null = |index| {
            values.is_null(index)
                || (!is_utf8
                    && !type_tags.is_null(index)
                    && type_tags.value(index) == JsonbType::Null.as_u8())
        };
        let converted: Arc<dyn Array> = match target_type {
            DataType::Boolean => {
                let mut builder = arrow_array::builder::BooleanBuilder::with_capacity(values.len());
                for index in 0..values.len() {
                    if is_null(index) {
                        builder.append_null();
                    } else {
                        let raw = jsonb::RawJsonb::new(values.value(index));
                        let value = jsonb::from_raw_jsonb::<bool>(&raw).map_err(|error| {
                            Error::invalid_input_source(
                                format!(
                                    "Failed to convert JSONPath '{path}' at batch row {index} to Boolean: {error}"
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
                let mut builder = arrow_array::builder::Int64Builder::with_capacity(values.len());
                for index in 0..values.len() {
                    if is_null(index) {
                        builder.append_null();
                    } else {
                        let raw = jsonb::RawJsonb::new(values.value(index));
                        let value = jsonb::from_raw_jsonb::<i64>(&raw).map_err(|error| {
                            Error::invalid_input_source(
                                format!(
                                    "Failed to convert JSONPath '{path}' at batch row {index} to Int64: {error}"
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
                let mut builder = arrow_array::builder::Float64Builder::with_capacity(values.len());
                for index in 0..values.len() {
                    if is_null(index) {
                        builder.append_null();
                    } else {
                        let raw = jsonb::RawJsonb::new(values.value(index));
                        let value = jsonb::from_raw_jsonb::<f64>(&raw).map_err(|error| {
                            Error::invalid_input_source(
                                format!(
                                    "Failed to convert JSONPath '{path}' at batch row {index} to Float64: {error}"
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
                    arrow_array::builder::StringBuilder::with_capacity(values.len(), 1024);
                for index in 0..values.len() {
                    if is_null(index) {
                        builder.append_null();
                    } else {
                        builder.append_value(jsonb::RawJsonb::new(values.value(index)).to_string());
                    }
                }
                Arc::new(builder.finish())
            }
            DataType::LargeBinary => Arc::new(LargeBinaryArray::from_iter(
                (0..values.len()).map(|index| (!is_null(index)).then(|| values.value(index))),
            )),
            _ => {
                return Err(Error::invalid_input_source(
                    format!("Unsupported JSON index target type: {target_type:?}").into(),
                ));
            }
        };

        let mut columns = Vec::with_capacity(output_schema.fields().len());
        columns.push(converted);
        columns.extend(
            passthrough_indices
                .iter()
                .map(|column_idx| batch.column(*column_idx).clone()),
        );
        Ok(RecordBatch::try_new(output_schema, columns)?)
    }

    fn convert_extracted_stream(
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
            Self::convert_extracted_batch(
                batch_result?,
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
        if !matches!(field.data_type(), DataType::Binary | DataType::LargeBinary) {
            return Err(Error::invalid_input_source(
                "A JSON index can only be created on a Binary or LargeBinary field.".into(),
            ));
        }

        let params = serde_json::from_str::<JsonIndexParameters>(params)?;
        // Validate the path now, but preserve its spelling in persisted metadata.
        // Released readers compare direct-key paths literally, while current
        // extraction and routing normalize them at their internal boundaries.
        normalize_json_path(&params.path).map_err(Error::from)?;
        // Initial builds infer the type from the data. Derived rebuild parameters
        // carry the learned type so every new segment uses the same target schema.
        let target_type = params
            .target_data_type
            .map(DataType::from)
            .unwrap_or(DataType::Utf8);
        let registry = self.registry()?;
        let target_plugin = registry.get_plugin_by_name(&params.target_index_type)?;
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

        let (converted_stream, target_type) =
            if let Some(target_data_type) = request.parameters.target_data_type {
                let target_type = DataType::from(target_data_type);
                let converted = Self::extract_json_typed(data, path.clone(), target_type.clone())?;
                (converted, target_type)
            } else {
                let (extracted, target_type) = Self::infer_json_type(data, path.clone()).await?;
                let converted =
                    Self::convert_extracted_stream(extracted, target_type.clone(), path.clone())?;
                (converted, target_type)
            };

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
            &Field::new("", target_type.clone(), true),
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

        let index_details = crate::pb::JsonIndexDetails {
            path,
            target_details: Some(target_index.index_details),
            target_data_type: Some(json_target_type_name(&target_type)?.to_string()),
            conversion: Some(JSON_INDEX_CONVERSION.to_string()),
        };
        Ok(CreatedIndex {
            index_details: prost_types::Any::from_msg(&index_details)?,
            index_version: index_version_for_target_type(&target_type),
            files: target_index.files,
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for JsonIndexPlugin {
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
        self.query_parser(index_name, index_details, None)
    }

    fn new_query_parser_with_training_data_type(
        &self,
        index_name: String,
        index_details: &prost_types::Any,
        training_data_type: Option<&DataType>,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        self.query_parser(index_name, index_details, training_data_type)
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let registry = self.registry().unwrap();
        let json_details = crate::pb::JsonIndexDetails::decode(index_details.value.as_slice())?;
        let target_details = json_details.target_details.as_ref().expect_ok()?;
        let target_plugin = registry.get_plugin_by_details(target_details).unwrap();
        let target_index = target_plugin
            .load_index(index_store, target_details, frag_reuse_index, cache)
            .await?;
        let (target_type, conversion) = match (
            json_details.target_data_type.as_deref(),
            json_details.conversion.as_deref(),
        ) {
            (None, None) => (None, JsonIndexConversion::LegacyV0),
            (Some(data_type), Some(conversion))
                if conversion == JSON_INDEX_LEGACY_NATIVE_CONVERSION =>
            {
                (
                    Some(parse_json_target_type(data_type).ok_or_else(|| {
                        Error::invalid_input(format!(
                            "JSON index path '{}' has unsupported target data type '{data_type}'",
                            json_details.path
                        ))
                    })?),
                    JsonIndexConversion::LegacyNativeV0,
                )
            }
            (Some(data_type), Some(conversion)) if conversion == JSON_INDEX_CONVERSION => (
                Some(parse_json_target_type(data_type).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "JSON index path '{}' has unsupported target data type '{data_type}'",
                        json_details.path
                    ))
                })?),
                JsonIndexConversion::TypedV1,
            ),
            (Some(_), Some(conversion)) => {
                return Err(Error::not_supported(format!(
                    "JSON index path '{}' uses unsupported conversion '{conversion}'",
                    json_details.path
                )));
            }
            _ => {
                return Err(Error::invalid_input(format!(
                    "JSON index path '{}' has incomplete conversion metadata",
                    json_details.path
                )));
            }
        };
        let index = match conversion {
            JsonIndexConversion::LegacyV0 => JsonIndex::new_legacy(target_index, json_details.path),
            JsonIndexConversion::LegacyNativeV0 => JsonIndex::new_upgraded_legacy(
                target_index,
                json_details.path,
                target_type.ok_or_else(|| {
                    Error::internal("Upgraded version-0 JSON index is missing its target type")
                })?,
            ),
            JsonIndexConversion::TypedV1 => JsonIndex::new(
                target_index,
                json_details.path,
                target_type.ok_or_else(|| {
                    Error::internal("Version-1 JSON index is missing its target type")
                })?,
            ),
        };
        Ok(Arc::new(index))
    }

    fn cache_namespace(&self, index_details: &prost_types::Any) -> Result<Option<String>> {
        crate::pb::JsonIndexDetails::decode(index_details.value.as_slice())?;

        // The wrapper's behavior is completely described by JsonIndexDetails.
        // Use its exact protobuf encoding so a metadata-only binding upgrade
        // cannot reuse (or race with) a wrapper loaded under the prior binding.
        const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut namespace = String::with_capacity(5 + index_details.value.len() * 2);
        namespace.push_str("json/");
        for byte in &index_details.value {
            namespace.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
            namespace.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
        }
        Ok(Some(namespace))
    }

    fn details_as_json(&self, details: &prost_types::Any) -> Result<serde_json::Value> {
        let registry = self.registry().unwrap();
        let json_details = crate::pb::JsonIndexDetails::decode(details.value.as_slice())?;
        let target_details = json_details.target_details.as_ref().expect_ok()?;
        let target_plugin = registry.get_plugin_by_details(target_details).unwrap();
        let target_details_json = target_plugin.details_as_json(target_details)?;
        Ok(serde_json::json!({
            "path": json_details.path,
            "target_data_type": json_details.target_data_type,
            "conversion": json_details.conversion,
            "target_details": target_details_json,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::NoOpMetricsCollector;
    use crate::progress::noop_progress;
    use crate::scalar::{SargableQuery, TextQuery};
    use arrow_array::{ArrayRef, RecordBatch, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_common::DataFusionError;
    use datafusion_expr::{col, expr::ScalarFunction, lit};
    use futures::stream;
    use lance_core::{ROW_ADDR, ROW_ID, utils::address::RowAddress};
    use lance_datafusion::udf::json::{
        json_extract_udf, json_get_bool_udf, json_get_float_udf, json_get_int_udf,
        json_get_string_udf,
    };
    use lance_select::RowAddrTreeMap;
    use rstest::rstest;
    use std::ops::Bound;
    use std::sync::Arc;

    // Note: The old test_detect_json_value_type test has been removed as we now use
    // JSONB's inherent type information instead of string-based type detection

    #[tokio::test]
    async fn test_json_type_inference() {
        use arrow_array::{LargeBinaryArray, UInt64Array};
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
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
            let owned_jsonb: jsonb::OwnedJsonb = json_str.parse().unwrap();
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
            JsonIndexPlugin::infer_json_type(stream, "$.age".to_string())
                .await
                .unwrap();

        assert_eq!(inferred_type, DataType::Int64);

        // Create new test stream for boolean field
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(LargeBinaryArray::from(vec![
                    json_data[0]
                        .parse::<jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                    json_data[1]
                        .parse::<jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                    json_data[2]
                        .parse::<jsonb::OwnedJsonb>()
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
        let (_, inferred_type) = JsonIndexPlugin::infer_json_type(stream2, "$.active".to_string())
            .await
            .unwrap();

        assert_eq!(inferred_type, DataType::Boolean);

        // Create test stream for string field
        let batch3 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(LargeBinaryArray::from(vec![
                    json_data[0]
                        .parse::<jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                    json_data[1]
                        .parse::<jsonb::OwnedJsonb>()
                        .ok()
                        .map(|j| j.to_vec())
                        .as_deref(),
                    json_data[2]
                        .parse::<jsonb::OwnedJsonb>()
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
        let (_, inferred_type) = JsonIndexPlugin::infer_json_type(stream3, "$.name".to_string())
            .await
            .unwrap();

        assert_eq!(inferred_type, DataType::Utf8);
    }

    #[tokio::test]
    async fn test_sparse_type_inference_releases_raw_json_batches() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::LargeBinary, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        let mut weak_values = Vec::new();
        let mut batches = Vec::new();
        for row_id in 0..3 {
            let document = format!(r#"{{"other":"{}"}}"#, "x".repeat(64 * 1024));
            let encoded = document.parse::<jsonb::OwnedJsonb>().unwrap().to_vec();
            let values: ArrayRef =
                Arc::new(LargeBinaryArray::from_iter_values([encoded.as_slice()]));
            weak_values.push(Arc::downgrade(&values));
            batches.push(
                RecordBatch::try_new(
                    schema.clone(),
                    vec![values, Arc::new(UInt64Array::from(vec![row_id]))],
                )
                .unwrap(),
            );
        }
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(batches.into_iter().map(Ok)),
        ));

        let (_extracted, inferred_type) =
            JsonIndexPlugin::infer_json_type(data, "$.missing".to_string())
                .await
                .unwrap();

        assert_eq!(inferred_type, DataType::Utf8);
        assert!(
            weak_values.iter().all(|values| values.upgrade().is_none()),
            "inference should retain only compact extracted values"
        );
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
            .map(|s| s.parse::<jsonb::OwnedJsonb>().unwrap().to_vec())
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

        let created = trainer
            .train_index(data, store.as_ref(), request, None, noop_progress())
            .await
            .unwrap();

        plugin
            .load_index(store, &created.index_details, None, &LanceCache::no_cache())
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

    async fn load_legacy_utf8_json_index(
        store: Arc<dyn IndexStore>,
        target_index_type: &str,
    ) -> Arc<dyn ScalarIndex> {
        let registry = IndexPluginRegistry::with_default_plugins();
        let target_plugin = registry.get_plugin_by_name(target_index_type).unwrap();
        let target_trainer = target_plugin.basic_trainer().unwrap();
        let target_request = target_trainer
            .new_training_request("{}", &Field::new("", DataType::Utf8, true))
            .unwrap();
        let mut fields = vec![Field::new(VALUE_COLUMN_NAME, DataType::Utf8, true)];
        let mut columns = vec![Arc::new(StringArray::from(vec!["foo"])) as ArrayRef];
        if target_request.criteria().needs_row_ids {
            fields.push(Field::new(ROW_ID, DataType::UInt64, false));
            columns.push(Arc::new(UInt64Array::from(vec![0])) as ArrayRef);
        }
        if target_request.criteria().needs_row_addrs {
            fields.push(Field::new(ROW_ADDR, DataType::UInt64, false));
            columns.push(Arc::new(UInt64Array::from(vec![0])) as ArrayRef);
        }
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let data = Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter([Ok(batch)]),
        ));
        let target_created = target_trainer
            .train_index(data, store.as_ref(), target_request, None, noop_progress())
            .await
            .unwrap();
        let legacy_details = crate::pb::JsonIndexDetails {
            path: "$.v".to_string(),
            target_details: Some(target_created.index_details),
            target_data_type: None,
            conversion: None,
        };
        let legacy_details = prost_types::Any::from_msg(&legacy_details).unwrap();
        let json_plugin = registry.get_plugin_by_name("json").unwrap();
        json_plugin
            .load_index(store, &legacy_details, None, &LanceCache::no_cache())
            .await
            .unwrap()
    }

    #[rstest]
    #[case::zonemap("zonemap")]
    #[case::fm("fm")]
    #[tokio::test]
    async fn test_legacy_json_index_loads_without_known_target_type(
        #[case] target_index_type: &str,
    ) {
        let (store, _dir) = local_json_index_store();
        let index = load_legacy_utf8_json_index(store, target_index_type).await;

        assert_eq!(index.training_data_type(), None);
        index.statistics().unwrap();
    }

    #[test]
    fn test_legacy_json_query_parser_requires_matching_physical_type() {
        let registry = IndexPluginRegistry::with_default_plugins();
        let legacy_details = crate::pb::JsonIndexDetails {
            path: "val".to_string(),
            target_details: Some(
                prost_types::Any::from_msg(&crate::pbold::BTreeIndexDetails::default()).unwrap(),
            ),
            target_data_type: None,
            conversion: None,
        };
        let legacy_details = prost_types::Any::from_msg(&legacy_details).unwrap();
        let plugin = registry.get_plugin_by_name("json").unwrap();
        assert!(
            plugin
                .new_query_parser("json_idx".to_string(), &legacy_details)
                .is_none(),
            "version-0 metadata alone must not claim a physical key type"
        );
        let parser = plugin
            .new_query_parser_with_training_data_type(
                "json_idx".to_string(),
                &legacy_details,
                Some(&DataType::Int64),
            )
            .expect("a loaded version-0 Int64 B-tree should retain a query parser");

        let getter = Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(json_get_int_udf()),
            vec![col("json"), lit("val")],
        ));
        assert_eq!(
            parser.is_valid_reference(&getter, &DataType::LargeBinary),
            Some(DataType::Int64)
        );

        let typed_int = Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(json_extract_typed_udf(&DataType::Int64).unwrap()),
            vec![col("json"), lit("$.val")],
        ));
        assert_eq!(
            parser.is_valid_reference(&typed_int, &DataType::LargeBinary),
            Some(DataType::Int64)
        );

        let float_getter = Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(json_get_float_udf()),
            vec![col("json"), lit("val")],
        ));
        assert_eq!(
            parser.is_valid_reference(&float_getter, &DataType::LargeBinary),
            None
        );

        for (target_type, getter) in [
            (DataType::Float64, json_get_float_udf()),
            (DataType::Boolean, json_get_bool_udf()),
        ] {
            let typed_parser = plugin
                .new_query_parser_with_training_data_type(
                    "json_idx".to_string(),
                    &legacy_details,
                    Some(&target_type),
                )
                .unwrap();
            let getter = Expr::ScalarFunction(ScalarFunction::new_udf(
                Arc::new(getter),
                vec![col("json"), lit("val")],
            ));
            assert_eq!(
                typed_parser.is_valid_reference(&getter, &DataType::LargeBinary),
                Some(target_type.clone())
            );
            let typed_extract = Expr::ScalarFunction(ScalarFunction::new_udf(
                Arc::new(json_extract_typed_udf(&target_type).unwrap()),
                vec![col("json"), lit("$.val")],
            ));
            assert_eq!(
                typed_parser.is_valid_reference(&typed_extract, &DataType::LargeBinary),
                Some(target_type)
            );
        }

        let json_path_getter = Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(json_get_int_udf()),
            vec![col("json"), lit("$.val")],
        ));
        assert_eq!(
            parser.is_valid_reference(&json_path_getter, &DataType::LargeBinary),
            None
        );

        let json_extract = Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(json_extract_udf()),
            vec![col("json"), lit("val")],
        ));
        assert_eq!(
            parser.is_valid_reference(&json_extract, &DataType::LargeBinary),
            None
        );

        let utf8_parser = plugin
            .new_query_parser_with_training_data_type(
                "json_idx".to_string(),
                &legacy_details,
                Some(&DataType::Utf8),
            )
            .unwrap();
        let string_getter = Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(json_get_string_udf()),
            vec![col("json"), lit("val")],
        ));
        assert_eq!(
            utf8_parser.is_valid_reference(&string_getter, &DataType::LargeBinary),
            Some(DataType::Utf8)
        );
        let typed_utf8 = Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(json_extract_typed_udf(&DataType::Utf8).unwrap()),
            vec![col("json"), lit("$.val")],
        ));
        assert_eq!(
            utf8_parser.is_valid_reference(&typed_utf8, &DataType::LargeBinary),
            None
        );
    }

    fn json_update_batch(json_docs: &[&str], row_ids: Vec<u64>) -> RecordBatch {
        use arrow_array::{LargeBinaryArray, UInt64Array};

        let jsonb = json_docs
            .iter()
            .map(|json| json.parse::<jsonb::OwnedJsonb>().unwrap().to_vec())
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::LargeBinary, true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(LargeBinaryArray::from_iter_values(
                    jsonb.iter().map(Vec::as_slice),
                )),
                Arc::new(UInt64Array::from(row_ids)),
            ],
        )
        .unwrap()
    }

    fn json_update_stream(json_docs: &[&str], row_ids: Vec<u64>) -> SendableRecordBatchStream {
        use futures::stream;

        let batch = json_update_batch(json_docs, row_ids);
        let schema = batch.schema();
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter([Ok(batch)]),
        ))
    }

    #[tokio::test]
    async fn test_legacy_json_index_maintenance_preserves_conversion() {
        let (source_store, _source_dir) = local_json_index_store();
        let legacy = load_legacy_utf8_json_index(source_store, "btree").await;

        let (remap_store, _remap_dir) = local_json_index_store();
        let remapped_created = legacy
            .remap(&RowAddrRemap::empty(), remap_store.as_ref())
            .await
            .unwrap();
        assert_eq!(remapped_created.index_version, 0);
        let remapped_details =
            crate::pb::JsonIndexDetails::decode(remapped_created.index_details.value.as_slice())
                .unwrap();
        assert_eq!(remapped_details.target_data_type, None);
        assert_eq!(remapped_details.conversion, None);

        let registry = IndexPluginRegistry::with_default_plugins();
        let json_plugin = registry.get_plugin_by_name("json").unwrap();
        let remapped = json_plugin
            .load_index(
                remap_store,
                &remapped_created.index_details,
                None,
                &LanceCache::no_cache(),
            )
            .await
            .unwrap();
        let decoded_key = JsonQuery::new(
            Arc::new(SargableQuery::Equals(ScalarValue::Utf8(Some(
                "foo".to_string(),
            )))),
            "$.v".to_string(),
        );
        let decoded_result = remapped
            .search(&decoded_key, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(
            decoded_result,
            SearchResult::exact(RowAddrTreeMap::from_iter([0]))
        );
        let typed_key = JsonQuery::new(
            Arc::new(SargableQuery::Equals(ScalarValue::Utf8(Some(
                r#""foo""#.to_string(),
            )))),
            "$.v".to_string(),
        );
        let typed_result = remapped
            .search(&typed_key, &NoOpMetricsCollector)
            .await
            .unwrap();
        assert_eq!(typed_result, SearchResult::exact(RowAddrTreeMap::default()));
        assert!(
            remapped.update_criteria().requires_old_data,
            "legacy maintenance must select a full rebuild"
        );
        assert!(
            remapped.requires_full_rebuild(),
            "legacy maintenance must replace every physical segment"
        );

        let (update_store, _update_dir) = local_json_index_store();
        let error = remapped
            .update(
                json_update_stream(&[r#"{"v": "bar"}"#], vec![1]),
                update_store.as_ref(),
                None,
            )
            .await
            .err()
            .expect("legacy incremental update should require a rebuild");
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(
            error
                .to_string()
                .contains("must be fully rebuilt before adding new data")
        );
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

        let mut converted =
            JsonIndexPlugin::extract_json_typed(raw_stream, "$.v".to_string(), DataType::Int64)
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
        assert!(message.contains("JSONPath '$.v'"), "{message}");
        assert!(message.contains("expected Int64"), "{message}");
        assert!(message.contains("JSON type Boolean"), "{message}");
    }

    #[tokio::test]
    #[serial_test::serial(LANCE_DF_SPILL_POOL)]
    async fn test_json_derived_params_preserve_wrapper() {
        let (store, _tmpdir) = local_json_index_store();
        let index = train_and_load_json_index(
            store,
            "btree",
            TrainingOrdering::None,
            "v",
            &[r#"{"v": 1}"#],
        )
        .await;

        let derived = index.derive_index_params().unwrap();
        assert_eq!(derived.index_type, "json");
        let parameters: JsonIndexParameters =
            serde_json::from_str(derived.params.as_deref().unwrap()).unwrap();
        assert_eq!(parameters.path, "v");
        assert_eq!(parameters.target_index_type, "btree");
        assert!(parameters.target_index_parameters.is_some());
        assert_eq!(
            parameters.target_data_type,
            Some(JsonIndexTargetType::Int64)
        );
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
        vec![0, 1, 2]
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
        assert_eq!(
            search(SargableQuery::Range(
                Bound::Excluded(ScalarValue::Float64(Some(0.0))),
                Bound::Unbounded,
            ))
            .await,
            SearchResult::exact(RowAddrTreeMap::from_iter([0u64, 3]))
                .with_nulls(RowAddrTreeMap::from_iter([1u64])),
            "> 0"
        );
        assert_eq!(
            search(SargableQuery::Equals(ScalarValue::Float64(Some(40.1)))).await,
            SearchResult::exact(RowAddrTreeMap::from_iter([0u64]))
                .with_nulls(RowAddrTreeMap::from_iter([1u64])),
            "= 40.1"
        );
        assert_eq!(
            search(SargableQuery::Range(
                Bound::Unbounded,
                Bound::Excluded(ScalarValue::Float64(Some(100.0))),
            ))
            .await,
            SearchResult::exact(RowAddrTreeMap::from_iter([0u64, 2, 3]))
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
