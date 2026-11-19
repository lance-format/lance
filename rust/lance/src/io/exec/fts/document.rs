// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execution node that normalizes schema-derived FTS documents.

use std::sync::Arc;

use arrow_schema::DataType;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion_physical_plan::ExecutionPlanProperties;
use lance_core::{Error, ROW_ID, Result, datatypes::Schema as LanceSchema};
use lance_index::scalar::inverted::DocumentGranularity;
use lance_index::scalar::registry::VALUE_COLUMN_NAME;

use crate::index::scalar::inverted::{
    ResolvedFtsField, fts_document_schema, resolve_fts_field, transform_fts_document_stream,
};

/// Expands a schema-derived nested FTS source into one canonical row per
/// logical document before flat search or index building consumes it.
#[derive(Debug)]
pub struct FtsDocumentExec {
    input: Arc<dyn ExecutionPlan>,
    resolved: ResolvedFtsField,
    properties: Arc<PlanProperties>,
}

impl FtsDocumentExec {
    /// Expand a source field into row or list-element documents.
    ///
    /// `source_schema` describes the field before projection or aliasing. The
    /// input must contain that field's complete root column (or the same column
    /// aliased to `value`) and a UInt64 `_rowid`. Null row IDs are selection
    /// masks and do not produce documents. List coordinates are derived from
    /// the schema, including physical positions occupied by null elements.
    ///
    /// A plan codec needs only the column path and document granularity in
    /// addition to its input and the schema of its pinned dataset snapshot:
    ///
    /// ```
    /// # use lance::{io::exec::fts::FtsDocumentExec, Result};
    /// # use lance_core::datatypes::Schema;
    /// # use datafusion::physical_plan::ExecutionPlan;
    /// # fn rebuild(plan: &FtsDocumentExec, schema: &Schema) -> Result<FtsDocumentExec> {
    /// FtsDocumentExec::try_new(
    ///     plan.children()[0].clone(),
    ///     schema,
    ///     plan.column(),
    ///     plan.document_granularity(),
    /// )
    /// # }
    /// ```
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        source_schema: &LanceSchema,
        column: &str,
        document_granularity: DocumentGranularity,
    ) -> Result<Self> {
        let resolved = resolve_fts_field(source_schema, column, document_granularity)?;
        let input_schema = input.schema();
        let row_id = input_schema.field_with_name(ROW_ID).map_err(|_| {
            Error::invalid_input("FTS document input is missing _rowid".to_string())
        })?;
        if row_id.data_type() != &DataType::UInt64 {
            return Err(Error::invalid_input(format!(
                "FTS document input _rowid must be UInt64, got {}",
                row_id.data_type()
            )));
        }
        let source = source_schema
            .fields
            .iter()
            .find(|field| field.name == resolved.root_column)
            .ok_or_else(|| Error::internal("Resolved FTS root field is missing"))?;
        let input_field = input_schema
            .field_with_name(&resolved.root_column)
            .or_else(|_| input_schema.field_with_name(VALUE_COLUMN_NAME))
            .map_err(|_| {
                Error::invalid_input(format!(
                    "FTS document input is missing root column '{}' or its {VALUE_COLUMN_NAME} alias",
                    resolved.root_column
                ))
            })?;
        if input_field.data_type() != &source.data_type() {
            return Err(Error::invalid_input(format!(
                "FTS document input column '{}' must have source type {}, got {}",
                input_field.name(),
                source.data_type(),
                input_field.data_type()
            )));
        }
        Ok(Self::new(input, resolved))
    }

    /// Canonical source field path, before the input root column is aliased.
    pub fn column(&self) -> &str {
        &self.resolved.canonical_path
    }

    /// Whether each row or each deepest list element is one document.
    pub fn document_granularity(&self) -> DocumentGranularity {
        self.resolved.document_granularity
    }

    pub(crate) fn new(input: Arc<dyn ExecutionPlan>, resolved: ResolvedFtsField) -> Self {
        let schema = fts_document_schema(resolved.coordinate_rank());
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            input.output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            input,
            resolved,
            properties,
        }
    }
}

impl DisplayAs for FtsDocumentExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "FtsDocument: column={}, granularity={:?}",
            self.resolved.canonical_path, self.resolved.document_granularity
        )
    }
}

impl ExecutionPlan for FtsDocumentExec {
    fn name(&self) -> &str {
        "FtsDocumentExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "FtsDocumentExec expects one child".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children.pop().unwrap(),
            self.resolved.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        transform_fts_document_stream(
            self.input.execute(partition, context)?,
            self.resolved.clone(),
        )
        .map_err(DataFusionError::from)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::buffer::NullBuffer;
    use arrow_array::builder::{ListBuilder, StringBuilder};
    use arrow_array::{
        ArrayRef, Int32Array, RecordBatch, StringArray, StructArray, UInt32Array, UInt64Array,
    };
    use arrow_schema::DataType;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::execution::TaskContext;
    use datafusion::physical_plan::{ExecutionPlan, empty::EmptyExec};
    use futures::TryStreamExt;
    use lance_core::{Error, ROW_ID, datatypes::Schema as LanceSchema};
    use lance_index::scalar::inverted::DocumentGranularity;
    use lance_index::scalar::registry::VALUE_COLUMN_NAME;
    use rstest::rstest;

    use super::FtsDocumentExec;

    #[rstest]
    #[case::rows(DocumentGranularity::Row)]
    #[case::elements(DocumentGranularity::ListElement)]
    #[tokio::test]
    async fn fts_document_plan_reconstruction(
        #[case] granularity: DocumentGranularity,
        #[values(false, true)] aliased: bool,
        #[values(false, true)] large_lists: bool,
    ) {
        let mut lists = ListBuilder::new(ListBuilder::new(StringBuilder::new()));
        for row in [
            Some(vec![
                Some(vec![Some("alpha"), None, Some("beta")]),
                Some(vec![]),
                None,
                Some(vec![Some("gamma")]),
            ]),
            Some(vec![Some(vec![Some("hidden by null struct")])]),
            Some(vec![]),
            Some(vec![Some(vec![Some("masked")])]),
            Some(vec![Some(vec![Some("delta")])]),
            None,
            Some(vec![None, Some(vec![None])]),
        ] {
            if let Some(row) = row {
                for list in row {
                    if let Some(values) = list {
                        for value in values {
                            lists.values().values().append_option(value);
                        }
                        lists.values().append(true);
                    } else {
                        lists.values().append(false);
                    }
                }
                lists.append(true);
            } else {
                lists.append(false);
            }
        }
        let lists = Arc::new(lists.finish()) as ArrayRef;
        let lists = if large_lists {
            arrow_cast::cast(
                &lists,
                &DataType::LargeList(Arc::new(arrow_schema::Field::new(
                    "item",
                    DataType::LargeList(Arc::new(arrow_schema::Field::new(
                        "item",
                        DataType::Utf8,
                        true,
                    ))),
                    true,
                ))),
            )
            .unwrap()
        } else {
            lists
        };
        let groups = Arc::new(StructArray::new(
            vec![
                Arc::new(arrow_schema::Field::new("unused", DataType::Int32, false)),
                Arc::new(arrow_schema::Field::new(
                    "docs",
                    lists.data_type().clone(),
                    true,
                )),
            ]
            .into(),
            vec![Arc::new(Int32Array::from(vec![0; 7])), lists],
            Some(NullBuffer::from(vec![
                true, false, true, true, true, true, true,
            ])),
        )) as ArrayRef;
        let source_schema = arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "groups",
            groups.data_type().clone(),
            true,
        )]);
        let source_schema = LanceSchema::try_from(&source_schema).unwrap();
        let input_schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(
                if aliased { VALUE_COLUMN_NAME } else { "groups" },
                groups.data_type().clone(),
                true,
            ),
            lance_core::ROW_ID_FIELD.clone(),
        ]));
        let batch = RecordBatch::try_new(
            input_schema.clone(),
            vec![
                groups,
                Arc::new(UInt64Array::from(vec![
                    Some(0),
                    Some(20),
                    Some(30),
                    None,
                    Some(50),
                    Some(60),
                    Some(70),
                ])),
            ],
        )
        .unwrap();
        let input = MemorySourceConfig::try_new_exec(
            &[vec![batch.slice(0, 4)], vec![batch.slice(4, 3)]],
            input_schema,
            None,
        )
        .unwrap();
        let original =
            FtsDocumentExec::try_new(input, &source_schema, "GROUPS.docs", granularity).unwrap();
        assert_eq!(original.column(), "groups.docs");
        let rebuilt = FtsDocumentExec::try_new(
            original.children()[0].clone(),
            &source_schema,
            original.column(),
            original.document_granularity(),
        )
        .unwrap();
        assert_eq!(
            rebuilt.properties().output_partitioning().partition_count(),
            2
        );
        let expected_names = if granularity.is_list_element() {
            vec!["value", "_rowid", "_doc_index_0", "_doc_index_1"]
        } else {
            vec!["value", "_rowid"]
        };
        assert_eq!(
            rebuilt
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            expected_names,
        );
        let (texts, ids) = match granularity {
            DocumentGranularity::Row => (
                vec!["alpha beta gamma", "", "", "delta", "", ""],
                vec![0, 20, 30, 50, 60, 70],
            ),
            DocumentGranularity::ListElement => (
                // A null deepest element is an empty document, unlike a null
                // parent container, which has no elements to expand.
                vec!["alpha", "", "beta", "gamma", "delta", ""],
                vec![0, 0, 0, 0, 50, 70],
            ),
        };
        let mut expected_columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(texts)),
            Arc::new(UInt64Array::from(ids)),
        ];
        if granularity.is_list_element() {
            expected_columns.extend([
                Arc::new(UInt32Array::from(vec![0, 0, 0, 3, 0, 1])) as ArrayRef,
                Arc::new(UInt32Array::from(vec![0, 1, 2, 0, 0, 0])) as ArrayRef,
            ]);
        }
        let expected = RecordBatch::try_new(rebuilt.schema(), expected_columns).unwrap();
        for plan in [original, rebuilt] {
            // Re-execution must preserve row IDs and physical list coordinates.
            for _ in 0..2 {
                let mut batches = Vec::new();
                for partition in 0..2 {
                    batches.extend(
                        plan.execute(partition, Arc::new(TaskContext::default()))
                            .unwrap()
                            .try_collect::<Vec<_>>()
                            .await
                            .unwrap(),
                    );
                }
                let actual =
                    arrow_select::concat::concat_batches(&plan.schema(), &batches).unwrap();
                assert_eq!(actual, expected);
            }
        }
    }

    #[rstest]
    #[case::missing_row_id(vec![("text", DataType::Utf8)], "missing _rowid")]
    #[case::wrong_row_id(vec![("text", DataType::Utf8), (ROW_ID, DataType::Int64)], "_rowid must be UInt64")]
    #[case::missing_source(vec![("other", DataType::Utf8), (ROW_ID, DataType::UInt64)], "missing root column")]
    #[case::wrong_source(vec![("text", DataType::Int32), (ROW_ID, DataType::UInt64)], "must have source type")]
    fn fts_document_plan_rejects_invalid_input(
        #[case] fields: Vec<(&str, DataType)>,
        #[case] message: &str,
    ) {
        let source =
            LanceSchema::try_from(&arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                "text",
                DataType::Utf8,
                true,
            )]))
            .unwrap();
        let input = Arc::new(EmptyExec::new(Arc::new(arrow_schema::Schema::new(
            fields
                .into_iter()
                .map(|(name, dtype)| arrow_schema::Field::new(name, dtype, true))
                .collect::<Vec<_>>(),
        ))));
        let error =
            FtsDocumentExec::try_new(input, &source, "text", DocumentGranularity::Row).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(message), "{error}");
    }
}
