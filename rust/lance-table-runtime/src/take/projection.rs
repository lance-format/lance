// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field};
use lance_arrow::RecordBatchExt;
use lance_arrow::json::convert_lance_json_to_arrow;
use lance_core::datatypes::{OnMissing, OnTypeMismatch, Projectable, Projection, Schema};
use lance_core::{ROW_ADDR, ROW_OFFSET, ROW_OFFSET_FIELD, Result, is_system_column};

use super::{TakeOptions, TakeSelection};
use crate::Table;

/// A resolved schema for a point take, preserving the requested column order.
///
/// Use this to validate a projection before executing a take with
/// [`Table::take_with_options`].
#[derive(Clone, Debug)]
pub struct TakeProjection {
    physical: Projection,
    columns: Vec<String>,
    with_row_offset: bool,
}

impl TakeProjection {
    /// Resolve data columns by name and type and include requested system columns.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance_table_runtime::{Table, TakeProjection};
    /// # fn example(table: Arc<Table>) -> lance_core::Result<()> {
    /// let requested = table.schema().project_preserve_system_columns(&["id", "_rowid"])?;
    /// let projection = TakeProjection::try_new(table, &requested)?;
    /// # let _ = projection;
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(base: Arc<dyn Projectable>, requested: &Schema) -> Result<Self> {
        // Preserve the schema-validation errors for requests with only data columns.
        let validated;
        let requested = if requested
            .fields
            .iter()
            .any(|field| is_system_column(&field.name))
        {
            requested
        } else {
            validated = base.schema().project_by_schema(
                requested,
                OnMissing::Error,
                OnTypeMismatch::Error,
            )?;
            &validated
        };
        let physical = Projection::from_schema(base, requested)?;
        let columns = requested
            .fields
            .iter()
            .map(|field| field.name.clone())
            .collect();
        let with_row_offset = requested
            .fields
            .iter()
            .any(|field| field.name == ROW_OFFSET);
        Ok(Self {
            physical,
            columns,
            with_row_offset,
        })
    }

    pub(super) async fn take(
        &self,
        table: &Table,
        selection: TakeSelection<'_>,
        options: TakeOptions,
    ) -> Result<RecordBatch> {
        let physical_batch = table
            .take_physical(selection, &self.physical, self.with_row_offset, options)
            .await?;

        let batch = match physical_batch {
            Some(batch) => batch,
            None => {
                let mut batch = RecordBatch::new_empty(Arc::new(self.physical.to_arrow_schema()));
                if self.with_row_offset {
                    batch = batch.try_with_column(
                        ROW_OFFSET_FIELD.clone().with_nullable(false),
                        Arc::new(UInt64Array::from(Vec::<u64>::new())),
                    )?;
                }
                if options.with_row_address && batch.column_by_name(ROW_ADDR).is_none() {
                    batch = batch.try_with_column(
                        Field::new(ROW_ADDR, DataType::UInt64, false),
                        Arc::new(UInt64Array::from(Vec::<u64>::new())),
                    )?;
                }
                batch
            }
        };

        let needs_address =
            options.with_row_address && !self.columns.iter().any(|name| name == ROW_ADDR);
        let columns = self
            .columns
            .iter()
            .map(String::as_str)
            .chain(needs_address.then_some(ROW_ADDR));
        let indices = columns
            .map(|name| batch.schema_ref().index_of(name))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(convert_lance_json_to_arrow(&batch.project(&indices)?)?)
    }
}
