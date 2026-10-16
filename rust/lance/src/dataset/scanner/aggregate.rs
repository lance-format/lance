// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use datafusion::common::DFSchema;
use datafusion::functions_aggregate;
use datafusion::logical_expr::{Expr, col, lit};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion_expr::ExprSchemable;
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_physical_expr::aggregate::LoweredAggregateBuilder;
use datafusion_physical_expr::{PhysicalExpr, create_physical_expr};
use futures::FutureExt;
use futures::future::BoxFuture;
use lance_datafusion::aggregate::Aggregate;

use crate::{Error, Result};

use super::Scanner;

/// Aggregate expression from Substrait or DataFusion.
#[derive(Debug, Clone)]
pub enum AggregateExpr {
    #[cfg(feature = "substrait")]
    Substrait(Vec<u8>),
    Datafusion {
        group_by: Vec<Expr>,
        aggregates: Vec<Expr>,
    },
}

impl AggregateExpr {
    /// Create a new builder for aggregate expressions.
    ///
    /// # Example
    /// ```ignore
    /// let agg = AggregateExpr::builder()
    ///     .group_by("category")
    ///     .count_star().alias("total_count")
    ///     .sum("amount").alias("total_amount")
    ///     .avg("price")
    ///     .build();
    /// scanner.aggregate(agg);
    /// ```
    pub fn builder() -> AggregateExprBuilder<false> {
        AggregateExprBuilder::new()
    }

    /// Create from Substrait Plan bytes.
    #[cfg(feature = "substrait")]
    pub fn substrait(bytes: impl Into<Vec<u8>>) -> Self {
        Self::Substrait(bytes.into())
    }

    /// Create from DataFusion expressions.
    /// Use `.alias()` on expressions to set output column names.
    pub fn datafusion(group_by: Vec<Expr>, aggregates: Vec<Expr>) -> Self {
        Self::Datafusion {
            group_by,
            aggregates,
        }
    }

    /// Parse into a unified Aggregate structure.
    ///
    /// For Substrait, this parses the bytes into DataFusion expressions.
    /// For DataFusion, this just wraps the expressions.
    ///
    /// The schema is used to resolve field references in Substrait expressions.
    pub(super) fn parse(
        self,
        #[allow(unused_variables)] schema: Arc<ArrowSchema>,
    ) -> Result<Aggregate> {
        match self {
            #[cfg(feature = "substrait")]
            Self::Substrait(bytes) => {
                use lance_datafusion::exec::{LanceExecutionOptions, get_session_context};
                use lance_datafusion::substrait::parse_substrait_aggregate;

                let ctx = get_session_context(&LanceExecutionOptions::default());
                parse_substrait_aggregate(&bytes, schema, &ctx.state())
                    .now_or_never()
                    .expect("could not parse the Substrait aggregate in a synchronous fashion")
            }
            Self::Datafusion {
                group_by,
                aggregates,
            } => Ok(Aggregate::new(group_by, aggregates)),
        }
    }
}

/// Builder for creating aggregate expressions without using DataFusion or Substrait directly.
///
/// The const generic `HAS_PENDING` tracks whether there's a pending aggregate that can be aliased.
/// When `HAS_PENDING` is `true`, the last item in `aggregates` is the pending aggregate.
#[derive(Debug, Clone)]
pub struct AggregateExprBuilder<const HAS_PENDING: bool> {
    group_by: Vec<Expr>,
    aggregates: Vec<Expr>,
}

impl Default for AggregateExprBuilder<false> {
    fn default() -> Self {
        Self {
            group_by: Vec::new(),
            aggregates: Vec::new(),
        }
    }
}

impl AggregateExprBuilder<false> {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the aggregate expression.
    pub fn build(self) -> AggregateExpr {
        AggregateExpr::Datafusion {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }
}

impl<const HAS_PENDING: bool> AggregateExprBuilder<HAS_PENDING> {
    /// Add a column to group by.
    ///
    /// Multiple invocations will add to the list (not replace it).
    /// E.g. `.group_by("x").group_by("y")` will group by both `x` and `y`.
    pub fn group_by(mut self, column: impl Into<String>) -> AggregateExprBuilder<false> {
        self.group_by.push(col(column.into()));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add multiple columns to group by.
    ///
    /// Multiple invocations will add to the list (not replace it).
    /// E.g. `.group_by("x").group_by_columns(["y", "z"])` will group by `x`, `y`, and `z`.
    pub fn group_by_columns(
        mut self,
        columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> AggregateExprBuilder<false> {
        for column in columns {
            self.group_by.push(col(column.into()));
        }
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add COUNT(*) aggregate that counts all rows.
    pub fn count_star(mut self) -> AggregateExprBuilder<true> {
        self.aggregates
            .push(functions_aggregate::count::count(lit(1)));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add COUNT(column) aggregate.
    ///
    /// Unlike `count_star`, this will only count the number of rows where `column`
    /// is not NULL.
    pub fn count(mut self, column: impl Into<String>) -> AggregateExprBuilder<true> {
        self.aggregates
            .push(functions_aggregate::count::count(col(column.into())));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add SUM(column) aggregate.
    pub fn sum(mut self, column: impl Into<String>) -> AggregateExprBuilder<true> {
        self.aggregates
            .push(functions_aggregate::sum::sum(col(column.into())));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add AVG(column) aggregate.
    pub fn avg(mut self, column: impl Into<String>) -> AggregateExprBuilder<true> {
        self.aggregates
            .push(functions_aggregate::average::avg(col(column.into())));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add MIN(column) aggregate.
    pub fn min(mut self, column: impl Into<String>) -> AggregateExprBuilder<true> {
        self.aggregates
            .push(functions_aggregate::min_max::min(col(column.into())));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Add MAX(column) aggregate.
    pub fn max(mut self, column: impl Into<String>) -> AggregateExprBuilder<true> {
        self.aggregates
            .push(functions_aggregate::min_max::max(col(column.into())));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }
}

impl AggregateExprBuilder<true> {
    /// Set an alias for the pending aggregate (the last added aggregate).
    pub fn alias(mut self, name: impl Into<String>) -> AggregateExprBuilder<false> {
        let pending = self.aggregates.pop().expect("pending aggregate must exist");
        self.aggregates.push(pending.alias(name.into()));
        AggregateExprBuilder {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }

    /// Build the aggregate expression.
    pub fn build(self) -> AggregateExpr {
        AggregateExpr::Datafusion {
            group_by: self.group_by,
            aggregates: self.aggregates,
        }
    }
}

impl Scanner {
    /// Create an execution plan with aggregation.
    ///
    /// Requires `aggregate()` to be called first.
    #[deprecated(note = "Use create_plan() instead, which now applies aggregate automatically")]
    pub fn create_aggregate_plan(&self) -> BoxFuture<'_, Result<Arc<dyn ExecutionPlan>>> {
        async move {
            if self.aggregate.is_none() {
                return Err(Error::invalid_input(
                    "create_aggregate_plan called but no aggregate was set",
                ));
            }
            // create_plan() now applies aggregate automatically when set
            self.create_plan().await
        }
        .boxed()
    }

    pub(super) async fn apply_aggregate(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        agg: &Aggregate,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion_physical_expr::aggregate::AggregateFunctionExpr;

        let schema = plan.schema();
        let df_schema = DFSchema::try_from(schema.as_ref().clone())?;

        let group_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = agg
            .group_by
            .iter()
            .map(|expr| {
                let name = expr.schema_name().to_string();
                let physical_expr =
                    create_physical_expr(expr, &df_schema, &ExecutionProps::default())?;
                Ok((physical_expr, name))
            })
            .collect::<Result<_>>()?;

        #[allow(clippy::type_complexity)]
        let aggr_results: Vec<(Arc<AggregateFunctionExpr>, Option<Arc<dyn PhysicalExpr>>)> = agg
            .aggregates
            .iter()
            .map(|expr| self.build_physical_aggregate_expr(expr, &df_schema, &schema))
            .collect::<Result<_>>()?;

        let (aggr_exprs, filters): (Vec<_>, Vec<_>) = aggr_results.into_iter().unzip();

        Ok(Arc::new(AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(group_exprs),
            aggr_exprs,
            filters,
            plan,
            schema,
        )?) as Arc<dyn ExecutionPlan>)
    }

    #[allow(clippy::type_complexity)]
    fn build_physical_aggregate_expr(
        &self,
        expr: &Expr,
        df_schema: &DFSchema,
        input_schema: &SchemaRef,
    ) -> Result<(
        Arc<datafusion_physical_expr::aggregate::AggregateFunctionExpr>,
        Option<Arc<dyn PhysicalExpr>>,
    )> {
        let coerced_expr = self.coerce_aggregate_expr(expr, df_schema)?;

        // Name and display the aggregate the way the deprecated
        // `create_aggregate_expr_and_maybe_filter` did, so plans and their
        // output column names stay the same.
        let (name, human_display) = match &coerced_expr {
            Expr::Alias(alias) => (
                Some(alias.name.clone()),
                coerced_expr.human_display().to_string(),
            ),
            Expr::AggregateFunction(_) => (
                Some(coerced_expr.schema_name().to_string()),
                coerced_expr.human_display().to_string(),
            ),
            _ => (None, String::new()),
        };
        let execution_props = ExecutionProps::default();
        let mut builder = LoweredAggregateBuilder::new(
            &coerced_expr,
            df_schema,
            input_schema.as_ref(),
            &execution_props,
        )
        .with_human_display(human_display);
        if let Some(name) = name {
            builder = builder.with_name(name);
        }

        // Note: order_by is already embedded in the AggregateFunctionExpr for ordered aggregates
        let lowered = builder.build()?;
        Ok((lowered.aggregate, lowered.filter))
    }

    /// Apply type coercion to aggregate arguments for UserDefined signature functions.
    ///
    /// Most aggregate functions (SUM, COUNT, MIN, MAX) have explicit type signatures that
    /// DataFusion handles automatically. However, some functions like AVG use UserDefined
    /// type signatures in the Substrait consumer, which means DataFusion doesn't know the
    /// expected input types and won't perform automatic coercion. We must explicitly coerce
    /// arguments to the types returned by `func.coerce_types()`.
    fn coerce_aggregate_expr(&self, expr: &Expr, schema: &DFSchema) -> Result<Expr> {
        Self::coerce_aggregate_expr_impl(expr, schema)
    }

    fn coerce_aggregate_expr_impl(expr: &Expr, schema: &DFSchema) -> Result<Expr> {
        use datafusion::logical_expr::Expr;
        use datafusion::logical_expr::expr::AggregateFunction;
        use datafusion::logical_expr::type_coercion::functions::fields_with_udf;

        match expr {
            Expr::AggregateFunction(agg_func) => {
                let func = &agg_func.func;
                let args = &agg_func.params.args;

                if args.is_empty() {
                    return Ok(expr.clone());
                }

                let current_fields: Vec<arrow_schema::FieldRef> = args
                    .iter()
                    .enumerate()
                    .map(|(i, e)| {
                        let dt = e.get_type(schema)?;
                        Ok(Arc::new(arrow_schema::Field::new(
                            format!("arg_{i}"),
                            dt,
                            true,
                        )))
                    })
                    .collect::<std::result::Result<_, datafusion::common::DataFusionError>>()?;

                let coerced_fields = fields_with_udf(&current_fields, func.as_ref())?;
                let coerced_args: Vec<Expr> = args
                    .iter()
                    .zip(coerced_fields.iter())
                    .map(|(arg, target_field)| {
                        let arg_type = arg.get_type(schema)?;
                        let target_type = target_field.data_type();
                        if arg_type == *target_type {
                            Ok(arg.clone())
                        } else {
                            arg.clone().cast_to(target_type, schema)
                        }
                    })
                    .collect::<std::result::Result<_, _>>()?;

                Ok(Expr::AggregateFunction(AggregateFunction::new_udf(
                    func.clone(),
                    coerced_args,
                    agg_func.params.distinct,
                    agg_func.params.filter.clone(),
                    agg_func.params.order_by.clone(),
                    agg_func.params.null_treatment,
                )))
            }
            Expr::Alias(alias) => {
                // Recursively coerce the inner expression and preserve the alias
                let coerced_inner = Self::coerce_aggregate_expr_impl(&alias.expr, schema)?;
                Ok(coerced_inner.alias(&alias.name))
            }
            other => Err(Error::invalid_input(format!(
                "Expected aggregate function expression, got {:?}",
                other.variant_name()
            ))),
        }
    }
}
