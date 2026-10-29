// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Datafusion user defined functions

#[cfg(not(feature = "geo"))]
use arrow_array::ArrayRef;
#[cfg(not(feature = "geo"))]
use arrow_schema::DataType;
#[cfg(not(feature = "geo"))]
use datafusion::logical_expr::{Volatility, create_udf};
use datafusion::prelude::SessionContext;
#[cfg(not(feature = "geo"))]
use datafusion_functions::utils::make_scalar_function;
#[cfg(not(feature = "geo"))]
use std::sync::Arc;

pub use lance_datafusion_exec::udf::{CONTAINS_TOKENS_UDF, json};

#[cfg(feature = "geo")]
mod geo;

/// Register UDF functions to datafusion context.
pub fn register_functions(ctx: &SessionContext) {
    ctx.register_udf(CONTAINS_TOKENS_UDF.clone());
    // JSON functions
    ctx.register_udf(json::json_extract_udf());
    ctx.register_udf(json::json_extract_with_type_udf());
    ctx.register_udf(json::json_exists_udf());
    ctx.register_udf(json::json_get_udf());
    ctx.register_udf(json::json_get_string_udf());
    ctx.register_udf(json::json_get_int_udf());
    ctx.register_udf(json::json_get_float_udf());
    ctx.register_udf(json::json_get_bool_udf());
    ctx.register_udf(json::json_array_contains_udf());
    ctx.register_udf(json::json_array_length_udf());
    // GEO functions
    #[cfg(feature = "geo")]
    {
        geodatafusion::register(ctx);
        geo::register_functions(ctx);
    }
    #[cfg(not(feature = "geo"))]
    register_geo_stub_functions(ctx);
}

/// When the `geo` feature is disabled, register stub UDFs for spatial SQL functions
/// so that users get a clear error mentioning the feature flag instead of
/// DataFusion's generic "Unknown function" error.
#[cfg(not(feature = "geo"))]
fn register_geo_stub_functions(ctx: &SessionContext) {
    let geo_funcs = [
        "st_intersects",
        "st_contains",
        "st_within",
        "st_touches",
        "st_crosses",
        "st_overlaps",
        "st_covers",
        "st_coveredby",
        "st_distance",
        "st_area",
        "st_length",
    ];

    for name in geo_funcs {
        let func_name = name.to_string();
        let stub = Arc::new(make_scalar_function(
            move |_args: &[ArrayRef]| {
                Err(datafusion::error::DataFusionError::Plan(format!(
                    "Function '{}' requires the `geo` feature. \
                     Rebuild with `--features geo` to enable geospatial functions.",
                    func_name
                )))
            },
            vec![],
        ));

        ctx.register_udf(create_udf(
            name,
            vec![DataType::Binary, DataType::Binary],
            DataType::Boolean,
            Volatility::Immutable,
            stub,
        ));
    }
}
