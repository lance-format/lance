// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub mod aggregate;
pub use lance_datafusion_exec::chunker;
pub mod dataframe;
#[cfg(feature = "datagen")]
pub use lance_datafusion_exec::datagen;
pub mod exec;
pub use lance_datafusion_exec::expr;
pub mod logical_expr;
pub mod planner;
pub mod projection;
pub mod pb {
    #![allow(clippy::all)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(unused)]
    #![allow(improper_ctypes)]
    #![allow(clippy::upper_case_acronyms)]
    #![allow(clippy::use_self)]
    include!(concat!(env!("OUT_DIR"), "/lance.datafusion.rs"));
}
mod numeric_coercion;
#[doc(hidden)]
pub mod signed_zero;
pub mod spill;
pub mod sql;
#[cfg(feature = "substrait")]
pub mod substrait;
pub mod udf;
pub use lance_datafusion_exec::utils;
