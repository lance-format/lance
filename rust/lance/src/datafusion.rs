// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Utilities for integrating Lance into DataFusion

pub(crate) mod dataframe;
pub(crate) mod index_join;
pub(crate) mod logical_plan;
pub(crate) mod planning_context;

pub use dataframe::LanceTableProvider;
