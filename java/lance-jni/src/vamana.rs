// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The graph parameters of the IVF_VAMANA_* index types.

use jni::JNIEnv;
use jni::objects::JObject;
use lance::index::vector::StageParams;
use lance_index::vector::vamana::VamanaBuildParams;

use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;

/// The Vamana stage of a Java `VectorIndexParams`, when it has one. Lance
/// validates the parameters when the build starts; only a negative count,
/// which no Rust parameter can hold, is refused here.
pub fn stage(env: &mut JNIEnv, vector_index_params: &JObject) -> Result<Option<StageParams>> {
    env.get_optional_from_method(vector_index_params, "getVamanaParams", |env, vamana| {
        let max_degree = env.call_method(&vamana, "getMaxDegree", "()I", &[])?.i()?;
        let search_list_size = env
            .call_method(&vamana, "getSearchListSize", "()I", &[])?
            .i()?;
        let alpha = env.call_method(&vamana, "getAlpha", "()F", &[])?.f()?;
        Ok(StageParams::Vamana(VamanaBuildParams {
            max_degree: non_negative("max_degree", max_degree)?,
            search_list_size: non_negative("search_list_size", search_list_size)? as usize,
            alpha,
        }))
    })
}

fn non_negative(name: &str, value: i32) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::input_error(format!("Vamana {name} must not be negative, got {value}")))
}
