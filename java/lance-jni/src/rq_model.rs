// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::str::FromStr;

use arrow::datatypes::Float32Type;
use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JObject};
use jni::sys::{jbyte, jbyteArray, jint, jintArray};
use lance_index::vector::bq::builder::RabitQuantizer;
use lance_index::vector::bq::storage::RabitQuantizationMetadata;
use lance_index::vector::bq::{RQBuildParams, RQRotationType, validate_rq_num_bits};
use lance_index::vector::quantizer::Quantization;

use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;

pub(crate) fn extract_rq_build_params(env: &mut JNIEnv, rq_obj: JObject) -> Result<RQBuildParams> {
    let num_bits = env.get_u8_from_method(&rq_obj, "getNumBits")?;
    let rotation_type = extract_rq_rotation_type(env, &rq_obj)?;
    let rotation = env.get_optional_from_method(&rq_obj, "getModel", |env, model| {
        let bytes = env.get_vec_u8_from_method(&model, "toBytes")?;
        let metadata = parse_rq_model(&bytes)?;
        if metadata.num_bits != num_bits {
            return Err(Error::input_error(format!(
                "RQ model num_bits={} does not match requested num_bits={}",
                metadata.num_bits, num_bits
            )));
        }
        if metadata.rotation_type != rotation_type {
            return Err(Error::input_error(format!(
                "RQ model rotation_type={:?} does not match requested rotation_type={:?}",
                metadata.rotation_type, rotation_type
            )));
        }
        Ok(metadata)
    })?;
    Ok(RQBuildParams {
        num_bits,
        rotation_type,
        rotation,
    })
}

fn extract_rq_rotation_type(env: &mut JNIEnv, rq_obj: &JObject) -> Result<RQRotationType> {
    let rotation_type_obj = env
        .call_method(
            rq_obj,
            "getRotationType",
            "()Lorg/lance/index/vector/RQRotationType;",
            &[],
        )?
        .l()?;
    RQRotationType::from_str(&env.get_string_from_method(&rotation_type_obj, "toRustString")?)
        .map_err(|e| Error::input_error(e.to_string()))
}

pub(crate) fn parse_rq_model(bytes: &[u8]) -> Result<RabitQuantizationMetadata> {
    let model: RabitQuantizationMetadata = serde_json::from_slice(bytes)
        .map_err(|e| Error::input_error(format!("Invalid RQ model: {e}")))?;
    validate_rq_num_bits(model.num_bits).map_err(|e| Error::input_error(e.to_string()))?;
    let dimension = model.rotated_dim();
    if dimension == 0 || dimension > i32::MAX as usize || !dimension.is_multiple_of(8) {
        return Err(Error::input_error(format!(
            "RQ model dimension must be positive and divisible by 8, got {dimension}"
        )));
    }
    if model.rotation_type != RQRotationType::Fast {
        return Err(Error::input_error(format!(
            "RQ model rotation type {:?} cannot be restored from serialized bytes; expected Fast",
            model.rotation_type
        )));
    }
    model
        .validate_rotation()
        .map_err(|e| Error::input_error(e.to_string()))?;
    Ok(model)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_RQModel_nativeBuild<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    dimension: jint,
    num_bits: jbyte,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        inner_build(&mut env, dimension, num_bits).map(|v| v.into_raw()),
        JByteArray::default().into_raw()
    )
}

fn inner_build<'local>(
    env: &mut JNIEnv<'local>,
    dimension: jint,
    num_bits: jbyte,
) -> Result<JByteArray<'local>> {
    if dimension <= 0 || dimension % 8 != 0 {
        return Err(Error::input_error(format!(
            "IVF_RQ dimension must be positive and divisible by 8, got {dimension}"
        )));
    }
    validate_rq_num_bits(num_bits as u8).map_err(|e| Error::input_error(e.to_string()))?;
    let quantizer = RabitQuantizer::new_with_rotation::<Float32Type>(
        num_bits as u8,
        dimension,
        RQRotationType::Fast,
    );
    let bytes = serde_json::to_vec(&quantizer.metadata(None))?;
    Ok(env.byte_array_from_slice(&bytes)?)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_RQModel_nativeInspect<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    bytes: JByteArray<'local>,
) -> jintArray {
    ok_or_throw_with_return!(
        env,
        inner_inspect(&mut env, bytes).map(|v| v.into_raw()),
        std::ptr::null_mut()
    )
}

fn inner_inspect<'local>(
    env: &mut JNIEnv<'local>,
    bytes: JByteArray,
) -> Result<jni::objects::JIntArray<'local>> {
    let bytes = env.convert_byte_array(&bytes)?;
    let model = parse_rq_model(&bytes)?;
    let result = env.new_int_array(2)?;
    env.set_int_array_region(
        &result,
        0,
        &[model.rotated_dim() as jint, model.num_bits as jint],
    )?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rq_model_validates_rotation_payload() {
        let mut model =
            RabitQuantizer::new_with_rotation::<Float32Type>(3, 32, RQRotationType::Fast)
                .metadata(None);
        model.fast_rotation_signs.as_mut().unwrap().pop();
        let bytes = serde_json::to_vec(&model).unwrap();

        let err = parse_rq_model(&bytes).unwrap_err();
        let message = err.to_string();
        assert!(message.starts_with("java/lang/IllegalArgumentException:"));
        assert!(message.contains("fast_rotation_signs length"));
    }
}
