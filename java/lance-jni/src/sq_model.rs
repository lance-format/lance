// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{
    error::{Error, Result},
    ffi::JNIEnvExt,
};
use jni::{
    JNIEnv,
    objects::{JByteArray, JClass, JObject},
    sys::jintArray,
};
use lance::index::vector::sq::SqModel;
use lance_index::vector::sq::{builder::SQBuildParams, storage::ScalarQuantizationMetadata};
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct SqModelPayload {
    #[serde(flatten)]
    metadata: ScalarQuantizationMetadata,
    distance_type: String,
}

pub(crate) fn serialize_sq_model(model: &SqModel) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&SqModelPayload {
        metadata: model.metadata().clone(),
        distance_type: model.distance_type().to_string(),
    })?)
}

fn parse_sq_model(bytes: &[u8]) -> Result<SqModel> {
    let model: SqModelPayload = serde_json::from_slice(bytes)
        .map_err(|e| Error::input_error(format!("Invalid SQ model: {e}")))?;
    let distance_type = DistanceType::try_from(model.distance_type.as_str())
        .map_err(|e| Error::input_error(format!("Invalid SQ model distance_type: {e}")))?;
    Ok(SqModel::try_new(model.metadata, distance_type)?)
}

pub(crate) fn extract_sq_build_params(
    env: &mut JNIEnv,
    sq_obj: JObject,
) -> Result<(SQBuildParams, Option<SqModel>)> {
    let num_bits = env.call_method(&sq_obj, "getNumBits", "()S", &[])?.s()? as u16;
    let model = env.get_optional_from_method(&sq_obj, "getModel", |env, model| {
        parse_sq_model(&env.get_vec_u8_from_method(&model, "toBytes")?)
    })?;
    let sample_rate = if model.is_some() {
        0
    } else {
        let rate = env.call_method(&sq_obj, "getSampleRate", "()I", &[])?.i()?;
        usize::try_from(rate).map_err(|_| {
            Error::input_error(format!("SQ sampleRate must be non-negative, got {rate}"))
        })?
    };
    Ok((
        SQBuildParams {
            num_bits,
            sample_rate,
        },
        model,
    ))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_index_vector_SQModel_nativeInspect<'local>(
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
    let model = parse_sq_model(&env.convert_byte_array(&bytes)?)?;
    let model = model.metadata();
    let result = env.new_int_array(2)?;
    env.set_int_array_region(&result, 0, &[model.dim as i32, model.num_bits as i32])?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::JavaExceptionClass;
    use serde_json::json;

    #[test]
    fn test_sq_model_transfer() {
        let metadata = ScalarQuantizationMetadata {
            dim: 32,
            num_bits: 8,
            // Without float_roundtrip, JSON rounds this upper bound to 1.0.
            bounds: -0.123456789012345..1.0_f64.next_down(),
        };
        let model = SqModel::try_new(metadata.clone(), DistanceType::Cosine).unwrap();
        let restored = parse_sq_model(&serialize_sq_model(&model).unwrap()).unwrap();
        assert_eq!(restored.metadata(), &metadata);
        assert_eq!(restored.distance_type(), model.distance_type());
    }

    #[test]
    fn test_rejects_invalid_sq_models() {
        let payload = json!({
            "dim": 32,
            "num_bits": 8,
            "bounds": {"start": 0.0, "end": 1.0},
            "distance_type": "l2",
        });
        for (field, value, expected) in [
            ("dim", json!(0), "SQ model dimension"),
            ("num_bits", json!(65544), "u16"),
            (
                "bounds",
                json!({"start": 2.0, "end": 1.0}),
                "SQ model bounds",
            ),
            (
                "distance_type",
                json!("unknown"),
                "Invalid SQ model distance_type",
            ),
        ] {
            let mut invalid = payload.clone();
            invalid[field] = value;
            let err = parse_sq_model(&serde_json::to_vec(&invalid).unwrap()).unwrap_err();
            assert!(
                err.to_string()
                    .starts_with(JavaExceptionClass::IllegalArgumentException.as_str())
            );
            assert!(err.to_string().contains(expected), "{err}");
        }

        let err = parse_sq_model(b"not JSON").unwrap_err();
        assert!(
            err.to_string()
                .starts_with(JavaExceptionClass::IllegalArgumentException.as_str())
        );
        assert!(err.to_string().contains("Invalid SQ model"));
    }
}
