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
use lance_index::{
    pb::VectorMetricType,
    vector::sq::{builder::SQBuildParams, storage::ScalarQuantizationMetadata},
};
use prost::Message;

// This message is a same-SDK transfer handle, not an on-disk index format.
#[derive(Clone, PartialEq, Message)]
struct SqModelProto {
    #[prost(uint32, tag = "1")]
    dimension: u32,
    #[prost(uint32, tag = "2")]
    num_bits: u32,
    #[prost(double, optional, tag = "3")]
    lower_bound: Option<f64>,
    #[prost(double, optional, tag = "4")]
    upper_bound: Option<f64>,
    #[prost(enumeration = "VectorMetricType", optional, tag = "5")]
    distance_type: Option<i32>,
}

pub(crate) fn serialize_sq_model(model: &SqModel) -> Vec<u8> {
    let metadata = model.metadata();
    SqModelProto {
        dimension: metadata.dim as u32,
        num_bits: metadata.num_bits as u32,
        lower_bound: Some(metadata.bounds.start),
        upper_bound: Some(metadata.bounds.end),
        distance_type: Some(VectorMetricType::from(model.distance_type()) as i32),
    }
    .encode_to_vec()
}

fn parse_sq_model(bytes: &[u8]) -> Result<SqModel> {
    let model = SqModelProto::decode(bytes)
        .map_err(|e| Error::input_error(format!("Invalid SQ model: {e}")))?;
    let metric = model
        .distance_type
        .and_then(|value| VectorMetricType::try_from(value).ok())
        .ok_or_else(|| {
            Error::input_error("SQ model has a missing or invalid distance type".into())
        })?;
    let num_bits = u16::try_from(model.num_bits).map_err(|_| {
        Error::input_error(format!(
            "SQ model num_bits must be 8, got {}",
            model.num_bits
        ))
    })?;
    let model = ScalarQuantizationMetadata {
        dim: model.dimension as usize,
        num_bits,
        bounds: model
            .lower_bound
            .ok_or_else(|| Error::input_error("SQ model lower_bound is missing".into()))?
            ..model
                .upper_bound
                .ok_or_else(|| Error::input_error("SQ model upper_bound is missing".into()))?,
    };
    Ok(SqModel::try_new(model, metric.into())?)
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

    #[test]
    fn test_sq_model_transfer() {
        let metadata = ScalarQuantizationMetadata {
            dim: 32,
            num_bits: 8,
            bounds: -0.123456789012345..9.87654321098765,
        };
        let model = SqModel::try_new(metadata.clone(), VectorMetricType::Cosine.into()).unwrap();
        let restored = parse_sq_model(&serialize_sq_model(&model)).unwrap();
        assert_eq!(restored.metadata(), &metadata);
        assert_eq!(restored.distance_type(), model.distance_type());
        for bytes in [
            vec![],
            vec![255],
            vec![8, 32, 16, 8],
            SqModelProto {
                dimension: 0,
                num_bits: 8,
                lower_bound: Some(0.0),
                upper_bound: Some(1.0),
                distance_type: Some(VectorMetricType::L2 as i32),
            }
            .encode_to_vec(),
            SqModelProto {
                dimension: 32,
                num_bits: 65544,
                lower_bound: Some(0.0),
                upper_bound: Some(1.0),
                distance_type: Some(VectorMetricType::L2 as i32),
            }
            .encode_to_vec(),
            SqModelProto {
                dimension: 32,
                num_bits: 8,
                lower_bound: Some(f64::NAN),
                upper_bound: Some(1.0),
                distance_type: Some(VectorMetricType::L2 as i32),
            }
            .encode_to_vec(),
        ] {
            let err = parse_sq_model(&bytes).unwrap_err();
            assert!(
                err.to_string()
                    .starts_with("java/lang/IllegalArgumentException:")
            );
        }
    }
}
