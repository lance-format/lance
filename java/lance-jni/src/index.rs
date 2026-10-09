// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::error::{Error, Result};
use crate::traits::{IntoJava, export_vec};
use jni::JNIEnv;
use jni::objects::{JObject, JValue};
use jni::sys::jbyte;
use lance::table::format::IndexMetadata;
use lance_index::IndexDescription;
use prost::Message;
use prost_types::Any;
use std::sync::Arc;

/// Build a `java.util.List<Integer>`.
///
/// Not `JLance<Vec<i32>>`'s `IntoJava`: that produces a primitive `int[]`, while
/// the Java constructors here take `List`.
fn int_list<'a>(env: &mut JNIEnv<'a>, ids: impl IntoIterator<Item = i32>) -> Result<JObject<'a>> {
    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for id in ids {
        let id_obj = env.new_object("java/lang/Integer", "(I)V", &[JValue::Int(id)])?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&id_obj)],
        )?;
    }
    Ok(array_list)
}

fn u64_to_jlong(field: &str, value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        Error::runtime_error(format!(
            "Cannot convert Rust index field {field}={value} to Java long"
        ))
    })
}

fn u32_to_jint(field: &str, value: u32) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        Error::runtime_error(format!(
            "Cannot convert Rust index field {field}={value} to Java int"
        ))
    })
}

impl IntoJava for &Arc<dyn IndexDescription> {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let field_ids_list = int_list(env, self.field_ids().iter().map(|id| *id as i32))?;
        let name = env.new_string(self.name())?;
        let type_url = env.new_string(self.type_url())?;
        let index_type = env.new_string(self.index_type())?;
        let rows_indexed = self.rows_indexed() as i64;
        let metadata_list = export_vec(env, self.metadata())?;
        let details_json = self.details()?;
        let details = env.new_string(details_json)?;
        let total_size_bytes = if let Some(size) = self.total_size_bytes() {
            env.new_object("java/lang/Long", "(J)V", &[JValue::Long(size as i64)])?
        } else {
            JObject::null()
        };

        let j_index_desc = env.new_object(
            "org/lance/index/IndexDescription",
            "(Ljava/lang/String;Ljava/util/List;Ljava/lang/String;Ljava/lang/String;JLjava/util/List;Ljava/lang/String;Ljava/lang/Long;)V",
            &[
                JValue::Object(&name),
                JValue::Object(&field_ids_list),
                JValue::Object(&type_url),
                JValue::Object(&index_type),
                JValue::Long(rows_indexed),
                JValue::Object(&metadata_list),
                JValue::Object(&details),
                JValue::Object(&total_size_bytes),
            ],
        )?;
        Ok(j_index_desc)
    }
}

impl IntoJava for &IndexMetadata {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let uuid = self.uuid.into_java(env)?;

        let fields = int_list(env, self.fields.iter().copied())?;
        let covering_fields = int_list(env, self.covering_fields.iter().copied())?;
        let name = env.new_string(&self.name)?;

        let fragments = match &self.fragment_bitmap {
            Some(bitmap) => {
                let ids = bitmap
                    .iter()
                    .map(|id| u32_to_jint("fragments", id))
                    .collect::<Result<Vec<_>>>()?;
                int_list(env, ids)?
            }
            None => JObject::null(),
        };

        // Convert index_details to byte array
        let index_details = if let Some(details) = &self.index_details {
            let bytes = details.encode_to_vec();
            let jbytes: &[jbyte] =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const jbyte, bytes.len()) };

            let byte_array = env.new_byte_array(bytes.len() as i32)?;
            env.set_byte_array_region(&byte_array, 0, jbytes)?;
            byte_array.into()
        } else {
            JObject::null()
        };

        // Convert created_at to Instant
        let created_at = if let Some(dt) = &self.created_at {
            let seconds = dt.timestamp();
            let nanos = dt.timestamp_subsec_nanos() as i64;
            env.call_static_method(
                "java/time/Instant",
                "ofEpochSecond",
                "(JJ)Ljava/time/Instant;",
                &[JValue::Long(seconds), JValue::Long(nanos)],
            )?
            .l()?
        } else {
            JObject::null()
        };

        // Convert base_id from Option<u32> to Integer for Java
        let base_id = if let Some(id) = self.base_id {
            env.new_object(
                "java/lang/Integer",
                "(I)V",
                &[JValue::Int(u32_to_jint("baseId", id)?)],
            )?
        } else {
            JObject::null()
        };

        let size_bytes = match &self.files {
            Some(files) => {
                let size = files.iter().try_fold(0_u64, |total, file| {
                    total.checked_add(file.size_bytes).ok_or_else(|| {
                        Error::runtime_error("Index file sizes overflow u64".to_string())
                    })
                })?;
                env.new_object(
                    "java/lang/Long",
                    "(J)V",
                    &[JValue::Long(u64_to_jlong("sizeBytes", size)?)],
                )?
            }
            None => JObject::null(),
        };
        let files = match &self.files {
            Some(files) => export_vec(env, files)?,
            None => JObject::null(),
        };

        // Determine index type from index_details type_url
        let index_type = determine_index_type(env, &self.index_details)?;

        // Create Index object
        Ok(env.new_object(
            "org/lance/index/Index",
            "(Ljava/util/UUID;Ljava/util/List;Ljava/util/List;Ljava/lang/String;JLjava/util/List;[BILjava/time/Instant;Ljava/lang/Integer;Ljava/lang/Long;Ljava/util/List;Lorg/lance/index/IndexType;)V",
            &[
                JValue::Object(&uuid),
                JValue::Object(&fields),
                JValue::Object(&covering_fields),
                JValue::Object(&name),
                JValue::Long(u64_to_jlong("datasetVersion", self.dataset_version)?),
                JValue::Object(&fragments),
                JValue::Object(&index_details),
                JValue::Int(self.index_version),
                JValue::Object(&created_at),
                JValue::Object(&base_id),
                JValue::Object(&size_bytes),
                JValue::Object(&files),
                JValue::Object(&index_type),
            ],
        )?)
    }
}

/// Determine the IndexType enum value from index_details protobuf
fn determine_index_type<'local>(
    env: &mut JNIEnv<'local>,
    index_details: &Option<Arc<Any>>,
) -> Result<JObject<'local>> {
    let type_name = if let Some(details) = index_details {
        // Extract type name from type_url (e.g., ".lance.index.BTreeIndexDetails" -> "BTREE")
        let type_url = &details.type_url;
        let type_part = type_url.split('.').next_back().unwrap_or("");
        let lower = type_part.to_lowercase();

        if lower.contains("btree") {
            Some("BTREE")
        } else if lower.contains("bitmap") {
            Some("BITMAP")
        } else if lower.contains("labellist") {
            Some("LABEL_LIST")
        } else if lower.contains("inverted") {
            Some("INVERTED")
        } else if lower.contains("ngram") {
            Some("NGRAM")
        } else if lower.contains("zonemap") {
            Some("ZONEMAP")
        } else if lower.contains("bloomfilter") {
            Some("BLOOM_FILTER")
        } else if lower.contains("rtree") {
            Some("RTREE")
        } else if lower.contains("ivfhnsw") {
            if lower.contains("sq") {
                Some("IVF_HNSW_SQ")
            } else if lower.contains("pq") {
                Some("IVF_HNSW_PQ")
            } else {
                Some("IVF_HNSW_FLAT")
            }
        } else if lower.contains("ivf") {
            if lower.contains("sq") {
                Some("IVF_SQ")
            } else if lower.contains("pq") {
                Some("IVF_PQ")
            } else {
                Some("IVF_FLAT")
            }
        } else if lower.contains("vector") {
            Some("VECTOR")
        } else {
            None
        }
    } else {
        None
    };

    match type_name {
        Some(name) => {
            let index_type = env
                .get_static_field(
                    "org/lance/index/IndexType",
                    name,
                    "Lorg/lance/index/IndexType;",
                )?
                .l()?;
            Ok(index_type)
        }
        None => Ok(JObject::null()),
    }
}
