// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::block_on;
use crate::error::{Error, Result};
use crate::traits::{IntoJava, export_vec};
use crate::utils::to_java_map;
use arrow_schema::DataType;
use jni::JNIEnv;
use jni::objects::{JObject, JValue};
use jni::sys::jbyte;
use lance::dataset::Dataset;
use lance::table::format::IndexMetadata;
use lance_index::pb::VectorIndexDetails;
use lance_index::{IndexDescription, IndexType};
use prost::Message;
use prost_types::Any;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Resolve fallback `index_details` for index segments whose manifest entry has
/// `index_details = None`, without mutating the raw `IndexMetadata.index_details`
/// field (so `Index.indexDetails()` continues to reflect whether protobuf details
/// were stored in the manifest).
pub(crate) fn resolve_missing_index_details(
    dataset: &Dataset,
    indexes: &[IndexMetadata],
) -> HashMap<Uuid, Arc<Any>> {
    if !indexes.iter().any(|idx| {
        idx.index_details.is_none() && lance_index::infer_system_index_type(idx).is_none()
    }) {
        return HashMap::new();
    }

    block_on(async {
        let mut resolved = HashMap::new();
        for idx in indexes {
            if idx.index_details.is_some() || lance_index::infer_system_index_type(idx).is_some() {
                continue;
            }
            let Some(field) = idx
                .keyed_field()
                .and_then(|field_id| dataset.schema().field_by_id(field_id))
            else {
                continue;
            };
            if matches!(field.data_type(), DataType::FixedSizeList(_, _)) {
                if let Ok(details) = Any::from_msg(&VectorIndexDetails::default()) {
                    resolved.insert(idx.uuid, Arc::new(details));
                }
                continue;
            }
            let column = dataset
                .schema()
                .field_path(field.id)
                .unwrap_or_else(|_| field.name.clone());
            if let Ok(details) =
                lance::index::scalar::fetch_index_details(dataset, &column, idx).await
            {
                resolved.insert(idx.uuid, details);
            }
        }
        resolved
    })
}

/// Build a `java.util.List<Integer>`.
///
/// Not `JLance<Vec<i32>>`'s `IntoJava`: that produces a primitive `int[]`, while
/// the Java constructors here take `List`.
pub(crate) fn int_list<'a>(
    env: &mut JNIEnv<'a>,
    ids: impl IntoIterator<Item = i32>,
) -> Result<JObject<'a>> {
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

pub(crate) fn u64_to_jlong(field: &str, value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        Error::runtime_error(format!(
            "Cannot convert Rust index field {field}={value} to Java long"
        ))
    })
}

pub(crate) fn u32_to_jint(field: &str, value: u32) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        Error::runtime_error(format!(
            "Cannot convert Rust index field {field}={value} to Java int"
        ))
    })
}

pub(crate) fn index_description_to_java<'a>(
    description: &Arc<dyn IndexDescription>,
    env: &mut JNIEnv<'a>,
    include_metadata: bool,
    fallback_details: &HashMap<Uuid, Arc<Any>>,
) -> Result<JObject<'a>> {
    let field_ids = description
        .field_ids()
        .iter()
        .map(|id| u32_to_jint("fieldIds", *id))
        .collect::<Result<Vec<_>>>()?;
    let field_ids_list = int_list(env, field_ids)?;
    let name = env.new_string(description.name())?;
    let type_url = env.new_string(description.type_url())?;
    let index_type = env.new_string(description.index_type())?;
    let rows_indexed = u64_to_jlong("rowsIndexed", description.rows_indexed())?;
    let metadata_list = if include_metadata {
        let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;
        for meta in description.metadata() {
            let java_index = index_metadata_to_java(
                env,
                meta,
                fallback_details.get(&meta.uuid).map(Arc::as_ref),
            )?;
            env.call_method(
                &array_list,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&java_index)],
            )?;
        }
        array_list
    } else {
        env.call_static_method(
            "java/util/Collections",
            "emptyList",
            "()Ljava/util/List;",
            &[],
        )?
        .l()?
    };
    let details_json = description.details()?;
    let details = env.new_string(details_json)?;
    let total_size_bytes = if let Some(size) = description.total_size_bytes() {
        env.new_object(
            "java/lang/Long",
            "(J)V",
            &[JValue::Long(u64_to_jlong("totalSizeBytes", size)?)],
        )?
    } else {
        JObject::null()
    };
    let fragment_coverage = if let Some(coverage) = description.fragment_coverage() {
        let covered_fragment_count =
            u64_to_jlong("coveredFragmentCount", coverage.covered_fragment_count)?;
        let current_fragment_count =
            u64_to_jlong("currentFragmentCount", coverage.current_fragment_count)?;
        let missing_fragment_count =
            u64_to_jlong("missingFragmentCount", coverage.missing_fragment_count)?;
        let stale_fragment_count =
            u64_to_jlong("staleFragmentCount", coverage.stale_fragment_count)?;
        let fragment_bitmap_size_bytes = u64_to_jlong(
            "fragmentBitmapSizeBytes",
            coverage.fragment_bitmap_size_bytes,
        )?;
        env.new_object(
            "org/lance/index/IndexFragmentCoverage",
            "(JJJJJ)V",
            &[
                JValue::Long(covered_fragment_count),
                JValue::Long(current_fragment_count),
                JValue::Long(missing_fragment_count),
                JValue::Long(stale_fragment_count),
                JValue::Long(fragment_bitmap_size_bytes),
            ],
        )?
    } else {
        JObject::null()
    };

    let j_index_desc = env.new_object(
        "org/lance/index/IndexDescription",
        "(Ljava/lang/String;Ljava/util/List;Ljava/lang/String;Ljava/lang/String;JLjava/util/List;Ljava/lang/String;Ljava/lang/Long;Lorg/lance/index/IndexFragmentCoverage;)V",
        &[
            JValue::Object(&name),
            JValue::Object(&field_ids_list),
            JValue::Object(&type_url),
            JValue::Object(&index_type),
            JValue::Long(rows_indexed),
            JValue::Object(&metadata_list),
            JValue::Object(&details),
            JValue::Object(&total_size_bytes),
            JValue::Object(&fragment_coverage),
        ],
    )?;
    Ok(j_index_desc)
}

impl IntoJava for &Arc<dyn IndexDescription> {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        index_description_to_java(self, env, true, &HashMap::new())
    }
}

/// JNI conversion for the summary-only API, which deliberately omits segment
/// metadata so fragment bitmaps are never expanded into Java collections.
pub(crate) struct IndexDescriptionSummary<'a>(pub &'a Arc<dyn IndexDescription>);

impl IntoJava for &IndexDescriptionSummary<'_> {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        index_description_to_java(self.0, env, false, &HashMap::new())
    }
}

pub(crate) fn index_metadata_to_java<'a>(
    env: &mut JNIEnv<'a>,
    index_meta: &IndexMetadata,
    fallback_details: Option<&Any>,
) -> Result<JObject<'a>> {
    let uuid = index_meta.uuid.into_java(env)?;

    let fields = int_list(env, index_meta.fields.iter().copied())?;
    let covering_fields = int_list(env, index_meta.covering_fields.iter().copied())?;
    let name = env.new_string(&index_meta.name)?;

    let fragments = match &index_meta.fragment_bitmap {
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
    let index_details = if let Some(details) = &index_meta.index_details {
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
    let created_at = if let Some(dt) = &index_meta.created_at {
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
    let base_id = if let Some(id) = index_meta.base_id {
        env.new_object(
            "java/lang/Integer",
            "(I)V",
            &[JValue::Int(u32_to_jint("baseId", id)?)],
        )?
    } else {
        JObject::null()
    };

    let size_bytes = match &index_meta.files {
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
    let files = match &index_meta.files {
        Some(files) => export_vec(env, files)?,
        None => JObject::null(),
    };

    // Determine index type from index_details type_url, system index name, or fallback metadata
    let index_type = determine_index_type(env, index_meta, fallback_details)?;

    // A u32 id fits a Java long losslessly; `Integer` would fold the upper
    // half of the range into negative values.
    let clustering_id = match index_meta.clustering_id {
        Some(id) => env.new_object("java/lang/Long", "(J)V", &[JValue::Long(i64::from(id))])?,
        None => JObject::null(),
    };
    let metadata = to_java_map(env, &index_meta.metadata)?;

    // Create Index object
    Ok(env.new_object(
        "org/lance/index/Index",
        "(Ljava/util/UUID;Ljava/util/List;Ljava/util/List;Ljava/lang/String;JLjava/util/List;[BILjava/time/Instant;Ljava/lang/Integer;Ljava/lang/Long;Ljava/util/List;Lorg/lance/index/IndexType;Ljava/lang/Long;Ljava/util/Map;)V",
        &[
            JValue::Object(&uuid),
            JValue::Object(&fields),
            JValue::Object(&covering_fields),
            JValue::Object(&name),
            JValue::Long(u64_to_jlong("datasetVersion", index_meta.dataset_version)?),
            JValue::Object(&fragments),
            JValue::Object(&index_details),
            JValue::Int(index_meta.index_version),
            JValue::Object(&created_at),
            JValue::Object(&base_id),
            JValue::Object(&size_bytes),
            JValue::Object(&files),
            JValue::Object(&index_type),
            JValue::Object(&clustering_id),
            JValue::Object(&metadata),
        ],
    )?)
}

impl IntoJava for &IndexMetadata {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        index_metadata_to_java(env, self, None)
    }
}

/// Determine the IndexType enum value from system index name, index_details protobuf, or segment files
fn determine_index_type<'local>(
    env: &mut JNIEnv<'local>,
    index_meta: &IndexMetadata,
    fallback_details: Option<&Any>,
) -> Result<JObject<'local>> {
    match resolve_index_type_name(index_meta, fallback_details) {
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

fn resolve_index_type_name(
    index_meta: &IndexMetadata,
    fallback_details: Option<&Any>,
) -> Option<&'static str> {
    match lance_index::infer_system_index_type(index_meta) {
        Some(IndexType::FragmentReuse) => Some("FRAGMENT_REUSE"),
        Some(IndexType::MemWal) => Some("MEM_WAL"),
        _ => {
            if let Some(details) = index_meta.index_details.as_deref() {
                determine_index_type_from_type_url(&details.type_url)
            } else {
                infer_index_type_from_files(index_meta).or_else(|| {
                    fallback_details
                        .and_then(|details| determine_index_type_from_type_url(&details.type_url))
                })
            }
        }
    }
}

fn determine_index_type_from_type_url(type_url: &str) -> Option<&'static str> {
    // Extract type name from type_url (e.g., ".lance.index.BTreeIndexDetails" -> "BTREE")
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
    } else if lower.contains("minhashlsh") {
        Some("MINHASH_LSH")
    } else if lower.contains("fmindex") {
        Some("FM")
    } else if lower.contains("fragmentreuse") {
        Some("FRAGMENT_REUSE")
    } else if lower.contains("memwal") {
        Some("MEM_WAL")
    } else if lower.contains("json") {
        Some("SCALAR")
    } else if lower.contains("ivfhnsw") {
        if lower.contains("sq") {
            Some("IVF_HNSW_SQ")
        } else if lower.contains("pq") {
            Some("IVF_HNSW_PQ")
        } else if lower.contains("rq") {
            Some("IVF_HNSW_RQ")
        } else {
            Some("IVF_HNSW_FLAT")
        }
    } else if lower.contains("ivf") {
        if lower.contains("sq") {
            Some("IVF_SQ")
        } else if lower.contains("pq") {
            Some("IVF_PQ")
        } else if lower.contains("rq") {
            Some("IVF_RQ")
        } else {
            Some("IVF_FLAT")
        }
    } else if lower.contains("vector") {
        Some("VECTOR")
    } else {
        None
    }
}

fn infer_index_type_from_files(index_meta: &IndexMetadata) -> Option<&'static str> {
    let files = index_meta.files.as_ref()?;
    let has_file = |name: &str| files.iter().any(|f| f.path == name);

    if has_file("zonemap.lance") {
        Some("ZONEMAP")
    } else if has_file("bloomfilter.lance") {
        Some("BLOOM_FILTER")
    } else if has_file("bitmap_page_lookup.lance") {
        Some("BITMAP")
    } else if has_file("metadata.lance") || has_file("invert.lance") {
        Some("INVERTED")
    } else if has_file("ngram_postings.lance") {
        Some("NGRAM")
    } else if has_file("nulls.lance") && has_file("page_data.lance") {
        Some("RTREE")
    } else if has_file("page_lookup.lance") || has_file("page_data.lance") {
        Some("BTREE")
    } else if has_file("signatures.lance") || has_file("bands.lance") {
        Some("MINHASH_LSH")
    } else if files.iter().any(|f| f.path.ends_with("_fm.lance")) {
        Some("FM")
    } else if has_file(lance_index::INDEX_FILE_NAME) {
        Some("VECTOR")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance::table::format::IndexFile;
    use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
    use lance_index::mem_wal::MEM_WAL_INDEX_NAME;

    fn empty_index_metadata(name: &str) -> IndexMetadata {
        IndexMetadata {
            uuid: Uuid::new_v4(),
            name: name.to_string(),
            fields: vec![0],
            covering_fields: vec![],
            dataset_version: 1,
            fragment_bitmap: None,
            index_details: None,
            index_version: 0,
            created_at: None,
            base_id: None,
            files: None,
            clustering_id: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_resolve_index_type_name_system_indices() {
        let frag_reuse = empty_index_metadata(FRAG_REUSE_INDEX_NAME);
        assert_eq!(
            resolve_index_type_name(&frag_reuse, None),
            Some("FRAGMENT_REUSE")
        );

        let mem_wal = empty_index_metadata(MEM_WAL_INDEX_NAME);
        assert_eq!(resolve_index_type_name(&mem_wal, None), Some("MEM_WAL"));
    }

    #[test]
    fn test_resolve_index_type_name_fallback_and_files() {
        let mut meta = empty_index_metadata("idx");
        assert_eq!(resolve_index_type_name(&meta, None), None);

        let fallback = Any {
            type_url: "/lance.table.BitmapIndexDetails".to_string(),
            value: vec![],
        };
        assert_eq!(
            resolve_index_type_name(&meta, Some(&fallback)),
            Some("BITMAP")
        );

        meta.files = Some(vec![IndexFile {
            path: "zonemap.lance".to_string(),
            size_bytes: 128,
            file_metadata_size_bytes: None,
        }]);
        assert_eq!(resolve_index_type_name(&meta, None), Some("ZONEMAP"));
    }
}
