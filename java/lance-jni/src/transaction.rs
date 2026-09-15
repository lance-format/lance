// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Error;
use crate::JNIEnvExt;
use crate::block_on;
use crate::blocking_dataset::{BlockingDataset, NATIVE_DATASET, extract_namespace_info};
use crate::error::Result;
use crate::traits::{
    FromJObjectWithEnv, FromJString, IntoJava, JLance, export_vec, import_vec_from_method,
};
use crate::utils::{to_java_map, to_rust_map};
use arrow::datatypes::Schema;
use arrow_schema::ffi::FFI_ArrowSchema;
use chrono::DateTime;
use jni::JNIEnv;
use jni::objects::{JByteArray, JIntArray, JLongArray, JMap, JObject, JString, JValue, JValueGen};
use jni::sys::{jboolean, jint, jlong};
use lance::dataset::CommitBuilder;
use lance::dataset::transaction::{
    DataOverlayGroup, DataReplacementGroup, Operation, RewriteGroup, RewrittenIndex, Transaction,
    TransactionBuilder, UpdateMap, UpdateMapEntry, UpdateMode, UpdatedFragmentOffsets,
};
use lance::io::ObjectStoreParams;
use lance::io::commit::namespace_manifest::LanceNamespaceExternalManifestStore;
use lance::table::format::key_existence::{FilterType, KeyExistenceFilter};
use lance::table::format::overlay::{DataOverlayFile, OverlayCoverage};
use lance::table::format::{DataFile, Fragment, IndexFile, IndexMetadata};
use lance_core::datatypes::Field;
use lance_core::datatypes::Schema as LanceSchema;
use lance_file::version::{LanceFileVersion, V2_FORMAT_2_0, V2_FORMAT_2_1, V2_FORMAT_2_2};
use lance_io::object_store::{LanceNamespaceStorageOptionsProvider, StorageOptionsProvider};
use lance_table::io::commit::CommitHandler;
use lance_table::io::commit::external_manifest::ExternalManifestCommitHandler;
use prost::Message;
use prost_types::Any;
use roaring::RoaringBitmap;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

fn u64_to_jlong(field: &str, value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| {
        Error::runtime_error(format!(
            "Cannot convert Rust transaction field {field}={value} to Java long"
        ))
    })
}

fn u32_to_jint(field: &str, value: u32) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        Error::runtime_error(format!(
            "Cannot convert Rust transaction field {field}={value} to Java int"
        ))
    })
}

fn checked_field_ids(field: &str, values: &[i64]) -> Result<Vec<u32>> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            u32::try_from(*value).map_err(|_| {
                Error::input_error(format!(
                    "Java transaction field {field}[{index}] must be between 0 and {}, got {value}",
                    u32::MAX
                ))
            })
        })
        .collect()
}

fn import_field_ids(
    env: &mut JNIEnv<'_>,
    object: &JObject<'_>,
    method: &str,
    field: &str,
) -> Result<Vec<u32>> {
    let array = env.call_method(object, method, "()[J", &[])?.l()?;
    let array = JLongArray::from(array);
    let mut values = vec![0_i64; env.get_array_length(&array)? as usize];
    env.get_long_array_region(&array, 0, &mut values)?;
    checked_field_ids(field, &values)
}

fn nonnegative_jlong_to_u64(field: &str, value: i64) -> Result<u64> {
    u64::try_from(value).map_err(|_| {
        Error::input_error(format!(
            "Java transaction field {field} must be non-negative, got {value}"
        ))
    })
}

fn import_unsigned_longs(
    env: &mut JNIEnv<'_>,
    object: &JObject<'_>,
    method: &str,
    field: &str,
) -> Result<Vec<u64>> {
    let index = Cell::new(0_usize);
    import_vec_from_method(env, object, method, |env, value| {
        let position = index.get();
        index.set(position + 1);
        nonnegative_jlong_to_u64(
            &format!("{field}[{position}]"),
            env.call_method(value, "longValue", "()J", &[])?.j()?,
        )
    })
}

fn export_unsigned_longs<'a>(
    env: &mut JNIEnv<'a>,
    values: &[u64],
    field: &str,
) -> Result<JObject<'a>> {
    let values = values
        .iter()
        .enumerate()
        .map(|(index, value)| u64_to_jlong(&format!("{field}[{index}]"), *value).map(JLance))
        .collect::<Result<Vec<_>>>()?;
    export_vec(env, &values)
}

impl IntoJava for &IndexFile {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let path = env.new_string(&self.path)?;
        Ok(env.new_object(
            "org/lance/index/IndexFile",
            "(Ljava/lang/String;J)V",
            &[
                JValue::Object(&path),
                JValue::Long(u64_to_jlong("newIndexFiles.sizeBytes", self.size_bytes)?),
            ],
        )?)
    }
}

impl FromJObjectWithEnv<IndexFile> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<IndexFile> {
        Ok(IndexFile {
            path: env.get_string_from_method(self, "getPath")?,
            size_bytes: nonnegative_jlong_to_u64(
                "newIndexFiles.sizeBytes",
                env.call_method(self, "getSizeBytes", "()J", &[])?.j()?,
            )?,
        })
    }
}

fn compacted_sstable_into_java<'a>(
    env: &mut JNIEnv<'a>,
    sstable: &lance_index::mem_wal::CompactedSsTable,
) -> Result<JObject<'a>> {
    let shard_id = env.new_string(sstable.shard_id.to_string())?;
    Ok(env.new_object(
        "org/lance/memwal/CompactedSsTable",
        "(Ljava/lang/String;J)V",
        &[
            JValue::Object(&shard_id),
            JValue::Long(u64_to_jlong(
                "compactedSstables.generation",
                sstable.generation,
            )?),
        ],
    )?)
}

fn compacted_sstable_from_java(
    env: &mut JNIEnv<'_>,
    object: &JObject<'_>,
) -> Result<lance_index::mem_wal::CompactedSsTable> {
    let shard_id = env.get_string_from_method(object, "getShardId")?;
    let shard_id = Uuid::parse_str(&shard_id).map_err(|e| {
        Error::input_error(format!(
            "Invalid compacted SSTable shardId '{shard_id}': {e}"
        ))
    })?;
    let generation = nonnegative_jlong_to_u64(
        "compactedSstables.generation",
        env.call_method(object, "getGeneration", "()J", &[])?.j()?,
    )?;
    Ok(lance_index::mem_wal::CompactedSsTable::new(
        shard_id, generation,
    ))
}

fn export_compacted_sstables<'a>(
    env: &mut JNIEnv<'a>,
    sstables: &[lance_index::mem_wal::CompactedSsTable],
) -> Result<JObject<'a>> {
    let list = env.new_object("java/util/ArrayList", "()V", &[])?;
    for sstable in sstables {
        let object = compacted_sstable_into_java(env, sstable)?;
        env.call_method(
            &list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&object)],
        )?;
    }
    Ok(list)
}

impl IntoJava for &RewriteGroup {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let old_fragments = export_vec(env, &self.old_fragments)?;
        let new_fragments = export_vec(env, &self.new_fragments)?;

        Ok(env.new_object(
            "org/lance/operation/RewriteGroup",
            "(Ljava/util/List;Ljava/util/List;)V",
            &[
                JValue::Object(&old_fragments),
                JValue::Object(&new_fragments),
            ],
        )?)
    }
}

impl IntoJava for &RewrittenIndex {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let old_id = self.old_id.into_java(env)?;
        let new_id = self.new_id.into_java(env)?;

        let new_index_details_type_url = env.new_string(self.new_index_details.type_url.clone())?;
        let new_index_details_value = env.byte_array_from_slice(&self.new_index_details.value)?;
        let new_index_files = match &self.new_index_files {
            Some(files) => export_vec(env, files)?,
            None => JObject::null(),
        };

        Ok(env.new_object(
            "org/lance/operation/RewrittenIndex",
            "(Ljava/util/UUID;Ljava/util/UUID;Ljava/lang/String;[BILjava/util/List;)V",
            &[
                JValue::Object(&old_id),
                JValue::Object(&new_id),
                JValue::Object(&new_index_details_type_url),
                JValue::Object(&new_index_details_value),
                JValue::Int(i32::try_from(self.new_index_version).map_err(|_| {
                    Error::runtime_error(format!(
                        "Cannot convert Rust transaction field newIndexVersion={} to Java int",
                        self.new_index_version
                    ))
                })?),
                JValue::Object(&new_index_files),
            ],
        )?)
    }
}

impl IntoJava for &DataReplacementGroup {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let fragment_id = self.0;
        let new_file = self.1.into_java(env)?;

        Ok(env.new_object(
            "org/lance/operation/DataReplacement$DataReplacementGroup",
            "(JLorg/lance/fragment/DataFile;)V",
            &[
                JValue::Long(u64_to_jlong("dataReplacement.fragmentId", fragment_id)?),
                JValue::Object(&new_file),
            ],
        )?)
    }
}

impl IntoJava for &UpdateMode {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let name = match self {
            UpdateMode::RewriteRows => "RewriteRows",
            UpdateMode::RewriteColumns => "RewriteColumns",
        };
        let update_mode_type_class = "org/lance/operation/Update$UpdateMode";
        env.get_static_field(
            update_mode_type_class,
            name,
            format!("L{};", update_mode_type_class),
        )?
        .l()
        .map_err(|e| {
            Error::runtime_error(format!("failed to get {}: {}", update_mode_type_class, e))
        })
    }
}

impl FromJObjectWithEnv<RewriteGroup> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<RewriteGroup> {
        let old_fragments: Vec<Fragment> =
            import_vec_from_method(env, self, "oldFragments", |env, fragment| {
                fragment.extract_object(env)
            })?;
        let new_fragments: Vec<Fragment> =
            import_vec_from_method(env, self, "newFragments", |env, fragment| {
                fragment.extract_object(env)
            })?;
        Ok(RewriteGroup {
            old_fragments,
            new_fragments,
        })
    }
}

impl FromJObjectWithEnv<RewrittenIndex> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<RewrittenIndex> {
        let java_old_id = env.get_field(self, "oldId", "Ljava/util/UUID;")?.l()?;
        let java_new_id = env.get_field(self, "newId", "Ljava/util/UUID;")?.l()?;
        let java_old_id = java_old_id.extract_object(env)?;
        let java_new_id = java_new_id.extract_object(env)?;

        let new_index_details_type_url = env
            .get_field(self, "newIndexDetailsTypeUrl", "Ljava/lang/String;")?
            .l()?;
        let new_index_details_type_url: String = env
            .get_string(&JString::from(new_index_details_type_url))?
            .to_str()?
            .to_string();

        let new_index_details_value = env.get_field(self, "newIndexDetailsValue", "[B")?.l()?;
        let new_index_details_value =
            env.convert_byte_array(JByteArray::from(new_index_details_value))?;

        let new_index_version = env.get_field(self, "newIndexVersion", "I")?.i()?;
        let new_index_files =
            env.get_optional_from_method(self, "getNewIndexFiles", |env, files| {
                crate::traits::import_vec_to_rust(env, &files, |env, file| file.extract_object(env))
            })?;
        Ok(RewrittenIndex {
            old_id: java_old_id,
            new_id: java_new_id,
            new_index_details: prost_types::Any {
                type_url: new_index_details_type_url,
                value: new_index_details_value,
            },
            new_index_version: u32::try_from(new_index_version).map_err(|_| {
                Error::input_error(format!(
                    "Java transaction field newIndexVersion must be non-negative, got {new_index_version}"
                ))
            })?,
            new_index_files,
        })
    }
}

fn key_existence_filter_into_java<'a>(
    env: &mut JNIEnv<'a>,
    filter: &KeyExistenceFilter,
) -> Result<JObject<'a>> {
    let field_ids = JLance(filter.field_ids.clone()).into_java(env)?;
    match &filter.filter {
        FilterType::ExactSet(hashes) => {
            let hashes = JLance(hashes.iter().map(|hash| *hash as i64).collect::<Vec<_>>())
                .into_java(env)?;
            env.call_static_method(
                "org/lance/operation/KeyExistenceFilter",
                "exact",
                "([I[J)Lorg/lance/operation/KeyExistenceFilter;",
                &[JValue::Object(&field_ids), JValue::Object(&hashes)],
            )?
            .l()
            .map_err(Into::into)
        }
        FilterType::Bloom {
            bitmap,
            num_bits,
            number_of_items,
            probability,
        } => {
            let bitmap = env.byte_array_from_slice(bitmap)?;
            env.call_static_method(
                "org/lance/operation/KeyExistenceFilter",
                "bloom",
                "([I[BIJD)Lorg/lance/operation/KeyExistenceFilter;",
                &[
                    JValue::Object(&field_ids),
                    JValue::Object(&bitmap),
                    JValue::Int(i32::try_from(*num_bits).map_err(|_| {
                        Error::runtime_error(format!(
                            "Cannot convert Rust transaction field insertedRowsFilter.numBits={num_bits} to Java int"
                        ))
                    })?),
                    JValue::Long(u64_to_jlong(
                        "insertedRowsFilter.numberOfItems",
                        *number_of_items,
                    )?),
                    JValue::Double(*probability),
                ],
            )?
            .l()
            .map_err(Into::into)
        }
    }
}

fn key_existence_filter_from_java(
    env: &mut JNIEnv<'_>,
    object: &JObject<'_>,
) -> Result<KeyExistenceFilter> {
    let field_ids = env.call_method(object, "getFieldIds", "()[I", &[])?.l()?;
    let field_ids = JIntArray::from(field_ids).extract_object(env)?;
    let filter_type = env
        .call_method(
            object,
            "getType",
            "()Lorg/lance/operation/KeyExistenceFilter$Type;",
            &[],
        )?
        .l()?;
    let filter_type = env.get_string_from_method(&filter_type, "name")?;
    let filter = match filter_type.as_str() {
        "EXACT" => {
            let hashes = env
                .call_method(object, "getExactKeyHashes", "()[J", &[])?
                .l()?;
            let hashes = JLongArray::from(hashes);
            let len = env.get_array_length(&hashes)?;
            let mut values = vec![0_i64; len as usize];
            env.get_long_array_region(&hashes, 0, &mut values)?;
            FilterType::ExactSet(values.into_iter().map(|value| value as u64).collect())
        }
        "BLOOM" => {
            let bitmap = env
                .call_method(object, "getBloomBitmap", "()[B", &[])?
                .l()?;
            let bitmap = env.convert_byte_array(JByteArray::from(bitmap))?;
            let num_bits = u32::try_from(
                env.call_method(object, "getBloomNumBits", "()I", &[])?
                    .i()?,
            )
            .map_err(|_| {
                Error::input_error("insertedRowsFilter.numBits must be positive".to_string())
            })?;
            let number_of_items = nonnegative_jlong_to_u64(
                "insertedRowsFilter.numberOfItems",
                env.call_method(object, "getBloomNumberOfItems", "()J", &[])?
                    .j()?,
            )?;
            let probability = env
                .call_method(object, "getBloomProbability", "()D", &[])?
                .d()?;
            let bitmap_bits = u32::try_from(bitmap.len())
                .ok()
                .and_then(|len| len.checked_mul(8))
                .ok_or_else(|| {
                    Error::input_error(
                        "insertedRowsFilter.bitmap is too large to represent".to_string(),
                    )
                })?;
            if num_bits == 0 || num_bits != bitmap_bits {
                return Err(Error::input_error(format!(
                    "insertedRowsFilter.numBits={num_bits} must equal bitmap length in bits ({bitmap_bits})"
                )));
            }
            if number_of_items == 0 {
                return Err(Error::input_error(
                    "insertedRowsFilter.numberOfItems must be positive".to_string(),
                ));
            }
            if !probability.is_finite() || !(0.0..1.0).contains(&probability) || probability == 0.0
            {
                return Err(Error::input_error(format!(
                    "insertedRowsFilter.probability must be finite and between 0 and 1, got {probability}"
                )));
            }
            FilterType::Bloom {
                bitmap,
                num_bits,
                number_of_items,
                probability,
            }
        }
        other => {
            return Err(Error::input_error(format!(
                "Unknown KeyExistenceFilter.Type: {other}"
            )));
        }
    };
    Ok(KeyExistenceFilter { field_ids, filter })
}

fn serialize_bitmap(bitmap: &RoaringBitmap) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(bitmap.serialized_size());
    bitmap.serialize_into(&mut bytes).map_err(|e| {
        Error::runtime_error(format!("failed to serialize overlay coverage bitmap: {e}"))
    })?;
    Ok(bytes)
}

impl IntoJava for &DataOverlayFile {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let data_file = self.data_file.into_java(env)?;
        let coverage = match &self.coverage {
            OverlayCoverage::Shared(bitmap) => {
                let bytes = serialize_bitmap(bitmap)?;
                let bytes = env.byte_array_from_slice(&bytes)?;
                env.call_static_method(
                    "org/lance/operation/DataOverlay$OverlayCoverage",
                    "shared",
                    "([B)Lorg/lance/operation/DataOverlay$OverlayCoverage;",
                    &[JValue::Object(&bytes)],
                )?
                .l()?
            }
            OverlayCoverage::PerField(bitmaps) => {
                let list = env.new_object("java/util/ArrayList", "()V", &[])?;
                for bitmap in bitmaps {
                    let bytes = serialize_bitmap(bitmap)?;
                    let bytes = env.byte_array_from_slice(&bytes)?;
                    env.call_method(
                        &list,
                        "add",
                        "(Ljava/lang/Object;)Z",
                        &[JValue::Object(&bytes)],
                    )?;
                }
                env.call_static_method(
                    "org/lance/operation/DataOverlay$OverlayCoverage",
                    "perField",
                    "(Ljava/util/List;)Lorg/lance/operation/DataOverlay$OverlayCoverage;",
                    &[JValue::Object(&list)],
                )?
                .l()?
            }
        };
        Ok(env.new_object(
            "org/lance/operation/DataOverlay$DataOverlayFile",
            "(Lorg/lance/fragment/DataFile;Lorg/lance/operation/DataOverlay$OverlayCoverage;J)V",
            &[
                JValue::Object(&data_file),
                JValue::Object(&coverage),
                JValue::Long(u64_to_jlong(
                    "dataOverlay.committedVersion",
                    self.committed_version,
                )?),
            ],
        )?)
    }
}

impl FromJObjectWithEnv<DataOverlayFile> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<DataOverlayFile> {
        let data_file: DataFile = env
            .call_method(self, "getDataFile", "()Lorg/lance/fragment/DataFile;", &[])?
            .l()?
            .extract_object(env)?;
        let coverage = env
            .call_method(
                self,
                "getCoverage",
                "()Lorg/lance/operation/DataOverlay$OverlayCoverage;",
                &[],
            )?
            .l()?;
        let is_shared = env.call_method(&coverage, "isShared", "()Z", &[])?.z()?;
        let bitmap_bytes: Vec<Vec<u8>> =
            import_vec_from_method(env, &coverage, "getBitmaps", |env, bytes| {
                env.convert_byte_array(JByteArray::from(bytes))
                    .map_err(Into::into)
            })?;
        let mut bitmaps = Vec::with_capacity(bitmap_bytes.len());
        for (position, bytes) in bitmap_bytes.into_iter().enumerate() {
            bitmaps.push(
                RoaringBitmap::deserialize_from(bytes.as_slice()).map_err(|e| {
                    Error::input_error(format!(
                        "invalid overlay coverage RoaringBitmap at position {position}: {e}"
                    ))
                })?,
            );
        }
        let coverage = if is_shared {
            let [bitmap]: [RoaringBitmap; 1] = bitmaps.try_into().map_err(|bitmaps: Vec<_>| {
                Error::input_error(format!(
                    "shared overlay coverage requires exactly one bitmap, got {}",
                    bitmaps.len()
                ))
            })?;
            OverlayCoverage::dense(bitmap)
        } else {
            if bitmaps.len() != data_file.fields.len() {
                return Err(Error::input_error(format!(
                    "per-field overlay coverage for {} has {} bitmaps but the data file has {} fields",
                    data_file.path,
                    bitmaps.len(),
                    data_file.fields.len()
                )));
            }
            OverlayCoverage::sparse(bitmaps)
        };
        Ok(DataOverlayFile {
            data_file,
            coverage,
            committed_version: nonnegative_jlong_to_u64(
                "dataOverlay.committedVersion",
                env.call_method(self, "getCommittedVersion", "()J", &[])?
                    .j()?,
            )?,
        })
    }
}

impl IntoJava for &DataOverlayGroup {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let overlays = export_vec(env, &self.overlays)?;
        Ok(env.new_object(
            "org/lance/operation/DataOverlay$DataOverlayGroup",
            "(JLjava/util/List;)V",
            &[
                JValue::Long(u64_to_jlong("dataOverlay.fragmentId", self.fragment_id)?),
                JValue::Object(&overlays),
            ],
        )?)
    }
}

impl FromJObjectWithEnv<DataOverlayGroup> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<DataOverlayGroup> {
        Ok(DataOverlayGroup {
            fragment_id: nonnegative_jlong_to_u64(
                "dataOverlay.fragmentId",
                env.call_method(self, "getFragmentId", "()J", &[])?.j()?,
            )?,
            overlays: import_vec_from_method(env, self, "getOverlays", |env, overlay| {
                overlay.extract_object(env)
            })?,
        })
    }
}

impl FromJObjectWithEnv<IndexMetadata> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<IndexMetadata> {
        let uuid = env
            .get_field(self, "uuid", "Ljava/util/UUID;")?
            .l()?
            .extract_object(env)?;

        let fields: Vec<i32> = import_vec_from_method(env, self, "fields", |env, field_id| {
            field_id.extract_object(env)
        })?;
        let covering_fields: Vec<i32> =
            import_vec_from_method(env, self, "coveringFields", |env, field_id| {
                field_id.extract_object(env)
            })?;

        let name = env.get_string_from_method(self, "name")?;
        let dataset_version = nonnegative_jlong_to_u64(
            "index.datasetVersion",
            env.get_field(self, "datasetVersion", "J")?.j()?,
        )?;

        let fragment_bitmap: Option<RoaringBitmap> =
            env.get_optional_from_method(self, "fragments", |env, fragments_obj| {
                let frag_ids = env.get_integers(&fragments_obj)?;
                let bitmap = frag_ids
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        u32::try_from(*value).map_err(|_| {
                            Error::input_error(format!(
                                "index.fragments[{index}] must be non-negative, got {value}"
                            ))
                        })
                    })
                    .collect::<Result<RoaringBitmap>>()?;
                Ok(bitmap)
            })?;

        let index_details: Option<Arc<Any>> =
            env.get_optional_from_method(self, "indexDetails", |env, details_obj| {
                let byte_array: JByteArray = details_obj.into();
                let bytes = env.convert_byte_array(&byte_array)?;
                let any = Any::decode(&bytes[..]).map_err(|e| {
                    Error::input_error(format!("Invalid index_details data: {}", e))
                })?;
                Ok(Arc::new(any))
            })?;

        let index_version = env.get_field(self, "indexVersion", "I")?.i()?;
        let created_at =
            env.get_optional_from_method(self, "createdAt", |env, created_at_obj| {
                let seconds = env
                    .call_method(&created_at_obj, "getEpochSecond", "()J", &[])?
                    .j()?;
                let nanos = env
                    .call_method(&created_at_obj, "getNano", "()I", &[])?
                    .i()? as u32;
                DateTime::from_timestamp(seconds, nanos).ok_or_else(|| {
                    Error::input_error(format!(
                        "Invalid index createdAt timestamp: seconds={seconds}, nanos={nanos}"
                    ))
                })
            })?;
        let base_id = env.get_optional_u32_from_method(self, "baseId")?;
        let files: Option<Vec<IndexFile>> =
            env.get_optional_from_method(self, "getFiles", |env, files| {
                crate::traits::import_vec_to_rust(env, &files, |env, file| file.extract_object(env))
            })?;
        let files_size = files
            .as_ref()
            .map(|files| {
                files.iter().try_fold(0_u64, |total, file| {
                    total.checked_add(file.size_bytes).ok_or_else(|| {
                        Error::input_error("index file sizes overflow u64".to_string())
                    })
                })
            })
            .transpose()?;
        if let Some(files_size) = files_size {
            i64::try_from(files_size).map_err(|_| {
                Error::input_error(format!(
                    "index file sizes total {files_size}, which cannot be represented by Java long"
                ))
            })?;
        }
        if let Some(size_bytes) = env.get_optional_u64_from_method(self, "getSizeBytes")? {
            let files_size = files_size.ok_or_else(|| {
                Error::input_error(format!(
                    "index sizeBytes={size_bytes} cannot be represented without files"
                ))
            })?;
            if size_bytes != files_size {
                return Err(Error::input_error(format!(
                    "index sizeBytes={size_bytes} does not match the sum of file sizes ({files_size})"
                )));
            }
        }

        Ok(IndexMetadata {
            uuid,
            fields,
            covering_fields,
            name,
            dataset_version,
            fragment_bitmap,
            index_details,
            index_version,
            created_at,
            base_id,
            files,
        })
    }
}

impl FromJObjectWithEnv<DataReplacementGroup> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<DataReplacementGroup> {
        let fragment_id = nonnegative_jlong_to_u64(
            "dataReplacement.fragmentId",
            env.call_method(self, "fragmentId", "()J", &[])?.j()?,
        )?;
        let new_file = env
            .call_method(self, "replacedFile", "()Lorg/lance/fragment/DataFile;", &[])?
            .l()?
            .extract_object(env)?;

        Ok(DataReplacementGroup(fragment_id, new_file))
    }
}

impl FromJObjectWithEnv<UpdateMode> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<UpdateMode> {
        let s = env
            .call_method(self, "toString", "()Ljava/lang/String;", &[])?
            .l()?;
        let s: String = env.get_string(&JString::from(s))?.into();
        let t = match s.as_str() {
            "RewriteRows" => UpdateMode::RewriteRows,
            "RewriteColumns" => UpdateMode::RewriteColumns,
            _ => {
                return Err(Error::input_error(format!("Unknown UpdateMode value: {s}")));
            }
        };
        Ok(t)
    }
}

impl IntoJava for Uuid {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let uuid_string = env.new_string(self.to_string())?;
        let uuid_class = env.find_class("java/util/UUID")?;

        env.call_static_method(
            uuid_class,
            "fromString",
            "(Ljava/lang/String;)Ljava/util/UUID;",
            &[JValue::Object(&uuid_string)],
        )?
        .l()
        .map_err(Into::into)
    }
}

impl FromJObjectWithEnv<Uuid> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<Uuid> {
        let uuid_string = env
            .call_method(self, "toString", "()Ljava/lang/String;", &[])?
            .l()?;
        let uuid_string = JString::from(uuid_string);
        let uuid_string: String = env.get_string(&uuid_string)?.into();
        let uuid = Uuid::parse_str(uuid_string.to_string().as_str()).map_err(|e| {
            Error::input_error(format!(
                "Invalid UUID string: {}, error: {}",
                uuid_string, e
            ))
        })?;
        Ok(uuid)
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeReadTransaction<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_read_transaction(&mut env, java_dataset))
}

fn inner_read_transaction<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let transaction = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        dataset_guard.read_transaction()?
    };

    let transaction = match transaction {
        Some(transaction) => convert_to_java_transaction(env, transaction)?,
        None => JObject::null(),
    };
    Ok(transaction)
}

pub(crate) fn convert_to_java_transaction<'local>(
    env: &mut JNIEnv<'local>,
    transaction: Transaction,
) -> Result<JObject<'local>> {
    let uuid = env.new_string(transaction.uuid)?;
    let tag = match transaction.tag {
        Some(tag) => JObject::from(env.new_string(tag)?),
        None => JObject::null(),
    };
    let transaction_properties = match transaction.transaction_properties {
        Some(properties) => to_java_map(env, &properties)?,
        _ => JObject::null(),
    };
    let operation = convert_to_java_operation(env, Some(transaction.operation))?;

    let java_transaction = env.new_object(
        "org/lance/Transaction",
        "(JLjava/lang/String;Lorg/lance/operation/Operation;Ljava/lang/String;Ljava/util/Map;)V",
        &[
            JValue::Long(u64_to_jlong(
                "transaction.readVersion",
                transaction.read_version,
            )?),
            JValue::Object(&uuid),
            JValue::Object(&operation),
            JValue::Object(&tag),
            JValue::Object(&transaction_properties),
        ],
    )?;
    Ok(java_transaction)
}

pub(crate) fn convert_to_java_operation<'local>(
    env: &mut JNIEnv<'local>,
    operation: Option<Operation>,
) -> Result<JObject<'local>> {
    let operation = match operation {
        Some(operation) => convert_to_java_operation_inner(env, operation)?,
        None => JObject::null(),
    };
    Ok(operation)
}

fn convert_to_java_operation_inner<'local>(
    env: &mut JNIEnv<'local>,
    operation: Operation,
) -> Result<JObject<'local>> {
    match operation {
        Operation::Append {
            fragments: rust_fragments,
        } => {
            let java_fragments = export_vec(env, &rust_fragments)?;

            Ok(env.new_object(
                "org/lance/operation/Append",
                "(Ljava/util/List;)V",
                &[JValue::Object(&java_fragments)],
            )?)
        }
        Operation::Delete {
            updated_fragments,
            deleted_fragment_ids,
            predicate,
        } => {
            let updated_fragments_obj = export_vec(env, &updated_fragments)?;

            let removed_fragment_ids_obj =
                export_unsigned_longs(env, &deleted_fragment_ids, "delete.deletedFragmentIds")?;

            let predicate_obj = env.new_string(&predicate)?;

            Ok(env.new_object(
                "org/lance/operation/Delete",
                "(Ljava/util/List;Ljava/util/List;Ljava/lang/String;)V",
                &[
                    JValue::Object(&updated_fragments_obj),
                    JValue::Object(&removed_fragment_ids_obj),
                    JValue::Object(&predicate_obj),
                ],
            )?)
        }
        Operation::Overwrite {
            fragments: rust_fragments,
            schema,
            config_upsert_values,
            initial_bases,
        } => {
            let java_fragments = export_vec(env, &rust_fragments)?;
            let java_schema = convert_to_java_schema(env, schema)?;
            let java_config = match config_upsert_values {
                Some(config_upsert_values) => to_java_map(env, &config_upsert_values)?,
                _ => JObject::null(),
            };
            let java_initial_bases = match initial_bases {
                Some(initial_bases) => export_vec(env, &initial_bases)?,
                None => JObject::null(),
            };

            Ok(env.new_object(
                "org/lance/operation/Overwrite",
                "(Ljava/util/List;Lorg/apache/arrow/vector/types/pojo/Schema;Ljava/util/Map;Ljava/util/List;)V",
                &[
                    JValue::Object(&java_fragments),
                    JValue::Object(&java_schema),
                    JValue::Object(&java_config),
                    JValue::Object(&java_initial_bases),
                ],
            )?)
        }
        Operation::CreateIndex {
            new_indices,
            removed_indices,
        } => {
            let java_new_indices = export_vec(env, &new_indices)?;
            let java_removed_indices = export_vec(env, &removed_indices)?;

            Ok(env.new_object(
                "org/lance/operation/CreateIndex",
                "(Ljava/util/List;Ljava/util/List;)V",
                &[
                    JValue::Object(&java_new_indices),
                    JValue::Object(&java_removed_indices),
                ],
            )?)
        }
        Operation::Update {
            removed_fragment_ids,
            updated_fragments,
            new_fragments,
            fields_modified,
            compacted_sstables,
            fields_for_preserving_frag_bitmap,
            update_mode,
            inserted_rows_filter,
            updated_fragment_offsets,
        } => {
            let removed_fragment_ids_obj =
                export_unsigned_longs(env, &removed_fragment_ids, "update.removedFragmentIds")?;
            let updated_fragments_obj = export_vec(env, &updated_fragments)?;
            let new_fragments_obj = export_vec(env, &new_fragments)?;
            let fields_modified = JLance(fields_modified.clone()).into_java(env)?;
            let fields_for_preserving_frag_bitmap =
                JLance(fields_for_preserving_frag_bitmap.clone()).into_java(env)?;
            let update_mode = match update_mode {
                Some(update_mode) => update_mode.into_java(env),
                None => Ok(JObject::null()),
            }?;
            let update_mode_optional = env
                .call_static_method(
                    "java/util/Optional",
                    "ofNullable",
                    "(Ljava/lang/Object;)Ljava/util/Optional;",
                    &[JValue::Object(&update_mode)],
                )?
                .l()?;
            let compacted_sstables = export_compacted_sstables(env, &compacted_sstables)?;
            let inserted_rows_filter = match inserted_rows_filter {
                Some(filter) => key_existence_filter_into_java(env, &filter)?,
                None => JObject::null(),
            };
            // Serialize updated_fragment_offsets to Java Map<Long, byte[]>.
            // Values are portable RoaringBitmap bytes so the JNI boundary stays O(bitmap size)
            // rather than O(n rows). Empty HashMap when None so the Java constructor always
            // receives a non-null map.
            let java_offsets_map = {
                let java_map = env.new_object("java/util/HashMap", "()V", &[])?;
                if let Some(UpdatedFragmentOffsets(ref map)) = updated_fragment_offsets {
                    for (frag_id, bitmap) in map {
                        let mut buf: Vec<u8> = Vec::new();
                        bitmap.serialize_into(&mut buf).map_err(|e| {
                            Error::runtime_error(format!(
                                "failed to serialize updatedFragmentOffsets for fragment \
                                 {frag_id}: {e}"
                            ))
                        })?;
                        // JNI byte arrays are signed i8; reinterpret without copying.
                        let buf_i8: &[i8] = unsafe {
                            std::slice::from_raw_parts(buf.as_ptr() as *const i8, buf.len())
                        };
                        env.with_local_frame(4, |env| {
                            let java_key = env.new_object(
                                "java/lang/Long",
                                "(J)V",
                                &[JValue::Long(u64_to_jlong(
                                    "update.updatedFragmentOffsets.fragmentId",
                                    *frag_id,
                                )?)],
                            )?;
                            let java_arr = env.new_byte_array(buf_i8.len() as i32)?;
                            env.set_byte_array_region(&java_arr, 0, buf_i8)?;
                            env.call_method(
                                &java_map,
                                "put",
                                "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
                                &[JValue::Object(&java_key), JValue::Object(&*java_arr)],
                            )?;
                            Ok::<JObject, Error>(JObject::null())
                        })?;
                    }
                }
                java_map
            };
            Ok(env.new_object(
                "org/lance/operation/Update",
                "(Ljava/util/List;Ljava/util/List;Ljava/util/List;[J[JLjava/util/Optional;Ljava/util/Map;Ljava/util/List;Lorg/lance/operation/KeyExistenceFilter;)V",
                &[
                    JValue::Object(&removed_fragment_ids_obj),
                    JValue::Object(&updated_fragments_obj),
                    JValue::Object(&new_fragments_obj),
                    JValueGen::Object(&fields_modified),
                    JValueGen::Object(&fields_for_preserving_frag_bitmap),
                    JValue::Object(&update_mode_optional),
                    JValue::Object(&java_offsets_map),
                    JValue::Object(&compacted_sstables),
                    JValue::Object(&inserted_rows_filter),
                ],
            )?)
        }
        Operation::Project {
            schema,
            preserves_nullability,
        } => {
            let java_schema = convert_to_java_schema(env, schema)?;

            Ok(env.new_object(
                "org/lance/operation/Project",
                "(Lorg/apache/arrow/vector/types/pojo/Schema;Z)V",
                &[
                    JValue::Object(&java_schema),
                    JValue::Bool(preserves_nullability as u8),
                ],
            )?)
        }
        Operation::Rewrite {
            groups,
            rewritten_indices,
            frag_reuse_index,
        } => {
            let java_groups = export_vec(env, &groups)?;
            let java_indices = export_vec(env, &rewritten_indices)?;
            let java_frag_reuse_index = match frag_reuse_index {
                Some(index) => index.into_java(env)?,
                None => JObject::null(),
            };

            Ok(env.new_object(
                "org/lance/operation/Rewrite",
                "(Ljava/util/List;Ljava/util/List;Lorg/lance/index/Index;)V",
                &[
                    JValue::Object(&java_groups),
                    JValue::Object(&java_indices),
                    JValue::Object(&java_frag_reuse_index),
                ],
            )?)
        }
        Operation::UpdateConfig {
            config_updates,
            table_metadata_updates,
            schema_metadata_updates,
            field_metadata_updates,
        } => {
            let config_updates_obj = export_update_map(env, &config_updates)?;
            let table_metadata_updates_obj = export_update_map(env, &table_metadata_updates)?;
            let schema_metadata_updates_obj = export_update_map(env, &schema_metadata_updates)?;

            // Handle field_metadata_updates
            let field_metadata_updates_obj = if field_metadata_updates.is_empty() {
                JObject::null()
            } else {
                let java_map = env.new_object("java/util/HashMap", "()V", &[])?;
                let map = JMap::from_env(env, &java_map)?;

                for (field_id, update_map) in field_metadata_updates {
                    let java_field_id =
                        env.new_object("java/lang/Integer", "(I)V", &[JValue::Int(field_id)])?;

                    let update_map_obj = export_update_map(env, &Some(update_map.clone()))?;
                    map.put(env, &java_field_id, &update_map_obj)?;
                }
                java_map
            };

            let java_operation = env.new_object(
                "org/lance/operation/UpdateConfig",
                "(Lorg/lance/operation/UpdateMap;Lorg/lance/operation/UpdateMap;Lorg/lance/operation/UpdateMap;Ljava/util/Map;)V",
                &[
                    JValue::Object(&config_updates_obj),
                    JValue::Object(&table_metadata_updates_obj),
                    JValue::Object(&schema_metadata_updates_obj),
                    JValue::Object(&field_metadata_updates_obj),
                ],
            )?;
            Ok(java_operation)
        }
        Operation::DataReplacement { replacements } => {
            let java_replacements = export_vec(env, &replacements)?;

            Ok(env.new_object(
                "org/lance/operation/DataReplacement",
                "(Ljava/util/List;)V",
                &[JValue::Object(&java_replacements)],
            )?)
        }
        Operation::DataOverlay { groups } => {
            let groups = export_vec(env, &groups)?;
            Ok(env.new_object(
                "org/lance/operation/DataOverlay",
                "(Ljava/util/List;)V",
                &[JValue::Object(&groups)],
            )?)
        }
        Operation::Merge {
            fragments: rust_fragments,
            schema,
            preserves_nullability,
        } => {
            let java_fragments = export_vec(env, &rust_fragments)?;
            let java_schema = convert_to_java_schema(env, schema)?;

            Ok(env.new_object(
                "org/lance/operation/Merge",
                "(Ljava/util/List;Lorg/apache/arrow/vector/types/pojo/Schema;Z)V",
                &[
                    JValue::Object(&java_fragments),
                    JValue::Object(&java_schema),
                    JValue::Bool(preserves_nullability as u8),
                ],
            )?)
        }
        Operation::Restore { version } => Ok(env.new_object(
            "org/lance/operation/Restore",
            "(J)V",
            &[JValue::Long(u64_to_jlong("restore.version", version)?)],
        )?),
        Operation::ReserveFragments { num_fragments } => Ok(env.new_object(
            "org/lance/operation/ReserveFragments",
            "(I)V",
            &[JValue::Int(u32_to_jint(
                "reserveFragments.numFragments",
                num_fragments,
            )?)],
        )?),
        Operation::UpdateMemWalState { compacted_sstables } => {
            let compacted_sstables = export_compacted_sstables(env, &compacted_sstables)?;
            Ok(env.new_object(
                "org/lance/operation/UpdateMemWalState",
                "(Ljava/util/List;)V",
                &[JValue::Object(&compacted_sstables)],
            )?)
        }
        Operation::Clone {
            is_shallow,
            ref_name,
            ref_version,
            ref_path,
            branch_name,
        } => {
            let ref_name = match ref_name {
                Some(ref_name) => env.new_string(ref_name)?.into(),
                None => JObject::null(),
            };
            let ref_path = env.new_string(ref_path)?;
            let branch_name = match branch_name {
                Some(branch_name) => env.new_string(branch_name)?.into(),
                None => JObject::null(),
            };
            Ok(env.new_object(
                "org/lance/operation/Clone",
                "(ZLjava/lang/String;JLjava/lang/String;Ljava/lang/String;)V",
                &[
                    JValue::Bool(is_shallow as u8),
                    JValue::Object(&ref_name),
                    JValue::Long(u64_to_jlong("clone.refVersion", ref_version)?),
                    JValue::Object(&ref_path),
                    JValue::Object(&branch_name),
                ],
            )?)
        }
        Operation::UpdateBases { new_bases } => {
            let new_bases = export_vec(env, &new_bases)?;
            Ok(env.new_object(
                "org/lance/operation/UpdateBases",
                "(Ljava/util/List;)V",
                &[JValue::Object(&new_bases)],
            )?)
        }
    }
}

pub(crate) fn convert_to_java_schema<'local>(
    env: &mut JNIEnv<'local>,
    schema: LanceSchema,
) -> Result<JObject<'local>> {
    let java_schema = schema.into_java(env)?;
    Ok(env
        .call_method(
            &java_schema,
            "asArrowSchema",
            "()Lorg/apache/arrow/vector/types/pojo/Schema;",
            &[],
        )?
        .l()?)
}

/// Parse a `CommitBuilder.storageFormat` string into a [`LanceFileVersion`].
///
/// The canonical spellings ("2.1", "stable", ...) are the ones every other Lance
/// binding accepts and the ones [`LanceFileVersion`]'s `Display` emits.
///
/// The `v`-prefixed spellings are a Java-only accident: this function originally
/// hand-rolled its match by walking the `LanceFileVersion` variant identifiers
/// (`V2_1` -> `"v2_1"`) instead of delegating to `FromStr`, so it accepted those
/// identifiers and rejected the canonical "2.1". They were documented on
/// `CommitBuilder.storageFormat` and shipped from 3.0.0, so they are translated
/// here for compatibility. The set is deliberately frozen to what shipped —
/// newer versions are reachable only by their canonical name.
fn parse_storage_format(name: &str) -> Result<LanceFileVersion> {
    let requested = name.to_lowercase();
    let canonical = match requested.as_str() {
        "v2_0" | "v2.0" => V2_FORMAT_2_0,
        "v2_1" | "v2.1" => V2_FORMAT_2_1,
        "v2_2" | "v2.2" => V2_FORMAT_2_2,
        _ => requested.as_str(),
    };

    if canonical != requested {
        log::warn!(
            "Storage format \"{}\" is deprecated and will be removed in a future release; use \"{}\" instead",
            name,
            canonical
        );
    }

    canonical
        .parse::<LanceFileVersion>()
        .map_err(|_| Error::input_error(format!("Unknown storage format: {}", name)))
}

/// Translate the Java `commitTimeoutNanos` sentinel into an
/// `Option<Duration>` for [`CommitBuilder::with_timeout`]. The Java side is
/// the source of truth for the default (30 minutes) and for rejecting
/// zero/negative-from-the-user inputs; here `< 0` simply means "disabled" and
/// any other value is the timeout in nanoseconds.
fn parse_commit_timeout(nanos: i64) -> Option<std::time::Duration> {
    if nanos < 0 {
        None
    } else {
        Some(std::time::Duration::from_nanos(nanos as u64))
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_lance_CommitBuilder_nativeCommitToDataset<'local>(
    mut env: JNIEnv<'local>,
    _cls: JObject,
    java_dataset: JObject,
    java_transaction: JObject,
    detached_jbool: jboolean,
    enable_v2_manifest_paths: jboolean,
    write_params_obj: JObject,
    use_stable_row_ids_obj: JObject,
    storage_format_obj: JObject,
    max_retries: jint,
    skip_auto_cleanup: jboolean,
    namespace_obj: JObject,
    table_id_obj: JObject,
    namespace_client_managed_versioning: jboolean,
    commit_timeout_nanos: jlong,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_commit_to_dataset(
            &mut env,
            java_dataset,
            java_transaction,
            detached_jbool != 0,
            enable_v2_manifest_paths != 0,
            write_params_obj,
            use_stable_row_ids_obj,
            storage_format_obj,
            max_retries as u32,
            skip_auto_cleanup != 0,
            namespace_obj,
            table_id_obj,
            namespace_client_managed_versioning != 0,
            commit_timeout_nanos,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_commit_to_dataset<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    java_transaction: JObject,
    detached: bool,
    enable_v2_manifest_paths: bool,
    write_params_obj: JObject,
    use_stable_row_ids_obj: JObject,
    storage_format_obj: JObject,
    max_retries: u32,
    skip_auto_cleanup: bool,
    namespace_obj: JObject,
    table_id_obj: JObject,
    namespace_client_managed_versioning: bool,
    commit_timeout_nanos: jlong,
) -> Result<JObject<'local>> {
    let commit_timeout = parse_commit_timeout(commit_timeout_nanos);
    let write_param = if write_params_obj.is_null() {
        HashMap::new()
    } else {
        let write_param_jmap = JMap::from_env(env, &write_params_obj)?;
        to_rust_map(env, &write_param_jmap)?
    };

    // Parse optional use_stable_row_ids (boxed Boolean)
    let use_stable_row_ids = if use_stable_row_ids_obj.is_null() {
        None
    } else {
        let val = env
            .call_method(&use_stable_row_ids_obj, "booleanValue", "()Z", &[])?
            .z()?;
        Some(val)
    };

    // Parse optional storage format string
    let storage_format = if storage_format_obj.is_null() {
        None
    } else {
        let format_str: String = JString::from(storage_format_obj).extract(env)?;
        Some(parse_storage_format(&format_str)?)
    };

    // Get the Dataset's storage_options_accessor and merge with write_param
    let storage_options_accessor = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        let existing_accessor = dataset_guard.inner.storage_options_accessor();

        // Merge write_param with existing accessor's initial options
        match existing_accessor {
            Some(accessor) => {
                let mut merged = accessor
                    .initial_storage_options()
                    .cloned()
                    .unwrap_or_default();
                merged.extend(write_param);
                if let Some(provider) = accessor.provider().cloned() {
                    Some(Arc::new(
                        lance::io::StorageOptionsAccessor::with_initial_and_provider(
                            merged, provider,
                        ),
                    ))
                } else {
                    Some(Arc::new(
                        lance::io::StorageOptionsAccessor::with_static_options(merged),
                    ))
                }
            }
            None => {
                if !write_param.is_empty() {
                    Some(Arc::new(
                        lance::io::StorageOptionsAccessor::with_static_options(write_param),
                    ))
                } else {
                    None
                }
            }
        }
    };

    // Build ObjectStoreParams using the merged accessor
    let store_params = ObjectStoreParams {
        storage_options_accessor,
        ..Default::default()
    };

    let java_allocator = env
        .call_method(
            &java_dataset,
            "allocator",
            "()Lorg/apache/arrow/memory/BufferAllocator;",
            &[],
        )?
        .l()?;

    // BlockingDataset from java dataset.
    let mut java_blocking_ds = {
        let dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        BlockingDataset::new(dataset_guard.inner.clone())
    };
    let transaction = convert_to_rust_transaction(
        env,
        java_transaction,
        Some(&java_allocator),
        Some(&mut java_blocking_ds),
    )?;

    // Set namespace commit handler only if namespace_client_managed_versioning is true
    let namespace_info = extract_namespace_info(env, &namespace_obj, &table_id_obj)?;
    let commit_handler = if namespace_client_managed_versioning {
        match namespace_info {
            Some((ns, tid)) => {
                // The store derives the branch a request targets from the base
                // path it is handed, resolved against the table root.
                let table_root = java_blocking_ds.inner.branch_location().find_main()?.path;
                let external_store = LanceNamespaceExternalManifestStore::new(ns, tid, table_root);
                Some(Arc::new(ExternalManifestCommitHandler {
                    external_manifest_store: Arc::new(external_store),
                }) as Arc<dyn CommitHandler>)
            }
            None => None,
        }
    } else {
        None
    };

    let new_blocking_ds = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(&java_dataset, NATIVE_DATASET) }?;
        dataset_guard.commit_transaction(
            transaction,
            store_params,
            detached,
            enable_v2_manifest_paths,
            use_stable_row_ids,
            storage_format,
            max_retries,
            skip_auto_cleanup,
            commit_handler,
            commit_timeout,
        )?
    };
    new_blocking_ds.into_java(env)
}

fn convert_to_rust_transaction(
    env: &mut JNIEnv,
    java_transaction: JObject,
    allocator: Option<&JObject>,
    dataset: Option<&mut BlockingDataset>,
) -> Result<Transaction> {
    let read_ver = nonnegative_jlong_to_u64(
        "transaction.readVersion",
        env.call_method(&java_transaction, "readVersion", "()J", &[])?
            .j()?,
    )?;
    let uuid = env.get_string_from_method(&java_transaction, "uuid")?;
    let op = env
        .call_method(
            &java_transaction,
            "operation",
            "()Lorg/lance/operation/Operation;",
            &[],
        )?
        .l()?;
    let op = convert_to_rust_operation(env, &op, allocator, dataset, read_ver)?;

    let tag = env.get_optional_from_method(&java_transaction, "tag", |env, tag_obj| {
        let tag_str = JString::from(tag_obj);
        tag_str.extract(env)
    })?;

    let transaction_properties = env.get_optional_from_method(
        &java_transaction,
        "transactionProperties",
        |env, transaction_properties| {
            let transaction_properties = JMap::from_env(env, &transaction_properties)?;
            to_rust_map(env, &transaction_properties)
        },
    )?;
    Ok(TransactionBuilder::new(read_ver, op)
        .uuid(uuid)
        .tag(tag)
        .transaction_properties(transaction_properties.map(Arc::new))
        .build())
}

fn convert_schema_from_operation(
    env: &mut JNIEnv,
    java_operation: &JObject,
    java_allocator: &JObject,
    dataset: Option<&mut BlockingDataset>,
    read_version: u64,
) -> Result<LanceSchema> {
    let schema_ptr = env
        .call_method(
            java_operation,
            "exportSchema",
            "(Lorg/apache/arrow/memory/BufferAllocator;)J",
            &[JValue::Object(java_allocator)],
        )?
        .j()?;
    let c_schema_ptr = schema_ptr as *mut FFI_ArrowSchema;
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };

    if let Some(dataset) = dataset {
        let arrow_schema = Schema::try_from(&c_schema)?;

        // Derive field ids based on the transaction read dataset schema.
        let read_schema = {
            if dataset.inner.version().version == read_version {
                dataset.inner.schema().clone()
            } else {
                let read_dataset = dataset.checkout_version(read_version)?;
                read_dataset.inner.schema().clone()
            }
        };

        let max_field_id = dataset.inner.manifest().max_field_id();
        let schema =
            LanceSchema::from_arrow_schema(&arrow_schema, Some(read_schema), Some(max_field_id))?;
        Ok(schema)
    } else {
        let schema = Schema::try_from(&c_schema)?;
        LanceSchema::try_from(&schema).map_err(|e| {
            Error::input_error(format!(
                "Failed to convert Arrow schema to Lance schema: {}",
                e
            ))
        })
    }
}

trait SchemaExt {
    /// Walk through the fields and assign a new field id to each field that does not have one
    /// (e.g. is set to -1)
    ///
    /// If this schema is on an existing dataset, pass the schema of the dataset to `base_schema`
    /// and the result of `Manifest::max_field_id` to `max_existing_id`.
    ///
    /// If this schema is not associated with a dataset, pass `None` to `base_schema` and
    /// `max_existing_id`.
    ///
    /// The rule of assigning id is:
    /// 1. If a lance field with same name exists in `base_schema` (including nested field), id is
    ///    derived from the field.
    /// 2. Otherwise, set field id based on max id, which is computed from `max_existing_id`,
    ///    `base_schema` max id and self max id.
    fn set_field_id_from_schema(
        &mut self,
        base_schema: Option<LanceSchema>,
        max_existing_id: Option<i32>,
    ) -> Result<()>;

    /// Create schema from `arrow_schema`, with field id priority below:
    /// 1. arrow metadata field id.
    /// 2. field id from `base_schema`.
    /// 3. field id from `max_existing_id`.
    fn from_arrow_schema(
        arrow_schema: &Schema,
        base_schema: Option<LanceSchema>,
        max_existing_id: Option<i32>,
    ) -> Result<LanceSchema>;
}

impl SchemaExt for LanceSchema {
    fn set_field_id_from_schema(
        &mut self,
        base_schema: Option<LanceSchema>,
        max_existing_id: Option<i32>,
    ) -> Result<()> {
        // Set id from base_schema
        if let Some(base_schema) = &base_schema {
            for field in self.fields.iter_mut() {
                if let Some(base_field) = base_schema.field(&field.name) {
                    field.set_field_id_from_field(-1, base_field)?;
                }
            }
        };

        // Set id from max_id
        let max_id = base_schema
            .map(|s| s.max_field_id().unwrap_or(-1))
            .unwrap_or(-1);
        let max_id = max_id.max(max_existing_id.unwrap_or(-1));
        self.set_field_id(Some(max_id));
        Ok(())
    }

    fn from_arrow_schema(
        arrow_schema: &Schema,
        base_schema: Option<LanceSchema>,
        max_existing_id: Option<i32>,
    ) -> Result<LanceSchema> {
        let mut schema = Self {
            fields: arrow_schema
                .fields
                .iter()
                .map(|f| Field::try_from(f.as_ref()))
                .collect::<lance_core::Result<_>>()?,
            metadata: arrow_schema.metadata.clone(),
        };
        schema.set_field_id_from_schema(base_schema, max_existing_id)?;
        schema.validate()?;
        schema.verify_primary_key()?;

        Ok(schema)
    }
}

trait FieldExt {
    /// Recursively set field ID and parent ID for this field and all its children.
    fn set_field_id_from_field(
        &mut self,
        parent_id: i32,
        base_field: &Field,
    ) -> lance_core::Result<()>;
}

impl FieldExt for Field {
    fn set_field_id_from_field(
        &mut self,
        parent_id: i32,
        base_field: &Field,
    ) -> lance_core::Result<()> {
        self.parent_id = parent_id;

        if self.name != base_field.name {
            return Ok(());
        }

        if self.logical_type != base_field.logical_type {
            return Err(lance_core::Error::invalid_input_source(
                format!(
                    "Expecting logical type {} but got {} for field {}",
                    base_field.logical_type, self.logical_type, self.name
                )
                .into(),
            ));
        }

        if self.id < 0 {
            // use id from base
            self.id = base_field.id;
        }

        for child in &mut self.children {
            if let Some(base_child) = base_field.children.iter().find(|f| f.name == child.name) {
                child.set_field_id_from_field(self.id, base_child)?;
            }
        }
        Ok(())
    }
}

fn convert_to_rust_operation(
    env: &mut JNIEnv<'_>,
    java_operation: &JObject<'_>,
    allocator: Option<&JObject<'_>>,
    dataset: Option<&mut BlockingDataset>,
    read_version: u64,
) -> Result<Operation> {
    let op_name = env.get_string_from_method(java_operation, "name")?;
    let op = match op_name.as_str() {
        "Project" => Operation::Project {
            preserves_nullability: env
                .get_boolean_from_method(java_operation, "preservesNullability")?,
            schema: convert_schema_from_operation(
                env,
                java_operation,
                allocator.ok_or_else(|| {
                    Error::input_error(
                        "BufferAllocator is required for Project operations".to_string(),
                    )
                })?,
                dataset,
                read_version,
            )?,
        },
        "UpdateConfig" => {
            let config_updates_obj = env
                .call_method(
                    java_operation,
                    "configUpdates",
                    "()Lorg/lance/operation/UpdateMap;",
                    &[],
                )?
                .l()?;
            let config_updates = if config_updates_obj.is_null() {
                None
            } else {
                extract_update_map(env, &config_updates_obj)?
            };

            let table_metadata_updates_obj = env
                .call_method(
                    java_operation,
                    "tableMetadataUpdates",
                    "()Lorg/lance/operation/UpdateMap;",
                    &[],
                )?
                .l()?;
            let table_metadata_updates = if table_metadata_updates_obj.is_null() {
                None
            } else {
                extract_update_map(env, &table_metadata_updates_obj)?
            };

            let schema_metadata_updates_obj = env
                .call_method(
                    java_operation,
                    "schemaMetadataUpdates",
                    "()Lorg/lance/operation/UpdateMap;",
                    &[],
                )?
                .l()?;
            let schema_metadata_updates = if schema_metadata_updates_obj.is_null() {
                None
            } else {
                extract_update_map(env, &schema_metadata_updates_obj)?
            };

            let field_metadata_updates_obj = env
                .call_method(
                    java_operation,
                    "fieldMetadataUpdates",
                    "()Ljava/util/Map;",
                    &[],
                )?
                .l()?;
            let mut field_metadata_updates = HashMap::new();
            if !field_metadata_updates_obj.is_null() {
                let field_metadata_map = JMap::from_env(env, &field_metadata_updates_obj)?;
                let mut iter = field_metadata_map.iter(env)?;
                env.with_local_frame(16, |env| {
                    while let Some((key, value)) = iter.next(env)? {
                        let field_id = env.call_method(&key, "intValue", "()I", &[])?.i()?;
                        if let Some(update_map) = extract_update_map(env, &value)? {
                            field_metadata_updates.insert(field_id, update_map);
                        }
                    }
                    Ok::<(), Error>(())
                })?;
            }

            Operation::UpdateConfig {
                config_updates,
                table_metadata_updates,
                schema_metadata_updates,
                field_metadata_updates,
            }
        }
        "Append" => {
            let fragments =
                import_vec_from_method(env, java_operation, "fragments", |env, fragment| {
                    fragment.extract_object(env)
                })?;
            Operation::Append { fragments }
        }
        "Delete" => {
            let updated_fragments: Vec<Fragment> = import_vec_from_method(
                env,
                java_operation,
                "updatedFragments",
                |env, fragment| fragment.extract_object(env),
            )?;

            let deleted_fragment_ids = import_unsigned_longs(
                env,
                java_operation,
                "deletedFragmentIds",
                "delete.deletedFragmentIds",
            )?;

            let predicate = env.get_string_from_method(java_operation, "predicate")?;

            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            }
        }
        "Overwrite" => {
            let fragments: Vec<Fragment> =
                import_vec_from_method(env, java_operation, "fragments", |env, fragment| {
                    fragment.extract_object(env)
                })?;

            let config_upsert_values = env.get_optional_from_method(
                java_operation,
                "configUpsertValues",
                |env, config_upsert_values| {
                    let config_upsert_values = JMap::from_env(env, &config_upsert_values)?;
                    to_rust_map(env, &config_upsert_values)
                },
            )?;
            let initial_bases =
                env.get_optional_from_method(java_operation, "getInitialBases", |env, bases| {
                    crate::traits::import_vec_to_rust(env, &bases, |env, base| {
                        base.extract_object(env)
                    })
                })?;
            // Pass None for dataset so that the new schema is not validated
            // against the old schema. Overwrite replaces the entire dataset,
            // so fields with the same name but different types are allowed.
            let schema = convert_schema_from_operation(
                env,
                java_operation,
                allocator.ok_or_else(|| {
                    Error::input_error(
                        "BufferAllocator is required for Overwrite operations".to_string(),
                    )
                })?,
                None,
                read_version,
            )?;
            Operation::Overwrite {
                fragments,
                schema,
                config_upsert_values,
                initial_bases,
            }
        }
        "Rewrite" => {
            let groups: Vec<RewriteGroup> =
                import_vec_from_method(env, java_operation, "groups", |env, group| {
                    group.extract_object(env)
                })?;

            let rewritten_indices: Vec<RewrittenIndex> =
                import_vec_from_method(env, java_operation, "rewrittenIndices", |env, index| {
                    index.extract_object(env)
                })?;

            let frag_reuse_index: Option<IndexMetadata> = env.get_optional_from_method(
                java_operation,
                "fragReuseIndex",
                |env, frag_reuse_index| frag_reuse_index.extract_object(env),
            )?;

            Operation::Rewrite {
                groups,
                rewritten_indices,
                frag_reuse_index,
            }
        }
        "Update" => {
            let removed_fragment_ids = import_unsigned_longs(
                env,
                java_operation,
                "removedFragmentIds",
                "update.removedFragmentIds",
            )?;

            let updated_fragments: Vec<Fragment> = import_vec_from_method(
                env,
                java_operation,
                "updatedFragments",
                |env, fragment| fragment.extract_object(env),
            )?;

            let new_fragments: Vec<Fragment> =
                import_vec_from_method(env, java_operation, "newFragments", |env, fragment| {
                    fragment.extract_object(env)
                })?;

            let fields_modified = import_field_ids(
                env,
                java_operation,
                "fieldsModified",
                "update.fieldsModified",
            )?;
            let fields_for_preserving_frag_bitmap = import_field_ids(
                env,
                java_operation,
                "fieldsForPreservingFragBitmap",
                "update.fieldsForPreservingFragBitmap",
            )?;

            let update_mode: Option<UpdateMode> =
                env.get_optional_from_method(java_operation, "updateMode", |env, update_mode| {
                    update_mode.extract_object(env)
                })?;
            if update_mode.is_none() {
                return Err(Error::input_error(
                    "update.updateMode must be specified because the transaction format cannot \
                     persist an absent update mode distinctly from RewriteRows"
                        .to_string(),
                ));
            }
            let compacted_sstables = import_vec_from_method(
                env,
                java_operation,
                "getCompactedSstables",
                |env, sstable| compacted_sstable_from_java(env, &sstable),
            )?;
            let inserted_rows_filter = env.get_optional_from_method(
                java_operation,
                "getInsertedRowsFilter",
                |env, filter| key_existence_filter_from_java(env, &filter),
            )?;

            let updated_fragment_offsets = {
                let offsets_obj = env
                    .call_method(
                        java_operation,
                        "updatedFragmentOffsets",
                        "()Ljava/util/Map;",
                        &[],
                    )?
                    .l()?;
                if offsets_obj.is_null() {
                    None
                } else {
                    let jmap = JMap::from_env(env, &offsets_obj)?;
                    let mut iter = jmap.iter(env)?;
                    let mut offsets: HashMap<u64, RoaringBitmap> = HashMap::new();
                    // Per-iteration local frame: iterator key/value JNI refs are released each
                    // loop so large multi-fragment maps cannot exhaust the local reference table.
                    loop {
                        let entry = env.with_local_frame(
                            8,
                            |env| -> Result<Option<(u64, RoaringBitmap)>> {
                                let Some((key, value)) = iter.next(env)? else {
                                    return Ok(None);
                                };
                                let frag_id = nonnegative_jlong_to_u64(
                                    "update.updatedFragmentOffsets.fragmentId",
                                    env.call_method(&key, "longValue", "()J", &[])?.j()?,
                                )?;
                                let buf: Vec<u8> =
                                    env.convert_byte_array(JByteArray::from(value))?;
                                let bitmap = RoaringBitmap::deserialize_from(buf.as_slice())
                                    .map_err(|e| {
                                        Error::input_error(format!(
                                            "invalid updatedFragmentOffsets RoaringBitmap bytes \
                                         for fragment {frag_id}: {e}"
                                        ))
                                    })?;
                                Ok(Some((frag_id, bitmap)))
                            },
                        )?;
                        match entry {
                            None => break,
                            Some((frag_id, bitmap)) => {
                                offsets.insert(frag_id, bitmap);
                            }
                        }
                    }
                    if offsets.is_empty() {
                        None
                    } else {
                        Some(UpdatedFragmentOffsets(offsets))
                    }
                }
            };

            Operation::Update {
                removed_fragment_ids,
                updated_fragments,
                new_fragments,
                fields_modified,
                compacted_sstables,
                fields_for_preserving_frag_bitmap,
                update_mode,
                inserted_rows_filter,
                updated_fragment_offsets,
            }
        }
        "DataReplacement" => {
            let replacements: Vec<DataReplacementGroup> =
                import_vec_from_method(env, java_operation, "replacements", |env, replacement| {
                    replacement.extract_object(env)
                })?;
            Operation::DataReplacement { replacements }
        }
        "DataOverlay" => {
            let groups = import_vec_from_method(env, java_operation, "getGroups", |env, group| {
                group.extract_object(env)
            })?;
            Operation::DataOverlay { groups }
        }
        "Merge" => {
            let fragments: Vec<Fragment> =
                import_vec_from_method(env, java_operation, "fragments", |env, fragment| {
                    fragment.extract_object(env)
                })?;
            Operation::Merge {
                fragments,
                preserves_nullability: env
                    .get_boolean_from_method(java_operation, "preservesNullability")?,
                schema: convert_schema_from_operation(
                    env,
                    java_operation,
                    allocator.ok_or_else(|| {
                        Error::input_error(
                            "BufferAllocator is required for Merge operations".to_string(),
                        )
                    })?,
                    dataset,
                    read_version,
                )?,
            }
        }
        "Restore" => {
            let version = nonnegative_jlong_to_u64(
                "restore.version",
                env.call_method(java_operation, "version", "()J", &[])?
                    .j()?,
            )?;
            return Ok(Operation::Restore { version });
        }
        "ReserveFragments" => {
            let java_num_fragments = env
                .call_method(java_operation, "numFragments", "()I", &[])?
                .i()?;
            let num_fragments = u32::try_from(java_num_fragments).map_err(|_| {
                Error::input_error(format!(
                    "reserveFragments.numFragments must be non-negative, got {java_num_fragments}"
                ))
            })?;
            return Ok(Operation::ReserveFragments { num_fragments });
        }
        "CreateIndex" => {
            let new_indices =
                import_vec_from_method(env, java_operation, "getNewIndices", |env, index| {
                    index.extract_object(env)
                })?;
            let removed_indices =
                import_vec_from_method(env, java_operation, "getRemovedIndices", |env, index| {
                    index.extract_object(env)
                })?;
            return Ok(Operation::CreateIndex {
                new_indices,
                removed_indices,
            });
        }
        "UpdateMemWalState" => {
            let compacted_sstables = import_vec_from_method(
                env,
                java_operation,
                "getCompactedSstables",
                |env, sstable| compacted_sstable_from_java(env, &sstable),
            )?;
            Operation::UpdateMemWalState { compacted_sstables }
        }
        "Clone" => Operation::Clone {
            is_shallow: env.get_boolean_from_method(java_operation, "isShallow")?,
            ref_name: env.get_optional_string_from_method(java_operation, "getRefName")?,
            ref_version: nonnegative_jlong_to_u64(
                "clone.refVersion",
                env.call_method(java_operation, "getRefVersion", "()J", &[])?
                    .j()?,
            )?,
            ref_path: env.get_string_from_method(java_operation, "getRefPath")?,
            branch_name: env.get_optional_string_from_method(java_operation, "getBranchName")?,
        },
        "UpdateBases" => {
            let new_bases =
                import_vec_from_method(env, java_operation, "getNewBases", |env, base| {
                    base.extract_object(env)
                })?;
            Operation::UpdateBases { new_bases }
        }
        _ => {
            return Err(Error::input_error(format!(
                "Unsupported Java transaction operation: {op_name}"
            )));
        }
    };
    Ok(op)
}

fn extract_update_map(env: &mut JNIEnv, update_map_obj: &JObject) -> Result<Option<UpdateMap>> {
    if update_map_obj.is_null() {
        return Ok(None);
    }

    let updates_obj = env
        .call_method(update_map_obj, "updates", "()Ljava/util/Map;", &[])?
        .l()?;
    let replace = env
        .call_method(update_map_obj, "replace", "()Z", &[])?
        .z()?;

    if updates_obj.is_null() {
        return Ok(None);
    }

    let updates_map = JMap::from_env(env, &updates_obj)?;
    let mut entries = Vec::new();
    let mut iter = updates_map.iter(env)?;

    env.with_local_frame(16, |env| {
        while let Some((key, value)) = iter.next(env)? {
            let key_jstring = JString::from(key);
            let key_string: String = env.get_string(&key_jstring)?.into();

            let value_string = if value.is_null() {
                None
            } else {
                let value_jstring = JString::from(value);
                let value_str = env.get_string(&value_jstring)?.into();
                Some(value_str)
            };

            entries.push(UpdateMapEntry {
                key: key_string,
                value: value_string,
            });
        }
        Ok::<(), Error>(())
    })?;

    Ok(Some(UpdateMap {
        update_entries: entries,
        replace,
    }))
}

fn export_update_map<'a>(
    env: &mut JNIEnv<'a>,
    update_map: &Option<UpdateMap>,
) -> Result<JObject<'a>> {
    match update_map {
        None => Ok(JObject::null()),
        Some(map) => {
            // Create a Java HashMap for the updates
            let updates_map = env.new_object("java/util/HashMap", "()V", &[])?;
            let jmap = JMap::from_env(env, &updates_map)?;

            for entry in &map.update_entries {
                let key = env.new_string(&entry.key)?;
                let value = match &entry.value {
                    Some(val) => JObject::from(env.new_string(val)?),
                    None => JObject::null(),
                };
                jmap.put(env, &key, &value)?;
            }

            // Create UpdateMap object
            let update_map_obj = env.new_object(
                "org/lance/operation/UpdateMap",
                "(Ljava/util/Map;Z)V",
                &[
                    JValue::Object(&updates_map),
                    JValue::Bool(map.replace as u8),
                ],
            )?;
            Ok(update_map_obj)
        }
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_org_lance_CommitBuilder_nativeCommitToUri<'local>(
    mut env: JNIEnv<'local>,
    _cls: JObject,
    uri: JString,
    java_transaction: JObject,
    detached_jbool: jboolean,
    enable_v2_manifest_paths: jboolean,
    namespace_obj: JObject,
    table_id_obj: JObject,
    allocator_obj: JObject,
    write_params_obj: JObject,
    use_stable_row_ids_obj: JObject,
    storage_format_obj: JObject,
    max_retries: jint,
    skip_auto_cleanup: jboolean,
    namespace_client_managed_versioning: jboolean,
    commit_timeout_nanos: jlong,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_commit_to_uri(
            &mut env,
            uri,
            java_transaction,
            detached_jbool != 0,
            enable_v2_manifest_paths != 0,
            namespace_obj,
            table_id_obj,
            allocator_obj,
            write_params_obj,
            use_stable_row_ids_obj,
            storage_format_obj,
            max_retries as u32,
            skip_auto_cleanup != 0,
            namespace_client_managed_versioning != 0,
            commit_timeout_nanos,
        )
    )
}

#[allow(clippy::too_many_arguments)]
fn inner_commit_to_uri<'local>(
    env: &mut JNIEnv<'local>,
    uri: JString,
    java_transaction: JObject,
    detached: bool,
    enable_v2_manifest_paths: bool,
    namespace_obj: JObject,
    table_id_obj: JObject,
    allocator_obj: JObject,
    write_params_obj: JObject,
    use_stable_row_ids_obj: JObject,
    storage_format_obj: JObject,
    max_retries: u32,
    skip_auto_cleanup: bool,
    namespace_client_managed_versioning: bool,
    commit_timeout_nanos: jlong,
) -> Result<JObject<'local>> {
    let commit_timeout = parse_commit_timeout(commit_timeout_nanos);
    let uri_str: String = uri.extract(env)?;

    // Extract write params from parameter
    let write_param = if write_params_obj.is_null() {
        HashMap::new()
    } else {
        let write_param_jmap = JMap::from_env(env, &write_params_obj)?;
        to_rust_map(env, &write_param_jmap)?
    };

    // Parse optional use_stable_row_ids (boxed Boolean)
    let use_stable_row_ids = if use_stable_row_ids_obj.is_null() {
        None
    } else {
        let val = env
            .call_method(&use_stable_row_ids_obj, "booleanValue", "()Z", &[])?
            .z()?;
        Some(val)
    };

    // Parse optional storage format string
    let storage_format = if storage_format_obj.is_null() {
        None
    } else {
        let format_str: String = JString::from(storage_format_obj).extract(env)?;
        Some(parse_storage_format(&format_str)?)
    };

    // Extract namespace info and create storage options provider if namespace is provided
    let namespace_info = extract_namespace_info(env, &namespace_obj, &table_id_obj)?;
    let storage_options_provider: Option<Arc<dyn StorageOptionsProvider>> =
        if let Some((ref ns, ref tid)) = namespace_info {
            Some(Arc::new(LanceNamespaceStorageOptionsProvider::new(
                ns.clone(),
                tid.clone(),
            )))
        } else {
            None
        };

    // Keep a copy of initial options for opening the read dataset.
    let initial_storage_options = write_param.clone();

    let accessor = match (write_param.is_empty(), storage_options_provider.clone()) {
        (false, Some(provider)) => Some(Arc::new(
            lance::io::StorageOptionsAccessor::with_initial_and_provider(write_param, provider),
        )),
        (false, None) => Some(Arc::new(
            lance::io::StorageOptionsAccessor::with_static_options(write_param),
        )),
        (true, Some(provider)) => Some(Arc::new(lance::io::StorageOptionsAccessor::with_provider(
            provider,
        ))),
        (true, None) => None,
    };

    let store_params = ObjectStoreParams {
        storage_options_accessor: accessor,
        ..Default::default()
    };

    let (open_namespace, open_table_id) = match &namespace_info {
        Some((namespace_client, tid)) => (Some(namespace_client.clone()), Some(tid.clone())),
        None => (None, None),
    };

    // Open the read dataset using the same storage options (and provider, if any) so that
    // `convert_to_rust_transaction` can derive schema/field ids based on the target dataset.
    let mut ds = BlockingDataset::open(
        &uri_str,
        None,
        None,
        6 * 1024 * 1024,
        1024 * 1024,
        initial_storage_options,
        HashMap::new(),
        None,
        storage_options_provider,
        None,
        open_namespace,
        open_table_id,
        namespace_client_managed_versioning,
    )
    .ok();

    // Convert Java transaction to Rust
    let allocator_ref = if allocator_obj.is_null() {
        None
    } else {
        Some(allocator_obj)
    };
    let transaction =
        convert_to_rust_transaction(env, java_transaction, allocator_ref.as_ref(), ds.as_mut())?;

    // Build CommitBuilder with URI
    let mut builder = CommitBuilder::new(&*uri_str)
        .with_store_params(store_params)
        .with_detached(detached)
        .enable_v2_manifest_paths(enable_v2_manifest_paths)
        .with_timeout(commit_timeout);

    if let Some(use_stable) = use_stable_row_ids {
        builder = builder.use_stable_row_ids(use_stable);
    }
    if let Some(format) = storage_format {
        builder = builder.with_storage_format(format);
    }
    if max_retries > 0 {
        builder = builder.with_max_retries(max_retries);
    }
    if skip_auto_cleanup {
        builder = builder.with_skip_auto_cleanup(true);
    }

    // Set namespace commit handler only if namespace_client_managed_versioning is true
    if namespace_client_managed_versioning && let Some((namespace_client, tid)) = namespace_info {
        let external_store =
            LanceNamespaceExternalManifestStore::for_table_uri(namespace_client, tid, &uri_str)?;
        let commit_handler: Arc<dyn CommitHandler> = Arc::new(ExternalManifestCommitHandler {
            external_manifest_store: Arc::new(external_store),
        });
        builder = builder.with_commit_handler(commit_handler);
    }

    let dataset = block_on(builder.execute(transaction))?;
    let blocking_ds = BlockingDataset { inner: dataset };
    blocking_ds.into_java(env)
}

#[cfg(test)]
mod tests {
    use arrow_schema::{
        DataType as ArrowDataType, Field as ArrowField, Fields as ArrowFields,
        Schema as ArrowSchema,
    };
    use std::{collections::HashMap, sync::Arc};

    use super::*;

    pub const LANCE_FIELD_ID_KEY: &str = "lance:field_id";

    #[test]
    fn test_create_schema_from_arrow() {
        // base_schema has an existing field id
        let mut base_a = Field::new_arrow("a", ArrowDataType::Int32, false).unwrap();
        base_a.set_id(-1, &mut 10);
        let mut base_b = Field::new_arrow("b", ArrowDataType::Int32, false).unwrap();
        base_b.set_id(-1, &mut 11);

        // base struct: s{x,y}
        let mut base_s = Field::try_from(&ArrowField::new(
            "s",
            ArrowDataType::Struct(ArrowFields::from(vec![
                ArrowField::new("x", ArrowDataType::Int32, false),
                ArrowField::new("y", ArrowDataType::Int32, false),
            ])),
            false,
        ))
        .unwrap();
        base_s.set_id(-1, &mut 20);
        let base_s_x = base_s.children.iter_mut().find(|c| c.name == "x").unwrap();
        base_s_x.set_id(20, &mut 21);
        let base_s_y = base_s.children.iter_mut().find(|c| c.name == "y").unwrap();
        base_s_y.set_id(20, &mut 22);

        // base list: l<item>
        let mut base_l = Field::try_from(&ArrowField::new(
            "l",
            ArrowDataType::List(Arc::new(ArrowField::new(
                "item",
                ArrowDataType::Int32,
                true,
            ))),
            true,
        ))
        .unwrap();
        base_l.set_id(-1, &mut 30);
        let base_l_item = base_l
            .children
            .iter_mut()
            .find(|c| c.name == "item")
            .unwrap();
        base_l_item.set_id(30, &mut 31);

        // base map: m<entries{key,value}>
        let base_map_entries = ArrowField::new(
            "entries",
            ArrowDataType::Struct(ArrowFields::from(vec![
                ArrowField::new("key", ArrowDataType::Utf8, false),
                ArrowField::new("value", ArrowDataType::Int32, true),
            ])),
            false,
        );
        let mut base_m = Field::try_from(&ArrowField::new(
            "m",
            ArrowDataType::Map(Arc::new(base_map_entries), false),
            true,
        ))
        .unwrap();
        base_m.set_id(-1, &mut 40);

        let base_m_entries = base_m
            .children
            .iter_mut()
            .find(|c| c.name == "entries")
            .unwrap();
        base_m_entries.set_id(40, &mut 41);

        let base_m_key = base_m_entries
            .children
            .iter_mut()
            .find(|c| c.name == "key")
            .unwrap();
        base_m_key.set_id(41, &mut 42);

        let base_m_val = base_m_entries
            .children
            .iter_mut()
            .find(|c| c.name == "value")
            .unwrap();
        base_m_val.set_id(41, &mut 43);

        let base_schema = LanceSchema {
            fields: vec![base_a, base_b, base_s, base_l, base_m],
            metadata: HashMap::from([("base_schema_k".to_string(), "base_schema_v".to_string())]),
        };

        // new_schema specifies:
        // - field a: manual field id
        // - field b: no id -> should inherit from base_schema
        // - field c: new field -> should be assigned based on max_field_id
        // - struct s: parent+child(x) manual, child(y) inherit, child(z) max_field_id
        // - list l: parent manual, child(item) inherit
        // - list l2: parent manual, child(item) max_field_id
        // - map m: parent manual, child(entries/key/value) inherit
        // - map m2: parent manual, child(entries/key/value) max_field_id
        let mut a_meta = HashMap::new();
        a_meta.insert(LANCE_FIELD_ID_KEY.to_string(), "5".to_string());
        let arrow_a = ArrowField::new("a", ArrowDataType::Int32, false).with_metadata(a_meta);
        let arrow_b = ArrowField::new("b", ArrowDataType::Int32, false);
        let arrow_c = ArrowField::new("c", ArrowDataType::Int32, false);

        // struct s: manual parent + manual child x
        let mut s_meta = HashMap::new();
        s_meta.insert(LANCE_FIELD_ID_KEY.to_string(), "50".to_string());
        let mut x_meta = HashMap::new();
        x_meta.insert(LANCE_FIELD_ID_KEY.to_string(), "51".to_string());
        let arrow_s = ArrowField::new(
            "s",
            ArrowDataType::Struct(ArrowFields::from(vec![
                ArrowField::new("x", ArrowDataType::Int32, false).with_metadata(x_meta),
                ArrowField::new("y", ArrowDataType::Int32, false),
                ArrowField::new("z", ArrowDataType::Int32, true),
            ])),
            false,
        )
        .with_metadata(s_meta);

        // list l: parent manual, item inherit
        let mut l_meta = HashMap::new();
        l_meta.insert(LANCE_FIELD_ID_KEY.to_string(), "60".to_string());
        let arrow_l = ArrowField::new(
            "l",
            ArrowDataType::List(Arc::new(ArrowField::new(
                "item",
                ArrowDataType::Int32,
                true,
            ))),
            true,
        )
        .with_metadata(l_meta);

        // list l2: parent manual, item max_field_id (no base match)
        let mut l2_meta = HashMap::new();
        l2_meta.insert(LANCE_FIELD_ID_KEY.to_string(), "61".to_string());
        let arrow_l2 = ArrowField::new(
            "l2",
            ArrowDataType::List(Arc::new(ArrowField::new(
                "item",
                ArrowDataType::Int32,
                true,
            ))),
            true,
        )
        .with_metadata(l2_meta);

        // map m: parent manual, entries/key/value inherit
        let map_entries = ArrowField::new(
            "entries",
            ArrowDataType::Struct(ArrowFields::from(vec![
                ArrowField::new("key", ArrowDataType::Utf8, false),
                ArrowField::new("value", ArrowDataType::Int32, true),
            ])),
            false,
        );
        let mut m_meta = HashMap::new();
        m_meta.insert(LANCE_FIELD_ID_KEY.to_string(), "70".to_string());
        let arrow_m = ArrowField::new("m", ArrowDataType::Map(Arc::new(map_entries), false), true)
            .with_metadata(m_meta);

        // map m2: parent manual, entries/key/value max_field_id (no base match)
        let map_entries = ArrowField::new(
            "entries",
            ArrowDataType::Struct(ArrowFields::from(vec![
                ArrowField::new("key", ArrowDataType::Utf8, false),
                ArrowField::new("value", ArrowDataType::Int32, true),
            ])),
            false,
        );
        let mut m2_meta = HashMap::new();
        m2_meta.insert(LANCE_FIELD_ID_KEY.to_string(), "71".to_string());
        let arrow_m2 =
            ArrowField::new("m2", ArrowDataType::Map(Arc::new(map_entries), false), true)
                .with_metadata(m2_meta);

        let arrow_schema = ArrowSchema::new_with_metadata(
            vec![
                arrow_a, arrow_b, arrow_c, arrow_s, arrow_l, arrow_l2, arrow_m, arrow_m2,
            ],
            HashMap::from([("new_schema_k".to_string(), "new_schema_v".to_string())]),
        );

        let schema =
            LanceSchema::from_arrow_schema(&arrow_schema, Some(base_schema), Some(100)).unwrap();

        // 1. Manually specified field id
        let got_a = schema.field("a").unwrap();
        assert_eq!(got_a.id, 5);
        assert!(!got_a.metadata.contains_key(LANCE_FIELD_ID_KEY));

        // 2. Inherit field id + metadata from base_schema (field b)
        let got_b = schema.field("b").unwrap();
        assert_eq!(got_b.id, 11);

        // 3. Assign a new field id using max_field_id (field c)
        let got_c = schema.field("c").unwrap();
        assert_eq!(got_c.id, 101);

        // 4. struct: parent+child(x) manual, child(y) inherit, child(z) max_field_id
        let got_s = schema.field("s").unwrap();
        assert_eq!(got_s.id, 50);
        let got_sx = schema.field("s.x").unwrap();
        assert_eq!(got_sx.id, 51);
        let got_sy = schema.field("s.y").unwrap();
        assert_eq!(got_sy.id, 22);
        let got_sz = schema.field("s.z").unwrap();
        assert_eq!(got_sz.id, 102);

        // 5. list l: parent manual, item inherit
        let got_l = schema.field("l").unwrap();
        assert_eq!(got_l.id, 60);
        let got_li = schema.field("l.item").unwrap();
        assert_eq!(got_li.id, 31);

        // 6. list l2: parent manual, item max_field_id
        let got_l2 = schema.field("l2").unwrap();
        assert_eq!(got_l2.id, 61);
        let got_l2i = schema.field("l2.item").unwrap();
        assert_eq!(got_l2i.id, 103);

        // 7. map m: parent manual, entries/key/value inherit
        let got_m = schema.field("m").unwrap();
        assert_eq!(got_m.id, 70);
        let got_me = schema.field("m.entries").unwrap();
        assert_eq!(got_me.id, 41);
        let got_mk = schema.field("m.entries.key").unwrap();
        assert_eq!(got_mk.id, 42);
        let got_mv = schema.field("m.entries.value").unwrap();
        assert_eq!(got_mv.id, 43);

        // 8. map m2: parent manual, entries/key/value max_field_id
        let got_m2 = schema.field("m2").unwrap();
        assert_eq!(got_m2.id, 71);
        let got_m2e = schema.field("m2.entries").unwrap();
        assert_eq!(got_m2e.id, 104);
        let got_m2k = schema.field("m2.entries.key").unwrap();
        assert_eq!(got_m2k.id, 105);
        let got_m2v = schema.field("m2.entries.value").unwrap();
        assert_eq!(got_m2v.id, 106);

        // 9. Schema metadata: when new_schema.metadata is non-empty, use new_schema metadata
        assert_eq!(
            schema.metadata,
            HashMap::from([("new_schema_k".to_string(), "new_schema_v".to_string())])
        );
    }

    #[test]
    fn test_parse_storage_format_canonical_forms() {
        let cases = [
            ("2.0", LanceFileVersion::V2_0),
            ("2.1", LanceFileVersion::V2_1),
            ("2.2", LanceFileVersion::V2_2),
            ("2.3", LanceFileVersion::V2_3),
            ("0.1", LanceFileVersion::Legacy),
            ("legacy", LanceFileVersion::Legacy),
            ("stable", LanceFileVersion::Stable),
            ("next", LanceFileVersion::Next),
        ];
        for (input, expected) in cases {
            assert_eq!(
                parse_storage_format(input).unwrap(),
                expected,
                "parse_storage_format({:?}) failed",
                input
            );
        }
    }

    /// The `v`-prefixed spellings shipped in the `CommitBuilder.storageFormat`
    /// Javadoc and must keep working for existing Java callers.
    #[test]
    fn test_parse_storage_format_deprecated_aliases() {
        let cases = [
            ("v2_0", LanceFileVersion::V2_0),
            ("v2.0", LanceFileVersion::V2_0),
            ("v2_1", LanceFileVersion::V2_1),
            ("v2.1", LanceFileVersion::V2_1),
            ("v2_2", LanceFileVersion::V2_2),
            ("v2.2", LanceFileVersion::V2_2),
        ];
        for (input, expected) in cases {
            assert_eq!(
                parse_storage_format(input).unwrap(),
                expected,
                "parse_storage_format({:?}) failed",
                input
            );
        }
    }

    /// The alias set is frozen to what shipped, so versions added after the
    /// aliases were deprecated are reachable only by their canonical name.
    #[test]
    fn test_parse_storage_format_does_not_extend_aliases_to_new_versions() {
        assert!(parse_storage_format("v2_3").is_err());
        assert!(parse_storage_format("v2.3").is_err());
        assert_eq!(parse_storage_format("2.3").unwrap(), LanceFileVersion::V2_3);
    }

    #[test]
    fn test_parse_storage_format_case_insensitive() {
        assert_eq!(
            parse_storage_format("LEGACY").unwrap(),
            LanceFileVersion::Legacy
        );
        assert_eq!(
            parse_storage_format("Stable").unwrap(),
            LanceFileVersion::Stable
        );
        assert_eq!(
            parse_storage_format("V2_1").unwrap(),
            LanceFileVersion::V2_1
        );
    }

    #[test]
    fn test_parse_storage_format_rejects_invalid() {
        assert!(parse_storage_format("v3.0").is_err());
        assert!(parse_storage_format("").is_err());
        assert!(parse_storage_format("foo").is_err());
    }

    #[test]
    fn test_checked_transaction_integer_conversions() {
        assert_eq!(
            u32_to_jint("basePath.id", i32::MAX as u32).unwrap(),
            i32::MAX
        );
        assert!(u32_to_jint("basePath.id", i32::MAX as u32 + 1).is_err());
        assert_eq!(
            checked_field_ids("update.fieldsModified", &[0, u32::MAX as i64]).unwrap(),
            vec![0, u32::MAX]
        );
        for invalid in [-1, u32::MAX as i64 + 1] {
            let error = checked_field_ids("update.fieldsModified", &[invalid]).unwrap_err();
            let message = error.to_string();
            assert!(message.contains("update.fieldsModified[0]"));
            assert!(message.contains(&invalid.to_string()));
        }
        assert_eq!(
            u64_to_jlong("dataOverlay.fragmentId", i64::MAX as u64).unwrap(),
            i64::MAX
        );
        assert!(u64_to_jlong("dataOverlay.fragmentId", i64::MAX as u64 + 1).is_err());
    }
}
