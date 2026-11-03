// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{BlockingDataset, NATIVE_DATASET};
use crate::error::{Error, Result};
use crate::ffi::JNIEnvExt;
use crate::namespace::{
    BlockingDirectoryNamespace, BlockingRestNamespace, create_java_lance_namespace,
};
use crate::traits::{FromJObjectWithEnv, IntoJava};
use jni::JNIEnv;
use jni::objects::{JMap, JObject, JValue};
use jni::sys::jlong;
use lance::dataset::Version;
use lance::table::format::BasePath;
use lance::table::format::WriterVersion;
use lance_namespace::LanceNamespace;
use lance_table::io::commit::{ManifestLocation, ManifestNamingScheme};
use std::sync::Arc;

impl FromJObjectWithEnv<BasePath> for JObject<'_> {
    fn extract_object(&self, env: &mut JNIEnv<'_>) -> Result<BasePath> {
        let java_id = env.call_method(self, "getId", "()I", &[])?.i()?;
        let id = u32::try_from(java_id).map_err(|_| {
            Error::input_error(format!("BasePath.id must be non-negative, got {java_id}"))
        })?;
        let name = env.get_optional_string_from_method(self, "getName")?;
        let path = env.get_string_from_method(self, "getPath")?;
        let is_dataset_root = env.get_boolean_from_method(self, "isDatasetRoot")?;
        Ok(BasePath {
            id,
            name,
            path,
            is_dataset_root,
        })
    }
}

impl IntoJava for &BasePath {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let id = i32::try_from(self.id).map_err(|_| {
            Error::runtime_error(format!("Base path id {} exceeds Java int range", self.id))
        })?;
        let name = match self.name.as_ref() {
            Some(name) => env.new_string(name)?.into(),
            None => JObject::null(),
        };
        let name = env
            .call_static_method(
                "java/util/Optional",
                "ofNullable",
                "(Ljava/lang/Object;)Ljava/util/Optional;",
                &[JValue::Object(&name)],
            )?
            .l()?;
        let path = env.new_string(&self.path)?;
        Ok(env.new_object(
            "org/lance/BasePath",
            "(ILjava/util/Optional;Ljava/lang/String;Z)V",
            &[
                JValue::Int(id),
                JValue::Object(&name),
                JValue::Object(&path),
                JValue::Bool(self.is_dataset_root.into()),
            ],
        )?)
    }
}

impl IntoJava for BlockingDataset {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        attach_native_dataset(env, self)
    }
}

impl IntoJava for Version {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let timestamp_str = self.timestamp.to_rfc3339();
        let jtimestamp = env.new_string(timestamp_str)?;
        let zdt = env
            .call_static_method(
                "java/time/ZonedDateTime",
                "parse",
                "(Ljava/lang/CharSequence;)Ljava/time/ZonedDateTime;",
                &[JValue::Object(&jtimestamp)],
            )?
            .l()?;

        let jmap = env.new_object("java/util/TreeMap", "()V", &[])?;
        let map = JMap::from_env(env, &jmap)?;

        for (k, v) in self.metadata {
            let jkey = env.new_string(k)?;
            let jval = env.new_string(v)?;
            map.put(env, &jkey, &jval).expect("ERROR: calling jmap.put");
        }

        let java_version = env.new_object(
            "org/lance/Version",
            "(JLjava/time/ZonedDateTime;Ljava/util/TreeMap;)V",
            &[
                JValue::Long(self.version as i64),
                JValue::Object(&zdt),
                JValue::Object(&jmap),
            ],
        )?;
        Ok(java_version)
    }
}

impl IntoJava for WriterVersion {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let library = env.new_string(self.library)?;
        let version = env.new_string(self.version)?;
        let prerelease = match self.prerelease {
            Some(value) => JObject::from(env.new_string(value)?),
            None => JObject::null(),
        };
        let build_metadata = match self.build_metadata {
            Some(value) => JObject::from(env.new_string(value)?),
            None => JObject::null(),
        };

        Ok(env.new_object(
            "org/lance/WriterVersion",
            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V",
            &[
                JValue::Object(&library),
                JValue::Object(&version),
                JValue::Object(&prerelease),
                JValue::Object(&build_metadata),
            ],
        )?)
    }
}

impl IntoJava for ManifestLocation {
    fn into_java<'a>(self, env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
        let size = self.size.ok_or_else(|| {
            Error::runtime_error(format!("Manifest size is unavailable for {}", self.path))
        })?;
        let path = env.new_string(self.path.to_string())?;
        let naming_scheme = env.new_string(match self.naming_scheme {
            ManifestNamingScheme::V1 => "V1",
            ManifestNamingScheme::V2 => "V2",
        })?;
        let e_tag = match self.e_tag {
            Some(value) => JObject::from(env.new_string(value)?),
            None => JObject::null(),
        };
        Ok(env.new_object(
            "org/lance/ManifestLocation",
            "(JLjava/lang/String;JLjava/lang/String;Ljava/lang/String;)V",
            &[
                JValue::Long(self.version as i64),
                JValue::Object(&path),
                JValue::Long(size as i64),
                JValue::Object(&naming_scheme),
                JValue::Object(&e_tag),
            ],
        )?)
    }
}

fn attach_native_dataset<'local>(
    env: &mut JNIEnv<'local>,
    dataset: BlockingDataset,
) -> Result<JObject<'local>> {
    let j_dataset = create_java_dataset_object(env)?;
    // This block sets a native Rust object (dataset) as a field in the Java object (j_dataset).
    // Caution: This creates a potential for memory leaks. The Rust object (dataset) is not
    // automatically garbage-collected by Java, and its memory will not be freed unless
    // explicitly handled.
    //
    // To prevent memory leaks, ensure the following:
    // 1. The Java object (`j_dataset`) should implement the `java.io.Closeable` interface.
    // 2. Users of this Java object should be instructed to always use it within a try-with-resources
    //    statement (or manually call the `close()` method) to ensure that `self.close()` is invoked.
    unsafe { env.set_rust_field(&j_dataset, NATIVE_DATASET, dataset) }?;
    Ok(j_dataset)
}

fn create_java_dataset_object<'a>(env: &mut JNIEnv<'a>) -> Result<JObject<'a>> {
    let object = env.new_object("org/lance/Dataset", "()V", &[])?;
    Ok(object)
}

/// Check if the Java object is an instance of DirectoryNamespace.
fn is_directory_namespace(env: &mut JNIEnv, namespace_obj: &JObject) -> Result<bool> {
    let class = env
        .find_class("org/lance/namespace/DirectoryNamespace")
        .map_err(|e| {
            Error::runtime_error(format!("Failed to find DirectoryNamespace class: {}", e))
        })?;
    env.is_instance_of(namespace_obj, class)
        .map_err(|e| Error::runtime_error(format!("Failed to check instanceof: {}", e)))
}

/// Check if the Java object is an instance of RestNamespace.
fn is_rest_namespace(env: &mut JNIEnv, namespace_obj: &JObject) -> Result<bool> {
    let class = env
        .find_class("org/lance/namespace/RestNamespace")
        .map_err(|e| Error::runtime_error(format!("Failed to find RestNamespace class: {}", e)))?;
    env.is_instance_of(namespace_obj, class)
        .map_err(|e| Error::runtime_error(format!("Failed to check instanceof: {}", e)))
}

/// Get the native handle from a Java LanceNamespace object.
fn get_native_namespace_handle(env: &mut JNIEnv, namespace_obj: &JObject) -> Result<jlong> {
    env.call_method(namespace_obj, "getNativeHandle", "()J", &[])
        .map_err(|e| Error::runtime_error(format!("Failed to call getNativeHandle: {}", e)))?
        .j()
        .map_err(|e| Error::runtime_error(format!("getNativeHandle did not return a long: {}", e)))
}

/// Extract namespace and table_id from Java objects into Rust types.
///
/// Returns `None` if `namespace_obj` is null, otherwise returns the namespace
/// and table_id pair.
#[allow(clippy::type_complexity)]
pub(crate) fn extract_namespace_info(
    env: &mut JNIEnv,
    namespace_obj: &JObject,
    table_id_obj: &JObject,
) -> Result<Option<(Arc<dyn LanceNamespace>, Vec<String>)>> {
    if namespace_obj.is_null() {
        return Ok(None);
    }

    let namespace_client: Arc<dyn LanceNamespace> = if is_directory_namespace(env, namespace_obj)? {
        let native_handle = get_native_namespace_handle(env, namespace_obj)?;
        let dir_namespace_client =
            unsafe { &*(native_handle as *const BlockingDirectoryNamespace) };
        dir_namespace_client.inner.clone()
    } else if is_rest_namespace(env, namespace_obj)? {
        let native_handle = get_native_namespace_handle(env, namespace_obj)?;
        let rest_namespace_client = unsafe { &*(native_handle as *const BlockingRestNamespace) };
        rest_namespace_client.inner.clone()
    } else {
        create_java_lance_namespace(env, namespace_obj)?
    };

    let table_id = env.get_strings(table_id_obj)?;
    Ok(Some((namespace_client, table_id)))
}
