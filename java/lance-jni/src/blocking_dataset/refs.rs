// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{BlockingDataset, NATIVE_DATASET, clone_dataset};
use crate::block_on;
use crate::error::Result;
use crate::ffi::JNIEnvExt;
use crate::traits::{FromJString, IntoJava};
use crate::utils::{to_java_map, to_rust_map};
use chrono::{DateTime, Utc};
use jni::JNIEnv;
use jni::objects::{JMap, JObject, JString, JValue};
use jni::sys::jlong;
use lance::dataset::refs::{Ref, TagContents};
use lance::io::ObjectStoreParams;
use std::collections::HashMap;
use std::sync::Arc;

impl BlockingDataset {
    pub fn checkout_version(&mut self, version: u64) -> Result<Self> {
        let inner = block_on(self.inner.checkout_version(version))?;
        Ok(Self { inner })
    }

    pub fn checkout_tag(&mut self, tag: &str) -> Result<Self> {
        let inner = block_on(self.inner.checkout_version(tag))?;
        Ok(Self { inner })
    }

    pub fn checkout_latest(&mut self) -> Result<()> {
        block_on(self.inner.checkout_latest())?;
        Ok(())
    }

    pub fn restore(&mut self) -> Result<()> {
        block_on(self.inner.restore())?;
        Ok(())
    }

    pub fn list_tags(&self) -> Result<HashMap<String, TagContents>> {
        let tags = block_on(self.inner.tags().list())?;
        Ok(tags)
    }

    pub fn list_branches(&self) -> Result<HashMap<String, lance::dataset::refs::BranchContents>> {
        let branches = block_on(self.inner.branches().list())?;
        Ok(branches)
    }

    pub fn delete_branch(&mut self, branch: &str) -> Result<()> {
        block_on(self.inner.branches().delete(branch, true))?;
        Ok(())
    }

    pub fn checkout_reference(
        &mut self,
        branch: Option<String>,
        version: Option<u64>,
        tag: Option<String>,
    ) -> Result<Self> {
        let reference = if let Some(tag_name) = tag {
            Ref::from(tag_name.as_str())
        } else {
            Ref::Version(branch, version)
        };
        let inner = block_on(self.inner.checkout_version(reference))?;
        Ok(Self { inner })
    }

    pub fn create_tag(&mut self, tag: &str, reference: Ref) -> Result<()> {
        block_on(self.inner.tags().create(tag, reference))?;
        Ok(())
    }

    pub fn delete_tag(&mut self, tag: &str) -> Result<()> {
        block_on(self.inner.tags().delete(tag))?;
        Ok(())
    }

    pub fn update_tag(&mut self, tag: &str, reference: Ref) -> Result<()> {
        block_on(self.inner.tags().update(tag, reference))?;
        Ok(())
    }

    pub fn replace_tag_metadata(
        &mut self,
        tag: &str,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        block_on(self.inner.tags().replace_metadata(tag, metadata))?;
        Ok(())
    }

    pub fn replace_branch_metadata(
        &mut self,
        branch: &str,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        block_on(self.inner.branches().replace_metadata(branch, metadata))?;
        Ok(())
    }

    pub fn get_version(&self, tag: &str) -> Result<u64> {
        let version = block_on(self.inner.tags().get_version(tag))?;
        Ok(version)
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCheckoutLatest(
    mut env: JNIEnv,
    java_dataset: JObject,
) {
    ok_or_throw_without_return!(env, inner_checkout_latest(&mut env, java_dataset));
}

fn inner_checkout_latest(env: &mut JNIEnv, java_dataset: JObject) -> Result<()> {
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.checkout_latest()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCheckoutVersion<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    version: jlong,
) -> JObject<'local> {
    ok_or_throw!(env, inner_checkout_version(&mut env, java_dataset, version))
}

fn inner_checkout_version<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    version: jlong,
) -> Result<JObject<'local>> {
    let new_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.checkout_version(version as u64)?
    };

    new_dataset.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCheckoutTag<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jtag: JString,
) -> JObject<'local> {
    ok_or_throw!(env, inner_checkout_tag(&mut env, java_dataset, jtag))
}

fn inner_checkout_tag<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jtag_name: JString,
) -> Result<JObject<'local>> {
    let tag_name = jtag_name.extract(env)?;
    let new_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.checkout_tag(tag_name.as_str())?
    };

    new_dataset.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeRestore(
    mut env: JNIEnv,
    java_dataset: JObject,
) {
    ok_or_throw_without_return!(env, inner_restore(&mut env, java_dataset))
}

fn inner_restore(env: &mut JNIEnv, java_dataset: JObject) -> Result<()> {
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.restore()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeShallowClone<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    target_path: JString,
    reference: JObject,
    storage_options: JObject,
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_shallow_clone(
            &mut env,
            java_dataset,
            target_path,
            reference,
            storage_options
        )
    )
}

fn inner_shallow_clone<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    target_path: JString,
    jref: JObject,
    storage_options: JObject,
) -> Result<JObject<'local>> {
    let target_path_str = target_path.extract(env)?;
    let reference = transform_jref_to_ref(jref, env)?;
    let storage_opts = transform_jstorage_options(storage_options, env)?;
    let new_ds = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        block_on(dataset_guard.inner.shallow_clone(
            target_path_str.as_str(),
            reference,
            storage_opts,
        ))?
    };

    BlockingDataset { inner: new_ds }.into_java(env)
}

//////////////////////////////
fn optional_datetime_to_java_instant<'local>(
    env: &mut JNIEnv<'local>,
    timestamp: Option<&DateTime<Utc>>,
) -> Result<JObject<'local>> {
    if let Some(timestamp) = timestamp {
        let seconds = timestamp.timestamp();
        let nanos = timestamp.timestamp_subsec_nanos() as i64;
        Ok(env
            .call_static_method(
                "java/time/Instant",
                "ofEpochSecond",
                "(JJ)Ljava/time/Instant;",
                &[JValue::Long(seconds), JValue::Long(nanos)],
            )?
            .l()?)
    } else {
        Ok(JObject::null())
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeListTags<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_list_tags(&mut env, java_dataset))
}

fn inner_list_tags<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let mut tags: Vec<_> = {
        let dataset = clone_dataset(env, &java_dataset)?;
        dataset.list_tags()?.into_iter().collect()
    };
    tags.sort_unstable_by(|(left_name, _), (right_name, _)| left_name.cmp(right_name));
    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;

    for (tag_name, tag_contents) in tags {
        let branch_name: JObject = if let Some(branch_name) = tag_contents.branch.as_ref() {
            env.new_string(branch_name)?.into()
        } else {
            JObject::null()
        };
        let created_at = optional_datetime_to_java_instant(env, tag_contents.created_at.as_ref())?;
        let updated_at = optional_datetime_to_java_instant(env, tag_contents.updated_at.as_ref())?;
        let java_metadata = to_java_map(env, &tag_contents.metadata)?;
        let java_tag = env.new_object(
            "org/lance/Tag",
            "(Ljava/lang/String;Ljava/lang/String;JILjava/time/Instant;Ljava/time/Instant;Ljava/util/Map;)V",
            &[
                JValue::Object(&env.new_string(tag_name)?.into()),
                JValue::Object(&branch_name),
                JValue::Long(tag_contents.version as i64),
                JValue::Int(tag_contents.manifest_size as i32),
                JValue::Object(&created_at),
                JValue::Object(&updated_at),
                JValue::Object(&java_metadata),
            ],
        )?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&java_tag)],
        )?;
    }
    Ok(array_list)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCreateTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jref: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_create_tag(&mut env, java_dataset, jtag_name, jref)
    )
}

fn inner_create_tag(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jref: JObject,
) -> Result<()> {
    let tag = jtag_name.extract(env)?;
    let reference = transform_jref_to_ref(jref, env)?;
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.create_tag(tag.as_str(), reference)?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeDeleteTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
) {
    ok_or_throw_without_return!(env, inner_delete_tag(&mut env, java_dataset, jtag_name))
}

fn inner_delete_tag(env: &mut JNIEnv, java_dataset: JObject, jtag_name: JString) -> Result<()> {
    let tag = { jtag_name.extract(env)? };
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.delete_tag(tag.as_str())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeUpdateTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jref: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_update_tag(&mut env, java_dataset, jtag_name, jref)
    )
}

fn inner_update_tag(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jref: JObject,
) -> Result<()> {
    let tag = jtag_name.extract(env)?;
    let reference = transform_jref_to_ref(jref, env)?;
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.update_tag(tag.as_str(), reference)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeReplaceTagMetadata(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jmetadata: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_replace_tag_metadata(&mut env, java_dataset, jtag_name, jmetadata)
    )
}

fn inner_replace_tag_metadata(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
    jmetadata: JObject,
) -> Result<()> {
    let tag = jtag_name.extract(env)?;
    let metadata = extract_metadata_map(env, &jmetadata)?;
    let mut dataset_guard =
        { unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }? };
    dataset_guard.replace_tag_metadata(tag.as_str(), metadata)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeGetVersionByTag(
    mut env: JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        inner_get_version_by_tag(&mut env, java_dataset, jtag_name),
        -1
    ) as jlong
}

fn inner_get_version_by_tag(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jtag_name: JString,
) -> Result<u64> {
    let tag = { jtag_name.extract(env)? };
    let dataset = clone_dataset(env, &java_dataset)?;
    dataset.get_version(tag.as_str())
}

//////////////////////////////
#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeListBranches<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
) -> JObject<'local> {
    ok_or_throw!(env, inner_list_branches(&mut env, java_dataset))
}

fn inner_list_branches<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
) -> Result<JObject<'local>> {
    let mut branches: Vec<_> = {
        let dataset = clone_dataset(env, &java_dataset)?;
        dataset.list_branches()?.into_iter().collect()
    };
    branches.sort_unstable_by(|(left_name, _), (right_name, _)| left_name.cmp(right_name));
    let array_list = env.new_object("java/util/ArrayList", "()V", &[])?;

    for (name, contents) in branches {
        let jname = env.new_string(name)?;
        let jparent = if let Some(p) = contents.parent_branch {
            env.new_string(p)?.into()
        } else {
            JObject::null()
        };
        let jbranch_identifier = env.new_object(
            "java/util/ArrayList",
            "(I)V",
            &[JValue::Int(contents.identifier.version_mapping.len() as i32)],
        )?;
        for (version, uuid) in contents.identifier.version_mapping.iter() {
            let juuid = env.new_string(uuid)?;
            let jmapping = env.new_object(
                "org/lance/Branch$BranchVersionMapping",
                "(JLjava/lang/String;)V",
                &[JValue::Long(*version as i64), JValue::Object(&juuid)],
            )?;
            env.call_method(
                &jbranch_identifier,
                "add",
                "(Ljava/lang/Object;)Z",
                &[JValue::Object(&jmapping)],
            )?;
        }
        let java_metadata = to_java_map(env, &contents.metadata)?;
        let jbranch = env.new_object(
            "org/lance/Branch",
            "(Ljava/lang/String;Ljava/lang/String;Ljava/util/List;JJILjava/util/Map;)V",
            &[
                JValue::Object(&jname),
                JValue::Object(&jparent),
                JValue::Object(&jbranch_identifier),
                JValue::Long(contents.parent_version as i64),
                JValue::Long(contents.create_at as i64),
                JValue::Int(contents.manifest_size as i32),
                JValue::Object(&java_metadata),
            ],
        )?;
        env.call_method(
            &array_list,
            "add",
            "(Ljava/lang/Object;)Z",
            &[JValue::Object(&jbranch)],
        )?;
    }
    Ok(array_list)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCreateBranch<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    jbranch: JString,
    jref: JObject,
    jstorage_options: JObject, // Optional<String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_create_branch(&mut env, java_dataset, jbranch, jref, jstorage_options)
    )
}

fn inner_create_branch<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    jbranch: JString,
    jref: JObject,
    jstorage_options: JObject, // Optional<String>
) -> Result<JObject<'local>> {
    let branch_name: String = jbranch.extract(env)?;
    let reference = transform_jref_to_ref(jref, env)?;
    let storage_opts = transform_jstorage_options(jstorage_options, env)?;

    let new_blocking_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        let inner = block_on(dataset_guard.inner.create_branch(
            branch_name.as_str(),
            reference,
            storage_opts,
        ))?;
        BlockingDataset { inner }
    };
    new_blocking_dataset.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeReplaceBranchMetadata(
    mut env: JNIEnv,
    java_dataset: JObject,
    jbranch: JString,
    jmetadata: JObject,
) {
    ok_or_throw_without_return!(
        env,
        inner_replace_branch_metadata(&mut env, java_dataset, jbranch, jmetadata)
    )
}

fn inner_replace_branch_metadata(
    env: &mut JNIEnv,
    java_dataset: JObject,
    jbranch: JString,
    jmetadata: JObject,
) -> Result<()> {
    let branch: String = jbranch.extract(env)?;
    let metadata = extract_metadata_map(env, &jmetadata)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.replace_branch_metadata(&branch, metadata)
}

fn extract_metadata_map(env: &mut JNIEnv, jmetadata: &JObject) -> Result<HashMap<String, String>> {
    let jmap = JMap::from_env(env, jmetadata)?;
    to_rust_map(env, &jmap)
}

pub(super) fn transform_jref_to_ref(jref: JObject, env: &mut JNIEnv) -> Result<Ref> {
    let source_tag_name = env.get_optional_string_from_method(&jref, "getTagName")?;
    let source_version_number = env.get_optional_u64_from_method(&jref, "getVersionNumber")?;
    let source_branch = env.get_optional_string_from_method(&jref, "getBranchName")?;
    if let Some(tag_name) = source_tag_name {
        Ok(Ref::Tag(tag_name))
    } else {
        Ok(Ref::Version(source_branch, source_version_number))
    }
}

fn transform_jstorage_options(
    jstorage_options: JObject,
    env: &mut JNIEnv,
) -> Result<Option<ObjectStoreParams>> {
    let storage_options = env.get_optional(&jstorage_options, |env, map_obj| {
        let jmap = JMap::from_env(env, &map_obj)?;
        to_rust_map(env, &jmap)
    })?;
    Ok(storage_options
        .map(|options| {
            Some(ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    lance::io::StorageOptionsAccessor::with_static_options(options),
                )),
                ..Default::default()
            })
        })
        .unwrap_or(None))
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeDeleteBranch(
    mut env: JNIEnv,
    java_dataset: JObject,
    jbranch: JString,
) {
    ok_or_throw_without_return!(env, inner_delete_branch(&mut env, java_dataset, jbranch))
}

fn inner_delete_branch(env: &mut JNIEnv, java_dataset: JObject, jbranch: JString) -> Result<()> {
    let branch_name: String = jbranch.extract(env)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    dataset_guard.delete_branch(branch_name.as_str())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeCheckout<'local>(
    mut env: JNIEnv<'local>,
    java_dataset: JObject,
    reference_obj: JObject, // Reference
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_checkout_ref(&mut env, java_dataset, reference_obj)
    )
}

fn inner_checkout_ref<'local>(
    env: &mut JNIEnv<'local>,
    java_dataset: JObject,
    reference_obj: JObject, // Reference
) -> Result<JObject<'local>> {
    // Extract Optional fields from Reference
    let branch_opt_obj = env
        .call_method(
            &reference_obj,
            "getBranchName",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let version_opt_obj = env
        .call_method(
            &reference_obj,
            "getVersionNumber",
            "()Ljava/util/Optional;",
            &[],
        )?
        .l()?;
    let tag_opt_obj = env
        .call_method(&reference_obj, "getTagName", "()Ljava/util/Optional;", &[])?
        .l()?;

    let branch_opt = env.get_string_opt(&branch_opt_obj)?;
    let version_opt = env.get_u64_opt(&version_opt_obj)?;
    let tag_opt = env.get_string_opt(&tag_opt_obj)?;

    let new_dataset = {
        let mut dataset_guard =
            unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
        dataset_guard.checkout_reference(branch_opt, version_opt, tag_opt)?
    };
    new_dataset.into_java(env)
}
