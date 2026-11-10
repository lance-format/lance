// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{BlockingDataset, NATIVE_DATASET};
use crate::block_on;
use crate::error::Result;
use crate::ffi::JNIEnvExt;
use crate::traits::{FromJObjectWithEnv, FromJString, IntoJava, import_vec};
use crate::utils::{extract_storage_options, to_rust_map};
use arrow::datatypes::Schema;
use arrow::ffi::FFI_ArrowSchema;
use jni::JNIEnv;
use jni::objects::{JMap, JObject, JString};
use jni::sys::jlong;
use lance::dataset::CommitBuilder;
use lance::dataset::transaction::{Operation, Transaction};
use lance::io::ObjectStoreParams;
use lance_core::datatypes::Schema as LanceSchema;
use lance_file::version::LanceFileVersion;
use lance_table::io::commit::CommitHandler;
use std::sync::Arc;

impl BlockingDataset {
    #[allow(clippy::too_many_arguments)]
    pub fn commit_transaction(
        &mut self,
        transaction: Transaction,
        store_params: ObjectStoreParams,
        detached: bool,
        enable_v2_manifest_paths: bool,
        use_stable_row_ids: Option<bool>,
        storage_format: Option<LanceFileVersion>,
        max_retries: u32,
        skip_auto_cleanup: bool,
        commit_handler: Option<Arc<dyn CommitHandler>>,
        commit_timeout: Option<std::time::Duration>,
    ) -> Result<Self> {
        let mut builder = CommitBuilder::new(Arc::new(self.clone().inner))
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
        if let Some(handler) = commit_handler {
            builder = builder.with_commit_handler(handler);
        }
        let new_dataset = block_on(builder.execute(transaction))?;
        Ok(BlockingDataset { inner: new_dataset })
    }

    pub fn read_transaction(&self) -> Result<Option<Transaction>> {
        let transaction = block_on(self.inner.read_transaction())?;
        Ok(transaction)
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_commitAppend<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<FragmentMetadata>
    storage_options_obj: JObject, // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_commit_append(
            &mut env,
            path,
            read_version_obj,
            fragments_obj,
            storage_options_obj
        )
    )
}

pub fn inner_commit_append<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    read_version_obj: JObject,    // Optional<Long>
    fragment_objs: JObject,       // List<FragmentMetadata>
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JObject<'local>> {
    let fragment_objs = import_vec(env, &fragment_objs)?;
    let mut fragments = Vec::with_capacity(fragment_objs.len());
    for f in fragment_objs {
        fragments.push(f.extract_object(env)?);
    }
    let op = Operation::Append { fragments };
    let path_str = path.extract(env)?;
    let read_version = env.get_u64_opt(&read_version_obj)?;
    let storage_options = extract_storage_options(env, &storage_options_obj)?;
    let dataset = BlockingDataset::commit(&path_str, op, read_version, storage_options)?;
    dataset.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_commitOverwrite<'local>(
    mut env: JNIEnv<'local>,
    _obj: JObject,
    path: JString,
    arrow_schema_addr: jlong,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<FragmentMetadata>
    storage_options_obj: JObject, // Map<String, String>
) -> JObject<'local> {
    ok_or_throw!(
        env,
        inner_commit_overwrite(
            &mut env,
            path,
            arrow_schema_addr,
            read_version_obj,
            fragments_obj,
            storage_options_obj
        )
    )
}

pub fn inner_commit_overwrite<'local>(
    env: &mut JNIEnv<'local>,
    path: JString,
    arrow_schema_addr: jlong,
    read_version_obj: JObject,    // Optional<Long>
    fragments_obj: JObject,       // List<FragmentMetadata>
    storage_options_obj: JObject, // Map<String, String>
) -> Result<JObject<'local>> {
    let fragment_objs = import_vec(env, &fragments_obj)?;
    let mut fragments = Vec::with_capacity(fragment_objs.len());
    for f in fragment_objs {
        fragments.push(f.extract_object(env)?);
    }
    let c_schema_ptr = arrow_schema_addr as *mut FFI_ArrowSchema;
    let c_schema = unsafe { FFI_ArrowSchema::from_raw(c_schema_ptr) };
    let arrow_schema = Schema::try_from(&c_schema)?;
    let schema = LanceSchema::try_from(&arrow_schema)?;

    let op = Operation::Overwrite {
        fragments,
        schema,
        config_upsert_values: None,
        initial_bases: None,
    };
    let path_str = path.extract(env)?;
    let read_version = env.get_u64_opt(&read_version_obj)?;
    let jmap = JMap::from_env(env, &storage_options_obj)?;
    let storage_options = to_rust_map(env, &jmap)?;
    let dataset = BlockingDataset::commit(&path_str, op, read_version, storage_options)?;
    dataset.into_java(env)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeDelete(
    mut env: JNIEnv,
    java_dataset: JObject,
    predicate: JString,
) {
    ok_or_throw_without_return!(env, inner_delete(&mut env, java_dataset, predicate))
}

fn inner_delete(env: &mut JNIEnv, java_dataset: JObject, predicate: JString) -> Result<()> {
    let predicate_str = predicate.extract(env)?;
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    block_on(dataset_guard.inner.delete(&predicate_str))?;
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_Dataset_nativeTruncateTable(
    mut env: JNIEnv,
    java_dataset: JObject,
) {
    ok_or_throw_without_return!(env, inner_truncate_table(&mut env, java_dataset))
}

fn inner_truncate_table(env: &mut JNIEnv, java_dataset: JObject) -> Result<()> {
    let mut dataset_guard =
        unsafe { env.get_rust_field::<_, _, BlockingDataset>(java_dataset, NATIVE_DATASET) }?;
    block_on(dataset_guard.inner.truncate_table())?;
    Ok(())
}
