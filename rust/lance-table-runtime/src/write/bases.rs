// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use lance_core::{Error, Result, datatypes::Schema};
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_table::format::BasePath;
use object_store::path::Path;

use super::{WriteMode, WriteParams, WriteParamsWithSession};
use crate::blob::{ExternalBaseCandidate, ExternalBaseResolver};
use crate::{Table, TableSession, base::log_unregistered_base_scoped_options};

/// Reserved base id that refers to the dataset's primary storage in
/// [`WriteParams::target_bases`]. Real base ids are assigned starting from 1,
/// so 0 is never a registered base. Files written through a primary slot
/// carry no base id, exactly like a write without target bases.
pub const PRIMARY_BASE_ID: u32 = 0;

/// Assign IDs to initial bases whose ID is zero.
///
/// Zero is both the automatic-assignment marker for `initial_bases` and the
/// ID that selects primary storage in `target_bases`, so a registered base
/// must never take it. Explicit IDs are reserved before any automatic ID is
/// chosen, which keeps the result independent of the order the caller lists
/// the bases in. Existing manifest bindings are reserved for the same reason.
///
/// Automatic IDs start after the highest existing ID, or at 1 when there are
/// no existing bases, so previously valid requests keep the IDs they received.
/// That range skips explicit IDs; once it is exhausted, allocation falls back
/// to the lowest free ID.
fn assign_initial_base_ids(
    initial_bases: &mut [BasePath],
    existing_base_paths: Option<&HashMap<u32, BasePath>>,
) -> Result<()> {
    let mut used_ids: BTreeSet<u32> = existing_base_paths
        .map(|bases| bases.keys().copied().collect())
        .unwrap_or_default();
    used_ids.insert(PRIMARY_BASE_ID);

    let mut explicit_ids = BTreeSet::new();
    for base_path in initial_bases.iter() {
        if base_path.id == PRIMARY_BASE_ID {
            continue;
        }
        if let Some(existing) = existing_base_paths.and_then(|bases| bases.get(&base_path.id)) {
            return Err(Error::invalid_input(format!(
                "Cannot register base ID {} for '{}': it is already bound to '{}'",
                base_path.id, base_path.path, existing.path
            )));
        }
        if !explicit_ids.insert(base_path.id) {
            return Err(Error::invalid_input(format!(
                "Duplicate explicit base ID {} in initial_bases. Base IDs must be unique.",
                base_path.id
            )));
        }
    }

    used_ids.extend(explicit_ids);
    let mut next_above_max = existing_base_paths
        .and_then(|bases| bases.keys().max().copied())
        .map_or(Some(1), |max_id| max_id.checked_add(1));
    for base_path in initial_bases.iter_mut() {
        if base_path.id != PRIMARY_BASE_ID {
            continue;
        }
        let id = match next_above_max.and_then(|start| first_unused_at_or_above(start, &used_ids)) {
            Some(id) => {
                next_above_max = id.checked_add(1);
                id
            }
            None => {
                // No ID above the highest existing one remains, so the free
                // IDs can only be below that range.
                next_above_max = None;
                BasePath::unused_id(used_ids.iter().copied())?
            }
        };
        base_path.id = id;
        used_ids.insert(id);
    }
    Ok(())
}

/// Find the first ID at or above `start` that is not in `used_ids`.
///
/// Returns `None` when every ID from `start` through `u32::MAX` is used.
fn first_unused_at_or_above(start: u32, used_ids: &BTreeSet<u32>) -> Option<u32> {
    let mut candidate = start;
    for &used_id in used_ids.range(start..) {
        if used_id != candidate {
            return Some(candidate);
        }
        candidate = candidate.checked_add(1)?;
    }
    Some(candidate)
}

pub async fn validate_and_resolve_target_bases<S: AsRef<TableSession> + Clone>(
    params: &mut WriteParamsWithSession<S>,
    existing_base_paths: Option<&HashMap<u32, BasePath>>,
) -> Result<Option<Vec<TargetBaseInfo>>> {
    // Step 1: Validations
    if !matches!(params.mode, WriteMode::Create) && params.initial_bases.is_some() {
        return Err(Error::invalid_input(format!(
            "Cannot register new bases in {:?} mode. Only CREATE mode can register new bases.",
            params.mode
        )));
    }

    if params.target_base_names_or_paths.is_some() && params.target_bases.is_some() {
        return Err(Error::invalid_input(
            "Cannot specify both target_base_names_or_paths and target_bases. Use one or the other.",
        ));
    }

    if params.target_all_bases.is_some() {
        return Err(Error::invalid_input(
            "target_all_bases requires dataset context to resolve; use the write or merge insert APIs to apply it.",
        ));
    }

    // Step 2: Assign IDs to initial_bases and add them to all_bases
    let mut all_bases: HashMap<u32, BasePath> = existing_base_paths.cloned().unwrap_or_default();
    if let Some(initial_bases) = &mut params.initial_bases {
        assign_initial_base_ids(initial_bases, existing_base_paths)?;
        for base_path in initial_bases.iter() {
            all_bases.insert(base_path.id, base_path.clone());
        }
    }
    log_unregistered_base_scoped_options(
        params.store_params.as_ref(),
        &all_bases,
        log::Level::Warn,
    );

    // Step 3: Resolve target_base_names_or_paths to IDs
    let target_base_ids = if let Some(ref names_or_paths) = params.target_base_names_or_paths {
        let mut resolved_ids = Vec::new();
        for reference in names_or_paths {
            let ref_str = reference.as_str();
            let id = all_bases
                .iter()
                .find(|(_, base)| {
                    base.name.as_ref().map(|n| n == ref_str).unwrap_or(false)
                        || base.path == ref_str
                })
                .map(|(&id, _)| id)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Base reference '{}' not found in available bases",
                        ref_str
                    ))
                })?;

            resolved_ids.push(id);
        }
        Some(resolved_ids)
    } else {
        params.target_bases.clone()
    };

    // Step 4: Prepare TargetBaseInfo structs
    let store_registry = params.store_registry();

    if let Some(target_bases) = &target_base_ids {
        // An empty list would panic in round-robin selection; reject it
        // instead of silently writing to primary storage.
        if target_bases.is_empty() {
            return Err(Error::invalid_input(
                "target_bases cannot be empty. Omit the option to write to primary storage.",
            ));
        }
        let mut bases_info = Vec::new();

        for &target_base_id in target_bases {
            let base_path = all_bases.get(&target_base_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Target base ID {} not found in available bases",
                    target_base_id
                ))
            })?;

            let store_params = write_store_params_for_base(params, base_path);
            let (target_object_store, extracted_path) = ObjectStore::from_uri_and_params(
                store_registry.clone(),
                &base_path.path,
                &store_params,
            )
            .await?;

            bases_info.push(TargetBaseInfo {
                base_id: target_base_id,
                object_store: target_object_store,
                base_dir: extracted_path,
                is_dataset_root: base_path.is_dataset_root,
            });
        }

        Ok(Some(bases_info))
    } else {
        Ok(None)
    }
}

/// Like [`validate_and_resolve_target_bases`], but also resolves references to
/// the dataset's primary storage: base id [`PRIMARY_BASE_ID`] in
/// `target_bases`, or an entry equal to `primary_uri` in
/// `target_base_names_or_paths`. Primary slots participate in the round-robin
/// like any other target base; files written through them carry no base id.
pub async fn validate_and_resolve_target_bases_with_primary<S: AsRef<TableSession> + Clone>(
    params: &mut WriteParamsWithSession<S>,
    existing_base_paths: Option<&HashMap<u32, BasePath>>,
    primary_object_store: &Arc<ObjectStore>,
    primary_base_dir: &Path,
    primary_uri: &str,
) -> Result<Option<Vec<TargetBaseInfo>>> {
    // Expand an all-bases request into an explicit id list (primary first,
    // then registered bases in ascending id order) and continue below.
    if let Some(include_primary) = params.target_all_bases {
        if params.target_bases.is_some() || params.target_base_names_or_paths.is_some() {
            return Err(Error::invalid_input(
                "Cannot specify target_all_bases together with target_bases or target_base_names_or_paths.",
            ));
        }
        let mut ids: Vec<u32> = existing_base_paths
            .map(|bases| bases.keys().copied().collect())
            .unwrap_or_default();
        // CREATE mode registers initial_bases in the same write; assign their
        // ids here (reserving explicit ids before automatic ones) so they join
        // the rotation.
        if let Some(initial_bases) = &mut params.initial_bases {
            assign_initial_base_ids(initial_bases, existing_base_paths)?;
            ids.extend(initial_bases.iter().map(|base_path| base_path.id));
        }
        ids.sort_unstable();
        ids.dedup();
        if include_primary {
            ids.insert(0, PRIMARY_BASE_ID);
        }
        if ids.is_empty() {
            return Err(Error::invalid_input(
                "target_all_bases found no registered bases and include_primary is false. \
                 Register bases or include primary storage.",
            ));
        }
        params.target_bases = Some(ids);
        params.target_all_bases = None;
    }

    let has_primary_ids = params
        .target_bases
        .as_ref()
        .is_some_and(|ids| ids.contains(&PRIMARY_BASE_ID));
    let has_primary_refs = params
        .target_base_names_or_paths
        .as_ref()
        .is_some_and(|refs| refs.iter().any(|r| r == primary_uri));
    if !has_primary_ids && !has_primary_refs {
        return validate_and_resolve_target_bases(params, existing_base_paths).await;
    }

    // The delegate below may be skipped when only primary slots remain, so
    // validate mutual exclusion here as well.
    if params.target_base_names_or_paths.is_some() && params.target_bases.is_some() {
        return Err(Error::invalid_input(
            "Cannot specify both target_base_names_or_paths and target_bases. Use one or the other.",
        ));
    }

    // Strip the primary slots, resolve the remaining references through the
    // normal path, then splice the primary slots back into their original
    // positions so the round-robin order matches what the caller asked for.
    let is_primary_slot: Vec<bool> = if let Some(ids) = &params.target_bases {
        ids.iter().map(|id| *id == PRIMARY_BASE_ID).collect()
    } else {
        params
            .target_base_names_or_paths
            .as_ref()
            .unwrap()
            .iter()
            .map(|r| r == primary_uri)
            .collect()
    };

    let mut shim = params.clone();
    if let Some(ids) = &params.target_bases {
        let rest: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| *id != PRIMARY_BASE_ID)
            .collect();
        shim.target_bases = if rest.is_empty() { None } else { Some(rest) };
    } else {
        let rest: Vec<String> = params
            .target_base_names_or_paths
            .as_ref()
            .unwrap()
            .iter()
            .filter(|r| *r != primary_uri)
            .cloned()
            .collect();
        shim.target_base_names_or_paths = if rest.is_empty() { None } else { Some(rest) };
    }

    let resolved_rest = validate_and_resolve_target_bases(&mut shim, existing_base_paths).await?;
    // The delegate assigns ids to initial_bases in place; propagate that side
    // effect back so CREATE-mode transactions register properly assigned ids.
    params.initial_bases = shim.initial_bases;

    let mut rest_iter = resolved_rest.unwrap_or_default().into_iter();
    let mut bases_info = Vec::with_capacity(is_primary_slot.len());
    for is_primary in is_primary_slot {
        if is_primary {
            bases_info.push(TargetBaseInfo {
                base_id: PRIMARY_BASE_ID,
                object_store: primary_object_store.clone(),
                base_dir: primary_base_dir.clone(),
                is_dataset_root: true,
            });
        } else {
            bases_info.push(rest_iter.next().ok_or_else(|| {
                Error::internal("target base resolution returned fewer bases than requested")
            })?);
        }
    }
    Ok(Some(bases_info))
}

fn append_external_base_candidate(
    base_path: &BasePath,
    store_prefix: String,
    extracted_path: Path,
    store_params: ObjectStoreParams,
    candidates: &mut Vec<ExternalBaseCandidate>,
    seen_base_ids: &mut HashSet<u32>,
) {
    if base_path.is_dataset_root {
        return;
    }
    if seen_base_ids.insert(base_path.id) {
        candidates.push(ExternalBaseCandidate {
            base_id: base_path.id,
            store_prefix,
            base_path: extracted_path,
            store_params,
        });
    }
}

fn write_store_params_for_base<S>(
    params: &WriteParamsWithSession<S>,
    base_path: &BasePath,
) -> ObjectStoreParams {
    // Exact per-URI bindings are used as-is. Otherwise the write-level default
    // params are resolved for the base scope: `base_<id>.<key>` storage
    // options overlay the shared defaults for that base.
    if let Some(store_params) = params
        .base_store_params
        .as_ref()
        .and_then(|base_store_params| base_store_params.get(&base_path.path))
    {
        return store_params.clone();
    }
    let default_params = params.store_params.clone().unwrap_or_default();
    match default_params.scoped_to_base(Some(base_path.id)) {
        Cow::Owned(scoped_params) => scoped_params,
        Cow::Borrowed(_) => default_params,
    }
}

async fn append_external_initial_bases(
    initial_bases: Option<&Vec<BasePath>>,
    store_registry: Arc<ObjectStoreRegistry>,
    params: &WriteParams,
    candidates: &mut Vec<ExternalBaseCandidate>,
    seen_base_ids: &mut HashSet<u32>,
) -> Result<()> {
    if let Some(initial_bases) = initial_bases {
        for base_path in initial_bases {
            let store_params = write_store_params_for_base(params, base_path);
            let (store, extracted_path) = ObjectStore::from_uri_and_params(
                store_registry.clone(),
                &base_path.path,
                &store_params,
            )
            .await?;
            append_external_base_candidate(
                base_path,
                store.store_prefix.clone(),
                extracted_path,
                store_params,
                candidates,
                seen_base_ids,
            );
        }
    }
    Ok(())
}

async fn build_external_base_resolver(
    dataset: Option<&Table>,
    params: &WriteParams,
) -> Result<ExternalBaseResolver> {
    let store_registry = dataset
        .map(|ds| ds.session().store_registry())
        .unwrap_or_else(|| params.store_registry());

    let mut seen_base_ids = HashSet::new();
    let mut candidates = vec![];

    if let Some(dataset) = dataset {
        for base_path in dataset.manifest.base_paths.values() {
            let store_params = dataset.store_params_for_base(Some(base_path));
            let (store, extracted_path) = ObjectStore::from_uri_and_params(
                store_registry.clone(),
                &base_path.path,
                &store_params,
            )
            .await?;
            append_external_base_candidate(
                base_path,
                store.store_prefix.clone(),
                extracted_path,
                store_params,
                &mut candidates,
                &mut seen_base_ids,
            );
        }
    }

    append_external_initial_bases(
        params.initial_bases.as_ref(),
        store_registry.clone(),
        params,
        &mut candidates,
        &mut seen_base_ids,
    )
    .await?;

    let mut resolver = ExternalBaseResolver::new(candidates, store_registry);
    resolver.unregistered_external_blob_policy = dataset
        .map(|dataset| dataset.state.session.as_ref())
        .or(params.session.as_deref())
        .map(|session| session.unregistered_external_blob_policy())
        .unwrap_or_default();
    resolver.registered_base_ids = dataset
        .into_iter()
        .flat_map(|dataset| dataset.manifest.base_paths.keys().copied())
        .chain(params.initial_bases.iter().flatten().map(|base| base.id))
        .collect();
    Ok(resolver)
}

#[doc(hidden)]
pub async fn blob_v2_external_base_resolver(
    dataset: Option<&Table>,
    params: &WriteParams,
    schema: &Schema,
) -> Result<Option<Arc<ExternalBaseResolver>>> {
    if schema.fields_pre_order().any(|field| field.is_blob_v2()) {
        Ok(Some(Arc::new(
            build_external_base_resolver(dataset, params).await?,
        )))
    } else {
        Ok(None)
    }
}

#[derive(Clone)]
pub struct TargetBaseInfo {
    /// The registered base id, or [`PRIMARY_BASE_ID`] for the dataset's
    /// primary storage.
    pub base_id: u32,
    pub object_store: Arc<ObjectStore>,
    /// The base directory path (without /data subdirectory)
    pub base_dir: Path,
    /// Whether this base path is a dataset root.
    /// If true, /data will be added when creating file paths.
    /// If false, files will be written directly to base_dir.
    pub is_dataset_root: bool,
}
