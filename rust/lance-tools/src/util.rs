// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::storage::path::Path;
use lance_core::{Error, Result};
use lance_io::storage::{Storage, StorageParams, StorageRegistry};
use std::sync::Arc;
use url::Url;

fn path_to_parent(path: &Path) -> Result<(Path, String)> {
    let mut parts = path.parts().collect::<Vec<_>>();
    if parts.is_empty() {
        return Err(Error::invalid_input(format!(
            "Path {} is not a valid path to a file",
            path
        )));
    }
    let filename = parts.pop().unwrap().as_ref().to_owned();
    Ok((Path::from_iter(parts), filename))
}

/// Get an object store and a path from a source string.
pub(crate) async fn get_storage_and_path(source: &String) -> Result<(Arc<Storage>, Path)> {
    if let Ok(mut url) = Url::parse(source)
        && url.scheme().len() > 1
    {
        let path = lance_core::storage::path::Path::parse(url.path()).map_err(Error::from)?;
        let (parent_path, filename) = path_to_parent(&path)?;
        url.set_path(parent_path.as_ref());
        let storage_registry = Arc::new(StorageRegistry::default());
        let storage_params = StorageParams::default();
        let (storage, dir_path) =
            Storage::from_uri_and_params(storage_registry, url.as_str(), &storage_params).await?;
        let child_path = dir_path.clone().join(filename);
        return Ok((storage, child_path));
    }
    let path = Path::from_filesystem_path(source)?;
    let storage = Arc::new(Storage::local());
    Ok((storage, path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_to_parent() {
        let (parent_path, filename) =
            path_to_parent(&lance_core::storage::path::Path::parse("/a/b/c").unwrap()).unwrap();
        assert_eq!("c", filename);
        let parts: Vec<_> = parent_path.parts().collect();
        assert_eq!(2, parts.len());
        assert_eq!("a", parts.first().unwrap().as_ref());
        assert_eq!("b", parts.get(1).unwrap().as_ref());
    }
}
