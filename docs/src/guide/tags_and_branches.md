# Manage Tags and Branches

Lance provides Git-like tag and branch capabilities through the `LanceDataset.tags` and `LanceDataset.branches` properties.

## Tags
Tags label specific versions within a branch's history.

`Tags` are particularly useful for tracking the evolution of datasets,
especially in machine learning workflows where datasets are frequently updated.
For example, you can `create`, `update`,
and `delete` or `list` tags.

The `reference` parameter (used in `create`, `update`, and `checkout_version`) accepts:

- An **integer**: version number in the **current branch** (e.g., `1`)
- A **string**: tag name (e.g., `"stable"`)
- A **tuple** `(branch_name, version)`: a specific version in a named branch
  - `(None, 2)` means version 2 on the main branch
  - `("main", 2)` means version 2 on the main branch (explicit)
  - `("experiment", 3)` means version 3 on the experiment branch
  - `("branch-name", None)` means the latest version on that branch

!!! note

    Creating or deleting tags does not generate new dataset versions.
    Tags exist as auxiliary metadata stored in a separate directory.

```python
import lance
import pyarrow as pa

ds = lance.dataset("./tags.lance")
print(len(ds.versions()))
# 2
print(ds.tags.list())
# {}
ds.tags.create("v1-prod", (None, 1))
print(ds.tags.list())
# {'v1-prod': {'version': 1, 'created_at': ..., 'updated_at': ..., 'manifest_size': ...}}
ds.tags.update("v1-prod", (None, 2))
print(ds.tags.list())
# {'v1-prod': {'version': 2, 'created_at': ..., 'updated_at': ..., 'manifest_size': ...}}
ds.tags.delete("v1-prod")
print(ds.tags.list())
# {}
print(ds.tags.list_ordered())
# []
ds.tags.create("v1-prod", (None, 1))
print(ds.tags.list_ordered())
# [('v1-prod', {'version': 1, 'created_at': ..., 'updated_at': ..., 'manifest_size': ...})]
ds.tags.update("v1-prod", (None, 2))
print(ds.tags.list_ordered())
# [('v1-prod', {'version': 2, 'created_at': ..., 'updated_at': ..., 'manifest_size': ...})]
ds.tags.delete("v1-prod")
print(ds.tags.list_ordered())
# []
```

!!! note

    Tagged versions are exempted from the `LanceDataset.cleanup_old_versions()`
    process.

    To remove a version that has been tagged, you must first `LanceDataset.tags.delete()`
    the associated tag. 

## Branches

Branches manage parallel lines of dataset evolution. You can create a branch from an existing version or tag, read and write to it independently, and checkout different branches. You can `create`, `delete`, `list`, and `checkout` branches.

The `reference` parameter works the same as for Tags (see above).

!!! note

    Creating or deleting branches does not generate new dataset versions.
    New versions are created by writes (append/overwrite/index operations).

    Each branch maintains its own linear version history, so version numbers may overlap across branches. Use `(branch_name, version_number)` tuples as global identifiers for operations like `checkout_version` and `tags.create`.

    "main" is a reserved branch name. Lance uses "main" to identify the default branch.

### Create and checkout branches
```python
import lance
import pyarrow as pa

# Open dataset
ds = lance.dataset("/tmp/test.lance")

# Create branch from the currently checked-out version
experiment_branch = ds.create_branch("experiment")
experimental_data = pa.Table.from_pydict({"a": [11], "b": [12]})
lance.write_dataset(experimental_data, experiment_branch, mode="append")

# Create tag on the latest version of the experimental branch
ds.tags.create("experiment-rc", ("experiment", None))

# Checkout by tag name
experiment_rc = ds.checkout_version("experiment-rc")
# Checkout the latest version of the experimental branch by tuple
experiment_latest = ds.checkout_version(("experiment", None))

# Create a new branch from a tag
new_experiment = ds.create_branch("new-experiment", "experiment-rc")
```

### List branches
```python
print(ds.branches.list())
# {'experiment': {...}, 'new-experiment': {...}}
```

### Delete a branch
```python
# Ensure the branch is no longer needed before deletion
ds.branches.delete("experiment")
print(ds.branches.list_ordered(order="desc"))
# {'new-experiment': {'parent_branch': 'experiment', 'parent_version': 2, 'create_at': ..., 'manifest_size': ...}, ...}
```

!!! note

    Branches hold references to data files. Lance ensures that cleanup does not delete files still referenced by any branch.

    Delete unused branches to allow their referenced files to be cleaned up by `cleanup_old_versions()`.
