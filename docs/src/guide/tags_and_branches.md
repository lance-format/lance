# Manage Tags and Branches

Lance provides Git-like tag and branch capabilities across the Python (`dataset.tags` / `dataset.branches`), Rust (`dataset.tags()` / `dataset.branches()`), and Java (`dataset.tags()` / `dataset.branches()`) SDKs.

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
  - `(None, 2)` means version 2 on the default branch
  - `("main", 2)` means version 2 on the default branch (explicit)
  - `("experiment", 3)` means version 3 on the experiment branch
  - `("branch-name", None)` means the latest version on that branch

In reference contexts, `"main"` is an alias for the default branch and is equivalent to `None`.

!!! note

    Creating or deleting tags does not generate new dataset versions.
    Tags exist as auxiliary metadata stored in a separate directory.

=== "Python"

    ```python
    import lance

    ds = lance.dataset("./tags.lance")
    ds.tags.create("v1-prod", (None, 1))
    print(ds.tags.list())
    # {'v1-prod': {'version': 1, 'created_at': ..., 'updated_at': ..., 'manifest_size': ...}}

    ds.tags.update("v1-prod", (None, 2))
    print(ds.tags.list_ordered())
    # [('v1-prod', {'version': 2, 'created_at': ..., 'updated_at': ..., 'manifest_size': ...})]

    ds.tags.delete("v1-prod")
    ```

=== "Rust"

    ```rust
    use lance::Dataset;

    let ds = Dataset::open("./tags.lance").await?;
    ds.tags().create("v1-prod", (None, Some(1))).await?;
    let tags = ds.tags().list().await?;

    ds.tags().update("v1-prod", (None, Some(2))).await?;
    ds.tags().delete("v1-prod").await?;
    ```

=== "Java"

    ```java
    import org.lance.Dataset;
    import org.lance.Ref;
    import org.lance.Tag;
    import java.util.List;

    try (Dataset ds = Dataset.open("./tags.lance", allocator)) {
      ds.tags().create("v1-prod", Ref.ofMain(1L));
      List<Tag> tags = ds.tags().list();

      ds.tags().update("v1-prod", Ref.ofMain(2L));
      ds.tags().delete("v1-prod");
    }
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

    `"main"` is reserved for the default branch. Use `"main"` or `None` when referring to the default branch in reference tuples or checkout APIs, but choose a different name when creating, deleting, or updating branches.

### Create and checkout branches

=== "Python"

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

    # Checkout by tag name or (branch, version) tuple
    experiment_rc = ds.checkout_version("experiment-rc")
    experiment_latest = ds.checkout_version(("experiment", None))
    main_latest = ds.checkout_version(("main", None))

    # Create a new branch from a tag
    new_experiment = ds.create_branch("new-experiment", "experiment-rc")
    ```

=== "Rust"

    ```rust
    use lance::Dataset;

    let mut ds = Dataset::open("/tmp/test.lance").await?;
    let current_version = ds.version().version;

    // Create branch from the current default-branch version and append data
    let mut experiment = ds
        .create_branch("experiment", (None, Some(current_version)), None)
        .await?;
    experiment.append(experimental_reader, None).await?;

    // Tag the latest version of the experimental branch
    ds.tags().create("experiment-rc", ("experiment", None)).await?;

    // Checkout by tag name or (branch, version) tuple
    let experiment_rc = ds.checkout_version("experiment-rc").await?;
    let experiment_latest = ds.checkout_version(("experiment", None)).await?;
    let main_latest = ds.checkout_version(("main", None)).await?;

    // Create a new branch from a tag
    let new_experiment = ds
        .create_branch("new-experiment", "experiment-rc", None)
        .await?;
    ```

=== "Java"

    ```java
    import org.lance.Dataset;
    import org.lance.Ref;

    try (Dataset ds = Dataset.open("/tmp/test.lance", allocator);
        Dataset experiment = ds.createBranch("experiment", Ref.ofMain(ds.version()))) {
      ds.tags().create("experiment-rc", Ref.ofBranch("experiment"));

      try (Dataset experimentRc = ds.checkout(Ref.ofTag("experiment-rc"));
          Dataset experimentLatest = ds.checkout(Ref.ofBranch("experiment"));
          Dataset mainLatest = ds.checkout(Ref.ofMain());
          Dataset newExperiment =
              ds.createBranch("new-experiment", Ref.ofTag("experiment-rc"))) {
        // read or write checked-out branches
      }
    }
    ```

### List branches

=== "Python"

    ```python
    print(ds.branches.list())
    # {'experiment': {...}, 'new-experiment': {...}}
    ```

=== "Rust"

    ```rust
    let branches = ds.branches().list().await?;
    ```

=== "Java"

    ```java
    import org.lance.Branch;
    import java.util.List;

    List<Branch> branches = ds.branches().list();
    ```

### Delete a branch

=== "Python"

    ```python
    # A branch cannot be deleted while other branches or tags reference it.
    # "new-experiment" was created from "experiment" and the "experiment-rc" tag
    # points into it, so remove those first.
    ds.branches.delete("new-experiment")
    ds.tags.delete("experiment-rc")
    ds.branches.delete("experiment")
    print(ds.branches.list())
    # {}
    ```

=== "Rust"

    ```rust
    ds.branches().delete("new-experiment", false).await?;
    ds.tags().delete("experiment-rc").await?;
    ds.branches().delete("experiment", false).await?;
    ```

=== "Java"

    ```java
    ds.branches().delete("new-experiment");
    ds.tags().delete("experiment-rc");
    ds.branches().delete("experiment");
    ```

!!! note

    Branches hold references to data files. Lance ensures that cleanup does not delete files still referenced by any branch.

    Delete unused branches to allow their referenced files to be cleaned up by `cleanup_old_versions()`.
