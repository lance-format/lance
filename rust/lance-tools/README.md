# lance-tools

`lance-tools` is a command-line tool for interacting with Lance files and tables.

## Commands

Display Lance file footer metadata:

```bash
lance-tools file meta --source path/to/file.lance
```

Display a Lance table manifest:

```bash
lance-tools table manifest --source path/to/table/_versions/1.manifest
```

## Dataset recovery

Build with the storage provider needed for your dataset, for example:

```shell
cargo build -p lance-tools --features aws
```

The `azure`, `gcp`, and `tos` features enable the corresponding providers.
All dataset commands accept `--storage-options` as a JSON object of string
values and `-s` / `--source` as the dataset URI or local directory.

Check every data file referenced by the latest manifest:

```shell
lance-tools dataset verify-data-files -s /path/to/table.lance --deep --fail
```

The default check reads object sizes and compares them with the manifest when
available. `--deep` also reads current-format Lance file metadata and checks row
counts. It does not read all data pages, blob payloads, deletion files, or index
files, so success is not a complete integrity guarantee. Deep verification of
legacy v1 files is reported as unsupported. Use `--version N` to inspect a
historical version. Verification reports all failing files with their fragment
and base IDs; `--fail` makes a failed check return a nonzero exit status.

Preview removing a known missing or damaged file, then commit the repair:

```shell
lance-tools dataset repair-manifest -s /path/to/table.lance \
  --remove-data-files file1.lance,file2.lance --dry-run
lance-tools dataset repair-manifest -s /path/to/table.lance \
  --remove-data-files file1.lance,file2.lance
```

Each selection must exactly match a manifest file path, `data/<manifest path>`,
or a resolved object key. The option can also be repeated. Every selection must
match before a repair can commit. The command removes **entire fragments and
all their rows**, including other data files containing columns of those rows.
Review the dry-run output to see the affected fragments and files.

Repair creates a new dataset version through the normal transaction commit
path. It does not delete physical objects or rewrite historical manifests.
Other retained versions may still reference the damaged files, and concurrent
conflicting writes can cause the commit to fail.

To find when a file first appeared in the retained history:

```shell
lance-tools dataset locate-data-file -s /path/to/table.lance --data-file file1.lance
```

The output includes the earliest retained version referencing the file and the
preceding retained version, if any. Versions already removed by cleanup cannot
be searched. An alternative to fragment removal is restoring an earlier snapshot:

```shell
lance-tools dataset restore -s /path/to/table.lance --version 10
```

Restore publishes that snapshot as a new latest version. Its referenced files
must still be available; restore does not reconstruct missing objects.
