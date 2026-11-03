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