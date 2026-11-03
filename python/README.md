# Lance Python SDK

`pylance` is the Python library for the [Lance](https://lance.org) format.

## Which Python package should I use?

Lance has two Python packages:

- **[`lancedb`](https://pypi.org/project/lancedb/)** (`pip install lancedb`): the recommended starting point for applications and data work on Lance datasets. It opens existing Lance datasets (`lancedb.connect(parent_dir).open_table(name)`), and covers filtered reads and scans, vector, full-text and hybrid search, upserts, versioning and tags. See the [LanceDB docs](https://docs.lancedb.com).
- **[`pylance`](https://pypi.org/project/pylance/)** (`pip install pylance`, `import lance`): the low-level format library, for building query engines and connectors that need fragment-level control, such as writing fragments and committing them manually.

If you are not building a query engine or connector, start with `lancedb`.

## Contributing

Please read the contribution guide at the [CONTRIBUTING.md](https://github.com/lance-format/lance/blob/main/python/CONTRIBUTING.md) page