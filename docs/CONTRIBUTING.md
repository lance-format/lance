## Contributing to Documentation

### Main website

The main documentation website is built using [mkdocs-material](https://squidfunk.github.io/mkdocs-material/).
To build the docs, first install requirements:

```bash
cd docs
uv sync --dev
```

Then build and start the docs server:

```bash
uv run mkdocs serve
```

### Python Generated Doc

Python code documentation is built using Sphinx in [lance-python-doc](https://github.com/lance-format/lance-python-doc),
and published through [Github Pages](https://lance-format.github.io/lance-python-doc/) in ReadTheDocs style.

### Rust Generated Doc

Rust code documentation is built and published to the [Rust official docs website](https://docs.rs/lance/latest/lance/)
as a part of the release process.

### Java Generated Doc

Java code documentation is built and published to Maven Central.
You can find the doc page for the specific project at [javadoc.io](https://javadoc.io).
