## Contributing to Documentation

### Main Website

The main documentation website is built with [MkDocs](https://www.mkdocs.org/) using a custom theme in `docs/theme/` and managed with [uv](https://docs.astral.sh/uv/).

To build and serve the documentation locally:

```bash
cd docs
make serve
```

The documentation server will be available at `http://localhost:8000`. To build the static site into `docs/site/`:

```bash
cd docs
make build
```

Both `make serve` and `make build` run `./make-full-website.sh` first. When sibling checkouts of external subprojects (such as `lance-namespace`, `lance-spark`, `lance-ray`, `lance-duckdb`, `lance-trino`, `lance-huggingface`, `lance-context`, or `lance-flink`) are available under `$HOME/oss/` or configured via `LANCE_*_REPO` environment variables, their documentation is assembled into the site; otherwise placeholder pages are used so the local site remains buildable. Run `./make-full-website.sh --help` for the full list of repository environment variables, and `./clean-full-website.sh` to reset generated multi-repo content.

### Python Generated Doc

Python code documentation is built using Sphinx in [lance-python-doc](https://github.com/lance-format/lance-python-doc),
and published through [Github Pages](https://lance-format.github.io/lance-python-doc/) in ReadTheDocs style.

### Rust Generated Doc

Rust code documentation is built and published to the [Rust official docs website](https://docs.rs/lance/latest/lance/)
as a part of the release process.

### Java Generated Doc

Java code documentation is built and published to Maven Central.
You can find the doc page for the specific project at [javadoc.io](https://javadoc.io).
