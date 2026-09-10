# Python Guidelines

Also see [root AGENTS.md](../AGENTS.md) for cross-language standards.

## Commands

* Environment: use `uv` for all local Python environment setup in this repository.
* First step in every new worktree or fresh checkout: run `make install` from `python/` before any Python command. This runs `uv sync` (dev and test dependencies are included by default via `[tool.uv] default-groups`) and sets up pre-commit hooks. Add `--group benchmarks`, `--extra torch`, or `--extra geo` only when needed.
* `uv sync` builds the local `pylance` Rust extension as part of environment setup. This can take a long time. Start it early, let it finish, and do not interrupt it or switch to a different setup path just because the build is slow.
* Only run `uv sync` again when dependencies change (e.g., after pulling new commits that update `pyproject.toml` or `uv.lock`).
* Command execution: always use `uv run ...` for Python-related repository commands. Do not rely on a globally activated environment.
* Never invoke bare `python`, `pytest`, `pip`, `maturin`, `make test`, `make doctest`, `make lint`, or `make format` for repository work.
* If a Python command fails outside `uv run`, that does not count as a dependency or test failure. Fix the environment usage first and rerun correctly.
* Build: `make build` (required after Rust changes)
* Test: `uv run make test`
* Run single test: `uv run pytest python/tests/<test_file>.py::<test_name>`
* Doctest: `uv run make doctest`
* Lint: `uv run make lint`
* Format: `uv run make format`

## Beta Releases

- Python RC and beta preview wheels are published on fury.io, not only PyPI. When a task needs a beta version such as `7.2.0b4`, use a disposable venv and install with `--pre` and the Lance fury indices:
  ```shell
  uv venv /path/to/venv
  uv pip install --python /path/to/venv/bin/python --pre \
      --extra-index-url https://pypi.fury.io/lance-format/ \
      "pylance==7.2.0b4"
  ```

## API Design

- Keep bindings as thin wrappers — centralize validation and logic in Rust core.
- Extend existing methods with named arguments instead of adding new methods that accept policy/config objects — the Python API should feel Pythonic (e.g., `cleanup_old_versions(..., retain_versions=N)`), not mirror Rust builder patterns.
- Pass all fields to Python dataclass constructors via PyO3, converting Rust `None` to `py.None()` instead of omitting args — dataclass constructors require all positional params.
- Use parameterized type hints (e.g., `list[DatasetBasePath]`, `Optional[Dict[str, str]]`) — never bare generics. Keep docstring type descriptions in sync with hints.

## Testing

- Add tests to existing `test_{module}.py` files rather than creating new test files for the same module.
