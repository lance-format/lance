# AGENTS.md

Lance is a modern columnar data format optimized for ML workflows and datasets, providing high-performance random access, vector search, zero-copy automatic versioning, and ecosystem integrations. The vision is to become the de facto standard columnar data format for machine learning and large language models.

Also see directory-specific guidelines: [rust/](rust/AGENTS.md) | [python/](python/AGENTS.md) | [java/](java/AGENTS.md) | [protos/](protos/AGENTS.md) | [docs/src/format/](docs/src/format/AGENTS.md)

## File Format Stability and Compatibility

- Treat every file format marked stable as a durable compatibility contract. All changes to a stable format must preserve both backward and forward compatibility.
- Treat every file format marked unstable as disposable. It may change freely; do not add compatibility code, migrations, fallbacks, or tests for files written by earlier unstable revisions.
- Evaluate compatibility against the latest released stable version while continuing to honor all stable format contracts. Changes that exist only on the current branch or `main` are not compatibility constraints; do not compromise a cleaner or more complete design to preserve those intermediate states.

### Legacy Compatibility Boundaries

- Treat formats and code paths that current writers no longer emit as frozen compatibility surfaces. Preserve their existing read behavior, but exclude them from new feature design unless legacy support is explicitly required.
- Implement new features in the current format and write paths. Do not extend legacy writers, retrofit new capabilities into legacy readers, or reuse legacy implementations as the foundation for new code.
- Avoid refactoring or otherwise modifying legacy code during feature work. If a shared boundary makes a legacy change unavoidable, isolate the change, preserve existing behavior, and add targeted regression coverage using released historical fixtures.

## Development Commands

### Rust

* Check: `cargo check --workspace --tests --benches`
* Test: `cargo test --workspace` or `cargo test -p <package> <test_name>`
* Lint: `cargo clippy --all --tests --benches -- -D warnings`
* Format: `cargo fmt --all`
* Coverage: `cargo +nightly llvm-cov -q -p <crate> --branch`
* Coverage HTML: `cargo +nightly llvm-cov -q -p <crate> --branch --html`
* Coverage for file: `python ci/coverage.py -p <crate> -f <file_path>`
* Use repository-defined Cargo profiles instead of ad hoc LTO overrides.
* Use `release-with-debug` for benchmarks and profiling so optimized builds keep debug symbols without a rebuild.
* Use `release-no-lto` only for local debugging, IO-bound benchmarks, or compile-time-sensitive performance investigation where LTO would not affect the measured bottleneck.

## Language-Specific Environment Contract

- For language-specific tasks, always follow the environment and command rules in the corresponding subdirectory guide before running build, test, lint, format, or tooling commands.
- Do not substitute a different environment manager or toolchain just because a command appears missing, unavailable, or slow.
- If a language-specific command fails outside the documented workflow, treat that as an environment usage mistake first. Fix the environment usage, rerun with the prescribed commands, and only then conclude that a dependency or tool is unavailable.

## Coding Standards

### General

- Always use English in code, examples, and comments.
- Code is for readability, not just execution. Only add meaningful comments and tests.
- Comments should explain non-obvious "why" reasoning, not restate what the code does.
- Remove debug prints (`println!`, `dbg!`, `print()`) before merging — use `tracing` or logging frameworks.
- Think carefully before adding a helper: only introduce one when it materially reduces cognitive load or eliminates substantial duplication, and do not add thin wrappers that only rename or forward existing calls.
- Keep PRs focused — no drive-by refactors, reformatting, or cosmetic changes.
- Be mindful of memory use: avoid collecting streams of `RecordBatch` into memory; use `RoaringBitmap` instead of `HashSet<u32>`.

### Cross-Language Bindings

- Keep Python and Java bindings as thin wrappers — centralize validation and logic in the Rust core.
- Keep parameter names consistent across all bindings (Rust, Python, Java) — rename everywhere or nowhere.
- Never break public API signatures — deprecate with `#[deprecated]`/`@deprecated` and add a new method.
- Replace mutually exclusive boolean flags with a single enum/mode parameter.

### Naming

- Name variables after what the value *is* (e.g., `partition_id` not `mask`) — precise names act as inline docs.
- Drop redundant prefixes when the struct/module already implies the domain.
- Use `indices` (not `indexes`) consistently in all APIs and docs.
- Use storage-agnostic terms in API names (e.g., `base` not `bucket`).
- When renaming a type/struct/enum, update all references (methods, fields, variables, test names).

### Error Handling

- Validate inputs and reject invalid values with descriptive errors at API boundaries — never silently clamp or adjust.
- Validate mutually exclusive options in builders/configs — throw a clear error if both are set.
- Include full context in error messages: variable names, values, sizes, types.

### Dependencies

- Prefer implementing functionality with the standard library or existing workspace dependencies before adding new external crates.
- Keep `Cargo.lock` changes intentional; revert unrelated dependency bumps. Pin broken deps with a comment linking the upstream issue.
- The repo has three lockfiles: the root `Cargo.lock`, `python/Cargo.lock`, and `java/lance-jni/Cargo.lock` (the latter two are excluded from the workspace). A `workspace.dependencies` change must be reflected in all three — refresh the excluded ones with `cargo check --manifest-path python/Cargo.toml` and `cargo check --manifest-path java/lance-jni/Cargo.toml`, then commit the updated lockfiles. The `cargo-lock-sync` pre-commit hook catches a miss offline.
- Gate optional/domain-specific deps behind Cargo feature flags. Prefer separate crates for domain functionality (geo, NLP).

## Testing Standards

- **All bugfixes and features must have corresponding tests. We do not merge code without tests.**
- Keep local unit tests lightweight: each test case should finish within one second on typical developer hardware. Split independent parameter matrices and use the smallest fixture or model that preserves the asserted behavior; do not relax assertions, coverage, or recall thresholds to meet the budget.
- Use `rstest` (Rust) or `@pytest.mark.parametrize` (Python) for tests that differ only in inputs. Use `#[case::{name}(...)]` for readable case names.
- Replace `print()` in tests with `assert` — prints don't catch regressions.
- Extend existing tests instead of adding overlapping new ones. Add to existing test files.
- Link a GitHub issue when skipping a test — never bare `@pytest.mark.skip` or `@Ignore` without a tracking URL.
- Include multi-fragment scenarios for dataset operations (reads, indexes, scans).
- Cover NULL edge cases in index tests: null items, all-null collections, empty collections, null columns.
- Vector index tests must assert recall metrics (>=0.5 threshold), not just verify creation succeeds.
- For backwards compatibility, use the `test_data` directory with checked-in datasets from older versions. Include a `datagen.py` that asserts the Lance version used. Use `copy_test_data_to_tmp` to read this data.
- Avoid `ignore` in doctests — write Rust doctests that compile a function instead:
  ```
  /// ```
  /// # use lance::{Dataset, Result};
  /// # async fn test(dataset: &Dataset) -> Result<()> {
  /// dataset.delete("id = 25").await?;
  /// # Ok(())
  /// # }
  /// ```
  ```
- Skip coverage for test utilities using `#[cfg_attr(coverage, coverage(off))]`.

## Documentation Standards

- All public APIs must have documentation with examples. Link to relevant structs and methods.
- Use ASCII tree diagrams for hierarchical structures (encoding layers, file formats, storage layouts).
- Keep doc examples in sync with actual API signatures — update when refactoring.
- Indent content under MkDocs admonition directives (`!!! note`, etc.) with 4 spaces.
- Proofread comments and docs for typos before committing.

## Filing Issues

- When opening an issue with `gh issue create` or the API, classify it and pass the matching label: `--label bug`, `--label feature`, or `--label performance`. These paths bypass the `.github/ISSUE_TEMPLATE` forms, so the label is not applied automatically.
- Prefix the title to match, e.g. `bug: ...`, `feature: ...`, or `perf: ...`. A content-based labeler (`.github/workflows/issue-labeler.yml`) uses this as a fallback signal, but an explicit `--label` is the reliable path.

## Pull Requests

- Before creating a PR, search for similar PRs and inspect any PRs linked to the issue being addressed. If a matching PR exists, verify its current status and scope before proceeding to avoid creating duplicate work.
- PR titles must follow the Conventional Commits specification because `.github/workflows/pr-title.yml` validates the PR title and body with commitlint. Use prefixes like `feat:`, `fix:`, `docs:`, `perf:`, `ci:`, `test:`, `build:`, `style:`, or `chore:`; add a scope when useful.
- Before creating or updating a PR, run the lint checks for every touched language surface, even when they are expensive. For Rust changes, run `cargo fmt --all` and `cargo clippy --all --tests --benches -- -D warnings`. For Python changes, follow the environment workflow in `python/AGENTS.md` and run `uv run make lint` from `python/`. If a required lint check cannot be run, state the blocker explicitly in the PR summary.

## Review Guidelines

Contributor and maintainer attention is the most valuable resource. Less is more.

- Be concise and clear. Focus on P0/P1 issues: severe bugs, performance degradation, security concerns.
- Do not reiterate detailed changes or repeat what's already well done.
- Check naming consistency, error handling patterns, and test coverage.
