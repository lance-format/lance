# Development

For local development, prefer [uv](https://docs.astral.sh/uv/) to create and manage the Python environment:

```shell
uv sync
```

Add `--group benchmarks` for benchmarks, or `--extra torch` / `--extra geo` for those optional features. After the environment is initialized, either activate it or use `uv run ...` for commands.

`uv sync` is not just downloading Python packages here. It also builds the local `pylance` Rust extension as part of the editable environment, so the first run, cache misses, or Rust dependency changes can make it noticeably slow. This is expected; let the build finish instead of interrupting it and switching to a different environment setup.

## Building the project

This project is built with [maturin](https://github.com/PyO3/maturin).

It can be built in development mode with:

```shell
uv run maturin develop --uv
```

This builds the Rust native module in place. You will need to re-run this
whenever you change the Rust code. But changing the Python code doesn't require
re-building.

As with `uv sync`, this may take a while because it is compiling Rust code for the local `pylance` extension. Slow builds are expected here, especially after dependency or toolchain changes.

## Running tests

To run the tests:

```shell
uv run make test
```

To check the documentation examples, use

```shell
uv run make doctest
```

## Formatting and linting

To run formatters, run:

```shell
uv run make format
```

(To run for just Python or just Rust, use `make format-python` or `cargo fmt`.)

To run format checker and linters, run:

```shell
uv run make lint
```

(To run for just Python or just Rust, use `make lint-python` or `make lint-rust`.)

### Format and lint on commit

If you would like to run the formatters and linters when you commit your code
then you can use the pre-commit tool. The project includes a pre-commit config
file already. First, install the pre-commit tool:

```shell
uv tool install pre-commit
```

Then install the hooks:

```shell
pre-commit install
```

From now any, any attempt to commit, will first run the linters against the
modified files:

```shell
$ git commit -m"Changed some python files"
black....................................................................Passed
isort (python)...........................................................Passed
ruff.....................................................................Passed
[main daf91ed] Changed some python files
 1 file changed, 1 insertion(+), 1 deletion(-)
```

## Cleaning build artifacts
To clean up build artifacts, run:
```shell
make clean
```

## Benchmarks

The benchmarks in `python/benchmarks` can be used to identify and diagnose
performance issues. They are run with [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/).
These benchmarks aren't mean to showcase performance on full-scale real world
datasets; rather they are meant to be useful for developers to iterate on
performance improvements and to catch performance regressions. Therefore, any
benchmarks added there should run in less than 5 seconds.

Before running benchmarks, you should build pylance in release mode:

```shell
uv sync --group benchmarks
uv run maturin develop --uv --profile release-with-debug --extras benchmarks --features datagen
```

(You can also use `--release` or `--profile release`, but `--profile release-with-debug`
will provide debug symbols for profiling.)

Then you can run the benchmarks with

```shell
uv run pytest python/benchmarks -m "not slow"
```

Note: the first time you run the benchmarks, they may take a while, since they
will write out test datasets and build vector indices. Once these are built,
they are re-used between benchmark runs.

Some benchmarks are especially slow, so they are skipped `-m "not slow"`. To run
the slow benchmarks, use:

```shell
uv run pytest python/benchmarks
```

### Run a particular benchmark

To filter benchmarks by name, use the usual pytest `-k` flag (this can be a
substring match, so you don't need to type the full name):

```shell
uv run pytest python/benchmarks -k test_ivf_pq_index_search
```

### Profile a benchmark

If you have [cargo-flamegraph](https://github.com/flamegraph-rs/flamegraph)
installed, you can create a flamegraph of a benchmark by running:

```shell
flamegraph -F 100 --no-inline -- $(uv run which python) \
    -m pytest python/benchmarks \
    --benchmark-min-time=2 \
    -k test_ivf_pq_index_search
```

Note the parameter `--benchmark-min-time`: this controls how many seconds to run
the benchmark in each round (default 5 rounds). The default is very low but you
can increase this so that the profile gets more samples.

You can drop the `--no-inline` to have the program try to identify which functions
were inlined to get more detail, though this will make the processing take
considerably longer.

This will only work on Linux.

Note that you'll want to run the benchmarks once prior to profiling, so that
the setup is complete and not captured as part of profiling.

### Compare benchmarks against previous version

You can easily compare the performance of the current version against `main`.
Checkout `main` branch, run the benchmarks, and save
the output using `--benchmark-save`. Then install the current version and run
the benchmarks again with `--benchmark-compare`.

```shell
CURRENT_BRANCH=$(git branch --show-current)
git checkout main
uv sync --group benchmarks
uv run maturin develop --uv --profile release-with-debug --features datagen
uv run pytest --benchmark-save=baseline python/benchmarks -m "not slow"
COMPARE_ID=$(ls .benchmarks/*/ | tail -1 | cut -c1-4)
git checkout $CURRENT_BRANCH
uv sync --group benchmarks
uv run maturin develop --uv --profile release-with-debug --features datagen
uv run pytest --benchmark-compare=$COMPARE_ID python/benchmarks -m "not slow"
```

## Tracing

Rust has great integration with tools like criterion and pprof which make it easy
to profile and debug CPU intensive tasks. However, these tools are not as effective
at profiling I/O intensive work or providing a high level trace of an operation.

To fill this gap the lance code utlizies the Rust tracing crate to provide tracing
information for lance operations. User applications can receive these events and
forward them on for logging purposes. Developers can also use this information to
get a sense of the I/O that happens during an operation.

### Instrumenting code

When instrumenting code you can use the `#[instrument]` macro from the Rust tracing
crate. See the crate docs for more information on the various parameters that can
be set. As a general guideline we should aim to instrument the following methods:

- Top-level methods that will often be called by external libraries and could be slow
- Compute intensive methods that will perform a significant amount of CPU compute
- Any point where we are waiting on external resources (e.g. disk)

To begin with, instrument methods as close to the user as possible and refine downwards
as you need. For example, start by instrumenting the entire dataset write operation
and then instrument any individual parts of the operation that you would like to see
details for.

### Tracing a unit test

If you would like tracing information for a rust unit test then you will need to
decorate your test with the lance_test_macros::test attribute. This will wrap any
existing test attributes that you are using:

```rust
#[lance_test_macros::test(tokio::test)]
async fn test() {
    ...
}
```

Then, when running your test, you will need to set the environment variable
LANCE_TRACING to the your desired verbosity level (trace, debug, info, warn, error):

```bash
LANCE_TESTING=debug cargo test dataset::tests::test_create_dataset
```

This will create a .json file (named with a timestamp) in your working directory. This
.json file can be loaded by chrome or by <https://ui.perfetto.dev>

### Tracing a python script

If you would like to trace a python script (application, benchmark, test) then you can easily
do so using the lance.tracing module. Simply call:

```python
from lance.tracing import trace_to_chrome

trace_to_chrome(level="debug")

# rest of script
```

A single .json trace file will be generated after python has exited.

You can use the `trace_to_chrome` function within the benchmarks, but for
sensible results you'll want to force the benchmark to just run only once.
To do this, rewrite the benchmark using the pedantic API:

```python
def run():
    "Put code to benchmark here"
    ...
benchmark.pedantic(run, iterations=1, rounds=1)
```

### Trace visualization limitations

The current tracing implementation is slightly flawed when it comes to async
operations that run in parallel. The rust tracing-chrome library emits
trace events into the chrome trace events JSON format. This format is not
sophisticated enough to represent asynchronous parallel work.

As a result, a single instrumented async method may appear as many different
spans in the UI.

## Running S3 Integration tests

The integration tests run against local minio and local dynamodb. To start the
services, run

```shell
docker compose up
```

Then you can run the tests with

```shell
pytest --run-integration python/tests/test_s3_ddb.py
```

## Building wheels locally

### Linux

On Mac or Linux, you can build manylinux wheels locally for Linux. The easiest
way to do this is to use `zig` with `maturin build`. Before you do this, you'll
need to make you (1) [install zig](https://github.com/ziglang/zig/wiki/Install-Zig-from-a-Package-Manager)
and (2) install the toolchains:

```shell
rustup target add x86_64-unknown-linux-gnu
rustup target add aarch64-unknown-linux-gnu
```

For x86 Linux:

```shell
maturin build --release --zig \
    --target x86_64-unknown-linux-gnu \
    --compatibility manylinux2014 \
    --out wheels
```

For ARM / aarch64 Linux:

```shell
maturin build --release --zig \
    --target aarch_64-unknown-linux-gnu \
    --compatibility manylinux2014 \
    --out wheels
```

### MacOS

On a Mac, you can build wheels locally for MacOS:

```shell
maturin build --release \
    --target aarch64-apple-darwin \
    --out wheels
```

```shell
maturin build --release \
    --target x86_64-apple-darwin \
    --out wheels
```

## Picking a thread pool

When an operation should run in parallel you typically need to specify how many threads
to use. For example, as input to `StreamExt::buffered`. There are two numbers you can
use. You can use `Storage::io_parallelism` or `get_num_compute_intensive_cpus`.

Often, operations will do a little of both compute and I/O, and you will need to make
a judgement call. If you are unsure, and you are doing any I/O, then picking the
`io_parallelism` is a good fallback behavior. The worst case is just that we over-parallelize
and there is more CPU contention then there needs to be. If this becomes a problem we
can always split the operation into two parts and use the two different thread pools.
