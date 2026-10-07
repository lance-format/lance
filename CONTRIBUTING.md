# Guide for New Contributors

This is a guide for new contributors to the Lance project.
Even if you have no previous experience with python, rust, and open source, you can still make a non-trivial
impact by helping us improve documentation, examples, and more.
For experienced developers, the issues you can work on run the gamut from warm-ups to serious challenges in python and rust.

If you have any questions, please join our [Discord](https://discord.gg/zMM32dvNtd) for real-time support. Your feedback is always welcome!

## Getting Started

1. Join our Discord and say hi
2. Setup your development environment
3. Pick an issue to work on. See https://github.com/lance-format/lance/contribute for good first issues.
4. Have fun!

## Development Environment

Currently Lance is implemented in Rust and comes with a Python wrapper. So you'll want to make sure you setup both.

1. Install Rust: https://www.rust-lang.org/tools/install
2. Install Python 3.10+: https://www.python.org/downloads/
3. Install protocol buffers: https://grpc.io/docs/protoc-installation/ (make sure you have version 3.20 or higher)
4. Install commit hooks:
    a. Install pre-commit: https://pre-commit.com/#install
    b. Run `pre-commit install` in the root of the repo

## Building for legacy x86_64 hosts (pre-Haswell)

The default workspace build targets `haswell` (AVX2 + FMA + F16C), matching the published wheels. To build a binary that runs on pre-Haswell silicon (Sandy Bridge / Ivy Bridge / Westmere on Intel, Bulldozer / Piledriver / Steamroller on AMD — i.e. CPUs without AVX2), set the baseline yourself at build time:

```sh
RUSTFLAGS="-C target-cpu=x86-64-v2" cargo build --release
```

Runtime SIMD dispatch in `lance-linalg::distance` will then pick the appropriate tier (scalar / AVX / AVX+FMA / AVX2+FMA / AVX-512) based on the host. From Python, use `lance.simd_info()` to verify which tier was selected.

## Sample Workflow

1. Fork the repo
2. Pick [Github issue](https://github.com/lance-format/lance/issues)
3. Create a branch for the issue
4. Make your changes
5. Create a pull request from your fork to lance-format/lance
6. Get feedback and iterate
7. Merge!
8. Go back to step 2

## Example Notebooks

Example notebooks are under `examples`.
These are standalone notebooks you should be able to download and run.

## Benchmarks

Our Rust benchmarks are run multiple times a day and the history can be found [here](https://github.com/lancedb/lance-benchmark-results).

Separately, we have vector index benchmarks that test against the sift1m dataset, as well as benchmarks for tpch.
These live under `benchmarks`.

## Reviewing issues and pull requests

Please consider the following when reviewing code contributions.

### Rust API design
* Design public APIs so they can be evolved easily in the future without breaking
  changes. Often this means using builder patterns or options structs instead of
  long argument lists.
* For public APIs, prefer inputs that use `Into<T>` or `AsRef<T>` traits to allow
  more flexible inputs. For example, use `name: Into<String>` instead of `name: String`,
  so we don't have to write `func("my_string".to_string())`.

### Testing
* Ensure all new public APIs have documentation and examples.
* Ensure that all bugfixes and features have corresponding tests. **We do not merge
  code without tests.**

### Important Labels

There are two important labels to apply to relevant issues and PRs:

1. `breaking-change`: Any PR that introduces a breaking change to the public API
  (Rust or Python) must be labelled as such. This is used to determine how to
  bump the version number when releasing. You can still add this label even
  after merging a PR.
2. `critical-fix`: Any PR that fixes a critical bug (e.g., security issue, data
  corruption, crash) should be labelled as such. These are bugs that users might
  have without realizing. Fixes that aren't critical include bugs that return
  an error message. These labels are used to determine whether a patch release
  is needed.

## Code of Conduct

We follow the Code of Conduct of [Python Foundation](https://www.python.org/psf/conduct/) and
[Rust Foundation](https://www.rust-lang.org/policies/code-of-conduct).
