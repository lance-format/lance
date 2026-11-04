# Rust-native extension probe

This workspace validates a Rust-native plugin boundary for Lance storage
providers:

- `runtime` is a shared Rust dylib containing `lance-io`;
- `goosefs-plugin` contains the existing GooseFS provider implementation and
  links the runtime;
- `host` loads the plugin and uses the existing `ObjectStoreRegistry`,
  `ObjectStore`, and `ObjectStoreProvider` APIs;
- the plugin registers `goosefs` and a `demo-memory` implementation through the
  same runtime trait object.

There is no plugin-defined `put`/`get` protocol, no runtime storage C ABI, and
no second storage trait. The host uses `ObjectStore::from_uri_and_params` and
the normal `ObjectStore::put` / `ObjectStore::read_one_all` methods. Plugins are
Rust dylibs and must be built with the same Rust toolchain and feature set as
the runtime.

The runtime has one Rust ABI entry point for loading a dylib,
`ExtensionRuntime::load_plugin`; the plugin receives the existing
`ObjectStoreRegistry` and registers providers with `ObjectStoreRegistry::insert`.
Language bindings for Python or Java should be added at the application edge and
use these same Rust APIs, rather than exposing a second storage protocol.

The GooseFS provider uses Lance's public
`object_store::providers::opendal::from_opendal_operator` helper. That helper
owns the existing `OpendalStore` path normalization, identity transport, and
retry setup, so the plugin does not carry a second OpenDAL adapter.

The host validates the registry and the normal `ObjectStore` read/write path
with the existing `MemoryStoreProvider`, then constructs a GooseFS store through
the same registry. Real GooseFS read/write requires a reachable cluster; set
`GOOSEFS_MASTER_ADDR` or `GOOSEFS_CONFIG_FILE` to run that path against it.

The runtime dylib is built with `-C prefer-dynamic` so the runtime and plugin
share Rust standard-library and runtime state. On macOS, `libstd` is resolved
from the Rust toolchain's target library directory; production packaging must
ship it or set an appropriate rpath.

Build and run:

```bash
cd rust/extension-probe-rust
CARGO_PROFILE_RELEASE_LTO=false RUSTFLAGS='-C prefer-dynamic' cargo build --release
LIBDIR=$(rustc --print target-libdir)
DYLD_LIBRARY_PATH="$LIBDIR:$PWD/target/release" \
  cargo run --release -- \
  target/release/liblance_extension_goosefs_plugin.dylib
```

`cargo run` rebuilds the crates with the same `RUSTFLAGS`. If the local Cargo
configuration enables LTO, use `CARGO_PROFILE_RELEASE_LTO=false`; `rustc`
rejects `-C prefer-dynamic` together with LTO.

On Linux, use `.so` paths.
