# Blob reference v1 conformance inputs

`conformance.json` defines logical payloads, successful demands, and invalid ranges
shared by the Rust, Python, and Java/JNI tests. Payload strings are UTF-8 bytes;
`null` and `""` are distinct. Request indices and invalid-range values are decimal
strings so runtimes without unsigned JSON integers preserve all 64 bits.

This is an execution API fixture, not a persisted Lance dataset. Each producer
writes a current Blob V2 dataset and captures real descriptors. It must pass encoded
references and source contexts to a worker that independently opens the exact
snapshot. Locators must never be reconstructed by the test or the engine.

Run the independent-worker handoff and I/O comparison from `python/`:

```shell
uv run python python/benchmarks/blob_reference_conformance.py
```

The runner also compares eager payload reads with metadata-only, selected-value,
and range workloads on separate payload objects. It checks equal logical output
and reports measured counters and serialized exchange sizes. It makes no latency
claim. See the [engine integration guide](../../docs/src/guide/blob_reference_integration.md)
for metric scope and deployment limits.
