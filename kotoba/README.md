# Kotoba binding for Lance (honest v1)

This tree is a first-class sibling of `java/`, `python/`, and `rust/` **on this fork**. It is not a replacement for those implementations and is not proposed to [lance-format/lance](https://github.com/lance-format/lance).

## Scope

v1 is **magic and version header only** (or, in other bindings, one record). This tree does the header:

- trailing 4-byte magic `LANC`
- the 4 bytes before it: `u16` major and `u16` minor, little-endian

It does **not** read a dataset, fragment body, manifest, page, or record. It is not a query engine. It is **not robotics-ready**.

The compiler contract is Kotoba CLI **0.7.2**, `wasm32-kotoba-v1`, value profile `i64-v1`. No FFI. No IEEE floats.

## Fixture

`fixtures/tiny.lance` is a 93-byte historical v1 Lance file copied from this repository's `test_data/v0.8.0/migrated_from_v0.7.5` data file. Last 8 bytes (little-endian):

```
00 00 01 00 4c 41 4e 43
major=0  minor=1  magic=LANC
```

`lance.kotoba` embeds those tail bytes as i64 scalars and returns:

```
magic_ok * 1000000 + major * 1000 + minor
= 1000001
```

## Check

Requires Kotoba CLI 0.7.2 on `PATH` (or `$KOTOBA`). On linux-amd64, `checks.sh` also pins the published v0.7.2 binary SHA-256. Missing CLI, checksum mismatch, compile failure, or field mismatch is a failure. The script does not skip those as success.

```sh
# from this directory, with kotoba 0.7.2 on PATH
./checks.sh
```
