#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
#
# Compile lance.kotoba with Kotoba 0.7.2 (wasm32 / i64-v1) and assert the
# vendored fixture's magic + version header. Missing tools or mismatches
# fail. This script does not treat a skip as a pass.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="${ROOT}/lance.kotoba"
FIXTURE="${ROOT}/fixtures/tiny.lance"
KOTOBA="${KOTOBA:-kotoba}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT

# Published kotoba-lang/kotoba v0.7.2 linux-amd64 executable digest
# (kotoba-linux-amd64.json -> binarySha256).
KOTOBA_072_LINUX_AMD64_SHA256="51f696d7d08b92d3d0f34ac5a32dc846ce63aeab3295b1baf74f8fc78a85601c"

# The repo has a kotoba/ directory. Directories are executable, so -x kotoba
# must not win over the CLI binary (CI failed: sha256sum: kotoba: Is a directory).
if [[ -f "${KOTOBA}" && -x "${KOTOBA}" ]]; then
  KOTOBA_BIN="${KOTOBA}"
elif command -v "${KOTOBA}" >/dev/null 2>&1; then
  KOTOBA_BIN="$(command -v "${KOTOBA}")"
else
  echo "fail: kotoba 0.7.2 is required and was not found (${KOTOBA})" >&2
  exit 1
fi
if [[ ! -f "${KOTOBA_BIN}" || -d "${KOTOBA_BIN}" ]]; then
  echo "fail: kotoba resolved to a directory, not the 0.7.2 CLI (${KOTOBA_BIN})" >&2
  exit 1
fi

if [[ ! -f "${SRC}" ]]; then
  echo "fail: missing module ${SRC}" >&2
  exit 1
fi
if [[ ! -f "${FIXTURE}" ]]; then
  echo "fail: missing fixture ${FIXTURE}" >&2
  exit 1
fi

if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
  actual_sha="$(sha256sum "${KOTOBA_BIN}" | awk '{print $1}')"
  if [[ "${actual_sha}" != "${KOTOBA_072_LINUX_AMD64_SHA256}" ]]; then
    echo "fail: kotoba binary is not v0.7.2 linux-amd64" >&2
    echo "  path:   ${KOTOBA_BIN}" >&2
    echo "  actual: ${actual_sha}" >&2
    echo "  want:   ${KOTOBA_072_LINUX_AMD64_SHA256}" >&2
    exit 1
  fi
  echo "kotoba binary matches v0.7.2 linux-amd64 (${actual_sha})"
else
  echo "note: binary SHA-256 pin is linux-amd64 only; host is $(uname -s)/$(uname -m)"
  echo "note: functional compile/run still decide pass/fail"
fi

python3 - "${SRC}" "${FIXTURE}" <<'PY'
import pathlib
import sys

src = pathlib.Path(sys.argv[1]).read_text()
blob = pathlib.Path(sys.argv[2]).read_bytes()
if len(blob) != 93:
    raise SystemExit(f"fail: fixture length {len(blob)} != 93")
tail = blob[-8:]
if tail[4:] != b"LANC":
    raise SystemExit(f"fail: fixture tail is not LANC: {tail.hex()}")
major = int.from_bytes(tail[0:2], "little")
minor = int.from_bytes(tail[2:4], "little")
if major != 0 or minor != 1:
    raise SystemExit(f"fail: unexpected fixture header major={major} minor={minor}")

required = [
    "(lance-magic? 76 65 78 67)",
    "(u16-le 0 0)",
    "(u16-le 1 0)",
]
missing = [frag for frag in required if frag not in src]
if missing:
    raise SystemExit(f"fail: lance.kotoba is missing fixture tail literals: {missing}")
if "(u16-le 65 0)" in src:
    raise SystemExit("fail: lance.kotoba still reads past the version header")
print(f"fixture header: size=93 major={major} minor={minor} magic=LANC")
PY

COMPILE_JSON="${WORKDIR}/compile.json"
WASM_OUT="${WORKDIR}/lance.wasm"
"${KOTOBA_BIN}" compile "${SRC}" --target wasm -o "${WASM_OUT}" --json >"${COMPILE_JSON}"

python3 - "${COMPILE_JSON}" "${WASM_OUT}" <<'PY'
import json
import pathlib
import sys

WASM_IMPORT_SECTION = 2


def read_uleb128(buf, i):
    shift = 0
    value = 0
    while True:
        if i >= len(buf):
            raise ValueError("truncated uleb128")
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return value, i
        shift += 7
        if shift > 35:
            raise ValueError("uleb128 too long")


def wasm_import_section(buf):
    if buf[:4] != b"\x00asm":
        raise ValueError(f"artifact magic {buf[:4]!r} is not wasm")
    if len(buf) < 8:
        raise ValueError("truncated wasm header")
    i = 8
    found = False
    import_count = None
    while i < len(buf):
        section_id = buf[i]
        i += 1
        size, i = read_uleb128(buf, i)
        end = i + size
        if end > len(buf):
            raise ValueError("truncated wasm section")
        payload = buf[i:end]
        i = end
        if section_id == WASM_IMPORT_SECTION:
            found = True
            import_count, _ = read_uleb128(payload, 0) if payload else (0, 0)
    return found, import_count


if wasm_import_section(b"\x00asm\x01\x00\x00\x00") != (False, None):
    raise SystemExit("fail: import-section checker failed on a no-section wasm")
if wasm_import_section(b"\x00asm\x01\x00\x00\x00\x02\x01\x00") != (True, 0):
    raise SystemExit("fail: import-section checker failed to see an import section")

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
if report.get("kotoba.cli/ok?") is not True:
    raise SystemExit(f"fail: compile did not succeed: {report}")
if report.get("kotoba.cli/code") != "emitted":
    raise SystemExit(
        f"fail: compile JSON code is {report.get('kotoba.cli/code')!r}, want 'emitted'"
    )
data = report.get("kotoba.cli/data") or {}
profile = data.get("value-profile")
target = (data.get("compatibility") or {}).get("target")
if profile != "i64-v1":
    raise SystemExit(f"fail: value-profile {profile!r} != 'i64-v1'")
if target != "wasm32-kotoba-v1":
    raise SystemExit(f"fail: target {target!r} != 'wasm32-kotoba-v1'")
wasm = pathlib.Path(sys.argv[2]).read_bytes()
if wasm[:4] != b"\0asm":
    raise SystemExit("fail: compile output is not a wasm32 module (missing \\0asm)")
has_imports, import_count = wasm_import_section(wasm)
if has_imports:
    raise SystemExit(
        f"fail: wasm has import section (id 2, count={import_count}); FFI is out of v1"
    )
print(f"compile: code=emitted target={target} value-profile={profile} import-section=absent")
PY

RUN_JSON="${WORKDIR}/run.json"
"${KOTOBA_BIN}" run "${SRC}" --json >"${RUN_JSON}"

python3 - "${RUN_JSON}" "${FIXTURE}" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
if report.get("kotoba.cli/ok?") is not True:
    raise SystemExit(f"fail: run did not succeed: {report}")
value = ((report.get("kotoba.cli/data") or {}).get("kotoba.runtime/result") or {}).get(
    "kotoba.runtime/value"
)
blob = pathlib.Path(sys.argv[2]).read_bytes()
tail = blob[-8:]
major = int.from_bytes(tail[0:2], "little")
minor = int.from_bytes(tail[2:4], "little")
magic_ok = 1 if tail[4:] == b"LANC" else 0
expected = magic_ok * 1000000 + major * 1000 + minor
if value != expected:
    raise SystemExit(
        f"fail: packed header {value!r} != {expected} "
        f"(magic_ok={magic_ok} major={major} minor={minor})"
    )
print(f"run: packed={value} magic_ok={magic_ok} major={major} minor={minor}")
PY

echo "kotoba v1 header checks passed"
