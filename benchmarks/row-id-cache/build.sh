#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
set -euo pipefail

if [[ $# != 2 || $(uname -s) != Linux ]]; then
    echo 'Usage (on the benchmark VM): build.sh <full-commit> <new-output-directory>' >&2
    exit 1
fi
helper_dir=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$helper_dir" rev-parse --show-toplevel)
revision=$(git -C "$repo" rev-parse "$1^{commit}")
[[ "$1" == "$revision" ]]
mkdir "$2"
output=$(cd "$2" && pwd)
mkdir "$output/source"
git -C "$repo" archive "$revision" | tar -x -C "$output/source"
mkdir -p "$output/source/rust/lance/src/bin"
cp "$helper_dir/harness.rs" "$output/source/rust/lance/src/bin/row_id_cache_bench.rs"

# Expose the actual resolver only in the exported benchmark copy. Main has no
# demand resolver; its adapter invokes the existing complete-index path.
python3 - "$output/source/rust/lance/src/dataset/rowids.rs" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
private = 'pub(crate) use resolve::resolve_row_ids;'
if private in source:
    source = source.replace(private, 'pub use resolve::resolve_row_ids;', 1)
else:
    assert 'fn resolve_row_ids(' not in source
    source += '''
/// Benchmark-only adapter for the complete-index lookup path.
pub async fn resolve_row_ids(
    dataset: &Dataset, ids: &[u64], with_deletions: bool,
) -> Result<Option<Vec<Option<RowAddress>>>> {
    get_row_id_index_with(dataset, with_deletions).await?
        .map(|index| index.get_many(ids)).transpose()
}
'''
path.write_text(source)
PY

export ROW_ID_BENCH_REVISION="$revision"
export CARGO_TERM_COLOR=always
cd "$output/source"
command -v cargo > "$output/cargo-path.txt"
rustc -Vv > "$output/rustc.txt"
uname -a > "$output/host.txt"
# Configure the repository's pinned toolchain and managed Cargo cache first.
cargo build --locked -p lance --bin row_id_cache_bench --profile release-with-debug \
    --message-format=json-render-diagnostics | tee "$output/build.jsonl"
python3 - "$output" <<'PY'
from pathlib import Path
import hashlib
import json
import shutil
import sys

root = Path(sys.argv[1])
artifacts = [json.loads(line) for line in (root / 'build.jsonl').read_text().splitlines()
             if line.startswith('{')]
binary = next(item['executable'] for item in artifacts
              if item.get('reason') == 'compiler-artifact'
              and item.get('target', {}).get('name') == 'row_id_cache_bench')
shutil.copy2(binary, root / 'row_id_cache_bench')
proof = {}
for name in ['row_id_cache_bench', 'source/rust/lance/src/bin/row_id_cache_bench.rs',
             'source/rust/lance/src/dataset/rowids.rs']:
    with (root / name).open('rb') as stream:
        proof[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
(root / 'sha256.json').write_text(json.dumps(proof, indent=2) + '\n')
PY
