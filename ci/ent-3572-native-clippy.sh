#!/usr/bin/env bash
#
# Copyright 2026 Lance Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

repo=${1:?usage: ent-3572-native-clippy.sh <repo> <log-dir>}
log_dir=${2:?usage: ent-3572-native-clippy.sh <repo> <log-dir>}
mkdir -p "$log_dir"
cd "$repo"

export PATH="/home/ec2-user/.local/share/mbx/bin:/home/ec2-user/.cargo/bin:$PATH"
export RUST_BACKTRACE=1

trap 'status=$?; echo "$status" > "$log_dir/native-all.exit"; exit "$status"' EXIT
exec > >(tee "$log_dir/native-run.log") 2>&1

fail() {
    echo "FAIL: $*"
    exit 1
}

run_step() {
    local name=$1
    shift
    echo "COMMAND $name: $*"
    set +e
    "$@" >"$log_dir/$name.log" 2>&1
    local status=$?
    set -e
    echo "$status" >"$log_dir/$name.exit"
    echo "RESULT $name exit=$status"
    if ((status != 0)); then
        tail -n 100 "$log_dir/$name.log"
        exit "$status"
    fi
}

echo "head=$(git rev-parse HEAD)"
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cargo=$(cargo --version)"
echo "rustc=$(rustc --version)"
echo "toolchain=$(rustup show active-toolchain)"
echo "cc=${CC:-unset}"
echo "cxx=${CXX:-unset}"
echo "status_begin=$(git status --porcelain | wc -l) changes"

if [[ -n $(git status --porcelain) ]]; then
    fail "source tree is not clean"
fi

features=$(cargo metadata --format-version=1 --no-deps | jq -r '.packages[] | .features | keys | .[]' | sort | uniq | paste -s -d, -)
echo "features=$features"

run_step clippy-native cargo clippy --profile ci --locked --features "$features" --all-targets -- -D warnings
run_step clippy-arrow cargo clippy --profile ci --locked --workspace --features "$features" --test arrow_ordering -- -D warnings

if [[ -n $(git status --porcelain) ]]; then
    fail "source tree changed during verification"
fi

echo "completed=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "verified_head=$(git rev-parse HEAD)"
