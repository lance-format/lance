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

usage="usage: ent-3572-public-append-ci.sh <repo> <log-dir> <expected-head>"
repo=${1:?$usage}
log_dir=${2:?$usage}
expected_head=${3:?$usage}
mkdir -p "$log_dir"
cd "$repo"

export PATH="/home/ec2-user/.local/share/mbx/bin:/home/ec2-user/.cargo/bin:$PATH"
export RUST_BACKTRACE=1
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

trap 'status=$?; echo "$status" > "$log_dir/all.exit"; exit "$status"' EXIT
exec > >(tee "$log_dir/run.log") 2>&1

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

head=$(git rev-parse HEAD)
if [[ "$head" != "$expected_head" ]]; then
    fail "expected head $expected_head, found $head"
fi
if [[ -n $(git status --porcelain) ]]; then
    fail "source tree is not clean"
fi

echo "head=$head"
echo "expected_head=$expected_head"
echo "branch=$(git branch --show-current)"
echo "hostname=$(hostname)"
echo "runner_sha256=$(sha256sum "$0" | awk '{print $1}')"
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cargo=$(cargo --version)"
echo "rustc=$(rustc --version)"
echo "toolchain=$(rustup show active-toolchain)"
echo "cc=$(clang --version | head -n 1)"
echo "cxx=$(clang++ --version | head -n 1)"
echo "status_begin=$(git status --porcelain | wc -l) changes"

run_step fmt cargo fmt --all -- --check
run_step test-list cargo test -p lance --lib --profile ci --locked \
    dataset::metadata::tests::field_metadata_ -- --list

listed=$(grep -c 'dataset::metadata::tests::field_metadata.*: test$' "$log_dir/test-list.log" || true)
echo "listed_tests=$listed"
if ((listed != 4)); then
    fail "expected four field metadata tests, listed $listed"
fi

run_step tests cargo test -p lance --lib --profile ci --locked \
    dataset::metadata::tests::field_metadata_ -- --nocapture
grep -Eq 'test result: ok\. 4 passed; 0 failed; 0 ignored' "$log_dir/tests.log" \
    || fail "unit test footer did not report four passing tests"

run_step clippy cargo clippy -p lance --profile ci --locked --all-targets -- -D warnings

if [[ -n $(git status --porcelain) ]]; then
    fail "source tree changed during verification"
fi

echo "completed=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "verified_head=$(git rev-parse HEAD)"
