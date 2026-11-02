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

repo=${1:?usage: ent-3572-public-ci.sh <repo> <log-dir>}
log_dir=${2:?usage: ent-3572-public-ci.sh <repo> <log-dir>}
mkdir -p "$log_dir"
cd "$repo"

export PATH="/home/ec2-user/.local/share/mbx/bin:/home/ec2-user/.cargo/bin:$PATH"
export RUST_BACKTRACE=1

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

echo "head=$(git rev-parse HEAD)"
echo "branch=$(git branch --show-current)"
echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cargo=$(cargo --version)"
echo "rustc=$(rustc --version)"
echo "toolchain=$(rustup show active-toolchain)"
echo "status_begin=$(git status --porcelain | wc -l) changes"

if [[ -n $(git status --porcelain) ]]; then
    fail "source tree is not clean"
fi

run_step fmt cargo fmt --all -- --check
run_step test-list cargo test -p lance --lib --profile ci --locked field_metadata_expected -- --list

listed=$(grep -c 'field_metadata_expected.*: test$' "$log_dir/test-list.log" || true)
echo "listed_tests=$listed"
if ((listed < 3)); then
    fail "expected at least three field_metadata_expected tests, listed $listed"
fi

run_step tests cargo test -p lance --lib --profile ci --locked field_metadata_expected -- --nocapture
grep -Eq 'test result: ok\. 3 passed' "$log_dir/tests.log" || fail "unit test footer did not report three passing tests"

run_step doctest cargo test -p lance --doc --profile ci --locked with_expected_latest_version
grep -Eq 'test result: ok\. 1 passed' "$log_dir/doctest.log" || fail "doctest footer did not report one passing test"

run_step clippy cargo clippy -p lance --profile ci --locked --all-targets -- -D warnings

if [[ -n $(git status --porcelain) ]]; then
    fail "source tree changed during verification"
fi

echo "completed=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "verified_head=$(git rev-parse HEAD)"
