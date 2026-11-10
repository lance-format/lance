#!/usr/bin/env bash

set -euo pipefail

log_dir=${1:?usage: ci/pr-10863-validation.sh <log-dir> <source-commit> <main-commit>}
source_commit=${2:?usage: ci/pr-10863-validation.sh <log-dir> <source-commit> <main-commit>}
main_commit=${3:?usage: ci/pr-10863-validation.sh <log-dir> <source-commit> <main-commit>}

mkdir -p "$log_dir"
exec > >(tee "$log_dir/runner.log") 2>&1

status_log="$log_dir/status.tsv"
printf 'name\tstatus\n' >"$status_log"

on_exit() {
    local status=$?
    printf 'exit_status=%s\n' "$status" | tee "$log_dir/exit_status.log"
}
trap on_exit EXIT

run_step() {
    local name=$1
    shift
    local log="$log_dir/$name.log"
    local command_text
    printf -v command_text '%q ' "$@"

    printf '\n>>> %s\ncommand=%s\n' "$name" "$command_text"
    set +e
    "$@" >"$log" 2>&1
    local status=$?
    set -e
    cat "$log"
    printf 'exit_status=%s\n' "$status" >>"$log"
    printf '%s\t%s\n' "$name" "$status" >>"$status_log"
    return "$status"
}

cd "$(git rev-parse --show-toplevel)"

{
    printf 'runner_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'source_commit=%s\n' "$source_commit"
    printf 'main_commit=%s\n' "$main_commit"
    printf 'status_before=%s\n' "$(git status --porcelain=v1)"
    printf 'runner_delta=%s\n' "$(git diff --name-only "$source_commit"..HEAD | paste -s -d, -)"
    rustup show active-toolchain
    cargo --version
    rustc --version
} | tee "$log_dir/environment.log"

test "$(git diff --name-only "$source_commit"..HEAD)" = "ci/pr-10863-validation.sh"
git merge-base --is-ancestor "$main_commit" HEAD
test -z "$(git status --porcelain=v1)"

run_step codegen_fmt \
    cargo fmt --manifest-path protos/codegen/Cargo.toml -- --check
run_step codegen_clippy \
    cargo clippy --locked --manifest-path protos/codegen/Cargo.toml -- -D warnings
run_step codegen_check \
    env PROTOC=/protoc-must-not-be-used \
    cargo run --locked --manifest-path protos/codegen/Cargo.toml -- --check
run_step proto_comments \
    python3 ci/check_proto_comments.py protos/filtered_read.proto
run_step root_fmt \
    cargo fmt --all -- --check
run_step options_none \
    cargo test --profile ci -p lance --lib test_options_roundtrip_basic
run_step options_nonempty_legacy \
    cargo test --profile ci -p lance --lib test_options_roundtrip_with_fragments
run_step exec_codec_none \
    cargo test --profile ci -p lance --lib test_exec_to_proto_roundtrip
run_step exec_empty_selection \
    cargo test --profile ci -p lance --lib test_exec_roundtrip_keeps_empty_fragment_selection
run_step root_clippy \
    cargo clippy --profile ci --all --tests --benches -- -D warnings

{
    printf 'runner_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'status_after=%s\n' "$(git status --porcelain=v1)"
} | tee "$log_dir/final-state.log"

test -z "$(git status --porcelain=v1)"
