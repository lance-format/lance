#!/usr/bin/env bash

set -euo pipefail

log_dir=${1:?usage: ci/pr-10863-validation.sh <log-dir> <source-commit> <main-commit> [all|tests-only]}
source_commit=${2:?usage: ci/pr-10863-validation.sh <log-dir> <source-commit> <main-commit> [all|tests-only]}
main_commit=${3:?usage: ci/pr-10863-validation.sh <log-dir> <source-commit> <main-commit> [all|tests-only]}
mode=${4:-all}

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

run_test_step() {
    local name=$1
    local test_name=$2
    shift 2

    run_step "$name" "$@"
    local log="$log_dir/$name.log"
    if ! grep -Eq "^test .*${test_name}.* \\.\\.\\. ok$" "$log"; then
        printf '%s did not execute %s\n' "$name" "$test_name" >&2
        printf '%s\tmissing-test\n' "$name" >>"$status_log"
        return 1
    fi
    if ! grep -Eq '^test result: ok\. [1-9][0-9]* passed; 0 failed;' "$log"; then
        printf '%s did not report a nonzero passing test count\n' "$name" >&2
        printf '%s\tzero-passed\n' "$name" >>"$status_log"
        return 1
    fi
}

run_substrait_tests() {
    run_test_step options_none test_options_roundtrip_basic \
        cargo test --profile ci -p lance --features substrait --lib test_options_roundtrip_basic
    run_test_step options_nonempty_legacy test_options_roundtrip_with_fragments \
        cargo test --profile ci -p lance --features substrait --lib test_options_roundtrip_with_fragments
    run_test_step exec_codec_none test_exec_to_proto_roundtrip \
        cargo test --profile ci -p lance --features substrait --lib test_exec_to_proto_roundtrip
    run_test_step exec_empty_selection test_exec_roundtrip_keeps_empty_fragment_selection \
        cargo test --profile ci -p lance --features substrait --lib test_exec_roundtrip_keeps_empty_fragment_selection
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

case "$mode" in
    all)
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
        run_substrait_tests
        run_step root_clippy \
            cargo clippy --profile ci --all --tests --benches -- -D warnings
        ;;
    tests-only)
        run_substrait_tests
        ;;
    *)
        printf 'invalid mode: %s\n' "$mode" >&2
        exit 2
        ;;
esac

{
    printf 'runner_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'status_after=%s\n' "$(git status --porcelain=v1)"
} | tee "$log_dir/final-state.log"

test -z "$(git status --porcelain=v1)"
