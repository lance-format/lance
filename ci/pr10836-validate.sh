#!/usr/bin/env bash

set -euo pipefail

readonly source_sha="d331235db44beabff5d37713529f3a1cd4075c4a"
readonly step="${1:-}"
readonly log_dir="${PR10836_LOG_DIR:?PR10836_LOG_DIR must name a stable output directory}"
readonly checkout_sha="$(git rev-parse HEAD)"
readonly log_file="${log_dir}/${step}.log"
readonly exit_file="${log_dir}/${step}.exit"
readonly log_tmp="${log_file}.tmp"
readonly exit_tmp="${exit_file}.tmp"

if ! git merge-base --is-ancestor "${source_sha}" HEAD; then
    printf 'source commit %s is not an ancestor of checkout %s\n' \
        "${source_sha}" "${checkout_sha}" >&2
    exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
    printf 'validation worktree is not clean at %s\n' "${checkout_sha}" >&2
    exit 2
fi

run_step() {
    case "${step}" in
        targeted-vector-filter-fts)
            cargo test --profile ci -p lance --lib test_vector_filter_fts_search
            ;;
        targeted-fts-filter-vector)
            cargo test --profile ci -p lance --lib test_fts_filter_vector_search
            ;;
        pure-fts-limit-offset)
            cargo test --profile ci -p lance --lib test_fts_limit_offset
            ;;
        fts-ordered-limit-offset)
            cargo test --profile ci -p lance --lib test_fts_ordered_limit_offset
            ;;
        explicit-fts-cap)
            cargo test --profile ci -p lance --lib test_fts_phrase_stale_rows_honor_query_limit
            ;;
        fmt-check)
            cargo fmt --all -- --check
            ;;
        clippy)
            cargo clippy --profile ci --all --tests --benches -- -D warnings
            ;;
        unit-lib)
            cargo test --profile ci -p lance --lib
            ;;
        *)
            printf 'unknown validation step: %s\n' "${step}" >&2
            return 2
            ;;
    esac
}

mkdir -p "${log_dir}"
printf 'checkout_sha=%s\nsource_sha=%s\nstep=%s\nstarted_at=%s\n' \
    "${checkout_sha}" "${source_sha}" "${step}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${log_tmp}"

set +e
run_step 2>&1 | tee -a "${log_tmp}"
status=${PIPESTATUS[0]}
set -e

printf 'finished_at=%s\nexit_status=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" >> "${log_tmp}"
mv -f "${log_tmp}" "${log_file}"
printf '%s\n' "${status}" > "${exit_tmp}"
mv -f "${exit_tmp}" "${exit_file}"
exit "${status}"
