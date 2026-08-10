#!/bin/bash
#
# Lance promises that non-AWS cloud builds can be built without aws-lc-rs.
# Consumers building slim images for a single backend (e.g. GCS-only) rely on
# this: aws-lc-rs compiles C/assembly through its cmake build dependency,
# which bloats builds, complicates cross-compilation, and forces a second
# crypto provider on deployments that standardize on ring. This script
# enforces the promise by resolving each non-AWS backend under the
# provider-neutral TLS feature (tls-no-provider) and failing if aws-lc-rs or
# cmake appears.
#
# It checks the FUNCTIONAL configuration (backend + tls-no-provider), not the
# bare backend: opendal's HTTP transport is opt-in, so a bare backend has no
# TLS stack and would pass this check vacuously while failing at runtime. To
# guard against that, each combination must also still resolve a reqwest HTTP
# transport (opendal-http-transport-reqwest); a missing transport is an error.
#
# The usual way this regresses is a dependency bump changing a TLS default,
# not a local change: opendal's default reqwest transport pins aws-lc-rs, and
# its reqwest-rustls-* feature aliases re-enable it, so lance-io wires the
# granular http-transport-reqwest-rustls-no-provider feature instead.
#
# If this check fails, fix the feature wiring rather than exempting the crate.
# Feature combos that include the AWS SDK are not checked: it links aws-lc-rs
# itself via aws-smithy-http-client.

set -euo pipefail

cd "$(dirname "$0")/.."

FORBIDDEN=(aws-lc-rs aws-lc-sys aws-lc-fips-sys cmake)
REQUIRED=(opendal-http-transport-reqwest)

check() {
    local desc="$1"
    shift

    local deps
    deps=$(cargo tree --locked -e normal,build --prefix none --format '{p}' "$@" | awk '{print $1}' | sort -u)

    local failed=0
    for crate in "${FORBIDDEN[@]}"; do
        if grep -qx "$crate" <<<"$deps"; then
            echo "error: forbidden dependency '$crate' found in $desc, pulled in via:"
            cargo tree --locked -e normal,build "$@" -i "$crate"
            failed=1
        fi
    done
    for crate in "${REQUIRED[@]}"; do
        if ! grep -qx "$crate" <<<"$deps"; then
            echo "error: $desc is missing HTTP transport '$crate'; the forbidden-dependency check would pass vacuously. Wire a provider-neutral TLS transport."
            failed=1
        fi
    done
    if [[ $failed -ne 0 ]]; then
        exit 1
    fi
    echo "ok: $desc is free of {${FORBIDDEN[*]}} and has an HTTP transport"
}

check "lance-io (gcp, tls-no-provider)" --manifest-path rust/lance-io/Cargo.toml --no-default-features --features gcp,tls-no-provider
check "lance-io (azure, tls-no-provider)" --manifest-path rust/lance-io/Cargo.toml --no-default-features --features azure,tls-no-provider
check "lance-io (oss, tls-no-provider)" --manifest-path rust/lance-io/Cargo.toml --no-default-features --features oss,tls-no-provider
