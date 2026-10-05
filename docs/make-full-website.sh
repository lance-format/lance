#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: make-full-website.sh

Override any repo path with the matching environment variable:
  LANCE_NAMESPACE_REPO
  LANCE_NAMESPACE_IMPLS_REPO
  LANCE_SPARK_REPO
  LANCE_RAY_REPO
  LANCE_TRINO_REPO
  LANCE_DUCKDB_REPO
  LANCE_HUGGINGFACE_REPO
  LANCE_CONTEXT_REPO
Defaults:
  LANCE_NAMESPACE_REPO=$HOME/oss/lance-namespace
  LANCE_NAMESPACE_IMPLS_REPO=$HOME/oss/lance-namespace-impls
  LANCE_SPARK_REPO=$HOME/oss/lance-spark
  LANCE_RAY_REPO=$HOME/oss/lance-ray
  LANCE_TRINO_REPO=$HOME/oss/lance-trino
  LANCE_DUCKDB_REPO=$HOME/oss/lance-duckdb
  LANCE_HUGGINGFACE_REPO=$HOME/oss/lance-huggingface
  LANCE_CONTEXT_REPO=$HOME/oss/lance-context
EOF
}

normalize_path() {
    local path="$1"

    case "$path" in
        \~)
            printf '%s\n' "$HOME"
            ;;
        \~/*)
            printf '%s\n' "$HOME/${path:2}"
            ;;
        *)
            printf '%s\n' "$path"
            ;;
    esac
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
docs_src="$script_dir/src"

if [ "$#" -gt 0 ]; then
    if [ "$#" -eq 1 ] && [ "$1" = "--help" ]; then
        usage
        exit 0
    fi
    usage
    exit 1
fi

namespace_repo_input=${LANCE_NAMESPACE_REPO:-$HOME/oss/lance-namespace}
namespace_impls_repo_input=${LANCE_NAMESPACE_IMPLS_REPO:-$HOME/oss/lance-namespace-impls}
spark_repo_input=${LANCE_SPARK_REPO:-$HOME/oss/lance-spark}
ray_repo_input=${LANCE_RAY_REPO:-$HOME/oss/lance-ray}
trino_repo_input=${LANCE_TRINO_REPO:-$HOME/oss/lance-trino}
duckdb_repo_input=${LANCE_DUCKDB_REPO:-$HOME/oss/lance-duckdb}
huggingface_repo_input=${LANCE_HUGGINGFACE_REPO:-$HOME/oss/lance-huggingface}
context_repo_input=${LANCE_CONTEXT_REPO:-$HOME/oss/lance-context}

copy_docs_dir() {
    local source_dir="$1"
    local target_dir="$2"

    if [ -d "$source_dir" ]; then
        mkdir -p "$(dirname "$target_dir")"
        rm -rf "$target_dir"
        cp -R "$source_dir" "$target_dir"
        return 0
    fi

    return 1
}

copy_file_if_exists() {
    local source_file="$1"
    local target_file="$2"

    if [ -f "$source_file" ]; then
        mkdir -p "$(dirname "$target_file")"
        cp "$source_file" "$target_file"
        return 0
    fi

    return 1
}

copy_duckdb_docs() {
    local source_dir="$1"
    local target_dir="$2"

    if [ ! -d "$source_dir" ]; then
        return 1
    fi

    rm -rf "$target_dir"
    mkdir -p "$target_dir"
    cp "$source_dir/.pages" "$target_dir/.pages"
    cp "$source_dir/index.md" "$target_dir/index.md"
    cp "$source_dir/sql.md" "$target_dir/sql.md"
    cp "$source_dir/rest.md" "$target_dir/rest.md"
    cp "$source_dir/cloud.md" "$target_dir/cloud.md"

    return 0
}

resolve_repo_dir() {
    local repo_path="$1"

    repo_path=$(normalize_path "$repo_path")

    if [ -d "$repo_path" ]; then
        (cd -- "$repo_path" && pwd)
        return
    fi

    printf '%s\n' "$repo_path"
}

warn_missing_repo() {
    local repo_label="$1"
    local repo_path="$2"

    echo "Warning: $repo_label repo not found at '$repo_path'; keeping placeholder docs." >&2
}

namespace_repo=$(resolve_repo_dir "$namespace_repo_input")
namespace_impls_repo=$(resolve_repo_dir "$namespace_impls_repo_input")
spark_repo=$(resolve_repo_dir "$spark_repo_input")
ray_repo=$(resolve_repo_dir "$ray_repo_input")
trino_repo=$(resolve_repo_dir "$trino_repo_input")
duckdb_repo=$(resolve_repo_dir "$duckdb_repo_input")
huggingface_repo=$(resolve_repo_dir "$huggingface_repo_input")
context_repo=$(resolve_repo_dir "$context_repo_input")

"$script_dir/clean-full-website.sh"

if copy_docs_dir "$namespace_repo/docs/src/catalog" "$docs_src/format/catalog"; then
    :
else
    warn_missing_repo "Lance Namespace catalog docs" "$namespace_repo/docs/src/catalog"
fi

if copy_docs_dir "$namespace_repo/docs/src/namespace" "$docs_src/format/namespace"; then
    copy_file_if_exists "$namespace_repo/docs/src/overview.png" "$docs_src/format/overview.png"
    copy_file_if_exists "$namespace_repo/docs/src/java-sdk-example.png" "$docs_src/format/java-sdk-example.png"
    :
else
    warn_missing_repo "Lance Namespace namespace docs" "$namespace_repo/docs/src/namespace"
fi

if [ -f "$docs_src/format/namespace/operations/index.md" ]; then
python3 - <<'PY' "$docs_src/format/namespace/operations/index.md"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
path.write_text(text.replace('[Models](models/)', '[Models](models/index.md)'))
PY
fi

mkdir -p "$docs_src/format/namespace/operations/models"
cat > "$docs_src/format/namespace/operations/models/index.md" <<'EOF'
# Operation Models

This section contains the generated request and response models for the Lance Namespace operations.

These pages define the JSON schemas referenced by the [operations overview](../index.md) and the [REST Catalog API specification](../../../catalog/rest/index.md).
EOF

supported_catalogs_dir="$docs_src/format/namespace/supported-catalogs"
if [ -d "$namespace_impls_repo/docs/src" ]; then
while IFS= read -r -d '' source_file; do
    target_file="$supported_catalogs_dir/$(basename "$source_file")"
    if [ -e "$target_file" ]; then
        echo "Refusing to overwrite existing supported catalog file: $target_file" >&2
        exit 1
    fi
    cp "$source_file" "$target_file"
done < <(find "$namespace_impls_repo/docs/src" -maxdepth 1 -type f -name '*.md' ! -name 'index.md' -print0 | sort -z)

python3 - <<'PY' "$namespace_impls_repo/docs/src/.pages" "$supported_catalogs_dir/.pages"
from pathlib import Path
import sys

impl_pages = Path(sys.argv[1]).read_text().splitlines()
target = Path(sys.argv[2])
existing = target.read_text().splitlines()

impl_entries = []
for line in impl_pages:
    stripped = line.strip()
    if not stripped.startswith('- '):
        continue
    entry = stripped[2:]
    if entry == 'Introduction: index.md':
        continue
    impl_entries.append(f'  - {entry}')

output = []
inserted = False
for line in existing:
    # Keep namespace-owned entries first and insert impl-owned entries immediately
    # before the template item when it exists.
    if line.strip() == '- Template: template.md' and not inserted:
        output.extend(impl_entries)
        inserted = True
    output.append(line)

if not inserted:
    output.extend(impl_entries)

target.write_text('\n'.join(output) + '\n')
PY
else
    warn_missing_repo "Lance Namespace Impls docs" "$namespace_impls_repo/docs/src"
fi

integration_entries=(
)

while IFS= read -r line; do
    integration_entries+=("$line")
done < <(sed -n '2,$p' "$docs_src/integrations/.pages")

if copy_duckdb_docs "$duckdb_repo/docs" "$docs_src/integrations/duckdb"; then
    integration_entries+=("  - DuckDB: duckdb")
else
    warn_missing_repo "Lance DuckDB docs" "$duckdb_repo/docs"
fi

if copy_docs_dir "$huggingface_repo/docs/src" "$docs_src/integrations/huggingface"; then
    integration_entries+=("  - Huggingface: huggingface")
else
    warn_missing_repo "Lance HuggingFace docs" "$huggingface_repo/docs/src"
fi

if copy_docs_dir "$spark_repo/docs/src" "$docs_src/integrations/spark"; then
    python3 - <<'PY' "$docs_src/integrations/spark/operations/ddl/create-index.md"
from pathlib import Path
import sys

path = Path(sys.argv[1])
if path.exists():
    text = path.read_text()
    path.write_text(text.replace('https://lance.org/format/table/index/scalar/fts/#tokenizers', 'https://lance.org/format/index/scalar/fts/#tokenizers'))
PY
    integration_entries+=("  - Apache Spark: spark")
else
    warn_missing_repo "Lance Spark docs" "$spark_repo/docs/src"
fi

if copy_docs_dir "$ray_repo/docs/src" "$docs_src/integrations/ray"; then
    integration_entries+=("  - Ray: ray")
else
    warn_missing_repo "Lance Ray docs" "$ray_repo/docs/src"
fi

if copy_docs_dir "$trino_repo/docs/src" "$docs_src/integrations/trino"; then
    integration_entries+=("  - Trino: trino")
else
    warn_missing_repo "Lance Trino docs" "$trino_repo/docs/src"
fi

if copy_docs_dir "$context_repo/docs/src" "$docs_src/integrations/context"; then
    integration_entries+=("  - Lance Context: context")
else
    warn_missing_repo "Lance Context docs" "$context_repo/docs/src"
fi

{
    echo "nav:"
    for entry in "${integration_entries[@]}"; do
        [ -n "$entry" ] || continue
        echo "$entry"
    done
} > "$docs_src/integrations/.pages"

mkdir -p "$docs_src/community/project-specific/lance"
copy_file_if_exists "$repo_root/CONTRIBUTING.md" "$docs_src/community/project-specific/lance/general.md"
copy_file_if_exists "$repo_root/release_process.md" "$docs_src/community/project-specific/lance/release.md"
copy_file_if_exists "$repo_root/rust/CONTRIBUTING.md" "$docs_src/community/project-specific/lance/rust.md"
copy_file_if_exists "$repo_root/python/CONTRIBUTING.md" "$docs_src/community/project-specific/lance/python.md"
copy_file_if_exists "$repo_root/docs/CONTRIBUTING.md" "$docs_src/community/project-specific/lance/docs.md"

project_entries=(
    "  - index.md"
    "  - Lance: lance"
)

if copy_file_if_exists "$namespace_repo/CONTRIBUTING.md" "$docs_src/community/project-specific/namespace.md"; then
    project_entries+=("  - Lance Namespace: namespace.md")
fi

if copy_file_if_exists "$namespace_impls_repo/CONTRIBUTING.md" "$docs_src/community/project-specific/namespace-impls.md"; then
    project_entries+=("  - Lance Namespace Impls: namespace-impls.md")
fi

if copy_file_if_exists "$ray_repo/CONTRIBUTING.md" "$docs_src/community/project-specific/ray.md"; then
    project_entries+=("  - Lance Ray: ray.md")
fi

if copy_file_if_exists "$spark_repo/CONTRIBUTING.md" "$docs_src/community/project-specific/spark.md"; then
    project_entries+=("  - Lance Spark: spark.md")
fi

if copy_file_if_exists "$trino_repo/CONTRIBUTING.md" "$docs_src/community/project-specific/trino.md"; then
    project_entries+=("  - Lance Trino: trino.md")
fi

if copy_file_if_exists "$context_repo/CONTRIBUTING.md" "$docs_src/community/project-specific/context.md"; then
    project_entries+=("  - Lance Context: context.md")
fi

{
    echo "nav:"
    for entry in "${project_entries[@]}"; do
        echo "$entry"
    done
} > "$docs_src/community/project-specific/.pages"

cat > "$docs_src/community/project-specific/lance/.pages" <<'EOF'
nav:
  - General: general.md
  - Release: release.md
  - Rust: rust.md
  - Python: python.md
  - Docs: docs.md
EOF
