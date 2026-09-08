#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
docs_src="$script_dir/src"

rm -rf "$docs_src/format/catalog"
rm -rf "$docs_src/format/namespace"
rm -f "$docs_src/format/layout.png"
rm -f "$docs_src/format/overview.png"
rm -f "$docs_src/format/java-sdk-example.png"
rm -rf "$docs_src/integrations/huggingface"
rm -rf "$docs_src/integrations/duckdb"
rm -rf "$docs_src/integrations/spark"
rm -rf "$docs_src/integrations/ray"
rm -rf "$docs_src/integrations/trino"
rm -rf "$docs_src/integrations/context"
rm -f "$docs_src/community/project-specific/.pages"
rm -rf "$docs_src/community/project-specific/lance"
rm -f "$docs_src/community/project-specific/namespace.md"
rm -f "$docs_src/community/project-specific/namespace-impls.md"
rm -f "$docs_src/community/project-specific/ray.md"
rm -f "$docs_src/community/project-specific/spark.md"
rm -f "$docs_src/community/project-specific/trino.md"
rm -f "$docs_src/community/project-specific/context.md"

cat > "$docs_src/format/.pages" <<'EOF'
nav:
  - Overview: index.md
  - File Format: file
  - Table Format: table
  - Index Formats: index
  - Catalog Specs: catalog
  - Namespace Client Spec: namespace
EOF

cat > "$docs_src/integrations/.pages" <<'EOF'
nav:
  - Overview: index.md
  - Apache DataFusion: datafusion.md
  - PostgreSQL: https://github.com/lance-format/pglance
  - PyTorch: pytorch.md
  - TensorFlow: tensorflow.md
EOF

mkdir -p "$docs_src/format/catalog/dir"
mkdir -p "$docs_src/format/catalog/rest"
mkdir -p "$docs_src/format/namespace/operations/models"
mkdir -p "$docs_src/format/namespace/supported-catalogs"

cat > "$docs_src/format/catalog/.pages" <<'EOF'
title: Catalog Specs
nav:
  - Overview: index.md
  - Directory Catalog: dir
  - REST Catalog: rest
EOF

cat > "$docs_src/format/catalog/index.md" <<'EOF'
# Catalog Specs

This section describes how Lance catalogs organize, discover, and coordinate Lance tables.

When a local `lance-namespace` checkout with split catalog docs is available, `docs/make-full-website.sh` replaces these placeholders with the latest source content.

See also:

- [Directory Catalog](dir/index.md)
- [REST Catalog](rest/index.md)
- [Namespace Client Spec](../namespace/index.md)
EOF

cat > "$docs_src/format/catalog/dir/index.md" <<'EOF'
# Directory Catalog

The Directory Catalog is the storage-native catalog format for Lance.

This placeholder page keeps the local website buildable when external catalog docs are not available.
Run `docs/make-full-website.sh` with `LANCE_NAMESPACE_REPO` pointing at a split `lance-namespace` checkout to populate the full specification.
EOF

cat > "$docs_src/format/catalog/rest/index.md" <<'EOF'
# REST Catalog

The REST Catalog is the service-oriented catalog specification for Lance.

This placeholder page keeps the local website buildable when external catalog docs are not available.
Run `docs/make-full-website.sh` with `LANCE_NAMESPACE_REPO` pointing at a split `lance-namespace` checkout to populate the full specification.
EOF

cat > "$docs_src/format/namespace/.pages" <<'EOF'
title: Namespace Client Spec
nav:
  - Overview: index.md
  - Objects & Relationships: object-relationship.md
  - Operations: operations
  - Supported Catalogs: supported-catalogs
EOF

cat > "$docs_src/format/namespace/index.md" <<'EOF'
# Namespace Client Spec

The Lance Namespace Client Spec defines the interface that engines and tools use to discover tables, resolve locations, and coordinate table operations through catalogs.

When a local `lance-namespace` checkout with split namespace docs is available, `docs/make-full-website.sh` replaces these placeholders with the latest source content.

See also:

- [Objects & Relationships](object-relationship.md)
- [Operations](operations/index.md)
- [Supported Catalogs](supported-catalogs/index.md)
EOF

cat > "$docs_src/format/namespace/object-relationship.md" <<'EOF'
# Objects & Relationships

This placeholder page keeps the local website buildable when external namespace docs are not available.

Run `docs/make-full-website.sh` with `LANCE_NAMESPACE_REPO` pointing at a split `lance-namespace` checkout to populate the full object model description.
EOF

cat > "$docs_src/format/namespace/operations/.pages" <<'EOF'
title: Operations
nav:
  - Overview: index.md
  - Models: models
EOF

cat > "$docs_src/format/namespace/operations/index.md" <<'EOF'
# Operations

This placeholder page keeps the local website buildable when external namespace docs are not available.

Run `docs/make-full-website.sh` with `LANCE_NAMESPACE_REPO` pointing at a split `lance-namespace` checkout to populate the operation reference.
EOF

cat > "$docs_src/format/namespace/operations/models/.pages" <<'EOF'
title: Models
EOF

cat > "$docs_src/format/namespace/operations/models/index.md" <<'EOF'
# Operation Models

This placeholder page keeps the local website buildable when external namespace docs are not available.
EOF

cat > "$docs_src/format/namespace/supported-catalogs/.pages" <<'EOF'
title: Supported Catalogs
nav:
  - Overview: index.md
  - Lance Directory Catalog: lance-dir.md
  - Lance REST Catalog: lance-rest.md
  - Template: template.md
EOF

cat > "$docs_src/format/namespace/supported-catalogs/index.md" <<'EOF'
# Supported Catalogs

This placeholder page keeps the local website buildable when external namespace docs are not available.

Run `docs/make-full-website.sh` with `LANCE_NAMESPACE_REPO` and `LANCE_NAMESPACE_IMPLS_REPO` set to local checkouts to populate the full integration catalog list.
EOF

cat > "$docs_src/format/namespace/supported-catalogs/lance-dir.md" <<'EOF'
# Lance Directory Catalog

This placeholder page keeps the local website buildable when external namespace docs are not available.
EOF

cat > "$docs_src/format/namespace/supported-catalogs/lance-rest.md" <<'EOF'
# Lance REST Catalog

This placeholder page keeps the local website buildable when external namespace docs are not available.
EOF

cat > "$docs_src/format/namespace/supported-catalogs/template.md" <<'EOF'
# Template

This placeholder page keeps the local website buildable when external namespace docs are not available.
EOF
