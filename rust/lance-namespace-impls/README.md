# lance-namespace-impls

Lance Namespace implementation backends.

## Overview

This crate provides concrete implementations of the Lance namespace trait:

- Unified connection interface for all implementations
- **REST Namespace** - REST API client for remote Lance namespace servers (feature: `rest`)
- **Directory Namespace** - File system-based namespace that stores tables as Lance datasets (always available)

## Features

### REST Namespace (feature: `rest`)

The REST namespace implementation provides a client for connecting to remote Lance namespace servers via REST API.

### Directory Namespace (always available)

The directory namespace implementation stores tables as Lance datasets in a directory structure on local or cloud storage.

Manifest object IDs separate namespace levels with `$`. Within each name, `%`
is encoded as `%25` and `$` as `%24`, so `a$b` as a single name is distinct from
the two-level ID `["a", "b"]`. A mutation that needs escaping atomically upgrades
the catalog IDs and sets reader and writer feature flags. Clients that do not
support these flags must upgrade before accessing the catalog. Existing
unescaped `$` keys retain their historical interpretation as namespace levels;
the original name boundaries cannot be recovered from those keys.

Directory migration preserves each `.lance` directory name as a single root
table name, including literal `$`. Register an explicit multipart ID to
place an existing table in a nested namespace.

Supported storage backends:
- Local filesystem (always available)
- AWS S3 (feature: `dir-aws`)
- Google Cloud Storage (feature: `dir-gcp`)
- Azure Blob Storage (feature: `dir-azure`)
- Alibaba Cloud OSS (feature: `dir-oss`)

## Usage

### Connecting to a Namespace

```rust
use lance_namespace_impls::connect;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to REST implementation
    let mut props = HashMap::new();
    props.insert("uri".to_string(), "http://localhost:8080".to_string());
    let namespace = connect("rest", props).await?;

    Ok(())
}
```

### Using Directory Namespace

```rust
use lance_namespace_impls::connect;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to local directory
    let mut props = HashMap::new();
    props.insert("root".to_string(), "/path/to/data".to_string());
    let namespace = connect("dir", props).await?;

    // Connect to S3
    let mut props = HashMap::new();
    props.insert("root".to_string(), "s3://my-bucket/path".to_string());
    props.insert("storage.region".to_string(), "us-west-2".to_string());
    let namespace = connect("dir", props).await?;

    Ok(())
}
```

## Configuration

### Directory Namespace Properties

- `root` - Root path for the namespace (local path or cloud storage URI)
- `storage.*` - Storage-specific options (e.g., `storage.region`, `storage.access_key_id`)

## Documentation

For more information about Lance and its namespace system, see the [Lance Namespace documentation](https://lance.org/format/namespace).
