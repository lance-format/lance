# Object Store Configuration

Lance supports object stores such as AWS S3 (and compatible stores), Azure Blob Store,
and Google Cloud Storage. Which object store to use is determined by the URI scheme of
the dataset path. For example, `s3://bucket/path` will use S3, `az://bucket/path`
will use Azure, and `gs://bucket/path` will use GCS.

These object stores take additional configuration objects. There are two ways to
specify these configurations: by setting environment variables or by passing them
to the `storage_options` parameter of `lance.dataset` and
`lance.write_dataset`. So for example, to globally set a higher timeout,
you would run in your shell:

```bash
export TIMEOUT=60s
```

If you only want to set the timeout for a single dataset, you can pass it as a
storage option:

```python
import lance
ds = lance.dataset("s3://path", storage_options={"timeout": "60s"})
```

## General Configuration

These options apply to all object stores.

| Key                          | Description                                                                                                                                                                                                                                                                                             |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `allow_http`                 | Allow non-TLS, i.e. non-HTTPS connections. Default, `False`.                                                                                                                                                                                                                                            |
| `download_retry_count`       | Number of times to retry a download. Default, `3`. This limit is applied when the HTTP request succeeds but the response is not fully downloaded, typically due to a violation of `timeout`.                                                                                                            |
| `allow_invalid_certificates` | Skip certificate validation on https connections. Default, `False`. Warning: This is insecure and should only be used for testing.                                                                                                                                                                      |
| `connect_timeout`            | Timeout for only the connect phase of a Client. Default, `5s`.                                                                                                                                                                                                                                          |
| `timeout`                    | Timeout for the entire request, from connection until the response body has finished. Default, `30s`. This applies to each individual request, so on a large write it must cover one complete multipart part upload; raise it alongside `LANCE_INITIAL_UPLOAD_SIZE`.                                    |
| `user_agent`                 | User agent string to use in requests.                                                                                                                                                                                                                                                                   |
| `proxy_url`                  | URL of a proxy server to use for requests. Default, `None`.                                                                                                                                                                                                                                             |
| `proxy_ca_certificate`       | PEM-formatted CA certificate for proxy connections                                                                                                                                                                                                                                                      |
| `proxy_excludes`             | List of hosts that bypass proxy. This is a comma separated list of domains and IP masks. Any subdomain of the provided domain will be bypassed. For example, `example.com, 192.168.1.0/24` would bypass `https://api.example.com`, `https://www.example.com`, and any IP in the range `192.168.1.0/24`. |
| `client_max_retries`         | Number of times for the object store client to retry the request. Default, `3`.                                                                                                                                                                                                                         |
| `client_retry_timeout`       | Timeout for the object store client to retry the request in seconds. Default, `180`.                                                                                                                                                                                                                    |
| `scheduler_error_mode`       | Scheduler error handling. `best_effort` preserves the default behavior. `fail_fast` returns the first observed error and cancels the remaining physical reads in the same logical request. Other requests are not cancelled.                                                                            |

An engine can apply `scheduler_error_mode` to every operation in a session by
adding it to the storage options used for every dataset open. For example, a
Python-based engine can merge the session setting before calling Lance:

```python
session_storage_options = {"scheduler_error_mode": "fail_fast"}
dataset = lance.dataset(uri, storage_options=session_storage_options)
```

The engine-facing configuration name is engine-specific. It must ultimately be
forwarded to Lance as the `scheduler_error_mode` storage option.

For example, a Lance Spark catalog named `lance` forwards settings below its
`storage` prefix:

```bash
spark-sql --conf "spark.sql.catalog.lance.storage.scheduler_error_mode=fail_fast"
```

### Bulk copy strategy

Lance streams bulk index-file movement and dataset deep-clone files through
read and write APIs by default. This avoids requiring a provider-native copy
operation and works across different object stores.

Set `LANCE_IO_SERVER_SIDE_COPY_ENABLED` to a truthy value (`1`, `true`, `on`,
`yes`, or `y`, case-insensitive) to opt cloud copies whose source and destination
share the same object-store client into the provider-native server-side copy
operation. Cross-client, cross-store, and local copies do not use this setting.
Native copy can reduce client bandwidth and transfer cost, but it requires copy
support from the object-store integration and is subject to the provider
request's timeout and retry behavior.

Deep clone bounds non-local file movement to four concurrent files by default.
Set `LANCE_DEEP_CLONE_STREAM_CONCURRENCY` to a positive integer to override this
operation-specific limit. The bound also applies when server-side copy is
enabled because S3 and GCS copies above the provider's single-copy size limit
fall back to streaming through Lance.

## Per-Base Configuration

A dataset can register additional base paths that store part of its data, and each
base may live in a different bucket, account, or storage provider. A storage option
key of the form `base_<id>.<key>` applies `<key>` only to the base path with that
manifest id. Every base inherits the unscoped options; base-scoped entries add to or
override them for that base only.

```python
import lance
ds = lance.dataset(
    "az://account-a/path",
    storage_options={
        # Shared defaults, used by the primary dataset and inherited by bases
        "account_name": "account-a",
        "account_key": "key-a",
        # Overrides for the base path with id 1
        "base_1.account_name": "account-b",
        "base_1.account_key": "key-b",
    },
)
```

Base ids are assigned when bases are registered (`initial_bases` ids are assigned
sequentially starting at 1, in order) and can be inspected through the manifest base
paths. Keys that do not match `base_<id>.<key>` exactly (e.g. `base_url`) are treated
as regular storage options. Exact per-base parameter maps (`base_store_params`,
keyed by base path URI) take precedence over base-scoped keys for that base.

## S3 Configuration

S3 (and S3-compatible stores) have additional configuration options that configure
authorization and S3-specific features (such as server-side encryption).

AWS credentials can be set in the environment variables `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`. Alternatively, they can be
passed as parameters to the `storage_options` parameter:

```python
import lance
ds = lance.dataset(
    "s3://bucket/path",
    storage_options={
        "access_key_id": "my-access-key",
        "secret_access_key": "my-secret-key",
        "session_token": "my-session-token",
    }
)
```

If you are using AWS SSO, you can specify the `AWS_PROFILE` environment variable.
It cannot be specified in the `storage_options` parameter.

The following keys can be used as both environment variables or keys in the
`storage_options` parameter:

| Key                                                                 | Description                                                                                                                                      |
|---------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `aws_region` / `region`                                             | The AWS region the bucket is in. This can be automatically detected when using AWS S3, but must be specified for S3-compatible stores.           |
| `aws_access_key_id` / `access_key_id`                               | The AWS access key ID to use.                                                                                                                    |
| `aws_secret_access_key` / `secret_access_key`                       | The AWS secret access key to use.                                                                                                                |
| `aws_session_token` / `session_token`                               | The AWS session token to use.                                                                                                                    |
| `aws_endpoint` / `endpoint`                                         | The endpoint to use for S3-compatible stores.                                                                                                    |
| `aws_virtual_hosted_style_request` / `virtual_hosted_style_request` | Whether to use virtual hosted-style requests, where bucket name is part of the endpoint. Meant to be used with `aws_endpoint`. Default, `False`. |
| `aws_s3_express` / `s3_express`                                     | Whether to use S3 Express One Zone endpoints. Default, `False`. See more details below.                                                          |
| `aws_server_side_encryption`                                        | The server-side encryption algorithm to use. Must be one of `"AES256"`, `"aws:kms"`, or `"aws:kms:dsse"`. Default, `None`.                       |
| `aws_sse_kms_key_id`                                                | The KMS key ID to use for server-side encryption. If set, `aws_server_side_encryption` must be `"aws:kms"` or `"aws:kms:dsse"`.                  |
| `aws_sse_bucket_key_enabled`                                        | Whether to use bucket keys for server-side encryption.                                                                                           |

### Credential provider selection

By default, Lance uses the standard AWS credential provider chain (environment
variables, shared config file, web identity tokens, ECS, EC2 instance metadata).

The `aws_provider_scheme` storage option pins a dataset to a specific credential
provider, which is useful when two datasets in the same process need different
AWS auth (for example, one bucket using IRSA and another using ECS container
credentials).

| Value | Behavior |
|-------|----------|
| `token` | Use static access-key credentials. Returns an error if `aws_access_key_id` and `aws_secret_access_key` are not set. |
| `ecs` | Use the ECS/Pod Identity container credential endpoint. Reads `AWS_CONTAINER_CREDENTIALS_FULL_URI` or `AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` from the environment. |
| `irsa` | Use IRSA (IAM Roles for Service Accounts) web identity token credentials. Reads `AWS_WEB_IDENTITY_TOKEN_FILE` and `AWS_ROLE_ARN` from the environment. |

```python
import lance

# Bucket A — use IRSA (web identity token from the environment)
ds_a = lance.dataset(
    "s3://bucket-a/path",
    storage_options={"aws_provider_scheme": "irsa"},
)

# Bucket B — use ECS container credentials
ds_b = lance.dataset(
    "s3://bucket-b/path",
    storage_options={"aws_provider_scheme": "ecs"},
)
```

### S3-compatible stores

Lance can also connect to S3-compatible stores, such as MinIO. To do so, you must
specify both region and endpoint:

```python
import lance
ds = lance.dataset(
    "s3://bucket/path",
    storage_options={
        "region": "us-east-1",
        "endpoint": "http://minio:9000",
    }
)
```

This can also be done with the `AWS_ENDPOINT` and `AWS_DEFAULT_REGION` environment variables.

### S3 Express (Directory Bucket)

Lance supports [S3 Express One Zone](https://aws.amazon.com/s3/storage-classes/express-one-zone/) buckets,
a.k.a. S3 directory buckets.
S3 Express buckets only support connecting from an EC2 instance within the same
region.
By default, Lance automatically recognize the `--x-s3` suffix of an express bucket,
there is no special configuration needed.

In case of an access point or private link that hides the bucket name,
you can configure express bucket access explicitly through storage option `s3_express`.

```python
import lance
ds = lance.dataset(
    "s3://my-bucket--use1-az4--x-s3/path/imagenet.lance",
    storage_options={
        "region": "us-east-1",
        "s3_express": "true",
    }
)
```

## Google Cloud Storage Configuration

GCS credentials are configured by setting the `GOOGLE_SERVICE_ACCOUNT` environment
variable to the path of a JSON file containing the service account credentials.
Alternatively, you can pass the path to the JSON file in the `storage_options`

```python
import lance
ds = lance.dataset(
    "gs://my-bucket/my-dataset",
    storage_options={
        "service_account": "path/to/service-account.json",
    }
)
```

!!! note

    By default, GCS uses HTTP/1 for communication, as opposed to HTTP/2. This improves
    maximum throughput significantly. However, if you wish to use HTTP/2 for some reason,
    you can set the environment variable `HTTP1_ONLY` to `false`.

The following keys can be used as both environment variables or keys in the
`storage_options` parameter:

| Key | Description |
|-----|-------------|
| `google_service_account` / `service_account` | Path to the service account JSON file. |
| `google_service_account_key` / `service_account_key` | The serialized service account key. |
| `google_application_credentials` / `application_credentials` | Path to the application credentials. |

## Azure Blob Storage Configuration

Azure Blob Storage credentials can be configured by setting the `AZURE_STORAGE_ACCOUNT_NAME`
and `AZURE_STORAGE_ACCOUNT_KEY` environment variables. Alternatively, you can pass
the account name and key in the `storage_options` parameter:

```python
import lance
ds = lance.dataset(
    "az://my-container/my-dataset",
    storage_options={
        "account_name": "some-account",
        "account_key": "some-key",
    }
)
```

These keys can be used as both environment variables or keys in the `storage_options` parameter:

| Key | Description |
|-----|-------------|
| `azure_storage_account_name` / `account_name` | The name of the azure storage account. |
| `azure_storage_account_key` / `account_key` | The serialized service account key. |
| `azure_client_id` / `client_id` | Service principal client id for authorizing requests. |
| `azure_client_secret` / `client_secret` | Service principal client secret for authorizing requests. |
| `azure_tenant_id` / `tenant_id` | Tenant id used in oauth flows. |
| `azure_storage_sas_key` / `azure_storage_sas_token` / `sas_key` / `sas_token` | Shared access signature. The signature is expected to be percent-encoded, much like they are provided in the azure storage explorer or azure portal. |
| `azure_storage_token` / `bearer_token` / `token` | Bearer token. |
| `azure_storage_use_emulator` / `object_store_use_emulator` / `use_emulator` | Use object store with azurite storage emulator. |
| `azure_endpoint` / `endpoint` | Override the endpoint used to communicate with blob storage. |
| `azure_use_fabric_endpoint` / `use_fabric_endpoint` | Use object store with url scheme account.dfs.fabric.microsoft.com. |
| `azure_msi_endpoint` / `azure_identity_endpoint` / `identity_endpoint` / `msi_endpoint` | Endpoint to request a imds managed identity token. |
| `azure_object_id` / `object_id` | Object id for use with managed identity authentication. |
| `azure_msi_resource_id` / `msi_resource_id` | Msi resource id for use with managed identity authentication. |
| `azure_federated_token_file` / `federated_token_file` | File containing token for Azure AD workload identity federation. |
| `azure_use_azure_cli` / `use_azure_cli` | Use azure cli for acquiring access token. |
| `azure_disable_tagging` / `disable_tagging` | Disables tagging objects. This can be desirable if not supported by the backing store. | 

## AliCloud Object Storage Service Configuration

OSS credentials can be set in the environment variables `OSS_ACCESS_KEY_ID`,
`OSS_ACCESS_KEY_SECRET`, `OSS_REGION`, and `OSS_SECURITY_TOKEN`. Alternatively, they can be
passed as parameters to the `storage_options` parameter:

```python
import lance
ds = lance.dataset(
    "oss://bucket/path",
    storage_options={
        "oss_region": "oss-region",
        "oss_endpoint": "oss-endpoint",
        "oss_access_key_id": "my-access-key",
        "oss_secret_access_key": "my-secret-key",
        "oss_security_token": "my-session-token",
    }
)
```

| Key | Description |
|-----|-------------|
| `oss_endpoint` | OSS endpoint. Required (for example, `https://oss-cn-hangzhou.aliyuncs.com`). |
| `oss_access_key_id` | Access key ID used for OSS authentication. Optional if credentials are provided by environment. |
| `oss_secret_access_key` | Access key secret used for OSS authentication. Optional if credentials are provided by environment. |
| `oss_region` | OSS region (for example, `cn-hangzhou`). Optional. |
| `oss_security_token` | Security token for temporary credentials (STS). Optional. |

## Volcengine TOS Configuration

TOS credentials can be set in the environment variables `TOS_ACCESS_KEY_ID`,
`TOS_SECRET_ACCESS_KEY`, `TOS_ENDPOINT`, `TOS_REGION`, and `TOS_SECURITY_TOKEN`.
Lance also accepts the corresponding `VOLCENGINE_` environment variable prefix.
Alternatively, credentials can be passed as parameters to the `storage_options`
parameter; explicit `storage_options` override environment variables:

```python
import lance
ds = lance.dataset(
    "tos://bucket/path",
    storage_options={
        "tos_endpoint": "https://tos-cn-beijing.volces.com",
        "tos_region": "cn-beijing",
        "tos_access_key_id": "my-access-key",
        "tos_secret_access_key": "my-secret-key",
        "tos_security_token": "my-session-token",
    }
)
```

| Key | Description |
|-----|-------------|
| `tos_endpoint` | TOS endpoint. Required (for example, `https://tos-cn-beijing.volces.com`). |
| `tos_region` | TOS signing region (for example, `cn-beijing`). Optional. |
| `tos_access_key_id` | Access key ID used for TOS authentication. Optional if credentials are provided by environment. |
| `tos_secret_access_key` | Secret access key used for TOS authentication. Optional if credentials are provided by environment. |
| `tos_security_token` | Security token for temporary credentials. Optional. |

## Tencent Cloud COS Configuration

[COS (Cloud Object Storage)](https://cloud.tencent.com/product/cos) credentials can be set in environment variables prefixed
with `COS_` or `TENCENTCLOUD_` (for example, `COS_ENDPOINT`, `COS_SECRET_ID`,
`COS_SECRET_KEY`, `TENCENTCLOUD_REGION`, `TENCENTCLOUD_SECURITY_TOKEN`).
Alternatively, credentials can be passed as parameters to the `storage_options`
parameter; explicit `storage_options` override environment variables:

=== "Python"

    ```python
    import lance
    ds = lance.dataset(
        "cos://bucket/path",
        storage_options={
            "cos_endpoint": "https://cos.ap-guangzhou.myqcloud.com",
            "cos_secret_id": "my-secret-id",
            "cos_secret_key": "my-secret-key",
        }
    )
    ```

=== "Rust"

    In this Lance distribution, `tencent` is already part of the **default
    features** of the `lance` crate, so simply depending on `lance` is enough:

    ```toml
    [dependencies]
    lance = "*"
    ```

    You only need to enable the `tencent` feature explicitly in the following
    cases:

    - You opted out of default features, e.g.
      `lance = { version = "*", default-features = false, features = ["tencent", ...] }`.
    - You depend on `lance-io` directly (without `lance`); `tencent` is **not**
      a default feature of `lance-io`:
      `lance-io = { version = "*", features = ["tencent"] }`.

| Key | Description |
|-----|-------------|
| `cos_endpoint` | COS endpoint. Required (for example, `https://cos.ap-guangzhou.myqcloud.com`). Can also be set via the `COS_ENDPOINT` environment variable. |
| `cos_secret_id` | Secret ID used for COS authentication. Optional if credentials are provided by environment. |
| `cos_secret_key` | Secret key used for COS authentication. Optional if credentials are provided by environment. |
| `cos_enable_versioning` | Whether to enable object versioning on the bucket. Optional. |

!!! warning

    Tencent COS does not reliably enforce put-if-not-exists on buckets that have
    ever had versioning enabled, even if versioning is now suspended. To prevent
    silent manifest overwrites, Lance requires a custom distributed commit lock
    for COS writes. Pass the same `commit_lock` implementation to every Python
    writer, or provide a custom `CommitHandler` in Rust. Reads do not require a
    commit lock.

!!! note

    The OpenDAL `CosConfig` currently exposes a limited set of options. Additional
    settings such as the security token (`TENCENTCLOUD_SECURITY_TOKEN`) and region
    (`TENCENTCLOUD_REGION`) must be configured via environment variables.

## GooseFS Configuration

[GooseFS](https://cloud.tencent.com/product/goosefs) is a distributed caching
filesystem. Lance accesses GooseFS through its Master gRPC service. The URL format
is `goosefs://host:port/path`, where `host:port` is the GooseFS Master address
(default port: `9200`, may be omitted, e.g. `goosefs://10.0.0.1/path`) and
`/path` is the filesystem path within GooseFS.

Manifest commits on `goosefs://` use `ConditionalPutCommitHandler`
(`PutMode::Create` / if-not-exists), backed by GooseFS master's atomic
no-replace rename so concurrent writers cannot clobber each other's
versioned manifests.

!!! warning "Mixed-version writers are NOT safe"

    The `if-not-exists` guarantee only holds when **every** writer for a
    dataset routes through this new handler. A writer running an older
    Lance release still selects `UnsafeCommitHandler` for `goosefs://`
    and writes the version path unconditionally, which can overwrite a
    manifest that an upgraded writer has already won. Safe concurrent
    commits therefore require:

    - all writers for the dataset run a Lance release that includes this
      routing change, **or**
    - writers share an external coordination boundary (e.g. a single-writer
      queue, table-level lock, or a gateway that serializes commits) that
      prevents the old code path from racing the new one.

    When upgrading in place, quiesce all writers (drain jobs, scale
    clients to zero, or route traffic through a writer coordinator) before
    rolling out the new Lance version, then bring writers back on the new
    version together.

!!! note "About the dataset path"

    `/path` is just an arbitrary directory inside GooseFS — Lance does **not**
    require the path to end with a `.lance` suffix. Any valid GooseFS directory
    works, for example:

    - `goosefs://10.0.0.1:9200/data/my-dataset`
    - `goosefs://10.0.0.1:9200/data/my-dataset.lance`
    - `goosefs://10.0.0.1:9200/lance-test/lance-io`

    The `.lance` suffix used in the examples below is only a naming convention
    that makes it easy to recognize a Lance dataset directory at a glance; it
    has no special meaning to Lance itself. The only requirement is that the
    same path is used consistently for reads and writes of a given dataset.

=== "Python"

    ```python
    import lance

    ds = lance.dataset(
        "goosefs://10.0.0.1:9200/data/my-dataset.lance",
        storage_options={
            "goosefs_auth_type": "simple",
            "goosefs_auth_username": "lance",
        },
    )
    ```

=== "Rust"

    In this Lance distribution, `goosefs` is already part of the **default
    features** of the `lance` crate, so simply depending on `lance` is enough:

    ```toml
    [dependencies]
    lance = "*"
    ```

    You only need to enable the `goosefs` feature explicitly in the following
    cases:

    - You opted out of default features, e.g.
      `lance = { version = "*", default-features = false, features = ["goosefs", ...] }`.
    - You depend on `lance-io` directly (without `lance`); `goosefs` is **not**
      a default feature of `lance-io`:
      `lance-io = { version = "*", features = ["goosefs"] }`.

    Open the underlying `lance_io::object_store::ObjectStore` directly (mirrors
    the integration test in `rust/lance-io/tests/goosefs_integration.rs`):

    ```rust
    use lance_io::object_store::ObjectStore;

    let uri = "goosefs://10.0.0.1:9200/lance-test/lance-io";
    let (store, path) = ObjectStore::from_uri(uri).await?;

    // Read / write through the underlying `object_store::ObjectStore` API
    store.inner.put(&path, (&b"hello"[..]).into()).await?;
    let result = store.inner.get(&path).await?;
    let bytes = result.bytes().await?;
    ```

    Open a Lance dataset with custom storage options:

    ```rust
    use std::collections::HashMap;
    use lance::dataset::DatasetBuilder;

    let mut storage_options = HashMap::new();
    storage_options.insert("goosefs_master_addr".to_string(), "10.0.0.1:9200".to_string());
    storage_options.insert("goosefs_auth_type".to_string(), "simple".to_string());
    storage_options.insert("goosefs_auth_username".to_string(), "lance".to_string());

    let dataset = DatasetBuilder::from_uri("goosefs://10.0.0.1:9200/data/my-dataset.lance")
        .with_storage_options(storage_options)
        .load()
        .await?;
    ```

=== "Java"

    Pass the GooseFS configuration through `ReadOptions.setStorageOptions`
    when opening the dataset:

    ```java
    import org.lance.Dataset;
    import org.lance.ReadOptions;

    import java.util.HashMap;
    import java.util.Map;

    Map<String, String> storageOptions = new HashMap<>();
    storageOptions.put("goosefs_master_addr", "10.0.0.1:9200");
    storageOptions.put("goosefs_auth_type", "simple");
    storageOptions.put("goosefs_auth_username", "lance");

    ReadOptions options = new ReadOptions.Builder()
        .setStorageOptions(storageOptions)
        .build();

    try (Dataset dataset = Dataset.open()
            .uri("goosefs://10.0.0.1:9200/data/my-dataset.lance")
            .readOptions(options)
            .build()) {
        // ... use the dataset
    }
    ```

    For writes, the same `storageOptions(...)` setter is available on
    `WriteDatasetBuilder` and `WriteFragmentBuilder`.

The Master address can be resolved from (in priority order):

1. The `goosefs_master_addr` storage option (supports HA: `"addr1:port,addr2:port"`).
2. The `GOOSEFS_MASTER_ADDR` environment variable.
3. The host and port from the URL authority.

`storage_options` keys **must be lowercase**. Uppercase or mixed-case spellings
such as `GOOSEFS_MASTER_ADDR` are rejected with an explicit error — they are
not ignored, and they are not treated as the matching environment variable.
Environment variables keep the `GOOSEFS_*` form.

| storage_options key | env var | Description |
|---------------------|---------|-------------|
| `goosefs_master_addr` | `GOOSEFS_MASTER_ADDR` | GooseFS Master address. Supports a single address (`host:port`) or comma-separated HA addresses (`addr1:port,addr2:port`). Optional if the address is provided in the URL. |
| `goosefs_write_type` | `GOOSEFS_WRITE_TYPE` | Write type, e.g. `MUST_CACHE`, `CACHE_THROUGH`, `THROUGH`, `ASYNC_THROUGH`. Optional. |
| `goosefs_block_size` | `GOOSEFS_BLOCK_SIZE` | GooseFS block size (this is the GooseFS-side block size, not Lance's I/O block size). Accepts a raw byte count or GooseFS suffixes such as `64MB` (binary units: `1KB = 1024`). Optional. |
| `goosefs_chunk_size` | `GOOSEFS_CHUNK_SIZE` | Chunk size used when reading or writing files. Accepts a raw byte count or GooseFS suffixes such as `4MB` (binary units: `1KB = 1024`). Optional. |
| `goosefs_auth_type` | `GOOSEFS_AUTH_TYPE` | Authentication type. Either `nosasl` or `simple` (case-insensitive; the value is passed through to OpenDAL). Optional. |
| `goosefs_auth_username` | `GOOSEFS_AUTH_USERNAME` | Username used in `simple` authentication mode. Optional. |

!!! note "Running the GooseFS integration tests"

    The Rust integration tests for GooseFS live at
    `rust/lance-io/tests/goosefs_integration.rs` and are gated behind feature
    flags. They require a reachable GooseFS cluster (configured via the
    `GOOSEFS_MASTER_ADDR` and `GOOSEFS_AUTH_TYPE` environment variables) and
    can be run with:

    ```bash
    cargo test -p lance-io --features "goosefs goosefs-test" \
        --test goosefs_integration -- --ignored --nocapture --test-threads=1
    ```
