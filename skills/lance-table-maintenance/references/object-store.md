## Tables in an object store

Use this file when a table is in an object store and the user asks how to give this machine access
to it, or when a command on this machine cannot reach such a table.

## What needs access from this machine

| What runs | Opens the table from |
| --- | --- |
| `scripts/doctor.py` | This machine |
| `scripts/maintain.py preview` without `--backend` | This machine |
| A job of the default backend | This machine, in a background process |
| A preview or a job of another execution backend | Wherever that backend runs it, with the access of that environment. Nothing in this file applies to it |

## Two ways to give access

| Way | When | What the scripts need |
| --- | --- | --- |
| Settings the machine already has | The environment variables of the store are set, such as `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, or the machine has a role that grants access | Nothing. Lance finds them |
| A storage options file | Anything else: credentials that are not in the environment, an S3-compatible store with its own endpoint, a table that needs other credentials than the rest of the machine | The path of the file in `LANCE_MAINTENANCE_STORAGE_OPTIONS_FILE` |

```bash
LANCE_MAINTENANCE_STORAGE_OPTIONS_FILE=~/.lance-maintenance/my-store.json python scripts/doctor.py s3://bucket/path/t.lance
```

Set the variable on every command that opens the table: `doctor.py`, `maintain.py preview` and
`maintain.py submit`. `status` and `cancel` do not open the table and do not need it.

## The storage options file

The file is a JSON object of Lance storage options, the same keys and values as the
`storage_options` argument of `lance.dataset`. Every value is a string, also a boolean or a number.

| Store | Example |
| --- | --- |
| AWS S3 | `{"aws_access_key_id": "...", "aws_secret_access_key": "...", "aws_region": "us-east-1"}` |
| S3-compatible store, such as MinIO | `{"aws_access_key_id": "...", "aws_secret_access_key": "...", "aws_region": "us-east-1", "aws_endpoint": "https://minio.example.com:9000"}`. Add `"aws_virtual_hosted_style_request": "true"` if the endpoint has the bucket in its host name |
| Google Cloud Storage | `{"google_service_account": "/path/to/service-account.json"}` |
| Azure Blob Storage | `{"azure_storage_account_name": "...", "azure_storage_account_key": "..."}` |

All options, also for other stores, are listed at <https://lance.org/guide/object_store/>. Timeouts
and retries are storage options too, for example `timeout` and `client_max_retries`: set them here
for a slow or unreliable connection.

| Rule | Reason |
| --- | --- |
| Write each option under its full name, the one that starts with the prefix of the store, such as `aws_access_key_id` and not `access_key_id` | An option under its full name always takes precedence over an environment variable of the same meaning. Under a short name it may not |
| Only the user can read the file, for example after `chmod 600 <path>` | It holds credentials |
| Credentials last at least as long as the job | A job reads the file once, when its background process starts. Temporary credentials that expire during the job make it fail |

## What to tell a user who has to write one

You never see the content of the file. Show the user the example for their store and ask them to
create the file themselves, for example:

```bash
mkdir -p ~/.lance-maintenance
touch ~/.lance-maintenance/my-store.json
chmod 600 ~/.lance-maintenance/my-store.json
# then fill it in with an editor
```

Then ask only for its path.

## When the table cannot be reached

Credentials that show up in an error message are replaced by `***`. This is best effort, so do not
repeat a message more widely than needed.

| Code | Message contains | Likely cause |
| --- | --- | --- |
| `invalid_input` | `cannot read the storage options` or `must hold a JSON object` | The path is wrong, the file is not JSON, or a value is not a string, such as `true` instead of `"true"` |
| `permission_denied` | `403 Forbidden` or `401 Unauthorized` | The credentials are wrong or expired, or they may not access this table. If the file is right, an environment variable with other credentials may be in the way: write the options under their full names |
| `permission_denied` | `Failed to get AWS credentials` | No credentials were found, neither in a file nor on the machine |
| `table_not_found` | `Bucket ... not found` | The endpoint or the region of an S3-compatible store is missing |
| `table_not_found` | `Dataset at path ... was not found` | The store was reached and has no table at this URI |
| `internal` | `error sending request` | The address in the message cannot be reached from this machine: the endpoint is missing or wrong, or the network does not allow it |
| `internal` | `404 Not Found` for a table that exists | The endpoint has the bucket in its host name and `aws_virtual_hosted_style_request` is not `"true"` |
| `internal` | The store asks to slow down, or says there are too many requests | The store limits how many requests it takes, and other work on the same account counts too. Wait, then submit the job again: it does what is left. The table is as usable as before |

A store that cannot be reached is tried several times, so such a command takes a while to fail.
