import copy
import uuid
import lance
import pyarrow as pa
from lance.namespace import DirectoryNamespace

CONFIG = {
    "allow_http": "true",
    "aws_access_key_id": "ACCESS_KEY",
    "aws_secret_access_key": "SECRET_KEY",
    "aws_endpoint": "http://localhost:4566",
    "aws_region": "us-east-1",
}

storage_options = copy.deepcopy(CONFIG)
storage_options_with_refresh = dict(storage_options)
storage_options_with_refresh["refresh_offset_millis"] = "1000"

dir_props = {f"storage.{k}": v for k, v in storage_options_with_refresh.items()}
dir_props["root"] = "s3://lance-namespace-integtest/namespace_root"
dir_props["ops_metrics_enabled"] = "true"
dir_props["vend_input_storage_options"] = "true"
dir_props["vend_input_storage_options_refresh_interval_millis"] = "3600000"

namespace = DirectoryNamespace(**dir_props)

table1 = pa.Table.from_pylist([{"a": 1, "b": 2}])
table_name = "debug_print3_" + uuid.uuid4().hex
table_id = ["test_ns", table_name]

print("=== Creating table ===")
ds = lance.write_dataset(
    table1,
    namespace_client=namespace,
    table_id=table_id,
    mode="create",
    storage_options=storage_options,
)
print("=== Table created ===")
print(
    "Describe count after write:",
    namespace.retrieve_ops_metrics().get("describe_table", 0),
)

print("")
print("=== Opening dataset via lance.dataset() ===")
ds_from_namespace = lance.dataset(
    namespace_client=namespace,
    table_id=table_id,
    storage_options=storage_options,
)
print("=== Dataset opened ===")
print(
    "Describe count after lance.dataset():",
    namespace.retrieve_ops_metrics().get("describe_table", 0),
)
