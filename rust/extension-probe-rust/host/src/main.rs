// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::path::PathBuf;

use lance_extension_runtime::ExtensionRuntime;
use lance_extension_runtime::lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_extension_runtime::object_store::path::Path;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let plugin_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("plugin path");
    let runtime = ExtensionRuntime::new();
    runtime.load_plugin(plugin_path).expect("load plugin");

    let params = ObjectStoreParams::default();
    let (store, _) =
        ObjectStore::from_uri_and_params(runtime.registry(), "demo-memory:///", &params)
            .await
            .expect("open demo-memory store");

    let key = Path::from("probe-key");
    let value = b"hello from a Rust storage provider";
    store.put(&key, value).await.expect("put");
    let result = store.read_one_all(&key).await.expect("get");
    assert_eq!(result.as_ref(), value);
    println!("{}", String::from_utf8_lossy(&result));

    let (_goosefs_store, _) =
        ObjectStore::from_uri_and_params(runtime.registry(), "goosefs://127.0.0.1:9200/", &params)
            .await
            .expect("construct GooseFS store");
    println!("GooseFS operator constructed through ObjectStoreProvider");
}
