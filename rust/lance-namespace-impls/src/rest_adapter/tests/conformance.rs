// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Checks RestNamespace and RestAdapter against the lance-namespace REST spec
//! of the lance-namespace-reqwest-client that cargo resolved.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::header::{CONTENT_LENGTH, CONTENT_TYPE, HOST};
use axum::http::{HeaderMap, Request, StatusCode};
use bytes::Bytes;
use futures::future::BoxFuture;
use lance_namespace::LanceNamespace;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tower::ServiceExt;
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::integration::RecordingNamespace;
use crate::{RestAdapter, RestAdapterConfig, RestNamespace, RestNamespaceBuilder};

/// The crate whose models RestNamespace and RestAdapter speak. Its release
/// pins the spec we check against.
const CLIENT: &str = "lance-namespace-reqwest-client";

/// The spec's path in the lance-namespace repository.
const SPEC_FILE: &str = "docs/src/spec.yaml";

/// A header that only our fallback sets, which marks a path the adapter has
/// no route for.
const UNROUTED: &str = "x-unrouted";

/// The spec of the client that cargo resolved, and where it came from, or why
/// it is unavailable.
async fn load_spec() -> std::result::Result<(Value, String), String> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .args(["metadata", "--format-version", "1"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .map_err(|e| format!("cannot run cargo metadata: {}", e))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let metadata: Value = serde_json::from_slice(&output.stdout).unwrap();
    let clients: Vec<&Value> = metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|package| package["name"] == CLIENT)
        .collect();
    let [client] = clients[..] else {
        panic!("cargo resolved {} versions of {}", clients.len(), CLIENT);
    };
    let version = client["version"].as_str().unwrap();

    // A [patch.crates-io] can point the client at a git or path checkout of
    // lance-namespace, which carries the spec it was generated from.
    if client["source"]
        .as_str()
        .is_none_or(|source| source.starts_with("git+"))
    {
        let manifest = Path::new(client["manifest_path"].as_str().unwrap());
        let path = manifest
            .ancestors()
            .map(|dir| dir.join(SPEC_FILE))
            .find(|path| path.is_file())
            .unwrap_or_else(|| panic!("no {} above {}", SPEC_FILE, manifest.display()));
        let from = format!("{} {} from {}", CLIENT, version, path.display());
        return Ok((parse(&std::fs::read_to_string(&path).unwrap(), &from), from));
    }

    let cache = Path::new(metadata["target_directory"].as_str().unwrap())
        .join("lance-namespace-spec")
        .join(version)
        .join("spec.yaml");
    if let Ok(yaml) = std::fs::read_to_string(&cache) {
        let from = format!("{} {} cached at {}", CLIENT, version, cache.display());
        return Ok((parse(&yaml, &from), from));
    }
    let url = format!(
        "https://raw.githubusercontent.com/lance-format/lance-namespace/v{}/{}",
        version, SPEC_FILE
    );
    let yaml = download(&url).await?;
    let spec = parse(&yaml, &url);
    // We write a temporary file and rename it, so that a concurrent run never
    // reads half a spec.
    std::fs::create_dir_all(cache.parent().unwrap()).unwrap();
    let partial = cache.with_extension(format!("yaml.{}", std::process::id()));
    std::fs::write(&partial, &yaml).unwrap();
    std::fs::rename(&partial, &cache).unwrap();
    let from = format!("{} {} downloaded from {}", CLIENT, version, url);
    Ok((spec, from))
}

fn parse(yaml: &str, from: &str) -> Value {
    serde_saphyr::from_str(yaml).unwrap_or_else(|e| panic!("cannot parse {}: {}", from, e))
}

/// Downloads `url`, retrying errors that may be transient: connection errors,
/// timeouts, 429 and 5xx.
async fn download(url: &str) -> std::result::Result<String, String> {
    const ATTEMPTS: u64 = 3;
    let client = reqwest::Client::builder()
        .user_agent("lance-rest-spec-conformance")
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let mut attempts = 0;
    let error = loop {
        attempts += 1;
        let result = async {
            client
                .get(url)
                .send()
                .await?
                .error_for_status()?
                .text()
                .await
        };
        let error = match result.await {
            Ok(yaml) => return Ok(yaml),
            Err(error) => error,
        };
        let permanent = error
            .status()
            .is_some_and(|status| status.is_client_error() && status.as_u16() != 429);
        if permanent || attempts == ATTEMPTS {
            break error;
        }
        tokio::time::sleep(Duration::from_secs(attempts)).await;
    };
    let mut message = error.to_string();
    let mut source = std::error::Error::source(&error);
    while let Some(cause) = source {
        message = format!("{}: {}", message, cause);
        source = cause.source();
    }
    Err(format!(
        "cannot download {} ({} attempts): {}",
        url, attempts, message
    ))
}

/// One operation's REST binding, read from the spec.
struct SpecOp {
    method: String,
    path: String,
    query: Vec<String>,
    /// The request content type and schema, and whether the body is required.
    body: Option<(String, Value, bool)>,
    status: u16,
    /// The response content type and schema, when the response has a body.
    response: Option<(String, Value)>,
}

fn resolve<'a>(spec: &'a Value, mut value: &'a Value) -> &'a Value {
    while let Some(target) = value["$ref"].as_str() {
        value = spec.pointer(target.trim_start_matches('#')).unwrap();
    }
    value
}

/// The first content type in `content`, and its schema.
fn media(spec: &Value, content: &Value) -> Option<(String, Value)> {
    let (content_type, media) = content.as_object()?.iter().next()?;
    Some((
        content_type.clone(),
        resolve(spec, &media["schema"]).clone(),
    ))
}

fn spec_ops(spec: &Value) -> BTreeMap<String, SpecOp> {
    let mut ops = BTreeMap::new();
    for (path, item) in spec["paths"].as_object().unwrap() {
        for (method, op) in item.as_object().unwrap() {
            if !matches!(method.as_str(), "get" | "post" | "put" | "patch" | "delete") {
                continue;
            }
            let params = item["parameters"].as_array().into_iter().flatten();
            let params = params.chain(op["parameters"].as_array().into_iter().flatten());
            let query = params
                .map(|param| resolve(spec, param))
                .filter(|param| param["in"] == "query")
                .map(|param| param["name"].as_str().unwrap().to_string())
                .collect();
            let request_body = resolve(spec, &op["requestBody"]);
            let body = media(spec, &request_body["content"]).map(|(content_type, schema)| {
                let required = request_body["required"].as_bool().unwrap_or(false);
                (content_type, schema, required)
            });
            let (status, response) = op["responses"]
                .as_object()
                .unwrap()
                .iter()
                .find(|(code, _)| code.starts_with('2'))
                .map(|(code, response)| {
                    let response = resolve(spec, response);
                    (code.parse().unwrap(), media(spec, &response["content"]))
                })
                .unwrap();
            let op_id = op["operationId"].as_str().unwrap().to_string();
            let op = SpecOp {
                method: method.to_uppercase(),
                path: path.clone(),
                query,
                body,
                status,
                response,
            };
            ops.insert(op_id, op);
        }
    }
    ops
}

/// The spec path with its parameters filled in as the calls below fill them.
fn fill(path: &str) -> String {
    let segments: Vec<&str> = path
        .split('/')
        .map(|segment| match segment {
            "{id}" => "ns%24t",
            "{index_name}" => "idx",
            param if param.starts_with('{') => "param",
            segment => segment,
        })
        .collect();
    segments.join("/")
}

/// Whether `router` sends the operation to a handler, rather than to the
/// fallback that answers paths it has no route for.
async fn serves(router: &Router, op: &SpecOp) -> bool {
    let request = Request::builder()
        .method(op.method.as_str())
        .uri(fill(&op.path))
        .body(Body::empty())
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    !response.headers().contains_key(UNROUTED)
}

/// Whether `value` has the top-level JSON type that `schema` defines.
fn has_shape(schema: &Value, value: &Value) -> bool {
    match schema["type"].as_str() {
        Some("integer") => value.is_i64() || value.is_u64(),
        Some("string") => value.is_string(),
        Some("object") if schema["additionalProperties"]["type"] == "string" => value
            .as_object()
            .is_some_and(|object| object.values().all(Value::is_string)),
        Some("object") => value.is_object(),
        _ => true,
    }
}

fn snake_case(op: &str) -> String {
    let mut out = String::new();
    for c in op.chars() {
        if c.is_ascii_uppercase() && !out.is_empty() {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

/// Builds an operation request from `fields`, on top of an id, an identity
/// and a context that every request schema has.
fn build<T: Serialize + DeserializeOwned>(fields: Value) -> T {
    let mut request = json!({
        "id": ["ns", "t"],
        "identity": {"api_key": "key", "auth_token": "token"},
        "context": {"header.x-trace-id": "trace"},
    });
    for (key, value) in fields.as_object().unwrap() {
        request[key] = value.clone();
    }
    let built: T = serde_json::from_value(request.clone()).unwrap();
    // A misspelled field would deserialize to nothing and test nothing.
    let rebuilt = serde_json::to_value(&built).unwrap();
    for (key, value) in request.as_object().unwrap() {
        assert!(
            value.is_null() || rebuilt.get(key).is_some(),
            "unknown field {}",
            key
        );
    }
    built
}

fn arrow() -> Bytes {
    Bytes::from_static(b"arrow ipc stream")
}

/// What one client call sent, and whether it parsed the reply.
struct Sent {
    request: Value,
    data: Option<Bytes>,
    result: std::result::Result<(), String>,
}

type Call = Box<dyn Fn(RestNamespace) -> BoxFuture<'static, Sent> + Send + Sync>;

macro_rules! call {
    ($method:ident, $fields:expr $(, $data:expr)?) => {
        Box::new(move |ns: RestNamespace| -> BoxFuture<'static, Sent> {
            Box::pin(async move {
                let request = build($fields);
                let sent = serde_json::to_value(&request).unwrap();
                let data: Option<Bytes> = None $(.or(Some($data)))?;
                let result = ns.$method(request $(, $data)?).await;
                Sent {
                    request: sent,
                    data,
                    result: result.map(drop).map_err(|e| e.to_string()),
                }
            })
        }) as Call
    };
}

/// One client call per spec operation that LanceNamespace has, setting every
/// field the spec moves into the path or query.
fn calls() -> BTreeMap<&'static str, Call> {
    BTreeMap::from([
        (
            "ListNamespaces",
            call!(list_namespaces, json!({"page_token": "p", "limit": 5})),
        ),
        ("DescribeNamespace", call!(describe_namespace, json!({}))),
        (
            "CreateNamespace",
            call!(
                create_namespace,
                json!({"mode": "create", "properties": {"k": "v"}})
            ),
        ),
        (
            "DropNamespace",
            call!(
                drop_namespace,
                json!({"mode": "Fail", "behavior": "Restrict"})
            ),
        ),
        ("NamespaceExists", call!(namespace_exists, json!({}))),
        (
            "ListTables",
            call!(
                list_tables,
                json!({"page_token": "p", "limit": 5, "include_declared": true})
            ),
        ),
        (
            "ListAllTables",
            call!(
                list_all_tables,
                json!({"id": null, "page_token": "p", "limit": 5, "include_declared": true})
            ),
        ),
        (
            "RegisterTable",
            call!(
                register_table,
                json!({"location": "t.lance", "mode": "create"})
            ),
        ),
        (
            "DescribeTable",
            call!(
                describe_table,
                json!({"version": 2, "branch": "b", "with_table_uri": true, "load_detailed_metadata": true, "check_declared": true})
            ),
        ),
        ("TableExists", call!(table_exists, json!({"version": 2}))),
        ("DropTable", call!(drop_table, json!({}))),
        ("DeregisterTable", call!(deregister_table, json!({}))),
        (
            "RestoreTable",
            call!(restore_table, json!({"version": 2, "branch": "b"})),
        ),
        (
            "RenameTable",
            call!(rename_table, json!({"new_table_name": "u"})),
        ),
        (
            "UpdateTableSchemaMetadata",
            call!(
                update_table_schema_metadata,
                json!({"metadata": {"k": "v"}, "branch": "b"})
            ),
        ),
        (
            "ListTableVersions",
            call!(
                list_table_versions,
                json!({"branch": "b", "page_token": "p", "limit": 5, "descending": true})
            ),
        ),
        (
            "CreateTableVersion",
            call!(
                create_table_version,
                json!({"version": 2, "manifest_path": "m", "naming_scheme": "V2", "branch": "b"})
            ),
        ),
        (
            "DescribeTableVersion",
            call!(describe_table_version, json!({"version": 2, "branch": "b"})),
        ),
        (
            "BatchDeleteTableVersions",
            call!(
                batch_delete_table_versions,
                json!({"ranges": [{"start_version": 1, "end_version": 2}], "branch": "b"})
            ),
        ),
        (
            "BatchCreateTableVersions",
            call!(
                batch_create_table_versions,
                json!({"id": null, "entries": [{"id": ["ns", "t"], "version": 2, "manifest_path": "m"}]})
            ),
        ),
        (
            "BatchCommitTables",
            call!(
                batch_commit_tables,
                json!({"id": null, "operations": [{"deregister_table": {"id": ["ns", "t"]}}]})
            ),
        ),
        (
            "UpdateFieldMetadata",
            call!(
                update_field_metadata,
                json!({"branch": "b", "updates": [{"path": "c", "metadata": {"k": "v"}}]})
            ),
        ),
        (
            "AlterTableAlterColumns",
            call!(
                alter_table_alter_columns,
                json!({"alterations": [{"path": "c", "nullable": true}]})
            ),
        ),
        (
            "AlterTableDropColumns",
            call!(
                alter_table_drop_columns,
                json!({"columns": ["c"], "branch": "b"})
            ),
        ),
        (
            "GetTableStats",
            call!(get_table_stats, json!({"branch": "b"})),
        ),
        (
            "InsertIntoTable",
            call!(
                insert_into_table,
                json!({"branch": "b", "mode": "append"}),
                arrow()
            ),
        ),
        (
            "MergeInsertIntoTable",
            call!(
                merge_insert_into_table,
                json!({
                    "branch": "b",
                    "on": ["c", "d"],
                    "when_matched_update_all": true,
                    "when_matched_update_all_filt": "c > 1",
                    "when_not_matched_insert_all": true,
                    "when_not_matched_by_source_delete": true,
                    "when_not_matched_by_source_delete_filt": "c < 1",
                    "timeout": "5s",
                    "use_index": true,
                }),
                arrow()
            ),
        ),
        (
            "UpdateTable",
            call!(
                update_table,
                json!({"updates": [["c", "1"]], "predicate": "c > 1", "branch": "b"})
            ),
        ),
        (
            "DeleteFromTable",
            call!(
                delete_from_table,
                json!({"predicate": "c > 1", "branch": "b"})
            ),
        ),
        (
            "QueryTable",
            call!(query_table, json!({"k": 1, "vector": {}})),
        ),
        (
            "CountTableRows",
            call!(
                count_table_rows,
                json!({"version": 2, "branch": "b", "predicate": "c > 1"})
            ),
        ),
        (
            "CreateTable",
            call!(
                create_table,
                json!({"mode": "create", "properties": {"k": "v"}, "storage_options": {"s": "o"}}),
                arrow()
            ),
        ),
        (
            "ExplainTableQueryPlan",
            call!(
                explain_table_query_plan,
                json!({"query": {"k": 1, "vector": {}}, "verbose": true})
            ),
        ),
        (
            "AnalyzeTableQueryPlan",
            call!(analyze_table_query_plan, json!({"k": 1, "vector": {}})),
        ),
        (
            "AlterTableAddColumns",
            call!(
                alter_table_add_columns,
                json!({"new_columns": [{"name": "c", "expression": "1"}]})
            ),
        ),
        (
            "AlterTableBackfillColumns",
            call!(
                alter_table_backfill_columns,
                json!({"column": "c", "branch": "b"})
            ),
        ),
        (
            "RefreshMaterializedView",
            call!(refresh_materialized_view, json!({"concurrency": 2})),
        ),
        (
            "CreateMaterializedView",
            call!(
                create_materialized_view,
                json!({"kind": "query", "source_query": "q", "output_schema": "s"})
            ),
        ),
        (
            "CreateTableIndex",
            call!(
                create_table_index,
                json!({"column": "c", "index_type": "BTREE", "branch": "b"})
            ),
        ),
        (
            "CreateTableScalarIndex",
            call!(
                create_table_scalar_index,
                json!({"column": "c", "index_type": "BTREE"})
            ),
        ),
        (
            "ListTableIndices",
            call!(list_table_indices, json!({"version": 2, "branch": "b"})),
        ),
        (
            "DescribeTableIndexStats",
            call!(
                describe_table_index_stats,
                json!({"index_name": "idx", "version": 2, "branch": "b"})
            ),
        ),
        (
            "DropTableIndex",
            call!(
                drop_table_index,
                json!({"index_name": "idx", "branch": "b"})
            ),
        ),
        (
            "ListTableTags",
            call!(list_table_tags, json!({"page_token": "p", "limit": 5})),
        ),
        (
            "GetTableTagVersion",
            call!(get_table_tag_version, json!({"tag": "v1"})),
        ),
        (
            "DeclareTable",
            call!(declare_table, json!({"location": "t.lance"})),
        ),
        (
            "CreateTableTag",
            call!(create_table_tag, json!({"tag": "v1", "version": 2})),
        ),
        (
            "DeleteTableTag",
            call!(delete_table_tag, json!({"tag": "v1"})),
        ),
        (
            "UpdateTableTag",
            call!(update_table_tag, json!({"tag": "v1", "version": 3})),
        ),
        (
            "ListTableBranches",
            call!(list_table_branches, json!({"page_token": "p", "limit": 5})),
        ),
        (
            "CreateTableBranch",
            call!(create_table_branch, json!({"name": "b", "from_version": 2})),
        ),
        (
            "DeleteTableBranch",
            call!(delete_table_branch, json!({"name": "b"})),
        ),
        (
            "DescribeTransaction",
            call!(describe_transaction, json!({})),
        ),
        (
            "AlterTransaction",
            call!(alter_transaction, json!({"actions": [{}]})),
        ),
    ])
}

/// Checks a request the client sent against the spec.
fn check_request(op: &str, spec: &SpecOp, sent: &Value, got: &wiremock::Request) -> Vec<String> {
    let mut failures = Vec::new();
    let mut fail = |what: String| failures.push(format!("{}: client {}", op, what));

    if got.method.as_str() != spec.method {
        fail(format!("sends {}, spec says {}", got.method, spec.method));
    }
    let path = fill(&spec.path);
    if got.url.path() != path {
        fail(format!("sends path {}, spec says {}", got.url.path(), path));
    }

    let sent_query: BTreeSet<String> = got.url.query_pairs().map(|(k, _)| k.into_owned()).collect();
    for key in &sent_query {
        if !spec.query.contains(key) {
            fail(format!(
                "sends query param {}, which the spec does not define",
                key
            ));
        }
    }
    for key in &spec.query {
        if !sent[key].is_null() && !sent_query.contains(key) {
            fail(format!(
                "drops {}, which the spec sends as a query param",
                key
            ));
        }
    }

    let content_type = got.headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok());
    match &spec.body {
        None if !got.body.is_empty() => fail("sends a body, the spec defines none".to_string()),
        None => {}
        Some((_, _, required)) if got.body.is_empty() => {
            if *required {
                fail("sends no body, the spec requires one".to_string());
            }
        }
        Some((want, schema, _)) => {
            if !content_type.is_some_and(|ct| ct.starts_with(want.as_str())) {
                fail(format!(
                    "sends content type {:?}, spec says {}",
                    content_type, want
                ));
            } else if want == "application/json" {
                let body: Value = serde_json::from_slice(&got.body).unwrap_or_default();
                if !has_shape(schema, &body) {
                    fail(format!(
                        "sends body {}, which does not match the spec's {} schema",
                        body, schema["type"]
                    ));
                }
            }
        }
    }

    for (header, want) in [
        ("x-api-key", "key"),
        ("authorization", "Bearer token"),
        ("x-trace-id", "trace"),
    ] {
        let value = got.headers.get(header).and_then(|v| v.to_str().ok());
        if value != Some(want) {
            fail(format!(
                "sends header {}: {:?}, the request maps it to {}",
                header, value, want
            ));
        }
    }
    failures
}

/// Strips what the adapter cannot get back from a request: the spec sends
/// context entries as headers, and only those with the header. prefix.
fn without_context(mut request: Value) -> Value {
    if let Some(request) = request.as_object_mut() {
        request.remove("context");
    }
    request
}

/// Sends `got` to the adapter as the spec defines the operation, with `body`
/// as the request body, and returns the answer.
async fn replay(
    spec: &SpecOp,
    got: &wiremock::Request,
    body: Vec<u8>,
    adapter: &str,
) -> (u16, HeaderMap, Bytes) {
    let method = reqwest::Method::from_bytes(spec.method.as_bytes()).unwrap();
    let mut url = format!("{}{}", adapter, got.url.path());
    if let Some(query) = got.url.query() {
        url = format!("{}?{}", url, query);
    }
    let mut request = reqwest::Client::new().request(method, url);
    for (name, value) in got.headers.iter() {
        if name != HOST && name != CONTENT_LENGTH && name != CONTENT_TYPE {
            request = request.header(name, value);
        }
    }
    if let Some((content_type, _, _)) = &spec.body {
        request = request.header(CONTENT_TYPE, content_type).body(body);
    }
    let response = request.send().await.unwrap();
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    (status, headers, response.bytes().await.unwrap())
}

/// Replays `got` to the adapter, and checks the answer and what reached the
/// backend.
async fn check_adapter(
    op: &str,
    spec: &SpecOp,
    sent: &Sent,
    got: &wiremock::Request,
    adapter: &str,
    recorder: &RecordingNamespace,
) -> Vec<String> {
    let mut failures = Vec::new();
    let mut fail = |what: String| failures.push(format!("{}: adapter {}", op, what));

    recorder.take();
    let (status, headers, body) = replay(spec, got, got.body.clone(), adapter).await;
    let content_type = headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok());
    if status != spec.status {
        fail(format!(
            "answers {} ({}), spec says {}",
            status,
            String::from_utf8_lossy(&body),
            spec.status
        ));
    }
    match &spec.response {
        None if !body.is_empty() => fail("answers with a body, the spec defines none".to_string()),
        None => {}
        Some((want, _)) if !content_type.is_some_and(|ct| ct.starts_with(want.as_str())) => {
            fail(format!(
                "answers content type {:?}, spec says {}",
                content_type, want
            ));
        }
        Some((want, schema)) if want == "application/json" => {
            match serde_json::from_slice::<Value>(&body) {
                Ok(body) if has_shape(schema, &body) => {}
                _ => fail(format!(
                    "answers {}, which does not match the spec's {} schema",
                    String::from_utf8_lossy(&body),
                    schema["type"]
                )),
            }
            // Object responses are models, whose context we send as headers.
            let served_by = headers.get("x-served-by").and_then(|v| v.to_str().ok());
            if schema["type"] == "object" && served_by != Some("recorder") {
                fail(format!(
                    "answers header x-served-by: {:?}, the response context maps it to recorder",
                    served_by
                ));
            }
        }
        Some(_) => {}
    }

    match recorder.take().pop() {
        None => fail("never called the backend".to_string()),
        Some((method, request, data)) => {
            if method != snake_case(op) {
                fail(format!("called backend {}", method));
            }
            let (want, got) = (
                without_context(sent.request.clone()),
                without_context(request),
            );
            if want != got {
                fail(format!(
                    "passed the backend {}, the client sent {}",
                    got, want
                ));
            }
            if data != sent.data {
                fail(format!(
                    "passed the backend data {:?}, the client sent {:?}",
                    data, sent.data
                ));
            }
        }
    }

    // The spec answers a malformed body, and a body that names another object
    // than the route, with its 400 error model.
    if let Some((content_type, schema, _)) = &spec.body
        && content_type == "application/json"
    {
        let body: Value = serde_json::from_slice(&got.body).unwrap();
        let mut bad = vec![("a malformed body".to_string(), b"{".to_vec())];
        for (param, other) in [("id", json!(["other"])), ("index_name", json!("other"))] {
            if spec.path.contains(&format!("{{{}}}", param))
                && !schema["properties"][param].is_null()
            {
                let mut body = body.clone();
                body[param] = other;
                let what = format!("a body with another {}", param);
                bad.push((what, body.to_string().into_bytes()));
            }
        }
        for (what, bad_body) in bad {
            let (status, _, answer) = replay(spec, got, bad_body, adapter).await;
            let answer: Value = serde_json::from_slice(&answer).unwrap_or_default();
            if status != 400 || answer["code"] != 13 {
                fail(format!(
                    "answers {} ({}) to {}, spec says 400 with error code 13",
                    status, answer, what
                ));
            }
        }
        if !recorder.take().is_empty() {
            fail("called the backend for a bad request".to_string());
        }
    }
    failures
}

// We print what we skip, rather than fail on it.
#[allow(clippy::print_stderr)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rest_spec_conformance() {
    let (spec, from) = match load_spec().await {
        Ok(loaded) => loaded,
        Err(why) if std::env::var_os("CI").is_some() => {
            panic!("{}\nCI is set, so this fails rather than skips.", why)
        }
        Err(why) => {
            eprintln!(
                "skipping the REST spec conformance test: {}\n\
                 The spec is downloaded once and cached in the target directory, \
                 so run this once with network access. With CI set, this fails \
                 rather than skips.",
                why
            );
            return;
        }
    };
    let ops = spec_ops(&spec);
    let calls = calls();
    let mut failures = Vec::new();
    let (mut unserved, mut not_in_trait) = (Vec::new(), Vec::new());

    let server = MockServer::start().await;
    let to_mock = RestNamespaceBuilder::new(server.uri()).build();

    let recorder = Arc::new(RecordingNamespace::default());
    let config = RestAdapterConfig {
        port: 0,
        ..Default::default()
    };
    let adapter = RestAdapter::new(recorder.clone(), config);
    let router = adapter
        .router()
        .fallback(|| async { (StatusCode::NOT_FOUND, [(UNROUTED, "true")]) });
    let handle = adapter.start().await.unwrap();
    let adapter = format!("http://127.0.0.1:{}", handle.port());
    let to_adapter = RestNamespaceBuilder::new(&adapter).build();

    for op in calls.keys() {
        if !ops.contains_key(*op) {
            failures.push(format!("{}: not a spec operation", op));
        }
    }

    for (op, spec_op) in &ops {
        let served = serves(&router, spec_op).await;
        let Some(call) = calls.get(op.as_str()) else {
            if served {
                failures.push(format!(
                    "{}: the adapter serves {} {}, but no call here checks it",
                    op, spec_op.method, spec_op.path
                ));
            } else {
                not_in_trait.push(op.as_str());
            }
            continue;
        };

        server.reset().await;
        Mock::given(any())
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;
        let sent = call(to_mock.clone()).await;
        let mut got = server.received_requests().await.unwrap();
        if got.len() != 1 {
            failures.push(format!("{}: client sent {} requests", op, got.len()));
            continue;
        }
        let got = got.pop().unwrap();
        failures.extend(check_request(op, spec_op, &sent.request, &got));

        if !served {
            unserved.push(op.as_str());
            continue;
        }
        failures.extend(check_adapter(op, spec_op, &sent, &got, &adapter, &recorder).await);

        // The client parses what the adapter answers.
        if let Err(e) = call(to_adapter.clone()).await.result {
            failures.push(format!(
                "{}: client fails on the adapter's answer: {}",
                op, e
            ));
        }
    }
    handle.shutdown();

    // A lance-namespace release can add operations before we implement them,
    // so we report those rather than fail on them.
    eprintln!(
        "checked {} of {} operations in the REST spec of {}",
        ops.len() - unserved.len() - not_in_trait.len(),
        ops.len(),
        from
    );
    for (what, names) in [
        ("not served by RestAdapter", &unserved),
        ("not in LanceNamespace", &not_in_trait),
    ] {
        if !names.is_empty() {
            eprintln!("{} (reported, not failed): {}", what, names.join(", "));
        }
    }

    assert!(
        failures.is_empty(),
        "{} spec mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
