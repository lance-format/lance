// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use jni::JNIEnv;
use jni::objects::{GlobalRef, JByteArray, JMap, JObject, JString, JValue};
use jni::sys::{jbyteArray, jlong, jobject, jstring};
use lance_namespace::LanceNamespace as LanceNamespaceTrait;
use lance_namespace::compat::merge_insert_request_from_json;
use lance_namespace::models::*;
use lance_namespace_impls::{
    ConnectBuilder, DirectoryNamespace, DirectoryNamespaceBuilder, DynamicContextProvider,
    OperationInfo, RestAdapter, RestAdapterConfig, RestNamespace, RestNamespaceBuilder,
};
use serde::{Deserialize, Serialize};

use crate::block_on;
use crate::error::{Error, Result};
use crate::utils::to_rust_map;

/// Java-implemented dynamic context provider.
///
/// Wraps a Java object that implements the DynamicContextProvider interface.
pub struct JavaDynamicContextProvider {
    java_provider: GlobalRef,
    jvm: Arc<jni::JavaVM>,
}

impl JavaDynamicContextProvider {
    /// Create a new Java context provider wrapper.
    pub fn new(env: &mut JNIEnv, java_provider: &JObject) -> Result<Self> {
        let java_provider = env.new_global_ref(java_provider)?;
        let jvm = Arc::new(env.get_java_vm()?);
        Ok(Self { java_provider, jvm })
    }
}

impl std::fmt::Debug for JavaDynamicContextProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JavaDynamicContextProvider")
    }
}

impl DynamicContextProvider for JavaDynamicContextProvider {
    fn provide_context(&self, info: &OperationInfo) -> HashMap<String, String> {
        // Attach to JVM
        let mut env = match self.jvm.attach_current_thread() {
            Ok(env) => env,
            Err(e) => {
                log::error!("Failed to attach to JVM: {}", e);
                return HashMap::new();
            }
        };

        // Create Java strings for parameters
        let operation = match env.new_string(&info.operation) {
            Ok(s) => s,
            Err(e) => {
                log::error!("Failed to create operation string: {}", e);
                return HashMap::new();
            }
        };

        let object_id = match env.new_string(&info.object_id) {
            Ok(s) => s,
            Err(e) => {
                log::error!("Failed to create object_id string: {}", e);
                return HashMap::new();
            }
        };

        // Call provideContext(String, String) -> Map<String, String>
        let result = env.call_method(
            &self.java_provider,
            "provideContext",
            "(Ljava/lang/String;Ljava/lang/String;)Ljava/util/Map;",
            &[JValue::Object(&operation), JValue::Object(&object_id)],
        );

        match result {
            Ok(jvalue) => match jvalue.l() {
                Ok(obj) if !obj.is_null() => {
                    // Convert Java Map to Rust HashMap
                    convert_java_map_to_hashmap(&mut env, &obj).unwrap_or_default()
                }
                Ok(_) => HashMap::new(),
                Err(e) => {
                    log::error!("provideContext did not return object: {}", e);
                    HashMap::new()
                }
            },
            Err(e) => {
                log::error!("Failed to call provideContext: {}", e);
                HashMap::new()
            }
        }
    }
}

fn convert_java_map_to_hashmap(
    env: &mut JNIEnv,
    map_obj: &JObject,
) -> Result<HashMap<String, String>> {
    let jmap = JMap::from_env(env, map_obj)?;
    let mut result = HashMap::new();

    let mut iter = jmap.iter(env)?;
    while let Some((key, value)) = iter.next(env)? {
        let key_str: String = env.get_string(&JString::from(key))?.into();
        let value_str: String = env.get_string(&JString::from(value))?.into();
        result.insert(key_str, value_str);
    }

    Ok(result)
}

/// Convert a Rust HashMap<String, u64> to a Java HashMap<String, Long>
fn rust_hashmap_to_java_map<'local>(
    env: &mut JNIEnv<'local>,
    map: &HashMap<String, u64>,
) -> Result<JObject<'local>> {
    let java_map = env.new_object("java/util/HashMap", "()V", &[])?;
    for (k, v) in map {
        let jkey = env.new_string(k)?;
        // Create a java.lang.Long object from the u64 value
        let jval = env.new_object("java/lang/Long", "(J)V", &[JValue::Long(*v as i64)])?;
        env.call_method(
            &java_map,
            "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            &[JValue::Object(&jkey), JValue::Object(&jval)],
        )?;
    }
    Ok(java_map)
}

/// Blocking wrapper for DirectoryNamespace
pub struct BlockingDirectoryNamespace {
    pub(crate) inner: Arc<DirectoryNamespace>,
}

/// Blocking wrapper for RestNamespace
pub struct BlockingRestNamespace {
    pub(crate) inner: Arc<RestNamespace>,
}

// ============================================================================
// JavaLanceNamespace - Generic wrapper for any Java LanceNamespace implementation
// ============================================================================

/// Java-implemented LanceNamespace wrapper.
///
/// This wraps any Java object that implements the LanceNamespace interface
/// and forwards calls to the Java implementation via JNI.
pub struct JavaLanceNamespace {
    java_namespace: GlobalRef,
    jvm: Arc<jni::JavaVM>,
    namespace_id: String,
}

impl std::fmt::Debug for JavaLanceNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JavaLanceNamespace({})", self.namespace_id)
    }
}

impl JavaLanceNamespace {
    /// Create a new wrapper for a Java LanceNamespace object.
    pub fn new(env: &mut JNIEnv, java_namespace: &JObject) -> Result<Self> {
        let java_namespace = env.new_global_ref(java_namespace)?;
        let jvm = Arc::new(env.get_java_vm()?);

        // Cache namespace_id since it's called frequently and won't change
        let namespace_id = Self::call_namespace_id_internal(env, &java_namespace)?;

        Ok(Self {
            java_namespace,
            jvm,
            namespace_id,
        })
    }

    fn call_namespace_id_internal(env: &mut JNIEnv, java_namespace: &GlobalRef) -> Result<String> {
        let result = env
            .call_method(java_namespace, "namespaceId", "()Ljava/lang/String;", &[])
            .map_err(|e| {
                Error::runtime_error(format!(
                    "Failed to call namespaceId on Java namespace: {}",
                    e
                ))
            })?;

        let jstring = result.l().map_err(|e| {
            Error::runtime_error(format!("namespaceId did not return an object: {}", e))
        })?;

        if jstring.is_null() {
            return Err(Error::runtime_error(
                "namespaceId returned null".to_string(),
            ));
        }

        let jstring_ref = JString::from(jstring);
        let java_string = env.get_string(&jstring_ref).map_err(|e| {
            Error::runtime_error(format!(
                "Failed to convert namespaceId to Rust string: {}",
                e
            ))
        })?;

        Ok(java_string.into())
    }
}

impl JavaLanceNamespace {
    /// Helper to deserialize JSON to Java object using ObjectMapper.
    fn deserialize_request<'a>(
        env: &mut JNIEnv<'a>,
        json: &str,
        request_class: &str,
    ) -> lance_core::Result<JObject<'a>> {
        let jrequest_json = env.new_string(json).map_err(|e| {
            lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                "Failed to create request JSON string: {}",
                e
            ))))
        })?;

        // Create ObjectMapper
        let object_mapper_class = env
            .find_class("com/fasterxml/jackson/databind/ObjectMapper")
            .map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to find ObjectMapper class: {}",
                    e
                ))))
            })?;

        let object_mapper = env
            .new_object(&object_mapper_class, "()V", &[])
            .map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to create ObjectMapper: {}",
                    e
                ))))
            })?;

        // Get request class
        let request_class_obj = env.find_class(request_class).map_err(|e| {
            lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                "Failed to find request class {}: {}",
                request_class, e
            ))))
        })?;

        // Call objectMapper.readValue(json, class)
        env.call_method(
            &object_mapper,
            "readValue",
            "(Ljava/lang/String;Ljava/lang/Class;)Ljava/lang/Object;",
            &[
                JValue::Object(&jrequest_json),
                JValue::Object(&request_class_obj),
            ],
        )
        .map_err(|e| {
            lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                "Failed to deserialize request via ObjectMapper: {}",
                e
            ))))
        })?
        .l()
        .map_err(|e| {
            lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                "ObjectMapper.readValue did not return an object: {}",
                e
            ))))
        })
    }

    /// Helper to serialize Java object to JSON using ObjectMapper.
    fn serialize_response(env: &mut JNIEnv, response_obj: &JObject) -> lance_core::Result<String> {
        // Create ObjectMapper
        let object_mapper_class = env
            .find_class("com/fasterxml/jackson/databind/ObjectMapper")
            .map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to find ObjectMapper class: {}",
                    e
                ))))
            })?;

        let object_mapper = env
            .new_object(&object_mapper_class, "()V", &[])
            .map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to create ObjectMapper: {}",
                    e
                ))))
            })?;

        // Call objectMapper.writeValueAsString(obj)
        let response_json_obj = env
            .call_method(
                &object_mapper,
                "writeValueAsString",
                "(Ljava/lang/Object;)Ljava/lang/String;",
                &[JValue::Object(response_obj)],
            )
            .map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to serialize response via ObjectMapper: {}",
                    e
                ))))
            })?
            .l()
            .map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "ObjectMapper.writeValueAsString did not return a string: {}",
                    e
                ))))
            })?;

        let response_str: String = env
            .get_string(&JString::from(response_json_obj))
            .map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to convert response JSON to string: {}",
                    e
                ))))
            })?
            .into();

        Ok(response_str)
    }

    /// Helper to call a Java method that takes a request object and returns a response object.
    /// JSON conversion is done via Jackson ObjectMapper.
    async fn call_json_method<Req, Resp>(
        &self,
        method_name: &'static str,
        request_class: &str,
        response_class: &str,
        request: Req,
    ) -> lance_core::Result<Resp>
    where
        Req: serde::Serialize + Send + 'static,
        Resp: serde::de::DeserializeOwned + Send + 'static,
    {
        let java_namespace = self.java_namespace.clone();
        let jvm = self.jvm.clone();
        let request_class = request_class.to_string();
        let response_class = response_class.to_string();

        tokio::task::spawn_blocking(move || {
            let mut env = jvm.attach_current_thread().map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to attach to JVM: {}",
                    e
                ))))
            })?;

            // Serialize request to JSON
            let request_json = serde_json::to_string(&request).map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to serialize request: {}",
                    e
                ))))
            })?;

            // Deserialize JSON to Java request object via ObjectMapper
            let request_obj = Self::deserialize_request(&mut env, &request_json, &request_class)?;

            // Call the interface method with request object
            let method_sig = format!("(L{};)L{};", request_class, response_class);
            let response_obj = env
                .call_method(
                    &java_namespace,
                    method_name,
                    &method_sig,
                    &[JValue::Object(&request_obj)],
                )
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "Failed to call {}: {}",
                        method_name, e
                    ))))
                })?
                .l()
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "{} did not return an object: {}",
                        method_name, e
                    ))))
                })?;

            if response_obj.is_null() {
                return Err(lance_core::Error::io_source(Box::new(
                    std::io::Error::other(format!("{} returned null", method_name)),
                )));
            }

            // Serialize Java response to JSON via ObjectMapper
            let response_str = Self::serialize_response(&mut env, &response_obj)?;

            serde_json::from_str(&response_str).map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to deserialize response: {}",
                    e
                ))))
            })
        })
        .await
        .map_err(|e| {
            lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                "Failed to spawn blocking task: {}",
                e
            ))))
        })?
    }

    /// Helper for void methods (return ()).
    async fn call_void_method<Req>(
        &self,
        method_name: &'static str,
        request_class: &str,
        request: Req,
    ) -> lance_core::Result<()>
    where
        Req: serde::Serialize + Send + 'static,
    {
        let java_namespace = self.java_namespace.clone();
        let jvm = self.jvm.clone();
        let request_class = request_class.to_string();

        tokio::task::spawn_blocking(move || {
            let mut env = jvm.attach_current_thread().map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to attach to JVM: {}",
                    e
                ))))
            })?;

            // Serialize request to JSON
            let request_json = serde_json::to_string(&request).map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to serialize request: {}",
                    e
                ))))
            })?;

            // Deserialize JSON to Java request object via ObjectMapper
            let request_obj = Self::deserialize_request(&mut env, &request_json, &request_class)?;

            // Call the interface method with request object
            let method_sig = format!("(L{};)V", request_class);
            env.call_method(
                &java_namespace,
                method_name,
                &method_sig,
                &[JValue::Object(&request_obj)],
            )
            .map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to call {}: {}",
                    method_name, e
                ))))
            })?;

            Ok(())
        })
        .await
        .map_err(|e| {
            lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                "Failed to spawn blocking task: {}",
                e
            ))))
        })?
    }

    /// Helper for methods returning a string directly.
    async fn call_string_method<Req>(
        &self,
        method_name: &'static str,
        request_class: &str,
        request: Req,
    ) -> lance_core::Result<String>
    where
        Req: serde::Serialize + Send + 'static,
    {
        let java_namespace = self.java_namespace.clone();
        let jvm = self.jvm.clone();
        let request_class = request_class.to_string();

        tokio::task::spawn_blocking(move || {
            let mut env = jvm.attach_current_thread().map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to attach to JVM: {}",
                    e
                ))))
            })?;

            // Serialize request to JSON
            let request_json = serde_json::to_string(&request).map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to serialize request: {}",
                    e
                ))))
            })?;

            // Deserialize JSON to Java request object via ObjectMapper
            let request_obj = Self::deserialize_request(&mut env, &request_json, &request_class)?;

            // Call the interface method with request object
            let method_sig = format!("(L{};)Ljava/lang/String;", request_class);
            let result = env
                .call_method(
                    &java_namespace,
                    method_name,
                    &method_sig,
                    &[JValue::Object(&request_obj)],
                )
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "Failed to call {}: {}",
                        method_name, e
                    ))))
                })?;

            let response_obj = result.l().map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "{} did not return an object: {}",
                    method_name, e
                ))))
            })?;

            if response_obj.is_null() {
                return Err(lance_core::Error::io_source(Box::new(
                    std::io::Error::other(format!("{} returned null", method_name)),
                )));
            }

            let response_str: String = env
                .get_string(&JString::from(response_obj))
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "Failed to convert response to string: {}",
                        e
                    ))))
                })?
                .into();

            Ok(response_str)
        })
        .await
        .map_err(|e| {
            lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                "Failed to spawn blocking task: {}",
                e
            ))))
        })?
    }

    /// Helper for methods returning Long (boxed).
    async fn call_long_method<Req>(
        &self,
        method_name: &'static str,
        request_class: &str,
        request: Req,
    ) -> lance_core::Result<i64>
    where
        Req: serde::Serialize + Send + 'static,
    {
        let java_namespace = self.java_namespace.clone();
        let jvm = self.jvm.clone();
        let request_class = request_class.to_string();

        tokio::task::spawn_blocking(move || {
            let mut env = jvm.attach_current_thread().map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to attach to JVM: {}",
                    e
                ))))
            })?;

            // Serialize request to JSON
            let request_json = serde_json::to_string(&request).map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to serialize request: {}",
                    e
                ))))
            })?;

            // Deserialize JSON to Java request object via ObjectMapper
            let request_obj = Self::deserialize_request(&mut env, &request_json, &request_class)?;

            // Call the interface method with request object - returns Long (boxed)
            let method_sig = format!("(L{};)Ljava/lang/Long;", request_class);
            let result = env
                .call_method(
                    &java_namespace,
                    method_name,
                    &method_sig,
                    &[JValue::Object(&request_obj)],
                )
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "Failed to call {}: {}",
                        method_name, e
                    ))))
                })?;

            let long_obj = result.l().map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "{} did not return an object: {}",
                    method_name, e
                ))))
            })?;

            if long_obj.is_null() {
                return Err(lance_core::Error::io_source(Box::new(
                    std::io::Error::other(format!("{} returned null", method_name)),
                )));
            }

            // Unbox Long to long
            let long_value = env
                .call_method(&long_obj, "longValue", "()J", &[])
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "Failed to call longValue: {}",
                        e
                    ))))
                })?
                .j()
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "longValue did not return a long: {}",
                        e
                    ))))
                })?;

            Ok(long_value)
        })
        .await
        .map_err(|e| {
            lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                "Failed to spawn blocking task: {}",
                e
            ))))
        })?
    }

    /// Helper for methods with Bytes parameter (request + byte[] data).
    async fn call_with_bytes_method<Req, Resp>(
        &self,
        method_name: &'static str,
        request_class: &str,
        response_class: &str,
        request: Req,
        data: Bytes,
    ) -> lance_core::Result<Resp>
    where
        Req: serde::Serialize + Send + 'static,
        Resp: serde::de::DeserializeOwned + Send + 'static,
    {
        let java_namespace = self.java_namespace.clone();
        let jvm = self.jvm.clone();
        let request_class = request_class.to_string();
        let response_class = response_class.to_string();

        tokio::task::spawn_blocking(move || {
            let mut env = jvm.attach_current_thread().map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to attach to JVM: {}",
                    e
                ))))
            })?;

            // Serialize request to JSON
            let request_json = serde_json::to_string(&request).map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to serialize request: {}",
                    e
                ))))
            })?;

            // Deserialize JSON to Java request object via ObjectMapper
            let request_obj = Self::deserialize_request(&mut env, &request_json, &request_class)?;

            let jdata = env.byte_array_from_slice(&data).map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to create byte array: {}",
                    e
                ))))
            })?;

            // Call the interface method with request object and byte array
            let method_sig = format!("(L{};[B)L{};", request_class, response_class);
            let response_obj = env
                .call_method(
                    &java_namespace,
                    method_name,
                    &method_sig,
                    &[JValue::Object(&request_obj), JValue::Object(&jdata)],
                )
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "Failed to call {}: {}",
                        method_name, e
                    ))))
                })?
                .l()
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "{} did not return an object: {}",
                        method_name, e
                    ))))
                })?;

            if response_obj.is_null() {
                return Err(lance_core::Error::io_source(Box::new(
                    std::io::Error::other(format!("{} returned null", method_name)),
                )));
            }

            // Serialize Java response to JSON via ObjectMapper
            let response_str = Self::serialize_response(&mut env, &response_obj)?;

            serde_json::from_str(&response_str).map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to deserialize response: {}",
                    e
                ))))
            })
        })
        .await
        .map_err(|e| {
            lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                "Failed to spawn blocking task: {}",
                e
            ))))
        })?
    }

    /// Helper for methods returning Bytes (byte[]).
    async fn call_bytes_method<Req>(
        &self,
        method_name: &'static str,
        request_class: &str,
        request: Req,
    ) -> lance_core::Result<Bytes>
    where
        Req: serde::Serialize + Send + 'static,
    {
        let java_namespace = self.java_namespace.clone();
        let jvm = self.jvm.clone();
        let request_class = request_class.to_string();

        tokio::task::spawn_blocking(move || {
            let mut env = jvm.attach_current_thread().map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to attach to JVM: {}",
                    e
                ))))
            })?;

            // Serialize request to JSON
            let request_json = serde_json::to_string(&request).map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to serialize request: {}",
                    e
                ))))
            })?;

            // Deserialize JSON to Java request object via ObjectMapper
            let request_obj = Self::deserialize_request(&mut env, &request_json, &request_class)?;

            // Call the interface method with request object - returns byte[]
            let method_sig = format!("(L{};)[B", request_class);
            let result = env
                .call_method(
                    &java_namespace,
                    method_name,
                    &method_sig,
                    &[JValue::Object(&request_obj)],
                )
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "Failed to call {}: {}",
                        method_name, e
                    ))))
                })?;

            let response_obj = result.l().map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "{} did not return an object: {}",
                    method_name, e
                ))))
            })?;

            if response_obj.is_null() {
                return Err(lance_core::Error::io_source(Box::new(
                    std::io::Error::other(format!("{} returned null", method_name)),
                )));
            }

            let byte_array = JByteArray::from(response_obj);
            let bytes = env.convert_byte_array(byte_array).map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to convert byte array: {}",
                    e
                ))))
            })?;

            Ok(Bytes::from(bytes))
        })
        .await
        .map_err(|e| {
            lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                "Failed to spawn blocking task: {}",
                e
            ))))
        })?
    }

    /// Helper for methods with request + extra String parameter (e.g., indexName).
    /// Extracts the extra string via getter_method on the request object.
    async fn call_json_method_with_extra_string<Req, Resp>(
        &self,
        method_name: &'static str,
        request_class: &str,
        response_class: &str,
        getter_method: &'static str,
        request: Req,
    ) -> lance_core::Result<Resp>
    where
        Req: serde::Serialize + Send + 'static,
        Resp: serde::de::DeserializeOwned + Send + 'static,
    {
        let java_namespace = self.java_namespace.clone();
        let jvm = self.jvm.clone();
        let request_class = request_class.to_string();
        let response_class = response_class.to_string();

        tokio::task::spawn_blocking(move || {
            let mut env = jvm.attach_current_thread().map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to attach to JVM: {}",
                    e
                ))))
            })?;

            // Serialize request to JSON
            let request_json = serde_json::to_string(&request).map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to serialize request: {}",
                    e
                ))))
            })?;

            // Deserialize JSON to Java request object via ObjectMapper
            let request_obj = Self::deserialize_request(&mut env, &request_json, &request_class)?;

            // Call getter method to extract extra string (e.g., getIndexName)
            let extra_string_obj = env
                .call_method(&request_obj, getter_method, "()Ljava/lang/String;", &[])
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "Failed to call {}: {}",
                        getter_method, e
                    ))))
                })?
                .l()
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "{} did not return an object: {}",
                        getter_method, e
                    ))))
                })?;

            // Call the interface method with request object and extra string
            let method_sig = format!(
                "(L{};Ljava/lang/String;)L{};",
                request_class, response_class
            );
            let response_obj = env
                .call_method(
                    &java_namespace,
                    method_name,
                    &method_sig,
                    &[
                        JValue::Object(&request_obj),
                        JValue::Object(&extra_string_obj),
                    ],
                )
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "Failed to call {}: {}",
                        method_name, e
                    ))))
                })?
                .l()
                .map_err(|e| {
                    lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                        "{} did not return an object: {}",
                        method_name, e
                    ))))
                })?;

            if response_obj.is_null() {
                return Err(lance_core::Error::io_source(Box::new(
                    std::io::Error::other(format!("{} returned null", method_name)),
                )));
            }

            // Serialize Java response to JSON via ObjectMapper
            let response_str = Self::serialize_response(&mut env, &response_obj)?;

            serde_json::from_str(&response_str).map_err(|e| {
                lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                    "Failed to deserialize response: {}",
                    e
                ))))
            })
        })
        .await
        .map_err(|e| {
            lance_core::Error::io_source(Box::new(std::io::Error::other(format!(
                "Failed to spawn blocking task: {}",
                e
            ))))
        })?
    }
}

const MODEL_PKG: &str = "org/lance/namespace/model";

#[async_trait]
impl LanceNamespaceTrait for JavaLanceNamespace {
    fn namespace_id(&self) -> String {
        self.namespace_id.clone()
    }

    async fn list_namespaces(
        &self,
        request: ListNamespacesRequest,
    ) -> lance_core::Result<ListNamespacesResponse> {
        self.call_json_method(
            "listNamespaces",
            &format!("{}/ListNamespacesRequest", MODEL_PKG),
            &format!("{}/ListNamespacesResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn describe_namespace(
        &self,
        request: DescribeNamespaceRequest,
    ) -> lance_core::Result<DescribeNamespaceResponse> {
        self.call_json_method(
            "describeNamespace",
            &format!("{}/DescribeNamespaceRequest", MODEL_PKG),
            &format!("{}/DescribeNamespaceResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn create_namespace(
        &self,
        request: CreateNamespaceRequest,
    ) -> lance_core::Result<CreateNamespaceResponse> {
        self.call_json_method(
            "createNamespace",
            &format!("{}/CreateNamespaceRequest", MODEL_PKG),
            &format!("{}/CreateNamespaceResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn drop_namespace(
        &self,
        request: DropNamespaceRequest,
    ) -> lance_core::Result<DropNamespaceResponse> {
        self.call_json_method(
            "dropNamespace",
            &format!("{}/DropNamespaceRequest", MODEL_PKG),
            &format!("{}/DropNamespaceResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn namespace_exists(&self, request: NamespaceExistsRequest) -> lance_core::Result<()> {
        self.call_void_method(
            "namespaceExists",
            &format!("{}/NamespaceExistsRequest", MODEL_PKG),
            request,
        )
        .await
    }

    async fn list_tables(
        &self,
        request: ListTablesRequest,
    ) -> lance_core::Result<ListTablesResponse> {
        self.call_json_method(
            "listTables",
            &format!("{}/ListTablesRequest", MODEL_PKG),
            &format!("{}/ListTablesResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn describe_table(
        &self,
        request: DescribeTableRequest,
    ) -> lance_core::Result<DescribeTableResponse> {
        self.call_json_method(
            "describeTable",
            &format!("{}/DescribeTableRequest", MODEL_PKG),
            &format!("{}/DescribeTableResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn register_table(
        &self,
        request: RegisterTableRequest,
    ) -> lance_core::Result<RegisterTableResponse> {
        self.call_json_method(
            "registerTable",
            &format!("{}/RegisterTableRequest", MODEL_PKG),
            &format!("{}/RegisterTableResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn table_exists(&self, request: TableExistsRequest) -> lance_core::Result<()> {
        self.call_void_method(
            "tableExists",
            &format!("{}/TableExistsRequest", MODEL_PKG),
            request,
        )
        .await
    }

    async fn drop_table(&self, request: DropTableRequest) -> lance_core::Result<DropTableResponse> {
        self.call_json_method(
            "dropTable",
            &format!("{}/DropTableRequest", MODEL_PKG),
            &format!("{}/DropTableResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn deregister_table(
        &self,
        request: DeregisterTableRequest,
    ) -> lance_core::Result<DeregisterTableResponse> {
        self.call_json_method(
            "deregisterTable",
            &format!("{}/DeregisterTableRequest", MODEL_PKG),
            &format!("{}/DeregisterTableResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn count_table_rows(&self, request: CountTableRowsRequest) -> lance_core::Result<i64> {
        self.call_long_method(
            "countTableRows",
            &format!("{}/CountTableRowsRequest", MODEL_PKG),
            request,
        )
        .await
    }

    async fn create_table(
        &self,
        request: CreateTableRequest,
        data: Bytes,
    ) -> lance_core::Result<CreateTableResponse> {
        self.call_with_bytes_method(
            "createTable",
            &format!("{}/CreateTableRequest", MODEL_PKG),
            &format!("{}/CreateTableResponse", MODEL_PKG),
            request,
            data,
        )
        .await
    }

    async fn declare_table(
        &self,
        request: DeclareTableRequest,
    ) -> lance_core::Result<DeclareTableResponse> {
        self.call_json_method(
            "declareTable",
            &format!("{}/DeclareTableRequest", MODEL_PKG),
            &format!("{}/DeclareTableResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn insert_into_table(
        &self,
        request: InsertIntoTableRequest,
        data: Bytes,
    ) -> lance_core::Result<InsertIntoTableResponse> {
        self.call_with_bytes_method(
            "insertIntoTable",
            &format!("{}/InsertIntoTableRequest", MODEL_PKG),
            &format!("{}/InsertIntoTableResponse", MODEL_PKG),
            request,
            data,
        )
        .await
    }

    async fn merge_insert_into_table(
        &self,
        request: MergeInsertIntoTableRequest,
        data: Bytes,
    ) -> lance_core::Result<MergeInsertIntoTableResponse> {
        self.call_with_bytes_method(
            "mergeInsertIntoTable",
            &format!("{}/MergeInsertIntoTableRequest", MODEL_PKG),
            &format!("{}/MergeInsertIntoTableResponse", MODEL_PKG),
            request,
            data,
        )
        .await
    }

    async fn update_table(
        &self,
        request: UpdateTableRequest,
    ) -> lance_core::Result<UpdateTableResponse> {
        self.call_json_method(
            "updateTable",
            &format!("{}/UpdateTableRequest", MODEL_PKG),
            &format!("{}/UpdateTableResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn delete_from_table(
        &self,
        request: DeleteFromTableRequest,
    ) -> lance_core::Result<DeleteFromTableResponse> {
        self.call_json_method(
            "deleteFromTable",
            &format!("{}/DeleteFromTableRequest", MODEL_PKG),
            &format!("{}/DeleteFromTableResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn query_table(&self, request: QueryTableRequest) -> lance_core::Result<Bytes> {
        self.call_bytes_method(
            "queryTable",
            &format!("{}/QueryTableRequest", MODEL_PKG),
            request,
        )
        .await
    }

    async fn create_table_index(
        &self,
        request: CreateTableIndexRequest,
    ) -> lance_core::Result<CreateTableIndexResponse> {
        self.call_json_method(
            "createTableIndex",
            &format!("{}/CreateTableIndexRequest", MODEL_PKG),
            &format!("{}/CreateTableIndexResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn list_table_indices(
        &self,
        request: ListTableIndicesRequest,
    ) -> lance_core::Result<ListTableIndicesResponse> {
        self.call_json_method(
            "listTableIndices",
            &format!("{}/ListTableIndicesRequest", MODEL_PKG),
            &format!("{}/ListTableIndicesResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn describe_table_index_stats(
        &self,
        request: DescribeTableIndexStatsRequest,
    ) -> lance_core::Result<DescribeTableIndexStatsResponse> {
        self.call_json_method_with_extra_string(
            "describeTableIndexStats",
            &format!("{}/DescribeTableIndexStatsRequest", MODEL_PKG),
            &format!("{}/DescribeTableIndexStatsResponse", MODEL_PKG),
            "getIndexName",
            request,
        )
        .await
    }

    async fn describe_transaction(
        &self,
        request: DescribeTransactionRequest,
    ) -> lance_core::Result<DescribeTransactionResponse> {
        self.call_json_method(
            "describeTransaction",
            &format!("{}/DescribeTransactionRequest", MODEL_PKG),
            &format!("{}/DescribeTransactionResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn alter_transaction(
        &self,
        request: AlterTransactionRequest,
    ) -> lance_core::Result<AlterTransactionResponse> {
        self.call_json_method(
            "alterTransaction",
            &format!("{}/AlterTransactionRequest", MODEL_PKG),
            &format!("{}/AlterTransactionResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn create_table_scalar_index(
        &self,
        request: CreateTableIndexRequest,
    ) -> lance_core::Result<CreateTableScalarIndexResponse> {
        self.call_json_method(
            "createTableScalarIndex",
            &format!("{}/CreateTableIndexRequest", MODEL_PKG),
            &format!("{}/CreateTableScalarIndexResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn drop_table_index(
        &self,
        request: DropTableIndexRequest,
    ) -> lance_core::Result<DropTableIndexResponse> {
        self.call_json_method_with_extra_string(
            "dropTableIndex",
            &format!("{}/DropTableIndexRequest", MODEL_PKG),
            &format!("{}/DropTableIndexResponse", MODEL_PKG),
            "getIndexName",
            request,
        )
        .await
    }

    async fn list_all_tables(
        &self,
        request: ListTablesRequest,
    ) -> lance_core::Result<ListTablesResponse> {
        self.call_json_method(
            "listAllTables",
            &format!("{}/ListTablesRequest", MODEL_PKG),
            &format!("{}/ListTablesResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn restore_table(
        &self,
        request: RestoreTableRequest,
    ) -> lance_core::Result<RestoreTableResponse> {
        self.call_json_method(
            "restoreTable",
            &format!("{}/RestoreTableRequest", MODEL_PKG),
            &format!("{}/RestoreTableResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn rename_table(
        &self,
        request: RenameTableRequest,
    ) -> lance_core::Result<RenameTableResponse> {
        self.call_json_method(
            "renameTable",
            &format!("{}/RenameTableRequest", MODEL_PKG),
            &format!("{}/RenameTableResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn list_table_versions(
        &self,
        request: ListTableVersionsRequest,
    ) -> lance_core::Result<ListTableVersionsResponse> {
        self.call_json_method(
            "listTableVersions",
            &format!("{}/ListTableVersionsRequest", MODEL_PKG),
            &format!("{}/ListTableVersionsResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn create_table_version(
        &self,
        request: CreateTableVersionRequest,
    ) -> lance_core::Result<CreateTableVersionResponse> {
        self.call_json_method(
            "createTableVersion",
            &format!("{}/CreateTableVersionRequest", MODEL_PKG),
            &format!("{}/CreateTableVersionResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn describe_table_version(
        &self,
        request: DescribeTableVersionRequest,
    ) -> lance_core::Result<DescribeTableVersionResponse> {
        self.call_json_method(
            "describeTableVersion",
            &format!("{}/DescribeTableVersionRequest", MODEL_PKG),
            &format!("{}/DescribeTableVersionResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn batch_delete_table_versions(
        &self,
        request: BatchDeleteTableVersionsRequest,
    ) -> lance_core::Result<BatchDeleteTableVersionsResponse> {
        self.call_json_method(
            "batchDeleteTableVersions",
            &format!("{}/BatchDeleteTableVersionsRequest", MODEL_PKG),
            &format!("{}/BatchDeleteTableVersionsResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn update_table_schema_metadata(
        &self,
        request: UpdateTableSchemaMetadataRequest,
    ) -> lance_core::Result<UpdateTableSchemaMetadataResponse> {
        self.call_json_method(
            "updateTableSchemaMetadata",
            &format!("{}/UpdateTableSchemaMetadataRequest", MODEL_PKG),
            &format!("{}/UpdateTableSchemaMetadataResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn get_table_stats(
        &self,
        request: GetTableStatsRequest,
    ) -> lance_core::Result<GetTableStatsResponse> {
        self.call_json_method(
            "getTableStats",
            &format!("{}/GetTableStatsRequest", MODEL_PKG),
            &format!("{}/GetTableStatsResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn explain_table_query_plan(
        &self,
        request: ExplainTableQueryPlanRequest,
    ) -> lance_core::Result<String> {
        self.call_string_method(
            "explainTableQueryPlan",
            &format!("{}/ExplainTableQueryPlanRequest", MODEL_PKG),
            request,
        )
        .await
    }

    async fn analyze_table_query_plan(
        &self,
        request: AnalyzeTableQueryPlanRequest,
    ) -> lance_core::Result<String> {
        self.call_string_method(
            "analyzeTableQueryPlan",
            &format!("{}/AnalyzeTableQueryPlanRequest", MODEL_PKG),
            request,
        )
        .await
    }

    async fn alter_table_add_columns(
        &self,
        request: AlterTableAddColumnsRequest,
    ) -> lance_core::Result<AlterTableAddColumnsResponse> {
        self.call_json_method(
            "alterTableAddColumns",
            &format!("{}/AlterTableAddColumnsRequest", MODEL_PKG),
            &format!("{}/AlterTableAddColumnsResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn alter_table_alter_columns(
        &self,
        request: AlterTableAlterColumnsRequest,
    ) -> lance_core::Result<AlterTableAlterColumnsResponse> {
        self.call_json_method(
            "alterTableAlterColumns",
            &format!("{}/AlterTableAlterColumnsRequest", MODEL_PKG),
            &format!("{}/AlterTableAlterColumnsResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn alter_table_drop_columns(
        &self,
        request: AlterTableDropColumnsRequest,
    ) -> lance_core::Result<AlterTableDropColumnsResponse> {
        self.call_json_method(
            "alterTableDropColumns",
            &format!("{}/AlterTableDropColumnsRequest", MODEL_PKG),
            &format!("{}/AlterTableDropColumnsResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn list_table_tags(
        &self,
        request: ListTableTagsRequest,
    ) -> lance_core::Result<ListTableTagsResponse> {
        self.call_json_method(
            "listTableTags",
            &format!("{}/ListTableTagsRequest", MODEL_PKG),
            &format!("{}/ListTableTagsResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn get_table_tag_version(
        &self,
        request: GetTableTagVersionRequest,
    ) -> lance_core::Result<GetTableTagVersionResponse> {
        self.call_json_method(
            "getTableTagVersion",
            &format!("{}/GetTableTagVersionRequest", MODEL_PKG),
            &format!("{}/GetTableTagVersionResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn create_table_tag(
        &self,
        request: CreateTableTagRequest,
    ) -> lance_core::Result<CreateTableTagResponse> {
        self.call_json_method(
            "createTableTag",
            &format!("{}/CreateTableTagRequest", MODEL_PKG),
            &format!("{}/CreateTableTagResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn delete_table_tag(
        &self,
        request: DeleteTableTagRequest,
    ) -> lance_core::Result<DeleteTableTagResponse> {
        self.call_json_method(
            "deleteTableTag",
            &format!("{}/DeleteTableTagRequest", MODEL_PKG),
            &format!("{}/DeleteTableTagResponse", MODEL_PKG),
            request,
        )
        .await
    }

    async fn update_table_tag(
        &self,
        request: UpdateTableTagRequest,
    ) -> lance_core::Result<UpdateTableTagResponse> {
        self.call_json_method(
            "updateTableTag",
            &format!("{}/UpdateTableTagRequest", MODEL_PKG),
            &format!("{}/UpdateTableTagResponse", MODEL_PKG),
            request,
        )
        .await
    }
}

/// Create a JavaLanceNamespace wrapper from a JNI environment and Java object.
pub fn create_java_lance_namespace(
    env: &mut JNIEnv,
    java_namespace: &JObject,
) -> Result<Arc<dyn LanceNamespaceTrait>> {
    let wrapper = JavaLanceNamespace::new(env, java_namespace)?;
    Ok(Arc::new(wrapper))
}

// ============================================================================
// DirectoryNamespace JNI Functions
// ============================================================================

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createNative(
    mut env: JNIEnv,
    _obj: JObject,
    properties_map: JObject,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        create_directory_namespace_internal(&mut env, properties_map, None),
        0
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createNativeWithProvider(
    mut env: JNIEnv,
    _obj: JObject,
    properties_map: JObject,
    context_provider: JObject,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        create_directory_namespace_internal(&mut env, properties_map, Some(context_provider)),
        0
    )
}

fn create_directory_namespace_internal(
    env: &mut JNIEnv,
    properties_map: JObject,
    context_provider: Option<JObject>,
) -> Result<jlong> {
    // Convert Java HashMap to Rust HashMap
    let jmap = JMap::from_env(env, &properties_map)?;
    let properties = to_rust_map(env, &jmap)?;

    // Build DirectoryNamespace using builder
    let mut builder =
        DirectoryNamespaceBuilder::from_properties(properties, None).map_err(|e| {
            Error::runtime_error(format!("Failed to create DirectoryNamespaceBuilder: {}", e))
        })?;

    // Add context provider if provided
    if let Some(provider_obj) = context_provider
        && !provider_obj.is_null()
    {
        let java_provider = JavaDynamicContextProvider::new(env, &provider_obj)?;
        builder = builder.context_provider(Arc::new(java_provider));
    }

    let namespace = block_on(builder.build())
        .map_err(|e| Error::runtime_error(format!("Failed to build DirectoryNamespace: {}", e)))?;

    let blocking_namespace = BlockingDirectoryNamespace {
        inner: Arc::new(namespace),
    };
    let handle = Box::into_raw(Box::new(blocking_namespace)) as jlong;
    Ok(handle)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_releaseNative(
    _env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            let _ = Box::from_raw(handle as *mut BlockingDirectoryNamespace);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_namespaceIdNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jstring {
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let namespace_id = namespace.inner.namespace_id();
    ok_or_throw_with_return!(
        env,
        env.new_string(namespace_id).map_err(Error::from),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_listNamespacesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.list_namespaces(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_describeNamespaceNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.describe_namespace(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createNamespaceNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.create_namespace(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_dropNamespaceNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.drop_namespace(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_namespaceExistsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) {
    ok_or_throw_without_return!(
        env,
        call_namespace_void_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.namespace_exists(req))
        })
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_listTablesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.list_tables(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_describeTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.describe_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_registerTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.register_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_tableExistsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) {
    ok_or_throw_without_return!(
        env,
        call_namespace_void_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.table_exists(req))
        })
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_dropTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.drop_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_deregisterTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.deregister_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_countTableRowsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        call_namespace_count_method(&mut env, handle, request_json),
        0
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_with_data_method(
            &mut env,
            handle,
            request_json,
            request_data,
            |namespace_client, req, data| {
                block_on(namespace_client.inner.create_table(req, data))
            }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_declareTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.declare_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_renameTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.rename_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_insertIntoTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_with_data_method(
            &mut env,
            handle,
            request_json,
            request_data,
            |namespace_client, req, data| {
                block_on(namespace_client.inner.insert_into_table(req, data))
            }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_mergeInsertIntoTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_with_data_method(
            &mut env,
            handle,
            request_json,
            request_data,
            |namespace_client, req: serde_json::Value, data| {
                let req = merge_insert_request_from_json(req)?;
                block_on(namespace_client.inner.merge_insert_into_table(req, data))
            }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_updateTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.update_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_deleteFromTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.delete_from_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_queryTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        call_namespace_query_method(&mut env, handle, request_json),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createTableIndexNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.create_table_index(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_listTableIndicesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.list_table_indices(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_describeTableIndexStatsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.describe_table_index_stats(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_describeTransactionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.describe_transaction(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_alterTransactionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.alter_transaction(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_listTableVersionsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.list_table_versions(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createTableVersionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.create_table_version(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_describeTableVersionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.describe_table_version(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_batchDeleteTableVersionsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.batch_delete_table_versions(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createTableScalarIndexNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.create_table_scalar_index(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_dropTableIndexNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.drop_table_index(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_listAllTablesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.list_all_tables(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_restoreTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.restore_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_updateTableSchemaMetadataNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.update_table_schema_metadata(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_getTableStatsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.get_table_stats(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_explainTableQueryPlanNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_string_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.explain_table_query_plan(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_analyzeTableQueryPlanNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_string_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.analyze_table_query_plan(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_alterTableAddColumnsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.alter_table_add_columns(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_alterTableAlterColumnsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.alter_table_alter_columns(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_alterTableDropColumnsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.alter_table_drop_columns(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_alterTableBackfillColumnsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.alter_table_backfill_columns(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_refreshMaterializedViewNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.refresh_materialized_view(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_listTableTagsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.list_table_tags(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_getTableTagVersionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.get_table_tag_version(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createTableTagNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.create_table_tag(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_deleteTableTagNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.delete_table_tag(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_updateTableTagNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.update_table_tag(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_createMaterializedViewNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.create_materialized_view(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_retrieveOpsMetricsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jobject {
    ok_or_throw_with_return!(
        env,
        {
            let namespace_client = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
            let metrics = namespace_client.inner.retrieve_ops_metrics();
            rust_hashmap_to_java_map(&mut env, &metrics)
        },
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_DirectoryNamespace_resetOpsMetricsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    ok_or_throw_without_return!(env, {
        let namespace_client = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
        namespace_client.inner.reset_ops_metrics();
        Ok::<(), Error>(())
    });
}

// ============================================================================
// RestNamespace JNI Functions
// ============================================================================

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createNative(
    mut env: JNIEnv,
    _obj: JObject,
    properties_map: JObject,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        create_rest_namespace_internal(&mut env, properties_map, None),
        0
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createNativeWithProvider(
    mut env: JNIEnv,
    _obj: JObject,
    properties_map: JObject,
    context_provider: JObject,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        create_rest_namespace_internal(&mut env, properties_map, Some(context_provider)),
        0
    )
}

fn create_rest_namespace_internal(
    env: &mut JNIEnv,
    properties_map: JObject,
    context_provider: Option<JObject>,
) -> Result<jlong> {
    // Convert Java HashMap to Rust HashMap
    let jmap = JMap::from_env(env, &properties_map)?;
    let properties = to_rust_map(env, &jmap)?;

    // Build RestNamespace using builder
    let mut builder = RestNamespaceBuilder::from_properties(properties).map_err(|e| {
        Error::runtime_error(format!("Failed to create RestNamespaceBuilder: {}", e))
    })?;

    // Add context provider if provided
    if let Some(provider_obj) = context_provider
        && !provider_obj.is_null()
    {
        let java_provider = JavaDynamicContextProvider::new(env, &provider_obj)?;
        builder = builder.context_provider(Arc::new(java_provider));
    }

    let namespace = builder.build();

    let blocking_namespace = BlockingRestNamespace {
        inner: Arc::new(namespace),
    };
    let handle = Box::into_raw(Box::new(blocking_namespace)) as jlong;
    Ok(handle)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_releaseNative(
    _env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            let _ = Box::from_raw(handle as *mut BlockingRestNamespace);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_namespaceIdNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jstring {
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let namespace_id = namespace.inner.namespace_id();
    ok_or_throw_with_return!(
        env,
        env.new_string(namespace_id).map_err(Error::from),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_listNamespacesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.list_namespaces(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_describeNamespaceNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.describe_namespace(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createNamespaceNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.create_namespace(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_dropNamespaceNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.drop_namespace(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_namespaceExistsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) {
    ok_or_throw_without_return!(
        env,
        call_rest_namespace_void_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.namespace_exists(req))
        })
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_listTablesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.list_tables(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_describeTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.describe_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_registerTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.register_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_tableExistsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) {
    ok_or_throw_without_return!(
        env,
        call_rest_namespace_void_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.table_exists(req))
        })
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_dropTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.drop_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_deregisterTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.deregister_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_countTableRowsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_count_method(&mut env, handle, request_json),
        0
    )
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_with_data_method(
            &mut env,
            handle,
            request_json,
            request_data,
            |namespace_client, req, data| {
                block_on(namespace_client.inner.create_table(req, data))
            }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_declareTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.declare_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_renameTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.rename_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_insertIntoTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_with_data_method(
            &mut env,
            handle,
            request_json,
            request_data,
            |namespace_client, req, data| {
                block_on(namespace_client.inner.insert_into_table(req, data))
            }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_mergeInsertIntoTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_with_data_method(
            &mut env,
            handle,
            request_json,
            request_data,
            |namespace_client, req: serde_json::Value, data| {
                let req = merge_insert_request_from_json(req)?;
                block_on(namespace_client.inner.merge_insert_into_table(req, data))
            }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_updateTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.update_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_deleteFromTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.delete_from_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_queryTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jbyteArray {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_query_method(&mut env, handle, request_json),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createTableIndexNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.create_table_index(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_listTableIndicesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.list_table_indices(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_describeTableIndexStatsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.describe_table_index_stats(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_describeTransactionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.describe_transaction(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_alterTransactionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.alter_transaction(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_listTableVersionsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.list_table_versions(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createTableVersionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.create_table_version(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_describeTableVersionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.describe_table_version(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_batchDeleteTableVersionsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.batch_delete_table_versions(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createTableScalarIndexNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.create_table_scalar_index(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_dropTableIndexNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.drop_table_index(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_listAllTablesNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.list_all_tables(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_restoreTableNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.restore_table(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_updateTableSchemaMetadataNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.update_table_schema_metadata(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_getTableStatsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.get_table_stats(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_explainTableQueryPlanNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_string_method(
            &mut env,
            handle,
            request_json,
            |namespace_client, req| {
                block_on(namespace_client.inner.explain_table_query_plan(req))
            }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_analyzeTableQueryPlanNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_string_method(
            &mut env,
            handle,
            request_json,
            |namespace_client, req| {
                block_on(namespace_client.inner.analyze_table_query_plan(req))
            }
        ),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_alterTableAddColumnsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.alter_table_add_columns(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_alterTableAlterColumnsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.alter_table_alter_columns(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_alterTableDropColumnsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.alter_table_drop_columns(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_alterTableBackfillColumnsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.alter_table_backfill_columns(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_refreshMaterializedViewNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.refresh_materialized_view(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_listTableTagsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.list_table_tags(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_getTableTagVersionNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.get_table_tag_version(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createTableTagNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.create_table_tag(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_deleteTableTagNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.delete_table_tag(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_updateTableTagNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.update_table_tag(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_createMaterializedViewNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
    request_json: JString,
) -> jstring {
    ok_or_throw_with_return!(
        env,
        call_rest_namespace_method(&mut env, handle, request_json, |namespace_client, req| {
            block_on(namespace_client.inner.create_materialized_view(req))
        }),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_retrieveOpsMetricsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jobject {
    ok_or_throw_with_return!(
        env,
        {
            let namespace_client = unsafe { &*(handle as *const BlockingRestNamespace) };
            let metrics = namespace_client.inner.retrieve_ops_metrics();
            rust_hashmap_to_java_map(&mut env, &metrics)
        },
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestNamespace_resetOpsMetricsNative(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    ok_or_throw_without_return!(env, {
        let namespace_client = unsafe { &*(handle as *const BlockingRestNamespace) };
        namespace_client.inner.reset_ops_metrics();
        Ok::<(), Error>(())
    });
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Helper function to call namespace methods that return a response object (DirectoryNamespace)
fn call_namespace_method<'local, Req, Resp, F>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
    f: F,
) -> Result<JString<'local>>
where
    Req: for<'de> Deserialize<'de>,
    Resp: Serialize,
    F: FnOnce(&BlockingDirectoryNamespace, Req) -> lance_core::Result<Resp>,
{
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let response = f(namespace, request).map_err(Error::from)?;

    let response_json = serde_json::to_string(&response)
        .map_err(|e| Error::runtime_error(format!("Failed to serialize response: {}", e)))?;

    env.new_string(response_json).map_err(Into::into)
}

/// Helper function for void methods (DirectoryNamespace)
fn call_namespace_void_method<Req, F>(
    env: &mut JNIEnv,
    handle: jlong,
    request_json: JString,
    f: F,
) -> Result<()>
where
    Req: for<'de> Deserialize<'de>,
    F: FnOnce(&BlockingDirectoryNamespace, Req) -> lance_core::Result<()>,
{
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    f(namespace, request).map_err(Error::from)?;

    Ok(())
}

/// Helper function for methods that return String directly (not JSON-serialized) (DirectoryNamespace)
fn call_namespace_string_method<'local, Req, F>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
    f: F,
) -> Result<JString<'local>>
where
    Req: for<'de> Deserialize<'de>,
    F: FnOnce(&BlockingDirectoryNamespace, Req) -> lance_core::Result<String>,
{
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let response = f(namespace, request).map_err(Error::from)?;

    env.new_string(response).map_err(Into::into)
}

/// Helper function for count methods (DirectoryNamespace)
fn call_namespace_count_method(
    env: &mut JNIEnv,
    handle: jlong,
    request_json: JString,
) -> Result<jlong> {
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: CountTableRowsRequest = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let count = block_on(namespace.inner.count_table_rows(request)).map_err(Error::from)?;

    Ok(count)
}

/// Helper function for methods with data parameter (DirectoryNamespace)
fn call_namespace_with_data_method<'local, Req, Resp, F>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
    f: F,
) -> Result<JString<'local>>
where
    Req: for<'de> Deserialize<'de>,
    Resp: Serialize,
    F: FnOnce(&BlockingDirectoryNamespace, Req, Bytes) -> lance_core::Result<Resp>,
{
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let data_vec = env.convert_byte_array(request_data)?;
    let data = bytes::Bytes::from(data_vec);

    let response = f(namespace, request, data).map_err(Error::from)?;

    let response_json = serde_json::to_string(&response)
        .map_err(|e| Error::runtime_error(format!("Failed to serialize response: {}", e)))?;

    env.new_string(response_json).map_err(Into::into)
}

/// Helper function for query methods that return byte arrays (DirectoryNamespace)
fn call_namespace_query_method<'local>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
) -> Result<JByteArray<'local>> {
    let namespace = unsafe { &*(handle as *const BlockingDirectoryNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: QueryTableRequest = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let result_bytes = block_on(namespace.inner.query_table(request)).map_err(Error::from)?;

    let byte_array = env.byte_array_from_slice(&result_bytes)?;
    Ok(byte_array)
}

/// Helper function to call namespace methods that return a response object (RestNamespace)
fn call_rest_namespace_method<'local, Req, Resp, F>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
    f: F,
) -> Result<JString<'local>>
where
    Req: for<'de> Deserialize<'de>,
    Resp: Serialize,
    F: FnOnce(&BlockingRestNamespace, Req) -> lance_core::Result<Resp>,
{
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let response = f(namespace, request).map_err(Error::from)?;

    let response_json = serde_json::to_string(&response)
        .map_err(|e| Error::runtime_error(format!("Failed to serialize response: {}", e)))?;

    env.new_string(response_json).map_err(Into::into)
}

/// Helper function for void methods (RestNamespace)
fn call_rest_namespace_void_method<Req, F>(
    env: &mut JNIEnv,
    handle: jlong,
    request_json: JString,
    f: F,
) -> Result<()>
where
    Req: for<'de> Deserialize<'de>,
    F: FnOnce(&BlockingRestNamespace, Req) -> lance_core::Result<()>,
{
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    f(namespace, request).map_err(Error::from)?;

    Ok(())
}

/// Helper function for methods that return String directly (not JSON-serialized) (RestNamespace)
fn call_rest_namespace_string_method<'local, Req, F>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
    f: F,
) -> Result<JString<'local>>
where
    Req: for<'de> Deserialize<'de>,
    F: FnOnce(&BlockingRestNamespace, Req) -> lance_core::Result<String>,
{
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let response = f(namespace, request).map_err(Error::from)?;

    env.new_string(response).map_err(Into::into)
}

/// Helper function for count methods (RestNamespace)
fn call_rest_namespace_count_method(
    env: &mut JNIEnv,
    handle: jlong,
    request_json: JString,
) -> Result<jlong> {
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: CountTableRowsRequest = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let count = block_on(namespace.inner.count_table_rows(request)).map_err(Error::from)?;

    Ok(count)
}

/// Helper function for methods with data parameter (RestNamespace)
fn call_rest_namespace_with_data_method<'local, Req, Resp, F>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
    request_data: JByteArray,
    f: F,
) -> Result<JString<'local>>
where
    Req: for<'de> Deserialize<'de>,
    Resp: Serialize,
    F: FnOnce(&BlockingRestNamespace, Req, Bytes) -> lance_core::Result<Resp>,
{
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: Req = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let data_vec = env.convert_byte_array(request_data)?;
    let data = bytes::Bytes::from(data_vec);

    let response = f(namespace, request, data).map_err(Error::from)?;

    let response_json = serde_json::to_string(&response)
        .map_err(|e| Error::runtime_error(format!("Failed to serialize response: {}", e)))?;

    env.new_string(response_json).map_err(Into::into)
}

/// Helper function for query methods that return byte arrays (RestNamespace)
fn call_rest_namespace_query_method<'local>(
    env: &mut JNIEnv<'local>,
    handle: jlong,
    request_json: JString,
) -> Result<JByteArray<'local>> {
    let namespace = unsafe { &*(handle as *const BlockingRestNamespace) };
    let request_str: String = env.get_string(&request_json)?.into();
    let request: QueryTableRequest = serde_json::from_str(&request_str)
        .map_err(|e| Error::input_error(format!("Failed to parse request JSON: {}", e)))?;

    let result_bytes = block_on(namespace.inner.query_table(request)).map_err(Error::from)?;

    let byte_array = env.byte_array_from_slice(&result_bytes)?;
    Ok(byte_array)
}
// ============================================================================
// RestAdapter - Server for testing
// ============================================================================

/// Wrapper for RestAdapter that manages the server lifecycle
pub struct BlockingRestAdapter {
    backend: Arc<dyn LanceNamespaceTrait>,
    config: RestAdapterConfig,
    server_handle: Option<lance_namespace_impls::RestAdapterHandle>,
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestAdapter_createNative(
    mut env: JNIEnv,
    _obj: JObject,
    namespace_client_impl: JString,
    properties_map: JObject,
    host: JString,
    port: JObject,
) -> jlong {
    ok_or_throw_with_return!(
        env,
        create_rest_adapter_internal(&mut env, namespace_client_impl, properties_map, host, port),
        0
    )
}

fn create_rest_adapter_internal(
    env: &mut JNIEnv,
    namespace_client_impl: JString,
    properties_map: JObject,
    host: JString,
    port: JObject,
) -> Result<jlong> {
    // Get namespace client implementation type
    let impl_str: String = env.get_string(&namespace_client_impl)?.into();

    // Convert Java HashMap to Rust HashMap
    let jmap = JMap::from_env(env, &properties_map)?;
    let properties = to_rust_map(env, &jmap)?;

    // Build backend namespace using ConnectBuilder
    let mut builder = ConnectBuilder::new(impl_str);
    for (k, v) in properties {
        builder = builder.property(k, v);
    }

    let backend = block_on(builder.connect())
        .map_err(|e| Error::runtime_error(format!("Failed to build backend namespace: {}", e)))?;

    // Build config with defaults, overriding if values provided
    let mut config = RestAdapterConfig::default();

    // Get host string if not null
    if !host.is_null() {
        config.host = env.get_string(&host)?.into();
    }

    // Get port if not null (Integer object)
    if !port.is_null() {
        let port_value = env
            .call_method(&port, "intValue", "()I", &[])?
            .i()
            .map_err(|e| Error::runtime_error(format!("Failed to get port value: {}", e)))?;
        config.port = port_value as u16;
    }

    let adapter = BlockingRestAdapter {
        backend,
        config,
        server_handle: None,
    };

    let handle = Box::into_raw(Box::new(adapter)) as jlong;
    Ok(handle)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestAdapter_start(
    mut env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    ok_or_throw_without_return!(env, start_internal(handle))
}

fn start_internal(handle: jlong) -> Result<()> {
    let adapter = unsafe { &mut *(handle as *mut BlockingRestAdapter) };
    let rest_adapter = RestAdapter::new(adapter.backend.clone(), adapter.config.clone());
    let server_handle = block_on(rest_adapter.start())?;
    adapter.server_handle = Some(server_handle);
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestAdapter_getPort(
    _env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) -> jni::sys::jint {
    let adapter = unsafe { &*(handle as *const BlockingRestAdapter) };
    adapter
        .server_handle
        .as_ref()
        .map(|h| h.port() as jni::sys::jint)
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestAdapter_stop(
    _env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    let adapter = unsafe { &mut *(handle as *mut BlockingRestAdapter) };

    if let Some(server_handle) = adapter.server_handle.take() {
        server_handle.shutdown();
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_namespace_RestAdapter_releaseNative(
    _env: JNIEnv,
    _obj: JObject,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            let mut adapter = Box::from_raw(handle as *mut BlockingRestAdapter);
            if let Some(server_handle) = adapter.server_handle.take() {
                server_handle.shutdown();
            }
        }
    }
}
