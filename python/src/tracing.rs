// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::CLIENT_VERSION;
use chrono::{SecondsFormat, Utc};
use datafusion_common::HashMap;
use pyo3::Bound;
use pyo3::IntoPyObject;
use pyo3::PyErr;
use pyo3::PyResult;
use pyo3::Python;
use pyo3::exceptions::PyRuntimeError;
use pyo3::pyfunction;
use pyo3::pymethods;
use pyo3::types::PyDict;
use pyo3::types::PyDictMethods;
use pyo3::types::PyTuple;
use pyo3::{Py, PyAny, pyclass};
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::mpsc::TryRecvError;
use std::sync::mpsc::TrySendError;
use std::thread::JoinHandle;
use tracing::Event;
use tracing::field::Field;
use tracing::field::Visit;
use tracing::span;
use tracing::subscriber;
use tracing_chrome::ChromeLayer;
use tracing_chrome::{ChromeLayerBuilder, TraceStyle};
use tracing_subscriber::Registry;
use tracing_subscriber::filter;
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;

static SUBSCRIBER: LazyLock<Arc<RwLock<Option<LoggingPassthroughState>>>> =
    LazyLock::new(|| Arc::new(RwLock::new(None)));

struct TraceEvent {
    target: String,
    args: HashMap<String, String>,
}

#[derive(Clone)]
struct PyTraceEventArgs(HashMap<String, String>);

impl<'py> IntoPyObject<'py> for PyTraceEventArgs {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let dict = PyDict::new(py);
        for (key, value) in self.0 {
            dict.set_item(key, value)?;
        }
        Ok(dict.into_pyobject(py)?)
    }
}

#[pyclass(get_all, name = "TraceEvent")]
pub struct PyTraceEvent {
    target: String,
    args: PyTraceEventArgs,
}

#[pymethods]
impl PyTraceEvent {
    pub fn __repr__(&self) -> String {
        format!("TraceEvent(target={}, args={:?})", self.target, self.args.0)
    }
}

impl From<TraceEvent> for PyTraceEvent {
    fn from(event: TraceEvent) -> Self {
        Self {
            target: event.target,
            args: PyTraceEventArgs(event.args),
        }
    }
}

struct LoggingPassthroughState {
    inner: Option<ChromeLayer<Registry>>,
    level: log::Level,
    callback_sender: Option<mpsc::SyncSender<TraceEvent>>,
    callback_handle: Option<JoinHandle<()>>,
    warned_about_overflow: AtomicBool,
}

impl LoggingPassthroughState {
    fn new(level: log::Level) -> Self {
        Self {
            inner: None,
            level,
            callback_sender: None,
            callback_handle: None,
            warned_about_overflow: AtomicBool::new(false),
        }
    }

    fn set_inner(&mut self, inner: ChromeLayer<Registry>) {
        self.inner = Some(inner);
    }

    fn set_callback(&mut self, callback: Py<PyAny>) -> PyResult<()> {
        if self.callback_sender.is_some() {
            // Panicking here would poison SUBSCRIBER and every later span would panic too.
            return Err(PyRuntimeError::new_err(
                "A trace event callback is already registered; call \
                 lance.tracing.shutdown_tracing() before registering another one",
            ));
        }
        let (sender, receiver) = mpsc::sync_channel(1000);
        self.callback_sender = Some(sender);
        self.callback_handle = Some(std::thread::spawn(move || {
            while let Ok(event) = receiver.recv() {
                Python::attach(|py| {
                    let call_python = |py: Python, event: TraceEvent| {
                        let py_event = PyTraceEvent::from(event);
                        let args = match PyTuple::new(py, [py_event]) {
                            Ok(args) => args,
                            Err(err) => {
                                log::error!("Failed to create PyTuple: {}", err);
                                return;
                            }
                        };
                        match callback.call1(py, args) {
                            Ok(_) => (),
                            Err(err) => {
                                log::error!("Failed to call trace event callback: {}", err);
                            }
                        }
                    };
                    call_python(py, event);
                    // Don't release the GIL if there are more events to process
                    loop {
                        match receiver.try_recv() {
                            Ok(event) => call_python(py, event),
                            // No more events to process, exit the loop
                            Err(TryRecvError::Empty) => break,
                            // Channel was disconnected, exit the thread
                            Err(TryRecvError::Disconnected) => {
                                return;
                            }
                        }
                    }
                });
            }
        }));
        Ok(())
    }

    fn shutdown(&mut self) {
        if let Some(handle) = self.callback_handle.take() {
            // This will signal the callback handler to exit
            self.callback_sender.take();
            // Wait for remaining trace events to be processed. A callback thread that
            // panicked must not panic here too: that would poison SUBSCRIBER.
            if handle.join().is_err() {
                log::warn!("The trace event callback thread panicked");
            }
        }
    }
}

#[derive(Default)]
struct EventToStr {
    str: String,
}

impl Visit for EventToStr {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let name = field.name();
        let name = if let Some(name_stripped) = name.strip_prefix("r#") {
            name_stripped
        } else {
            name
        };
        self.str += &format!("{}={:?} ", name, value);
    }
}

#[derive(Default)]
struct EventToMap {
    args: HashMap<String, String>,
}

impl EventToMap {
    fn field_to_str(field: &Field) -> &str {
        let name = field.name();
        if let Some(name_stripped) = name.strip_prefix("r#") {
            name_stripped
        } else {
            name
        }
    }
}

impl Visit for EventToMap {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        let name = Self::field_to_str(field);
        let value = format!("{:?}", value);
        self.args.insert(name.to_string(), value);
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        let name = Self::field_to_str(field);
        self.args.insert(name.to_string(), value.to_string());
    }
}

fn trace_event_log_target(target: &str) -> String {
    let target = target.strip_prefix("lance::").unwrap_or(target);
    format!("lance::events::{target}")
}

#[derive(Clone)]
pub struct LoggingPassthroughRef(Arc<RwLock<Option<LoggingPassthroughState>>>);

impl LoggingPassthroughRef {
    fn inner_do<F: FnOnce(&ChromeLayer<Registry>)>(&self, f: F) {
        let state_guard = self.0.read().unwrap();
        if let Some(state) = state_guard.as_ref()
            && let Some(inner) = state.inner.as_ref()
        {
            f(inner)
        }
    }
}

impl tracing_subscriber::Layer<Registry> for LoggingPassthroughRef {
    fn on_enter(&self, id: &span::Id, ctx: Context<'_, Registry>) {
        self.inner_do(|inner| inner.on_enter(id, ctx));
    }

    fn on_record(&self, id: &span::Id, values: &span::Record<'_>, ctx: Context<'_, Registry>) {
        self.inner_do(|inner| inner.on_record(id, values, ctx));
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, Registry>) {
        let state_guard = self.0.read().unwrap();
        let state = state_guard.as_ref().unwrap();

        let mut fields = EventToStr::default();
        event.record(&mut fields);
        let log_target = trace_event_log_target(event.metadata().target());
        log::log!(
            target: &log_target,
            state.level,
            "target=\"{}\" {}",
            event.metadata().target(),
            fields.str
        );

        if let Some(callback_sender) = state.callback_sender.as_ref() {
            let mut args = EventToMap::default();
            event.record(&mut args);

            let now = Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);
            args.args.insert("timestamp".to_string(), now.clone());
            args.args
                .insert("client_version".to_string(), CLIENT_VERSION.to_string());

            match callback_sender.try_send(TraceEvent {
                target: event.metadata().target().to_string(),
                args: args.args,
            }) {
                Ok(_) => (),
                Err(TrySendError::Full(_)) => {
                    if state
                        .warned_about_overflow
                        .compare_exchange(
                            false,
                            true,
                            std::sync::atomic::Ordering::Relaxed,
                            std::sync::atomic::Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        log::warn!("Trace event queue overflowed, some events may be dropped");
                    }
                }
                Err(TrySendError::Disconnected(_)) => (),
            }
        }

        if let Some(inner) = state.inner.as_ref() {
            inner.on_event(event, ctx);
        }
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, Registry>) {
        self.inner_do(|inner| inner.on_exit(id, ctx));
    }

    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, Registry>) {
        self.inner_do(|inner| inner.on_new_span(attrs, id, ctx));
    }

    fn on_close(&self, id: span::Id, ctx: Context<'_, Registry>) {
        self.inner_do(|inner| inner.on_close(id, ctx));
    }
}

#[pyclass]
pub struct TraceGuard {
    guard: Arc<Mutex<Option<tracing_chrome::FlushGuard>>>,
}

#[pymethods]
impl TraceGuard {
    pub fn finish_tracing(&self) {
        // We're exiting anyways, so discard the result
        let _ = self.guard.lock().map(|mut g| g.take());
    }
}

#[pyfunction]
#[pyo3(signature=(path=None))]
pub fn trace_to_chrome(path: Option<&str>) -> PyResult<TraceGuard> {
    let mut builder = ChromeLayerBuilder::new()
        .trace_style(TraceStyle::Async)
        .include_args(true);
    if let Some(path) = path {
        builder = builder.file(path);
    }
    let (chrome_layer, guard) = builder.build();

    SUBSCRIBER
        .write()
        .unwrap()
        .as_mut()
        .unwrap()
        .set_inner(chrome_layer);

    Ok(TraceGuard {
        guard: Arc::new(Mutex::new(Some(guard))),
    })
}

pub fn initialize_tracing(level: log::Level) {
    let level_filter = match level {
        log::Level::Trace => filter::LevelFilter::TRACE,
        log::Level::Debug => filter::LevelFilter::DEBUG,
        log::Level::Info => filter::LevelFilter::INFO,
        log::Level::Warn => filter::LevelFilter::WARN,
        log::Level::Error => filter::LevelFilter::ERROR,
    };
    // Narrow down to just our targets, otherwise we get a lot of spam from
    // our dependencies. The target check is based on a prefix, so `lance` is
    // sufficient to match `lance_*`.
    let filter = filter::Targets::new()
        .with_target("lance", level_filter)
        .with_target("pylance", level_filter);

    SUBSCRIBER
        .write()
        .unwrap()
        .replace(LoggingPassthroughState::new(level));

    let subscriber =
        Registry::default().with(LoggingPassthroughRef(SUBSCRIBER.clone()).with_filter(filter));
    subscriber::set_global_default(subscriber).unwrap();
}

#[pyfunction]
#[pyo3(signature=(callback))]
pub fn capture_trace_events(callback: Py<PyAny>, py: Python<'_>) -> PyResult<()> {
    SUBSCRIBER
        .write()
        .unwrap()
        .as_mut()
        .unwrap()
        .set_callback(callback.clone_ref(py))
}

#[pyfunction]
#[pyo3(signature=())]
pub fn shutdown_tracing(py: Python<'_>) {
    // Release Python GIL to avoid deadlock between current thread with the receiver thread.
    py.detach(|| {
        SUBSCRIBER.write().unwrap().as_mut().unwrap().shutdown();
    });
}
