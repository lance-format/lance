// Copyright 2023 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use arrow::pyarrow::*;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use half::bf16;
use lance::arrow::bfloat16::BFLOAT16_EXT_NAME;
use lance::arrow::bfloat16::BFloat16Array;
use lance::arrow::{ARROW_EXT_META_KEY, ARROW_EXT_NAME_KEY};
use pyo3::{exceptions::PyValueError, prelude::*, pyclass::CompareOp, types::PyType};

#[pyclass]
pub struct BFloat16(bf16);

#[pymethods]
impl BFloat16 {
    #[new]
    fn new(value: f32) -> Self {
        Self(bf16::from_f32(value))
    }

    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, bytes: &[u8]) -> PyResult<Self> {
        if bytes.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "BFloat16::from_bytes: expected 2 bytes, got {}",
                bytes.len()
            )));
        }
        Ok(Self(bf16::from_bits(u16::from_ne_bytes([
            bytes[0], bytes[1],
        ]))))
    }

    fn as_float(&self) -> f32 {
        self.0.to_f32()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.0))
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => self.0 == other.0,
            CompareOp::Ge => self.0 >= other.0,
            CompareOp::Le => self.0 <= other.0,
            CompareOp::Gt => self.0 > other.0,
            CompareOp::Lt => self.0 < other.0,
            CompareOp::Ne => self.0 != other.0,
        }
    }
}

const EXPORT_METADATA: [(&str, &str); 2] = [
    (ARROW_EXT_NAME_KEY, BFLOAT16_EXT_NAME),
    (ARROW_EXT_META_KEY, ""),
];

#[pyfunction]
pub fn bfloat16_array<'py>(
    values: Vec<Option<f32>>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let array =
        BFloat16Array::from_iter(values.into_iter().map(|v| v.map(bf16::from_f32))).into_inner();

    // Create a record batch with a single column and an annotated schema
    let field = Field::new("bfloat16", DataType::FixedSizeBinary(2), true).with_metadata(
        EXPORT_METADATA
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<arrow_schema::Metadata>(),
    );
    let schema = Schema::new(vec![field]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)])
        .map_err(|err| PyValueError::new_err(format!("Failed to build array: {}", err)))?;

    let pyarrow_batch = batch.to_pyarrow(py)?;
    pyarrow_batch.call_method1("__getitem__", ("bfloat16",))
}
