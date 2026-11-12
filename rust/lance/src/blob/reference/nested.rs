// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::io::Cursor;

use arrow_array::{Array, ArrayRef, RecordBatch, cast::AsArray, types::UInt64Type};
use arrow_ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_schema::DataType;
use lance_core::{Error, Result};
use prost::Message;

use super::{BlobRefBatch, VERSION, check_version, pb};

/// Flat references plus an Arrow mapping back to a nested scan's shape.
///
/// The mapping has one column and retains only the path to the selected Blob V2
/// leaf. Struct names/nulls and List/LargeList offsets/nulls are preserved. Each
/// non-null uint64 leaf indexes [`Self::references`]; null leaves denote null
/// blobs or hidden children of null containers. Row order and list ordinals
/// identify elements even when several blobs belong to one source row.
///
/// Transport this envelope, or transport the mapping in the engine's Arrow stream
/// alongside the reference batch. Neither encoding acquires snapshot retention.
///
/// ```
/// # use lance::Result;
/// # use lance::blob::reference::NestedBlobRefBatch;
/// # fn example(captured: &NestedBlobRefBatch) -> Result<()> {
/// let restored = NestedBlobRefBatch::decode(&captured.encode()?)?;
/// assert_eq!(restored.indices(), captured.indices());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct NestedBlobRefBatch {
    references: BlobRefBatch,
    indices: RecordBatch,
}

impl NestedBlobRefBatch {
    /// Validate a transported mapping and its reference table. Structs must
    /// contain exactly the selected child; Map, unions and fixed-size lists are unsupported.
    pub fn try_new(references: BlobRefBatch, indices: RecordBatch) -> Result<Self> {
        if indices.num_columns() != 1 {
            return Err(Error::invalid_input(
                "Nested blob indices require exactly one column",
            ));
        }
        validate_indices(indices.column(0), references.array().len())?;
        Ok(Self {
            references,
            indices,
        })
    }

    /// Flat nullable references accepted by the existing whole/range reader.
    pub fn references(&self) -> &BlobRefBatch {
        &self.references
    }

    /// One Arrow column with the original selected container path and uint64 leaves.
    pub fn indices(&self) -> &RecordBatch {
        &self.indices
    }

    /// Encode the references and shape mapping for an independent process.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut ipc = Vec::new();
        let mut writer = StreamWriter::try_new(&mut ipc, self.indices.schema().as_ref())?;
        writer.write(&self.indices)?;
        writer.finish()?;
        Ok(pb::NestedReferenceBatch {
            protocol_version: VERSION,
            references: self.references.encode()?.into(),
            indices_ipc: ipc.into(),
        }
        .encode_to_vec())
    }

    /// Decode and validate both the shape mapping and every portable locator.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let proto = pb::NestedReferenceBatch::decode(bytes).map_err(|error| {
            Error::invalid_input(format!("Invalid nested blob reference protobuf: {error}"))
        })?;
        check_version(proto.protocol_version, "nested reference batch")?;
        let references = BlobRefBatch::decode(&proto.references)?;
        let mut reader = StreamReader::try_new(Cursor::new(proto.indices_ipc), None)?;
        let indices = reader
            .next()
            .transpose()?
            .ok_or_else(|| Error::invalid_input("Missing nested blob index Arrow batch"))?;
        if reader.next().transpose()?.is_some() {
            return Err(Error::invalid_input(
                "Nested blob indices require exactly one Arrow batch",
            ));
        }
        Self::try_new(references, indices)
    }
}

fn validate_indices(array: &ArrayRef, count: usize) -> Result<()> {
    match array.data_type() {
        DataType::UInt64 => {
            for index in array.as_primitive::<UInt64Type>().iter().flatten() {
                if index >= count as u64 {
                    return Err(Error::invalid_input(format!(
                        "Nested blob index {index} exceeds reference count {count}"
                    )));
                }
            }
        }
        DataType::Struct(fields) if fields.len() == 1 => {
            validate_indices(array.as_struct().column(0), count)?;
        }
        DataType::List(_) => validate_indices(array.as_list::<i32>().values(), count)?,
        DataType::LargeList(_) => validate_indices(array.as_list::<i64>().values(), count)?,
        data_type => {
            return Err(Error::invalid_input(format!(
                "Unsupported nested blob index layout {data_type}; expected uint64 leaves below single-child structs and List/LargeList"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, UInt64Array};
    use arrow_schema::{Field, Schema};
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::unknown_index(Arc::new(UInt64Array::from(vec![Some(u64::MAX)])), "exceeds reference count")]
    #[case::unsupported_layout(Arc::new(Int32Array::from(vec![0])), "Unsupported nested blob index layout")]
    fn nested_mapping_rejects_invalid_indices(#[case] array: ArrayRef, #[case] message: &str) {
        let references = BlobRefBatch::try_new(vec![], vec![None]).unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "indices",
                array.data_type().clone(),
                true,
            )])),
            vec![array],
        )
        .unwrap();
        let error = NestedBlobRefBatch::try_new(references, batch).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message));
    }

    #[test]
    fn nested_mapping_rejects_unknown_version_and_multiple_columns() {
        let references = BlobRefBatch::try_new(vec![], vec![None]).unwrap();
        let field = Field::new("indices", DataType::UInt64, true);
        let array: ArrayRef = Arc::new(UInt64Array::from(vec![None]));
        let indices = RecordBatch::try_new(
            Arc::new(Schema::new(vec![field.clone()])),
            vec![array.clone()],
        )
        .unwrap();
        let nested = NestedBlobRefBatch::try_new(references.clone(), indices).unwrap();
        let mut envelope =
            pb::NestedReferenceBatch::decode(nested.encode().unwrap().as_slice()).unwrap();
        envelope.protocol_version = 99;
        let error = NestedBlobRefBatch::decode(&envelope.encode_to_vec()).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("protocol version 99"));
        let indices = RecordBatch::try_new(
            Arc::new(Schema::new(vec![field.clone(), field.with_name("extra")])),
            vec![array.clone(), array],
        )
        .unwrap();
        let error = NestedBlobRefBatch::try_new(references, indices).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("exactly one column"));
    }
}
