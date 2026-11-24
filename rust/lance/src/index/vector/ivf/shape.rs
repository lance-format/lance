// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;

use arrow_schema::DataType;
use bytes::Bytes;
use lance_arrow::DataTypeExt;
use lance_core::{Error, Result, cache::CacheKey};
use lance_file::reader::{BufferDescriptor, FileReader};
use lance_index::{
    INDEX_FILE_NAME, INDEX_METADATA_SCHEMA_KEY, IndexMetadata,
    metrics::NoOpMetricsCollector,
    pb,
    vector::ivf::storage::{IVF_METADATA_KEY, IvfModel},
};
use lance_io::{
    scheduler::{FileScheduler, ScanScheduler, SchedulerConfig},
    utils::CachedFileSize,
};
use prost::{
    Message,
    encoding::{DecodeContext, WireType, decode_key, decode_varint},
};
use uuid::Uuid;

use super::compatibility::VectorQueryContract;
use crate::{
    Dataset,
    index::{DatasetIndexExt, DatasetIndexInternalExt},
};

struct IvfShapeKey;

impl CacheKey for IvfShapeKey {
    type ValueType = Option<(usize, usize)>;

    fn key(&self) -> Cow<'_, str> {
        "ivf_shape".into()
    }

    fn type_name() -> &'static str {
        "IvfPartitionShape"
    }
}

impl Dataset {
    /// Returns the partition count and the centroid dimension of one query-visible
    /// vector index segment. The dimension is in the index's coordinate space.
    /// A model without centroids reports its partition count and a zero dimension.
    ///
    /// Current built-in IVF indices need only their index file. Other layouts
    /// obtain the shape from the query opener, including registered extensions.
    pub async fn ivf_partition_shape(&self, uuid: &Uuid) -> Result<(usize, usize)> {
        let indices = self.load_indices().await?;
        let segment = indices
            .iter()
            .find(|index| index.uuid == *uuid)
            .ok_or_else(|| Error::index_not_found(format!("uuid={uuid}")))?;
        let store = self.object_store_for_index(segment).await?;
        let path = self
            .indice_files_dir(segment)?
            .join(uuid.to_string())
            .join(INDEX_FILE_NAME);
        let cache = self
            .metadata_cache
            .file_metadata_cache(&path)
            .with_key_prefix(&store.store_prefix);
        let shape = cache
            .get_or_insert_with_key(IvfShapeKey, || async {
                let size = segment
                    .files
                    .iter()
                    .flatten()
                    .find(|file| file.path == INDEX_FILE_NAME)
                    .map(|file| CachedFileSize::new(file.size_bytes))
                    .unwrap_or_else(CachedFileSize::unknown);
                let scheduler =
                    ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
                let file = scheduler
                    .open_file(&path, &size)
                    .await?
                    .with_metadata_read_size(4096)?;
                let (schema, buffers) =
                    match FileReader::read_schema_and_global_buffers(&file).await {
                        Ok(result) => result,
                        Err(Error::VersionConflict {
                            major_version: 0,
                            minor_version: 0..=2,
                            ..
                        }) => return Ok(None),
                        Err(err) => return Err(err),
                    };
                let raw = schema
                    .metadata
                    .get(INDEX_METADATA_SCHEMA_KEY)
                    .ok_or_else(|| {
                        Error::index(format!("Missing {INDEX_METADATA_SCHEMA_KEY} in {path}"))
                    })?;
                let metadata: IndexMetadata = serde_json::from_str(raw)?;
                if VectorQueryContract::from_file_metadata(&metadata).is_err() {
                    return Ok(None);
                }
                let ivf_buffer = schema
                    .metadata
                    .get(IVF_METADATA_KEY)
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|&p| p > 0)
                    .and_then(|p| buffers.get(p))
                    .ok_or_else(|| {
                        Error::index(format!(
                            "Invalid {IVF_METADATA_KEY} buffer reference in {path}"
                        ))
                    })?;
                read_shape(&file, ivf_buffer).await.map(Some)
            })
            .await?;
        if let Some(shape) = *shape {
            return Ok(shape);
        }
        let field = segment
            .fields
            .first()
            .ok_or_else(|| Error::index(format!("Segment {uuid} has no indexed field")))?;
        let column = self.schema().field_path(*field)?;
        let index = self
            .open_vector_index_from_metadata(&column, segment, &NoOpMetricsCollector)
            .await?;
        Ok((
            index.ivf_model().num_partitions(),
            index.ivf_model().dimension(),
        ))
    }
}

// A current writer puts the two packed partition arrays first, followed by
// the Tensor and an optional fixed64 loss. Other protobuf field orders and
// representations are valid and go through the normal IVF decoder.
enum Probe {
    Shape(usize, usize),
    NeedPrefix(usize),
}

fn probe_shape(prefix: &[u8], loss_tail: &[u8], len: usize) -> Option<Probe> {
    let mut input = prefix;
    for tag in [2, 3] {
        let mut field = input;
        if decode_key(&mut field).ok()? != (tag, WireType::LengthDelimited) {
            continue;
        }
        let size = usize::try_from(decode_varint(&mut field).ok()?).ok()?;
        let end = (prefix.len() - field.len()).checked_add(size)?;
        // A canonical Tensor header (two uint32 dimensions and length prefixes)
        // fits in 64 bytes. Grow to the declared end of this array in one read.
        let needed = end.checked_add(64)?.min(len);
        if needed > prefix.len() {
            return Some(Probe::NeedPrefix(needed));
        }
        input = prefix.get(end..)?;
    }
    let arrays = &prefix[..prefix.len() - input.len()];
    if decode_key(&mut input).ok()? != (4, WireType::LengthDelimited) {
        return None;
    }
    let tensor_len = usize::try_from(decode_varint(&mut input).ok()?).ok()?;
    let tensor_end = (prefix.len() - input.len()).checked_add(tensor_len)?;
    match len.checked_sub(tensor_end)? {
        0 => {}
        // Verify that no second tensor can override or merge the first one.
        9 if loss_tail.first() == Some(&0x29) => {}
        _ => return None,
    }

    let mut tensor = pb::Tensor::default();
    for tag in [1, 2] {
        let (actual_tag, wire) = decode_key(&mut input).ok()?;
        if actual_tag != tag {
            return None;
        }
        tensor
            .merge_field(tag, wire, &mut input, DecodeContext::default())
            .ok()?;
    }
    if decode_key(&mut input).ok()? != (3, WireType::LengthDelimited) {
        return None;
    }
    let data_len = usize::try_from(decode_varint(&mut input).ok()?).ok()?;
    let &[partitions, dimension] = tensor.shape.as_slice() else {
        return None;
    };
    let dimension = i32::try_from(dimension).ok()? as usize;
    let partitions = partitions as usize;
    let dtype = pb::tensor::DataType::try_from(tensor.data_type).ok()?;
    if dtype == pb::tensor::DataType::Bfloat16 || dimension == 0 {
        return None;
    }
    let expected = partitions
        .checked_mul(dimension)?
        .checked_mul(DataType::from(dtype).byte_width())?;
    if expected != data_len || (prefix.len() - input.len()).checked_add(data_len)? != tensor_end {
        return None;
    }
    // Decode and validate the partition arrays exactly as the model reader does.
    IvfModel::try_from(pb::Ivf::decode(arrays).ok()?).ok()?;
    Some(Probe::Shape(partitions, dimension))
}

async fn read_shape(file: &FileScheduler, buffer: &BufferDescriptor) -> Result<(usize, usize)> {
    // GBO bounds (including overflow) have been checked against the file.
    let len = usize::try_from(buffer.size)
        .map_err(|_| Error::index("IVF buffer exceeds address space"))?;
    if len == 0 {
        // An empty IVF message encodes an empty partition table.
        return Ok((0, 0));
    }
    let start = buffer.position;
    let end = start + buffer.size;
    let initial = len.min(4096);
    let (mut prefix, tail) = if len <= initial + 9 {
        let bytes = file.submit_single(start..end, 0).await?;
        let tail = bytes.slice(len.saturating_sub(9)..);
        (bytes, tail)
    } else {
        // Keep the tail and header reads concurrent without coalescing the
        // intervening centroid values on stores with a large block size.
        futures::try_join!(
            file.submit_single(start..start + initial as u64, 0),
            file.submit_single(end - 9..end, 0)
        )?
    };
    loop {
        match probe_shape(&prefix, &tail, len) {
            Some(Probe::Shape(partitions, dimension)) => return Ok((partitions, dimension)),
            Some(Probe::NeedPrefix(size)) if size < len => {
                let rest = file
                    .submit_single(start + prefix.len() as u64..start + size as u64, 0)
                    .await?;
                let mut extended = Vec::with_capacity(size);
                extended.extend_from_slice(&prefix);
                extended.extend_from_slice(&rest);
                prefix = Bytes::from(extended);
            }
            _ => break,
        }
    }

    // No centroids, deprecated float arrays, or an alternate protobuf encoding.
    // The existing model reader owns their semantics and validation.
    let bytes = if prefix.len() == len {
        prefix
    } else {
        file.submit_single(start..end, 0).await?
    };
    let proto = pb::Ivf::decode(bytes)?;
    if let Some(tensor) = &proto.centroids_tensor
        && !pb::tensor::DataType::try_from(tensor.data_type)
            .is_ok_and(|t| t != pb::tensor::DataType::Bfloat16)
    {
        return Err(Error::index(format!(
            "Unsupported IVF centroid data type {}",
            tensor.data_type
        )));
    }
    let model = IvfModel::try_from(proto)?;
    Ok((model.num_partitions(), model.dimension()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_io::{object_store::ObjectStore, scheduler::IoStats};
    use object_store::path::Path;
    use rstest::rstest;
    use std::sync::Arc;

    async fn shape_from_bytes(bytes: &[u8]) -> (Result<(usize, usize)>, u64) {
        let store = Arc::new(ObjectStore::memory());
        let path = Path::from("shape.idx");
        // A global buffer is not necessarily at offset zero or at EOF.
        let mut file_bytes = vec![42; 17];
        file_bytes.extend_from_slice(bytes);
        file_bytes.extend_from_slice(&[42; 11]);
        store.put(&path, &file_bytes).await.unwrap();
        let scheduler = ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
        let io = IoStats::new();
        let file = scheduler
            .open_file(&path, &CachedFileSize::new(file_bytes.len() as u64))
            .await
            .unwrap()
            .with_io_stats(io.recorder());
        let shape = read_shape(
            &file,
            &BufferDescriptor {
                position: 17,
                size: bytes.len() as u64,
            },
        )
        .await;
        (shape, io.snapshot().bytes_read)
    }

    #[rstest]
    #[case::small_metadata(64, None)]
    #[case::long_partition_arrays(8192, Some(0.5))]
    #[tokio::test]
    async fn ivf_partition_shape_reads_headers(
        #[case] partitions: usize,
        #[case] loss: Option<f64>,
    ) {
        let dimension = 64;
        let proto = pb::Ivf {
            offsets: (0..partitions as u64).collect(),
            lengths: vec![1; partitions],
            centroids_tensor: Some(pb::Tensor {
                data_type: pb::tensor::DataType::Float32 as i32,
                shape: vec![partitions as u32, dimension as u32],
                data: vec![0; partitions * dimension * 4],
            }),
            loss,
            ..Default::default()
        };
        let (actual, bytes_read) = shape_from_bytes(&proto.encode_to_vec()).await;
        assert_eq!(actual.unwrap(), (partitions, dimension));
        assert!(
            bytes_read < (partitions * 3 + 8192) as u64,
            "read {bytes_read} bytes"
        );
    }

    #[tokio::test]
    async fn ivf_partition_shape_preserves_other_protobuf_layouts_and_errors() {
        assert_eq!(shape_from_bytes(&[]).await.0.unwrap(), (0, 0));
        let partitions = pb::Ivf {
            lengths: vec![1, 2, 3],
            ..Default::default()
        };
        let (shape, _) = shape_from_bytes(&partitions.encode_to_vec()).await;
        assert_eq!(shape.unwrap(), (3, 0));

        let flat = pb::Ivf {
            centroids: vec![0.0; 12],
            ..partitions.clone()
        };
        let (shape, _) = shape_from_bytes(&flat.encode_to_vec()).await;
        assert_eq!(shape.unwrap(), (3, 4));

        let tensor = pb::Tensor {
            data_type: pb::tensor::DataType::Uint8 as i32,
            shape: vec![3, 4],
            data: vec![0; 12],
        };
        let mut reordered = pb::Ivf {
            centroids_tensor: Some(tensor.clone()),
            ..Default::default()
        }
        .encode_to_vec();
        reordered.extend(partitions.encode_to_vec());
        let (shape, _) = shape_from_bytes(&reordered).await;
        assert_eq!(shape.unwrap(), (3, 4));

        let mut corrupt = pb::Ivf {
            centroids_tensor: Some(tensor),
            ..partitions
        };
        corrupt.offsets = vec![0, 1];
        assert!(shape_from_bytes(&corrupt.encode_to_vec()).await.0.is_err());
        corrupt.offsets.clear();
        corrupt.centroids_tensor.as_mut().unwrap().data.pop();
        assert!(shape_from_bytes(&corrupt.encode_to_vec()).await.0.is_err());
        let mut truncated = corrupt.encode_to_vec();
        truncated.pop();
        assert!(shape_from_bytes(&truncated).await.0.is_err());
    }
}
