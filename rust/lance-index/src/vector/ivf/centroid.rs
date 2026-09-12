// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Persisted routing graphs over the ordered centroids of a current IVF model.

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array};
use arrow_ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow_schema::DataType;
use bytes::Bytes;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use roaring::RoaringBitmap;

use crate::vector::flat::storage::FlatFloatStorage;
use crate::vector::graph::VisitedGenerator;
use crate::vector::hnsw::builder::{HNSW_METADATA_KEY, HnswBuildParams, HnswQueryParams};
use crate::vector::hnsw::{HNSW, HnswMetadata};
use crate::vector::storage::VectorStore;
use crate::vector::v3::subindex::IvfSubIndex;

/// Schema metadata key pointing directly to the graph's one-based global buffer ID.
pub const CENTROID_HNSW_KEY: &str = "lance:ivf:centroid_hnsw";

/// Parse the positive decimal global buffer ID stored in schema metadata.
pub fn graph_buffer_id(value: &str) -> Result<u32> {
    if !value.bytes().all(|c| c.is_ascii_digit()) {
        return Err(corrupt(format!("invalid graph buffer ID {value:?}")));
    }
    value
        .parse::<u32>()
        .ok()
        .filter(|id| *id != 0)
        .ok_or_else(|| {
            corrupt(format!(
                "graph buffer ID must be in 1..=u32::MAX, got {value:?}"
            ))
        })
}

/// A loaded graph with the containing IVF model's centroid storage.
#[derive(Debug, DeepSizeOf)]
pub struct CentroidIndex {
    graph: HNSW,
    store: FlatFloatStorage,
    dimension: usize,
}

impl CentroidIndex {
    /// Serialize a graph built from the final Float32/L2 centroid model.
    ///
    /// The payload retains the ordinary HNSW batch schema and JSON metadata.
    pub fn build(
        centroids: FixedSizeListArray,
        metric: DistanceType,
        params: HnswBuildParams,
    ) -> Result<Bytes> {
        validate_centroids(&centroids, metric)?;
        let store = FlatFloatStorage::new(centroids, metric);
        let graph = HNSW::index_vectors(&store, params)?;
        let batch = graph.to_batch()?;
        let mut bytes = Vec::new();
        let options = IpcWriteOptions::try_new(8, false, arrow_ipc::MetadataVersion::V5)?;
        let mut writer =
            StreamWriter::try_new_with_options(&mut bytes, batch.schema_ref(), options)?;
        writer.write(&batch)?;
        writer.finish()?;
        Ok(bytes.into())
    }

    /// Decode an advertised graph, rejecting malformed framing and topology.
    pub fn load(bytes: Bytes, centroids: FixedSizeListArray, metric: DistanceType) -> Result<Self> {
        validate_centroids(&centroids, metric).map_err(|e| corrupt(e.to_string()))?;
        validate_stream(&bytes)?;
        let batch = lance_arrow::ipc::read_ipc_stream_single(&bytes)
            .map_err(|e| corrupt(format!("invalid IPC graph: {e}")))?;
        validate_graph(&batch, centroids.len())?;
        let graph = HNSW::load(batch).map_err(|e| corrupt(e.to_string()))?;
        Ok(Self {
            graph,
            dimension: centroids.value_length() as usize,
            store: FlatFloatStorage::new(centroids, metric),
        })
    }

    /// Carry a graph into a rewritten index. Preserve the bytes when ordered
    /// centroids match, otherwise rebuild with the stored construction parameters.
    /// Call from a blocking task because rebuilding uses the HNSW builder.
    pub fn reuse_or_rebuild(
        bytes: Bytes,
        previous: FixedSizeListArray,
        current: FixedSizeListArray,
        metric: DistanceType,
    ) -> Result<Bytes> {
        validate_centroids(&current, metric)?;
        let is_same_model = previous == current;
        let graph = Self::load(bytes.clone(), previous, metric)?;
        if is_same_model {
            Ok(bytes)
        } else {
            Self::build(current, metric, graph.graph.metadata().params)
        }
    }

    /// Search centroid IDs directly, without mapping them to dataset row IDs.
    pub fn search(
        &self,
        query: ArrayRef,
        nprobes: usize,
        ef: usize,
    ) -> Result<(UInt32Array, Float32Array)> {
        if query.data_type() != &DataType::Float32
            || query.len() != self.dimension
            || query.null_count() != 0
        {
            return Err(Error::invalid_input(format!(
                "centroid HNSW query requires {} non-null Float32 values, got type={}, length={}, nulls={}",
                self.dimension,
                query.data_type(),
                query.len(),
                query.null_count()
            )));
        }
        if nprobes == 0 || nprobes > self.store.len() {
            return Err(Error::invalid_input(format!(
                "centroid HNSW nprobes must be in 1..={}, got {nprobes}",
                self.store.len()
            )));
        }
        if ef < nprobes {
            return Err(Error::invalid_input(format!(
                "centroid_ef must be >= maximum_nprobes, got centroid_ef={ef}, maximum_nprobes={nprobes}"
            )));
        }
        let params = HnswQueryParams {
            ef,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
            use_acorn: false,
        };
        // Scratch belongs to this query, so concurrent searches cannot grow the
        // cached graph beyond its accounted size.
        let mut visited = VisitedGenerator::new(self.store.len());
        let mut nodes = self.graph.search_inner(
            query,
            nprobes,
            &params,
            None,
            &mut visited,
            &self.store,
            Some(2),
        )?;
        nodes.sort_unstable_by(|a, b| a.dist.cmp(&b.dist).then(a.id.cmp(&b.id)));
        let (ids, distances): (Vec<_>, Vec<_>) =
            nodes.into_iter().map(|n| (n.id, n.dist.0)).unzip();
        Ok((UInt32Array::from(ids), Float32Array::from(distances)))
    }
}

fn corrupt(message: impl Into<String>) -> Error {
    Error::corrupt_file(
        "index.idx".into(),
        format!("centroid HNSW: {}", message.into()),
    )
}

/// Validate the model before opt-in construction, independently of graph size.
fn validate_centroids(centroids: &FixedSizeListArray, metric: DistanceType) -> Result<()> {
    if metric != DistanceType::L2 || centroids.value_type() != DataType::Float32 {
        return Err(Error::invalid_input(format!(
            "centroid_hnsw requires Float32/L2 centroids, got type={}, metric={metric}",
            centroids.value_type()
        )));
    }
    if centroids.is_empty() || centroids.value_length() <= 0 || centroids.len() > u32::MAX as usize
    {
        return Err(Error::invalid_input(format!(
            "centroid_hnsw requires 1..=u32::MAX centroids and positive dimension, got count={}, dimension={}",
            centroids.len(),
            centroids.value_length()
        )));
    }
    let values = centroids
        .values()
        .as_primitive::<arrow_array::types::Float32Type>();
    if centroids.null_count() != 0
        || values.null_count() != 0
        || values.values().iter().any(|v| !v.is_finite())
    {
        return Err(Error::invalid_input(
            "centroid_hnsw requires non-null finite centroids",
        ));
    }
    Ok(())
}

fn validate_stream(bytes: &[u8]) -> Result<()> {
    let mut offset = 0usize;
    let mut messages = 0;
    loop {
        let header = bytes
            .get(offset..offset.saturating_add(8))
            .ok_or_else(|| corrupt("truncated IPC header or missing EOS"))?;
        if header[..4] != [0xff; 4] {
            return Err(corrupt("IPC continuation marker is missing"));
        }
        let size =
            u32::from_le_bytes(header[4..8].try_into().map_err(|_| corrupt("IPC size"))?) as usize;
        offset += 8;
        if size == 0 {
            return if messages == 2 && offset == bytes.len() {
                Ok(())
            } else {
                Err(corrupt(
                    "expected schema, one batch, EOS and no trailing bytes",
                ))
            };
        }
        let end = offset
            .checked_add(size)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| corrupt("truncated IPC metadata"))?;
        let message = arrow_ipc::root_as_message(&bytes[offset..end])
            .map_err(|e| corrupt(format!("invalid IPC message: {e}")))?;
        let expected = match messages {
            0 => arrow_ipc::MessageHeader::Schema,
            1 => arrow_ipc::MessageHeader::RecordBatch,
            _ => return Err(corrupt("unexpected extra IPC message")),
        };
        if message.version() != arrow_ipc::MetadataVersion::V5 || message.header_type() != expected
        {
            return Err(corrupt(
                "expected IPC V5 schema followed by one record batch",
            ));
        }
        if let Some(schema) = message.header_as_schema()
            && schema.endianness() != arrow_ipc::Endianness::Little
        {
            return Err(corrupt("expected little-endian IPC schema"));
        }
        if let Some(batch) = message.header_as_record_batch()
            && batch.compression().is_some()
        {
            return Err(corrupt("compressed IPC graph is not supported"));
        }
        let body =
            usize::try_from(message.bodyLength()).map_err(|_| corrupt("negative IPC body size"))?;
        offset = end
            .checked_add(body)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| corrupt("truncated IPC body"))?;
        messages += 1;
    }
}

fn validate_graph(batch: &RecordBatch, count: usize) -> Result<()> {
    if batch.schema().fields() != HNSW::schema().fields() {
        return Err(corrupt("graph must use the existing HNSW schema"));
    }
    let metadata = batch
        .schema_ref()
        .metadata()
        .get(HNSW_METADATA_KEY)
        .ok_or_else(|| corrupt("missing lance:hnsw metadata"))?;
    let metadata: HnswMetadata = serde_json::from_str(metadata)
        .map_err(|e| corrupt(format!("invalid HNSW metadata: {e}")))?;
    metadata
        .params
        .validate()
        .map_err(|e| corrupt(e.to_string()))?;
    let offsets = &metadata.level_offsets;
    if metadata.params.max_level == 0
        || offsets.len() != metadata.params.max_level as usize + 1
        || offsets.first() != Some(&0)
        || offsets.last() != Some(&batch.num_rows())
        || offsets.windows(2).any(|w| w[0] > w[1])
        || offsets.get(1) != Some(&count)
    {
        return Err(corrupt("invalid level offsets or centroid count"));
    }
    let ids = batch
        .column_by_name("__vector_id")
        .ok_or_else(|| corrupt("missing node IDs"))?
        .as_primitive::<arrow_array::types::UInt32Type>();
    let neighbors = batch
        .column_by_name("__neighbors")
        .ok_or_else(|| corrupt("missing neighbors"))?
        .as_list::<i32>();
    let distances = batch
        .column_by_name("_distance")
        .ok_or_else(|| corrupt("missing distances"))?
        .as_list::<i32>();
    if ids.null_count() != 0
        || neighbors.null_count() != 0
        || neighbors.values().null_count() != 0
        || distances.null_count() != 0
        || distances.values().null_count() != 0
    {
        return Err(corrupt("null graph data"));
    }
    let mut previous = RoaringBitmap::new();
    for (level, range) in offsets.windows(2).enumerate() {
        let mut members = RoaringBitmap::new();
        for row in range[0]..range[1] {
            let id = ids.value(row);
            if id as usize >= count
                || !members.insert(id)
                || (level == 0 && id as usize != row)
                || (level > 0 && !previous.contains(id))
            {
                return Err(corrupt(format!("invalid node {id} at level {level}")));
            }
        }
        if !members.is_empty() && !members.contains(metadata.entry_point) {
            return Err(corrupt(format!("entry point missing at level {level}")));
        }
        for row in range[0]..range[1] {
            let edges = neighbors.value(row);
            let edges = edges.as_primitive::<arrow_array::types::UInt32Type>();
            let mut seen = RoaringBitmap::new();
            for &id in edges.values() {
                if !members.contains(id) || id == ids.value(row) || !seen.insert(id) {
                    return Err(corrupt(format!(
                        "invalid edge {id} at level {level}, row {row}"
                    )));
                }
            }
            if edges.len() != distances.value_length(row) as usize {
                return Err(corrupt("neighbor and distance counts differ"));
            }
        }
        previous = members;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_arrow::FixedSizeListArrayExt;
    use rstest::rstest;
    use std::sync::Arc;

    fn centroids() -> FixedSizeListArray {
        FixedSizeListArray::try_new_from_values(
            Float32Array::from((0..32).map(|i| i as f32).collect::<Vec<_>>()),
            1,
        )
        .unwrap()
    }

    #[test]
    fn roundtrip_preserves_centroid_ids_and_recall() {
        let centroids = centroids();
        let bytes = CentroidIndex::build(
            centroids.clone(),
            DistanceType::L2,
            HnswBuildParams::default(),
        )
        .unwrap();
        let graph = CentroidIndex::load(bytes, centroids, DistanceType::L2).unwrap();
        let size = graph.deep_size_of();
        let (ids, distances) = graph
            .search(Arc::new(Float32Array::from(vec![12.1])), 3, 32)
            .unwrap();
        let recall = ids
            .values()
            .iter()
            .filter(|id| [11, 12, 13].contains(id))
            .count() as f32
            / 3.0;
        assert!(recall >= 0.99, "recall={recall}");
        assert_eq!(ids.value(0), 12);
        assert!((distances.value(0) - 0.01).abs() < 1e-5);
        assert_eq!(graph.deep_size_of(), size);
    }

    #[rstest]
    #[case::missing_eos(false)]
    #[case::trailing_bytes(true)]
    fn rejects_invalid_framing(#[case] append: bool) {
        let centroids = centroids();
        let mut bytes = CentroidIndex::build(
            centroids.clone(),
            DistanceType::L2,
            HnswBuildParams::default(),
        )
        .unwrap()
        .to_vec();
        if append {
            bytes.push(0);
        } else {
            bytes.truncate(bytes.len() - 8);
        }
        let error = CentroidIndex::load(bytes.into(), centroids, DistanceType::L2).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("centroid HNSW"));
    }

    #[test]
    fn rejects_wrong_centroid_count() {
        let bytes = CentroidIndex::build(centroids(), DistanceType::L2, HnswBuildParams::default())
            .unwrap();
        let error =
            CentroidIndex::load(bytes, centroids().slice(0, 31), DistanceType::L2).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("centroid count"));
    }

    #[test]
    fn rewrite_reuses_or_rebuilds_for_the_final_model() {
        let previous = centroids();
        let bytes = CentroidIndex::build(
            previous.clone(),
            DistanceType::L2,
            HnswBuildParams::default(),
        )
        .unwrap();
        let reused = CentroidIndex::reuse_or_rebuild(
            bytes.clone(),
            previous.clone(),
            previous.clone(),
            DistanceType::L2,
        )
        .unwrap();
        assert_eq!(reused, bytes);
        let current = FixedSizeListArray::try_new_from_values(
            Float32Array::from((0..33).map(|i| i as f32).collect::<Vec<_>>()),
            1,
        )
        .unwrap();
        let rewritten =
            CentroidIndex::reuse_or_rebuild(bytes, previous, current.clone(), DistanceType::L2)
                .unwrap();
        let graph = CentroidIndex::load(rewritten, current, DistanceType::L2).unwrap();
        let (ids, _) = graph
            .search(Arc::new(Float32Array::from(vec![32.0])), 1, 16)
            .unwrap();
        assert_eq!(ids.value(0), 32);
    }

    #[test]
    fn rejects_invalid_entry_point() {
        let centroids = centroids();
        let bytes = CentroidIndex::build(
            centroids.clone(),
            DistanceType::L2,
            HnswBuildParams::default(),
        )
        .unwrap();
        let batch = lance_arrow::ipc::read_ipc_stream_single(&bytes).unwrap();
        let mut metadata = batch.schema().metadata().clone();
        let mut hnsw: HnswMetadata = serde_json::from_str(&metadata[HNSW_METADATA_KEY]).unwrap();
        hnsw.entry_point = centroids.len() as u32;
        metadata.insert(
            HNSW_METADATA_KEY.into(),
            serde_json::to_string(&hnsw).unwrap(),
        );
        let schema = Arc::new(batch.schema().as_ref().clone().with_metadata(metadata));
        let batch = RecordBatch::try_new(schema, batch.columns().to_vec()).unwrap();
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, batch.schema_ref()).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
        let error = CentroidIndex::load(bytes.into(), centroids, DistanceType::L2).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("entry point"));
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::all_null(vec![None])]
    #[case::null_value(vec![Some(1.0), None])]
    #[case::nan(vec![Some(f32::NAN)])]
    fn rejects_invalid_centroids(#[case] values: Vec<Option<f32>>) {
        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), 1).unwrap();
        let error = CentroidIndex::build(centroids, DistanceType::L2, HnswBuildParams::default())
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("centroid_hnsw requires"));
    }

    #[rstest]
    #[case::zero("0")]
    #[case::signed("+1")]
    #[case::empty("")]
    #[case::overflow("4294967296")]
    fn rejects_invalid_buffer_ids(#[case] value: &str) {
        let error = graph_buffer_id(value).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("buffer ID"));
    }
}
