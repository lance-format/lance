# Indexing a dataset with HNSW (Hierarchical Navigable Small World)

HNSW is a graph based algorithm for approximate neighbor search in high-dimensional spaces. In this example, we will demonstrate how to build an HNSW vector index against a Lance dataset.

This example will show how to:

1. Generate synthetic test data of specified dimensions
2. Build a hierarchical graph structure for efficient vector search using Lance API
3. Perform vector search with different parameters and compute the ground truth using L2 distance search

The example sets the search beam width `ef` explicitly. A query that leaves it
unset gets `1.5 * k * refine_factor`, floored at 256; the floor is what keeps
recall usable on a single large partition, which is the layout
`IVF_HNSW_*` builds by default.

## Complete Example

```rust
use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{types::Float32Type, Array, FixedSizeListArray};
use arrow::array::{AsArray, FixedSizeListBuilder, Float32Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::record_batch::RecordBatchIterator;
use arrow_select::concat::concat;
use futures::stream::StreamExt;
use lance::Dataset;
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_index::vector::{
    flat::storage::FlatFloatStorage,
    hnsw::{builder::HnswBuildParams, HNSW},
};
use lance_linalg::distance::DistanceType;

fn ground_truth(fsl: &FixedSizeListArray, query: &[f32], k: usize) -> HashSet<u32> {
    let mut dists = vec![];
    for i in 0..fsl.len() {
        let dist = lance_linalg::distance::l2_distance(
            query,
            fsl.value(i).as_primitive::<Float32Type>().values(),
        );
        dists.push((dist, i as u32));
    }
    dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    dists.truncate(k);
    dists.into_iter().map(|(_, i)| i).collect()
}

pub async fn create_test_vector_dataset(output: &str, num_rows: usize, dim: i32) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
        false,
    )]));

    let mut batches = Vec::new();

    // Create a few batches
    for _ in 0..2 {
        let v_builder = Float32Builder::new();
        let mut list_builder = FixedSizeListBuilder::new(v_builder, dim);

        for _ in 0..num_rows {
            for _ in 0..dim {
                list_builder.values().append_value(rand::random::<f32>());
            }
            list_builder.append(true);
        }
        let array = Arc::new(list_builder.finish());
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        batches.push(batch);
    }
    let batch_reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());
    println!("Writing dataset to {}", output);
    Dataset::write(batch_reader, output, None).await.unwrap();
}

#[tokio::main]
async fn main() {
    let uri: Option<String> = None; // None means generate test data
    let column = "vector";
    let ef = 100;
    let max_edges = 30;
    let max_level = 7;

    // 1. Generate a synthetic test data of specified dimensions
    let dataset = if uri.is_none() {
        println!("No uri is provided, generating test dataset...");
        let output = "test_vectors.lance";
        create_test_vector_dataset(output, 1000, 64).await;
        Dataset::open(output).await.expect("Failed to open dataset")
    } else {
        Dataset::open(uri.as_ref().unwrap())
            .await
            .expect("Failed to open dataset")
    };

    println!("Dataset schema: {:#?}", dataset.schema());
    let batches = dataset
        .scan()
        .project(&[column])
        .unwrap()
        .try_into_stream()
        .await
        .unwrap()
        .then(|batch| async move { batch.unwrap().column_by_name(column).unwrap().clone() })
        .collect::<Vec<_>>()
        .await;
    let arrs = batches.iter().map(|b| b.as_ref()).collect::<Vec<_>>();
    let fsl = concat(&arrs).unwrap().as_fixed_size_list().clone();
    println!("Loaded {:?} batches", fsl.len());

    let vector_store = Arc::new(FlatFloatStorage::new(fsl.clone(), DistanceType::L2));

    let q = fsl.value(0);
    let k = 10;
    let gt = ground_truth(&fsl, q.as_primitive::<Float32Type>().values(), k);

    for ef_construction in [15, 30, 50] {
        let now = std::time::Instant::now();
        // 2. Build a hierarchical graph structure for efficient vector search using Lance API
        let hnsw = HNSW::index_vectors(
            vector_store.as_ref(),
            HnswBuildParams::default()
                .max_level(max_level)
                .num_edges(max_edges)
                .ef_construction(ef_construction),
        )
        .unwrap();
        let construct_time = now.elapsed().as_secs_f32();
        let now = std::time::Instant::now();
        // 3. Perform vector search with different parameters and compute the ground truth using L2 distance search
        let results: HashSet<u32> = hnsw
            .search_basic(q.clone(), k, ef, None, vector_store.as_ref())
            .unwrap()
            .iter()
            .map(|node| node.id)
            .collect();
        let search_time = now.elapsed().as_micros();
        println!(
            "level={}, ef_construct={}, ef={} recall={}: construct={:.3}s search={:.3} us",
            max_level,
            ef_construction,
            ef,
            results.intersection(&gt).count() as f32 / k as f32,
            construct_time,
            search_time
        );
    }
}
```
