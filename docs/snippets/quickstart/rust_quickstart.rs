use std::sync::Arc;

use arrow_array::{
    FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let uri = "/tmp/lance-rust-quickstart.lance";

    // 1. Define an Arrow schema with scalar and vector columns
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("price", DataType::Float32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(item_field.clone(), 4),
            false,
        ),
    ]));

    let vectors = FixedSizeListArray::try_new(
        item_field.clone(),
        4,
        Arc::new(Float32Array::from(vec![
            1.0, 0.0, 0.0, 0.25, // row 1
            0.875, 0.125, 0.0, 0.375, // row 2
            0.0, 1.0, 0.25, 0.0, // row 3
        ])),
        None,
    )?;

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![
                "red running shoes",
                "trail running shoes",
                "rain jacket",
            ])),
            Arc::new(Float32Array::from(vec![89.0, 120.0, 99.0])),
            Arc::new(vectors),
        ],
    )?;

    // 2. Write a new Lance dataset
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Overwrite,
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, uri, Some(write_params)).await?;
    println!("Created dataset at version {}", dataset.version().version);

    // 3. Filter and project columns with SQL predicates
    let filtered: Vec<RecordBatch> = dataset
        .scan()
        .filter("price < 100.0")?
        .project(&["id", "name", "price"])?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    println!(
        "Filtered rows: {}",
        filtered.iter().map(|b| b.num_rows()).sum::<usize>()
    );

    // 4. Run nearest-neighbor vector search
    let query = Float32Array::from(vec![0.875, 0.125, 0.0, 0.25]);
    let nearest: Vec<RecordBatch> = dataset
        .scan()
        .project(&["id", "name", "_distance"])?
        .nearest("vector", &query, 2)?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    println!(
        "Nearest matches: {}",
        nearest.iter().map(|b| b.num_rows()).sum::<usize>()
    );

    // 5. Append rows and time travel to an earlier version
    let new_vectors = FixedSizeListArray::try_new(
        item_field,
        4,
        Arc::new(Float32Array::from(vec![0.75, 0.25, 0.25, 0.25])),
        None,
    )?;
    let append_batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4])),
            Arc::new(StringArray::from(vec!["trail socks"])),
            Arc::new(Float32Array::from(vec![15.0])),
            Arc::new(new_vectors),
        ],
    )?;
    dataset
        .append(
            RecordBatchIterator::new(vec![Ok(append_batch)], schema),
            None,
        )
        .await?;
    println!(
        "Latest version: {} ({} rows)",
        dataset.version().version,
        dataset.count_rows(None).await?
    );

    let v1 = dataset.checkout_version(1_u64).await?;
    println!("Version 1 rows: {}", v1.count_rows(None).await?);

    Ok(())
}
