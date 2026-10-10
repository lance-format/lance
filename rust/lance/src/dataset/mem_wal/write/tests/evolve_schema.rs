// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Moving a live writer onto a schema change, and the replay that has to
//! rebuild the same generation boundaries afterwards.

/// Replay starts each memtable where the WAL entry it reads says to.
mod replay {
    use super::super::*;

    /// Replay starts a new memtable where the recorded generation changes, even
    /// when no size limit or Blob v2 target would.
    #[tokio::test]
    async fn test_replay_rotates_on_a_recorded_generation_alone() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let config = ShardWriterConfig {
            shard_id: Uuid::new_v4(),
            durable_write: true,
            max_wal_buffer_size: 1,
            max_wal_flush_interval: Some(Duration::from_millis(10)),
            max_memtable_rows: 100,
            ..Default::default()
        };
        let before = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let after = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("note", DataType::Utf8, true),
        ]));
        let row = |schema: &Arc<ArrowSchema>, id: i32| {
            let mut columns: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(vec![id]))];
            if schema.fields().len() > 1 {
                columns.push(Arc::new(StringArray::from(vec![None::<&str>])));
            }
            RecordBatch::try_new(schema.clone(), columns).unwrap()
        };

        let live_generation;
        {
            let writer = ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                config.clone(),
                before.clone(),
                Vec::new(),
            )
            .await
            .unwrap();
            writer.put(vec![row(&before, 0)]).await.unwrap();
            writer
                .evolve_schema(after.clone(), Vec::new())
                .await
                .unwrap();
            writer.put(vec![row(&after, 1)]).await.unwrap();
            live_generation = writer.active_memtable_ref().await.unwrap().generation;
            // Dropped without a flush, so only the WAL holds the rows.
            std::mem::forget(writer);
        }

        let reopened = ShardWriter::open(store, base_path, base_uri, config, after, Vec::new())
            .await
            .unwrap();
        assert_eq!(
            reopened.active_memtable_ref().await.unwrap().generation,
            live_generation,
            "replay rebuilds the generation the writer recorded"
        );
        assert_eq!(
            reopened.manifest().await.unwrap().unwrap().sstables.len(),
            1,
            "the generation it left must have been flushed, not folded into this one"
        );
        reopened.close().await.unwrap();
    }

    async fn manifest_generations(
        store: &Arc<ObjectStore>,
        base_path: &Path,
        shard_id: Uuid,
    ) -> Vec<u64> {
        ShardManifestStore::new(store.clone(), base_path, shard_id, 10)
            .latest()
            .await
            .unwrap()
            .map(|m| m.sstables.iter().map(|s| s.generation).collect())
            .unwrap_or_default()
    }

    /// Rows that replay widened past the byte limit are not treated as a full
    /// memtable.
    #[tokio::test]
    async fn test_replay_does_not_refuse_rows_a_schema_change_widened() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let shard_id = Uuid::new_v4();
        let narrow = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        // Only the widened rows cross this byte limit.
        let config = ShardWriterConfig {
            max_memtable_size: 128,
            ..memtable_config_with_pk(shard_id)
        };

        {
            let writer = ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                config.clone(),
                narrow.clone(),
                vec![],
            )
            .await
            .unwrap();
            for id in 0..8i64 {
                let batch = RecordBatch::try_new(
                    narrow.clone(),
                    vec![Arc::new(arrow_array::Int64Array::from(vec![id]))],
                )
                .unwrap();
                writer.put(vec![batch]).await.unwrap();
            }
        }

        // A column added since, so replay fills it with nulls.
        let widened = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 64),
                true,
            ),
        ]));
        let reopened = ShardWriter::open(store, base_path, base_uri, config, widened, vec![])
            .await
            .expect("a widened replay must not be read as a full memtable");
        reopened.close().await.unwrap();
    }

    /// Replay refuses a recorded generation too big for the memtable, publishes
    /// nothing, and succeeds again at the original size.
    #[tokio::test]
    async fn test_replay_refuses_a_recorded_generation_it_cannot_hold() {
        let (store, base_path, base_uri, _temp_dir) = create_local_store().await;
        let schema = schema_with_pk();
        let shard_id = Uuid::new_v4();
        // Large enough that all rows stay in one generation.
        let config = ShardWriterConfig {
            max_memtable_rows: 10_000,
            ..memtable_config_with_pk(shard_id)
        };

        {
            let writer = ShardWriter::open(
                store.clone(),
                base_path.clone(),
                base_uri.clone(),
                config.clone(),
                schema.clone(),
                vec![],
            )
            .await
            .unwrap();
            for id in 0..4i32 {
                writer
                    .put(vec![create_test_batch(&schema, id, 1)])
                    .await
                    .unwrap();
            }
            // Dropped without a close, so only the WAL survives.
        }
        assert!(
            manifest_generations(&store, &base_path, shard_id)
                .await
                .is_empty(),
            "the writer must have flushed nothing for this to be about replay"
        );

        // Room for two of the generation's four rows.
        let err = ShardWriter::open(
            store.clone(),
            base_path.clone(),
            base_uri.clone(),
            ShardWriterConfig {
                max_memtable_rows: 2,
                ..config.clone()
            },
            schema.clone(),
            vec![],
        )
        .await
        .err()
        .expect("a recorded generation the memtable cannot hold must be refused");
        assert!(
            err.to_string()
                .contains("past what the memtable replaying it can hold"),
            "unexpected error: {err}"
        );
        assert!(
            manifest_generations(&store, &base_path, shard_id)
                .await
                .is_empty(),
            "the refusal must come before replay publishes a generation"
        );

        let reopened = ShardWriter::open(store, base_path, base_uri, config, schema, vec![])
            .await
            .expect("the shard is unchanged and reopens at its original cap");
        assert_eq!(
            reopened.memtable_stats().await.unwrap().row_count,
            4,
            "every row the WAL held must be back in the memtable"
        );
        reopened.close().await.unwrap();
    }
}

/// A live writer moved onto a new schema keeps writing and reading.
mod evolve {
    use super::super::*;
    use crate::dataset::WriteParams;
    use crate::dataset::mem_wal::DatasetMemWalExt;
    use crate::dataset::mem_wal::write::shard_writer_tests::{
        create_test_batch, create_test_schema,
    };
    use crate::index::DatasetIndexExt;
    use arrow_array::{Int64Array, RecordBatchIterator};
    use lance_index::IndexType;
    use lance_index::scalar::FullTextSearchQuery;
    use lance_index::scalar::inverted::InvertedIndexParams;

    /// A writer on an `id`/`vector`/`text` table with an FTS index on `text`.
    /// Frozen memtables stay in memory for an hour after flush. `maintained`
    /// lists the maintained indexes; `None` maintains all of them.
    async fn evolving_writer(
        name: &str,
        maintained: Option<&[&str]>,
    ) -> (Dataset, ShardWriter, Uuid, Arc<ArrowSchema>) {
        let vector_dim = 4;
        let schema = create_test_schema(vector_dim);
        // The test reopens the flushed generation by URI.
        let uri = format!("shared-memory://{name}-{}/", Uuid::new_v4().simple());
        let initial = create_test_batch(&schema, 0, 10, vector_dim);
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(initial)], schema.clone()),
            &uri,
            Some(WriteParams::default()),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some("text_fts".to_string()),
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        let init = dataset.initialize_mem_wal();
        match maintained {
            Some(names) => init.maintained_indexes(names.iter().copied()),
            None => init,
        }
        .execute()
        .await
        .unwrap();
        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(
                shard_id,
                ShardWriterConfig::new(shard_id)
                    .with_max_wal_flush_interval(std::time::Duration::from_millis(10))
                    .with_frozen_memtable_grace(std::time::Duration::from_secs(3600)),
            )
            .await
            .unwrap();
        (dataset, writer, shard_id, schema)
    }

    /// A batch of `(id, text)` rows under `schema`, whatever its column names.
    fn rows_under(schema: &ArrowSchema, rows: &[(i64, &str)]) -> RecordBatch {
        use arrow_array::ArrayRef;
        let vector_dim = 4;
        let columns: Vec<ArrayRef> = schema
            .fields()
            .iter()
            .map(|field| -> ArrayRef {
                match field.data_type() {
                    DataType::Int64 if field.name() == "id" => {
                        Arc::new(Int64Array::from_iter_values(rows.iter().map(|(id, _)| *id)))
                    }
                    DataType::Int64 => Arc::new(Int64Array::from_iter_values(
                        rows.iter().map(|(id, _)| id * 10),
                    )),
                    DataType::FixedSizeList(..) => Arc::new(
                        FixedSizeListArray::try_new_from_values(
                            Float32Array::from_iter_values(
                                rows.iter().flat_map(|(id, _)| vec![*id as f32; vector_dim]),
                            ),
                            vector_dim as i32,
                        )
                        .unwrap(),
                    ),
                    _ => Arc::new(StringArray::from_iter_values(
                        rows.iter().map(|(_, text)| *text),
                    )),
                }
            })
            .collect();
        RecordBatch::try_new(Arc::new(schema.clone()), columns).unwrap()
    }

    /// The schema a writer is evolved to: the dataset's, carrying field ids.
    fn writer_schema_of(dataset: &Dataset) -> Arc<ArrowSchema> {
        Arc::new(crate::dataset::mem_wal::arrow_schema_with_field_ids(
            dataset.schema(),
        ))
    }

    /// The dataset's FTS index as a writer index spec.
    async fn fts_config_of(dataset: &Dataset) -> Vec<crate::dataset::mem_wal::index::MemIndexSpec> {
        use crate::dataset::mem_wal::index::{FtsMemIndexPlugin, MemIndexSpec};
        let meta = dataset
            .load_indices_by_name("text_fts")
            .await
            .unwrap()
            .remove(0);
        let resolved =
            FtsMemIndexPlugin::resolve_from_metadata("text_fts", dataset.schema(), &meta).unwrap();
        vec![MemIndexSpec {
            name: "text_fts".to_string(),
            field_ids: resolved.field_ids.unwrap(),
            columns: resolved.columns,
            plugin: Arc::new(FtsMemIndexPlugin),
            params: resolved.params,
            index_details: meta.index_details,
        }]
    }

    #[tokio::test]
    async fn test_evolve_schema_seals_under_the_old_schema_and_writes_under_the_new() {
        use crate::dataset::ColumnAlteration;

        let (mut dataset, writer, shard_id, schema) =
            evolving_writer("evolve-seal", Some(&["text_fts"])).await;
        writer
            .put(vec![rows_under(&schema, &[(100, "before")])])
            .await
            .unwrap();

        dataset
            .alter_columns(&[ColumnAlteration::new("text".into()).rename("body".into())])
            .await
            .unwrap();
        let renamed: ArrowSchema = dataset.schema().into();
        writer
            .put(vec![rows_under(&renamed, &[(101, "early")])])
            .await
            .expect_err("a batch shaped to a schema the writer has not been evolved to");

        let sealed_generation = writer.active_memtable_ref().await.unwrap().generation;
        writer
            .evolve_schema(writer_schema_of(&dataset), fts_config_of(&dataset).await)
            .await
            .unwrap();
        let memtables = writer.in_memory_memtable_refs().await.unwrap();
        assert_eq!(memtables.active.generation, sealed_generation + 1);
        assert_eq!(memtables.frozen.len(), 1, "the old memtable is sealed");
        assert!(
            memtables.frozen[0]
                .schema
                .column_with_name("text")
                .is_some()
        );
        assert!(memtables.active.schema.column_with_name("body").is_some());

        writer
            .put(vec![rows_under(&renamed, &[(101, "after")])])
            .await
            .unwrap();
        writer
            .put(vec![rows_under(&schema, &[(102, "stale")])])
            .await
            .expect_err("a batch shaped to the schema the writer has moved off");

        // Same schema again: nothing is sealed.
        writer
            .evolve_schema(writer_schema_of(&dataset), fts_config_of(&dataset).await)
            .await
            .unwrap();
        let memtables = writer.in_memory_memtable_refs().await.unwrap();
        assert_eq!(memtables.active.generation, sealed_generation + 1);
        assert_eq!(memtables.frozen.len(), 1);

        // The sealed memtable flushes under its old schema and index.
        writer.wait_for_flush_drain().await.unwrap();
        let manifest = writer.manifest().await.unwrap().unwrap();
        let sstable = manifest
            .sstables
            .iter()
            .find(|s| s.generation == sealed_generation)
            .expect("the sealed generation is in the manifest");
        let generation = Dataset::open(&format!(
            "{}/_mem_wal/{shard_id}/{}",
            dataset.uri().trim_end_matches('/'),
            sstable.path
        ))
        .await
        .unwrap();
        assert!(generation.schema().field("text").is_some());
        assert!(generation.schema().field("body").is_none());
        let indices = generation.load_indices().await.unwrap();
        assert!(
            indices.iter().any(|index| index.name == "text_fts"),
            "the generation carries the index its memtable was built with: {:?}",
            indices.iter().map(|index| &index.name).collect::<Vec<_>>()
        );
        writer.close().await.unwrap();
    }

    /// A generation flushed after a column drop keeps the table's field ids,
    /// whether or not the table has migrated to non-reusable ids.
    ///
    /// The drop leaves a gap in the table's ids. Reads and the merge into base
    /// resolve a generation's columns by id, so numbering the generation's
    /// fields afresh would hand one column's values to another.
    #[rstest::rstest]
    #[tokio::test]
    async fn test_a_generation_flushed_after_a_drop_keeps_the_table_field_ids(
        #[values(false, true)] non_reusable_ids: bool,
    ) {
        let (mut dataset, writer, shard_id, _) =
            evolving_writer("evolve-drop", Some(&["text_fts"])).await;
        if non_reusable_ids {
            dataset.migrate_to_non_reusable_field_ids().await.unwrap();
        }
        dataset.drop_columns(&["vector"]).await.unwrap();
        writer.evolve_to(&dataset).await.unwrap();
        let dropped: ArrowSchema = dataset.schema().into();
        writer
            .put(vec![rows_under(&dropped, &[(100, "after the drop")])])
            .await
            .unwrap();
        let generation = writer.active_memtable_ref().await.unwrap().generation;
        writer.force_seal_active().await.unwrap();
        writer.wait_for_flush_drain().await.unwrap();

        let manifest = writer.manifest().await.unwrap().unwrap();
        let sstable = manifest
            .sstables
            .iter()
            .find(|s| s.generation == generation)
            .expect("the sealed generation is flushed");
        let flushed = Dataset::open(&format!(
            "{}/_mem_wal/{shard_id}/{}",
            dataset.uri().trim_end_matches('/'),
            sstable.path
        ))
        .await
        .unwrap();
        let ids = |schema: &crate::datatypes::Schema| -> Vec<(String, i32)> {
            schema
                .fields_pre_order()
                .map(|field| (field.name.clone(), field.id))
                .collect()
        };
        let table_ids = ids(dataset.schema());
        assert!(
            table_ids.iter().map(|(_, id)| *id).collect::<Vec<_>>()
                != (0..table_ids.len() as i32).collect::<Vec<_>>(),
            "the drop leaves a gap in the table's ids: {table_ids:?}"
        );
        let generation_ids: Vec<_> = ids(flushed.schema())
            .into_iter()
            .filter(|(name, _)| table_ids.iter().any(|(table, _)| table == name))
            .collect();
        assert_eq!(generation_ids, table_ids);
        // The data files carry the same ids, so readers find each column.
        let file_ids: Vec<i32> = flushed.get_fragments()[0].metadata().files[0]
            .fields
            .to_vec();
        for (name, id) in &table_ids {
            assert!(
                file_ids.contains(id),
                "{name} (id {id}) not in {file_ids:?}"
            );
        }
        assert!(
            !flushed.manifest.uses_non_reusable_field_ids(),
            "the generation takes the table's ids without allocating its own"
        );

        // Read the row back from the flushed generation alone: by scan, by key,
        // and through the generation's FTS index.
        use crate::dataset::mem_wal::scanner::{LsmScanner, ShardSnapshot};
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int64Type;
        use datafusion::prelude::{col, lit};
        let mut snapshot =
            ShardSnapshot::new(shard_id).with_current_generation(manifest.current_generation);
        for sstable in &manifest.sstables {
            snapshot = snapshot.with_sstable(sstable.generation, sstable.path.clone());
        }
        let scanner = || {
            LsmScanner::without_base_table(
                Arc::new(dropped.clone()),
                dataset.uri(),
                vec![snapshot.clone()],
                vec!["id".to_string()],
            )
            .with_identity_schema(writer_schema_of(&dataset))
        };
        let rows = |batch: RecordBatch| -> Vec<(i64, String)> {
            let ids = batch["id"].as_primitive::<Int64Type>();
            let texts = batch["text"].as_string::<i32>();
            (0..batch.num_rows())
                .map(|i| (ids.value(i), texts.value(i).to_string()))
                .collect()
        };
        let expected = vec![(100, "after the drop".to_string())];
        assert_eq!(rows(scanner().try_into_batch().await.unwrap()), expected);
        let by_key = scanner()
            .filter_expr(col("id").eq(lit(100i64)))
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(rows(by_key), expected);
        let by_text = scanner()
            .full_text_search(
                FullTextSearchQuery::new("drop".to_string())
                    .with_column("text".to_string())
                    .unwrap(),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            by_text["id"].as_primitive::<Int64Type>().values().to_vec(),
            vec![100]
        );
        writer.close().await.unwrap();
    }

    /// When the table maintains every index (an empty name list), a schema
    /// change keeps them all.
    #[tokio::test]
    async fn test_evolve_to_keeps_every_index_when_the_table_maintains_all() {
        use crate::dataset::ColumnAlteration;

        let (mut dataset, writer, _, _) = evolving_writer("evolve-all", None).await;
        assert!(
            dataset
                .mem_wal_index_details()
                .await
                .unwrap()
                .expect("initialized")
                .maintained_indexes
                .is_empty(),
            "maintaining every index carries no names"
        );
        assert_eq!(
            writer.maintained_index_names().await,
            vec!["text_fts".to_string()]
        );

        dataset
            .alter_columns(&[ColumnAlteration::new("text".into()).rename("body".into())])
            .await
            .unwrap();
        writer.evolve_to(&dataset).await.unwrap();
        assert_eq!(
            writer.maintained_index_names().await,
            vec!["text_fts".to_string()],
            "the index the table maintains survives the schema the writer moved onto"
        );
        writer.close().await.unwrap();
    }

    /// `replace_index_configs` keeps the writer's evolved schema.
    #[tokio::test]
    async fn test_replace_index_configs_keeps_the_schema_the_writer_holds() {
        use crate::dataset::ColumnAlteration;

        let (mut dataset, writer, _, schema) =
            evolving_writer("evolve-keep-schema", Some(&["text_fts"])).await;
        dataset
            .alter_columns(&[ColumnAlteration::new("text".into()).rename("body".into())])
            .await
            .unwrap();
        writer.evolve_to(&dataset).await.unwrap();
        let renamed: ArrowSchema = dataset.schema().into();

        writer.replace_index_configs(Vec::new()).await.unwrap();
        assert!(writer.maintained_index_names().await.is_empty());

        writer
            .put(vec![rows_under(&renamed, &[(200, "after")])])
            .await
            .expect("the evolved schema survives an index-set change");
        writer
            .put(vec![rows_under(&schema, &[(201, "stale")])])
            .await
            .expect_err("the schema the writer moved off is still refused");
        writer.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_evolve_schema_replaces_an_empty_memtable_and_keeps_the_key() {
        use crate::dataset::ColumnAlteration;

        let (mut dataset, writer, _, _) =
            evolving_writer("evolve-empty", Some(&["text_fts"])).await;
        let generation = writer.active_memtable_ref().await.unwrap().generation;
        dataset
            .alter_columns(&[ColumnAlteration::new("text".into()).rename("body".into())])
            .await
            .unwrap();
        writer.evolve_to(&dataset).await.unwrap();
        let memtables = writer.in_memory_memtable_refs().await.unwrap();
        assert_eq!(
            memtables.active.generation, generation,
            "nothing was sealed"
        );
        assert!(memtables.frozen.is_empty());
        assert!(memtables.active.schema.column_with_name("body").is_some());
        let renamed: ArrowSchema = dataset.schema().into();
        writer
            .put(vec![rows_under(&renamed, &[(100, "after")])])
            .await
            .unwrap();

        // Drop the primary-key metadata from `id`.
        let keyless = ArrowSchema::new(
            writer_schema_of(&dataset)
                .fields()
                .iter()
                .map(|f| {
                    let mut metadata = f.metadata().clone();
                    metadata.remove("lance-schema:unenforced-primary-key");
                    f.as_ref().clone().with_metadata(metadata)
                })
                .collect::<Vec<_>>(),
        );
        let err = writer
            .evolve_schema(Arc::new(keyless), Vec::new())
            .await
            .expect_err("the primary key cannot change under a live writer");
        assert!(
            err.to_string().contains("cannot change the primary key"),
            "unexpected error: {err}"
        );
        writer.close().await.unwrap();
    }

    /// A memtable written before a rename answers a filter on the new name from
    /// its indexes on the old one, keeping each key's newest version.
    #[tokio::test]
    async fn test_memtable_filter_indexes_answer_across_a_rename() {
        use crate::dataset::ColumnAlteration;
        use crate::dataset::mem_wal::scanner::LsmScanner;
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int64Type;
        use datafusion::prelude::{col, lit};
        use lance_index::scalar::ScalarIndexParams;

        use crate::dataset::mem_wal::memtable::scanner::newest_checks;

        let schema = create_test_schema(4);
        let uri = format!("shared-memory://evolve-filter-{}/", Uuid::new_v4().simple());
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(create_test_batch(&schema, 0, 10, 4))], schema.clone()),
            &uri,
            Some(WriteParams::default()),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["text"],
                IndexType::BTree,
                Some("text_btree".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        dataset
            .initialize_mem_wal()
            .maintained_indexes(["text_btree"])
            .execute()
            .await
            .unwrap();
        let shard_id = Uuid::new_v4();
        let writer = dataset
            .mem_wal_writer(
                shard_id,
                ShardWriterConfig::new(shard_id)
                    .with_max_wal_flush_interval(std::time::Duration::from_millis(10))
                    .with_frozen_memtable_grace(std::time::Duration::from_secs(3600)),
            )
            .await
            .unwrap();
        // Filler rows, sorting below every filter, give each memtable a match
        // budget above zero, so a selective filter is read from the indexes.
        let filler = |first: i64| -> Vec<(i64, String)> {
            (first..first + 64)
                .map(|id| (id, format!("aaa{id}")))
                .collect()
        };
        fn as_rows(rows: &[(i64, String)]) -> Vec<(i64, &str)> {
            rows.iter().map(|(id, text)| (*id, text.as_str())).collect()
        }
        let (old_filler, new_filler) = (filler(1000), filler(2000));
        // Key 102 is rewritten under the old name, key 103 under the new one.
        writer
            .put(vec![rows_under(
                &schema,
                &[(101, "alpha"), (102, "beta"), (103, "gamma")],
            )])
            .await
            .unwrap();
        writer
            .put(vec![rows_under(&schema, &as_rows(&old_filler))])
            .await
            .unwrap();
        writer
            .put(vec![rows_under(&schema, &[(102, "beta again")])])
            .await
            .unwrap();
        dataset
            .alter_columns(&[ColumnAlteration::new("text".into()).rename("body".into())])
            .await
            .unwrap();
        writer.evolve_to(&dataset).await.unwrap();
        let evolved: ArrowSchema = dataset.schema().into();
        writer
            .put(vec![rows_under(&evolved, &[(103, "gamma again")])])
            .await
            .unwrap();
        writer
            .put(vec![rows_under(&evolved, &as_rows(&new_filler))])
            .await
            .unwrap();

        let memtables = writer.in_memory_memtable_refs().await.unwrap();
        assert_eq!(
            memtables.frozen.len(),
            1,
            "the pre-rename memtable is in memory"
        );
        for (filter, expected) in [
            (col("body").eq(lit("beta")), vec![]),
            (col("body").eq(lit("beta again")), vec![102]),
            (col("body").eq(lit("gamma")), vec![]),
            (col("body").eq(lit("gamma again")), vec![103]),
            (col("body").gt_eq(lit("alpha")), vec![101, 102, 103]),
        ] {
            for enabled in [false, true] {
                let scanner = LsmScanner::without_base_table(
                    Arc::new(evolved.clone()),
                    dataset.uri(),
                    vec![],
                    vec!["id".to_string()],
                )
                .with_identity_schema(writer_schema_of(&dataset))
                .with_in_memory_memtables(shard_id, memtables.clone())
                .with_memtable_filter_indexes(enabled)
                .filter_expr(filter.clone());
                let plan = scanner.create_plan().await.unwrap();
                let batches = datafusion::physical_plan::collect(
                    plan.clone(),
                    datafusion::prelude::SessionContext::new().task_ctx(),
                )
                .await
                .unwrap();
                let mut ids: Vec<i64> = batches
                    .iter()
                    .flat_map(|batch| batch["id"].as_primitive::<Int64Type>().values().to_vec())
                    .collect();
                ids.sort_unstable();
                assert_eq!(ids, expected, "{filter} with filter indexes {enabled}");
                assert_eq!(
                    newest_checks(&plan) > 0,
                    enabled,
                    "{filter}: the indexes' matches are checked only when enabled"
                );
            }
        }
        writer.close().await.unwrap();
    }

    /// Rows in a memtable written before a rename and an added column read back
    /// under the new names, with the new column null, on every read path.
    #[tokio::test]
    async fn test_reads_resolve_a_memtable_written_before_a_schema_change() {
        use crate::dataset::mem_wal::scanner::LsmScanner;
        use crate::dataset::{ColumnAlteration, NewColumnTransform};
        use arrow_array::Array;
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int64Type;
        use datafusion::prelude::{col, lit};

        let (mut dataset, writer, shard_id, schema) =
            evolving_writer("evolve-read", Some(&["text_fts"])).await;
        writer
            .put(vec![rows_under(
                &schema,
                &[(1, "alpha one"), (2, "beta two"), (3, "gamma three")],
            )])
            .await
            .unwrap();

        dataset
            .alter_columns(&[
                ColumnAlteration::new("text".into()).rename("body".into()),
                ColumnAlteration::new("vector".into()).rename("embedding".into()),
            ])
            .await
            .unwrap();
        dataset
            .add_columns(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![Field::new(
                    "extra",
                    DataType::Int64,
                    true,
                )]))),
                None,
                None,
            )
            .await
            .unwrap();
        writer.evolve_to(&dataset).await.unwrap();
        let evolved: ArrowSchema = dataset.schema().into();
        // `extra` is `id * 10` for rows written from here on.
        writer
            .put(vec![rows_under(
                &evolved,
                &[(2, "delta two"), (4, "alpha four")],
            )])
            .await
            .unwrap();

        let memtables = writer.in_memory_memtable_refs().await.unwrap();
        assert_eq!(
            memtables.frozen.len(),
            1,
            "the pre-change memtable is still in memory"
        );
        let scanner = || {
            LsmScanner::without_base_table(
                Arc::new(evolved.clone()),
                dataset.uri(),
                vec![],
                vec!["id".to_string()],
            )
            .with_identity_schema(writer_schema_of(&dataset))
            .with_in_memory_memtables(shard_id, memtables.clone())
        };
        let rows = |batch: RecordBatch| {
            let ids = batch["id"].as_primitive::<Int64Type>();
            let bodies = batch["body"].as_string::<i32>();
            let extras = batch["extra"].as_primitive::<Int64Type>();
            let mut rows: Vec<(i64, String, Option<i64>)> = (0..batch.num_rows())
                .map(|i| {
                    (
                        ids.value(i),
                        bodies.value(i).to_string(),
                        extras.is_valid(i).then(|| extras.value(i)),
                    )
                })
                .collect();
            rows.sort();
            rows
        };
        let row = |id: i64, body: &str, extra: Option<i64>| (id, body.to_string(), extra);

        // Scan: the newest version of each key wins.
        let all = scanner().try_into_batch().await.unwrap();
        assert_eq!(
            rows(all),
            vec![
                row(1, "alpha one", None),
                row(2, "delta two", Some(20)),
                row(3, "gamma three", None),
                row(4, "alpha four", Some(40)),
            ]
        );

        // Filter on a renamed column.
        let gamma = scanner()
            .filter_expr(col("body").eq(lit("gamma three")))
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(rows(gamma), vec![row(3, "gamma three", None)]);

        // Filter on the added column, which the old memtable lacks. It must
        // apply after the newest version of key 2 wins.
        let unset = scanner()
            .filter_expr(col("extra").is_null())
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            rows(unset),
            vec![row(1, "alpha one", None), row(3, "gamma three", None)]
        );

        // Point lookup on a key only in the old memtable, and one rewritten since.
        for (id, expected) in [
            (1, row(1, "alpha one", None)),
            (2, row(2, "delta two", Some(20))),
        ] {
            let found = scanner()
                .filter_expr(col("id").eq(lit(id)))
                .try_into_batch()
                .await
                .unwrap();
            assert_eq!(rows(found), vec![expected]);
        }

        // FTS on the renamed column uses the old memtable's index.
        let alpha = scanner()
            .full_text_search(
                FullTextSearchQuery::new("alpha".to_string())
                    .with_column("body".to_string())
                    .unwrap(),
            )
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let mut alpha_ids: Vec<i64> = alpha["id"].as_primitive::<Int64Type>().values().to_vec();
        alpha_ids.sort_unstable();
        assert_eq!(alpha_ids, vec![1, 4]);

        // Vector search with a filter the old memtable cannot answer.
        let nearest = scanner()
            .nearest("embedding", &Float32Array::from(vec![3.0f32; 4]), 1)
            .unwrap()
            .filter_expr(col("extra").is_null())
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(
            nearest["id"].as_primitive::<Int64Type>().values().to_vec(),
            vec![3],
            "the nearest row the predicate admits"
        );
        writer.close().await.unwrap();
    }
}
