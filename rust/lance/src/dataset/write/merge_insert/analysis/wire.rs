// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::super::{
    MergeInsertAnalysis, MergeInsertJoin, MergeInsertJoinStrategy, MergeInsertWriteStrategy,
};
use crate::pb;
use lance_datafusion::exec::{ExecutionAnalysis, ExecutionMetricValue, ExecutionOutcome};
use prost::Message;

impl MergeInsertAnalysis {
    /// Encode the runtime report as the `MergeAnalysisReport` protobuf wire message.
    ///
    /// This representation is for binding transport, not persisted dataset metadata.
    /// All integer measurements retain full precision. Optional fields preserve availability.
    ///
    /// ```
    /// # use lance::dataset::MergeInsertAnalysis;
    /// # fn example(report: &MergeInsertAnalysis) {
    /// let bytes = report.to_bytes();
    /// assert!(!bytes.is_empty());
    /// # }
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_proto().encode_to_vec()
    }

    /// Convert to the runtime protobuf schema, preserving optional-field presence.
    ///
    /// ```
    /// # use lance::dataset::MergeInsertAnalysis;
    /// # fn example(report: &MergeInsertAnalysis) {
    /// assert_eq!(report.to_proto().operation_id, report.operation_id.to_string());
    /// # }
    /// ```
    pub fn to_proto(&self) -> pb::MergeAnalysisReport {
        let operator = &self.operator_spills;
        let replay = &self.replay_spills;
        pb::MergeAnalysisReport {
            operation_id: self.operation_id.to_string(),
            execution: Some(execution(&self.execution)),
            write_mode: match self.write_mode {
                MergeInsertWriteStrategy::RewriteRows => "rewrite_rows",
                MergeInsertWriteStrategy::RewriteColumns => "rewrite_columns",
                MergeInsertWriteStrategy::DeleteOnly => "delete_only",
            }
            .to_owned(),
            stats: self.stats.as_ref().map(|stats| pb::MergeChangeStats {
                num_inserted_rows: stats.num_inserted_rows,
                num_updated_rows: stats.num_updated_rows,
                num_deleted_rows: stats.num_deleted_rows,
                bytes_written: stats.bytes_written,
                num_files_written: stats.num_files_written,
                num_skipped_duplicates: stats.num_skipped_duplicates,
            }),
            fragments_touched: self.fragments_touched.map(|v| v as u64),
            target_rows_read: self.target_rows_read.map(|v| v as u64),
            fragments_read: self.fragments_read.map(|v| v as u64),
            index_used: self.index_used,
            join_index_segments: self.join_index_segments.map(|v| v as u64),
            join_duration_ns: self.join_duration.map(|v| v.as_nanos() as u64),
            merge_write_duration_ns: self.merge_write_duration.map(|v| v.as_nanos() as u64),
            joined_rows: self.joined_rows.map(|v| v as u64),
            mapped_target_rows: self.mapped_target_rows.map(|v| v as u64),
            operator_spills: Some(pb::MergeSpillUsage {
                spill_count: Some(operator.spill_count as u64),
                bytes_written: Some(operator.bytes_written as u64),
                rows: Some(operator.rows as u64),
                bytes_read: operator.bytes_read.map(|v| v as u64),
                write_duration_ns: operator.write_duration.map(|v| v.as_nanos() as u64),
                read_duration_ns: operator.read_duration.map(|v| v.as_nanos() as u64),
                ..Default::default()
            }),
            replay_spills: Some(pb::MergeSpillUsage {
                spill_count: Some(replay.spill_count),
                bytes_written: Some(replay.bytes_written),
                bytes_read: Some(replay.bytes_read),
                write_duration_ns: Some(replay.write_duration.as_nanos() as u64),
                read_duration_ns: Some(replay.read_duration.as_nanos() as u64),
                ..Default::default()
            }),
            scratch_spills: self
                .scratch_spills
                .as_ref()
                .map(|scratch| pb::MergeSpillUsage {
                    bytes_written: Some(scratch.bytes_written),
                    bytes_read: Some(scratch.bytes_read),
                    write_duration_ns: Some(scratch.write_duration.as_nanos() as u64),
                    read_duration_ns: Some(scratch.read_duration.as_nanos() as u64),
                    accepted_bytes: Some(scratch.accepted_bytes),
                    charged_bytes: Some(scratch.charged_bytes),
                    peak_charged_bytes: Some(scratch.peak_charged_bytes),
                    live_files: Some(scratch.live_files),
                    files_created: Some(scratch.files_created),
                    quota_failures: Some(scratch.quota_failures),
                    cleanup_failures: Some(scratch.cleanup_failures),
                    ..Default::default()
                }),
            subplans: self
                .subplans
                .iter()
                .map(|plan| pb::MergeAnalysisSubplan {
                    id: plan.id as u64,
                    purpose: plan.purpose.clone(),
                    execution: Some(execution(&plan.execution)),
                    joins: plan.joins.iter().map(join).collect(),
                })
                .collect(),
            joins: self.joins.iter().map(join).collect(),
        }
    }
}

fn join(join: &MergeInsertJoin) -> pb::MergePhysicalJoin {
    let (strategy, partition_mode) = match join.strategy {
        MergeInsertJoinStrategy::Hash(mode) => ("hash", Some(format!("{mode:?}"))),
        MergeInsertJoinStrategy::SortMerge => ("sort_merge", None),
    };
    pb::MergePhysicalJoin {
        path: join.path.iter().map(|v| *v as u64).collect(),
        strategy: strategy.to_owned(),
        partition_mode,
        join_type: format!("{:?}", join.join_type),
        subplan_id: join.subplan_id as u64,
    }
}

fn metric(name: &str, kind: &str, value: u64) -> pb::MergeMetric {
    pb::MergeMetric {
        name: name.to_owned(),
        kind: kind.to_owned(),
        value,
        partition: None,
        labels: Vec::new(),
    }
}

fn execution(execution: &ExecutionAnalysis) -> pb::MergeExecutionReport {
    let summary = &execution.summary;
    let mut counts: Vec<_> = summary
        .all_counts
        .iter()
        .map(|(name, value)| metric(name, "count", *value as u64))
        .collect();
    counts.extend(
        [
            ("iops", summary.iops),
            ("requests", summary.requests),
            ("bytes_read", summary.bytes_read),
            ("indices_loaded", summary.indices_loaded),
            ("parts_loaded", summary.parts_loaded),
            ("index_comparisons", summary.index_comparisons),
        ]
        .into_iter()
        .map(|(name, value)| metric(name, "count", value as u64)),
    );
    counts.sort_by(|a, b| a.name.cmp(&b.name));
    let mut times: Vec<_> = summary
        .all_times
        .iter()
        .map(|(name, value)| metric(name, "time_ns", *value as u64))
        .collect();
    times.sort_by(|a, b| a.name.cmp(&b.name));
    pb::MergeExecutionReport {
        plan: execution.plan.clone(),
        outcome: match execution.outcome {
            ExecutionOutcome::Complete => "complete",
            ExecutionOutcome::Failed => "failed",
            ExecutionOutcome::Cancelled => "cancelled",
        }
        .to_owned(),
        error: execution.error.clone(),
        elapsed_ns: execution.elapsed.as_nanos() as u64,
        pool_peak_reserved_bytes: execution.pool_peak_reserved_bytes.map(|v| v as u64),
        counts,
        times,
        operators: execution
            .operators
            .iter()
            .map(|node| pb::MergeOperatorMetrics {
                path: node.path.iter().map(|v| *v as u64).collect(),
                name: node.name.clone(),
                metrics: node
                    .metrics
                    .iter()
                    .map(|value| {
                        let (kind, number) = match value.value {
                            ExecutionMetricValue::Count(v) => ("count", v as u64),
                            ExecutionMetricValue::Gauge(v) => ("gauge", v as u64),
                            ExecutionMetricValue::PeakMemoryBytes(v) => {
                                ("peak_memory_bytes", v as u64)
                            }
                            ExecutionMetricValue::Time(v) => ("time_ns", v.as_nanos() as u64),
                        };
                        pb::MergeMetric {
                            partition: value.partition.map(|v| v as u64),
                            labels: value
                                .labels
                                .iter()
                                .map(|(name, value)| pb::MergeMetricLabel {
                                    name: name.clone(),
                                    value: value.clone(),
                                })
                                .collect(),
                            ..metric(&value.name, kind, number)
                        }
                    })
                    .collect(),
            })
            .collect(),
    }
}
