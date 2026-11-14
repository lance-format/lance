/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.merge;

import org.lance.proto.MergeAnalysisWire;

import com.google.protobuf.InvalidProtocolBufferException;

import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.stream.Collectors;

/**
 * Immutable protobuf-backed snapshot of an uncommitted streaming merge.
 *
 * <p>Counts come from Rust producers. Missing readings remain absent; zero means measured zero.
 * Join and write timings overlap. Pool peaks count DataFusion reservations, not process RSS. Failed
 * reports are partial; outstanding I/O and cleanup can continue beyond their snapshot.
 *
 * <p>For example, {@code report.getReplaySpills().getBytesWritten()} measures replay traffic,
 * separately from {@code report.getOperatorSpills().getBytesWritten()} and data-file bytes.
 */
public final class MergeInsertAnalysis {
  private final MergeAnalysisWire.MergeAnalysisReport report;

  /** Decode the runtime wire format returned by JNI, without retaining native resources. */
  public MergeInsertAnalysis(byte[] encoded) {
    try {
      report = MergeAnalysisWire.MergeAnalysisReport.parseFrom(encoded);
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException("Invalid merge analysis protobuf", e);
    }
  }

  /** Physical writer selected by Rust, distinct from the caller's auto rewrite policy. */
  public enum WriteMode {
    /** Write whole rows, replacing matched rows or appending unmatched inserts. */
    REWRITE_ROWS,
    /** Patch columns into existing fragments, including mixed updates and inserts. */
    REWRITE_COLUMNS,
    DELETE_ONLY
  }

  /** Whether the output stream completed, failed, or was dropped before exhaustion. */
  public enum Outcome {
    COMPLETE,
    FAILED,
    CANCELLED
  }

  /** Physical join algorithm; partition mode is available separately for hash joins. */
  public enum JoinStrategy {
    HASH,
    SORT_MERGE
  }

  /** Identifier also attached to execution tracing, never to exported metric labels. */
  public String getOperationId() {
    return report.getOperationId();
  }

  /** Root execution diagnostics; indexed execution details live in subplans. */
  public Execution getExecution() {
    return new Execution(report.getExecution());
  }

  /** Actual write behavior, including deletion-only writes. */
  public WriteMode getWriteMode() {
    return WriteMode.valueOf(report.getWriteMode().toUpperCase(Locale.ROOT));
  }

  /** Complete logical changes, absent after an execution failure. */
  public Optional<ChangeStats> getStats() {
    return report.hasStats() ? Optional.of(new ChangeStats(report.getStats())) : Optional.empty();
  }

  /** Existing fragments modified or removed, excluding newly inserted fragments. */
  public OptionalLong getFragmentsTouched() {
    return optional(report.hasFragmentsTouched(), report.getFragmentsTouched());
  }

  /** Logical target read visits, excluding deleted rows; repeated reads count again. */
  public OptionalLong getTargetRowsRead() {
    return optional(report.hasTargetRowsRead(), report.getTargetRowsRead());
  }

  /** Scanner fragment visits; repeated visits are not deduplicated. */
  public OptionalLong getFragmentsRead() {
    return optional(report.hasFragmentsRead(), report.getFragmentsRead());
  }

  /** True only when an index probe ran, regardless of unrelated dataset indices. */
  public Optional<Boolean> getIndexUsed() {
    return report.hasIndexUsed() ? Optional.of(report.getIndexUsed()) : Optional.empty();
  }

  /** Distinct physical segments selected for index opening, including cache-served opens. */
  public OptionalLong getJoinIndexSegments() {
    return optional(report.hasJoinIndexSegments(), report.getJoinIndexSegments());
  }

  /** Cumulative actual join-node time in nanoseconds; overlaps inclusive write time. */
  public OptionalLong getJoinDurationNs() {
    return optional(report.hasJoinDurationNs(), report.getJoinDurationNs());
  }

  /** Inclusive sink/orchestration duration, overlapping join time, in nanoseconds. */
  public OptionalLong getMergeWriteDurationNs() {
    return optional(report.hasMergeWriteDurationNs(), report.getMergeWriteDurationNs());
  }

  /** Rows emitted by join operators before merge actions, including repeated matches. */
  public OptionalLong getJoinedRows() {
    return optional(report.hasJoinedRows(), report.getJoinedRows());
  }

  /** Distinct mapped target addresses, an upper bound on matching target rows. */
  public OptionalLong getMappedTargetRows() {
    return optional(report.hasMappedTargetRows(), report.getMappedTargetRows());
  }

  /** Operator traffic across root and subplans, counted once per physical producer. */
  public SpillUsage getOperatorSpills() {
    return new SpillUsage(report.getOperatorSpills());
  }

  /** Replay traffic, including IPC metadata and repeated reads. */
  public SpillUsage getReplaySpills() {
    return new SpillUsage(report.getReplaySpills());
  }

  /** Absent when this route has no diagnostic scratch-store owner. */
  public Optional<SpillUsage> getScratchSpills() {
    return report.hasScratchSpills()
        ? Optional.of(new SpillUsage(report.getScratchSpills()))
        : Optional.empty();
  }

  /** Separately executed physical plans, each with its own operator path namespace. */
  public List<Subplan> getSubplans() {
    return report.getSubplansList().stream().map(Subplan::new).collect(Collectors.toList());
  }

  /** Root joins, or indexed-route joins carrying their subplan ids. */
  public List<Join> getJoins() {
    return report.getJoinsList().stream().map(Join::new).collect(Collectors.toList());
  }

  /** Copy of the opaque, version-compatible runtime wire message. */
  public byte[] toBytes() {
    return report.toByteArray();
  }

  private static OptionalLong optional(boolean present, long value) {
    return present ? OptionalLong.of(value) : OptionalLong.empty();
  }

  /** Execution measurements with producer identity and units retained. */
  public static final class Execution {
    private final MergeAnalysisWire.MergeExecutionReport value;

    private Execution(MergeAnalysisWire.MergeExecutionReport value) {
      this.value = value;
    }

    public String getPlan() {
      return value.getPlan();
    }

    public Outcome getOutcome() {
      return Outcome.valueOf(value.getOutcome().toUpperCase(Locale.ROOT));
    }

    public Optional<String> getError() {
      return value.hasError() ? Optional.of(value.getError()) : Optional.empty();
    }

    public long getElapsedNs() {
      return value.getElapsedNs();
    }

    public OptionalLong getPoolPeakReservedBytes() {
      return optional(value.hasPoolPeakReservedBytes(), value.getPoolPeakReservedBytes());
    }

    public List<Metric> getCounts() {
      return value.getCountsList().stream().map(Metric::new).collect(Collectors.toList());
    }

    public List<Metric> getTimes() {
      return value.getTimesList().stream().map(Metric::new).collect(Collectors.toList());
    }

    public List<Operator> getOperators() {
      return value.getOperatorsList().stream().map(Operator::new).collect(Collectors.toList());
    }
  }

  /** An operation-unique subplan id scopes all child-index operator paths. */
  public static final class Subplan {
    private final MergeAnalysisWire.MergeAnalysisSubplan value;

    private Subplan(MergeAnalysisWire.MergeAnalysisSubplan value) {
      this.value = value;
    }

    public long getId() {
      return value.getId();
    }

    public String getPurpose() {
      return value.getPurpose();
    }

    public Execution getExecution() {
      return new Execution(value.getExecution());
    }

    public List<Join> getJoins() {
      return value.getJoinsList().stream().map(Join::new).collect(Collectors.toList());
    }
  }

  /** Physical join choice after optimization, without plan-text parsing. */
  public static final class Join {
    private final MergeAnalysisWire.MergePhysicalJoin value;

    private Join(MergeAnalysisWire.MergePhysicalJoin value) {
      this.value = value;
    }

    public List<Long> getPath() {
      return value.getPathList();
    }

    public long getSubplanId() {
      return value.getSubplanId();
    }

    public JoinStrategy getStrategy() {
      return JoinStrategy.valueOf(value.getStrategy().toUpperCase(Locale.ROOT));
    }

    public Optional<String> getPartitionMode() {
      return value.hasPartitionMode() ? Optional.of(value.getPartitionMode()) : Optional.empty();
    }

    public String getJoinType() {
      return value.getJoinType();
    }
  }

  /** Single producer measurement; time_ns and peak_memory_bytes carry explicit units. */
  public static final class Metric {
    private final MergeAnalysisWire.MergeMetric value;

    private Metric(MergeAnalysisWire.MergeMetric value) {
      this.value = value;
    }

    public String getName() {
      return value.getName();
    }

    public String getKind() {
      return value.getKind();
    }

    public long getValue() {
      return value.getValue();
    }

    public OptionalLong getPartition() {
      return optional(value.hasPartition(), value.getPartition());
    }

    public List<Label> getLabels() {
      return value.getLabelsList().stream().map(Label::new).collect(Collectors.toList());
    }
  }

  /** One producer label. A list preserves ordering and repeated names. */
  public static final class Label {
    private final MergeAnalysisWire.MergeMetricLabel value;

    private Label(MergeAnalysisWire.MergeMetricLabel value) {
      this.value = value;
    }

    public String getName() {
      return value.getName();
    }

    public String getValue() {
      return value.getValue();
    }
  }

  /** Operator identity and unaggregated measurements, including memory peaks. */
  public static final class Operator {
    private final MergeAnalysisWire.MergeOperatorMetrics value;

    private Operator(MergeAnalysisWire.MergeOperatorMetrics value) {
      this.value = value;
    }

    public List<Long> getPath() {
      return value.getPathList();
    }

    public String getName() {
      return value.getName();
    }

    public List<Metric> getMetrics() {
      return value.getMetricsList().stream().map(Metric::new).collect(Collectors.toList());
    }
  }

  /** Complete logical changes and data-file bytes, excluding deletion/blob/spill files. */
  public static final class ChangeStats {
    private final MergeAnalysisWire.MergeChangeStats value;

    private ChangeStats(MergeAnalysisWire.MergeChangeStats value) {
      this.value = value;
    }

    public long getNumInsertedRows() {
      return value.getNumInsertedRows();
    }

    public long getNumUpdatedRows() {
      return value.getNumUpdatedRows();
    }

    public long getNumDeletedRows() {
      return value.getNumDeletedRows();
    }

    public long getBytesWritten() {
      return value.getBytesWritten();
    }

    public long getNumFilesWritten() {
      return value.getNumFilesWritten();
    }

    public long getNumSkippedDuplicates() {
      return value.getNumSkippedDuplicates();
    }
  }

  /** Producer-specific traffic and retained storage; unavailable capabilities stay absent. */
  public static final class SpillUsage {
    private final MergeAnalysisWire.MergeSpillUsage value;

    private SpillUsage(MergeAnalysisWire.MergeSpillUsage value) {
      this.value = value;
    }

    public OptionalLong getSpillCount() {
      return optional(value.hasSpillCount(), value.getSpillCount());
    }

    public OptionalLong getBytesWritten() {
      return optional(value.hasBytesWritten(), value.getBytesWritten());
    }

    public OptionalLong getBytesRead() {
      return optional(value.hasBytesRead(), value.getBytesRead());
    }

    public OptionalLong getWriteDurationNs() {
      return optional(value.hasWriteDurationNs(), value.getWriteDurationNs());
    }

    public OptionalLong getReadDurationNs() {
      return optional(value.hasReadDurationNs(), value.getReadDurationNs());
    }

    public OptionalLong getRows() {
      return optional(value.hasRows(), value.getRows());
    }

    public OptionalLong getAcceptedBytes() {
      return optional(value.hasAcceptedBytes(), value.getAcceptedBytes());
    }

    public OptionalLong getChargedBytes() {
      return optional(value.hasChargedBytes(), value.getChargedBytes());
    }

    public OptionalLong getPeakChargedBytes() {
      return optional(value.hasPeakChargedBytes(), value.getPeakChargedBytes());
    }

    public OptionalLong getLiveFiles() {
      return optional(value.hasLiveFiles(), value.getLiveFiles());
    }

    public OptionalLong getFilesCreated() {
      return optional(value.hasFilesCreated(), value.getFilesCreated());
    }

    public OptionalLong getQuotaFailures() {
      return optional(value.hasQuotaFailures(), value.getQuotaFailures());
    }

    public OptionalLong getCleanupFailures() {
      return optional(value.hasCleanupFailures(), value.getCleanupFailures());
    }
  }
}
