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
package org.lance.memwal;

import com.google.common.base.Preconditions;

import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Optional configuration for a {@link ShardWriter}.
 *
 * <p>Every value is optional; unset values fall back to the Lance default writer configuration. Use
 * the {@code with*} methods to build a configuration, then pass it to {@link
 * org.lance.Dataset#memWalWriter} or {@link InitializeMemWalParams#withWriterConfigDefaults}.
 */
public class ShardWriterConfig {
  private Optional<Boolean> durableWrite = Optional.empty();
  private Optional<Long> maxWalBufferSize = Optional.empty();
  private Optional<Long> maxWalFlushIntervalMs = Optional.empty();
  private Optional<Long> maxMemtableSize = Optional.empty();
  private Optional<Long> maxMemtableRows = Optional.empty();
  private Optional<Long> maxMemtableBatches = Optional.empty();
  private Optional<Long> maxUnflushedMemtableBytes = Optional.empty();
  private Optional<Long> manifestScanBatchSize = Optional.empty();
  private Optional<Long> backpressureLogIntervalMs = Optional.empty();
  private Optional<Long> statsLogIntervalMs = Optional.empty();
  private List<MemWalHnswParams> hnswParams = Collections.emptyList();

  /** Whether each {@code put} is durably persisted to the WAL before returning. */
  public ShardWriterConfig withDurableWrite(boolean durableWrite) {
    this.durableWrite = Optional.of(durableWrite);
    return this;
  }

  /** Maximum size of the in-memory WAL buffer, in bytes. */
  public ShardWriterConfig withMaxWalBufferSize(long maxWalBufferSize) {
    Preconditions.checkArgument(
        maxWalBufferSize >= 0, "maxWalBufferSize must not be negative, got %s", maxWalBufferSize);
    this.maxWalBufferSize = Optional.of(maxWalBufferSize);
    return this;
  }

  /** Maximum interval between WAL flushes, in milliseconds. */
  public ShardWriterConfig withMaxWalFlushIntervalMs(long maxWalFlushIntervalMs) {
    Preconditions.checkArgument(
        maxWalFlushIntervalMs >= 0,
        "maxWalFlushIntervalMs must not be negative, got %s",
        maxWalFlushIntervalMs);
    this.maxWalFlushIntervalMs = Optional.of(maxWalFlushIntervalMs);
    return this;
  }

  /** Maximum size of a MemTable before it is flushed, in bytes. */
  public ShardWriterConfig withMaxMemtableSize(long maxMemtableSize) {
    Preconditions.checkArgument(
        maxMemtableSize >= 0, "maxMemtableSize must not be negative, got %s", maxMemtableSize);
    this.maxMemtableSize = Optional.of(maxMemtableSize);
    return this;
  }

  /** Maximum number of rows in a MemTable before it is flushed. */
  public ShardWriterConfig withMaxMemtableRows(long maxMemtableRows) {
    Preconditions.checkArgument(
        maxMemtableRows >= 0, "maxMemtableRows must not be negative, got %s", maxMemtableRows);
    this.maxMemtableRows = Optional.of(maxMemtableRows);
    return this;
  }

  /** Maximum number of record batches in a MemTable before it is flushed. */
  public ShardWriterConfig withMaxMemtableBatches(long maxMemtableBatches) {
    Preconditions.checkArgument(
        maxMemtableBatches >= 0,
        "maxMemtableBatches must not be negative, got %s",
        maxMemtableBatches);
    this.maxMemtableBatches = Optional.of(maxMemtableBatches);
    return this;
  }

  /** Maximum unflushed MemTable bytes allowed before applying backpressure. */
  public ShardWriterConfig withMaxUnflushedMemtableBytes(long maxUnflushedMemtableBytes) {
    Preconditions.checkArgument(
        maxUnflushedMemtableBytes >= 0,
        "maxUnflushedMemtableBytes must not be negative, got %s",
        maxUnflushedMemtableBytes);
    this.maxUnflushedMemtableBytes = Optional.of(maxUnflushedMemtableBytes);
    return this;
  }

  /** Batch size used when scanning the WAL manifest. */
  public ShardWriterConfig withManifestScanBatchSize(long manifestScanBatchSize) {
    Preconditions.checkArgument(
        manifestScanBatchSize >= 0,
        "manifestScanBatchSize must not be negative, got %s",
        manifestScanBatchSize);
    this.manifestScanBatchSize = Optional.of(manifestScanBatchSize);
    return this;
  }

  /** Interval between backpressure log messages, in milliseconds. */
  public ShardWriterConfig withBackpressureLogIntervalMs(long backpressureLogIntervalMs) {
    Preconditions.checkArgument(
        backpressureLogIntervalMs >= 0,
        "backpressureLogIntervalMs must not be negative, got %s",
        backpressureLogIntervalMs);
    this.backpressureLogIntervalMs = Optional.of(backpressureLogIntervalMs);
    return this;
  }

  /** Interval between statistics log messages, in milliseconds; {@code 0} disables logging. */
  public ShardWriterConfig withStatsLogIntervalMs(long statsLogIntervalMs) {
    Preconditions.checkArgument(
        statsLogIntervalMs >= 0,
        "statsLogIntervalMs must not be negative, got %s",
        statsLogIntervalMs);
    this.statsLogIntervalMs = Optional.of(statsLogIntervalMs);
    return this;
  }

  /**
   * Per-index HNSW build-parameter overrides for the MemTable this writer builds. Each {@link
   * MemWalHnswParams#indexName} should reference a maintained vector index.
   */
  public ShardWriterConfig withHnswParams(List<MemWalHnswParams> hnswParams) {
    Preconditions.checkNotNull(hnswParams, "hnswParams must not be null");
    this.hnswParams = hnswParams;
    return this;
  }

  public Optional<Boolean> durableWrite() {
    return durableWrite;
  }

  public Optional<Long> maxWalBufferSize() {
    return maxWalBufferSize;
  }

  public Optional<Long> maxWalFlushIntervalMs() {
    return maxWalFlushIntervalMs;
  }

  public Optional<Long> maxMemtableSize() {
    return maxMemtableSize;
  }

  public Optional<Long> maxMemtableRows() {
    return maxMemtableRows;
  }

  public Optional<Long> maxMemtableBatches() {
    return maxMemtableBatches;
  }

  public Optional<Long> maxUnflushedMemtableBytes() {
    return maxUnflushedMemtableBytes;
  }

  public Optional<Long> manifestScanBatchSize() {
    return manifestScanBatchSize;
  }

  public Optional<Long> backpressureLogIntervalMs() {
    return backpressureLogIntervalMs;
  }

  public Optional<Long> statsLogIntervalMs() {
    return statsLogIntervalMs;
  }

  public List<MemWalHnswParams> hnswParams() {
    return hnswParams;
  }
}
