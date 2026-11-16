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
package org.lance.compaction;

import org.lance.DataStorageVersion;

import com.google.common.base.MoreObjects;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OptionalDataException;
import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Compaction options. All fields are optional in Java side. We do not set default value for them to
 * avoid conflicting with those default values in Rust side. Please check the <a
 * href="https://docs.rs/lance/latest/src/lance/dataset/optimize.rs.html#118">rust code</a> for all
 * default values.
 */
public class CompactionOptions implements Serializable {
  // Pinned to the UID generated before maxSourceRows/maxSourceBytes were added, so that
  // CompactionTask streams queued by older workers still deserialize during a rolling upgrade.
  private static final long serialVersionUID = 3114922060085417942L;

  // these fields are effectively final, but not marked as final for de/ser
  private Optional<Long> targetRowsPerFragment;
  private Optional<Long> maxRowsPerGroup;
  private Optional<Long> maxBytesPerFile;
  private Optional<Long> minBytesPerFile;
  private Optional<Boolean> materializeDeletions;
  private Optional<Float> materializeDeletionsThreshold;
  private Optional<Long> numThreads;
  private Optional<Long> batchSize;
  private Optional<Boolean> deferIndexRemap;
  private Optional<CompactionMode> compactionMode;
  private Optional<Long> binaryCopyReadBatchBytes;
  private Optional<Long> maxSourceFragments;
  private Optional<Long> maxSourceRows;
  private Optional<Long> maxSourceBytes;
  private List<Long> excludedFragmentIds;
  private Optional<DataStorageVersion> dataStorageVersion;
  private Optional<SourceBudgetMode> sourceBudgetMode;

  private CompactionOptions(
      Optional<Long> targetRowsPerFragment,
      Optional<Long> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
      Optional<Long> minBytesPerFile,
      Optional<Boolean> materializeDeletions,
      Optional<Float> materializeDeletionsThreshold,
      Optional<Long> numThreads,
      Optional<Long> batchSize,
      Optional<Boolean> deferIndexRemap,
      Optional<CompactionMode> compactionMode,
      Optional<Long> binaryCopyReadBatchBytes,
      Optional<Long> maxSourceFragments,
      Optional<Long> maxSourceRows,
      Optional<Long> maxSourceBytes,
      List<Long> excludedFragmentIds,
      Optional<String> dataStorageVersion,
      Optional<SourceBudgetMode> sourceBudgetMode) {
    this.targetRowsPerFragment = targetRowsPerFragment;
    this.maxRowsPerGroup = maxRowsPerGroup;
    this.maxBytesPerFile = maxBytesPerFile;
    this.minBytesPerFile = minBytesPerFile;
    this.materializeDeletions = materializeDeletions;
    this.materializeDeletionsThreshold = materializeDeletionsThreshold;
    this.numThreads = numThreads;
    this.batchSize = batchSize;
    this.deferIndexRemap = deferIndexRemap;
    this.compactionMode = compactionMode;
    this.binaryCopyReadBatchBytes = binaryCopyReadBatchBytes;
    this.maxSourceFragments = maxSourceFragments;
    this.maxSourceRows = maxSourceRows;
    this.maxSourceBytes = maxSourceBytes;
    this.excludedFragmentIds = List.copyOf(excludedFragmentIds);
    this.dataStorageVersion = dataStorageVersion.map(DataStorageVersion::fromRustString);
    this.sourceBudgetMode = sourceBudgetMode;
  }

  public Optional<Boolean> getDeferIndexRemap() {
    return deferIndexRemap;
  }

  /** Returns the compaction mode as its string value for the native layer. */
  public Optional<String> getCompactionMode() {
    return compactionMode.map(CompactionMode::getValue);
  }

  public Optional<Long> getBinaryCopyReadBatchBytes() {
    return binaryCopyReadBatchBytes;
  }

  public Optional<Long> getMaxSourceFragments() {
    return maxSourceFragments;
  }

  public Optional<Long> getMaxSourceRows() {
    return maxSourceRows;
  }

  public Optional<Long> getMaxSourceBytes() {
    return maxSourceBytes;
  }

  public List<Long> getExcludedFragmentIds() {
    return excludedFragmentIds;
  }

  /** Returns the source budget mode as its string value for the native layer. */
  public Optional<String> getSourceBudgetMode() {
    return sourceBudgetMode.map(SourceBudgetMode::getValue);
  }

  /** Returns the version selector as its string value for the native layer. */
  public Optional<String> getDataStorageVersion() {
    return dataStorageVersion.map(DataStorageVersion::toRustString);
  }

  public Optional<Boolean> getMaterializeDeletions() {
    return materializeDeletions;
  }

  public Optional<Float> getMaterializeDeletionsThreshold() {
    return materializeDeletionsThreshold;
  }

  public Optional<Long> getBatchSize() {
    return batchSize;
  }

  public Optional<Long> getMaxBytesPerFile() {
    return maxBytesPerFile;
  }

  /** Returns the optional file-size threshold in bytes for small-file compaction. */
  public Optional<Long> getMinBytesPerFile() {
    return minBytesPerFile;
  }

  public Optional<Long> getMaxRowsPerGroup() {
    return maxRowsPerGroup;
  }

  public Optional<Long> getNumThreads() {
    return numThreads;
  }

  public Optional<Long> getTargetRowsPerFragment() {
    return targetRowsPerFragment;
  }

  public static Builder builder() {
    return new Builder();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("targetRowsPerFragment", targetRowsPerFragment.orElse(null))
        .add("maxRowsPerGroup", maxRowsPerGroup.orElse(null))
        .add("maxBytesPerFile", maxBytesPerFile.orElse(null))
        .add("minBytesPerFile", minBytesPerFile.orElse(null))
        .add("materializeDeletions", materializeDeletions.orElse(null))
        .add("materializeDeletionsThreshold", materializeDeletionsThreshold.orElse(null))
        .add("numThreads", numThreads.orElse(null))
        .add("batchSize", batchSize.orElse(null))
        .add("deferIndexRemap", deferIndexRemap.orElse(null))
        .add("compactionMode", compactionMode.orElse(null))
        .add("binaryCopyReadBatchBytes", binaryCopyReadBatchBytes.orElse(null))
        .add("maxSourceFragments", maxSourceFragments.orElse(null))
        .add("maxSourceRows", maxSourceRows.orElse(null))
        .add("maxSourceBytes", maxSourceBytes.orElse(null))
        .add("excludedFragmentIds", excludedFragmentIds)
        .add("dataStorageVersion", dataStorageVersion.orElse(null))
        .add("sourceBudgetMode", sourceBudgetMode.orElse(null))
        .toString();
  }

  private void writeObject(ObjectOutputStream output) throws IOException {
    output.writeObject(targetRowsPerFragment.orElse(null));
    output.writeObject(maxRowsPerGroup.orElse(null));
    output.writeObject(maxBytesPerFile.orElse(null));
    output.writeObject(materializeDeletions.orElse(null));
    output.writeObject(materializeDeletionsThreshold.orElse(null));
    output.writeObject(numThreads.orElse(null));
    output.writeObject(batchSize.orElse(null));
    output.writeObject(deferIndexRemap.orElse(null));
    output.writeObject(compactionMode.map(CompactionMode::getValue).orElse(null));
    output.writeObject(binaryCopyReadBatchBytes.orElse(null));
    output.writeObject(maxSourceFragments.orElse(null));
    output.writeObject(maxSourceRows.orElse(null));
    output.writeObject(maxSourceBytes.orElse(null));
    output.writeObject(excludedFragmentIds);
    output.writeObject(getDataStorageVersion().orElse(null));
    output.writeObject(minBytesPerFile.orElse(null));
    output.writeObject(sourceBudgetMode.map(SourceBudgetMode::getValue).orElse(null));
  }

  private void readObject(ObjectInputStream input) throws IOException, ClassNotFoundException {
    this.targetRowsPerFragment = Optional.ofNullable((Long) input.readObject());
    this.maxRowsPerGroup = Optional.ofNullable((Long) input.readObject());
    this.maxBytesPerFile = Optional.ofNullable((Long) input.readObject());
    this.materializeDeletions = Optional.ofNullable((Boolean) input.readObject());
    this.materializeDeletionsThreshold = Optional.ofNullable((Float) input.readObject());
    this.numThreads = Optional.ofNullable((Long) input.readObject());
    this.batchSize = Optional.ofNullable((Long) input.readObject());
    this.deferIndexRemap = Optional.ofNullable((Boolean) input.readObject());
    String modeStr = (String) input.readObject();
    this.compactionMode = Optional.empty();
    if (modeStr != null) {
      for (CompactionMode m : CompactionMode.values()) {
        if (m.getValue().equals(modeStr)) {
          this.compactionMode = Optional.of(m);
          break;
        }
      }
    }
    this.binaryCopyReadBatchBytes = Optional.ofNullable((Long) input.readObject());
    this.maxSourceFragments = Optional.ofNullable((Long) input.readObject());
    this.maxSourceRows = readTrailingLong(input);
    this.maxSourceBytes = readTrailingLong(input);
    this.excludedFragmentIds = readTrailingLongList(input);
    this.dataStorageVersion = readTrailingString(input).map(DataStorageVersion::fromRustString);
    this.minBytesPerFile = readTrailingLong(input);
    this.sourceBudgetMode = readTrailingSourceBudgetMode(input);
  }

  /**
   * Reads a trailing Long field that older writers did not emit. Streams written before the field
   * was added end here, which surfaces as an {@link OptionalDataException} with {@code eof} set; in
   * that case the field is treated as unset.
   */
  private static Optional<Long> readTrailingLong(ObjectInputStream input)
      throws IOException, ClassNotFoundException {
    try {
      return Optional.ofNullable((Long) input.readObject());
    } catch (OptionalDataException e) {
      if (!e.eof) {
        throw e;
      }
      return Optional.empty();
    }
  }

  @SuppressWarnings("unchecked")
  private static List<Long> readTrailingLongList(ObjectInputStream input)
      throws IOException, ClassNotFoundException {
    try {
      List<Long> fragmentIds = (List<Long>) input.readObject();
      return fragmentIds == null ? Collections.emptyList() : List.copyOf(fragmentIds);
    } catch (OptionalDataException e) {
      if (!e.eof) {
        throw e;
      }
      return Collections.emptyList();
    }
  }

  private static Optional<String> readTrailingString(ObjectInputStream input)
      throws IOException, ClassNotFoundException {
    try {
      return Optional.ofNullable((String) input.readObject());
    } catch (OptionalDataException e) {
      if (!e.eof) {
        throw e;
      }
      return Optional.empty();
    }
  }

  private static Optional<SourceBudgetMode> readTrailingSourceBudgetMode(ObjectInputStream input)
      throws IOException, ClassNotFoundException {
    Optional<String> value = readTrailingString(input);
    if (value.isEmpty()) {
      return Optional.empty();
    }
    for (SourceBudgetMode mode : SourceBudgetMode.values()) {
      if (mode.getValue().equals(value.get())) {
        return Optional.of(mode);
      }
    }
    return Optional.empty();
  }

  /** Builder for CompactionOptions. */
  public static class Builder {
    private Optional<Long> targetRowsPerFragment = Optional.empty();
    private Optional<Long> maxRowsPerGroup = Optional.empty();
    private Optional<Long> maxBytesPerFile = Optional.empty();
    private Optional<Long> minBytesPerFile = Optional.empty();
    private Optional<Boolean> materializeDeletions = Optional.empty();
    private Optional<Float> materializeDeletionsThreshold = Optional.empty();
    private Optional<Long> numThreads = Optional.empty();
    private Optional<Long> batchSize = Optional.empty();
    private Optional<Boolean> deferIndexRemap = Optional.empty();
    private Optional<CompactionMode> compactionMode = Optional.empty();
    private Optional<Long> binaryCopyReadBatchBytes = Optional.empty();
    private Optional<Long> maxSourceFragments = Optional.empty();
    private Optional<Long> maxSourceRows = Optional.empty();
    private Optional<Long> maxSourceBytes = Optional.empty();
    private List<Long> excludedFragmentIds = Collections.emptyList();
    private Optional<DataStorageVersion> dataStorageVersion = Optional.empty();
    private Optional<SourceBudgetMode> sourceBudgetMode = Optional.empty();

    private Builder() {}

    public Builder withTargetRowsPerFragment(long targetRowsPerFragment) {
      this.targetRowsPerFragment = Optional.of(targetRowsPerFragment);
      return this;
    }

    public Builder withMaxRowsPerGroup(long maxRowsPerGroup) {
      this.maxRowsPerGroup = Optional.of(maxRowsPerGroup);
      return this;
    }

    /**
     * Maximum bytes per output file. Defaults to the dataset config value, or the <a
     * href="https://docs.rs/lance/latest/lance/dataset/write/struct.WriteParams.html#structfield.max_bytes_per_file">writer's
     * default</a>.
     */
    public Builder withMaxBytesPerFile(long maxBytesPerFile) {
      this.maxBytesPerFile = Optional.of(maxBytesPerFile);
      return this;
    }

    /**
     * Files at or above this size are excluded from small-file compaction. Defaults to the dataset
     * config value, or 95% of {@link #withMaxBytesPerFile(long) maxBytesPerFile}. Must be greater
     * than zero and no greater than that limit. Files needing cleanup may still be compacted.
     *
     * <p>Example: {@code CompactionOptions.builder().withMinBytesPerFile(512L * 1024 *
     * 1024).build()}.
     */
    public Builder withMinBytesPerFile(long minBytesPerFile) {
      this.minBytesPerFile = Optional.of(minBytesPerFile);
      return this;
    }

    public Builder withMaterializeDeletions(boolean materializeDeletions) {
      this.materializeDeletions = Optional.of(materializeDeletions);
      return this;
    }

    public Builder withMaterializeDeletionsThreshold(float materializeDeletionsThreshold) {
      this.materializeDeletionsThreshold = Optional.of(materializeDeletionsThreshold);
      return this;
    }

    public Builder withNumThreads(long numThreads) {
      this.numThreads = Optional.of(numThreads);
      return this;
    }

    public Builder withBatchSize(long batchSize) {
      this.batchSize = Optional.of(batchSize);
      return this;
    }

    public Builder withDeferIndexRemap(boolean deferIndexRemap) {
      this.deferIndexRemap = Optional.of(deferIndexRemap);
      return this;
    }

    public Builder withCompactionMode(CompactionMode compactionMode) {
      this.compactionMode = Optional.of(compactionMode);
      return this;
    }

    public Builder withBinaryCopyReadBatchBytes(long binaryCopyReadBatchBytes) {
      this.binaryCopyReadBatchBytes = Optional.of(binaryCopyReadBatchBytes);
      return this;
    }

    /**
     * Maximum number of source fragments to compact in a single run. Tasks are included until
     * adding the next task would exceed this limit, allowing for incremental compaction. Fragments
     * are processed oldest first.
     *
     * @throws IllegalArgumentException if {@code maxSourceFragments} is not positive
     */
    public Builder withMaxSourceFragments(long maxSourceFragments) {
      this.maxSourceFragments =
          Optional.of(positiveBudget("maxSourceFragments", maxSourceFragments));
      return this;
    }

    /**
     * Maximum number of source rows to compact in a single run. Rows are counted as live rows
     * (physical rows minus soft-deleted rows). Tasks are included until adding the next task would
     * exceed this limit.
     *
     * @throws IllegalArgumentException if {@code maxSourceRows} is not positive
     */
    public Builder withMaxSourceRows(long maxSourceRows) {
      this.maxSourceRows = Optional.of(positiveBudget("maxSourceRows", maxSourceRows));
      return this;
    }

    /**
     * Maximum number of source bytes to compact in a single run, measured as the total size of the
     * source fragments' data and overlay files. Tasks are included until adding the next task would
     * exceed this limit. Blob v2 payloads live in separate blob files and are not counted, so this
     * is not a cap on total compaction I/O for datasets with blob columns.
     *
     * @throws IllegalArgumentException if {@code maxSourceBytes} is not positive
     */
    public Builder withMaxSourceBytes(long maxSourceBytes) {
      this.maxSourceBytes = Optional.of(positiveBudget("maxSourceBytes", maxSourceBytes));
      return this;
    }

    /**
     * Select the output data file version without changing the dataset's default write version.
     * Omission uses the compaction config target when set, otherwise the dataset default. The
     * planner resolves release selectors before distributing tasks. V1/V2 cross-family targets are
     * rejected by the engine.
     */
    public Builder withDataStorageVersion(DataStorageVersion version) {
      this.dataStorageVersion = Optional.of(Objects.requireNonNull(version, "version"));
      return this;
    }

    /**
     * Sets how planning applies all configured max-source budgets to cumulative totals across the
     * tasks selected for one plan. {@link SourceBudgetMode#HARD} rejects a task when adding it
     * would exceed a cumulative budget. {@link SourceBudgetMode#SOFT} always admits the first task;
     * if it exceeds a budget the plan stops there, otherwise later tasks use the same cumulative
     * checks as hard mode. If a byte budget is configured but the first task has a source file
     * without a recorded size, soft mode admits that task and stops; hard mode reports an error.
     */
    public Builder withSourceBudgetMode(SourceBudgetMode sourceBudgetMode) {
      this.sourceBudgetMode =
          Optional.of(Objects.requireNonNull(sourceBudgetMode, "sourceBudgetMode"));
      return this;
    }

    /**
     * Fragment IDs to exclude from compaction planning. Excluded fragments remain unchanged and act
     * as boundaries, so fragments on opposite sides are not combined into the same task. Duplicate
     * and unknown IDs are ignored.
     *
     * @throws IllegalArgumentException if an ID is negative or exceeds the unsigned 32-bit range
     */
    public Builder withExcludedFragmentIds(List<Long> excludedFragmentIds) {
      Objects.requireNonNull(excludedFragmentIds, "excludedFragmentIds");
      for (Long fragmentId : excludedFragmentIds) {
        if (fragmentId == null || fragmentId < 0 || fragmentId > 0xFFFF_FFFFL) {
          throw new IllegalArgumentException(
              "excludedFragmentIds must contain values between 0 and 4294967295, got "
                  + fragmentId);
        }
      }
      this.excludedFragmentIds = List.copyOf(excludedFragmentIds);
      return this;
    }

    /**
     * A max source budget of zero admits no work and a negative value would wrap around to an
     * effectively unlimited budget on the Rust side, so both are rejected here. Leave the option
     * unset for no limit.
     */
    private static long positiveBudget(String name, long value) {
      if (value <= 0) {
        throw new IllegalArgumentException(
            name + " must be greater than 0, got " + value + " (leave unset for no limit)");
      }
      return value;
    }

    public CompactionOptions build() {
      return new CompactionOptions(
          targetRowsPerFragment,
          maxRowsPerGroup,
          maxBytesPerFile,
          minBytesPerFile,
          materializeDeletions,
          materializeDeletionsThreshold,
          numThreads,
          batchSize,
          deferIndexRemap,
          compactionMode,
          binaryCopyReadBatchBytes,
          maxSourceFragments,
          maxSourceRows,
          maxSourceBytes,
          excludedFragmentIds,
          dataStorageVersion.map(DataStorageVersion::toRustString),
          sourceBudgetMode);
    }
  }
}
