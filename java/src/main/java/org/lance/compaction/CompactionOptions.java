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

  private CompactionOptions(
      Optional<Long> targetRowsPerFragment,
      Optional<Long> maxRowsPerGroup,
      Optional<Long> maxBytesPerFile,
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
      List<Long> excludedFragmentIds) {
    this.targetRowsPerFragment = targetRowsPerFragment;
    this.maxRowsPerGroup = maxRowsPerGroup;
    this.maxBytesPerFile = maxBytesPerFile;
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

  /** Builder for CompactionOptions. */
  public static class Builder {
    private Optional<Long> targetRowsPerFragment = Optional.empty();
    private Optional<Long> maxRowsPerGroup = Optional.empty();
    private Optional<Long> maxBytesPerFile = Optional.empty();
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

    private Builder() {}

    public Builder withTargetRowsPerFragment(long targetRowsPerFragment) {
      this.targetRowsPerFragment = Optional.of(targetRowsPerFragment);
      return this;
    }

    public Builder withMaxRowsPerGroup(long maxRowsPerGroup) {
      this.maxRowsPerGroup = Optional.of(maxRowsPerGroup);
      return this;
    }

    public Builder withMaxBytesPerFile(long maxBytesPerFile) {
      this.maxBytesPerFile = Optional.of(maxBytesPerFile);
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
     * Maximum number of fragments whose identities may change in a single run. This includes
     * compacted source fragments and following fragments relabeled to preserve row order. The
     * planner selects candidates from a suffix within this limit.
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
     * Fragment IDs to exclude from compaction planning. Excluded fragments remain unchanged and act
     * as boundaries, so fragments on opposite sides are not combined into the same task. To
     * preserve row order, only candidates after the last present excluded fragment are eligible.
     * Duplicate and unknown IDs are ignored.
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
          excludedFragmentIds);
    }
  }
}
