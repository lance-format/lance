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
package org.lance.operation;

import org.lance.fragment.DataFile;

import com.google.common.base.MoreObjects;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Attach overlay files that update selected cells without rewriting fragment base files. */
public final class DataOverlay implements Operation {
  private final List<DataOverlayGroup> groups;

  private DataOverlay(List<DataOverlayGroup> groups) {
    this.groups = Objects.requireNonNull(groups);
  }

  public List<DataOverlayGroup> getGroups() {
    return groups;
  }

  @Override
  public String name() {
    return "DataOverlay";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("groups", groups).toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    DataOverlay that = (DataOverlay) o;
    return Objects.equals(groups, that.groups);
  }

  @Override
  public int hashCode() {
    return Objects.hash(groups);
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private List<DataOverlayGroup> groups = Collections.emptyList();

    public Builder groups(List<DataOverlayGroup> groups) {
      this.groups = groups;
      return this;
    }

    public DataOverlay build() {
      return new DataOverlay(groups);
    }
  }

  /** Overlay files appended to a single fragment, ordered oldest to newest. */
  public static final class DataOverlayGroup {
    private final long fragmentId;
    private final List<DataOverlayFile> overlays;

    public DataOverlayGroup(long fragmentId, List<DataOverlayFile> overlays) {
      this.fragmentId = fragmentId;
      this.overlays = Objects.requireNonNull(overlays);
    }

    public long getFragmentId() {
      return fragmentId;
    }

    public List<DataOverlayFile> getOverlays() {
      return overlays;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      DataOverlayGroup that = (DataOverlayGroup) o;
      return fragmentId == that.fragmentId && Objects.equals(overlays, that.overlays);
    }

    @Override
    public int hashCode() {
      return Objects.hash(fragmentId, overlays);
    }
  }

  /**
   * A data file together with the physical-row coverage it supplies.
   *
   * <p>When this object is used to commit an overlay, Lance ignores the supplied committed version
   * and stamps the overlay stored in the manifest with the new dataset version (including after a
   * retry). The value remains part of the transaction model so transactions read from native code
   * can represent it exactly.
   */
  public static final class DataOverlayFile {
    private final DataFile dataFile;
    private final OverlayCoverage coverage;
    private final long committedVersion;

    public DataOverlayFile(DataFile dataFile, OverlayCoverage coverage, long committedVersion) {
      this.dataFile = Objects.requireNonNull(dataFile);
      this.coverage = Objects.requireNonNull(coverage);
      if (!coverage.isShared() && coverage.getBitmaps().size() != dataFile.getFields().length) {
        throw new IllegalArgumentException(
            String.format(
                "per-field overlay coverage for %s has %d bitmaps but the data file has %d fields",
                dataFile.getPath(), coverage.getBitmaps().size(), dataFile.getFields().length));
      }
      this.committedVersion = committedVersion;
    }

    /** Creates an overlay for commit; Lance stamps its manifest version during the commit. */
    public static DataOverlayFile forCommit(DataFile dataFile, OverlayCoverage coverage) {
      return new DataOverlayFile(dataFile, coverage, 0);
    }

    public DataFile getDataFile() {
      return dataFile;
    }

    public OverlayCoverage getCoverage() {
      return coverage;
    }

    public long getCommittedVersion() {
      return committedVersion;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      DataOverlayFile that = (DataOverlayFile) o;
      return committedVersion == that.committedVersion
          && Objects.equals(dataFile, that.dataFile)
          && Objects.equals(coverage, that.coverage);
    }

    @Override
    public int hashCode() {
      int dataFileHash =
          Objects.hash(
              dataFile.getPath(),
              dataFile.getFileMajorVersion(),
              dataFile.getFileMinorVersion(),
              dataFile.getFileSizeBytes());
      dataFileHash = 31 * dataFileHash + Arrays.hashCode(dataFile.getFields());
      dataFileHash = 31 * dataFileHash + Arrays.hashCode(dataFile.getColumnIndices());
      return Objects.hash(dataFileHash, coverage, committedVersion);
    }
  }

  /**
   * Portable RoaringBitmap bytes for shared or per-field overlay coverage. Per-field bitmaps are
   * ordered to match the overlay data file's fields.
   */
  public static final class OverlayCoverage {
    private final boolean shared;
    private final List<byte[]> bitmaps;

    private OverlayCoverage(boolean shared, List<byte[]> bitmaps) {
      this.shared = shared;
      this.bitmaps = Objects.requireNonNull(bitmaps);
      if (shared && bitmaps.size() != 1) {
        throw new IllegalArgumentException("shared overlay coverage requires exactly one bitmap");
      }
    }

    public static OverlayCoverage shared(byte[] bitmap) {
      return new OverlayCoverage(true, Collections.singletonList(bitmap));
    }

    public static OverlayCoverage perField(List<byte[]> bitmaps) {
      return new OverlayCoverage(false, new ArrayList<>(bitmaps));
    }

    public boolean isShared() {
      return shared;
    }

    public List<byte[]> getBitmaps() {
      return bitmaps;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      OverlayCoverage that = (OverlayCoverage) o;
      if (shared != that.shared || bitmaps.size() != that.bitmaps.size()) return false;
      for (int i = 0; i < bitmaps.size(); i++) {
        if (!Arrays.equals(bitmaps.get(i), that.bitmaps.get(i))) return false;
      }
      return true;
    }

    @Override
    public int hashCode() {
      int result = Boolean.hashCode(shared);
      for (byte[] bitmap : bitmaps) result = 31 * result + Arrays.hashCode(bitmap);
      return result;
    }
  }
}
