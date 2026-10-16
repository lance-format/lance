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

import java.util.List;
import java.util.Objects;

/**
 * Replace the data files backing some fields of existing fragments with new files, without moving
 * rows. Each group names a fragment and one new data file for it. At commit, a file holding exactly
 * the new file's fields is swapped for it; otherwise the new file's fields are tombstoned where
 * they live and the new file is appended. A fragment can take several groups, one per new file.
 * Used for null column population, and by compaction to repack columns into fewer files.
 *
 * <p>{@code dataChange == false} declares that the new files hold the same values as the files they
 * replace: indices keep their coverage, overlays keep shadowing, and no row is reported as updated.
 */
public class DataReplacement implements Operation {
  private final List<DataReplacementGroup> replacements;
  private final boolean dataChange;

  private DataReplacement(List<DataReplacementGroup> replacements, boolean dataChange) {
    this.replacements = replacements;
    this.dataChange = dataChange;
  }

  /**
   * Get the list of data replacement groups.
   *
   * @return the list of data replacement groups
   */
  public List<DataReplacementGroup> replacements() {
    return replacements;
  }

  /**
   * Whether the new files change any value.
   *
   * @return false if the values were only moved to new files
   */
  public boolean dataChange() {
    return dataChange;
  }

  @Override
  public String name() {
    return "DataReplacement";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("replacements", replacements)
        .add("dataChange", dataChange)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    DataReplacement that = (DataReplacement) o;
    return dataChange == that.dataChange && Objects.equals(replacements, that.replacements);
  }

  @Override
  public int hashCode() {
    return Objects.hash(replacements, dataChange);
  }

  /**
   * Create a new builder for DataReplacement.
   *
   * @return a new builder
   */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder for DataReplacement. */
  public static class Builder {
    private List<DataReplacementGroup> replacements;
    private boolean dataChange = true;

    public Builder() {}

    /**
     * Set the list of data replacement groups.
     *
     * @param replacements the list of data replacement groups
     * @return this builder
     */
    public Builder replacements(List<DataReplacementGroup> replacements) {
      this.replacements = replacements;
      return this;
    }

    /**
     * Set whether the new files change any value. Defaults to true; pass false only when the new
     * files hold the same values as the files they replace.
     *
     * @param dataChange whether the new files change any value
     * @return this builder
     */
    public Builder dataChange(boolean dataChange) {
      this.dataChange = dataChange;
      return this;
    }

    /**
     * Build a new DataReplacement.
     *
     * @return a new DataReplacement
     */
    public DataReplacement build() {
      return new DataReplacement(replacements, dataChange);
    }
  }

  /** A group of data replacement, containing a fragment ID and a new data file. */
  public static class DataReplacementGroup {
    private final long fragmentId;
    private final DataFile replacedFile;

    /**
     * Create a new DataReplacementGroup.
     *
     * @param fragmentId the fragment ID
     * @param replacedFile the new data file to replace old file of the fragment id
     */
    public DataReplacementGroup(long fragmentId, DataFile replacedFile) {
      this.fragmentId = fragmentId;
      this.replacedFile = replacedFile;
    }

    /**
     * Get the fragment ID.
     *
     * @return the fragment ID
     */
    public long fragmentId() {
      return fragmentId;
    }

    /**
     * Get the new data file.
     *
     * @return the new data file
     */
    public DataFile replacedFile() {
      return replacedFile;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      DataReplacementGroup that = (DataReplacementGroup) o;
      return fragmentId == that.fragmentId && Objects.equals(replacedFile, that.replacedFile);
    }

    @Override
    public int hashCode() {
      return Objects.hash(fragmentId, replacedFile);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("fragmentId", fragmentId)
          .add("replacedFile", replacedFile)
          .toString();
    }
  }
}
