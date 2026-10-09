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

import org.lance.memwal.CompactedSsTable;

import com.google.common.base.MoreObjects;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Update SSTable compaction progress in the MemWAL index. */
public final class UpdateMemWalState implements Operation {
  private final List<CompactedSsTable> compactedSstables;

  private UpdateMemWalState(List<CompactedSsTable> compactedSstables) {
    this.compactedSstables = Objects.requireNonNull(compactedSstables);
  }

  public List<CompactedSsTable> getCompactedSstables() {
    return compactedSstables;
  }

  @Override
  public String name() {
    return "UpdateMemWalState";
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("compactedSstables", compactedSstables).toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    UpdateMemWalState that = (UpdateMemWalState) o;
    if (compactedSstables.size() != that.compactedSstables.size()) return false;
    for (int i = 0; i < compactedSstables.size(); i++) {
      CompactedSsTable left = compactedSstables.get(i);
      CompactedSsTable right = that.compactedSstables.get(i);
      if (left.getGeneration() != right.getGeneration()
          || !Objects.equals(left.getShardId(), right.getShardId())) {
        return false;
      }
    }
    return true;
  }

  @Override
  public int hashCode() {
    int result = 1;
    for (CompactedSsTable sstable : compactedSstables) {
      result = 31 * result + Objects.hash(sstable.getShardId(), sstable.getGeneration());
    }
    return result;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private List<CompactedSsTable> compactedSstables = Collections.emptyList();

    public Builder compactedSstables(List<CompactedSsTable> compactedSstables) {
      this.compactedSstables = compactedSstables;
      return this;
    }

    public UpdateMemWalState build() {
      return new UpdateMemWalState(compactedSstables);
    }
  }
}
