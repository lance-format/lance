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
package org.lance.index.vector;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;

import java.util.Optional;

/**
 * Parameters for building an HNSW index in each IVF partition. This speeds up the search in a large
 * dataset.
 */
public class HnswBuildParams {
  private final short maxLevel;
  private final int m;
  private final int efConstruction;
  private final Optional<Integer> prefetchDistance;

  private HnswBuildParams(Builder builder) {
    this.maxLevel = builder.maxLevel;
    this.m = builder.m;
    this.efConstruction = builder.efConstruction;
    this.prefetchDistance = builder.prefetchDistance;
  }

  public static class Builder {
    private short maxLevel = 7;
    private int m = 20;
    private int efConstruction = 150;
    private Optional<Integer> prefetchDistance = Optional.of(2);

    /**
     * Create a new builder for HNSW index parameters. Each IVF partition will be built with an HNSW
     * index.
     */
    public Builder() {}

    /**
     * @param maxLevel the maximum number of levels in the graph
     * @return Builder
     */
    public Builder setMaxLevel(short maxLevel) {
      Preconditions.checkArgument(maxLevel > 0, "maxLevel must be greater than 0");
      this.maxLevel = maxLevel;
      return this;
    }

    /**
     * @param m the number of edges per node in the graph
     * @return Builder
     */
    public Builder setM(int m) {
      Preconditions.checkArgument(m > 0, "m must be greater than 0");
      this.m = m;
      return this;
    }

    /**
     * @param efConstruction the number of nodes to examine during the construction.
     * @return Builder
     */
    public Builder setEfConstruction(int efConstruction) {
      Preconditions.checkArgument(efConstruction > 0, "efConstruction must be greater than 0");
      this.efConstruction = efConstruction;
      return this;
    }

    /**
     * @param prefetchDistance number of vectors ahead to prefetch while building the graph
     * @return Builder
     */
    public Builder setPrefetchDistance(Integer prefetchDistance) {
      this.prefetchDistance = Optional.ofNullable(prefetchDistance);
      return this;
    }

    public HnswBuildParams build() {
      return new HnswBuildParams(this);
    }
  }

  public short getMaxLevel() {
    return maxLevel;
  }

  public int getM() {
    return m;
  }

  public int getEfConstruction() {
    return efConstruction;
  }

  public Optional<Integer> getPrefetchDistance() {
    return prefetchDistance;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("maxLevel", maxLevel)
        .add("m", m)
        .add("efConstruction", efConstruction)
        .add("prefetchDistance", prefetchDistance.orElse(null))
        .toString();
  }
}
