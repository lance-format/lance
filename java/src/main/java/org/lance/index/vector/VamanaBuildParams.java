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

/**
 * Parameters for building a Vamana graph in each IVF partition of an {@code IVF_VAMANA_FLAT},
 * {@code IVF_VAMANA_SQ} or {@code IVF_VAMANA_PQ} index. A query walks the graph of every partition
 * it probes and keeps its {@code ef} best candidates while it walks.
 *
 * <p>These indices support the l2 and cosine distances only. Lance validates the parameters when
 * the build starts. The defaults are the best of the pairs of {@code maxDegree} and {@code
 * searchListSize} measured on SIFT1M against Lance's HNSW at matched memory.
 */
public class VamanaBuildParams {
  private final int maxDegree;
  private final int searchListSize;
  private final float alpha;

  private VamanaBuildParams(Builder builder) {
    this.maxDegree = builder.maxDegree;
    this.searchListSize = builder.searchListSize;
    this.alpha = builder.alpha;
  }

  public static class Builder {
    private int maxDegree = 64;
    private int searchListSize = 100;
    private float alpha = 1.2f;

    /** Create a new builder for the graph parameters, starting from the defaults. */
    public Builder() {}

    /**
     * @param maxDegree the most out-edges a vertex of the graph keeps, between 1 and 1024. Default
     *     is 64.
     * @return Builder
     */
    public Builder setMaxDegree(int maxDegree) {
      this.maxDegree = maxDegree;
      return this;
    }

    /**
     * @param searchListSize the candidates each search keeps while the graph is built, greater than
     *     zero. Unrelated to the candidates a query keeps, which its {@code ef} sets. Default is
     *     100.
     * @return Builder
     */
    public Builder setSearchListSize(int searchListSize) {
      this.searchListSize = searchListSize;
      return this;
    }

    /**
     * @param alpha the pruning slack of the second build pass, a finite value of at least 1.0; the
     *     first pass prunes at 1.0. Default is 1.2.
     * @return Builder
     */
    public Builder setAlpha(float alpha) {
      this.alpha = alpha;
      return this;
    }

    public VamanaBuildParams build() {
      return new VamanaBuildParams(this);
    }
  }

  public int getMaxDegree() {
    return maxDegree;
  }

  public int getSearchListSize() {
    return searchListSize;
  }

  public float getAlpha() {
    return alpha;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("maxDegree", maxDegree)
        .add("searchListSize", searchListSize)
        .add("alpha", alpha)
        .toString();
  }
}
