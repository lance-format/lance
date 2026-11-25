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
package org.lance.index.scalar;

import org.lance.util.JsonUtils;

import java.util.Collections;

/**
 * Builder-style configuration for FM-Index scalar index parameters.
 *
 * <p>An FM-Index answers {@code contains(column, 'substring')} filters exactly, including
 * single-character patterns.
 */
public final class FmIndexParams {
  private static final String INDEX_TYPE = "fm";

  private FmIndexParams() {}

  /** Create a new builder for FM-Index parameters. */
  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private Integer numSegments;

    /**
     * Split the indexed fragments, in order, into at most this many segments, each an independent
     * FM-Index. There are never more segments than fragments.
     *
     * <p>The default is one segment. Only applies when the index type is {@code IndexType.FM}, and
     * is ignored when the build is restricted to fragment ids, which always writes one segment.
     *
     * @param numSegments number of segments, must be positive
     * @return this builder
     * @throws IllegalArgumentException if {@code numSegments} is not positive
     */
    public Builder numSegments(int numSegments) {
      if (numSegments <= 0) {
        throw new IllegalArgumentException("numSegments must be positive");
      }
      this.numSegments = numSegments;
      return this;
    }

    /** Build a {@link ScalarIndexParams} instance for an FM-Index. */
    public ScalarIndexParams build() {
      if (numSegments == null) {
        return ScalarIndexParams.create(INDEX_TYPE);
      }
      return ScalarIndexParams.create(
          INDEX_TYPE,
          JsonUtils.toJson(Collections.<String, Object>singletonMap("num_segments", numSegments)));
    }
  }
}
