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

import java.util.HashMap;
import java.util.Map;

/** Builder-style configuration for Bitmap scalar index parameters. */
public final class BitmapIndexParams {
  private static final String INDEX_TYPE = "bitmap";

  private BitmapIndexParams() {}

  /** Create a new builder for Bitmap index parameters. */
  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private Integer shardId;

    /**
     * Retain a legacy shard ID so index creation reports that shard builds are unsupported.
     *
     * @param shardId non-negative shard identifier
     * @return this builder
     * @throws IllegalArgumentException
     * @deprecated Bitmap builds create independent segments. Omit this option and use the
     *     uncommitted segment build API; index creation rejects any supplied shard ID.
     */
    @Deprecated
    public Builder shardId(int shardId) {
      if (shardId < 0) {
        throw new IllegalArgumentException("shardId must be non-negative");
      }
      this.shardId = shardId;
      return this;
    }

    /** Build a {@link ScalarIndexParams} instance for a Bitmap index. */
    public ScalarIndexParams build() {
      Map<String, Object> params = new HashMap<>();
      if (shardId != null) {
        params.put("shard_id", shardId);
      }

      if (params.isEmpty()) {
        return ScalarIndexParams.create(INDEX_TYPE);
      }

      String json = JsonUtils.toJson(params);
      return ScalarIndexParams.create(INDEX_TYPE, json);
    }
  }
}
