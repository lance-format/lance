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
package org.lance.index;

import java.io.Serializable;
import java.util.Objects;
import java.util.UUID;

/**
 * Statistics at the collecting dataset's read version, not the segment's build version. Detached
 * versions may be negative.
 *
 * <p>The JSON payload is opaque. Natively maintained segments with no fragment coverage have an
 * empty type URI and {@code "{}"} payload.
 */
public final class IndexSegmentStatistics implements Serializable {
  private static final long serialVersionUID = 1L;

  private final long readVersion;
  private final UUID indexUuid;
  private final String indexTypeUri;
  private final String statisticsJson;

  private IndexSegmentStatistics(
      long readVersion, String indexUuid, String indexTypeUri, String statisticsJson) {
    this.readVersion = readVersion;
    this.indexUuid = UUID.fromString(Objects.requireNonNull(indexUuid, "indexUuid"));
    this.indexTypeUri = Objects.requireNonNull(indexTypeUri, "indexTypeUri");
    this.statisticsJson = Objects.requireNonNull(statisticsJson, "statisticsJson");
  }

  public long getReadVersion() {
    return readVersion;
  }

  public UUID getIndexUuid() {
    return indexUuid;
  }

  public String getIndexTypeUri() {
    return indexTypeUri;
  }

  public String getStatisticsJson() {
    return statisticsJson;
  }
}
