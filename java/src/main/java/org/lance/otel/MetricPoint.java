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
package org.lance.otel;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/** A cumulative Lance metric data point captured by the native metrics recorder. */
public final class MetricPoint {
  private final String name;
  private final String kind;
  private final Map<String, String> attributes;
  private final Double value;
  private final List<MetricBucket> buckets;
  private final Long count;
  private final Double sum;

  public MetricPoint(
      String name,
      String kind,
      Map<String, String> attributes,
      Double value,
      List<MetricBucket> buckets,
      Long count,
      Double sum) {
    this.name = name;
    this.kind = kind;
    this.attributes =
        attributes == null ? Collections.emptyMap() : Collections.unmodifiableMap(attributes);
    this.value = value;
    this.buckets = buckets == null ? null : Collections.unmodifiableList(buckets);
    this.count = count;
    this.sum = sum;
  }

  public String getName() {
    return name;
  }

  public String getKind() {
    return kind;
  }

  public Map<String, String> getAttributes() {
    return attributes;
  }

  public Double getValue() {
    return value;
  }

  public List<MetricBucket> getBuckets() {
    return buckets;
  }

  public Long getCount() {
    return count;
  }

  public Double getSum() {
    return sum;
  }
}
