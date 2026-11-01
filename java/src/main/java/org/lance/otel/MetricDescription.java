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

/** Metadata for one Lance metric available through the OpenTelemetry bridge. */
public final class MetricDescription {
  private final String name;
  private final String kind;
  private final String unit;
  private final String description;

  public MetricDescription(String name, String kind, String unit, String description) {
    this.name = name;
    this.kind = kind;
    this.unit = unit;
    this.description = description;
  }

  public String getName() {
    return name;
  }

  public String getKind() {
    return kind;
  }

  public String getUnit() {
    return unit;
  }

  public String getDescription() {
    return description;
  }
}
