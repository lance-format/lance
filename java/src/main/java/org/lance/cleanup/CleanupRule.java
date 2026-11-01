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
package org.lance.cleanup;

import org.apache.arrow.util.Preconditions;

import java.time.Duration;
import java.util.Map;

/** An immutable age-band rule for cleanup, ordered by increasing olderThan age. */
public final class CleanupRule {
  private final Duration perInterval;
  private final Duration olderThan;

  private CleanupRule(Duration perInterval, Duration olderThan) {
    this.perInterval = perInterval;
    this.olderThan = Preconditions.checkNotNull(olderThan, "olderThan cannot be null");
    Preconditions.checkArgument(
        !olderThan.isNegative() && olderThan.equals(Duration.ofMillis(olderThan.toMillis())),
        "olderThan must be non-negative with millisecond precision, got %s",
        olderThan);
  }

  /**
   * Keep one newest eligible commit per UTC epoch-aligned interval after olderThan. The age must be
   * at least two days, relative to the hinted commit (or snapshot without a hint). The oldest
   * applicable rule selects the interval. Tags, current versions and branch dependencies stay
   * protected.
   *
   * @param perInterval positive interval with millisecond precision
   * @param olderThan thinning age with millisecond precision, at least two days
   * @return immutable sampling rule; policy validation occurs on explain/execute
   */
  public static CleanupRule keepOne(Duration perInterval, Duration olderThan) {
    Preconditions.checkNotNull(perInterval, "perInterval cannot be null");
    Preconditions.checkArgument(
        perInterval.toMillis() > 0 && perInterval.equals(Duration.ofMillis(perInterval.toMillis())),
        "keep_one.per_interval must be positive with millisecond precision, got %s",
        perInterval);
    return new CleanupRule(perInterval, olderThan);
  }

  /**
   * Expire all eligible versions after a wall-clock age, including sampled commits. This rule must
   * be last. Existing current/tag/branch protections remain.
   *
   * @param olderThan non-negative expiration age with millisecond precision
   * @return immutable expiration rule
   */
  public static CleanupRule removeAll(Duration olderThan) {
    return new CleanupRule(null, olderThan);
  }

  /** Internal configuration form shared with automatic cleanup. */
  public Map<String, Object> toConfig() {
    if (perInterval == null) {
      return Map.of("type", "remove_all", "older_than", durationString(olderThan));
    }
    return Map.of(
        "type",
        "keep_one",
        "per_interval",
        durationString(perInterval),
        "older_than",
        durationString(olderThan));
  }

  private static String durationString(Duration duration) {
    return duration.getSeconds() + "s " + duration.getNano() + "ns";
  }
}
