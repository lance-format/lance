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
package org.lance;

/**
 * Statistics for a session cache.
 *
 * <p>A snapshot of cache activity, useful for monitoring hit/miss rates over time.
 */
public class CacheStats {
  private final long hits;
  private final long misses;
  private final long numEntries;
  private final long sizeBytes;

  /**
   * Constructs cache statistics. Instances are created by the native layer.
   *
   * @param hits number of cache lookups that found an item
   * @param misses number of cache lookups that did not find an item
   * @param numEntries number of entries currently in the cache
   * @param sizeBytes total size in bytes of all entries in the cache
   */
  public CacheStats(long hits, long misses, long numEntries, long sizeBytes) {
    if (hits < 0 || misses < 0 || numEntries < 0 || sizeBytes < 0) {
      throw new IllegalArgumentException(
          String.format(
              "Cache statistics must be non-negative: "
                  + "hits=%d, misses=%d, numEntries=%d, sizeBytes=%d",
              hits, misses, numEntries, sizeBytes));
    }
    this.hits = hits;
    this.misses = misses;
    this.numEntries = numEntries;
    this.sizeBytes = sizeBytes;
  }

  /**
   * Returns the number of cache lookups that found an item in the cache.
   *
   * @return the number of cache hits
   */
  public long getHits() {
    return hits;
  }

  /**
   * Returns the number of cache lookups that did not find an item in the cache.
   *
   * @return the number of cache misses
   */
  public long getMisses() {
    return misses;
  }

  /**
   * Returns the number of entries currently in the cache.
   *
   * @return the number of cache entries
   */
  public long getNumEntries() {
    return numEntries;
  }

  /**
   * Returns the total size in bytes of all entries in the cache.
   *
   * @return the cache size in bytes
   */
  public long getSizeBytes() {
    return sizeBytes;
  }

  /**
   * Returns the ratio of hits to total lookups, or 0 if there have been no lookups.
   *
   * @return the cache hit ratio in the range [0, 1]
   */
  public double getHitRatio() {
    // Sum in double to avoid long overflow for very large counters
    double total = (double) hits + (double) misses;
    if (total == 0) {
      return 0.0;
    }
    return hits / total;
  }

  @Override
  public String toString() {
    return String.format(
        "CacheStats(hits=%d, misses=%d, numEntries=%d, sizeBytes=%d)",
        hits, misses, numEntries, sizeBytes);
  }
}
