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
package org.lance.ipc;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Summary counts collected after executing a scan plan.
 *
 * <p>These statistics are populated when the scan stream is fully consumed and closed.
 */
public final class ScanStats {
  private final long iops;
  private final long requests;
  private final long bytesRead;
  private final long indicesLoaded;
  private final long partsLoaded;
  private final long indexComparisons;

  /** Number of index cache page lookups served from memory in this scan. */
  private final long indexCacheHits;

  /** Number of index cache page lookups that had to load from storage in this scan. */
  private final long indexCacheMisses;

  private final Map<String, Long> allCounts;
  private final Map<String, Long> allTimes;

  public ScanStats(
      long iops,
      long requests,
      long bytesRead,
      long indicesLoaded,
      long partsLoaded,
      long indexComparisons,
      long indexCacheHits,
      long indexCacheMisses,
      Map<String, Long> allCounts,
      Map<String, Long> allTimes) {
    this.iops = iops;
    this.requests = requests;
    this.bytesRead = bytesRead;
    this.indicesLoaded = indicesLoaded;
    this.partsLoaded = partsLoaded;
    this.indexComparisons = indexComparisons;
    this.indexCacheHits = indexCacheHits;
    this.indexCacheMisses = indexCacheMisses;
    this.allCounts = freezeMap(allCounts);
    this.allTimes = freezeMap(allTimes);
  }

  /**
   * Backwards-compatible constructor kept for existing callers that predate the addition of
   * per-query index cache statistics. New code should use the 10-argument constructor that also
   * accepts {@code indexCacheHits} and {@code indexCacheMisses}.
   *
   * @deprecated Use {@link #ScanStats(long, long, long, long, long, long, long, long, Map, Map)}.
   */
  @Deprecated
  public ScanStats(
      long iops,
      long requests,
      long bytesRead,
      long indicesLoaded,
      long partsLoaded,
      long indexComparisons,
      Map<String, Long> allCounts,
      Map<String, Long> allTimes) {
    this(
        iops,
        requests,
        bytesRead,
        indicesLoaded,
        partsLoaded,
        indexComparisons,
        0L,
        0L,
        allCounts,
        allTimes);
  }

  private static <K, V> Map<K, V> freezeMap(Map<K, V> map) {
    if (map == null || map.isEmpty()) {
      return Collections.emptyMap();
    }
    return Collections.unmodifiableMap(new HashMap<>(map));
  }

  public long getIops() {
    return iops;
  }

  public long getRequests() {
    return requests;
  }

  public long getBytesRead() {
    return bytesRead;
  }

  public long getIndicesLoaded() {
    return indicesLoaded;
  }

  public long getPartsLoaded() {
    return partsLoaded;
  }

  public long getIndexComparisons() {
    return indexComparisons;
  }

  /**
   * Number of index cache page lookups where the loader was not executed in this scan.
   *
   * <p>Counts both true cache hits on already-populated entries and coalesced concurrent loads (a
   * follower attached to another caller's in-flight load).
   *
   * <p>Instrumented boundaries in this release: BTree, IVF v2 (write-cache scan path), inverted
   * posting list (grouped and per-token) and its per-token metadata, inverted phrase positions,
   * bitmap (Equals / Range / IsIn), ngram, rtree.
   *
   * <p>Caveats:
   *
   * <ul>
   *   <li>IVF v2 streaming scans and legacy v1 IVF partitions bypass the cache by design and are
   *       therefore reported as a miss on every call.
   *   <li>A cold posting-list lookup on the grouped inverted layout can record up to two misses
   *       (group + per-token metadata) for a single term.
   * </ul>
   *
   * <p>Uninstrumented paths (HNSW graph pages, quantizer codebooks) do not contribute to either
   * counter. See the sibling {@link #getIndexCacheMisses()} for the paired counter.
   */
  public long getIndexCacheHits() {
    return indexCacheHits;
  }

  /**
   * Number of index cache page lookups where the loader ran in this scan (the page was not resident
   * and had to be materialised, typically from storage). See {@link #getIndexCacheHits()} for the
   * paired counter and the list of instrumented boundaries.
   */
  public long getIndexCacheMisses() {
    return indexCacheMisses;
  }

  public Map<String, Long> getAllCounts() {
    return allCounts;
  }

  public Map<String, Long> getAllTimes() {
    return allTimes;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof ScanStats)) {
      return false;
    }
    ScanStats that = (ScanStats) o;
    return iops == that.iops
        && requests == that.requests
        && bytesRead == that.bytesRead
        && indicesLoaded == that.indicesLoaded
        && partsLoaded == that.partsLoaded
        && indexComparisons == that.indexComparisons
        && indexCacheHits == that.indexCacheHits
        && indexCacheMisses == that.indexCacheMisses
        && Objects.equals(allCounts, that.allCounts)
        && Objects.equals(allTimes, that.allTimes);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        iops,
        requests,
        bytesRead,
        indicesLoaded,
        partsLoaded,
        indexComparisons,
        indexCacheHits,
        indexCacheMisses,
        allCounts,
        allTimes);
  }

  @Override
  public String toString() {
    return "ScanStats{"
        + "iops="
        + iops
        + ", requests="
        + requests
        + ", bytesRead="
        + bytesRead
        + ", indicesLoaded="
        + indicesLoaded
        + ", partsLoaded="
        + partsLoaded
        + ", indexComparisons="
        + indexComparisons
        + ", indexCacheHits="
        + indexCacheHits
        + ", indexCacheMisses="
        + indexCacheMisses
        + ", allCounts="
        + allCounts
        + ", allTimes="
        + allTimes
        + '}';
  }
}
