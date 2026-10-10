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
package org.lance.compaction;

import org.lance.FragmentMetadata;
import org.lance.fragment.DataFile;

import javax.annotation.Nullable;

import java.io.Serializable;
import java.util.List;

/**
 * Rewrite Result of a single compaction task. It will be passed across different workers and be
 * committed later.
 */
public class RewriteResult implements Serializable {
  // Pinned to the UID generated before the repack fields were added, so that results produced by
  // older workers still deserialize during a rolling upgrade.
  private static final long serialVersionUID = 4501818269828675274L;

  private final CompactionMetrics metrics;
  private final List<FragmentMetadata> newFragments;
  private final List<FragmentMetadata> originalFragments;
  private final long readVersion;

  // Serialized RoaringTreemap of row addresses read from the original fragments.
  // null for stable row IDs.
  @Nullable private final byte[] rowAddrs;

  // Set only for a column repack: the fragment it repacked and the data files it wrote.
  @Nullable private final Long repackedFragmentId;
  @Nullable private final List<DataFile> repackedFiles;

  public RewriteResult(
      CompactionMetrics metrics,
      List<FragmentMetadata> newFragments,
      List<FragmentMetadata> originalFragments,
      long readVersion,
      byte[] rowAddrs) {
    this(metrics, newFragments, originalFragments, readVersion, rowAddrs, null, null);
  }

  public RewriteResult(
      CompactionMetrics metrics,
      List<FragmentMetadata> newFragments,
      List<FragmentMetadata> originalFragments,
      long readVersion,
      byte[] rowAddrs,
      @Nullable Long repackedFragmentId,
      @Nullable List<DataFile> repackedFiles) {
    this.metrics = metrics;
    this.newFragments = newFragments;
    this.originalFragments = originalFragments;
    this.readVersion = readVersion;
    this.rowAddrs = rowAddrs;
    this.repackedFragmentId = repackedFragmentId;
    this.repackedFiles = repackedFiles;
  }

  /**
   * The fragment a column repack wrote new data files for.
   *
   * @return null for the result of a fragment rewrite
   */
  @Nullable
  public Long getRepackedFragmentId() {
    return repackedFragmentId;
  }

  /**
   * The data files a column repack wrote, empty when the fragment had no rows to write.
   *
   * @return null for the result of a fragment rewrite
   */
  @Nullable
  public List<DataFile> getRepackedFiles() {
    return repackedFiles;
  }

  public long getReadVersion() {
    return readVersion;
  }

  public CompactionMetrics getMetrics() {
    return metrics;
  }

  @Nullable
  public byte[] getRowAddrs() {
    return rowAddrs;
  }

  public List<FragmentMetadata> getNewFragments() {
    return newFragments;
  }

  public List<FragmentMetadata> getOriginalFragments() {
    return originalFragments;
  }
}
