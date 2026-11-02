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

/** Statistics returned by dataset cleanup. */
public class RemovalStats {
  private final long bytesRemoved;
  private final long oldVersions;
  private final long dataFilesRemoved;
  private final long transactionFilesRemoved;
  private final long indexFilesRemoved;
  private final long deletionFilesRemoved;
  private final long failedDeletes;

  public RemovalStats(long bytesRemoved, long oldVersions) {
    this(bytesRemoved, oldVersions, 0L, 0L, 0L, 0L, 0L);
  }

  public RemovalStats(
      long bytesRemoved,
      long oldVersions,
      long dataFilesRemoved,
      long transactionFilesRemoved,
      long indexFilesRemoved,
      long deletionFilesRemoved) {
    this(
        bytesRemoved,
        oldVersions,
        dataFilesRemoved,
        transactionFilesRemoved,
        indexFilesRemoved,
        deletionFilesRemoved,
        0L);
  }

  public RemovalStats(
      long bytesRemoved,
      long oldVersions,
      long dataFilesRemoved,
      long transactionFilesRemoved,
      long indexFilesRemoved,
      long deletionFilesRemoved,
      long failedDeletes) {
    this.bytesRemoved = bytesRemoved;
    this.oldVersions = oldVersions;
    this.dataFilesRemoved = dataFilesRemoved;
    this.transactionFilesRemoved = transactionFilesRemoved;
    this.indexFilesRemoved = indexFilesRemoved;
    this.deletionFilesRemoved = deletionFilesRemoved;
    this.failedDeletes = failedDeletes;
  }

  public long getBytesRemoved() {
    return bytesRemoved;
  }

  public long getOldVersions() {
    return oldVersions;
  }

  public long getDataFilesRemoved() {
    return dataFilesRemoved;
  }

  public long getTransactionFilesRemoved() {
    return transactionFilesRemoved;
  }

  public long getIndexFilesRemoved() {
    return indexFilesRemoved;
  }

  public long getDeletionFilesRemoved() {
    return deletionFilesRemoved;
  }

  /**
   * Number of objects cleanup tried and failed to delete.
   *
   * <p>Cleanup is best effort: a failed delete is counted here and the sweep continues, so a
   * non-zero value means the cleanup completed but did not remove everything it identified. The
   * files it could not remove are still unreferenced and a later cleanup will retry them. Callers
   * that treat cleanup as all-or-nothing should check this rather than rely on the call throwing.
   */
  public long getFailedDeletes() {
    return failedDeletes;
  }
}
