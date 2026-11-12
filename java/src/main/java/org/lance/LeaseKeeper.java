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

import org.apache.arrow.util.Preconditions;

/**
 * Renews a retention lease in the background while long-running work uses it.
 *
 * <p>The keeper never interrupts the work: a failed renewal is retried, and a lease that ends
 * anyway is reported through {@link #getStatus()}. Whether the work can still be published is
 * decided when it commits. Publishing a result for the lease (for example with {@link
 * Dataset#commitExistingIndexSegments(String, String, java.util.List, RetentionLease)} or {@link
 * Transaction.Builder#forLease}) releases it.
 *
 * <p>Each renewal extends the lease by its original time to live ({@code expiresAt - createdAt}),
 * once half of the remaining time has passed. {@link #close()} stops renewing without releasing the
 * lease; call {@link #release()} to abandon the work instead.
 *
 * <pre>{@code
 * RetentionLease lease = dataset.acquireRetentionLease(LeaseOptions.builder().build());
 * try (Dataset leased = dataset.checkoutLease(lease);
 *     LeaseKeeper keeper = LeaseKeeper.start(leased, lease)) {
 *   List<Index> segments = buildSegments(leased);
 *   keeper.getStatus().getLostReason().ifPresent(reason -> log.warn("may conflict: " + reason));
 *   dataset.commitExistingIndexSegments("value_idx", "value", segments, keeper.getLease());
 * }
 * }</pre>
 */
public final class LeaseKeeper implements AutoCloseable {
  static {
    JniLoader.ensureLoaded();
  }

  private long nativeLeaseKeeperHandle;

  private RetentionLease stoppedLease;
  private LeaseStatus stoppedStatus;

  private final LockManager lockManager = new LockManager();

  // Constructed from native code only.
  private LeaseKeeper() {}

  /**
   * Start renewing {@code lease} on {@code dataset}'s branch.
   *
   * @param dataset a dataset on the lease's branch, typically the leased dataset from {@link
   *     Dataset#checkoutLease}
   * @param lease the lease to renew
   * @return the running keeper
   */
  public static LeaseKeeper start(Dataset dataset, RetentionLease lease) {
    Preconditions.checkNotNull(dataset, "dataset cannot be null");
    Preconditions.checkNotNull(lease, "lease cannot be null");
    return nativeStart(dataset, lease);
  }

  private static native LeaseKeeper nativeStart(Dataset dataset, RetentionLease lease);

  /**
   * @return the lease as of its latest renewal
   */
  public RetentionLease getLease() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      return nativeLeaseKeeperHandle == 0 ? stoppedLease : nativeGetLease();
    }
  }

  private native RetentionLease nativeGetLease();

  /**
   * @return whether the keeper still holds its lease
   */
  public LeaseStatus getStatus() {
    try (LockManager.ReadLock readLock = lockManager.acquireReadLock()) {
      return nativeLeaseKeeperHandle == 0 ? stoppedStatus : new LeaseStatus(nativeGetLostReason());
    }
  }

  private native String nativeGetLostReason();

  /**
   * Stop renewing and release the lease, for example when the work is abandoned. Not needed after
   * publishing, since publication releases the lease.
   *
   * @throws IllegalStateException if the keeper was already closed or released
   */
  public void release() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      Preconditions.checkState(
          nativeLeaseKeeperHandle != 0,
          "LeaseKeeper is closed; release its lease with Dataset.releaseRetentionLease");
      recordStoppedState();
      nativeRelease();
    }
  }

  private native void nativeRelease();

  /** Stop renewing without releasing the lease. */
  @Override
  public void close() {
    try (LockManager.WriteLock writeLock = lockManager.acquireWriteLock()) {
      if (nativeLeaseKeeperHandle != 0) {
        recordStoppedState();
        nativeClose();
      }
    }
  }

  private native void nativeClose();

  private void recordStoppedState() {
    stoppedLease = nativeGetLease();
    stoppedStatus = new LeaseStatus(nativeGetLostReason());
  }
}
