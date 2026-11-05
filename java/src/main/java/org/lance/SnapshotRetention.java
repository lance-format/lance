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

import java.io.IOException;

/**
 * Explicit-release snapshot protection for ordinary scans and deferred blob reads. All maintenance
 * clients must participate. Query tags do not expire on drop or time.
 *
 * <p>Maintenance owners expire after 120 hours by default. The positive integer table config {@code
 * lance.snapshot_retention.lock_expiration_hours} overrides this lifetime in hours. Operations and
 * storage requests must finish before expiry and clocks must agree. Expiration cannot cancel an
 * already submitted delete or fence a paused process.
 *
 * <p>Persist acquisition intent before acquiring, persist the token before issuing work, and record
 * release intent after every consumer and retry has stopped. Retry lost release responses with the
 * saved token. See {@link SnapshotIdentity} for an example.
 */
public final class SnapshotRetention {
  static {
    JniLoader.ensureLoaded();
  }

  private final byte[] encoded;

  private SnapshotRetention(byte[] encoded) {
    this.encoded = encoded;
  }

  /** Protect the exact snapshot before work. retentionId must identify one query lifetime. */
  public static SnapshotRetention acquire(
      Dataset dataset, SnapshotIdentity snapshot, String retentionId) throws IOException {
    Preconditions.checkNotNull(dataset);
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return new SnapshotRetention(acquireNative(dataset, snapshot.encode(), retentionId));
    }
  }

  /** Validate a token without acquiring or renewing protection. */
  public static SnapshotRetention decode(byte[] encoded) {
    return new SnapshotRetention(validateNative(encoded));
  }

  /** Serialize release ownership without storage credentials. */
  public byte[] encode() {
    return encoded.clone();
  }

  /** Exact identity protected by this token. */
  public SnapshotIdentity getSnapshot() {
    return SnapshotIdentity.decode(snapshotNative(encoded));
  }

  /** Protected tag name for diagnostics and recovery. */
  public String getTagName() {
    return tagNative(encoded);
  }

  /** Idempotently release after all workers, reads, and retries have stopped. */
  public void release(Dataset dataset) throws IOException {
    Preconditions.checkNotNull(dataset);
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      releaseNative(dataset, encoded);
    }
  }

  private static native byte[] acquireNative(Dataset dataset, byte[] snapshot, String retentionId)
      throws IOException;

  private static native byte[] validateNative(byte[] encoded);

  private static native byte[] snapshotNative(byte[] encoded);

  private static native String tagNative(byte[] encoded);

  private static native void releaseNative(Dataset dataset, byte[] encoded) throws IOException;
}
