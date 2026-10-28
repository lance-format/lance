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
 * Exact dataset snapshot identity shared by ordinary scans and blob references. Contains no
 * credentials. Holding or decoding this value does not protect files.
 *
 * <pre>{@code
 * SnapshotIdentity snapshot = SnapshotIdentity.fromDataset(dataset, "catalog/table-incarnation");
 * SnapshotRetention retention = SnapshotRetention.acquire(dataset, snapshot, uniqueQueryId);
 * // Run readers and wait for every consumer and retry to stop.
 * retention.release(dataset);
 * }</pre>
 */
public final class SnapshotIdentity {
  static {
    JniLoader.ensureLoaded();
  }

  private final byte[] encoded;

  private SnapshotIdentity(byte[] encoded) {
    this.encoded = encoded;
  }

  /** Capture the loaded snapshot; tableIdentity must distinguish table incarnations. */
  public static SnapshotIdentity fromDataset(Dataset dataset, String tableIdentity)
      throws IOException {
    Preconditions.checkNotNull(dataset);
    try (LockManager.ReadLock readLock = dataset.acquireReadLock()) {
      return new SnapshotIdentity(fromDatasetNative(dataset, tableIdentity));
    }
  }

  /** Validate a transported identity without acquiring protection. */
  public static SnapshotIdentity decode(byte[] encoded) {
    return new SnapshotIdentity(validateNative(encoded));
  }

  /** Serialize for another process or a durable recovery record. */
  public byte[] encode() {
    return encoded.clone();
  }

  private static native byte[] fromDatasetNative(Dataset dataset, String tableIdentity)
      throws IOException;

  private static native byte[] validateNative(byte[] encoded);
}
