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

import java.util.concurrent.locks.ReentrantReadWriteLock;

/** The LockManager class provides a way to manage read and write locks. */
public class LockManager {
  private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
  private final Dataset dataset;

  public LockManager() {
    this(null);
  }

  LockManager(Dataset dataset) {
    this.dataset = dataset;
  }

  /**
   * Represents a read lock for the LockManager. This lock allows multiple threads to read
   * concurrently, but prevents write access.
   */
  public class ReadLock implements AutoCloseable {
    private final boolean callbackLease;

    /** Acquires a read lock on the lock manager. */
    public ReadLock() {
      // The outer index create already pins the Dataset with a read lock. Callback re-entry uses
      // that lease instead of blocking behind a writer queued on the same lock.
      callbackLease = dataset != null && ContextIndexBuildProgress.isActive(dataset);
      if (callbackLease) {
        return;
      }
      if (isCallbackFromAnotherDataset()) {
        // An untimed tryLock can join active readers even when a writer is queued, but never waits
        // for another Dataset whose operation may itself be waiting for this callback.
        if (!lock.readLock().tryLock()) {
          throw new IllegalStateException(
              "Dataset lifecycle read lock is busy in an index progress callback");
        }
      } else {
        lock.readLock().lock();
      }
    }

    @Override
    public void close() {
      if (!callbackLease) {
        lock.readLock().unlock();
      }
    }
  }

  /** Represents a write lock that can be acquired and released. */
  public class WriteLock implements AutoCloseable {
    /** Constructs a new WriteLock and acquires the write lock. */
    public WriteLock() {
      if (dataset != null && ContextIndexBuildProgress.isActive(dataset)) {
        throw new IllegalStateException("Dataset is busy in an index progress callback");
      }
      if (isCallbackFromAnotherDataset()) {
        if (!lock.writeLock().tryLock()) {
          throw new IllegalStateException(
              "Dataset lifecycle write lock is busy in an index progress callback");
        }
      } else {
        lock.writeLock().lock();
      }
    }

    @Override
    public void close() {
      lock.writeLock().unlock();
    }
  }

  /**
   * Acquires a read lock on the LockManager.
   *
   * @return the acquired ReadLock object
   */
  public ReadLock acquireReadLock() {
    return new ReadLock();
  }

  /**
   * Acquires a write lock on the LockManager.
   *
   * @return the acquired WriteLock object
   */
  public WriteLock acquireWriteLock() {
    return new WriteLock();
  }

  private boolean isCallbackFromAnotherDataset() {
    return dataset != null
        && ContextIndexBuildProgress.isCallbackActive()
        && !ContextIndexBuildProgress.isActive(dataset);
  }
}
