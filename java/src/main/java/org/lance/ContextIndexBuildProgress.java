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

import org.lance.index.IndexBuildProgress;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Optional;
import java.util.Set;

/**
 * Marks the current thread as executing an index progress callback for one or more Datasets.
 *
 * <p>The active contexts are tracked by Dataset identity and nesting count. This lets Dataset give
 * callback threads a scoped read lease and reject conflicting write re-entry without rejecting
 * unrelated concurrent callers.
 */
final class ContextIndexBuildProgress implements IndexBuildProgress {
  private static final ThreadLocal<IdentityHashMap<Dataset, int[]>> ACTIVE_CALLBACKS =
      new ThreadLocal<>();

  private final Dataset dataset;
  private final IndexBuildProgress delegate;
  private final Set<Dataset> inheritedDatasets;

  ContextIndexBuildProgress(Dataset dataset, IndexBuildProgress delegate) {
    this.dataset = dataset;
    this.delegate = delegate;
    IdentityHashMap<Dataset, int[]> activeCallbacks = ACTIVE_CALLBACKS.get();
    if (activeCallbacks == null || activeCallbacks.isEmpty()) {
      this.inheritedDatasets = Collections.emptySet();
    } else {
      Set<Dataset> inheritedDatasets = Collections.newSetFromMap(new IdentityHashMap<>());
      inheritedDatasets.addAll(activeCallbacks.keySet());
      this.inheritedDatasets = inheritedDatasets;
    }
  }

  static boolean isActive(Dataset dataset) {
    IdentityHashMap<Dataset, int[]> activeCallbacks = ACTIVE_CALLBACKS.get();
    return activeCallbacks != null && activeCallbacks.containsKey(dataset);
  }

  static boolean isCallbackActive() {
    IdentityHashMap<Dataset, int[]> activeCallbacks = ACTIVE_CALLBACKS.get();
    return activeCallbacks != null && !activeCallbacks.isEmpty();
  }

  private static IdentityHashMap<Dataset, int[]> activeCallbacks() {
    IdentityHashMap<Dataset, int[]> activeCallbacks = ACTIVE_CALLBACKS.get();
    if (activeCallbacks == null) {
      activeCallbacks = new IdentityHashMap<>();
      ACTIVE_CALLBACKS.set(activeCallbacks);
    }
    return activeCallbacks;
  }

  private static void begin(Dataset dataset) {
    IdentityHashMap<Dataset, int[]> activeCallbacks = activeCallbacks();
    activeCallbacks.computeIfAbsent(dataset, key -> new int[1])[0]++;
  }

  private static void end(Dataset dataset) {
    IdentityHashMap<Dataset, int[]> activeCallbacks = ACTIVE_CALLBACKS.get();
    int[] count = activeCallbacks.get(dataset);
    if (count == null) {
      throw new IllegalStateException("Callback context is not active");
    }
    count[0]--;
    if (count[0] == 0) {
      activeCallbacks.remove(dataset);
    }
    if (activeCallbacks.isEmpty()) {
      ACTIVE_CALLBACKS.remove();
    }
  }

  private void beginCallback() {
    for (Dataset inheritedDataset : inheritedDatasets) {
      begin(inheritedDataset);
    }
    begin(dataset);
  }

  private void endCallback() {
    // Release in reverse acquisition order so nested contexts unwind cleanly.
    end(dataset);
    for (Dataset inheritedDataset : inheritedDatasets) {
      end(inheritedDataset);
    }
  }

  @Override
  public void stageStart(String stage, Optional<Long> total, String unit) {
    beginCallback();
    try {
      delegate.stageStart(stage, total, unit);
    } finally {
      endCallback();
    }
  }

  @Override
  public void stageProgress(String stage, long completed) {
    beginCallback();
    try {
      delegate.stageProgress(stage, completed);
    } finally {
      endCallback();
    }
  }

  @Override
  public void stageComplete(String stage) {
    beginCallback();
    try {
      delegate.stageComplete(stage);
    } finally {
      endCallback();
    }
  }
}
