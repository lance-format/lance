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

import javax.annotation.Nullable;

import java.io.Serializable;
import java.util.List;

/**
 * Data of compaction task.
 *
 * <p>A task either rewrites its fragments into new fragments, or, when {@link #getRepackFiles()} is
 * non-null, rewrites some columns of its one fragment into new data files and leaves the fragment
 * id, rows, deletions, overlays and index coverage as they are.
 */
public class TaskData implements Serializable {
  // Pinned to the UID generated before repackFiles was added, so that tasks queued by older workers
  // still deserialize (as fragment rewrites) during a rolling upgrade.
  private static final long serialVersionUID = -4884632518342713596L;

  private final List<FragmentMetadata> fragments;

  // One entry per new data file: the field ids of the top-level columns it holds.
  @Nullable private final List<List<Integer>> repackFiles;

  public TaskData(List<FragmentMetadata> fragments) {
    this(fragments, null);
  }

  public TaskData(List<FragmentMetadata> fragments, @Nullable List<List<Integer>> repackFiles) {
    this.fragments = fragments;
    this.repackFiles = repackFiles;
  }

  public List<FragmentMetadata> getFragments() {
    return fragments;
  }

  /**
   * The new data files of a column repack, each the field ids of the top-level columns it holds.
   *
   * @return null for a task that rewrites its fragments
   */
  @Nullable
  public List<List<Integer>> getRepackFiles() {
    return repackFiles;
  }
}
