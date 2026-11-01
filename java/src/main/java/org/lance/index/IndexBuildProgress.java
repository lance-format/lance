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
package org.lance.index;

import java.util.Optional;

/** Receives stage-level progress while Lance builds or finalizes an index. */
public interface IndexBuildProgress {

  /**
   * Reports that a stage has started.
   *
   * <p>Implementations must be thread-safe. Lance may invoke callbacks concurrently from native
   * runtime threads. Callbacks may re-enter read-only methods on the same {@code Dataset} through
   * JNI. Conflicting write re-entry from a callback is rejected; unrelated concurrent callers keep
   * their normal wait behavior. Concurrent create/merge index builds on the same Dataset handle are
   * serialized. An exception thrown by this method terminates the index operation.
   *
   * @param stage stable, index-type-specific stage name
   * @param total number of work units, or empty when the total is unknown
   * @param unit description of the work unit, such as {@code files} or {@code partitions}
   */
  void stageStart(String stage, Optional<Long> total, String unit);

  /**
   * Reports completed work within a stage.
   *
   * <p>An exception thrown by this method terminates the index operation.
   *
   * @param stage stage name previously reported to {@link #stageStart}
   * @param completed number of completed work units
   */
  void stageProgress(String stage, long completed);

  /**
   * Reports that a stage has completed.
   *
   * <p>The stage work has already succeeded when this method is called. Lance therefore logs and
   * ignores exceptions thrown by this callback instead of failing the completed operation.
   *
   * @param stage completed stage name
   */
  void stageComplete(String stage);
}
