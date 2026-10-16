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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/** Unit tests for {@link CompactionOptions.Builder} input validation. */
public class CompactionOptionsTest {
  @Test
  public void testBuilderRejectsNonPositiveThreadsAndBatchSize() {
    // numThreads and batchSize must be positive; a negative value would wrap to a
    // huge unsigned value in the JNI layer, matching the maxSource* budgets.
    Assertions.assertThrows(
        IllegalArgumentException.class, () -> CompactionOptions.builder().withNumThreads(0));
    Assertions.assertThrows(
        IllegalArgumentException.class, () -> CompactionOptions.builder().withNumThreads(-1));
    Assertions.assertThrows(
        IllegalArgumentException.class, () -> CompactionOptions.builder().withBatchSize(0));
    Assertions.assertThrows(
        IllegalArgumentException.class, () -> CompactionOptions.builder().withBatchSize(-1));
    // A positive setting still builds.
    CompactionOptions.builder().withNumThreads(4).withBatchSize(1024).build();
  }
}
