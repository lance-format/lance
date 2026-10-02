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
package org.lance.index.vector;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/** Unit tests for {@link HnswBuildParams.Builder} input validation. */
public class HnswBuildParamsTest {
  @Test
  public void testBuilderRejectsNonPositiveParams() {
    // maxLevel, m, and efConstruction must be positive, matching the sibling
    // MemWalHnswParams which carries the same HNSW parameters.
    Assertions.assertThrows(
        IllegalArgumentException.class, () -> new HnswBuildParams.Builder().setMaxLevel((short) 0));
    Assertions.assertThrows(
        IllegalArgumentException.class, () -> new HnswBuildParams.Builder().setM(0));
    Assertions.assertThrows(
        IllegalArgumentException.class, () -> new HnswBuildParams.Builder().setEfConstruction(-1));
    // A valid build still succeeds and keeps the defaults.
    Assertions.assertEquals(20, new HnswBuildParams.Builder().build().getM());
  }
}
