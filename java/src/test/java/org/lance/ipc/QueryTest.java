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
package org.lance.ipc;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/** Unit tests for {@link Query.Builder} input validation. */
public class QueryTest {
  @Test
  public void testBuilderRejectsNonPositiveEfAndRefineFactor() {
    // ef (HNSW candidate count) and refineFactor must be positive, like the other
    // search parameters the Query constructor validates.
    Assertions.assertThrows(IllegalArgumentException.class, () -> baseBuilder().setEf(0).build());
    Assertions.assertThrows(IllegalArgumentException.class, () -> baseBuilder().setEf(-1).build());
    Assertions.assertThrows(
        IllegalArgumentException.class, () -> baseBuilder().setRefineFactor(0).build());
    Assertions.assertThrows(
        IllegalArgumentException.class, () -> baseBuilder().setRefineFactor(-1).build());
    // Positive values, and leaving them unset, still build.
    baseBuilder().setEf(100).setRefineFactor(2).build();
    baseBuilder().build();
  }

  private static Query.Builder baseBuilder() {
    return new Query.Builder().setColumn("vector").setKey(new float[] {1.0f, 2.0f});
  }
}
