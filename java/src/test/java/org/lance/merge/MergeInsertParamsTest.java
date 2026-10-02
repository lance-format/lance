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
package org.lance.merge;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Collections;

/** Unit tests for {@link MergeInsertParams} input validation. */
public class MergeInsertParamsTest {
  @Test
  public void testRejectsNullJoinKeysAndNegativeRetries() {
    // Null join keys, negative retries, and a negative timeout must be rejected up
    // front, matching sibling UpdateParams. An empty `on` stays valid (the core
    // falls back to the primary key), so it is deliberately not rejected.
    NullPointerException nullOn =
        Assertions.assertThrows(NullPointerException.class, () -> new MergeInsertParams(null));
    Assertions.assertTrue(nullOn.getMessage().contains("on must not be null"));

    MergeInsertParams params = new MergeInsertParams(Collections.singletonList("id"));
    IllegalArgumentException negativeRetries =
        Assertions.assertThrows(
            IllegalArgumentException.class, () -> params.withConflictRetries(-1));
    Assertions.assertTrue(negativeRetries.getMessage().contains("non-negative"));
    IllegalArgumentException negativeTimeout =
        Assertions.assertThrows(
            IllegalArgumentException.class, () -> params.withRetryTimeoutMs(-1L));
    Assertions.assertTrue(negativeTimeout.getMessage().contains("non-negative"));

    Assertions.assertTrue(new MergeInsertParams(Collections.emptyList()).on().isEmpty());
  }
}
