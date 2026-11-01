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
package org.lance.fragment;

import org.lance.FragmentMetadata;

import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Equality contract of fragment metadata value classes. */
public class DataFileEqualityTest {

  private static DataFile dataFile(Integer baseId) {
    return new DataFile("data/a.lance", new int[] {0, 1}, new int[] {0, 1}, 2, 1, 1024L, baseId);
  }

  private static DeletionFile deletionFile(Integer baseId) {
    return new DeletionFile(7L, 3L, 5L, DeletionFileType.BITMAP, baseId);
  }

  @Test
  void testDataFileEqualsAndHashCodeIncludeBaseId() {
    for (Integer baseId : new Integer[] {null, 1}) {
      assertEquals(dataFile(baseId), dataFile(baseId));
      assertEquals(dataFile(baseId).hashCode(), dataFile(baseId).hashCode());
    }
    // Same relative path under different bases refers to different physical files.
    assertNotEquals(dataFile(1), dataFile(2));
    assertNotEquals(dataFile(null), dataFile(1));
  }

  @Test
  void testDeletionFileEqualsAndHashCodeIncludeBaseId() {
    for (Integer baseId : new Integer[] {null, 1}) {
      assertEquals(deletionFile(baseId), deletionFile(baseId));
      assertEquals(deletionFile(baseId).hashCode(), deletionFile(baseId).hashCode());
    }
    assertNotEquals(deletionFile(1), deletionFile(2));
    assertNotEquals(deletionFile(null), deletionFile(1));
  }

  @Test
  void testEqualFragmentMetadataWorksInHashSet() {
    FragmentMetadata a =
        new FragmentMetadata(
            0, Collections.singletonList(dataFile(null)), 100L, deletionFile(null), null);
    FragmentMetadata b =
        new FragmentMetadata(
            0, Collections.singletonList(dataFile(null)), 100L, deletionFile(null), null);
    assertEquals(a, b);
    assertEquals(a.hashCode(), b.hashCode());
    Set<FragmentMetadata> set = new HashSet<>(Collections.singletonList(a));
    assertTrue(set.contains(b));
  }
}
