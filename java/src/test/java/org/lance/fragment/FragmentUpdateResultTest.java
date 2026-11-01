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

import org.lance.CommitBuilder;
import org.lance.Dataset;
import org.lance.Fragment;
import org.lance.FragmentMetadata;
import org.lance.TestUtils;
import org.lance.Transaction;
import org.lance.operation.Append;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FragmentUpdateResultTest {

  /** Portable RoaringBitmap bytes for offsets {1, 3, 5} (see UpdateTest round-trip fixture). */
  private static final byte[] PORTABLE_ROARING_BYTES_135 =
      new byte[] {
        (byte) 0x3A,
        (byte) 0x30,
        (byte) 0x00,
        (byte) 0x00,
        (byte) 0x01,
        (byte) 0x00,
        (byte) 0x00,
        (byte) 0x00,
        (byte) 0x00,
        (byte) 0x00,
        (byte) 0x02,
        (byte) 0x00,
        (byte) 0x10,
        (byte) 0x00,
        (byte) 0x00,
        (byte) 0x00,
        (byte) 0x01,
        (byte) 0x00,
        (byte) 0x03,
        (byte) 0x00,
        (byte) 0x05,
        (byte) 0x00
      };

  @Test
  void testGetUpdatedRowOffsetBytesRoundTripViaDeprecatedGetter() {
    FragmentUpdateResult result =
        FragmentUpdateResult.create(null, new long[0], PORTABLE_ROARING_BYTES_135);
    assertArrayEquals(PORTABLE_ROARING_BYTES_135, result.getUpdatedRowOffsetBytes());
    assertArrayEquals(new long[] {1, 3, 5}, result.getUpdatedRowOffsets());
  }

  @Test
  void testDeprecatedLongArrayConstructorEncodesToBytes() {
    FragmentUpdateResult result = new FragmentUpdateResult(null, new long[0], new long[] {1, 3, 5});
    assertArrayEquals(new long[] {1, 3, 5}, result.getUpdatedRowOffsets());

    // Stored bytes from the deprecated long[] constructor decode to the same offsets.
    FragmentUpdateResult fromEncodedBytes =
        FragmentUpdateResult.create(null, new long[0], result.getUpdatedRowOffsetBytes());
    assertArrayEquals(new long[] {1, 3, 5}, fromEncodedBytes.getUpdatedRowOffsets());
  }

  @Test
  void testUpdateColumnsReturnsMatchedRowOffsetBytes(@TempDir Path tempDir) throws Exception {
    String datasetPath = tempDir.resolve("testUpdateColumnsRowOffsetBytes").toString();
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      TestUtils.UpdateColumnTestDataset testDataset =
          new TestUtils.UpdateColumnTestDataset(allocator, datasetPath);
      try (Dataset dataset = testDataset.createEmptyDataset()) {
        FragmentMetadata fragmentMeta = testDataset.createNewFragment(6);
        try (Transaction appendTxn =
            new Transaction.Builder()
                .readVersion(dataset.version())
                .operation(
                    Append.builder().fragments(Collections.singletonList(fragmentMeta)).build())
                .build()) {
          try (Dataset appended = new CommitBuilder(dataset).execute(appendTxn)) {
            Fragment fragment = appended.getFragments().get(0);
            FragmentUpdateResult updateResult = testDataset.updateColumn(fragment, 4);
            assertTrue(updateResult.getUpdatedRowOffsetBytes().length > 0);
            assertArrayEquals(new long[] {0, 1, 2, 3}, updateResult.getUpdatedRowOffsets());
          }
        }
      }
    }
  }
}
