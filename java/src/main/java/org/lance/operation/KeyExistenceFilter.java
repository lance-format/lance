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
package org.lance.operation;

import java.util.Arrays;
import java.util.Objects;

/** Inserted-key conflict filter carried by an {@link Update} operation. */
public final class KeyExistenceFilter {
  public enum Type {
    EXACT,
    BLOOM
  }

  private final int[] fieldIds;
  private final Type type;
  private final long[] exactKeyHashes;
  private final byte[] bloomBitmap;
  private final int bloomNumBits;
  private final long bloomNumberOfItems;
  private final double bloomProbability;

  private KeyExistenceFilter(
      int[] fieldIds,
      Type type,
      long[] exactKeyHashes,
      byte[] bloomBitmap,
      int bloomNumBits,
      long bloomNumberOfItems,
      double bloomProbability) {
    this.fieldIds = Objects.requireNonNull(fieldIds);
    this.type = Objects.requireNonNull(type);
    this.exactKeyHashes = exactKeyHashes;
    this.bloomBitmap = bloomBitmap;
    this.bloomNumBits = bloomNumBits;
    this.bloomNumberOfItems = bloomNumberOfItems;
    this.bloomProbability = bloomProbability;
  }

  public static KeyExistenceFilter exact(int[] fieldIds, long[] keyHashes) {
    long[] canonicalHashes = Objects.requireNonNull(keyHashes).clone();
    Arrays.sort(canonicalHashes);
    return new KeyExistenceFilter(fieldIds, Type.EXACT, canonicalHashes, null, 0, 0, 0);
  }

  public static KeyExistenceFilter bloom(
      int[] fieldIds, byte[] bitmap, int numBits, long numberOfItems, double probability) {
    Objects.requireNonNull(bitmap, "bitmap must not be null");
    if (numBits <= 0 || numberOfItems <= 0) {
      throw new IllegalArgumentException("numBits and numberOfItems must be positive");
    }
    if (numBits != bitmap.length * Byte.SIZE) {
      throw new IllegalArgumentException(
          String.format(
              "numBits (%d) must equal the bitmap length in bits (%d)",
              numBits, bitmap.length * Byte.SIZE));
    }
    if (!Double.isFinite(probability) || probability <= 0.0 || probability >= 1.0) {
      throw new IllegalArgumentException("probability must be finite and between 0 and 1");
    }
    return new KeyExistenceFilter(
        fieldIds, Type.BLOOM, null, bitmap, numBits, numberOfItems, probability);
  }

  public int[] getFieldIds() {
    return fieldIds;
  }

  public Type getType() {
    return type;
  }

  public long[] getExactKeyHashes() {
    return exactKeyHashes;
  }

  public byte[] getBloomBitmap() {
    return bloomBitmap;
  }

  public int getBloomNumBits() {
    return bloomNumBits;
  }

  public long getBloomNumberOfItems() {
    return bloomNumberOfItems;
  }

  public double getBloomProbability() {
    return bloomProbability;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    KeyExistenceFilter that = (KeyExistenceFilter) o;
    return bloomNumBits == that.bloomNumBits
        && bloomNumberOfItems == that.bloomNumberOfItems
        && Double.compare(bloomProbability, that.bloomProbability) == 0
        && Arrays.equals(fieldIds, that.fieldIds)
        && type == that.type
        && Arrays.equals(exactKeyHashes, that.exactKeyHashes)
        && Arrays.equals(bloomBitmap, that.bloomBitmap);
  }

  @Override
  public int hashCode() {
    int result = Objects.hash(type, bloomNumBits, bloomNumberOfItems, bloomProbability);
    result = 31 * result + Arrays.hashCode(fieldIds);
    result = 31 * result + Arrays.hashCode(exactKeyHashes);
    return 31 * result + Arrays.hashCode(bloomBitmap);
  }
}
