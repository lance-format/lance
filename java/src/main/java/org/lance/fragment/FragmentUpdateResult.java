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
import org.lance.JniLoader;

import com.google.common.base.MoreObjects;
import org.apache.arrow.c.ArrowArrayStream;

/**
 * Result of {@link org.lance.Fragment#updateColumns(ArrowArrayStream, String, String)
 * Fragment.updateColumns()}.
 */
public class FragmentUpdateResult {
  static {
    JniLoader.ensureLoaded();
  }

  private final FragmentMetadata updatedFragment;
  private final long[] fieldsModified;

  /**
   * Matched physical row offsets within the fragment, serialized as portable RoaringBitmap bytes
   * (little-endian, same format as {@link org.lance.operation.Update#updatedFragmentOffsets()}).
   */
  private final byte[] updatedRowOffsetBytes;

  /** Primary public API for constructing a result with portable RoaringBitmap offset bytes. */
  public static FragmentUpdateResult create(
      FragmentMetadata updatedFragment, long[] updatedFieldIds, byte[] updatedRowOffsetBytes) {
    return new FragmentUpdateResult(updatedFragment, updatedFieldIds, updatedRowOffsetBytes);
  }

  /** Two-argument form for callers that do not track per-row offsets; offsets default to empty. */
  public FragmentUpdateResult(FragmentMetadata updatedFragment, long[] updatedFieldIds) {
    this(updatedFragment, updatedFieldIds, new byte[0]);
  }

  private FragmentUpdateResult(
      FragmentMetadata updatedFragment, long[] updatedFieldIds, byte[] updatedRowOffsetBytes) {
    this.updatedFragment = updatedFragment;
    this.fieldsModified = updatedFieldIds;
    this.updatedRowOffsetBytes =
        updatedRowOffsetBytes != null ? updatedRowOffsetBytes : new byte[0];
  }

  /**
   * @deprecated Use {@link #create(FragmentMetadata, long[], byte[])} instead. This constructor
   *     encodes the expanded {@code long[]} offsets into portable RoaringBitmap bytes via JNI and
   *     is retained for backward compatibility with callers compiled against the prior long[]-based
   *     API.
   */
  @Deprecated
  public FragmentUpdateResult(
      FragmentMetadata updatedFragment, long[] updatedFieldIds, long[] updatedRowOffsets) {
    this(
        updatedFragment,
        updatedFieldIds,
        encodeRowOffsetsToBytes(updatedRowOffsets != null ? updatedRowOffsets : new long[0]));
  }

  public FragmentMetadata getUpdatedFragment() {
    return updatedFragment;
  }

  public long[] getFieldsModified() {
    return fieldsModified;
  }

  /**
   * Physical row offsets (0-based within the fragment) whose columns were rewritten, as portable
   * RoaringBitmap bytes.
   */
  public byte[] getUpdatedRowOffsetBytes() {
    return updatedRowOffsetBytes;
  }

  /**
   * Physical row offsets (0-based within the fragment) whose columns were rewritten.
   *
   * @deprecated Use {@link #getUpdatedRowOffsetBytes()} instead.
   */
  @Deprecated
  public long[] getUpdatedRowOffsets() {
    return expandRowOffsetsFromBytes(updatedRowOffsetBytes);
  }

  private static native byte[] encodeRowOffsetsToBytes(long[] rowOffsets);

  private static native long[] expandRowOffsetsFromBytes(byte[] rowOffsetBytes);

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fragmentMetadata", updatedFragment)
        .add("updatedFieldIds", fieldsModified)
        .add("updatedRowOffsetBytesLength", updatedRowOffsetBytes.length)
        .toString();
  }
}
