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

import org.lance.JniLoader;

import java.util.Arrays;
import java.util.Objects;

/**
 * A reusable IVF_RQ rotation model. Send {@link #toBytes()} to other builders and restore it with
 * {@link #fromBytes(byte[])}. Physical merging also requires the same IVF centroids and compatible
 * index parameters across segments.
 *
 * <p>Only {@link RQRotationType#FAST} can be serialized because the matrix rotation is omitted by
 * the model format. Dimension must be positive and divisible by 8; numBits must be between 1 and 9
 * inclusive.
 *
 * <pre>{@code
 * RQModel model = RQModel.build(128, (byte) 5);
 * byte[] transferred = model.toBytes();
 * RQBuildParams params = new RQBuildParams.Builder()
 *     .setNumBits((byte) 5)
 *     .setModel(RQModel.fromBytes(transferred))
 *     .build();
 * }</pre>
 */
public final class RQModel {
  static {
    JniLoader.ensureLoaded();
  }

  private final byte[] bytes;
  private final int dimension;
  private final byte numBits;

  private RQModel(byte[] bytes, int dimension, byte numBits) {
    this.bytes = bytes;
    this.dimension = dimension;
    this.numBits = numBits;
  }

  /** Generate a model with fast rotation for the given vector dimension and number of bits. */
  public static RQModel build(int dimension, byte numBits) {
    return new RQModel(nativeBuild(dimension, numBits), dimension, numBits);
  }

  /**
   * Restore a serialized model received from another process. The rotation payload is validated
   * immediately; the index build also checks its dimension against the indexed vectors.
   */
  public static RQModel fromBytes(byte[] bytes) {
    Objects.requireNonNull(bytes, "bytes");
    byte[] copy = Arrays.copyOf(bytes, bytes.length);
    int[] metadata = nativeInspect(copy);
    return new RQModel(copy, metadata[0], (byte) metadata[1]);
  }

  /** Return a copy of the opaque serialized model. */
  public byte[] toBytes() {
    return Arrays.copyOf(bytes, bytes.length);
  }

  public int getDimension() {
    return dimension;
  }

  public byte getNumBits() {
    return numBits;
  }

  public RQRotationType getRotationType() {
    return RQRotationType.FAST;
  }

  private static native byte[] nativeBuild(int dimension, byte numBits);

  private static native int[] nativeInspect(byte[] bytes);
}
