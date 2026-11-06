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
import org.lance.index.DistanceType;

import java.util.Arrays;
import java.util.Objects;

/**
 * A reusable SQ8 model containing the vector dimension and trained quantization bounds. Train it
 * once with {@link VectorTrainer#trainSqModel} and send {@link #toBytes()} to workers. All segments
 * to be physically merged must also share IVF centroids and distance metric. Cosine models must be
 * trained with {@link DistanceType#Cosine} because encoding normalizes vectors.
 *
 * <p>The opaque protobuf representation is for transfer between workers using the same SDK version,
 * not persistent storage across SDK upgrades.
 *
 * <pre>{@code
 * SQModel model = VectorTrainer.trainSqModel(
 *     dataset, "vector", new SQBuildParams.Builder().build());
 * SQBuildParams params = new SQBuildParams.Builder()
 *     .setModel(SQModel.fromBytes(model.toBytes()))
 *     .build();
 * }</pre>
 */
public final class SQModel {
  static {
    JniLoader.ensureLoaded();
  }

  private final byte[] bytes;
  private final int dimension;
  private final short numBits;

  private SQModel(byte[] bytes, int dimension, short numBits) {
    this.bytes = bytes;
    this.dimension = dimension;
    this.numBits = numBits;
  }

  /** Restore and validate a model received from another process. Invalid metadata is rejected. */
  public static SQModel fromBytes(byte[] bytes) {
    Objects.requireNonNull(bytes, "bytes");
    byte[] copy = Arrays.copyOf(bytes, bytes.length);
    int[] metadata = nativeInspect(copy);
    return new SQModel(copy, metadata[0], (short) metadata[1]);
  }

  /** Return a copy of the opaque serialized model. */
  public byte[] toBytes() {
    return Arrays.copyOf(bytes, bytes.length);
  }

  /** Return the vector dimension against which builds are validated. */
  public int getDimension() {
    return dimension;
  }

  /** Return the number of bits per component (currently 8). */
  public short getNumBits() {
    return numBits;
  }

  private static native int[] nativeInspect(byte[] bytes);
}
